"""
Multi-Agent Research System - Async Implementation
Implements Anthropic's orchestrator-worker architecture with async DSPy
"""

import os
import asyncio
import warnings
from typing import List, Dict, Optional, Literal, Any
from enum import Enum
from pydantic import BaseModel, Field
import dspy
from dspy.adapters import JSONAdapter
import wikipedia
from brave_search_python_client import BraveSearch, WebSearchRequest
from dotenv import load_dotenv
import json
import logging
from utils import setup_langfuse

# Suppress noisy warnings and logs
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# ========== LANGFUSE SETUP ==========
langfuse = setup_langfuse()
# =====================================

# Configuration
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


SMALL_MODEL = os.getenv("GEMINI_2.5_FLASH_LITE")
BIG_MODEL = os.getenv("O4_MINI")


TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))
BIG_MODEL_MAX_TOKENS = int(os.getenv("BIG_MODEL_MAX_TOKENS", "20000"))
SMALL_MODEL_MAX_TOKENS = int(os.getenv("SMALL_MODEL_MAX_TOKENS", "4000"))

# ---------- Data Models (Unchanged) ----------

class SubagentTask(BaseModel):
    """Atomic research micro-task allocated to one sub-agent in the plan."""
    task_id: int = Field(description="Identifier used to track this task across iterations")
    objective: str = Field(description="Crisp single-focus goal the subagent must accomplish")
    tool_guidance: Dict[Literal["web_search", "memory_read", "memory_list"], str] = Field(description="Mapping of allowed tool names to concise usage instructions")
    tool_budget: int = Field(default=8, ge=3, le=15, description="Maximum number of tool calls the subagent may issue")
    expected_output: str = Field(description="Exact artifact or information the subagent must return for completion")
    tip: Optional[str] = Field(default=None, description="Optional hint to improve quality or efficiency while executing the task")


class SubagentResult(BaseModel):
    """Structured report a subagent returns after finishing its task."""
    task_id: int = Field(description="Identifier of the task that produced this result")
    summary: str = Field(description="High-density 2-4 sentence overview of the key findings")
    finding: str = Field(description="Full detailed answer directly addressing the task objective")
    debug_info: Optional[List[str]] = Field(default=None, description="Optional list of raw tool call traces for debugging")


class Memory(BaseModel):
    """In-memory store for research artifacts with lightweight summaries."""
    store: Dict[str, str] = Field(default_factory=dict, description="Full JSON/text storage")
    summaries: Dict[str, str] = Field(default_factory=dict, description="Condensed index cards to reference the memory storage")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._summarizer_lm = dspy.LM(
            model=SMALL_MODEL,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_BASE_URL,
            temperature=0.3,
            max_tokens=SMALL_MODEL_MAX_TOKENS,
        )
        self._summarizer = dspy.Predict(MemorySummary)

    async def summarize(self, text: str) -> str:
        """Create a concise summary for the index card."""
        with dspy.settings.context(lm=self._summarizer_lm):
            result = await self._summarizer.acall(raw_context=text)
            return result.summary
        
    async def write(self, cycle: int, type: str, content: str, task_id: Optional[int] = None) -> str:
        """Write content to memory and create summary index card.
        
        Args:
            cycle: Current cycle index
            type: Stage type ('plan', 'task', 'synthesis')
            content: Full content to store (typically JSON)
            task_id: Optional task ID for subagent results
            
        Returns:
            Key used to store the content
        """
        # Build deterministic key
        key = f"{cycle}-{type}" if task_id is None else f"{cycle}-{type}-{task_id}"
        
        # Store full content
        self.store[key] = content
        self.summaries[key] = await self.summarize(content)
        return key
    
    def read(self, key: str) -> str:
        """Retrieve full content by key.
        
        Args:
            key: Memory key (e.g., '1-plan', '2-task-3')
            
        Returns:
            Full stored content or error message if not found
        """
        return self.store.get(key, f"[ERROR] Key not found: {key}")
    
    def list_keys(self) -> str:
        """List all available memory keys for debugging."""
        if not self.store:
            return "No memory records available yet."
        
        keys = sorted(self.store.keys())
        return f"Available memory keys: {', '.join(keys)}"
    



# ---------- DSPy Signatures (Unchanged) ----------

class MemorySummary(dspy.Signature):
    """Summarize the raw context in a way to easily retrieve and distil the most important information"""
    raw_context: str = dspy.InputField(desc="The raw context to summarize")
    summary: str = dspy.OutputField(desc="The summary of the raw context")    

class PlanResearch(dspy.Signature):
    """Generate strategic reasoning and a parallel task list for subagents from the user's query.
    
    IMPORTANT: Use consistent formatting for ALL field headers with ## on BOTH sides:
    [[ ## next_thought ## ]]
    [[ ## next_tool_name ## ]]  
    [[ ## next_tool_args ## ]]
    [[ ## reasoning ## ]]
    [[ ## tasks ## ]]  
    """
    query: str = dspy.InputField(desc="Original user question providing context")
    memory_summaries: Dict[str, str] = dspy.InputField(default={}, desc="Index of memory records with summaries. If empty, memory_read tool can't be used.")
    reasoning: str = dspy.OutputField(desc="Extended strategic thinking to persist in memory")
    tasks: List[SubagentTask] = dspy.OutputField(desc="Complete list of SubagentTask objects to execute in parallel")

class ExecuteSubagentTask(dspy.Signature):
    """Run one subagent task with its tools and return a structured result.
    
    IMPORTANT: Use consistent formatting for ALL field headers with ## on BOTH sides:
    [[ ## next_thought ## ]]
    [[ ## next_tool_name ## ]]  
    [[ ## next_tool_args ## ]]
    """
    task: SubagentTask = dspy.InputField(desc="SubagentTask definition to be executed")
    final_result: SubagentResult = dspy.OutputField(desc="Structured outcome generated by the subagent")
    reasoning: str = dspy.OutputField(desc="Subagent's internal rationale for the result")

class SynthesizeAndDecide(dspy.Signature):
    """Synthesize completed research results and determine if investigation is complete or requires additional cycles."""
    query: str = dspy.InputField(desc="Original user question guiding evaluation")
    memory_summaries: Dict[str, str] = dspy.InputField(default={}, desc="Index of memory records with summaries")
    completed_results: List[SubagentResult] = dspy.InputField(desc="List of SubagentResult objects ready for synthesis")
    synthesis: str = dspy.OutputField(desc="New consolidated insights from the completed results")
    is_done: bool = dspy.OutputField(desc="True if research is complete and ready to generate final report, False to continue with additional cycles")
    gap_analysis: Optional[str] = dspy.OutputField(desc="If not done, summarize gaps/incompletes/conflicts to address")
    refined_query: Optional[str] = dspy.OutputField(desc="If not done, provide refined query for next iteration")

class FinalReport(dspy.Signature):
    """Distill memory into a final report. Use tools to read details.
    
    IMPORTANT: Use consistent formatting for ALL field headers with ## on BOTH sides:
    [[ ## next_thought ## ]]
    [[ ## next_tool_name ## ]]  
    [[ ## next_tool_args ## ]]
    """
    query: str = dspy.InputField(desc="Original query")
    memory_summaries: Dict[str, str] = dspy.InputField(desc="Memory index summaries")
    final_synthesis: str = dspy.InputField(desc="Final synthesis of the last cycle")
    steps_trace: List[str] = dspy.InputField(desc="List of steps taken by the agent. Helpful to understand entire process taken to answer the query")
    report: str = dspy.OutputField(desc="Complete Markdown report")

# ---------- Async Tool Implementations ----------

async def web_search(query: str, count: int = 5) -> str:
    """Search the web using Brave Search.
    
    Args:
        query: Search query
        count: Number of results (default: 5)
        
    Returns:
        Formatted search results
    """
    if count > 5:
        count = 5
    
    client = BraveSearch(api_key=BRAVE_SEARCH_API_KEY)
    
    try:
        response = await client.web(WebSearchRequest(q=query, count=count))
        
        if not response.web or not response.web.results:
            return f"No results found for '{query}'"

        results = []
        for i, result in enumerate(response.web.results[:count], 1):
            results.append(f"{i}. {result.title}\\n   {result.description}\\n   {result.url}")
        
        return f"Search results for '{query}':\\n\\n" + "\\n\\n".join(results)
    except Exception as e:
        return f"Search error: {e}"


def wikipedia_search(query: str, sentences: int = 3) -> str:
    """
    Return a concise English summary for `query` (≤ `sentences` sentences).

    If Wikipedia returns multiple possible pages (disambiguation), we list the
    top 5 options so the calling agent can decide what to do next.
    """
    try:
        wikipedia.set_lang("en")
        titles = wikipedia.search(query, results=1)
        if not titles:
            return f"No Wikipedia article found for '{query}'."

        title = titles[0]
        summary = wikipedia.summary(title, sentences=sentences, auto_suggest=False)
        return f"Wikipedia – {title}\\n\\n{summary}"

    except wikipedia.exceptions.DisambiguationError as e:
        # Show a short disambiguation list
        opts = "\\n • ".join(e.options[:5])
        return f"Wikipedia disambiguation for '{query}'. Try one of:\\n • {opts}"
    except Exception as err:
        return f"Wikipedia error: {err}"


async def async_batch_call(calls: list[dict]) -> list[str]:
    """
    Execute multiple tool calls in parallel for efficiency using async.
    
    Args:
        calls: List of dicts, each with:
            - tool_name: Name of the tool ('web_search' or 'wikipedia_search')
            - args: Dictionary of arguments for that tool
            
    Example:
        calls = [
            {"tool_name": "web_search", "args": {"query": "Lamine Yamal stats", "count": 2}},
            {"tool_name": "web_search", "args": {"query": "Desire Doue stats", "count": 2}},
            {"tool_name": "wikipedia_search", "args": {"query": "Lamine Yamal", "sentences": 5}},
            {"tool_name": "wikipedia_search", "args": {"query": "Desire Doue", "sentences": 5}}
        ]
    
    Returns:
        List of results in the same order as input calls
    """
    tasks = []
    formatted_results = []
    
    for call in calls:
        tool_name = call.get("tool_name")
        args = call.get("args", {})
        
        if tool_name == "web_search":
            # Async function - call directly
            task = web_search(**args)
        elif tool_name == "wikipedia_search":
            # Sync function - run in thread pool
            task = asyncio.to_thread(wikipedia_search, **args)
        else:
            # Invalid tool
            formatted_results.append(f"[ERROR] Unknown tool: {tool_name}")
            continue

        tasks.append(task)
    
    # Execute all tasks concurrently with error handling
    # return_exceptions=True ensures failed tools don't crash everything
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Format results with tool names
    for i, output in enumerate(results):
        tool_name = calls[i].get("tool_name", "unknown")
        if isinstance(output, Exception):
            # Convert exception to string so LLM can reason about the failure
            error_msg = f"[ERROR] {type(output).__name__}: {str(output)}"
            formatted_results.append(f"{tool_name}: {error_msg}")
        else:
            formatted_results.append(f"{tool_name}: {output}")
    
    return formatted_results


# ---------- Helper Functions ----------

def prediction_to_json(prediction) -> str:
    """Convert DSPy Prediction object to JSON string."""
    data = {}
    # Extract all fields from the prediction's _store
    if hasattr(prediction, '_store'):
        for key, value in prediction._store.items():
            # Handle Pydantic models
            if hasattr(value, 'model_dump'):
                data[key] = value.model_dump()
            # Handle lists of Pydantic models
            elif isinstance(value, list) and value and hasattr(value[0], 'model_dump'):
                data[key] = [item.model_dump() for item in value]
            else:
                data[key] = value
    return json.dumps(data)


# ---------- Lead Agent DSPy Modules ----------

class LeadAgent(dspy.Module):
    """Lead agent: plans once, launches parallel subagents, synthesizes, then decides."""

    def __init__(self):
        super().__init__()
        self.memory = Memory()
        self.cycle_idx = 0  # Track cycles for memory keys
        self.steps_trace: List[Dict[str, Any]] = []  # High-level trace of steps

        # Initialize all tools - simplified for better ReAct compatibility
        self.tools = {
            "web_search": dspy.Tool(
                web_search,
                name="web_search",
                desc="Search the web for information. Default 5 results."
            ),
            "wikipedia_search": dspy.Tool(
                wikipedia_search,
                name="wikipedia_search",
                desc="Search Wikipedia for information. Max 3 sentences."
            ),
            # "parallel_search": dspy.Tool(
            #     async_batch_call,
            #     name="parallel_search",
            #     desc="Run multiple searches in parallel. Provide tool_name ('web_search' or 'wikipedia_search') and args for each call."
            # ),
            "memory_read": dspy.Tool(
                lambda key: self.memory.read(key),
                name="memory_read",
                desc="Retrieve full memory record by key. Use format: '{cycle}-{type}' or '{cycle}-task-{task_id}'. Examples: '1-plan', '1-synthesis', '1-task-0', '1-task-1'."
            ),
            "memory_list": dspy.Tool(
                lambda: self.memory.list_keys(),
                name="memory_list",
                desc="List all available memory keys to see what data has been stored."
            )
        }

        # Language models
        self.planner_lm = dspy.LM(
            model=BIG_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=BIG_MODEL_MAX_TOKENS,
        )

        self.subagent_lm = dspy.LM(
            model=BIG_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=BIG_MODEL_MAX_TOKENS,
        )

        self.synthesizer_lm = dspy.LM(
            model=SMALL_MODEL,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_BASE_URL,
            temperature=TEMPERATURE,
            max_tokens=SMALL_MODEL_MAX_TOKENS,
        )


        # Create modules without context - we'll set context during execution
        planning_tools = [self.tools["web_search"], self.tools["memory_read"], self.tools["memory_list"]]
        # Create ReAct with JSONAdapter for better parsing reliability
        self.planner = dspy.ReAct(PlanResearch, tools=planning_tools, max_iters=3)
        self.synthesizer = dspy.ChainOfThought(SynthesizeAndDecide)

    async def aforward(self, query: str):
        """Plan → execute tasks in parallel → synthesize/decide (single cycle)."""
        
        # Increment cycle counter
        self.cycle_idx += 1

        # 1. Generate plan
        print(f"\n🔍 CYCLE {self.cycle_idx}: Starting planning phase...")
        print(f"📝 Query: {query}")
        print(f"💾 Memory summaries available: {len(self.memory.summaries)} items")
        
        with dspy.context(lm=self.planner_lm):
            plan = await self.planner.acall(query=query, memory_summaries=self.memory.summaries)
        
        print(f"✅ Plan generated successfully!")
        print(f"🎯 Reasoning: {plan.reasoning[:150]}...")
        print(f"📋 Tasks created: {len(plan.tasks)}")
        for i, task in enumerate(plan.tasks):
            print(f"   Task {i}: {task.objective[:100]}...")
        
        # Write plan to memory (non-blocking)
        plan_key = await self.memory.write(self.cycle_idx, "plan", prediction_to_json(plan))
        self.steps_trace.append({
            'cycle': self.cycle_idx,
            'action': 'Planned',
            'summary': f"Generated {len(plan.tasks)} tasks",
            'memory_key': plan_key
        })

        # 2. Launch subagents in parallel
        print(f"\n🚀 Launching {len(plan.tasks)} subagents in parallel...")
        task_calls = []
        for task in plan.tasks:
            # Include memory tools for all subagents
            permitted_tools = [self.tools[name] for name in task.tool_guidance.keys() if name in self.tools]
            permitted_tools.extend([self.tools["memory_read"], self.tools["memory_list"]])
            
            async def run_sub(task=task, permitted_tools=permitted_tools):
                with dspy.context(lm=self.subagent_lm):
                    sub = dspy.ReAct(ExecuteSubagentTask, tools=permitted_tools, max_iters=task.tool_budget)
                    sub.adapter = JSONAdapter()  # Use JSON adapter for better parsing
                    return await sub.acall(task=task)
            
            task_calls.append(run_sub())

        raw_results = await asyncio.gather(*task_calls, return_exceptions=True)
        print(f"\n📊 Processing {len(raw_results)} subagent results...")
        
        results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                print(f"❌ Subagent {i} error: {str(r)}")
                continue
            if hasattr(r, "final_result") and r.final_result is not None:
                results.append(r.final_result)
                await self.memory.write(self.cycle_idx, "task", r.final_result.model_dump_json(), task_id=r.final_result.task_id)
                print(f"✅ Subagent {i} completed - Task ID: {r.final_result.task_id}")
                print(f"   Summary: {r.final_result.summary[:100]}...")
            else:
                print(f"⚠️  Subagent {i} returned invalid result")
        
        print(f"📈 Successfully collected {len(results)} valid results")

        # 3. Synthesize and decide
        print(f"\n🧠 Starting synthesis phase...")
        print(f"🔗 Synthesizing {len(results)} results")
        
        with dspy.context(lm=self.synthesizer_lm):
            decision = await self.synthesizer.acall(query=query, memory_summaries=self.memory.summaries, completed_results=results)
        
        print(f"🎯 Synthesis completed!")
        print(f"✅ Decision: {'DONE' if decision.is_done else 'CONTINUE'}")
        print(f"📝 Synthesis: {decision.synthesis[:150]}...")
        if not decision.is_done:
            print(f"🔍 Gap analysis: {decision.gap_analysis[:100]}...")
            print(f"🔄 Refined query: {decision.refined_query[:100]}...")
        
        # Write synthesis to memory (non-blocking)
        synth_key = await self.memory.write(self.cycle_idx, "synthesis", prediction_to_json(decision))
        self.steps_trace.append({
            'cycle': self.cycle_idx,
            'action': 'Synthesized',
            'summary': f"Decision: {'DONE' if decision.is_done else 'CONTINUE'}, {len(results)} results",
            'memory_key': synth_key
        })

        return {
            "is_done": decision.is_done,
            "synthesis": decision.synthesis,
            "gap_analysis": decision.gap_analysis,
            "refined_query": decision.refined_query,
            "results": results,
        }

    async def run(self, query: str):
        """Runs a single minimal-cycle research and returns the best available answer."""
        print(f"\n🎬 STARTING RESEARCH SESSION")
        print(f"🎯 Original query: {query}")
        print("=" * 80)
        
        result = None
        cycle_count = 0
        
        while result is None or not result["is_done"]:
            current_query = query
            if result and "refined_query" in result and result["refined_query"]:
                current_query = result["refined_query"]  # Use refined if provided
                print(f"\n🔄 REFINED QUERY (Cycle {cycle_count}): {current_query}")
            
            result = await self.aforward(current_query)
            print(f"\n📊 CYCLE {cycle_count} COMPLETE - Decision: {'DONE ✅' if result['is_done'] else 'CONTINUE 🔄'}")
            
            
        # When done, use ReAct to distill into report
        if result["is_done"]:
            print(f"\n📄 GENERATING FINAL REPORT...")
            print(f"📚 Available memory keys: {len(self.memory.summaries)}")
            print(f"📋 Steps taken: {len(self.steps_trace)}")
            
            final_react = dspy.ReAct(FinalReport, tools=[self.tools["memory_read"], self.tools["memory_list"]], max_iters=3)
            final_react.adapter = JSONAdapter()  # Use JSON adapter for better parsing
            with dspy.context(lm=self.synthesizer_lm):
                final_result = await final_react.acall(
                    query=query,
                    memory_summaries=self.memory.summaries,
                    final_synthesis=result["synthesis"],
                    steps_trace=self.steps_trace  # Use for indexing memory reads
                )
            
            print(f"📝 Final report generated!")
            print(f"📏 Report length: {len(final_result.report)} characters")
            print("🎉 RESEARCH SESSION COMPLETE!")
            print("=" * 80)
            return final_result.report
            
        print("⚠️  Research incomplete, returning synthesis")
        return result["synthesis"]  # Fallback
    


if __name__ == "__main__":
    print("🤖 Initializing LeadAgent...")
    agent = LeadAgent()
    print("✅ Agent initialized successfully!")
    
    result = asyncio.run(agent.run("Doue vs Yamal who's better? Be objective"))
    
    print("\n" + "="*80)
    print("📋 FINAL RESULT:")
    print("="*80)
    print(result)