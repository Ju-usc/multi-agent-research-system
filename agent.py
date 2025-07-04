"""
Multi-Agent Research System - Async Implementation
Implements Anthropic's orchestrator-worker architecture with async DSPy
"""

import os
import asyncio
from typing import List, Dict, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field
import dspy
from dspy.adapters import JSONAdapter
import wikipedia
from brave_search_python_client import BraveSearch, WebSearchRequest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

SMALL_MODEL = os.getenv("GEMINI_2.5_FLASH_LITE")
BIG_MODEL = os.getenv("GPT_4.1_MINI")

TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))

# ---------- Data Models (Unchanged) ----------

class QueryAnalysis(BaseModel):
    """Analyze and break down the user's prompt to make sure you fully understand it."""
    query_type: Literal["depth_first", "breadth_first", "straightforward"] = Field(
        description="The type of query to be answered"
    )
    complexity: Literal["simple", "medium", "complex"] = Field(
        description="The complexity of the query"
    )
    main_concepts: List[str] = Field(description="Key concepts to research")
    key_entities: List[str] = Field(description="Important entities mentioned")
    relationships: List[str] = Field(description="Relationships in the task")
    notes: Optional[str] = Field(description="Any temporal or contextual constraints on the question")
    answer_format: Optional[str] = Field(
        description="The recommended format of the answer such as detailed report, a list of entities, an analysis of different perspectives, etc."
    )


class PlanStep(BaseModel):
    """A single step in the research plan"""
    id: int
    description: str
    depends_on: List[int] = Field(default_factory=list)
    complexity_hint: Optional[Literal["simple", "medium", "complex"]] = None


class SubagentTask(PlanStep):
    """Executable research *micro-task* assigned to a single sub-agent."""
    tools_to_use: List[str] = Field(
        default=["web_search", "wikipedia_search", "parallel_search"],
        description="Which tools this subagent should use"
    )
    tool_budget: int = Field(
        default=8, 
        ge=3, 
        le=15,
        description="Maximum number of tool calls allowed"
    )
    complexity: Literal["simple", "medium", "complex"] = Field(
        description="The complexity of the task"
    )
    execution_notes: Optional[str] = None
    expected_output: str = Field(
        description="What kind of output/information is expected from this task"
    )


class TaskAllocation(BaseModel):
    """Bundle of sub-agent tasks plus simple scheduling hints."""
    tasks: List[SubagentTask] = Field(
        description="List of tasks to be executed by subagents"
    )
    execution_strategy: str = Field(
        description="Optional free-text scheduling guidance"
    )
    max_concurrent: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum number of concurrent subagents (Brave API safe default)"
    )


class SubagentResult(BaseModel):
    """Compact summary returned by a sub-agent when done."""
    task_id: str = Field(description="ID of the task that was executed")
    summary: str = Field(description="Summary of findings from the subagent")
    finding: str = Field(description="Detailed findings from the subagent which answers the assigned task")
    debug_info: Optional[List[str]] = Field(
        default=None,
        description="Tool call traces for debugging only."
    )


# ---------- DSPy Signatures (Unchanged) ----------

class AnalyzeQuery(dspy.Signature):
    """Analyze query to determine research strategy. Categorize as depth-first (one topic, multiple angles), 
    breadth-first (multiple independent topics), or straightforward (simple fact)."""
    query: str = dspy.InputField(desc="The user's research query")
    analysis: QueryAnalysis = dspy.OutputField(desc="Strategic analysis for delegation planning")


class PlanResearch(dspy.Signature):
    """Create delegation plan for subagents based on analysis. Use tools sparingly for reconnaissance only 
    (verify entities, assess scope). Output specific research tasks for subagents, not research results."""
    query: str = dspy.InputField(desc="The user's research query")
    analysis: QueryAnalysis = dspy.InputField(desc="Strategic analysis from previous step")
    reasoning: str = dspy.OutputField(desc="Reasoning about the plan")
    plans: List[PlanStep] = dspy.OutputField(desc="A list of delegation plans with specific tasks for subagents")
    # context: str = dspy.OutputField(desc="Any Relevant or Necessary Context to pass to subagents")


class DecomposeToTasks(dspy.Signature):
    """Break down a *single* plan step into smaller, independent SubagentTasks.

    You receive the full ResearchPlan for global awareness, but **must create tasks only for `current_step`**.
    Decompose complex steps into 1-4 atomic tasks that can run concurrently for subagents. Do not invent tasks for other steps."""
    query: str = dspy.InputField(desc="The original user query for context")
    plans: List[PlanStep] = dspy.InputField(desc="Full research plan for reference")
    current_step: PlanStep = dspy.InputField(desc="The single plan step to decompose now")
    completed_results: List[SubagentResult] = dspy.InputField(
        desc="Results from already completed tasks (empty list if none)"
    )
    allocation: TaskAllocation = dspy.OutputField(
        desc="Allocation of subagent tasks with execution strategy"
    )

class ExecuteSubagentTask(dspy.Signature):
    """Execute a single subagent task using tools and return the result."""
    task: SubagentTask = dspy.InputField(desc="The subagent task to execute")
    final_result: SubagentResult = dspy.OutputField(desc="The result of the subagent task")
    reasoning: str = dspy.OutputField(desc="Reasoning about the result")

class SynthesizeResults(dspy.Signature):
    """Synthesize all subagent results into a comprehensive answer to the user's query. 
    Focus on answering the original question completely while highlighting key findings, 
    patterns, and relationships discovered across all research tasks."""
    query: str = dspy.InputField(desc="The original user query")
    analysis: QueryAnalysis = dspy.InputField(desc="Initial strategic analysis")
    plans: List[PlanStep] = dspy.InputField(desc="Full research plan for reference")
    current_step: PlanStep = dspy.InputField(desc="The single plan step to synthesize now")
    completed_results: List[SubagentResult] = dspy.InputField(desc="Results from already completed tasks (empty list if none)")

    synthesis: str = dspy.OutputField(desc="Comprehensive synthesized answer addressing the user's query")
    key_findings: List[str] = dspy.OutputField(desc="Key findings from the synthesis")
    reflection: str = dspy.OutputField(desc="Reflection ")
    gap_analysis: str = dspy.OutputField(desc="Gap analysis")
    reasoning: str = dspy.OutputField(desc="Reasoning about the synthesis")
    next_decision: Literal["DONE", "REPLAN", "CONTINUE"] = dspy.OutputField(desc="Next decision to make")


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


# Register tools with DSPy
TOOLS = {
    "web_search": dspy.Tool(web_search),
    "wikipedia_search": dspy.Tool(wikipedia_search),
    "parallel_search": dspy.Tool(
        async_batch_call,
        name="parallel_search",
        desc="Run multiple searches in parallel. Provide tool_name ('web_search' or 'wikipedia_search') and args for each call."
    ),
    "quick_search": dspy.Tool(
        web_search,
        name="quick_search",
        desc="Single quick search to check information availability. Max 2 results."
    )
}


# ---------- Async DSPy Modules ----------

class AsyncLeadAgent(dspy.Module):
    """Lead agent that analyzes queries and creates research plans"""
        
    def __init__(self):
        super().__init__()

        # Configure DSPy with OpenRouter
        analysis_lm = dspy.LM(
            model=SMALL_MODEL,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_BASE_URL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        planner_lm = dspy.LM(
            model=BIG_MODEL,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_BASE_URL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        decomposer_lm = dspy.LM(
            model=BIG_MODEL,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_BASE_URL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            adapter=JSONAdapter()  # Use JSON adapter for structured outputs
        )

        subagent_lm = dspy.LM(
            model=BIG_MODEL,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_BASE_URL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        # Create modules with specific LMs using context managers
        with dspy.settings.context(lm=analysis_lm):
            self.query_analyzer = dspy.ChainOfThought(AnalyzeQuery)

        with dspy.settings.context(lm=planner_lm):
            planning_tools = [TOOLS["parallel_search"], TOOLS["quick_search"]]
            self.planner = dspy.ReAct(
                PlanResearch,
                tools=planning_tools,
                max_iters=3  # Minimal tool usage for planning
            )
        
        with dspy.settings.context(lm=decomposer_lm):
            self.decomposer = dspy.ChainOfThought(DecomposeToTasks)

        # Store subagent LM for dynamic configuration
        self.subagent_lm = subagent_lm

    
    async def aforward(self, query: str):
        """Async forward pass for lead agent"""
        # Phase 1: Analyze query
        analysis = await self.query_analyzer.acall(query=query)
        
        # Phase 2: Create research plan
        research_plan = await self.planner.acall(
            query=query,
            analysis=analysis.analysis
        )

        # Phase 3: Decompose the current working step
        current_step = research_plan.plans.pop(0)
        allocation = await self.decomposer.acall(
            query=query,
            plans=research_plan.plans,
            current_step=current_step,
            completed_results=[],
        )

        # Phase 4: run tasks with subagents in parallel
        subagent_tasks = []
        for task in allocation.allocation.tasks:
            # Create task-specific subagent with dynamic configuration
            task_tools = [TOOLS[tool] for tool in task.tools_to_use if tool in TOOLS]
            
            with dspy.settings.context(lm=self.subagent_lm):
                task_subagent = dspy.ReAct(
                    ExecuteSubagentTask,
                    tools=task_tools,
                    max_iters=task.tool_budget
                )
            
            subagent_tasks.append(task_subagent.acall(task=task))
        
        subagent_results = await asyncio.gather(*subagent_tasks, return_exceptions=True)
        
        # Phase 5: Synthesize results and reflect results to decide next step
        # DONE, REPLAN, CONTINUE



        return analysis, research_plan, allocation, subagent_results


