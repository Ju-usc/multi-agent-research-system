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
    budget_calls: int = Field(default=5, ge=1, le=10)


class ResearchPlan(BaseModel):
    """Complete research plan with steps"""
    steps: List[PlanStep]


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
    plan: ResearchPlan = dspy.OutputField(desc="Delegation plan with specific tasks for subagents")


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
            max_tokens=MAX_TOKENS
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


    
    async def aforward(self, query: str):
        """Async forward pass for lead agent"""
        # Phase 1: Analyze query
        analysis = await self.query_analyzer.acall(query=query)
        
        # Phase 2: Create research plan
        plan = await self.planner.acall(
            query=query,
            analysis=analysis.analysis
        )
        
        return analysis, plan


# ---------- Main Orchestration Function ----------

async def run_research(query: str, planner_lm=None, verbose=True):
    """
    Main entry point for running async research.
    
    Args:
        query: The research query
        planner_lm: Optional LM for planning phase
        verbose: Whether to print progress
    
    Returns:
        Tuple of (analysis, plan)
    """
    # Create lead agent
    agent = AsyncLeadAgent(planner_lm=planner_lm)
    
    # Run research
    if verbose:
        print("=== Starting Async Research ===")
        print(f"Query: {query}\\n")
    
    analysis, plan = await agent.aforward(query)
    
    if verbose:
        print("=== Query Analysis ===")
        print(f"Type: {analysis.analysis.query_type}")
        print(f"Complexity: {analysis.analysis.complexity}")
        print(f"Main concepts: {analysis.analysis.main_concepts}")
        print(f"Key entities: {analysis.analysis.key_entities}")
        
        print("\\n=== Research Plan ===")
        print(f"Plan has {len(plan.plan.steps)} steps:")
        for step in plan.plan.steps:
            print(f"\\nStep {step.id}: {step.description}")
            print(f"  Budget: {step.budget_calls} tool calls")
            print(f"  Depends on: {step.depends_on}")
    
    return analysis, plan


# ---------- Synchronous Wrapper (for backwards compatibility) ----------

def run_research_sync(query: str, planner_lm=None, verbose=True):
    """Synchronous wrapper for run_research"""
    return asyncio.run(run_research(query, planner_lm, verbose))