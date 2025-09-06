"""
Multi-Agent Research System - Async Implementation
Implements Anthropic's orchestrator-worker architecture with async DSPy
"""

import asyncio
from typing import List, Dict, Optional, Any
import dspy
from dspy.adapters.baml_adapter import BAMLAdapter
import logging

# Import from new modules
from config import (
    BRAVE_SEARCH_API_KEY, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENAI_API_KEY,
    SMALL_MODEL, BIG_MODEL, TEMPERATURE, BIG_MODEL_MAX_TOKENS, SMALL_MODEL_MAX_TOKENS
)
from models import (
    SubagentTask, SubagentResult, FileSystem,
    PlanResearch, ExecuteSubagentTask, SynthesizeAndDecide, FinalReport
)
from tools import (
    WebSearchTool, FileSystemTool
)
from utils import setup_langfuse, prediction_to_markdown, log_call
from langfuse import observe

# Setup logging
logger = logging.getLogger(__name__)

# ========== LANGFUSE SETUP ==========
langfuse = setup_langfuse()
# =====================================

 # ---------- Lead Agent DSPy Module ----------

class LeadAgent(dspy.Module):
    """Lead agent: plans once, launches parallel subagents, synthesizes, then decides."""

    def __init__(self):
        super().__init__()
        self.fs = FileSystem()
        self.cycle_idx = 0  # Track cycles
        # Single BAML adapter instance used across all modules
        self.baml_adapter = BAMLAdapter()

        # Initialize tool instances
        self.web_search_tool = WebSearchTool(BRAVE_SEARCH_API_KEY)
        self.fs_tool = FileSystemTool(self.fs)

        # Create DSPy tools from class instances
        self.tools = {
            "web_search": dspy.Tool(
                self.web_search_tool,
                name="web_search",
                desc="Search the web for information. Default 5 results."
            ),
            "filesystem_read": dspy.Tool(
                self.fs_tool.read,
                name="filesystem_read",
                desc="Read file from research memory. Use path relative to memory/, e.g., 'cycle_003/foo/bar.md' (do NOT include leading 'memory/')."
            ),
            "filesystem_tree": dspy.Tool(
                self.fs_tool.tree,
                name="filesystem_tree",
                desc="Show filesystem structure. Tree shows entries under 'memory/'. When calling filesystem_read, drop the 'memory/' prefix."
            )
        }

        # Initialize language models
        self.planner_lm = dspy.LM(
            model=BIG_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=BIG_MODEL_MAX_TOKENS,
        )
        # Create modules without context - we'll set context during execution
        planning_tools = [self.tools["web_search"], self.tools["filesystem_read"], self.tools["filesystem_tree"]]
        self.planner = dspy.ReAct(PlanResearch, tools=planning_tools, max_iters=3)
        self.planner.adapter = self.baml_adapter

        # Subagents run on the small model
        self.subagent_lm = dspy.LM(
            model=SMALL_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=SMALL_MODEL_MAX_TOKENS,
        )

        self.synthesizer_lm = dspy.LM(
            model=SMALL_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=SMALL_MODEL_MAX_TOKENS,
        )
        self.synthesizer = dspy.ChainOfThought(SynthesizeAndDecide)
        self.synthesizer.adapter = self.baml_adapter

        self.final_report_lm = dspy.LM(
            model=BIG_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=BIG_MODEL_MAX_TOKENS,
        )
        
        # Final report generation module
        final_report_tools = [self.tools["filesystem_read"], self.tools["filesystem_tree"]]
        self.final_reporter = dspy.ReAct(FinalReport, tools=final_report_tools, max_iters=5)
        self.final_reporter.adapter = self.baml_adapter




    @observe(name="execute_subagent", capture_input=True, capture_output=True)
    async def execute_subagent_task(self, task: SubagentTask, subagent_lm) -> Optional[SubagentResult]:
        """Execute a single subagent task with proper tool setup and error handling.
        
        Args:
            task: The SubagentTask to execute
            subagent_lm: Language model to use for the subagent
            
        Returns:
            SubagentResult on success, None on failure
        """
        try:
            # Setup permitted tools for this task
            permitted_tools = [self.tools[name] for name in task.tool_guidance.keys() if name in self.tools]
            # Always include filesystem tools
            permitted_tools.extend([self.tools["filesystem_read"], self.tools["filesystem_tree"]])
            
            # Execute subagent with DSPy context
            with dspy.context(lm=subagent_lm):
                sub = dspy.ReAct(ExecuteSubagentTask, tools=permitted_tools, max_iters=task.tool_budget)
                # Use BAML adapter for improved structured outputs on nested Pydantic models
                sub.adapter = self.baml_adapter
                result = await sub.acall(task=task)
        
            # Return result - final_result is guaranteed by ExecuteSubagentTask signature
            if result.final_result is not None:
                return result.final_result
            else:
                logger.warning(f"Task {task.task_name} returned invalid result")
                return None
                
        except Exception as e:
            logger.error(f"Task {task.task_name} error: {str(e)}")
            return None

    @log_call
    @observe(name="execute_parallel_tasks", capture_input=True, capture_output=True)
    async def execute_tasks_parallel(self, tasks: List[SubagentTask]) -> List[SubagentResult]:
        """Execute multiple subagent tasks in parallel and collect results.
        
        Args:
            tasks: List of SubagentTask objects to execute
            
        Returns:
            List of valid SubagentResult objects (excludes failures)
        """
        # Create task calls
        task_calls = [self.execute_subagent_task(task, self.subagent_lm) for task in tasks]

        # Execute all tasks in parallel with error handling
        raw_results = await asyncio.gather(*task_calls, return_exceptions=True)
        
        # Filter and process valid results
        results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.error(f"Subagent {i} error: {str(r)}")
                continue
            if r is not None:
                results.append(r)
                # Update task_name in result
                r.task_name = tasks[i].task_name
                # Write result to filesystem (generic markdown rendering)
                result_path = f"cycle_{self.cycle_idx:03d}/{r.task_name}/result.md"
                self.fs.write(result_path, prediction_to_markdown(r, title=r.task_name))
            else:
                logger.warning(f"Subagent {i} returned invalid result")
        
        return results

    @log_call
    @observe(name="plan_research", capture_input=True, capture_output=True)
    async def plan_research(self, query: str) -> dspy.Prediction:
        """Execute planning phase and return plan.
        
        Args:
            query: Research query to plan for
            
        Returns:
            PlanResult prediction containing tasks and reasoning
        """
        # Get current filesystem structure
        memory_tree = self.fs.tree()

        with dspy.context(lm=self.planner_lm):
            plan = await self.planner.acall(query=query, memory_tree=memory_tree)

        # Determine a safe filename (tests may not set plan.plan_filename)
        plan_filename = getattr(plan, "plan_filename", None) or "test-plan"
        # Backfill on the object for downstream references
        try:
            plan.plan_filename = plan_filename
        except Exception:
            pass

        # Write plan to filesystem
        plan_path = f"cycle_{self.cycle_idx:03d}/{plan_filename}.md"
        self.fs.write(plan_path, prediction_to_markdown(plan, title="Plan"))
        
        return plan
    
    @log_call
    @observe(name="synthesize", capture_input=True, capture_output=True)
    async def synthesize_results(self, query: str, results: List[SubagentResult]) -> dspy.Prediction:
        """Execute synthesis phase and return decision.
        
        Args:
            query: Original research query
            results: List of completed subagent results
            
        Returns:
            SynthesisResult prediction containing decision and synthesis
        """
        # Get updated filesystem structure
        memory_tree = self.fs.tree()

        with dspy.context(lm=self.synthesizer_lm):
            decision = await self.synthesizer.acall(
                query=query, memory_tree=memory_tree, completed_results=results
            )

        # Write synthesis to filesystem
        synthesis_path = f"cycle_{self.cycle_idx:03d}/synthesis.md"
        self.fs.write(synthesis_path, prediction_to_markdown(decision, title="Synthesis"))

        return decision

    @log_call
    @observe(name="generate_final_report", capture_input=True, capture_output=True)
    async def generate_final_report(self, query: str, final_synthesis: str) -> str:
        """Generate final markdown report from memory.
        
        Args:
            query: Original research query
            final_synthesis: Final synthesis from last cycle
            
        Returns:
            Markdown formatted final report
        """
        # Get full filesystem structure
        memory_tree = self.fs.tree(max_depth=None)
        
        with dspy.context(lm=self.final_report_lm):
            final_result = await self.final_reporter.acall(
                query=query,
                memory_tree=memory_tree,
                final_synthesis=final_synthesis,
            )

        # Write final report under active cycle and a root copy for convenience
        self.fs.write(f"cycle_{self.cycle_idx:03d}/final_report.md", final_result.report)

        return final_result.report

    @log_call
    @observe(name="lead_agent_main", capture_input=True, capture_output=True)
    async def aforward(self, query: str):
        """Single cycle: plan → execute (parallel) → synthesize, optional final_report.

        Returns a dict with decision surface and an 'artifacts' dict for tracing.
        """

        # Start in idle state; cycle increments when planning begins
        artifacts: Dict[str, Any] = {
            "cycle": None,
            "query": query,
            "plan": None,
            "results": [],
            "decision": None,
            "final_report": None,
        }

        next_action = "plan"

        while True:
            if next_action == "plan":
                # Begin a new cycle and record it
                self.cycle_idx += 1
                artifacts["cycle"] = self.cycle_idx
                plan = await self.plan_research(query)
                artifacts["plan"] = plan
                # Use tasks provided by the planner
                tasks = plan.tasks
                next_action = "execute"

            elif next_action == "execute":
                results = await self.execute_tasks_parallel(tasks)
                artifacts["results"] = results
                next_action = "synthesize"

            elif next_action == "synthesize":
                decision = await self.synthesize_results(query, artifacts["results"])
                artifacts["decision"] = decision

                if decision.is_done:
                    next_action = "final_report"
                else:
                    next_action = "plan"

            elif next_action == "final_report":
                # Generate and store final report
                final_report = await self.generate_final_report(query, artifacts["decision"].synthesis)
                artifacts["final_report"] = final_report
                return {
                    "is_done": True,
                    "synthesis": artifacts["decision"].synthesis if hasattr(artifacts["decision"], "synthesis") else "",
                    "gap_analysis": getattr(artifacts["decision"], "gap_analysis", None),
                    "refined_query": getattr(artifacts["decision"], "refined_query", None),
                    "final_report": final_report,
                    "results": artifacts["results"],
                    "artifacts": artifacts,
                }


if __name__ == "__main__":
    # Configure logging outside library modules to avoid side effects
    try:
        from logging_config import configure_logging
        configure_logging()
    except Exception:
        pass
    
    logger.info("🤖 Initializing LeadAgent...")
    agent = LeadAgent()
    logger.info("✅ Agent initialized successfully!")
    
    # Run a single-cycle state machine directly for tracing
    query = "Doue vs Yamal who's better? Be objective"
    logger.info(f"🎬 STARTING SINGLE CYCLE | Query: {query}")
    state_result = asyncio.run(agent.aforward(query))

    logger.info("\n" + "="*80)
    logger.info("📋 STATE MACHINE RESULT (single cycle):")
    logger.info("="*80)
    logger.info(state_result)
