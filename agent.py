"""
Multi-Agent Research System - Async Implementation
Implements Anthropic's orchestrator-worker architecture with async DSPy
"""

import asyncio
from typing import List, Dict, Optional, Any
import dspy
from dspy.adapters import JSONAdapter, TwoStepAdapter
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
    WebSearchTool, ParallelSearchTool, FileSystemTool
)
from utils import setup_langfuse, prediction_to_json
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
        self.steps_trace: List[Dict[str, Any]] = []  # High-level trace of steps
        self.tool_logs: Dict[str, List[str]] = {}  # Store tool logs per task

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
                desc="Read file from research memory"
            ),
            "filesystem_tree": dspy.Tool(
                self.fs_tool.tree,
                name="filesystem_tree",
                desc="Show filesystem structure to see available research data"
            )
        }

        # Initialize language models
        self.init_language_models()

        # Create modules without context - we'll set context during execution
        planning_tools = [self.tools["web_search"], self.tools["filesystem_read"], self.tools["filesystem_tree"]]
        # Create ReAct with JSONAdapter for better parsing reliability
        self.planner = dspy.ReAct(PlanResearch, tools=planning_tools, max_iters=3)
        self.synthesizer = dspy.ChainOfThought(SynthesizeAndDecide)
    
    def init_language_models(self):
        """Initialize all language models in one place."""
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

        self.final_report_lm = dspy.LM(
            model=BIG_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=BIG_MODEL_MAX_TOKENS,
        )


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
                sub.adapter = JSONAdapter()  # Use JSON adapter for better parsing
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

    @observe(name="execute_parallel_tasks", capture_input=True, capture_output=True)
    async def execute_tasks_parallel(self, tasks: List[SubagentTask]) -> List[SubagentResult]:
        """Execute multiple subagent tasks in parallel and collect results.
        
        Args:
            tasks: List of SubagentTask objects to execute
            
        Returns:
            List of valid SubagentResult objects (excludes failures)
        """
        logger.info(f"🚀 Launching {len(tasks)} subagents in parallel...")
        
        # Create task calls
        task_calls = []
        for task in tasks:
            task_calls.append(self.execute_subagent_task(task, self.subagent_lm))
        
        # Execute all tasks in parallel with error handling
        raw_results = await asyncio.gather(*task_calls, return_exceptions=True)
        logger.info(f"📊 Processing {len(raw_results)} subagent results...")
        
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
                # Write result to filesystem
                result_path = f"cycle_{self.cycle_idx:03d}/{r.task_name}/result.md"
                self.fs.write(result_path, str(r))
                logger.info(f"✅ Subagent {i} completed - Task: {r.task_name}")
                logger.debug(f"Summary: {r.summary[:100]}...")
            else:
                logger.warning(f"Subagent {i} returned invalid result")
        
        logger.info(f"📈 Successfully collected {len(results)} valid results")
        return results

    @observe(name="plan_research", capture_input=True, capture_output=True)
    async def plan_research(self, query: str) -> dspy.Prediction:
        """Execute planning phase and return plan.
        
        Args:
            query: Research query to plan for
            
        Returns:
            PlanResult prediction containing tasks and reasoning
        """
        logger.info(f"🔍 CYCLE {self.cycle_idx + 1}: Starting planning phase...")
        logger.debug(f"Query: {query}")
        
        # Get current filesystem structure
        memory_tree = self.fs.tree()
        logger.debug(f"Current memory structure:\n{memory_tree}")
        
        with dspy.context(lm=self.planner_lm):
            plan = await self.planner.acall(query=query, memory_tree=memory_tree)
        
        logger.info(f"✅ Plan generated with {len(plan.tasks)} tasks")
        logger.debug(f"Reasoning: {plan.reasoning[:150]}...")
        
        for i, task in enumerate(plan.tasks):
            logger.debug(f"Task {i}: {task.objective[:100]}...")
        
        # Write plan to filesystem
        plan_path = f"cycle_{self.cycle_idx:03d}/{plan.plan_filename}.md"
        self.fs.write(plan_path, str(plan))
        self.steps_trace.append({
            'cycle': self.cycle_idx,
            'action': 'Planned',
            'summary': f"Generated {len(plan.tasks)} tasks",
            'path': plan_path
        })
        
        return plan
    
    @observe(name="synthesize", capture_input=True, capture_output=True)
    async def synthesize_results(self, query: str, results: List[SubagentResult]) -> dspy.Prediction:
        """Execute synthesis phase and return decision.
        
        Args:
            query: Original research query
            results: List of completed subagent results
            
        Returns:
            SynthesisResult prediction containing decision and synthesis
        """
        logger.info(f"🧠 Starting synthesis phase...")
        logger.debug(f"Synthesizing {len(results)} results")
        
        # Get updated filesystem structure
        memory_tree = self.fs.tree()
        
        with dspy.context(lm=self.synthesizer_lm):
            decision = await self.synthesizer.acall(query=query, memory_tree=memory_tree, completed_results=results)
        
        logger.info(f"✅ Synthesis completed - Decision: {'DONE' if decision.is_done else 'CONTINUE'}")
        logger.debug(f"Synthesis: {decision.synthesis[:150]}...")
        
        if not decision.is_done:
            logger.info(f"🔍 Gap analysis: {decision.gap_analysis[:100]}...")
            logger.info(f"🔄 Refined query: {decision.refined_query[:100]}...")
        
        # Write synthesis to filesystem
        synthesis_path = f"cycle_{self.cycle_idx:03d}/synthesis.md"
        self.fs.write(synthesis_path, str(decision))
        self.steps_trace.append({
            'cycle': self.cycle_idx,
            'action': 'Synthesized',
            'summary': f"Decision: {'DONE' if decision.is_done else 'CONTINUE'}, {len(results)} results",
            'path': synthesis_path
        })
        
        return decision

    @observe(name="generate_final_report", capture_input=True, capture_output=True)
    async def generate_final_report(self, query: str, final_synthesis: str) -> str:
        """Generate final markdown report from memory.
        
        Args:
            query: Original research query
            final_synthesis: Final synthesis from last cycle
            
        Returns:
            Markdown formatted final report
        """
        logger.info(f"📄 GENERATING FINAL REPORT...")
        logger.debug(f"Steps taken: {len(self.steps_trace)}")
        
        # Get full filesystem structure
        memory_tree = self.fs.tree(max_depth=None)
        
        final_react = dspy.ReAct(FinalReport, tools=[self.tools["filesystem_read"], self.tools["filesystem_tree"]], max_iters=5)
        final_react.adapter = JSONAdapter()  # Use JSONAdapter for compatibility
        
        with dspy.context(lm=self.final_report_lm):
            final_result = await final_react.acall(
                query=query,
                memory_tree=memory_tree,
                final_synthesis=final_synthesis,
                steps_trace=self.steps_trace
            )
        
        logger.info(f"✅ Final report generated!")
        logger.debug(f"Report length: {len(final_result.report)} characters")
        
        # Write final report to filesystem
        self.fs.write("final_report.md", final_result.report)
        
        return final_result.report

    @observe(name="lead_agent_main", capture_input=True, capture_output=True)
    async def aforward(self, query: str):
        """Plan → execute tasks in parallel → synthesize/decide (single cycle)."""
        
        # Increment cycle counter
        self.cycle_idx += 1

        # 1. Generate plan
        plan = await self.plan_research(query)

        # 2. Launch subagents in parallel
        results = await self.execute_tasks_parallel(plan.tasks)

        # 3. Synthesize and decide
        decision = await self.synthesize_results(query, results)
        
        return {
            "is_done": decision.is_done,
            "synthesis": decision.synthesis,
            "gap_analysis": decision.gap_analysis,
            "refined_query": decision.refined_query,
            "results": results,
        }

    async def run(self, query: str):
        """Runs a single minimal-cycle research and returns the best available answer."""
        logger.info(f"🎬 STARTING RESEARCH SESSION")
        logger.info(f"🎯 Original query: {query}")
        logger.info("=" * 80)
        
        result = None
        cycle_count = 0
        
        while result is None or not result["is_done"]:
            current_query = query
            if result and "refined_query" in result and result["refined_query"]:
                current_query = result["refined_query"]  # Use refined if provided
                logger.info(f"🔄 REFINED QUERY (Cycle {cycle_count}): {current_query}")
            
            result = await self.aforward(current_query)
            cycle_count += 1
            logger.info(f"📊 CYCLE {cycle_count} COMPLETE - Decision: {'DONE ✅' if result['is_done'] else 'CONTINUE 🔄'}")
            
            
        # When done, use ReAct to distill into report
        if result["is_done"]:
            final_report = await self.generate_final_report(query, result["synthesis"])
            logger.info("🎉 RESEARCH SESSION COMPLETE!")
            
            # Log final memory structure
            logger.info("\n📁 Final memory structure:")
            logger.info(self.fs.tree())
            
            logger.info("=" * 80)
            return final_report
            
        logger.warning("Research incomplete, returning synthesis")
        return result["synthesis"]  # Fallback
    


if __name__ == "__main__":
    logger.info("🤖 Initializing LeadAgent...")
    agent = LeadAgent()
    logger.info("✅ Agent initialized successfully!")
    
    result = asyncio.run(agent.run("Doue vs Yamal who's better? Be objective"))
    
    logger.info("\n" + "="*80)
    logger.info("📋 FINAL RESULT:")
    logger.info("="*80)
    logger.info(result)