"""
Data models and DSPy signatures for the multi-agent research system.
"""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
import dspy
from pathlib import Path
from config import SMALL_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, SMALL_MODEL_MAX_TOKENS


# ---------- Data Models ----------

class SubagentTask(BaseModel):
    """Atomic research micro-task allocated to one sub-agent in the plan."""
    task_name: str = Field(
        description="Filesystem-friendly directory name (e.g., 'research-python-async', 'analyze-security-trends')",
        max_length=50,
        pattern="^[a-z0-9-]+$"
    )
    prompt: str = Field(description="Prompt for the subagent to complete the task", exclude=True) # exclude prompt as an input field of subagentResult as it is configured directly via instruction 
    description: str = Field(description="Description of the task")
    tool_budget: int = Field(default=3, ge=1, le=15, description="Maximum number of tool calls the subagent may issue")
    expected_output: str = Field(description="Exact artifact or information the subagent must return for completion")
    tip: Optional[str] = Field(default=None, description="Optional hint to improve quality or efficiency while executing the task")


class SubagentResult(BaseModel):
    """Structured report a subagent returns after finishing its task."""
    # task_name is populated from the originating SubagentTask, not the LLM output
    task_name: str = Field(
        default="",
        description="Task directory name that produced this result",
        exclude=True,
    )
    summary: str = Field(description="High-density 2-4 sentence overview of the key findings")
    detail: Optional[str] = Field(default=None, description="Optional short-form detail that supplements the summary")
    artifact_path: Optional[str] = Field(
        default=None,
        description="Optional path to a filesystem artifact containing the full report when detail does not fit inline",
    )

# ---------- Todo List ----------

class Todo(BaseModel):
    """Todo list item."""
    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"]
    priority: Literal["low", "medium", "high"]

# ---------- DSPy Signatures ----------    


class PlanResearch(dspy.Signature):
    """Generate strategic reasoning and a parallel task list for subagents from the user's query.
    """
    query: str = dspy.InputField(desc="User's research question")
    plan_filename: str = dspy.OutputField(desc="Filesystem-friendly filename for this plan (max 50 chars, e.g., 'compare-llm-frameworks')")
    reasoning: str = dspy.OutputField(desc="Strategic analysis of query decomposition approach")
    tasks: List[SubagentTask] = dspy.OutputField(desc="List of 3-5 parallel tasks with unique task_names for subagents") 


class ExecuteSubagentTask(dspy.Signature):
    """Execute a focused micro-task using permitted tools and return structured findings.
    """
    task: SubagentTask = dspy.InputField(desc="The atomic task this subagent must complete")
    final_result: SubagentResult = dspy.OutputField(desc="Structured output summarizing task completion")


class SynthesizeAndDecide(dspy.Signature):
    """Synthesize all findings from the current cycle and decide whether to continue iteration."""
    query: str = dspy.InputField(desc="Original user query being addressed")
    completed_results: List[SubagentResult] = dspy.InputField(desc="Results from this cycle's subagents")
    synthesis: str = dspy.OutputField(desc="Comprehensive analysis integrating all findings so far")
    is_done: bool = dspy.OutputField(desc="True if we have sufficient information to answer the query")
    gap_analysis: Optional[str] = dspy.OutputField(desc="If not done, what critical information gaps remain")
    refined_query: Optional[str] = dspy.OutputField(desc="If not done, focused query for the next iteration")


class FinalReport(dspy.Signature):
    """Generate a well-structured final research report from all memory artifacts.
    """
    query: str = dspy.InputField(desc="The original user query")
    final_synthesis: str = dspy.InputField(desc="The final synthesis from the lead agent")
    report: str = dspy.OutputField(desc="Complete research report in markdown format")
