"""Data models and DSPy signatures."""

from typing import Optional, Literal
from pydantic import BaseModel, Field
import dspy


class SubagentTask(BaseModel):
    """Atomic research task for a subagent."""
    task_name: str = Field(
        description="Filesystem-friendly directory name",
        max_length=50,
        pattern="^[a-z0-9-]+$"
    )
    prompt: str = Field(description="Prompt for the subagent", exclude=True)
    description: str = Field(description="Description of the task")
    tool_budget: int = Field(default=3, ge=1, le=15, description="Max tool calls")
    expected_output: Optional[str] = Field(default=None, description="Expected artifact")
    tip: Optional[str] = Field(default=None, description="Hint for quality/efficiency")


class SubagentResult(BaseModel):
    """Subagent output."""
    task_name: str = Field(default="", description="Task name", exclude=True)
    summary: str = Field(description="2-4 sentence overview of findings")
    detail: Optional[str] = Field(default=None, description="Supplemental detail")
    artifact_path: Optional[str] = Field(default=None, description="Path relative to workspace root")


class Todo(BaseModel):
    """Todo list item."""
    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"]
    priority: Literal["low", "medium", "high"]


class ExecuteSubagentTask(dspy.Signature):
    """Execute a micro-task and return findings."""
    task: SubagentTask = dspy.InputField(desc="The task to complete")
    final_result: SubagentResult = dspy.OutputField(desc="Structured output")
