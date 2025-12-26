"""Data models and DSPy signatures."""

from typing import Optional, Literal
from pydantic import BaseModel, Field
import dspy


class ToolResponse(BaseModel):
    """Unified response format for all tools."""
    isError: bool
    message: str

    def __str__(self) -> str:
        return self.model_dump_json()


class SubagentTask(BaseModel):
    """Atomic research task for a subagent."""
    name: str = Field(description="Task identifier for matching results", max_length=50)
    # exclude=True: prompt is injected into signature.instructions, not serialized to LLM
    prompt: str = Field(description="Prompt for the subagent", exclude=True)
    description: str = Field(description="Description of the task")
    tool_budget: int = Field(default=3, ge=1, le=15, description="Max tool calls")


class SubagentResult(BaseModel):
    """Subagent output."""
    name: str = Field(default="", description="Task identifier for matching parallel results")
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


class LLMJudgeAnswer(BaseModel):
    """Answer from LLM judge on prediction correctness."""
    is_correct: bool
    extracted_answer: str
    reasoning: str


class BrowseCompJudge(dspy.Signature):
    """Judge whether the research report correctly answers the question.
    
    Focus only on whether the report contains the correct answer, not on quality of reasoning.
    Allow small variations in wording or format.
    Answer False if answer is missing, incorrect, or significantly different.
    """
    question: str = dspy.InputField()
    report: str = dspy.InputField()
    correct_answer: str = dspy.InputField()
    
    answer: LLMJudgeAnswer = dspy.OutputField()
