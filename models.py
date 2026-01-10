"""Data models and DSPy signatures."""

from typing import Literal
from pydantic import BaseModel, Field
import dspy


class SubagentTask(BaseModel):
    """Atomic research task for a subagent."""
    name: str = Field(description="Name of the task or subagent", max_length=100)
    # exclude=True: instructions are appended to signature.instructions, not serialized to LLM
    instructions: str = Field(description="Task instructions appended to subagent signature", exclude=True)
    max_steps: int = Field(default=3, ge=1, le=15, description="Max steps to complete task")
    expected_output: str | None = Field(default=None, description="Expected output format or artifact when known")

class SubagentResult(BaseModel):
    """Subagent output."""
    name: str = Field(default="", description="Name of the task or subagent")
    summary: str = Field(description="Summary of the findings")
    artifact_path: str | None = Field(default=None, description="Path to the artifact")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Confidence in correctness (0-1)")

class Todo(BaseModel):
    """Todo list item."""
    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"] = "pending"
    priority: Literal["low", "medium", "high"] = "medium"

class LLMJudgeAnswer(BaseModel):
    """Answer from LLM judge on prediction correctness."""
    extracted_answer: str
    reasoning: str
    is_correct: bool

class AgentSignature(dspy.Signature):
    """Lead agent contract."""
    query: str = dspy.InputField(desc="User query or research request")
    answer: str = dspy.OutputField(desc="Answer to the query")

class ExecuteSubagentTask(dspy.Signature):
    """Execute task defined by lead agent and return findings back to lead agent."""
    task: SubagentTask = dspy.InputField(desc="The task to complete")
    final_result: SubagentResult = dspy.OutputField(desc="Final result of the task")

class BrowseCompJudge(dspy.Signature):
    """Judge if the report CONTAINS the correct answer.

    is_correct = True if:
    - The correct answer (or its core identifier) appears in the report
    - A reasonable variant of the answer appears (abbreviations, full names, with/without suffixes like Ltd/Inc)
    - The semantic meaning matches even if wording differs

    is_correct = False ONLY if:
    - The answer is completely absent
    - A different/wrong entity is named
    - The report explicitly states it cannot find the answer
    """
    question: str = dspy.InputField()
    report: str = dspy.InputField()
    correct_answer: str = dspy.InputField()
    answer: LLMJudgeAnswer = dspy.OutputField()