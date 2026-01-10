from typing import Literal

import dspy
from pydantic import BaseModel, Field


class SubagentTask(BaseModel):
    name: str = Field(description="Name of the task or subagent", max_length=100)
    instructions: str = Field(description="Task instructions", exclude=True)
    max_steps: int = Field(default=3, ge=1, le=15, description="Max steps")
    expected_output: str | None = Field(default=None, description="Expected output format")


class SubagentResult(BaseModel):
    name: str = Field(default="", description="Task name")
    summary: str = Field(description="Summary of findings")
    artifact_path: str | None = Field(default=None, description="Artifact path")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Confidence 0-1")


class Todo(BaseModel):
    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"] = "pending"
    priority: Literal["low", "medium", "high"] = "medium"


class LLMJudgeAnswer(BaseModel):
    extracted_answer: str
    reasoning: str
    is_correct: bool


class AgentSignature(dspy.Signature):
    """Lead agent contract."""
    query: str = dspy.InputField(desc="User query")
    answer: str = dspy.OutputField(desc="Answer")


class ExecuteSubagentTask(dspy.Signature):
    """Execute task and return findings."""
    task: SubagentTask = dspy.InputField(desc="Task to complete")
    final_result: SubagentResult = dspy.OutputField(desc="Result")


class BrowseCompJudge(dspy.Signature):
    """Judge if the report CONTAINS the correct answer.

    is_correct = True if:
    - The correct answer (or its core identifier) appears in the report
    - A reasonable variant appears (abbreviations, full names, suffixes)
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
