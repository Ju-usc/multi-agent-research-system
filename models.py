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
    objective: str = Field(description="Crisp single-focus goal the subagent must accomplish")
    tool_guidance: Dict[Literal["web_search", "filesystem_read", "filesystem_tree"], str] = Field(description="Mapping of allowed tool names to concise usage instructions")
    tool_budget: int = Field(default=8, ge=3, le=15, description="Maximum number of tool calls the subagent may issue")
    expected_output: str = Field(description="Exact artifact or information the subagent must return for completion")
    tip: Optional[str] = Field(default=None, description="Optional hint to improve quality or efficiency while executing the task")


class SubagentResult(BaseModel):
    """Structured report a subagent returns after finishing its task."""
    task_name: str = Field(description="Task directory name that produced this result")
    summary: str = Field(description="High-density 2-4 sentence overview of the key findings")
    finding: str = Field(description="Full detailed answer directly addressing the task objective")
    debug_info: Optional[List[str]] = Field(default=None, description="Optional list of raw tool call traces for debugging")


# ---------- DSPy Signatures ----------    


class PlanResearch(dspy.Signature):
    """Generate strategic reasoning and a parallel task list for subagents from the user's query.
    
    IMPORTANT: Use consistent formatting for ALL field headers with ## on BOTH sides:
    [[ ## next_thought ## ]]
    [[ ## next_tool_name ## ]]  
    [[ ## next_tool_args ## ]]
    [[ ## plan_filename ## ]]
    [[ ## reasoning ## ]]
    [[ ## tasks ## ]]
    """
    query: str = dspy.InputField(desc="User's research question")
    memory_tree: str = dspy.InputField(desc="Current filesystem structure showing available research data")
    plan_filename: str = dspy.OutputField(desc="Filesystem-friendly filename for this plan (max 50 chars, e.g., 'compare-llm-frameworks')")
    reasoning: str = dspy.OutputField(desc="Strategic analysis of query decomposition approach")
    tasks: List[SubagentTask] = dspy.OutputField(desc="List of 3-5 parallel tasks with unique task_names for subagents") 


class ExecuteSubagentTask(dspy.Signature):
    """Execute a focused micro-task using permitted tools and return structured findings.
    
    IMPORTANT: Use consistent formatting for ALL field headers with ## on BOTH sides:
    [[ ## next_thought ## ]]
    [[ ## next_tool_name ## ]]  
    [[ ## next_tool_args ## ]]
    [[ ## final_result ## ]]
    """
    task: SubagentTask = dspy.InputField(desc="The atomic task this subagent must complete")
    final_result: SubagentResult = dspy.OutputField(desc="Structured output summarizing task completion")


class SynthesizeAndDecide(dspy.Signature):
    """Synthesize all findings from the current cycle and decide whether to continue iteration."""
    query: str = dspy.InputField(desc="Original user query being addressed")
    memory_tree: str = dspy.InputField(desc="Current filesystem structure showing all research data")
    completed_results: List[SubagentResult] = dspy.InputField(desc="Results from this cycle's subagents")
    synthesis: str = dspy.OutputField(desc="Comprehensive analysis integrating all findings so far")
    is_done: bool = dspy.OutputField(desc="True if we have sufficient information to answer the query")
    gap_analysis: Optional[str] = dspy.OutputField(desc="If not done, what critical information gaps remain")
    refined_query: Optional[str] = dspy.OutputField(desc="If not done, focused query for the next iteration")


class FinalReport(dspy.Signature):
    """Generate a well-structured final research report from all memory artifacts.
    
    IMPORTANT: Use consistent formatting for ALL field headers with ## on BOTH sides:
    [[ ## next_thought ## ]]
    [[ ## next_tool_name ## ]]
    [[ ## next_tool_args ## ]]
    [[ ## synthesis ## ]]
    [[ ## report ## ]]
    """
    query: str = dspy.InputField(desc="The original user query")
    final_synthesis: str = dspy.InputField(desc="The final synthesis from the lead agent")
    memory_tree: str = dspy.InputField(desc="Filesystem structure showing all research artifacts")
    report: str = dspy.OutputField(desc="Complete research report in markdown format")


# ---------- FileSystem ----------

class FileSystem:
    """File system for research memory."""
    
    def __init__(self, root: str = "memory"):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)
    
    def write(self, path: str, content: str) -> Path:
        """Write content to path."""
        file_path = self.root / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
    
    def read(self, path: str) -> str:
        """Read content from path."""
        file_path = self.root / path
        if not file_path.exists():
            return f"[ERROR] File not found: {path}"
        return file_path.read_text()
    
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        return (self.root / path).exists()
    
    def tree(self, max_depth: Optional[int] = 3) -> str:
        """Get file structure listing."""
        paths = []
        self._collect_paths(self.root, "", paths, max_depth, 0)
        
        if not paths:
            return "memory/ (empty)"
        
        return "\n".join(["memory/"] + sorted(paths))
    
    def _collect_paths(self, path: Path, relative_path: str, paths: list,
                      max_depth: Optional[int], current_depth: int) -> None:
        """Collect paths recursively."""
        if max_depth and current_depth >= max_depth:
            return
        
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for item in items:
            item_path = f"{relative_path}{item.name}" if relative_path else item.name
            
            if item.is_dir():
                paths.append(f"{item_path}/")
                self._collect_paths(item, f"{item_path}/", paths, max_depth, current_depth + 1)
            else:
                paths.append(item_path)
    
    def clear(self) -> None:
        """Clear all files."""
        import shutil
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(exist_ok=True)