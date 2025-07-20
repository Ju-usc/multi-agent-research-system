"""
Data models and DSPy signatures for the multi-agent research system.
"""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
import dspy
from config import SMALL_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, SMALL_MODEL_MAX_TOKENS


# ---------- Data Models ----------

class SubagentTask(BaseModel):
    """Atomic research micro-task allocated to one sub-agent in the plan."""
    task_id: int = Field(description="Identifier used to track this task across iterations")
    objective: str = Field(description="Crisp single-focus goal the subagent must accomplish")
    tool_guidance: Dict[Literal["web_search", "memory_read", "memory_list"], str] = Field(description="Mapping of allowed tool names to concise usage instructions")
    tool_budget: int = Field(default=8, ge=3, le=15, description="Maximum number of tool calls the subagent may issue")
    expected_output: str = Field(description="Exact artifact or information the subagent must return for completion")
    tip: Optional[str] = Field(default=None, description="Optional hint to improve quality or efficiency while executing the task")


class SubagentResult(BaseModel):
    """Structured report a subagent returns after finishing its task."""
    task_id: int = Field(description="Identifier of the task that produced this result")
    summary: str = Field(description="High-density 2-4 sentence overview of the key findings")
    finding: str = Field(description="Full detailed answer directly addressing the task objective")
    debug_info: Optional[List[str]] = Field(default=None, description="Optional list of raw tool call traces for debugging")


class Memory(BaseModel):
    """In-memory store for research artifacts with lightweight summaries."""
    store: Dict[str, str] = Field(default_factory=dict, description="Full JSON/text storage")
    summaries: Dict[str, str] = Field(default_factory=dict, description="Condensed index cards to reference the memory storage")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._summarizer_lm = dspy.LM(
            model=SMALL_MODEL,
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_BASE_URL,
            temperature=0.3,
            max_tokens=SMALL_MODEL_MAX_TOKENS,
        )
        self._summarizer = dspy.Predict(MemorySummary)

    async def summarize(self, text: str) -> str:
        """Create a concise summary for the index card."""
        with dspy.settings.context(lm=self._summarizer_lm):
            result = await self._summarizer.acall(raw_context=text)
            return result.summary
        
    async def write(self, cycle: int, type: str, content: str, task_id: Optional[int] = None) -> str:
        """Write content to memory and create summary index card.
        
        Args:
            cycle: Current cycle index
            type: Stage type ('plan', 'task', 'synthesis')
            content: Full content to store (typically JSON)
            task_id: Optional task ID for subagent results
            
        Returns:
            Key used to store the content
        """
        # Build deterministic key
        key = f"{cycle}-{type}" if task_id is None else f"{cycle}-{type}-{task_id}"
        
        # Store full content
        self.store[key] = content
        self.summaries[key] = await self.summarize(content)
        return key
    
    def read(self, key: str) -> str:
        """Retrieve full content by key.
        
        Args:
            key: Memory key (e.g., '1-plan', '2-task-3')
            
        Returns:
            Full stored content or error message if not found
        """
        return self.store.get(key, f"[ERROR] Key not found: {key}")
    
    def list_keys(self) -> str:
        """List all available memory keys for debugging."""
        if not self.store:
            return "No memory records available yet."
        
        keys = sorted(self.store.keys())
        return f"Available memory keys: {', '.join(keys)}"


# ---------- DSPy Signatures ----------

class MemorySummary(dspy.Signature):
    """Summarize the raw context in a way to easily retrieve and distil the most important information"""
    raw_context: str = dspy.InputField(desc="The raw context to summarize")
    summary: str = dspy.OutputField(desc="The summary of the raw context")    


class PlanResearch(dspy.Signature):
    """Generate strategic reasoning and a parallel task list for subagents from the user's query.
    
    IMPORTANT: Use consistent formatting for ALL field headers with ## on BOTH sides:
    [[ ## next_thought ## ]]
    [[ ## next_tool_name ## ]]  
    [[ ## next_tool_args ## ]]
    [[ ## reasoning ## ]]
    [[ ## tasks ## ]]
    """
    query: str = dspy.InputField(desc="User's research question")
    memory_summaries: Dict[str, str] = dspy.InputField(desc="Summaries of previously stored memory artifacts")
    reasoning: str = dspy.OutputField(desc="Strategic analysis of query decomposition approach")
    tasks: List[SubagentTask] = dspy.OutputField(desc="List of 3-5 parallel tasks for subagents") 


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
    memory_summaries: Dict[str, str] = dspy.InputField(desc="Summaries of all memory artifacts")
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
    memory_summaries: Dict[str, str] = dspy.InputField(desc="All memory summaries for reference")
    report: str = dspy.OutputField(desc="Complete research report in markdown format")