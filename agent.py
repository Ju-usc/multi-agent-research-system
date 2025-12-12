import logging
from pathlib import Path

import dspy
from dspy.adapters.chat_adapter import ChatAdapter

from config import (
    BIG_MODEL,
    SMALL_MODEL,
    BIG_MODEL_MAX_TOKENS,
    SMALL_MODEL_MAX_TOKENS,
    TEMPERATURE,
    resolve_model_config,
    lm_kwargs_for,
)
from tools import WebSearchTool, FileSystemTool, TodoListTool, SubagentTool, ParallelToolCall
from utils import create_model_cli_parser

logger = logging.getLogger(__name__)


class AgentSignature(dspy.Signature):
    """Lead agent contract."""
    query: str = dspy.InputField(desc="User query or research request")
    answer: str = dspy.OutputField(desc="Answer to the query")


class Agent(dspy.Module):
    def __init__(
        self,
        *,
        big_model: str = BIG_MODEL,
        small_model: str = SMALL_MODEL,
        temperature: float = TEMPERATURE,
        big_max_tokens: int = BIG_MODEL_MAX_TOKENS,
        small_max_tokens: int = SMALL_MODEL_MAX_TOKENS,
        work_dir: Path | str | None = None,
    ) -> None:
        super().__init__()
        self.web_search_tool = WebSearchTool()
        if work_dir is None:
            work_dir = Path("memory")
        elif isinstance(work_dir, str):
            work_dir = Path(work_dir)
        self.fs_tool = FileSystemTool(root=work_dir)
        self.todo_list_tool = TodoListTool()

        big_kwargs = lm_kwargs_for(big_model)
        small_kwargs = lm_kwargs_for(small_model)

        self.agent_lm = dspy.LM(
            model=big_model,
            temperature=temperature,
            max_tokens=big_max_tokens,
            **big_kwargs,
        )
        self.subagent_lm = dspy.LM(
            model=small_model,
            temperature=temperature,
            max_tokens=small_max_tokens,
            **small_kwargs,
        )


        self.subagent_tools = {
            "web_search": dspy.Tool(
                self.web_search_tool,
                name="web_search",
                desc="Search the web. Returns JSON: {isError: bool, message: str} with search results in message.",
            ),
            "filesystem_write": dspy.Tool(
                self.fs_tool.write,
                name="filesystem_write",
                desc="Write content to path. Returns JSON: {isError: bool, message: str}. Use relative paths like 'results/data.json'.",
            ),
        }

        subagent_parallel_tool = ParallelToolCall(self.subagent_tools, num_threads=4)
        self.subagent_tools["parallel_tool_call"] = dspy.Tool(
            subagent_parallel_tool,
            name="parallel_tool_call",
            desc="Run multiple subagent tools in parallel with a single call.",
        )

        self.subagent_tool = SubagentTool(
            tools=list(self.subagent_tools.values()),
            lm=self.subagent_lm,
            adapter=ChatAdapter(),
        )

        self.lead_agent_tools = {
            "filesystem_read": dspy.Tool(
                self.fs_tool.read,
                name="filesystem_read",
                desc="Read artifacts. Returns JSON: {isError: bool, message: str} with file content in message.",
            ),
            "filesystem_tree": dspy.Tool(
                self.fs_tool.tree,
                name="filesystem_tree",
                desc="List workspace tree. Returns JSON: {isError: bool, message: str} with directory listing in message.",
            ),
            "todo_list_read": dspy.Tool(
                self.todo_list_tool.read,
                name="todo_list_read",
                desc="Fetch To-Do list. Returns JSON: {isError: bool, message: str} with todos in message.",
            ),
            "todo_list_write": dspy.Tool(
                self.todo_list_tool.write,
                name="todo_list_write",
                desc="Write To-Do list. Returns JSON: {isError: bool, message: str}. Always update status after completing tasks.",
            ),
            "subagent_run": dspy.Tool(
                self.subagent_tool,
                name="subagent_run",
                desc="Execute a subagent task. Returns JSON with summary, detail, artifact_path. Use parallel_tool_call for concurrent execution.",
            ),
        }
        
        lead_parallel_tool = ParallelToolCall(self.lead_agent_tools, num_threads=4)
        self.lead_agent_tools["parallel_tool_call"] = dspy.Tool(
            lead_parallel_tool,
            name="parallel_tool_call",
            desc="Run multiple lead agent tools in parallel. Useful for spawning multiple subagents concurrently.",
        )

        self.lead_agent = dspy.ReAct(
            AgentSignature,
            tools=list(self.lead_agent_tools.values()),
        )

    def forward(self, query: str) -> dspy.Prediction:
        return self.lead_agent(query=query)

    def reset_workspace(self, work_dir: Path) -> None:
        """Reset agent state for a new evaluation run.

        Args:
            work_dir: New workspace directory (will be created if needed)
        """
        self.fs_tool.root = work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        self.web_search_tool.call_count = 0
        self.todo_list_tool._todos = []


def parse_args():
    parser = create_model_cli_parser(
        "Run the single-loop research agent.",
        query=(
            None,  # Required - no default
            "Research query to run through the agent.",
        ),
    )
    args = parser.parse_args()
    if args.query is None:
        parser.error("--query is required")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    try:
        preset = resolve_model_config(args.model, args.model_big, args.model_small)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    logger.info(
        "Selected models | big=%s small=%s | big_max_tokens=%s | small_max_tokens=%s",
        preset.big,
        preset.small,
        preset.big_max_tokens,
        preset.small_max_tokens,
    )

    dspy.configure(
        lm=dspy.LM(
            model=preset.big,
            temperature=TEMPERATURE,
            max_tokens=preset.big_max_tokens,
            **lm_kwargs_for(preset.big),
        ),
        adapter=ChatAdapter(),
    )

    agent = Agent(
        big_model=preset.big,
        small_model=preset.small,
        temperature=TEMPERATURE,
        big_max_tokens=preset.big_max_tokens,
        small_max_tokens=preset.small_max_tokens,
    )
    result = agent(query=args.query)
    dspy.inspect_history(n=5)

    print(result.answer)

if __name__ == "__main__":
    main()
