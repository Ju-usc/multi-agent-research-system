import logging
from pathlib import Path

import dspy
from dspy.adapters.chat_adapter import ChatAdapter

from config import ModelConfig, lm_kwargs_for
from tools import WebSearchTool, FileSystemTool, TodoListTool, SubagentTool, ParallelToolCall
from tracer import trace
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
        config: ModelConfig | None = None,
        work_dir: Path | str | None = None,
    ) -> None:
        super().__init__()
        config = config or ModelConfig()

        # Core tools
        self.web_search_tool = WebSearchTool()

        # Use isolated work directory or default to shared "memory"
        if work_dir is None:
            work_dir = Path("memory")
        elif isinstance(work_dir, str):
            work_dir = Path(work_dir)
        self.fs_tool = FileSystemTool(root=work_dir)
        self.todo_list_tool = TodoListTool()

        # Lead / subagent language models
        lead_kwargs = lm_kwargs_for(config.lead)
        sub_kwargs = lm_kwargs_for(config.sub)

        self.agent_lm = dspy.LM(
            model=config.lead,
            temperature=config.temperature,
            max_tokens=config.lead_max_tokens,
            **lead_kwargs,
        )
        self.subagent_lm = dspy.LM(
            model=config.sub,
            temperature=config.temperature,
            max_tokens=config.sub_max_tokens,
            **sub_kwargs,
        )


        subagent_tools = [
            dspy.Tool(
                self.web_search_tool,
                name="web_search",
                desc="Search the web. Returns JSON: {isError: bool, message: str} with search results in message.",
            ),
            dspy.Tool(
                self.fs_tool.write,
                name="filesystem_write",
                desc="Write content to path. Returns JSON: {isError: bool, message: str}. Use relative paths like 'results/data.json'.",
            ),
        ]

        self.subagent_tool = SubagentTool(
            tools=subagent_tools,
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
        
        lead_parallel_tool = ParallelToolCall(self.lead_agent_tools)
        self.lead_agent_tools["parallel_tool_call"] = dspy.Tool(
            lead_parallel_tool,
            name="parallel_tool_call",
            desc="Run multiple lead agent tools in parallel. Useful for spawning multiple subagents concurrently.",
        )

        self.lead_agent = dspy.ReAct(
            AgentSignature,
            tools=list(self.lead_agent_tools.values()),
        )
        self.lead_agent.lm = self.agent_lm
        self.lead_agent.adapter = ChatAdapter()

    @trace
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
        self.todo_list_tool.clear()


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

    config = ModelConfig(lead=args.lead, sub=args.sub)
    logger.info("Models | lead=%s sub=%s", config.lead, config.sub)

    agent = Agent(config=config)
    result = agent(query=args.query)
    if logger.isEnabledFor(logging.DEBUG):
        dspy.inspect_history(n=10)

    print(result.answer)

if __name__ == "__main__":
    main()
