import logging
from pathlib import Path

import dspy
from dspy.adapters.chat_adapter import ChatAdapter

import config as cfg
from config import ModelConfig
from logger import AgentLogger, AgentLoggingCallback
from tools import WebSearchTool, WebFetchTool, FileSystemTool, TodoListTool, SubagentTool, ParallelToolCall
from tracer import trace
from models import AgentSignature
from utils import create_model_cli_parser

logger = logging.getLogger(__name__)

class Agent(dspy.Module):
    def __init__(
        self,
        *,
        config: ModelConfig = ModelConfig(),
        work_dir: Path = Path("memory"),
        log_dir: str,
    ) -> None:
        super().__init__()
        self.log_dir = log_dir

        self.fs_tool = FileSystemTool(root=work_dir)
        self.todo_list_tool = TodoListTool()

        if config.offline:
            from browsecompplus import LocalSearchTool, LocalFetchTool
            self.search_tool = LocalSearchTool()
            self.fetch_tool = LocalFetchTool()
        else:
            self.search_tool = WebSearchTool()
            self.fetch_tool = WebFetchTool()

        self.leadagent_lm = dspy.LM(
            model=config.lead,
            temperature=config.temperature,
            max_tokens=config.lead_max_tokens,
            api_key=cfg.OPENROUTER_API_KEY,
        )
        self.subagent_lm = dspy.LM(
            model=config.sub,
            temperature=config.temperature,
            max_tokens=config.sub_max_tokens,
            api_key=cfg.OPENROUTER_API_KEY,
        )

        subagent_tools = [
            dspy.Tool(
                self.search_tool,
                name="search",
                desc="Search for information.",
            ),
            dspy.Tool(
                self.fetch_tool,
                name="fetch",
                desc="Fetch document content.",
            ),
            dspy.Tool(
                self.fs_tool.write,
                name="filesystem_write",
                desc="Write content to path.",
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
                desc="Read file content.",
            ),
            "filesystem_tree": dspy.Tool(
                self.fs_tool.tree,
                name="filesystem_tree",
                desc="List workspace tree.",
            ),
            "todo_list_read": dspy.Tool(
                self.todo_list_tool.read,
                name="todo_list_read",
                desc="Read To-Do list.",
            ),
            "todo_list_write": dspy.Tool(
                self.todo_list_tool.write,
                name="todo_list_write",
                desc="Write To-Do list.",
            ),
            "subagent_run": dspy.Tool(
                self.subagent_tool,
                name="subagent_run",
                desc="Execute a subagent task.",
            ),
        }
        
        lead_parallel_tool = ParallelToolCall(self.lead_agent_tools)
        self.lead_agent_tools["parallel_tool_call"] = dspy.Tool(
            lead_parallel_tool,
            name="parallel_tool_call",
            desc="Run multiple tools in parallel.",
        )

        self.lead_agent = dspy.ReAct(
            AgentSignature,
            tools=list(self.lead_agent_tools.values()),
        )
        self.lead_agent.lm = self.leadagent_lm
        self.lead_agent.adapter = ChatAdapter()

    @trace
    def forward(self, query: str) -> dspy.Prediction:
        agent_logger = AgentLogger(
            log_dir=self.log_dir,
            lead_model=self.leadagent_lm.model,
            sub_model=self.subagent_lm.model,
            query=query,
        )
        with dspy.context(
            callbacks=[AgentLoggingCallback(agent_logger)],
            agent_type="lead",
            agent_name="lead_agent",
        ):
            return self.lead_agent(query=query)

    def reset_workspace(self, work_dir: Path) -> None:
        """Reset agent state for a new evaluation run.

        Args:
            work_dir: New workspace directory (will be created if needed)
        """
        self.fs_tool.root = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        self.search_tool.total_cost_usd = 0.0
        self.fetch_tool.total_cost_usd = 0.0
        self.todo_list_tool.clear()


def parse_args():
    parser = create_model_cli_parser(
        "Run the single-loop research agent.",
        query=(
            None,  # Required - no default
            "Research query to run through the agent.",
        ),
    )
    parser.add_argument("--log-dir", default="logs", help="Directory for agent execution logs (JSONL).")
    args = parser.parse_args()
    if args.query is None:
        parser.error("--query is required")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    model_config = ModelConfig(lead=args.lead, sub=args.sub)
    logger.info("Models | lead=%s sub=%s", model_config.lead, model_config.sub)

    dspy.configure(
        lm=dspy.LM(
            model=model_config.lead,
            temperature=model_config.temperature,
            max_tokens=model_config.lead_max_tokens,
            api_key=cfg.OPENROUTER_API_KEY,
        ),
        adapter=ChatAdapter(),
    )

    agent = Agent(config=model_config, log_dir=args.log_dir)
    result = agent(query=args.query)
    if logger.isEnabledFor(logging.DEBUG):
        dspy.inspect_history(n=10)

    print(result.answer)

if __name__ == "__main__":
    main()
