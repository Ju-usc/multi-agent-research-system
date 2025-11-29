import logging

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
    """Minimal single-loop agent contract."""

    query: str = dspy.InputField(desc="User query or research request")
    answer: str = dspy.OutputField(desc="answer to the user's query")


class Agent(dspy.Module):
    def __init__(
        self,
        *,
        big_model: str = BIG_MODEL,
        small_model: str = SMALL_MODEL,
        temperature: float = TEMPERATURE,
        big_max_tokens: int = BIG_MODEL_MAX_TOKENS,
        small_max_tokens: int = SMALL_MODEL_MAX_TOKENS,
        work_dir: str | None = None,
    ) -> None:
        super().__init__()
        # Shared adapter for structured outputs
        # Core tools
        self.web_search_tool = WebSearchTool()
        
        # Use isolated work directory or default to shared "memory"
        # Enables parallel evaluation without filesystem conflicts
        if work_dir is None:
            work_dir = "memory"  # Backward compatible default
        self.fs_tool = FileSystemTool(root=work_dir)
        self.todo_list_tool = TodoListTool()
        self.fs = self.fs_tool  # provide backward-compatible alias

        # Lead / subagent language models
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
                desc="Search the web for supporting information (<=5 results).",
            ),
            "filesystem_write": dspy.Tool(
                self.fs_tool.write,
                name="filesystem_write",
                desc="Write content to path relative to workspace root. Use simple relative paths like 'results/data.json', NOT 'memory/results/data.json'.",
            ),
        }

        subagent_parallel_tool = ParallelToolCall(self.subagent_tools, num_threads=4)
        self.subagent_tools["parallel_tool_call"] = dspy.Tool(
            subagent_parallel_tool,
            name="parallel_tool_call",
            desc="Run multiple subagent tools in parallel with a single call.",
        )

        # Subagent execution tool 
        self.subagent_tool = SubagentTool(
            tools=list(self.subagent_tools.values()),
            lm=self.subagent_lm,
            adapter=ChatAdapter(),
        )

        # Lead tool registry (populate base tools first)
        self.lead_agent_tools = {
            "filesystem_read": dspy.Tool(
                self.fs_tool.read,
                name="filesystem_read",
                desc="Read artifacts using paths relative to workspace root (e.g., 'results/data.json').",
            ),
            "filesystem_tree": dspy.Tool(
                self.fs_tool.tree,
                name="filesystem_tree",
                desc="List workspace tree to see available artifacts. Returns paths relative to workspace root.",
            ),
            "todo_list_read": dspy.Tool(
                self.todo_list_tool.read,
                name="todo_list_read",
                desc="Fetch the current status of the To-Do list. Useful when you need to see what you have planned.",
            ),
            "todo_list_write": dspy.Tool(
                self.todo_list_tool.write,
                name="todo_list_write",
                desc="Write the To-Do list. Useful when you need to plan something. You should always try to update the To-Do list status.",
            ),
            "subagent_run": dspy.Tool(
                self.subagent_tool,
                name="subagent_run",
                desc="Execute a single subagent research task. Returns JSON with summary, detail, and artifact_path. For parallel execution of multiple subagents, use parallel_tool_call.",
            ),
        }
        
        # Add parallel_tool_call for lead agent to enable parallel subagent execution
        lead_parallel_tool = ParallelToolCall(self.lead_agent_tools, num_threads=4)
        self.lead_agent_tools["parallel_tool_call"] = dspy.Tool(
            lead_parallel_tool,
            name="parallel_tool_call",
            desc="Run multiple lead agent tools in parallel. Useful for spawning multiple subagents concurrently.",
        )



        # Single-loop agent program
        self.lead_agent = dspy.ReAct(
            AgentSignature,
            tools=list(self.lead_agent_tools.values()),
        )

    def forward(self, query: str) -> dspy.Prediction:
        return self.lead_agent(query=query)

def parse_args():
    parser = create_model_cli_parser(
        "Run the single-loop research agent.",
        query=(
            "Lamine vs Doue? Be objective and keep it research short and concise DO NOT ASK ANYTHING ELSE. Try to use tools provided to you to test our system first. Yet there will be no memory artifacts as this is the first session.",
            "Query to run through the agent.",
        ),
    )
    return parser.parse_args()


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
