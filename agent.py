import argparse
import logging

import dspy
from dspy.adapters.chat_adapter import ChatAdapter

from config import (
    MODEL_PRESETS,
    BIG_MODEL,
    SMALL_MODEL,
    BIG_MODEL_MAX_TOKENS,
    SMALL_MODEL_MAX_TOKENS,
    TEMPERATURE,
    resolve_model_config,
    lm_kwargs_for,
)
from tools import WebSearchTool, FileSystemTool, TodoListTool, SubagentTool, ParallelToolCall
from logging_config import trace_call, configure_logging
from utils import setup_langfuse
from langfuse import observe


# Initialize Langfuse tracing once per process (no-op if disabled)
langfuse = setup_langfuse()

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
    ) -> None:
        super().__init__()
        # Shared adapter for structured outputs
        # Core tools
        self.web_search_tool = WebSearchTool()
        self.fs_tool = FileSystemTool()
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
                desc="Write content to specific path in the filesystem. Drop the leading 'memory/' prefix in paths.",
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
                desc="Read artifacts from subagents under memory/. Drop the leading 'memory/' prefix in paths.",
            ),
            "filesystem_tree": dspy.Tool(
                self.fs_tool.tree,
                name="filesystem_tree",
                desc="List the current memory tree to see available artifacts from subagents.",
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
            "subagent_parallel_run": dspy.Tool(
                self.subagent_tool.parallel_run,
                name="subagent_parallel_run",
                desc="Kick off several subagents at once; each runs web search and writes back findings.",
            ),
        }

        lead_parallel_tool = ParallelToolCall(self.lead_agent_tools, num_threads=4)
        self.lead_agent_tools["parallel_tool_call"] = dspy.Tool(
            lead_parallel_tool,
            name="parallel_tool_call",
            desc="Run multiple lead tools in parallel with a single call.",
        )



        # Single-loop agent program
        self.lead_agent = dspy.ReAct(
            AgentSignature,
            tools=list(self.lead_agent_tools.values()),
            max_iters=3,
        )

    @trace_call("agent.forward")
    @observe(name="lead_agent", capture_input=True, capture_output=True)
    def forward(self, query: str) -> dspy.Prediction:
        return self.lead_agent(query=query)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-loop research agent.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Model preset to use for both big and small slots.",
    )
    parser.add_argument("--model-big", dest="model_big", help="Override the big model identifier.")
    parser.add_argument("--model-small", dest="model_small", help="Override the small model identifier.")
    parser.add_argument(
        "--query",
        default="Lamine vs Doue? Be objective and keep it research short and concise DO NOT ASK ANYTHING ELSE. Try to use tools provided to you to test our system first. Yet there will be no memory artifacts as this is the first session.",
        help="Query to run through the agent.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()

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
