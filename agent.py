import dspy
from dspy.adapters.baml_adapter import BAMLAdapter

from config import (
    BRAVE_SEARCH_API_KEY,
    OPENAI_API_KEY,
    SMALL_MODEL,
    BIG_MODEL,
    TEMPERATURE,
    BIG_MODEL_MAX_TOKENS,
    SMALL_MODEL_MAX_TOKENS,
)
from tools import WebSearchTool, FileSystemTool, TodoListTool, SubagentTool
from utils import setup_langfuse
from langfuse import observe


# Initialize Langfuse tracing once per process (no-op if disabled)
langfuse = setup_langfuse()


class AgentSignature(dspy.Signature):
    """Minimal single-loop agent contract."""

    query: str = dspy.InputField(desc="User query or research request")
    answer: str = dspy.OutputField(desc="answer to the user's query")


class Agent(dspy.Module):
    def __init__(self) -> None:
        super().__init__()

        # Shared adapter for structured outputs
        self.baml_adapter = BAMLAdapter()

        # Core tools
        self.web_search_tool = WebSearchTool(BRAVE_SEARCH_API_KEY)
        self.fs_tool = FileSystemTool()
        self.todo_list_tool = TodoListTool()
        self.fs = self.fs_tool  # provide backward-compatible alias

        # Lead / subagent language models
        self.agent_lm = dspy.LM(
            model=BIG_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=BIG_MODEL_MAX_TOKENS,
        )
        self.subagent_lm = dspy.LM(
            model=SMALL_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=SMALL_MODEL_MAX_TOKENS,
        )

        # Lead tool registry
        self.tools = {
            "web_search": dspy.Tool(
                self.web_search_tool,
                name="web_search",
                desc="Search the web for supporting information (<=5 results).",
            ),
            "filesystem_read": dspy.Tool(
                self.fs_tool.read,
                name="filesystem_read",
                desc="Read markdown artifacts under memory/. Drop the leading 'memory/' prefix in paths.",
            ),
            "filesystem_tree": dspy.Tool(
                self.fs_tool.tree,
                name="filesystem_tree",
                desc="List the current research memory tree to see available artifacts.",
            ),
            "todo_list_read": dspy.Tool(
                self.todo_list_tool.read,
                name="todo_list_read",
                desc="Fetch the in-memory To-Do list snapshot.",
            ),
            "todo_list_write": dspy.Tool(
                self.todo_list_tool.write,
                name="todo_list_write",
                desc="Replace the To-Do list with your updated list of items.",
            ),
        }

        # Subagent execution tool (web-search only for now)
        self.subagent_tool = SubagentTool(
            tools=[self.web_search_tool],
            lm=self.subagent_lm,
            adapter=self.baml_adapter,
        )

        self.tools["subagent_parallel_run"] = dspy.Tool(
            self.subagent_tool.parallel_run,
            name="subagent_parallel_run",
            desc="Kick off several research subagents at once; each runs web search and writes back findings.",
        )

        # Single-loop agent program
        self.agent_program = dspy.ReAct(
            AgentSignature,
            tools=list(self.tools.values()),
            max_iters=3,
        )
        self.agent_program.adapter = self.baml_adapter

    @observe(name="single_loop_agent", capture_input=True, capture_output=True)
    def forward(self, query: str) -> dspy.Prediction:
        with dspy.context(lm=self.agent_lm):
            return self.agent_program(query=query)


def main() -> None:
    print("Initializing agent...")
    agent = Agent()
    print("Starting agent...")
    result = agent(query="Lamine vs Doue who's better?")
    print(result.answer)

if __name__ == "__main__":
    main()
