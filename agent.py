import dspy
from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.adapters.chat_adapter import ChatAdapter


from config import (
    EXA_API_KEY,
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
        # Core tools
        self.web_search_tool = WebSearchTool(EXA_API_KEY)
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

        dspy.configure(lm=self.agent_lm, adapter=ChatAdapter())


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



        # Single-loop agent program
        self.lead_agent = dspy.ReAct(
            AgentSignature,
            tools=list(self.lead_agent_tools.values()),
            max_iters=3,
        )

    @observe(name="single_loop_agent", capture_input=True, capture_output=True)
    async def aforward(self, query: str) -> dspy.Prediction:
        return await self.lead_agent.acall(query=query)


async def main() -> None:
    print("Initializing agent...")
    agent = Agent()
    print("Starting agent...")
    result = await agent.aforward(query="Lamine vs Doue who's better? Be objective and keep it research short and concise DO NOT ASK ANYTHING ELSE. Try to use tools provided to you to test our system first.")
    print(result.answer)
    dspy.inspect_history(n=10)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
