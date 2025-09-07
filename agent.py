import dspy
from config import (
    BRAVE_SEARCH_API_KEY, OPENAI_API_KEY, SMALL_MODEL, BIG_MODEL, TEMPERATURE, BIG_MODEL_MAX_TOKENS, SMALL_MODEL_MAX_TOKENS
)
from models import (
    FileSystem, 
)
from tools import (
    WebSearchTool, FileSystemTool, TodoListTool
)
from dspy.adapters.baml_adapter import BAMLAdapter

class Agent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.fs = FileSystem()
        self.cycle_idx = 0  # Track cycles
        # Single BAML adapter instance used across all modules
        self.baml_adapter = BAMLAdapter()

        # Initialize tool instances
        self.web_search_tool = WebSearchTool(BRAVE_SEARCH_API_KEY)
        self.fs_tool = FileSystemTool(self.fs)
        self.todo_list_tool = TodoListTool()

        # Create DSPy tools from class instances
        self.tools = {
            "web_search": dspy.Tool(
                self.web_search_tool,
                name="web_search",
                desc="Search the web for information. Default 5 results."
            ),
            "filesystem_read": dspy.Tool(
                self.fs_tool.read,
                name="filesystem_read",
                desc="Read file from research memory. Use path relative to memory/, e.g., 'cycle_003/foo/bar.md' (do NOT include leading 'memory/')."
            ),
            "filesystem_tree": dspy.Tool(
                self.fs_tool.tree,
                name="filesystem_tree",
                desc="Show filesystem structure. Tree shows entries under 'memory/'. When calling filesystem_read, drop the 'memory/' prefix."
            ),
            "todo_list_read": dspy.Tool(
                self.todo_list_tool.read,
                name="todo_list_read",
                desc="Read the todo list. Useful to keep track of the tasks you need to complete."
            ),
            "todo_list_write": dspy.Tool(
                self.todo_list_tool.write,
                name="todo_list_write",
                desc="Write to the todo list. Useful to keep track of the tasks you need to complete."
            )
        }

        # Initialize language models
        self.agent_lm = dspy.LM(
            model=BIG_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=BIG_MODEL_MAX_TOKENS,
        )
        
        self.agent = dspy.ReAct(Agent, tools=self.tools, max_iters=3)

    def aforward(self, query: str) -> dspy.Prediction:
        return self.agent.acall(query=query)
