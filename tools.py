"""
Tool implementations for the multi-agent research system.
All tools are implemented as classes with __call__ methods unless class methods are used to call the tool.
"""

import asyncio
import functools
from typing import Dict, List, Any, Optional
import json
import dspy
from pathlib import Path
from brave_search_python_client import BraveSearch, WebSearchRequest
from models import (
    Todo,
    SubagentTask,
    SubagentResult,
    ExecuteSubagentTask,
)

# ---------- WebSearch ----------

class WebSearchTool:
    """Web search tool using Brave Search API."""
    
    def __init__(self, api_key: str):
        """Initialize with Brave Search API key."""
        self.api_key = api_key
        self.client = BraveSearch(api_key=api_key)
    
    async def __call__(self, query: str, count: int = 5) -> str:
        """Execute web search.
        
        Args:
            query: Search query
            count: Number of results (default: 5, max: 5)
            
        Returns:
            Formatted search results as markdown
        """
        if count > 5:
            count = 5
        
        try:
            response = await self.client.web(WebSearchRequest(q=query, count=count))
            
            if not response.web or not response.web.results:
                return f"No results found for '{query}'"

            # Format results as markdown
            content_lines = [f"# Search results for '{query}'", ""]
            
            for i, result in enumerate(response.web.results[:count], 1):
                content_lines.extend([
                    f"## {i}. {result.title}",
                    f"{result.description}",
                    f"**URL:** {result.url}",
                    ""
                ])
            
            return "\n".join(content_lines)
        except Exception as e:
            return f"# Search Error\n\nError searching for '{query}': {e}"




# ---------- ParallelSearch ----------

class ParallelSearchTool:
    """Execute multiple searches in parallel for efficiency."""
    
    def __init__(self, tools: Dict[str, Any]):
        """Initialize with available tools dictionary."""
        self.tools = tools
    
    async def __call__(self, calls: list[dict]) -> list[str]:
        """Execute multiple tool calls in parallel.
        
        Args:
            calls: List of dicts with 'tool_name' and 'args' keys
                Example: [
                    {"tool_name": "web_search", "args": {"query": "Python programming"}},
                    {"tool_name": "memory_read", "args": {"key": "1-plan"}}
                ]
        
        Returns:
            List of results from each tool call
        """
        async def execute_call(call):
            tool_name = call.get("tool_name")
            args = call.get("args", {})
            
            if tool_name not in self.tools:
                return f"Unknown tool: {tool_name}"
            
            tool = self.tools[tool_name]
            
            # Check if tool is async
            if asyncio.iscoroutinefunction(tool):
                return await tool(**args)
            else:
                # Run sync functions in executor to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, functools.partial(tool, **args))
        
        # Execute all calls in parallel
        results = await asyncio.gather(*[execute_call(call) for call in calls], return_exceptions=True)
        
        # Convert exceptions to error messages
        return [
            str(r) if isinstance(r, Exception) else r
            for r in results
        ]



# ---------- FileSystem ----------

class FileSystemTool:
    """File system for research memory.

    - Defaults to root="memory" when no argument is provided
    - Methods: write, read, exists, tree, clear
    """

    def __init__(self, root: str = "memory"):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)

        # Cache resolved root for simple path safety checks
        try:
            self._resolved_root = self.root.resolve()
        except Exception:
            self._resolved_root = self.root

    def write(self, path: str, content: str) -> Path:
        file_path = self.root / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    def read(self, path: str) -> str:
        file_path = self.root / path
        if not file_path.exists():
            return f"[ERROR] File not found: {path}"
        return file_path.read_text()

    def exists(self, path: str) -> bool:
        return (self.root / path).exists()

    def tree(self, max_depth: Optional[int] = 3) -> str:
        paths: List[str] = []
        self._collect_paths(self.root, "", paths, max_depth, 0)

        root_label = f"{str(self.root).rstrip('/')}/"
        if not paths:
            return f"{root_label} (empty)"
        return "\n".join([root_label] + sorted(paths))

    def _collect_paths(self, path: Path, relative_path: str, paths: list,
                       max_depth: Optional[int], current_depth: int) -> None:
        if max_depth is not None and current_depth >= max_depth:
            return

        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except FileNotFoundError:
            return

        for item in items:
            item_path = f"{relative_path}{item.name}" if relative_path else item.name
            if item.is_dir():
                paths.append(f"{item_path}/")
                self._collect_paths(item, f"{item_path}/", paths, max_depth, current_depth + 1)
            else:
                paths.append(item_path)

    def clear(self) -> None:
        import shutil
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(exist_ok=True)

# ---------- TodoList ----------

class TodoListTool:
    """Run-scoped todo store. Accepts our built Todo model list; minimal and strict."""

    def __init__(self) -> None:
        self._todos: List[Todo] = []

    def write(self, todos: List[Todo]) -> str:
        """Replace the todo list with the given list[Todo] and return Json string"""
        try:
            self._todos = todos
            return json.dumps({
                "success": True,
                "count": len(self._todos),  
                "message": f"Updated {len(self._todos)} todos items",
            })
            
            
        except Exception as e:
            return f"Error writing todos: {e}"
        
    def read(self) -> str:
        """Return the current todos as Json string"""
        try:
            return json.dumps({
                "success": True,
                "count": len(self._todos),
                "message": f"Read {len(self._todos)} todos items",
                "todos": [t.model_dump() for t in self._todos]
            })
        except Exception as e:
            return f"Error reading todos: {e}"


# ---------- SubagentTool ----------

class SubagentTool:
    """Execute subagent tasks with per-task instruction via with_instruction.

    - Input: tasks: list[SubagentTask]
    - Prompt source: task.prompt (excluded from task serialization)
    - Tools: provided by caller (agent-configured allowlist)
    - Output: SubagentResult or list thereof (via parallel_run)
    - Persistence: none; caller handles filesystem writes
    """

    def __init__(self, tools: Dict[str, Any], lm: Any, adapter: Optional[Any] = None) -> None:
        self._tools = tools
        self._lm = lm
        self._adapter = adapter
    
    async def run(self, task: SubagentTask) -> SubagentResult:
        # Prepare instruction-layered signature
        output_signature = ExecuteSubagentTask.with_instruction(task.prompt)

        try:
            # Execute with provided LM and optional adapter
            with dspy.context(lm=self._lm):
                subAgent = dspy.ReAct(output_signature, tools=self._tools, max_iters=task.tool_budget)
                subAgent.adapter = self._adapter
                result = await subAgent.acall(task=task)

            final = result.final_result
            # Preserve the originating task_name instead of relying on LLM output
            final.task_name = task.task_name
            return final
        except Exception as e:
            raise e

    async def parallel_run(self, tasks: List[SubagentTask]) -> List[SubagentResult]:
        return await asyncio.gather(*[self.run(task) for task in tasks], return_exceptions=True)
