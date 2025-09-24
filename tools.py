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
from exa_py import Exa
from langfuse import observe
from config import EXA_API_KEY
from models import (
    Todo,
    SubagentTask,
    SubagentResult,
    ExecuteSubagentTask,
)

# ---------- WebSearch ----------

class WebSearchTool:
    """Call Exa's search API and format the hits as a markdown list."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        max_results: int = 10,
        max_snippet_length: int = 10_000,
        search_type: str = "auto",
    ) -> None:
        key = api_key or EXA_API_KEY
        if not key:
            raise ValueError("EXA_API_KEY is required to initialize WebSearchTool")

        self.client = Exa(key)
        self.max_results = max(1, max_results)
        self.max_snippet_length = max_snippet_length
        self.search_type = search_type

    @observe(name="tool_web_search", capture_input=True, capture_output=True)
    def __call__(self, query: str, count: int = 3, snippet_length: int = 2000) -> str:
        """Return up to `count` results with snippets trimmed to the requested length."""
        query = query.strip()
        if not query:
            return "# Search Error\n\nQuery cannot be empty."

        num_results = max(1, min(count, self.max_results))
        snippet_length = max(100, min(snippet_length, self.max_snippet_length))

        kwargs: Dict[str, Any] = {
            "num_results": num_results,
            "type": self.search_type,
            "text": True,
        }

        try:
            response = self.client.search_and_contents(
                query,
                **kwargs,
            )
        except Exception as exc:
            return f"# Search Error\n\nError searching for '{query}': {exc}"

        if not response.results:
            return f"No results found for '{query}'"
        lines: List[str] = []
        for idx, item in enumerate(response.results, 1):
            title = item.title or "Untitled"
            context = (item.summary or item.text or "").strip()[:snippet_length]
            lines.append(f"{idx}. {title}\n{context}\n{item.url}")
        output = "\n\n".join(lines)
        return output

class ParallelSearchTool:
    """Run several tool invocations concurrently and return their outputs."""

    def __init__(self, tools: Dict[str, Any]) -> None:
        self.tools = tools

    async def __call__(self, calls: list[dict]) -> list[str]:
        """Calls must be dicts of the form {'tool_name': str, 'args': dict}."""

        async def execute(call: dict) -> str:
            name = call.get("tool_name")
            args = call.get("args", {})
            if name not in self.tools:
                return f"Unknown tool: {name}"

            tool = self.tools[name]
            if asyncio.iscoroutinefunction(tool):
                return await tool(**args)

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, functools.partial(tool, **args))

        results = await asyncio.gather(*(execute(call) for call in calls), return_exceptions=True)
        return [str(r) if isinstance(r, Exception) else r for r in results]


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

    @observe(name="tool_filesystem_write", capture_input=True, capture_output=True)
    def write(self, path: str, content: str) -> Path:
        file_path = self.root / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    @observe(name="tool_filesystem_read", capture_input=True, capture_output=True)
    def read(self, path: str) -> str:
        file_path = self.root / path
        if not file_path.exists():
            return f"[ERROR] File not found: {path}"
        return file_path.read_text()

    def exists(self, path: str) -> bool:
        return (self.root / path).exists()

    @observe(name="tool_filesystem_tree", capture_input=True, capture_output=True)
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

    @observe(name="tool_todo_write", capture_input=True, capture_output=True)
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
        
    @observe(name="tool_todo_read", capture_input=True, capture_output=True)
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

class SubagentTool(dspy.Module):
    """Execute subagent tasks with per-task instruction via with_instruction."""

    def __init__(self, tools: List[dspy.Tool], lm: Any, adapter: Optional[Any] = None) -> None:
        super().__init__()
        self._tools = tools
        self._lm = lm
        self._adapter = adapter

    @observe(name="tool_subagent_forward", capture_input=True, capture_output=True)
    def forward(self, task: SubagentTask) -> Optional[SubagentResult]:
        # Append the task prompt from the lead agent to the existing instructions
        current_instructions = ExecuteSubagentTask.instructions
        new_instructions = current_instructions + "\n" + task.prompt
        new_signature = ExecuteSubagentTask.with_instructions(instructions=new_instructions)

        subagent = dspy.ReAct(new_signature, tools=self._tools, max_iters=task.tool_budget)
        subagent.lm = self._lm
        subagent.adapter = self._adapter
        with dspy.context(lm=self._lm, adapter=self._adapter):
            prediction = subagent(task=task)
        final = prediction.final_result
        final.task_name = task.task_name
        return final

    @observe(name="tool_subagent_parallel_run", capture_input=True, capture_output=True)
    def parallel_run(self, tasks: List[SubagentTask]) -> str:
        if not tasks:
            return json.dumps({"successes": [], "failures": []}, indent=2)

        examples = [dspy.Example(task=task).with_inputs("task") for task in tasks]

        results, failed_examples, exceptions = self.batch(
            examples,
            return_failed_examples=True,
            max_errors=len(examples),
            provide_traceback=True,
        )

        summary = {
            "successes": [res.model_dump() for res in results],
            "failures": [
                {"task_name": ex.task.task_name, "error": str(err)}
                for ex, err in zip(failed_examples, exceptions)
            ],
        }

        return json.dumps(summary, indent=2)
