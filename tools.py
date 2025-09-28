"""
Tool implementations for the multi-agent research system.
All tools are implemented as classes with __call__ methods unless class methods are used to call the tool.
"""

import logging
from typing import Dict, List, Any, Optional
import json
import dspy
from pathlib import Path
from perplexity import Perplexity
from langfuse import observe
from config import PERPLEXITY_API_KEY
from logging_config import trace_call
from models import (
    Todo,
    SubagentTask,
    SubagentResult,
    ExecuteSubagentTask,
)


logger = logging.getLogger(__name__)
# ---------- WebSearch ----------

class WebSearchTool:
    """Perplexity search supporting up to 5 queries (max 20 results)."""

    def __init__(self) -> None:
        if not PERPLEXITY_API_KEY:
            raise RuntimeError("PERPLEXITY_API_KEY must be set to use WebSearchTool")

        self.client = Perplexity(api_key=PERPLEXITY_API_KEY)

    @trace_call("tool_web_search")
    @observe(name="tool_web_search", capture_input=True, capture_output=True)
    def __call__(
        self,
        queries: List[str],
        max_results: Optional[int] = 5,
        max_tokens_per_page: Optional[int] = 1024,
    ) -> str:
        try:
            response = self.client.search.create(
                queries=queries,
                max_results=max_results,
                max_tokens_per_page=max_tokens_per_page,
            )
            results = response.results
        except Exception as exc:
            return f"Error searching for '{queries}': {exc}"

        lines: List[str] = []
        for idx, result in enumerate(results, 1):
            title = result.title
            snippet = result.snippet
            url = result.url
            date = result.date
            last_updated = result.last_updated
            lines.append(f"{idx}. {title}\n{snippet}\n{url}\n{date}\n{last_updated}")

        return "\n\n".join(lines)

class ParallelToolCall:
    """Run several tool invocations concurrently and return their outputs.

    Up to four calls execute in parallel threads. Example:

        [
            {"tool": "web_search", "args": {"query": "multi-agent"}},
            {"tool": "filesystem_read", "args": {"path": "summary.md"}},
        ]
    """

    def __init__(self, tools: Dict[str, Any], *, num_threads: int = 4) -> None:
        self.tools = tools
        self._num_threads = num_threads

    @trace_call("tool_parallel_tool_call")
    @observe(name="tool_parallel_tool_call", capture_input=True, capture_output=True)
    def __call__(self, calls: list[dict]) -> list[str]:
        """Calls must be dicts of the form {'tool': str, 'args': dict}."""
        if not calls:
            return []

        parallel = dspy.Parallel(num_threads=self._num_threads, provide_traceback=True)
        exec_pairs = [(self._invoke, (call,)) for call in calls]
        results = parallel(exec_pairs)
        return [str(result) for result in results]

    def _invoke(self, call: dict) -> Any:
        name = call.get("tool")
        args = call.get("args", {})
        tool = self.tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"

        try:
            return tool(**args)
        except Exception as error:
            logger.exception("Tool %s failed", name)
            return f"Tool error ({name}): {error}"


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

    @trace_call("tool_filesystem_write")
    @observe(name="tool_filesystem_write", capture_input=True, capture_output=True)
    def write(self, path: str, content: str) -> Path:
        file_path = self.root / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    @trace_call("tool_filesystem_read")
    @observe(name="tool_filesystem_read", capture_input=True, capture_output=True)
    def read(self, path: str) -> str:
        file_path = self.root / path
        if not file_path.exists():
            return f"[ERROR] File not found: {path}"
        return file_path.read_text()

    def exists(self, path: str) -> bool:
        return (self.root / path).exists()

    @trace_call("tool_filesystem_tree")
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

    @trace_call("tool_todo_write")
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
        
    @trace_call("tool_todo_read")
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

    @trace_call("tool_subagent")
    @observe(name="tool_subagent", capture_input=True, capture_output=True)
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

    @trace_call("tool_subagent_parallel_run")
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
