"""
Tool implementations for the multi-agent research system.
All tools are implemented as classes with __call__ methods unless class methods are used to call the tool.
"""

import logging
import shutil
from typing import Any, Optional
import json
import dspy
from pathlib import Path
from perplexity import Perplexity
from config import (
    PERPLEXITY_API_KEY,
    WEBSEARCH_MAX_RESULTS,
    WEBSEARCH_MAX_TOKENS_PER_PAGE,
    PARALLEL_THREADS,
    FILESYSTEM_TREE_MAX_DEPTH,
)
from models import (
    Todo,
    SubagentTask,
    ExecuteSubagentTask,
)
from tracer import trace


logger = logging.getLogger(__name__)


def tool_response(is_error: bool, message: str) -> str:
    """Unified JSON response for primitive tools."""
    return json.dumps({"isError": is_error, "message": message})


# ---------- WebSearch ----------

class WebSearchTool:
    """Perplexity search supporting up to 5 queries (max 20 results)."""

    def __init__(self) -> None:
        if not PERPLEXITY_API_KEY:
            raise RuntimeError("PERPLEXITY_API_KEY must be set to use WebSearchTool")

        self.client = Perplexity(api_key=PERPLEXITY_API_KEY)
        self.call_count = 0

    @trace
    def __call__(
        self,
        queries: list[str],
        max_results: Optional[int] = WEBSEARCH_MAX_RESULTS,
        max_tokens_per_page: Optional[int] = WEBSEARCH_MAX_TOKENS_PER_PAGE,
    ) -> str:
        """Search web via Perplexity. Batch queries for efficiency."""
        self.call_count += 1
        try:
            query_param = queries if len(queries) != 1 else queries[0]
            response = self.client.search.create(
                query=query_param,
                max_results=max_results,
                max_tokens_per_page=max_tokens_per_page,
            )
            results = response.results
        except Exception as exc:
            return tool_response(True, f"Search failed for {queries}: {exc}")

        lines: list[str] = []
        for idx, result in enumerate(results, 1):
            title = result.title
            snippet = result.snippet
            url = result.url
            date = result.date
            last_updated = result.last_updated
            lines.append(f"{idx}. {title}\n{snippet}\n{url}\n{date}\n{last_updated}")

        return tool_response(False, "\n\n".join(lines))

class ParallelToolCall:
    """Run multiple tool invocations concurrently."""

    def __init__(self, tools: dict[str, Any], *, num_threads: int = PARALLEL_THREADS) -> None:
        self.tools = tools
        self._num_threads = num_threads

    @trace
    def __call__(self, calls: list[dict]) -> list[str]:
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
            return tool_response(True, f"Unknown tool: {name}")

        try:
            return tool(**args)
        except Exception as error:
            logger.exception("Tool %s failed", name)
            return tool_response(True, f"Tool '{name}' failed: {error}")


# ---------- FileSystem ----------

class FileSystemTool:
    """Sandboxed file system for research artifacts."""

    def __init__(self, root: Path | str = "memory"):
        self.root = Path(root) if isinstance(root, str) else root
        self.root.mkdir(parents=True, exist_ok=True)

    @trace
    def write(self, path: str, content: str) -> str:
        file_path = self.root / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return tool_response(False, f"Written to {path}")

    @trace
    def read(self, path: str) -> str:
        file_path = self.root / path
        if not file_path.exists():
            return tool_response(True, f"File not found: {path}. The subagent may have returned artifact_path without actually writing the file.")
        return tool_response(False, file_path.read_text())

    @trace
    def exists(self, path: str) -> bool:
        return (self.root / path).exists()

    @trace
    def tree(self, max_depth: Optional[int] = FILESYSTEM_TREE_MAX_DEPTH) -> str:
        paths: list[str] = []
        self._collect_paths(self.root, "", paths, max_depth, 0)

        root_label = f"{str(self.root).rstrip('/')}/"
        if not paths:
            return tool_response(False, f"{root_label} (empty)")
        return tool_response(False, "\n".join([root_label] + sorted(paths)))

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
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

# ---------- TodoList ----------

class TodoListTool:
    """Run-scoped todo store."""

    def __init__(self) -> None:
        self._todos: list[Todo] = []

    @trace
    def write(self, todos: list[Todo]) -> str:
        try:
            self._todos = todos
            count = len(self._todos)
            return tool_response(False, f"Updated {count} todo item{'s' if count != 1 else ''}")
        except Exception as e:
            return tool_response(True, f"Failed to write todos: {e}")

    @trace
    def read(self) -> str:
        try:
            todos_json = json.dumps([t.model_dump() for t in self._todos], indent=2)
            return tool_response(False, f"Todos ({len(self._todos)} items):\n{todos_json}")
        except Exception as e:
            return tool_response(True, f"Failed to read todos: {e}")


# ---------- SubagentTool ----------

class SubagentTool:
    """Execute a single subagent research task via ReAct."""

    def __init__(self, tools: list[dspy.Tool], lm: Any, adapter: Optional[Any] = None) -> None:
        self._tools = tools
        self._lm = lm
        self._adapter = adapter

    @trace
    def __call__(self, task: SubagentTask) -> str:
        """Execute task and return SubagentResult JSON."""
        current_instructions = ExecuteSubagentTask.instructions
        new_instructions = current_instructions + "\n" + task.prompt
        new_signature = ExecuteSubagentTask.with_instructions(instructions=new_instructions)

        subagent = dspy.ReAct(new_signature, tools=self._tools, max_iters=task.tool_budget)
        subagent.lm = self._lm
        subagent.adapter = self._adapter
        
        with dspy.context(lm=self._lm, adapter=self._adapter):
            prediction = subagent(task=task)

        result = prediction.final_result
        result.task_name = task.task_name
        if result.artifact_path:
            result.artifact_path = result.artifact_path.lstrip('/').removeprefix('memory/')
        
        return json.dumps(result.model_dump(), indent=2)
