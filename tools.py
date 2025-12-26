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
    FILESYSTEM_TREE_MAX_DEPTH,
)
from models import (
    ToolResponse,
    Todo,
    SubagentTask,
    ExecuteSubagentTask,
)
from tracer import trace


logger = logging.getLogger(__name__)


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
            return str(ToolResponse(isError=True, message=f"Search failed for {queries}: {exc}"))

        lines: list[str] = []
        for idx, result in enumerate(results, 1):
            title = result.title
            snippet = result.snippet
            url = result.url
            date = result.date
            last_updated = result.last_updated
            lines.append(f"{idx}. {title}\n{snippet}\n{url}\n{date}\n{last_updated}")

        return str(ToolResponse(isError=False, message="\n\n".join(lines)))

class ParallelToolCall:
    """Run multiple tool invocations concurrently."""

    def __init__(self, tools: dict[str, Any], *, num_threads: int = 4) -> None:
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
            return str(ToolResponse(isError=True, message=f"Unknown tool: {name}"))

        try:
            return tool(**args)
        except Exception as error:
            logger.exception("Tool %s failed", name)
            return str(ToolResponse(isError=True, message=f"Tool '{name}' failed: {error}"))


# ---------- FileSystem ----------

class FileSystemTool:
    """Sandboxed file system for research artifacts."""

    def __init__(self, root: Path | str = "memory"):
        self.root = Path(root).resolve() if isinstance(root, str) else root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, path: str) -> Path | None:
        """Resolve path and verify it's inside sandbox. Returns None if invalid."""
        resolved = (self.root / path).resolve()
        if resolved.is_relative_to(self.root):
            return resolved
        return None
    @trace
    def write(self, path: str, content: str) -> str:
        file_path = self._safe_path(path)
        if file_path is None:
            return str(ToolResponse(isError=True, message=f"Invalid path: {path}"))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return str(ToolResponse(isError=False, message=f"Written to {path}"))

    @trace
    def read(self, path: str) -> str:
        file_path = self._safe_path(path)
        if file_path is None:
            return str(ToolResponse(isError=True, message=f"Invalid path: {path}"))
        if not file_path.exists():
            return str(ToolResponse(isError=True, message=f"File not found: {path}"))
        return str(ToolResponse(isError=False, message=file_path.read_text()))

    @trace
    def tree(self, max_depth: Optional[int] = FILESYSTEM_TREE_MAX_DEPTH) -> str:
        paths = []
        for p in sorted(self.root.rglob("*")):
            relative = p.relative_to(self.root)
            if max_depth is None or len(relative.parts) <= max_depth:
                paths.append(str(relative) + ("/" if p.is_dir() else ""))

        root_label = f"{str(self.root).rstrip('/')}/"
        if not paths:
            return str(ToolResponse(isError=False, message=f"{root_label} (empty)"))
        return str(ToolResponse(isError=False, message="\n".join([root_label] + sorted(paths))))

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
        self._todos = todos
        count = len(todos)
        return str(ToolResponse(isError=False, message=f"Updated {count} todo item{'s' if count != 1 else ''}"))

    @trace
    def read(self) -> str:
        todos_json = json.dumps([t.model_dump() for t in self._todos], indent=2)
        return str(ToolResponse(isError=False, message=f"Todos ({len(self._todos)} items):\n{todos_json}"))

    def clear(self) -> None:
        self._todos = []


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

        with dspy.context(lm=self._lm, adapter=self._adapter):
            prediction = subagent(task=task)

        result = prediction.final_result
        result.name = task.name
        if result.artifact_path:
            result.artifact_path = result.artifact_path.lstrip('/').removeprefix('memory/')

        return str(ToolResponse(isError=False, message=json.dumps(result.model_dump(), indent=2)))
