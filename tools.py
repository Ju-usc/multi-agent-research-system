"""
Tool implementations for the multi-agent research system.
All tools are implemented as classes with __call__ methods unless class methods are used to call the tool.
"""

import logging
import shutil
from typing import Any
import json
import dspy
from pathlib import Path
from parallel import Parallel
from config import (
    PARALLEL_API_KEY,
    FILESYSTEM_TREE_MAX_DEPTH,
    WEBSEARCH_COST_USD,
    WEBFETCH_COST_USD,
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
    """Web search via Parallel AI."""

    def __init__(self) -> None:
        if not PARALLEL_API_KEY:
            raise RuntimeError("PARALLEL_API_KEY must be set")
        self._client: Parallel | None = None
        self.total_cost_usd = 0.0

    @trace
    def __call__(
        self,
        queries: list[str],
        objective: str,
        max_results: int | None = None,
    ) -> str:
        """Search web via Parallel AI."""
        self.total_cost_usd += WEBSEARCH_COST_USD
        
        if self._client is None:
            self._client = Parallel(api_key=PARALLEL_API_KEY)

        try:
            response = self._client.beta.search(
                objective=objective,
                search_queries=queries,
                max_results=max_results,
            )
        except Exception as error:
            return str(ToolResponse(isError=True, message=f"Search failed: {error}"))

        lines: list[str] = []
        for idx, result in enumerate(response.results, 1):
            title = result.title or "Untitled"
            excerpt = "\n".join(result.excerpts or [])
            lines.append(f"{idx}. {title}\n{excerpt}\n{result.url}")

        return str(ToolResponse(isError=False, message="\n\n".join(lines)))

class WebFetchTool:
    """Fetch URL content via Parallel AI Extract API."""

    def __init__(self) -> None:
        if not PARALLEL_API_KEY:
            raise RuntimeError("PARALLEL_API_KEY must be set")
        self._client: Parallel | None = None
        self.total_cost_usd = 0.0

    @trace
    def __call__(
        self,
        urls: list[str],
        objective: str,
    ) -> str:
        """Fetch and extract content from URLs."""
        if len(urls) > 5:
            return str(ToolResponse(isError=True, message="Too many URLs. Max 5 allowed."))

        self.total_cost_usd += WEBFETCH_COST_USD

        if self._client is None:
            self._client = Parallel(api_key=PARALLEL_API_KEY)

        try:
            response = self._client.beta.extract(
                urls=urls,
                objective=objective,
                excerpts=True,
                full_content=False,
            )
        except Exception as error:
            return str(ToolResponse(isError=True, message=f"Fetch failed: {error}"))

        lines: list[str] = []
        for result in response.results:
            title = result.title or "Untitled"
            content = "\n".join(result.excerpts or [])
            lines.append(f"# {title}\nURL: {result.url}\n\n<fetched_content>\n{content}\n</fetched_content>")

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
        name = list(call.keys())[0]
        args = call[name]
        tool = self.tools[name]

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
            return str(ToolResponse(isError=True, message=f"Invalid path: '{path}' must be relative"))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return str(ToolResponse(isError=False, message=f"Written to {path}"))

    @trace
    def read(self, path: str) -> str:
        file_path = self._safe_path(path)
        if file_path is None:
            return str(ToolResponse(isError=True, message=f"Invalid path: '{path}' must be relative"))
        if not file_path.exists():
            return str(ToolResponse(isError=True, message=f"File not found: {path}"))
        return str(ToolResponse(isError=False, message=file_path.read_text()))

    @trace
    def tree(self, max_depth: int | None = FILESYSTEM_TREE_MAX_DEPTH) -> str:
        paths = []
        for p in sorted(self.root.rglob("*")):
            relative = p.relative_to(self.root)
            if max_depth is None or len(relative.parts) <= max_depth:
                paths.append(str(relative) + ("/" if p.is_dir() else ""))

        if not paths:
            return str(ToolResponse(isError=False, message="(empty)"))
        return str(ToolResponse(isError=False, message="\n".join(paths)))

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
        return str(ToolResponse(isError=False, message=f"Updated {len(todos)} todos"))

    @trace
    def read(self) -> str:
        todos_json = json.dumps([t.model_dump() for t in self._todos], indent=2)
        return str(ToolResponse(isError=False, message=f"Todos ({len(self._todos)} items):\n{todos_json}"))

    def clear(self) -> None:
        self._todos = []


# ---------- SubagentTool ----------

class SubagentTool:
    """Execute a single subagent research task via ReAct."""

    def __init__(self, tools: list[dspy.Tool], lm: Any, adapter: Any | None = None) -> None:
        self._tools = tools
        self._lm = lm
        self._adapter = adapter

    @trace
    def __call__(self, task: SubagentTask) -> str:
        """Execute task and return SubagentResult JSON."""
        current_instructions = ExecuteSubagentTask.instructions
        new_instructions = current_instructions + "\n" + task.instructions
        new_signature = ExecuteSubagentTask.with_instructions(instructions=new_instructions)

        subagent = dspy.ReAct(new_signature, tools=self._tools, max_iters=task.max_steps)

        with dspy.context(lm=self._lm, adapter=self._adapter):
            prediction = subagent(task=task)

        result = prediction.final_result
        result.name = task.name

        return str(ToolResponse(isError=False, message=json.dumps(result.model_dump(), indent=2)))
