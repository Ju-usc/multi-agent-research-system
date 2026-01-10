import logging
import shutil
from pathlib import Path
from typing import Any

import dspy
from parallel import Parallel

from config import PARALLEL_API_KEY
from models import ExecuteSubagentTask, SubagentTask, Todo

logger = logging.getLogger(__name__)


class WebSearchTool:
    COST_USD = 0.005

    def __init__(self) -> None:
        if not PARALLEL_API_KEY:
            raise RuntimeError("PARALLEL_API_KEY must be set")
        self._client: Parallel | None = None
        self.total_cost_usd = 0.0

    def __call__(
        self, queries: list[str], objective: str, max_results: int | None = None
    ) -> list[dict] | str:
        self.total_cost_usd += self.COST_USD
        if self._client is None:
            self._client = Parallel(api_key=PARALLEL_API_KEY)
        try:
            response = self._client.beta.search(
                objective=objective, search_queries=queries, max_results=max_results
            )
        except Exception as error:
            return f"Error: Search failed: {error}"
        return [
            {"title": r.title or "Untitled", "excerpt": "\n".join(r.excerpts or []), "url": r.url}
            for r in response.results
        ]


class WebFetchTool:
    COST_USD = 0.001

    def __init__(self) -> None:
        if not PARALLEL_API_KEY:
            raise RuntimeError("PARALLEL_API_KEY must be set")
        self._client: Parallel | None = None
        self.total_cost_usd = 0.0

    def __call__(self, urls: list[str], objective: str) -> list[dict] | str:
        if len(urls) > 5:
            return "Error: Too many URLs. Max 5 allowed."
        self.total_cost_usd += self.COST_USD
        if self._client is None:
            self._client = Parallel(api_key=PARALLEL_API_KEY)
        try:
            response = self._client.beta.extract(
                urls=urls, objective=objective, excerpts=True, full_content=False
            )
        except Exception as error:
            return f"Error: Fetch failed: {error}"
        return [
            {"title": r.title or "Untitled", "url": r.url, "content": "\n".join(r.excerpts or [])}
            for r in response.results
        ]


class ParallelToolCall:
    def __init__(self, tools: dict[str, Any], *, num_threads: int = 4) -> None:
        self.tools = tools
        self._num_threads = num_threads

    def __call__(self, calls: list[dict]) -> list[Any]:
        if not calls:
            return []
        parallel = dspy.Parallel(num_threads=self._num_threads, provide_traceback=True)
        exec_pairs = [(self._invoke, (call,)) for call in calls]
        return parallel(exec_pairs)

    def _invoke(self, call: dict) -> Any:
        name = None
        try:
            name = list(call.keys())[0]
            args = call[name]
            return self.tools[name](**args)
        except Exception as error:
            logger.exception("Tool call failed: %s", call)
            return f"Error: Tool '{name}' failed: {error}" if name else f"Error: Tool call failed: {error}"


class FileSystemTool:
    DEFAULT_TREE_DEPTH = 3

    def __init__(self, root: Path | str = "memory"):
        self.root = Path(root).resolve() if isinstance(root, str) else root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, path: str) -> Path | None:
        resolved = (self.root / path).resolve()
        return resolved if resolved.is_relative_to(self.root) else None

    def write(self, path: str, content: str) -> str:
        file_path = self._safe_path(path)
        if file_path is None:
            return f"Error: Invalid path: '{path}' must be relative"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Written to {path}"

    def read(self, path: str) -> str:
        file_path = self._safe_path(path)
        if file_path is None:
            return f"Error: Invalid path: '{path}' must be relative"
        if not file_path.exists():
            return f"Error: File not found: {path}"
        return file_path.read_text()

    def tree(self, max_depth: int | None = None) -> str:
        if max_depth is None:
            max_depth = self.DEFAULT_TREE_DEPTH
        paths = []
        for p in sorted(self.root.rglob("*")):
            relative = p.relative_to(self.root)
            if len(relative.parts) <= max_depth:
                paths.append(str(relative) + ("/" if p.is_dir() else ""))
        return "\n".join(paths) or "(empty)"

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)


class TodoListTool:
    def __init__(self) -> None:
        self._todos: list[Todo] = []

    def write(self, todos: list[Todo]) -> str:
        self._todos = todos
        return f"Updated {len(todos)} todos"

    def read(self) -> list[dict]:
        return [t.model_dump() for t in self._todos]

    def clear(self) -> None:
        self._todos = []


class SubagentTool:
    def __init__(self, tools: list[dspy.Tool], lm: Any, adapter: Any | None = None) -> None:
        self._tools = tools
        self._lm = lm
        self._adapter = adapter

    def __call__(self, task: SubagentTask) -> dict:
        current_instructions = ExecuteSubagentTask.instructions
        new_instructions = current_instructions + "\n" + task.instructions
        new_signature = ExecuteSubagentTask.with_instructions(instructions=new_instructions)
        subagent = dspy.ReAct(new_signature, tools=self._tools, max_iters=task.max_steps)
        with dspy.context(
            lm=self._lm, adapter=self._adapter, agent_type="subagent", agent_name=task.name
        ):
            prediction = subagent(task=task)
        result = prediction.final_result
        result.name = task.name
        return result.model_dump()
