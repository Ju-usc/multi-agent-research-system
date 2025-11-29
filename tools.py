"""
Tool implementations for the multi-agent research system.
All tools are implemented as classes with __call__ methods unless class methods are used to call the tool.
"""

import logging
from typing import Any, Optional
import json
import dspy
from pathlib import Path
from perplexity import Perplexity
from config import PERPLEXITY_API_KEY
from models import (
    Todo,
    SubagentTask,
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
        self.call_count = 0

    def __call__(
        self,
        queries: list[str],  # Supports batch queries - more efficient (1 API call vs N calls)
        max_results: Optional[int] = 5,
        max_tokens_per_page: Optional[int] = 1024,
    ) -> str:
        """Search the web using Perplexity API.
        
        Args:
            queries: List of query strings to search (batched in single API request for efficiency)
            max_results: Max results per query (default 5)
            max_tokens_per_page: Max content tokens per page (default 1024)
            
        Returns:
            Formatted search results with titles, snippets, URLs, dates
        """
        self.call_count += 1
        try:
            # Perplexity API accepts either string or List[str] for batch searching
            # Batch is more efficient: pricing is per API request, not per query
            query_param = queries if len(queries) != 1 else queries[0]
            response = self.client.search.create(
                query=query_param,
                max_results=max_results,
                max_tokens_per_page=max_tokens_per_page,
            )
            results = response.results
        except Exception as exc:
            return f"Error searching for {queries}: {exc}"

        lines: list[str] = []
        for idx, result in enumerate(results, 1):
            title = result.title
            snippet = result.snippet
            url = result.url
            date = result.date
            last_updated = result.last_updated
            lines.append(f"{idx}. {title}\n{snippet}\n{url}\n{date}\n{last_updated}")

        return "\n\n".join(lines)

class ParallelToolCall:
    """Run multiple tool invocations concurrently (max 4 parallel threads).
    
    IMPORTANT: Available tools depend on context:
    - Lead agent can parallelize: subagent_run, filesystem_read, todo_list_read
    - Subagents can parallelize: web_search, filesystem_write
    
    Example - Lead spawning 3 subagents in parallel:
        [
            {"tool": "subagent_run", "args": {"task": {"task_name": "research-ams", ...}}},
            {"tool": "subagent_run", "args": {"task": {"task_name": "research-rollo", ...}}},
            {"tool": "subagent_run", "args": {"task": {"task_name": "find-papers", ...}}}
        ]
    
    Example - Subagent batching multiple web searches in ONE call (more efficient):
        {"tool": "web_search", "args": {"queries": ["AMS Fellows 2006-2019", "Rollo Davidson winners 1991-2004"]}}
    
    Example - Subagent using parallel_tool_call for different tools:
        [
            {"tool": "web_search", "args": {"queries": ["query1", "query2"]}},
            {"tool": "filesystem_write", "args": {"path": "findings.txt", "content": "..."}}
        ]
    
    Note: Each tool must specify correct args as defined in its signature.
    - subagent_run: task (SubagentTask dict with task_name, prompt, description, tool_budget, etc.)
    - web_search: queries (List[str]), max_results (int, optional), max_tokens_per_page (int, optional)
    - filesystem_write: path (str), content (str)
    - filesystem_read: path (str)
    
    IMPORTANT: web_search accepts List[str] for batch efficiency (1 API call = N queries).
    """

    def __init__(self, tools: dict[str, Any], *, num_threads: int = 4) -> None:
        self.tools = tools
        self._num_threads = num_threads

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

    def write(self, path: str, content: str) -> Path:
        file_path = self.root / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    def read(self, path: str) -> str:
        file_path = self.root / path
        if not file_path.exists():
            return f"[ERROR] File not found: {path}. The subagent may have returned artifact_path without actually writing the file. Use the inline 'detail' field instead, or verify file was written."
        return file_path.read_text()

    def exists(self, path: str) -> bool:
        return (self.root / path).exists()

    def tree(self, max_depth: Optional[int] = 3) -> str:
        paths: list[str] = []
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
        self._todos: list[Todo] = []

    def write(self, todos: list[Todo]) -> str:
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
    """Execute a single subagent research task.
    
    This is a regular tool (not a dspy.Module) that creates and runs a ReAct agent
    for one specific research task. For parallel execution of multiple subagents,
    use the parallel_tool_call tool in the lead agent.
    """

    def __init__(self, tools: list[dspy.Tool], lm: Any, adapter: Optional[Any] = None) -> None:
        self._tools = tools
        self._lm = lm
        self._adapter = adapter

    def __call__(self, task: SubagentTask) -> str:
        """Execute one subagent task and return JSON-formatted result.
        
        Args:
            task: SubagentTask with research objective and constraints
            
        Returns:
            JSON string with SubagentResult containing summary, detail, artifact_path
        """
        # Append the task prompt from the lead agent to the existing instructions
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
        
        # Normalize artifact_path: strip any "memory/" prefix to ensure workspace-relative path
        if result.artifact_path:
            result.artifact_path = result.artifact_path.lstrip('/').removeprefix('memory/')
        
        return json.dumps(result.model_dump(), indent=2)
