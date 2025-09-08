"""
Tool implementations for the multi-agent research system.
All tools are implemented as classes with __call__ methods unless class methods are used to call the tool.
"""

import asyncio
import functools
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import re
import json
import dspy
from brave_search_python_client import BraveSearch, WebSearchRequest
from models import (
    FileSystem,
    Todo,
    SubagentTask,
    SubagentResult,
    ExecuteSubagentTask,
)


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


class FileSystemTool:
    """Tool for interacting with the research filesystem."""
    
    def __init__(self, fs: FileSystem):
        """Initialize with FileSystem instance."""
        self.fs = fs
    
    def tree(self, max_depth: int = 3) -> str:
        """Show filesystem structure.
        
        Args:
            max_depth: Maximum depth to display (default: 3)
            
        Returns:
            ASCII tree representation of research memory
        """
        return self.fs.tree(max_depth)
    
    def read(self, path: str) -> str:
        """Read file content by path.
        
        Args:
            path: Relative path from memory root (e.g., 'cycle_001/plan.md')
            
        Returns:
            File content or error message if not found
        """
        return self.fs.read(path)


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
