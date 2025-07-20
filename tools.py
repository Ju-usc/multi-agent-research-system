"""
Tool implementations for the multi-agent research system.
All tools are implemented as classes with __call__ methods.
"""

import asyncio
from typing import Dict, List, Any
import wikipedia
from brave_search_python_client import BraveSearch, WebSearchRequest
from models import Memory


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
            Formatted search results
        """
        if count > 5:
            count = 5
        
        try:
            response = await self.client.web(WebSearchRequest(q=query, count=count))
            
            if not response.web or not response.web.results:
                return f"No results found for '{query}'"

            results = []
            for i, result in enumerate(response.web.results[:count], 1):
                results.append(f"{i}. {result.title}\\n   {result.description}\\n   {result.url}")
            
            return f"Search results for '{query}':\\n\\n" + "\\n\\n".join(results)
        except Exception as e:
            return f"Search error: {e}"


class WikipediaSearchTool:
    """Wikipedia search tool."""
    
    def __call__(self, query: str, sentences: int = 3) -> str:
        """Execute Wikipedia search.
        
        Return a concise English summary for `query` (≤ `sentences` sentences).
        If Wikipedia returns multiple possible pages (disambiguation), we list the
        top 5 options so the calling agent can decide what to do next.
        
        Args:
            query: Search query
            sentences: Maximum sentences in summary (default: 3)
            
        Returns:
            Wikipedia summary or disambiguation options
        """
        try:
            wikipedia.set_lang("en")
            titles = wikipedia.search(query, results=1)
            if not titles:
                return f"No Wikipedia article found for '{query}'."

            title = titles[0]
            summary = wikipedia.summary(title, sentences=sentences, auto_suggest=False)
            return f"Wikipedia – {title}\\n\\n{summary}"

        except wikipedia.exceptions.DisambiguationError as e:
            # Show a short disambiguation list
            opts = "\\n • ".join(e.options[:5])
            return f"Wikipedia disambiguation for '{query}'. Try one of:\\n • {opts}"
        except Exception as err:
            return f"Wikipedia error: {err}"


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
                    {"tool_name": "wikipedia_search", "args": {"query": "Python", "sentences": 5}}
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
                return await loop.run_in_executor(None, tool, **args)
        
        # Execute all calls in parallel
        results = await asyncio.gather(*[execute_call(call) for call in calls], return_exceptions=True)
        
        # Convert exceptions to error messages
        return [
            str(r) if isinstance(r, Exception) else r
            for r in results
        ]


class MemoryTool:
    """Unified memory tool with multiple operations."""
    
    def __init__(self, memory: Memory):
        """Initialize with memory instance."""
        self.memory = memory
    
    def read(self, key: str) -> str:
        """Read memory content by key.
        
        Args:
            key: Memory key (e.g., '1-plan', '1-synthesis', '1-task-0')
            
        Returns:
            Full stored content or error message if not found
        """
        return self.memory.read(key)
    
    def list(self) -> str:
        """List all available memory keys.
        
        Returns:
            Formatted string with all available keys
        """
        return self.memory.list_keys()