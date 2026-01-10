"""BrowseComp-Plus dataset and local search tools."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy
from fastmcp import Client

from tracer import trace

MCP_URL = "http://localhost:8000/mcp"


# ---------- Dataset ----------


@dataclass
class BrowseCompTask:
    query_id: str
    query: str
    answer: str
    gold_docs: set[str]
    evidence_docs: set[str]
    negative_docs: set[str]


class BrowseCompPlusDataset:
    """Load decrypted BrowseComp-Plus tasks."""

    def __init__(self, jsonl_path: str | Path, num_examples: int | None = None, seed: int = 42):
        self.seed = seed
        self.num_examples = num_examples
        self._tasks = self._load(jsonl_path)
        self._examples = None

    def _load(self, path: str | Path) -> list[BrowseCompTask]:
        tasks = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                tasks.append(
                    BrowseCompTask(
                        query_id=str(row["query_id"]),
                        query=row["query"],
                        answer=row["answer"],
                        gold_docs={d["docid"] for d in row["gold_docs"]},
                        evidence_docs={d["docid"] for d in row["evidence_docs"]},
                        negative_docs={d["docid"] for d in row["negative_docs"]},
                    )
                )
        return tasks

    def load(self) -> list[dspy.Example]:
        """Load dataset as dspy.Example list (compatible with BrowseCompDataset)."""
        if self._examples is not None:
            return self._examples

        examples = [
            dspy.Example(problem=t.query, answer=t.answer).with_inputs("problem")
            for t in self._tasks
        ]

        if self.num_examples is not None:
            rng = random.Random(self.seed)
            examples = rng.sample(examples, min(self.num_examples, len(examples)))

        self._examples = examples
        return examples

    def split(self, train_size: float = 0.5) -> tuple[list, list]:
        """Split into train/val sets."""
        examples = self.load()
        rng = random.Random(self.seed)
        shuffled = rng.sample(examples, len(examples))
        split_idx = int(len(shuffled) * train_size)
        return shuffled[:split_idx], shuffled[split_idx:]

    def __len__(self) -> int:
        return len(self._tasks)


# ---------- LocalSearch ----------


class LocalSearchTool:
    """Search via local BM25 MCP server."""

    COST_USD = 0.0

    def __init__(self, mcp_url: str = MCP_URL):
        self.mcp_url = mcp_url
        self.total_cost_usd = 0.0

    @trace
    def __call__(self, queries: list[str]) -> list[dict[str, Any]]:
        results = []
        for query in queries:
            hits = asyncio.run(self._search(query))
            results.extend(hits)
        return results

    async def _search(self, query: str) -> list[dict[str, Any]]:
        async with Client(self.mcp_url) as client:
            result = await client.call_tool("search", {"query": query})
            hits = json.loads(result.content[0].text)
            return [
                {"title": f"Doc {h['docid']}", "excerpt": h["snippet"], "url": h["docid"]}
                for h in hits
            ]


class LocalFetchTool:
    """Fetch document via local MCP server."""

    COST_USD = 0.0

    def __init__(self, mcp_url: str = MCP_URL):
        self.mcp_url = mcp_url
        self.total_cost_usd = 0.0

    @trace
    def __call__(self, docids: list[str]) -> list[dict[str, Any]]:
        if len(docids) > 5:
            return "Error: Too many docids. Max 5 allowed."
        return [asyncio.run(self._fetch(docid)) for docid in docids]

    async def _fetch(self, docid: str) -> dict[str, Any]:
        async with Client(self.mcp_url) as client:
            result = await client.call_tool("get_document", {"docid": docid})
            doc = json.loads(result.content[0].text)
            return {"title": f"Doc {docid}", "url": docid, "content": doc["text"]}
