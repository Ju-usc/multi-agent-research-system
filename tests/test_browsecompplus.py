"""E2E tests for BrowseComp-Plus local tools.

Requires MCP server running:
    cd ../BrowseComp-Plus
    uv run python searcher/mcp_server.py --searcher-type bm25 --index-path ./indexes/bm25/ --get-document --port 8000

Run with:
    RUN_E2E=1 uv run pytest tests/test_browsecompplus_e2e.py -v
"""

import os
import pytest

from browsecompplus import LocalSearchTool, LocalFetchTool, BrowseCompPlusDataset

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_E2E"),
    reason="E2E test requires RUN_E2E=1 and MCP server running"
)


def test_search_returns_results():
    tool = LocalSearchTool()
    results = tool(queries=["Vikings TV show"])
    assert isinstance(results, list)
    assert len(results) > 0
    assert "url" in results[0]


def test_search_multiple_queries():
    tool = LocalSearchTool()
    results = tool(queries=["Vikings", "Georgia Hirst"])
    assert len(results) > 0


def test_fetch_document():
    search = LocalSearchTool()
    results = search(queries=["Vikings"])
    docid = results[0]["url"]

    fetch = LocalFetchTool()
    docs = fetch(docids=[docid])
    assert len(docs) == 1
    assert "content" in docs[0]
    assert len(docs[0]["content"]) > 0


def test_dataset_load():
    path = "../BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl"
    if not os.path.exists(path):
        pytest.skip("Decrypted dataset not found")

    dataset = BrowseCompPlusDataset(path, num_examples=5, seed=42)
    assert len(dataset) > 0

    examples = dataset.load()
    assert len(examples) == 5
    assert examples[0].problem
    assert examples[0].answer
