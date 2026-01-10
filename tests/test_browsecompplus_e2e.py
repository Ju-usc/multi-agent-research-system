"""E2E test for BrowseComp-Plus local tools.

Requires MCP server running:
    cd ../BrowseComp-Plus
    python searcher/mcp_server.py --searcher-type bm25 --index-path ./indexes/bm25/ --get-document --port 8000

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


class TestLocalSearchTool:
    def test_search_returns_results(self):
        tool = LocalSearchTool()
        results = tool(queries=["Vikings TV show"])
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert "url" in results[0]

    def test_search_multiple_queries(self):
        tool = LocalSearchTool()
        results = tool(queries=["Vikings TV show", "Georgia Hirst actress"])
        
        assert isinstance(results, list)
        assert len(results) > 0


class TestLocalFetchTool:
    def test_fetch_document(self):
        search_tool = LocalSearchTool()
        search_results = search_tool(queries=["Vikings"])
        
        assert len(search_results) > 0
        docid = search_results[0]["url"]
        
        fetch_tool = LocalFetchTool()
        results = fetch_tool(docids=[docid])
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert "content" in results[0]
        assert len(results[0]["content"]) > 0


class TestBrowseCompPlusDataset:
    def test_load_decrypted(self):
        dataset_path = "../BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl"
        if not os.path.exists(dataset_path):
            pytest.skip("Decrypted dataset not found")
        
        dataset = BrowseCompPlusDataset(dataset_path, num_examples=10, seed=42)
        assert len(dataset) == 830  # total tasks
        
        examples = dataset.load()
        assert len(examples) == 10  # sampled
        assert examples[0].problem
        assert examples[0].answer
