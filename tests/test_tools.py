from types import SimpleNamespace
import pathlib
import sys
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tools


def test_web_search_tool_formats_results(monkeypatch):
    class _FakeSearch:
        def __init__(self):
            self.last_kwargs = None

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        title="Result One",
                        snippet="Snippet One",
                        url="https://one.example",
                        date="2024-01-01",
                        last_updated="2024-01-02",
                    ),
                    SimpleNamespace(
                        title="Result Two",
                        snippet="Snippet Two",
                        url="https://two.example",
                        date="2024-01-03",
                        last_updated="2024-01-04",
                    ),
                ]
            )

    class _FakePerplexity:
        def __init__(self, api_key: str | None = None) -> None:
            self.api_key = api_key
            self.search = _FakeSearch()

    fake_client = _FakePerplexity()

    monkeypatch.setattr(tools, "PERPLEXITY_API_KEY", "fake-key")

    def _factory(api_key=None):
        fake_client.api_key = api_key
        return fake_client

    monkeypatch.setattr(tools, "Perplexity", _factory)

    tool = tools.WebSearchTool()

    output = tool(["test query"])
    result = json.loads(output)

    expected_message = (
        "1. Result One\n"
        "Snippet One\n"
        "https://one.example\n"
        "2024-01-01\n"
        "2024-01-02\n\n"
        "2. Result Two\n"
        "Snippet Two\n"
        "https://two.example\n"
        "2024-01-03\n"
        "2024-01-04"
    )

    assert result["isError"] is False
    assert result["message"] == expected_message
    assert fake_client.api_key == "fake-key"
    assert fake_client.search.last_kwargs == {
        "query": "test query",
        "max_results": 5,
        "max_tokens_per_page": 1024,
    }


def test_web_search_tool_call_count(monkeypatch):
    class _FakeSearch:
        def create(self, **_):
            return SimpleNamespace(results=[])

    class _FakePerplexity:
        def __init__(self, api_key: str | None = None) -> None:
            self.api_key = api_key
            self.search = _FakeSearch()

    monkeypatch.setattr(tools, "PERPLEXITY_API_KEY", "fake-key")
    monkeypatch.setattr(tools, "Perplexity", lambda api_key=None: _FakePerplexity(api_key))

    tool = tools.WebSearchTool()

    assert tool.call_count == 0
    tool(["first"])
    tool(["second"])

    assert tool.call_count == 2


class _FakeParallel:
    last_num_threads = None

    def __init__(self, *, num_threads=None, **__):
        self.num_threads = num_threads
        _FakeParallel.last_num_threads = num_threads

    def __call__(self, exec_pairs):
        return [func(*args) for func, args in exec_pairs]


def test_parallel_tool_call_invokes_tools(monkeypatch):
    monkeypatch.setattr(tools.dspy, "Parallel", _FakeParallel)

    calls = [
        {"tool": "alpha", "args": {"value": "A"}},
        {"tool": "beta", "args": {"value": 123}},
        {"tool": "missing", "args": {}},
    ]

    def alpha(value: str) -> str:
        return f"alpha:{value}"

    def beta(value: int) -> str:
        return f"beta:{value}"

    tool = tools.ParallelToolCall({
        "alpha": alpha,
        "beta": beta,
    })

    results = tool(calls)

    # alpha and beta return plain strings (tools return their own format)
    assert results[0] == "alpha:A"
    assert results[1] == "beta:123"
    # Missing tool returns JSON error
    missing_result = json.loads(results[2])
    assert missing_result["isError"] is True
    assert "Unknown tool: missing" in missing_result["message"]
    assert _FakeParallel.last_num_threads == 4


def test_parallel_tool_call_reports_failures(monkeypatch):
    monkeypatch.setattr(tools.dspy, "Parallel", _FakeParallel)

    def ok() -> str:
        return "fine"

    def boom() -> str:
        raise RuntimeError("kaboom")

    tool = tools.ParallelToolCall({
        "ok": ok,
        "boom": boom,
    })

    results = tool([
        {"tool": "ok", "args": {}},
        {"tool": "boom", "args": {}},
    ])

    assert results[0] == "fine"
    error_result = json.loads(results[1])
    assert error_result["isError"] is True
    assert "boom" in error_result["message"]
    assert "kaboom" in error_result["message"]

