from types import SimpleNamespace
import pathlib
import sys

import pytest

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

    expected = (
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

    assert output == expected
    assert fake_client.api_key == "fake-key"
    assert fake_client.search.last_kwargs == {
        "queries": ["test query"],
        "max_results": 5,
        "max_tokens_per_page": 1024,
    }


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

    assert results == [
        "alpha:A",
        "beta:123",
        "Unknown tool: missing",
    ]
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

    assert results == ["fine", "Tool error (boom): kaboom"]

