from types import SimpleNamespace
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tools


class _FakeExa:
    def __init__(self, key: str) -> None:
        self.key = key

    def search_and_contents(self, query: str, **kwargs):
        assert query == "Test query"
        assert kwargs["num_results"] == 2
        assert kwargs["type"] == "auto"
        assert kwargs["text"] is True
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    title="Document One",
                    summary="Concise summary",
                    text="Full body of the first document.",
                    url="https://one.example",
                ),
                SimpleNamespace(
                    title=None,
                    summary="",
                    text="Full body of the second document.",
                    highlights=["Notable highlight from the second document."],
                    url="https://two.example",
                ),
            ]
        )


def test_web_search_tool_formats_results(monkeypatch):
    monkeypatch.setattr(tools, "Exa", _FakeExa)

    tool = tools.WebSearchTool(api_key="fake-key", max_results=5)

    output = tool("  Test query  ", count=2, snippet_length=30)

    expected = (
        "1. Document One\n"
        "Concise summary\n"
        "https://one.example\n\n"
        "2. Untitled\n"
        "Full body of the second document.\n"
        "https://two.example"
    )

    assert output == expected


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

