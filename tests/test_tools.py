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


@pytest.mark.asyncio
async def test_web_search_tool_formats_results(monkeypatch):
    monkeypatch.setattr(tools, "Exa", _FakeExa)

    tool = tools.WebSearchTool(api_key="fake-key", max_results=5)

    output = await tool("  Test query  ", count=2, snippet_length=30)

    expected = (
        "1. Document One\n"
        "Concise summary\n"
        "https://one.example\n\n"
        "2. Untitled\n"
        "Full body of the second document.\n"
        "https://two.example"
    )

    assert output == expected
