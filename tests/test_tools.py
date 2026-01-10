from types import SimpleNamespace
import pytest
import tools


@pytest.fixture
def mock_parallel(monkeypatch):
    """Mock Parallel client for web search tests."""
    results = [
        SimpleNamespace(title="Result One", excerpts=["Excerpt One"], url="https://one.example"),
        SimpleNamespace(title="Result Two", excerpts=["Excerpt Two"], url="https://two.example"),
    ]

    class MockParallel:
        def __init__(self, api_key=None):
            self.beta = SimpleNamespace(
                search=lambda **kw: SimpleNamespace(results=results),
            )
            self._last_search_kwargs = None

        def _search(self, **kwargs):
            self._last_search_kwargs = kwargs
            return SimpleNamespace(results=results)

    monkeypatch.setattr(tools, "PARALLEL_API_KEY", "fake-key")
    client = MockParallel()
    monkeypatch.setattr(tools, "Parallel", lambda api_key=None: client)
    return client


def test_web_search_tool(mock_parallel):
    tool = tools.WebSearchTool()
    result = tool(queries=["test"], objective="test objective")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["title"] == "Result One"


class MockDspyParallel:
    def __init__(self, *, num_threads=None, **__):
        pass

    def __call__(self, exec_pairs):
        return [func(*args) for func, args in exec_pairs]


def test_parallel_tool_call_invokes_tools(monkeypatch):
    monkeypatch.setattr(tools.dspy, "Parallel", MockDspyParallel)
    tool = tools.ParallelToolCall({
        "alpha": lambda value: f"alpha:{value}",
        "beta": lambda value: f"beta:{value}",
    })
    results = tool([{"alpha": {"value": "A"}}, {"beta": {"value": 123}}])
    assert results == ["alpha:A", "beta:123"]


def test_parallel_tool_call_missing_tool_returns_error(monkeypatch):
    monkeypatch.setattr(tools.dspy, "Parallel", MockDspyParallel)
    tool = tools.ParallelToolCall({"alpha": lambda: "ok"})
    results = tool([{"missing": {}}])
    assert "Error:" in results[0]


def test_parallel_tool_call_bad_args_returns_error(monkeypatch):
    monkeypatch.setattr(tools.dspy, "Parallel", MockDspyParallel)
    tool = tools.ParallelToolCall({"needs_arg": lambda required: f"got:{required}"})
    results = tool([{"needs_arg": {"wrong": "x"}}])
    assert "Error:" in results[0]


def test_filesystem_tool_blocks_path_traversal(tmp_path):
    fs = tools.FileSystemTool(root=tmp_path / "sandbox")

    # Valid paths work
    assert fs.write("valid.txt", "content") == "Written to valid.txt"
    assert fs.read("valid.txt") == "content"

    # Path traversal blocked
    for path in ["../escape.txt", "/etc/passwd", "subdir/../../escape.txt"]:
        result = fs.write(path, "bad")
        assert result.startswith("Error:") and "Invalid path" in result

