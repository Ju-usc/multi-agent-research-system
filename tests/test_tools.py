from types import SimpleNamespace
import pytest
import tools


class FakeBetaSearch:
    """Mock Parallel beta.search."""
    def __init__(self, results=None):
        self.last_kwargs = None
        self._results = results or []

    def search(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(results=self._results)


class FakeParallelClient:
    """Mock Parallel client."""
    def __init__(self, api_key=None, results=None):
        self.api_key = api_key
        self.beta = FakeBetaSearch(results)


@pytest.fixture
def mock_parallel(monkeypatch):
    """Fixture to mock Parallel client."""
    results = [
        SimpleNamespace(title="Result One", excerpts=["Excerpt One"], url="https://one.example"),
        SimpleNamespace(title="Result Two", excerpts=["Excerpt Two"], url="https://two.example"),
    ]
    client = FakeParallelClient(results=results)
    monkeypatch.setattr(tools, "PARALLEL_API_KEY", "fake-key")
    monkeypatch.setattr(tools, "Parallel", lambda api_key=None: setattr(client, 'api_key', api_key) or client)
    return client


def test_web_search_tool(mock_parallel):
    tool = tools.WebSearchTool()
    result = tool(queries=["test query"], objective="Find test results")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["title"] == "Result One"
    assert mock_parallel.beta.last_kwargs["objective"] == "Find test results"


class _FakeParallel:
    def __init__(self, *, num_threads=None, **__):
        pass

    def __call__(self, exec_pairs):
        return [func(*args) for func, args in exec_pairs]


def test_parallel_tool_call_invokes_tools(monkeypatch):
    monkeypatch.setattr(tools.dspy, "Parallel", _FakeParallel)

    tool = tools.ParallelToolCall({
        "alpha": lambda value: f"alpha:{value}",
        "beta": lambda value: f"beta:{value}",
    })

    results = tool([{"alpha": {"value": "A"}}, {"beta": {"value": 123}}])

    assert results == ["alpha:A", "beta:123"]


@pytest.mark.parametrize("call,expected_substr", [
    ({"missing": {}}, "missing"),
    ({}, "Error:"),
    ({"needs_arg": {"wrong": "x"}}, "Error:"),
])
def test_parallel_tool_call_errors(monkeypatch, call, expected_substr):
    """Tool errors return error strings."""
    monkeypatch.setattr(tools.dspy, "Parallel", _FakeParallel)

    def needs_arg(required: str) -> str:
        return f"got: {required}"

    tool = tools.ParallelToolCall({"alpha": lambda: "ok", "needs_arg": needs_arg})
    results = tool([call])

    assert results[0].startswith("Error:") or expected_substr in results[0]


def test_filesystem_tool_blocks_path_traversal(tmp_path):
    fs = tools.FileSystemTool(root=tmp_path / "sandbox")

    # Valid paths work
    assert fs.write("valid.txt", "content") == "Written to valid.txt"
    assert fs.read("valid.txt") == "content"

    # Path traversal blocked
    for path in ["../escape.txt", "/etc/passwd", "subdir/../../escape.txt"]:
        result = fs.write(path, "bad")
        assert result.startswith("Error:") and "Invalid path" in result

