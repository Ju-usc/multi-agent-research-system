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
        SimpleNamespace(
            title="Result One",
            excerpts=["Excerpt One"],
            url="https://one.example",
        ),
        SimpleNamespace(
            title="Result Two",
            excerpts=["Excerpt Two"],
            url="https://two.example",
        ),
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
    assert result[0]["url"] == "https://one.example"
    assert mock_parallel.beta.last_kwargs["objective"] == "Find test results"
    assert mock_parallel.beta.last_kwargs["search_queries"] == ["test query"]








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
        {"alpha": {"value": "A"}},
        {"beta": {"value": 123}},
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

    assert results[0] == "alpha:A"
    assert results[1] == "beta:123"
    assert _FakeParallel.last_num_threads == 4


def test_parallel_tool_call_unknown_tool_returns_error(monkeypatch):
    """Unknown tool returns error response (LLM can self-correct)."""
    monkeypatch.setattr(tools.dspy, "Parallel", _FakeParallel)

    tool = tools.ParallelToolCall({"alpha": lambda: "ok"})
    results = tool([{"missing": {}}])

    assert results[0].startswith("Error:")
    assert "missing" in results[0]


def test_parallel_tool_call_empty_dict_returns_error(monkeypatch):
    """Empty call dict returns error response."""
    monkeypatch.setattr(tools.dspy, "Parallel", _FakeParallel)

    tool = tools.ParallelToolCall({"alpha": lambda: "ok"})
    results = tool([{}])

    assert results[0].startswith("Error:")


def test_parallel_tool_call_no_args_tool_works(monkeypatch):
    """Tool with no args works when called with empty args dict."""
    monkeypatch.setattr(tools.dspy, "Parallel", _FakeParallel)

    tool = tools.ParallelToolCall({"no_args": lambda: "success"})
    results = tool([{"no_args": {}}])

    assert results[0] == "success"


def test_parallel_tool_call_wrong_args_returns_error(monkeypatch):
    """Tool called with wrong args returns error response."""
    monkeypatch.setattr(tools.dspy, "Parallel", _FakeParallel)

    def needs_arg(required: str) -> str:
        return f"got: {required}"

    tool = tools.ParallelToolCall({"needs_arg": needs_arg})
    results = tool([{"needs_arg": {"wrong_param": "value"}}])

    assert results[0].startswith("Error:")


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
        {"ok": {}},
        {"boom": {}},
    ])

    assert results[0] == "fine"
    assert results[1].startswith("Error:")
    assert "boom" in results[1]
    assert "kaboom" in results[1]


def test_filesystem_tool_blocks_path_traversal(tmp_path):
    fs = tools.FileSystemTool(root=tmp_path / "sandbox")

    # Valid paths work
    result = fs.write("valid.txt", "content")
    assert result == "Written to valid.txt"

    result = fs.read("valid.txt")
    assert result == "content"

    # Path traversal blocked
    result = fs.write("../escape.txt", "bad")
    assert result.startswith("Error:")
    assert "Invalid path" in result

    result = fs.read("../escape.txt")
    assert result.startswith("Error:")
    assert "Invalid path" in result

    # Absolute paths blocked
    result = fs.write("/etc/passwd", "bad")
    assert result.startswith("Error:")
    assert "Invalid path" in result

    # Nested traversal blocked
    result = fs.write("subdir/../../escape.txt", "bad")
    assert result.startswith("Error:")
    assert "Invalid path" in result

