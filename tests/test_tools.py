from types import SimpleNamespace
import json
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
    output = tool(queries=["test query"], objective="Find test results")
    result = json.loads(output)

    assert result["isError"] is False
    assert "Result One" in result["message"]
    assert mock_parallel.beta.last_kwargs["objective"] == "Find test results"
    assert mock_parallel.beta.last_kwargs["search_queries"] == ["test query"]


def test_web_search_tool_real_api():
    tool = tools.WebSearchTool()
    output = tool(queries=["Python programming"], objective="What is Python?")
    result = json.loads(output)

    assert result["isError"] is False
    assert len(result["message"]) > 0





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


def test_filesystem_tool_blocks_path_traversal(tmp_path):
    fs = tools.FileSystemTool(root=tmp_path / "sandbox")
    
    # Valid paths work
    result = json.loads(fs.write("valid.txt", "content"))
    assert result["isError"] is False
    
    result = json.loads(fs.read("valid.txt"))
    assert result["isError"] is False
    assert result["message"] == "content"
    
    # Path traversal blocked
    result = json.loads(fs.write("../escape.txt", "bad"))
    assert result["isError"] is True
    assert "Invalid path" in result["message"]
    
    result = json.loads(fs.read("../escape.txt"))
    assert result["isError"] is True
    assert "Invalid path" in result["message"]
    
    # Absolute paths blocked
    result = json.loads(fs.write("/etc/passwd", "bad"))
    assert result["isError"] is True
    assert "Invalid path" in result["message"]
    
    # Nested traversal blocked
    result = json.loads(fs.write("subdir/../../escape.txt", "bad"))
    assert result["isError"] is True
    assert "Invalid path" in result["message"]

