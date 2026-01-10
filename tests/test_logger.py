import json
from concurrent.futures import ThreadPoolExecutor

import dspy

from logger import AgentLogger, AgentIteration, AgentLoggingCallback


def test_agent_logger_thread_safety(tmp_path):
    logger = AgentLogger(
        log_dir=str(tmp_path),
        lead_model="lead",
        sub_model="sub",
        query="q",
    )

    logger.log(AgentIteration(agent_type="lead", agent_name="lead_agent", event="start", data={"query": "q"}))

    def log_tool(i: int) -> None:
        logger.log(AgentIteration(
            agent_type="lead",
            agent_name="lead_agent",
            event="tool",
            data={"tool": "noop", "i": i},
        ))

    n = 200
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(log_tool, range(n)))

    logger.log(AgentIteration(agent_type="lead", agent_name="lead_agent", event="finish", data={"answer": "done"}))

    lines = logger.path.read_text().splitlines()
    assert len(lines) == 1 + (n + 2)  # 1 metadata + n tool + start + finish

    records = [json.loads(line) for line in lines]
    assert records[0]["type"] == "metadata"

    iters = [r for r in records[1:] if r["type"] == "iteration"]
    assert len(iters) == n + 2
    assert {r["iteration"] for r in iters} == set(range(1, n + 3))


def test_agent_logger_metadata(tmp_path):
    """Test that metadata is written correctly."""
    logger = AgentLogger(
        log_dir=str(tmp_path),
        lead_model="gpt-4",
        sub_model="gpt-3.5",
        query="test query",
    )

    lines = logger.path.read_text().splitlines()
    assert len(lines) == 1

    metadata = json.loads(lines[0])
    assert metadata["type"] == "metadata"
    assert metadata["lead_model"] == "gpt-4"
    assert metadata["sub_model"] == "gpt-3.5"
    assert metadata["query"] == "test query"


def test_agent_logging_callback_logs_tool_event(tmp_path):
    agent_logger = AgentLogger(
        log_dir=str(tmp_path),
        lead_model="lead",
        sub_model="sub",
        query="q",
    )

    tool = dspy.Tool(lambda x: str(x), name="echo")

    with dspy.context(
        callbacks=[AgentLoggingCallback(agent_logger)],
        agent_type="lead",
        agent_name="lead_agent",
    ):
        tool(x=1)

    records = [json.loads(line) for line in agent_logger.path.read_text().splitlines()]
    assert len(records) == 2  # metadata + tool event

    rec = records[1]
    assert rec["type"] == "iteration"
    assert rec["event"] == "tool"
    assert rec["agent_name"] == "lead_agent"
    assert rec["data"]["tool"] == "echo"
    assert rec["data"]["args"] == {"x": 1}
    assert rec["data"]["result"] == "1"
    assert "duration_ms" in rec["data"]


def test_agent_logging_callback_ignores_finish_tool(tmp_path):
    """Finish tool should not be logged."""
    agent_logger = AgentLogger(
        log_dir=str(tmp_path),
        lead_model="lead",
        sub_model="sub",
        query="q",
    )

    finish_tool = dspy.Tool(lambda: "done", name="finish")
    callback = AgentLoggingCallback(agent_logger)

    with dspy.context(
        callbacks=[callback],
        agent_type="lead",
        agent_name="lead_agent",
    ):
        finish_tool()

    records = [json.loads(line) for line in agent_logger.path.read_text().splitlines()]
    # Only metadata, no tool event for "finish"
    assert len(records) == 1
    assert records[0]["type"] == "metadata"


def test_agent_logging_callback_module_end_without_start(tmp_path):
    """on_module_end should gracefully handle missing start."""
    agent_logger = AgentLogger(
        log_dir=str(tmp_path),
        lead_model="lead",
        sub_model="sub",
        query="q",
    )

    callback = AgentLoggingCallback(agent_logger)
    # Call on_module_end with unknown call_id
    callback.on_module_end("unknown_call_id", outputs=None)

    # Should not crash, only metadata logged
    records = [json.loads(line) for line in agent_logger.path.read_text().splitlines()]
    assert len(records) == 1


def test_agent_logging_callback_tool_end_without_start(tmp_path):
    """on_tool_end should gracefully handle missing start."""
    agent_logger = AgentLogger(
        log_dir=str(tmp_path),
        lead_model="lead",
        sub_model="sub",
        query="q",
    )

    callback = AgentLoggingCallback(agent_logger)
    # Call on_tool_end with unknown call_id
    callback.on_tool_end("unknown_call_id", outputs=None)

    # Should not crash, only metadata logged
    records = [json.loads(line) for line in agent_logger.path.read_text().splitlines()]
    assert len(records) == 1


def test_agent_logging_callback_with_verbose_printer(tmp_path):
    """Test callback works with verbose printer."""
    agent_logger = AgentLogger(
        log_dir=str(tmp_path),
        lead_model="lead",
        sub_model="sub",
        query="q",
    )

    # Mock verbose printer
    class MockVerbosePrinter:
        def __init__(self):
            self.iterations = []

        def print_iteration(self, iteration):
            self.iterations.append(iteration)

    mock_verbose = MockVerbosePrinter()
    callback = AgentLoggingCallback(agent_logger, verbose_printer=mock_verbose)

    tool = dspy.Tool(lambda: "result", name="test_tool")

    with dspy.context(
        callbacks=[callback],
        agent_type="lead",
        agent_name="lead_agent",
    ):
        tool()

    # Verify verbose printer received the iteration
    assert len(mock_verbose.iterations) == 1
    assert mock_verbose.iterations[0].event == "tool"
    assert mock_verbose.iterations[0].data["tool"] == "test_tool"


def test_agent_logging_callback_with_exception(tmp_path):
    """Test callback logs errors properly."""
    agent_logger = AgentLogger(
        log_dir=str(tmp_path),
        lead_model="lead",
        sub_model="sub",
        query="q",
    )

    def failing_tool():
        raise ValueError("tool failed")

    tool = dspy.Tool(failing_tool, name="failing")
    callback = AgentLoggingCallback(agent_logger)

    with dspy.context(
        callbacks=[callback],
        agent_type="lead",
        agent_name="lead_agent",
    ):
        try:
            tool()
        except ValueError:
            pass

    records = [json.loads(line) for line in agent_logger.path.read_text().splitlines()]
    # metadata + tool event with error
    assert len(records) == 2
    assert records[1]["error"] is not None or records[1]["data"].get("result") is None
