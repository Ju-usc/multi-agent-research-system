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
