"""Lightweight agent-path tests to keep the branch focused on sync behaviour."""

import json
from types import SimpleNamespace


def test_agent_forward_invokes_lead_agent():
    import agent

    calls: list[dict] = []

    class StubLead:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(answer="stubbed")

    agent_instance = agent.Agent.__new__(agent.Agent)
    agent_instance.lead_agent = StubLead()

    result = agent.Agent.forward(agent_instance, query="quick check")

    assert result.answer == "stubbed"
    assert calls == [{"query": "quick check"}]


def test_todo_list_round_trip():
    from tools import TodoListTool
    from models import Todo

    tool = TodoListTool()
    todos = [
        Todo(id="1", content="Sketch experiment plan", status="pending", priority="high"),
    ]

    write_response = json.loads(tool.write(todos))
    assert write_response == {
        "success": True,
        "count": 1,
        "message": "Updated 1 todos items",
    }

    read_response = json.loads(tool.read())
    assert read_response["count"] == 1
    assert read_response["todos"][0]["content"] == "Sketch experiment plan"

