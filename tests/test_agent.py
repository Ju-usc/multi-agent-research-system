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
    assert write_response["isError"] is False
    assert "Updated 1 todo item" in write_response["message"]

    read_response = json.loads(tool.read())
    assert read_response["isError"] is False
    assert "Sketch experiment plan" in read_response["message"]


def test_agent_reset_workspace(tmp_path):
    """Test that reset_workspace correctly resets all agent state."""
    import agent
    from tools import FileSystemTool, WebSearchTool, TodoListTool
    from models import Todo

    # Create agent instance manually (avoid LM initialization)
    agent_instance = agent.Agent.__new__(agent.Agent)
    agent_instance.fs_tool = FileSystemTool(root=tmp_path / "initial")
    agent_instance.web_search_tool = WebSearchTool.__new__(WebSearchTool)
    agent_instance.web_search_tool.call_count = 5
    agent_instance.todo_list_tool = TodoListTool()
    agent_instance.todo_list_tool._todos = [
        Todo(id="1", content="test", status="pending", priority="high")
    ]

    # Reset to new workspace
    new_dir = tmp_path / "new_workspace"
    agent_instance.reset_workspace(new_dir)

    assert agent_instance.fs_tool.root == new_dir
    assert new_dir.exists()
    assert agent_instance.web_search_tool.call_count == 0
    assert agent_instance.todo_list_tool._todos == []

