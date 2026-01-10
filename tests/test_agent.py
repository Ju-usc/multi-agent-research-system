"""Lightweight agent-path tests to keep the branch focused on sync behaviour."""

from types import SimpleNamespace


def test_agent_forward_invokes_lead_agent(tmp_path):
    import agent

    calls: list[dict] = []

    class StubLead:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(answer="stubbed")

    class StubLM:
        model = "stub-model"

    agent_instance = agent.Agent.__new__(agent.Agent)
    agent_instance.lead_agent = StubLead()
    agent_instance.leadagent_lm = StubLM()
    agent_instance.subagent_lm = StubLM()
    agent_instance.log_dir = str(tmp_path)

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

    write_response = tool.write(todos)
    assert write_response == "Updated 1 todos"

    read_response = tool.read()
    assert isinstance(read_response, list)
    assert len(read_response) == 1
    assert read_response[0]["content"] == "Sketch experiment plan"


def test_agent_reset_workspace(tmp_path):
    """Test that reset_workspace correctly resets all agent state."""
    import agent
    from tools import FileSystemTool, WebSearchTool, WebFetchTool, TodoListTool
    from models import Todo

    # Create agent instance manually (avoid LM initialization)
    agent_instance = agent.Agent.__new__(agent.Agent)
    agent_instance.fs_tool = FileSystemTool(root=tmp_path / "initial")
    agent_instance.search_tool = WebSearchTool.__new__(WebSearchTool)
    agent_instance.search_tool.total_cost_usd = 0.05
    agent_instance.fetch_tool = WebFetchTool.__new__(WebFetchTool)
    agent_instance.fetch_tool.total_cost_usd = 0.01
    agent_instance.todo_list_tool = TodoListTool()
    agent_instance.todo_list_tool._todos = [
        Todo(id="1", content="test", status="pending", priority="high")
    ]

    # Reset to new workspace
    new_dir = tmp_path / "new_workspace"
    agent_instance.reset_workspace(new_dir)

    assert agent_instance.fs_tool.root == new_dir
    assert new_dir.exists()
    assert agent_instance.search_tool.total_cost_usd == 0.0
    assert agent_instance.fetch_tool.total_cost_usd == 0.0
    assert agent_instance.todo_list_tool._todos == []

