"""Tests for Agent module."""

from types import SimpleNamespace
import pytest

import agent
from tools import FileSystemTool, WebSearchTool, WebFetchTool, TodoListTool
from models import Todo


class StubLM:
    model = "stub-model"


@pytest.fixture
def stub_agent(tmp_path):
    """Create agent with stubbed dependencies (no LM init)."""
    a = agent.Agent.__new__(agent.Agent)
    a.leadagent_lm = StubLM()
    a.subagent_lm = StubLM()
    a.log_dir = str(tmp_path)
    a.fs_tool = FileSystemTool(root=tmp_path / "memory")
    a.search_tool = WebSearchTool.__new__(WebSearchTool)
    a.search_tool.total_cost_usd = 0.0
    a.fetch_tool = WebFetchTool.__new__(WebFetchTool)
    a.fetch_tool.total_cost_usd = 0.0
    a.todo_list_tool = TodoListTool()
    return a


def test_forward_invokes_lead_agent(stub_agent):
    calls = []

    class StubLead:
        def __call__(self, **kw):
            calls.append(kw)
            return SimpleNamespace(answer="stubbed")

    stub_agent.lead_agent = StubLead()
    result = agent.Agent.forward(stub_agent, query="test")

    assert result.answer == "stubbed"
    assert calls == [{"query": "test"}]


def test_reset_workspace(stub_agent, tmp_path):
    stub_agent.search_tool.total_cost_usd = 0.05
    stub_agent.fetch_tool.total_cost_usd = 0.01
    stub_agent.todo_list_tool._todos = [Todo(id="1", content="x", status="pending", priority="high")]

    new_dir = tmp_path / "new_workspace"
    stub_agent.reset_workspace(new_dir)

    assert stub_agent.fs_tool.root == new_dir
    assert new_dir.exists()
    assert stub_agent.search_tool.total_cost_usd == 0.0
    assert stub_agent.fetch_tool.total_cost_usd == 0.0
    assert stub_agent.todo_list_tool._todos == []


def test_todo_list_round_trip():
    tool = TodoListTool()
    todos = [Todo(id="1", content="Plan", status="pending", priority="high")]

    assert tool.write(todos) == "Updated 1 todos"
    assert tool.read()[0]["content"] == "Plan"
