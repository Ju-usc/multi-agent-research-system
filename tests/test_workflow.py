"""Single-call integration tests for LeadAgent.aforward (fast + comprehensive)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import importlib
import sys
import types
import os
from pathlib import Path
 


@pytest.fixture
def lead_agent():
    """Create a LeadAgent instance for testing without Langfuse side effects."""
    # Ensure project root is on sys.path
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Stub external deps BEFORE importing workflow
    sys.modules['langfuse'] = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
    sys.modules['utils'] = types.SimpleNamespace(
        setup_langfuse=lambda: None,
        prediction_to_markdown=lambda obj, title=None: "stub",
        log_call=(lambda f: f),
    )
    class _WebSearchTool:
        def __init__(self, api_key: str):
            pass
        async def __call__(self, *a, **k):
            return ""
    class _FileSystemTool:
        def __init__(self, fs):
            self.fs = fs
        def tree(self, max_depth: int = 3) -> str:
            return self.fs.tree(max_depth)
        def read(self, path: str) -> str:
            return self.fs.read(path)
    sys.modules['tools'] = types.SimpleNamespace(WebSearchTool=_WebSearchTool, FileSystemTool=_FileSystemTool)

    agent_module = importlib.import_module('workflow')
    LeadAgent = getattr(agent_module, 'LeadAgent')
    return LeadAgent()


@pytest.mark.asyncio
async def test_end_to_end_single_call(lead_agent):
    """One aforward() call validates plan write, subagent storage, synthesis, and final report."""
    query = "What is the capital of France?"

    # Pre-seed filesystem to verify persistence doesn't regress
    lead_agent.fs.write("custom/initial.md", "Initial data")

    # Mock plan
    mock_plan = MagicMock()
    mock_plan.reasoning = "Need to find capital of France"
    mock_plan.tasks = [
        types.SimpleNamespace(
            task_name="task-1",
            objective="Find the capital of France",
            tool_guidance={"web_search": "Search for France capital"},
            tool_budget=5,
            expected_output="The capital city name",
            tip=None,
        )
    ]
    mock_plan.plan_filename = "test-plan"

    # Mock subagent result
    mock_result = types.SimpleNamespace(
        task_name="task-1",
        summary="Paris is the capital of France",
        finding="Paris is the capital and largest city of France",
    )

    # Mock synthesis (trigger finalization)
    mock_synthesis = MagicMock(is_done=True, synthesis="The capital of France is Paris")

    # Short-circuit final report to avoid real LLM
    with patch.object(lead_agent.planner, 'acall', new=AsyncMock(return_value=mock_plan)):
        with patch.object(lead_agent, 'execute_subagent_task', new=AsyncMock(return_value=mock_result)):
            with patch.object(lead_agent.synthesizer, 'acall', new=AsyncMock(return_value=mock_synthesis)):
                with patch.object(
                    lead_agent.final_reporter,
                    'acall',
                    new=AsyncMock(return_value=types.SimpleNamespace(report="# Report\nParis")),
                ):
                    result = await lead_agent.aforward(query)

    # Assertions: decision surface
    assert result is not None
    assert result["is_done"] is True

    # Filesystem artifacts for this single cycle
    last_cycle = lead_agent.cycle_idx
    tree = lead_agent.fs.tree(max_depth=None)
    assert f"cycle_{last_cycle:03d}/synthesis.md" in tree
    assert f"cycle_{last_cycle:03d}/final_report.md" in tree
    assert f"cycle_{last_cycle:03d}/test-plan.md" in tree
    assert f"cycle_{last_cycle:03d}/task-1/result.md" in tree

    # Persistence of pre-seeded file
    assert lead_agent.fs.exists("custom/initial.md")
    assert lead_agent.fs.read("custom/initial.md") == "Initial data"
