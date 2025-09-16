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
        def __init__(self, root: str = "memory"):
            self.root = Path(root)
            self.root.mkdir(exist_ok=True)
        def write(self, path: str, content: str):
            fp = self.root / path
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content)
            return fp
        def read(self, path: str) -> str:
            fp = self.root / path
            if not fp.exists():
                return f"[ERROR] File not found: {path}"
            return fp.read_text()
        def exists(self, path: str) -> bool:
            return (self.root / path).exists()
        def tree(self, max_depth: int = 3) -> str:
            paths = []
            def _collect(p: Path, rel: str, depth: int):
                if max_depth is not None and depth >= max_depth:
                    return
                items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name)) if p.exists() else []
                for item in items:
                    ip = f"{rel}{item.name}" if rel else item.name
                    if item.is_dir():
                        paths.append(f"{ip}/")
                        _collect(item, f"{ip}/", depth+1)
                    else:
                        paths.append(ip)
            _collect(self.root, "", 0)
            if not paths:
                return "memory/ (empty)"
            return "\n".join(["memory/"] + sorted(paths))

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
        detail="Paris is the capital and largest city of France",
        artifact_path=None,
    )

    # Mock synthesis (trigger finalization)
    mock_synthesis = MagicMock(is_done=True, synthesis="The capital of France is Paris")

    # Short-circuit final report to avoid real LLM
    with patch.object(lead_agent.planner, 'acall', new=AsyncMock(return_value=mock_plan)):
        with patch.object(lead_agent, 'execute_subagent_task', new=AsyncMock(return_value=mock_result)):
            with patch.object(lead_agent.synthesizer, 'acall', new=AsyncMock(return_value=mock_synthesis)):
                async def fake_final_report(query: str, final_synthesis: str) -> str:
                    path = f"cycle_{lead_agent.cycle_idx:03d}/final_report.md"
                    lead_agent.fs.write(path, "# Report\nParis")
                    return "# Report\nParis"
                with patch.object(lead_agent, 'generate_final_report', new=fake_final_report):
                    result = await lead_agent.aforward(query)

    # Assertions: decision surface
    assert result == "# Report\nParis"
    
    # Verify plan was stored
    plan_content = lead_agent.fs.read("cycle_001/plans/test-plan.md")
    assert "Need to find capital of France" in plan_content
    
    # Verify subagent result stored
    task_content = lead_agent.fs.read("cycle_001/subagent_results/task-1.md")
    assert "Paris is the capital of France" in task_content
    
    # Verify synthesis stored
    synthesis_content = lead_agent.fs.read("cycle_001/synthesis.md")
    assert "The capital of France is Paris" in synthesis_content
    
    # Verify final report stored
    report_content = lead_agent.fs.read("cycle_001/final_report.md")
    assert "# Report\nParis" in report_content
    
    # Verify pre-seeded data persisted
    assert lead_agent.fs.read("custom/initial.md") == "Initial data"
