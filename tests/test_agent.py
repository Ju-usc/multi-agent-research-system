"""Test the Multi-Agent Research System."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import dspy
import warnings
import logging
import importlib
import sys
import types
import asyncio
from dataset import BrowseCompDataset
# eval imports are done lazily inside tests to avoid importing agent at module import time

# Suppress noisy warnings and logs
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

@pytest.mark.asyncio
async def test_lead_agent_basic_run():
    # Stub langfuse before importing agent
    fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
    sys.modules['langfuse'] = fake_langfuse
    import utils as _utils
    with patch.object(_utils, 'setup_langfuse', return_value=None):
        agent_module = importlib.import_module('agent')
        LeadAgent = getattr(agent_module, 'LeadAgent')
        agent = LeadAgent()
    
    # Mock internal calls to avoid network
    mock_plan = MagicMock()
    mock_plan.tasks = []
    mock_plan.plan_filename = "test-plan"
    mock_synth = MagicMock()
    mock_synth.is_done = True
    mock_synth.synthesis = "Done"
    
    with patch.object(agent.planner, 'acall', new=AsyncMock(return_value=mock_plan)):
        with patch.object(agent.synthesizer, 'acall', new=AsyncMock(return_value=mock_synth)):
            with patch.object(agent, 'generate_final_report', new=AsyncMock(return_value="Final report")):
                artifacts = await agent.aforward("Test simple research query")
                report = await agent.generate_final_report("Test simple research query", artifacts["synthesis"])
    
    assert isinstance(report, str), "Report should be a string"
    assert "Final report" in report

# ========== BrowseComp Evaluation Tests ==========

def test_browsecomp_dataset():
    """Test BrowseComp dataset loading."""
    
    dataset = BrowseCompDataset(num_examples=5)
    examples = dataset.load()
    
    assert len(examples) == 5
    assert all(hasattr(ex, 'problem') and ex.problem for ex in examples)
    assert all(hasattr(ex, 'answer') and ex.answer for ex in examples)


def test_browsecomp_program_wrapper():
    """Test DSPy program wrapper for LeadAgent (with fake agent)."""
    fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
    sys.modules['langfuse'] = fake_langfuse
    import utils as _utils
    with patch.object(_utils, 'setup_langfuse', return_value=None):
        from eval import BrowseCompProgram

    class FakeAgent:
        async def run(self, problem: str) -> str:
            return f"Report for: {problem}"

    program = BrowseCompProgram(FakeAgent())
    assert hasattr(program, 'forward')
    assert callable(program.forward)


def test_browsecomp_metric():
    """Test BrowseComp metric function."""
    fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
    sys.modules['langfuse'] = fake_langfuse
    import utils as _utils
    with patch.object(_utils, 'setup_langfuse', return_value=None):
        from eval import browsecomp_metric
    
    # Create mock example and prediction
    example = dspy.Example(
        problem="What is 2+2?",
        answer="4"
    )
    
    # Test correct prediction
    pred_correct = dspy.Prediction(report="The answer is 4.")
    score_correct = browsecomp_metric(example, pred_correct)
    assert isinstance(score_correct, float)
    assert 0.0 <= score_correct <= 1.0
    
    # Test incorrect prediction  
    pred_incorrect = dspy.Prediction(report="The answer is 5.")
    score_incorrect = browsecomp_metric(example, pred_incorrect)
    assert isinstance(score_incorrect, float)
    assert 0.0 <= score_incorrect <= 1.0


def test_browsecomp_evaluation_framework():
    """Test the DSPy evaluation framework integration with a fake agent (no network)."""
    fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
    sys.modules['langfuse'] = fake_langfuse
    import utils as _utils
    with patch.object(_utils, 'setup_langfuse', return_value=None):
        from eval import run_browsecomp_evaluation
    class FakeAgent:
        async def run(self, problem: str) -> str:
            # Echo minimal answer to pass some cases
            if "2+2" in problem:
                return "The answer is 4."
            return "Report: N/A"

    results = run_browsecomp_evaluation(
        num_examples=2,
        num_threads=1,
        agent=FakeAgent(),
    )

    assert "accuracy" in results
    assert "num_examples" in results
    assert "results" in results
    assert isinstance(results["accuracy"], (int, float))
    assert results["num_examples"] == 2
    assert len(results["results"]) == 2


@pytest.mark.asyncio
async def test_leadagent_basic():
    """Test basic LeadAgent functionality without real LLM calls."""
    fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
    sys.modules['langfuse'] = fake_langfuse
    import utils as _utils
    with patch.object(_utils, 'setup_langfuse', return_value=None):
        agent_module = importlib.import_module('agent')
        LeadAgent = getattr(agent_module, 'LeadAgent')
        agent = LeadAgent()

    mock_plan = MagicMock()
    mock_plan.tasks = []
    mock_plan.plan_filename = "basic"
    mock_synth = MagicMock()
    mock_synth.is_done = True
    mock_synth.synthesis = "Answer"

    with patch.object(agent.planner, 'acall', new=AsyncMock(return_value=mock_plan)):
        with patch.object(agent.synthesizer, 'acall', new=AsyncMock(return_value=mock_synth)):
            with patch.object(agent, 'generate_final_report', new=AsyncMock(return_value="Report")):
                artifacts = await agent.aforward("What is the capital of France?")
                report = await agent.generate_final_report("What is the capital of France?", artifacts["synthesis"])

    assert isinstance(report, str)
    assert len(report) > 0


if __name__ == "__main__":
    # Run BrowseComp tests
    test_browsecomp_dataset()
    test_browsecomp_program_wrapper()
    test_browsecomp_metric()
    test_browsecomp_evaluation_framework()
    asyncio.run(test_leadagent_basic())
    print("✅ All BrowseComp tests passed!")
