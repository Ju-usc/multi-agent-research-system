"""Test the Multi-Agent Research System."""

import pytest
import dspy
import warnings
import logging
from agent import LeadAgent, SubagentTask, SubagentResult, Memory
import asyncio

# Suppress noisy warnings and logs
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

@pytest.mark.asyncio
async def test_lead_agent_basic_run():
    agent = LeadAgent()  # Init with defaults/mocks if needed
    query = "Test simple research query"
    result = await agent.run(query)
    print(result)
    assert isinstance(result, str), "Result should be a string report"
    assert len(agent.steps_trace) > 0, "Steps trace should have entries"
    assert any('Planned' in step['action'] for step in agent.steps_trace), "Should have a planning step"

# ========== BrowseComp Evaluation Tests ==========

def test_browsecomp_dataset():
    """Test BrowseComp dataset loading."""
    from dataset import BrowseCompDataset
    
    dataset = BrowseCompDataset(num_examples=5)
    examples = dataset.load()
    
    assert len(examples) == 5
    assert all(hasattr(ex, 'problem') and ex.problem for ex in examples)
    assert all(hasattr(ex, 'answer') and ex.answer for ex in examples)


def test_browsecomp_program_wrapper():
    """Test DSPy program wrapper for LeadAgent."""
    from eval import BrowseCompProgram
    
    agent = LeadAgent()
    program = BrowseCompProgram(agent)
    
    # Test that it has the forward method
    assert hasattr(program, 'forward')
    assert callable(program.forward)


def test_browsecomp_metric():
    """Test BrowseComp metric function."""
    import dspy
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
    """Test the DSPy evaluation framework integration."""
    from eval import run_browsecomp_evaluation
    
    # Test with very small sample for speed
    results = run_browsecomp_evaluation(
        num_examples=2,
        num_threads=1
    )
    
    # Check result structure
    assert "accuracy" in results
    assert "num_examples" in results  
    assert "results" in results
    assert isinstance(results["accuracy"], (int, float))
    assert results["num_examples"] == 2
    assert len(results["results"]) == 2


@pytest.mark.asyncio
async def test_leadagent_basic():
    """Test basic LeadAgent functionality."""
    agent = LeadAgent()
    
    # Simple test query
    result = await agent.run("What is the capital of France?")
    
    assert isinstance(result, str)
    assert len(result) > 0


if __name__ == "__main__":
    # Run BrowseComp tests
    test_browsecomp_dataset()
    test_browsecomp_program_wrapper()
    test_browsecomp_metric()
    test_browsecomp_evaluation_framework()
    asyncio.run(test_leadagent_basic())
    print("✅ All BrowseComp tests passed!")
