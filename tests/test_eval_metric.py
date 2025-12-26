"""
Tests for BrowseComp evaluation metrics.

Tests the BrowseCompEvaluator class methods for grading predictions
and calculating metrics.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import dspy
import pytest

from eval import BrowseCompEvaluator
from models import LLMJudgeAnswer


@pytest.fixture
def mock_args():
    """Create mock args for BrowseCompEvaluator."""
    return SimpleNamespace(
        optimize=False,
        num_threads=1,
    )


@pytest.fixture
def mock_judge():
    """Mock judge that can be configured per test."""
    return MagicMock()


@pytest.fixture
def evaluator(mock_args, mock_judge, monkeypatch):
    """Create BrowseCompEvaluator with mocked DSPy boundary."""
    # Mock at DSPy boundary (external lib that makes API calls)
    monkeypatch.setattr("eval.dspy.LM", lambda **kwargs: MagicMock())
    monkeypatch.setattr("eval.dspy.ChainOfThought", lambda sig: mock_judge)
    
    evaluator = BrowseCompEvaluator(mock_args)
    return evaluator


def test_calculate_lm_cost_basic(evaluator):
    """Test LM cost calculation with basic token usage."""
    usage = {
        "openrouter/x-ai/grok-4.1-fast": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 0}
        }
    }
    
    cost = evaluator.calculate_lm_cost(usage)
    
    # grok-4.1-fast: $0.20 per 1M input, $0.50 per 1M output
    # 1000 tokens / 1M * $0.20 = $0.0002
    # 500 tokens / 1M * $0.50 = $0.00025
    # Total: $0.00045
    assert cost == pytest.approx(0.00045)


def test_calculate_lm_cost_with_caching(evaluator):
    """Test LM cost calculation with cached tokens (no cache discount for grok)."""
    usage = {
        "openrouter/x-ai/grok-4.1-fast": {
            "prompt_tokens": 2000,
            "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 1000}
        }
    }
    
    cost = evaluator.calculate_lm_cost(usage)
    
    # grok-4.1-fast: $0.20 per 1M input, $0.50 per 1M output (no cached_input pricing)
    # Input: 2000 tokens / 1M * $0.20 = $0.0004
    # Output: 500 tokens / 1M * $0.50 = $0.00025
    # Total: $0.00065
    assert cost == pytest.approx(0.00065)


def test_calculate_lm_cost_unknown_model(evaluator):
    """Test LM cost calculation gracefully handles unknown models."""
    usage = {
        "unknown-model": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        }
    }
    
    cost = evaluator.calculate_lm_cost(usage)
    
    # Unknown model should log warning and return 0 cost
    assert cost == 0.0


def test_calculate_lm_cost_multiple_models(evaluator):
    """Test LM cost calculation with multiple models."""
    usage = {
        "openrouter/x-ai/grok-4.1-fast": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 0}
        },
        "openrouter/deepseek/deepseek-v3.2": {
            "prompt_tokens": 500,
            "completion_tokens": 200,
            "prompt_tokens_details": {"cached_tokens": 0}
        }
    }
    
    cost = evaluator.calculate_lm_cost(usage)
    
    # grok-4.1-fast: (1000/1M * $0.20) + (500/1M * $0.50) = $0.0002 + $0.00025 = $0.00045
    # deepseek-v3.2: (500/1M * $0.24) + (200/1M * $0.38) = $0.00012 + $0.000076 = $0.000196
    # Total: $0.000646
    assert cost == pytest.approx(0.000646)


def test_metric_correct_answer(evaluator, mock_judge):
    """Test metric with correct answer."""
    example = dspy.Example(problem="What is 2+2?", answer="4")
    pred = dspy.Prediction(report="The answer is 4")
    pred.elapsed_seconds = 2.0
    pred.websearch_calls = 1
    pred.get_lm_usage = lambda: {
        "openrouter/x-ai/grok-4.1-fast": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 0}
        }
    }
    
    # Configure mock judge to return correct answer
    mock_judge.return_value = SimpleNamespace(
        answer=LLMJudgeAnswer(
            is_correct=True,
            extracted_answer="4",
            reasoning="The answer matches."
        )
    )
    
    result = evaluator.metric(example, pred)
    
    assert float(result) == 1.0
    assert pred.metrics["accuracy"] == 1.0
    assert pred.metrics["elapsed_seconds"] == 2.0
    assert "Expected: 4" in result.feedback
    assert "Extracted: 4" in result.feedback


def test_metric_incorrect_answer(evaluator, mock_judge):
    """Test metric with incorrect answer."""
    example = dspy.Example(problem="What is 2+2?", answer="4")
    pred = dspy.Prediction(report="The answer is 5")
    pred.elapsed_seconds = 1.0
    pred.websearch_calls = 0
    pred.get_lm_usage = lambda: {}
    
    # Configure mock judge to return incorrect
    mock_judge.return_value = SimpleNamespace(
        answer=LLMJudgeAnswer(
            is_correct=False,
            extracted_answer="5",
            reasoning="The answer is wrong."
        )
    )
    
    result = evaluator.metric(example, pred)
    
    assert float(result) == 0.0
    assert pred.metrics["accuracy"] == 0.0


def test_metric_stores_feedback(evaluator, mock_judge):
    """Test metric stores feedback in ScoreWithFeedback."""
    example = dspy.Example(problem="Q", answer="A")
    pred = dspy.Prediction(report="A")
    pred.elapsed_seconds = 1.0
    pred.websearch_calls = 1
    pred.get_lm_usage = lambda: {}
    
    mock_judge.return_value = SimpleNamespace(
        answer=LLMJudgeAnswer(
            is_correct=True,
            extracted_answer="A",
            reasoning="Correct."
        )
    )
    
    result = evaluator.metric(example, pred)
    
    assert float(result) == 1.0
    assert "Accuracy: 1/1" in result.feedback
    assert "Reasoning: Correct." in result.feedback


def test_grade_prediction_error_handling(evaluator, mock_judge):
    """Test grade_prediction handles errors gracefully."""
    example = dspy.Example(problem="Q", answer="A")
    pred = dspy.Prediction(report="A")
    pred.elapsed_seconds = 1.0
    pred.websearch_calls = 0
    pred.get_lm_usage = lambda: {}
    
    # Configure mock judge to raise exception
    mock_judge.side_effect = Exception("Judge failed")
    
    result = evaluator.metric(example, pred)
    
    assert float(result) == 0.0  # Returns 0 on error
    assert "Grading failed" in result.feedback
