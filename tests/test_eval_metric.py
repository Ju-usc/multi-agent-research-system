"""
Tests for BrowseComp evaluation metrics.

Tests the BrowseCompEvaluator class methods for calculating accuracy,
efficiency, and cost metrics from agent predictions.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import dspy
import pytest

from eval import BrowseCompEvaluator
from config import WEBSEARCH_COST_PER_CALL_USD


@pytest.fixture
def mock_args():
    """Create mock args for BrowseCompEvaluator."""
    return SimpleNamespace(
        metric="efficiency",
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


def test_calculate_metrics_correct_answer(evaluator, mock_judge):
    """Test calculate_metrics with correct answer."""
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
    
    # Configure mock judge at boundary to return correct
    mock_judge.return_value = SimpleNamespace(is_correct=True)
    
    metrics = evaluator.calculate_metrics(example, pred)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["elapsed_seconds"] == 2.0
    assert metrics["lm_cost_usd"] == pytest.approx(0.00045)  # grok pricing
    assert metrics["web_cost_usd"] == pytest.approx(WEBSEARCH_COST_PER_CALL_USD)
    assert metrics["total_cost_usd"] == pytest.approx(0.00045 + WEBSEARCH_COST_PER_CALL_USD)
    assert metrics["websearch_calls"] == 1
    assert "efficiency_temp" in metrics


def test_calculate_metrics_incorrect_answer(evaluator, mock_judge):
    """Test calculate_metrics with incorrect answer."""
    example = dspy.Example(problem="What is 2+2?", answer="4")
    pred = dspy.Prediction(report="The answer is 5")
    pred.elapsed_seconds = 1.0
    pred.websearch_calls = 0
    pred.get_lm_usage = lambda: {}
    
    # Configure mock judge at boundary to return incorrect
    mock_judge.return_value = SimpleNamespace(is_correct=False)
    
    metrics = evaluator.calculate_metrics(example, pred)
    
    assert metrics["accuracy"] == 0.0
    assert metrics["efficiency_temp"] == 0.0  # Efficiency is 0 when incorrect


def test_accuracy_metric(evaluator, mock_judge):
    """Test accuracy_metric returns judge result."""
    example = dspy.Example(problem="Q", answer="A")
    pred = dspy.Prediction(report="A")
    
    # Configure mock judge at boundary
    mock_judge.return_value = SimpleNamespace(is_correct=True)
    
    score = evaluator.accuracy_metric(example, pred)
    
    assert score == 1.0
    mock_judge.assert_called_once()


def test_efficiency_metric_stores_metrics(evaluator, mock_judge):
    """Test efficiency_metric stores full metrics in prediction."""
    example = dspy.Example(problem="Q", answer="A")
    pred = dspy.Prediction(report="A")
    pred.elapsed_seconds = 1.0
    pred.websearch_calls = 1
    pred.get_lm_usage = lambda: {}
    
    # Configure mock judge at boundary
    mock_judge.return_value = SimpleNamespace(is_correct=True)
    
    score = evaluator.efficiency_metric(example, pred)
    
    assert score == 1.0  # Returns accuracy
    assert pred.metrics["accuracy"] == 1.0
    assert pred.metrics["elapsed_seconds"] == 1.0
    assert pred.metrics["websearch_calls"] == 1


def test_judge_prediction_error_handling(evaluator, mock_judge):
    """Test judge_prediction handles errors gracefully."""
    example = dspy.Example(problem="Q", answer="A")
    pred = dspy.Prediction(report="A")
    
    # Configure mock judge at boundary to raise exception
    mock_judge.side_effect = Exception("Judge failed")
    
    score = evaluator.judge_prediction(example, pred)
    
    assert score == 0.0  # Returns 0 on error
