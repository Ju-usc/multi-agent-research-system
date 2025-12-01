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
from utils import calculate_lm_cost


@pytest.fixture
def mock_args():
    """Create mock args for BrowseCompEvaluator."""
    args = SimpleNamespace(
        metric="efficiency",
        optimize=False,
        num_threads=1,
    )
    return args


@pytest.fixture
def mock_config():
    """Create mock config for BrowseCompEvaluator."""
    config = SimpleNamespace(
        big="gpt-4o-mini",
        small="gpt-4o-mini",
        big_max_tokens=4096,
        small_max_tokens=4096,
    )
    return config


@pytest.fixture
def evaluator(mock_config, mock_args, monkeypatch):
    """Create BrowseCompEvaluator with mocked LM initialization."""
    # Mock dspy.LM to avoid actual API calls
    mock_lm = MagicMock()
    monkeypatch.setattr("eval.dspy.LM", lambda **kwargs: mock_lm)
    
    # Mock dspy.ChainOfThought
    mock_judge = MagicMock()
    monkeypatch.setattr("eval.dspy.ChainOfThought", lambda sig: mock_judge)
    
    evaluator = BrowseCompEvaluator(mock_args)
    evaluator.judge = mock_judge
    return evaluator


def test_calculate_lm_cost_basic():
    """Test LM cost calculation with basic token usage."""
    usage = {
        "openrouter/openai/gpt-5-mini": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 0}
        }
    }
    
    cost = calculate_lm_cost(usage)
    
    # openrouter/openai/gpt-5-mini: $0.25 per 1M input, $2.00 per 1M output
    # 1000 tokens / 1M * $0.25 = $0.00025
    # 500 tokens / 1M * $2.00 = $0.001
    # Total: $0.00125
    assert cost == pytest.approx(0.00125)


def test_calculate_lm_cost_with_caching():
    """Test LM cost calculation with cached tokens."""
    usage = {
        "openrouter/openai/gpt-5-mini": {
            "prompt_tokens": 2000,
            "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 1000}
        }
    }
    
    cost = calculate_lm_cost(usage)
    
    # openrouter/openai/gpt-5-mini: $0.25 per 1M input, $0.025 per 1M cached, $2.00 per 1M output
    # Non-cached: 1000 tokens / 1M * $0.25 = $0.00025
    # Cached: 1000 tokens / 1M * $0.025 = $0.000025
    # Output: 500 tokens / 1M * $2.00 = $0.001
    # Total: $0.001275
    assert cost == pytest.approx(0.001275)


def test_calculate_lm_cost_unknown_model():
    """Test LM cost calculation gracefully handles unknown models."""
    usage = {
        "unknown-model": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        }
    }
    
    cost = calculate_lm_cost(usage)
    
    # Unknown model should log warning and return 0 cost
    assert cost == 0.0


def test_calculate_lm_cost_multiple_models():
    """Test LM cost calculation with multiple models."""
    usage = {
        "openrouter/openai/gpt-5-mini": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 0}
        },
        "openrouter/openai/gpt-5": {
            "prompt_tokens": 500,
            "completion_tokens": 200,
            "prompt_tokens_details": {"cached_tokens": 0}
        }
    }
    
    cost = calculate_lm_cost(usage)
    
    # gpt-5-mini: (1000/1M * $0.25) + (500/1M * $2.00) = $0.00025 + $0.001 = $0.00125
    # gpt-5: (500/1M * $1.25) + (200/1M * $10.00) = $0.000625 + $0.002 = $0.002625
    # Total: $0.003875
    assert cost == pytest.approx(0.003875)


def test_calculate_metrics_correct_answer(evaluator):
    """Test calculate_metrics with correct answer."""
    example = dspy.Example(problem="What is 2+2?", answer="4")
    pred = dspy.Prediction(report="The answer is 4")
    pred.elapsed_seconds = 2.0
    pred.websearch_calls = 1
    pred.get_lm_usage = lambda: {
        "openrouter/openai/gpt-5-mini": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 0}
        }
    }
    
    # Mock judge to return correct
    evaluator.judge_prediction = MagicMock(return_value=1.0)
    
    metrics = evaluator.calculate_metrics(example, pred)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["elapsed_seconds"] == 2.0
    assert metrics["lm_cost_usd"] == pytest.approx(0.00125)  # $0.00125 as calculated above
    assert metrics["web_cost_usd"] == pytest.approx(WEBSEARCH_COST_PER_CALL_USD)
    assert metrics["total_cost_usd"] == pytest.approx(0.00125 + WEBSEARCH_COST_PER_CALL_USD)
    assert metrics["websearch_calls"] == 1
    assert "efficiency_temp" in metrics


def test_calculate_metrics_incorrect_answer(evaluator):
    """Test calculate_metrics with incorrect answer."""
    example = dspy.Example(problem="What is 2+2?", answer="4")
    pred = dspy.Prediction(report="The answer is 5")
    pred.elapsed_seconds = 1.0
    pred.websearch_calls = 0
    pred.get_lm_usage = lambda: {}
    
    # Mock judge to return incorrect
    evaluator.judge_prediction = MagicMock(return_value=0.0)
    
    metrics = evaluator.calculate_metrics(example, pred)
    
    assert metrics["accuracy"] == 0.0
    assert metrics["efficiency_temp"] == 0.0  # Efficiency is 0 when incorrect


def test_accuracy_metric(evaluator):
    """Test accuracy_metric returns judge result."""
    example = dspy.Example(problem="Q", answer="A")
    pred = dspy.Prediction(report="A")
    
    # Mock judge to return 1.0
    evaluator.judge_prediction = MagicMock(return_value=1.0)
    
    score = evaluator.accuracy_metric(example, pred)
    
    assert score == 1.0
    evaluator.judge_prediction.assert_called_once_with(example, pred)


def test_efficiency_metric_stores_metrics(evaluator):
    """Test efficiency_metric stores full metrics in prediction."""
    example = dspy.Example(problem="Q", answer="A")
    pred = dspy.Prediction(report="A")
    pred.elapsed_seconds = 1.0
    pred.websearch_calls = 1
    pred.get_lm_usage = lambda: {}
    
    # Mock calculate_metrics
    expected_metrics = {
        "accuracy": 1.0,
        "elapsed_seconds": 1.0,
        "total_cost_usd": 0.01,
        "lm_cost_usd": 0.0,
        "web_cost_usd": 0.01,
        "websearch_calls": 1,
        "lm_usage": {},
        "efficiency_temp": 100.0,
    }
    evaluator.calculate_metrics = MagicMock(return_value=expected_metrics)
    
    score = evaluator.efficiency_metric(example, pred)
    
    assert score == 1.0  # Returns accuracy
    assert pred.metrics == expected_metrics  # Stores full metrics


def test_judge_prediction_error_handling(evaluator):
    """Test judge_prediction handles errors gracefully."""
    example = dspy.Example(problem="Q", answer="A")
    pred = dspy.Prediction(report="A")
    
    # Mock judge to raise exception
    evaluator.judge = MagicMock(side_effect=Exception("Judge failed"))
    
    score = evaluator.judge_prediction(example, pred)
    
    assert score == 0.0  # Returns 0 on error
