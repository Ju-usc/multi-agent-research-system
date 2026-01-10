"""Tests for BrowseComp evaluation metrics."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import dspy
import pytest

from eval import BrowseCompEvaluator
from models import LLMJudgeAnswer


@pytest.fixture
def mock_judge():
    return MagicMock()


@pytest.fixture
def evaluator(mock_judge, monkeypatch):
    args = SimpleNamespace(optimize=False, num_threads=1)
    monkeypatch.setattr("eval.dspy.LM", lambda **kw: MagicMock())
    monkeypatch.setattr("eval.dspy.ChainOfThought", lambda sig: mock_judge)
    return BrowseCompEvaluator(args)


@pytest.fixture
def make_prediction():
    """Factory for creating test predictions."""
    def _make(answer="A", usage=None):
        pred = dspy.Prediction(answer=answer, report=answer)
        pred.elapsed_seconds = 1.0
        pred.tool_cost_usd = 0.0
        pred.get_lm_usage = lambda: usage or {}
        return pred
    return _make


class TestCalculateLmCost:
    def test_basic(self, evaluator):
        usage = {
            "openrouter/x-ai/grok-4.1-fast": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        }
        # grok: $0.20/1M input + $0.50/1M output = $0.00045
        assert evaluator.calculate_lm_cost(usage) == pytest.approx(0.00045)

    def test_with_caching(self, evaluator):
        usage = {
            "openrouter/x-ai/grok-4.1-fast": {
                "prompt_tokens": 2000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 1000},
            }
        }
        # No cache discount for grok: $0.0004 + $0.00025 = $0.00065
        assert evaluator.calculate_lm_cost(usage) == pytest.approx(0.00065)

    def test_unknown_model_raises(self, evaluator):
        with pytest.raises(KeyError):
            evaluator.calculate_lm_cost({"unknown": {"prompt_tokens": 100, "completion_tokens": 50}})

    def test_multiple_models(self, evaluator):
        usage = {
            "openrouter/x-ai/grok-4.1-fast": {
                "prompt_tokens": 1000, "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "openrouter/deepseek/deepseek-v3.2": {
                "prompt_tokens": 500, "completion_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        # grok: $0.00045 + deepseek: $0.000196 = $0.000646
        assert evaluator.calculate_lm_cost(usage) == pytest.approx(0.000646)


class TestMetric:
    def test_correct_answer(self, evaluator, mock_judge, make_prediction):
        example = dspy.Example(problem="2+2?", answer="4")
        pred = make_prediction(answer="4")
        mock_judge.return_value = SimpleNamespace(
            answer=LLMJudgeAnswer(is_correct=True, extracted_answer="4", reasoning="Correct")
        )

        result = evaluator.metric(example, pred)

        assert float(result) > 0.9
        assert pred.metrics["accuracy"] == 1.0

    def test_incorrect_answer(self, evaluator, mock_judge, make_prediction):
        example = dspy.Example(problem="2+2?", answer="4")
        pred = make_prediction(answer="5")
        mock_judge.return_value = SimpleNamespace(
            answer=LLMJudgeAnswer(is_correct=False, extracted_answer="5", reasoning="Wrong")
        )

        result = evaluator.metric(example, pred)

        assert float(result) == 0.0
        assert pred.metrics["accuracy"] == 0.0

    def test_judge_error_returns_zero(self, evaluator, mock_judge, make_prediction):
        example = dspy.Example(problem="Q", answer="A")
        pred = make_prediction()
        mock_judge.side_effect = Exception("Judge failed")

        result = evaluator.metric(example, pred)

        assert float(result) == 0.0
        assert "Grading failed" in result.feedback
