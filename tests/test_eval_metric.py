import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import dspy
import pytest

import eval as eval_module


def test_efficiency_accuracy_metric_basic(monkeypatch):
    """Test efficiency metric with correct answer and typical costs."""
    monkeypatch.setattr(eval_module, "LM_COST_PER_1K_TOKENS", {"model": 0.2}, raising=False)
    monkeypatch.setattr(eval_module, "WEBSEARCH_COST_PER_CALL_USD", 0.01, raising=False)
    monkeypatch.setattr(
        eval_module.dspy,
        "ChainOfThought",
        lambda *_, **__: lambda **__: SimpleNamespace(is_correct=True),
        raising=False,
    )

    example = dspy.Example(problem="Q", answer="A")

    class Prediction(SimpleNamespace):
        def get_lm_usage(self):
            return {"model": {"total_tokens": 100}}

    pred = Prediction(report="A", elapsed_seconds=2.0, websearch_calls=1)

    score = eval_module.efficiency_accuracy_metric(example, pred)

    assert score == pytest.approx(1.0 / (2.0 * 0.03))
    breakdown = pred.efficiency_breakdown
    assert breakdown["accuracy"] == 1.0
    assert breakdown["elapsed_seconds"] == pytest.approx(2.0)
    assert breakdown["lm_cost_usd"] == pytest.approx(0.02)
    assert breakdown["web_cost_usd"] == pytest.approx(0.01)
    assert breakdown["total_cost_usd"] == pytest.approx(0.03)
    assert breakdown["score"] == pytest.approx(score)


def test_efficiency_accuracy_metric_incorrect_answer(monkeypatch):
    """Test efficiency metric returns 0 for incorrect answers."""
    monkeypatch.setattr(eval_module, "LM_COST_PER_1K_TOKENS", {"model": 0.2}, raising=False)
    monkeypatch.setattr(eval_module, "WEBSEARCH_COST_PER_CALL_USD", 0.01, raising=False)
    monkeypatch.setattr(
        eval_module.dspy,
        "ChainOfThought",
        lambda *_, **__: lambda **__: SimpleNamespace(is_correct=False),
        raising=False,
    )

    example = dspy.Example(problem="Q", answer="A")

    class Prediction(SimpleNamespace):
        def get_lm_usage(self):
            return {"model": {"total_tokens": 100}}

    pred = Prediction(report="Wrong", elapsed_seconds=2.0, websearch_calls=1)

    score = eval_module.efficiency_accuracy_metric(example, pred)

    # When incorrect, score is 0 and breakdown is still set
    assert score == 0.0
    assert hasattr(pred, "efficiency_breakdown")
    assert pred.efficiency_breakdown["accuracy"] == 0.0


def test_efficiency_accuracy_metric_free_model(monkeypatch):
    """Test efficiency metric with zero-cost model."""
    monkeypatch.setattr(eval_module, "LM_COST_PER_1K_TOKENS", {"free-model": 0.0}, raising=False)
    monkeypatch.setattr(eval_module, "WEBSEARCH_COST_PER_CALL_USD", 0.0, raising=False)
    monkeypatch.setattr(
        eval_module.dspy,
        "ChainOfThought",
        lambda *_, **__: lambda **__: SimpleNamespace(is_correct=True),
        raising=False,
    )

    example = dspy.Example(problem="Q", answer="A")

    class Prediction(SimpleNamespace):
        def get_lm_usage(self):
            return {"free-model": {"total_tokens": 100}}

    pred = Prediction(report="A", elapsed_seconds=1.5, websearch_calls=0)

    score = eval_module.efficiency_accuracy_metric(example, pred)

    # With epsilon protection, score = 1.0 / (1.5 * 1e-6)
    assert score > 0
    breakdown = pred.efficiency_breakdown
    assert breakdown["lm_cost_usd"] == 0.0
    assert breakdown["web_cost_usd"] == 0.0
    assert breakdown["total_cost_usd"] == 0.0


def test_efficiency_accuracy_metric_missing_usage(monkeypatch):
    """Test efficiency metric handles missing usage data gracefully."""
    monkeypatch.setattr(eval_module, "LM_COST_PER_1K_TOKENS", {}, raising=False)
    monkeypatch.setattr(eval_module, "WEBSEARCH_COST_PER_CALL_USD", 0.01, raising=False)
    monkeypatch.setattr(
        eval_module.dspy,
        "ChainOfThought",
        lambda *_, **__: lambda **__: SimpleNamespace(is_correct=True),
        raising=False,
    )

    example = dspy.Example(problem="Q", answer="A")

    class Prediction(SimpleNamespace):
        def get_lm_usage(self):
            return None

    pred = Prediction(report="A", elapsed_seconds=2.0, websearch_calls=2)

    score = eval_module.efficiency_accuracy_metric(example, pred)

    # Should handle None usage gracefully
    assert score > 0
    breakdown = pred.efficiency_breakdown
    assert breakdown["lm_cost_usd"] == 0.0
    assert breakdown["web_cost_usd"] == 0.02


def test_efficiency_accuracy_metric_separate_prompt_completion_tokens(monkeypatch):
    """Test efficiency metric correctly sums prompt + completion tokens."""
    monkeypatch.setattr(eval_module, "LM_COST_PER_1K_TOKENS", {"model": 0.1}, raising=False)
    monkeypatch.setattr(eval_module, "WEBSEARCH_COST_PER_CALL_USD", 0.0, raising=False)
    monkeypatch.setattr(
        eval_module.dspy,
        "ChainOfThought",
        lambda *_, **__: lambda **__: SimpleNamespace(is_correct=True),
        raising=False,
    )

    example = dspy.Example(problem="Q", answer="A")

    class Prediction(SimpleNamespace):
        def get_lm_usage(self):
            # total_tokens missing, should sum prompt + completion
            return {"model": {"prompt_tokens": 60, "completion_tokens": 40}}

    pred = Prediction(report="A", elapsed_seconds=1.0, websearch_calls=0)

    score = eval_module.efficiency_accuracy_metric(example, pred)

    breakdown = pred.efficiency_breakdown
    # 100 tokens * 0.1 per 1k = 0.01
    assert breakdown["lm_cost_usd"] == pytest.approx(0.01)


def test_browsecomp_metric_correct(monkeypatch):
    """Test basic accuracy metric with correct answer."""
    monkeypatch.setattr(
        eval_module.dspy,
        "ChainOfThought",
        lambda *_, **__: lambda **__: SimpleNamespace(is_correct=True),
        raising=False,
    )

    example = dspy.Example(problem="What is 2+2?", answer="4")
    pred = dspy.Prediction(report="The answer is 4")

    score = eval_module.browsecomp_metric(example, pred)

    assert score == 1.0


def test_browsecomp_metric_incorrect(monkeypatch):
    """Test basic accuracy metric with incorrect answer."""
    monkeypatch.setattr(
        eval_module.dspy,
        "ChainOfThought",
        lambda *_, **__: lambda **__: SimpleNamespace(is_correct=False),
        raising=False,
    )

    example = dspy.Example(problem="What is 2+2?", answer="4")
    pred = dspy.Prediction(report="The answer is 5")

    score = eval_module.browsecomp_metric(example, pred)

    assert score == 0.0


def test_browsecomp_metric_error_handling(monkeypatch):
    """Test accuracy metric handles judge errors gracefully."""
    def failing_judge(*_, **__):
        raise ValueError("Judge failed")

    monkeypatch.setattr(
        eval_module.dspy,
        "ChainOfThought",
        lambda *_, **__: failing_judge,
        raising=False,
    )

    example = dspy.Example(problem="Q", answer="A")
    pred = dspy.Prediction(report="A")

    score = eval_module.browsecomp_metric(example, pred)

    assert score == 0.0
