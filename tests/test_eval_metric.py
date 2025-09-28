from types import SimpleNamespace

import dspy
import pytest

import eval as eval_module


def test_efficiency_accuracy_metric_basic(monkeypatch):
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
