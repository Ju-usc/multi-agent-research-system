"""Tests for per-module usage tracking."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tracking import PerModuleUsageTracker


class TestComputeDelta:
    """Tests for _compute_delta method."""

    def test_basic_delta(self):
        tracker = PerModuleUsageTracker()
        before = {
            "openai/gpt-5": {"prompt_tokens": 100, "completion_tokens": 50}
        }
        after = {
            "openai/gpt-5": {"prompt_tokens": 200, "completion_tokens": 80}
        }

        delta = tracker._compute_delta(before, after)

        assert delta["openai/gpt-5"]["prompt_tokens"] == 100
        assert delta["openai/gpt-5"]["completion_tokens"] == 30

    def test_new_model_in_after(self):
        tracker = PerModuleUsageTracker()
        before = {}
        after = {
            "openai/gpt-5": {"prompt_tokens": 100, "completion_tokens": 50}
        }

        delta = tracker._compute_delta(before, after)

        assert delta["openai/gpt-5"]["prompt_tokens"] == 100
        assert delta["openai/gpt-5"]["completion_tokens"] == 50

    def test_multiple_models(self):
        tracker = PerModuleUsageTracker()
        before = {
            "model-a": {"prompt_tokens": 100, "completion_tokens": 50},
            "model-b": {"prompt_tokens": 200, "completion_tokens": 100},
        }
        after = {
            "model-a": {"prompt_tokens": 150, "completion_tokens": 60},
            "model-b": {"prompt_tokens": 300, "completion_tokens": 150},
        }

        delta = tracker._compute_delta(before, after)

        assert delta["model-a"]["prompt_tokens"] == 50
        assert delta["model-b"]["prompt_tokens"] == 100


class TestBuildModulePath:
    """Tests for _build_module_path method."""

    def test_lead_agent_react(self):
        tracker = PerModuleUsageTracker()
        instance = MagicMock()
        instance.__class__.__name__ = "ReAct"

        path = tracker._build_module_path(instance, {"query": "test"})

        assert path == "lead_agent"

    def test_subagent_react_with_task(self):
        tracker = PerModuleUsageTracker()
        instance = MagicMock()
        instance.__class__.__name__ = "ReAct"
        task = SimpleNamespace(task_name="market-research")

        path = tracker._build_module_path(instance, {"task": task})

        assert path == "subagent_market-research"

    def test_grader_chain_of_thought(self):
        tracker = PerModuleUsageTracker()
        instance = MagicMock()
        instance.__class__.__name__ = "ChainOfThought"
        instance.signature = "BrowseCompJudge"

        path = tracker._build_module_path(instance, {})

        assert path == "grader"

    def test_untracked_module(self):
        tracker = PerModuleUsageTracker()
        instance = MagicMock()
        instance.__class__.__name__ = "SomeOtherModule"

        path = tracker._build_module_path(instance, {})

        assert path is None


class TestCalculateCost:
    """Tests for _calculate_cost method."""

    def test_known_model(self):
        tracker = PerModuleUsageTracker()
        tokens = {"prompt_tokens": 1000, "completion_tokens": 500}

        # openai/gpt-5: input=$1.25/M, output=$10.00/M
        cost = tracker._calculate_cost("openai/gpt-5", tokens)

        # 1000/1M * 1.25 = 0.00125
        # 500/1M * 10.00 = 0.005
        # Total: 0.00625
        assert cost == pytest.approx(0.00625)

    def test_unknown_model(self):
        tracker = PerModuleUsageTracker()
        tokens = {"prompt_tokens": 1000, "completion_tokens": 500}

        cost = tracker._calculate_cost("unknown-model", tokens)

        assert cost == 0.0


class TestMergeUsage:
    """Tests for _merge_usage method."""

    def test_new_path(self):
        tracker = PerModuleUsageTracker()
        delta = {"openai/gpt-5": {"prompt_tokens": 100, "completion_tokens": 50}}

        tracker._merge_usage("lead_agent", delta)

        assert "lead_agent" in tracker.module_usage
        assert tracker.module_usage["lead_agent"]["prompt_tokens"] == 100
        assert tracker.module_usage["lead_agent"]["completion_tokens"] == 50

    def test_existing_path_accumulates(self):
        tracker = PerModuleUsageTracker()
        tracker.module_usage["lead_agent"] = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost_usd": 0.01,
        }
        delta = {"openai/gpt-5": {"prompt_tokens": 100, "completion_tokens": 50}}

        tracker._merge_usage("lead_agent", delta)

        assert tracker.module_usage["lead_agent"]["prompt_tokens"] == 200
        assert tracker.module_usage["lead_agent"]["completion_tokens"] == 100


class TestFullFlow:
    """Integration tests for the full tracking flow."""

    def test_on_module_start_and_end(self):
        tracker = PerModuleUsageTracker()

        # Mock instance
        instance = MagicMock()
        instance.__class__.__name__ = "ReAct"

        # Mock _get_snapshot to return controlled values
        snapshots = [
            {"openai/gpt-5": {"prompt_tokens": 0, "completion_tokens": 0}},
            {"openai/gpt-5": {"prompt_tokens": 100, "completion_tokens": 50}},
        ]
        tracker._get_snapshot = MagicMock(side_effect=snapshots)

        # Simulate module execution
        tracker.on_module_start("call-1", instance, {"query": "test"})
        tracker.on_module_end("call-1", {}, None)

        # Verify tracking
        assert "lead_agent" in tracker.module_usage
        assert tracker.module_usage["lead_agent"]["prompt_tokens"] == 100
        assert tracker.module_usage["lead_agent"]["completion_tokens"] == 50

    def test_get_usage_returns_module_usage(self):
        tracker = PerModuleUsageTracker()
        tracker.module_usage = {"test": {"prompt_tokens": 100}}

        result = tracker.get_usage()

        assert result == {"test": {"prompt_tokens": 100}}

    def test_reset_clears_state(self):
        tracker = PerModuleUsageTracker()
        tracker.module_usage = {"test": {}}
        tracker.module_trackers = {"call-1": ("test", {})}

        tracker.reset()

        assert tracker.module_usage == {}
        assert tracker.module_trackers == {}
