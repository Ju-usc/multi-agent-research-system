"""
Per-module token usage tracking for multi-agent research system.

Uses DSPy's BaseCallback to capture usage snapshots before/after each module,
compute deltas, and track cost by phase (lead_agent, subagents, grader).

Based on: https://www.elicited.blog/posts/dspy-track-token-usage-per-module
"""

import copy

import dspy
from dspy.utils.callback import BaseCallback

from config import LM_PRICING


class PerModuleUsageTracker(BaseCallback):
    """Track token usage per module phase."""

    def __init__(self):
        self.module_usage = {}
        self.module_trackers = {}

    def on_module_start(self, call_id, instance, inputs):
        path = self._build_module_path(instance, inputs)
        if path:
            self.module_trackers[call_id] = (path, self._get_snapshot())

    def on_module_end(self, call_id, outputs, exception):
        if call_id not in self.module_trackers:
            return
        path, before = self.module_trackers.pop(call_id)
        delta = self._compute_delta(before, self._get_snapshot())
        self._merge_usage(path, delta)

    def _build_module_path(self, instance, inputs):
        """Identify module: lead_agent, subagent_X, or grader."""
        class_name = instance.__class__.__name__

        if class_name == "ReAct":
            task = inputs.get("task")
            if task and hasattr(task, "task_name"):
                return f"subagent_{task.task_name}"
            return "lead_agent"

        if class_name == "ChainOfThought":
            sig = getattr(instance, "signature", None)
            if sig and "BrowseCompJudge" in str(sig):
                return "grader"

        return None

    def _get_snapshot(self):
        """Get current usage from DSPy tracker."""
        tracker = dspy.settings.usage_tracker
        if tracker is None:
            return {}
        return copy.deepcopy(tracker.get_total_tokens())

    def _compute_delta(self, before, after):
        """Compute token delta between snapshots."""
        delta = {}
        for model, stats in after.items():
            prev = before.get(model, {})
            delta[model] = {
                "prompt_tokens": stats.get("prompt_tokens", 0) - prev.get("prompt_tokens", 0),
                "completion_tokens": stats.get("completion_tokens", 0) - prev.get("completion_tokens", 0),
            }
        return delta

    def _merge_usage(self, path, delta):
        """Merge delta into module_usage with cost calculation."""
        if path not in self.module_usage:
            self.module_usage[path] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            }

        for model, tokens in delta.items():
            self.module_usage[path]["prompt_tokens"] += tokens["prompt_tokens"]
            self.module_usage[path]["completion_tokens"] += tokens["completion_tokens"]
            self.module_usage[path]["cost_usd"] += self._calculate_cost(model, tokens)

    def _calculate_cost(self, model, tokens):
        """Calculate cost using LM_PRICING."""
        pricing = LM_PRICING.get(model, {})
        input_cost = (tokens["prompt_tokens"] / 1_000_000) * pricing.get("input", 0)
        output_cost = (tokens["completion_tokens"] / 1_000_000) * pricing.get("output", 0)
        return input_cost + output_cost

    def get_usage(self) -> dict:
        """Return usage breakdown."""
        return self.module_usage

    def reset(self):
        """Reset for new run."""
        self.module_usage.clear()
        self.module_trackers.clear()
