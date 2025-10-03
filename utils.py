"""
Utility functions for the multi-agent research system
"""

import argparse
import os
import json
import logging
from functools import wraps
from typing import Any, Iterable, Tuple

from dotenv import load_dotenv

from config import MODEL_PRESETS


def setup_langfuse():
    """
    Setup Langfuse tracing for DSPy following official documentation.
    
    Returns:
        langfuse client if successful, None otherwise
    """
    # Load environment variables
    load_dotenv()
    
    # Step 1: Set environment variables
    os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-...")
    os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-...")
    os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # Step 2: Initialize Langfuse client
    try:
        from langfuse import get_client
        langfuse = get_client()
        
        # Verify connection
        if langfuse.auth_check():
            print("✅ Langfuse client is authenticated and ready!")
        else:
            print("⚠️  Authentication failed. Please check your credentials and host.")
            return None
            
    except ImportError:
        print("⚠️  Langfuse not available. Run: uv sync")
        return None
    
    # Step 3: Enable tracing for DSPy
    try:
        from openinference.instrumentation.dspy import DSPyInstrumentor
        DSPyInstrumentor().instrument()
        print("✅ DSPy tracing enabled")
    except ImportError:
        print("⚠️  DSPy instrumentation not available. Run: uv sync")
        return None
    
    return langfuse


 


def prediction_to_markdown(obj: Any, title: str | None = None) -> str:
    """Generic Markdown preview (nested bullets) + Raw JSON.

    Single path:
    - Serialize once with a small default serializer.
    - If JSON parses to a dict, render the entire structure as nested bullets
      (dicts/lists handled recursively; scalars inline). No headings, no
      domain keys.
    - Always append a Raw JSON section.
    - The `title` parameter is ignored for content (kept for API compatibility).
    """

    # --- Serialization -----------------------------------------------------
    def _default(o: Any):
        if hasattr(o, "_store") and isinstance(getattr(o, "_store"), dict):
            return o._store
        if hasattr(o, "model_dump") and callable(getattr(o, "model_dump")):
            try:
                return o.model_dump()
            except Exception:
                pass
        return str(o)

    body = json.dumps(obj, default=_default, indent=2, ensure_ascii=False)

    try:
        parsed = json.loads(body)
    except Exception:
        parsed = None

    lines: list[str] = []

    def _choose_fence(text: str) -> str:
        return "~~~" if "```" in text else "```"

    def _render(value: Any, indent: int = 0, key: str | None = None) -> None:
        pad = "  " * indent
        bullet = f"{pad}- "

        if isinstance(value, dict):
            if key is not None:
                lines.append(f"{bullet}{key}:")
                indent += 1
            for k, v in value.items():
                _render(v, indent, k)
            return

        if isinstance(value, list):
            if key is not None:
                lines.append(f"{bullet}{key}:")
                indent += 1
            for i, item in enumerate(value, 1):
                if isinstance(item, (dict, list)):
                    _render(item, indent, f"item {i}")
                else:
                    _render(item, indent)
            return

        if isinstance(value, str):
            if "\n" in value:
                fence = _choose_fence(value)
                if key is not None:
                    lines.append(f"{bullet}{key}:")
                else:
                    lines.append(f"{bullet.rstrip()}")
                pad = "  " * (indent + 1)
                lines.append(f"{pad}{fence}text")
                for line in value.splitlines():
                    lines.append(f"{pad}{line}")
                lines.append(f"{pad}{fence}")
            else:
                if key is not None:
                    lines.append(f"{bullet}{key}: {value}")
                else:
                    lines.append(f"{bullet}{value}")
            return

        txt = json.dumps(value, ensure_ascii=False)
        if key is not None:
            lines.append(f"{bullet}{key}: {txt}")
        else:
            lines.append(f"{bullet}{txt}")

    if isinstance(parsed, dict):
        _render(parsed)

    lines.append("## Raw")
    lines.append("")
    lines.append("```json")
    lines.append(body)
    lines.append("```")

    return "\n".join(lines)


def log_call(func):
    """Log entry and exit of async functions to cut boilerplate."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.info("Starting %s", func.__name__)
        result = await func(*args, **kwargs)
        logger.info("Finished %s", func.__name__)
        return result

    return wrapper


def create_model_cli_parser(
    description: str,
    *,
    include_list: bool = False,
    query: Tuple[str, str] | None = None,
) -> argparse.ArgumentParser:
    """Return an ArgumentParser with shared model arguments.

    Args:
        description: CLI description string.
        include_list: add ``--list-models`` when True.
        query: optional tuple of (default, help) to add ``--query``.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Model preset to use for both big and small slots.",
    )
    parser.add_argument("--model-big", dest="model_big", help="Override the big model identifier.")
    parser.add_argument("--model-small", dest="model_small", help="Override the small model identifier.")
    if include_list:
        parser.add_argument(
            "--list-models",
            action="store_true",
            help="List available model presets and exit.",
        )
    if query is not None:
        default, help_text = query
        parser.add_argument("--query", default=default, help=help_text)
    return parser


def iter_model_presets() -> Iterable[tuple[str, Any]]:
    """Yield model presets sorted by key."""

    return sorted(MODEL_PRESETS.items())


# ========== EXPERIMENT TRACKING ==========

from datetime import datetime
from pathlib import Path
import numpy as np


def save_experiment_results(
    result,
    examples,
    predictions,
    config,
    args,
    output_dir="experiments"
):
    """
    Save comprehensive experiment results with full context.
    Organizes experiments by ID with metadata, results, and summary stats.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{args.model}_{args.num_examples}ex_{timestamp}"
    exp_dir = output_path / exp_id
    exp_dir.mkdir(exist_ok=True)
    
    # 1. Save metadata (experiment config with clear model roles)
    metadata = {
        "experiment_id": exp_id,
        "timestamp": timestamp,
        "config": {
            "model_preset": args.model,
            "lead_agent_model": config.big,      # Clear: which model for lead
            "subagent_model": config.small,      # Clear: which model for subagents
            "num_examples": args.num_examples,
            "metric": args.metric,
            "num_threads": args.num_threads,
        },
        "summary": {
            "score": result.score,
            "total_examples": len(examples),
            "passed": sum(1 for p in predictions if p.get('metrics', {}).get('accuracy', 0) > 0),
            "failed": sum(1 for p in predictions if p.get('metrics', {}).get('accuracy', 0) == 0),
        }
    }
    
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # 2. Save detailed results (all predictions with metrics)
    detailed_results = []
    for example, pred in zip(examples, predictions):
        detailed_results.append({
            "example": {
                "problem": example.problem,
                "answer": example.answer,
            },
            "prediction": pred.toDict(),  # Includes our custom pred.metrics!
        })
    
    with open(exp_dir / "results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    # 3. Save summary statistics (for quick analysis)
    metrics_list = [p.get('metrics', {}) for p in predictions]
    if metrics_list and metrics_list[0]:
        def safe_stats(values):
            vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if not vals:
                return None
            return {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "median": float(np.median(vals)),
            }
        
        summary_stats = {
            "accuracy": safe_stats([m.get('accuracy') for m in metrics_list]),
            "elapsed_seconds": safe_stats([m.get('elapsed_seconds') for m in metrics_list]),
            "total_cost_usd": safe_stats([m.get('total_cost_usd') for m in metrics_list]),
            "lm_cost_usd": safe_stats([m.get('lm_cost_usd') for m in metrics_list]),
            "web_cost_usd": safe_stats([m.get('web_cost_usd') for m in metrics_list]),
        }
        
        with open(exp_dir / "summary_stats.json", "w") as f:
            json.dump(summary_stats, f, indent=2)
    
    # 4. Update experiments manifest (index of all runs)
    manifest_path = output_path / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"experiments": []}
    
    manifest["experiments"].append({
        "experiment_id": exp_id,
        "timestamp": timestamp,
        "model_preset": args.model,
        "lead_agent_model": config.big,
        "subagent_model": config.small,
        "num_examples": args.num_examples,
        "score": result.score,
        "path": str(exp_dir),
    })
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Results saved to: {exp_dir}")
    print(f"   - metadata.json (experiment config)")
    print(f"   - results.json (detailed predictions)")
    print(f"   - summary_stats.json (statistical summary)")
    
    return exp_dir


class ExperimentAnalyzer:
    """Load and analyze saved experiments."""
    
    def __init__(self, experiments_dir="experiments"):
        self.experiments_dir = Path(experiments_dir)
        manifest_path = self.experiments_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {"experiments": []}
    
    def list_experiments(self):
        """List all experiments as DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame(self.manifest["experiments"])
        except ImportError:
            print("⚠️ pandas not available. Install with: uv add pandas")
            return self.manifest["experiments"]
    
    def load_experiment(self, experiment_id):
        """Load full results for an experiment."""
        exp_dir = self.experiments_dir / experiment_id
        
        with open(exp_dir / "metadata.json") as f:
            metadata = json.load(f)
        
        with open(exp_dir / "results.json") as f:
            results = json.load(f)
        
        with open(exp_dir / "summary_stats.json") as f:
            stats = json.load(f)
        
        return {
            "metadata": metadata,
            "results": results,
            "stats": stats,
        }
    
    def aggregate_metrics(self, experiment_ids=None):
        """Aggregate all metrics across experiments into single DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            print("⚠️ pandas not available. Install with: uv add pandas")
            return None
        
        if experiment_ids is None:
            experiment_ids = [e["experiment_id"] for e in self.manifest["experiments"]]
        
        all_metrics = []
        for exp_id in experiment_ids:
            exp = self.load_experiment(exp_id)
            for result in exp["results"]:
                metrics = result["prediction"].get("metrics", {})
                metrics["experiment_id"] = exp_id
                metrics["model_preset"] = exp["metadata"]["config"]["model_preset"]
                metrics["lead_agent_model"] = exp["metadata"]["config"]["lead_agent_model"]
                metrics["subagent_model"] = exp["metadata"]["config"]["subagent_model"]
                all_metrics.append(metrics)
        
        return pd.DataFrame(all_metrics)


def analyze_experiments(experiments_dir="experiments"):
    """
    Analyze all baseline experiments and print summary.
    Convenience function for quick analysis.
    """
    from pathlib import Path
    
    analyzer = ExperimentAnalyzer(experiments_dir)
    
    print("=" * 70)
    print("BASELINE PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # List all experiments
    experiments_df = analyzer.list_experiments()
    print("\n1. ALL EXPERIMENTS")
    if hasattr(experiments_df, 'to_string'):
        print(experiments_df)
    else:
        print(json.dumps(experiments_df, indent=2))
    
    # Analyze each experiment
    for exp_info in analyzer.manifest["experiments"]:
        exp_id = exp_info["experiment_id"]
        print(f"\n" + "=" * 70)
        print(f"EXPERIMENT: {exp_id}")
        print("=" * 70)
        
        # Load experiment (handle missing summary_stats)
        exp_dir = Path(experiments_dir) / exp_id
        with open(exp_dir / "metadata.json") as f:
            metadata = json.load(f)
        with open(exp_dir / "results.json") as f:
            results = json.load(f)
        
        print(f"\nConfiguration:")
        print(f"  Model Preset: {metadata['config']['model_preset']}")
        print(f"  Lead Agent: {metadata['config']['lead_agent_model']}")
        print(f"  Subagent: {metadata['config']['subagent_model']}")
        print(f"  Examples: {metadata['config']['num_examples']}")
        print(f"  Metric: {metadata['config']['metric']}")
        
        print(f"\nResults:")
        print(f"  Score: {metadata['summary']['score']:.2%}")
        print(f"  Passed: {metadata['summary']['passed']}")
        print(f"  Failed: {metadata['summary']['failed']}")
        
        # Extract raw metrics from predictions
        times = []
        web_calls = []
        has_metrics = []
        
        for result in results:
            pred = result["prediction"]
            # Check for metrics field first
            if "metrics" in pred:
                has_metrics.append(True)
                metrics = pred["metrics"]
                if "elapsed_seconds" in metrics:
                    times.append(metrics["elapsed_seconds"])
                if "websearch_calls" in metrics:
                    web_calls.append(metrics["websearch_calls"])
            else:
                has_metrics.append(False)
                # Fall back to top-level fields
                if "elapsed_seconds" in pred:
                    times.append(pred["elapsed_seconds"])
                if "websearch_calls" in pred:
                    web_calls.append(pred["websearch_calls"])
        
        if times:
            print(f"\nTime Distribution (seconds):")
            print(f"  Mean: {np.mean(times):.2f}s")
            print(f"  Median: {np.median(times):.2f}s")
            print(f"  Std: {np.std(times):.2f}s")
            print(f"  Min: {np.min(times):.2f}s")
            print(f"  Max: {np.max(times):.2f}s")
            print(f"  Range: {np.max(times) / max(np.min(times), 0.001):.1f}x")
        
        if web_calls:
            print(f"\nWeb Search Calls:")
            print(f"  Mean: {np.mean(web_calls):.1f}")
            print(f"  Median: {np.median(web_calls):.1f}")
            print(f"  Total: {sum(web_calls)}")
        
        # Diagnostic info
        metrics_count = sum(has_metrics)
        if metrics_count < len(results):
            print(f"\n⚠️  WARNING: Only {metrics_count}/{len(results)} predictions have 'metrics' field")
            print(f"   This suggests predictions were re-run after evaluation without capturing metrics.")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total experiments: {len(analyzer.manifest['experiments'])}")
    print("\nDIAGNOSTIC INFO:")
    print("- If elapsed_seconds ~0.03s and websearch_calls=0: predictions are fresh runs without real execution")
    print("- If 'metrics' field missing: predictions weren't captured during evaluation phase")
    print("- Real agent execution should take 30-120s and make 5-20 web search calls")
    
    return analyzer


def start_cleanup_watchdog(grace_period_seconds: int = 30) -> None:
    """Start a watchdog timer that forces exit if cleanup hangs.
    
    This prevents the process from hanging indefinitely during executor cleanup
    (e.g., DSPy's ParallelExecutor shutdown interacting with LiteLLM callbacks).
    
    Call this AFTER all important results are saved. The watchdog runs in a daemon
    thread and will force-exit the process if normal cleanup takes too long.
    
    Args:
        grace_period_seconds: How long to wait before forcing exit (default: 30s)
    """
    import threading
    import time
    import os
    
    def force_exit():
        time.sleep(grace_period_seconds)
        print(f"\n⚠️  Cleanup took >{grace_period_seconds}s, forcing exit")
        print("   (Results already saved - this is just stuck cleanup)")
        os._exit(0)  # Hard exit, bypass hanging cleanup
    
    watchdog = threading.Thread(target=force_exit, daemon=True)
    watchdog.start()
