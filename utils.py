"""CLI utilities."""

import argparse
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Tuple

from config import MODEL_PRESETS, LM_PRICING

logger = logging.getLogger(__name__)


def create_model_cli_parser(
    description: str,
    *,
    include_list: bool = False,
    query: Tuple[str, str] | None = None,
) -> argparse.ArgumentParser:
    """Return an ArgumentParser with shared model arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Model preset to use for both big and small slots.",
    )
    parser.add_argument("--model-big", dest="model_big", help="Override the big model identifier.")
    parser.add_argument("--model-small", dest="model_small", help="Override the small model identifier.")
    if include_list:
        parser.add_argument("--list-models", action="store_true", help="List available model presets and exit.")
    if query is not None:
        default, help_text = query
        parser.add_argument("--query", default=default, help=help_text)
    return parser


def iter_model_presets() -> Iterable[tuple[str, Any]]:
    """Yield model presets sorted by key."""
    return sorted(MODEL_PRESETS.items())


def create_isolated_workspace(base_dir: str = "memory_eval") -> Path:
    """Create unique workspace directory for parallel-safe operations."""
    work_dir = Path(base_dir) / str(uuid.uuid4())[:8]
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def cleanup_workspace(work_dir: Path) -> None:
    """Best-effort cleanup of workspace directory."""
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass


def start_cleanup_watchdog(grace_period_seconds: int = 30) -> None:
    """Force exit if cleanup hangs (workaround for DSPy/LiteLLM bug)."""
    def force_exit():
        time.sleep(grace_period_seconds)
        print(f"\n⚠️  Cleanup took >{grace_period_seconds}s, forcing exit")
        os._exit(0)

    threading.Thread(target=force_exit, daemon=True).start()


def calculate_lm_cost(usage: dict) -> float:
    """Calculate LM cost with accurate input/output/cached token pricing.
    
    Pricing in LM_PRICING is per 1M tokens (industry standard).
    Formula: (tokens / 1,000,000) * price_per_1M = cost in USD
    """
    total_cost = 0.0
    
    for model_name, stats in usage.items():
        pricing = LM_PRICING.get(model_name, {})
        if not pricing:
            logger.warning(f"No pricing configured for model: {model_name}")
            continue
        
        prompt_tokens = stats.get("prompt_tokens", 0)
        completion_tokens = stats.get("completion_tokens", 0)
        prompt_details = stats.get("prompt_tokens_details", {})
        # Handle case where prompt_details is int instead of dict (some providers)
        if isinstance(prompt_details, dict):
            cached_tokens = prompt_details.get("cached_tokens", 0)
        else:
            cached_tokens = 0
        non_cached_input = prompt_tokens - cached_tokens
        
        # Pricing is per 1M tokens, so divide by 1,000,000
        input_cost = (non_cached_input / 1_000_000) * pricing.get("input", 0.0)
        cached_cost = (cached_tokens / 1_000_000) * pricing.get("cached_input", pricing.get("input", 0.0))
        output_cost = (completion_tokens / 1_000_000) * pricing.get("output", 0.0)
        
        total_cost += input_cost + cached_cost + output_cost
    
    return total_cost
