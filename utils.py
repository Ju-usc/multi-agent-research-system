"""CLI utilities."""

import argparse
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Tuple

from config import MODEL_PRESETS


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
