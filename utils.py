"""CLI utilities."""

import argparse
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Tuple

from config import (
    DEFAULT_LEAD_MODEL,
    DEFAULT_SUB_MODEL,
    WORKSPACE_UUID_LENGTH,
    CLEANUP_WATCHDOG_TIMEOUT_SECONDS,
)


def create_model_cli_parser(
    description: str,
    *,
    query: Tuple[str, str] | None = None,
) -> argparse.ArgumentParser:
    """Return an ArgumentParser with shared model arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--lead", default=DEFAULT_LEAD_MODEL, help="Lead agent model.")
    parser.add_argument("--sub", default=DEFAULT_SUB_MODEL, help="Subagent model.")
    if query is not None:
        default, help_text = query
        parser.add_argument("--query", default=default, help=help_text)
    return parser


def create_isolated_workspace(base_dir: str = "memory_eval") -> Path:
    """Create unique workspace directory for parallel-safe operations."""
    work_dir = Path(base_dir) / str(uuid.uuid4())[:WORKSPACE_UUID_LENGTH]
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def cleanup_workspace(work_dir: Path) -> None:
    """Best-effort cleanup of workspace directory."""
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass


def start_cleanup_watchdog(grace_period_seconds: int = CLEANUP_WATCHDOG_TIMEOUT_SECONDS) -> None:
    """Force exit if cleanup hangs (workaround for DSPy/LiteLLM bug)."""
    def force_exit():
        time.sleep(grace_period_seconds)
        print(f"\n⚠️  Cleanup took >{grace_period_seconds}s, forcing exit")
        os._exit(0)

    threading.Thread(target=force_exit, daemon=True).start()
