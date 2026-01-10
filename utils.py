import argparse
import os
import threading
import time
import uuid
from pathlib import Path

from config import DEFAULT_LEAD_MODEL, DEFAULT_SUB_MODEL


def create_model_cli_parser(
    description: str, *, query: tuple[str, str] | None = None
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--lead", default=DEFAULT_LEAD_MODEL, help="Lead agent model.")
    parser.add_argument("--sub", default=DEFAULT_SUB_MODEL, help="Subagent model.")
    if query is not None:
        default, help_text = query
        parser.add_argument("--query", default=default, help=help_text)
    return parser


def create_isolated_workspace(base_dir: str = "memory_eval") -> Path:
    work_dir = Path(base_dir) / str(uuid.uuid4())[:8]
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def start_cleanup_watchdog(grace_period_seconds: int = 30) -> None:
    def force_exit():
        time.sleep(grace_period_seconds)
        print(f"\n⚠️  Cleanup took >{grace_period_seconds}s, forcing exit")
        os._exit(0)
    threading.Thread(target=force_exit, daemon=True).start()
