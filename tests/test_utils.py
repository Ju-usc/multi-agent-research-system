"""Tests for utils module."""

import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from utils import (
    create_model_cli_parser,
    create_isolated_workspace,
    cleanup_workspace,
    start_cleanup_watchdog,
)
from config import DEFAULT_LEAD_MODEL, DEFAULT_SUB_MODEL


def test_create_model_cli_parser_defaults():
    parser = create_model_cli_parser("Test parser")
    args = parser.parse_args([])

    assert args.lead == DEFAULT_LEAD_MODEL
    assert args.sub == DEFAULT_SUB_MODEL


def test_create_model_cli_parser_overrides():
    parser = create_model_cli_parser("Test parser")
    args = parser.parse_args(["--lead", "custom/lead", "--sub", "custom/sub"])

    assert args.lead == "custom/lead"
    assert args.sub == "custom/sub"


def test_create_model_cli_parser_with_query():
    parser = create_model_cli_parser(
        "Test parser",
        query=("default query", "Query help text"),
    )
    args = parser.parse_args([])
    assert args.query == "default query"


def test_create_model_cli_parser_with_query_override():
    parser = create_model_cli_parser(
        "Test parser",
        query=(None, "Query help text"),
    )
    args = parser.parse_args(["--query", "my query"])
    assert args.query == "my query"


def test_create_isolated_workspace(tmp_path):
    workspace = create_isolated_workspace(str(tmp_path))

    assert workspace.exists()
    assert workspace.parent == tmp_path
    # Should be a uuid-like directory name (8 chars)
    assert len(workspace.name) == 8


def test_create_isolated_workspace_unique():
    workspaces = [create_isolated_workspace() for _ in range(3)]
    # All should be unique
    assert len(set(workspaces)) == 3


def test_cleanup_workspace(tmp_path):
    workspace = tmp_path / "to_clean"
    workspace.mkdir()
    (workspace / "file.txt").write_text("content")

    cleanup_workspace(workspace)

    assert not workspace.exists()


def test_cleanup_workspace_nonexistent(tmp_path):
    workspace = tmp_path / "nonexistent"
    # Should not raise
    cleanup_workspace(workspace)


def test_cleanup_workspace_permission_error(tmp_path, monkeypatch):
    """Cleanup should handle errors gracefully."""
    workspace = tmp_path / "protected"
    workspace.mkdir()

    def fail_rmtree(path):
        raise PermissionError("cannot remove")

    monkeypatch.setattr("shutil.rmtree", fail_rmtree)

    # Should not raise
    cleanup_workspace(workspace)


def test_start_cleanup_watchdog_runs_in_background():
    """Verify watchdog thread starts without blocking."""
    # Use a very long grace period so it doesn't actually exit
    start_cleanup_watchdog(grace_period_seconds=9999)

    # Verify a daemon thread was started
    threads = [t for t in threading.enumerate() if t.daemon]
    # At least one daemon thread should exist (the watchdog)
    assert len(threads) >= 1
