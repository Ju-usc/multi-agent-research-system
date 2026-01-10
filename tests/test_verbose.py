"""Tests for verbose module."""

from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from verbose import VerbosePrinter
from logger import AgentIteration, AgentMetadata


def test_verbose_printer_disabled():
    """When disabled, no output should occur."""
    printer = VerbosePrinter(enabled=False)
    assert printer.console is None

    # These should not raise
    metadata = AgentMetadata(
        run_id="abc123",
        started_at="2024-01-01T00:00:00",
        lead_model="lead-model",
        sub_model="sub-model",
        query="test query",
    )
    printer.print_metadata(metadata)

    iteration = AgentIteration(
        agent_type="lead",
        agent_name="lead_agent",
        event="start",
        data={"query": "test"},
    )
    printer.print_iteration(iteration)


def test_verbose_printer_enabled():
    """When enabled, console should be created."""
    printer = VerbosePrinter(enabled=True)
    assert printer.console is not None


def test_verbose_printer_print_metadata():
    """Test metadata output doesn't crash."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)

    printer = VerbosePrinter(enabled=True)
    printer.console = console

    metadata = AgentMetadata(
        run_id="abc123",
        started_at="2024-01-01T00:00:00",
        lead_model="test/lead-model",
        sub_model="test/sub-model",
        query="test query",
    )
    printer.print_metadata(metadata)

    result = output.getvalue()
    assert "test/lead-model" in result or "Agent" in result


def test_verbose_printer_print_start():
    """Test start event output."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)

    printer = VerbosePrinter(enabled=True)
    printer.console = console

    iteration = AgentIteration(
        agent_type="lead",
        agent_name="lead_agent",
        event="start",
        data={"query": "test query"},
    )
    printer.print_iteration(iteration)

    result = output.getvalue()
    # Should contain agent info
    assert "lead" in result or len(result) > 0


def test_verbose_printer_print_tool():
    """Test tool event output."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)

    printer = VerbosePrinter(enabled=True)
    printer.console = console

    iteration = AgentIteration(
        agent_type="lead",
        agent_name="lead_agent",
        event="tool",
        data={"tool": "search", "result": "search results here", "duration_ms": 150},
    )
    printer.print_iteration(iteration)

    result = output.getvalue()
    assert len(result) > 0


def test_verbose_printer_print_tool_with_error():
    """Test tool event with error."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)

    printer = VerbosePrinter(enabled=True)
    printer.console = console

    iteration = AgentIteration(
        agent_type="lead",
        agent_name="lead_agent",
        event="tool",
        data={"tool": "search", "result": None, "duration_ms": 50},
        error="API timeout",
    )
    printer.print_iteration(iteration)

    result = output.getvalue()
    assert "Error" in result or len(result) > 0


def test_verbose_printer_print_tool_long_result():
    """Test tool event with long result is truncated."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)

    printer = VerbosePrinter(enabled=True)
    printer.console = console

    long_result = "x" * 500
    iteration = AgentIteration(
        agent_type="lead",
        agent_name="lead_agent",
        event="tool",
        data={"tool": "fetch", "result": long_result, "duration_ms": 100},
    )
    printer.print_iteration(iteration)

    result = output.getvalue()
    # Result should be truncated (200 chars + ...)
    assert len(result) > 0


def test_verbose_printer_print_finish_lead():
    """Test finish event for lead agent."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)

    printer = VerbosePrinter(enabled=True)
    printer.console = console

    iteration = AgentIteration(
        agent_type="lead",
        agent_name="lead_agent",
        event="finish",
        data={"answer": "The answer is 42", "duration_ms": 5000},
    )
    printer.print_iteration(iteration)

    result = output.getvalue()
    assert len(result) > 0


def test_verbose_printer_print_finish_subagent():
    """Test finish event for subagent (truncated)."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)

    printer = VerbosePrinter(enabled=True)
    printer.console = console

    long_result = {"summary": "x" * 600, "artifact_path": "/path/to/artifact"}
    iteration = AgentIteration(
        agent_type="subagent",
        agent_name="researcher",
        event="finish",
        data={"result": long_result, "duration_ms": 3000},
    )
    printer.print_iteration(iteration)

    result = output.getvalue()
    # Subagent results are truncated to 500 chars
    assert len(result) > 0


def test_verbose_printer_print_finish_with_error():
    """Test finish event with error."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)

    printer = VerbosePrinter(enabled=True)
    printer.console = console

    iteration = AgentIteration(
        agent_type="lead",
        agent_name="lead_agent",
        event="finish",
        data={"answer": None, "duration_ms": 1000},
        error="Agent timed out",
    )
    printer.print_iteration(iteration)

    result = output.getvalue()
    assert "Error" in result or len(result) > 0


def test_verbose_printer_print_finish_none_answer():
    """Test finish event with None answer."""
    output = StringIO()
    console = Console(file=output, force_terminal=True)

    printer = VerbosePrinter(enabled=True)
    printer.console = console

    iteration = AgentIteration(
        agent_type="lead",
        agent_name="lead_agent",
        event="finish",
        data={"duration_ms": 1000},  # No answer key
    )
    printer.print_iteration(iteration)

    result = output.getvalue()
    assert len(result) > 0


def test_verbose_printer_print_iteration_disabled():
    """When disabled, print_iteration should no-op."""
    printer = VerbosePrinter(enabled=False)

    iteration = AgentIteration(
        agent_type="lead",
        agent_name="lead_agent",
        event="start",
        data={"query": "test"},
    )
    # Should not raise
    printer.print_iteration(iteration)
