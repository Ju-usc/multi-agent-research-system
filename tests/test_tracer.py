"""Unit tests for the tracer module."""

import json
import pytest
import tracer


@pytest.fixture
def trace_file(tmp_path):
    """Enable tracing with file output."""
    log_file = tmp_path / "trace.jsonl"
    tracer.trace.configure(level="debug", log_path=str(log_file))
    yield log_file
    tracer.trace.configure(level="", log_path="")


def test_trace_function(trace_file):
    """@trace on function: returns correct value, logs to file."""
    @tracer.trace
    def add(a, b):
        return a + b

    assert add(2, 3) == 5

    records = [json.loads(line) for line in trace_file.read_text().strip().split("\n")]
    assert any(r.get("event") == "enter" for r in records)
    assert any(r.get("event") == "exit" for r in records)


def test_trace_method(trace_file):
    """@trace on method: wraps method and tracks class name."""
    class Calculator:
        @tracer.trace
        def add(self, a,  b):
            return a + b

    calc = Calculator()
    assert calc.add(2, 3) == 5

    records = [json.loads(line) for line in trace_file.read_text().strip().split("\n")]
    names = [r.get("name", "") for r in records]
    assert any("Calculator.add" in n for n in names)


def test_trace_hierarchy(trace_file):
    """Nested calls track parent-child relationships."""
    @tracer.trace
    def outer():
        return inner()

    @tracer.trace
    def inner():
        return "done"

    outer()

    records = [json.loads(line) for line in trace_file.read_text().strip().split("\n")]
    enter_events = [r for r in records if r.get("event") == "enter"]

    outer_id = next(r["call_id"] for r in enter_events if "outer" in r.get("name", ""))
    inner_record = next(r for r in enter_events if "inner" in r.get("name", ""))
    assert inner_record.get("parent_id") == outer_id


def test_trace_disabled():
    """Disabled tracing has zero overhead."""
    tracer.trace.configure(level="", log_path="")

    @tracer.trace
    def simple():
        return 42

    assert simple() == 42


def test_trace_captures_errors(trace_file):
    """Tracer logs exceptions."""
    @tracer.trace
    def failing():
        raise ValueError("oops")

    with pytest.raises(ValueError):
        failing()

    records = [json.loads(line) for line in trace_file.read_text().strip().split("\n")]
    exit_record = next(r for r in records if r.get("event") == "exit")
    assert "oops" in exit_record.get("error", "")
