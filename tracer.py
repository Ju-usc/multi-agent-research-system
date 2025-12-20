"""
Non-invasive tracing via @trace decorator.

Usage:
    from tracer import trace

    class MyClass:
        @trace
        def method(self): ...

    @trace
    def my_function(): ...

Config (env vars):
    TRACE_LEVEL=info|debug  # terminal output level
    TRACE_LOG=logs/trace.jsonl  # file output (captures all)
"""

import functools
import inspect
import json
import os
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any


class Tracer:
    """Observe function calls and emit trace records."""

    LEVELS = {"info": 1, "debug": 2}
    TERMINAL_MAX_LEN = 200
    FILE_MAX_LEN = 100

    def __init__(self, level: str = "", log_path: str = ""):
        self.level = level.lower()
        self.log_path = log_path
        self._stack: ContextVar[list] = ContextVar("stack", default=[])

    @property
    def enabled(self) -> bool:
        return bool(self.level or self.log_path)

    @property
    def is_debug(self) -> bool:
        return self.LEVELS.get(self.level, 0) >= self.LEVELS["debug"]

    def __call__(self, func):
        """Decorator to trace function calls."""
        name = func.__qualname__

        # Detect if first param is self/cls
        try:
            params = list(inspect.signature(func).parameters.keys())
            skip_first = params and params[0] in ("self", "cls")
        except (ValueError, TypeError):
            skip_first = False

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            call_id = uuid.uuid4().hex[:8]
            stack = self._stack.get()
            parent_id = stack[-1]["id"] if stack else None
            depth = len(stack)

            # Prepare args (skip self/cls for methods)
            log_args = kwargs.copy()
            arg_start = 1 if skip_first else 0
            for i, a in enumerate(args[arg_start:]):
                log_args[f"arg{i}"] = a

            # Push to stack and execute
            self._stack.set(stack + [{"id": call_id, "name": name}])
            start = datetime.now()
            error = None
            result = None
            try:
                self._log_terminal("enter", name, depth, start, args=log_args)
                self._log_file("enter", name, call_id, parent_id, depth, start, args=log_args)
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                end = datetime.now()
                duration_ms = (end - start).total_seconds() * 1000
                self._log_terminal("exit", name, depth, end, duration_ms=duration_ms, error=error, result=result)
                self._log_file("exit", name, call_id, parent_id, depth, end, duration_ms=duration_ms, error=error, result=result)
                self._stack.set(self._stack.get()[:-1])

        return wrapper

    def _format_value(self, value: Any, max_len: int) -> Any:
        """Format value for output, truncating unless debug mode."""
        if self.is_debug:
            return value
        return self._truncate(value, max_len)

    def _log_terminal(self, event: str, name: str, depth: int, timestamp: datetime,
                      args: dict = None, duration_ms: float = None, error: str = None, result: Any = None):
        """Human-readable output to terminal."""
        if not self.level:
            return

        prefix = f"[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] {'  ' * depth}"

        if event == "enter":
            suffix = ""
            if args:
                pairs = ", ".join(f"{k}={self._format_value(v, self.TERMINAL_MAX_LEN)}" for k, v in args.items())
                suffix = f"({pairs})"
            print(f"{prefix}-> {name}{suffix}")
            return

        status = "x" if error else "ok"
        suffix = f" -> {self._format_value(result, self.TERMINAL_MAX_LEN)}" if result is not None else ""
        print(f"{prefix}<- {name} [{duration_ms:.0f}ms] {status}{suffix}")

    def _log_file(self, event: str, name: str, call_id: str, parent_id: str,
                  depth: int, timestamp: datetime, args: dict = None, duration_ms: float = None,
                  error: str = None, result: Any = None):
        """JSON trace record to file."""
        if not self.log_path:
            return

        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        lines = []

        # Info record
        rec = {"level": "info", "ts": timestamp.isoformat(), "event": event,
               "name": name, "call_id": call_id, "depth": depth}
        if parent_id:
            rec["parent_id"] = parent_id
        if duration_ms is not None:
            rec["duration_ms"] = round(duration_ms, 2)
        if error:
            rec["error"] = error
        lines.append(json.dumps(rec, default=str))

        # Debug record (truncated unless debug mode)
        debug_fields = {}
        if args is not None:
            debug_fields["args"] = self._format_value(args, self.FILE_MAX_LEN)
        if result is not None:
            debug_fields["result"] = self._format_value(result, self.FILE_MAX_LEN)
        if debug_fields:
            debug_rec = {"level": "debug", "ts": timestamp.isoformat(), "call_id": call_id, **debug_fields}
            lines.append(json.dumps(debug_rec, default=str))

        with open(self.log_path, "a") as f:
            f.write("\n".join(lines) + "\n")

    def _truncate(self, value: Any, max_len: int = 100) -> Any:
        """Truncate values: prefix......suffix. Handles nested structures."""
        if isinstance(value, str):
            if len(value) <= max_len * 2:
                return value
            return f"{value[:max_len]}......{value[-max_len:]}"
        if isinstance(value, dict):
            return {k: self._truncate(v, max_len) for k, v in value.items()}
        if isinstance(value, list):
            return [self._truncate(v, max_len) for v in value]
        return value


# Default instance from environment
tracer = Tracer(
    level=os.getenv("TRACE_LEVEL", ""),
    log_path=os.getenv("TRACE_LOG", "")
)

# Backward-compatible alias
trace = tracer
