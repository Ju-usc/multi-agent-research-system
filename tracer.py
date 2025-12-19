"""
Non-invasive tracing via @trace decorator.

Usage:
    from tracer import trace

    @trace
    class MyClass:
        def method(self): ...

    @trace
    def my_function(): ...

Config (env vars):
    TRACE_LEVEL=info|debug  # terminal output level
    TRACE_LOG=logs/trace.jsonl  # file output (captures all)
"""

import functools
import json
import os
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path


# --- Config ---
TRACE_LEVEL = os.getenv("TRACE_LEVEL", "").lower()
TRACE_LOG = os.getenv("TRACE_LOG", "")
TRACE_ENABLED = bool(TRACE_LEVEL or TRACE_LOG)
LEVELS = {"info": 1, "debug": 2}

# --- Call stack for hierarchy ---
_stack: ContextVar[list] = ContextVar("stack", default=[])


# --- Public API ---

def trace(target=None, *, exclude: list = None):
    """Smart decorator for functions and classes."""
    def decorator(t):
        if isinstance(t, type):
            return _wrap_class(t, exclude or [])
        if callable(t):
            return _wrap_func(t)
        raise TypeError(f"@trace cannot decorate {type(t)}")

    return decorator(target) if target else decorator


# --- Internal: wrap function ---

def _wrap_func(func, class_name: str = None):
    """Wrap function to trace calls."""
    name = f"{class_name}.{func.__name__}" if class_name else func.__name__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not TRACE_ENABLED:
            return func(*args, **kwargs)

        # Setup
        call_id = uuid.uuid4().hex[:8]
        stack = _stack.get()
        parent_id = stack[-1]["id"] if stack else None
        depth = len(stack)

        # Push to stack
        _stack.set(stack + [{"id": call_id, "name": name}])

        # Prepare args (skip self for methods)
        log_args = kwargs.copy()
        start = 1 if class_name else 0
        for i, a in enumerate(args[start:]):
            log_args[f"arg{i}"] = a

        # Entry
        _emit("enter", name, call_id, parent_id, depth, args=log_args)

        # Execute
        t0 = time.time()
        error = None
        result = None
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error = str(e)
            raise
        finally:
            ms = (time.time() - t0) * 1000
            _emit("exit", name, call_id, parent_id, depth, ms=ms, error=error, result=result)
            # Pop from stack
            _stack.set(_stack.get()[:-1])

    return wrapper


# --- Internal: wrap class ---

def _wrap_class(cls, exclude: list):
    """Wrap all public methods of a class."""
    for attr_name in dir(cls):
        if attr_name in exclude:
            continue
        if attr_name.startswith("__") and attr_name != "__call__":
            continue
        attr = getattr(cls, attr_name, None)
        if callable(attr) and not isinstance(attr, type):
            setattr(cls, attr_name, _wrap_func(attr, cls.__name__))
    return cls


# --- Internal: emit trace ---

def _emit(event, name, call_id, parent_id, depth, args=None, ms=None, error=None, result=None):
    """Emit to terminal and/or file."""
    ts = datetime.now()
    is_debug = LEVELS.get(TRACE_LEVEL, 0) >= LEVELS.get("debug", 0)

    # Terminal output
    if TRACE_LEVEL and LEVELS.get("info", 0) <= LEVELS.get(TRACE_LEVEL, 0):
        indent = "  " * depth
        time_str = ts.strftime("%H:%M:%S.%f")[:-3]
        if event == "enter":
            # info: truncated args (200 chars), debug: full args
            preview = ""
            if args:
                if is_debug:
                    preview = "(" + ", ".join(f"{k}={v}" for k, v in args.items()) + ")"
                else:
                    preview = "(" + ", ".join(f"{k}={_trunc(v, 200)}" for k, v in args.items()) + ")"
            print(f"[{time_str}] {indent}-> {name}{preview}")
        else:
            status = "x" if error else "ok"
            # info: truncated result (200 chars), debug: full result
            result_preview = ""
            if result is not None:
                if is_debug:
                    result_preview = f" -> {result}"
                else:
                    result_preview = f" -> {_trunc(result, 200)}"
            print(f"[{time_str}] {indent}<- {name} [{ms:.0f}ms] {status}{result_preview}")

    # File output (all levels)
    if TRACE_LOG:
        Path(TRACE_LOG).parent.mkdir(parents=True, exist_ok=True)
        # Info record
        rec = {"level": "info", "ts": ts.isoformat(), "event": event,
               "name": name, "call_id": call_id, "depth": depth}
        if parent_id:
            rec["parent_id"] = parent_id
        if ms is not None:
            rec["duration_ms"] = round(ms, 2)
        if error:
            rec["error"] = error
        with open(TRACE_LOG, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        # Debug record (args or result)
        if args is not None or result is not None:
            debug_rec = {"level": "debug", "ts": ts.isoformat(), "call_id": call_id}
            if args is not None:
                debug_rec["args"] = args
            if result is not None:
                debug_rec["result"] = result
            with open(TRACE_LOG, "a") as f:
                f.write(json.dumps(debug_rec, default=str) + "\n")


def _trunc(v, n=50):
    """Truncate for display."""
    s = str(v)
    return s[:n] + "..." if len(s) > n else s
