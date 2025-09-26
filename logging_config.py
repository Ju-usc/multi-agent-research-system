"""
Central logging configuration for the project.

Best practice: avoid calling basicConfig in libraries. Configure logging
once at process entrypoints (CLI, __main__, tests) using this module.
"""

import logging
import os
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

TRACE_LOGGER_NAME = "trace"
MAX_PREVIEW_CHARS = 100
LOG_DIR = Path("logs")
TRACE_RUN_ID = os.getenv("TRACE_RUN_ID") or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
TRACE_LOG_FILENAME = os.getenv("TRACE_LOG_FILENAME") or f"trace-{TRACE_RUN_ID}.log"
TRACE_STACK: ContextVar[tuple[tuple[str, Optional[str]], ...]] = ContextVar("trace_stack", default=())


def configure_logging(level: Optional[str] = None, fmt: Optional[str] = None) -> None:
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt = fmt or os.getenv("LOG_FORMAT", "%(levelname)s: %(message)s")

    # Initialize root logger
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO), format=fmt)

    # Quiet noisy deps if present
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    trace_logger.setLevel(logging.DEBUG)
    trace_logger.propagate = False

    _attach_trace_console_handler()
    _attach_trace_file_handler()

    # Example: project-wide default
    logging.getLogger(__name__).debug("Logging configured: %s", level_name)


def trace_call(scope: str, *, level: int = logging.DEBUG, logger: Optional[logging.Logger] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    trace_logger = logger or logging.getLogger(TRACE_LOGGER_NAME)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            previous_stack = TRACE_STACK.get()
            display_args = _strip_bound_self(func, wrapped, args)
            label = _extract_label(scope, display_args, kwargs)
            token = TRACE_STACK.set(previous_stack + ((scope, label),))
            current_stack = TRACE_STACK.get()
            depth = max(len(current_stack) - 1, 0)
            active_subagent = _active_subagent(current_stack)

            if trace_logger.isEnabledFor(logging.INFO):
                trace_logger.info(
                    _format_summary(
                        scope,
                        "call",
                        depth=depth,
                        label=label,
                        active_subagent=active_subagent,
                        args=display_args,
                        kwargs=kwargs,
                    )
                )
            if trace_logger.isEnabledFor(level):
                trace_logger.log(
                    level,
                    _format_entry(scope, "call", args=display_args, kwargs=kwargs),
                )

            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                if trace_logger.isEnabledFor(logging.ERROR):
                    trace_logger.error(
                        _format_summary(
                            scope,
                            "error",
                            depth=depth,
                            label=label,
                            active_subagent=active_subagent,
                            args=display_args,
                            kwargs=kwargs,
                            error=exc,
                        )
                    )
                trace_logger.exception(
                    _format_entry(scope, "error", args=display_args, kwargs=kwargs, error=exc),
                )
                raise
            else:
                if trace_logger.isEnabledFor(logging.INFO):
                    trace_logger.info(
                        _format_summary(
                            scope,
                            "return",
                            depth=depth,
                            label=label,
                            active_subagent=active_subagent,
                            args=display_args,
                            kwargs=kwargs,
                            result=result,
                        )
                    )
                if trace_logger.isEnabledFor(level):
                    trace_logger.log(
                        level,
                        _format_entry(scope, "return", args=display_args, kwargs=kwargs, result=result),
                    )
                return result
            finally:
                TRACE_STACK.reset(token)

        return wrapped

    return decorator


def _format_entry(scope: str, event: str, *, args: tuple[Any, ...] | None = None,
                  kwargs: dict[str, Any] | None = None,
                  result: Any | None = None, error: Exception | None = None) -> str:
    parts = [f"{scope}::{event}"]

    if args is not None or kwargs is not None:
        arg_summary = ", ".join(_summarize_parts(args or (), kwargs or {}))
        if arg_summary:
            parts.append(f"input={arg_summary}")

    if result is not None:
        parts.append(f"output={_preview_value(result)}")

    if error is not None:
        parts.append(f"error={_preview_value(error)}")

    return " | ".join(parts)


def _format_summary(scope: str, event: str, *, depth: int,
                    label: Optional[str], active_subagent: Optional[str],
                    args: tuple[Any, ...] | None = None,
                    kwargs: dict[str, Any] | None = None,
                    result: Any | None = None, error: Exception | None = None) -> str:
    indent = "  " * depth
    context_bits = []
    if active_subagent and active_subagent != label:
        context_bits.append(f"subagent={active_subagent}")
    if label and scope == "tool.subagent_forward":
        context_bits.append(f"task={label}")

    context = f" ({', '.join(context_bits)})" if context_bits else ""
    lines = [f"{indent}{scope} [{event}]{context}"]

    for part in _summarize_parts(args or (), kwargs or {}, labelled=True):
        lines.append(f"{indent}    {part}")

    if result is not None:
        lines.append(f"{indent}    -> {_preview_value(result)}")

    if error is not None:
        lines.append(f"{indent}    !! {_preview_value(error)}")

    return "\n".join(lines)


def _summarize_parts(args: tuple[Any, ...], kwargs: dict[str, Any], *, labelled: bool = False) -> list[str]:
    parts: list[str] = []

    for idx, value in enumerate(args):
        preview = _preview_value(value)
        if preview:
            parts.append(f"arg{idx}: {preview}" if labelled else preview)

    for key, value in kwargs.items():
        preview = _preview_value(value)
        if preview:
            parts.append(f"{key}={preview}")

    return parts


def _extract_label(scope: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[str]:
    if scope != "tool.subagent_forward":
        return None

    task = kwargs.get("task")
    if task is None:
        for value in args:
            if hasattr(value, "task_name"):
                task = value
                break

    if task is not None and hasattr(task, "task_name"):
        return getattr(task, "task_name")

    return None


def _active_subagent(stack: tuple[tuple[str, Optional[str]], ...]) -> Optional[str]:
    for scope, label in reversed(stack):
        if scope == "tool.subagent_forward" and label:
            return label
    return None


def _preview_value(value: Any) -> str:
    if isinstance(value, str):
        cleaned = " ".join(value.split())
        return f'"{_truncate(cleaned)}"'

    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)

    if isinstance(value, dict):
        return f"<dict len={len(value)}>"

    if isinstance(value, (list, tuple, set)):
        return f"<{type(value).__name__} len={len(value)}>"

    try:
        return _truncate(repr(value))
    except Exception:
        return _truncate(object.__repr__(value))


def _truncate(text: str) -> str:
    if len(text) <= MAX_PREVIEW_CHARS:
        return text
    return text[:MAX_PREVIEW_CHARS] + "…"


def _strip_bound_self(original: Callable[..., Any], wrapper: Callable[..., Any], args: tuple[Any, ...]) -> tuple[Any, ...]:
    if not args:
        return args

    method_name = getattr(original, "__name__", getattr(wrapper, "__name__", None))
    if not method_name:
        return args

    first = args[0]
    if _bound_target_matches(first, method_name, original, wrapper):
        return args[1:]

    return args


def _bound_target_matches(candidate: Any, method_name: str, original: Callable[..., Any], wrapper: Callable[..., Any]) -> bool:
    try:
        bound = getattr(candidate, method_name)
    except AttributeError:
        return False

    target = getattr(bound, "__func__", bound)
    if target is wrapper:
        return True

    wrapped_target = getattr(target, "__wrapped__", None)
    return wrapped_target is original


def _attach_trace_file_handler() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_path = (LOG_DIR / TRACE_LOG_FILENAME).resolve()

    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)

    for handler in trace_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            existing_path = Path(handler.baseFilename).resolve()
            if existing_path == file_path:
                return

    handler = logging.FileHandler(file_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    trace_logger.addHandler(handler)
    trace_logger.debug("Trace file attached: %s", handler.baseFilename)


def _attach_trace_console_handler() -> None:
    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)

    for handler in trace_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            return

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))

    trace_logger.addHandler(handler)

