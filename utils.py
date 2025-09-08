"""
Utility functions for the multi-agent research system
"""

import os
import json
import logging
from functools import wraps
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel


def setup_langfuse():
    """
    Setup Langfuse tracing for DSPy following official documentation.
    
    Returns:
        langfuse client if successful, None otherwise
    """
    # Load environment variables
    load_dotenv()
    
    # Step 1: Set environment variables
    os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-...")
    os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-...")
    os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # Step 2: Initialize Langfuse client
    try:
        from langfuse import get_client
        langfuse = get_client()
        
        # Verify connection
        if langfuse.auth_check():
            print("✅ Langfuse client is authenticated and ready!")
        else:
            print("⚠️  Authentication failed. Please check your credentials and host.")
            return None
            
    except ImportError:
        print("⚠️  Langfuse not available. Run: uv sync")
        return None
    
    # Step 3: Enable tracing for DSPy
    try:
        from openinference.instrumentation.dspy import DSPyInstrumentor
        DSPyInstrumentor().instrument()
        print("✅ DSPy tracing enabled")
    except ImportError:
        print("⚠️  DSPy instrumentation not available. Run: uv sync")
        return None
    
    return langfuse


 


def prediction_to_markdown(obj: Any, title: str | None = None) -> str:
    """Generic Markdown preview (nested bullets) + Raw JSON.

    Single path:
    - Serialize once with a small default serializer.
    - If JSON parses to a dict, render the entire structure as nested bullets
      (dicts/lists handled recursively; scalars inline). No headings, no
      domain keys.
    - Always append a Raw JSON section.
    - The `title` parameter is ignored for content (kept for API compatibility).
    """

    # --- Serialization -----------------------------------------------------
    def _default(o: Any):
        # dspy.Prediction
        if hasattr(o, "_store") and isinstance(getattr(o, "_store"), dict):
            return o._store
        # Pydantic v2 BaseModel (duck-typed)
        if hasattr(o, "model_dump") and callable(getattr(o, "model_dump")):
            try:
                return o.model_dump()
            except Exception:
                pass
        # Last resort: string representation
        return str(o)

    body = json.dumps(obj, default=_default, indent=2, ensure_ascii=False)

    # --- Parse back to JSON types -----------------------------------------
    try:
        parsed = json.loads(body)
    except Exception:
        parsed = None

    lines: list[str] = []

    # --- Small rendering helpers ------------------------------------------
    def _choose_fence(text: str) -> str:
        return "~~~" if "```" in text else "```"

    def _render_str(value: str, indent: int, *, key: str | None = None, label: str | None = None) -> None:
        pad = "  " * indent
        if "\n" in value:
            fence = _choose_fence(value)
            if key is not None:
                lines.append(f"{pad}- {key}:")
            elif label is not None:
                lines.append(f"{pad}- {label}:")
            else:
                lines.append(f"{pad}-")
            lines.append(f"{pad}  {fence}text")
            for line in value.splitlines():
                lines.append(f"{pad}  {line}")
            lines.append(f"{pad}  {fence}")
        else:
            if key is not None:
                lines.append(f"{pad}- {key}: {value}")
            else:
                lines.append(f"{pad}- {value}")

    def _render_scalar(value: Any, indent: int, *, key: str | None = None) -> None:
        pad = "  " * indent
        txt = json.dumps(value, ensure_ascii=False)
        if key is not None:
            lines.append(f"{pad}- {key}: {txt}")
        else:
            lines.append(f"{pad}- {txt}")

    def _render_dict(d: dict[str, Any], indent: int) -> None:
        pad = "  " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{pad}- {k}:")
                _render_dict(v, indent + 1)
            elif isinstance(v, list):
                _render_list(v, indent, parent_key=k)
            elif isinstance(v, str):
                _render_str(v, indent, key=k)
            else:
                _render_scalar(v, indent, key=k)

    def _render_list(items: list[Any], indent: int, *, parent_key: str | None = None) -> None:
        pad = "  " * indent
        if parent_key is not None:
            lines.append(f"{pad}- {parent_key}:")
            base = indent + 1
        else:
            base = indent

        for i, item in enumerate(items, 1):
            if isinstance(item, dict):
                lines.append(f"{'  ' * base}- item {i}:")
                _render_dict(item, base + 1)
            elif isinstance(item, list):
                lines.append(f"{'  ' * base}- item {i}:")
                # Render nested lists: complex -> item:, scalars -> bullets
                for sub in item:
                    if isinstance(sub, dict):
                        lines.append(f"{'  ' * (base + 1)}- item:")
                        _render_dict(sub, base + 2)
                    elif isinstance(sub, list):
                        lines.append(f"{'  ' * (base + 1)}- item:")
                        for subsub in sub:
                            if isinstance(subsub, str):
                                if "\n" in subsub:
                                    _render_str(subsub, base + 2)
                                else:
                                    lines.append(f"{'  ' * (base + 2)}- {subsub}")
                            else:
                                lines.append(f"{'  ' * (base + 2)}- {json.dumps(subsub, ensure_ascii=False)}")
                    else:
                        if isinstance(sub, str):
                            if "\n" in sub:
                                _render_str(sub, base + 1)
                            else:
                                lines.append(f"{'  ' * (base + 1)}- {sub}")
                        else:
                            lines.append(f"{'  ' * (base + 1)}- {json.dumps(sub, ensure_ascii=False)}")
            else:
                if isinstance(item, str):
                    if "\n" in item:
                        _render_str(item, base, label=f"item {i}")
                    else:
                        lines.append(f"{'  ' * base}- {item}")
                else:
                    _render_scalar(item, base)

    # --- Render preview if we have a top-level dict ------------------------
    if isinstance(parsed, dict):
        _render_dict(parsed, 0)

    # --- Always include Raw JSON ------------------------------------------
    lines.append("## Raw")
    lines.append("")
    lines.append("```json")
    lines.append(body)
    lines.append("```")

    return "\n".join(lines)


def log_call(func=None, *, return_attr: str | None = None):
    """Log entry/exit plus debug details for async functions.

    Args:
        func: Async function being decorated.
        return_attr: Optional dotted path to log a nested attribute from the
            returned object instead of the full object.

    Emits the arguments (excluding ``self``) and a JSON representation of the
    return value (or the specified nested attribute) at DEBUG level so callers
    can inspect intermediate planning artifacts without polluting core logic.
    """

    if func is None:
        return lambda f: log_call(f, return_attr=return_attr)

    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.info("Starting %s", func.__name__)

        # Log inputs without the first arg (typically ``self``)
        call_args = args[1:] if args else []
        logger.debug("%s args=%s kwargs=%s", func.__name__, call_args, kwargs)

        result = await func(*args, **kwargs)

        # Resolve nested attribute if requested
        to_log = result
        if return_attr:
            for attr in return_attr.split('.'):
                to_log = getattr(to_log, attr, None)
                if to_log is None:
                    break

        # Attempt to serialize DSPy predictions or arbitrary objects
        try:
            serialized = json.dumps(getattr(to_log, "_store", to_log), default=str, ensure_ascii=False)
        except Exception:  # pragma: no cover - best effort logging
            serialized = repr(to_log)
        logger.debug("%s result=%s", func.__name__, serialized)

        logger.info("Finished %s", func.__name__)
        return result

    return wrapper
