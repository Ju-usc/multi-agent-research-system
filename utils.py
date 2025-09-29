"""
Utility functions for the multi-agent research system
"""

import argparse
import os
import json
import logging
from functools import wraps
from typing import Any, Iterable, Tuple

from dotenv import load_dotenv

from config import MODEL_PRESETS


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
        if hasattr(o, "_store") and isinstance(getattr(o, "_store"), dict):
            return o._store
        if hasattr(o, "model_dump") and callable(getattr(o, "model_dump")):
            try:
                return o.model_dump()
            except Exception:
                pass
        return str(o)

    body = json.dumps(obj, default=_default, indent=2, ensure_ascii=False)

    try:
        parsed = json.loads(body)
    except Exception:
        parsed = None

    lines: list[str] = []

    def _choose_fence(text: str) -> str:
        return "~~~" if "```" in text else "```"

    def _render(value: Any, indent: int = 0, key: str | None = None) -> None:
        pad = "  " * indent
        bullet = f"{pad}- "

        if isinstance(value, dict):
            if key is not None:
                lines.append(f"{bullet}{key}:")
                indent += 1
            for k, v in value.items():
                _render(v, indent, k)
            return

        if isinstance(value, list):
            if key is not None:
                lines.append(f"{bullet}{key}:")
                indent += 1
            for i, item in enumerate(value, 1):
                if isinstance(item, (dict, list)):
                    _render(item, indent, f"item {i}")
                else:
                    _render(item, indent)
            return

        if isinstance(value, str):
            if "\n" in value:
                fence = _choose_fence(value)
                if key is not None:
                    lines.append(f"{bullet}{key}:")
                else:
                    lines.append(f"{bullet.rstrip()}")
                pad = "  " * (indent + 1)
                lines.append(f"{pad}{fence}text")
                for line in value.splitlines():
                    lines.append(f"{pad}{line}")
                lines.append(f"{pad}{fence}")
            else:
                if key is not None:
                    lines.append(f"{bullet}{key}: {value}")
                else:
                    lines.append(f"{bullet}{value}")
            return

        txt = json.dumps(value, ensure_ascii=False)
        if key is not None:
            lines.append(f"{bullet}{key}: {txt}")
        else:
            lines.append(f"{bullet}{txt}")

    if isinstance(parsed, dict):
        _render(parsed)

    lines.append("## Raw")
    lines.append("")
    lines.append("```json")
    lines.append(body)
    lines.append("```")

    return "\n".join(lines)


def log_call(func):
    """Log entry and exit of async functions to cut boilerplate."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.info("Starting %s", func.__name__)
        result = await func(*args, **kwargs)
        logger.info("Finished %s", func.__name__)
        return result

    return wrapper


def create_model_cli_parser(
    description: str,
    *,
    include_list: bool = False,
    query: Tuple[str, str] | None = None,
) -> argparse.ArgumentParser:
    """Return an ArgumentParser with shared model arguments.

    Args:
        description: CLI description string.
        include_list: add ``--list-models`` when True.
        query: optional tuple of (default, help) to add ``--query``.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Model preset to use for both big and small slots.",
    )
    parser.add_argument("--model-big", dest="model_big", help="Override the big model identifier.")
    parser.add_argument("--model-small", dest="model_small", help="Override the small model identifier.")
    if include_list:
        parser.add_argument(
            "--list-models",
            action="store_true",
            help="List available model presets and exit.",
        )
    if query is not None:
        default, help_text = query
        parser.add_argument("--query", default=default, help=help_text)
    return parser


def iter_model_presets() -> Iterable[tuple[str, Any]]:
    """Yield model presets sorted by key."""

    return sorted(MODEL_PRESETS.items())
