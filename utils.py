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
    """Generic Markdown preview + Raw JSON.

    - Serialize once with a small default serializer.
    - If the JSON parses to a dict, render a simple, uniform Markdown view:
      each top-level key becomes a section with its value shown in a minimal
      way (scalars inline, lists as bullets, dicts as one-level bullets).
    - Always append a Raw JSON section for traceability.
    - The `title` parameter is ignored for content (kept for API compatibility).
    """

    def _default(o: Any):
        # dspy.Prediction
        if hasattr(o, "_store") and isinstance(getattr(o, "_store"), dict):
            return o._store
        # Pydantic v2 BaseModel
        if isinstance(o, BaseModel):
            return o.model_dump()
        # Last resort: string representation
        return str(o)

    # Serialize once
    body = json.dumps(obj, default=_default, indent=2, ensure_ascii=False)

    lines: list[str] = []

    # Try to parse back to a dict for a generic preview
    parsed = None
    try:
        parsed = json.loads(body)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        for key, value in parsed.items():
            lines.append(f"## {key}")
            if isinstance(value, dict):
                for sk, sv in value.items():
                    lines.append(f"- {sk}: {json.dumps(sv, ensure_ascii=False)}")
                lines.append("")
            elif isinstance(value, list):
                for item in value:
                    lines.append(f"- {json.dumps(item, ensure_ascii=False)}")
                lines.append("")
            else:
                lines.append(str(json.dumps(value, ensure_ascii=False)))
                lines.append("")

    # Always include raw JSON
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
