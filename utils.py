"""
Utility functions for the multi-agent research system
"""

import os
import json
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
    """Reliable, single-output renderer: one JSON block (optionally titled).

    Uses json.dumps with a small default serializer so we don't branch over
    types manually. This keeps the function short and predictable.
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

    body = json.dumps(obj, default=_default, indent=2, ensure_ascii=False)
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    lines.append("```json")
    lines.append(body)
    lines.append("```")
    return "\n".join(lines)
