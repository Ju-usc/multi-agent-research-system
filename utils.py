"""
Utility functions for the multi-agent research system
"""

import os
import json
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


def prediction_to_json(prediction) -> str:
    """Convert DSPy Prediction object to JSON string.
    
    Args:
        prediction: DSPy Prediction object
        
    Returns:
        JSON string representation of the prediction
    """
    data = {}
    # Extract all fields from the prediction's _store (always exists on dspy.Prediction)
    for key, value in prediction._store.items():
        # Handle Pydantic models (all inherit from BaseModel which has model_dump)
        if isinstance(value, BaseModel):
            data[key] = value.model_dump()
        # Handle lists of Pydantic models
        elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
            data[key] = [item.model_dump() for item in value]
        else:
            data[key] = value
    return json.dumps(data)
