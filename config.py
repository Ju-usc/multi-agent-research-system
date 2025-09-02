"""
Configuration module for multi-agent research system.
Loads environment variables and exposes model settings.
Logging configuration is handled in logging_config.py to avoid side effects.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ========== API KEYS ==========
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ========== MODEL CONFIGURATION ==========
"""Models: use one small and one big model (must differ)."""
SMALL_MODEL = os.getenv("SMALL_MODEL", "gpt-4o-mini")
BIG_MODEL = os.getenv("BIG_MODEL", "gpt-4o")

# ========== MODEL PARAMETERS ==========
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))
BIG_MODEL_MAX_TOKENS = int(os.getenv("BIG_MODEL_MAX_TOKENS", "20000"))
SMALL_MODEL_MAX_TOKENS = int(os.getenv("SMALL_MODEL_MAX_TOKENS", "4000"))
