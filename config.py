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
EXA_API_KEY = os.getenv("EXA_API_KEY")

# ========== MODEL CONFIGURATION ==========
"""Models: use one small and one big model (must differ)."""
# Migrate defaults to GPT‑5 tiers: small -> gpt-5-nano, big -> gpt-5-mini
SMALL_MODEL = "gpt-5-mini"
BIG_MODEL = "gpt-5-mini"

# ========== MODEL PARAMETERS ==========
TEMPERATURE = 1.0
BIG_MODEL_MAX_TOKENS = 50000
# For OpenAI reasoning models, DSPy expects >= 16000
SMALL_MODEL_MAX_TOKENS = 50000
