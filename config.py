"""
Configuration module for multi-agent research system.
Contains environment variables, model settings, and logging configuration.
"""

import os
import logging
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ========== API KEYS ==========
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ========== MODEL CONFIGURATION ==========
SMALL_MODEL = os.getenv("GEMINI_2.5_FLASH_LITE")
BIG_MODEL = os.getenv("O4_MINI")

# ========== MODEL PARAMETERS ==========
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))
BIG_MODEL_MAX_TOKENS = int(os.getenv("BIG_MODEL_MAX_TOKENS", "20000"))
SMALL_MODEL_MAX_TOKENS = int(os.getenv("SMALL_MODEL_MAX_TOKENS", "4000"))

# ========== LOGGING CONFIGURATION ==========
# Suppress noisy warnings and logs
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)