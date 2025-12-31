"""
Configuration module for multi-agent research system.
Loads environment variables and exposes model settings.
"""

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# ========== API KEYS ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY")

# ========== MODEL CONFIGURATION ==========

# Default models
# Lead: Grok 4.1 Fast - best agentic tool calling, 2M context, 30K max output
# Sub: Grok 4.1 Fast - same model for consistency in agentic tasks
DEFAULT_LEAD_MODEL = "openrouter/x-ai/grok-4.1-fast"
DEFAULT_SUB_MODEL = "openrouter/x-ai/grok-4.1-fast"
DEFAULT_LEAD_MAX_TOKENS = 40000
DEFAULT_SUB_MAX_TOKENS = 40000
DEFAULT_TEMPERATURE = 1.0

class ModelConfig:
    """Model configuration bundle for lead agent and subagents."""
    def __init__(
        self,
        lead: str = DEFAULT_LEAD_MODEL,
        sub: str = DEFAULT_SUB_MODEL,
        lead_max_tokens: int = DEFAULT_LEAD_MAX_TOKENS,
        sub_max_tokens: int = DEFAULT_SUB_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.lead = lead
        self.sub = sub
        self.lead_max_tokens = lead_max_tokens
        self.sub_max_tokens = sub_max_tokens
        self.temperature = temperature

# ========== TOOL DEFAULTS ==========

# Filesystem tree display
FILESYSTEM_TREE_MAX_DEPTH = 3

# Workspace isolation
WORKSPACE_UUID_LENGTH = 8  # Characters from UUID for directory naming

# Cleanup watchdog
CLEANUP_WATCHDOG_TIMEOUT_SECONDS = 30  # Force exit if DSPy/LiteLLM cleanup hangs

# Query timeout (matches OpenAI Deep Research)
QUERY_TIMEOUT_SECONDS = 600  # 10 min

# ========== EVALUATION MODELS (Fixed for experimental consistency) ==========
# These models are used for evaluation/optimization across all experiments
# to eliminate judge/optimizer variance as a confounding variable.
GRADER_MODEL = "openrouter/openai/gpt-5.2"  # Judges answer correctness
GRADER_MAX_TOKENS = 40000  # Large budget for reasoning chains

OPTIMIZER_MODEL = "openrouter/openai/gpt-5.2"  # GEPA prompt optimization
OPTIMIZER_MAX_TOKENS = 40000  # Large budget for prompt refinement

# ========== COST CONFIGURATION ==========

WEBSEARCH_COST_USD = 0.005
WEBFETCH_COST_USD = 0.001

# Model pricing per 1M tokens (for cost tracking in eval.py)
# Add your model here if using --lead/--sub with a custom model
# Unknown models will log a warning and skip cost calculation
LM_PRICING = {
    "openrouter/x-ai/grok-4.1-fast": {"input": 0.20, "output": 0.50},
    "openrouter/deepseek/deepseek-v3.2": {"input": 0.24, "output": 0.38},
    "openrouter/google/gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
    "openrouter/google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "openrouter/openai/gpt-oss-120b": {"input": 0.039, "output": 0.19},
    "openrouter/qwen/qwen3-235b-a22b-2507": {"input": 0.071, "output": 0.463},
    "openrouter/openai/gpt-5.2": {"input": 1.75, "output": 14.0, "cached_input": 0.175},
}
