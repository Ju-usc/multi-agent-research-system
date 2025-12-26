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
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# ========== MODEL CONFIGURATION ==========

# Default models - Grok 4.1 Fast (Dec 2025)
# Best agentic tool calling, 2M context, $0.20/M in, $0.50/M out
DEFAULT_LEAD_MODEL = "openrouter/x-ai/grok-4.1-fast"
DEFAULT_SUB_MODEL = "openrouter/x-ai/grok-4.1-fast"
DEFAULT_LEAD_MAX_TOKENS = 16000
DEFAULT_SUB_MAX_TOKENS = 16000
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


def lm_kwargs_for(model_id: str) -> dict[str, str]:
    if model_id.startswith("openrouter/"):
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY must be set to use OpenRouter models.")

        kwargs = {"api_key": OPENROUTER_API_KEY}
        if OPENROUTER_BASE_URL:
            kwargs["base_url"] = OPENROUTER_BASE_URL
        return kwargs

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY must be set to use OpenAI models.")

    return {"api_key": OPENAI_API_KEY}


# ========== TOOL DEFAULTS ==========
# WebSearch defaults
WEBSEARCH_MAX_RESULTS = 5  # Results per query
WEBSEARCH_MAX_TOKENS_PER_PAGE = 1024  # Content truncation limit

# Parallel execution
PARALLEL_THREADS = 4  # Max concurrent tool invocations

# Filesystem tree display
FILESYSTEM_TREE_MAX_DEPTH = 3

# Workspace isolation
WORKSPACE_UUID_LENGTH = 8  # Characters from UUID for directory naming

# Cleanup watchdog
CLEANUP_WATCHDOG_TIMEOUT_SECONDS = 30  # Force exit if DSPy/LiteLLM cleanup hangs

# ========== EVALUATION MODELS (Fixed for experimental consistency) ==========
# These models are used for evaluation/optimization across all experiments
# to eliminate judge/optimizer variance as a confounding variable.
GRADER_MODEL = "openrouter/google/gemini-3-flash-preview"  # Judges answer correctness
GRADER_MAX_TOKENS = 16000  # Large budget for reasoning chains

OPTIMIZER_MODEL = "openrouter/google/gemini-3-flash-preview"  # GEPA prompt optimization
OPTIMIZER_MAX_TOKENS = 32000  # Large budget for prompt refinement

# ========== COST CONFIGURATION ==========

WEBSEARCH_COST_PER_CALL_USD = float(os.getenv("WEBSEARCH_COST_PER_CALL_USD", "0.005"))

# Model pricing per 1M tokens (for cost tracking in eval.py)
# Add your model here if using --lead/--sub with a custom model
# Unknown models will log a warning and skip cost calculation
LM_PRICING = {
    "openrouter/x-ai/grok-4.1-fast": {"input": 0.20, "output": 0.50},
    "openrouter/deepseek/deepseek-v3.2": {"input": 0.24, "output": 0.38},
    "openrouter/google/gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
}
