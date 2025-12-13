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

# Default models (placeholder - will revisit after analyzing feature/metric-eval branch)
DEFAULT_LEAD_MODEL = "openai/gpt-5-mini"
DEFAULT_SUB_MODEL = "openai/gpt-5-mini"
DEFAULT_LEAD_MAX_TOKENS = 30000
DEFAULT_SUB_MAX_TOKENS = 30000
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
GRADER_MODEL = "openai/gpt-5"  # Judges answer correctness
GRADER_MAX_TOKENS = 16000  # Large budget for reasoning chains

OPTIMIZER_MODEL = "openai/gpt-5"  # GEPA prompt optimization
OPTIMIZER_MAX_TOKENS = 32000  # Large budget for prompt refinement

# ========== COST CONFIGURATION ==========

WEBSEARCH_COST_PER_CALL_USD = float(os.getenv("WEBSEARCH_COST_PER_CALL_USD", "0.005"))

# Model pricing per 1 MILLION tokens (industry standard format)
# Matches OpenAI/Anthropic/Google pricing display conventions
# Note: Even when using free tier, we track AS IF paying for meaningful cost comparisons
LM_PRICING = {
    # OpenAI GPT-5 Models (Standard Tier - verified 2025)
    "openai/gpt-5-mini": {
        "input": 0.25,           # $0.25 per 1M tokens
        "output": 2.00,          # $2.00 per 1M tokens
        "cached_input": 0.025    # $0.025 per 1M tokens (90% discount)
    },
    "openai/gpt-5": {
        "input": 1.25,           # $1.25 per 1M tokens
        "output": 10.00,         # $10.00 per 1M tokens
        "cached_input": 0.125    # $0.125 per 1M tokens (90% discount)
    },
    
    # DeepSeek v3.1 (verified 2025 - OpenRouter free tier uses same pricing)
    "openrouter/deepseek/deepseek-chat-v3.1:free": {
        "input": 0.28,           # $0.28 per 1M tokens (cache miss)
        "output": 0.42,          # $0.42 per 1M tokens
        "cached_input": 0.028    # $0.028 per 1M tokens (cache hit, 90% discount)
    },
    
    # Moonshot Kimi K2 (verified 2025)
    "openrouter/moonshotai/kimi-k2:free": {
        "input": 0.60,           # $0.60 per 1M tokens (cache miss)
        "output": 2.50,          # $2.50 per 1M tokens
        "cached_input": 0.15     # $0.15 per 1M tokens (cache hit, 75% discount)
    },
    
    # Qwen3 Coder (verified 2025 - OpenRouter pricing)
    "openrouter/qwen/qwen3-coder:free": {
        "input": 0.22,           # $0.22 per 1M tokens
        "output": 0.95,          # $0.95 per 1M tokens
        "cached_input": 0.22     # $0.22 per 1M tokens (no separate cache discount)
    }
}
