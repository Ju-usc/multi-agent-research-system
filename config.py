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

# Default models - based on feature/metrics-eval testing:
# kimi-k2: Reliable tool calling, ~5-6 min/example, no stability issues
# grok-4.1-fast: Very thorough (~30 searches), but slow (~29 min/example)
# See MODEL_NOTES below for full testing results
DEFAULT_LEAD_MODEL = "openrouter/moonshotai/kimi-k2:free"
DEFAULT_SUB_MODEL = "openrouter/moonshotai/kimi-k2:free"
DEFAULT_GRADER_MODEL = "openai/gpt-5"
DEFAULT_REFLECTOR_MODEL = "openai/gpt-5"
DEFAULT_LEAD_MAX_TOKENS = 16000
DEFAULT_SUB_MAX_TOKENS = 16000
DEFAULT_GRADER_MAX_TOKENS = 16000
DEFAULT_REFLECTOR_MAX_TOKENS = 32000
DEFAULT_TEMPERATURE = 1.0

# Testing notes from feature/metrics-eval branch (for reference when choosing models):
# ┌─────────────────────┬─────────────┬─────────────────────────────────────────────┐
# │ Model               │ Recommended │ Notes                                       │
# ├─────────────────────┼─────────────┼─────────────────────────────────────────────┤
# │ kimi-k2             │ ✓ DEFAULT   │ Reliable, ~5-6 min, good speed/quality      │
# │ grok-4.1-fast       │ ✓ Lead      │ Very thorough, ~29 min, best for quality    │
# │ deepseek-v3.2       │ ✓ Paid      │ ~6.5 min, 8 searches, good balance          │
# │ qwen3-coder         │ ~ Coding    │ Works, but coding-focused                   │
# │ deepseek-r1t        │ ✗           │ Did NOT use websearch (0 calls)             │
# │ glm-4.5-air         │ ✗           │ Unstable ("cannot schedule new futures")    │
# └─────────────────────┴─────────────┴─────────────────────────────────────────────┘


class ModelConfig:
    """Model configuration bundle for lead agent, subagents, grader, and reflector."""
    def __init__(
        self,
        lead: str = DEFAULT_LEAD_MODEL,
        sub: str = DEFAULT_SUB_MODEL,
        grader: str = DEFAULT_GRADER_MODEL,
        reflector: str = DEFAULT_REFLECTOR_MODEL,
        lead_max_tokens: int = DEFAULT_LEAD_MAX_TOKENS,
        sub_max_tokens: int = DEFAULT_SUB_MAX_TOKENS,
        grader_max_tokens: int = DEFAULT_GRADER_MAX_TOKENS,
        reflector_max_tokens: int = DEFAULT_REFLECTOR_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.lead = lead
        self.sub = sub
        self.grader = grader
        self.reflector = reflector
        self.lead_max_tokens = lead_max_tokens
        self.sub_max_tokens = sub_max_tokens
        self.grader_max_tokens = grader_max_tokens
        self.reflector_max_tokens = reflector_max_tokens
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

# ========== COST CONFIGURATION ==========

WEBSEARCH_COST_PER_CALL_USD = float(os.getenv("WEBSEARCH_COST_PER_CALL_USD", "0.005"))

# Model pricing per 1 MILLION tokens (industry standard format)
# Matches OpenAI/Anthropic/Google pricing display conventions
# Note: Even when using free tier, we track AS IF paying for meaningful cost comparisons
LM_PRICING = {
    # === FREE TIER MODELS (OpenRouter) - priced "as-if paid" for meaningful comparisons ===
    # xAI Grok 4.1 Fast
    "openrouter/x-ai/grok-4.1-fast:free": {
        "input": 0.20,           # $0.20 per 1M tokens (paid rate)
        "output": 0.50,          # $0.50 per 1M tokens (paid rate)
        "cached_input": 0.02     # estimated 90% cache discount
    },
    # Moonshot Kimi K2
    "openrouter/moonshotai/kimi-k2:free": {
        "input": 0.60,           # $0.60 per 1M tokens (paid rate)
        "output": 2.50,          # $2.50 per 1M tokens (paid rate)
        "cached_input": 0.15     # $0.15 per 1M tokens (75% cache discount)
    },

    # === PAID MODELS ===
    # OpenAI GPT-5 Models (Standard Tier)
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
    # DeepSeek V3.2 (via OpenRouter)
    "openrouter/deepseek/deepseek-v3.2": {
        "input": 0.25,           # $0.25 per 1M tokens
        "output": 0.38,          # $0.38 per 1M tokens
        "cached_input": 0.025    # estimated 90% cache discount
    },
    # DeepSeek V3.2 Speciale - high-compute variant for max reasoning
    "openrouter/deepseek/deepseek-v3.2-speciale": {
        "input": 0.27,           # $0.27 per 1M tokens
        "output": 0.41,          # $0.41 per 1M tokens
        "cached_input": 0.027    # estimated 90% cache discount
    }
}
