"""
Configuration module for multi-agent research system.
Loads environment variables and exposes model settings.
Logging configuration is handled in logging_config.py to avoid side effects.
"""

import os
from dataclasses import dataclass
from typing import Final

from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# ========== API KEYS ==========
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# ========== MODEL CONFIGURATION ==========

@dataclass(frozen=True)
class ModelPreset:
    big: str
    small: str
    big_max_tokens: int
    small_max_tokens: int


MODEL_PRESETS: Final[dict[str, ModelPreset]] = {
    # === FREE TIER MODELS (50 req/day without credits, 1000/day with $10+) ===
    
    # xAI Grok 4.1 Fast - Best agentic tool calling, 2M context
    # TESTED: Excellent tool calling. Very thorough researcher (~30 web searches).
    # Slow for BrowseComp (~29 min/example). Some Perplexity 400 errors on complex queries.
    # RECOMMENDED: Lead agent, judge/grader roles (quality over speed)
    "grok-4.1-fast": ModelPreset(
        big="openrouter/x-ai/grok-4.1-fast:free",
        small="openrouter/x-ai/grok-4.1-fast:free",
        big_max_tokens=16000,
        small_max_tokens=16000,
    ),
    # Moonshot Kimi K2 - Strong reasoning, tool use
    # TESTED: Reliable tool calling. Good balance of speed and quality.
    # Completed BrowseComp in ~5-6 min. No stability issues.
    # RECOMMENDED: Subagent workhorse, primary testing model
    "kimi-k2": ModelPreset(
        big="openrouter/moonshotai/kimi-k2:free",
        small="openrouter/moonshotai/kimi-k2:free",
        big_max_tokens=16000,
        small_max_tokens=16000,
    ),
    # Qwen3 Coder - Optimized for agentic coding, 262K context
    # TESTED: Tool calling works. Coding-focused, may be less suited for research.
    # Consider for code analysis subagents.
    "qwen3-coder": ModelPreset(
        big="openrouter/qwen/qwen3-coder:free",
        small="openrouter/qwen/qwen3-coder:free",
        big_max_tokens=12000,
        small_max_tokens=12000,
    ),
    # OpenAI gpt-oss-20b - Open-weight, agentic capabilities
    # TESTED: Tool calling works. Limited testing on BrowseComp.
    "gpt-oss-20b": ModelPreset(
        big="openrouter/openai/gpt-oss-20b:free",
        small="openrouter/openai/gpt-oss-20b:free",
        big_max_tokens=16000,
        small_max_tokens=16000,
    ),
    # DeepSeek R1T Chimera - Reasoning + efficiency merged model
    # TESTED: Tool calling works BUT did NOT use websearch tool in BrowseComp (0 calls).
    # NOT RECOMMENDED for research tasks requiring web search.
    "deepseek-r1t": ModelPreset(
        big="openrouter/tngtech/deepseek-r1t-chimera:free",
        small="openrouter/tngtech/deepseek-r1t-chimera:free",
        big_max_tokens=16000,
        small_max_tokens=16000,
    ),
    # GLM 4.5 Air - Hybrid reasoning, agent-centric
    # TESTED: Tool calling works. UNSTABLE in eval - "cannot schedule new futures
    # after shutdown" errors. NOT RECOMMENDED until stability issues resolved.
    "glm-4.5-air": ModelPreset(
        big="openrouter/z-ai/glm-4.5-air:free",
        small="openrouter/z-ai/glm-4.5-air:free",
        big_max_tokens=16000,
        small_max_tokens=16000,
    ),
    
    # === PAID MODELS (via OpenRouter) ===
    
    # DeepSeek V3.2 - GPT-5 class reasoning, agentic tool-use, 164K context
    # Released Dec 1, 2025. IMO/IOI gold medal level. DSA sparse attention.
    # TESTED: Tool calling works. ~6.5 min/example, 8 websearches. Paid model.
    "deepseek-v3.2": ModelPreset(
        big="openrouter/deepseek/deepseek-v3.2",
        small="openrouter/deepseek/deepseek-v3.2",
        big_max_tokens=16000,
        small_max_tokens=16000,
    ),
    
    # MiniMax M2 - Strong on BrowseComp, agentic workflows
    # TESTED: Tool calling works. Untested on full BrowseComp eval.
    "minimax-m2": ModelPreset(
        big="openrouter/minimax/minimax-m2",
        small="openrouter/minimax/minimax-m2",
        big_max_tokens=16000,
        small_max_tokens=16000,
    ),
    # OpenAI GPT-5 Nano - Fast, cheap, good tool support
    # TESTED: Tool calling works. Minor quirk: outputs 'finish' with quotes.
    "gpt-5-nano": ModelPreset(
        big="openrouter/openai/gpt-5-nano",
        small="openrouter/openai/gpt-5-nano",
        big_max_tokens=16000,
        small_max_tokens=16000,
    ),
    # OpenAI GPT-5 Mini - Balanced performance (via OpenRouter)
    "gpt-5-mini": ModelPreset(
        big="openrouter/openai/gpt-5-mini",
        small="openrouter/openai/gpt-5-mini",
        big_max_tokens=16000,
        small_max_tokens=16000,
    ),
}

# === MODEL ROLE DEFAULTS ===
# Lead agent: orchestrates, plans, synthesizes - needs high reasoning quality
# Subagent: executes research tasks - needs speed + reliability
# Judge/Grader: evaluates answers - needs consistency
DEFAULT_LEAD_MODEL: Final[str] = "grok-4.1-fast"
DEFAULT_SUBAGENT_MODEL: Final[str] = "kimi-k2"
DEFAULT_MODEL_PRESET: Final[str] = "kimi-k2"  # Default for CLI --model flag

# === AGENT BEHAVIOR CONFIGURATION ===
# Reflection: Lead agent self-checks answer completeness before finalizing
# Adds ~1 extra LLM call but improves answer quality on complex queries
ENABLE_REFLECTION: Final[bool] = True
MAX_REFLECTION_ITERATIONS: Final[int] = 2  # Max self-correction loops

def _resolve_override(
    override: str | None,
    *,
    slot: str,
    fallback: ModelPreset,
) -> tuple[str, int]:
    """Resolve model id and token cap for a slot using preset aliases."""

    fallback_tokens = getattr(fallback, f"{slot}_max_tokens")

    if not override:
        return getattr(fallback, slot), fallback_tokens

    candidate = override.lower()
    if candidate in MODEL_PRESETS:
        preset = MODEL_PRESETS[candidate]
        tokens = getattr(preset, f"{slot}_max_tokens")
        return getattr(preset, slot), tokens

    # Accept fully qualified model identifiers while defaulting to the preset token cap.
    return override, fallback_tokens


def resolve_model_config(
    preset: str | None = None,
    big_override: str | None = None,
    small_override: str | None = None,
) -> ModelPreset:
    resolved_name = (preset or DEFAULT_MODEL_PRESET).lower()
    if resolved_name not in MODEL_PRESETS:
        valid = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown model preset '{resolved_name}'. Valid options: {valid}.")

    preset_config = MODEL_PRESETS[resolved_name]

    big_id, big_tokens = _resolve_override(big_override, slot="big", fallback=preset_config)
    small_id, small_tokens = _resolve_override(small_override, slot="small", fallback=preset_config)

    return ModelPreset(
        big=big_id,
        small=small_id,
        big_max_tokens=big_tokens,
        small_max_tokens=small_tokens,
    )


def lm_kwargs_for(model_id: str) -> dict[str, str]:
    """Return API kwargs for a model. All models route through OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY must be set.")

    kwargs = {"api_key": OPENROUTER_API_KEY}
    if OPENROUTER_BASE_URL:
        kwargs["base_url"] = OPENROUTER_BASE_URL
    return kwargs


_DEFAULT_PRESET = resolve_model_config()
BIG_MODEL = _DEFAULT_PRESET.big
SMALL_MODEL = _DEFAULT_PRESET.small
BIG_MODEL_MAX_TOKENS = _DEFAULT_PRESET.big_max_tokens
SMALL_MODEL_MAX_TOKENS = _DEFAULT_PRESET.small_max_tokens

# ========== MODEL PARAMETERS ==========
TEMPERATURE = 1.0
# Max token limits are derived from the selected preset above.
# Do not override them here; models like OpenRouter free tiers enforce 8–16k.

# ========== EVALUATION MODELS (Fixed for experimental consistency) ==========
# These models are used for evaluation/optimization across all experiments
# to eliminate judge/optimizer variance as a confounding variable.
# Using Grok 4.1 Fast: thorough, high reasoning quality, free tier
GRADER_MODEL: Final[str] = "openrouter/x-ai/grok-4.1-fast:free"  # Judges answer correctness
GRADER_MAX_TOKENS: Final[int] = 40000  # Grader needs room for detailed reasoning

OPTIMIZER_MODEL: Final[str] = "openrouter/x-ai/grok-4.1-fast:free"  # GEPA prompt optimization
OPTIMIZER_MAX_TOKENS: Final[int] = 40000  # GEPA reflection needs room for analysis

# ========== COST CONFIGURATION ==========

WEBSEARCH_COST_PER_CALL_USD = float(os.getenv("WEBSEARCH_COST_PER_CALL_USD", "0.005"))

# Model pricing per 1 MILLION tokens (industry standard format)
# Note: Free tier models track "as if paying" for meaningful cost comparisons
LM_PRICING: Final[dict[str, dict[str, float]]] = {
    # === FREE TIER MODELS ===
    
    # xAI Grok 4.1 Fast (free tier)
    "openrouter/x-ai/grok-4.1-fast:free": {
        "input": 0.0,
        "output": 0.0,
        "cached_input": 0.0
    },
    # Moonshot Kimi K2 (free tier)
    "openrouter/moonshotai/kimi-k2:free": {
        "input": 0.60,
        "output": 2.50,
        "cached_input": 0.15
    },
    # Qwen3 Coder (free tier)
    "openrouter/qwen/qwen3-coder:free": {
        "input": 0.22,
        "output": 0.95,
        "cached_input": 0.22
    },
    # OpenAI gpt-oss-20b (free tier)
    "openrouter/openai/gpt-oss-20b:free": {
        "input": 0.0,
        "output": 0.0,
        "cached_input": 0.0
    },
    # DeepSeek R1T Chimera (free tier)
    "openrouter/tngtech/deepseek-r1t-chimera:free": {
        "input": 0.0,
        "output": 0.0,
        "cached_input": 0.0
    },
    # GLM 4.5 Air (free tier)
    "openrouter/z-ai/glm-4.5-air:free": {
        "input": 0.0,
        "output": 0.0,
        "cached_input": 0.0
    },
    
    # === PAID MODELS ===
    
    # DeepSeek V3.2 (released Dec 1, 2025)
    "openrouter/deepseek/deepseek-v3.2": {
        "input": 0.28,
        "output": 0.40,
        "cached_input": 0.028
    },
    # MiniMax M2
    "openrouter/minimax/minimax-m2": {
        "input": 0.255,
        "output": 1.02,
        "cached_input": 0.255
    },
    # OpenAI GPT-5 Nano
    "openrouter/openai/gpt-5-nano": {
        "input": 0.05,
        "output": 0.40,
        "cached_input": 0.005
    },
    # OpenAI GPT-5 Mini (via OpenRouter)
    "openrouter/openai/gpt-5-mini": {
        "input": 0.25,
        "output": 2.00,
        "cached_input": 0.025
    },
    # OpenAI GPT-5 (for grader/optimizer)
    "openrouter/openai/gpt-5": {
        "input": 1.25,
        "output": 10.00,
        "cached_input": 0.125
    },
}
