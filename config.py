"""
Configuration module for multi-agent research system.
Loads environment variables and exposes model settings.
Logging configuration is handled in logging_config.py to avoid side effects.
"""

import os
import json
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
    "gpt-5-mini": ModelPreset(
        big="openai/gpt-5-mini",
        small="openai/gpt-5-mini",
        big_max_tokens=30000,
        small_max_tokens=30000,
    ),
    "kimi-k2": ModelPreset(
        big="openrouter/moonshotai/kimi-k2:free",
        small="openrouter/moonshotai/kimi-k2:free",
        big_max_tokens=30000,
        small_max_tokens=30000,
    ),
    "qwen3-coder": ModelPreset(
        big="openrouter/qwen/qwen3-coder:free",
        small="openrouter/qwen/qwen3-coder:free",
        big_max_tokens=12000,
        small_max_tokens=12000,
    ),
    "gpt-oss-120b": ModelPreset(
        big="openrouter/openai/gpt-oss-120b:free",
        small="openrouter/openai/gpt-oss-120b:free",
        big_max_tokens=30000,
        small_max_tokens=30000,
    ),
    "deepseek-v3.1": ModelPreset(
        big="openrouter/deepseek/deepseek-chat-v3.1:free",
        small="openrouter/deepseek/deepseek-chat-v3.1:free",
        big_max_tokens=30000,
        small_max_tokens=30000,
    ),
}

DEFAULT_MODEL_PRESET: Final[str] = "gpt-5-mini"

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
GRADER_MODEL: Final[str] = "openai/gpt-5"  # Judges answer correctness
GRADER_MAX_TOKENS: Final[int] = 16000  # Large budget for reasoning chains

OPTIMIZER_MODEL: Final[str] = "openai/gpt-5"  # GEPA prompt optimization
OPTIMIZER_MAX_TOKENS: Final[int] = 32000  # Large budget for prompt refinement

# ========== COST CONFIGURATION ==========

WEBSEARCH_COST_PER_CALL_USD = float(os.getenv("WEBSEARCH_COST_PER_CALL_USD", "0.005"))

# Model pricing (per 1K tokens) - uses actual pricing for efficiency metrics
# Note: Even when using free tier, we track AS IF paying for meaningful cost comparisons
LM_PRICING: Final[dict[str, dict[str, float]]] = {
    "openai/gpt-5-mini": {
        "input": 0.00025,        # $0.250 per 1M tokens
        "output": 0.002,         # $2.000 per 1M tokens
        "cached_input": 0.000025 # $0.025 per 1M (90% discount)
    },
    "openai/gpt-5": {
        "input": 0.00125,        # $1.25 per 1M
        "output": 0.01,          # $10.00 per 1M
        "cached_input": 0.000125 # $0.125 per 1M
    },
    "openrouter/deepseek/deepseek-chat-v3.1:free": {
        "input": 0.00027,        # $0.27 per 1M (actual DeepSeek pricing)
        "output": 0.0011,        # $1.10 per 1M
        "cached_input": 0.00007  # $0.07 per 1M
    },
    "openrouter/moonshotai/kimi-k2:free": {
        "input": 0.0006,         # $0.60 per 1M (actual Kimi pricing)
        "output": 0.0025,        # $2.50 per 1M
        "cached_input": 0.00015  # $0.15 per 1M
    },
    "openrouter/qwen/qwen3-coder:free": {
        "input": 0.00022,        # $0.22 per 1M (actual Qwen pricing)
        "output": 0.00095,       # $0.95 per 1M
        "cached_input": 0.00022  # No separate cache pricing
    }
}
