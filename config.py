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

# ========== EVALUATION MODELS (Fixed for experimental consistency) ==========
# These models are used for evaluation/optimization across all experiments
# to eliminate judge/optimizer variance as a confounding variable.
GRADER_MODEL = "openrouter/openai/gpt-5.2"  # Judges answer correctness
GRADER_MAX_TOKENS = 40000  # Large budget for reasoning chains

OPTIMIZER_MODEL = "openrouter/openai/gpt-5.2"  # GEPA prompt optimization
OPTIMIZER_MAX_TOKENS = 40000  # Large budget for prompt refinement


