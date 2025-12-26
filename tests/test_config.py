"""Tests for config ModelConfig."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    ModelConfig,
    DEFAULT_LEAD_MODEL,
    DEFAULT_SUB_MODEL,
    DEFAULT_LEAD_MAX_TOKENS,
    DEFAULT_SUB_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
)


def test_model_config_uses_defaults():
    config = ModelConfig()
    assert config.lead == DEFAULT_LEAD_MODEL
    assert config.sub == DEFAULT_SUB_MODEL
    assert config.lead_max_tokens == DEFAULT_LEAD_MAX_TOKENS
    assert config.sub_max_tokens == DEFAULT_SUB_MAX_TOKENS
    assert config.temperature == DEFAULT_TEMPERATURE


def test_model_config_accepts_custom_models():
    config = ModelConfig(
        lead="custom/lead",
        sub="custom/sub",
    )
    assert config.lead == "custom/lead"
    assert config.sub == "custom/sub"
