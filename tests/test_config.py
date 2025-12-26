"""Tests for config ModelConfig."""

from config import (
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
