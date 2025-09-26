"""Tests for config resolve_model_config overrides."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import resolve_model_config, MODEL_PRESETS


def test_override_accepts_preset_alias_for_big_model():
    base = resolve_model_config(preset="gpt-5-mini", big_override="kimi-k2")
    assert base.big == MODEL_PRESETS["kimi-k2"].big


def test_override_accepts_preset_alias_for_small_model():
    base = resolve_model_config(preset="gpt-5-mini", small_override="qwen3-coder")
    assert base.small == MODEL_PRESETS["qwen3-coder"].small


def test_override_passes_through_full_identifier():
    custom_small = "openrouter/openai/gpt-5-mini:free"
    base = resolve_model_config(preset="gpt-5-mini", small_override=custom_small)
    assert base.small == custom_small
