import os
import sys

# Ensure repository root is on sys.path for direct module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import prediction_to_markdown


def test_prediction_to_markdown_generic_preview_and_raw():
    obj = {
        "title": "My Plan",
        "reasoning": "Need data.",
        "is_done": False,
        "tasks": [
            {"task_name": "t1"},
            "note",
        ],
    }

    out = prediction_to_markdown(obj)

    # Nested bullets for top-level keys
    assert "- title: My Plan" in out
    assert "- reasoning: Need data." in out
    assert "- is_done: false" in out

    # List rendered with nested bullets
    assert "- tasks:" in out
    assert "- item 1:" in out
    assert "- task_name: t1" in out
    assert "- note" in out

    # Raw JSON section always present
    assert "## Raw" in out
    assert "```json" in out
