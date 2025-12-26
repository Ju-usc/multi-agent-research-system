"""
Pytest configuration - automatically loaded for all test files.

Mocks external observability/telemetry tools to prevent:
- API calls during tests (costly, pollutes production data)
- Network requests (slower tests, requires credentials)
- Import errors when dependencies not configured

NOTE: This is the standard pytest pattern for mocking external dependencies.
Langfuse mocking is centralized HERE (not in individual test files) following
pytest best practices. This conftest.py is automatically loaded before any test
execution, ensuring all test files benefit from these mocks without duplication.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Mock langfuse before any test imports
# This prevents langfuse from authenticating during test suite runs
# IMPORTANT: conftest.py is the correct location for this (not individual test files)
sys.modules['langfuse'] = MagicMock()
sys.modules['langfuse.callback'] = MagicMock()


def mock_observe(*args, **kwargs):
    """No-op decorator that replaces @observe during tests."""
    def decorator(func):
        return func
    return decorator if not args else decorator(args[0])


sys.modules['langfuse'].observe = mock_observe
