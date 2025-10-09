"""
Pytest configuration - automatically loaded for all test files.

Mocks external observability/telemetry tools to prevent:
- API calls during tests (costly, pollutes production data)
- Network requests (slower tests, requires credentials)
- Import errors when dependencies not configured
"""
import sys
from unittest.mock import MagicMock


# Mock langfuse before any test imports
# This prevents langfuse from authenticating during test suite runs
sys.modules['langfuse'] = MagicMock()
sys.modules['langfuse.callback'] = MagicMock()


def mock_observe(*args, **kwargs):
    """No-op decorator that replaces @observe during tests."""
    def decorator(func):
        return func
    return decorator if not args else decorator(args[0])


sys.modules['langfuse'].observe = mock_observe
