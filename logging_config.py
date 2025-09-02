"""
Central logging configuration for the project.

Best practice: avoid calling basicConfig in libraries. Configure logging
once at process entrypoints (CLI, __main__, tests) using this module.
"""

import logging
import os
from typing import Optional


def configure_logging(level: Optional[str] = None, fmt: Optional[str] = None) -> None:
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt = fmt or os.getenv("LOG_FORMAT", "%(levelname)s: %(message)s")

    # Initialize root logger
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO), format=fmt)

    # Quiet noisy deps if present
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    # Example: project-wide default
    logging.getLogger(__name__).debug("Logging configured: %s", level_name)

