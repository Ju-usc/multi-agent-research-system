#!/bin/bash
# Simple script to refresh dependencies and run tests
# Run this periodically to let upstream fixes resolve warnings

uv sync --refresh
uv run pytest tests/ 