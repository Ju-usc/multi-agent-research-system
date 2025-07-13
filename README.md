# Multi-Agent Research System

A minimal multi-agent research system built with DSPy and OpenRouter, designed to reverse-engineer Claude's research capabilities.

# GOAL

- Create the right abstraction for multi-agent research system to utilize dspy's signature
- let the dspy optimization to automatically emerge the specific detailed instructions for each module

## Langfuse Tracing Setup

This system includes **automatic DSPy tracing** with Langfuse for observability and debugging.

### Quick Setup (following official docs):

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set environment variables:**
   ```bash
   export LANGFUSE_PUBLIC_KEY="pk-lf-..."
   export LANGFUSE_SECRET_KEY="sk-lf-..."
   export LANGFUSE_HOST="https://cloud.langfuse.com"
   ```

   Or create a `.env` file:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

### Get Langfuse credentials:
- Sign up at https://cloud.langfuse.com
- Create a project
- Copy your API keys from project settings

### That's it! 🚀
DSPy operations are automatically traced via OpenInference instrumentation. View traces at https://cloud.langfuse.com

**Official documentation:** https://langfuse.com/docs/integrations/dspy
