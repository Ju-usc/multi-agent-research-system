# Multi-Agent Research System

A minimal multi-agent research system built with DSPy and OpenRouter, designed to reverse-engineer Claude's research capabilities.

## Architecture

This system implements a lead–subagent pattern:
- **Lead Agent** (`workflow.py`): Plans tasks, launches subagents, and synthesizes results
- **Subagents**: Execute research micro-tasks in parallel
- **Memory Store**: Filesystem that keeps artifacts with lightweight summaries

## Features

- 🔄 Async parallel execution of research tasks
- 🧠 Filesystem artifact storage with Markdown outputs
- 🔍 Web search integration
- 📊 Iterative refinement based on synthesis decisions
- 🎯 Task-specific tool guidance and budgets
- 📝 Todo list tool for tracking pending work

## BrowseComp Evaluation

Evaluate the multi-agent research system on OpenAI's BrowseComp dataset:

### Dataset (`dataset.py`)
- Downloads and decrypts the official BrowseComp dataset from OpenAI
- Handles XOR decryption with automatic canary value detection
- Creates DSPy Example objects for seamless integration

### Evaluation (`eval.py`)
- **DSPy Framework**: Uses `dspy.Evaluate` for standardized evaluation
- **LLM Judge**: Intelligent answer evaluation with reasoning
- **Parallel Execution**: Multi-threaded evaluation for efficiency
- **Program Wrapper**: `BrowseCompProgram` adapts async LeadAgent for DSPy

### Usage

```bash
uv run python -c "from eval import run_browsecomp_evaluation; results = run_browsecomp_evaluation(num_examples=20, num_threads=4); print(f'Accuracy: {results[\"accuracy\"]:.1f}%')"
```

## Installation

```bash
uv sync
```

## Environment Variables

Create a `.env` file with:
```
BRAVE_SEARCH_API_KEY=your_key
OPENROUTER_API_KEY=your_key
OPENAI_API_KEY=your_key
```

## Usage

```bash
uv run python -c "from workflow import LeadAgent; import asyncio; agent = LeadAgent(); print(asyncio.run(agent.aforward('Your research question here')))"
```

## Testing

```bash
uv run pytest tests/
```

## Dependencies

- dspy-ai
- python-dotenv
- brave-search-python-client
- pandas (for BrowseComp dataset)

## License

MIT
