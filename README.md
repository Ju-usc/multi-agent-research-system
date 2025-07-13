# Multi-Agent Research System

A minimal multi-agent research system built with DSPy and OpenRouter, designed to reverse-engineer Claude's research capabilities.

## Architecture

This system implements a lead-subagent research pattern where:
- **Lead Agent**: Plans research tasks, manages memory, and synthesizes results
- **Subagents**: Execute specific research micro-tasks in parallel
- **Memory Store**: Maintains research artifacts with lightweight summaries

## Features

- 🔄 Async parallel execution of research tasks
- 🧠 In-memory artifact storage with automatic summarization
- 🔍 Web search and Wikipedia integration
- 📊 Iterative refinement based on synthesis decisions
- 🎯 Task-specific tool guidance and budgets

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

```python
from eval import run_browsecomp_evaluation

# Quick evaluation
results = run_browsecomp_evaluation(
    num_examples=20,
    num_threads=4
)

print(f"Accuracy: {results['accuracy']:.1f}%")
```

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with:
```
BRAVE_SEARCH_API_KEY=your_key
OPENROUTER_API_KEY=your_key
OPENAI_API_KEY=your_key
```

## Usage

```python
from agent import LeadAgent
import asyncio

agent = LeadAgent()
result = asyncio.run(agent.run("Your research question here"))
print(result)
```

## Testing

```bash
pytest tests/
```

## Dependencies

- dspy-ai
- python-dotenv
- wikipedia-api
- brave-search-python
- pandas (for BrowseComp dataset)

## License

MIT
