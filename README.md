# Multi-Agent Research System

A minimal multi-agent research system, built with DSPy, inspired by [Anthropic’s multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).


> Documentation:
> - **architecture-agent.md** — flexible, subagent-spawning design (primary direction)

> - **AGENTS.md** — contributor setup, coding style, and testing checklist

## Core ideas

- **Separation of concerns:** Lead plans, tracks To-Dos, reads artifacts, synthesizes. Subagents research and optionally write a report.
- **Least privilege:** Subagents use a fixed tool set: `web_search` and optional `fs_write_report`. No filesystem reads for subagents.
- **Minimal contracts:** Subagent return surface is a single field:  
  `{"summary": "<brief findings; may include 'artifact: <path>'>"}`  
  Citations (`refs`) can be added later without changing this surface.
- **To-Do–driven planning:** High-level plans are written and tracked via a To-Do List tool to manage decomposition and progress.
- **Flexibility:** Loops and tactics are examples, not prescriptions. Prompting guidelines will evolve through optimization (e.g., SIMBA/GEPA).

## Conceptual flow

1. Create a **To-Do** of high-level plans.  
2. Spawn **Subagents** for tasks (a subagent may decompose a single To-Do into multiple tasks or handle multiple To-Dos in parallel).  
3. Subagents use **web_search** and may `fs_write_report(...)`; each returns a **summary**.  
4. Lead **reads artifacts**, updates the To-Do, and **synthesizes** the answer; iterate if gaps remain.

## Evaluation philosophy

The default `efficiency` metric drives **lean correctness** — right answers with minimal waste:

```
efficiency = accuracy / (time × cost)
```

This penalizes over-search (20 queries vs 5), over-decomposition (10 subagents vs 3), and slow synthesis. Wrong answers score zero regardless of speed.

**GEPA optimization** discovers prompts that maximize efficiency. Patterns like "use 3-5 focused tasks" and "stop when returns diminish" emerge naturally from optimizing `accuracy / (time × cost)` — they're not hardcoded.

Trade-off: optimizing for `accuracy` alone ignores cost and produces verbose reports.

## Quick CLI run

Run the lead agent from the repository root.

```bash
uv run python agent.py --query "Summarize recent innovations in AI agent collaboration frameworks."
```

Swap models with CLI flags:

```bash
uv run python agent.py --lead openrouter/deepseek/deepseek-v3.2 --query "..."
```

## Tracing

Non-invasive call tracing via `@trace` decorator. Zero overhead when disabled.

```bash
# Terminal output (human-readable)
TRACE_LEVEL=info uv run python agent.py --query "..."

# File output (JSON lines for analysis)
TRACE_LOG=logs/run.jsonl uv run python agent.py --query "..."

# Both
TRACE_LEVEL=debug TRACE_LOG=logs/run.jsonl uv run python agent.py --query "..."
```

**Levels:**
| Level | Shows |
|-------|-------|
| `info` | Args + result snippets (truncated to 200 chars) |
| `debug` | Full args + results (no truncation) |

**Example output (info):**
```
[10:23:45.123] -> Agent.forward(query=What is AI?)
[10:23:45.130]   -> WebSearchTool.search(query=AI definition...)
[10:23:45.635]   <- WebSearchTool.search [505ms] ok -> {'results': [{'title': '...
[10:23:46.500] <- Agent.forward [1377ms] ok -> The answer is...
```

See `tracer.py` for implementation details.

## Evaluation

Run BrowseComp evaluation with efficiency metrics and GEPA optimization.

```bash
# Basic evaluation
uv run python eval.py

# Custom settings
uv run python eval.py --num-examples 20 --metric accuracy

# GEPA optimization (auto train/test split)
uv run python eval.py --optimize --optimize-steps 10

# Save results
uv run python eval.py --save-metrics results.json
```

**Metrics:**
- `accuracy`: Binary correctness (1.0 or 0.0)
- `efficiency`: accuracy / (time × cost) - default

**Cost config** (`.env`):
```bash
WEBSEARCH_COST_PER_CALL_USD=0.005
LM_COST_PER_1K_TOKENS_JSON='{"openai/gpt-4o": 0.005}'
```

See `.env.template` for full example.

## License

MIT
