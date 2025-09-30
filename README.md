# Multi-Agent Research System

A minimal multi-agent research system, built with DSPy, inspired by [Anthropic’s multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).


> Documentation:
> - **architecture-agent.md** — flexible, subagent-spawning design (primary direction)
> - **architecture-workflow.md** — structured plan → execute → synthesize pipeline
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

Swap models with the preset flag when you want to test different providers.

```bash
uv run python agent.py --model kimi-k2 --query "Summarize recent innovations in AI agent collaboration frameworks."
```

## Logs

CLI runs emit structured traces under `logs/` by default.

Use `TRACE_LOG_FILENAME` to pick an easy-to-remember name.

```bash
TRACE_LOG_FILENAME=trace-ai-collab.log LOG_LEVEL=DEBUG \
  uv run python agent.py --query "Summarize recent innovations in AI agent collaboration frameworks."
```

The example above writes `logs/trace-ai-collab.log`. Create directories ahead of time if you specify a path.

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
