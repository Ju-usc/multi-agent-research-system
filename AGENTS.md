## Quick Reference

- **README.md** — project overview
- **architecture-agent.md** — system design (source of truth for contracts)

---

## Dev Commands

```bash
# Setup
uv sync

# Run tests
uv run pytest -q

# Run single test
uv run pytest -k <pattern> -q

# Lint
uv run ruff check .

# Run agent
uv run python agent.py --query "Your query here"

# With different model
uv run python agent.py --lead openrouter/deepseek/deepseek-v3.2 --query "..."
```

---

## Tracing

Non-invasive call tracing via `@trace` decorator. Zero overhead when disabled.

```bash
# Terminal output (human-readable)
TRACE_LEVEL=info uv run python agent.py --query "..."

# File output (JSON lines)
TRACE_LOG=logs/run.jsonl uv run python agent.py --query "..."

# Both
TRACE_LEVEL=debug TRACE_LOG=logs/run.jsonl uv run python agent.py --query "..."
```

Levels: `info` (snippets, 200 chars), `debug` (full output).

---

## This Project

DSPy-based multi-agent research system:
- **Lead Agent** — plans via To-Do List, spawns subagents, synthesizes
- **Subagents** — focused researchers with web search + write capability

See `architecture-agent.md` for contracts and tool surfaces.

---

## Repo Pointers

- `agent.py` — ReAct-style agent loop
- `tools.py` — tool definitions
- `tracer.py` — `@trace` decorator for call tracing
- `models.py` — data schemas
- `config.py` — env/config switches
- `tests/` — unit tests + one opt-in e2e test

---

## Coding Principles

- Follow contracts in architecture-agent.md
- Single-shape I/O; validate at boundaries, fail fast
- Keep modules small; pass dependencies explicitly
- Trust contracts; avoid defensive `isinstance` chains
- Reuse existing patterns; delete dead code

---

## Testing

- **Default:** Unit tests only, no network, deterministic
- **E2E (opt-in):** `RUN_E2E=1 uv run pytest -q`

---

## Workflow

1. **Research** — explore codebase, find existing patterns
2. **Plan** — write plan, get approval before implementing
3. **Implement** — execute plan, group questions at end
4. **Verify** — run tests, check output before declaring done

---

## Learning from Mistakes

When something unexpected happens:

1. **Observe** — capture in memories.md, don't assume cause yet
2. **Finish task** — stay in flow, don't derail
3. **Investigate** — after task, find root cause
4. **Persist** — route to right place:

| Learning Type | Where |
|---------------|-------|
| Personal preference/observation | `~/.factory/memories.md` |
| Quick project note | `.factory/memories.md` |
| Formal project knowledge | `docs/` (via doc-maintainer) |
| Reusable patterns | relevant skill |
