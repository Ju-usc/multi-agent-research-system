## Useful Docs

Start with README.md for the high-level tour.

> Architecture details such as tool names or model choices evolve. Always cross-check the architecture docs for the current truth.

See also:
- @README.md — project overview (Before starting any task, read this file)
- @architecture-agent.md — agent architecture (primary direction)

- @.codex/ — additional useful rules/guidelines (Before starting any task, read this file)

---

## Dev setup & common commands

```bash
# Create/refresh the virtual env and install deps
uv sync

# Run fast tests (no network, deterministic)
uv run pytest -q

# Run a single test
uv run pytest -k <pattern> -q

# Lint (if configured)
uv run ruff check .
```

> Prefer `uv` over direct `python/pip` to stay inside the project venv.

---

## Useful quick iteration mode

* **One-liner shell**: fastest loop for micro-checks.
* **Spike test**: when logic is unclear, write a temporary test with debug logs; pick the simplest working branch, then delete.
* **Unit test**: for stable contracts.
* **E2E**: single opt-in integration test; run only when explicitly requested.

Examples:

  ```bash
  # one-off check
  uv run python - <<'PY'
  from module import fn
  print(fn("input"))
  PY

# run a focused test
uv run pytest -k "<pattern>" -q
```

### Run the agent CLI

Use the repo root as the working directory.

```bash
uv run python agent.py --query "Summarize recent innovations in AI agent collaboration frameworks."
```

Change models with the preset shortcut when you need to verify another provider.

```bash
uv run python agent.py --model kimi-k2 --query "Summarize recent innovations in AI agent collaboration frameworks."
```

### Tracing

Non-invasive call tracing via `@trace` decorator. Zero overhead when disabled.

```bash
# Terminal output (human-readable)
TRACE_LEVEL=info uv run python agent.py --query "..."

# File output (JSON lines)
TRACE_LOG=logs/run.jsonl uv run python agent.py --query "..."

# Both
TRACE_LEVEL=debug TRACE_LOG=logs/run.jsonl uv run python agent.py --query "..."
```

Levels: `info` (enter/exit), `debug` (+ args), `verbose` (+ results).

---

## Testing strategy

* **Default:** unit tests only; no network; tool I/O stubbed; deterministic.
* **Integration (opt-in):** full agent invocation is expensive. Trigger explicitly:

  ```bash
  uv run pytest -m e2e -q
  # or
  RUN_E2E=1 uv run pytest -q
  ```
* Keep integration to **one** comprehensive test; everything else stays unit-level.

---

## Logging & instrumentation

* Use `@trace` decorator from `tracer.py` for call hierarchy tracing.
* Keep logging outside core logic; decorators wrap functions non-invasively.
* File output (`TRACE_LOG`) writes JSONL with full args/results for analysis.
* Never log secrets or raw tool payloads.
* Increase detail when needed:

  ```bash
  TRACE_LEVEL=verbose uv run python agent.py --query "..."
  ```

---

## Coding conventions (enforced by review)

* Follow contracts in @architecture-agent.md; do not add ad-hoc fields or tools.
* Single-shape I/O per function/module; validate at boundaries and **fail fast**.
* Keep modules small; avoid ambient globals; pass dependencies explicitly.
* Prefer straight-line logic: one input type → one output type. If variants are required, use an explicit `type/kind` field and switch once.
* Reuse helpers/utilities instead of rewriting from scratch; delete duplicate or dead code.
* Trust contracts: avoid defensive ladders (`isinstance`, `hasattr` chains); write less code, rely on defined models.

---

## Writing style

- Break up long sentences. After each long sentence, insert two newline characters.
- Avoid long bullet lists.
- Write in plain, natural English. Be conversational.
- Keep sentences short and simple. Use concise, easy-to-understand language.
- Do not use overly complex words or structures.
- Write in complete, clear sentences.
- Speak like a Senior Developer mentoring a junior engineer.
- Provide enough context for the User to understand, but keep explanations short.
- Always state your assumptions and conclusions clearly.
  
---

## Help the user learn

- when coding, always explain what you are doing and why
- your job is to help the user learn & upskill himself, above all
- assume the user is an intelligent, tech savvy person -- but do not assume he knows the details
- explain everything clearly, simply, in easy-to-understand language. write in short sentences.
- Always consider multiple different approaches, and analyze their tradeoffs just like a Senior Developer would

---

## Repo pointers (orientation, not coupling)

* `agent.py` — flexible a single ReAct-style agent; plans, executes, synthesizes in one loop (see @architecture-agent.md).

* `tools.py` — tool definitions; usage documented in architecture files.
* `tracer.py` — `@trace` decorator for non-invasive call hierarchy tracing.
* `models.py` — data schemas and signatures.
* `config.py` — env/config switches.
* `tests/` — unit tests plus one opt-in end-to-end test.

---

## PR checklist (minimal)

* Tests pass locally: `uv run pytest -q` (E2E only when explicitly triggered).
* No secrets in code/logs; limits respected.
* Interfaces unchanged or documented; update @architecture-\*.md if contracts changed.
* Logs are structured and concise.

---

## Agentic Coding Workflow Guidlines

0. Tasks

- Operating on a task basis. Store all intermediate context in markdown files in tasks/<task-id>/ folders.
- Use semantic task id slugs

1. Research

- Find existing patterns in this codebase
- Search internet, mcp tools if relevant (PREFER using nia tools than relying only on generic web search)
- Start by asking follow up questions to set the direction of research
- Report findings in research.md file

2. Planning

- Read the research.md in tasks for <task-id>.
- Based on the research come up with a plan for implementing the user request. We should reuse existing patterns, components and code where possible.
- If needed, ask clarifying questions to user to understand the scope of the task
- Write the comprehensive plan to plan.md. The plan should include all context required for an engineer to implement the feature.
- Wait for user to read and approve the plan

3. Implementation

- Read. plan.md and create a todo-list with all items, then execute on the plan.
- Go for as long as possible. If ambiguous, leave all questions to the end and group them.

4. Verification

- Once implementation is complete, you must verify that the implementation meets the requirements and is free of bugs.
- Do this by running tests, making tool calls and checking the output.
- If there are any issues, go back to the implementation step and make the necessary changes.
- Once verified, update the task status to "verified".
