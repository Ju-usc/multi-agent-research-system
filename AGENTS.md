## Useful Docs

See also:
- @README.md — project overview
- @architecture-agent.md — agent architecture (primary direction)
- @architecture-workflow.md — workflow architecture (structured pipeline)

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
````

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

* Keep logging outside core logic; use decorators, wrappers, or hooks.
* Use `logging` (no `print`); emit structured, JSONL-friendly lines.
* Never log secrets or raw tool payloads.
* Increase detail when needed:

  ```bash
  LOG_LEVEL=DEBUG uv run pytest -q
  ```

---

## Coding conventions (enforced by review)

* Follow contracts in @architecture-agent.md and @architecture-workflow\.md; do not add ad-hoc fields or tools.
* Single-shape I/O per function/module; validate at boundaries and **fail fast**.
* Keep modules small; avoid ambient globals; pass dependencies explicitly.
* Prefer straight-line logic: one input type → one output type. If variants are required, use an explicit `type/kind` field and switch once.
* Reuse helpers/utilities instead of rewriting from scratch; delete duplicate or dead code.
* Trust contracts: avoid defensive ladders (`isinstance`, `hasattr` chains); write less code, rely on defined models.

---

# WRITING STYLE

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

# HELP THE USER LEARN

- when coding, always explain what you are doing and why
- your job is to help the user learn & upskill himself, above all
- assume the user is an intelligent, tech savvy person -- but do not assume he knows the details
- explain everything clearly, simply, in easy-to-understand language. write in short sentences.
- Always consider multiple different approaches, and analyze their tradeoffs just like a Senior Developer would

---

## Repo pointers (orientation, not coupling)

* `agent.py` — flexible a single ReAct-style agent; plans, executes, synthesizes in one loop (see @architecture-agent.md).
* `workflow.py` — structured Plan → Execute → Synthesize multi modules pipeline (see @architecture-workflow\.md).
* `tools.py` — tool definitions; usage documented in architecture files.
* `models.py` — data schemas and signatures.
* `config.py` — env/config switches.
* `tests/` — unit tests plus one opt-in end-to-end test.

---

## PR checklist (minimal)

* Tests pass locally: `uv run pytest -q` (E2E only when explicitly triggered).
* No secrets in code/logs; limits respected.
* Interfaces unchanged or documented; update @architecture-\*.md if contracts changed.
* Logs are structured and concise.
