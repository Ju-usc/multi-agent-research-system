# AGENTS.md — Coding‑Agent Guide (Principles, Architecture, Flow)

Audience: Coding agents operating in IDE/CI that implement and evolve this multi‑agent research system.
Non‑goals: Human quickstarts, vendor‑specific recipes, or low‑level API docs.

## North Star
- Simplicity > Cleverness: small modules, right abstraction, narrow interfaces.
- Plan → Act → Synthesize → Verify: always plan first, then execute narrowly.
- Separation of Concerns: planning, tools, memory, and instrumentation stay distinct.
- Replaceable Internals: model/provider/tooling can swap without changing contracts.

## Core Roles (Concepts)
- Planner: outputs a 3–6 step checklist + success criteria; pure, no side effects.
- Executor: runs steps by calling Tools; deterministic, idempotent when possible.
- Synthesizer: turns intermediate results into final artifacts; may merge with Planner for small tasks.
- Memory: keeps just‑enough context (K‑bounded + optional summary/artifacts).
- Instrumentation: logs plans/decisions via hooks; never lives in business logic.

## Operating Principles
- Async‑first: avoid blocking; use executors for any sync IO.
- Determinism & Idempotence: fixed seeds, stable ordering, explicit clocks.
- Explicit Contracts: simple tool signatures and typed IO models.
- Stateless Tools: pass deps explicitly; no hidden globals or ambient state.
- Config‑driven: environment/config for models, keys, and limits; no vendor locks in core.

## Coding Standards (Global)
- Single‑Shape Interfaces: functions, classes, and modules accept exactly one input schema and return exactly one output schema.
- No Ad‑Hoc Polymorphism: avoid hasattr/isinstance branching to support multiple shapes; enforce explicit schemas and validation instead.
- Fail Fast: when inputs don’t match the expected schema, return/raise a clear error; do not auto‑normalize or guess.
- JSON at Boundaries: for serialized payloads, prefer a single top‑level object; do not mix structured data with prose in the same field.
- Purity by Default: prefer pure logic; isolate side effects behind explicit adapters; keep inner code straight‑line with minimal branching.

## Execution Flow
1) Plan: produce checklist, inputs, and acceptance criteria.
2) Act: call tools per step; bound concurrency; collect intermediate artifacts.
3) Synthesize: compress results into a final answer/report with attributions as needed.
4) Verify: run cheap checks (schemas, invariants, tests). If unmet, update plan and loop.
5) Emit: write artifacts to memory and return the final answer.

## Instrumentation
- Emit structured events: planning_started, tool_called, synthesis_started, verification_failed, artifact_written.
- Log JSONL/structured lines safe for CI; redact secrets.
- For debugging, decorate async functions with `utils.log_call`; set
  `LOG_LEVEL=DEBUG` or call `configure_logging("DEBUG")` to emit inputs and
  JSON-serialized results. Pass `return_attr="path.to.attr"` to log a nested
  field from the returned object.

```python
from logging_config import configure_logging
from utils import log_call

@log_call(return_attr="prediction.text")
async def plan(...): ...

configure_logging("DEBUG")
```

## Testing & CI
- Unit tests per tool (with fakes) and an end‑to‑end async flow test.
- keep the tests as minimal and clean as possbile.
- no need to test every single trival details in the code.
- All network usage mocked; tests deterministic and parallel‑friendly.
- Use `uv` for env and test runs.

## Security & Limits
- Secrets via env only; never log raw keys; redact on error.
- Enforce ceilings: tokens, runtime, parallelism; fail safe and explain why.

## Repository Pointers (for orientation, not coupling)
- Current mapping: `agent.py` (orchestrates), `tools.py` (tool impls), `models.py` (data/signatures), `config.py` (settings), `tests/` (pytest), `memory/` (artifacts from running the agent).
- Treat these as reference anchors; the conceptual contracts above remain stable even if files change.

## PR Checklist (lightweight)
- Plan included in description; tests added/updated and deterministic.
- No secrets; logs redacted; limits enforced.
- Docs touched when contracts change (this file if principles shift).
