# Agent Architecture

ReAct-style orchestration: a **Lead Agent** plans via a **To-Do List**, uses filesystem tools to inspect artifacts, and spawns **Subagents** (as a tool) to research. Subagents are ReAct agents with web search and write-artifact capability. The loop is flexible; the Lead decides when to finalize.

## Roles

- **Lead Agent (ReAct)**
  - Maintains a **To-Do List** of high-level plans.
  - Inspects existing artifacts to guide decisions.
  - Spawns Subagents as needed (parallel when useful).
  - Synthesizes and decides when to stop.

- **Subagent (ReAct)**
  - Focused researcher for a task.
  - Uses web search; optionally writes a Markdown report.
  - Returns a minimal summary (may include inline artifact reference).

## Tools (fixed surfaces)

All primitive tools return JSON: `{"isError": bool, "message": str}`.
- `isError: false` → success, result in `message`
- `isError: true` → failure, error details in `message`

**Lead tools**
- `todo_list_read()` — returns `{isError, message}` with todos JSON in message.
- `todo_list_write(todos)` — write the entire To-Do list, returns `{isError, message}`.
- `filesystem_read(path)` — read artifact, returns `{isError, message}` with content in message.
- `filesystem_tree(max_depth)` — list artifacts, returns `{isError, message}` with directory tree.
- `subagent_run(task)` — execute subagent, returns `SubagentResult` JSON (summary, detail, artifact_path).
- `parallel_tool_call(calls)` — run tools concurrently, returns list of individual tool results.

**Subagent tools**
- `web_search(query, …)` — search web, returns `{isError, message}` with results in message.
- `filesystem_write(path, content)` — write artifact, returns `{isError, message}`.
- `parallel_tool_call(calls)` — run tools concurrently, returns list of individual tool results.

## Contracts (minimal)

- **Subagent input:** `{ task }`
- **Subagent output:** structured JSON with `summary`.

  Optional `detail` and `artifact_path` appear when the subagent writes a report under `memory/`.
- **Task shape:** `task_name`, `prompt`, `description`, `tool_budget` stay required. Add `expected_output` when you know the artifact; skip it when exploring.
- **No filesystem reads by Subagents.** Artifacts are written by Subagents, read by the Lead.

## High-level loop (illustrative, not prescriptive)

1) **Decompose → To-Do:** capture high-level plans as To-Dos.
2) **Inspect:** call `filesystem_tree` then `filesystem_read` to reuse artifacts.
3) **Dispatch:** use `parallel_tool_call` with multiple `subagent_run` calls to execute tasks in parallel when beneficial.
4) **Collect:** aggregate subagent summaries and note any referenced artifacts.
5) **Update:** adjust To-Dos through `todo_list_write` and add follow-ups for gaps.
6) **Synthesize & stop:** produce the answer and stop when the Lead judges it complete.

> Prompting guidelines and control heuristics emerge via optimization (e.g., SIMBA/GEPA); the loop remains flexible.

## Diagram

```mermaid
flowchart LR
  U[User Query] --> L[Lead Agent ReAct]
  L --> T[To-Do List]
  L --> A[Artifacts *.md]
  L --> S1[Subagent]
  L --> S2[Subagent]

  S1 -->|web_search| W[(Web)]
  S2 -->|web_search| W
  S1 -->|filesystem_write| A
  S2 -->|filesystem_write| A

  S1 -->|summary| L
  S2 -->|summary| L

  L -->|filesystem_read| A
  L --> F[Finalize when sufficient]
```
