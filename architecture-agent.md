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

**Lead tools**
- `todo.add|update|mark` — manage To-Dos.
- `fs_list(path)` — list artifacts (filesystem tree discovery).
- `fs_read(path)` — read artifact contents.
- `spawn_subagent(task, [artifact_path])` → `{ summary }`
  - Executes a Subagent with its own tools.
  - Can be invoked in parallel for multiple tasks.

**Subagent tools**
- `web_search(query, …)` — research on the web.
- `fs_write_report(path, markdown)` → `{ path, bytes }` (sandboxed write)

## Contracts (minimal)

- **Subagent input:** `{ task, (optional) artifact_path }`
- **Subagent output:**  
  `{"summary": "<brief findings; may include 'artifact: <path>'>"}`
- **No filesystem reads by Subagents.** Artifacts are written by Subagents, read by the Lead.

## High-level loop (illustrative, not prescriptive)

1) **Decompose → To-Do:** capture high-level plans as To-Dos.  
2) **Inspect:** `fs_list` / `fs_read` to reuse existing artifacts.  
3) **Dispatch:** `spawn_subagent` for active To-Dos or decomposed sub-tasks (parallel allowed).  
4) **Collect:** aggregate Subagent `summary` outputs; artifacts may be referenced inline.  
5) **Update:** mark/adjust To-Dos; add follow-ups for uncovered aspects.  
6) **Synthesize & stop:** produce the answer; finalize when the Lead judges it complete.

> Prompting guidelines and control heuristics emerge via optimization (e.g., SIMBA/GEPA); the loop remains flexible.

## Diagram

```mermaid
flowchart LR
  U[User Query] --> L[Lead Agent ReAct]

  subgraph ARTIFACTS["Artifacts (agent-written files)"]
    T[To-Do List]
    A[Reports: *.md]
  end

  L -->|create/update/mark| T
  L -->|fs_list / fs_read| A
  L -->|spawn_subagent (parallel)| S1[Subagent]
  L -->|spawn_subagent (parallel)| S2[Subagent]

  S1 -->|web_search| W[(Web)]
  S2 -->|web_search| W
  S1 -->|fs_write_report| A
  S2 -->|fs_write_report| A

  S1 -->|summary| L
  S2 -->|summary| L
  L -->|read| A
  L --> F[Finalize when sufficient]
