# Workflow Architecture

Structured pipeline reflecting the same roles with a fixed sequence.

---

## Sequence

1. **Plan (ReAct)**
   - A plan module/agent maintains a **To-Do List** of high-level plans (using a To-Do tool).
   - Reads existing artifacts to maintain context.
   - Emits a set of **parallelizable tasks** as artifacts for subagents.

2. **Execute (parallel)**
   - Spawn Subagents to complete tasks.
   - Each Subagent uses `web_search` and may `fs_write_report(path, markdown)`.
   - Each returns a minimal `summary`.

3. **Synthesize (decide)**
   - Lead reads artifacts and integrates results.
   - **Outputs either**:
     - **Done** → finalize answer, or
     - **Plan** → re-plan with a **refined query / updated To-Dos** and loop again.

4. **Iterate (optional)**
   - If “Plan,” update the To-Do List and generate additional tasks; then repeat Execute → Synthesize.

---

## Diagram

```mermaid
flowchart TD
  Q[User Query] --> P[Plan Agent (ReAct + To-Do)]
  P --> T[Task Artifacts]
  T --> E[Execute Subagents in Parallel]
  E --> S[Synthesize & Decide]
  S -->|refined plan / refined query| P
  S --> F[Finalize Answer]
