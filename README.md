# Multi-Agent Research System

A minimal multi-agent research system, built with DSPy, inspired by [Anthropic’s multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).

reference: https://www.anthropic.com/engineering/multi-agent-research-system

> Documentation:
> - **architecture-agent.md** — flexible, subagent-spawning design (primary direction)
> - **architecture-workflow.md** — structured plan → execute → synthesize pipeline

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

## License

MIT
