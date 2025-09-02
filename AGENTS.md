# Repository Guidelines

Contributions here should stay lean, well-tested, and consistent with the async, tool-driven research flow.

## Project Structure & Module Organization
- `agent.py`: Orchestrates planning, parallel subagents, synthesis, final report; writes artifacts to `memory/`.
- `models.py`: Pydantic data models and DSPy signatures; includes `FileSystem` for research memory.
- `tools.py`: Tool implementations (`WebSearchTool`, `FileSystemTool`, `ParallelSearchTool`).
- `config.py`: API keys, model names, and token/temperature settings.
- `dataset.py`, `eval.py`: BrowseComp dataset loader and DSPy evaluation program/metric.
- `tests/`: Pytest suite (unit + integration). Naming: `test_*.py`.
- `memory/`: Filesystem-based artifacts (e.g., `cycle_001/<task_name>/result.md`, `plan.md`, `synthesis.md`, `final_report.md`). Do not commit personal runs.
- `source_docs/`: Additional documentation.

## Build, Test, and Development Commands
- `uv sync`: Install/update all dependencies from `pyproject.toml` and `uv.lock`.
- `uv run pytest tests/`: Run the full test suite (async enabled).
- `uv run python agent.py`: Run the demo entrypoint (writes under `memory/`).
- `uv run python -c "from agent import LeadAgent; import asyncio; print(asyncio.run(LeadAgent().run('Your question')))"`: Ad‑hoc run.
- `./update.sh`: Refresh deps and run tests.

## Coding Style & Naming Conventions
- Python 3.11, PEP 8, 4‑space indentation; prefer type hints throughout.
- Functions/variables: `snake_case`; classes: `PascalCase`; modules: `snake_case.py`.
- Keep agent flows async-first; avoid blocking calls in tools (use `async`/executors).

- Simple Utilities (helpers/mini-functions): Default to a single-path solution with one output shape, no branching or speculative edge-case handling, and no extra state. If unsure, write quick tests and try potentially multiple variants; keep the simplest variant that passes and delete the rest.

## Testing Guidelines
- Frameworks: `pytest` + `pytest-asyncio` (`asyncio_mode=auto`).
- Async tests use `@pytest.mark.asyncio`. Name tests `test_*` and keep them deterministic (mock external APIs/tools).
- Examples: `uv run pytest -q`, or a subset `uv run pytest tests/test_agent_units.py -k test_full_agent_flow`.

## Commit & Pull Request Guidelines
- Use Conventional Commits where possible: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`.
- PRs: concise description, linked issues (`#123`), before/after notes (logs or small diffs), and passing tests (`uv run pytest`).
- Secrets: never commit `.env` or keys. Validate `.gitignore` and redact logs.

## Security & Configuration Tips
- `.env` must provide `BRAVE_SEARCH_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`.
- Networked tools (e.g., Brave Search) should be mocked/faked in tests.
- Generated `memory/` artifacts are local working data; clean as needed.
