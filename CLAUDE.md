# Multi-Agent Research System - Design Specification

## System Overview
A production-grade multi-agent research system implementing Anthropic's orchestrator-worker architecture for complex research tasks. The system emphasizes parallel processing, token efficiency, and production reliability while maintaining research quality comparable to Claude's native capabilities.

## Project Structure

```
multi-agent-research-system/
├── agent.py                    # Core multi-agent orchestration system with LeadAgent and subagent implementations
├── eval.py                     # BrowseComp evaluation framework with DSPy integration and LLM-as-judge metrics
├── dataset.py                  # BrowseComp dataset loader with XOR decryption and DSPy Example creation
├── utils.py                    # Utility functions for model configuration and Langfuse observability setup
├── README.md                   # Project overview and basic usage instructions
├── CLAUDE.md                   # This file - detailed design specs and guidelines for Claude Code
│
├── source_docs/                # Reference documentation from Anthropic's multi-agent research
│   ├── article.md             # Technical article explaining multi-agent architecture principles
│   ├── leadagent_prompt.md    # Reference prompt patterns for lead agent behavior
│   ├── subagent_prompt.md     # Reference prompt patterns for subagent task execution
│   └── open_ai_browsecomp_eval.py  # OpenAI's original BrowseComp evaluation code
│
├── tests/                      # Test suite
│   └── test_agent.py          # Unit tests for agent functionality and BrowseComp evaluation
│
└── .claude/                    # Claude Code configuration
    ├── commands/              # Custom slash commands
    │   ├── gitpush.md        # Git commit and push workflow command
    │   └── research.md       # Research task execution command
    └── settings.local.json    # Local Claude Code settings
```

### Key Files Overview

- **agent.py**: The heart of the system - implements LeadAgent (orchestrator), subagent workers, memory system, and all DSPy signatures for planning, execution, and synthesis
- **eval.py**: Evaluation pipeline that wraps LeadAgent in DSPy's Evaluate framework, implements answer correctness metrics, and supports multi-threaded evaluation
- **dataset.py**: Handles BrowseComp dataset operations including downloading, XOR decryption with canary checking, and conversion to DSPy Examples
- **utils.py**: Configuration hub for model selection (O4_MINI, GEMINI_2.5_FLASH) and Langfuse observability initialization

## Design Principles

### 1. Strategic Orchestration
- **Lead Agent as Coordinator**: Focuses exclusively on strategy, planning, and synthesis - never conducts primary research
- **Smart Query Classification**: Categorizes queries as depth-first (multiple perspectives), breadth-first (independent sub-questions), or straightforward
- **Detailed Task Decomposition**: Each subagent receives specific objectives, output formats, tool guidance, and clear boundaries

### 2. Efficiency Through Parallelization
- **Multi-level Concurrency**: Lead spawns 3-5 subagents in parallel; each subagent uses 3+ tools simultaneously
- **Token Budget Management**: Subagents have explicit "research budgets" (5-15 tool calls) to prevent inefficient exploration
- **Diminishing Returns Detection**: Agents recognize when further research yields minimal new information

### 3. Advanced Prompt Engineering
- **Extended Thinking Mode**: Visible thinking processes serve as controllable scratchpads for planning
- **OODA Loop Implementation**: Subagents follow Observe-Orient-Decide-Act cycles for systematic research
- **Scaling Heuristics**: 1 agent for simple queries, 2-4 for comparisons, 10+ for complex multi-part tasks

### 4. Production Reliability
- **Stateful Error Handling**: Resume from checkpoints rather than restarting expensive operations
- **Context Window Management**: Agents summarize and store essential information before hitting context limits
- **Memory Persistence**: Lead agents save plans to survive context truncation at 200k tokens

### 5. Quality Control
- **End-State Over Process**: Evaluate whether agents achieved correct outcomes, not prescribed paths
- **Multi-Criteria Evaluation**: Assess factual accuracy, citation quality, completeness, and tool efficiency
- **Source Quality Heuristics**: Prioritize primary sources, avoid SEO farms, distinguish speculation from facts

## Implementation Guidelines

### When Reviewing Code
1. **Verify Orchestration Pattern**: Lead agent should only plan and synthesize, never research directly
2. **Check Parallelization**: Ensure both agent-level and tool-level concurrent execution
3. **Validate Token Budgets**: Each subagent must have explicit limits on tool usage
4. **Assess Task Decomposition**: Subagent instructions should be highly specific and bounded

### Key Performance Targets
- **Token Multiplier**: Expect ~15x more tokens than single-agent chat
- **Quality Improvement**: Target 90%+ better performance on research tasks
- **Context Efficiency**: Each subagent acts as intelligent filter with own context window

### Source Documentation
- `source_docs/leadagent_prompt.md`: Reference implementation for lead agent behavior
- `source_docs/subagent_prompt.md`: Pattern for subagent task execution
- `source_docs/article.md`: Technical deep-dive on multi-agent system architecture

### General Guidelines

- Always try to explain in the planning phase what you are going to do as the user wants to learn technical details.
- Do not overcomplicate the planning phase try to simplify approaches only to contain the most important and necessary components.

## Important General MCP Guidelines
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing existing files over creating new ones
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested
- Use MCP tools proactively when they can enhance accuracy or provide better context
### mcp__Context7__resolve-library-id, get-library-docs
**When to use:**
- When needing up-to-date documentation for libraries/frameworks
- For researching API references and implementation patterns
- When implementing features using third-party libraries
- To understand best practices and usage examples for specific packages
**Best practices:**
- ALWAYS call `resolve-library-id` first to get the Context7-compatible library ID
- Use the returned library ID with `get-library-docs` to fetch documentation
- Specify a topic parameter to focus on specific aspects (e.g., 'hooks', 'routing')
- Adjust tokens parameter based on needed context (default: 10000)
- For ambiguous library names, select based on name similarity, description relevance, and trust score
**Important notes:**
- Only bypass `resolve-library-id` if user explicitly provides a library ID in format '/org/project' or '/org/project/version'
- Consider documentation coverage and trust scores when selecting from multiple matches
- Request clarification for highly ambiguous queries before proceeding
### mcp__deepwiki__read_wiki_structure, read_wiki_contents, ask_question
**When to use:**
- When working with GitHub repositories requiring documentation understanding
- Before implementing features to understand existing patterns/conventions
- To discover API references, installation guides, or architectural decisions
- When user asks about specific library/framework usage in a repo
**Best practices:**
- Always start with `read_wiki_structure` to understand available topics
- Use `read_wiki_contents` for detailed documentation on specific sections
- Use `ask_question` for targeted queries about the repository
- Combine these tools for comprehensive understanding before coding
**Important limitations:**
- Some repositories (e.g., stanfordnlp/dspy) have documentation exceeding the 25,000 token limit for `read_wiki_contents`
- When encountering token limit errors, use `ask_question` for specific queries instead
- For large documentation sets, prefer targeted questions over attempting to read entire contents

# Important Clean Code Rules

## Naming
- Use descriptive variable names
- Functions should do one thing
- Avoid abbreviations

## Functions
- Keep functions small (< 20 lines)
- Use early returns
- Avoid nested conditionals

## Comments
- Code should be self-documenting
- Comments explain why, not what
- Keep comments up to date

## Project Execution

**This project uses `uv` instead of `pip`/`python`**:
- Run files: `uv run python <file.py>`
- Run tests: `uv run pytest`
- Install deps: `uv sync`

## General Workflow

1. **Review plans.md** → Finalize/adjust plans
2. **Create/update tasks.md** → Based on finalized plans.md
3. **Implement** → Refer to tasks.md for specific steps
4. **Update tracking** → Mark completed tasks in tasks.md
5. **Update CLAUDE.md** → Update the CLAUDE.md file with the new information