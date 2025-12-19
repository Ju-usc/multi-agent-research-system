# Multi-Agent Research System - Design Specification

## System Overview
A production-grade multi-agent research system implementing Anthropic's orchestrator-worker architecture for complex research tasks. The system emphasizes parallel processing, token efficiency, and production reliability while maintaining research quality comparable to Claude's native capabilities.

## Project Structure

```
multi-agent-research-system/
├── agent.py                    # Refactored orchestration system - LeadAgent with helper methods
├── config.py                   # Environment variables, model constants, and logging configuration
├── models.py                   # Pydantic models and DSPy signatures for the multi-agent system
├── tools.py                    # Class-based tool implementations with async support
├── utils.py                    # CLI utilities and workspace helpers
├── eval.py                     # BrowseComp evaluation with accuracy metrics and GEPA optimization
├── dataset.py                  # BrowseComp dataset loader with XOR decryption and DSPy Example creation
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
│   ├── test_agent.py          # Integration tests for agent functionality and BrowseComp evaluation
│   └── test_agent_units.py    # Comprehensive unit tests for refactored LeadAgent methods
│
└── .claude/                    # Claude Code configuration
    ├── commands/              # Custom slash commands
    │   ├── gitpush.md        # Git commit and push workflow command
    │   └── research.md       # Research task execution command
    └── settings.local.json    # Local Claude Code settings
```

### Key Files Overview

- **agent.py**: Refactored orchestration system - implements LeadAgent with clean separation of concerns through helper methods for planning, execution, and synthesis
- **config.py**: Centralized configuration for environment variables, model constants, and logging setup
- **models.py**: All Pydantic models (SubagentTask, SubagentResult, Memory) and DSPy signatures in one place
- **tools.py**: Class-based tool implementations (WebSearchTool, MemoryTool) with async support
- **utils.py**: CLI utilities including argument parsing and workspace helpers
- **eval.py**: Evaluation with accuracy metrics, GEPA optimization, and DSPy's built-in result saving
- **dataset.py**: Handles BrowseComp dataset operations including downloading, XOR decryption with canary checking, and conversion to DSPy Examples

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

## Refactored Architecture

### Modular Design
The codebase has been refactored from a monolithic 611-line agent.py into focused modules:

1. **Separation of Concerns**
   - Configuration separated into `config.py`
   - Data models extracted to `models.py`
   - Tool implementations moved to `tools.py`
   - Helper utilities in `utils.py`
   - Agent logic focused purely on orchestration

2. **Clean Code Improvements**
   - Helper methods for complex operations (execute_subagent_task, execute_tasks_parallel)
   - Consistent logging using Python's standard logging module
   - Memory operations wrapped for cleaner JSON handling
   - Main methods (aforward, run) reduced to under 50 lines each

3. **Testing Structure**
   - `test_agent.py`: Integration tests for end-to-end functionality
   - `test_agent_units.py`: Comprehensive unit tests with proper mocking
   - 19 unit tests covering all refactored methods
   - Tests for error handling, parallel execution, and memory persistence

### Key Refactored Methods

1. **execute_subagent_task()**: Handles individual subagent execution with error handling
2. **execute_tasks_parallel()**: Manages concurrent subagent execution and result collection
3. **store_to_memory()**: Simplified memory storage with automatic JSON conversion
4. **plan_research()**: Extracted planning logic with trace updates
5. **synthesize_results()**: Isolated synthesis phase with decision making
6. **generate_final_report()**: Separated final report generation using ReAct
7. **init_language_models()**: Centralized LM initialization

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

### Design Before Implementation

For non-trivial changes:
1. Write a design spec with options and tradeoffs
2. Get user approval before writing code
3. Only then implement

This prevents wasted effort and ensures alignment.

### Clarify Before Implementing

Before starting implementation:
1. Make sure the user understands what's going to change
2. Explain the approach in plain language
3. Ask clarifying questions if requirements are ambiguous
4. Don't jump into code while the user is still asking questions

The user should feel informed, not surprised.

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

# Clean Code: Newspaper Article Style

Write code like a newspaper article: **most important information first, details later, plain language throughout.**

## The Inverted Pyramid

```
┌─────────────────────────────────┐
│     PUBLIC API / MAIN LOGIC     │  ← Reader sees this first
├─────────────────────────────────┤
│      SUPPORTING METHODS         │  ← Important helpers
├─────────────────────────────────┤
│    PRIVATE IMPLEMENTATION       │  ← Implementation details
└─────────────────────────────────┘
```

**In practice:**
- Put `__init__`, public methods, and main logic at TOP of class
- Put private helpers (`_method`) BELOW public methods
- Put constants and utilities at module level, ABOVE classes

## Naming: Write Headlines, Not Puzzles

```python
# ❌ Bad: Reader must decode
def proc_tkn(t, m):
    return t / 1e6 * m

# ✅ Good: Self-explanatory headline
def calculate_cost_usd(token_count, price_per_million):
    return token_count / 1_000_000 * price_per_million
```

**Rules:**
- Names should read like newspaper headlines
- Avoid abbreviations (`tkn` → `token`, `proc` → `process`)
- Use domain language the team understands
- Boolean variables: `is_`, `has_`, `can_`, `should_`
- Functions: verb + noun (`calculate_cost`, `register_module`)

## Functions: One Paragraph, One Idea

```python
# ❌ Bad: Multiple ideas, hard to scan
def process(data):
    # validate
    if not data: raise ValueError()
    if len(data) > 100: data = data[:100]
    # transform
    result = []
    for item in data:
        if item.valid:
            result.append(item.value * 2)
    # aggregate
    return sum(result) / len(result) if result else 0

# ✅ Good: One idea per function
def process(data):
    validated = validate_data(data)
    transformed = transform_items(validated)
    return calculate_average(transformed)
```

**Rules:**
- Functions do ONE thing (single responsibility)
- Keep under 20 lines (ideally under 10)
- Use early returns to reduce nesting
- If you need a comment to explain a block, extract it to a function

## Structure: Answer WHO/WHAT/WHY First

```python
# ✅ Good: Module docstring answers the big questions
"""
Per-module token usage tracking for multi-agent research system.

WHO: Used by eval.py to track costs during evaluation
WHAT: Captures token usage per registered DSPy module
WHY: Provides cost breakdown for GEPA optimization feedback
"""

class PerModuleUsageTracker:
    """Track token usage per registered module.

    Usage:
        tracker = PerModuleUsageTracker()
        tracker.register("lead_agent", my_module)
    """
```

**Rules:**
- Module docstring: What this file does, who uses it, why it exists
- Class docstring: What it is + minimal usage example
- Method docstring: Only if the name isn't self-explanatory
- Inline comments: Explain WHY, never WHAT

## Data Flow: Make It Obvious

```python
# ❌ Bad: Hidden state changes
class Tracker:
    def process(self, data):
        self._internal = transform(data)  # Hidden side effect
        self._cache[id(data)] = self._internal  # Another hidden change
        return self._internal

# ✅ Good: Explicit data flow
class Tracker:
    def process(self, data):
        result = transform(data)
        self._cache_result(data, result)  # Named side effect
        return result
```

**Rules:**
- Make inputs and outputs explicit
- Name side effects clearly
- Prefer pure functions when possible
- Use `_` prefix for internal state

## Quick Checklist

Before committing, ask:

1. **Can a beginner understand this in 30 seconds?**
2. **Is the most important code at the top?**
3. **Do names read like headlines?**
4. **Does each function do exactly one thing?**
5. **Are there any magic numbers or cryptic abbreviations?**

---

# nanoGPT Style (Advanced)

For code that needs to be **minimal, dense, logical, and easy to reverse-engineer**.

Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - code so clean that anyone can read it, understand it, and reproduce the logic.

## Core Principles

1. **Every line earns its place** - no boilerplate, no over-abstraction
2. **Logic flows linearly** - read top-to-bottom, understand the algorithm
3. **Comments explain WHY, not WHAT** - only where you diverge from standard practice
4. **Visible intermediate steps** - don't chain everything into clever one-liners

## Example: nanoGPT Style

```python
class Block(nn.Module):
    """Transformer block: attention + feedforward with residual connections."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

**Why this works:**
- Dense but readable - every line does something meaningful
- Modular - small composable pieces
- Linear flow - input → attention → mlp → output
- No comments needed - the code explains itself

## Quick Checklist (nanoGPT)

1. **Can someone reproduce this logic from reading the code?**
2. **Does every line earn its place?**
3. **Does the logic flow linearly (top to bottom)?**
4. **Are comments explaining WHY, not WHAT?**
5. **Would Karpathy approve?**

## Project Execution

**This project uses `uv` instead of `pip`/`python`**:
- Run files: `uv run python <file.py>`
- Run tests: `uv run pytest`
- Run specific test file: `uv run pytest tests/test_agent_units.py -v`
- Run evaluation: `uv run python eval.py --num-examples 5`
- Run with GEPA optimization: `uv run python eval.py --optimize --optimize-steps 10`
- Install deps: `uv sync`

## Testing

The project includes comprehensive test coverage:

### Unit Tests (`tests/test_agent_units.py`)
- Tests for all refactored helper methods with proper mocking
- Coverage for error handling, parallel execution, and memory operations
- 19 tests covering execute_subagent_task, execute_tasks_parallel, store_to_memory, plan_research, synthesize_results
- Integration tests for full agent flow including multiple cycles and error recovery

### Integration Tests (`tests/test_agent.py`)
- End-to-end agent functionality tests
- BrowseComp evaluation framework tests
- Real agent execution with actual LLM calls

## General Workflow

1. **Review plans.md** → Finalize/adjust plans
2. **Create/update tasks.md** → Based on finalized plans.md
3. **Implement** → Refer to tasks.md for specific steps
4. **Update tracking** → Mark completed tasks in tasks.md
5. **Update CLAUDE.md** → Update the CLAUDE.md file with the new information

## Git Worktree Workflow

### Setup
```bash
# Create worktree structure
mkdir ~/project-worktrees && cd ~/project-worktrees
git clone <repo> main && cd main
git worktree add ../feature-x -b feature/x
git worktree add ../experiment --detach  # For quick experiments
```

### Key Commands
- **Add worktree**: `git worktree add ../name -b branch-name`
- **List worktrees**: `git worktree list`
- **Remove worktree**: `git worktree remove ../name`
- **Switch branches**: `cd ../experiment && git checkout <branch>` (test without affecting main)

### Productivity Pattern
```
project-worktrees/
├── main/          # Primary development
├── plans.md       # High-level strategy (shared)
├── feature-x/
│   └── tasks.md   # Feature-specific tasks
├── feature-y/
│   └── tasks.md   # Feature-specific tasks
└── experiment/    # Temporary experiments (detached HEAD)
```

### Best Practices
- Run Claude instances in parallel: one per worktree
- Each worktree maintains independent state (no stashing needed)
- Experiment freely without creating branches: `git worktree add -d ../experiment`
- Clean up worktrees after merging: `git worktree remove ../feature-x`