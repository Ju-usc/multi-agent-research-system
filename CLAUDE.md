# Multi-Agent Research System - Design Specification

## System Overview
A production-grade multi-agent research system implementing Anthropic's orchestrator-worker architecture for complex research tasks. The system emphasizes parallel processing, token efficiency, and production reliability while maintaining research quality comparable to Claude's native capabilities.

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
- Always verify the specific syntax of library you are using via deepwiki mcp(mostly preferred), websearch, ask perpexlity, etc when implementing or desiging phases. 

### Tool guidelines

- use deepwiki mcp to ask any questions to refer official documentation of the library we are using (e.g. dspy, etc)
