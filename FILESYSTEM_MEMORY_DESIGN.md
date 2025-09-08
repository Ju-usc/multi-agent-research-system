# FileSystem Memory Design

## Overview
Replace the complex key-value memory system with a simple file-based approach using markdown files.

## Architecture

### Directory Structure
```
memory/
├── cycle_001/
│   ├── compare-python-javascript.md     # Plan (LLM-generated filename)
│   ├── research-python-async/           # Task directory (LLM-generated name)
│   │   ├── result.md                   # Task execution result
│   │   └── tools/                      # Tool outputs
│   │       ├── async-guide-2024.md     # Web search results
│   │       └── python-coroutines.md    # More search results
│   ├── analyze-javascript-promises/     # Another task
│   │   └── result.md
│   └── synthesis.md                    # Cycle synthesis & decision
├── cycle_002/
│   └── ...
└── final_report.md
```

### Implementation

**FileSystemTool** (tools.py)
- Canonical filesystem implementation for research memory
- Default root is `"memory"` via `__init__(root="memory")`
- Methods: `write()`, `read()`, `tree()`, `exists()`, `clear()`
- Tree output: Simple path listing for LLM parsing

**Key Changes**
- `SubagentTask.task_name`: Filesystem-friendly directory name
- `PlanResearch.plan_filename`: Descriptive plan filename  
- `memory_tree` replaces `memory_summaries` in DSPy signatures
- WebSearchTool returns plain text (DSPy compatibility)

### Benefits
- No JSON serialization complexity
- File structure serves as memory overview
- Human-readable markdown files
- Direct pathlib usage
- Easy debugging and inspection
