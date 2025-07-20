# Multi-Agent Research System Plans

## Overview
Strategic plans for multi-agent research system development. Each major feature gets its own worktree with dedicated tasks.md.

## Git Worktree Strategy
- **main/**: Core system maintenance and integration
- **feature branches**: Independent worktrees for parallel development
- **experiment/**: Detached HEAD worktree for quick tests (no branch needed)

## Active Plans

### 1. Extract Subagent Execution Logic ✅
**Goal**: Create dedicated helper function to handle subagent execution
**Status**: Active
**Benefits**: 
- Reduces complexity in main flow
- Centralizes error handling
- Makes testing easier

### 2. ~~Extract Tool Configuration~~ ❌
**Status**: SKIPPED (per user request - would worsen readability)

### 3. Simplify Parallel Execution ✅
**Goal**: Create helper for parallel task execution
**Status**: Active
**Benefits**:
- Cleaner async/await handling
- Centralized result processing
- Better error handling

### 4. Extract Logging/Debug Output ✅
**Goal**: Create consistent logging helpers
**Status**: Active
**Benefits**:
- Consistent output format
- Easier to control verbosity
- Cleaner main logic

### 5. Simplify Memory Operations ✅
**Goal**: Create wrappers for memory operations (JSON conversion only)
**Status**: Active
**Benefits**:
- Hide JSON conversion complexity
- Consistent memory key format
- Easier memory debugging
**Note**: No new functionality - just wrapping existing calls

### 6. Extract Model Configuration ✅
**Goal**: Centralize LM initialization (simple extraction only)
**Status**: Active
**Benefits**:
- Single place for model config
- Easier to swap models
- Cleaner initialization
**Note**: Just move existing LM initialization - no new ModelConfig class

### 7. Improve Main Flow in aforward() ✅
**Goal**: Simplify main orchestration logic
**Status**: Active
**Benefits**:
- Clear separation of phases
- Easier to understand flow
- Better maintainability

### 8. ~~Extract Response Preparation~~ ❌
**Status**: SKIPPED (per user request - would worsen readability)

### 9. Simplify Final Report Generation ✅
**Goal**: Extract final report logic from run()
**Status**: Active
**Benefits**:
- Cleaner run() method
- Reusable report generation
- Better separation of concerns

### 10. Consider Using Dataclasses for State ⏸️
**Goal**: Use dataclass for agent state
**Status**: Consider Later
**Benefits**:
- Cleaner state management
- Type safety
- Immutability options

### 11. Modularize Agent into Multiple Files ✅
**Goal**: Break down monolithic agent.py (611 lines) into focused modules
**Status**: Completed
**Benefits**:
- Better organization and maintainability
- Easier testing of individual components
- Clear separation of concerns
- Reusable components

**New File Structure**:
1. **config.py** - Environment variables, model constants, logging setup
2. **models.py** - Pydantic models and DSPy signatures 
3. **tools.py** - Class-based tool implementations with `__call__` methods
4. **utils.py** - Helper functions including prediction_to_json
5. **agent.py** - Refactored to use new modules, focus on orchestration

### 12. Unify Memory Tools for Better Code Organization 🚀
**Goal**: Consolidate MemoryReadTool and MemoryListTool into single MemoryTool class
**Status**: Completed
**Benefits**:
- Cleaner code structure with single memory tool class
- LLM-friendly interface maintained (separate tool names)
- Easier to extend with new memory operations
- Better encapsulation of memory-related functionality

**Implementation**:
- Single `MemoryTool` class with `read()` and `list()` methods
- Agent exposes methods as separate DSPy tools for LLM clarity
- No breaking changes to existing tool interfaces

## Plan Adjustments Log

### 2024-01-20
- Removed Plan #2 (Extract Tool Configuration) - would add unnecessary abstraction
- Removed Plan #8 (Extract Response Preparation) - response dict is simple enough inline
- Prioritizing plans that significantly reduce complexity without adding layers
- Adjusted Plan #5 - Remove memory state summary feature (new functionality, not refactoring)
- Adjusted Plan #6 - Simple LM initialization extraction only (no ModelConfig class)
- Clarified error handling and logging must preserve exact current behavior

## Success Metrics
- [ ] Main methods (aforward, run) under 50 lines each
- [ ] Helper functions under 20 lines each
- [ ] Clear separation between orchestration and execution
- [ ] Improved error handling and logging
- [ ] Maintained or improved performance