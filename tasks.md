# Refactoring Tasks Breakdown

## Task Group 1: Extract Subagent Execution Logic ✅ COMPLETED

### Task 1.1: Create execute_subagent_task() helper ✅
```python
async def execute_subagent_task(self, task: SubagentTask, subagent_lm) -> Optional[SubagentResult]:
    """Execute a single subagent task with proper tool setup and error handling."""
    # Implementation details
```
- [x] Move subagent execution logic from aforward()
- [x] Include tool setup (permitted_tools)
- [x] Include DSPy context management
- [x] Add try/except for error handling
- [x] Return None on failure, SubagentResult on success

### Task 1.2: Update aforward() to use helper ✅
- [x] Replace inline subagent execution with helper call
- [x] Maintain existing parallel execution structure

## Task Group 2: Simplify Parallel Execution ✅ COMPLETED

### Task 2.1: Create execute_tasks_parallel() helper ✅
```python
async def execute_tasks_parallel(self, tasks: List[SubagentTask]) -> List[SubagentResult]:
    """Execute multiple subagent tasks in parallel and collect results."""
    # Implementation details
```
- [x] Move asyncio.gather logic
- [x] Maintain EXACT current error handling (return_exceptions=True)
- [x] Filter valid results
- [x] Write results to memory
- [x] Return only valid SubagentResults

### Task 2.2: Simplify result processing ✅
- [x] Move result validation logic into helper
- [x] Clean up logging for results

## Task Group 3: Extract Logging Helpers ✅ COMPLETED

### Implementation Summary:
Instead of creating separate logging methods, we implemented the simplest approach using Python's built-in logging module:
- Added `logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')`
- Created `logger = logging.getLogger(__name__)`
- Replaced all `print()` statements with appropriate `logger.info()`, `logger.debug()`, `logger.error()`, and `logger.warning()` calls
- Preserved exact output format including emojis
- Used log levels appropriately:
  - INFO: Main progress updates with emojis
  - DEBUG: Detailed information (summaries, reasoning)
  - ERROR: Subagent failures
  - WARNING: Invalid results or incomplete research

Benefits achieved:
- Standard Python logging (no custom methods needed)
- Can easily adjust verbosity with log level
- Preserved visual output format
- Clean, industry-standard approach

## Task Group 4: Simplify Memory Operations ✅ COMPLETED

### Task 4.1: Create store_to_memory() wrapper ✅
```python
async def store_to_memory(self, cycle: int, type: str, content: Any, task_id: Optional[int] = None) -> str:
    """Store content to memory with automatic JSON conversion."""
```
- [x] Wrap memory.write()
- [x] Handle JSON conversion for predictions
- [x] Handle Pydantic model conversion
- [x] Return memory key
- [x] Update all memory.write() calls to use wrapper

### Task 4.2: ~~Add memory state helpers~~ ❌ REMOVED
**Reason**: This adds new functionality, not refactoring. Removed from scope.

## Task Group 5: Extract Model Configuration ✅ COMPLETED

### Task 5.1: Create init_language_models() method ✅
```python
def init_language_models(self):
    """Initialize all language models in one place."""
```
- [x] Move all LM initialization from __init__
- [x] Keep same configuration values
- [x] No new class or abstraction

### Task 5.2: Update LeadAgent __init__ to use helper ✅
- [x] Replace inline LM creation with method call
- [x] Keep everything else the same

## Task Group 6: Improve Main Flow ✅ COMPLETED

### Task 6.1: Create plan_research() method ✅
```python
async def plan_research(self, query: str) -> PlanResult:
    """Execute planning phase and return plan."""
```
- [x] Extract planning logic from aforward()
- [x] Include logging
- [x] Include memory write
- [x] Update steps_trace

### Task 6.2: Create synthesize_results() method ✅
```python
async def synthesize_results(self, query: str, results: List[SubagentResult]) -> SynthesisResult:
    """Execute synthesis phase and return decision."""
```
- [x] Extract synthesis logic
- [x] Include logging
- [x] Include memory write
- [x] Update steps_trace

### Task 6.3: Refactor aforward() to use new methods ✅
- [x] Replace inline logic with method calls
- [x] Maintain return structure
- [x] Keep under 50 lines (achieved ~19 lines)

## Task Group 7: Simplify Final Report Generation ✅ COMPLETED

### Task 7.1: Create generate_final_report() method ✅
```python
async def generate_final_report(self, query: str, final_synthesis: str) -> str:
    """Generate final markdown report from memory."""
```
- [x] Extract from run() method
- [x] Include ReAct setup
- [x] Handle memory summaries
- [x] Return markdown report

### Task 7.2: Simplify run() method ✅
- [x] Use generate_final_report()
- [x] Clean up flow logic
- [x] Improve readability

## Task Group 8: Modularize Agent into Multiple Files ✅ COMPLETED

### Task 8.1: Create config.py ✅
```python
# Environment variables and constants
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# ... other config
```
- [x] Move all environment variable loading
- [x] Move model configuration constants
- [x] Move temperature and token limits
- [x] Set up logging configuration
- [x] Add necessary imports

### Task 8.2: Create models.py ✅
```python
# Data models and DSPy signatures
class SubagentTask(BaseModel): ...
class SubagentResult(BaseModel): ...
class Memory(BaseModel): ...
# DSPy signatures
class MemorySummary(dspy.Signature): ...
```
- [x] Move all Pydantic models (SubagentTask, SubagentResult, Memory)
- [x] Move all DSPy signatures
- [x] Import config for model initialization in Memory
- [x] Ensure all imports are correct

### Task 8.3: Create tools.py with class-based tools ✅
```python
class WebSearchTool:
    def __init__(self, api_key: str):
        self.client = BraveSearch(api_key=api_key)
    
    async def __call__(self, query: str, count: int = 5) -> str:
        # Implementation
```
- [x] Create WebSearchTool class
- [x] Create WikipediaSearchTool class
- [x] Create ParallelSearchTool class
- [x] Create MemoryReadTool and MemoryListTool classes
- [x] Ensure proper async/sync handling
- [x] Add error handling

### Task 8.4: Update utils.py ✅
- [x] Move prediction_to_json() function from agent.py
- [x] Keep existing setup_langfuse() function
- [x] Add any other helper functions

### Task 8.5: Refactor agent.py ✅
- [x] Update imports to use new modules
- [x] Initialize tools as class instances
- [x] Remove all moved code
- [x] Ensure LeadAgent still works correctly

### Task 8.6: Update dependent files ✅
- [x] Fix imports in eval.py (no changes needed)
- [x] Fix imports in test_agent.py
- [x] Run tests to ensure everything works

## Implementation Order

1. **Phase 1**: Logging helpers (Task Group 3) ✅ - COMPLETED
2. **Phase 2**: Memory operations (Task Group 4) ✅ - COMPLETED  
3. **Phase 3**: Subagent execution (Task Group 1) ✅ - COMPLETED
4. **Phase 4**: Parallel execution (Task Group 2) ✅ - COMPLETED
5. **Phase 5**: Modularization (Task Group 8) ✅ - COMPLETED
6. **Phase 6**: Model configuration (Task Group 5) ✅ - COMPLETED
7. **Phase 7**: Main flow improvement (Task Group 6) ✅ - COMPLETED
8. **Phase 8**: Final report (Task Group 7) ✅ - COMPLETED

## Testing Strategy

### Unit Tests to Add
- [ ] Test execute_subagent_task with mock task
- [ ] Test execute_tasks_parallel with various results
- [ ] Test memory helpers
- [ ] Test logging helpers (output capture)

### Integration Tests
- [ ] Ensure refactored agent produces same results
- [ ] Test error handling paths
- [ ] Verify memory consistency

## Rollback Plan
- Implement changes incrementally
- Run tests after each phase
- Revert phase if tests fail