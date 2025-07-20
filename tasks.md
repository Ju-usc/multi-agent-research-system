# Refactoring Tasks Breakdown

## Task Group 1: Extract Subagent Execution Logic

### Task 1.1: Create execute_subagent_task() helper
```python
async def execute_subagent_task(self, task: SubagentTask, subagent_lm) -> Optional[SubagentResult]:
    """Execute a single subagent task with proper tool setup and error handling."""
    # Implementation details
```
- [ ] Move subagent execution logic from aforward()
- [ ] Include tool setup (permitted_tools)
- [ ] Include DSPy context management
- [ ] Add try/except for error handling
- [ ] Return None on failure, SubagentResult on success

### Task 1.2: Update aforward() to use helper
- [ ] Replace inline subagent execution with helper call
- [ ] Maintain existing parallel execution structure

## Task Group 2: Simplify Parallel Execution

### Task 2.1: Create execute_tasks_parallel() helper
```python
async def execute_tasks_parallel(self, tasks: List[SubagentTask]) -> List[SubagentResult]:
    """Execute multiple subagent tasks in parallel and collect results."""
    # Implementation details
```
- [ ] Move asyncio.gather logic
- [ ] Handle exceptions from gather
- [ ] Filter valid results
- [ ] Write results to memory
- [ ] Return only valid SubagentResults

### Task 2.2: Simplify result processing
- [ ] Move result validation logic into helper
- [ ] Clean up logging for results

## Task Group 3: Extract Logging Helpers

### Task 3.1: Create log_plan_summary()
```python
def log_plan_summary(self, plan, cycle_idx: int):
    """Log planning phase results in consistent format."""
```
- [ ] Extract all print statements from planning phase
- [ ] Standardize output format
- [ ] Include cycle number, task count, reasoning preview

### Task 3.2: Create log_task_results()
```python
def log_task_results(self, results: List[SubagentResult], raw_results: List):
    """Log subagent execution results."""
```
- [ ] Extract result logging from aforward()
- [ ] Show success/failure counts
- [ ] Include error summaries

### Task 3.3: Create log_synthesis_summary()
```python
def log_synthesis_summary(self, decision):
    """Log synthesis phase results."""
```
- [ ] Extract synthesis logging
- [ ] Show decision (DONE/CONTINUE)
- [ ] Include gap analysis when continuing

## Task Group 4: Simplify Memory Operations

### Task 4.1: Create store_to_memory() wrapper
```python
async def store_to_memory(self, cycle: int, type: str, content: Any, task_id: Optional[int] = None) -> str:
    """Store content to memory with automatic JSON conversion."""
```
- [ ] Wrap memory.write()
- [ ] Handle JSON conversion for predictions
- [ ] Handle Pydantic model conversion
- [ ] Return memory key

### Task 4.2: Add memory state helpers
```python
def get_memory_state_summary(self) -> Dict[str, int]:
    """Get summary of current memory state."""
```
- [ ] Count items by type
- [ ] Return summary dict

## Task Group 5: Extract Model Configuration

### Task 5.1: Create ModelConfig class
```python
class ModelConfig:
    """Centralized model configuration."""
    def __init__(self):
        # Load from env
    
    def get_planner_lm(self):
    def get_subagent_lm(self):
    def get_synthesizer_lm(self):
```
- [ ] Move all LM initialization
- [ ] Handle API keys and env vars
- [ ] Provide factory methods

### Task 5.2: Update LeadAgent to use ModelConfig
- [ ] Replace inline LM creation
- [ ] Use factory methods
- [ ] Maintain existing functionality

## Task Group 6: Improve Main Flow

### Task 6.1: Create plan_research() method
```python
async def plan_research(self, query: str) -> PlanResult:
    """Execute planning phase and return plan."""
```
- [ ] Extract planning logic from aforward()
- [ ] Include logging
- [ ] Include memory write
- [ ] Update steps_trace

### Task 6.2: Create synthesize_results() method
```python
async def synthesize_results(self, query: str, results: List[SubagentResult]) -> SynthesisResult:
    """Execute synthesis phase and return decision."""
```
- [ ] Extract synthesis logic
- [ ] Include logging
- [ ] Include memory write
- [ ] Update steps_trace

### Task 6.3: Refactor aforward() to use new methods
- [ ] Replace inline logic with method calls
- [ ] Maintain return structure
- [ ] Keep under 50 lines

## Task Group 7: Simplify Final Report Generation

### Task 7.1: Create generate_final_report() method
```python
async def generate_final_report(self, query: str, final_synthesis: str) -> str:
    """Generate final markdown report from memory."""
```
- [ ] Extract from run() method
- [ ] Include ReAct setup
- [ ] Handle memory summaries
- [ ] Return markdown report

### Task 7.2: Simplify run() method
- [ ] Use generate_final_report()
- [ ] Clean up flow logic
- [ ] Improve readability

## Implementation Order

1. **Phase 1**: Logging helpers (Task Group 3) - Low risk, immediate benefit
2. **Phase 2**: Memory operations (Task Group 4) - Simplifies later tasks
3. **Phase 3**: Subagent execution (Task Group 1) - Core complexity reduction
4. **Phase 4**: Parallel execution (Task Group 2) - Builds on Phase 3
5. **Phase 5**: Main flow improvement (Task Group 6) - Uses all previous helpers
6. **Phase 6**: Model configuration (Task Group 5) - Clean up initialization
7. **Phase 7**: Final report (Task Group 7) - Final cleanup

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
- Keep original agent.py as agent_original.py before starting
- Implement changes incrementally
- Run tests after each phase
- Revert phase if tests fail