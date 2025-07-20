"""Unit tests for refactored LeadAgent methods."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agent import LeadAgent
from models import SubagentTask, SubagentResult, Memory
import json


class TestExecuteSubagentTask:
    """Test suite for execute_subagent_task method."""
    
    @pytest.fixture
    def lead_agent(self):
        """Create a LeadAgent instance for testing."""
        with patch('agent.setup_langfuse'):
            agent = LeadAgent()
            return agent
    
    @pytest.fixture
    def mock_task(self):
        """Create a mock SubagentTask."""
        return SubagentTask(
            task_id=1,
            objective="Test objective",
            tool_guidance={"web_search": "Search for test info"},
            tool_budget=5,
            expected_output="Test output",
            tip="Test tip"
        )
    
    @pytest.fixture
    def mock_subagent_lm(self):
        """Create a mock language model for subagent."""
        mock_lm = MagicMock()
        mock_lm.kwargs = {'model': 'test-model'}
        return mock_lm
    
    @pytest.mark.asyncio
    async def test_execute_subagent_task_success(self, lead_agent, mock_task, mock_subagent_lm):
        """Test successful execution of a subagent task."""
        # Mock the ReAct predict to return a successful result
        mock_result = SubagentResult(
            task_id=1,
            summary="Test summary",
            finding="Test finding",
            debug_info=["debug1", "debug2"]
        )
        
        mock_result = SubagentResult(
            task_id=1,
            summary="Test summary",
            finding="Test finding"
        )
        
        with patch('dspy.ReAct') as mock_react_class:
            mock_react = AsyncMock()
            mock_react.acall.return_value = MagicMock(
                final_result=mock_result
            )
            mock_react_class.return_value = mock_react
            
            result = await lead_agent.execute_subagent_task(mock_task, mock_subagent_lm)
            
            assert result is not None
            assert result.task_id == 1
            assert result.summary == "Test summary"
            assert result.finding == "Test finding"
    
    @pytest.mark.asyncio
    async def test_execute_subagent_task_failure(self, lead_agent, mock_task, mock_subagent_lm):
        """Test handling of subagent task execution failure."""
        with patch('dspy.ReAct') as mock_react_class:
            mock_react = AsyncMock()
            mock_react.acall.side_effect = Exception("Test error")
            mock_react_class.return_value = mock_react
            
            result = await lead_agent.execute_subagent_task(mock_task, mock_subagent_lm)
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_execute_subagent_task_with_tools(self, lead_agent, mock_task, mock_subagent_lm):
        """Test that tools are properly configured for subagent."""
        mock_result = SubagentResult(
            task_id=1,
            summary="Test summary",
            finding="Test finding"
        )
        
        with patch('dspy.ReAct') as mock_react_class:
            mock_react = AsyncMock()
            mock_react.acall.return_value = MagicMock(
                final_result=mock_result
            )
            
            # Check that ReAct is called with the right tools
            def check_tools(*_, **kwargs):
                tools = kwargs.get('tools', [])
                # Should have web_search and memory tools
                assert any(hasattr(tool, 'name') and 'search' in tool.name for tool in tools)
                assert any(hasattr(tool, 'name') and 'memory' in tool.name for tool in tools)
                return mock_react
                
            mock_react_class.side_effect = check_tools
            
            await lead_agent.execute_subagent_task(mock_task, mock_subagent_lm)


class TestExecuteTasksParallel:
    """Test suite for execute_tasks_parallel method."""
    
    @pytest.fixture
    def lead_agent(self):
        """Create a LeadAgent instance for testing."""
        with patch('agent.setup_langfuse'):
            agent = LeadAgent()
            return agent
    
    @pytest.fixture
    def mock_tasks(self):
        """Create mock SubagentTasks."""
        return [
            SubagentTask(
                task_id=i,
                objective=f"Test objective {i}",
                tool_guidance={"web_search": f"Search for test {i}"},
                tool_budget=5,
                expected_output=f"Test output {i}",
                tip=f"Test tip {i}"
            )
            for i in range(1, 4)
        ]
    
    @pytest.mark.asyncio
    async def test_execute_tasks_parallel_all_success(self, lead_agent, mock_tasks):
        """Test parallel execution with all tasks succeeding."""
        mock_results = [
            SubagentResult(
                task_id=i,
                summary=f"Summary {i}",
                finding=f"Finding {i}"
            )
            for i in range(1, 4)
        ]
        
        with patch.object(lead_agent, 'execute_subagent_task', new=AsyncMock()) as mock_execute:
            mock_execute.side_effect = mock_results
            
            results = await lead_agent.execute_tasks_parallel(mock_tasks)
            
            assert len(results) == 3
            assert all(isinstance(r, SubagentResult) for r in results)
            assert mock_execute.call_count == 3
    
    @pytest.mark.asyncio
    async def test_execute_tasks_parallel_partial_failure(self, lead_agent, mock_tasks):
        """Test parallel execution with some tasks failing."""
        mock_results = [
            SubagentResult(task_id=1, summary="Summary 1", finding="Finding 1"),
            None,  # Failed task
            SubagentResult(task_id=3, summary="Summary 3", finding="Finding 3")
        ]
        
        with patch.object(lead_agent, 'execute_subagent_task', new=AsyncMock()) as mock_execute:
            mock_execute.side_effect = mock_results
            
            results = await lead_agent.execute_tasks_parallel(mock_tasks)
            
            assert len(results) == 2  # Only successful results
            assert results[0].task_id == 1
            assert results[1].task_id == 3
    
    @pytest.mark.asyncio
    async def test_execute_tasks_parallel_memory_storage(self, lead_agent, mock_tasks):
        """Test that results are stored in memory."""
        mock_results = [
            SubagentResult(task_id=1, summary="Summary 1", finding="Finding 1")
        ]
        
        with patch.object(lead_agent, 'execute_subagent_task', new=AsyncMock()) as mock_execute:
            mock_execute.side_effect = mock_results
            
            _ = await lead_agent.execute_tasks_parallel(mock_tasks[:1])
            
            # Check memory was updated
            assert len(lead_agent.memory.store) > 0
            # Memory key format: '{cycle}-task-{task_id}'
            memory_keys = list(lead_agent.memory.store.keys())
            assert any('task-1' in key for key in memory_keys)


class TestStoreToMemory:
    """Test suite for store_to_memory method."""
    
    @pytest.fixture
    def lead_agent(self):
        """Create a LeadAgent instance for testing."""
        with patch('agent.setup_langfuse'):
            agent = LeadAgent()
            return agent
    
    @pytest.mark.asyncio
    async def test_store_to_memory_basic(self, lead_agent):
        """Test basic memory storage."""
        key = await lead_agent.store_to_memory(
            cycle=1,
            type="test",
            content="Test content"
        )
        
        assert key == "1-test"
        assert lead_agent.memory.store[key] == "Test content"
    
    @pytest.mark.asyncio
    async def test_store_to_memory_with_task_id(self, lead_agent):
        """Test memory storage with task ID."""
        key = await lead_agent.store_to_memory(
            cycle=1,
            type="result",
            content="Test content",
            task_id=5
        )
        
        assert key == "1-result-5"
        assert lead_agent.memory.store[key] == "Test content"
    
    @pytest.mark.asyncio
    async def test_store_to_memory_json_conversion(self, lead_agent):
        """Test JSON conversion for predictions."""
        # Create a mock prediction that has _store attribute
        class MockPrediction:
            def __init__(self):
                self._store = {"test": "data"}
        
        mock_prediction = MockPrediction()
        
        with patch('agent.prediction_to_json', return_value='{"test": "data"}') as mock_json:
            key = await lead_agent.store_to_memory(
                cycle=1,
                type="prediction",
                content=mock_prediction
            )
            
            assert key == "1-prediction"
            assert lead_agent.memory.store[key] == '{"test": "data"}'
            mock_json.assert_called_once_with(mock_prediction)
    
    @pytest.mark.asyncio
    async def test_store_to_memory_pydantic_conversion(self, lead_agent):
        """Test Pydantic model conversion."""
        task = SubagentTask(
            task_id=1,
            objective="Test",
            tool_guidance={"web_search": "test"},
            tool_budget=5,
            expected_output="output"
        )
        
        key = await lead_agent.store_to_memory(
            cycle=1,
            type="task",
            content=task
        )
        
        stored_content = json.loads(lead_agent.memory.store[key])
        assert stored_content["task_id"] == 1
        assert stored_content["objective"] == "Test"
    
    @pytest.mark.asyncio
    async def test_store_to_memory_with_summary(self, lead_agent):
        """Test that summaries are created for stored content."""
        # Patch the summarize method on the Memory class itself
        original_summarize = Memory.summarize
        
        async def mock_summarize(_, __):
            return "Test summary"
        
        Memory.summarize = mock_summarize
        
        try:
            key = await lead_agent.store_to_memory(
                cycle=1,
                type="test",
                content="Long test content that needs summarization"
            )
            
            assert key == "1-test"
            assert lead_agent.memory.summaries[key] == "Test summary"
        finally:
            # Restore original method
            Memory.summarize = original_summarize


class TestPlanResearch:
    """Test suite for plan_research method."""
    
    @pytest.fixture
    def lead_agent(self):
        """Create a LeadAgent instance for testing."""
        with patch('agent.setup_langfuse'):
            agent = LeadAgent()
            return agent
    
    @pytest.mark.asyncio
    async def test_plan_research_basic(self, lead_agent):
        """Test basic planning functionality."""
        mock_plan = MagicMock()
        mock_plan.reasoning = "Test reasoning"
        mock_plan.tasks = [
            SubagentTask(
                task_id=1,
                objective="Test objective",
                tool_guidance={"web_search": "test"},
                tool_budget=5,
                expected_output="output"
            )
        ]
        
        with patch.object(lead_agent.planner, 'acall', new=AsyncMock()) as mock_plan_call:
            mock_plan_call.return_value = mock_plan
            
            result = await lead_agent.plan_research("Test query")
            
            assert result == mock_plan
            mock_plan_call.assert_called_once()
            
            # Check that plan was stored in memory
            memory_keys = list(lead_agent.memory.store.keys())
            assert any('plan' in key for key in memory_keys)
    
    @pytest.mark.asyncio
    async def test_plan_research_updates_trace(self, lead_agent):
        """Test that planning updates the steps trace."""
        mock_plan = MagicMock()
        mock_plan.reasoning = "Test reasoning"
        mock_plan.tasks = []
        
        with patch.object(lead_agent.planner, 'acall', new=AsyncMock()) as mock_plan_call:
            mock_plan_call.return_value = mock_plan
            
            await lead_agent.plan_research("Test query")
            
            assert len(lead_agent.steps_trace) > 0
            assert lead_agent.steps_trace[-1]['action'] == 'Planned'


class TestSynthesizeResults:
    """Test suite for synthesize_results method."""
    
    @pytest.fixture
    def lead_agent(self):
        """Create a LeadAgent instance for testing."""
        with patch('agent.setup_langfuse'):
            agent = LeadAgent()
            return agent
    
    @pytest.fixture
    def mock_results(self):
        """Create mock SubagentResults."""
        return [
            SubagentResult(
                task_id=1,
                summary="Summary 1",
                finding="Finding 1"
            ),
            SubagentResult(
                task_id=2,
                summary="Summary 2", 
                finding="Finding 2"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_synthesize_results_basic(self, lead_agent, mock_results):
        """Test basic synthesis functionality."""
        mock_synthesis = MagicMock()
        mock_synthesis.decision = "continue"
        mock_synthesis.reasoning = "Test reasoning"
        mock_synthesis.synthesis = "Test synthesis"
        
        with patch.object(lead_agent.synthesizer, 'acall', new=AsyncMock()) as mock_synth_call:
            mock_synth_call.return_value = mock_synthesis
            
            result = await lead_agent.synthesize_results("Test query", mock_results)
            
            assert result == mock_synthesis
            mock_synth_call.assert_called_once()
            
            # Check that synthesis was stored in memory
            memory_keys = list(lead_agent.memory.store.keys())
            assert any('synthesis' in key for key in memory_keys)
    
    @pytest.mark.asyncio
    async def test_synthesize_results_updates_trace(self, lead_agent, mock_results):
        """Test that synthesis updates the steps trace."""
        mock_synthesis = MagicMock()
        mock_synthesis.decision = "continue"
        mock_synthesis.reasoning = "Test reasoning"
        mock_synthesis.synthesis = "Test synthesis"
        
        with patch.object(lead_agent.synthesizer, 'acall', new=AsyncMock()) as mock_synth_call:
            mock_synth_call.return_value = mock_synthesis
            
            await lead_agent.synthesize_results("Test query", mock_results)
            
            assert len(lead_agent.steps_trace) > 0
            assert lead_agent.steps_trace[-1]['action'] == 'Synthesized'


class TestIntegration:
    """Integration tests for end-to-end agent flow."""
    
    @pytest.fixture
    def lead_agent(self):
        """Create a LeadAgent instance for testing."""
        with patch('agent.setup_langfuse'):
            agent = LeadAgent()
            return agent
    
    @pytest.mark.asyncio
    async def test_full_agent_flow(self, lead_agent):
        """Test complete agent flow from query to synthesis."""
        query = "What is the capital of France?"
        
        # Mock plan
        mock_plan = MagicMock()
        mock_plan.reasoning = "Need to find capital of France"
        mock_plan.tasks = [
            SubagentTask(
                task_id=1,
                objective="Find the capital of France",
                tool_guidance={"web_search": "Search for France capital"},
                tool_budget=5,
                expected_output="The capital city name"
            )
        ]
        
        # Mock subagent result
        mock_result = SubagentResult(
            task_id=1,
            summary="Paris is the capital of France",
            finding="Paris is the capital and largest city of France"
        )
        
        # Mock synthesis
        mock_synthesis = MagicMock()
        mock_synthesis.is_done = True
        mock_synthesis.reasoning = "Found complete answer"
        mock_synthesis.synthesis = "The capital of France is Paris"
        
        with patch.object(lead_agent.planner, 'acall', new=AsyncMock(return_value=mock_plan)):
            with patch.object(lead_agent, 'execute_subagent_task', new=AsyncMock(return_value=mock_result)):
                with patch.object(lead_agent.synthesizer, 'acall', new=AsyncMock(return_value=mock_synthesis)):
                    
                    result = await lead_agent.aforward(query)
                    
                    assert result is not None
                    assert 'synthesis' in result
                    assert result['synthesis'] == "The capital of France is Paris"
                    assert result['is_done'] == True
                    assert result['results'] == [mock_result]
    
    @pytest.mark.asyncio
    async def test_multiple_cycles(self, lead_agent):
        """Test agent handling multiple research cycles."""
        query = "Compare Python and JavaScript"
        
        # First cycle plan
        plan1 = MagicMock()
        plan1.reasoning = "Need to research both languages"
        plan1.subagent_tasks = [
            SubagentTask(
                task_id=1,
                objective="Research Python features",
                tool_guidance={"web_search": "Python programming language"},
                tool_budget=5,
                expected_output="Python features"
            )
        ]
        
        # First cycle result
        result1 = SubagentResult(
            task_id=1,
            summary="Python is high-level language",
            finding="Python details..."
        )
        
        # First synthesis (continue)
        synthesis1 = MagicMock()
        synthesis1.is_done = False
        synthesis1.reasoning = "Need JavaScript info"
        synthesis1.synthesis = "Python research complete, need JavaScript"
        
        # Second cycle plan
        plan2 = MagicMock()
        plan2.reasoning = "Research JavaScript"
        plan2.subagent_tasks = [
            SubagentTask(
                task_id=2,
                objective="Research JavaScript features",
                tool_guidance={"web_search": "JavaScript programming"},
                tool_budget=5,
                expected_output="JavaScript features"
            )
        ]
        
        # Second cycle result
        result2 = SubagentResult(
            task_id=2,
            summary="JavaScript is web language",
            finding="JavaScript details..."
        )
        
        # Final synthesis
        synthesis2 = MagicMock()
        synthesis2.is_done = True
        synthesis2.reasoning = "Have complete comparison"
        synthesis2.synthesis = "Python vs JavaScript comparison complete"
        
        # Need to simulate two full cycles by calling aforward twice
        with patch.object(lead_agent.planner, 'acall', new=AsyncMock(side_effect=[plan1, plan2])):
            with patch.object(lead_agent, 'execute_subagent_task', new=AsyncMock(side_effect=[result1, result2])):
                with patch.object(lead_agent.synthesizer, 'acall', new=AsyncMock(side_effect=[synthesis1, synthesis2])):
                    
                    # First cycle
                    result1_dict = await lead_agent.aforward(query)
                    assert result1_dict['is_done'] == False
                    assert lead_agent.cycle_idx == 1
                    
                    # Second cycle  
                    result2_dict = await lead_agent.aforward(query)
                    assert result2_dict['is_done'] == True
                    assert lead_agent.cycle_idx == 2
    
    @pytest.mark.asyncio  
    async def test_error_recovery(self, lead_agent):
        """Test agent handling errors gracefully."""
        query = "Test error handling"
        
        # Mock plan
        mock_plan = MagicMock()
        mock_plan.reasoning = "Test plan"
        mock_plan.tasks = [
            SubagentTask(
                task_id=1,
                objective="Task that will fail",
                tool_guidance={"web_search": "test"},
                tool_budget=5,
                expected_output="output"
            ),
            SubagentTask(
                task_id=2,
                objective="Task that will succeed",
                tool_guidance={"web_search": "test"},
                tool_budget=5,
                expected_output="output"
            )
        ]
        
        # Mock results - one fails, one succeeds
        results = [
            None,  # Failed task
            SubagentResult(
                task_id=2,
                summary="Success",
                finding="Successful finding"
            )
        ]
        
        # Mock synthesis
        mock_synthesis = MagicMock()
        mock_synthesis.is_done = True
        mock_synthesis.reasoning = "Partial success"
        mock_synthesis.synthesis = "Got partial results"
        
        with patch.object(lead_agent.planner, 'acall', new=AsyncMock(return_value=mock_plan)):
            with patch.object(lead_agent, 'execute_subagent_task', new=AsyncMock(side_effect=results)):
                with patch.object(lead_agent.synthesizer, 'acall', new=AsyncMock(return_value=mock_synthesis)):
                    
                    result = await lead_agent.aforward(query)
                    
                    # Should continue despite one failure
                    assert result is not None
                    assert len(result['results']) == 1  # Only successful result
                    assert result['results'][0].task_id == 2
    
    @pytest.mark.asyncio
    async def test_memory_persistence(self, lead_agent):
        """Test that memory persists across cycles."""
        query = "Test memory"
        
        # Store initial data in memory
        await lead_agent.store_to_memory(0, "initial", "Initial data")
        
        # Mock simple flow
        mock_plan = MagicMock()
        mock_plan.reasoning = "Test"
        mock_plan.tasks = []
        
        mock_synthesis = MagicMock()
        mock_synthesis.is_done = True
        mock_synthesis.reasoning = "Done"
        mock_synthesis.synthesis = "Complete"
        
        with patch.object(lead_agent.planner, 'acall', new=AsyncMock(return_value=mock_plan)):
            with patch.object(lead_agent.synthesizer, 'acall', new=AsyncMock(return_value=mock_synthesis)):
                
                await lead_agent.aforward(query)
                
                # Check initial data still in memory
                assert "0-initial" in lead_agent.memory.store
                assert lead_agent.memory.store["0-initial"] == "Initial data"
                
                # Check new data was added
                assert any('plan' in key for key in lead_agent.memory.store.keys())
                assert any('synthesis' in key for key in lead_agent.memory.store.keys())