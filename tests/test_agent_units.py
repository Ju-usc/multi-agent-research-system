"""Unit tests for refactored LeadAgent methods (FileSystem-based)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import importlib
import sys
import types
from models import SubagentTask, SubagentResult
import utils as _utils


class TestExecuteSubagentTask:
    """Test suite for execute_subagent_task method."""
    
    @pytest.fixture
    def lead_agent(self):
        """Create a LeadAgent instance for testing without Langfuse side effects."""
        fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
        sys.modules['langfuse'] = fake_langfuse
        with patch.object(_utils, 'setup_langfuse', return_value=None):
            agent_module = importlib.import_module('agent')
            LeadAgent = getattr(agent_module, 'LeadAgent')
            agent = LeadAgent()
            return agent
    
    @pytest.fixture
    def mock_task(self):
        """Create a mock SubagentTask."""
        return SubagentTask(
            task_name="test-task-1",
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
            task_name=mock_task.task_name,
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
            assert result.task_name == mock_task.task_name
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
            task_name=mock_task.task_name,
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
                # Should include web_search and filesystem tools
                assert any(getattr(tool, 'name', '') == 'web_search' for tool in tools)
                assert any(getattr(tool, 'name', '') == 'filesystem_read' for tool in tools)
                assert any(getattr(tool, 'name', '') == 'filesystem_tree' for tool in tools)
                return mock_react
                
            mock_react_class.side_effect = check_tools
            
            await lead_agent.execute_subagent_task(mock_task, mock_subagent_lm)


class TestExecuteTasksParallel:
    """Test suite for execute_tasks_parallel method."""
    
    @pytest.fixture
    def lead_agent(self):
        """Create a LeadAgent instance for testing without Langfuse side effects."""
        fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
        sys.modules['langfuse'] = fake_langfuse
        with patch.object(_utils, 'setup_langfuse', return_value=None):
            agent_module = importlib.import_module('agent')
            LeadAgent = getattr(agent_module, 'LeadAgent')
            agent = LeadAgent()
            return agent
    
    @pytest.fixture
    def mock_tasks(self):
        """Create mock SubagentTasks."""
        return [
            SubagentTask(
                task_name=f"task-{i}",
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
                task_name=f"task-{i}",
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
            SubagentResult(task_name="task-1", summary="Summary 1", finding="Finding 1"),
            None,  # Failed task
            SubagentResult(task_name="task-3", summary="Summary 3", finding="Finding 3")
        ]
        
        with patch.object(lead_agent, 'execute_subagent_task', new=AsyncMock()) as mock_execute:
            mock_execute.side_effect = mock_results
            
            results = await lead_agent.execute_tasks_parallel(mock_tasks)
            
            assert len(results) == 2  # Only successful results
            assert results[0].task_name == "task-1"
            assert results[1].task_name == "task-3"
    
    @pytest.mark.asyncio
    async def test_execute_tasks_parallel_memory_storage(self, lead_agent, mock_tasks):
        """Test that results are stored in memory."""
        mock_results = [
            SubagentResult(task_name="task-1", summary="Summary 1", finding="Finding 1")
        ]
        
        with patch.object(lead_agent, 'execute_subagent_task', new=AsyncMock()) as mock_execute:
            mock_execute.side_effect = mock_results
            
            lead_agent.cycle_idx = 1
            _ = await lead_agent.execute_tasks_parallel(mock_tasks[:1])
            
            # Check filesystem was updated
            tree = lead_agent.fs.tree(max_depth=None)
            assert "cycle_001/" in tree
            assert "cycle_001/task-1/result.md" in tree


class TestAforwardFlow:
    @pytest.fixture
    def lead_agent(self):
        fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
        sys.modules['langfuse'] = fake_langfuse
        with patch.object(_utils, 'setup_langfuse', return_value=None):
            agent_module = importlib.import_module('agent')
            LeadAgent = getattr(agent_module, 'LeadAgent')
            return LeadAgent()

    @pytest.mark.asyncio
    async def test_aforward_two_cycles(self, lead_agent):
        """Simulate two cycles: continue then done."""
        query = "Compare Python vs JavaScript"

        # First cycle plan
        plan1 = MagicMock()
        plan1.reasoning = "Research Python"
        plan1.tasks = [
            SubagentTask(
                task_name="task-1",
                objective="Research Python async",
                tool_guidance={"web_search": "Python async"},
                tool_budget=5,
                expected_output="Python async features"
            )
        ]
        plan1.plan_filename = "compare-python-javascript"

        # First cycle result
        result1 = SubagentResult(
            task_name="task-1",
            summary="Python summary",
            finding="Python details..."
        )

        # First cycle synthesis: continue
        synthesis1 = MagicMock()
        synthesis1.is_done = False
        synthesis1.reasoning = "Need JavaScript info"
        synthesis1.synthesis = "Python research complete, need JavaScript"
        synthesis1.gap_analysis = "Missing JS"
        synthesis1.refined_query = query

        # Second cycle plan
        plan2 = MagicMock()
        plan2.reasoning = "Research JavaScript"
        plan2.tasks = [
            SubagentTask(
                task_name="task-2",
                objective="Research JavaScript features",
                tool_guidance={"web_search": "JavaScript programming"},
                tool_budget=5,
                expected_output="JavaScript features"
            )
        ]
        plan2.plan_filename = "compare-python-javascript"

        # Second cycle result
        result2 = SubagentResult(
            task_name="task-2",
            summary="JavaScript is web language",
            finding="JavaScript details..."
        )

        # Final synthesis
        synthesis2 = MagicMock()
        synthesis2.is_done = True
        synthesis2.reasoning = "Have complete comparison"
        synthesis2.synthesis = "Python vs JavaScript comparison complete"

        # Orchestrate two cycles
        with patch.object(lead_agent.planner, 'acall', new=AsyncMock(side_effect=[plan1, plan2])):
            with patch.object(lead_agent, 'execute_subagent_task', new=AsyncMock(side_effect=[result1, result2])):
                with patch.object(lead_agent.synthesizer, 'acall', new=AsyncMock(side_effect=[synthesis1, synthesis2])):
                    # First cycle
                    res1 = await lead_agent.aforward(query)
                    assert res1['is_done'] is False
                    assert lead_agent.cycle_idx == 1
                    # Second cycle
                    res2 = await lead_agent.aforward(query)
                    assert res2['is_done'] is True
                    assert lead_agent.cycle_idx == 2


class TestPlanResearch:
    """Test suite for plan_research method."""
    
    @pytest.fixture
    def lead_agent(self):
        """Create a LeadAgent instance for testing without Langfuse side effects."""
        fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
        sys.modules['langfuse'] = fake_langfuse
        with patch.object(_utils, 'setup_langfuse', return_value=None):
            agent_module = importlib.import_module('agent')
            LeadAgent = getattr(agent_module, 'LeadAgent')
            agent = LeadAgent()
            return agent
    
    @pytest.mark.asyncio
    async def test_plan_research_basic(self, lead_agent):
        """Test basic planning functionality."""
        mock_plan = MagicMock()
        mock_plan.reasoning = "Test reasoning"
        mock_plan.tasks = [
            SubagentTask(
                task_name="task-1",
                objective="Test objective",
                tool_guidance={"web_search": "test"},
                tool_budget=5,
                expected_output="output"
            )
        ]
        mock_plan.plan_filename = "test-plan"
        
        with patch.object(lead_agent.planner, 'acall', new=AsyncMock()) as mock_plan_call:
            mock_plan_call.return_value = mock_plan
            
            result = await lead_agent.plan_research("Test query")
            
            assert result == mock_plan
            mock_plan_call.assert_called_once()
            
            # Check that plan file was written
            tree = lead_agent.fs.tree(max_depth=None)
            assert "cycle_000/test-plan.md" in tree
    
    @pytest.mark.asyncio
    async def test_plan_research_writes_plan_file_again(self, lead_agent):
        """Ensure planning writes a plan file to the filesystem (no steps trace)."""
        mock_plan = MagicMock()
        mock_plan.reasoning = "Test reasoning"
        mock_plan.tasks = []
        mock_plan.plan_filename = "test-plan-2"

        with patch.object(lead_agent.planner, 'acall', new=AsyncMock(return_value=mock_plan)):
            await lead_agent.plan_research("Test query")
            tree = lead_agent.fs.tree(max_depth=None)
            assert "cycle_000/test-plan-2.md" in tree


class TestSynthesizeResults:
    """Test suite for synthesize_results method."""
    
    @pytest.fixture
    def lead_agent(self):
        """Create a LeadAgent instance for testing without Langfuse side effects."""
        fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
        sys.modules['langfuse'] = fake_langfuse
        with patch.object(_utils, 'setup_langfuse', return_value=None):
            agent_module = importlib.import_module('agent')
            LeadAgent = getattr(agent_module, 'LeadAgent')
            agent = LeadAgent()
            return agent
    
    @pytest.fixture
    def mock_results(self):
        """Create mock SubagentResults."""
        return [
            SubagentResult(
                task_name="task-1",
                summary="Summary 1",
                finding="Finding 1"
            ),
            SubagentResult(
                task_name="task-2",
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
            
            # Check that synthesis was written to filesystem
            lead_agent.cycle_idx = 1
            _ = await lead_agent.synthesize_results("Test query", mock_results)
            tree = lead_agent.fs.tree(max_depth=None)
            assert "cycle_001/synthesis.md" in tree
    
    @pytest.mark.asyncio
    async def test_synthesize_results_writes_file_again(self, lead_agent, mock_results):
        """Ensure synthesis writes a synthesis file to the filesystem (no steps trace)."""
        mock_synthesis = MagicMock()
        mock_synthesis.decision = "continue"
        mock_synthesis.reasoning = "Test reasoning"
        mock_synthesis.synthesis = "Test synthesis"

        with patch.object(lead_agent.synthesizer, 'acall', new=AsyncMock(return_value=mock_synthesis)):
            lead_agent.cycle_idx = 1
            await lead_agent.synthesize_results("Test query", mock_results)
            tree = lead_agent.fs.tree(max_depth=None)
            assert "cycle_001/synthesis.md" in tree


class TestIntegration:
    """Integration tests for end-to-end agent flow."""
    
    @pytest.fixture
    def lead_agent(self):
        """Create a LeadAgent instance for testing without Langfuse side effects."""
        fake_langfuse = types.SimpleNamespace(observe=lambda *a, **k: (lambda f: f))
        sys.modules['langfuse'] = fake_langfuse
        with patch.object(_utils, 'setup_langfuse', return_value=None):
            agent_module = importlib.import_module('agent')
            LeadAgent = getattr(agent_module, 'LeadAgent')
            return LeadAgent()
    
    @pytest.mark.asyncio
    async def test_full_agent_flow(self, lead_agent):
        """Test complete agent flow from query to synthesis."""
        query = "What is the capital of France?"
        
        # Mock plan
        mock_plan = MagicMock()
        mock_plan.reasoning = "Need to find capital of France"
        mock_plan.tasks = [
            SubagentTask(
                task_name="task-1",
                objective="Find the capital of France",
                tool_guidance={"web_search": "Search for France capital"},
                tool_budget=5,
                expected_output="The capital city name"
            )
        ]
        
        # Mock subagent result
        mock_result = SubagentResult(
            task_name="task-1",
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
                task_name="task-1",
                objective="Research Python features",
                tool_guidance={"web_search": "Python programming language"},
                tool_budget=5,
                expected_output="Python features"
            )
        ]
        
        # First cycle result
        result1 = SubagentResult(
            task_name="task-1",
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
                task_name="task-2",
                objective="Research JavaScript features",
                tool_guidance={"web_search": "JavaScript programming"},
                tool_budget=5,
                expected_output="JavaScript features"
            )
        ]
        
        # Second cycle result
        result2 = SubagentResult(
            task_name="task-2",
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
                task_name="task-1",
                objective="Task that will fail",
                tool_guidance={"web_search": "test"},
                tool_budget=5,
                expected_output="output"
            ),
            SubagentTask(
                task_name="task-2",
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
                task_name="task-2",
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
                    assert result['results'][0].task_name == "task-2"
    
    @pytest.mark.asyncio
    async def test_filesystem_persistence(self, lead_agent):
        """Test that filesystem artifacts persist across cycles."""
        query = "Test memory"
        
        # Pre-seed filesystem with a file
        lead_agent.fs.write("custom/initial.md", "Initial data")
        
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

        # Check initial data still present
        assert lead_agent.fs.exists("custom/initial.md")
        assert lead_agent.fs.read("custom/initial.md") == "Initial data"

        # Check new data was added to filesystem
        tree = lead_agent.fs.tree(max_depth=None)
        assert "cycle_001/test-plan.md" in tree
        assert "cycle_001/synthesis.md" in tree
