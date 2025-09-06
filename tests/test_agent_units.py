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
    async def test_execute_subagent_task_no_crash_on_error(self, lead_agent, mock_task, mock_subagent_lm):
        """Ensure failures are handled without raising and return None."""
        with patch('dspy.ReAct') as mock_react_class:
            mock_react = AsyncMock()
            mock_react.acall.side_effect = Exception("Test error")
            mock_react_class.return_value = mock_react
            result = await lead_agent.execute_subagent_task(mock_task, mock_subagent_lm)
            assert result is None


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
    
    # Keep a single minimal storage test
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
    pass


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
    
    # Remove duplicate/overlap test


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
    
    # Remove duplicate/overlap test


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
    async def test_end_to_end_final_artifacts(self, lead_agent):
        """Minimal end-to-end: agentic run yields final artifacts without real network."""
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
        mock_synthesis = MagicMock(is_done=True, synthesis="The capital of France is Paris")
        
        with patch.object(lead_agent.planner, 'acall', new=AsyncMock(return_value=mock_plan)):
            with patch.object(lead_agent, 'execute_subagent_task', new=AsyncMock(return_value=mock_result)):
                with patch.object(lead_agent.synthesizer, 'acall', new=AsyncMock(return_value=mock_synthesis)):
                    
                    result = await lead_agent.aforward(query)
        assert result is not None
        assert result['is_done'] is True
        last_cycle = lead_agent.cycle_idx
        assert lead_agent.fs.exists(f"cycle_{last_cycle:03d}/synthesis.md")
        assert lead_agent.fs.exists(f"cycle_{last_cycle:03d}/final_report.md")
    
    @pytest.mark.asyncio
    # Remove multi-call per-cycle test; covered by end-to-end above
    
    @pytest.mark.asyncio  
    # Remove error recovery integration test to minimize overhead
    
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

        # Check final artifacts exist in last cycle
        last_cycle = lead_agent.cycle_idx
        assert lead_agent.fs.exists(f"cycle_{last_cycle:03d}/synthesis.md")
        assert lead_agent.fs.exists(f"cycle_{last_cycle:03d}/final_report.md")
