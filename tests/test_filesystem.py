"""Comprehensive tests for FileSystem and FileSystemTool integration."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from models import FileSystem
from tools import FileSystemTool
from agent import LeadAgent


class TestFileSystem:
    """Test FileSystem class functionality."""
    
    @pytest.fixture
    def temp_fs(self):
        """Create a temporary filesystem for testing."""
        temp_dir = tempfile.mkdtemp()
        fs = FileSystem(root=temp_dir)
        yield fs
        shutil.rmtree(temp_dir)
    
    def test_write_and_read(self, temp_fs):
        """Test basic write and read operations."""
        path = "test/file.md"
        content = "Test content"
        
        # Write content
        file_path = temp_fs.write(path, content)
        assert file_path.exists()
        
        # Read content
        result = temp_fs.read(path)
        assert result == content
    
    def test_read_nonexistent_file(self, temp_fs):
        """Test reading non-existent file returns error."""
        result = temp_fs.read("nonexistent/file.md")
        assert "[ERROR] File not found" in result
    
    def test_exists_check(self, temp_fs):
        """Test file existence checking."""
        path = "test/exists.md"
        
        # Initially doesn't exist
        assert not temp_fs.exists(path)
        
        # Write file
        temp_fs.write(path, "content")
        assert temp_fs.exists(path)
    
    def test_tree_structure(self, temp_fs):
        """Test tree structure display."""
        # Write test files
        temp_fs.write("cycle_001/plan.md", "Plan content")
        temp_fs.write("cycle_001/task1/result.md", "Result content")
        temp_fs.write("cycle_002/synthesis.md", "Synthesis content")
        
        tree = temp_fs.tree()
        assert "cycle_001/" in tree
        assert "cycle_002/" in tree
        assert "plan.md" in tree
        assert "task1/" in tree
        assert "synthesis.md" in tree
    
    def test_tree_empty_filesystem(self, temp_fs):
        """Test tree display for empty filesystem."""
        tree = temp_fs.tree()
        assert tree == "memory/ (empty)"
    
    def test_tree_max_depth(self, temp_fs):
        """Test tree structure with depth limit."""
        # Create nested structure
        temp_fs.write("level1/level2/level3/deep.md", "Deep content")
        
        # Test with depth limit
        tree_shallow = temp_fs.tree(max_depth=2)
        tree_deep = temp_fs.tree(max_depth=None)
        
        assert "level1/" in tree_shallow
        assert "level2/" in tree_shallow
        assert "deep.md" not in tree_shallow  # Too deep
        
        assert "deep.md" in tree_deep  # No depth limit


class TestFileSystemTool:
    """Test FileSystemTool wrapper functionality."""
    
    @pytest.fixture
    def temp_fs_tool(self):
        """Create a temporary filesystem tool for testing."""
        temp_dir = tempfile.mkdtemp()
        fs = FileSystem(root=temp_dir)
        tool = FileSystemTool(fs)
        yield tool, fs
        shutil.rmtree(temp_dir)
    
    def test_tool_tree(self, temp_fs_tool):
        """Test FileSystemTool tree method."""
        tool, fs = temp_fs_tool
        
        # Write test files
        fs.write("test1/file1.md", "Content 1")
        fs.write("test2/file2.md", "Content 2")
        
        tree = tool.tree()
        assert "test1/" in tree
        assert "test2/" in tree
        assert "file1.md" in tree
        assert "file2.md" in tree
    
    def test_tool_tree_with_depth(self, temp_fs_tool):
        """Test FileSystemTool tree method with custom depth."""
        tool, fs = temp_fs_tool
        
        # Create nested structure
        fs.write("a/b/c/d/deep.md", "Deep content")
        
        tree_shallow = tool.tree(max_depth=2)
        tree_deep = tool.tree(max_depth=5)
        
        assert "a/" in tree_shallow
        assert "b/" in tree_shallow
        assert "deep.md" not in tree_shallow
        
        assert "deep.md" in tree_deep
    
    def test_tool_read(self, temp_fs_tool):
        """Test FileSystemTool read method."""
        tool, fs = temp_fs_tool
        
        # Write test file
        path = "test/read.md"
        content = "Test read content"
        fs.write(path, content)
        
        # Read via tool
        result = tool.read(path)
        assert result == content
    
    def test_tool_read_nonexistent(self, temp_fs_tool):
        """Test FileSystemTool read method with non-existent file."""
        tool, fs = temp_fs_tool
        
        result = tool.read("nonexistent.md")
        assert "[ERROR] File not found" in result


class TestAgentFileSystemIntegration:
    """Test LeadAgent integration with FileSystem."""
    
    @pytest.fixture
    def lead_agent(self):
        """Create a LeadAgent instance for testing."""
        with patch('agent.setup_langfuse'):
            agent = LeadAgent()
            return agent
    
    def test_agent_has_filesystem(self, lead_agent):
        """Test that LeadAgent has filesystem instance."""
        assert hasattr(lead_agent, 'fs')
        assert isinstance(lead_agent.fs, FileSystem)
    
    def test_agent_has_filesystem_tool(self, lead_agent):
        """Test that LeadAgent has filesystem tool."""
        assert hasattr(lead_agent, 'fs_tool')
        assert isinstance(lead_agent.fs_tool, FileSystemTool)
    
    def test_agent_filesystem_tools_available(self, lead_agent):
        """Test that filesystem tools are available in agent tools."""
        assert 'filesystem_read' in lead_agent.tools
        assert 'filesystem_tree' in lead_agent.tools
    
    def test_agent_cycle_counter_starts_zero(self, lead_agent):
        """Test that cycle counter starts at zero."""
        assert lead_agent.cycle_idx == 0
    
    def test_agent_filesystem_operations(self, lead_agent):
        """Test basic filesystem operations through agent."""
        # Write content
        lead_agent.fs.write("test_cycle/plan.md", "Test plan content")
        
        # Verify it exists
        assert lead_agent.fs.exists("test_cycle/plan.md")
        
        # Read it back
        content = lead_agent.fs.read("test_cycle/plan.md")
        assert content == "Test plan content"
        
        # Check tree structure
        tree = lead_agent.fs.tree()
        assert "test_cycle/" in tree
        assert "plan.md" in tree


class TestFileSystemCycleOperations:
    """Test filesystem operations during agent cycles."""
    
    @pytest.fixture
    def lead_agent(self):
        """Create a LeadAgent instance for testing."""
        with patch('agent.setup_langfuse'):
            agent = LeadAgent()
            return agent
    
    def test_plan_storage_structure(self, lead_agent):
        """Test that plans are stored with correct cycle structure."""
        lead_agent.cycle_idx = 1
        
        plan_content = "# Research Plan\nObjective: Test research"
        plan_path = f"cycle_{lead_agent.cycle_idx:03d}/test_plan.md"
        
        lead_agent.fs.write(plan_path, plan_content)
        
        assert lead_agent.fs.exists(plan_path)
        content = lead_agent.fs.read(plan_path)
        assert "Research Plan" in content
    
    def test_result_storage_structure(self, lead_agent):
        """Test that results are stored with correct task structure."""
        lead_agent.cycle_idx = 2
        
        result_content = "# Task Result\nSummary: Completed successfully"
        result_path = f"cycle_{lead_agent.cycle_idx:03d}/web_search_task/result.md"
        
        lead_agent.fs.write(result_path, result_content)
        
        assert lead_agent.fs.exists(result_path)
        content = lead_agent.fs.read(result_path)
        assert "Task Result" in content
    
    def test_synthesis_storage_structure(self, lead_agent):
        """Test that synthesis is stored with correct cycle structure."""
        lead_agent.cycle_idx = 3
        
        synthesis_content = "# Synthesis\nDecision: Continue research"
        synthesis_path = f"cycle_{lead_agent.cycle_idx:03d}/synthesis.md"
        
        lead_agent.fs.write(synthesis_path, synthesis_content)
        
        assert lead_agent.fs.exists(synthesis_path)
        content = lead_agent.fs.read(synthesis_path)
        assert "Synthesis" in content
    
    def test_final_report_storage(self, lead_agent):
        """Test that final report is stored at root level."""
        report_content = "# Final Research Report\nConclusion: Research complete"
        
        lead_agent.fs.write("final_report.md", report_content)
        
        assert lead_agent.fs.exists("final_report.md")
        content = lead_agent.fs.read("final_report.md")
        assert "Final Research Report" in content
    
    def test_multi_cycle_structure(self, lead_agent):
        """Test filesystem structure across multiple cycles."""
        # Simulate multiple cycles
        for cycle in range(1, 4):
            lead_agent.cycle_idx = cycle
            
            # Write plan
            plan_path = f"cycle_{cycle:03d}/plan.md"
            lead_agent.fs.write(plan_path, f"Plan for cycle {cycle}")
            
            # Write synthesis
            synthesis_path = f"cycle_{cycle:03d}/synthesis.md"
            lead_agent.fs.write(synthesis_path, f"Synthesis for cycle {cycle}")
            
            # Write task result
            task_path = f"cycle_{cycle:03d}/task_{cycle}/result.md"
            lead_agent.fs.write(task_path, f"Result for cycle {cycle}")
        
        # Check tree structure
        tree = lead_agent.fs.tree(max_depth=None)
        
        for cycle in range(1, 4):
            assert f"cycle_{cycle:03d}/" in tree
            assert "plan.md" in tree
            assert "synthesis.md" in tree
            assert f"task_{cycle}/" in tree


if __name__ == "__main__":
    pytest.main([__file__, "-v"])