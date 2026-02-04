"""
Test module for FlowbookAgent.
"""

import json
import pytest
import tempfile
import os
import time as time_module
from unittest.mock import AsyncMock, Mock, patch
from typing import List

from flowbook.agent.agent import FlowbookAgent, FlowbookContext, FlowbookStats
from agents import Tool
from pydantic import BaseModel


class SampleOutput(BaseModel):
    """Sample output model for testing."""
    message: str
    count: int


class TestFlowbookAgent:
    """Test cases for FlowbookAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset counters before each test
        FlowbookAgent.counters = {}
        
    def test_initialization_basic(self):
        """Test basic FlowbookAgent initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = FlowbookAgent(
                key="test-agent",
                model="openai/gpt-4o-mini",
                instructions="You are a test agent.",
                log_dir=temp_dir
            )
            
            assert agent.name == "test-agent"
            assert agent.log_dir == temp_dir
            assert os.path.exists(temp_dir)
            
    def test_initialization_with_output_type(self):
        """Test FlowbookAgent initialization with output type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = FlowbookAgent(
                key="test-agent",
                model="openai/gpt-4o-mini", 
                instructions="You are a test agent.",
                output_type=SampleOutput,
                log_dir=temp_dir
            )
            
            assert agent.name == "test-agent"
            assert agent.output_type == SampleOutput
            
    def test_initialization_with_tools(self):
        """Test FlowbookAgent initialization with tools."""
        mock_tool = Mock(spec=Tool)
        mock_tool.name = "test_tool"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = FlowbookAgent(
                key="test-agent",
                model="openai/gpt-4o-mini",
                instructions="You are a test agent.",
                tools=[mock_tool],
                log_dir=temp_dir
            )
            
            assert len(agent.tools) == 1
            assert agent.tools[0].name == "test_tool"
            
    def test_make_unique(self):
        """Test the make_unique class method."""
        # First call should return "test-1"
        unique1 = FlowbookAgent.make_unique("test")
        assert unique1 == "test-1"
        
        # Second call should return "test-2"
        unique2 = FlowbookAgent.make_unique("test")
        assert unique2 == "test-2"
        
        # Different key should start from 1
        unique3 = FlowbookAgent.make_unique("other")
        assert unique3 == "other-1"
        
    def test_transform_and_dump(self):
        """Test the transform_and_dump method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = FlowbookAgent(
                key="test-agent",
                model="openai/gpt-4o-mini",
                instructions="You are a test agent.",
                log_dir=temp_dir
            )
            
            # Test with simple object
            test_obj = {"key": "value", "number": 42}
            result = agent.transform_and_dump(test_obj)
            
            # Should return JSON string
            assert isinstance(result, str)
            parsed = json.loads(result)
            assert parsed == test_obj
            
    def test_flowbook_context(self):
        """Test FlowbookContext initialization."""
        context = FlowbookContext()
        
        assert context.start is None
        assert context.time is None
        assert context.usage is None
        
    def test_flowbook_stats_initialization(self):
        """Test FlowbookStats initialization."""
        from agents import Usage
        
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        
        # Test with string model
        stats = FlowbookStats(
            model="openai/gpt-4o-mini",
            time=1.5,
            usage=usage,
            log_path="/test/path.txt"
        )
        
        assert stats.model == "openai/gpt-4o-mini"
        assert stats.time == 1.5
        assert stats.usage == usage
        assert stats.log_path == "/test/path.txt"
        assert isinstance(stats.cost, float)
        
    @pytest.mark.asyncio
    @patch('flowbook.agent.agent.Runner')
    async def test_run_method_mock(self, mock_runner):
        """Test the run method with mocked dependencies."""
        from agents import Usage

        # Mock the Runner.run method to also set context fields
        mock_result = Mock()
        mock_result.final_output = "test output"

        async def fake_run(agent, input, context, hooks, session, max_turns):
            # context here is the raw FlowbookContext (Runner wraps it internally)
            context.start = time_module.time()
            context.time = 0.1
            context.usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
            return mock_result

        mock_runner.run = AsyncMock(side_effect=fake_run)

        with tempfile.TemporaryDirectory() as temp_dir:
            agent = FlowbookAgent(
                key="test-agent",
                model="openai/gpt-4o-mini",
                instructions="You are a test agent.",
                log_dir=temp_dir
            )

            # Mock the session and cost function
            with patch('flowbook.agent.agent.SQLiteSession') as mock_session_class, \
                 patch('flowbook.agent.agent.cost', return_value=0.001):
                mock_session = Mock()
                mock_session.get_items = AsyncMock(return_value=[])
                mock_session_class.return_value = mock_session

                result, stats = await agent.run("test input")

                assert result == "test output"
                assert isinstance(stats, FlowbookStats)
                assert mock_runner.run.called


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestFlowbookAgent()
    test_instance.setup_method()
    
    print("Running basic tests...")
    
    # Test initialization
    test_instance.test_initialization_basic()
    print("✓ Basic initialization test passed")
    
    test_instance.test_initialization_with_output_type()
    print("✓ Initialization with output type test passed")
    
    test_instance.test_initialization_with_tools()
    print("✓ Initialization with tools test passed")
    
    # Test make_unique
    test_instance.test_make_unique()
    print("✓ make_unique test passed")
    
    # Test transform_and_dump
    test_instance.test_transform_and_dump()
    print("✓ transform_and_dump test passed")
    
    # Test context and stats
    test_instance.test_flowbook_context()
    print("✓ FlowbookContext test passed")
    
    test_instance.test_flowbook_stats_initialization()
    print("✓ FlowbookStats test passed")
    
    print("\nAll basic tests passed! 🎉")
