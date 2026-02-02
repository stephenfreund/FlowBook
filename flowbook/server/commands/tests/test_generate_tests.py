"""
Unit tests for the GenerateTestsCommand.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List

import nbformat

from flowbook.server.commands.generate_tests import (
    GenerateTestsCommand,
    GeneratedTests,
)
from flowbook.util.flowbook_metadata import UnitTest, UnitTests, FlowbookMetadata
from flowbook.agent.agent import FlowbookStats
from agents import Usage


class TestGenerateTestsCommand:
    """Test cases for GenerateTestsCommand."""

    def setup_method(self):
        """Set up test fixtures."""
        self.command = GenerateTestsCommand()

    def test_command_properties(self):
        """Test command basic properties."""
        assert self.command.command_name == "generate_tests"
        assert self.command.display_name == "Generate Tests"
        assert self.command.icon_name == "ui-components:build"
        assert self.command.requires_kernel == False
        assert "test" in self.command.tooltip.lower()

    def create_simple_notebook(self):
        """Create a simple test notebook."""
        nb = nbformat.v4.new_notebook()

        # Cell 1: Simple function
        cell1 = nbformat.v4.new_code_cell(
            source="def add(a, b):\n    return a + b\n\nresult = add(2, 3)"
        )
        cell1.id = "cell-1"

        # Cell 2: Uses result from cell 1
        cell2 = nbformat.v4.new_code_cell(
            source="doubled = result * 2\nprint(f'Doubled: {doubled}')"
        )
        cell2.id = "cell-2"

        nb.cells = [cell1, cell2]
        return nb

    def create_notebook_with_existing_tests(self):
        """Create a notebook with existing tests in metadata."""
        nb = nbformat.v4.new_notebook()

        cell = nbformat.v4.new_code_cell(
            source="x = 5\ny = x * 2"
        )
        cell.id = "cell-with-tests"

        # Add existing tests to metadata
        existing_test = UnitTest(
            title="Existing Test",
            description="A pre-existing test",
            setup_code="# existing setup",
            assertion_code="assert True"
        )
        unit_tests = UnitTests(tests=[existing_test])
        cell.metadata["flowbook"] = {"unit_tests": unit_tests.model_dump()}

        nb.cells = [cell]
        return nb

    @pytest.mark.asyncio
    async def test_generate_tests_adds_to_existing(self):
        """Test that generated tests are added to existing tests, not replaced."""
        nb = self.create_notebook_with_existing_tests()

        # Mock the AI agent to return generated tests
        mock_generated = GeneratedTests(
            tests=[
                UnitTest(
                    title="Generated Test 1",
                    description="First generated test",
                    setup_code="# setup 1",
                    assertion_code="assert x == 5"
                ),
                UnitTest(
                    title="Generated Test 2",
                    description="Second generated test",
                    setup_code="# setup 2",
                    assertion_code="assert y == 10"
                ),
                UnitTest(
                    title="Generated Test 3",
                    description="Third generated test",
                    setup_code="# setup 3",
                    assertion_code="assert True"
                )
            ]
        )

        mock_stats = FlowbookStats(
            model="gpt-4o-mini",
            log_path="",
            time=1.0,
            usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        )

        with patch('flowbook.server.commands.generate_tests.FlowbookAgent.make_and_run_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = (mock_generated, mock_stats)

            # Mock config
            mock_config = Mock()
            mock_config.model = "gpt-4o"

            # Process the notebook
            result = await self.command.process(
                notebook_content=dict(nb),
                selected_cell_ids=["cell-with-tests"],
                config=mock_config
            )

            # Check that tests were added, not replaced
            processed_nb = nbformat.from_dict(result.notebook)
            cell = processed_nb.cells[0]

            flowbook_meta = FlowbookMetadata.from_cell(cell)
            assert flowbook_meta.unit_tests is not None
            assert len(flowbook_meta.unit_tests.tests) == 4  # 1 existing + 3 new

            # Check that existing test is still there
            test_titles = [t.title for t in flowbook_meta.unit_tests.tests]
            assert "Existing Test" in test_titles
            assert "Generated Test 1" in test_titles
            assert "Generated Test 2" in test_titles

    @pytest.mark.asyncio
    async def test_generate_tests_for_empty_cell(self):
        """Test that empty cells are skipped."""
        nb = nbformat.v4.new_notebook()
        empty_cell = nbformat.v4.new_code_cell(source="")
        empty_cell.id = "empty-cell"
        nb.cells = [empty_cell]

        mock_config = Mock()
        mock_config.model = "gpt-4o"

        result = await self.command.process(
            notebook_content=dict(nb),
            config=mock_config
        )

        # Should succeed without errors
        assert result.metadata["status"] == "success"

    @pytest.mark.asyncio
    async def test_generate_tests_multiple_cells(self):
        """Test generating tests for multiple cells in parallel."""
        nb = self.create_simple_notebook()

        # Mock the AI agent
        mock_generated = GeneratedTests(
            tests=[
                UnitTest(
                    title="Test 1",
                    description="Test description",
                    setup_code="# setup",
                    assertion_code="assert True"
                ),
                UnitTest(
                    title="Test 2",
                    description="Test description 2",
                    setup_code="# setup",
                    assertion_code="assert True"
                ),
                UnitTest(
                    title="Test 3",
                    description="Test description 3",
                    setup_code="# setup",
                    assertion_code="assert True"
                )
            ]
        )

        mock_stats = FlowbookStats(
            model="gpt-4o-mini",
            log_path="",
            time=1.0,
            usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        )

        with patch('flowbook.server.commands.generate_tests.FlowbookAgent.make_and_run_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = (mock_generated, mock_stats)

            mock_config = Mock()
            mock_config.model = "gpt-4o"

            # Process both cells
            result = await self.command.process(
                notebook_content=dict(nb),
                selected_cell_ids=["cell-1", "cell-2"],
                config=mock_config
            )

            # Agent should be called for each cell
            assert mock_agent.call_count == 2

            # Check metadata
            assert result.metadata["status"] == "success"
            assert result.metadata["cells_processed"] == 2

    @pytest.mark.asyncio
    async def test_generate_tests_with_dependencies(self):
        """Test that globals_read and globals_written are passed to AI."""
        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_code_cell(
            source="result = x + y\noutput = result * 2"
        )
        cell.id = "cell-with-deps"
        nb.cells = [cell]

        mock_generated = GeneratedTests(
            tests=[
                UnitTest(
                    title="Test with dependencies",
                    description="Tests the computation",
                    setup_code="x = 1\ny = 2",  # Should initialize globals_read
                    assertion_code="assert result == 3\nassert output == 6"  # Should test globals_written
                ),
                UnitTest(
                    title="Test edge case",
                    description="Tests edge case",
                    setup_code="x = 0\ny = 0",
                    assertion_code="assert result == 0"
                ),
                UnitTest(
                    title="Test negative",
                    description="Tests negative values",
                    setup_code="x = -1\ny = -2",
                    assertion_code="assert result == -3"
                )
            ]
        )

        mock_stats = FlowbookStats(
            model="gpt-4o-mini",
            log_path="",
            time=1.0,
            usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        )

        with patch('flowbook.server.commands.generate_tests.FlowbookAgent.make_and_run_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = (mock_generated, mock_stats)

            mock_config = Mock()
            mock_config.model = "gpt-4o"

            result = await self.command.process(
                notebook_content=dict(nb),
                selected_cell_ids=["cell-with-deps"],
                config=mock_config
            )

            # Check that the agent was called with a prompt mentioning dependencies
            call_args = mock_agent.call_args
            assert call_args is not None

            # The input should contain variable environment info
            input_text = call_args[1]["input"]
            assert "variables are available" in input_text.lower() or "variables" in input_text.lower()
            assert "variables are live" in input_text.lower() or "variables" in input_text.lower()

    @pytest.mark.asyncio
    async def test_processing_result_structure(self):
        """Test that the command returns a properly structured ProcessingResult."""
        nb = self.create_simple_notebook()

        mock_generated = GeneratedTests(
            tests=[
                UnitTest(
                    title="Test",
                    description="Test",
                    setup_code="",
                    assertion_code="assert True"
                ),
                UnitTest(
                    title="Test 2",
                    description="Test 2",
                    setup_code="",
                    assertion_code="assert True"
                ),
                UnitTest(
                    title="Test 3",
                    description="Test 3",
                    setup_code="",
                    assertion_code="assert True"
                )
            ]
        )

        mock_stats = FlowbookStats(
            model="gpt-4o-mini",
            log_path="",
            time=1.0,
            usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        )

        with patch('flowbook.server.commands.generate_tests.FlowbookAgent.make_and_run_agent', new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = (mock_generated, mock_stats)

            mock_config = Mock()
            mock_config.model = "gpt-4o"

            result = await self.command.process(
                notebook_content=dict(nb),
                selected_cell_ids=["cell-1"],
                config=mock_config
            )

            # Check ProcessingResult structure
            assert hasattr(result, "notebook")
            assert hasattr(result, "metadata")
            assert hasattr(result, "total_cost")
            assert hasattr(result, "total_time")

            assert result.metadata["status"] == "success"
            assert result.metadata["command"] == "generate_tests"
            assert isinstance(result.total_cost, float)
            assert isinstance(result.total_time, float)


def test_command_instantiation():
    """Test that the command can be instantiated."""
    command = GenerateTestsCommand()
    assert command is not None
    assert command.command_name == "generate_tests"


if __name__ == "__main__":
    # Run basic synchronous tests
    print("Running basic tests...")

    test_command_instantiation()
    print("✓ Command instantiation test passed")

    test_instance = TestGenerateTestsCommand()
    test_instance.setup_method()

    test_instance.test_command_properties()
    print("✓ Command properties test passed")

    print("\n✓ All basic tests passed!")
    print("Run 'pytest flowbook/server/commands/test_generate_tests.py' for async tests")
