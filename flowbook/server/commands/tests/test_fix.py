"""
Unit tests for the FixCommand.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

import nbformat

from flowbook.server.commands.fix import (
    FixCommand,
    FixProposal,
)
from flowbook.util.flowbook_metadata import (
    ProposedFix,
    ProposedFixEntry,
    FlowbookMetadata,
    set_proposed_fix_flowbook_metadata,
)
from flowbook.util.notebook_to_python import notebook_to_python, python_to_notebook_cells
from flowbook.agent.agent import FlowbookStats
from agents import Usage


class TestFixCommand:
    """Test cases for FixCommand."""

    def setup_method(self):
        """Set up test fixtures."""
        self.command = FixCommand()

    def test_command_properties(self):
        """Test command basic properties."""
        assert self.command.command_name == "fix"
        assert self.command.display_name == "Fix Violation"
        assert self.command.icon_name == "ui-components:refresh"
        assert self.command.requires_kernel is False
        assert "fix" in self.command.tooltip.lower()

    def create_notebook_with_violation(self):
        """Create a notebook with a backward violation in cell metadata."""
        nb = nbformat.v4.new_notebook()

        # Cell 1: Reads df
        cell1 = nbformat.v4.new_code_cell(
            source="result = df['price'].sum()"
        )
        cell1.id = "abcd"

        # Cell 2: Modifies df (causes violation)
        cell2 = nbformat.v4.new_code_cell(
            source="df['new_col'] = 1"
        )
        cell2.id = "efgh"
        cell2.metadata["flowbook"] = {
            "errors": [{
                "error_type": "no_write_after_read",
                "cell_id": "efgh",
                "locations": ["df"],
                "message": "Cell @B modified `df` read by earlier cell @A",
                "accepted": False,
                "causer_cell": "abcd",
            }]
        }

        nb.cells = [cell1, cell2]
        return nb

    def create_notebook_with_proposed_fix(self):
        """Create a notebook with a proposed fix in cell metadata."""
        nb = self.create_notebook_with_violation()

        # Add proposed fix to cell 2
        proposed_fix = ProposedFix(
            violation_type="backward_mutation",
            mutating_cell="efgh",
            affected_cell="abcd",
            strategy="alpha_rename",
            fix_entries=[
                ProposedFixEntry(
                    cell_ids=["efgh"],
                    modified_source="df_processed = df.copy()\ndf_processed['new_col'] = 1",
                    explanation="Renamed df to df_processed to avoid modifying the original"
                )
            ],
            explanation="Alpha-rename the variable to avoid modifying the original DataFrame"
        )
        set_proposed_fix_flowbook_metadata(nb.cells[1], proposed_fix)

        return nb

    def test_get_violation_from_cell(self):
        """Test extracting violation info from cell metadata."""
        nb = self.create_notebook_with_violation()
        cell = nb.cells[1]

        violation = self.command._get_violation_from_cell(dict(cell))
        assert violation is not None
        assert violation["error_type"] == "no_write_after_read"
        assert violation["cell_id"] == "efgh"
        assert violation["causer_cell"] == "abcd"
        assert "df" in violation["locations"]

    def test_get_proposed_fix_from_cell(self):
        """Test extracting proposed fix from cell metadata."""
        nb = self.create_notebook_with_proposed_fix()
        cell = nb.cells[1]

        proposed_fix = self.command._get_proposed_fix_from_cell(dict(cell))
        assert proposed_fix is not None
        assert proposed_fix.strategy == "alpha_rename"
        assert len(proposed_fix.fix_entries) == 1
        assert "df_processed" in proposed_fix.fix_entries[0].modified_source

    def test_no_violation_returns_none(self):
        """Test that cells without violations return None."""
        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_code_cell(source="x = 1")
        cell.id = "test"
        nb.cells = [cell]

        violation = self.command._get_violation_from_cell(dict(cell))
        assert violation is None

    def test_no_proposed_fix_returns_none(self):
        """Test that cells without proposed fix return None."""
        nb = self.create_notebook_with_violation()
        cell = nb.cells[1]

        proposed_fix = self.command._get_proposed_fix_from_cell(dict(cell))
        assert proposed_fix is None

    @pytest.mark.asyncio
    async def test_apply_existing_fix(self):
        """Test applying an existing proposed fix."""
        nb = self.create_notebook_with_proposed_fix()

        mock_config = Mock()
        mock_config.model = "gpt-4o"

        result = await self.command.process(
            notebook_content=dict(nb),
            cell_id="efgh",
            config=mock_config
        )

        # Check that the fix was applied
        assert result.metadata["status"] == "success"
        assert result.metadata["fixes_applied"] == 1

        # Check that cell source was updated
        processed_nb = nbformat.from_dict(result.notebook)
        cell2 = processed_nb.cells[1]
        assert "df_processed" in cell2.source
        assert "df_processed['new_col'] = 1" in cell2.source

    @pytest.mark.asyncio
    async def test_generate_fix_proposal(self):
        """Test generating a fix proposal via LLM."""
        nb = self.create_notebook_with_violation()

        # Mock the LLM agent
        mock_proposal = FixProposal(
            strategy="alpha_rename",
            fix_entries=[
                ProposedFixEntry(
                    cell_ids=["efgh"],
                    modified_source="df_modified = df.copy()\ndf_modified['new_col'] = 1",
                    explanation="Create a copy before modification"
                )
            ],
            explanation="Use a copy to avoid modifying the original"
        )

        mock_stats = FlowbookStats(
            model="gpt-4o-mini",
            log_path="",
            time=1.0,
            usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        )

        with patch.object(self.command, '_generate_fix_proposal', new_callable=AsyncMock) as mock_gen:
            from flowbook.server.commands.fix import FixProposalAndStats
            mock_gen.return_value = FixProposalAndStats(proposal=mock_proposal, stats=mock_stats)

            mock_config = Mock()
            mock_config.model = "gpt-4o"

            result = await self.command.process(
                notebook_content=dict(nb),
                cell_id="efgh",
                config=mock_config
            )

            # Check that proposal was generated and stored
            assert result.metadata["status"] == "success"
            assert result.metadata["proposals_generated"] == 1

            # Check that proposed_fix was added to metadata
            processed_nb = nbformat.from_dict(result.notebook)
            cell2 = processed_nb.cells[1]
            flowbook_meta = FlowbookMetadata.from_cell(cell2)
            assert flowbook_meta.proposed_fix is not None
            assert flowbook_meta.proposed_fix.strategy == "alpha_rename"

    @pytest.mark.asyncio
    async def test_silent_mode_generates_and_applies(self):
        """Test that silent mode generates and applies in one shot."""
        nb = self.create_notebook_with_violation()

        mock_proposal = FixProposal(
            strategy="copy_value",
            fix_entries=[
                ProposedFixEntry(
                    cell_ids=["efgh"],
                    modified_source="df_copy = df.copy()\ndf_copy['new_col'] = 1",
                    explanation="Insert copy before mutation"
                )
            ],
            explanation="Copy before modify"
        )

        mock_stats = FlowbookStats(
            model="gpt-4o-mini",
            log_path="",
            time=1.0,
            usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        )

        with patch.object(self.command, '_generate_fix_proposal', new_callable=AsyncMock) as mock_gen:
            from flowbook.server.commands.fix import FixProposalAndStats
            mock_gen.return_value = FixProposalAndStats(proposal=mock_proposal, stats=mock_stats)

            mock_config = Mock()
            mock_config.model = "gpt-4o"

            result = await self.command.process(
                notebook_content=dict(nb),
                cell_id="efgh",
                config=mock_config,
                silent=True
            )

            # Check that fix was applied immediately
            assert result.metadata["status"] == "success"
            assert result.metadata["proposals_generated"] == 1
            assert result.metadata["fixes_applied"] == 1

            # Check that cell source was updated
            processed_nb = nbformat.from_dict(result.notebook)
            cell2 = processed_nb.cells[1]
            assert "df_copy" in cell2.source

    @pytest.mark.asyncio
    async def test_no_violation_no_changes(self):
        """Test that cells without violations are unchanged."""
        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_code_cell(source="x = 1")
        cell.id = "test"
        nb.cells = [cell]

        mock_config = Mock()
        mock_config.model = "gpt-4o"

        result = await self.command.process(
            notebook_content=dict(nb),
            cell_id="test",
            config=mock_config
        )

        # No proposals or fixes since no violation
        assert result.metadata["status"] == "success"
        assert result.metadata["proposals_generated"] == 0
        assert result.metadata["fixes_applied"] == 0

    @pytest.mark.asyncio
    async def test_merge_cells_fix(self):
        """Test applying a merge cells fix."""
        nb = self.create_notebook_with_violation()

        # Create a fix that merges cells
        proposed_fix = ProposedFix(
            violation_type="backward_mutation",
            mutating_cell="efgh",
            affected_cell="abcd",
            strategy="merge_cells",
            fix_entries=[
                ProposedFixEntry(
                    cell_ids=["abcd", "efgh"],
                    modified_source="df['new_col'] = 1\nresult = df['price'].sum()",
                    explanation="Merged cells to fix execution order"
                )
            ],
            explanation="Merge cells so modification happens before read"
        )
        set_proposed_fix_flowbook_metadata(nb.cells[1], proposed_fix)

        mock_config = Mock()
        mock_config.model = "gpt-4o"

        result = await self.command.process(
            notebook_content=dict(nb),
            cell_id="efgh",
            config=mock_config
        )

        # Check that merge was applied
        assert result.metadata["fixes_applied"] == 1

        # Check that only one cell remains
        processed_nb = nbformat.from_dict(result.notebook)
        assert len(processed_nb.cells) == 1
        assert "df['new_col'] = 1" in processed_nb.cells[0].source
        assert "result = df['price'].sum()" in processed_nb.cells[0].source

    @pytest.mark.asyncio
    async def test_delete_cell_fix(self):
        """Test applying a fix that deletes a cell."""
        nb = self.create_notebook_with_violation()

        # Create a fix that deletes the mutating cell
        proposed_fix = ProposedFix(
            violation_type="backward_mutation",
            mutating_cell="efgh",
            affected_cell="abcd",
            strategy="reorder",
            fix_entries=[
                ProposedFixEntry(
                    cell_ids=["efgh"],
                    modified_source="",  # Empty = delete
                    explanation="Remove the problematic cell"
                )
            ],
            explanation="Delete the cell causing the violation"
        )
        set_proposed_fix_flowbook_metadata(nb.cells[1], proposed_fix)

        mock_config = Mock()
        mock_config.model = "gpt-4o"

        result = await self.command.process(
            notebook_content=dict(nb),
            cell_id="efgh",
            config=mock_config
        )

        # Check that cell was deleted
        assert result.metadata["fixes_applied"] == 1
        processed_nb = nbformat.from_dict(result.notebook)
        assert len(processed_nb.cells) == 1
        assert processed_nb.cells[0].id == "abcd"


class TestNotebookToPython:
    """Test cases for notebook_to_python conversion utilities."""

    def test_simple_conversion(self):
        """Test basic notebook to Python conversion."""
        nb = nbformat.v4.new_notebook()
        cell1 = nbformat.v4.new_code_cell(source="x = 1")
        cell1.id = "abcd"
        cell2 = nbformat.v4.new_code_cell(source="y = x + 1")
        cell2.id = "efgh"
        nb.cells = [cell1, cell2]

        python_source = notebook_to_python(dict(nb))

        assert "# ====== CELL [abcd] (code) ======" in python_source
        assert "x = 1" in python_source
        assert "# ====== CELL [efgh] (code) ======" in python_source
        assert "y = x + 1" in python_source

    def test_roundtrip(self):
        """Test that conversion is roundtrip-safe."""
        nb = nbformat.v4.new_notebook()
        cell1 = nbformat.v4.new_code_cell(source="x = 1\ny = 2")
        cell1.id = "abcd"
        cell2 = nbformat.v4.new_code_cell(source="z = x + y")
        cell2.id = "efgh"
        nb.cells = [cell1, cell2]

        python_source = notebook_to_python(dict(nb))
        parsed_cells = python_to_notebook_cells(python_source)

        assert len(parsed_cells) == 2
        assert parsed_cells[0]["id"] == "abcd"
        assert parsed_cells[0]["source"].strip() == "x = 1\ny = 2"
        assert parsed_cells[1]["id"] == "efgh"
        assert parsed_cells[1]["source"].strip() == "z = x + y"

    def test_markdown_cell_conversion(self):
        """Test markdown cell conversion with # prefix."""
        nb = nbformat.v4.new_notebook()
        md_cell = nbformat.v4.new_markdown_cell(source="# Title\n\nSome text")
        md_cell.id = "mdcl"
        nb.cells = [md_cell]

        python_source = notebook_to_python(dict(nb))

        assert "# ====== CELL [mdcl] (markdown) ======" in python_source
        # Each line should be prefixed with #
        assert "# # Title" in python_source
        assert "# Some text" in python_source

    def test_empty_cell(self):
        """Test handling of empty cells."""
        nb = nbformat.v4.new_notebook()
        empty_cell = nbformat.v4.new_code_cell(source="")
        empty_cell.id = "empt"
        nb.cells = [empty_cell]

        python_source = notebook_to_python(dict(nb))

        assert "# ====== CELL [empt] (code) ======" in python_source


def test_command_instantiation():
    """Test that the command can be instantiated."""
    command = FixCommand()
    assert command is not None
    assert command.command_name == "fix"


if __name__ == "__main__":
    print("Running basic tests...")

    test_command_instantiation()
    print("✓ Command instantiation test passed")

    test_instance = TestFixCommand()
    test_instance.setup_method()

    test_instance.test_command_properties()
    print("✓ Command properties test passed")

    test_instance.test_get_violation_from_cell()
    print("✓ Get violation from cell test passed")

    test_instance.test_get_proposed_fix_from_cell()
    print("✓ Get proposed fix from cell test passed")

    test_instance.test_no_violation_returns_none()
    print("✓ No violation returns None test passed")

    test_instance.test_no_proposed_fix_returns_none()
    print("✓ No proposed fix returns None test passed")

    # Test notebook_to_python
    notebook_tests = TestNotebookToPython()
    notebook_tests.test_simple_conversion()
    print("✓ Simple conversion test passed")

    notebook_tests.test_roundtrip()
    print("✓ Roundtrip test passed")

    notebook_tests.test_markdown_cell_conversion()
    print("✓ Markdown cell conversion test passed")

    notebook_tests.test_empty_cell()
    print("✓ Empty cell test passed")

    print("\n✓ All basic tests passed!")
    print("Run 'pytest flowbook/server/commands/tests/test_fix.py' for async tests")
