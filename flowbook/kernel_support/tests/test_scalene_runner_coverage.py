"""Tests for scalene_runner.py - Targeting uncovered helper methods.

Coverage gaps include:
- _read_profile (lines 173-179)
- _replace_filenames_with_cell_ids (lines 183-189)
- StopJupyterExecution class (line 15)
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock

from flowbook.kernel_support.scalene_runner import ScaleneRunner, StopJupyterExecution


class MockShell:
    """Mock IPython shell for testing."""
    execution_count = 5


class TestStopJupyterExecution:
    """Tests for StopJupyterExecution exception."""

    def test_is_exception(self):
        """StopJupyterExecution is an Exception subclass."""
        exc = StopJupyterExecution()
        assert isinstance(exc, Exception)

    def test_can_be_raised_and_caught(self):
        """StopJupyterExecution can be raised and caught."""
        with pytest.raises(StopJupyterExecution):
            raise StopJupyterExecution()


class TestReplaceFilenamesWithCellIds:
    """Tests for _replace_filenames_with_cell_ids."""

    def test_replace_known_cell(self):
        """Replaces known cell filenames with cell ID."""
        runner = ScaleneRunner(MockShell(), {1: "abcd", 2: "efgh"})
        text = "File /some/path/_ipython-input-1-profile line 5"
        result = runner._replace_filenames_with_cell_ids(text)
        assert "Cell abcd" in result
        assert "_ipython-input-1-profile" not in result

    def test_replace_unknown_cell(self):
        """Replaces unknown cell filenames with numeric fallback."""
        runner = ScaleneRunner(MockShell(), {})
        text = "File /some/path/_ipython-input-99-profile line 5"
        result = runner._replace_filenames_with_cell_ids(text)
        assert "Cell 99" in result

    def test_no_replacement_needed(self):
        """Text without profile filenames is unchanged."""
        runner = ScaleneRunner(MockShell(), {})
        text = "Normal text without any filenames"
        result = runner._replace_filenames_with_cell_ids(text)
        assert result == text

    def test_multiple_replacements(self):
        """Multiple filenames are all replaced."""
        runner = ScaleneRunner(MockShell(), {1: "aaaa", 2: "bbbb"})
        text = (
            "/path/_ipython-input-1-profile\n"
            "/path/_ipython-input-2-profile\n"
        )
        result = runner._replace_filenames_with_cell_ids(text)
        assert "Cell aaaa" in result
        assert "Cell bbbb" in result


class TestReadProfile:
    """Tests for _read_profile."""

    def test_read_existing_file(self):
        """Read profile from existing file."""
        runner = ScaleneRunner(MockShell(), {1: "cell1"})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Profile output for /path/_ipython-input-1-profile\n")
            f.write("Some data\n")
            fname = f.name
        try:
            result = runner._read_profile(fname)
            assert result is not None
            assert "Cell cell1" in result
            assert "Some data" in result
        finally:
            os.unlink(fname)

    def test_read_missing_file(self):
        """Read profile returns None for missing file."""
        runner = ScaleneRunner(MockShell(), {})
        result = runner._read_profile("/nonexistent/file.txt")
        assert result is None
