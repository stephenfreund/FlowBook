"""
Test suite for test_code crash handling.

Tests the kernel's test_code functionality with crash scenarios to ensure
proper error capture and reporting.

Run with: python -m pytest data_ferret/kernel/test_test_code_crashes.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock
from data_ferret.kernel.types import (
    TestCodeSuccess,
    TestCodeOriginalCrash,
    TestCodeModifiedCrash,
    ExecutionError,
)


class MockComm:
    """Mock comm object for testing."""

    def __init__(self):
        self.messages = []

    def send(self, data):
        """Record sent messages."""
        self.messages.append(data)


class MockShellResult:
    """Mock IPython shell result."""

    def __init__(self, error=None):
        self.error_in_exec = error


class TestTestCodeCrashes:
    """Test crash handling in test_code method."""

    @pytest.fixture
    def mock_kernel(self):
        """Create a mock FerretKernel with necessary methods."""
        from data_ferret.kernel.ferret_kernel import FerretKernel

        # Create a real kernel instance (it needs proper initialization)
        kernel = FerretKernel.instance()

        # Mock the shell.run_cell method
        kernel.shell.run_cell = MagicMock()

        return kernel

    def test_both_codes_succeed(self, mock_kernel):
        """Test when both original and modified code execute successfully."""
        comm = MockComm()

        # Mock successful execution for both codes
        mock_kernel.shell.run_cell.return_value = MockShellResult(error=None)

        # Mock checkpoint methods
        mock_kernel.checkpoint = MagicMock()

        # Create mock Checkpoint objects with user_ns attribute
        mock_checkpoint_obj = type('Checkpoint', (), {'user_ns': {}})()
        mock_kernel._checkpoint = MagicMock()
        mock_kernel._checkpoint.get = MagicMock(return_value=mock_checkpoint_obj)

        # Mock checkpoint_diff to return DiffResult with no differences
        from data_ferret.kernel import checkpoint
        from data_ferret.kernel.types import DiffResult
        original_checkpoint_diff = checkpoint.checkpoint_diff
        checkpoint.checkpoint_diff = MagicMock(
            return_value=DiffResult(differences={})
        )

        try:
            result = mock_kernel.test_code(
                comm,
                "x = 1",
                "x = 1",
                {"x"}
            )

            # Should return TestCodeSuccess
            assert "status" in result
            assert result["status"] == "success"
            assert "diff" in result
            assert "original_duration" in result
            assert "modified_duration" in result
            assert "speedup" in result

            # Check that progress messages were sent
            progress_messages = [m for m in comm.messages if m.get("type") == "progress"]
            assert len(progress_messages) > 0

        finally:
            # Restore original function
            checkpoint.checkpoint_diff = original_checkpoint_diff

    def test_original_code_crashes(self, mock_kernel):
        """Test when original code crashes during execution."""
        comm = MockComm()

        # Mock original code to crash
        def mock_run_cell(code):
            if "original" in code or "x = 1/0" in code:
                raise ZeroDivisionError("division by zero")
            return MockShellResult(error=None)

        mock_kernel.shell.run_cell = MagicMock(side_effect=mock_run_cell)
        mock_kernel.checkpoint = MagicMock()

        result = mock_kernel.test_code(
            comm,
            "x = 1/0  # This will crash",
            "x = 5",
            {"x"}
        )

        # Should return TestCodeOriginalCrash
        assert "status" in result
        assert result["status"] == "original_crash"
        assert "error" in result
        assert result["error"]["error_type"] == "ZeroDivisionError"
        assert result["error"]["error_message"] == "division by zero"
        assert "traceback" in result["error"]
        assert "original_duration" in result

        # Check that crash message was sent
        crash_messages = [
            m for m in comm.messages
            if "crashed" in m.get("message", "").lower()
        ]
        assert len(crash_messages) > 0

    def test_modified_code_crashes(self, mock_kernel):
        """Test when modified code crashes but original succeeds."""
        comm = MockComm()

        # Counter to track which execution we're on
        execution_count = [0]

        def mock_run_cell(code):
            execution_count[0] += 1
            # First call (original) succeeds
            if execution_count[0] == 1:
                return MockShellResult(error=None)
            # Second call (modified) crashes
            else:
                raise ValueError("Invalid value")

        mock_kernel.shell.run_cell = MagicMock(side_effect=mock_run_cell)

        # Mock checkpoint methods
        mock_kernel.checkpoint = MagicMock()
        mock_kernel._checkpoint = MagicMock()
        mock_kernel._checkpoint.get = MagicMock(return_value={})

        result = mock_kernel.test_code(
            comm,
            "x = 5",
            "x = int('invalid')  # This will crash",
            {"x"}
        )

        # Should return TestCodeModifiedCrash
        assert "status" in result
        assert result["status"] == "modified_crash"
        assert "error" in result
        assert result["error"]["error_type"] == "ValueError"
        assert result["error"]["error_message"] == "Invalid value"
        assert "traceback" in result["error"]
        assert "original_duration" in result
        assert "modified_duration" in result

        # Check that crash message was sent
        crash_messages = [
            m for m in comm.messages
            if "crashed" in m.get("message", "").lower()
        ]
        assert len(crash_messages) > 0

    def test_original_code_syntax_error(self, mock_kernel):
        """Test when original code has a syntax error."""
        comm = MockComm()

        # Mock syntax error
        def mock_run_cell(code):
            if "if x =" in code:
                raise SyntaxError("invalid syntax")
            return MockShellResult(error=None)

        mock_kernel.shell.run_cell = MagicMock(side_effect=mock_run_cell)
        mock_kernel.checkpoint = MagicMock()

        result = mock_kernel.test_code(
            comm,
            "if x = 5:",  # Invalid syntax
            "x = 5",
            {"x"}
        )

        # Should return TestCodeOriginalCrash
        assert "status" in result
        assert result["status"] == "original_crash"
        assert result["error"]["error_type"] == "SyntaxError"

    def test_modified_code_name_error(self, mock_kernel):
        """Test when modified code references undefined variable."""
        comm = MockComm()

        execution_count = [0]

        def mock_run_cell(code):
            execution_count[0] += 1
            if execution_count[0] == 1:
                return MockShellResult(error=None)
            else:
                raise NameError("name 'undefined_var' is not defined")

        mock_kernel.shell.run_cell = MagicMock(side_effect=mock_run_cell)
        mock_kernel.checkpoint = MagicMock()
        mock_kernel._checkpoint = MagicMock()
        mock_kernel._checkpoint.get = MagicMock(return_value={})

        result = mock_kernel.test_code(
            comm,
            "x = 5",
            "y = undefined_var + 1",
            {"x", "y"}
        )

        # Should return TestCodeModifiedCrash
        assert "status" in result
        assert result["status"] == "modified_crash"
        assert result["error"]["error_type"] == "NameError"
        assert "undefined_var" in result["error"]["error_message"]


class TestExecutionErrorModel:
    """Test the ExecutionError BaseModel."""

    def test_execution_error_creation(self):
        """Test creating an ExecutionError instance."""
        error = ExecutionError(
            error_type="ValueError",
            error_message="Test error",
            traceback="Traceback line 1\nTraceback line 2",
            code_snippet="x = invalid()"
        )

        assert error.error_type == "ValueError"
        assert error.error_message == "Test error"
        assert error.traceback == "Traceback line 1\nTraceback line 2"
        assert error.code_snippet == "x = invalid()"

    def test_execution_error_optional_code_snippet(self):
        """Test that code_snippet is optional."""
        error = ExecutionError(
            error_type="RuntimeError",
            error_message="Test error",
            traceback="Traceback"
        )

        assert error.code_snippet is None


class TestResultModels:
    """Test the result BaseModels."""

    def test_test_code_success_discriminator(self):
        """Test TestCodeSuccess has correct discriminator."""
        from data_ferret.kernel.types import DiffResult

        result = TestCodeSuccess(
            diff=DiffResult(differences={}),
            original_duration=1.0,
            modified_duration=0.5,
            speedup=2.0
        )

        assert result.status == "success"

    def test_test_code_original_crash_discriminator(self):
        """Test TestCodeOriginalCrash has correct discriminator."""
        error = ExecutionError(
            error_type="ValueError",
            error_message="Test",
            traceback="Traceback"
        )
        result = TestCodeOriginalCrash(error=error, original_duration=0.1)

        assert result.status == "original_crash"

    def test_test_code_modified_crash_discriminator(self):
        """Test TestCodeModifiedCrash has correct discriminator."""
        error = ExecutionError(
            error_type="ValueError",
            error_message="Test",
            traceback="Traceback"
        )
        result = TestCodeModifiedCrash(
            error=error,
            original_duration=1.0,
            modified_duration=0.5
        )

        assert result.status == "modified_crash"

    def test_serialization_deserialization(self):
        """Test that results can be serialized and deserialized."""
        from data_ferret.kernel.types import DiffResult
        import json

        # Test TestCodeSuccess
        success = TestCodeSuccess(
            diff=DiffResult(differences={}),
            original_duration=1.0,
            modified_duration=0.5,
            speedup=2.0
        )
        success_json = success.model_dump_json()
        success_dict = json.loads(success_json)
        assert success_dict["status"] == "success"

        # Test TestCodeOriginalCrash
        crash = TestCodeOriginalCrash(
            error=ExecutionError(
                error_type="ValueError",
                error_message="Test",
                traceback="Traceback"
            ),
            original_duration=0.1
        )
        crash_json = crash.model_dump_json()
        crash_dict = json.loads(crash_json)
        assert crash_dict["status"] == "original_crash"
