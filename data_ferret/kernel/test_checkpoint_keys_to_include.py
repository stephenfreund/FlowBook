"""
Unit tests for checkpoint comparison with keys_to_include parameter.

These tests verify that the keys_to_include feature works correctly
at multiple levels: Checkpoint.diff, handler, and client.

To run these tests:
    pytest data_ferret/kernel/test_checkpoint_keys_to_include.py -v
"""

import pytest
from unittest.mock import Mock

from data_ferret.kernel.checkpoint import Checkpoint
from data_ferret.kernel.kernel_command_handlers import KernelCommandHandlers
from data_ferret.kernel.kernel_commands import CheckpointCompareRequest
from data_ferret.kernel.types import DiffResult


class TestCheckpointDiffKeysToInclude:
    """Test Checkpoint.diff method with keys_to_include parameter."""

    def test_checkpoint_diff_all_keys_when_none(self):
        """Test that Checkpoint.diff compares all keys when keys_to_include is None."""
        # Create two checkpoints with different values
        cp1 = Checkpoint("cp1", {'x': 1, 'y': 2, 'z': 3}, {})
        cp2 = Checkpoint("cp2", {'x': 1, 'y': 999, 'z': 3}, {})

        # Compare without keys_to_include (should compare all)
        result = Checkpoint.diff(cp1, cp2, keys_to_include=None)

        # Should detect difference in 'y'
        assert isinstance(result, DiffResult)
        assert 'y' in result.differences
        # x and z should not be in differences (they match)
        assert 'x' not in result.differences
        assert 'z' not in result.differences

    def test_checkpoint_diff_specific_keys_only(self):
        """Test that Checkpoint.diff only compares specified keys."""
        # Create two checkpoints with differences in multiple variables
        cp1 = Checkpoint("cp1", {'x': 1, 'y': 2, 'z': 3}, {})
        cp2 = Checkpoint("cp2", {'x': 999, 'y': 999, 'z': 3}, {})

        # Compare only 'z' (which is the same in both)
        result = Checkpoint.diff(cp1, cp2, keys_to_include={'z'})

        # Should have no differences (z is the same)
        assert isinstance(result, DiffResult)
        assert len(result.differences) == 0

    def test_checkpoint_diff_includes_only_different_specified_keys(self):
        """Test that only differences in specified keys are reported."""
        # Create two checkpoints with differences in multiple variables
        cp1 = Checkpoint("cp1", {'x': 1, 'y': 2, 'z': 3}, {})
        cp2 = Checkpoint("cp2", {'x': 999, 'y': 999, 'z': 3}, {})

        # Compare only 'x' and 'z'
        result = Checkpoint.diff(cp1, cp2, keys_to_include={'x', 'z'})

        # Should only report difference in 'x', not 'y'
        assert isinstance(result, DiffResult)
        assert 'x' in result.differences
        assert 'y' not in result.differences
        assert 'z' not in result.differences  # z is the same

    def test_checkpoint_diff_empty_keys_to_include(self):
        """Test that empty keys_to_include set results in no comparison."""
        cp1 = Checkpoint("cp1", {'x': 1, 'y': 2}, {})
        cp2 = Checkpoint("cp2", {'x': 999, 'y': 999}, {})

        # Compare with empty set
        result = Checkpoint.diff(cp1, cp2, keys_to_include=set())

        # Should have no differences (nothing compared)
        assert isinstance(result, DiffResult)
        assert len(result.differences) == 0

    def test_checkpoint_diff_nonexistent_keys(self):
        """Test that specifying nonexistent keys doesn't cause errors."""
        cp1 = Checkpoint("cp1", {'x': 1, 'y': 2}, {})
        cp2 = Checkpoint("cp2", {'x': 1, 'y': 2}, {})

        # Compare with keys that don't exist
        result = Checkpoint.diff(cp1, cp2, keys_to_include={'nonexistent', 'also_missing'})

        # Should have no differences (keys don't exist in either checkpoint)
        assert isinstance(result, DiffResult)
        assert len(result.differences) == 0

    def test_checkpoint_diff_mixed_existing_nonexisting_keys(self):
        """Test with mix of existing and nonexistent keys."""
        cp1 = Checkpoint("cp1", {'x': 1, 'y': 2, 'z': 3}, {})
        cp2 = Checkpoint("cp2", {'x': 999, 'y': 2, 'z': 3}, {})

        # Compare with mix of existing and nonexistent keys
        result = Checkpoint.diff(cp1, cp2, keys_to_include={'x', 'y', 'nonexistent'})

        # Should report difference in 'x', not 'y', and ignore nonexistent
        assert isinstance(result, DiffResult)
        assert 'x' in result.differences
        assert 'y' not in result.differences
        assert 'z' not in result.differences
        assert 'nonexistent' not in result.differences

    def test_checkpoint_diff_complex_types(self):
        """Test keys_to_include with complex data types."""
        import pandas as pd
        import numpy as np

        cp1 = Checkpoint("cp1", {
            'df': pd.DataFrame({'a': [1, 2, 3]}),
            'arr': np.array([1, 2, 3]),
            'num': 42
        }, {})

        cp2 = Checkpoint("cp2", {
            'df': pd.DataFrame({'a': [1, 2, 999]}),  # Different
            'arr': np.array([1, 2, 3]),  # Same
            'num': 42  # Same
        }, {})

        # Compare only 'arr' and 'num' (both the same)
        result = Checkpoint.diff(cp1, cp2, keys_to_include={'arr', 'num'})

        # Should have no differences
        assert isinstance(result, DiffResult)
        assert len(result.differences) == 0

        # Now compare all to verify df is different
        result_all = Checkpoint.diff(cp1, cp2, keys_to_include=None)
        assert 'df' in result_all.differences


class TestHandlerCheckpointCompareKeysToInclude:
    """Test handler's checkpoint_compare with keys_to_include parameter."""

    @pytest.fixture
    def mock_kernel(self):
        """Create a mock FerretKernel for testing."""
        kernel = Mock()
        kernel._checkpoint = Mock()
        return kernel

    @pytest.fixture
    def handlers(self, mock_kernel):
        """Create KernelCommandHandlers with mock kernel."""
        return KernelCommandHandlers(mock_kernel)

    def test_handle_checkpoint_compare_with_keys_to_include(self, handlers, mock_kernel):
        """Test that handler passes keys_to_include to Checkpoint.diff."""
        # Setup mocks
        cp1 = Checkpoint("cp1", {'x': 1, 'y': 2, 'z': 3}, {})
        cp2 = Checkpoint("cp2", {'x': 999, 'y': 2, 'z': 3}, {})
        mock_kernel._checkpoint.get.side_effect = [cp1, cp2]

        # Create request with keys_to_include
        req = CheckpointCompareRequest(
            name1="cp1",
            name2="cp2",
            keys_to_include={'y', 'z'}
        )
        response = handlers.handle_checkpoint_compare(req)

        # Verify response
        assert response.status == "ok"
        assert isinstance(response.diff, DiffResult)

        # Should only compare y and z, both are the same, so no differences
        assert len(response.diff.differences) == 0

    def test_handle_checkpoint_compare_without_keys_to_include(self, handlers, mock_kernel):
        """Test that handler works correctly when keys_to_include is None."""
        # Setup mocks
        cp1 = Checkpoint("cp1", {'x': 1, 'y': 2}, {})
        cp2 = Checkpoint("cp2", {'x': 999, 'y': 2}, {})
        mock_kernel._checkpoint.get.side_effect = [cp1, cp2]

        # Create request without keys_to_include
        req = CheckpointCompareRequest(name1="cp1", name2="cp2")
        response = handlers.handle_checkpoint_compare(req)

        # Verify response
        assert response.status == "ok"
        assert isinstance(response.diff, DiffResult)

        # Should detect difference in 'x'
        assert 'x' in response.diff.differences

    def test_handle_checkpoint_compare_empty_keys_to_include(self, handlers, mock_kernel):
        """Test handler with empty keys_to_include set."""
        # Setup mocks
        cp1 = Checkpoint("cp1", {'x': 1, 'y': 2}, {})
        cp2 = Checkpoint("cp2", {'x': 999, 'y': 999}, {})
        mock_kernel._checkpoint.get.side_effect = [cp1, cp2]

        # Create request with empty keys_to_include
        req = CheckpointCompareRequest(
            name1="cp1",
            name2="cp2",
            keys_to_include=set()
        )
        response = handlers.handle_checkpoint_compare(req)

        # Verify response
        assert response.status == "ok"
        assert isinstance(response.diff, DiffResult)

        # Should have no differences (nothing compared)
        assert len(response.diff.differences) == 0


class TestRequestModel:
    """Test CheckpointCompareRequest model with keys_to_include."""

    def test_request_with_keys_to_include(self):
        """Test creating request with keys_to_include."""
        req = CheckpointCompareRequest(
            name1="cp1",
            name2="cp2",
            keys_to_include={'x', 'y', 'z'}
        )

        assert req.name1 == "cp1"
        assert req.name2 == "cp2"
        assert req.keys_to_include == {'x', 'y', 'z'}
        assert req.command == "checkpoint_compare"

    def test_request_without_keys_to_include(self):
        """Test creating request without keys_to_include (defaults to None)."""
        req = CheckpointCompareRequest(name1="cp1", name2="cp2")

        assert req.name1 == "cp1"
        assert req.name2 == "cp2"
        assert req.keys_to_include is None
        assert req.command == "checkpoint_compare"

    def test_request_serialization(self):
        """Test that request can be serialized and deserialized."""
        req = CheckpointCompareRequest(
            name1="cp1",
            name2="cp2",
            keys_to_include={'x', 'y'}
        )

        # Serialize
        data = req.model_dump()

        # Verify serialized data
        assert data['name1'] == "cp1"
        assert data['name2'] == "cp2"
        assert data['keys_to_include'] == {'x', 'y'}

        # Deserialize
        req2 = CheckpointCompareRequest(**data)
        assert req2.name1 == req.name1
        assert req2.name2 == req.name2
        assert req2.keys_to_include == req.keys_to_include


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
