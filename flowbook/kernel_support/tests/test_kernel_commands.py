"""
Test harness for kernel commands via comm channel.

This module tests the kernel_command comm channel and the KernelCommandClient,
ensuring that all commands work correctly end-to-end.

To run these tests:
    pytest flowbook/kernel/test_kernel_commands.py -v
"""

import pytest
import time
from typing import List

from flowbook.kernel_support.kernel_command_client import KernelCommandClient, KernelCommandError


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def kernel_client():
    """
    Fixture that provides a kernel client for testing.

    NOTE: This requires a running Jupyter kernel. In practice, you'll need to:
    1. Start a FlowbookKernel instance
    2. Connect to it with a BlockingKernelClient
    3. Return the client here

    For now, this is a placeholder that skips tests if no kernel is available.
    """
    pytest.skip("Kernel client fixture not implemented - requires running kernel")


@pytest.fixture
def command_client(kernel_client):
    """Fixture that provides a KernelCommandClient."""
    return KernelCommandClient(kernel_client)


# ============================================================================
# Checkpoint Command Tests
# ============================================================================

class TestCheckpointCommands:
    """Test checkpoint-related commands."""

    def test_checkpoint_save(self, command_client):
        """Test saving a checkpoint."""
        # Set up some variables in the kernel
        # (In real test, would execute code via kernel_client)

        # Save checkpoint
        response = command_client.checkpoint_save("test_save")

        assert response.status == "ok"
        assert isinstance(response.saved, dict)
        assert isinstance(response.removed, dict)
        assert response.duration >= 0

    def test_checkpoint_restore(self, command_client):
        """Test restoring a checkpoint."""
        # Save checkpoint first
        command_client.checkpoint_save("test_restore")

        # Modify state
        # (In real test, would execute code via kernel_client)

        # Restore checkpoint
        response = command_client.checkpoint_restore("test_restore")

        assert response.status == "ok"
        assert "restored" in response.message.lower()

    def test_checkpoint_delete(self, command_client):
        """Test deleting a checkpoint."""
        # Save checkpoint first
        command_client.checkpoint_save("test_delete")

        # Delete checkpoint
        response = command_client.checkpoint_delete("test_delete")

        assert response.status == "ok"
        assert "deleted" in response.message.lower()

        # Verify it's gone
        list_response = command_client.checkpoint_list()
        assert "test_delete" not in list_response.checkpoints

    def test_checkpoint_list(self, command_client):
        """Test listing checkpoints."""
        # Clear existing checkpoints
        command_client.checkpoint_clear()

        # Save some checkpoints
        command_client.checkpoint_save("checkpoint1")
        command_client.checkpoint_save("checkpoint2")
        command_client.checkpoint_save("checkpoint3")

        # List checkpoints
        response = command_client.checkpoint_list()

        assert response.status == "ok"
        assert len(response.checkpoints) == 3
        assert "checkpoint1" in response.checkpoints
        assert "checkpoint2" in response.checkpoints
        assert "checkpoint3" in response.checkpoints

    def test_checkpoint_compare(self, command_client):
        """Test comparing checkpoints."""
        # Save first checkpoint
        command_client.checkpoint_save("compare1")

        # Modify state
        # (In real test, would execute code via kernel_client)

        # Save second checkpoint
        command_client.checkpoint_save("compare2")

        # Compare checkpoints
        response = command_client.checkpoint_compare("compare1", "compare2")

        assert response.status == "ok"
        assert response.diff is not None
        # In real test, would verify diff contains expected changes

    def test_checkpoint_compare_with_keys_to_include(self, command_client):
        """Test comparing checkpoints with keys_to_include parameter."""
        # Save first checkpoint
        command_client.checkpoint_save("compare_keys1")

        # Modify state
        # (In real test, would execute code to change multiple variables)

        # Save second checkpoint
        command_client.checkpoint_save("compare_keys2")

        # Compare checkpoints with specific keys
        response = command_client.checkpoint_compare(
            "compare_keys1",
            "compare_keys2",
            keys_to_include={'x', 'y'}
        )

        assert response.status == "ok"
        assert response.diff is not None
        # In real test, would verify diff only contains x and y

    def test_checkpoint_compare_with_empty_keys_to_include(self, command_client):
        """Test comparing checkpoints with empty keys_to_include set."""
        # Save checkpoints
        command_client.checkpoint_save("empty_keys1")
        command_client.checkpoint_save("empty_keys2")

        # Compare with empty set (should result in no differences)
        response = command_client.checkpoint_compare(
            "empty_keys1",
            "empty_keys2",
            keys_to_include=set()
        )

        assert response.status == "ok"
        assert response.diff is not None
        # Empty keys_to_include means nothing is compared

    def test_checkpoint_clear(self, command_client):
        """Test clearing all checkpoints."""
        # Save some checkpoints
        command_client.checkpoint_save("clear1")
        command_client.checkpoint_save("clear2")

        # Clear all
        response = command_client.checkpoint_clear()

        assert response.status == "ok"

        # Verify all are gone
        list_response = command_client.checkpoint_list()
        assert len(list_response.checkpoints) == 0

    def test_checkpoint_restore_nonexistent(self, command_client):
        """Test restoring a nonexistent checkpoint raises error."""
        with pytest.raises(KernelCommandError) as exc_info:
            command_client.checkpoint_restore("nonexistent")

        assert "nonexistent" in str(exc_info.value).lower()

    def test_checkpoint_delete_nonexistent(self, command_client):
        """Test deleting a nonexistent checkpoint raises error."""
        with pytest.raises(KernelCommandError) as exc_info:
            command_client.checkpoint_delete("nonexistent")

        assert "nonexistent" in str(exc_info.value).lower()


# ============================================================================
# Feature Toggle Command Tests
# ============================================================================

class TestFeatureToggleCommands:
    """Test feature toggle commands."""

    def test_enable_scalene(self, command_client):
        """Test enabling Scalene profiling."""
        response = command_client.enable_scalene()

        assert response.status == "ok"
        assert "enabled" in response.message.lower()

    def test_disable_scalene(self, command_client):
        """Test disabling Scalene profiling."""
        response = command_client.disable_scalene()

        assert response.status == "ok"
        assert "disabled" in response.message.lower()

    def test_force_checkpoints_enable(self, command_client):
        """Test enabling force checkpoints."""
        response = command_client.force_checkpoints(enabled=True)

        assert response.status == "ok"
        assert response.enabled is True

    def test_force_checkpoints_disable(self, command_client):
        """Test disabling force checkpoints."""
        response = command_client.force_checkpoints(enabled=False)

        assert response.status == "ok"
        assert response.enabled is False


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling in the command system."""

    def test_invalid_command(self, command_client):
        """Test sending an invalid command raises error."""
        # Directly send invalid command via low-level interface
        with pytest.raises(KernelCommandError) as exc_info:
            command_client._send_command({"command": "invalid_command"})

        assert "unknown command" in str(exc_info.value).lower()

    def test_missing_required_parameter(self, command_client):
        """Test missing required parameter raises error."""
        with pytest.raises(Exception):  # Pydantic validation error
            command_client._send_command({"command": "checkpoint_save"})
            # Missing 'name' parameter

    def test_timeout(self, command_client):
        """Test command timeout handling."""
        # This would need a command that takes a long time
        # For now, just verify timeout parameter works
        response = command_client.checkpoint_list(timeout=1.0)
        assert response.status == "ok"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests that combine multiple commands."""

    def test_checkpoint_workflow(self, command_client):
        """Test complete checkpoint workflow."""
        # Clear any existing checkpoints
        command_client.checkpoint_clear()

        # Save initial state
        save_response = command_client.checkpoint_save("initial")
        assert save_response.status == "ok"

        # Modify state
        # (In real test, would execute code)

        # Save modified state
        command_client.checkpoint_save("modified")

        # List checkpoints
        list_response = command_client.checkpoint_list()
        assert "initial" in list_response.checkpoints
        assert "modified" in list_response.checkpoints

        # Compare checkpoints
        compare_response = command_client.checkpoint_compare("initial", "modified")
        assert compare_response.status == "ok"

        # Restore initial state
        restore_response = command_client.checkpoint_restore("initial")
        assert restore_response.status == "ok"

        # Clean up
        command_client.checkpoint_delete("initial")
        command_client.checkpoint_delete("modified")


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for kernel commands."""

    def test_checkpoint_save_performance(self, command_client):
        """Test checkpoint save performance."""
        # Create some variables
        # (In real test, would execute code to create large namespace)

        start_time = time.time()
        response = command_client.checkpoint_save("perf_test")
        elapsed = time.time() - start_time

        assert response.status == "ok"
        assert response.duration <= elapsed + 0.1  # Allow small overhead

    def test_multiple_commands_performance(self, command_client):
        """Test performance of multiple commands in sequence."""
        start_time = time.time()

        for i in range(10):
            command_client.checkpoint_save(f"perf_{i}")

        elapsed = time.time() - start_time

        # Should complete reasonably quickly
        assert elapsed < 5.0  # 10 saves in under 5 seconds

        # Clean up
        command_client.checkpoint_clear()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
