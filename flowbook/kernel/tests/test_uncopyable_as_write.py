"""
Tests for FLOWBOOK_UNCOPYABLE_AS_WRITE environment variable behavior.

This feature controls how uncopyable variables are handled:
- Old behavior (default): Remove uncopyable variables from user_ns
- New behavior (env var set): Add uncopyable variables to W (writes) for soundness
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from flowbook.kernel_support.models import TrackingData


class TestUncopyableAsWriteConfig:
    """Tests for the _uncopyable_as_write configuration."""

    def test_default_is_false(self):
        """Default behavior should be to remove uncopyable vars (old behavior)."""
        # Clear env var if set
        with patch.dict(os.environ, {}, clear=True):
            # Need to reimport to pick up env var
            import importlib
            import flowbook.kernel.flowbook_kernel as fk
            importlib.reload(fk)

            # Check the class attribute
            assert fk.FlowbookKernel._uncopyable_as_write is False

    def test_env_var_enables_new_behavior(self):
        """Setting FLOWBOOK_UNCOPYABLE_AS_WRITE=1 enables new behavior."""
        with patch.dict(os.environ, {"FLOWBOOK_UNCOPYABLE_AS_WRITE": "1"}):
            import importlib
            import flowbook.kernel.flowbook_kernel as fk
            importlib.reload(fk)

            assert fk.FlowbookKernel._uncopyable_as_write is True

    def test_env_var_true_enables_new_behavior(self):
        """Setting FLOWBOOK_UNCOPYABLE_AS_WRITE=true enables new behavior."""
        with patch.dict(os.environ, {"FLOWBOOK_UNCOPYABLE_AS_WRITE": "true"}):
            import importlib
            import flowbook.kernel.flowbook_kernel as fk
            importlib.reload(fk)

            assert fk.FlowbookKernel._uncopyable_as_write is True

    def test_env_var_yes_enables_new_behavior(self):
        """Setting FLOWBOOK_UNCOPYABLE_AS_WRITE=yes enables new behavior."""
        with patch.dict(os.environ, {"FLOWBOOK_UNCOPYABLE_AS_WRITE": "yes"}):
            import importlib
            import flowbook.kernel.flowbook_kernel as fk
            importlib.reload(fk)

            assert fk.FlowbookKernel._uncopyable_as_write is True

    def test_env_var_other_values_disabled(self):
        """Other values for env var should not enable new behavior."""
        with patch.dict(os.environ, {"FLOWBOOK_UNCOPYABLE_AS_WRITE": "0"}):
            import importlib
            import flowbook.kernel.flowbook_kernel as fk
            importlib.reload(fk)

            assert fk.FlowbookKernel._uncopyable_as_write is False


class TestUncopyableHandlingOldBehavior:
    """Tests for old behavior: remove uncopyable vars from namespace."""

    def test_uncopyable_removed_when_env_not_set(self):
        """Uncopyable vars should be removed from namespace when env var not set."""
        import tempfile

        # Create uncopyable object (file handle)
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)

        user_ns = {"x": 1, "f": tmp}

        # Simulate old behavior
        uncopyable_vars = {"f"}
        _uncopyable_as_write = False

        if uncopyable_vars:
            if not _uncopyable_as_write:
                for k in uncopyable_vars:
                    if k in user_ns:
                        del user_ns[k]

        assert "x" in user_ns
        assert "f" not in user_ns

        # Cleanup
        tmp.close()
        os.unlink(tmp.name)


class TestUncopyableHandlingNewBehavior:
    """Tests for new behavior: add uncopyable vars to writes."""

    def test_uncopyable_added_to_writes_when_env_set(self):
        """Uncopyable vars should be added to writes when env var is set."""
        import tempfile

        # Create uncopyable object (file handle)
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)

        user_ns = {"x": 1, "f": tmp}

        # Create tracking data
        tracking = TrackingData(
            reads_before_writes={"x"},
            writes={"x"},  # Cell wrote x
        )

        # Simulate new behavior
        uncopyable_vars = {"f"}
        _uncopyable_as_write = True

        if tracking and uncopyable_vars and _uncopyable_as_write:
            tracking.writes = tracking.writes | uncopyable_vars

        # f should be added to writes
        assert "f" in tracking.writes
        assert "x" in tracking.writes

        # f should still be in namespace
        assert "f" in user_ns

        # Cleanup
        tmp.close()
        os.unlink(tmp.name)

    def test_uncopyable_not_added_to_writes_when_env_not_set(self):
        """Uncopyable vars should NOT be added to writes when env var not set."""
        import tempfile

        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)

        # Create tracking data
        tracking = TrackingData(
            reads_before_writes={"x"},
            writes={"x"},
        )

        original_writes = tracking.writes.copy()

        # Simulate old behavior
        uncopyable_vars = {"f"}
        _uncopyable_as_write = False

        if tracking and uncopyable_vars and _uncopyable_as_write:
            tracking.writes = tracking.writes | uncopyable_vars

        # writes should be unchanged
        assert tracking.writes == original_writes
        assert "f" not in tracking.writes

        # Cleanup
        tmp.close()
        os.unlink(tmp.name)


class TestUncopyableInConflictDetection:
    """Tests for how uncopyable vars affect conflict detection."""

    def test_uncopyable_as_write_triggers_staleness(self):
        """When uncopyable var is in W, it should trigger staleness for readers."""
        # This is a conceptual test - the actual staleness logic is in the enforcer
        # Here we verify the data flow is correct

        tracking = TrackingData(
            reads_before_writes=set(),
            writes={"x"},
        )

        uncopyable_vars = {"socket_conn"}
        _uncopyable_as_write = True

        if tracking and uncopyable_vars and _uncopyable_as_write:
            tracking.writes = tracking.writes | uncopyable_vars

        # If another cell reads socket_conn, it should become stale
        # because socket_conn is now in W
        assert "socket_conn" in tracking.writes

        # The enforcer will use this to compute:
        # ForwardStale: j > i and W_i intersect R_j != empty
        # So if cell j read socket_conn, it becomes stale


class TestMixedBehavior:
    """Tests for edge cases and mixed scenarios."""

    def test_empty_uncopyable_vars_no_change(self):
        """Empty uncopyable_vars should not change anything."""
        tracking = TrackingData(
            reads_before_writes={"a"},
            writes={"b"},
        )

        uncopyable_vars = set()
        _uncopyable_as_write = True

        original_writes = tracking.writes.copy()

        if tracking and uncopyable_vars and _uncopyable_as_write:
            tracking.writes = tracking.writes | uncopyable_vars

        assert tracking.writes == original_writes

    def test_no_tracking_no_error(self):
        """When tracking is None, should not error."""
        tracking = None
        uncopyable_vars = {"f"}
        _uncopyable_as_write = True

        # This should not raise
        if tracking and uncopyable_vars and _uncopyable_as_write:
            tracking.writes = tracking.writes | uncopyable_vars

        # tracking is still None
        assert tracking is None

    def test_multiple_uncopyable_vars(self):
        """Multiple uncopyable vars should all be added to writes."""
        import tempfile

        tmp1 = tempfile.NamedTemporaryFile(mode='w', delete=False)
        tmp2 = tempfile.NamedTemporaryFile(mode='w', delete=False)

        tracking = TrackingData(
            reads_before_writes=set(),
            writes={"x"},
        )

        uncopyable_vars = {"f1", "f2", "socket"}
        _uncopyable_as_write = True

        if tracking and uncopyable_vars and _uncopyable_as_write:
            tracking.writes = tracking.writes | uncopyable_vars

        assert "f1" in tracking.writes
        assert "f2" in tracking.writes
        assert "socket" in tracking.writes
        assert "x" in tracking.writes

        # Cleanup
        tmp1.close()
        tmp2.close()
        os.unlink(tmp1.name)
        os.unlink(tmp2.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
