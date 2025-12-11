"""
Integration tests for kernel execution flow.

These tests verify the complete execution pipeline including:
- Tracking during execution
- Monotonicity enforcement flow
- Execution context handling
- Error handling paths
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch, AsyncMock

from data_ferret.kernel.checkpoint import Checkpoints
from data_ferret.kernel.tracking import TrackingDict
from data_ferret.kernel.monotonicity import MonotonicityEnforcer
from data_ferret.kernel.models import (
    TrackingData,
    ExecutionContext,
    ExecutionProfile,
    ExecutionMetadata,
    MonotonicityViolation,
)


class TestTrackingExecutionFlow:
    """Integration tests for tracking during execution."""

    def test_full_tracking_flow(self):
        """Test complete tracking flow: setup, execution, data capture."""
        # Setup namespace with initial values
        initial = {"x": 1, "y": 2, "z": 3}
        user_ns = TrackingDict(initial)

        # Simulate cell execution with tracking
        with user_ns.track_execution():
            # Read x and y
            val_x = user_ns["x"]
            val_y = user_ns["y"]
            # Write new variable
            user_ns["result"] = val_x + val_y
            # Overwrite z
            user_ns["z"] = 100

        # Get tracking data
        data = user_ns.get_tracking_data()

        # Verify tracking
        assert "x" in data.reads_before_writes
        assert "y" in data.reads_before_writes
        assert "z" not in data.reads_before_writes  # Was written without read
        assert "result" in data.writes
        assert "z" in data.writes

    def test_tracking_with_dataframe_columns(self):
        """Test tracking with DataFrame column access."""
        df = pd.DataFrame({
            "price": [10, 20, 30],
            "quantity": [1, 2, 3],
        })
        user_ns = TrackingDict({"df": df})

        with user_ns.track_execution():
            # Read columns
            prices = user_ns["df"]["price"]
            quantities = user_ns["df"]["quantity"]
            # Write new column
            user_ns["df"]["total"] = prices * quantities

        data = user_ns.get_tracking_data()

        # Variable-level tracking
        assert "df" in data.reads_before_writes

        # Column-level tracking
        assert "df" in data.column_reads_before_writes
        assert "price" in data.column_reads_before_writes["df"]
        assert "quantity" in data.column_reads_before_writes["df"]

    def test_tracking_with_nested_data(self):
        """Test tracking with nested data structures."""
        user_ns = TrackingDict({
            "config": {"model": "linear", "params": {"alpha": 0.1}},
            "data": [1, 2, 3],
        })

        with user_ns.track_execution():
            cfg = user_ns["config"]
            user_ns["new_config"] = {**cfg, "version": 2}

        data = user_ns.get_tracking_data()
        assert "config" in data.reads_before_writes
        assert "new_config" in data.writes


class TestMonotonicityEnforcementFlow:
    """Integration tests for monotonicity enforcement."""

    def test_monotonicity_pass_flow(self):
        """Test full monotonicity flow when check passes."""
        checkpoints = Checkpoints()
        user_ns = {"x": 10, "y": 20}

        # Save pre-state
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("cell_001")

        # Simulate execution that reads x but writes different var
        tracking = TrackingData(
            reads_before_writes=["x"],
            writes=["z"],
        )
        user_ns["z"] = 30  # Add new variable

        # Check monotonicity
        result = enforcer.check_and_enforce(tracking, "cell_001")

        assert result is None  # No violation
        assert user_ns["z"] == 30  # New var kept

    def test_monotonicity_fail_flow(self):
        """Test full monotonicity flow when check fails."""
        checkpoints = Checkpoints()
        user_ns = {"x": 10}

        # Save pre-state
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("cell_001")

        # Simulate execution that modifies read variable
        original_x = user_ns["x"]
        user_ns["x"] = 999  # Modify!

        tracking = TrackingData(
            reads_before_writes=["x"],
            writes=["x"],
        )

        # Check monotonicity
        result = enforcer.check_and_enforce(tracking, "cell_001")

        assert result is not None
        assert isinstance(result, MonotonicityViolation)
        assert "x" in result.violated_vars
        assert user_ns["x"] == original_x  # State restored!

    def test_monotonicity_with_column_tracking(self):
        """Test monotonicity with column-level tracking."""
        checkpoints = Checkpoints()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        user_ns = {"df": df.copy()}

        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("cell")

        # Only read column 'a', write column 'c'
        user_ns["df"]["c"] = [7, 8, 9]

        tracking = TrackingData(
            reads_before_writes=["df"],
            writes=["df"],
            column_reads_before_writes={"df": ["a"]},
            column_writes={"df": ["c"]},
        )

        result = enforcer.check_and_enforce(tracking, "cell")

        # Should pass - we didn't modify 'a'
        assert result is None


class TestExecutionContextFlow:
    """Integration tests for execution context handling."""

    def test_context_creation_and_usage(self):
        """Test creating and using execution context."""
        # Simulate context creation
        code = "# timeout 120\nx = 1 + 1"
        cell_id = "abcd"

        # Parse timeout
        import re
        match = re.match(r"# timeout (\d+)\n", code)
        timeout = int(match.group(1)) if match else 1800
        parsed_code = code.replace(match.group(0), "", 1) if match else code

        # Create context
        ctx = ExecutionContext(
            cell_id=cell_id,
            code=parsed_code,
            timeout=float(timeout),
            original_code=code,
        )

        assert ctx.cell_id == "abcd"
        assert ctx.code == "x = 1 + 1"
        assert ctx.timeout == 120.0
        assert ctx.should_profile is True

    def test_context_magic_detection(self):
        """Test context detects magic commands."""
        # Regular code
        ctx1 = ExecutionContext(
            cell_id="a", code="x = 1", timeout=30, original_code="x = 1"
        )
        assert ctx1.has_cell_magics is False
        assert ctx1.should_profile is True

        # Magic code
        ctx2 = ExecutionContext(
            cell_id="b", code="%time x = 1", timeout=30, original_code="%time x = 1"
        )
        assert ctx2.has_cell_magics is True
        assert ctx2.should_profile is False

        # Shell code
        ctx3 = ExecutionContext(
            cell_id="c", code="!pip list", timeout=30, original_code="!pip list"
        )
        assert ctx3.has_shell_magics is True
        assert ctx3.should_profile is False


class TestMetadataCreationFlow:
    """Integration tests for metadata creation."""

    def test_full_metadata_creation(self):
        """Test creating full execution metadata."""
        # Create profile
        profile = ExecutionProfile(
            duration=2.5,
            profile="CPU: 80%, Memory: 200MB",
            env={"x": "int", "df": "DataFrame[100x5]"},
            env_after={"x": "int", "df": "DataFrame[100x5]", "result": "float"},
        )

        # Create tracking
        tracking = TrackingData(
            reads_before_writes=["x", "df"],
            writes=["result"],
            column_reads_before_writes={"df": ["price", "qty"]},
            column_writes={"df": ["total"]},
        )

        # Create metadata
        metadata = ExecutionMetadata(
            profile=profile,
            dynamic_dependencies=tracking,
        )

        # Convert to display format
        display = metadata.to_display_metadata()

        assert display["profile"]["duration"] == 2.5
        assert "CPU" in display["profile"]["profile"]
        assert display["dynamic_dependencies"]["reads_before_writes"] == {"x", "df"}
        assert display["dynamic_dependencies"]["column_reads_before_writes"]["df"] == {"price", "qty"}


class TestErrorHandlingFlow:
    """Integration tests for error handling paths."""

    def test_monotonicity_error_to_result(self):
        """Test converting monotonicity violation to kernel error result."""
        violation = MonotonicityViolation(
            violated_vars=["x", "y"],
            diff_details="x: 1 -> 2\ny: 3 -> 4",
            error_summary="Monotonicity violation: ['x', 'y']",
        )

        result = violation.to_error_result(execution_count=42)

        assert result["status"] == "error"
        assert result["execution_count"] == 42
        assert result["ename"] == "MonotonicityError"
        assert "x" in result["evalue"]
        assert result["traceback"][0] == "x: 1 -> 2\ny: 3 -> 4"

    def test_tracking_error_recovery(self):
        """Test tracking recovers from errors in execution."""
        user_ns = TrackingDict({"x": 1})

        with pytest.raises(ValueError):
            with user_ns.track_execution():
                _ = user_ns["x"]  # Read
                user_ns["y"] = 2  # Write
                raise ValueError("Simulated error")

        # Tracking should still work
        data = user_ns.get_tracking_data()
        assert "x" in data.reads_before_writes
        assert "y" in data.writes


class TestCompleteExecutionScenarios:
    """Complete execution scenarios testing multiple components."""

    def test_scenario_data_analysis_cell(self):
        """Simulate a typical data analysis cell."""
        # Setup
        df = pd.DataFrame({
            "product": ["A", "B", "C"],
            "price": [10.0, 20.0, 30.0],
            "quantity": [100, 50, 25],
        })
        user_ns = TrackingDict({"df": df, "tax_rate": 0.1})

        # Execute with tracking
        with user_ns.track_execution():
            # Read data
            prices = user_ns["df"]["price"]
            quantities = user_ns["df"]["quantity"]
            tax = user_ns["tax_rate"]

            # Compute
            user_ns["df"]["subtotal"] = prices * quantities
            user_ns["df"]["total"] = user_ns["df"]["subtotal"] * (1 + tax)
            user_ns["summary"] = user_ns["df"]["total"].sum()

        # Check tracking
        data = user_ns.get_tracking_data()
        assert "df" in data.reads_before_writes
        assert "tax_rate" in data.reads_before_writes
        assert "summary" in data.writes

        # Column tracking
        assert "price" in data.column_reads_before_writes.get("df", [])
        assert "quantity" in data.column_reads_before_writes.get("df", [])

    def test_scenario_model_training_cell(self):
        """Simulate a model training cell."""
        # Setup
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        user_ns = TrackingDict({
            "X_train": X,
            "y_train": y,
            "learning_rate": 0.01,
        })

        # Execute
        with user_ns.track_execution():
            # Read training data
            X = user_ns["X_train"]
            y = user_ns["y_train"]
            lr = user_ns["learning_rate"]

            # "Train" a simple model
            user_ns["weights"] = np.linalg.lstsq(X, y, rcond=None)[0]
            user_ns["model_config"] = {"lr": lr, "n_features": X.shape[1]}

        data = user_ns.get_tracking_data()
        assert "X_train" in data.reads_before_writes
        assert "y_train" in data.reads_before_writes
        assert "weights" in data.writes
        assert "model_config" in data.writes

    def test_scenario_monotonicity_violation_in_loop(self):
        """Simulate monotonicity violation in accumulation loop."""
        checkpoints = Checkpoints()
        user_ns = {"total": 0, "items": [1, 2, 3, 4, 5]}

        # Save pre-state
        enforcer = MonotonicityEnforcer(checkpoints, user_ns)
        enforcer.save_pre_state("loop_cell")

        # Simulate cell that reads and modifies total
        original_total = user_ns["total"]
        for item in user_ns["items"]:
            user_ns["total"] += item  # Modifies read variable!

        tracking = TrackingData(
            reads_before_writes=["total", "items"],
            writes=["total"],
        )

        result = enforcer.check_and_enforce(tracking, "loop_cell")

        # Should detect violation
        assert result is not None
        assert "total" in result.violated_vars
        # State should be restored
        assert user_ns["total"] == original_total


class TestEdgeCasesIntegration:
    """Integration tests for edge cases."""

    def test_empty_cell_execution(self):
        """Test executing empty cell."""
        user_ns = TrackingDict({"x": 1})

        with user_ns.track_execution():
            pass  # Empty execution

        data = user_ns.get_tracking_data()
        assert data.reads_before_writes == set()
        assert data.writes == set()

    def test_readonly_cell(self):
        """Test cell that only reads."""
        user_ns = TrackingDict({"a": 1, "b": 2, "c": 3})

        with user_ns.track_execution():
            _ = user_ns["a"]
            _ = user_ns["b"]
            _ = user_ns["c"]

        data = user_ns.get_tracking_data()
        assert data.reads_before_writes == {"a", "b", "c"}
        assert data.writes == set()

    def test_writeonly_cell(self):
        """Test cell that only writes."""
        user_ns = TrackingDict()

        with user_ns.track_execution():
            user_ns["x"] = 1
            user_ns["y"] = 2
            user_ns["z"] = 3

        data = user_ns.get_tracking_data()
        assert data.reads_before_writes == set()
        assert data.writes == {"x", "y", "z"}

    def test_same_var_read_write_cycle(self):
        """Test reading and writing same variable."""
        user_ns = TrackingDict({"counter": 0})

        with user_ns.track_execution():
            val = user_ns["counter"]  # Read
            user_ns["counter"] = val + 1  # Write

        data = user_ns.get_tracking_data()
        assert "counter" in data.reads_before_writes
        assert "counter" in data.writes

    def test_large_dataframe_tracking(self):
        """Test tracking with large DataFrame."""
        df = pd.DataFrame(
            np.random.randn(10000, 100),
            columns=[f"col_{i}" for i in range(100)]
        )
        user_ns = TrackingDict({"big_df": df})

        with user_ns.track_execution():
            # Access several columns
            _ = user_ns["big_df"]["col_0"]
            _ = user_ns["big_df"]["col_50"]
            user_ns["big_df"]["new_col"] = 0

        data = user_ns.get_tracking_data()
        assert "big_df" in data.reads_before_writes
        assert "col_0" in data.column_reads_before_writes.get("big_df", [])
        assert "col_50" in data.column_reads_before_writes.get("big_df", [])
