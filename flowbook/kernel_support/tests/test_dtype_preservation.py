"""Tests for dtype preservation across checkpoint save/restore cycles."""

import numpy as np
import pandas as pd
import pytest

from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints


class TestSaveDoesNotMutateOriginal:
    """Saving a checkpoint must not modify the original DataFrame/Series dtypes."""

    def test_dataframe_object_columns_preserved(self):
        """Save does not modify original DataFrame dtypes."""
        checkpoints = MemoryCheckpoints()
        df = pd.DataFrame({
            "strings": pd.Series(["a", "b", "c"], dtype=object),
            "ints": pd.Series([1, 2, 3], dtype=object),
        })
        user_ns = {"df": df}

        checkpoints.save("cp1", user_ns)

        assert user_ns["df"]["strings"].dtype == object
        assert user_ns["df"]["ints"].dtype == object

    def test_series_object_dtype_preserved(self):
        """Save does not modify original Series dtype."""
        checkpoints = MemoryCheckpoints()
        s = pd.Series(["x", "y", "z"], dtype=object)
        user_ns = {"s": s}

        checkpoints.save("cp1", user_ns)

        assert user_ns["s"].dtype == object


class TestCheckpointStoresConvertedDtypes:
    """The checkpoint internally should store converted (specialized) dtypes."""

    def test_checkpoint_has_specialized_dtypes(self):
        """Checkpoint internally stores converted dtypes for efficiency."""
        checkpoints = MemoryCheckpoints()
        df = pd.DataFrame({
            "strings": pd.Series(["a", "b", "c"], dtype=object),
            "ints": pd.Series([1, 2, 3], dtype=object),
        })
        user_ns = {"df": df}

        checkpoints.save("cp1", user_ns)

        cp = checkpoints.get("cp1")
        # Internally, the checkpoint should have converted dtypes
        assert cp.user_ns["df"]["strings"].dtype == pd.StringDtype()
        assert cp.user_ns["df"]["ints"].dtype == pd.Int64Dtype()


class TestRestoreReturnsOriginalDtypes:
    """Restoring a checkpoint must return variables with original dtypes."""

    def test_restore_dataframe_object_columns(self):
        """Restore returns DataFrame with original object dtypes."""
        checkpoints = MemoryCheckpoints()
        df = pd.DataFrame({
            "strings": pd.Series(["a", "b", "c"], dtype=object),
            "ints": pd.Series([1, 2, 3], dtype=object),
            "floats": pd.Series([1.5, 2.5, 3.5], dtype=object),
        })
        user_ns = {"df": df}

        checkpoints.save("cp1", user_ns)

        # Mutate user_ns to simulate execution
        user_ns["df"] = pd.DataFrame({"x": [1]})

        checkpoints.restore("cp1", user_ns)

        assert user_ns["df"]["strings"].dtype == object
        assert user_ns["df"]["ints"].dtype == object
        assert user_ns["df"]["floats"].dtype == object

    def test_restore_series_object_dtype(self):
        """Restore preserves original Series dtype."""
        checkpoints = MemoryCheckpoints()
        s = pd.Series(["hello", "world", None], dtype=object)
        user_ns = {"s": s}

        checkpoints.save("cp1", user_ns)

        user_ns["s"] = pd.Series([999])

        checkpoints.restore("cp1", user_ns)

        assert user_ns["s"].dtype == object

    def test_restore_values_correct(self):
        """Round-trip values are correct including None/NaN."""
        checkpoints = MemoryCheckpoints()
        df = pd.DataFrame({
            "strings": pd.Series(["a", None, "c"], dtype=object),
            "ints": pd.Series([1, None, 3], dtype=object),
        })
        user_ns = {"df": df}

        checkpoints.save("cp1", user_ns)
        user_ns["df"] = pd.DataFrame({"x": [1]})
        checkpoints.restore("cp1", user_ns)

        # Values should be correct
        assert user_ns["df"]["strings"][0] == "a"
        assert user_ns["df"]["strings"][2] == "c"
        assert user_ns["df"]["ints"][0] == 1
        assert user_ns["df"]["ints"][2] == 3
        # None/NA values should be missing (pd.isna covers both None and pd.NA)
        assert pd.isna(user_ns["df"]["strings"][1])
        assert pd.isna(user_ns["df"]["ints"][1])


class TestMultipleCycles:
    """Multiple save/restore cycles must preserve dtypes."""

    def test_multiple_save_restore_cycles(self):
        """Multiple save/restore cycles preserve dtypes."""
        checkpoints = MemoryCheckpoints()
        df = pd.DataFrame({
            "col": pd.Series(["a", "b", "c"], dtype=object),
        })
        user_ns = {"df": df}

        for i in range(3):
            checkpoints.save(f"cp{i}", user_ns)
            user_ns["df"] = pd.DataFrame({"x": [1]})
            checkpoints.restore(f"cp{i}", user_ns)
            assert user_ns["df"]["col"].dtype == object, \
                f"Cycle {i}: expected object, got {user_ns['df']['col'].dtype}"


class TestNonObjectDtypesUnaffected:
    """Non-object dtype columns should pass through unchanged."""

    def test_numeric_columns_unchanged(self):
        """Non-object dtype columns are unaffected by save/restore."""
        checkpoints = MemoryCheckpoints()
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "obj_col": pd.Series(["a", "b", "c"], dtype=object),
        })
        user_ns = {"df": df}

        checkpoints.save("cp1", user_ns)
        user_ns["df"] = pd.DataFrame({"x": [1]})
        checkpoints.restore("cp1", user_ns)

        assert user_ns["df"]["int_col"].dtype == np.int64
        assert user_ns["df"]["float_col"].dtype == np.float64
        assert user_ns["df"]["obj_col"].dtype == object


class TestLightGBMScenario:
    """Simulate the LightGBM scenario: object string columns must survive round-trip."""

    def test_lightgbm_like_object_strings(self):
        """Object-dtype string columns survive checkpoint round-trip for LightGBM compat."""
        checkpoints = MemoryCheckpoints()
        # LightGBM expects object dtype for string columns, not StringDtype
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "category": pd.Series(["cat_a", "cat_b", "cat_a"], dtype=object),
            "label": [0, 1, 0],
        })
        user_ns = {"train_df": df}

        # Save checkpoint
        checkpoints.save("before_train", user_ns)

        # Simulate training that modifies df
        user_ns["train_df"]["category"] = pd.Series(["modified"], dtype=object)

        # Restore
        checkpoints.restore("before_train", user_ns)

        # The category column must be object dtype (not StringDtype)
        assert user_ns["train_df"]["category"].dtype == object, \
            f"Expected object dtype for LightGBM compat, got {user_ns['train_df']['category'].dtype}"
        assert list(user_ns["train_df"]["category"]) == ["cat_a", "cat_b", "cat_a"]
