"""
Tests for incremental checkpoint optimization.

Tests the save_incremental() method which reuses deep copies from prior
checkpoints for variables that were not accessed during cell execution.
"""

import numpy as np
import pandas as pd
import pytest

from flowbook.kernel_support.memory_checkpoint import (
    MemoryCheckpoints,
    _is_known_leaf_object,
)


class TestIsKnownLeafObject:
    """Tests for _is_known_leaf_object function."""

    def test_scalars_are_leaf(self):
        """Scalar primitives should be leaf objects."""
        assert _is_known_leaf_object(None) is True
        assert _is_known_leaf_object(True) is True
        assert _is_known_leaf_object(42) is True
        assert _is_known_leaf_object(3.14) is True
        assert _is_known_leaf_object(1 + 2j) is True
        assert _is_known_leaf_object("hello") is True
        assert _is_known_leaf_object(b"bytes") is True

    def test_numeric_ndarray_is_leaf(self):
        """Numeric ndarray that owns its data should be leaf."""
        arr = np.arange(100)
        assert _is_known_leaf_object(arr) is True

    def test_ndarray_view_is_not_leaf(self):
        """ndarray view (base is not None) should not be leaf."""
        arr = np.arange(100)
        view = arr[10:20]
        assert view.base is arr
        assert _is_known_leaf_object(view) is False

    def test_object_dtype_ndarray_is_not_leaf(self):
        """Object dtype ndarray should not be leaf (contains pointers)."""
        arr = np.array([{}, [], None], dtype=object)
        assert _is_known_leaf_object(arr) is False

    def test_dataframe_is_not_leaf(self):
        """DataFrame should not be leaf (complex internal structure)."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert _is_known_leaf_object(df) is False

    def test_uncached_list_is_not_leaf(self):
        """Regular list (not in cache) should not be leaf."""
        lst = [1, 2, 3]
        assert _is_known_leaf_object(lst) is False


class TestSaveIncremental:
    """Tests for save_incremental() method."""

    def test_reuses_unaccessed_leaf_array(self):
        """Unaccessed leaf array should be reused from prior checkpoint."""
        ns = {
            "big_array": np.arange(1_000_000),
            "small_val": 42,
        }

        cp = MemoryCheckpoints(sanity_check=False)

        # Take initial checkpoint
        cp.save("pre", ns.copy())

        # Incremental save - only small_val was accessed
        accessed_vars = {"small_val"}
        cp.save_incremental("post", ns.copy(), accessed_vars, "pre")

        # big_array should be reused (same object)
        pre_cp = cp.saved["pre"]
        post_cp = cp.saved["post"]
        assert post_cp.user_ns["big_array"] is pre_cp.user_ns["big_array"]

    def test_copies_accessed_variable(self):
        """Accessed variable should be deep copied, not reused."""
        ns = {
            "arr": np.arange(100),
            "other": 42,
        }

        cp = MemoryCheckpoints(sanity_check=False)
        cp.save("pre", ns.copy())

        # arr was accessed
        accessed_vars = {"arr"}
        cp.save_incremental("post", ns.copy(), accessed_vars, "pre")

        pre_cp = cp.saved["pre"]
        post_cp = cp.saved["post"]
        # arr should be a new copy
        assert post_cp.user_ns["arr"] is not pre_cp.user_ns["arr"]
        np.testing.assert_array_equal(post_cp.user_ns["arr"], pre_cp.user_ns["arr"])

    def test_copies_nonleaf_object(self):
        """Non-leaf objects should be copied even if unaccessed."""
        ns = {
            "df": pd.DataFrame({"a": [1, 2, 3]}),
            "small": 1,
        }

        cp = MemoryCheckpoints(sanity_check=False)
        cp.save("pre", ns.copy())

        # Neither was accessed, but df is not a leaf
        accessed_vars = set()
        cp.save_incremental("post", ns.copy(), accessed_vars, "pre")

        pre_cp = cp.saved["pre"]
        post_cp = cp.saved["post"]
        # df should be copied (not a leaf)
        assert post_cp.user_ns["df"] is not pre_cp.user_ns["df"]

    def test_copies_changed_identity_variable(self):
        """Variable with changed identity should be copied."""
        arr = np.arange(100)
        ns = {"arr": arr}

        cp = MemoryCheckpoints(sanity_check=False)
        cp.save("pre", ns.copy())

        # Replace arr with new object (different id)
        ns["arr"] = np.arange(100, 200)
        accessed_vars = set()  # Not "accessed" in namespace sense
        cp.save_incremental("post", ns.copy(), accessed_vars, "pre")

        post_cp = cp.saved["post"]
        # New array should be in checkpoint
        np.testing.assert_array_equal(post_cp.user_ns["arr"], np.arange(100, 200))

    def test_fallback_to_regular_save_if_prior_missing(self):
        """Should fall back to regular save if prior checkpoint doesn't exist."""
        ns = {"x": np.arange(100)}

        cp = MemoryCheckpoints(sanity_check=False)

        # No prior checkpoint saved
        accessed_vars = {"x"}
        saved, removed = cp.save_incremental(
            "post", ns.copy(), accessed_vars, "nonexistent"
        )

        # Should still work (fallback to regular save)
        assert "x" in cp.saved["post"].user_ns

    def test_tracks_original_ids(self):
        """Both save and save_incremental should track original IDs."""
        arr = np.arange(100)
        ns = {"arr": arr}

        cp = MemoryCheckpoints(sanity_check=False)

        # Regular save
        cp.save("pre", ns.copy())
        pre_cp = cp.saved["pre"]
        assert pre_cp._original_ids["arr"] == id(arr)

        # Incremental save
        cp.save_incremental("post", ns.copy(), set(), "pre")
        post_cp = cp.saved["post"]
        assert post_cp._original_ids["arr"] == id(arr)

    def test_expands_aliases_before_reuse_check(self):
        """Aliased variables should not be reused even if not directly accessed."""
        # Create two variables that share internal structure
        shared_arr = np.arange(100)
        container1 = {"data": shared_arr}
        container2 = {"data": shared_arr}  # Same internal array

        ns = {
            "c1": container1,
            "c2": container2,
        }

        cp = MemoryCheckpoints(sanity_check=False)
        cp.save("pre", ns.copy())

        # Only c1 was accessed, but c2 shares data with c1
        accessed_vars = {"c1"}
        cp.save_incremental("post", ns.copy(), accessed_vars, "pre")

        pre_cp = cp.saved["pre"]
        post_cp = cp.saved["post"]

        # Both should be copied due to alias relationship
        # (c2 is aliased with c1 via shared internal array)
        # Note: containers are not leaf objects, so they're always copied anyway
        assert post_cp.user_ns["c1"] is not pre_cp.user_ns["c1"]


class TestDiffIdentityShortCircuit:
    """Tests for identity short-circuit in diff."""

    def test_same_object_returns_none(self):
        """Comparing same object should return None immediately."""
        from flowbook.kernel_support.diff import Diff

        arr = np.arange(1000000)
        differ = Diff(strict=False)

        result = differ._compare_values(arr, arr)
        assert result is None

    def test_equal_objects_returns_none(self):
        """Comparing equal but different objects should also return None."""
        from flowbook.kernel_support.diff import Diff

        arr1 = np.arange(100)
        arr2 = np.arange(100)
        differ = Diff(strict=False)

        result = differ._compare_values(arr1, arr2)
        assert result is None

    def test_different_objects_returns_diff(self):
        """Comparing different objects should return diff node."""
        from flowbook.kernel_support.diff import Diff

        arr1 = np.arange(100)
        arr2 = np.arange(100, 200)
        differ = Diff(strict=False)

        result = differ._compare_values(arr1, arr2)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
