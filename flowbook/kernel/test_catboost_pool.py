"""
Tests for CatBoost Pool deepcopy handling.

CatBoost Pool objects explicitly block __deepcopy__ with a CatBoostError,
but our custom deepcopy implementation handles them via pool.slice() workaround.
"""

import numpy as np
import pytest

# Skip all tests if catboost is not installed
catboost = pytest.importorskip("catboost")
from catboost import Pool as CatBoostPool
from _catboost import _PoolBase

from flowbook.kernel.deepcopyable import check_deepcopyable
from flowbook.kernel.deepcopy import deepcopy


class TestCatBoostPoolDeepCopyable:
    """Tests that check_deepcopyable recognizes CatBoost Pool as copyable."""

    def test_pool_is_deepcopyable(self):
        """CatBoost Pool should be recognized as deepcopyable."""
        pool = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        assert check_deepcopyable(pool) is None

    def test_pool_with_cat_features_is_deepcopyable(self):
        """CatBoost Pool with categorical features should be deepcopyable."""
        pool = CatBoostPool(
            [[1, "a"], [2, "b"], [3, "a"]],
            label=[0, 1, 0],
            cat_features=[1],
        )
        assert check_deepcopyable(pool) is None

    def test_pool_with_feature_names_is_deepcopyable(self):
        """CatBoost Pool with feature names should be deepcopyable."""
        pool = CatBoostPool(
            [[1, 2], [3, 4]],
            label=[0, 1],
            feature_names=["feat1", "feat2"],
        )
        assert check_deepcopyable(pool) is None

    def test_pool_with_weights_is_deepcopyable(self):
        """CatBoost Pool with sample weights should be deepcopyable."""
        pool = CatBoostPool(
            [[1, 2], [3, 4]],
            label=[0, 1],
            weight=[1.0, 2.0],
        )
        assert check_deepcopyable(pool) is None

    def test_pool_instance_is_pool_base(self):
        """Verify Pool instances are also _PoolBase instances."""
        pool = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        assert isinstance(pool, _PoolBase)

    def test_pool_base_type_recognized(self):
        """_PoolBase type should be recognized in check_deepcopyable."""
        pool = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        # Check that the base class module/name detection works
        assert type(pool).__mro__[1].__name__ == "_PoolBase"
        assert type(pool).__mro__[1].__module__ == "_catboost"


class TestCatBoostPoolDeepCopy:
    """Tests that our custom deepcopy correctly copies CatBoost Pool objects."""

    def test_pool_deepcopy_creates_independent_copy(self):
        """Deepcopy should create an independent Pool copy."""
        original = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        copied = deepcopy(original)

        # Should be different objects
        assert copied is not original
        assert id(copied) != id(original)

    def test_pool_deepcopy_preserves_features(self):
        """Deepcopy should preserve feature values."""
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        original = CatBoostPool(data, label=[0, 1, 0])
        copied = deepcopy(original)

        # Get features from both pools
        original_features = np.array(original.get_features())
        copied_features = np.array(copied.get_features())

        np.testing.assert_array_equal(original_features, copied_features)

    def test_pool_deepcopy_preserves_labels(self):
        """Deepcopy should preserve label values."""
        labels = [0.0, 1.0, 0.0]
        original = CatBoostPool([[1, 2], [3, 4], [5, 6]], label=labels)
        copied = deepcopy(original)

        original_labels = original.get_label()
        copied_labels = copied.get_label()

        np.testing.assert_array_equal(original_labels, copied_labels)

    def test_pool_deepcopy_preserves_weights(self):
        """Deepcopy should preserve sample weights."""
        weights = [1.0, 2.0, 0.5]
        original = CatBoostPool(
            [[1, 2], [3, 4], [5, 6]],
            label=[0, 1, 0],
            weight=weights,
        )
        copied = deepcopy(original)

        original_weights = original.get_weight()
        copied_weights = copied.get_weight()

        np.testing.assert_array_equal(original_weights, copied_weights)

    def test_pool_deepcopy_preserves_cat_features(self):
        """Deepcopy should preserve categorical feature indices."""
        original = CatBoostPool(
            [[1, "a", 2], [2, "b", 3], [3, "a", 4]],
            label=[0, 1, 0],
            cat_features=[1],
        )
        copied = deepcopy(original)

        original_cat = original.get_cat_feature_indices()
        copied_cat = copied.get_cat_feature_indices()

        assert original_cat == copied_cat

    def test_pool_deepcopy_preserves_feature_names(self):
        """Deepcopy should preserve feature names."""
        feature_names = ["feat_a", "feat_b"]
        original = CatBoostPool(
            [[1, 2], [3, 4]],
            label=[0, 1],
            feature_names=feature_names,
        )
        copied = deepcopy(original)

        original_names = original.get_feature_names()
        copied_names = copied.get_feature_names()

        assert original_names == copied_names

    def test_pool_deepcopy_preserves_row_count(self):
        """Deepcopy should preserve the number of rows."""
        original = CatBoostPool([[1, 2], [3, 4], [5, 6]], label=[0, 1, 0])
        copied = deepcopy(original)

        assert copied.num_row() == original.num_row()

    def test_pool_deepcopy_preserves_col_count(self):
        """Deepcopy should preserve the number of columns."""
        original = CatBoostPool([[1, 2, 3], [4, 5, 6]], label=[0, 1])
        copied = deepcopy(original)

        assert copied.num_col() == original.num_col()

    def test_pool_deepcopy_memo_prevents_duplicate_copy(self):
        """Same Pool referenced multiple times should only be copied once."""
        pool = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        container = {"pool1": pool, "pool2": pool}

        memo = {}
        copied = deepcopy(container, memo)

        # Both references should point to the same copied Pool
        assert copied["pool1"] is copied["pool2"]
        assert copied["pool1"] is not pool

    def test_pool_in_list_deepcopy(self):
        """Pool inside a list should be properly deep copied."""
        pool = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        original_list = [pool, "other", 123]

        copied_list = deepcopy(original_list)

        assert copied_list[0] is not pool
        assert isinstance(copied_list[0], CatBoostPool)
        assert copied_list[1] == "other"
        assert copied_list[2] == 123

    def test_pool_in_dict_deepcopy(self):
        """Pool inside a dict should be properly deep copied."""
        pool = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        original_dict = {"pool": pool, "name": "test"}

        copied_dict = deepcopy(original_dict)

        assert copied_dict["pool"] is not pool
        assert isinstance(copied_dict["pool"], CatBoostPool)
        assert copied_dict["name"] == "test"


class TestCatBoostPoolStdlibDeepCopyFails:
    """Verify that stdlib deepcopy fails on CatBoost Pool (our handler is needed)."""

    def test_stdlib_deepcopy_raises_catboost_error(self):
        """Standard library deepcopy should raise CatBoostError on Pool."""
        import copy

        pool = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])

        with pytest.raises(catboost.CatBoostError, match="Can't deepcopy"):
            copy.deepcopy(pool)
