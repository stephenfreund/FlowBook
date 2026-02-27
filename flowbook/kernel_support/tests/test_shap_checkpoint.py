"""Tests for SHAP checkpoint optimization (deepcopy and diff handlers)."""

import pytest
import numpy as np

# Skip all tests if SHAP is not installed
shap = pytest.importorskip("shap")

from flowbook.kernel_support.deepcopy import (
    deepcopy as flowbook_deepcopy,
    _is_shap_explanation,
    _is_shap_tree_explainer,
    _register_shap_handlers_if_needed,
    reset_shap_deepcopy_handler,
)
from flowbook.kernel_support.diff import (
    Diff,
    _is_shap_explanation as diff_is_shap_explanation,
    _is_shap_tree_explainer as diff_is_shap_tree_explainer,
    _register_shap_dispatch_if_needed,
)
from flowbook.kernel_support.types import ValueComparison, CompoundDiff


@pytest.fixture(autouse=True)
def reset_handlers():
    """Reset the SHAP handlers before each test."""
    reset_shap_deepcopy_handler()
    yield
    reset_shap_deepcopy_handler()


@pytest.fixture
def sample_data():
    """Create sample data for SHAP explanations."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.1
    return X, y


@pytest.fixture
def tree_model(sample_data):
    """Create a fitted tree model for TreeExplainer."""
    sklearn = pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor

    X, y = sample_data
    model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=3)
    model.fit(X, y)
    return model, X


@pytest.fixture
def shap_explanation(tree_model):
    """Create a SHAP Explanation object."""
    model, X = tree_model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values, explainer, X


class TestSHAPDetection:
    """Tests for SHAP object detection."""

    def test_detect_explanation(self, shap_explanation):
        """Should detect SHAP Explanation."""
        shap_values, _, _ = shap_explanation
        assert _is_shap_explanation(shap_values)
        assert diff_is_shap_explanation(shap_values)

    def test_detect_tree_explainer(self, shap_explanation):
        """Should detect SHAP TreeExplainer."""
        _, explainer, _ = shap_explanation
        assert _is_shap_tree_explainer(explainer)
        assert diff_is_shap_tree_explainer(explainer)

    def test_non_shap_not_detected(self):
        """Should not detect non-SHAP objects."""
        assert not _is_shap_explanation("string")
        assert not _is_shap_explanation([1, 2, 3])
        assert not _is_shap_explanation({'key': 'value'})
        assert not _is_shap_explanation(np.array([1, 2, 3]))

        assert not diff_is_shap_explanation("string")
        assert not diff_is_shap_explanation([1, 2, 3])

        assert not _is_shap_tree_explainer("string")
        assert not _is_shap_tree_explainer([1, 2, 3])


class TestSHAPExplanationDeepCopy:
    """Tests for SHAP Explanation deepcopy handler."""

    def test_deepcopy_explanation(self, shap_explanation):
        """Should deepcopy Explanation correctly."""
        shap_values, _, _ = shap_explanation
        copy = flowbook_deepcopy(shap_values)

        # Different wrapper objects
        assert copy is not shap_values

        # SHARED numpy arrays (immutable, so safe to share)
        assert copy.values is shap_values.values
        assert copy.base_values is shap_values.base_values
        assert copy.data is shap_values.data

    def test_deepcopy_explanation_same_shape(self, shap_explanation):
        """Copy should have same shape as original."""
        shap_values, _, _ = shap_explanation
        copy = flowbook_deepcopy(shap_values)

        assert copy.values.shape == shap_values.values.shape
        if hasattr(shap_values, 'base_values') and shap_values.base_values is not None:
            assert copy.base_values.shape == shap_values.base_values.shape

    def test_deepcopy_explanation_same_values(self, shap_explanation):
        """Copy should have same values as original."""
        shap_values, _, _ = shap_explanation
        copy = flowbook_deepcopy(shap_values)

        np.testing.assert_array_equal(copy.values, shap_values.values)

    def test_memo_sharing(self, shap_explanation):
        """Same Explanation referenced twice should share copy in memo."""
        shap_values, _, _ = shap_explanation
        data = {
            'shap1': shap_values,
            'shap2': shap_values  # Same reference
        }
        data_copy = flowbook_deepcopy(data)

        # Both should point to the same copy
        assert data_copy['shap1'] is data_copy['shap2']
        # But different from original
        assert data_copy['shap1'] is not shap_values

    def test_lazy_registration(self, shap_explanation):
        """Handler should be registered lazily on first explanation."""
        shap_values, _, _ = shap_explanation

        # Reset to unregistered state
        reset_shap_deepcopy_handler()

        # Deepcopy should work (triggers registration)
        copy = flowbook_deepcopy(shap_values)

        # Verify copy is valid
        assert copy is not shap_values
        np.testing.assert_array_equal(copy.values, shap_values.values)


class TestSHAPTreeExplainerDeepCopy:
    """Tests for SHAP TreeExplainer deepcopy handler."""

    def test_deepcopy_explainer(self, shap_explanation):
        """Should return the same object (explainer is immutable)."""
        _, explainer, _ = shap_explanation
        copy = flowbook_deepcopy(explainer)

        # Should be SAME object (shared, since immutable)
        assert copy is explainer

    def test_explainer_usable_after_copy(self, shap_explanation, tree_model):
        """Copied explainer should still be usable."""
        _, explainer, X = shap_explanation
        copy = flowbook_deepcopy(explainer)

        # Should be able to compute new explanations
        new_values = copy(X[:5])
        assert new_values.values.shape[0] == 5


class TestSHAPExplanationDiff:
    """Tests for SHAP Explanation diff handler."""

    def test_diff_equal_explanations(self, shap_explanation):
        """Equal explanations should have no differences."""
        shap_values, _, _ = shap_explanation
        copy = flowbook_deepcopy(shap_values)

        diff = Diff()
        result = diff.diff({'shap': shap_values}, {'shap': copy})

        assert 'shap' not in result.differences

    def test_diff_different_explanations(self, tree_model):
        """Different explanations should be detected."""
        model, X = tree_model
        explainer = shap.TreeExplainer(model)

        # Create two different explanations
        values1 = explainer(X[:50])
        values2 = explainer(X[50:])

        diff = Diff()
        result = diff.diff({'shap': values1}, {'shap': values2})

        assert 'shap' in result.differences

    def test_diff_pointer_comparison_fast(self, shap_explanation):
        """Pointer comparison should be O(1) when arrays are shared."""
        shap_values, _, _ = shap_explanation
        copy = flowbook_deepcopy(shap_values)

        # Verify arrays are shared
        assert shap_values.values is copy.values

        # Diff should be fast (pointer comparison)
        diff = Diff()
        result = diff.diff({'shap': shap_values}, {'shap': copy})

        assert 'shap' not in result.differences


class TestSHAPTreeExplainerDiff:
    """Tests for SHAP TreeExplainer diff handler."""

    def test_diff_equal_explainers(self, shap_explanation):
        """Same explainer should have no differences."""
        _, explainer, _ = shap_explanation
        copy = flowbook_deepcopy(explainer)

        diff = Diff()
        result = diff.diff({'explainer': explainer}, {'explainer': copy})

        assert 'explainer' not in result.differences

    def test_diff_different_explainers(self, tree_model, sample_data):
        """Different explainers should be detected."""
        sklearn = pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestRegressor

        model1, X = tree_model
        X2, y2 = sample_data

        # Create a different model
        model2 = RandomForestRegressor(n_estimators=5, random_state=999, max_depth=2)
        model2.fit(X2, y2)

        explainer1 = shap.TreeExplainer(model1)
        explainer2 = shap.TreeExplainer(model2)

        diff = Diff()
        result = diff.diff({'explainer': explainer1}, {'explainer': explainer2})

        assert 'explainer' in result.differences


class TestSHAPCheckpointIntegration:
    """Integration tests for SHAP with checkpoint system."""

    def test_explanation_in_namespace(self, shap_explanation):
        """Explanation in namespace should deepcopy correctly."""
        shap_values, explainer, X = shap_explanation
        namespace = {
            'shap_values': shap_values,
            'explainer': explainer,
            'X': X,
        }

        namespace_copy = flowbook_deepcopy(namespace)

        # Explanation should be copied (different wrapper, shared arrays)
        assert namespace_copy['shap_values'] is not shap_values
        assert namespace_copy['shap_values'].values is shap_values.values

        # Explainer should be shared (immutable)
        assert namespace_copy['explainer'] is explainer

        # Data should be copied
        assert namespace_copy['X'] is not X

    def test_explanation_diff_in_namespace(self, tree_model):
        """Explanation changes should be detected in namespace diff."""
        model, X = tree_model
        explainer = shap.TreeExplainer(model)

        # First checkpoint
        values1 = explainer(X[:50])
        ns1 = {'shap_values': values1, 'X': X[:50]}
        ns1_copy = flowbook_deepcopy(ns1)

        # Second checkpoint with different data
        values2 = explainer(X[50:])
        ns2 = {'shap_values': values2, 'X': X[50:]}

        diff = Diff()
        result = diff.diff(ns1_copy, ns2)

        # SHAP values should be detected as different
        assert 'shap_values' in result.differences
        # X should also be different
        assert 'X' in result.differences


class TestSHAPEdgeCases:
    """Edge case tests for SHAP handlers."""

    def test_explanation_with_feature_names(self, tree_model):
        """Should handle explanations with feature names."""
        model, X = tree_model
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        shap_values.feature_names = feature_names

        copy = flowbook_deepcopy(shap_values)

        assert copy.feature_names == feature_names
        # Feature names list should be a copy (mutable)
        assert copy.feature_names is not shap_values.feature_names

    def test_explanation_with_data(self, tree_model):
        """Should handle explanations with data attribute."""
        model, X = tree_model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)

        copy = flowbook_deepcopy(shap_values)

        # Data array should be SHARED (immutable)
        if hasattr(shap_values, 'data') and shap_values.data is not None:
            assert copy.data is shap_values.data

    def test_small_explanation(self, tree_model):
        """Should handle very small explanations."""
        model, X = tree_model
        explainer = shap.TreeExplainer(model)

        # Single sample explanation
        shap_values = explainer(X[:1])
        copy = flowbook_deepcopy(shap_values)

        assert copy is not shap_values
        np.testing.assert_array_equal(copy.values, shap_values.values)
