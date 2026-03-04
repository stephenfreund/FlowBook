"""Tests for sklearn StackingRegressor/Classifier checkpoint optimization (diff only).

NOTE: Unlike SHAP/TargetEncoder, Stacking estimators do NOT have a custom deepcopy
handler because fitted model objects are mutable (partial_fit, attribute modification).
We only optimize the diff comparison using identity checks.
"""

import pytest
import numpy as np

# Skip all tests if sklearn is not available
sklearn = pytest.importorskip("sklearn")
from sklearn.ensemble import (
    StackingRegressor,
    StackingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

from flowbook.kernel_support.deepcopy import (
    deepcopy as flowbook_deepcopy,
    _is_sklearn_stacking_estimator,
)
from flowbook.kernel_support.diff import (
    Diff,
    _is_sklearn_stacking_estimator as diff_is_sklearn_stacking_estimator,
    _register_stacking_dispatch_if_needed,
)
from flowbook.kernel_support.types import ValueComparison, CompoundDiff


@pytest.fixture
def regression_data():
    """Create sample regression data."""
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def classification_data():
    """Create sample classification data."""
    X, y = make_classification(n_samples=200, n_features=10, n_classes=3,
                               n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def unfitted_stacking_regressor():
    """Create an unfitted StackingRegressor."""
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=5, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=5, random_state=42)),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=2
    )


@pytest.fixture
def fitted_stacking_regressor(regression_data):
    """Create a fitted StackingRegressor."""
    X_train, X_test, y_train, y_test = regression_data
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=5, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=5, random_state=42)),
    ]
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=2
    )
    stacking.fit(X_train, y_train)
    return stacking, X_test


@pytest.fixture
def unfitted_stacking_classifier():
    """Create an unfitted StackingClassifier."""
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=5, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=5, random_state=42)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=2
    )


@pytest.fixture
def fitted_stacking_classifier(classification_data):
    """Create a fitted StackingClassifier."""
    X_train, X_test, y_train, y_test = classification_data
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=5, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=5, random_state=42)),
    ]
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=2
    )
    stacking.fit(X_train, y_train)
    return stacking, X_test


class TestStackingDetection:
    """Tests for Stacking estimator detection."""

    def test_detect_unfitted_regressor(self, unfitted_stacking_regressor):
        """Should detect unfitted StackingRegressor."""
        assert _is_sklearn_stacking_estimator(unfitted_stacking_regressor)
        assert diff_is_sklearn_stacking_estimator(unfitted_stacking_regressor)

    def test_detect_fitted_regressor(self, fitted_stacking_regressor):
        """Should detect fitted StackingRegressor."""
        stacking, _ = fitted_stacking_regressor
        assert _is_sklearn_stacking_estimator(stacking)
        assert diff_is_sklearn_stacking_estimator(stacking)

    def test_detect_unfitted_classifier(self, unfitted_stacking_classifier):
        """Should detect unfitted StackingClassifier."""
        assert _is_sklearn_stacking_estimator(unfitted_stacking_classifier)
        assert diff_is_sklearn_stacking_estimator(unfitted_stacking_classifier)

    def test_detect_fitted_classifier(self, fitted_stacking_classifier):
        """Should detect fitted StackingClassifier."""
        stacking, _ = fitted_stacking_classifier
        assert _is_sklearn_stacking_estimator(stacking)
        assert diff_is_sklearn_stacking_estimator(stacking)

    def test_non_stacking_not_detected(self):
        """Should not detect non-Stacking objects."""
        assert not _is_sklearn_stacking_estimator("string")
        assert not _is_sklearn_stacking_estimator([1, 2, 3])
        assert not _is_sklearn_stacking_estimator({'key': 'value'})
        assert not _is_sklearn_stacking_estimator(np.array([1, 2, 3]))

        assert not diff_is_sklearn_stacking_estimator("string")
        assert not diff_is_sklearn_stacking_estimator([1, 2, 3])

    def test_base_estimators_not_detected(self):
        """Should not detect base estimators as Stacking."""
        rf = RandomForestRegressor(n_estimators=5)
        gb = GradientBoostingRegressor(n_estimators=5)
        ridge = Ridge()

        assert not _is_sklearn_stacking_estimator(rf)
        assert not _is_sklearn_stacking_estimator(gb)
        assert not _is_sklearn_stacking_estimator(ridge)


class TestStackingDeepCopy:
    """Tests for Stacking estimator deepcopy (standard deepcopy, no sharing)."""

    def test_deepcopy_unfitted_regressor(self, unfitted_stacking_regressor):
        """Should deepcopy unfitted regressor correctly."""
        copy = flowbook_deepcopy(unfitted_stacking_regressor)

        # Different objects
        assert copy is not unfitted_stacking_regressor

        # Same number of estimators
        assert len(copy.estimators) == len(unfitted_stacking_regressor.estimators)

    def test_deepcopy_fitted_regressor(self, fitted_stacking_regressor):
        """Should deepcopy fitted regressor - creates independent copy."""
        stacking, X_test = fitted_stacking_regressor
        copy = flowbook_deepcopy(stacking)

        # Different objects (not shared for mutation safety)
        assert copy is not stacking

        # Fitted estimators are also different (deep copied)
        assert copy.estimators_ is not stacking.estimators_
        for orig_est, copy_est in zip(stacking.estimators_, copy.estimators_):
            assert copy_est is not orig_est

        # Final estimator is also different
        assert copy.final_estimator_ is not stacking.final_estimator_

    def test_deepcopy_fitted_same_predictions(self, fitted_stacking_regressor):
        """Copy should produce same predictions as original."""
        stacking, X_test = fitted_stacking_regressor
        copy = flowbook_deepcopy(stacking)

        # Both should produce same output
        np.testing.assert_array_almost_equal(
            stacking.predict(X_test),
            copy.predict(X_test)
        )

    def test_deepcopy_fitted_classifier(self, fitted_stacking_classifier):
        """Should deepcopy fitted classifier correctly."""
        stacking, X_test = fitted_stacking_classifier
        copy = flowbook_deepcopy(stacking)

        # Different objects
        assert copy is not stacking

        # Same predictions
        np.testing.assert_array_equal(
            stacking.predict(X_test),
            copy.predict(X_test)
        )

    def test_memo_sharing(self, fitted_stacking_regressor):
        """Same stacking referenced twice should share copy in memo."""
        stacking, _ = fitted_stacking_regressor
        data = {
            'stacking1': stacking,
            'stacking2': stacking  # Same reference
        }
        data_copy = flowbook_deepcopy(data)

        # Both should point to the same copy
        assert data_copy['stacking1'] is data_copy['stacking2']
        # But different from original
        assert data_copy['stacking1'] is not stacking

    def test_mutation_isolation(self, fitted_stacking_regressor, regression_data):
        """Mutations to copy should not affect original."""
        stacking, X_test = fitted_stacking_regressor
        X_train, _, y_train, _ = regression_data
        copy = flowbook_deepcopy(stacking)

        # Get original predictions
        original_preds = stacking.predict(X_test).copy()

        # Mutate the copy's final estimator (simulate partial refit)
        # Note: This modifies internal state
        copy.final_estimator_.coef_ = copy.final_estimator_.coef_ * 2

        # Original should be unchanged
        np.testing.assert_array_almost_equal(
            stacking.predict(X_test),
            original_preds
        )


class TestStackingDiff:
    """Tests for Stacking estimator diff handler."""

    def test_diff_same_reference(self, fitted_stacking_regressor):
        """Same reference should have no differences (O(1) identity check)."""
        stacking, _ = fitted_stacking_regressor

        diff = Diff()
        result = diff.diff({'stacking': stacking}, {'stacking': stacking})

        assert 'stacking' not in result.differences

    def test_diff_copied_estimators_different(self, fitted_stacking_regressor):
        """Copied estimators should be detected as different (different objects)."""
        stacking, _ = fitted_stacking_regressor
        copy = flowbook_deepcopy(stacking)

        diff = Diff()
        result = diff.diff({'stacking': stacking}, {'stacking': copy})

        # Since they're different objects, they should be detected as different
        assert 'stacking' in result.differences

    def test_diff_different_estimators(self, regression_data):
        """Different stacking estimators should be detected."""
        X_train, _, y_train, _ = regression_data

        # Create two different stacking models
        estimators1 = [
            ('rf', RandomForestRegressor(n_estimators=5, random_state=42)),
        ]
        stacking1 = StackingRegressor(
            estimators=estimators1,
            final_estimator=Ridge(),
            cv=2
        )
        stacking1.fit(X_train, y_train)

        estimators2 = [
            ('gb', GradientBoostingRegressor(n_estimators=5, random_state=42)),
        ]
        stacking2 = StackingRegressor(
            estimators=estimators2,
            final_estimator=Ridge(),
            cv=2
        )
        stacking2.fit(X_train, y_train)

        diff = Diff()
        result = diff.diff({'stacking': stacking1}, {'stacking': stacking2})

        assert 'stacking' in result.differences

    def test_diff_fitted_vs_unfitted(self, fitted_stacking_regressor, unfitted_stacking_regressor):
        """Fitted vs unfitted should be different."""
        stacking, _ = fitted_stacking_regressor

        diff = Diff()
        result = diff.diff({'stacking': stacking}, {'stacking': unfitted_stacking_regressor})

        assert 'stacking' in result.differences
        # Should indicate fitted status mismatch
        stacking_diff = result.differences['stacking']
        assert isinstance(stacking_diff, ValueComparison)
        assert 'fitted' in stacking_diff.message.lower()

    def test_diff_unfitted_same_reference(self):
        """Unfitted estimators with same reference should be equal."""
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=5, random_state=42)),
        ]
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(),
            cv=2
        )

        # Same reference - should be equal
        diff = Diff()
        result = diff.diff({'stacking': stacking}, {'stacking': stacking})

        assert 'stacking' not in result.differences

    def test_diff_unfitted_different_instances_same_config(self):
        """Unfitted estimators with same config but different instances are different.

        This is expected behavior - even if the configuration is identical,
        different object instances are considered different because:
        1. The estimator objects in the 'estimators' param are different instances
        2. We can't reliably determine semantic equality without deep comparison
        """
        estimators1 = [
            ('rf', RandomForestRegressor(n_estimators=5, random_state=42)),
        ]
        stacking1 = StackingRegressor(
            estimators=estimators1,
            final_estimator=Ridge(),
            cv=2
        )

        estimators2 = [
            ('rf', RandomForestRegressor(n_estimators=5, random_state=42)),
        ]
        stacking2 = StackingRegressor(
            estimators=estimators2,
            final_estimator=Ridge(),
            cv=2
        )

        diff = Diff()
        result = diff.diff({'stacking': stacking1}, {'stacking': stacking2})

        # Different instances are detected as different
        assert 'stacking' in result.differences

    def test_diff_unfitted_different_params(self):
        """Unfitted estimators with different params should be different."""
        estimators1 = [
            ('rf', RandomForestRegressor(n_estimators=5, random_state=42)),
        ]
        stacking1 = StackingRegressor(
            estimators=estimators1,
            final_estimator=Ridge(),
            cv=2
        )

        estimators2 = [
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),  # Different
        ]
        stacking2 = StackingRegressor(
            estimators=estimators2,
            final_estimator=Ridge(),
            cv=3  # Different
        )

        diff = Diff()
        result = diff.diff({'stacking': stacking1}, {'stacking': stacking2})

        assert 'stacking' in result.differences


class TestStackingCheckpointIntegration:
    """Integration tests for Stacking with checkpoint system."""

    def test_stacking_in_namespace(self, fitted_stacking_regressor, regression_data):
        """Stacking in namespace should deepcopy correctly."""
        stacking, X_test = fitted_stacking_regressor
        X_train, _, y_train, _ = regression_data
        namespace = {
            'stacking': stacking,
            'X_train': X_train,
            'y_train': y_train
        }

        namespace_copy = flowbook_deepcopy(namespace)

        # Stacking should be copied (not shared)
        assert namespace_copy['stacking'] is not stacking

        # But predictions should match
        np.testing.assert_array_almost_equal(
            stacking.predict(X_test),
            namespace_copy['stacking'].predict(X_test)
        )

        # Data should be copied
        assert namespace_copy['X_train'] is not X_train
        assert namespace_copy['y_train'] is not y_train

    def test_stacking_replacement_detected(self, regression_data):
        """Replacing stacking model should be detected in namespace diff."""
        X_train, X_test, y_train, y_test = regression_data

        # First checkpoint
        estimators1 = [
            ('rf', RandomForestRegressor(n_estimators=5, random_state=42)),
        ]
        stacking1 = StackingRegressor(
            estimators=estimators1,
            final_estimator=Ridge(),
            cv=2
        )
        stacking1.fit(X_train, y_train)
        ns1 = {'stacking': stacking1}
        ns1_copy = flowbook_deepcopy(ns1)

        # Second checkpoint - different model
        estimators2 = [
            ('gb', GradientBoostingRegressor(n_estimators=5, random_state=42)),
        ]
        stacking2 = StackingRegressor(
            estimators=estimators2,
            final_estimator=Ridge(),
            cv=2
        )
        stacking2.fit(X_train, y_train)
        ns2 = {'stacking': stacking2}

        diff = Diff()
        result = diff.diff(ns1_copy, ns2)

        # Should be detected as different
        assert 'stacking' in result.differences


class TestStackingEdgeCases:
    """Edge case tests for Stacking handlers."""

    def test_single_base_estimator(self, regression_data):
        """Should handle stacking with single base estimator."""
        X_train, X_test, y_train, y_test = regression_data

        estimators = [
            ('rf', RandomForestRegressor(n_estimators=5, random_state=42)),
        ]
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(),
            cv=2
        )
        stacking.fit(X_train, y_train)

        copy = flowbook_deepcopy(stacking)

        np.testing.assert_array_almost_equal(
            stacking.predict(X_test),
            copy.predict(X_test)
        )

    def test_many_base_estimators(self, regression_data):
        """Should handle stacking with many base estimators."""
        X_train, X_test, y_train, y_test = regression_data

        estimators = [
            ('rf1', RandomForestRegressor(n_estimators=3, random_state=42)),
            ('rf2', RandomForestRegressor(n_estimators=3, random_state=43)),
            ('gb1', GradientBoostingRegressor(n_estimators=3, random_state=42)),
            ('gb2', GradientBoostingRegressor(n_estimators=3, random_state=43)),
        ]
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(),
            cv=2
        )
        stacking.fit(X_train, y_train)

        copy = flowbook_deepcopy(stacking)

        np.testing.assert_array_almost_equal(
            stacking.predict(X_test),
            copy.predict(X_test)
        )

    def test_passthrough_enabled(self, regression_data):
        """Should handle stacking with passthrough=True."""
        X_train, X_test, y_train, y_test = regression_data

        estimators = [
            ('rf', RandomForestRegressor(n_estimators=5, random_state=42)),
        ]
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(),
            cv=2,
            passthrough=True
        )
        stacking.fit(X_train, y_train)

        copy = flowbook_deepcopy(stacking)

        np.testing.assert_array_almost_equal(
            stacking.predict(X_test),
            copy.predict(X_test)
        )

    def test_classifier_with_proba(self, classification_data):
        """Should handle classifier with predict_proba."""
        X_train, X_test, y_train, y_test = classification_data

        estimators = [
            ('rf', RandomForestClassifier(n_estimators=5, random_state=42)),
        ]
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=2
        )
        stacking.fit(X_train, y_train)

        copy = flowbook_deepcopy(stacking)

        # Check predictions
        np.testing.assert_array_equal(
            stacking.predict(X_test),
            copy.predict(X_test)
        )

        # Check probabilities
        np.testing.assert_array_almost_equal(
            stacking.predict_proba(X_test),
            copy.predict_proba(X_test)
        )

    def test_identity_comparison_performance(self, fitted_stacking_regressor):
        """Identity comparison should be fast O(1) for same reference."""
        stacking, _ = fitted_stacking_regressor

        # Same reference - should be instant
        diff = Diff()
        result = diff.diff({'stacking': stacking}, {'stacking': stacking})

        # No differences expected
        assert 'stacking' not in result.differences
