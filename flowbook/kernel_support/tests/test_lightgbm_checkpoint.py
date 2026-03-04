"""Tests for LightGBM checkpoint optimization (deepcopy and diff handlers)."""

import pytest
import numpy as np

# Skip all tests if LightGBM is not installed
lgb = pytest.importorskip("lightgbm")

from flowbook.kernel_support.deepcopy import (
    deepcopy as flowbook_deepcopy,
    _is_lightgbm_model,
    _register_lightgbm_handlers_if_needed,
    reset_lightgbm_deepcopy_handler,
)
from flowbook.kernel_support.diff import (
    Diff,
    _is_lightgbm_model as diff_is_lightgbm_model,
    _register_lightgbm_dispatch_if_needed,
)
from flowbook.kernel_support.types import ValueComparison, CompoundDiff


@pytest.fixture(autouse=True)
def reset_handlers():
    """Reset the LightGBM handlers before each test."""
    reset_lightgbm_deepcopy_handler()
    yield
    reset_lightgbm_deepcopy_handler()


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.1
    return X, y


@pytest.fixture
def classification_data():
    """Create sample classification data."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def unfitted_regressor():
    """Create an unfitted LGBMRegressor."""
    return lgb.LGBMRegressor(n_estimators=10, verbose=-1)


@pytest.fixture
def fitted_regressor(sample_data):
    """Create a fitted LGBMRegressor."""
    X, y = sample_data
    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    model.fit(X, y)
    return model, X


@pytest.fixture
def fitted_classifier(classification_data):
    """Create a fitted LGBMClassifier."""
    X, y = classification_data
    model = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
    model.fit(X, y)
    return model, X


class TestLightGBMDetection:
    """Tests for LightGBM model detection."""

    def test_detect_unfitted_regressor(self, unfitted_regressor):
        """Should detect unfitted LGBMRegressor."""
        assert _is_lightgbm_model(unfitted_regressor)
        assert diff_is_lightgbm_model(unfitted_regressor)

    def test_detect_fitted_regressor(self, fitted_regressor):
        """Should detect fitted LGBMRegressor."""
        model, _ = fitted_regressor
        assert _is_lightgbm_model(model)
        assert diff_is_lightgbm_model(model)

    def test_detect_classifier(self, fitted_classifier):
        """Should detect LGBMClassifier."""
        model, _ = fitted_classifier
        assert _is_lightgbm_model(model)
        assert diff_is_lightgbm_model(model)

    def test_detect_ranker(self):
        """Should detect LGBMRanker."""
        model = lgb.LGBMRanker(n_estimators=10, verbose=-1)
        assert _is_lightgbm_model(model)
        assert diff_is_lightgbm_model(model)

    def test_non_lightgbm_not_detected(self):
        """Should not detect non-LightGBM objects."""
        assert not _is_lightgbm_model("string")
        assert not _is_lightgbm_model([1, 2, 3])
        assert not _is_lightgbm_model({'key': 'value'})
        assert not _is_lightgbm_model(np.array([1, 2, 3]))

        assert not diff_is_lightgbm_model("string")
        assert not diff_is_lightgbm_model([1, 2, 3])


class TestLightGBMDeepCopy:
    """Tests for LightGBM deepcopy handler."""

    def test_deepcopy_unfitted_regressor(self, unfitted_regressor):
        """Should deepcopy unfitted model correctly."""
        copy = flowbook_deepcopy(unfitted_regressor)

        # Different objects
        assert copy is not unfitted_regressor

        # Same parameters
        assert copy.get_params() == unfitted_regressor.get_params()

    def test_deepcopy_fitted_regressor(self, fitted_regressor):
        """Should deepcopy fitted model with same predictions."""
        model, X = fitted_regressor
        copy = flowbook_deepcopy(model)

        # Wrapper objects are different
        assert copy is not model

        # Booster is SHARED (immutable, so safe to share)
        assert copy.booster_ is model.booster_

        # Same predictions
        np.testing.assert_array_almost_equal(
            model.predict(X),
            copy.predict(X),
            decimal=10
        )

    def test_deepcopy_fitted_classifier(self, fitted_classifier):
        """Should deepcopy fitted classifier with same predictions."""
        model, X = fitted_classifier
        copy = flowbook_deepcopy(model)

        # Wrapper objects are different
        assert copy is not model

        # Booster is SHARED (immutable, so safe to share)
        assert copy.booster_ is model.booster_

        # Same predictions
        np.testing.assert_array_equal(
            model.predict(X),
            copy.predict(X)
        )

        # Same probability predictions
        np.testing.assert_array_almost_equal(
            model.predict_proba(X),
            copy.predict_proba(X),
            decimal=10
        )

    def test_deepcopy_independence(self, fitted_regressor):
        """Copy wrapper should be independent of original wrapper."""
        model, X = fitted_regressor
        copy = flowbook_deepcopy(model)

        # Get predictions before any modification
        orig_pred = model.predict(X).copy()
        copy_pred = copy.predict(X).copy()

        # Both should produce same predictions
        np.testing.assert_array_almost_equal(orig_pred, copy_pred, decimal=10)

        # Wrapper objects are different (allows independent __dict__ modification)
        assert id(copy) != id(model)

        # Booster is SHARED - this is the optimization!
        # Since the booster is immutable after fit(), sharing is safe
        assert id(copy.booster_) == id(model.booster_)

    def test_deepcopy_preserves_sklearn_attrs(self, fitted_regressor):
        """Should preserve sklearn fitted attributes."""
        model, X = fitted_regressor
        copy = flowbook_deepcopy(model)

        # Check common sklearn attributes
        if hasattr(model, 'n_features_in_'):
            assert copy.n_features_in_ == model.n_features_in_

        if hasattr(model, 'feature_name_'):
            assert copy.feature_name_ == model.feature_name_

        if hasattr(model, 'best_iteration_'):
            assert copy.best_iteration_ == model.best_iteration_

    def test_deepcopy_classifier_preserves_classes(self, fitted_classifier):
        """Should preserve classes_ attribute for classifier."""
        model, _ = fitted_classifier
        copy = flowbook_deepcopy(model)

        if hasattr(model, 'classes_'):
            np.testing.assert_array_equal(copy.classes_, model.classes_)

    def test_memo_sharing(self, fitted_regressor):
        """Same model referenced twice should share copy in memo."""
        model, _ = fitted_regressor
        data = {
            'model1': model,
            'model2': model  # Same reference
        }
        data_copy = flowbook_deepcopy(data)

        # Both should point to the same copy
        assert data_copy['model1'] is data_copy['model2']
        # But different from original
        assert data_copy['model1'] is not model

    def test_lazy_registration(self, fitted_regressor):
        """Handler should be registered lazily on first model."""
        model, X = fitted_regressor

        # Reset to unregistered state
        reset_lightgbm_deepcopy_handler()

        # Deepcopy should work (triggers registration)
        copy = flowbook_deepcopy(model)

        # Verify copy works
        np.testing.assert_array_almost_equal(
            model.predict(X),
            copy.predict(X),
            decimal=10
        )


class TestLightGBMDiff:
    """Tests for LightGBM diff handler."""

    def test_diff_equal_models(self, fitted_regressor):
        """Equal models should have no differences."""
        model, _ = fitted_regressor
        copy = flowbook_deepcopy(model)

        diff = Diff()
        result = diff.diff({'model': model}, {'model': copy})

        assert 'model' not in result.differences

    def test_diff_different_models(self, sample_data):
        """Different models should be detected."""
        X, y = sample_data

        model1 = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
        model1.fit(X, y)

        # Train with different random seed
        model2 = lgb.LGBMRegressor(n_estimators=10, verbose=-1, random_state=999)
        model2.fit(X, y)

        diff = Diff()
        result = diff.diff({'model': model1}, {'model': model2})

        assert 'model' in result.differences

    def test_diff_unfitted_equal_params(self):
        """Unfitted models with same params should be equal."""
        model1 = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
        model2 = lgb.LGBMRegressor(n_estimators=10, verbose=-1)

        diff = Diff()
        result = diff.diff({'model': model1}, {'model': model2})

        assert 'model' not in result.differences

    def test_diff_unfitted_different_params(self):
        """Unfitted models with different params should be different."""
        model1 = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
        model2 = lgb.LGBMRegressor(n_estimators=20, verbose=-1)

        diff = Diff()
        result = diff.diff({'model': model1}, {'model': model2})

        assert 'model' in result.differences

    def test_diff_fitted_vs_unfitted(self, fitted_regressor, unfitted_regressor):
        """Fitted vs unfitted models should be different."""
        model, _ = fitted_regressor

        diff = Diff()
        result = diff.diff({'model': model}, {'model': unfitted_regressor})

        assert 'model' in result.differences
        # Should indicate fitted status mismatch
        model_diff = result.differences['model']
        assert isinstance(model_diff, ValueComparison)
        assert 'fitted' in model_diff.message.lower()

    def test_diff_type_mismatch(self, fitted_regressor, fitted_classifier):
        """Different model types should be different."""
        reg_model, _ = fitted_regressor
        clf_model, _ = fitted_classifier

        diff = Diff()
        result = diff.diff({'model': reg_model}, {'model': clf_model})

        assert 'model' in result.differences
        model_diff = result.differences['model']
        assert isinstance(model_diff, ValueComparison)
        assert 'type' in model_diff.message.lower() or 'mismatch' in model_diff.message.lower()


class TestLightGBMCheckpointIntegration:
    """Integration tests for LightGBM with checkpoint system."""

    def test_model_in_namespace(self, fitted_regressor):
        """Model in namespace should deepcopy correctly."""
        model, X = fitted_regressor
        namespace = {
            'model': model,
            'X': X,
            'predictions': model.predict(X)
        }

        namespace_copy = flowbook_deepcopy(namespace)

        # Model should be copied
        assert namespace_copy['model'] is not model
        # Data should be copied
        assert namespace_copy['X'] is not X
        # Predictions should match
        np.testing.assert_array_almost_equal(
            namespace_copy['predictions'],
            namespace['predictions'],
            decimal=10
        )

    def test_model_diff_in_namespace(self, sample_data):
        """Model changes should be detected in namespace diff."""
        X, y = sample_data

        # First checkpoint
        model = lgb.LGBMRegressor(n_estimators=5, verbose=-1)
        model.fit(X, y)
        ns1 = {'model': model, 'X': X}
        ns1_copy = flowbook_deepcopy(ns1)

        # Second checkpoint after more training
        model2 = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
        model2.fit(X, y)
        ns2 = {'model': model2, 'X': X}

        diff = Diff()
        result = diff.diff(ns1_copy, ns2)

        # Model should be detected as different
        assert 'model' in result.differences
        # X should be unchanged
        assert 'X' not in result.differences


class TestLightGBMEdgeCases:
    """Edge case tests for LightGBM handlers."""

    def test_model_with_custom_params(self, sample_data):
        """Should handle models with many custom parameters."""
        X, y = sample_data
        model = lgb.LGBMRegressor(
            n_estimators=10,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_weight=1,
            verbose=-1
        )
        model.fit(X, y)

        copy = flowbook_deepcopy(model)

        # Should preserve all parameters
        assert copy.get_params() == model.get_params()

        # Should produce same predictions
        np.testing.assert_array_almost_equal(
            model.predict(X),
            copy.predict(X),
            decimal=10
        )

    def test_model_with_categorical_features(self, sample_data):
        """Should handle models trained with categorical features."""
        X, y = sample_data
        # Create categorical column
        X_cat = np.column_stack([X, np.random.randint(0, 5, len(X))])

        model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
        model.fit(X_cat, y, categorical_feature=[5])

        copy = flowbook_deepcopy(model)

        np.testing.assert_array_almost_equal(
            model.predict(X_cat),
            copy.predict(X_cat),
            decimal=10
        )

    def test_model_with_eval_result(self, sample_data):
        """Should handle models with evaluation results."""
        X, y = sample_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        copy = flowbook_deepcopy(model)

        # Should preserve best iteration if available
        if hasattr(model, 'best_iteration_'):
            assert copy.best_iteration_ == model.best_iteration_

        # Should produce same predictions
        np.testing.assert_array_almost_equal(
            model.predict(X_train),
            copy.predict(X_train),
            decimal=10
        )

    def test_empty_model_params(self):
        """Should handle model with default params only."""
        model = lgb.LGBMRegressor()
        copy = flowbook_deepcopy(model)

        assert copy is not model
        # Default params should match
        orig_params = model.get_params()
        copy_params = copy.get_params()
        for key in orig_params:
            if key != 'verbose':  # verbose can differ
                assert copy_params[key] == orig_params[key], f"Param {key} differs"
