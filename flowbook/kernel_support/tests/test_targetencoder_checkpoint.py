"""Tests for sklearn TargetEncoder checkpoint optimization (deepcopy and diff handlers)."""

import pytest
import numpy as np
import pandas as pd

# Skip all tests if sklearn TargetEncoder is not available
sklearn = pytest.importorskip("sklearn")
try:
    from sklearn.preprocessing import TargetEncoder
except ImportError:
    pytest.skip("TargetEncoder not available (requires sklearn >= 1.3)", allow_module_level=True)

from flowbook.kernel_support.deepcopy import (
    deepcopy as flowbook_deepcopy,
    _is_sklearn_target_encoder,
    _register_target_encoder_handlers_if_needed,
    reset_target_encoder_deepcopy_handler,
)
from flowbook.kernel_support.diff import (
    Diff,
    _is_sklearn_target_encoder as diff_is_sklearn_target_encoder,
    _register_target_encoder_dispatch_if_needed,
)
from flowbook.kernel_support.types import ValueComparison, CompoundDiff


@pytest.fixture(autouse=True)
def reset_handlers():
    """Reset the TargetEncoder handlers before each test."""
    reset_target_encoder_deepcopy_handler()
    yield
    reset_target_encoder_deepcopy_handler()


@pytest.fixture
def sample_data():
    """Create sample data for TargetEncoder."""
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame({
        'cat1': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'num': np.random.randn(n_samples)
    })
    y = np.random.randn(n_samples)
    return X, y


@pytest.fixture
def unfitted_encoder():
    """Create an unfitted TargetEncoder."""
    return TargetEncoder(smooth='auto', target_type='continuous')


@pytest.fixture
def fitted_encoder(sample_data):
    """Create a fitted TargetEncoder."""
    X, y = sample_data
    encoder = TargetEncoder(smooth='auto', target_type='continuous')
    encoder.fit(X[['cat1', 'cat2']], y)
    return encoder, X[['cat1', 'cat2']]


class TestTargetEncoderDetection:
    """Tests for TargetEncoder detection."""

    def test_detect_unfitted_encoder(self, unfitted_encoder):
        """Should detect unfitted TargetEncoder."""
        assert _is_sklearn_target_encoder(unfitted_encoder)
        assert diff_is_sklearn_target_encoder(unfitted_encoder)

    def test_detect_fitted_encoder(self, fitted_encoder):
        """Should detect fitted TargetEncoder."""
        encoder, _ = fitted_encoder
        assert _is_sklearn_target_encoder(encoder)
        assert diff_is_sklearn_target_encoder(encoder)

    def test_non_target_encoder_not_detected(self):
        """Should not detect non-TargetEncoder objects."""
        assert not _is_sklearn_target_encoder("string")
        assert not _is_sklearn_target_encoder([1, 2, 3])
        assert not _is_sklearn_target_encoder({'key': 'value'})
        assert not _is_sklearn_target_encoder(np.array([1, 2, 3]))

        assert not diff_is_sklearn_target_encoder("string")
        assert not diff_is_sklearn_target_encoder([1, 2, 3])

    def test_other_sklearn_encoders_not_detected(self):
        """Should not detect other sklearn encoders as TargetEncoder."""
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        le = LabelEncoder()
        ohe = OneHotEncoder()

        assert not _is_sklearn_target_encoder(le)
        assert not _is_sklearn_target_encoder(ohe)


class TestTargetEncoderDeepCopy:
    """Tests for TargetEncoder deepcopy handler."""

    def test_deepcopy_unfitted_encoder(self, unfitted_encoder):
        """Should deepcopy unfitted encoder correctly."""
        copy = flowbook_deepcopy(unfitted_encoder)

        # Different objects
        assert copy is not unfitted_encoder

        # Same parameters
        assert copy.get_params() == unfitted_encoder.get_params()

    def test_deepcopy_fitted_encoder(self, fitted_encoder):
        """Should deepcopy fitted encoder with shared arrays."""
        encoder, X = fitted_encoder
        copy = flowbook_deepcopy(encoder)

        # Wrapper objects are different
        assert copy is not encoder

        # Fitted arrays are SHARED (immutable, so safe to share)
        assert copy.encodings_ is encoder.encodings_
        assert copy.categories_ is encoder.categories_
        assert copy.target_mean_ is encoder.target_mean_

    def test_deepcopy_fitted_same_transform(self, fitted_encoder):
        """Copy should produce same transform results as original."""
        encoder, X = fitted_encoder
        copy = flowbook_deepcopy(encoder)

        # Both should produce same output
        np.testing.assert_array_equal(
            encoder.transform(X),
            copy.transform(X)
        )

    def test_deepcopy_preserves_fitted_attrs(self, fitted_encoder):
        """Should preserve fitted attributes."""
        encoder, X = fitted_encoder
        copy = flowbook_deepcopy(encoder)

        # Check common fitted attributes
        if hasattr(encoder, 'n_features_in_'):
            assert copy.n_features_in_ == encoder.n_features_in_

        if hasattr(encoder, 'feature_names_in_'):
            np.testing.assert_array_equal(
                copy.feature_names_in_, encoder.feature_names_in_
            )

    def test_memo_sharing(self, fitted_encoder):
        """Same encoder referenced twice should share copy in memo."""
        encoder, _ = fitted_encoder
        data = {
            'encoder1': encoder,
            'encoder2': encoder  # Same reference
        }
        data_copy = flowbook_deepcopy(data)

        # Both should point to the same copy
        assert data_copy['encoder1'] is data_copy['encoder2']
        # But different from original
        assert data_copy['encoder1'] is not encoder

    def test_lazy_registration(self, fitted_encoder):
        """Handler should be registered lazily on first encoder."""
        encoder, X = fitted_encoder

        # Reset to unregistered state
        reset_target_encoder_deepcopy_handler()

        # Deepcopy should work (triggers registration)
        copy = flowbook_deepcopy(encoder)

        # Verify copy works
        np.testing.assert_array_equal(
            encoder.transform(X),
            copy.transform(X)
        )


class TestTargetEncoderDiff:
    """Tests for TargetEncoder diff handler."""

    def test_diff_equal_encoders(self, fitted_encoder):
        """Equal encoders should have no differences."""
        encoder, _ = fitted_encoder
        copy = flowbook_deepcopy(encoder)

        diff = Diff()
        result = diff.diff({'encoder': encoder}, {'encoder': copy})

        assert 'encoder' not in result.differences

    def test_diff_different_encoders(self, sample_data):
        """Different encoders should be detected."""
        X, y = sample_data

        encoder1 = TargetEncoder(smooth='auto', target_type='continuous')
        encoder1.fit(X[['cat1']], y)

        encoder2 = TargetEncoder(smooth='auto', target_type='continuous')
        encoder2.fit(X[['cat2']], y)

        diff = Diff()
        result = diff.diff({'encoder': encoder1}, {'encoder': encoder2})

        assert 'encoder' in result.differences

    def test_diff_unfitted_equal_params(self):
        """Unfitted encoders with same params should be equal."""
        encoder1 = TargetEncoder(smooth='auto', target_type='continuous')
        encoder2 = TargetEncoder(smooth='auto', target_type='continuous')

        diff = Diff()
        result = diff.diff({'encoder': encoder1}, {'encoder': encoder2})

        assert 'encoder' not in result.differences

    def test_diff_unfitted_different_params(self):
        """Unfitted encoders with different params should be different."""
        encoder1 = TargetEncoder(smooth=0.5, target_type='continuous')
        encoder2 = TargetEncoder(smooth=1.0, target_type='continuous')

        diff = Diff()
        result = diff.diff({'encoder': encoder1}, {'encoder': encoder2})

        assert 'encoder' in result.differences

    def test_diff_fitted_vs_unfitted(self, fitted_encoder, unfitted_encoder):
        """Fitted vs unfitted encoders should be different."""
        encoder, _ = fitted_encoder

        diff = Diff()
        result = diff.diff({'encoder': encoder}, {'encoder': unfitted_encoder})

        assert 'encoder' in result.differences
        # Should indicate fitted status mismatch
        encoder_diff = result.differences['encoder']
        assert isinstance(encoder_diff, ValueComparison)
        assert 'fitted' in encoder_diff.message.lower()

    def test_diff_pointer_comparison_fast(self, fitted_encoder):
        """Pointer comparison should be O(1) when arrays are shared."""
        encoder, _ = fitted_encoder
        copy = flowbook_deepcopy(encoder)

        # Verify arrays are shared
        assert encoder.encodings_ is copy.encodings_

        # Diff should be fast (pointer comparison)
        diff = Diff()
        result = diff.diff({'encoder': encoder}, {'encoder': copy})

        assert 'encoder' not in result.differences


class TestTargetEncoderCheckpointIntegration:
    """Integration tests for TargetEncoder with checkpoint system."""

    def test_encoder_in_namespace(self, fitted_encoder, sample_data):
        """Encoder in namespace should deepcopy correctly."""
        encoder, X = fitted_encoder
        _, y = sample_data
        namespace = {
            'encoder': encoder,
            'X': X,
            'y': y
        }

        namespace_copy = flowbook_deepcopy(namespace)

        # Encoder should be copied (different wrapper, shared arrays)
        assert namespace_copy['encoder'] is not encoder
        assert namespace_copy['encoder'].encodings_ is encoder.encodings_

        # Data should be copied
        assert namespace_copy['X'] is not X
        assert namespace_copy['y'] is not y

    def test_encoder_diff_in_namespace(self, sample_data):
        """Encoder changes should be detected in namespace diff."""
        X, y = sample_data

        # First checkpoint - fit on partial data
        encoder1 = TargetEncoder(smooth='auto', target_type='continuous')
        encoder1.fit(X[['cat1']].iloc[:50], y[:50])
        ns1 = {'encoder': encoder1, 'X': X[['cat1']]}
        ns1_copy = flowbook_deepcopy(ns1)

        # Second checkpoint - fit on different data
        encoder2 = TargetEncoder(smooth='auto', target_type='continuous')
        encoder2.fit(X[['cat1']].iloc[50:], y[50:])
        ns2 = {'encoder': encoder2, 'X': X[['cat1']]}

        diff = Diff()
        result = diff.diff(ns1_copy, ns2)

        # Encoder should be detected as different
        assert 'encoder' in result.differences


class TestTargetEncoderEdgeCases:
    """Edge case tests for TargetEncoder handlers."""

    def test_encoder_with_many_categories(self, sample_data):
        """Should handle encoders with many categories."""
        X, y = sample_data

        # Create data with many categories
        np.random.seed(42)
        X_many = pd.DataFrame({
            'cat': [f'cat_{i}' for i in np.random.randint(0, 50, 100)]
        })

        encoder = TargetEncoder(smooth='auto', target_type='continuous')
        encoder.fit(X_many, y)

        copy = flowbook_deepcopy(encoder)

        # Should work correctly
        np.testing.assert_array_equal(
            encoder.transform(X_many),
            copy.transform(X_many)
        )

    def test_encoder_with_infrequent_categories(self, sample_data):
        """Should handle encoders with infrequent category handling."""
        X, y = sample_data

        # Create data with infrequent categories
        np.random.seed(42)
        cats = ['common'] * 90 + [f'rare_{i}' for i in range(10)]
        X_rare = pd.DataFrame({'cat': cats})

        # Note: TargetEncoder doesn't have min_frequency parameter,
        # but it may handle rare categories differently
        encoder = TargetEncoder(smooth='auto', target_type='continuous')
        encoder.fit(X_rare, y)

        copy = flowbook_deepcopy(encoder)

        # Should produce same results
        np.testing.assert_array_equal(
            encoder.transform(X_rare),
            copy.transform(X_rare)
        )

    def test_encoder_single_category(self, sample_data):
        """Should handle encoders with single-category features."""
        _, y = sample_data

        X_single = pd.DataFrame({
            'cat': ['A'] * 100
        })

        encoder = TargetEncoder(smooth='auto', target_type='continuous')
        encoder.fit(X_single, y)

        copy = flowbook_deepcopy(encoder)

        np.testing.assert_array_equal(
            encoder.transform(X_single),
            copy.transform(X_single)
        )

    def test_encoder_binary_target(self):
        """Should handle encoders with binary classification target."""
        np.random.seed(42)
        X = pd.DataFrame({
            'cat': np.random.choice(['A', 'B', 'C'], 100)
        })
        y = np.random.randint(0, 2, 100)

        encoder = TargetEncoder(smooth='auto', target_type='binary')
        encoder.fit(X, y)

        copy = flowbook_deepcopy(encoder)

        np.testing.assert_array_equal(
            encoder.transform(X),
            copy.transform(X)
        )

    def test_encoder_multiclass_target(self):
        """Should handle encoders with multiclass target."""
        np.random.seed(42)
        X = pd.DataFrame({
            'cat': np.random.choice(['A', 'B', 'C'], 100)
        })
        y = np.random.randint(0, 5, 100)

        encoder = TargetEncoder(smooth='auto', target_type='multiclass')
        encoder.fit(X, y)

        copy = flowbook_deepcopy(encoder)

        np.testing.assert_array_equal(
            encoder.transform(X),
            copy.transform(X)
        )

    def test_default_params(self):
        """Should handle encoder with default params only."""
        encoder = TargetEncoder()
        copy = flowbook_deepcopy(encoder)

        assert copy is not encoder
        # Default params should match
        orig_params = encoder.get_params()
        copy_params = copy.get_params()
        for key in orig_params:
            assert copy_params[key] == orig_params[key], f"Param {key} differs"
