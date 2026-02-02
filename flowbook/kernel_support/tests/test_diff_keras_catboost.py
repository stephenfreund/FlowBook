"""
Comprehensive tests for diff functionality with Keras models and CatBoost Pools.

These tests verify that the diff module correctly:
1. Detects when objects are unchanged after deepcopy
2. Detects changes when objects are modified after deepcopy
3. Handles edge cases for each object type
"""

import numpy as np
import pytest

from flowbook.kernel_support.diff import Diff
from flowbook.kernel_support.deepcopy import deepcopy


# ============================================================================
# CatBoost Pool Tests
# ============================================================================

catboost = pytest.importorskip("catboost")
from catboost import Pool as CatBoostPool


class TestCatBoostPoolDiffIdentical:
    """Tests that identical CatBoost pools show no diff."""

    def test_identical_pools_no_diff(self):
        """Two identical pools should have no diff."""
        pool1 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        pool2 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert result == {}, f"Expected no diff, got: {result}"

    def test_deepcopy_pool_no_diff(self):
        """Deepcopy of pool should have no diff from original."""
        original = CatBoostPool([[1, 2, 3], [4, 5, 6], [7, 8, 9]], label=[0, 1, 0])
        copied = deepcopy(original)

        differ = Diff()
        result = differ.diff({"pool": original}, {"pool": copied})
        assert result == {}, f"Expected no diff, got: {result}"

    def test_pool_with_weights_no_diff(self):
        """Pools with identical weights should have no diff."""
        pool1 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1], weight=[1.0, 2.0])
        pool2 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1], weight=[1.0, 2.0])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert result == {}, f"Expected no diff, got: {result}"

    def test_pool_with_feature_names_no_diff(self):
        """Pools with identical feature names should have no diff."""
        pool1 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1], feature_names=["a", "b"])
        pool2 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1], feature_names=["a", "b"])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert result == {}, f"Expected no diff, got: {result}"

    def test_pool_with_cat_features_no_diff(self):
        """Pools with identical categorical features should have no diff."""
        pool1 = CatBoostPool([[1, "a"], [2, "b"]], label=[0, 1], cat_features=[1])
        pool2 = CatBoostPool([[1, "a"], [2, "b"]], label=[0, 1], cat_features=[1])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert result == {}, f"Expected no diff, got: {result}"


class TestCatBoostPoolDiffDifferent:
    """Tests that different CatBoost pools show correct diffs."""

    def test_different_row_count(self):
        """Pools with different row counts should show diff."""
        pool1 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        pool2 = CatBoostPool([[1, 2], [3, 4], [5, 6]], label=[0, 1, 0])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert "pool" in result, f"Expected diff on 'pool', got: {result}"

    def test_different_col_count(self):
        """Pools with different column counts should show diff."""
        pool1 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        pool2 = CatBoostPool([[1, 2, 5], [3, 4, 6]], label=[0, 1])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert "pool" in result, f"Expected diff on 'pool', got: {result}"

    def test_different_feature_values(self):
        """Pools with different feature values should show diff."""
        pool1 = CatBoostPool([[1.0, 2.0], [3.0, 4.0]], label=[0, 1])
        pool2 = CatBoostPool([[1.0, 2.0], [3.0, 999.0]], label=[0, 1])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert "pool" in result, f"Expected diff on 'pool', got: {result}"

    def test_different_labels(self):
        """Pools with different labels should show diff."""
        pool1 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        pool2 = CatBoostPool([[1, 2], [3, 4]], label=[1, 0])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert "pool" in result, f"Expected diff on 'pool', got: {result}"

    def test_different_weights(self):
        """Pools with different weights should show diff."""
        pool1 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1], weight=[1.0, 1.0])
        pool2 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1], weight=[1.0, 2.0])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert "pool" in result, f"Expected diff on 'pool', got: {result}"

    def test_different_feature_names(self):
        """Pools with different feature names should show diff."""
        pool1 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1], feature_names=["a", "b"])
        pool2 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1], feature_names=["x", "y"])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert "pool" in result, f"Expected diff on 'pool', got: {result}"

    def test_different_cat_feature_indices(self):
        """Pools with different categorical feature indices should show diff."""
        # Use numeric data with different cat_features designations
        pool1 = CatBoostPool([[1, 2, 3], [4, 5, 6]], label=[0, 1], cat_features=[0])
        pool2 = CatBoostPool([[1, 2, 3], [4, 5, 6]], label=[0, 1], cat_features=[1])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert "pool" in result, f"Expected diff on 'pool', got: {result}"

    def test_one_pool_has_labels_other_doesnt(self):
        """Pool with labels vs without should show diff."""
        pool1 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        pool2 = CatBoostPool([[1, 2], [3, 4]])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert "pool" in result, f"Expected diff on 'pool', got: {result}"

    def test_one_pool_has_weights_other_doesnt(self):
        """Pool with weights vs without should show diff."""
        pool1 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1], weight=[1.0, 2.0])
        pool2 = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert "pool" in result, f"Expected diff on 'pool', got: {result}"


class TestCatBoostPoolDiffAfterDeepCopy:
    """Tests that modifications after deepcopy are detected."""

    def test_deepcopy_then_recreate_with_different_data(self):
        """After deepcopy, creating a new pool with different data should show diff."""
        original = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        copied = deepcopy(original)

        # Create a new pool with different data (simulating modification scenario)
        modified = CatBoostPool([[1, 2], [3, 999]], label=[0, 1])

        differ = Diff()
        # Compare copied (unchanged) vs modified
        result = differ.diff({"pool": copied}, {"pool": modified})
        assert "pool" in result, f"Expected diff on 'pool', got: {result}"

    def test_pool_in_dict_deepcopy_independence(self):
        """Pool in dict should be independent after deepcopy."""
        original_pool = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        container = {"pool": original_pool, "name": "test"}

        copied = deepcopy(container)

        # Verify copies are identical
        differ = Diff()
        result = differ.diff(container, copied)
        assert result == {}, f"Expected no diff, got: {result}"

        # Verify pool in copy is independent
        assert copied["pool"] is not original_pool

    def test_multiple_pools_in_namespace(self):
        """Multiple pools in namespace should be handled correctly."""
        pool1 = CatBoostPool([[1, 2]], label=[0])
        pool2 = CatBoostPool([[3, 4]], label=[1])
        pool3 = CatBoostPool([[5, 6]], label=[0])

        ns1 = {"pool_a": pool1, "pool_b": pool2, "pool_c": pool3}
        ns2 = deepcopy(ns1)

        differ = Diff()
        result = differ.diff(ns1, ns2)
        assert result == {}, f"Expected no diff, got: {result}"


class TestCatBoostPoolDiffEdgeCases:
    """Edge cases for CatBoost pool diff."""

    def test_empty_pool(self):
        """Empty pools should compare correctly."""
        # CatBoost requires at least one row, so test single row
        pool1 = CatBoostPool([[1, 2]], label=[0])
        pool2 = CatBoostPool([[1, 2]], label=[0])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert result == {}, f"Expected no diff, got: {result}"

    def test_large_pool(self):
        """Large pools should compare correctly."""
        np.random.seed(42)
        data = np.random.randn(1000, 50)
        labels = np.random.randint(0, 2, 1000)

        pool1 = CatBoostPool(data, label=labels)
        pool2 = deepcopy(pool1)

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert result == {}, f"Expected no diff, got: {result}"

    def test_pool_with_nan_values(self):
        """Pools with NaN values should compare correctly."""
        pool1 = CatBoostPool([[1.0, np.nan], [3.0, 4.0]], label=[0, 1])
        pool2 = CatBoostPool([[1.0, np.nan], [3.0, 4.0]], label=[0, 1])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        assert result == {}, f"Expected no diff, got: {result}"

    def test_float_tolerance(self):
        """Pools with tiny floating point differences should be equal within tolerance."""
        pool1 = CatBoostPool([[1.0, 2.0]], label=[0])
        pool2 = CatBoostPool([[1.0 + 1e-15, 2.0]], label=[0])

        differ = Diff()
        result = differ.diff({"pool": pool1}, {"pool": pool2})
        # Tiny difference should be within tolerance
        assert result == {}, f"Expected no diff for tiny float difference, got: {result}"


# ============================================================================
# Keras Model Tests
# ============================================================================

keras = pytest.importorskip("keras")


class TestKerasModelDiffIdentical:
    """Tests that identical Keras models show no diff."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple Sequential model."""
        model = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(5,)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def test_deepcopy_model_no_diff(self, simple_model):
        """Deepcopy of model should have no diff from original."""
        copied = deepcopy(simple_model)

        differ = Diff()
        result = differ.diff({"model": simple_model}, {"model": copied})
        # Models should be equivalent after deepcopy
        assert result == {}, f"Expected no diff, got: {result}"

    def test_deepcopy_trained_model_no_diff(self, simple_model):
        """Deepcopy of trained model should have no diff from original."""
        # Train the model
        X = np.random.randn(50, 5)
        y = np.random.randn(50, 1)
        simple_model.fit(X, y, epochs=2, verbose=0)

        copied = deepcopy(simple_model)

        differ = Diff()
        result = differ.diff({"model": simple_model}, {"model": copied})
        assert result == {}, f"Expected no diff, got: {result}"


class TestKerasModelDiffDifferent:
    """Tests that different Keras models show correct diffs.

    The dedicated _compare_keras_model handler in diff.py compares:
    - Number of layers
    - Layer configurations (type, units, activation, etc.)
    - Model weights (layer by layer comparison using _compare_ndarray)
    """

    def test_different_weights_detected(self):
        """Models with different weights should show diff."""
        model1 = keras.Sequential([
            keras.layers.Dense(8, input_shape=(3,)),
            keras.layers.Dense(1)
        ])
        model1.compile(optimizer='adam', loss='mse')

        model2 = keras.Sequential([
            keras.layers.Dense(8, input_shape=(3,)),
            keras.layers.Dense(1)
        ])
        model2.compile(optimizer='adam', loss='mse')

        # Models have random weight initialization, so they should differ
        differ = Diff()
        result = differ.diff({"model": model1}, {"model": model2})
        # Two separately initialized models should have different weights
        assert "model" in result, f"Expected diff on 'model' due to different random weights, got: {result}"

    def test_weights_modified_after_copy(self):
        """Modifying weights after copy should show diff."""
        model = keras.Sequential([
            keras.layers.Dense(8, input_shape=(3,)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        copied = deepcopy(model)

        # Modify original's weights
        weights = model.get_weights()
        weights[0] = weights[0] * 2  # Double the first weight matrix
        model.set_weights(weights)

        differ = Diff()
        result = differ.diff({"model": model}, {"model": copied})
        assert "model" in result, f"Expected diff on 'model' after weight modification, got: {result}"

    def test_training_changes_model(self):
        """Training a model should change it compared to its pre-training copy."""
        model = keras.Sequential([
            keras.layers.Dense(8, input_shape=(3,)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Copy before training
        copied = deepcopy(model)

        # Train original
        X = np.random.randn(100, 3)
        y = np.random.randn(100, 1)
        model.fit(X, y, epochs=5, verbose=0)

        differ = Diff()
        result = differ.diff({"model": model}, {"model": copied})
        assert "model" in result, f"Expected diff on 'model' after training, got: {result}"


class TestKerasModelDiffNested:
    """Tests for Keras models in nested structures."""

    @pytest.fixture
    def model(self):
        """Create a simple model."""
        m = keras.Sequential([
            keras.layers.Dense(4, input_shape=(2,)),
            keras.layers.Dense(1)
        ])
        m.compile(optimizer='adam', loss='mse')
        return m

    def test_model_in_dict(self, model):
        """Model in dict should be copied and compared correctly."""
        container = {"model": model, "epochs": 10, "lr": 0.001}
        copied = deepcopy(container)

        differ = Diff()
        result = differ.diff(container, copied)
        assert result == {}, f"Expected no diff, got: {result}"

    def test_model_in_nested_dict(self, model):
        """Model in nested dict should be copied and compared correctly."""
        container = {
            "config": {
                "model": model,
                "params": {"lr": 0.001}
            },
            "name": "experiment1"
        }
        copied = deepcopy(container)

        differ = Diff()
        result = differ.diff(container, copied)
        assert result == {}, f"Expected no diff, got: {result}"

    def test_model_in_list(self, model):
        """Model in list should be copied and compared correctly."""
        container = [model, "config", 123]
        copied = deepcopy(container)

        differ = Diff()
        result = differ.diff({"data": container}, {"data": copied})
        assert result == {}, f"Expected no diff, got: {result}"

    def test_multiple_models_same_reference(self, model):
        """Same model referenced multiple times should work correctly."""
        container = {"model1": model, "model2": model}
        copied = deepcopy(container)

        # Both should point to same copy
        assert copied["model1"] is copied["model2"]

        differ = Diff()
        result = differ.diff(container, copied)
        assert result == {}, f"Expected no diff, got: {result}"


class TestKerasModelDiffEdgeCases:
    """Edge cases for Keras model diff."""

    def test_uncompiled_model(self):
        """Uncompiled model should be handled correctly."""
        model = keras.Sequential([
            keras.layers.Dense(4, input_shape=(2,))
        ])
        # Not compiled

        copied = deepcopy(model)

        differ = Diff()
        result = differ.diff({"model": model}, {"model": copied})
        assert result == {}, f"Expected no diff, got: {result}"

    def test_functional_model(self):
        """Functional API model should be handled correctly."""
        inputs = keras.Input(shape=(5,))
        x = keras.layers.Dense(8, activation='relu')(inputs)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')

        copied = deepcopy(model)

        differ = Diff()
        result = differ.diff({"model": model}, {"model": copied})
        assert result == {}, f"Expected no diff, got: {result}"

    def test_model_with_dropout(self):
        """Model with dropout layers should be handled correctly."""
        model = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(5,)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        copied = deepcopy(model)

        differ = Diff()
        result = differ.diff({"model": model}, {"model": copied})
        assert result == {}, f"Expected no diff, got: {result}"

    def test_model_with_batch_norm(self):
        """Model with batch normalization should be handled correctly."""
        model = keras.Sequential([
            keras.layers.Dense(16, input_shape=(5,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        copied = deepcopy(model)

        differ = Diff()
        result = differ.diff({"model": model}, {"model": copied})
        assert result == {}, f"Expected no diff, got: {result}"


# ============================================================================
# Mixed Tests (Both CatBoost and Keras)
# ============================================================================

class TestMixedObjectsDiff:
    """Tests with both CatBoost and Keras objects."""

    def test_namespace_with_both_types(self):
        """Namespace containing both pool and model should work correctly."""
        pool = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        model = keras.Sequential([
            keras.layers.Dense(4, input_shape=(2,)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        namespace = {
            "pool": pool,
            "model": model,
            "config": {"epochs": 10}
        }
        copied = deepcopy(namespace)

        differ = Diff()
        result = differ.diff(namespace, copied)
        assert result == {}, f"Expected no diff, got: {result}"

    def test_nested_structure_with_both_types(self):
        """Nested structure with both types should work correctly."""
        pool = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        model = keras.Sequential([
            keras.layers.Dense(4, input_shape=(2,)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        structure = {
            "data": {
                "train_pool": pool,
                "val_pool": CatBoostPool([[5, 6]], label=[1])
            },
            "models": {
                "main": model,
                "backup": deepcopy(model)
            }
        }
        copied = deepcopy(structure)

        differ = Diff()
        result = differ.diff(structure, copied)
        assert result == {}, f"Expected no diff, got: {result}"

    def test_one_type_modified_other_unchanged(self):
        """When only one type is modified, diff should reflect that."""
        pool = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        model = keras.Sequential([
            keras.layers.Dense(4, input_shape=(2,)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        ns1 = {"pool": pool, "model": model}
        ns2 = deepcopy(ns1)

        # Modify only the model weights in ns1
        weights = ns1["model"].get_weights()
        weights[0] = weights[0] * 2
        ns1["model"].set_weights(weights)

        differ = Diff()
        result = differ.diff(ns1, ns2)

        # Model should show diff, pool should not
        assert "model" in result, f"Expected diff on 'model', got: {result}"
        assert "pool" not in result, f"Pool should not have diff, got: {result}"

    def test_pool_modified_model_unchanged(self):
        """When pool is recreated with different data, diff should detect it."""
        pool = CatBoostPool([[1, 2], [3, 4]], label=[0, 1])
        model = keras.Sequential([
            keras.layers.Dense(4, input_shape=(2,)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        ns1 = {"pool": pool, "model": model}
        ns2 = deepcopy(ns1)

        # Replace pool with different data in ns1
        ns1["pool"] = CatBoostPool([[1, 2], [3, 999]], label=[0, 1])

        differ = Diff()
        result = differ.diff(ns1, ns2)

        # Pool should show diff (CatBoost diff works properly)
        assert "pool" in result, f"Expected diff on 'pool', got: {result}"
