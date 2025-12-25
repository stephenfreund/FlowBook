"""Tests for Keras model deepcopy support."""

import pytest
import numpy as np

# Skip all tests if Keras is not installed
keras = pytest.importorskip("keras")

from data_ferret.kernel.deepcopy import deepcopy as ferret_deepcopy
from data_ferret.kernel.deepcopyable import check_deepcopyable


@pytest.fixture
def simple_model():
    """Create a simple Sequential model."""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


@pytest.fixture
def trained_model(simple_model):
    """Create a trained Sequential model."""
    X = np.random.randn(100, 10)
    y = np.random.randn(100, 1)
    simple_model.fit(X, y, epochs=2, verbose=0)
    return simple_model


class TestCheckDeepCopyable:
    """Tests for check_deepcopyable with Keras models."""

    def test_sequential_is_copyable(self, simple_model):
        """Sequential model should be detected as copyable."""
        result = check_deepcopyable(simple_model)
        assert result is None, f"Expected None (copyable), got: {result}"

    def test_trained_model_is_copyable(self, trained_model):
        """Trained model should also be detected as copyable."""
        result = check_deepcopyable(trained_model)
        assert result is None, f"Expected None (copyable), got: {result}"

    def test_model_in_dict_is_copyable(self, simple_model):
        """Dict containing model should be copyable."""
        data = {'model': simple_model, 'name': 'test'}
        result = check_deepcopyable(data)
        assert result is None, f"Expected None (copyable), got: {result}"

    def test_model_in_list_is_copyable(self, simple_model):
        """List containing model should be copyable."""
        data = [simple_model, 'test']
        result = check_deepcopyable(data)
        assert result is None, f"Expected None (copyable), got: {result}"


class TestFerretDeepCopy:
    """Tests for ferret_deepcopy with Keras models."""

    def test_simple_model_copy(self, simple_model):
        """Simple model should be copyable."""
        model_copy = ferret_deepcopy(simple_model)
        assert len(model_copy.layers) == len(simple_model.layers)

    def test_trained_model_copy(self, trained_model):
        """Trained model copy should preserve weights."""
        model_copy = ferret_deepcopy(trained_model)

        orig_weights = trained_model.get_weights()
        copy_weights = model_copy.get_weights()

        assert len(orig_weights) == len(copy_weights)
        for o, c in zip(orig_weights, copy_weights):
            np.testing.assert_array_almost_equal(o, c)

    def test_copy_independence(self, trained_model):
        """Modifying original should not affect copy."""
        model_copy = ferret_deepcopy(trained_model)

        X_test = np.random.randn(5, 10)
        pred_copy_before = model_copy.predict(X_test, verbose=0)

        # Modify original
        weights = trained_model.get_weights()
        weights[0] = weights[0] * 2
        trained_model.set_weights(weights)

        # Copy should be unchanged
        pred_copy_after = model_copy.predict(X_test, verbose=0)
        np.testing.assert_array_almost_equal(pred_copy_before, pred_copy_after)

    def test_model_in_dict(self, simple_model):
        """Model in dict should be copied correctly."""
        data = {'model': simple_model, 'epochs': 10}
        data_copy = ferret_deepcopy(data)

        assert 'model' in data_copy
        assert len(data_copy['model'].layers) == len(simple_model.layers)
        assert data_copy['model'] is not simple_model

    def test_model_in_nested_structure(self, simple_model):
        """Model in nested structure should be copied correctly."""
        data = {
            'config': {
                'model': simple_model,
                'params': [1, 2, 3]
            },
            'name': 'test'
        }
        data_copy = ferret_deepcopy(data)

        assert data_copy['config']['model'] is not simple_model
        assert len(data_copy['config']['model'].layers) == len(simple_model.layers)

    def test_memo_sharing(self, simple_model):
        """Same model referenced twice should share copy."""
        data = {
            'model1': simple_model,
            'model2': simple_model  # Same reference
        }
        data_copy = ferret_deepcopy(data)

        # Both should point to the same copy
        assert data_copy['model1'] is data_copy['model2']
        # But different from original
        assert data_copy['model1'] is not simple_model


class TestFunctionalModel:
    """Tests for Keras Functional API models."""

    @pytest.fixture
    def functional_model(self):
        """Create a simple Functional API model."""
        inputs = keras.Input(shape=(10,))
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.Dense(32, activation='relu')(x)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def test_functional_is_copyable(self, functional_model):
        """Functional model should be detected as copyable."""
        result = check_deepcopyable(functional_model)
        assert result is None, f"Expected None (copyable), got: {result}"

    def test_functional_copy(self, functional_model):
        """Functional model should be copyable."""
        model_copy = ferret_deepcopy(functional_model)
        assert len(model_copy.layers) == len(functional_model.layers)


class TestEdgeCases:
    """Edge case tests."""

    def test_uncompiled_model(self):
        """Uncompiled model should be copyable."""
        model = keras.Sequential([
            keras.layers.Dense(10, input_shape=(5,))
        ])
        # Not compiled
        result = check_deepcopyable(model)
        assert result is None

        model_copy = ferret_deepcopy(model)
        assert len(model_copy.layers) == 1

    def test_model_with_custom_layer(self):
        """Model with registered custom layer should be copyable."""
        # Custom layers must be registered with Keras for serialization
        @keras.saving.register_keras_serializable(package='test')
        class ScaledDense(keras.layers.Dense):
            def __init__(self, units, scale=1.0, **kwargs):
                super().__init__(units, **kwargs)
                self.scale = scale

            def call(self, inputs):
                return super().call(inputs) * self.scale

            def get_config(self):
                config = super().get_config()
                config['scale'] = self.scale
                return config

        model = keras.Sequential([
            ScaledDense(10, scale=2.0, input_shape=(5,))
        ])

        result = check_deepcopyable(model)
        assert result is None

        model_copy = ferret_deepcopy(model)
        assert len(model_copy.layers) == 1
