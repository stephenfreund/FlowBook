"""Tests for opaque handler pattern with Keras models."""

import pytest
import numpy as np

# Skip all tests if Keras is not installed
keras = pytest.importorskip("keras")

from flowbook.kernel.opaque import (
    OpaqueRegistry,
    KerasModelHandler,
    reset_keras_handler,
)
from flowbook.kernel.deepcopy import deepcopy as flowbook_deepcopy
from flowbook.kernel.deepcopyable import check_deepcopyable
from flowbook.kernel.checkpoint import _collect_reachable_ids


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the opaque registry before each test."""
    OpaqueRegistry.clear()
    reset_keras_handler()
    yield
    OpaqueRegistry.clear()
    reset_keras_handler()


@pytest.fixture
def unbuilt_model():
    """Create an unbuilt Sequential model."""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    # Model is NOT built (no input_shape specified, no .build() called)
    assert not model.built
    return model


@pytest.fixture
def built_model():
    """Create a built Sequential model."""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    assert model.built
    return model


@pytest.fixture
def trained_model(built_model):
    """Create a trained Sequential model."""
    X = np.random.randn(100, 10)
    y = np.random.randn(100, 1)
    built_model.fit(X, y, epochs=2, verbose=0)
    return built_model


class TestKerasModelHandler:
    """Tests for KerasModelHandler."""

    def test_can_handle_sequential(self, built_model):
        """Handler should recognize Sequential models."""
        handler = KerasModelHandler()
        assert handler.can_handle(built_model)

    def test_can_handle_functional(self):
        """Handler should recognize Functional models."""
        inputs = keras.Input(shape=(10,))
        x = keras.layers.Dense(64)(inputs)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        handler = KerasModelHandler()
        assert handler.can_handle(model)

    def test_cannot_handle_non_keras(self):
        """Handler should not recognize non-Keras objects."""
        handler = KerasModelHandler()
        assert not handler.can_handle("string")
        assert not handler.can_handle([1, 2, 3])
        assert not handler.can_handle({'key': 'value'})
        assert not handler.can_handle(np.array([1, 2, 3]))

    def test_unbuilt_not_checkpointable(self, unbuilt_model):
        """Unbuilt model should not be checkpointable."""
        handler = KerasModelHandler()
        can_cp, error = handler.is_checkpointable(unbuilt_model)
        assert not can_cp
        assert "not built" in error.lower()

    def test_built_is_checkpointable(self, built_model):
        """Built model should be checkpointable."""
        handler = KerasModelHandler()
        can_cp, error = handler.is_checkpointable(built_model)
        assert can_cp
        assert error is None

    def test_get_mutable_state(self, trained_model):
        """Should extract weights as mutable state."""
        handler = KerasModelHandler()
        state = handler.get_mutable_state(trained_model)

        assert 'weights' in state
        assert len(state['weights']) == len(trained_model.get_weights())
        assert 'input_shape' in state

        # Weights should be deep copies
        for orig, state_w in zip(trained_model.get_weights(), state['weights']):
            np.testing.assert_array_equal(orig, state_w)
            assert orig is not state_w  # Different objects

    def test_copy_with_state(self, trained_model):
        """Should create copy with independent weights."""
        handler = KerasModelHandler()
        state = handler.get_mutable_state(trained_model)
        memo = {}
        copy = handler.copy_with_state(trained_model, state, memo)

        # Copy should have same architecture
        assert len(copy.layers) == len(trained_model.layers)

        # Copy should have same weights
        for orig, copy_w in zip(trained_model.get_weights(), copy.get_weights()):
            np.testing.assert_array_almost_equal(orig, copy_w)

        # Copy should be in memo
        assert id(trained_model) in memo
        assert memo[id(trained_model)] is copy

    def test_states_equal(self, trained_model):
        """Should correctly compare states."""
        handler = KerasModelHandler()
        state1 = handler.get_mutable_state(trained_model)
        state2 = handler.get_mutable_state(trained_model)

        assert handler.states_equal(state1, state2)

        # Modify state2
        state2['weights'][0][0, 0] += 1.0
        assert not handler.states_equal(state1, state2)


class TestOpaqueRegistry:
    """Tests for OpaqueRegistry."""

    def test_register_and_get_handler(self, built_model):
        """Should register and retrieve handlers."""
        # Initially no handler
        OpaqueRegistry.clear()
        reset_keras_handler()

        # Get handler triggers lazy registration
        handler = OpaqueRegistry.get_handler(built_model)
        assert handler is not None
        assert isinstance(handler, KerasModelHandler)

    def test_is_opaque(self, built_model):
        """Should detect opaque objects."""
        assert OpaqueRegistry.is_opaque(built_model)
        assert not OpaqueRegistry.is_opaque("string")
        assert not OpaqueRegistry.is_opaque([1, 2, 3])

    def test_non_keras_not_opaque(self):
        """Non-Keras objects should not be opaque."""
        assert not OpaqueRegistry.is_opaque(np.array([1, 2, 3]))
        assert not OpaqueRegistry.is_opaque({'key': 'value'})


class TestUnbuiltModelErrors:
    """Tests for unbuilt model error handling."""

    def test_check_deepcopyable_unbuilt(self, unbuilt_model):
        """check_deepcopyable should report unbuilt models as not copyable."""
        result = check_deepcopyable(unbuilt_model)
        assert result is not None
        assert "not built" in result.lower()

    def test_deepcopy_unbuilt_raises(self, unbuilt_model):
        """flowbook_deepcopy should raise for unbuilt models."""
        with pytest.raises(TypeError) as excinfo:
            flowbook_deepcopy(unbuilt_model)
        assert "not built" in str(excinfo.value).lower()

    def test_check_deepcopyable_built(self, built_model):
        """check_deepcopyable should accept built models."""
        result = check_deepcopyable(built_model)
        assert result is None


class TestAliasIndexOptimization:
    """Tests for alias index optimization with opaque objects."""

    def test_model_has_single_id(self, trained_model):
        """Built model should have only 1 ID in alias tracking."""
        visited = set()
        _collect_reachable_ids(trained_model, visited)

        # Should have exactly 1 ID (the model itself)
        assert len(visited) == 1
        assert id(trained_model) in visited

    def test_model_in_dict_minimal_ids(self, trained_model):
        """Dict containing model should have minimal IDs."""
        data = {'model': trained_model, 'name': 'test'}
        visited = set()
        _collect_reachable_ids(data, visited)

        # Should have: dict, model, string 'name' (if tracked), but NOT
        # the millions of internal keras objects
        assert len(visited) < 10
        assert id(data) in visited
        assert id(trained_model) in visited

    def test_callbacks_referencing_model(self, trained_model):
        """Callbacks referencing model should not explode ID count."""
        # Simulate what happens with EarlyStopping callback
        class MockCallback:
            def __init__(self, model):
                self.model = model
                self.patience = 10

        callback = MockCallback(trained_model)

        visited = set()
        _collect_reachable_ids(callback, visited)

        # Should have: callback, model, patience (int - not tracked)
        # Not the millions of keras internals
        assert len(visited) < 10
        assert id(callback) in visited
        assert id(trained_model) in visited


class TestDeepCopyWithOpaqueHandler:
    """Tests for deepcopy using opaque handler."""

    def test_copy_preserves_weights(self, trained_model):
        """Copy should preserve all weights."""
        copy = flowbook_deepcopy(trained_model)

        orig_weights = trained_model.get_weights()
        copy_weights = copy.get_weights()

        assert len(orig_weights) == len(copy_weights)
        for o, c in zip(orig_weights, copy_weights):
            np.testing.assert_array_almost_equal(o, c)

    def test_copy_is_independent(self, trained_model):
        """Modifying original should not affect copy."""
        copy = flowbook_deepcopy(trained_model)

        X_test = np.random.randn(5, 10)
        pred_copy_before = copy.predict(X_test, verbose=0)

        # Modify original weights
        weights = trained_model.get_weights()
        weights[0] = weights[0] * 2
        trained_model.set_weights(weights)

        # Copy should be unchanged
        pred_copy_after = copy.predict(X_test, verbose=0)
        np.testing.assert_array_almost_equal(pred_copy_before, pred_copy_after)

    def test_modifying_copy_doesnt_affect_original(self, trained_model):
        """Modifying copy should not affect original."""
        copy = flowbook_deepcopy(trained_model)

        X_test = np.random.randn(5, 10)
        pred_orig_before = trained_model.predict(X_test, verbose=0)

        # Modify copy weights
        weights = copy.get_weights()
        weights[0] = weights[0] * 2
        copy.set_weights(weights)

        # Original should be unchanged
        pred_orig_after = trained_model.predict(X_test, verbose=0)
        np.testing.assert_array_almost_equal(pred_orig_before, pred_orig_after)

    def test_memo_sharing(self, trained_model):
        """Same model referenced twice should share copy in memo."""
        data = {
            'model1': trained_model,
            'model2': trained_model  # Same reference
        }
        data_copy = flowbook_deepcopy(data)

        # Both should point to the same copy
        assert data_copy['model1'] is data_copy['model2']
        # But different from original
        assert data_copy['model1'] is not trained_model


class TestFunctionalModel:
    """Tests for Functional API models with opaque handler."""

    @pytest.fixture
    def functional_model(self):
        """Create a Functional API model."""
        inputs = keras.Input(shape=(10,))
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.Dense(32, activation='relu')(x)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def test_functional_is_checkpointable(self, functional_model):
        """Functional model should be checkpointable."""
        result = check_deepcopyable(functional_model)
        assert result is None

    def test_functional_copy(self, functional_model):
        """Functional model should copy correctly."""
        copy = flowbook_deepcopy(functional_model)
        assert len(copy.layers) == len(functional_model.layers)

    def test_functional_alias_tracking(self, functional_model):
        """Functional model should have minimal alias IDs."""
        visited = set()
        _collect_reachable_ids(functional_model, visited)
        assert len(visited) == 1


class TestKerasCustomAttributes:
    """Tests for custom __dict__ attribute capture and restoration."""

    def test_model_custom_int_preserved(self, built_model):
        """Custom integer attribute should be preserved after copy."""
        built_model.custom_epochs = 42
        copy = flowbook_deepcopy(built_model)

        assert hasattr(copy, 'custom_epochs')
        assert copy.custom_epochs == 42

    def test_model_custom_list_preserved(self, built_model):
        """Custom list attribute should be preserved after copy."""
        built_model.loss_history = [0.5, 0.3, 0.1]
        copy = flowbook_deepcopy(built_model)

        assert hasattr(copy, 'loss_history')
        assert copy.loss_history == [0.5, 0.3, 0.1]
        # Should be independent copy
        assert copy.loss_history is not built_model.loss_history

    def test_model_custom_dict_preserved(self, built_model):
        """Custom dict attribute should be preserved after copy."""
        built_model.config = {'lr': 0.01, 'batch_size': 32}
        copy = flowbook_deepcopy(built_model)

        assert hasattr(copy, 'config')
        assert copy.config == {'lr': 0.01, 'batch_size': 32}
        # Should be independent copy
        assert copy.config is not built_model.config

    def test_layer_custom_attribute_preserved(self, built_model):
        """Custom attribute on a layer should be preserved after copy."""
        # Add custom attribute to the first Dense layer
        built_model.layers[0].custom_scale = 2.5
        copy = flowbook_deepcopy(built_model)

        assert hasattr(copy.layers[0], 'custom_scale')
        assert copy.layers[0].custom_scale == 2.5

    def test_keras_internals_not_captured_as_custom(self, built_model):
        """Keras internal attributes should not be captured as custom attrs."""
        handler = KerasModelHandler()
        state = handler.get_mutable_state(built_model)

        # These are Keras internals and should NOT be in model_dict
        model_dict = state.get('model_dict', {})
        assert 'layers' not in model_dict
        assert 'optimizer' not in model_dict
        assert 'built' not in model_dict
        assert 'trainable' not in model_dict

    def test_custom_attr_independent_after_copy(self, built_model):
        """Modifying custom attr on original should not affect copy."""
        built_model.my_list = [1, 2, 3]
        copy = flowbook_deepcopy(built_model)

        # Modify original
        built_model.my_list.append(4)

        # Copy should be unchanged
        assert copy.my_list == [1, 2, 3]

    def test_custom_attr_aliased_for_tracking(self, built_model):
        """Custom attributes should be traversed for alias detection."""
        shared_list = [1, 2, 3]
        built_model.custom_data = shared_list

        visited = set()
        _collect_reachable_ids(built_model, visited)

        # Should find the shared_list since it's a custom attribute
        assert id(shared_list) in visited

    def test_state_includes_custom_attrs(self, built_model):
        """get_mutable_state should include custom attributes."""
        built_model.my_value = 100
        built_model.my_dict = {'a': 1}

        handler = KerasModelHandler()
        state = handler.get_mutable_state(built_model)

        assert 'model_dict' in state
        assert state['model_dict'].get('my_value') == 100
        assert state['model_dict'].get('my_dict') == {'a': 1}

    def test_states_equal_considers_custom_attrs(self, built_model):
        """states_equal should compare custom attributes."""
        built_model.custom_value = 100

        handler = KerasModelHandler()
        state1 = handler.get_mutable_state(built_model)

        # Modify and get new state
        built_model.custom_value = 200
        state2 = handler.get_mutable_state(built_model)

        # States should not be equal due to custom attr difference
        assert not handler.states_equal(state1, state2)
