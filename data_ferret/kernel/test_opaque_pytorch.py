"""Tests for opaque handler pattern with PyTorch models."""

import pytest
import numpy as np

# Skip all tests if PyTorch is not installed
torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from data_ferret.kernel.opaque import (
    OpaqueRegistry,
    PyTorchModelHandler,
    reset_pytorch_handler,
    reset_keras_handler,
)
from data_ferret.kernel.deepcopy import deepcopy as ferret_deepcopy
from data_ferret.kernel.deepcopyable import check_deepcopyable
from data_ferret.kernel.checkpoint import _collect_reachable_ids


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the opaque registry before each test."""
    OpaqueRegistry.clear()
    reset_pytorch_handler()
    reset_keras_handler()
    yield
    OpaqueRegistry.clear()
    reset_pytorch_handler()
    reset_keras_handler()


@pytest.fixture
def simple_model():
    """Create a simple Sequential model."""
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    return model


@pytest.fixture
def trained_model(simple_model):
    """Create a model with modified weights (simulating training)."""
    # Just do a forward pass to ensure everything is initialized
    x = torch.randn(5, 10)
    _ = simple_model(x)
    return simple_model


class TestPyTorchModelHandler:
    """Tests for PyTorchModelHandler."""

    def test_can_handle_sequential(self, simple_model):
        """Handler should recognize Sequential models."""
        handler = PyTorchModelHandler()
        assert handler.can_handle(simple_model)

    def test_can_handle_linear(self):
        """Handler should recognize basic nn.Module subclasses."""
        handler = PyTorchModelHandler()
        model = nn.Linear(10, 5)
        assert handler.can_handle(model)

    def test_can_handle_custom_module(self):
        """Handler should recognize custom nn.Module subclasses."""
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        handler = PyTorchModelHandler()
        model = CustomModel()
        assert handler.can_handle(model)

    def test_cannot_handle_non_pytorch(self):
        """Handler should not recognize non-PyTorch objects."""
        handler = PyTorchModelHandler()
        assert not handler.can_handle("string")
        assert not handler.can_handle([1, 2, 3])
        assert not handler.can_handle({'key': 'value'})
        assert not handler.can_handle(np.array([1, 2, 3]))

    def test_regular_model_is_checkpointable(self, simple_model):
        """Regular initialized model should be checkpointable."""
        handler = PyTorchModelHandler()
        can_cp, error = handler.is_checkpointable(simple_model)
        assert can_cp
        assert error is None

    def test_lazy_module_uninitialized_not_checkpointable(self):
        """Uninitialized lazy module should not be checkpointable."""
        # LazyLinear defers weight initialization until first forward pass
        model = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

        handler = PyTorchModelHandler()
        can_cp, error = handler.is_checkpointable(model)
        assert not can_cp
        assert "uninitialized lazy module" in error.lower()

    def test_lazy_module_initialized_is_checkpointable(self):
        """Initialized lazy module should be checkpointable."""
        model = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(1)
        )
        # Initialize by running a forward pass
        x = torch.randn(5, 10)
        _ = model(x)

        handler = PyTorchModelHandler()
        can_cp, error = handler.is_checkpointable(model)
        assert can_cp
        assert error is None

    def test_get_mutable_state(self, trained_model):
        """Should extract state_dict and training mode as mutable state."""
        handler = PyTorchModelHandler()
        state = handler.get_mutable_state(trained_model)

        assert 'state_dict' in state
        assert 'training' in state
        assert 'device' in state
        assert 'model_dict' in state
        assert 'submodule_dicts' in state

        # State dict should have all parameters
        orig_sd = trained_model.state_dict()
        assert set(state['state_dict'].keys()) == set(orig_sd.keys())

        # State dict tensors should be independent copies
        for key in state['state_dict']:
            assert state['state_dict'][key] is not orig_sd[key]
            assert torch.allclose(state['state_dict'][key], orig_sd[key].cpu())

    def test_copy_with_state(self, trained_model):
        """Should create copy with independent weights."""
        handler = PyTorchModelHandler()
        state = handler.get_mutable_state(trained_model)
        memo = {}
        copy = handler.copy_with_state(trained_model, state, memo)

        # Copy should have same structure
        orig_params = list(trained_model.parameters())
        copy_params = list(copy.parameters())
        assert len(orig_params) == len(copy_params)

        # Copy should have same weights
        for orig, copy_p in zip(orig_params, copy_params):
            assert torch.allclose(orig, copy_p)

        # Copy should be in memo
        assert id(trained_model) in memo
        assert memo[id(trained_model)] is copy

    def test_states_equal(self, trained_model):
        """Should correctly compare states."""
        handler = PyTorchModelHandler()
        state1 = handler.get_mutable_state(trained_model)
        state2 = handler.get_mutable_state(trained_model)

        assert handler.states_equal(state1, state2)

        # Modify state2
        first_key = list(state2['state_dict'].keys())[0]
        state2['state_dict'][first_key] = state2['state_dict'][first_key] + 1.0
        assert not handler.states_equal(state1, state2)


class TestOpaqueRegistryPyTorch:
    """Tests for OpaqueRegistry with PyTorch models."""

    def test_lazy_registration_on_first_model(self, simple_model):
        """Handler should be registered lazily on first model encounter."""
        # Initially no handler
        OpaqueRegistry.clear()
        reset_pytorch_handler()

        # Get handler triggers lazy registration
        handler = OpaqueRegistry.get_handler(simple_model)
        assert handler is not None
        assert isinstance(handler, PyTorchModelHandler)

    def test_is_opaque(self, simple_model):
        """Should detect opaque objects."""
        assert OpaqueRegistry.is_opaque(simple_model)
        assert not OpaqueRegistry.is_opaque("string")
        assert not OpaqueRegistry.is_opaque([1, 2, 3])

    def test_non_pytorch_not_opaque(self):
        """Non-PyTorch objects should not be opaque."""
        assert not OpaqueRegistry.is_opaque(np.array([1, 2, 3]))
        assert not OpaqueRegistry.is_opaque({'key': 'value'})


class TestPyTorchAliasTracking:
    """Tests for alias tracking with PyTorch models."""

    def test_model_has_single_id(self, trained_model):
        """Model should have only 1 ID in alias tracking (internal structure skipped)."""
        visited = set()
        _collect_reachable_ids(trained_model, visited)

        # Should have exactly 1 ID (the model itself)
        # Internal structure (layers, parameters) is NOT traversed
        assert len(visited) == 1
        assert id(trained_model) in visited

    def test_model_in_dict_minimal_ids(self, trained_model):
        """Dict containing model should have minimal IDs."""
        data = {'model': trained_model, 'name': 'test'}
        visited = set()
        _collect_reachable_ids(data, visited)

        # Should have: dict, model, string 'name', string 'test'
        # NOT the millions of internal PyTorch objects
        assert len(visited) < 10
        assert id(data) in visited
        assert id(trained_model) in visited

    def test_custom_attr_traversed_for_aliasing(self, trained_model):
        """Custom attributes on model should be traversed for alias detection."""
        shared_list = [1, 2, 3]
        trained_model.custom_data = shared_list

        visited = set()
        _collect_reachable_ids(trained_model, visited)

        # Should find the shared_list since it's a custom attribute
        assert id(shared_list) in visited

    def test_nested_models_tracked_separately(self, simple_model):
        """Nested models should each be tracked."""
        class ContainerModel(nn.Module):
            def __init__(self, submodel):
                super().__init__()
                self.submodel = submodel
                self.fc = nn.Linear(1, 1)

            def forward(self, x):
                return self.fc(self.submodel(x))

        container = ContainerModel(simple_model)
        visited = set()
        _collect_reachable_ids(container, visited)

        # Container is opaque, so we only get its ID
        # (submodel is part of container's structure, handled internally)
        assert id(container) in visited


class TestPyTorchCustomAttributes:
    """Tests for custom attribute preservation."""

    def test_custom_int_preserved(self, simple_model):
        """Custom integer attribute should be preserved after copy."""
        simple_model.custom_value = 42
        copy = ferret_deepcopy(simple_model)

        assert hasattr(copy, 'custom_value')
        assert copy.custom_value == 42

    def test_custom_list_preserved(self, simple_model):
        """Custom list attribute should be preserved after copy."""
        simple_model.custom_list = [1, 2, 3]
        copy = ferret_deepcopy(simple_model)

        assert hasattr(copy, 'custom_list')
        assert copy.custom_list == [1, 2, 3]
        # Should be independent copy
        assert copy.custom_list is not simple_model.custom_list

    def test_custom_dict_preserved(self, simple_model):
        """Custom dict attribute should be preserved after copy."""
        simple_model.config = {'lr': 0.01, 'epochs': 10}
        copy = ferret_deepcopy(simple_model)

        assert hasattr(copy, 'config')
        assert copy.config == {'lr': 0.01, 'epochs': 10}
        # Should be independent copy
        assert copy.config is not simple_model.config


class TestDeepCopyWithOpaqueHandler:
    """Tests for deepcopy using opaque handler."""

    def test_copy_preserves_weights(self, trained_model):
        """Copy should preserve all weights."""
        copy = ferret_deepcopy(trained_model)

        orig_sd = trained_model.state_dict()
        copy_sd = copy.state_dict()

        assert set(orig_sd.keys()) == set(copy_sd.keys())
        for key in orig_sd.keys():
            assert torch.allclose(orig_sd[key], copy_sd[key])

    def test_copy_is_independent(self, trained_model):
        """Modifying original should not affect copy."""
        copy = ferret_deepcopy(trained_model)

        X_test = torch.randn(5, 10)
        pred_copy_before = copy(X_test).detach().clone()

        # Modify original weights
        with torch.no_grad():
            for param in trained_model.parameters():
                param.mul_(2.0)

        # Copy should be unchanged
        pred_copy_after = copy(X_test).detach()
        assert torch.allclose(pred_copy_before, pred_copy_after)

    def test_modifying_copy_doesnt_affect_original(self, trained_model):
        """Modifying copy should not affect original."""
        copy = ferret_deepcopy(trained_model)

        X_test = torch.randn(5, 10)
        pred_orig_before = trained_model(X_test).detach().clone()

        # Modify copy weights
        with torch.no_grad():
            for param in copy.parameters():
                param.mul_(2.0)

        # Original should be unchanged
        pred_orig_after = trained_model(X_test).detach()
        assert torch.allclose(pred_orig_before, pred_orig_after)

    def test_memo_sharing(self, trained_model):
        """Same model referenced twice should share copy in memo."""
        data = {
            'model1': trained_model,
            'model2': trained_model  # Same reference
        }
        data_copy = ferret_deepcopy(data)

        # Both should point to the same copy
        assert data_copy['model1'] is data_copy['model2']
        # But different from original
        assert data_copy['model1'] is not trained_model


class TestTrainingModePreservation:
    """Tests for training mode preservation."""

    def test_training_mode_preserved(self, simple_model):
        """Training mode should be preserved."""
        simple_model.train()
        assert simple_model.training

        copy = ferret_deepcopy(simple_model)
        assert copy.training

    def test_eval_mode_preserved(self, simple_model):
        """Eval mode should be preserved."""
        simple_model.eval()
        assert not simple_model.training

        copy = ferret_deepcopy(simple_model)
        assert not copy.training


class TestCheckDeepCopyable:
    """Tests for check_deepcopyable with PyTorch models."""

    def test_regular_model_copyable(self, simple_model):
        """Regular model should be detected as copyable."""
        result = check_deepcopyable(simple_model)
        assert result is None, f"Expected None (copyable), got: {result}"

    def test_lazy_uninitialized_not_copyable(self):
        """Uninitialized lazy module should not be copyable."""
        model = nn.Sequential(
            nn.LazyLinear(64),
            nn.LazyLinear(1)
        )

        result = check_deepcopyable(model)
        assert result is not None
        assert "uninitialized lazy module" in result.lower()

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
