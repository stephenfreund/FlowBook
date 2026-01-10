"""Tests for PyTorch model diff/comparison functionality."""

import pytest
import numpy as np

# Skip all tests if PyTorch is not installed
torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from flowbook.kernel.diff import Diff


@pytest.fixture
def simple_model():
    """Create a simple Sequential model."""
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    return model


@pytest.fixture
def another_model():
    """Create another simple Sequential model with same architecture."""
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    return model


class TestPyTorchDiff:
    """Tests for PyTorch model comparison."""

    def test_identical_models_no_diff(self, simple_model):
        """Identical models should have no difference."""
        differ = Diff()
        result = differ._compare_pytorch_model(simple_model, simple_model, "model")
        assert result is None

    def test_same_architecture_same_weights_no_diff(self, simple_model):
        """Models with same architecture and weights should have no diff."""
        import copy
        model_copy = copy.deepcopy(simple_model)

        differ = Diff()
        result = differ._compare_pytorch_model(simple_model, model_copy, "model")
        assert result is None

    def test_different_weights_detected(self, simple_model, another_model):
        """Different weights should be detected."""
        # Models have same architecture but different random weights
        differ = Diff()
        result = differ._compare_pytorch_model(simple_model, another_model, "model")

        # Should have differences in parameters
        assert result is not None
        assert hasattr(result, 'children')
        # At least one parameter should differ
        param_diffs = [k for k in result.children.keys() if k.startswith('_param_')]
        assert len(param_diffs) > 0

    def test_different_architecture_detected(self):
        """Different architectures should be detected."""
        model_a = nn.Sequential(
            nn.Linear(10, 64),
            nn.Linear(64, 1)
        )
        model_b = nn.Sequential(
            nn.Linear(10, 32),
            nn.Linear(32, 1)
        )

        differ = Diff()
        result = differ._compare_pytorch_model(model_a, model_b, "model")

        assert result is not None
        # Should detect parameter shape differences
        assert hasattr(result, 'children')

    def test_different_training_mode_detected(self, simple_model):
        """Different training modes should be detected."""
        import copy
        model_copy = copy.deepcopy(simple_model)

        simple_model.train()
        model_copy.eval()

        differ = Diff()
        result = differ._compare_pytorch_model(simple_model, model_copy, "model")

        assert result is not None
        assert '_training' in result.children

    def test_same_weights_after_copy(self, simple_model):
        """Copied model should have no diff from original."""
        from flowbook.kernel.deepcopy import deepcopy as flowbook_deepcopy

        model_copy = flowbook_deepcopy(simple_model)

        differ = Diff()
        result = differ._compare_pytorch_model(simple_model, model_copy, "model")
        assert result is None

    def test_modified_weights_after_copy(self, simple_model):
        """Modified weights after copy should be detected."""
        from flowbook.kernel.deepcopy import deepcopy as flowbook_deepcopy

        model_copy = flowbook_deepcopy(simple_model)

        # Modify original
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(1.0)

        differ = Diff()
        result = differ._compare_pytorch_model(simple_model, model_copy, "model")

        assert result is not None
        assert hasattr(result, 'children')
        param_diffs = [k for k in result.children.keys() if k.startswith('_param_')]
        assert len(param_diffs) > 0

    def test_tolerance_for_float_weights(self, simple_model):
        """Small numerical differences within tolerance should not be detected."""
        import copy
        model_copy = copy.deepcopy(simple_model)

        # Add very small noise (within tolerance)
        with torch.no_grad():
            for param in model_copy.parameters():
                param.add_(1e-10)

        differ = Diff()
        result = differ._compare_pytorch_model(simple_model, model_copy, "model")

        # Should be considered equal due to tolerance
        assert result is None


class TestPyTorchDiffNested:
    """Tests for PyTorch models in nested structures."""

    def test_model_in_dict(self, simple_model, another_model):
        """Models in dicts should be compared correctly."""
        dict_a = {'model': simple_model, 'epochs': 10}
        dict_b = {'model': another_model, 'epochs': 10}

        differ = Diff()
        result = differ._compare_values(dict_a, dict_b, "data")

        # Should detect the model difference
        assert result is not None

    def test_multiple_models(self, simple_model):
        """Multiple models should each be compared."""
        model_2 = nn.Linear(5, 3)

        dict_a = {'model1': simple_model, 'model2': model_2}
        # Create copies with same weights
        import copy
        dict_b = {
            'model1': copy.deepcopy(simple_model),
            'model2': copy.deepcopy(model_2)
        }

        differ = Diff()
        result = differ._compare_values(dict_a, dict_b, "data")

        # Should be equal (same architecture and weights)
        assert result is None


class TestPyTorchDiffCustomModules:
    """Tests for custom nn.Module subclasses."""

    def test_custom_module_comparison(self):
        """Custom modules should be compared by state_dict."""
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 1)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model_a = CustomModel()
        model_b = CustomModel()

        differ = Diff()
        result = differ._compare_pytorch_model(model_a, model_b, "model")

        # Different random weights
        assert result is not None

    def test_custom_module_same_weights(self):
        """Custom modules with same weights should have no diff."""
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)

            def forward(self, x):
                return self.fc(x)

        model_a = CustomModel()
        import copy
        model_b = copy.deepcopy(model_a)

        differ = Diff()
        result = differ._compare_pytorch_model(model_a, model_b, "model")

        assert result is None


class TestPyTorchDiffDispatch:
    """Tests for dispatch mechanism with PyTorch models."""

    def test_dispatch_detects_pytorch_model(self, simple_model):
        """Dispatch should correctly route PyTorch models."""
        import copy
        model_copy = copy.deepcopy(simple_model)

        differ = Diff()
        # Use the main compare_values which goes through dispatch
        result = differ._compare_values(simple_model, model_copy, "model")

        # Same model should have no diff
        assert result is None

    def test_dispatch_works_for_modified_model(self, simple_model):
        """Dispatch should work for modified models."""
        import copy
        model_copy = copy.deepcopy(simple_model)

        # Modify copy
        with torch.no_grad():
            for param in model_copy.parameters():
                param.add_(100.0)

        differ = Diff()
        result = differ._compare_values(simple_model, model_copy, "model")

        # Should detect difference
        assert result is not None
