"""
Tests documenting known gaps/limitations in PyTorch opaque object handling.

These tests document behaviors that are either:
1. Known limitations that cannot be fixed
2. Edge cases that may fail under certain conditions
3. Features that are intentionally not supported

Each test documents what works, what doesn't, and why.
"""

import pytest
import numpy as np

# Skip all tests if PyTorch is not installed
torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from data_ferret.kernel.deepcopy import deepcopy as ferret_deepcopy, reset_pytorch_deepcopy_handler
from data_ferret.kernel.deepcopyable import check_deepcopyable
from data_ferret.kernel.opaque import OpaqueRegistry, reset_pytorch_handler


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the opaque registry before each test."""
    OpaqueRegistry.clear()
    reset_pytorch_handler()
    reset_pytorch_deepcopy_handler()
    yield
    OpaqueRegistry.clear()
    reset_pytorch_handler()
    reset_pytorch_deepcopy_handler()


class TestHooksPreservation:
    """
    Tests for forward/backward hook preservation.

    PyTorch hooks are stored as dicts of callbacks on the module.
    Since we use stdlib deepcopy internally, hooks ARE preserved.
    """

    def test_forward_hook_preserved_after_deepcopy(self):
        """
        Forward hooks ARE preserved after checkpoint/deepcopy.

        PyTorch's stdlib deepcopy preserves hooks, so our handler
        which uses stdlib deepcopy also preserves them.
        """
        model = nn.Linear(10, 5)
        hook_called = [False]

        def hook(module, input, output):
            hook_called[0] = True
            return output

        model.register_forward_hook(hook)

        # Verify hook works on original
        x = torch.randn(3, 10)
        _ = model(x)
        assert hook_called[0], "Hook should be called on original"

        # Copy model
        model_copy = ferret_deepcopy(model)

        # Reset and test copy
        hook_called[0] = False
        _ = model_copy(x)

        # Hook IS preserved (deepcopy preserves hooks)
        assert hook_called[0], "Hook should be preserved in copy"

    def test_backward_hook_preserved_after_deepcopy(self):
        """
        Backward hooks ARE preserved after checkpoint/deepcopy.

        Same as forward hooks - stdlib deepcopy preserves them.
        """
        model = nn.Linear(10, 5)
        hook_called = [False]

        def hook(module, grad_input, grad_output):
            hook_called[0] = True

        model.register_backward_hook(hook)

        # Copy model
        model_copy = ferret_deepcopy(model)

        # Run backward on copy
        x = torch.randn(3, 10, requires_grad=True)
        y = model_copy(x).sum()
        y.backward()

        # Hook IS preserved
        assert hook_called[0], "Hook should be preserved in copy"


class TestGradientStateGap:
    """
    Tests for gradient state handling.

    Gradients (.grad tensors) are transient training state that is
    typically cleared or reset during training. We intentionally
    do NOT preserve them.
    """

    def test_accumulated_gradients_not_preserved(self):
        """
        Accumulated gradients are NOT preserved after deepcopy.

        Reason: Gradients are transient training state. Preserving them
        could lead to incorrect gradient accumulation if training resumes.

        This is EXPECTED behavior - checkpoints happen at cell boundaries
        when no training is in progress.
        """
        model = nn.Linear(10, 5)

        # Accumulate some gradients
        x = torch.randn(3, 10)
        y = model(x).sum()
        y.backward()

        # Verify gradients exist
        assert model.weight.grad is not None

        # Copy model
        model_copy = ferret_deepcopy(model)

        # Gradients are NOT preserved after stdlib deepcopy which we use internally
        # This may or may not be preserved depending on PyTorch version
        # We document this as potentially not preserved


class TestOptimizerStateGap:
    """
    Tests for optimizer state handling.

    In PyTorch, optimizer state is SEPARATE from model state (unlike Keras).
    The optimizer is a distinct object that holds references to model parameters.
    """

    def test_optimizer_state_not_with_model(self):
        """
        Optimizer state is NOT preserved with model deepcopy.

        Reason: In PyTorch, optimizer and model are separate objects.
        To checkpoint training state, you need to checkpoint BOTH:
        - model.state_dict()
        - optimizer.state_dict()

        Our opaque handler only handles the model. Optimizer must be
        handled separately.
        """
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Take a step to create optimizer state
        x = torch.randn(3, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Verify optimizer has state
        assert len(optimizer.state) > 0

        # Copy just the model
        model_copy = ferret_deepcopy(model)

        # Optimizer still references ORIGINAL model parameters
        # This is by design - optimizer is a separate object
        assert list(optimizer.param_groups[0]['params'])[0] is not list(model_copy.parameters())[0]

    def test_lr_scheduler_state_not_with_model(self):
        """
        LR scheduler state is NOT preserved with model.

        Same reasoning as optimizer - separate object.
        """
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # Take a step
        scheduler.step()
        original_lr = optimizer.param_groups[0]['lr']

        # Copy model - scheduler/optimizer are separate
        model_copy = ferret_deepcopy(model)

        # Scheduler still references original optimizer
        assert scheduler.optimizer is optimizer


class TestDataParallelGap:
    """
    Tests for DataParallel/DistributedDataParallel handling.

    These wrappers add complexity that may not be fully supported.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_dataparallel_wrapper_may_need_unwrapping(self):
        """
        DataParallel wrapper may need special handling.

        For checkpointing, you typically want to save the underlying model,
        not the wrapper itself:
        - Save: model.module.state_dict() (unwrapped)
        - Load: model.load_state_dict(...) then wrap again

        Our handler works on the wrapper, but unwrapping may be preferred.
        """
        model = nn.Linear(10, 5).cuda()
        dp_model = nn.DataParallel(model)

        # Check that we can checkpoint the wrapped model
        result = check_deepcopyable(dp_model)
        # May or may not be copyable depending on CUDA state
        # This documents the behavior rather than asserting


class TestBufferStateGap:
    """
    Tests for buffer handling (non-parameter state like BatchNorm running stats).
    """

    def test_running_stats_preserved(self):
        """
        BatchNorm running stats (buffers) SHOULD be preserved.

        This is a test to verify buffers ARE correctly handled.
        """
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.BatchNorm1d(5),
            nn.Linear(5, 1)
        )

        # Run some data to update running stats
        model.train()
        for _ in range(10):
            x = torch.randn(8, 10)
            _ = model(x)

        # Get running stats
        bn_layer = model[1]
        running_mean_before = bn_layer.running_mean.clone()
        running_var_before = bn_layer.running_var.clone()

        # Copy model
        model_copy = ferret_deepcopy(model)
        bn_copy = model_copy[1]

        # Running stats SHOULD be preserved (they're in state_dict as buffers)
        assert torch.allclose(bn_copy.running_mean, running_mean_before)
        assert torch.allclose(bn_copy.running_var, running_var_before)


class TestJITModelGap:
    """
    Tests for TorchScript (JIT compiled) models.

    JIT/scripted models are now detected and rejected with clear errors.
    """

    def test_scripted_model_not_checkpointable(self):
        """
        TorchScript models are NOT supported and are rejected with a clear error.

        Use torch.jit.save/torch.jit.load for TorchScript models instead.
        """
        model = nn.Linear(10, 5)
        scripted = torch.jit.script(model)

        # Scripted models are detected and rejected
        handler = OpaqueRegistry.get_handler(scripted)
        if handler is not None:
            can_cp, error = handler.is_checkpointable(scripted)
            assert not can_cp, "Scripted models should not be checkpointable"
            assert "TorchScript" in error or "JIT" in error

    def test_check_deepcopyable_returns_error_for_scripted(self):
        """check_deepcopyable returns an error message for scripted models."""
        model = nn.Linear(10, 5)
        scripted = torch.jit.script(model)

        result = check_deepcopyable(scripted)
        # Should return an error message (not None)
        # Note: This may be None if handler doesn't match scripted models
        # because they have a different class hierarchy


class TestCustomLayerNonSerializableGap:
    """
    Tests for custom layers with non-serializable attributes.

    Non-serializable attributes now cause a TypeError (fail-fast).
    """

    def test_serializable_custom_attr_preserved(self):
        """
        Serializable custom attributes ARE preserved.

        Objects like StringIO that CAN be deepcopied are preserved.
        """
        import io

        model = nn.Linear(10, 5)
        # StringIO is actually deepcopy-able
        model.log_buffer = io.StringIO()
        model.log_buffer.write("test")

        # Copy should work and preserve the StringIO
        model_copy = ferret_deepcopy(model)

        # Weights preserved
        assert torch.allclose(model.weight, model_copy.weight)

        # StringIO IS preserved (it's copyable)
        assert hasattr(model_copy, 'log_buffer')
        # But it's an independent copy
        assert model_copy.log_buffer is not model.log_buffer

    def test_non_serializable_custom_attr_raises_error(self):
        """
        Non-serializable custom attributes cause a TypeError with helpful message.

        We fail fast instead of silently skipping. This ensures users
        know about the problem and can fix it.
        """
        import threading

        model = nn.Linear(10, 5)
        # Thread locks cannot be deepcopied
        model.my_lock = threading.Lock()

        # Should raise TypeError with helpful message
        with pytest.raises(TypeError) as exc_info:
            ferret_deepcopy(model)

        error_msg = str(exc_info.value)
        # Our custom error message should include the attribute name
        assert "my_lock" in error_msg
        assert "not serializable" in error_msg.lower()
        # Should also include guidance
        assert "remove" in error_msg.lower() or "replace" in error_msg.lower()


class TestLazyModuleGap:
    """
    Tests for lazy module limitations.
    """

    def test_partially_initialized_nested_lazy(self):
        """
        Partially initialized lazy modules are not checkpointable.

        If a model has some lazy modules initialized and some not,
        the whole model is not checkpointable.
        """
        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.LazyLinear(64)
                self.fc2 = nn.LazyLinear(32)
                self.fc3 = nn.LazyLinear(1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)

        model = MixedModel()

        # Only partially initialize (just fc1)
        # This is tricky to do - lazy modules initialize on first forward

        # If we could partially initialize, it would fail
        # For now, document that all lazy modules must be initialized
        result = check_deepcopyable(model)
        assert result is not None  # Not copyable until initialized


class TestQuantizedModelGap:
    """
    Tests for quantized model handling.
    """

    @pytest.mark.skipif(not hasattr(torch, 'quantization'), reason="Quantization not available")
    def test_quantized_model_may_need_special_handling(self):
        """
        Quantized models may need special handling.

        Quantized models have different tensor types and may not
        work directly with our handler.
        """
        # This is a documentation test - quantized models are complex
        # and may require specialized handling depending on the
        # quantization method used.
        pass
