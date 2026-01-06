"""
Tests for Keras opaque object handling with custom attribute capture.

HISTORY: These tests were originally written to demonstrate correctness gaps
where custom __dict__ attributes were NOT preserved through checkpoint/restore.

UPDATE: With the enhanced KerasModelHandler that captures custom __dict__
attributes, most of these tests should now PASS. Tests that still fail
document truly unfixable limitations (multi-GPU context, etc.).

Categories:
1. FIXED: Custom model/layer __dict__ attributes - NOW PRESERVED
2. PARTIALLY FIXED: Some layer state restored via __dict__ capture
3. STILL GAPS: Multi-GPU strategy context, device placement (external to model)
"""

import pytest
import numpy as np

keras = pytest.importorskip("keras")

from data_ferret.kernel.opaque import OpaqueRegistry, reset_keras_handler
from data_ferret.kernel.deepcopy import deepcopy as ferret_deepcopy


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the opaque registry before each test."""
    OpaqueRegistry.clear()
    reset_keras_handler()
    yield
    OpaqueRegistry.clear()
    reset_keras_handler()


# =============================================================================
# FIXED: Custom State Now Captured via __dict__
# =============================================================================
# These tests previously failed because custom __dict__ attributes were not
# captured. Now they should PASS because KerasModelHandler captures and
# restores custom model and layer __dict__ attributes.

# Register all custom models at module level so Keras can find them
@keras.saving.register_keras_serializable(package='test_gaps_models')
class CountingModel(keras.Model):
    """Model that counts how many times it's been called."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(1)
        self.call_count = 0  # Custom state - NOT in get_config()!

    def call(self, inputs):
        # Note: Keras may call this multiple times internally, so we use
        # an explicit increment method instead
        return self.dense(inputs)

    def increment_count(self):
        """Explicitly increment the call count."""
        self.call_count += 1

    def get_config(self):
        config = super().get_config()
        # Note: call_count is NOT included - it's runtime state
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='test_gaps_models')
class ModelWithHistory(keras.Model):
    """Model that tracks its own loss history."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(1)
        self.loss_history = []  # Custom state for tracking losses

    def call(self, inputs):
        return self.dense(inputs)

    def record_loss(self, loss):
        self.loss_history.append(loss)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='test_gaps_models')
class CurriculumModel(keras.Model):
    """Model with curriculum learning state."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(1)
        self.difficulty_level = 1  # Starts easy
        self.samples_seen = 0

    def call(self, inputs):
        return self.dense(inputs)

    def update_curriculum(self, num_samples):
        """Update curriculum state after processing samples."""
        self.samples_seen += num_samples
        # Increase difficulty every 100 samples
        self.difficulty_level = 1 + self.samples_seen // 100

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='test_gaps_models')
class ModelWithGradientScaling(keras.Model):
    """Model with custom gradient scaling factor."""

    def __init__(self, gradient_scale=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(1)
        self.gradient_scale = gradient_scale  # Custom training parameter

    def call(self, inputs):
        return self.dense(inputs)

    def get_config(self):
        config = super().get_config()
        # BUG: gradient_scale should be in config but we "forgot"
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TestCustomStateGap:
    """
    Tests for custom state captured via __dict__.

    FIXED: These tests should now PASS because KerasModelHandler captures
    custom __dict__ attributes on both models and layers.

    Previously this was a CORRECTNESS issue - now it's fixed.
    """

    def test_subclassed_model_with_counter(self):
        """
        Subclassed model with a counter attribute loses counter on deepcopy.

        This demonstrates that custom Python attributes on model subclasses
        are NOT preserved through checkpoint/restore.
        """
        model = CountingModel()
        # Build the model by calling it
        X = np.random.randn(10, 5).astype(np.float32)
        model(X)

        # Explicitly increment counter (simulating training iterations)
        for _ in range(5):
            model.increment_count()

        assert model.call_count == 5, "Sanity check: counter should be 5"

        # Checkpoint via deepcopy
        model_copy = ferret_deepcopy(model)

        # CORRECTNESS ISSUE: counter is lost!
        # The copy should have the same call_count as original
        assert model_copy.call_count == 5, (
            f"CORRECTNESS GAP: Custom state lost! "
            f"Original call_count=5, copy call_count={model_copy.call_count}"
        )

    def test_model_with_training_history_attribute(self):
        """
        Model with custom training history attribute loses it on deepcopy.

        Common pattern: storing training metrics on the model itself.
        """
        model = ModelWithHistory()
        # Build the model
        X = np.random.randn(10, 5).astype(np.float32)
        model(X)

        # Simulate recording training losses
        model.record_loss(1.5)
        model.record_loss(1.2)
        model.record_loss(0.9)

        assert len(model.loss_history) == 3, "Sanity check"

        # Checkpoint
        model_copy = ferret_deepcopy(model)

        # CORRECTNESS ISSUE: history is lost!
        assert len(model_copy.loss_history) == 3, (
            f"CORRECTNESS GAP: Training history lost! "
            f"Original has 3 entries, copy has {len(model_copy.loss_history)}"
        )

    def test_model_with_adaptive_learning_state(self):
        """
        Model with adaptive state (e.g., curriculum learning step) loses state.

        This is a real-world pattern where models adjust behavior based on
        training progress.
        """
        model = CurriculumModel()

        # Build model
        X = np.random.randn(10, 5).astype(np.float32)
        model(X)

        # Simulate training for a while (250 samples across batches)
        model.update_curriculum(100)
        model.update_curriculum(100)
        model.update_curriculum(50)

        assert model.samples_seen == 250
        assert model.difficulty_level == 3  # 1 + 250//100 = 3

        # Checkpoint
        model_copy = ferret_deepcopy(model)

        # CORRECTNESS ISSUE: curriculum state is lost!
        # Model would restart from difficulty_level=1
        assert model_copy.difficulty_level == 3, (
            f"CORRECTNESS GAP: Curriculum state lost! "
            f"Original difficulty=3, copy difficulty={model_copy.difficulty_level}"
        )
        assert model_copy.samples_seen == 250, (
            f"CORRECTNESS GAP: samples_seen lost! "
            f"Original=250, copy={model_copy.samples_seen}"
        )


# =============================================================================
# FIXED: Custom Training Behavior Attributes Now Captured
# =============================================================================

class TestHooksGap:
    """
    Tests for custom training behavior attributes.

    FIXED: Custom attributes like gradient_scale are now captured via __dict__.

    Note: Actual Keras callbacks (like keras.callbacks.EarlyStopping) are
    separate objects, not model attributes. They need to be checkpointed
    separately if needed.
    """

    def test_model_with_gradient_scaling_attribute(self):
        """
        Model with custom gradient_scale attribute loses it on deepcopy.

        This attribute would affect training behavior - after restore,
        gradients would use default scaling instead of custom value.
        """
        model = ModelWithGradientScaling(gradient_scale=0.1)
        # Build model
        X = np.random.randn(10, 5).astype(np.float32)
        model(X)

        # Change gradient scale
        model.gradient_scale = 0.01

        # Checkpoint
        model_copy = ferret_deepcopy(model)

        # CORRECTNESS ISSUE: gradient_scale is custom state, not preserved
        assert model_copy.gradient_scale == 0.01, (
            f"CORRECTNESS GAP: Custom gradient_scale lost! "
            f"Original=0.01, copy={model_copy.gradient_scale}"
        )

    def test_sequential_model_with_custom_layer_state(self):
        """
        Sequential model containing a layer with custom runtime state.

        Even layers inside Sequential models can have state that's lost.
        """
        @keras.saving.register_keras_serializable(package='test_gaps_layers')
        class StatefulDense(keras.layers.Dense):
            """Dense layer that tracks activation statistics."""

            def __init__(self, units, **kwargs):
                super().__init__(units, **kwargs)
                self.activation_sum = 0.0
                self.activation_count = 0

            def call(self, inputs):
                output = super().call(inputs)
                # Track statistics (runtime state)
                self.activation_sum += float(keras.ops.sum(output))
                self.activation_count += int(keras.ops.shape(output)[0])
                return output

            def get_mean_activation(self):
                if self.activation_count == 0:
                    return 0.0
                return self.activation_sum / self.activation_count

        model = keras.Sequential([
            keras.layers.Input(shape=(5,)),
            StatefulDense(10),
            keras.layers.Dense(1)
        ])

        # Run some data through to accumulate statistics
        X = np.random.randn(100, 5).astype(np.float32)
        model(X)

        original_mean = model.layers[0].get_mean_activation()
        original_count = model.layers[0].activation_count

        assert original_count == 100, "Sanity check"
        assert original_mean != 0.0, "Sanity check"

        # Checkpoint
        model_copy = ferret_deepcopy(model)

        # CORRECTNESS ISSUE: activation statistics are lost
        copy_count = model_copy.layers[0].activation_count
        assert copy_count == 100, (
            f"CORRECTNESS GAP: Layer runtime state lost! "
            f"Original activation_count={original_count}, copy={copy_count}"
        )


# =============================================================================
# PARTIALLY FIXED: Custom Layer Attributes via __dict__
# =============================================================================

class TestCustomLayerConfigGap:
    """
    Tests for custom layer attributes not in get_config().

    PARTIALLY FIXED: Layer custom __dict__ attributes are now captured.
    However, attributes that rely on get_config() for clone_model() may
    still have issues if the cloned architecture differs from original.

    Best practice: Always include all attributes in get_config() AND
    register with @keras.saving.register_keras_serializable.
    """

    def test_custom_layer_missing_config_attribute(self):
        """
        Custom layer that doesn't include all attributes in get_config().

        This is a common mistake - users add attributes but forget to
        include them in get_config(), so clone_model() loses them.
        """
        @keras.saving.register_keras_serializable(package='test_gaps_partial')
        class PartialConfigLayer(keras.layers.Layer):
            def __init__(self, units, dropout_rate=0.5, use_special_init=False, **kwargs):
                super().__init__(**kwargs)
                self.units = units
                self.dropout_rate = dropout_rate
                self.use_special_init = use_special_init  # FORGOT to add to get_config!

            def build(self, input_shape):
                initializer = 'zeros' if self.use_special_init else 'glorot_uniform'
                self.w = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer=initializer,
                    trainable=True
                )
                self.built = True

            def call(self, inputs):
                return inputs @ self.w

            def get_config(self):
                config = super().get_config()
                config['units'] = self.units
                config['dropout_rate'] = self.dropout_rate
                # BUG: Forgot to include use_special_init!
                return config

        # Create model with special initialization
        model = keras.Sequential([
            keras.layers.Input(shape=(5,)),
            PartialConfigLayer(10, use_special_init=True)
        ])

        assert model.layers[0].use_special_init == True, "Sanity check"

        # Checkpoint
        model_copy = ferret_deepcopy(model)

        # CORRECTNESS ISSUE: use_special_init defaults to False in copy
        assert model_copy.layers[0].use_special_init == True, (
            f"CORRECTNESS GAP: use_special_init lost! "
            f"Original=True, copy={model_copy.layers[0].use_special_init}. "
            f"This happened because get_config() doesn't include use_special_init."
        )

    def test_custom_layer_with_runtime_computed_state(self):
        """
        Custom layer with attribute computed at runtime, not in config.

        Some attributes are computed during __init__ and not serializable.
        """
        @keras.saving.register_keras_serializable(package='test_gaps_computed')
        class LayerWithComputedState(keras.layers.Layer):
            def __init__(self, base_units, multiplier=2, **kwargs):
                super().__init__(**kwargs)
                self.base_units = base_units
                self.multiplier = multiplier
                # Computed attribute - how do we serialize this?
                self.effective_units = base_units * multiplier
                self._initialization_timestamp = np.random.randint(0, 1000000)

            def build(self, input_shape):
                self.w = self.add_weight(
                    shape=(input_shape[-1], self.effective_units),
                    initializer='glorot_uniform',
                    trainable=True
                )
                self.built = True

            def call(self, inputs):
                return inputs @ self.w

            def get_config(self):
                config = super().get_config()
                config['base_units'] = self.base_units
                config['multiplier'] = self.multiplier
                # effective_units is computed, so we don't include it
                # _initialization_timestamp is runtime state, not config
                return config

        model = keras.Sequential([
            keras.layers.Input(shape=(10,)),
            LayerWithComputedState(5, multiplier=3)
        ])

        original_timestamp = model.layers[0]._initialization_timestamp

        # Checkpoint
        model_copy = ferret_deepcopy(model)

        # effective_units should be recomputed correctly (5*3=15)
        assert model_copy.layers[0].effective_units == 15, (
            "effective_units should be recomputed correctly"
        )

        # But timestamp is runtime state that gets reset
        # This might be OK or not depending on use case
        copy_timestamp = model_copy.layers[0]._initialization_timestamp

        # This test documents the behavior - timestamp IS different
        # Whether this is a "gap" depends on whether timestamp matters
        assert copy_timestamp == original_timestamp, (
            f"CORRECTNESS GAP: Runtime state _initialization_timestamp changed! "
            f"Original={original_timestamp}, copy={copy_timestamp}. "
            f"This may or may not be a problem depending on use case."
        )

    def test_unregistered_custom_layer_state_lost(self):
        """
        Custom layer that isn't registered - custom state is lost.

        Even if Keras manages to copy it, custom attributes won't persist.
        """
        # Deliberately NOT using @keras.saving.register_keras_serializable
        class UnregisteredLayerWithState(keras.layers.Layer):
            def __init__(self, units, **kwargs):
                super().__init__(**kwargs)
                self.units = units
                self.custom_scale = 2.5  # Custom attribute

            def build(self, input_shape):
                self.w = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer='glorot_uniform',
                    trainable=True
                )
                self.built = True

            def call(self, inputs):
                return (inputs @ self.w) * self.custom_scale

        model = keras.Sequential([
            keras.layers.Input(shape=(5,)),
            UnregisteredLayerWithState(10)
        ])

        # Modify the custom scale
        model.layers[0].custom_scale = 5.0

        try:
            model_copy = ferret_deepcopy(model)

            # If copy succeeded, check if custom_scale is preserved
            assert model_copy.layers[0].custom_scale == 5.0, (
                f"CORRECTNESS GAP: custom_scale lost! "
                f"Original=5.0, copy={model_copy.layers[0].custom_scale}"
            )
        except (TypeError, ValueError) as e:
            # If it failed, that's also a gap - can't checkpoint at all
            pytest.fail(
                f"CORRECTNESS GAP: Cannot checkpoint model with unregistered layer. "
                f"Error: {e}"
            )


# =============================================================================
# Gap 1: Multi-GPU / Distributed Wrappers
# =============================================================================

class TestMultiGPUGap:
    """
    Tests for multi-GPU and distributed training wrappers.

    These can cause correctness issues because:
    1. The wrapper itself may have state
    2. Device placement may be lost
    3. Strategy configuration may not persist
    """

    def test_mirrored_strategy_model(self):
        """
        Model created under MirroredStrategy loses strategy context.

        After checkpoint/restore, the model is no longer distributed.
        """
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not available")

        # Check if we have multiple devices (or can simulate)
        # Even with 1 GPU, MirroredStrategy works
        try:
            strategy = tf.distribute.MirroredStrategy()
        except Exception as e:
            pytest.skip(f"MirroredStrategy not available: {e}")

        with strategy.scope():
            model = keras.Sequential([
                keras.layers.Input(shape=(5,)),
                keras.layers.Dense(10, activation='relu'),
                keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

        # Verify model knows about strategy
        assert hasattr(model, 'distribute_strategy'), "Model should have distribute_strategy"

        # Checkpoint
        model_copy = ferret_deepcopy(model)

        # CORRECTNESS ISSUE: Strategy context may be lost
        # The copy might not be aware it was created under a strategy
        original_strategy = getattr(model, 'distribute_strategy', None)
        copy_strategy = getattr(model_copy, 'distribute_strategy', None)

        # Check if strategies match
        if original_strategy is not None:
            assert copy_strategy is not None, (
                "CORRECTNESS GAP: distribute_strategy lost after checkpoint! "
                "Model was created under MirroredStrategy but copy has no strategy."
            )

            # Even if both have strategies, check if they're the same type
            assert type(copy_strategy).__name__ == type(original_strategy).__name__, (
                f"CORRECTNESS GAP: Strategy type changed! "
                f"Original={type(original_strategy).__name__}, "
                f"Copy={type(copy_strategy).__name__}"
            )

    def test_model_device_placement(self):
        """
        Model with explicit device placement may lose device info.

        After checkpoint, model might end up on different device.
        """
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not available")

        # Create model on specific device
        try:
            # Try to place on GPU if available, otherwise CPU
            devices = tf.config.list_physical_devices('GPU')
            if devices:
                device = '/GPU:0'
            else:
                device = '/CPU:0'

            with tf.device(device):
                model = keras.Sequential([
                    keras.layers.Input(shape=(5,)),
                    keras.layers.Dense(10),
                    keras.layers.Dense(1)
                ])
                # Build the model
                model.build((None, 5))
        except Exception as e:
            pytest.skip(f"Device placement test not possible: {e}")

        # Get device info - Keras 3 uses different API
        try:
            # Try TensorFlow-style device access
            original_device = str(model.layers[0].kernel.device)
        except AttributeError:
            # Keras 3 with different backend - skip device test
            pytest.skip("Device info not accessible in this Keras version")

        # Checkpoint
        model_copy = ferret_deepcopy(model)

        # Check device of copied weight
        try:
            copy_device = str(model_copy.layers[0].kernel.device)
        except AttributeError:
            pytest.skip("Device info not accessible after copy")

        # Document behavior: device may or may not be preserved
        # This is informational - TensorFlow usually handles this correctly
        if original_device != copy_device:
            pytest.fail(
                f"CORRECTNESS GAP: Device placement changed! "
                f"Original device={original_device}, copy device={copy_device}. "
                f"This could cause issues in multi-device setups."
            )

    def test_model_with_distribution_state(self):
        """
        Model with custom distribution-related attributes loses them.

        Real-world example: custom batch splitting or gradient aggregation state.
        """
        @keras.saving.register_keras_serializable(package='test_gaps_distributed')
        class DistributedAwareModel(keras.Model):
            """Model that tracks distribution-related state."""

            def __init__(self, num_replicas=1, **kwargs):
                super().__init__(**kwargs)
                self.dense = keras.layers.Dense(1)
                self.num_replicas = num_replicas
                self.local_batch_size = 32
                self.global_batch_size = self.local_batch_size * num_replicas
                self.gradient_accumulation_steps = 0

            def call(self, inputs):
                return self.dense(inputs)

            def accumulate_gradients(self):
                """Simulate gradient accumulation step."""
                self.gradient_accumulation_steps += 1

            def get_config(self):
                config = super().get_config()
                config['num_replicas'] = self.num_replicas
                # Note: gradient_accumulation_steps is runtime state, not config
                return config

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        model = DistributedAwareModel(num_replicas=4)
        X = np.random.randn(10, 5).astype(np.float32)
        model(X)

        # Simulate some gradient accumulation
        for _ in range(10):
            model.accumulate_gradients()

        assert model.gradient_accumulation_steps == 10
        assert model.global_batch_size == 128  # 32 * 4

        # Checkpoint
        model_copy = ferret_deepcopy(model)

        # CORRECTNESS GAP: gradient_accumulation_steps is lost
        assert model_copy.gradient_accumulation_steps == 10, (
            f"CORRECTNESS GAP: gradient_accumulation_steps lost! "
            f"Original=10, copy={model_copy.gradient_accumulation_steps}"
        )

        # num_replicas should be preserved (it's in get_config)
        assert model_copy.num_replicas == 4, (
            f"num_replicas should be preserved: got {model_copy.num_replicas}"
        )


# =============================================================================
# Bonus: Test that demonstrates weights ARE preserved (not a gap)
# =============================================================================

class TestWeightsArePreserved:
    """Verify that model weights ARE correctly preserved (this should pass)."""

    def test_weights_preserved_on_sequential(self):
        """Standard Sequential model weights should be preserved."""
        model = keras.Sequential([
            keras.layers.Input(shape=(5,)),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(1)
        ])

        # Set some specific weights
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(np.ones_like(layer.kernel.numpy()) * 0.5)

        # Checkpoint
        model_copy = ferret_deepcopy(model)

        # Weights should be identical
        for orig_layer, copy_layer in zip(model.layers, model_copy.layers):
            if hasattr(orig_layer, 'kernel'):
                np.testing.assert_array_almost_equal(
                    orig_layer.kernel.numpy(),
                    copy_layer.kernel.numpy(),
                    err_msg="Weights should be preserved"
                )

        # Modifications should be independent
        if hasattr(model.layers[0], 'kernel'):
            model.layers[0].kernel.assign(np.zeros_like(model.layers[0].kernel.numpy()))
            assert not np.allclose(
                model.layers[0].kernel.numpy(),
                model_copy.layers[0].kernel.numpy()
            ), "Copy should be independent"


# =============================================================================
# Summary: Fixed vs Remaining Gaps
# =============================================================================

class TestSummary:
    """Summary of what's fixed and what remains as gaps."""

    def test_print_gap_summary(self):
        """Print a summary of fixed items and remaining gaps."""
        fixed = [
            "FIXED: Subclassed model custom attributes (call_count, loss_history)",
            "FIXED: Curriculum/adaptive learning state on model",
            "FIXED: Custom gradient scaling factors on model",
            "FIXED: Layer custom runtime statistics (activation tracking)",
            "FIXED: Custom layer __dict__ attributes (even without get_config)",
            "FIXED: Runtime-computed layer state preserved via __dict__",
            "FIXED: Unregistered custom layer state (via __dict__ capture)",
        ]

        remaining_gaps = [
            "GAP: MirroredStrategy context is external to model - not preserved",
            "GAP: Device placement managed by TensorFlow - may change",
            "GAP: Keras callbacks (EarlyStopping, etc.) are separate objects",
            "GAP: Optimizer state requires separate handling",
        ]

        print("\n" + "=" * 70)
        print("KERAS OPAQUE OBJECT HANDLING STATUS")
        print("=" * 70)
        print("\nFIXED (via __dict__ capture):")
        for item in fixed:
            print(f"  + {item}")
        print("\nREMAINING GAPS (external to model):")
        for gap in remaining_gaps:
            print(f"  - {gap}")
        print("=" * 70)
        print("\nPyTorch has similar fixed/remaining patterns.")
        print("=" * 70 + "\n")

        # This test always passes - it's just for documentation
        assert True
