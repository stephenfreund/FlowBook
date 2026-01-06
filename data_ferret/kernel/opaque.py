"""
Opaque object handling for checkpoint/diff operations.

================================================================================
OVERVIEW
================================================================================

This module provides infrastructure for handling "opaque" objects - objects with
complex internal structures that should not be deeply traversed for alias detection
or standard deepcopy. Instead, these objects are treated as atomic units with
extractable mutable state.

Examples of opaque objects:
- Keras models (millions of internal TensorFlow objects, but only weights change)
- PyTorch models (similar complexity)
- Database connections (can't be copied, should be excluded)

The key insight is that many complex objects have:
1. Immutable structure (architecture, configuration)
2. Mutable state (weights, parameters)
3. Custom attributes (user-defined state on the object)

By extracting mutable state AND custom attributes, we can:
- Reduce alias tracking from O(millions) to O(1) for internal structure
- Still traverse custom attributes for alias detection
- Make deepcopy fast via structure sharing + state copying
- Preserve user-defined attributes that aren't part of the framework's state
- Provide clear extension points for new object types

================================================================================
SUPPORTED FEATURES
================================================================================

Keras Models (KerasModelHandler)
--------------------------------
- Sequential, Functional, and Model subclasses
- Model weights via get_weights() / set_weights()
- Optimizer state (best-effort, may fail for some optimizers)
- Custom model-level __dict__ attributes (e.g., model.my_counter = 0)
- Custom layer-level __dict__ attributes (e.g., model.layers[0].my_scale = 2.0)
- Alias tracking for custom attributes (detects shared references)

PyTorch Models (PyTorchModelHandler)
------------------------------------
- All nn.Module subclasses (Linear, Sequential, custom modules, etc.)
- Model state via state_dict() / load_state_dict()
- Training mode preservation (model.train() / model.eval())
- Device placement (best-effort restoration to original device)
- Forward/backward hooks (preserved via stdlib deepcopy)
- Custom module-level __dict__ attributes (e.g., model.custom_data = [...])
- Custom submodule __dict__ attributes (e.g., model.fc1.my_attr = 42)
- Alias tracking for custom attributes

================================================================================
KNOWN LIMITATIONS
================================================================================

Keras Limitations
-----------------
1. UNBUILT MODELS: Models must be built (have defined input shape) before
   checkpointing. Unbuilt models will raise an error.

2. DISTRIBUTE STRATEGY: Models created under tf.distribute.MirroredStrategy
   lose their strategy context after checkpoint. The strategy is external
   to the model object and cannot be captured.

3. DEVICE PLACEMENT: TensorFlow manages device placement. After restore,
   tensors may be on different devices depending on TF's auto-placement.

4. KERAS CALLBACKS: Callbacks like EarlyStopping are separate objects,
   not model attributes. They must be checkpointed separately if needed.

5. NON-SERIALIZABLE CUSTOM ATTRS: Custom attributes that can't be deepcopied
   (e.g., file handles, database connections) will cause a TypeError.
   Remove or replace these attributes before checkpointing.

6. OPTIMIZER STATE: Optimizer weight restoration is best-effort. Some
   optimizers with complex state may not restore perfectly.

PyTorch Limitations
-------------------
1. LAZY MODULES: Models with uninitialized lazy modules (e.g., LazyLinear
   that hasn't seen input) cannot be checkpointed. Run a forward pass first.

2. FORWARD/BACKWARD HOOKS: Hooks registered via register_forward_hook() or
   register_backward_hook() ARE preserved (stdlib deepcopy preserves them).
   Note: If hooks reference external state, the copy will share that state.

3. OPTIMIZER STATE: PyTorch optimizers are separate objects (by design).
   To checkpoint training state, checkpoint both model AND optimizer:
     - model.state_dict()
     - optimizer.state_dict()

4. GRADIENTS: Accumulated gradients (.grad tensors) are NOT preserved.
   They are transient training state typically cleared between steps.

5. DATAPARALLEL: DataParallel/DistributedDataParallel wrappers are NOT
   supported. Checkpointing will fail with a clear error directing you to
   checkpoint model.module (the underlying model) instead.

6. JIT/SCRIPTED MODELS: TorchScript (JIT compiled) models are NOT supported.
   Checkpointing will fail with a clear error. Checkpoint the original
   nn.Module before scripting, or use torch.jit.save/load.

7. QUANTIZED MODELS: Quantized models are NOT supported. Checkpointing
   will fail with a clear error. Checkpoint the model before quantization,
   or use torch.save/load for quantized models.

8. NON-SERIALIZABLE CUSTOM ATTRS: Custom attributes that can't be deepcopied
   (e.g., file handles, sockets) will cause a TypeError. Remove or replace
   these attributes before checkpointing.

================================================================================
ARCHITECTURE
================================================================================

OpaqueHandler (ABC)
    Abstract base class defining the handler interface:
    - can_handle(obj) -> bool: Check if handler applies
    - is_checkpointable(obj) -> (bool, error_msg): Validate checkpointability
    - get_mutable_state(obj) -> state: Extract copyable state
    - copy_with_state(obj, state, memo) -> copy: Create copy from state
    - states_equal(state1, state2) -> bool: Compare states
    - get_traversable_attrs(obj) -> dict: Attrs for alias detection

OpaqueRegistry
    Singleton registry managing handlers:
    - register(handler): Add a handler
    - get_handler(obj) -> handler: Get handler for object (triggers lazy reg)
    - is_opaque(obj) -> bool: Check if object has a handler

Lazy Registration
    Handlers are registered lazily on first encounter to avoid:
    - Expensive imports at module load time (TensorFlow ~3s, PyTorch ~1s)
    - Import errors when frameworks aren't installed
    - Matplotlib backend initialization issues

================================================================================
USAGE
================================================================================

Basic usage (automatic via deepcopy/checkpoint):

    from data_ferret.kernel.deepcopy import deepcopy

    # Keras
    model = keras.Sequential([...])
    model.custom_attr = 42  # Will be preserved!
    model_copy = deepcopy(model, {})

    # PyTorch
    model = nn.Linear(10, 5)
    model.my_data = [1, 2, 3]  # Will be preserved!
    model_copy = deepcopy(model, {})

Direct handler usage:

    from data_ferret.kernel.opaque import OpaqueRegistry

    handler = OpaqueRegistry.get_handler(model)
    if handler:
        can_cp, error = handler.is_checkpointable(model)
        if can_cp:
            state = handler.get_mutable_state(model)
            copy = handler.copy_with_state(model, state, {})

Adding custom handlers:

    class MyHandler(OpaqueHandler):
        def can_handle(self, obj):
            return isinstance(obj, MyComplexType)
        # ... implement other methods ...

    OpaqueRegistry.register(MyHandler())

================================================================================
TESTING
================================================================================

Test files:
- test_opaque_keras.py: Core Keras handler tests
- test_opaque_pytorch.py: Core PyTorch handler tests
- test_opaque_keras_gaps.py: Documents Keras limitations (many now fixed)
- test_opaque_pytorch_gaps.py: Documents PyTorch limitations
- test_keras_deepcopy.py: Keras deepcopy integration tests
- test_pytorch_diff.py: PyTorch diff comparison tests
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Tuple, Dict
import numpy as np


class OpaqueHandler(ABC):
    """
    Handler for objects that should not be deeply traversed.

    Implementations define how to:
    - Check if an object can be checkpointed
    - Extract mutable state (including custom attributes)
    - Create copies with given state
    - Compare states for equality
    - Return attributes that should be traversed for alias detection
    """

    @abstractmethod
    def can_handle(self, obj: Any) -> bool:
        """Return True if this handler applies to obj."""
        pass

    @abstractmethod
    def is_checkpointable(self, obj: Any) -> Tuple[bool, Optional[str]]:
        """
        Check if object can be checkpointed.

        Returns:
            Tuple of (can_checkpoint, error_message_if_not)
        """
        pass

    @abstractmethod
    def get_mutable_state(self, obj: Any) -> Any:
        """
        Extract the mutable state that needs to be copied.

        The returned state should be fully independent (deep copied)
        so that mutations to the original don't affect the state.

        This should include:
        - Framework-specific state (weights, parameters)
        - Custom user-defined attributes on the object
        """
        pass

    @abstractmethod
    def copy_with_state(self, obj: Any, state: Any, memo: dict) -> Any:
        """
        Create a copy of obj with the given state.

        The copy should:
        - Share immutable structure with original where possible
        - Have independent mutable state from the provided state
        - Restore custom user-defined attributes
        - Be registered in memo to handle circular references
        """
        pass

    @abstractmethod
    def states_equal(self, state1: Any, state2: Any) -> bool:
        """Compare two states for equality."""
        pass

    def get_alias_id(self, obj: Any) -> int:
        """
        Return a single ID to represent this object in alias tracking.

        Default: use the object's id(). Opaque objects are treated as atomic,
        so we don't recurse into their internals.
        """
        return id(obj)

    def get_traversable_attrs(self, obj: Any) -> Dict[str, Any]:
        """
        Return custom attributes that should be traversed for alias detection.

        These are user-defined attributes that may contain references to other
        objects in the namespace. They should be traversed to detect aliasing,
        even though the internal framework structure is skipped.

        Returns:
            Dict mapping attribute name/path to attribute value.
            Default: empty dict (fully opaque, no traversal).
        """
        return {}


class OpaqueRegistry:
    """
    Registry of handlers for opaque objects.

    Handlers are checked in registration order. First matching handler wins.
    """

    _handlers: List[OpaqueHandler] = []
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls):
        """Lazily initialize handlers to avoid import-time side effects."""
        if cls._initialized:
            return
        cls._initialized = True

        # Register built-in handlers
        # Keras handler is registered lazily when first Keras object is seen
        # to avoid importing TensorFlow at module load time

    @classmethod
    def register(cls, handler: OpaqueHandler):
        """Register a handler. Earlier handlers take priority."""
        cls._handlers.append(handler)

    @classmethod
    def get_handler(cls, obj: Any) -> Optional[OpaqueHandler]:
        """Get the handler for an object, or None if not opaque."""
        cls._ensure_initialized()

        # Check if this looks like a Keras model and register handler if needed
        _maybe_register_keras_handler(obj)

        # Check if this looks like a PyTorch model and register handler if needed
        _maybe_register_pytorch_handler(obj)

        for handler in cls._handlers:
            if handler.can_handle(obj):
                return handler
        return None

    @classmethod
    def is_opaque(cls, obj: Any) -> bool:
        """Check if an object is opaque (has a registered handler)."""
        return cls.get_handler(obj) is not None

    @classmethod
    def clear(cls):
        """Clear all handlers. Mainly for testing."""
        cls._handlers = []
        cls._initialized = False


# =============================================================================
# Keras Model Handler
# =============================================================================

# Keras internal attributes that should NOT be captured as custom state
KERAS_MODEL_INTERNALS = frozenset({
    # Core Keras model attributes
    'layers', 'inputs', 'outputs', 'input', 'output',
    'input_spec', 'output_names', 'input_names',
    'built', 'trainable', 'dtype', 'name',
    'optimizer', 'loss', 'metrics', 'compiled_loss', 'compiled_metrics',
    'distribute_strategy', 'run_eagerly', 'jit_compile',
    # Training-related
    'history', 'stop_training',
    # Common attribute names that are Keras-managed
    'supports_masking', 'stateful', 'dynamic',
})

KERAS_LAYER_INTERNALS = frozenset({
    # Core layer attributes
    'kernel', 'bias', 'weights', 'trainable_weights', 'non_trainable_weights',
    'input_spec', 'built', 'trainable', 'dtype', 'name',
    'supports_masking', 'stateful', 'dynamic',
    # Recurrent layer specifics
    'states', 'cell', 'return_sequences', 'return_state',
})


class KerasModelHandler(OpaqueHandler):
    """
    Handler for Keras Sequential and Functional models.

    Keras models have millions of internal TensorFlow objects for tracking,
    but only the weights actually change during training. This handler:

    1. Rejects unbuilt models (architecture not frozen yet)
    2. Extracts weights via get_weights() for checkpointing
    3. Captures custom user-defined attributes on model and layers
    4. Creates copies via clone_model() + set_weights() + restore custom attrs
    5. Compares models by weight and custom attribute equality
    """

    def can_handle(self, obj: Any) -> bool:
        """Check if obj is a Keras model."""
        type_name = type(obj).__name__
        module = type(obj).__module__ or ""

        if not (module.startswith("keras") or module.startswith("tensorflow.keras")):
            return False

        # Handle Sequential, Functional, and Model subclasses
        return type_name in ("Sequential", "Functional") or "Model" in type_name

    def is_checkpointable(self, obj: Any) -> Tuple[bool, Optional[str]]:
        """Check if model is built and can be checkpointed."""
        if not hasattr(obj, 'built'):
            return False, "Object does not have 'built' attribute"

        if not obj.built:
            return False, (
                f"Keras model '{type(obj).__name__}' is not built. "
                "Call model.build(input_shape) or run model.fit()/model.predict() first."
            )

        return True, None

    def _capture_custom_dict(self, obj: Any, internals: frozenset, obj_desc: str = "model") -> Dict[str, Any]:
        """
        Capture custom attributes from an object's __dict__.

        Filters out:
        - Private attributes (starting with _)
        - Known framework internals

        Raises:
            TypeError: If any custom attribute cannot be deepcopied
        """
        from data_ferret.kernel.deepcopy import deepcopy as ferret_deepcopy

        captured = {}
        for key, value in obj.__dict__.items():
            # Skip private attributes
            if key.startswith('_'):
                continue
            # Skip known internals
            if key in internals:
                continue
            # Attempt to deep copy - fail if not possible
            try:
                captured[key] = ferret_deepcopy(value, {})
            except Exception as e:
                raise TypeError(
                    f"Cannot checkpoint Keras {obj_desc}: custom attribute '{key}' "
                    f"(type: {type(value).__name__}) is not serializable. "
                    f"Remove or replace this attribute before checkpointing. "
                    f"Original error: {e}"
                ) from e

        return captured

    def _capture_model_dict(self, model: Any) -> Dict[str, Any]:
        """Capture custom attributes from model's __dict__."""
        return self._capture_custom_dict(model, KERAS_MODEL_INTERNALS)

    def _capture_layer_dicts(self, model: Any) -> List[Dict[str, Any]]:
        """Capture custom attributes from each layer's __dict__."""
        layer_dicts = []
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                layer_name = getattr(layer, 'name', f'layer_{i}')
                layer_dicts.append(
                    self._capture_custom_dict(layer, KERAS_LAYER_INTERNALS, f"layer '{layer_name}'")
                )
        return layer_dicts

    def get_mutable_state(self, obj: Any) -> Dict[str, Any]:
        """
        Extract weights, optimizer state, and custom attributes.

        Returns dict with:
        - 'weights': list of numpy arrays (deep copied)
        - 'optimizer_weights': optimizer state if compiled, else None
        - 'input_shape': for rebuilding
        - 'model_dict': custom model-level attributes
        - 'layer_dicts': custom layer-level attributes
        """
        # Deep copy weights (they're numpy arrays)
        weights = obj.get_weights()
        state = {
            'weights': [w.copy() for w in weights],
            'input_shape': obj.input_shape,
            'optimizer_weights': None,
            'optimizer_config': None,
            'loss': None,
            'metrics': None,
            # NEW: Capture custom attributes
            'model_dict': self._capture_model_dict(obj),
            'layer_dicts': self._capture_layer_dicts(obj),
        }

        # Capture optimizer state if model is compiled
        if hasattr(obj, 'optimizer') and obj.optimizer is not None:
            try:
                opt_weights = obj.optimizer.get_weights()
                if opt_weights:
                    state['optimizer_weights'] = [w.copy() for w in opt_weights]
                state['optimizer_config'] = obj.optimizer.get_config()
                state['loss'] = obj.loss
                # Get metric names for recompilation
                if obj.metrics:
                    state['metrics'] = [m.name if hasattr(m, 'name') else str(m)
                                       for m in obj.metrics]
            except Exception:
                pass  # Optimizer state capture is best-effort

        return state

    def copy_with_state(self, obj: Any, state: Dict[str, Any], memo: dict) -> Any:
        """
        Create a copy by cloning architecture and applying saved state.

        Strategy:
        1. Clone the model architecture (shares layer class definitions)
        2. Build with same input shape
        3. Set weights from state
        4. Restore custom model and layer attributes
        5. Optionally restore optimizer state
        """
        from data_ferret.kernel.deepcopy import deepcopy as ferret_deepcopy

        obj_id = id(obj)
        if obj_id in memo:
            return memo[obj_id]

        # Import here to avoid loading TensorFlow at module import time
        try:
            from tensorflow.keras.models import clone_model
        except ImportError:
            from keras.models import clone_model

        # Clone architecture (fast - just recreates layer structure)
        model_copy = clone_model(obj)

        # Build with same input shape
        if state['input_shape'] is not None:
            model_copy.build(state['input_shape'])

        # Restore weights
        model_copy.set_weights(state['weights'])

        # Register in memo before restoring custom attrs
        memo[obj_id] = model_copy

        # Restore custom model-level attributes
        if state.get('model_dict'):
            for key, value in state['model_dict'].items():
                try:
                    setattr(model_copy, key, ferret_deepcopy(value, {}))
                except Exception:
                    pass

        # Restore custom layer-level attributes
        if state.get('layer_dicts') and hasattr(model_copy, 'layers'):
            for i, layer_dict in enumerate(state['layer_dicts']):
                if i < len(model_copy.layers):
                    for key, value in layer_dict.items():
                        try:
                            setattr(model_copy.layers[i], key, ferret_deepcopy(value, {}))
                        except Exception:
                            pass

        # Optionally restore optimizer state
        if state.get('optimizer_config') is not None:
            try:
                # Get optimizer class
                opt_class = type(obj.optimizer)
                new_optimizer = opt_class.from_config(state['optimizer_config'])

                # Recompile
                model_copy.compile(
                    optimizer=new_optimizer,
                    loss=state['loss'],
                    metrics=state['metrics'],
                )

                # Restore optimizer weights if available
                if state['optimizer_weights'] is not None:
                    try:
                        model_copy.optimizer.set_weights(state['optimizer_weights'])
                    except Exception:
                        pass  # Optimizer weight restoration is best-effort
            except Exception:
                pass  # Compilation restoration is best-effort

        return model_copy

    def states_equal(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> bool:
        """Compare weights and custom attributes for equality."""
        # Compare weights
        weights1 = state1['weights']
        weights2 = state2['weights']

        if len(weights1) != len(weights2):
            return False

        for w1, w2 in zip(weights1, weights2):
            if w1.shape != w2.shape:
                return False
            if not np.allclose(w1, w2, rtol=1e-5, atol=1e-8):
                return False

        # Compare custom model attributes
        if state1.get('model_dict') != state2.get('model_dict'):
            return False

        # Compare custom layer attributes
        if state1.get('layer_dicts') != state2.get('layer_dicts'):
            return False

        return True

    def get_traversable_attrs(self, obj: Any) -> Dict[str, Any]:
        """
        Return custom attributes that should be traversed for alias detection.

        These are user-defined attributes on the model and its layers that
        may contain references to objects in the notebook namespace.
        """
        attrs = {}

        # Model-level custom attributes
        for key, value in obj.__dict__.items():
            if key.startswith('_'):
                continue
            if key in KERAS_MODEL_INTERNALS:
                continue
            attrs[key] = value

        # Layer-level custom attributes
        if hasattr(obj, 'layers'):
            for i, layer in enumerate(obj.layers):
                for key, value in layer.__dict__.items():
                    if key.startswith('_'):
                        continue
                    if key in KERAS_LAYER_INTERNALS:
                        continue
                    attrs[f"layers[{i}].{key}"] = value

        return attrs


# =============================================================================
# Lazy Registration
# =============================================================================

_keras_handler_registered = False


def _maybe_register_keras_handler(obj: Any) -> None:
    """Register Keras handler if obj looks like a Keras model."""
    global _keras_handler_registered
    if _keras_handler_registered:
        return

    module = type(obj).__module__ or ""
    if module.startswith("keras") or module.startswith("tensorflow.keras"):
        type_name = type(obj).__name__
        if type_name in ("Sequential", "Functional") or "Model" in type_name:
            OpaqueRegistry.register(KerasModelHandler())
            _keras_handler_registered = True


def reset_keras_handler():
    """Reset Keras handler registration. For testing."""
    global _keras_handler_registered
    _keras_handler_registered = False


# =============================================================================
# PyTorch Model Handler
# =============================================================================

# PyTorch internal attributes that should NOT be captured as custom state
PYTORCH_MODULE_INTERNALS = frozenset({
    # Core nn.Module attributes
    'training', 'forward',
    # Parameter/buffer related (managed by state_dict)
    'weight', 'bias',
})


class PyTorchModelHandler(OpaqueHandler):
    """
    Handler for PyTorch nn.Module objects.

    PyTorch models are simpler than Keras in some ways:
    - copy.deepcopy() works directly (unlike Keras)
    - state_dict() captures all parameters and buffers

    But have unique challenges:
    - Lazy modules that aren't initialized
    - Device management (CPU/GPU)
    - DataParallel wrappers
    - Forward/backward hooks (not serializable)

    This handler:
    1. Rejects uninitialized lazy modules
    2. Extracts state via state_dict() + custom attributes
    3. Creates copies via deepcopy + load_state_dict
    4. Preserves training mode and device placement
    """

    def can_handle(self, obj: Any) -> bool:
        """Check if obj is a PyTorch nn.Module.

        Handles both built-in torch modules (nn.Linear, nn.Sequential, etc.)
        and custom subclasses defined in user code.
        """
        # Check MRO for nn.Module base class
        # This handles both torch modules and custom subclasses
        for base in type(obj).__mro__:
            base_module = getattr(base, '__module__', '') or ''
            if base.__name__ == 'Module' and 'torch.nn' in base_module:
                return True

        return False

    def _is_scripted(self, obj: Any) -> bool:
        """Check if obj is a TorchScript (JIT compiled) model."""
        cls_name = type(obj).__name__
        # ScriptModule, RecursiveScriptModule, etc.
        if 'Script' in cls_name:
            return True
        # Check module path
        module = getattr(type(obj), '__module__', '') or ''
        if 'torch.jit' in module:
            return True
        return False

    def _is_quantized(self, obj: Any) -> bool:
        """Check if obj is a quantized model."""
        module = getattr(type(obj), '__module__', '') or ''
        if 'torch.ao.quantization' in module or 'torch.quantization' in module:
            return True
        # Check for quantized submodules
        for name, submodule in obj.named_modules():
            submodule_name = type(submodule).__name__
            # Common quantized layer patterns
            if any(q in submodule_name for q in ['Quantized', 'QLinear', 'QConv']):
                return True
        return False

    def _is_data_parallel(self, obj: Any) -> bool:
        """Check if obj is a DataParallel or DistributedDataParallel wrapper."""
        cls_name = type(obj).__name__
        return cls_name in ('DataParallel', 'DistributedDataParallel')

    def is_checkpointable(self, obj: Any) -> Tuple[bool, Optional[str]]:
        """
        Check if model is ready for checkpointing.

        Rejects:
        - JIT/scripted models (TorchScript)
        - Quantized models
        - DataParallel/DistributedDataParallel wrappers
        - Models with uninitialized lazy modules
        """
        # Check for JIT/scripted models
        if self._is_scripted(obj):
            return False, (
                f"TorchScript (JIT) models are not supported. "
                "Checkpoint the original nn.Module before scripting, or use "
                "torch.jit.save/torch.jit.load for TorchScript models."
            )

        # Check for quantized models
        if self._is_quantized(obj):
            return False, (
                f"Quantized models are not supported. "
                "Checkpoint the model before quantization, or use "
                "torch.save/torch.load for quantized models."
            )

        # Check for DataParallel wrappers
        if self._is_data_parallel(obj):
            return False, (
                f"DataParallel/DistributedDataParallel wrappers are not supported. "
                "Checkpoint the underlying model instead: use model.module for "
                "checkpointing, then re-wrap after restore."
            )

        # Check for uninitialized lazy modules in any submodule
        for name, module in obj.named_modules():
            if hasattr(module, 'has_uninitialized_params'):
                try:
                    if module.has_uninitialized_params():
                        return False, (
                            f"PyTorch model has uninitialized lazy module '{name}'. "
                            "Call model(sample_input) to initialize all lazy layers first."
                        )
                except Exception:
                    pass

        return True, None

    def _get_device(self, obj: Any) -> Optional[str]:
        """Get the device of the first parameter, or None if no params."""
        try:
            first_param = next(obj.parameters(), None)
            if first_param is not None:
                return str(first_param.device)
        except StopIteration:
            pass
        return None

    def _capture_custom_dict(self, obj: Any) -> Dict[str, Any]:
        """
        Capture custom attributes from a module's __dict__.

        Filters out:
        - Private attributes (starting with _)
        - Known framework internals

        Raises:
            TypeError: If any custom attribute cannot be deepcopied
        """
        from data_ferret.kernel.deepcopy import deepcopy as ferret_deepcopy

        captured = {}
        for key, value in obj.__dict__.items():
            # Skip private/internal PyTorch attributes
            if key.startswith('_'):
                continue
            # Skip known internals
            if key in PYTORCH_MODULE_INTERNALS:
                continue
            # Attempt to deep copy - fail if not possible
            try:
                captured[key] = ferret_deepcopy(value, {})
            except Exception as e:
                raise TypeError(
                    f"Cannot checkpoint PyTorch model: custom attribute '{key}' "
                    f"(type: {type(value).__name__}) is not serializable. "
                    f"Remove or replace this attribute before checkpointing. "
                    f"Original error: {e}"
                ) from e

        return captured

    def _capture_submodule_dicts(self, model: Any) -> Dict[str, Dict[str, Any]]:
        """Capture custom attributes from all named submodules."""
        from data_ferret.kernel.deepcopy import deepcopy as ferret_deepcopy

        submodule_dicts = {}
        for name, module in model.named_modules():
            if name == '':  # Skip root module (handled by model_dict)
                continue

            # Capture custom attrs for this submodule
            captured = {}
            for key, value in module.__dict__.items():
                if key.startswith('_'):
                    continue
                if key in PYTORCH_MODULE_INTERNALS:
                    continue
                try:
                    captured[key] = ferret_deepcopy(value, {})
                except Exception as e:
                    raise TypeError(
                        f"Cannot checkpoint PyTorch model: custom attribute '{name}.{key}' "
                        f"(type: {type(value).__name__}) is not serializable. "
                        f"Remove or replace this attribute before checkpointing. "
                        f"Original error: {e}"
                    ) from e

            if captured:  # Only store if there are custom attrs
                submodule_dicts[name] = captured

        return submodule_dicts

    def get_mutable_state(self, obj: Any) -> Dict[str, Any]:
        """
        Extract state_dict, training mode, device, and custom attributes.

        Returns dict with:
        - 'state_dict': model parameters and buffers (cloned tensors)
        - 'training': training mode flag
        - 'device': device of first parameter
        - 'model_dict': custom model-level attributes
        - 'submodule_dicts': custom submodule attributes
        """
        # Clone state_dict tensors to ensure independence
        state_dict = {}
        for key, tensor in obj.state_dict().items():
            state_dict[key] = tensor.clone().detach().cpu()

        return {
            'state_dict': state_dict,
            'training': obj.training,
            'device': self._get_device(obj),
            'model_dict': self._capture_custom_dict(obj),
            'submodule_dicts': self._capture_submodule_dicts(obj),
        }

    def copy_with_state(self, obj: Any, state: Dict[str, Any], memo: dict) -> Any:
        """
        Create a copy of the PyTorch model with the given state.

        Strategy:
        1. Use copy.deepcopy() for architecture (works for PyTorch)
        2. Load state_dict
        3. Restore training mode
        4. Restore device placement
        5. Restore custom attributes
        """
        import copy as stdlib_copy
        from data_ferret.kernel.deepcopy import deepcopy as ferret_deepcopy

        obj_id = id(obj)
        if obj_id in memo:
            return memo[obj_id]

        # Import torch here to avoid loading at module import time
        import torch

        # Deep copy creates independent model with same architecture
        model_copy = stdlib_copy.deepcopy(obj)

        # Register in memo early
        memo[obj_id] = model_copy

        # Convert state_dict tensors back and load
        converted_state_dict = {}
        for key, tensor in state['state_dict'].items():
            converted_state_dict[key] = tensor.clone()
        model_copy.load_state_dict(converted_state_dict)

        # Restore training mode
        if state['training']:
            model_copy.train()
        else:
            model_copy.eval()

        # Restore device if specified
        if state['device'] is not None:
            try:
                model_copy.to(state['device'])
            except Exception:
                pass  # Device restoration is best-effort

        # Restore custom model-level attributes
        if state.get('model_dict'):
            for key, value in state['model_dict'].items():
                try:
                    setattr(model_copy, key, ferret_deepcopy(value, {}))
                except Exception:
                    pass

        # Restore custom submodule attributes
        if state.get('submodule_dicts'):
            named_modules = dict(model_copy.named_modules())
            for name, module_dict in state['submodule_dicts'].items():
                if name in named_modules:
                    submodule = named_modules[name]
                    for key, value in module_dict.items():
                        try:
                            setattr(submodule, key, ferret_deepcopy(value, {}))
                        except Exception:
                            pass

        return model_copy

    def states_equal(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> bool:
        """Compare PyTorch model states for equality."""
        import torch

        # Compare state_dicts
        sd1 = state1['state_dict']
        sd2 = state2['state_dict']

        if set(sd1.keys()) != set(sd2.keys()):
            return False

        for key in sd1.keys():
            t1, t2 = sd1[key], sd2[key]
            if t1.shape != t2.shape:
                return False
            if t1.dtype != t2.dtype:
                return False
            if not torch.allclose(t1.float(), t2.float(), rtol=1e-5, atol=1e-8):
                return False

        # Compare training mode
        if state1['training'] != state2['training']:
            return False

        # Compare custom model attributes
        if state1.get('model_dict') != state2.get('model_dict'):
            return False

        # Compare custom submodule attributes
        if state1.get('submodule_dicts') != state2.get('submodule_dicts'):
            return False

        return True

    def get_traversable_attrs(self, obj: Any) -> Dict[str, Any]:
        """
        Return custom attributes that should be traversed for alias detection.

        These are user-defined attributes on the module and its submodules
        that may contain references to objects in the notebook namespace.
        """
        attrs = {}

        # Model-level custom attributes
        for key, value in obj.__dict__.items():
            if key.startswith('_'):
                continue
            if key in PYTORCH_MODULE_INTERNALS:
                continue
            attrs[key] = value

        # Submodule custom attributes
        for name, module in obj.named_modules():
            if name == '':
                continue
            for key, value in module.__dict__.items():
                if key.startswith('_'):
                    continue
                if key in PYTORCH_MODULE_INTERNALS:
                    continue
                attrs[f"{name}.{key}"] = value

        return attrs


# =============================================================================
# PyTorch Lazy Registration
# =============================================================================

_pytorch_handler_registered = False


def _maybe_register_pytorch_handler(obj: Any) -> None:
    """Register PyTorch handler if obj looks like a PyTorch nn.Module."""
    global _pytorch_handler_registered
    if _pytorch_handler_registered:
        return

    module = type(obj).__module__ or ""
    if module.startswith("torch"):
        # Check MRO for nn.Module base class
        for base in type(obj).__mro__:
            base_module = getattr(base, '__module__', '') or ''
            if base.__name__ == 'Module' and 'torch.nn' in base_module:
                OpaqueRegistry.register(PyTorchModelHandler())
                _pytorch_handler_registered = True
                return


def reset_pytorch_handler():
    """Reset PyTorch handler registration. For testing."""
    global _pytorch_handler_registered
    _pytorch_handler_registered = False
