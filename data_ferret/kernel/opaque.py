"""
Opaque object handling for checkpoint/diff operations.

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
2. Mutable state (weights, cursors)

By extracting only the mutable state, we can:
- Reduce alias tracking from O(millions) to O(1)
- Make deepcopy fast via structure sharing + state copying
- Provide clear extension points for new object types
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Tuple, Dict
import numpy as np


class OpaqueHandler(ABC):
    """
    Handler for objects that should not be deeply traversed.

    Implementations define how to:
    - Check if an object can be checkpointed
    - Extract mutable state
    - Create copies with given state
    - Compare states for equality
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
        """
        pass

    @abstractmethod
    def copy_with_state(self, obj: Any, state: Any, memo: dict) -> Any:
        """
        Create a copy of obj with the given state.

        The copy should:
        - Share immutable structure with original where possible
        - Have independent mutable state from the provided state
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

class KerasModelHandler(OpaqueHandler):
    """
    Handler for Keras Sequential and Functional models.

    Keras models have millions of internal TensorFlow objects for tracking,
    but only the weights actually change during training. This handler:

    1. Rejects unbuilt models (architecture not frozen yet)
    2. Extracts weights via get_weights() for checkpointing
    3. Creates copies via clone_model() + set_weights()
    4. Compares models by weight equality
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

    def get_mutable_state(self, obj: Any) -> Dict[str, Any]:
        """
        Extract weights and optimizer state.

        Returns dict with:
        - 'weights': list of numpy arrays (deep copied)
        - 'optimizer_weights': optimizer state if compiled, else None
        - 'input_shape': for rebuilding
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
        4. Optionally restore optimizer state
        """
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

        # Register in memo before attempting optimizer restoration
        # (in case optimizer restoration fails)
        memo[obj_id] = model_copy

        # Optionally restore optimizer state
        if state['optimizer_config'] is not None:
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
                    # Need to do a dummy training step to initialize optimizer slots
                    # Or try direct weight setting
                    try:
                        model_copy.optimizer.set_weights(state['optimizer_weights'])
                    except Exception:
                        pass  # Optimizer weight restoration is best-effort
            except Exception:
                pass  # Compilation restoration is best-effort

        return model_copy

    def states_equal(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> bool:
        """Compare weights for equality."""
        weights1 = state1['weights']
        weights2 = state2['weights']

        if len(weights1) != len(weights2):
            return False

        for w1, w2 in zip(weights1, weights2):
            if w1.shape != w2.shape:
                return False
            if not np.allclose(w1, w2, rtol=1e-5, atol=1e-8):
                return False

        return True


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
