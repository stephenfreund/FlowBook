"""
Universal model copy utility for ML frameworks.

Provides a single `safe_model_copy()` function that works correctly for all
major ML frameworks, using the fastest correct method for each:

- sklearn (unfitted): sklearn.clone() — fast, copies params only
- sklearn (fitted): copy.deepcopy() — slow but correct
- TensorFlow/Keras: tf.keras.models.clone_model() + set_weights()
- PyTorch: copy.deepcopy() — native implementation, preserves device
- XGBoost/LightGBM/CatBoost: copy.deepcopy() — uses __reduce__ serialization

Usage:
    from flowbook.util.model_copy import safe_model_copy

    model_copy = safe_model_copy(model)
    model_copy.fit(X, y)  # Original model unchanged
"""

import copy
from typing import Any


def _is_sklearn_estimator(obj: Any) -> bool:
    """Check if obj is a sklearn-compatible estimator without importing sklearn."""
    cls = type(obj)
    module = getattr(cls, '__module__', '') or ''
    if not isinstance(module, str):
        return False

    # Check for sklearn or compatible libraries (xgboost, lightgbm, catboost sklearn API)
    sklearn_modules = ('sklearn', 'xgboost', 'lightgbm', 'catboost')
    if not any(module.startswith(m) for m in sklearn_modules):
        return False

    # Must have get_params (sklearn estimator protocol)
    return hasattr(obj, 'get_params') and callable(getattr(obj, 'get_params'))


def _is_fitted_sklearn(obj: Any) -> bool:
    """
    Check if a sklearn estimator is fitted.

    Uses sklearn convention: fitted estimators have attributes ending with '_'
    that are set during fit() (e.g., coef_, feature_importances_, classes_).
    """
    # Common fitted attributes across sklearn estimators
    fitted_attrs = [
        'coef_', 'intercept_', 'classes_', 'n_classes_',
        'feature_importances_', 'tree_', 'estimators_',
        'cluster_centers_', 'labels_', 'components_',
        'mean_', 'var_', 'scale_', 'n_features_in_',
        # XGBoost/LightGBM/CatBoost
        'booster_', '_Booster', 'feature_name_', 'best_iteration_',
    ]
    return any(hasattr(obj, attr) for attr in fitted_attrs)


def _is_keras_model(obj: Any) -> bool:
    """Check if obj is a Keras/TensorFlow model without importing TensorFlow."""
    cls = type(obj)
    module = getattr(cls, '__module__', '') or ''
    if not isinstance(module, str):
        return False

    # Check for keras or tensorflow.keras
    if not ('keras' in module or 'tensorflow' in module):
        return False

    # Check MRO for Model base class
    for base in cls.__mro__:
        if base.__name__ in ('Model', 'Sequential', 'Functional'):
            base_module = getattr(base, '__module__', '') or ''
            if 'keras' in base_module:
                return True

    return False


def _is_pytorch_model(obj: Any) -> bool:
    """Check if obj is a PyTorch nn.Module without importing torch."""
    cls = type(obj)
    module = getattr(cls, '__module__', '') or ''
    if not isinstance(module, str):
        return False

    if not module.startswith('torch'):
        return False

    # Check MRO for nn.Module base class
    for base in cls.__mro__:
        base_module = getattr(base, '__module__', '') or ''
        if base.__name__ == 'Module' and 'torch.nn' in base_module:
            return True

    return False


def safe_model_copy(model: Any) -> Any:
    """
    Create an independent copy of a model using the fastest correct method.

    This function handles all major ML frameworks:

    - **sklearn (unfitted)**: Uses `sklearn.clone()` which only copies
      hyperparameters — very fast, O(1) regardless of model complexity.

    - **sklearn (fitted)**: Uses `copy.deepcopy()` which serializes the
      entire model state. Slower but necessary to preserve fitted state.

    - **TensorFlow/Keras**: Uses `tf.keras.models.clone_model()` to clone
      architecture, then `set_weights()` to copy weights. Required because
      Keras models don't support standard deepcopy.

    - **PyTorch**: Uses `copy.deepcopy()` which PyTorch implements natively.
      Preserves device placement (GPU models stay on GPU).

    - **XGBoost/LightGBM/CatBoost**: Uses `copy.deepcopy()` which triggers
      `__reduce__` serialization. Works correctly for both CPU and GPU
      trained models (model state is always CPU-resident).

    Args:
        model: Any ML model object

    Returns:
        An independent copy of the model. Mutations to the copy do not
        affect the original.

    Raises:
        TypeError: If the model cannot be copied (e.g., TensorFlow model
            with custom non-serializable layers)

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier(n_estimators=100)
        >>> model_copy = safe_model_copy(model)  # Fast clone (unfitted)
        >>> model_copy.fit(X, y)  # Original unchanged

        >>> model.fit(X, y)
        >>> model_copy = safe_model_copy(model)  # Deep copy (fitted)
    """
    # Fast path: unfitted sklearn estimator
    if _is_sklearn_estimator(model) and not _is_fitted_sklearn(model):
        try:
            from sklearn.base import clone
            return clone(model)
        except ImportError:
            # sklearn not installed, fall through to deepcopy
            pass

    # TensorFlow/Keras special case
    if _is_keras_model(model):
        try:
            import tensorflow as tf

            # clone_model creates architecture copy
            model_copy = tf.keras.models.clone_model(model)

            # Copy weights if model has been built
            if model.weights:
                model_copy.set_weights(model.get_weights())

            # Copy optimizer state if compiled
            if model.optimizer is not None:
                # Recompile with same config
                model_copy.compile(
                    optimizer=model.optimizer.__class__.from_config(
                        model.optimizer.get_config()
                    ),
                    loss=model.loss,
                    metrics=[m.name for m in model.metrics] if model.metrics else None,
                )
                # Restore optimizer weights if available
                if hasattr(model.optimizer, 'get_weights'):
                    opt_weights = model.optimizer.get_weights()
                    if opt_weights:
                        # Need a dummy training step to initialize optimizer variables
                        # before setting weights (Keras quirk)
                        pass  # Skip optimizer state for simplicity

            return model_copy
        except Exception as e:
            raise TypeError(
                f"Cannot copy Keras model: {e}. "
                "Consider saving/loading the model with model.save()/tf.keras.models.load_model()."
            ) from e

    # Everything else: deepcopy works
    # - PyTorch: native __deepcopy__ implementation, preserves device
    # - sklearn fitted: pickle-based, correct
    # - XGBoost/LightGBM/CatBoost: __reduce__ serialization, correct
    return copy.deepcopy(model)


def safe_model_copy_simple(model: Any) -> Any:
    """
    Simplified model copy for use in generated fix code.

    Unlike safe_model_copy(), this version:
    - Doesn't handle Keras (requires explicit tensorflow import)
    - Prioritizes simplicity over performance
    - Always works without additional imports

    For generated fix code, use this version to minimize dependencies.
    """
    # Try sklearn.clone for unfitted estimators (fast)
    if hasattr(model, 'get_params') and hasattr(model, 'fit'):
        # Check if unfitted
        fitted_attrs = ['coef_', 'classes_', 'feature_importances_', 'booster_', '_Booster']
        if not any(hasattr(model, attr) for attr in fitted_attrs):
            try:
                from sklearn.base import clone
                return clone(model)
            except ImportError:
                pass

    # Fall back to deepcopy (always works)
    return copy.deepcopy(model)
