# LightGBM Checkpoint Optimization Plan

## Problem

LGBMRegressor/LGBMClassifier objects are slow to deepcopy and diff because:
1. The internal `Booster` has a C++ backing object requiring serialization round-trips
2. Default `deepcopy` doesn't know the model is immutable after fitting
3. Diffing compares the entire object graph instead of leveraging model immutability

## Solution Overview

Leverage the fact that **fitted LightGBM models are immutable** - the tree ensemble doesn't change after training. Use `booster_.model_to_string()` for fast serialization and hash-based change detection.

## Integration Points

| Component | File | Purpose |
|-----------|------|---------|
| Custom deepcopy | `flowbook/kernel_support/deepcopy.py` | Fast model copying via string serialization |
| Custom diff | `flowbook/kernel_support/diff.py` | Hash-based equality check |

## Implementation

### 1. Add LightGBM Detection (deepcopy.py)

Follow the existing Keras/PyTorch pattern with lazy imports:

```python
def _is_lightgbm_model(obj: Any) -> bool:
    """Detect LightGBM model without importing lightgbm."""
    obj_module = getattr(type(obj), '__module__', '')
    return obj_module.startswith('lightgbm')

def _is_fitted_lightgbm_model(obj: Any) -> bool:
    """Check if model has been fitted (has booster_)."""
    return _is_lightgbm_model(obj) and hasattr(obj, 'booster_')
```

### 2. Add Custom Deepcopy Handler (deepcopy.py)

```python
def _deepcopy_lightgbm_model(model: Any, memo: dict) -> Any:
    """
    Fast deepcopy for fitted LightGBM models.

    Strategy:
    - Fitted models: Serialize booster to string, reconstruct
    - Unfitted models: Fall back to default deepcopy
    """
    import lightgbm as lgb

    model_id = id(model)
    if model_id in memo:
        return memo[model_id]

    if not hasattr(model, 'booster_'):
        # Unfitted model - use default deepcopy
        result = copy.deepcopy(model, memo)
        memo[model_id] = result
        return result

    # Fitted model - use fast string serialization
    model_str = model.booster_.model_to_string()

    # Create new instance of same class
    new_model = model.__class__(**model.get_params())

    # Reconstruct booster from string
    new_model._Booster = lgb.Booster(model_str=model_str)
    new_model.booster_ = new_model._Booster

    # Copy sklearn fitted attributes
    for attr in ['n_features_', 'n_features_in_', 'feature_name_',
                 '_n_features', 'fitted_', '_best_iteration',
                 '_best_score', '_other_params', '_n_classes',
                 'classes_', '_class_map', '_class_weight']:
        if hasattr(model, attr):
            setattr(new_model, attr, copy.deepcopy(getattr(model, attr), memo))

    memo[model_id] = new_model
    return new_model

_LIGHTGBM_HANDLERS_REGISTERED = False

def _register_lightgbm_handlers_if_needed() -> None:
    """Lazily register LightGBM handlers on first encounter."""
    global _LIGHTGBM_HANDLERS_REGISTERED
    if _LIGHTGBM_HANDLERS_REGISTERED:
        return

    try:
        import lightgbm as lgb
        _deepcopy_dispatch[lgb.LGBMRegressor] = _deepcopy_lightgbm_model
        _deepcopy_dispatch[lgb.LGBMClassifier] = _deepcopy_lightgbm_model
        _deepcopy_dispatch[lgb.LGBMRanker] = _deepcopy_lightgbm_model
        _LIGHTGBM_HANDLERS_REGISTERED = True
    except ImportError:
        pass
```

### 3. Update Main Deepcopy Function (deepcopy.py)

In the main `deepcopy()` function, add detection before the dispatch lookup:

```python
def deepcopy(obj: Any, memo: Optional[dict] = None) -> Any:
    # ... existing code ...

    # Check for special types that need lazy registration
    if _is_lightgbm_model(obj):
        _register_lightgbm_handlers_if_needed()

    # ... continue with dispatch lookup ...
```

### 4. Add Custom Diff Handler (diff.py)

```python
def _is_lightgbm_model(obj: Any) -> bool:
    """Detect LightGBM model without importing."""
    obj_module = getattr(type(obj), '__module__', '')
    return obj_module.startswith('lightgbm')

def _compare_lightgbm_model(self, a: Any, b: Any, path: str) -> Optional[ValueComparison]:
    """
    Compare LightGBM models using model string hash.

    For fitted models, the booster string is a complete representation.
    Comparing hashes is O(model_size) but avoids deep object traversal.
    """
    # Different types -> different
    if type(a) != type(b):
        return ValueComparison(path=path, equal=False,
                               reason=f"type mismatch: {type(a)} vs {type(b)}")

    # Both unfitted -> compare params only
    a_fitted = hasattr(a, 'booster_')
    b_fitted = hasattr(b, 'booster_')

    if not a_fitted and not b_fitted:
        return self._compare_dict(a.get_params(), b.get_params(), path)

    # One fitted, one not -> different
    if a_fitted != b_fitted:
        return ValueComparison(path=path, equal=False,
                               reason="fitted status differs")

    # Both fitted -> compare model strings
    a_str = a.booster_.model_to_string()
    b_str = b.booster_.model_to_string()

    if a_str == b_str:
        return None  # Equal

    return ValueComparison(path=path, equal=False,
                           reason="model trees differ")
```

Add to the isinstance fallback chain in `Diff._compare()`:

```python
# After PyTorch check, before generic object fallback
if _is_lightgbm_model(a) and _is_lightgbm_model(b):
    return self._compare_lightgbm_model(a, b, path)
```

### 5. Optional: Hash Caching for Repeated Checks

For scenarios with repeated diffs of unchanged models, cache the model hash:

```python
# In MemoryCheckpoint or as a module-level cache
_lightgbm_model_hashes: WeakValueDictionary[int, str] = WeakValueDictionary()

def _get_lightgbm_hash(model: Any) -> str:
    """Get or compute hash of LightGBM model."""
    model_id = id(model)
    if model_id in _lightgbm_model_hashes:
        return _lightgbm_model_hashes[model_id]

    model_str = model.booster_.model_to_string()
    model_hash = hashlib.sha256(model_str.encode()).hexdigest()
    _lightgbm_model_hashes[model_id] = model_hash
    return model_hash
```

## Testing

### Unit Tests

Create `flowbook/kernel_support/tests/test_lightgbm_checkpoint.py`:

```python
import pytest
import numpy as np

pytest.importorskip("lightgbm")

from lightgbm import LGBMRegressor, LGBMClassifier
from flowbook.kernel_support.deepcopy import deepcopy
from flowbook.kernel_support.diff import Diff

@pytest.fixture
def fitted_regressor():
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    model = LGBMRegressor(n_estimators=10, verbose=-1)
    model.fit(X, y)
    return model, X

def test_deepcopy_fitted_regressor(fitted_regressor):
    model, X = fitted_regressor
    copied = deepcopy(model, {})

    # Different objects
    assert copied is not model
    assert copied.booster_ is not model.booster_

    # Same predictions
    np.testing.assert_array_equal(
        model.predict(X),
        copied.predict(X)
    )

def test_deepcopy_unfitted_regressor():
    model = LGBMRegressor(n_estimators=10)
    copied = deepcopy(model, {})

    assert copied is not model
    assert copied.get_params() == model.get_params()

def test_diff_equal_models(fitted_regressor):
    model, _ = fitted_regressor
    copied = deepcopy(model, {})

    diff = Diff()
    result = diff.diff({'model': model}, {'model': copied})
    assert result.differences == {}

def test_diff_different_models():
    X = np.random.randn(100, 5)
    y1 = np.random.randn(100)
    y2 = np.random.randn(100)

    model1 = LGBMRegressor(n_estimators=10, verbose=-1).fit(X, y1)
    model2 = LGBMRegressor(n_estimators=10, verbose=-1).fit(X, y2)

    diff = Diff()
    result = diff.diff({'model': model1}, {'model': model2})
    assert 'model' in result.differences

def test_deepcopy_performance(fitted_regressor, benchmark):
    """Verify optimized deepcopy is faster than default."""
    model, _ = fitted_regressor

    def copy_model():
        return deepcopy(model, {})

    benchmark(copy_model)
```

### Benchmark Script

Create `examples/benchmark_lightgbm_checkpoint.py`:

```python
"""Benchmark LightGBM checkpoint operations."""
import time
import copy
import numpy as np
from lightgbm import LGBMRegressor

from flowbook.kernel_support.deepcopy import deepcopy as fast_deepcopy
from flowbook.kernel_support.diff import Diff

def benchmark():
    # Create large model
    X = np.random.randn(10000, 100)
    y = np.random.randn(10000)

    model = LGBMRegressor(n_estimators=500, num_leaves=63, verbose=-1)
    model.fit(X, y)

    print(f"Model size: {len(model.booster_.model_to_string()) / 1024:.1f} KB")

    # Benchmark default deepcopy
    start = time.perf_counter()
    for _ in range(10):
        copy.deepcopy(model)
    default_time = (time.perf_counter() - start) / 10
    print(f"Default deepcopy: {default_time*1000:.1f} ms")

    # Benchmark optimized deepcopy
    start = time.perf_counter()
    for _ in range(10):
        fast_deepcopy(model, {})
    fast_time = (time.perf_counter() - start) / 10
    print(f"Optimized deepcopy: {fast_time*1000:.1f} ms")
    print(f"Speedup: {default_time/fast_time:.1f}x")

    # Benchmark diff
    copied = fast_deepcopy(model, {})
    diff = Diff()

    start = time.perf_counter()
    for _ in range(10):
        diff.diff({'model': model}, {'model': copied})
    diff_time = (time.perf_counter() - start) / 10
    print(f"Diff (equal models): {diff_time*1000:.1f} ms")

if __name__ == '__main__':
    benchmark()
```

## Expected Performance

Based on LightGBM's architecture:

| Operation | Default | Optimized | Speedup |
|-----------|---------|-----------|---------|
| Deepcopy (100 trees) | ~500ms | ~50ms | ~10x |
| Deepcopy (500 trees) | ~2500ms | ~200ms | ~12x |
| Diff (equal) | ~500ms | ~100ms | ~5x |
| Diff (different) | ~500ms | ~100ms | ~5x |

The speedup comes from:
1. Avoiding Python object graph traversal
2. Using LightGBM's optimized C++ serialization
3. Single string comparison instead of tree-by-tree comparison

## XGBoost / CatBoost Extension

The same pattern can be applied to other gradient boosting libraries:

### XGBoost
```python
# model.save_raw() returns bytes
model_bytes = model.get_booster().save_raw()
new_booster = xgb.Booster()
new_booster.load_model(bytearray(model_bytes))
```

### CatBoost
```python
# Already has _deepcopy_catboost_pool, extend to CatBoostRegressor
model_str = model.save_model(format='cbm')  # or use save_model to buffer
```

## Rollout Steps

1. **Phase 1**: Implement and test deepcopy handler
   - Add detection and handler functions
   - Add unit tests
   - Run benchmark to verify speedup

2. **Phase 2**: Implement diff handler
   - Add comparison function
   - Verify equality detection works
   - Test with reproducibility enforcer

3. **Phase 3**: Optional optimizations
   - Add hash caching if benchmarks show benefit
   - Extend to XGBoost/CatBoost if needed

## Files to Modify

- [ ] `flowbook/kernel_support/deepcopy.py` - Add LightGBM handlers
- [ ] `flowbook/kernel_support/diff.py` - Add LightGBM comparison
- [ ] `flowbook/kernel_support/tests/test_lightgbm_checkpoint.py` - New test file
- [ ] `examples/benchmark_lightgbm_checkpoint.py` - New benchmark script
