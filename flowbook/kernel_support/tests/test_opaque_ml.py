"""Tests for alias-only opaque handler for ML library objects."""

import importlib

import pytest
import numpy as np

from flowbook.kernel_support.opaque import (
    OpaqueRegistry,
    MLModelOpaqueHandler,
    reset_ml_handler,
    reset_keras_handler,
    reset_pytorch_handler,
)
from flowbook.kernel_support.memory_checkpoint import _collect_reachable_ids


def _has_module(name: str) -> bool:
    """Check if a module is importable without importing it."""
    return importlib.util.find_spec(name) is not None


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the opaque registry before each test."""
    OpaqueRegistry.clear()
    reset_ml_handler()
    reset_keras_handler()
    reset_pytorch_handler()
    yield
    OpaqueRegistry.clear()
    reset_ml_handler()
    reset_keras_handler()
    reset_pytorch_handler()


# =============================================================================
# CatBoost
# =============================================================================


@pytest.mark.skipif(not _has_module("catboost"), reason="catboost not installed")
class TestCatBoostHandler:
    def test_can_handle_regressor(self):
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=5, verbose=0)
        handler = MLModelOpaqueHandler()
        assert handler.can_handle(model)

    def test_can_handle_classifier(self):
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=5, verbose=0)
        handler = MLModelOpaqueHandler()
        assert handler.can_handle(model)

    def test_registry_returns_handler(self):
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=5, verbose=0)
        handler = OpaqueRegistry.get_handler(model)
        assert handler is not None
        assert isinstance(handler, MLModelOpaqueHandler)

    def test_traversable_attrs_empty(self):
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=5, verbose=0)
        handler = MLModelOpaqueHandler()
        assert handler.get_traversable_attrs(model) == {}

    def test_collect_reachable_ids_short_circuits(self):
        """Alias collection should visit only the top-level model ID."""
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=5, verbose=0)
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        model.fit(X, y, verbose=0)

        OpaqueRegistry.register(MLModelOpaqueHandler())
        visited = set()
        _collect_reachable_ids(model, visited)

        # With handler: should be just the top-level ID
        assert len(visited) == 1
        assert id(model) in visited

    def test_can_handle_pool(self):
        from catboost import Pool
        X = np.random.rand(10, 3)
        pool = Pool(X)
        handler = MLModelOpaqueHandler()
        assert handler.can_handle(pool)

    def test_registry_returns_handler_for_pool(self):
        from catboost import Pool
        X = np.random.rand(10, 3)
        pool = Pool(X)
        handler = OpaqueRegistry.get_handler(pool)
        assert handler is not None
        assert isinstance(handler, MLModelOpaqueHandler)


# =============================================================================
# XGBoost
# =============================================================================


@pytest.mark.skipif(not _has_module("xgboost"), reason="xgboost not installed")
class TestXGBoostHandler:
    def test_can_handle_regressor(self):
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=5)
        handler = MLModelOpaqueHandler()
        assert handler.can_handle(model)

    def test_can_handle_classifier(self):
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=5)
        handler = MLModelOpaqueHandler()
        assert handler.can_handle(model)

    def test_registry_returns_handler(self):
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=5)
        handler = OpaqueRegistry.get_handler(model)
        assert handler is not None
        assert isinstance(handler, MLModelOpaqueHandler)

    def test_collect_reachable_ids_short_circuits(self):
        """Fitted XGBoost model should only have top-level ID collected."""
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=5)
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        model.fit(X, y)

        OpaqueRegistry.register(MLModelOpaqueHandler())
        visited = set()
        _collect_reachable_ids(model, visited)
        assert len(visited) == 1
        assert id(model) in visited


# =============================================================================
# LightGBM
# =============================================================================


@pytest.mark.skipif(not _has_module("lightgbm"), reason="lightgbm not installed")
class TestLightGBMHandler:
    def test_can_handle_regressor(self):
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=5, verbose=-1)
        handler = MLModelOpaqueHandler()
        assert handler.can_handle(model)

    def test_can_handle_classifier(self):
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=5, verbose=-1)
        handler = MLModelOpaqueHandler()
        assert handler.can_handle(model)

    def test_registry_returns_handler(self):
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=5, verbose=-1)
        handler = OpaqueRegistry.get_handler(model)
        assert handler is not None
        assert isinstance(handler, MLModelOpaqueHandler)

    def test_collect_reachable_ids_short_circuits(self):
        """Fitted LightGBM model should only have top-level ID collected."""
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=5, verbose=-1)
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        model.fit(X, y)

        OpaqueRegistry.register(MLModelOpaqueHandler())
        visited = set()
        _collect_reachable_ids(model, visited)
        assert len(visited) == 1
        assert id(model) in visited


# =============================================================================
# SHAP
# =============================================================================


@pytest.mark.skipif(not _has_module("shap"), reason="shap not installed")
class TestSHAPHandler:
    def test_can_handle_explanation(self):
        import shap
        values = np.random.rand(10, 3)
        explanation = shap.Explanation(values=values)
        handler = MLModelOpaqueHandler()
        assert handler.can_handle(explanation)

    def test_registry_returns_handler_for_explanation(self):
        import shap
        values = np.random.rand(10, 3)
        explanation = shap.Explanation(values=values)
        handler = OpaqueRegistry.get_handler(explanation)
        assert handler is not None
        assert isinstance(handler, MLModelOpaqueHandler)

    def test_collect_reachable_ids_short_circuits(self):
        """SHAP Explanation should only have top-level ID collected."""
        import shap
        values = np.random.rand(10, 3)
        explanation = shap.Explanation(values=values)

        OpaqueRegistry.register(MLModelOpaqueHandler())
        visited = set()
        _collect_reachable_ids(explanation, visited)
        assert len(visited) == 1
        assert id(explanation) in visited


# =============================================================================
# sklearn
# =============================================================================


@pytest.mark.skipif(not _has_module("sklearn"), reason="sklearn not installed")
class TestSklearnHandler:
    def test_can_handle_target_encoder(self):
        from sklearn.preprocessing import TargetEncoder
        enc = TargetEncoder()
        handler = MLModelOpaqueHandler()
        assert handler.can_handle(enc)

    def test_can_handle_stacking_regressor(self):
        from sklearn.linear_model import Ridge
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import StackingRegressor
        estimators = [('ridge', Ridge()), ('dt', DecisionTreeRegressor())]
        stacker = StackingRegressor(estimators=estimators)
        handler = MLModelOpaqueHandler()
        assert handler.can_handle(stacker)

    def test_registry_returns_handler_for_target_encoder(self):
        from sklearn.preprocessing import TargetEncoder
        enc = TargetEncoder()
        handler = OpaqueRegistry.get_handler(enc)
        assert handler is not None
        assert isinstance(handler, MLModelOpaqueHandler)

    def test_collect_reachable_ids_short_circuits(self):
        """Fitted TargetEncoder should only have top-level ID collected."""
        from sklearn.preprocessing import TargetEncoder
        enc = TargetEncoder()
        X = np.array([[1], [2], [3], [1], [2], [3]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
        enc.fit(X, y)

        OpaqueRegistry.register(MLModelOpaqueHandler())
        visited = set()
        _collect_reachable_ids(enc, visited)
        assert len(visited) == 1
        assert id(enc) in visited


# =============================================================================
# Negative tests
# =============================================================================


class TestMLHandlerNegative:
    def test_does_not_handle_plain_dict(self):
        handler = MLModelOpaqueHandler()
        assert not handler.can_handle({"a": 1})

    def test_does_not_handle_dataframe(self):
        import pandas as pd
        handler = MLModelOpaqueHandler()
        assert not handler.can_handle(pd.DataFrame({"a": [1, 2]}))

    def test_does_not_handle_numpy_array(self):
        handler = MLModelOpaqueHandler()
        assert not handler.can_handle(np.array([1, 2, 3]))

    def test_does_not_handle_plain_object(self):
        handler = MLModelOpaqueHandler()

        class MyObj:
            pass
        assert not handler.can_handle(MyObj())

    def test_lazy_registration_only_on_match(self):
        """Handler should not be registered for non-ML objects."""
        OpaqueRegistry.get_handler({"a": 1})
        OpaqueRegistry.get_handler(np.array([1]))
        assert not any(
            isinstance(h, MLModelOpaqueHandler)
            for h in OpaqueRegistry._handlers
        )
