"""
Regression tests for fitted models reported as mutated by cells that only
use them (predict, feature importances, SHAP), found in the evaluation.

- CatBoost: copy.deepcopy drops wrapper attributes (``_n_features_in`` comes
  back 0), so every checkpoint copy differed from its model.
- sklearn Stacking*: the comparator compared base estimators by identity,
  but checkpoint copies do not share them, so every copy differed.
- Namedtuples holding arrays (IterativeImputer's ``_ImputerTriplet``): the
  object fallback's ``!=`` raised on the arrays and reported a difference.

Each case also checks that a real change is still detected.
"""

import collections
import copy
import warnings

import numpy as np
import pandas as pd
import pytest

from flowbook.kernel_support.deepcopy import deepcopy as flowbook_deepcopy
from flowbook.kernel_support.diff import Diff
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint, MemoryCheckpoints


def _diff_after(ns, action=None):
    """Checkpoint ns, run action(ns), and diff the checkpoint against ns."""
    checkpoints = MemoryCheckpoints(sanity_check=False, warn_classes=False)
    checkpoints.save("pre", ns, max_size_mb=None)
    if action is not None:
        action(ns)
    return MemoryCheckpoint.diff(checkpoints.get("pre"), ns)


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((200, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f0"] + rng.random(200) > 1).astype(int).to_numpy()
    return X, y


# =============================================================================
# CatBoost
# =============================================================================


class TestCatBoostCopy:
    @pytest.fixture
    def catboost(self):
        return pytest.importorskip("catboost")

    def test_deepcopy_keeps_wrapper_attributes(self, catboost, classification_data):
        X, y = classification_data
        model = catboost.CatBoostClassifier(iterations=5, verbose=0, allow_writing_files=False).fit(X, y)
        model_copy = flowbook_deepcopy(model)
        assert model_copy._n_features_in == model._n_features_in == X.shape[1]
        assert model_copy is not model
        assert model_copy._object is not model._object
        np.testing.assert_allclose(model_copy.predict_proba(X), model.predict_proba(X))

    @pytest.mark.parametrize("action", [
        lambda ns: ns["model"].predict_proba(ns["X"]),
        lambda ns: ns["model"].get_feature_importance(),
        lambda ns: None,
    ], ids=["predict_proba", "get_feature_importance", "no-op"])
    def test_using_the_model_is_not_a_change(self, catboost, classification_data, action):
        X, y = classification_data
        model = catboost.CatBoostClassifier(iterations=5, verbose=0, allow_writing_files=False).fit(X, y)
        result = _diff_after({"model": model, "X": X}, action)
        assert "model" not in result.differences

    def test_model_inside_a_dict_is_not_a_change(self, catboost, classification_data):
        X, y = classification_data
        models = {"cb": catboost.CatBoostRegressor(iterations=5, verbose=0, allow_writing_files=False).fit(X, y.astype(float))}
        result = _diff_after({"models": models, "X": X}, lambda ns: ns["models"]["cb"].predict(ns["X"]))
        assert "models" not in result.differences

    def test_refit_is_a_change(self, catboost, classification_data):
        X, y = classification_data
        model = catboost.CatBoostClassifier(iterations=5, verbose=0, allow_writing_files=False, random_seed=0).fit(X, y)
        result = _diff_after(
            {"model": model, "X": X},
            lambda ns: ns["model"].fit(ns["X"].iloc[:, :3], 1 - y),
        )
        assert "model" in result.differences


# =============================================================================
# sklearn Stacking
# =============================================================================


class TestStackingCopy:
    @pytest.fixture
    def stacking(self, classification_data):
        from sklearn.ensemble import RandomForestClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression

        X, y = classification_data
        return StackingClassifier(
            [("rf", RandomForestClassifier(n_estimators=5, random_state=0)),
             ("lr", LogisticRegression())],
            cv=2,
        ).fit(X, y)

    def test_copy_compares_equal(self, stacking):
        result = Diff().diff({"stacking": stacking}, {"stacking": flowbook_deepcopy(stacking)})
        assert "stacking" not in result.differences

    def test_predict_is_not_a_change(self, stacking, classification_data):
        X, _ = classification_data
        result = _diff_after({"stacking": stacking, "X": X}, lambda ns: ns["stacking"].predict(ns["X"]))
        assert "stacking" not in result.differences

    def test_refit_is_a_change(self, stacking, classification_data):
        X, y = classification_data
        result = _diff_after(
            {"stacking": stacking, "X": X},
            lambda ns: ns["stacking"].fit(ns["X"].iloc[:, :3], 1 - y),
        )
        assert "stacking" in result.differences

    def test_changed_base_estimator_is_reported(self, stacking):
        stacking_copy = flowbook_deepcopy(stacking)
        stacking_copy.estimators_[1].coef_ = stacking_copy.estimators_[1].coef_ + 1.0
        result = Diff().diff({"stacking": stacking}, {"stacking": stacking_copy})
        assert "estimators_[1]" in result.differences["stacking"].children


# =============================================================================
# Namedtuples (IterativeImputer)
# =============================================================================


Triplet = collections.namedtuple("Triplet", ["feat_idx", "neighbor_feat_idx", "weights"])


class TestNamedTupleWithArrays:
    def test_equal_namedtuples_compare_equal(self):
        a = Triplet(1, np.array([0, 2]), np.array([0.5, 1.5]))
        result = Diff().diff({"t": a}, {"t": copy.deepcopy(a)})
        assert "t" not in result.differences

    def test_different_array_element_is_reported(self):
        a = Triplet(1, np.array([0, 2]), np.array([0.5, 1.5]))
        b = Triplet(1, np.array([0, 2]), np.array([0.5, 9.5]))
        result = Diff().diff({"t": a}, {"t": b})
        assert "t" in result.differences

    def test_iterative_imputer_pipeline_predict_is_not_a_change(self):
        lgb = pytest.importorskip("lightgbm")
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.compose import make_column_transformer
        from sklearn.impute import IterativeImputer
        from sklearn.pipeline import make_pipeline

        rng = np.random.default_rng(0)
        n = 300
        x = pd.DataFrame({
            "a": np.where(rng.random(n) < 0.1, np.nan, rng.random(n)),
            "b": rng.random(n),
            "c": rng.random(n),
        })
        y = rng.random(n)
        prepro = make_column_transformer((make_pipeline(IterativeImputer()), ["a", "b", "c"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = make_pipeline(prepro, lgb.LGBMRegressor(n_estimators=10, verbose=-1)).fit(x, y)
            result = _diff_after(
                {"model": model, "prepro": prepro, "x": x},
                lambda ns: ns["model"].predict(ns["x"]),
            )
        assert "model" not in result.differences
        assert "prepro" not in result.differences
