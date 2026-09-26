"""
Enforcer-level regressions for cells that only read state but were flagged
UNRECOVERABLE_MUTATION in the evaluation (see the checkpoint/diff-level tests
in kernel_support/tests/test_model_copy_diff_regressions.py and
test_cudf_regressions.py for the root causes):

- predict / feature importances on a fitted CatBoost model,
- predict on a fitted sklearn StackingClassifier,
- reading a cuDF DataFrame whose float column holds NaN.

Each cell reads the model/frame and writes only a new variable, so the
enforcer must report no error at all.
"""

import numpy as np
import pandas as pd
import pytest

from flowbook.kernel.tests.conftest import ReproducibilityTestHelper, make_tracking


def _run_reading_cell(ns, cell, reads, writes, column_reads=None):
    """Check one cell that runs cell(ns) between its pre-checkpoint and check()."""
    helper = ReproducibilityTestHelper()
    helper.set_cell_order(["a"])
    helper.save_pre_checkpoint("a", ns)
    cell(ns)
    return helper.sdc.check(
        cell_id="a",
        pre_checkpoint=helper.get_pre_checkpoint("a"),
        namespace=ns,
        tracking=make_tracking(reads=reads, writes=writes, column_reads=column_reads),
        continue_on_violation=True,
    )


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((200, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f0"] + rng.random(200) > 1).astype(int).to_numpy()
    return X, y


def test_catboost_predict_cell_has_no_error(classification_data):
    catboost = pytest.importorskip("catboost")
    X, y = classification_data
    model = catboost.CatBoostClassifier(iterations=5, verbose=0, allow_writing_files=False).fit(X, y)

    def cell(ns):
        ns["p"] = ns["model"].predict_proba(ns["X"])
        ns["imp"] = ns["model"].get_feature_importance()

    result = _run_reading_cell({"model": model, "X": X}, cell, reads={"model", "X"}, writes={"p", "imp"})
    assert result.errors == []


def test_catboost_in_dict_predict_cell_has_no_error(classification_data):
    catboost = pytest.importorskip("catboost")
    from sklearn.ensemble import RandomForestClassifier

    X, y = classification_data
    models = {
        "rf": RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y),
        "cb": catboost.CatBoostClassifier(iterations=5, verbose=0, allow_writing_files=False).fit(X, y),
    }

    def cell(ns):
        ns["best"] = max(ns["models"].items(), key=lambda kv: kv[1].predict_proba(ns["X"])[:, 1].sum())[1]

    result = _run_reading_cell({"models": models, "X": X}, cell, reads={"models", "X"}, writes={"best"})
    assert result.errors == []


def test_stacking_predict_cell_has_no_error(classification_data):
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression

    X, y = classification_data
    stacking = StackingClassifier(
        [("rf", RandomForestClassifier(n_estimators=5, random_state=0)), ("lr", LogisticRegression())],
        cv=2,
    ).fit(X, y)

    def cell(ns):
        ns["p"] = ns["stacking"].predict(ns["X"])

    result = _run_reading_cell({"stacking": stacking, "X": X}, cell, reads={"stacking", "X"}, writes={"p"})
    assert result.errors == []


def test_cudf_frame_with_nan_read_only_cell_has_no_error():
    cudf = pytest.importorskip("cudf")
    try:
        cudf.Series([1])
    except Exception:  # pragma: no cover - no usable GPU
        pytest.skip("cudf is installed but no GPU is usable")
    from flowbook.kernel_support import cudf_compat

    train = cudf.DataFrame({"a": np.arange(100) % 7})
    train["w"] = cudf.Series(np.where(np.arange(100) % 10 == 0, np.nan, 1.5), nan_as_null=False)
    train["Price"] = np.arange(100, dtype="float64")

    def cell(ns):
        ns["weights"] = ns["train"]["w"].values

    previous = cudf_compat._CUDF_GPU_CHECKPOINT
    cudf_compat.set_gpu_checkpoint_mode(True)
    try:
        result = _run_reading_cell(
            {"train": train}, cell, reads={"train"}, writes={"weights"}, column_reads={"train": {"w"}},
        )
    finally:
        cudf_compat.set_gpu_checkpoint_mode(previous)
    assert result.errors == []
