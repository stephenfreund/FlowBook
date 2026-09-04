"""Diffing a checkpoint against a live namespace must materialize subset relations.

Regression: a DataFrame produced by a boolean-mask filter is stored in the
checkpoint as a relation to its parent, not a copy. Diffing that checkpoint
against the live namespace reported the subset as different even when nothing
had run, which the enforcer turned into a phantom write of the subset variable
by any cell that merely read it.
"""
import numpy as np
import pandas as pd
import pytest

from flowbook.kernel_support.checkpoint import Checkpoint, Checkpoints


def _ns():
    rng = np.random.default_rng(0)
    n = 5000  # large enough for the subset optimization to kick in
    raw = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.integers(0, 5, n)})
    ns = {"raw": raw}
    ns["sub"] = raw[raw["a"] >= 0]
    return ns


def _save_with_subsets(ns):
    cp = Checkpoints()
    cp.memory.set_df_subset_optimization(True)
    saved, _ = cp.save("pre", dict(ns), max_size_mb=None)
    return cp, saved


def test_subset_is_stored_as_relation():
    ns = _ns()
    _, saved = _save_with_subsets(ns)
    assert saved.memory._df_subset_relations, "test setup: subset optimization did not trigger"
    assert "sub" not in saved.memory.user_ns


def test_live_diff_sees_no_change_without_execution():
    ns = _ns()
    _, saved = _save_with_subsets(ns)
    d = Checkpoint.diff(saved, ns, use_leq=False)
    assert "sub" not in d.differences
    assert "raw" not in d.differences


def test_live_diff_after_reading_subset_sees_no_change():
    ns = _ns()
    _, saved = _save_with_subsets(ns)
    exec('x = sub[["a", "b"]]\ny = sub["c"]\nn = len(sub)', ns)
    d = Checkpoint.diff(saved, ns, use_leq=False)
    assert "sub" not in d.differences
    assert {"x", "y", "n"} <= set(d.differences)


def test_live_diff_after_mutating_subset_sees_change():
    ns = _ns()
    _, saved = _save_with_subsets(ns)
    exec('sub = sub.copy()\nsub["a"] = 0.0', ns)
    d = Checkpoint.diff(saved, ns, use_leq=False)
    assert "sub" in d.differences


def test_keys_to_include_pulls_in_parent_chain():
    ns = _ns()
    ns["sub2"] = ns["sub"][ns["sub"]["c"] == 1]
    _, saved = _save_with_subsets(ns)
    mat = saved.memory.materialized_ns({"sub2"})
    assert "sub2" in mat
    pd.testing.assert_frame_equal(mat["sub2"], ns["sub2"])
