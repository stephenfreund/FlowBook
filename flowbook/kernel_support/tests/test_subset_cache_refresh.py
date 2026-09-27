"""The subset detector's cache must not serve stale values.

A child frame that is a row subset of a parent is checkpointed as (parent, row indices,
extra columns). The relation is cached by object id and shape; an in-place rewrite of an
extra column keeps the shape, so a cache hit used to return the *old* extra-column data.
Restoring or diffing against that checkpoint then made a read-only cell look like it had
mutated the column (unrecoverable_mutation).
"""
import numpy as np
import pandas as pd

from flowbook.kernel_support.df_subset_detector import DataFrameSubsetDetector


def _detector():
    return DataFrameSubsetDetector(min_rows=10, min_savings_bytes=0)


def test_cache_hit_refreshes_extra_column_data():
    parent = pd.DataFrame({"a": np.arange(100, dtype=float), "q": np.arange(100)})
    child = parent[parent["a"] >= 20].copy()
    child["b"] = child["a"] * 2
    det = _detector()
    r1 = det.detect({"parent": parent, "child": child}).relations[0]
    assert r1.extra_columns == ["b"]
    assert list(r1.extra_data["b"]) == list(child["b"])
    child["b"] = child["a"] * 5           # same shape, new values
    r2 = det.detect({"parent": parent, "child": child}).relations[0]
    assert list(r2.extra_data["b"]) == list(child["b"]), "cached extra data is stale"


def test_cache_hit_revalidates_common_columns():
    parent = pd.DataFrame({"a": np.arange(100, dtype=float)})
    child = parent[parent["a"] >= 20].copy()
    det = _detector()
    assert det.detect({"parent": parent, "child": child}).relations
    child["a"] = child["a"] + 1           # no longer a subset of the parent, same shape
    assert not det.detect({"parent": parent, "child": child}).relations
