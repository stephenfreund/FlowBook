"""
Regression tests for cuDF / cudf.pandas false positives found in the
evaluation's two RAPIDS notebooks.

- cudf.pandas proxy DataFrames: column reads/writes were never tracked when
  cudf.pandas was installed after the tracker (the kernel order: `%load_ext
  cudf.pandas` runs in a cell), so `df[c] = ...` was reported as an
  unrecoverable mutation.
- GPU DataFrame diff: cudf's equals() treats NaN != NaN, so every float column
  holding NaN compared as changed and every cell reading the frame was flagged.
- GPU DataFrame diff: `_structural_columns` listed every column as added.
- A pandas-backed proxy Index was checkpointed as a pandas Index and then
  compared against the proxy: "Type mismatch at COLS: Index vs Index".
- A row-subset DataFrame (`test = train.iloc[:n].copy()`) could not be rebuilt
  from its checkpoint relation (proxy ndarray row indices), so it vanished from
  the checkpoint and its mutations went undetected. This needs cudf.pandas to
  be installed before FlowBook is imported.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

cudf = pytest.importorskip("cudf")
try:
    cudf.Series([1])
except Exception:  # pragma: no cover - no usable GPU
    pytest.skip("cudf is installed but no GPU is usable", allow_module_level=True)

from flowbook.kernel_support import cudf_compat
from flowbook.kernel.change_detector import detect_changes
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint, MemoryCheckpoints
from flowbook.kernel.changes import ColumnAdded, ColumnModified, ColumnRemoved


@pytest.fixture
def gpu_checkpoint_mode():
    """Run with GPU-side checkpoints (the kernel default), restoring the flag."""
    previous = cudf_compat._CUDF_GPU_CHECKPOINT
    cudf_compat.set_gpu_checkpoint_mode(True)
    yield
    cudf_compat.set_gpu_checkpoint_mode(previous)


def _diff_after(ns, mutate=None):
    """Checkpoint ns, apply mutate(ns), and diff the checkpoint against ns."""
    checkpoints = MemoryCheckpoints(sanity_check=False, warn_classes=False)
    checkpoints.save("pre", ns, max_size_mb=None)
    if mutate is not None:
        mutate(ns)
    return MemoryCheckpoint.diff(checkpoints.get("pre"), ns)


def _nan_series(values):
    """A float cudf Series holding real NaN values (not nulls)."""
    return cudf.Series(values, nan_as_null=False)


# =============================================================================
# GPU DataFrame diff: NaN equality
# =============================================================================


@pytest.mark.usefixtures("gpu_checkpoint_mode")
class TestGpuDiffNaN:
    def test_nan_column_unchanged_has_no_diff(self):
        df = cudf.DataFrame({"a": [1, 2, 3]})
        df["w"] = _nan_series([1.0, np.nan, 3.0])
        result = _diff_after({"df": df})
        assert "df" not in result.differences

    def test_null_column_unchanged_has_no_diff(self):
        df = cudf.DataFrame({"a": [1, 2, 3], "w": [1.0, None, 3.0]})
        result = _diff_after({"df": df})
        assert "df" not in result.differences

    def test_nan_series_unchanged_has_no_diff(self):
        s = _nan_series([np.nan, 2.0, np.nan])
        result = _diff_after({"s": s})
        assert "s" not in result.differences

    def test_nan_replaced_by_value_detected(self):
        df = cudf.DataFrame({"a": [1, 2, 3]})
        df["w"] = _nan_series([1.0, np.nan, 3.0])

        def mutate(ns):
            ns["df"]["w"] = _nan_series([1.0, 2.0, 3.0])

        result = _diff_after({"df": df}, mutate)
        assert "['w']" in result.differences["df"].children

    def test_value_change_in_nan_column_detected(self):
        df = cudf.DataFrame({"a": [1, 2, 3]})
        df["w"] = _nan_series([1.0, np.nan, 3.0])

        def mutate(ns):
            ns["df"]["w"] = _nan_series([1.0, np.nan, 30.0])

        result = _diff_after({"df": df}, mutate)
        children = result.differences["df"].children
        assert "['w']" in children
        assert "['a']" not in children

    def test_nan_series_change_detected(self):
        s = _nan_series([np.nan, 2.0, np.nan])

        def mutate(ns):
            ns["s"][1] = 5.0

        result = _diff_after({"s": s}, mutate)
        assert "s" in result.differences

    def test_index_change_detected(self):
        df = cudf.DataFrame({"a": [1, 2, 3]})

        def mutate(ns):
            ns["df"].index = cudf.Index([10, 11, 12])

        result = _diff_after({"df": df}, mutate)
        assert "_index" in result.differences["df"].children


# =============================================================================
# GPU DataFrame diff: added / removed columns match the pandas path
# =============================================================================


@pytest.mark.usefixtures("gpu_checkpoint_mode")
class TestGpuDiffColumns:
    def test_added_column_is_the_only_column_added(self):
        df = cudf.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})

        def mutate(ns):
            ns["df"]["new"] = ns["df"]["a"] * 2

        result = _diff_after({"df": df}, mutate)
        changes = detect_changes(result)
        added = {c.column for c in changes if isinstance(c, ColumnAdded)}
        assert added == {"new"}
        assert not any(isinstance(c, ColumnModified) for c in changes)

    def test_removed_column_reported_as_removed(self):
        df = cudf.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})

        def mutate(ns):
            del ns["df"]["b"]

        result = _diff_after({"df": df}, mutate)
        changes = detect_changes(result)
        assert {c.column for c in changes if isinstance(c, ColumnRemoved)} == {"b"}
        assert not any(isinstance(c, ColumnAdded) for c in changes)

    def test_structural_columns_lists_only_added_names(self):
        from flowbook.kernel_support.diff import Diff
        from flowbook.kernel_support.structural_tracking import StructuralTrackingMode

        before = cudf.DataFrame({"a": [1, 2], "b": [3, 4]})
        after = before.copy(deep=True)
        after["c"] = [5, 6]
        differ = Diff(
            structural_reads={"df": {"columns"}},
            structural_mode=StructuralTrackingMode.ENFORCE,
        )
        node = cudf_compat._diff_cudf_gpu_dataframe(before, after, "df", differ)
        structural = node.children["_structural_columns"]
        assert structural.value1 == []
        assert structural.value2 == ["c"]


# =============================================================================
# cudf.pandas proxies (install() is process-global and irreversible, so these
# run in-process only where the order does not matter, and in a subprocess
# where it does)
# =============================================================================


@pytest.fixture
def cudf_pandas():
    import cudf.pandas

    cudf.pandas.install()
    import pandas as pd

    return pd


@pytest.mark.usefixtures("gpu_checkpoint_mode")
class TestCudfPandasProxyCheckpoints:
    def test_pandas_backed_proxy_index_unchanged_has_no_diff(self, cudf_pandas):
        pd = cudf_pandas
        train = pd.DataFrame({"a": np.arange(200) % 7, "w": np.random.rand(200)})
        cols = train.columns[train.columns != "w"]
        result = _diff_after({"train": train, "COLS": cols})
        assert "COLS" not in result.differences

    def test_proxy_index_rebinding_detected(self, cudf_pandas):
        pd = cudf_pandas
        train = pd.DataFrame({"a": np.arange(200) % 7, "w": np.random.rand(200)})

        def mutate(ns):
            ns["COLS"] = ns["train"].columns

        ns = {"train": train, "COLS": train.columns[train.columns != "w"]}
        result = _diff_after(ns, mutate)
        assert "COLS" in result.differences


_KERNEL_ORDER_SCRIPT = textwrap.dedent(
    """
    import json
    from flowbook.kernel_support.tracking import TrackingDict

    ns = TrackingDict()
    # First cell: the tracker's patches are installed before cudf.pandas
    with ns.track_execution("c0"):
        pass

    import cudf.pandas
    cudf.pandas.install()
    import pandas as pd

    with ns.track_execution("c1"):
        ns["df"] = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": [0, 0, 0]})

    with ns.track_execution("c2"):
        df = ns["df"]
        df["b"] = df["a"] * 2
        del df["c"]
    data = ns.get_tracking_data()

    print("RESULT " + json.dumps({
        "writes": {k: sorted(v) for k, v in data.column_writes.items()},
        "reads": {k: sorted(v) for k, v in data.column_reads_before_writes.items()},
        "deletions": {k: sorted(v) for k, v in data.column_deletions.items()},
    }))
    """
)


def test_proxy_column_tracking_when_cudf_pandas_installed_after_tracker():
    """`%load_ext cudf.pandas` runs after the tracker patched pandas."""
    import json

    result = subprocess.run(
        [sys.executable, "-c", _KERNEL_ORDER_SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    result_line = [l for l in result.stdout.splitlines() if l.startswith("RESULT ")][-1]
    data = json.loads(result_line[len("RESULT "):])
    assert data["writes"] == {"df": ["b"]}
    assert data["reads"] == {"df": ["a"]}
    assert data["deletions"] == {"df": ["c"]}


_SUBSET_SCRIPT = textwrap.dedent(
    """
    import json
    # cudf.pandas before FlowBook (e.g. a kernel started under
    # `python -m cudf.pandas`): proxies are then seen as DataFrames and the
    # subset detector stores `test` as a relation to `train`.
    import cudf.pandas
    cudf.pandas.install()
    import numpy as np
    import pandas as pd
    from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoint, MemoryCheckpoints

    def diff_keys(mutate):
        ns = {"train": pd.DataFrame({"a": np.arange(20000) % 7, "b": np.random.rand(20000)})}
        ns["test"] = ns["train"].iloc[:5000].copy()
        checkpoints = MemoryCheckpoints(sanity_check=False, warn_classes=False)
        checkpoints.save("pre", ns, max_size_mb=None)
        relations = [r.child_var for r in checkpoints.get("pre")._df_subset_relations]
        if mutate:
            ns["test"].loc[3, "b"] = -1.0
        result = MemoryCheckpoint.diff(checkpoints.get("pre"), ns)
        node = result.differences.get("test")
        return {
            "relations": relations,
            "changed": sorted(result.differences),
            "test_was_added": bool(node is not None and getattr(node, "value1", 0) is None),
        }

    print("RESULT " + json.dumps({"noop": diff_keys(False), "mutated": diff_keys(True)}))
    """
)


def test_row_subset_checkpointed_and_mutation_detected():
    """A proxy row subset is rebuilt from its checkpoint relation."""
    import json

    result = subprocess.run(
        [sys.executable, "-c", _SUBSET_SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    result_line = [l for l in result.stdout.splitlines() if l.startswith("RESULT ")][-1]
    data = json.loads(result_line[len("RESULT "):])
    assert data["noop"]["relations"] == ["test"]
    assert data["noop"]["changed"] == []
    assert data["mutated"]["changed"] == ["test"]
    assert not data["mutated"]["test_was_added"]


def test_is_dataframe_recognizes_proxy_dataframes(cudf_pandas):
    from flowbook.kernel_support.column_tracking import _is_dataframe, _is_series

    pd = cudf_pandas
    df = pd.DataFrame({"a": [1, 2]})
    assert _is_dataframe(df)
    assert _is_series(df["a"])
    assert not _is_dataframe(df["a"])
    assert not _is_series(df)
