"""Diff-derived write locations are recorded for mutated variables, not for rebound ones.

Wᵢ is the tracking writes plus the locations a checkpoint diff finds changed
(notebook_state.record_execution). For a variable the cell rebinds to a different
object, that diff compares the new value with whatever an earlier cell left in the
name: `submission = <XGB predictions>` after a cell that bound `submission` to LGBM
predictions recorded Col(submission, num_sold) on its first run and not on a rerun.
Var(x) already conflicts with every read of x, so these locations are no longer
recorded for rebound variables; in-place mutations keep them.
"""
import pandas as pd

from flowbook.kernel.reproducibility_enforcer import ReproducibilityEnforcer
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints
from flowbook.kernel_support.models import TrackingData
from flowbook.kernel_support.tracking import TrackingDict


def _run(enforcer, checkpoints, cell_id, pre_ns, post_ns, tracking):
    checkpoints.save(f"_pre_{cell_id}", pre_ns, max_size_mb=None)
    enforcer.check(cell_id, checkpoints.saved[f"_pre_{cell_id}"], post_ns, tracking,
                   continue_on_violation=True)
    return {str(w) for w in enforcer._notebook_state.writes[cell_id]}


def _enforcer():
    checkpoints = MemoryCheckpoints()
    enforcer = ReproducibilityEnforcer(checkpoints)
    enforcer.set_cell_order(["c"])
    return enforcer, checkpoints


def test_rebinding_records_only_var_write():
    enforcer, checkpoints = _enforcer()
    old = pd.DataFrame({"id": [1, 2], "num_sold": [1.0, 2.0]})
    new = pd.DataFrame({"id": [1, 2], "num_sold": [5.0, 6.0]})
    w = _run(enforcer, checkpoints, "c", {"submission": old}, {"submission": new},
             TrackingData(writes={"submission"}, rebound={"submission"}))
    assert w == {"Var(submission)"}


def test_in_place_mutation_keeps_diff_derived_column_write():
    enforcer, checkpoints = _enforcer()
    df = pd.DataFrame({"id": [1, 2], "num_sold": [1.0, 2.0]})
    pre = {"submission": df}
    checkpoints.save("_pre_c", pre, max_size_mb=None)
    df["num_sold"] = [5.0, 6.0]  # same object, changed in place
    enforcer.check("c", checkpoints.saved["_pre_c"], {"submission": df},
                   TrackingData(writes=set(), column_writes={"submission": {"num_sold"}}),
                   continue_on_violation=True)
    w = {str(x) for x in enforcer._notebook_state.writes["c"]}
    assert "Col(submission, num_sold)" in w


def test_rebinding_to_same_object_is_not_rebound():
    """`x = x` then an in-place change: the diff still finds the mutated column."""
    enforcer, checkpoints = _enforcer()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    checkpoints.save("_pre_c", {"x": df}, max_size_mb=None)
    df["b"] = [9, 9]
    enforcer.check("c", checkpoints.saved["_pre_c"], {"x": df},
                   TrackingData(writes={"x"}, rebound=set()), continue_on_violation=True)
    w = {str(x) for x in enforcer._notebook_state.writes["c"]}
    assert "Col(x, b)" in w and "Var(x)" in w


def test_tracking_dict_reports_rebound_by_identity():
    td = TrackingDict({})
    td._tracking_enabled = False
    td["x"] = pd.DataFrame({"a": [1]})
    td["y"] = pd.DataFrame({"a": [1]})
    with td.track_execution(cell_id="c"):
        exec("x = pd.DataFrame({'a': [2]})\ny = y\ny['a'] = [5]\nz = 1", {"pd": pd, "__builtins__": __builtins__}, td)
    t = td.get_tracking_data()
    assert "x" in t.rebound          # new object
    assert "z" in t.rebound          # newly bound
    assert "y" not in t.rebound      # same object, mutated in place
