"""Column accesses are copied to the aliases of the object the cell accessed.

check() copies a cell's column writes (and column/structural reads) from the name
the column tracker resolved them to onto that name's aliases. The aliases used to
come only from the pre-cell checkpoint, i.e. from what the names held before the
cell. For a frame the cell creates,

    train_transformed = create_corr_features(train_temp)   # columns written inside
    X = train_transformed

the tracker credits the writes to X. On a first run neither name existed before,
so nothing was copied; on a rerun both names held the previous run's shared frame,
so the writes were copied to train_transformed. Wᵢ depended on history.

Now pre-state aliasing is followed only between names the cell did not rebind,
and column writes also go to the names holding the frame after the cell.
"""
import pandas as pd

from flowbook.kernel.reproducibility_enforcer import ReproducibilityEnforcer
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints
from flowbook.kernel_support.models import TrackingData


def _enforcer():
    checkpoints = MemoryCheckpoints()
    enforcer = ReproducibilityEnforcer(checkpoints)
    enforcer.set_cell_order(["c"])
    return enforcer, checkpoints


def _check(enforcer, checkpoints, pre_ns, post_ns, tracking):
    checkpoints.save("_pre_c", pre_ns, max_size_mb=None)
    enforcer.check("c", checkpoints.saved["_pre_c"], post_ns, tracking,
                   continue_on_violation=True)
    return enforcer._notebook_state.tracking_data["c"].column_writes


def _corr_cell(train_temp):
    """What the cell leaves behind, and the TrackingData it produces."""
    train_transformed = train_temp.copy()
    train_transformed["f1_x_f2"] = train_transformed["f1"] * train_transformed["f2"]
    X = train_transformed
    post = {"train_temp": train_temp, "train_transformed": train_transformed, "X": X}
    tracking = TrackingData(
        reads_before_writes={"train_temp"},
        writes={"train_transformed", "X"},
        rebound={"train_transformed", "X"},
        column_reads_before_writes={"train_temp": {"f1", "f2"}},
        column_writes={"X": {"f1_x_f2"}},  # per object, last registered name
    )
    return post, tracking


def test_new_frame_writes_same_on_first_run_and_rerun():
    train_temp = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    expected = {"X": {"f1_x_f2"}, "train_transformed": {"f1_x_f2"}}

    # First run: neither name bound before the cell.
    enforcer, checkpoints = _enforcer()
    post1, tracking = _corr_cell(train_temp)
    first = _check(enforcer, checkpoints, {"train_temp": train_temp}, post1, tracking)
    assert first == expected

    # Rerun: both names hold the previous run's shared frame before the cell.
    post2, tracking = _corr_cell(train_temp)
    rerun = _check(enforcer, checkpoints, post1, post2, tracking)
    assert rerun == expected

    w = {str(x) for x in enforcer._notebook_state.writes["c"]}
    assert {"Col(X, f1_x_f2)", "Col(train_transformed, f1_x_f2)"} <= w


def test_new_frame_does_not_take_reads_through_old_aliases():
    """A read of p is not copied to x when x was p's alias but the cell rebound x."""
    enforcer, checkpoints = _enforcer()
    p = pd.DataFrame({"a": [1, 2]})
    pre = {"p": p, "x": p}
    post = {"p": p, "x": pd.DataFrame({"b": [0, 0]})}
    checkpoints.save("_pre_c", pre, max_size_mb=None)
    enforcer.check("c", checkpoints.saved["_pre_c"], post,
                   TrackingData(reads_before_writes={"p"}, writes={"x"}, rebound={"x"},
                                column_reads_before_writes={"p": {"a"}},
                                structural_reads={"p": {"shape"}}),
                   continue_on_violation=True)
    td = enforcer._notebook_state.tracking_data["c"]
    assert td.column_reads_before_writes == {"p": {"a"}}
    assert td.structural_reads == {"p": {"shape"}}


def test_in_place_write_to_existing_alias_still_expands():
    """p and x share a frame before the cell; p['col'] = ... also writes x['col']."""
    enforcer, checkpoints = _enforcer()
    p = pd.DataFrame({"a": [1, 2]})
    checkpoints.save("_pre_c", {"p": p, "x": p}, max_size_mb=None)
    p["col"] = [3, 4]
    enforcer.check("c", checkpoints.saved["_pre_c"], {"p": p, "x": p},
                   TrackingData(column_writes={"p": {"col"}}), continue_on_violation=True)
    td = enforcer._notebook_state.tracking_data["c"]
    assert td.column_writes == {"p": {"col"}, "x": {"col"}}


def test_in_place_write_reaches_deep_alias():
    """A container holding the frame before the cell is still a (deep) alias."""
    enforcer, checkpoints = _enforcer()
    p = pd.DataFrame({"a": [1, 2]})
    data = {"train": p}
    checkpoints.save("_pre_c", {"p": p, "data": data}, max_size_mb=None)
    p["col"] = [3, 4]
    enforcer.check("c", checkpoints.saved["_pre_c"], {"p": p, "data": data},
                   TrackingData(column_writes={"p": {"col"}}), continue_on_violation=True)
    assert enforcer._notebook_state.tracking_data["c"].column_writes == {
        "p": {"col"}, "data": {"col"}}


def test_new_frame_in_new_container_same_on_rerun():
    """`X = f(...); data = {'train': X}`: data is credited on the first run and on a rerun."""
    base = pd.DataFrame({"a": [1, 2]})

    def cell():
        X = base.copy()
        X["b"] = X["a"] + 1
        post = {"base": base, "X": X, "data": {"train": X}}
        return post, TrackingData(reads_before_writes={"base"}, writes={"X", "data"},
                                  rebound={"X", "data"}, column_writes={"X": {"b"}})

    enforcer, checkpoints = _enforcer()
    post1, t = cell()
    first = _check(enforcer, checkpoints, {"base": base}, post1, t)
    post2, t = cell()
    rerun = _check(enforcer, checkpoints, post1, post2, t)
    assert first == rerun == {"X": {"b"}, "data": {"b"}}


def test_rebinding_to_existing_frame_credits_its_holders():
    """`X = train; X['c'] = 1`: train is written on the first run, not only on a rerun."""
    train = pd.DataFrame({"a": [1, 2]})
    tracking = TrackingData(writes={"X"}, rebound={"X"}, column_writes={"X": {"c"}})

    enforcer, checkpoints = _enforcer()
    checkpoints.save("_pre_c", {"train": train}, max_size_mb=None)
    train["c"] = 1
    enforcer.check("c", checkpoints.saved["_pre_c"], {"train": train, "X": train}, tracking,
                   continue_on_violation=True)
    assert enforcer._notebook_state.tracking_data["c"].column_writes == {
        "X": {"c"}, "train": {"c"}}
