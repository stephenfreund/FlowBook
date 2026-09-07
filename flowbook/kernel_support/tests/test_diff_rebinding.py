"""Rebinding a name that shared an object is not a mutation of the objects that still hold it.

Regression: the diff tracked pointer structure and reported "Pointer structure
mismatch" as a difference. After `cols = [...]` was re-run, every fitted model
whose ColumnTransformer still referenced the old list was reported as changed,
and the enforcer raised UNRECOVERABLE_MUTATION on a cell that only rebound a
name. Mutations through an alias must still be detected, and cycles must
terminate.
"""
from flowbook.kernel_support.checkpoint import Checkpoint, Checkpoints


class Holder:
    def __init__(self, cols):
        self.cols = cols


def _save(ns):
    cp = Checkpoints()
    saved, _ = cp.save("pre", dict(ns), max_size_mb=None)
    return saved


def test_rebinding_shared_name_is_not_a_change_to_the_holder():
    ns = {"cats": ["region"]}
    ns["holder"] = Holder(ns["cats"])
    saved = _save(ns)
    ns["cats"] = ["region"]  # new, equal list; holder still references the old one
    d = Checkpoint.diff(saved, ns, use_leq=False)
    assert "holder" not in d.differences
    assert "cats" not in d.differences


def test_mutation_through_alias_is_still_detected():
    ns = {"cats": ["region"]}
    ns["holder"] = Holder(ns["cats"])
    saved = _save(ns)
    ns["cats"].append("city")  # in place, visible through holder
    d = Checkpoint.diff(saved, ns, use_leq=False)
    assert {"cats", "holder"} <= set(d.differences)


def test_rebinding_to_a_different_value_reports_only_the_rebound_name():
    ns = {"cats": ["region"]}
    ns["holder"] = Holder(ns["cats"])
    saved = _save(ns)
    ns["cats"] = ["city"]
    d = Checkpoint.diff(saved, ns, use_leq=False)
    assert "cats" in d.differences
    assert "holder" not in d.differences


def test_cyclic_structures_with_mismatched_aliasing_terminate():
    a = []
    a.append(a)
    ns = {"x": a, "y": Holder(a)}
    saved = _save(ns)
    b = []
    b.append(b)
    ns["x"] = b  # rebind x to a fresh cycle; y still holds the old one
    d = Checkpoint.diff(saved, ns, use_leq=False)  # must not recurse forever
    assert "y" not in d.differences
