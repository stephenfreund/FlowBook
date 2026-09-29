"""TrackingDict as exec globals: class bodies, `global` statements, __module__, class-body read tracking.

Regression for 288f146, which made the kernel run cells with the TrackingDict as
globals. CPython reads/writes a globals dict subclass's own storage directly for
class-body lookups (LOAD_NAME), STORE_GLOBAL/DELETE_GLOBAL and a new function's or
class's __module__, and that storage was empty: `class A: v = q` raised NameError,
`global counter; counter += 1` was lost, `global x; del x` raised NameError, and
notebook functions/classes had __module__ None/'builtins'.

These tests exec cells the way the kernel and the simulator do (TrackingDict as
globals, begin_mirror/end_mirror around the code, class_scope.compile_cell).
"""
import ast
import builtins
import types

import pytest

from flowbook.kernel_support import class_scope
from flowbook.kernel_support.tracking import TrackingDict


class Notebook:
    def __init__(self):
        mod = types.ModuleType("__main__")
        mod.__dict__["__builtins__"] = builtins
        self.real = mod.__dict__
        self.td = TrackingDict(self.real)

    def run(self, src):
        """Execute one cell; return (reads_before_writes, writes)."""
        with self.td.track_execution(cell_id="c"):
            self.td.begin_mirror()
            try:
                exec(class_scope.compile_cell(src), self.td)
            finally:
                self.td.end_mirror()
        t = self.td.get_tracking_data()
        return set(t.reads_before_writes), set(t.writes)


@pytest.fixture
def nb():
    n = Notebook()
    n.run("q = 1\ncounter = 0\ngone = 1")
    return n


def test_class_body_reads_global_and_is_tracked(nb):
    reads, writes = nb.run("class A:\n    v = q\nres = A.v")
    assert nb.real["res"] == 1
    assert "q" in reads


def test_class_body_own_names_shadow_globals(nb):
    reads, _ = nb.run("class B:\n    q = 5\n    w = q * 2\nres = B.w")
    assert nb.real["res"] == 10
    assert "q" not in reads


def test_class_body_undefined_name_raises_nameerror(nb):
    with pytest.raises(NameError, match="name 'undefined_name' is not defined"):
        nb.run("class D:\n    v = undefined_name")


def test_module_of_notebook_functions_and_classes(nb):
    nb.run("class K:\n    pass\ndef f():\n    pass\nres = (K.__module__, f.__module__)")
    assert nb.real["res"] == ("__main__", "__main__")


def test_global_augassign_twice_in_one_cell_and_across_cells(nb):
    reads, writes = nb.run("def inc():\n    global counter\n    counter += 1\ninc()\ninc()\nres = counter")
    assert nb.real["res"] == 2 and nb.real["counter"] == 2
    assert "counter" in reads and "counter" in writes
    nb.run("inc()\nres = counter")
    assert nb.real["counter"] == 3


def test_global_creation_is_a_write_before_the_read(nb):
    reads, writes = nb.run("def mk():\n    global made\n    made = 42\nmk()\nres = made")
    assert nb.real["made"] == 42 and nb.real["res"] == 42
    assert "made" in writes and "made" not in reads


def test_global_created_and_not_read_is_applied_at_cell_end(nb):
    _, writes = nb.run("def mk2():\n    global later\n    later = 7\nmk2()")
    assert nb.real["later"] == 7 and "later" in writes


def test_global_delete(nb):
    _, writes = nb.run("def rm():\n    global gone\n    del gone\nrm()\nres = 'gone' in dir()")
    assert nb.real["res"] is False and "gone" not in nb.real
    assert "gone" in writes


def test_comprehension_first_iterable_in_class_body(nb):
    reads, _ = nb.run("class C:\n    items = [q for _ in range(2)]\nres = C.items")
    assert nb.real["res"] == [1, 1] and "q" in reads


def test_enum_dataclass_and_custom_prepare(nb):
    reads, _ = nb.run("import enum\nclass E(enum.Enum):\n    X = q\n    Y = q + 1\nres = E.Y.value")
    assert nb.real["res"] == 2 and "q" in reads
    reads, _ = nb.run("from dataclasses import dataclass\n@dataclass\nclass P:\n    a: int = q\nres = P().a")
    assert nb.real["res"] == 1 and "q" in reads
    reads, _ = nb.run("class Meta(type):\n    @classmethod\n    def __prepare__(m, n, b):\n"
                      "        return {'injected': 5}\nclass U(metaclass=Meta):\n    v = injected + q\nres = U.v")
    assert nb.real["res"] == 6 and "q" in reads


def test_class_in_function_sees_enclosing_locals_and_tracks_globals(nb):
    reads, _ = nb.run("def outer(p):\n    z = 7\n    class In:\n        v = z + q + p\n    return In.v\nres = outer(1)")
    assert nb.real["res"] == 9 and "q" in reads


def test_super_and_class_cell(nb):
    nb.run("class S:\n    def m(self):\n        return __class__.__name__\n"
           "class T(S):\n    def m(self):\n        return super().m()\nres = T().m()")
    assert nb.real["res"] == "S"


def test_raw_storage_holds_no_copy_between_cells(nb):
    nb.run("x = 1\ndef f():\n    return x")
    assert set(dict.keys(nb.td)) <= {"__name__", "__builtins__"}
    # FlowBook restores checkpoints into the real namespace directly between cells
    nb.real["x"] = 99
    nb.run("res = f()")
    assert nb.real["res"] == 99


def test_transform_scope_rules():
    src = ("class M:\n    a = q\n    if flag:\n        def m(self, d=q2):\n            class Inner:\n"
           "                v = self_var + q3\n            self_var = 1\n            return Inner\n"
           "def outer(p):\n    z = 7\n    class In:\n        v = z + q + p\n"
           "        w = [q4 * i for i in range(n)]\n    return In\n")
    out = ast.unparse(class_scope.transform(ast.parse(src)))
    load = class_scope.CLASS_LOAD
    for name in ("q", "flag", "q2", "q3", "n"):
        assert f"{load}('{name}')" in out, name
    for name in ("z", "p", "self_var", "q4"):  # enclosing-function variables; comprehension body
        assert f"{load}('{name}')" not in out, name
