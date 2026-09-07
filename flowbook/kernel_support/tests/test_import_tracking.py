"""Module bindings created by import are tracked locations; re-importing the same module is not a write."""
import json as _json_mod
import types

import pytest

from flowbook.kernel_support.tracking import TrackingDict, rollback_module_bindings


def _run(ns, code, td=None):
    td = td or TrackingDict(ns)
    with td.track_execution(cell_id="c"):
        exec(code, td)
    return td, td.get_tracking_data()


def test_first_import_is_a_write_and_use_is_a_read():
    ns = {}
    _, t = _run(ns, "import json")
    assert "json" in t.writes
    td, t2 = _run(ns, "s = json.dumps({})")
    assert "json" in t2.reads_before_writes
    assert "json" not in t2.writes


def test_reimport_of_the_same_module_is_an_idempotent_write():
    ns = {}
    _run(ns, "import json")
    _, t = _run(ns, "import json\ns = json.dumps({})")
    assert "json" in t.writes
    assert t.rebound_same == {"json"}
    assert "json" not in t.reads_before_writes  # bound before use inside the cell


def test_first_import_is_not_idempotent():
    _, t = _run({}, "import json")
    assert t.rebound_same == set()


def test_rebinding_a_name_to_a_different_module_is_a_write():
    ns = {}
    _run(ns, "import json")
    _, t = _run(ns, "import os as json")
    assert "json" in t.writes and "json" not in t.rebound_same


def test_deleting_a_module_binding_is_a_write():
    ns = {}
    _run(ns, "import json")
    _, t = _run(ns, "del json")
    assert "json" in t.writes


def test_env_flag_restores_old_behaviour(monkeypatch):
    monkeypatch.setenv("FLOWBOOK_TRACK_IMPORTS", "0")
    ns = {}
    _, t = _run(ns, "import json")
    assert "json" not in t.writes
    _, t2 = _run(ns, "s = json.dumps({})")
    assert "json" not in t2.reads_before_writes


def test_rollback_module_bindings_undoes_new_import_and_rebinding():
    ns = {}
    _run(ns, "import os")
    td, _ = _run(ns, "import json\nimport sys as os\nx = 1")
    touched = rollback_module_bindings(td)
    assert sorted(touched) == ["json", "os"]
    assert "json" not in ns and ns["os"].__name__ == "os" and ns["x"] == 1


def test_rollback_module_bindings_leaves_an_idempotent_reimport_alone():
    ns = {}
    _run(ns, "import json")
    td, _ = _run(ns, "import json")
    assert rollback_module_bindings(td) == []
    assert ns["json"] is _json_mod
