"""
Tests for the liveness analysis module.
"""

import pytest
from data_ferret.util.liveness import (
    analyze_notebook_liveness,
    get_live_out_variables,
    get_live_in_variables,
    get_dead_after_cell,
    get_never_used_variables,
)


def test_simple_variable_chain():
    """Test basic variable usage chain."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "y = x + 10"},
            {"id": "c3", "cell_type": "code", "source": "z = y * 2"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Cell 1: x is live (needed by c2)
    assert "x" in liveness["c1"].live_out
    assert "y" not in liveness["c1"].live_out

    # Cell 2: y is live (needed by c3), x is dead
    assert "y" in liveness["c2"].live_out
    assert "x" not in liveness["c2"].live_out

    # Cell 3: nothing is live after last cell
    assert len(liveness["c3"].live_out) == 0


def test_variable_redefinition():
    """Test that variable redefinition kills old value."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "x = 10"},
            {"id": "c3", "cell_type": "code", "source": "y = x + 1"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Cell 1: x is killed by c2, so NOT live
    assert "x" not in liveness["c1"].live_out

    # Cell 2: x (new def) is live (needed by c3)
    assert "x" in liveness["c2"].live_out


def test_unused_variable():
    """Test variable defined but never used."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "y = 10"},
            {"id": "c3", "cell_type": "code", "source": "z = y * 2"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # x is never used
    assert "x" not in liveness["c1"].live_out
    assert "x" not in liveness["c2"].live_out

    # y is used by c3
    assert "y" in liveness["c2"].live_out


def test_multiple_reads():
    """Test variable read by multiple cells."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "y = x + 10"},
            {"id": "c3", "cell_type": "code", "source": "z = x * 2"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # x is live after c1 (used by c2 and c3)
    assert "x" in liveness["c1"].live_out
    # x is still live after c2 (used by c3)
    assert "x" in liveness["c2"].live_out
    # x becomes dead after c3
    assert "x" not in liveness["c3"].live_out


def test_function_definition_and_call():
    """Test function defined and called - dependencies stay live."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "def f():\n    return x + 10"},
            {"id": "c3", "cell_type": "code", "source": "result = f()"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Cell 1: x is live (needed by f in c3)
    assert "x" in liveness["c1"].live_out

    # Cell 2: both f and x are live (needed by c3)
    assert "f" in liveness["c2"].live_out
    assert "x" in liveness["c2"].live_out

    # Cell 3: nothing live after
    assert len(liveness["c3"].live_out) == 0


def test_function_definition_unused():
    """Test function defined but never called - function and deps are dead."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "def f():\n    return x + 10"},
            {"id": "c3", "cell_type": "code", "source": "y = 20"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Cell 1: x is NOT live (f is never called)
    assert "x" not in liveness["c1"].live_out

    # Cell 2: f is NOT live (never called)
    assert "f" not in liveness["c2"].live_out
    # x is also NOT live
    assert "x" not in liveness["c2"].live_out


def test_function_with_variable_redefinition():
    """Test variable redefined between function def and call."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "def f():\n    return x + 10"},
            {"id": "c3", "cell_type": "code", "source": "x = 20"},
            {"id": "c4", "cell_type": "code", "source": "result = f()"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Cell 1: x (old value) is NOT live (killed by c3)
    assert "x" not in liveness["c1"].live_out

    # Cell 2: only f is live
    assert "f" in liveness["c2"].live_out
    assert "x" not in liveness["c2"].live_out

    # Cell 3: both f and x (new value) are live
    assert "f" in liveness["c3"].live_out
    assert "x" in liveness["c3"].live_out


def test_nested_function_calls():
    """Test nested function calls with transitive dependencies."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "def g():\n    return y"},
            {"id": "c2", "cell_type": "code", "source": "def f():\n    return g() + 10"},
            {"id": "c3", "cell_type": "code", "source": "y = 5"},
            {"id": "c4", "cell_type": "code", "source": "result = f()"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Cell 1: g is live (called by f)
    assert "g" in liveness["c1"].live_out

    # Cell 2: both f and g are live
    assert "f" in liveness["c2"].live_out
    assert "g" in liveness["c2"].live_out

    # Cell 3: f, g, and y are all live
    assert "f" in liveness["c3"].live_out
    assert "g" in liveness["c3"].live_out
    assert "y" in liveness["c3"].live_out


def test_function_called_in_same_cell():
    """Test function defined and called in same cell."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "def f():\n    return x + 10\nresult = f()"},
            {"id": "c3", "cell_type": "code", "source": "y = 20"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Cell 1: x is live (used by f in c2)
    assert "x" in liveness["c1"].live_out

    # Cell 2: x is read, so it's in gen, and it's dead after
    assert "x" not in liveness["c2"].live_out


def test_class_definition_and_usage():
    """Test class definition and instantiation."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "class MyClass:\n    pass"},
            {"id": "c2", "cell_type": "code", "source": "obj = MyClass()"},
            {"id": "c3", "cell_type": "code", "source": "y = 10"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Cell 1: MyClass is live (used in c2)
    assert "MyClass" in liveness["c1"].live_out

    # Cell 2: MyClass is dead, obj might be live if used later
    assert "MyClass" not in liveness["c2"].live_out


def test_method_calls_with_dependencies():
    """Test methods calling functions with dependencies."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "data = [1, 2, 3]"},
            {"id": "c2", "cell_type": "code", "source": "def helper(x):\n    return x * 2"},
            {
                "id": "c3",
                "cell_type": "code",
                "source": "class Processor:\n    def transform(self, items):\n        return [helper(x) for x in items]"
            },
            {"id": "c4", "cell_type": "code", "source": "p = Processor()\nresult = p.transform(data)"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Cell 1: data is live (used in c4)
    assert "data" in liveness["c1"].live_out

    # Cell 2: helper is live (called by Processor.transform)
    assert "helper" in liveness["c2"].live_out

    # Cell 3: Processor is live
    assert "Processor" in liveness["c3"].live_out
    assert "helper" in liveness["c3"].live_out


def test_empty_cells():
    """Test that empty cells don't cause errors."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": ""},
            {"id": "c3", "cell_type": "code", "source": "y = x + 10"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Should have entries for all cells
    assert "c1" in liveness
    assert "c2" in liveness
    assert "c3" in liveness

    # x should be live through c2
    assert "x" in liveness["c1"].live_out
    assert "x" in liveness["c2"].live_out


def test_get_live_out_variables():
    """Test simplified API for live_out."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "y = x + 10"}
        ]
    }

    live_out = get_live_out_variables(notebook)

    assert "c1" in live_out
    assert "x" in live_out["c1"]
    assert "c2" in live_out
    assert len(live_out["c2"]) == 0


def test_get_live_in_variables():
    """Test simplified API for live_in."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "y = x + 10"}
        ]
    }

    live_in = get_live_in_variables(notebook)

    # c1 has nothing live before it
    assert "c1" in live_in
    assert len(live_in["c1"]) == 0

    # c2 needs x
    assert "c2" in live_in
    assert "x" in live_in["c2"]


def test_get_dead_after_cell():
    """Test getting dead variables after each cell."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5\ny = 10"},
            {"id": "c2", "cell_type": "code", "source": "z = x + y"},
            {"id": "c3", "cell_type": "code", "source": "w = z * 2"}
        ]
    }

    dead = get_dead_after_cell(notebook)

    # After c2, both x and y become dead
    assert "c2" in dead
    assert "x" in dead["c2"]
    assert "y" in dead["c2"]

    # After c3, z becomes dead
    assert "c3" in dead
    assert "z" in dead["c3"]


def test_get_never_used_variables():
    """Test detection of variables that are never used."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5\nunused = 10"},
            {"id": "c2", "cell_type": "code", "source": "y = x + 20"}
        ]
    }

    unused = get_never_used_variables(notebook)

    # unused variable is written in c1 but never read
    assert "c1" in unused
    assert "unused" in unused["c1"]
    assert "x" not in unused["c1"]  # x is used


def test_complex_dataflow():
    """Test complex dataflow pattern."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "a = 1\nb = 2\nc = 3"},
            {"id": "c2", "cell_type": "code", "source": "d = a + b"},
            {"id": "c3", "cell_type": "code", "source": "e = c + d"},
            {"id": "c4", "cell_type": "code", "source": "f = e * 2"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # c1: a, b, c all live
    assert "a" in liveness["c1"].live_out
    assert "b" in liveness["c1"].live_out
    assert "c" in liveness["c1"].live_out

    # c2: a, b dead; c, d live
    assert "a" not in liveness["c2"].live_out
    assert "b" not in liveness["c2"].live_out
    assert "c" in liveness["c2"].live_out
    assert "d" in liveness["c2"].live_out

    # c3: c, d dead; e live
    assert "c" not in liveness["c3"].live_out
    assert "d" not in liveness["c3"].live_out
    assert "e" in liveness["c3"].live_out

    # c4: e dead
    assert "e" not in liveness["c4"].live_out


def test_function_passed_as_callback():
    """Test function passed as callback."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "shared_value = 42"},
            {"id": "c2", "cell_type": "code", "source": "def callback(x):\n    return x + shared_value"},
            {"id": "c3", "cell_type": "code", "source": "result = df.apply(callback)"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # shared_value is live (used by callback which is called in c3)
    assert "shared_value" in liveness["c1"].live_out
    assert "shared_value" in liveness["c2"].live_out

    # callback is live
    assert "callback" in liveness["c2"].live_out


def test_variable_redefinition_multiple_times():
    """Test variable redefined multiple times."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 1"},
            {"id": "c2", "cell_type": "code", "source": "x = 2"},
            {"id": "c3", "cell_type": "code", "source": "x = 3"},
            {"id": "c4", "cell_type": "code", "source": "y = x + 10"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Only the x from c3 is live
    assert "x" not in liveness["c1"].live_out
    assert "x" not in liveness["c2"].live_out
    assert "x" in liveness["c3"].live_out


def test_gen_kill_sets():
    """Test that gen/kill sets are correctly populated."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "y = x + 10"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # c1: kills x, generates nothing
    assert "x" in liveness["c1"].kill
    assert len(liveness["c1"].gen) == 0

    # c2: kills y, generates x
    assert "y" in liveness["c2"].kill
    assert "x" in liveness["c2"].gen


def test_last_cell_live_out_empty():
    """Test that last cell always has empty live_out."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "y = x + 10"},
            {"id": "c3", "cell_type": "code", "source": "z = y * 2"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Last cell always has empty live_out
    assert len(liveness["c3"].live_out) == 0


def test_import_handling():
    """Test that imports are handled correctly."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "import pandas as pd"},
            {"id": "c2", "cell_type": "code", "source": "df = pd.DataFrame()"},
            {"id": "c3", "cell_type": "code", "source": "result = df.describe()"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # pd should NOT appear in liveness (it's a module, filtered out by dependencies.py)
    assert "pd" not in liveness["c1"].live_out

    # df should be live
    assert "df" in liveness["c2"].live_out


def test_multiple_functions_sharing_dependency():
    """Test two functions sharing a dependency."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "shared = 100"},
            {"id": "c2", "cell_type": "code", "source": "def f1():\n    return shared * 2"},
            {"id": "c3", "cell_type": "code", "source": "def f2():\n    return shared * 3"},
            {"id": "c4", "cell_type": "code", "source": "a = f1()"},
            {"id": "c5", "cell_type": "code", "source": "b = f2()"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # shared stays live throughout (needed by both functions)
    assert "shared" in liveness["c1"].live_out
    assert "shared" in liveness["c2"].live_out
    assert "shared" in liveness["c3"].live_out
    assert "shared" in liveness["c4"].live_out

    # f1 becomes dead after c4
    assert "f1" in liveness["c3"].live_out
    assert "f1" not in liveness["c4"].live_out

    # f2 is live through c4
    assert "f2" in liveness["c4"].live_out


def test_conditional_writes_in_cell():
    """Test that conditional writes are handled by dependencies.py."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "if True:\n    x = 5"},
            {"id": "c2", "cell_type": "code", "source": "y = x + 10"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # x is treated as written (dependencies.py handles conditionals conservatively)
    assert "x" in liveness["c1"].kill
    # x is live (needed by c2)
    assert "x" in liveness["c1"].live_out


def test_global_in_function():
    """Test global declaration in function."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "counter = 0"},
            {"id": "c2", "cell_type": "code", "source": "def increment():\n    global counter\n    counter += 1"},
            {"id": "c3", "cell_type": "code", "source": "increment()"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # counter is live (read by increment which is called in c3)
    assert "counter" in liveness["c1"].live_out
    assert "counter" in liveness["c2"].live_out


def test_lambda_with_closure():
    """Test lambda capturing variables.

    Note: This test demonstrates a limitation inherited from dependencies.py.
    Lambdas don't get transitive closure tracking (they're not in function_defs),
    so offset is only in c2's gen set, not propagated to c3's reads.
    When c3 calls f(), dependencies.py only includes {f}, not {f, offset}.
    """
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "offset = 10"},
            {"id": "c2", "cell_type": "code", "source": "f = lambda x: x + offset"},
            {"id": "c3", "cell_type": "code", "source": "result = f(5)"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # c2 reads offset directly (in gen set)
    assert "offset" in liveness["c2"].gen

    # offset is live before c2 (in live_in)
    assert "offset" in liveness["c2"].live_in

    # But offset is NOT in c2's live_out because dependencies.py doesn't
    # track transitive dependencies through lambdas (limitation)
    # So c3 only reads {f}, not {f, offset}
    assert "offset" not in liveness["c2"].live_out
    assert "f" in liveness["c2"].live_out

    # offset is killed after being read in c2
    dead_after_c2 = liveness["c2"].live_in - liveness["c2"].live_out
    assert "offset" in dead_after_c2


def test_list_comprehension():
    """Test list comprehension with dependencies."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "data = [1, 2, 3]\nmultiplier = 2"},
            {"id": "c2", "cell_type": "code", "source": "result = [x * multiplier for x in data]"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Both data and multiplier are live
    assert "data" in liveness["c1"].live_out
    assert "multiplier" in liveness["c1"].live_out


def test_no_code_cells():
    """Test notebook with no code cells."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "markdown", "source": "# Title"},
            {"id": "c2", "cell_type": "markdown", "source": "Some text"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    assert len(liveness) == 0


def test_mixed_cell_types():
    """Test notebook with mixed cell types."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "markdown", "source": "# Comment"},
            {"id": "c3", "cell_type": "code", "source": "y = x + 10"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # Should only have code cells
    assert "c1" in liveness
    assert "c2" not in liveness
    assert "c3" in liveness

    # x should be live
    assert "x" in liveness["c1"].live_out


def test_self_referential_update():
    """Test x = x + 1 pattern."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "x = x + 1"},
            {"id": "c3", "cell_type": "code", "source": "y = x * 2"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)

    # c1: x is live (read by c2)
    assert "x" in liveness["c1"].live_out

    # c2: x is live (needed by c3)
    # c2 both reads (gen) and writes (kill) x
    assert "x" in liveness["c2"].gen
    assert "x" in liveness["c2"].kill
    assert "x" in liveness["c2"].live_out


def test_to_dict_method():
    """Test CellLiveness.to_dict() method."""
    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5\ny = 10"},
            {"id": "c2", "cell_type": "code", "source": "z = x + y"}
        ]
    }

    liveness = analyze_notebook_liveness(notebook)
    cell_dict = liveness["c1"].to_dict()

    assert "cell_id" in cell_dict
    assert "live_in" in cell_dict
    assert "live_out" in cell_dict
    assert "gen" in cell_dict
    assert "kill" in cell_dict
    assert "dead_after" in cell_dict

    # Verify values are sorted lists
    assert isinstance(cell_dict["live_out"], list)
    assert isinstance(cell_dict["kill"], list)


def test_reuse_dependencies():
    """Test that we can provide pre-computed dependencies."""
    from data_ferret.util.dependencies import analyze_notebook

    notebook = {
        "cells": [
            {"id": "c1", "cell_type": "code", "source": "x = 5"},
            {"id": "c2", "cell_type": "code", "source": "y = x + 10"}
        ]
    }

    # Compute dependencies once
    dependencies = analyze_notebook(notebook)

    # Reuse for liveness analysis
    liveness = analyze_notebook_liveness(notebook, dependencies)

    assert "c1" in liveness
    assert "x" in liveness["c1"].live_out
