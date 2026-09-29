"""A read of IPython's _/__/___ must not take over the path of the DataFrame it aliases.

When a cell ends in an expression, IPython's display hook reads _, __ and ___
through user_ns (the TrackingDict). If one of them holds the same DataFrame as a
real variable X (an earlier cell displayed X), TrackingDict._track_read used to
register that DataFrame under '___', so the column reads the cell made on X were
attributed to '___' and then dropped when the display hook wrote ___.
"""
import builtins

import pandas as pd

from flowbook.kernel_support.tracking import TrackingDict


def _run_cell(td, src, display_hook_reads=()):
    with td.track_execution(cell_id="c"):
        exec(src, {"__builtins__": builtins}, td)
        for name in display_hook_reads:  # what IPython's DisplayHook.update_user_ns does
            td.get(name)
        if display_hook_reads:
            td["___"], td["__"], td["_"] = td.get("__"), td.get("_"), 0
    return td.get_tracking_data()


def test_display_hook_read_of_result_var_keeps_real_variable_column_reads():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    td = TrackingDict({"X": df, "_": 1, "__": 2, "___": df})
    t = _run_cell(td, "v = X['a'].sum()", display_hook_reads=("_", "__", "___"))
    assert t.column_reads_before_writes.get("X") == {"a"}

