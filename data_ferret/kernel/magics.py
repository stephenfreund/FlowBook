# ferret_kernel/magics.py
import os, re, json, types, copy
from typing import Any
import numpy as np
import pandas as pd
import dill

from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.display import display, Javascript

from scalene import scalene_profiler
from scalene.scalene_arguments import ScaleneArguments

from data_ferret.profiler.user_ns import UserNS, CellNS
from data_ferret.util.output import log, print, error, timer
from data_ferret.analysis.idempotent_cells import check_idempotent


@magics_class
class FerretMagics(Magics):
    """IPython (Jupyter) support for magics for DataFerret (%scalene)."""

    def _values_equal(self, a: Any, b: Any) -> bool:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.array_equiv(a, b)
        if isinstance(a, pd.Timestamp) and isinstance(b, pd.Timestamp):
            return a == b or (pd.isna(a) and pd.isna(b))
        if isinstance(a, pd.DatetimeIndex) and isinstance(b, pd.DatetimeIndex):
            return a.equals(b)
        if isinstance(a, (pd.DataFrame, pd.Series)) and isinstance(b, (pd.DataFrame, pd.Series)):
            return a.equals(b)
        if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
            return False
        if isinstance(a, (list, tuple, np.ndarray)) and isinstance(b, (list, tuple, np.ndarray)):
            if len(a) != len(b):
                return False
            return all(self._values_equal(x, y) for x, y in zip(a, b))
        if isinstance(a, set) and isinstance(b, set):
            return a == b
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(self._values_equal(a[k], b[k]) for k in a)
        return bool(a == b)

    def __init__(self, shell):
        super().__init__(shell)
        self.executed_cell_ids = {}
        self.cached_ns: dict[int, CellNS] = {}
        self.cached_cell_id = None
        self._last_state = {}
        self.cached_deltas: dict[str, dict[str, Any]] = {}
        # clean up any old files (keep your original behavior)
        for filename in os.listdir():
            if filename.startswith("_ipython"):
                try:
                    os.remove(filename)
                except Exception:
                    pass

    def _report(self, e: Exception, k: Any, current: Any, last: Any):
        error(f"Exception: {e}")
        error(f"Type of {k}: {type(current)}")
        error("Value of current:")
        error(current)
        error("Value of last:")
        error(last)
        error(f"Type of last {k}: {type(last)}")

    def snapshot_delta(self, cell_id: str):
        ip = get_ipython()
        current = {
            k: v for k, v in ip.user_ns.items()
            if not k.startswith("_")
            and k not in ("get_ipython", "In", "Out", "exit", "quit")
            and not isinstance(v, types.ModuleType)
        }
        added_keys = set(current) - set(self._last_state)
        removed_keys = set(self._last_state) - set(current)
        common = set(current) & set(self._last_state)

        changed = {}
        for k in common:
            try:
                if not self._values_equal(current[k], self._last_state[k]):
                    changed[k] = current[k]
            except Exception as e:
                self._report(e, k, current[k], self._last_state[k])
                changed[k] = current[k]

        self.cached_deltas[cell_id] = {
            "added": {k: current[k] for k in added_keys},
            "removed": {k: self._last_state[k] for k in removed_keys},
            "changed": {k: current[k] for k in changed},
        }
        if changed:
            error(f"Dynamic Idempotency check failed: {', '.join(changed.keys())}")

        with open(f"_ipython-delta-{cell_id}.json", "w", encoding="utf-8") as delta_file:
            delta = {"added": list(added_keys), "removed": list(removed_keys), "changed": list(changed.keys())}
            delta_file.write(json.dumps(delta))

        for k in removed_keys:
            self._last_state.pop(k, None)
        for k in changed:
            try:
                self._last_state[k] = copy.deepcopy(current[k])
            except Exception as e:
                self._report(e, k, current.get(k, "?"), self._last_state.get(k, "?"))
        for k in added_keys:
            try:
                self._last_state[k] = copy.deepcopy(current[k])
            except Exception as e:
                self._report(e, k, current.get(k, "?"), self._last_state.get(k, "?"))

    def run_code(self, args: ScaleneArguments, code: str) -> None:
        import IPython
        args.gpu = False
        args.memory = False
        args.json = False
        args.html = False
        args.web = False
        args.no_browser = True

        ip = IPython.get_ipython()

        if self.cached_cell_id is not None:
            cell_id = self.cached_cell_id
        else:
            hdr = ip.kernel._parent_header
            cell_id = hdr.get("metadata", {}).get("cellId")

        args.outfile = f"_ipython-profile-{cell_id}.txt"

        n = len(ip.history_manager.input_hist_raw) - 1
        filename = f"_ipython-input-{n}-profile"
        with open(filename, "w") as tmpfile:
            tmpfile.write(code)

        with timer(key="capture_namespace", message="Capturing namespace"):
            in_vars = UserNS.from_dict(ip.user_ns)

        scalene_profiler.Scalene.set_initialized()
        with timer(message="Running profiler"):
            scalene_profiler.Scalene.run_profiler(args, [filename], is_jupyter=True)

        with timer(key="capture_namespace", message="Capturing namespace"):
            out_vars = UserNS.from_dict(ip.user_ns)

        self.executed_cell_ids[n] = cell_id
        self.cached_ns[n] = CellNS(cell_id=cell_id, in_vars=in_vars, out_vars=out_vars)

        with timer(message="Saving namespace"):
            with open(f"_ipython-ns-{cell_id}.json", "w", encoding="utf-8") as ns_file:
                ns_file.write(self.cached_ns[n].model_dump_json(indent=2))

        with timer(message="Saving profile"):
            if os.path.exists(args.outfile):
                with open(args.outfile, "r", encoding="utf-8") as profile_file:
                    profile = profile_file.read()
                profile = self.replace_filenames_with_cell_ids(profile)
                with open(args.outfile, "w", encoding="utf-8") as profile_file:
                    profile_file.write(profile)
            else:
                error(f"Profile file {args.outfile} does not exist")

    def replace_filenames_with_cell_ids(self, text: str) -> str:
        pattern = r"/[^\s]*?_ipython-input-(\d+)-profile"
        def repl(m):
            n = int(m.group(1))
            return f"Cell {self.executed_cell_ids.get(n, n)}"
        return re.sub(pattern, repl, text)

    @line_cell_magic
    def scalene(self, line: str, cell: str = "") -> None:
        args = ScaleneArguments()
        if cell:
            try:
                print(cell)
                self.run_code(args, cell)
            except Exception as e:
                error(f"Error running code: {e}")
                raise e

    @line_cell_magic
    def cell_id(self, line: str, cell: str = "") -> None:
        self.cached_cell_id = line
