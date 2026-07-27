"""Read/write footprint comparison for the run-until-clean algorithm.

Implements the check from the paper's Progress theorem (RunToClean;
see FORMAL_DEVELOPMENT.md): the run-until-clean loop executes the first
stale cell in notebook order, recording the set E of cells it has
executed. If it executes a cell in E a second time, the new run's read
and write sets must match those recorded by the cell's previous run.
If they do not, the loop may fail to terminate — a rerun that drops a
write can mark an earlier cell stale (BackwardStale) again and again —
so the guard reports the rerun as a *warning* at that cell and the
loop continues. The hard stop is the per-cell run counter: once a cell
has run more than ``MAX_RERUNS_PER_CELL`` times in one sweep, the loop
fails with a potential non-termination error.

Deterministic cells never trigger the report, and neither do
nondeterministic cells whose read and write sets are fixed while the
values they write vary (e.g. cells that draw random numbers): the check
compares only the location sets, never values.

The comparison is at *name level*. Serialized loc dicts carry an
object-identity ``qualifier`` (a stable_id from
``flowbook.kernel.loc_ids``) that legitimately changes when a rerun
recreates an object (``df = pd.read_csv(...)`` allocates a new
DataFrame on every run), so integer qualifiers are dropped and each
location is keyed by ``(type, owner, name)`` where the owner is the
accessing variable name.
"""

from typing import Any, Dict, FrozenSet, List, Optional, Tuple

# Maximum runs of a single cell within one run-until-clean sweep. The
# paper's Progress theorem is qualitative (no run bound exists for
# footprint-unstable cells), so this is a pragmatic limit: warnings are
# issued while under it, and exceeding it stops the sweep.
MAX_RERUNS_PER_CELL = 10

# A name-level location key: (loc type, owning variable or None, name).
FootprintKey = Tuple[Optional[str], Optional[str], Optional[str]]


def canonical_loc_key(loc: Dict[str, Any]) -> FootprintKey:
    """Reduce a serialized ReadLoc/WriteLoc dict to a name-level key.

    String qualifiers are variable names and are kept; integer
    qualifiers are object stable_ids and are replaced by the recorded
    ``var_name`` (the variable used to access the object), which is
    stable across reruns of the same source.
    """
    qualifier = loc.get("qualifier")
    if isinstance(qualifier, str):
        owner: Optional[str] = qualifier
    else:
        owner = loc.get("var_name")
    return (loc.get("type"), owner, loc.get("name"))


def canonical_footprint(
    locs: Optional[List[Dict[str, Any]]],
) -> FrozenSet[FootprintKey]:
    """Canonicalize a list of serialized loc dicts to a comparable set."""
    return frozenset(
        canonical_loc_key(loc) for loc in (locs or []) if isinstance(loc, dict)
    )


def format_footprint_key(key: FootprintKey) -> str:
    """Human-readable rendering of a name-level location key."""
    loc_type, owner, name = key
    if loc_type == "var":
        return str(name)
    if loc_type == "col":
        return f"{owner}.{name}" if owner else f".{name}"
    if loc_type in ("cols", "rows"):
        return f"{loc_type}({owner or name})"
    if loc_type == "file":
        return f"file:{name}"
    return f"{loc_type}:{name}"


class RunToCleanGuard:
    """Implements the RunToClean rerun check for one sweep.

    Call :meth:`note_run` after every committed cell execution. The
    check triggers — returning a warning report — exactly when a
    *re-executed* cell (one already run this sweep) leaves a cell
    before itself stale: the only way staleness moves backward is a run
    dropping a write owned by an earlier cell, and when a rerun does
    that, the sweep may never terminate. The caller warns and
    continues, stopping only when :meth:`cap_exceeded` becomes true.

    The guard also remembers each cell's name-level read/write
    footprint, purely to *explain* a report (the trigger never compares
    footprints, so the report works even without tracking metadata for
    earlier runs).
    """

    def __init__(self) -> None:
        self._executed: set = set()
        self._recorded: Dict[str, Tuple[FrozenSet, FrozenSet]] = {}
        self._run_counts: Dict[str, int] = {}

    def run_count(self, cell_id: str) -> int:
        """Number of times ``note_run`` has seen ``cell_id`` this sweep."""
        return self._run_counts.get(cell_id, 0)

    def cap_exceeded(self, cell_id: str) -> bool:
        """True once ``cell_id`` has run more than MAX_RERUNS_PER_CELL times."""
        return self.run_count(cell_id) > MAX_RERUNS_PER_CELL

    def _note_footprint(
        self, cell_id: str, fb_meta: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, List[str]]]:
        """Record this run's footprint; describe the change if any.

        Returns None when this is the cell's first recorded footprint,
        when the footprint matches the previous run, or when no tracking
        metadata is available (the record is dropped so a later run is
        not compared against stale data).
        """
        if not fb_meta or (
            "read_locs" not in fb_meta and "write_locs" not in fb_meta
        ):
            self._recorded.pop(cell_id, None)
            return None
        reads = canonical_footprint(fb_meta.get("read_locs"))
        writes = canonical_footprint(fb_meta.get("write_locs"))
        previous = self._recorded.get(cell_id)
        self._recorded[cell_id] = (reads, writes)
        if previous is None:
            return None
        prev_reads, prev_writes = previous
        if prev_reads == reads and prev_writes == writes:
            return None

        def _fmt_all(keys):
            return sorted(format_footprint_key(k) for k in keys)

        return {
            "prev_reads": _fmt_all(prev_reads),
            "prev_writes": _fmt_all(prev_writes),
            "new_reads": _fmt_all(reads),
            "new_writes": _fmt_all(writes),
            "reads_added": _fmt_all(reads - prev_reads),
            "reads_removed": _fmt_all(prev_reads - reads),
            "writes_added": _fmt_all(writes - prev_writes),
            "writes_removed": _fmt_all(prev_writes - writes),
        }

    def note_run(
        self,
        cell_id: str,
        fb_meta: Optional[Dict[str, Any]],
        cell_order: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Note a committed run of ``cell_id``; report if it should warn.

        Returns None for first executions, and for re-executions that
        leave every cell before ``cell_id`` clean. Returns a report dict
        when a re-execution marked an earlier cell stale: key
        ``backward_stale`` lists those cells (document order),
        ``run_count`` gives this cell's run count this sweep, and the
        footprint-change keys (``prev_reads`` etc.) are included when
        tracking metadata allows the change to be spelled out.

        The caller must run cells first-stale-first: ``cell_id`` must
        have been the first stale (or unexecuted) cell when selected, so
        that any stale cell before it afterwards was marked by this run.
        """
        rerun = cell_id in self._executed
        self._executed.add(cell_id)
        self._run_counts[cell_id] = self._run_counts.get(cell_id, 0) + 1
        change = self._note_footprint(cell_id, fb_meta)
        if not rerun or not fb_meta:
            return None
        if cell_id not in cell_order:
            return None
        position = cell_order.index(cell_id)
        positions = {cid: idx for idx, cid in enumerate(cell_order)}
        backward = sorted(
            (
                cid
                for cid in fb_meta.get("stale_cells", [])
                if cid in positions and positions[cid] < position
            ),
            key=lambda cid: positions[cid],
        )
        if not backward:
            return None
        report: Dict[str, Any] = {
            "backward_stale": backward,
            "run_count": self._run_counts[cell_id],
        }
        if change is not None:
            report.update(change)
        return report


def _format_set(items: List[str]) -> str:
    return ", ".join(items) if items else "nothing"


def format_footprint_change(change: Dict[str, List[str]]) -> str:
    """Describe a footprint change by spelling out both runs' read and
    write sets in full."""
    return (
        f"the previous run read {_format_set(change.get('prev_reads', []))} "
        f"and wrote {_format_set(change.get('prev_writes', []))}, "
        f"but this run read {_format_set(change.get('new_reads', []))} "
        f"and wrote {_format_set(change.get('new_writes', []))}"
    )
