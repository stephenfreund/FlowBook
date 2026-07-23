"""Read/write footprint comparison for the run-until-clean algorithm.

Implements the check from the paper's Progress theorem (RunToClean;
see FORMAL_DEVELOPMENT.md): the run-until-clean loop executes the first
stale cell in notebook order, recording the set E of cells it has
executed. If it executes a cell in E a second time, the new run's read
and write sets must match those recorded by the cell's previous run.
If they do not, the loop may fail to terminate — a rerun that drops a
write can mark an earlier cell stale (BackwardStale) again and again —
so the loop reports potential non-termination at that cell.

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
    """Tracks per-cell footprints across one run-until-clean sweep.

    Call :meth:`note_run` after every committed cell execution. The
    first execution of a cell records its footprint and returns None;
    a re-execution whose read and write sets match returns None; a
    re-execution whose sets changed returns a change summary, which the
    loop reports as potential non-termination.
    """

    def __init__(self) -> None:
        self._recorded: Dict[str, Tuple[FrozenSet, FrozenSet]] = {}

    def note_run(
        self, cell_id: str, fb_meta: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, List[str]]]:
        """Record this run's footprint; report a change on re-execution.

        Returns None when this is the cell's first execution this sweep,
        when the footprint matches the previous run, or when no tracking
        metadata is available (nothing to compare — the cell's record is
        dropped so a later run is not compared against stale data).
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
        return {
            "reads_added": sorted(
                format_footprint_key(k) for k in reads - prev_reads
            ),
            "reads_removed": sorted(
                format_footprint_key(k) for k in prev_reads - reads
            ),
            "writes_added": sorted(
                format_footprint_key(k) for k in writes - prev_writes
            ),
            "writes_removed": sorted(
                format_footprint_key(k) for k in prev_writes - writes
            ),
        }


def format_footprint_change(change: Dict[str, List[str]]) -> str:
    """One-line description of a footprint change for reports."""
    parts = []
    for label, key in (
        ("reads +", "reads_added"),
        ("reads -", "reads_removed"),
        ("writes +", "writes_added"),
        ("writes -", "writes_removed"),
    ):
        if change.get(key):
            parts.append(f"{label}{{{', '.join(change[key])}}}")
    return "; ".join(parts) if parts else "footprint changed"
