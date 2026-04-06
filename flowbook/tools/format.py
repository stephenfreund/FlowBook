"""Shared formatting functions for FlowBook tool output.

These produce human+LLM-readable text from raw NotebookSession results.
Used by both the MCP server and NBI extension.
"""

from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Location formatting
# ---------------------------------------------------------------------------

def format_loc(loc) -> str:
    """Format a ReadLoc/WriteLoc dict (or pre-formatted string) as readable text.

    {"type": "var", "name": "df"} -> "df"
    {"type": "col", "name": "df", "qualifier": "price"} -> "df.price"
    {"type": "struct", "name": "df", "qualifier": "columns"} -> "df[columns]"
    "df['age']" -> "df['age']"  (already formatted)
    """
    if isinstance(loc, str):
        return loc
    name = loc.get("name", "?")
    loc_type = loc.get("type", "var")
    qualifier = loc.get("qualifier")
    if loc_type == "col" and qualifier:
        return f"{name}.{qualifier}"
    if loc_type == "struct" and qualifier:
        return f"{name}[{qualifier}]"
    return name


def format_loc_list(locs: list) -> str:
    """Format a list of locs as comma-separated readable names."""
    if not locs:
        return "(none)"
    return ", ".join(format_loc(loc) for loc in locs)


# ---------------------------------------------------------------------------
# Error / staleness formatting
# ---------------------------------------------------------------------------

def format_error(error: Dict[str, Any]) -> str:
    """Format a reproducibility error dict as a readable string."""
    etype = error.get("error_type", "unknown")
    msg = error.get("message", "")
    locs = error.get("locations", [])
    causer = error.get("causer_cell", "")
    parts = [f"{etype}: {msg}"]
    if locs:
        parts.append(f"  Locations: {format_loc_list(locs)}")
    if causer:
        parts.append(f"  Causer cell: {causer}")
    return "\n".join(parts)


def format_staleness_reasons(reasons: Dict[str, List[Dict[str, Any]]]) -> str:
    """Format staleness reasons dict as readable text."""
    if not reasons:
        return "(none)"
    lines = []
    for cell_id, reason_list in reasons.items():
        reason_strs = []
        for r in reason_list:
            rtype = r.get("type", "unknown")
            loc = r.get("loc", "")
            cause = r.get("cell_id", "")
            s = rtype
            if loc:
                s += f": {loc}"
            if cause:
                s += f" (from cell {cause})"
            reason_strs.append(s)
        lines.append(f"  {cell_id}: {', '.join(reason_strs)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metadata formatting
# ---------------------------------------------------------------------------

def format_flowbook_meta(meta: Dict[str, Any]) -> str:
    """Format raw flowbook metadata into human+LLM-readable text."""
    lines = []

    reads = meta.get("read_locs", [])
    writes = meta.get("write_locs", [])
    changed = meta.get("changed_locs", [])
    errors = meta.get("errors", [])
    stale = meta.get("stale_cells", [])
    reasons = meta.get("staleness_reasons", {})

    lines.append(f"Reads: {format_loc_list(reads)}")
    lines.append(f"Writes: {format_loc_list(writes)}")
    if changed:
        lines.append(f"Changed: {format_loc_list(changed)}")

    if errors:
        lines.append("Errors:")
        for e in errors:
            lines.append(f"  - {format_error(e)}")
    else:
        lines.append("Errors: (none)")

    if stale:
        lines.append(f"Stale cells: {', '.join(stale)}")
        if reasons:
            lines.append(f"Staleness reasons:\n{format_staleness_reasons(reasons)}")
    else:
        lines.append("Stale cells: (none)")

    # Timing
    exec_ms = meta.get("execute_duration_ms")
    code_ms = meta.get("code_duration_ms")
    if exec_ms is not None:
        timing_parts = [f"total={exec_ms:.0f}ms"]
        if code_ms is not None:
            timing_parts.append(f"code={code_ms:.0f}ms")
        lines.append(f"Timing: {', '.join(timing_parts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _to_str(value) -> str:
    """Convert a value that may be a string or list of strings to a string."""
    if isinstance(value, list):
        return "".join(value)
    return str(value) if value is not None else ""


def format_outputs_text(outputs: List[Dict[str, Any]]) -> str:
    """Extract human-readable text from cell outputs."""
    parts = []
    for output in outputs:
        otype = output.get("output_type", "")
        if otype == "stream":
            parts.append(_to_str(output.get("text", "")))
        elif otype == "execute_result":
            data = output.get("data", {})
            if "text/plain" in data:
                parts.append(_to_str(data["text/plain"]))
        elif otype == "display_data":
            data = output.get("data", {})
            if "text/plain" in data:
                parts.append(_to_str(data["text/plain"]))
            elif "text/html" in data:
                parts.append("[HTML output]")
            elif "image/png" in data:
                parts.append("[Image output]")
        elif otype == "error":
            ename = output.get("ename", "Error")
            evalue = output.get("evalue", "")
            parts.append(f"{ename}: {evalue}")
    return "\n".join(parts) if parts else "(no output)"


# ---------------------------------------------------------------------------
# Composite tool-result formatters
# ---------------------------------------------------------------------------

def format_run_result(
    result: Dict[str, Any],
    label: str,
    meta: Optional[Dict[str, Any]] = None,
    output_preview_len: int = 200,
) -> str:
    """Format the result of a run_cell call."""
    cell_id = result.get("cell_id", "?")
    status = result.get("status", "?")
    line = f"{label} [{cell_id}]: {status}"

    if result.get("error_message"):
        line += f" — {result['error_message']}"

    output_text = result.get("outputs_text", "").strip()
    if output_text:
        preview = output_text[:output_preview_len]
        if len(output_text) > output_preview_len:
            preview += "..."
        line += f"\nOutput: {preview}"

    if meta:
        line += f"\n{format_flowbook_meta(meta)}"
    elif "flowbook" in result:
        line += f"\n{result['flowbook']}"

    return line


def format_violation_line(
    error: Dict[str, Any],
    label: str,
    cell_id: str,
) -> str:
    """Format a single violation for inclusion in a summary."""
    etype = error.get("error_type", "?")
    locs = error.get("locations", [])
    loc_str = ", ".join(format_loc(l) for l in locs) if locs else ""
    line = f"  {label} [{cell_id}]: {etype}"
    if loc_str:
        line += f" [{loc_str}]"
    return line


def format_rename_result(
    old_name: str,
    new_name: str,
    modified_labels: List[str],
    start_label: str,
) -> str:
    """Format the result of an alpha_rename call."""
    if not modified_labels:
        return f"No occurrences of '{old_name}' found from cell {start_label} onwards."
    return (
        f"Renamed '{old_name}' \u2192 '{new_name}'\n"
        f"Modified {len(modified_labels)} cells: {', '.join(modified_labels)}"
    )


def format_status(
    status: Dict[str, Any],
    label_fn: Callable[[str], str],
) -> str:
    """Format the result of a get_status call.

    Args:
        status: Dict from NotebookSession.get_status()
        label_fn: Converts cell_id to @A label
    """
    n_exec = status["executed"]
    n_total = status["total_code_cells"]
    violations = status["violations"]
    stale = status["stale_cells"]

    line = f"{n_exec}/{n_total} executed | {len(violations)} violations | {len(stale)} stale"

    if violations:
        for v in violations:
            etype = v.get("error_type", "?")
            cid = v.get("cell_id", "?")
            label = label_fn(cid)
            locs = v.get("locations", [])
            loc_str = ", ".join(format_loc(l) for l in locs) if locs else ""
            line += f"\n  {label} [{cid}]: {etype}"
            if loc_str:
                line += f" [{loc_str}]"

    if stale:
        for cid, reasons in stale.items():
            label = label_fn(cid)
            reason_strs = []
            for r in reasons:
                rtype = r.get("type", "?")
                loc = r.get("loc", "")
                s = rtype
                if loc:
                    s += f": {loc}"
                reason_strs.append(s)
            line += f"\n  {label} [{cid}] stale: {', '.join(reason_strs)}"

    return line
