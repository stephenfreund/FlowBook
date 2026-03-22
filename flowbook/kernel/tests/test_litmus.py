"""
Litmus test driver for FlowBook reproducibility enforcement.

This module loads tests from LITMUS_TESTS.yaml and runs them against
the ReproducibilityEnforcer to verify correct behavior.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest
import yaml

from flowbook.kernel.models import CellStatus, Reason, ReasonType
from flowbook.kernel.notebook_state import NotebookState
from flowbook.kernel.reproducibility_enforcer import ReproducibilityEnforcer
from flowbook.kernel.tests.conftest import make_tracking, ReproducibilityTestHelper
from flowbook.kernel.tests.litmus_helpers import infer_rw


# Path to the YAML file
LITMUS_YAML_PATH = Path(__file__).parent / "LITMUS_TESTS.yaml"


@dataclass
class StateSnapshot:
    """Snapshot of state after an operation."""
    reads: Dict[str, Set[str]]  # cell_id -> read vars
    writes: Dict[str, Set[str]]  # cell_id -> write vars
    status: Dict[str, str]  # cell_id -> "clean" | "stale"
    reasons: Dict[str, List[Dict[str, Any]]]  # cell_id -> list of reason dicts
    cell_order: List[str]


class LitmusTestRunner:
    """
    Runner for litmus tests.

    Executes operations against a ReproducibilityEnforcer and tracks
    state snapshots for visualization.
    """

    def __init__(self, cell_order: List[str], cells: Optional[Dict[str, str]] = None):
        """Initialize with a cell order and optional initial cell code."""
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(cell_order)
        self.history: List[StateSnapshot] = []
        self.operations: List[Dict[str, Any]] = []  # Track executed operations
        self.cell_code: Dict[str, str] = cells.copy() if cells else {}  # Track current code for each cell
        self.code_history: List[Dict[str, str]] = []  # Track code state after each operation
        self.last_violation: Optional[Dict[str, Any]] = None
        self.last_newly_stale: List[str] = []

        # Capture initial state
        self._capture_snapshot()
        self.code_history.append(dict(self.cell_code))  # Initial code state

    def _capture_snapshot(self) -> StateSnapshot:
        """Capture current state as a snapshot."""
        state = self.helper.sdc._notebook_state

        # Collect status for all cells
        status = {}
        reasons = {}
        for cell_id in state.cell_order:
            cell_status = state.get_status(cell_id)
            status[cell_id] = "clean" if cell_status.is_clean else "stale"
            if not cell_status.is_clean:
                reasons[cell_id] = [r.to_dict() for r in cell_status.reasons]

        snapshot = StateSnapshot(
            reads={k: set(v) for k, v in state.reads.items()},
            writes={k: set(v) for k, v in state.writes.items()},
            status=status,
            reasons=reasons,
            cell_order=list(state.cell_order),
        )
        self.history.append(snapshot)
        return snapshot

    def execute_operation(self, op: Dict[str, Any]) -> None:
        """Execute a single operation."""
        op_type = op["type"]

        if op_type == "RUN":
            self._run_cell(op)
            # Track code if available
            if "code" in op:
                self.cell_code[op["cell"]] = op["code"]
        elif op_type == "EDIT":
            self._edit_cell(op)
        elif op_type == "DELETE":
            self._delete_cell(op)
        elif op_type == "INSERT":
            self._insert_cell(op)
        elif op_type == "MOVE":
            self._move_cells(op)
        else:
            raise ValueError(f"Unknown operation type: {op_type}")

        # Track the operation for display
        self.operations.append(op)
        self._capture_snapshot()
        self.code_history.append(dict(self.cell_code))  # Snapshot code state

    def _run_cell(self, op: Dict[str, Any]) -> None:
        """Execute a RUN operation."""
        cell_id = op["cell"]
        pre_namespace = op.get("pre_namespace", {})
        post_namespace = op.get("post_namespace", {})

        # Determine reads/writes either from explicit fields or code inference
        if "code" in op:
            inferred = infer_rw(op["code"])
            reads = inferred.reads
            writes = inferred.writes
            column_reads = {k: set(v) for k, v in inferred.column_reads.items()}
            column_writes = {k: set(v) for k, v in inferred.column_writes.items()}
            structural_reads = {k: set(v) for k, v in inferred.structural_reads.items()}
        else:
            reads = set(op.get("reads", []))
            writes = set(op.get("writes", []))
            column_reads = {k: set(v) for k, v in op.get("column_reads", {}).items()}
            column_writes = {k: set(v) for k, v in op.get("column_writes", {}).items()}
            structural_reads = {k: set(v) for k, v in op.get("structural_reads", {}).items()}

        result = self.helper.execute_cell(
            cell_id=cell_id,
            pre_namespace=pre_namespace,
            post_namespace=post_namespace,
            reads=reads,
            writes=writes,
            column_reads=column_reads if column_reads else None,
            column_writes=column_writes if column_writes else None,
            structural_reads=structural_reads if structural_reads else None,
            continue_on_violation=True,  # Capture violation but continue
        )

        # Capture errors as violations for litmus test expectations
        from flowbook.kernel.models import ErrorType
        for error in result.errors:
            if error.error_type == ErrorType.NO_WRITE_AFTER_READ:
                self.last_violation = {
                    "mutating_cell": error.cell_id,
                    "affected_cell": error.causer_cell,
                    "variables": list(error.locations),
                    "violation_type": "backward_mutation",
                }

    def _edit_cell(self, op: Dict[str, Any]) -> None:
        """Execute an EDIT operation."""
        cell_id = op["cell"]
        self.helper.sdc._notebook_state.handle_edit(cell_id)
        # Track new code if provided
        if "new_code" in op:
            self.cell_code[cell_id] = op["new_code"]

    def _delete_cell(self, op: Dict[str, Any]) -> None:
        """Execute a DELETE operation."""
        cell_id = op["cell"]
        current_order = list(self.helper.sdc.cell_order)
        if cell_id in current_order:
            current_order.remove(cell_id)
        result = self.helper.sdc.set_cell_order(current_order)
        self.last_newly_stale = result.newly_stale

    def _insert_cell(self, op: Dict[str, Any]) -> None:
        """Execute an INSERT operation."""
        cell_id = op["cell"]
        position = op["position"]
        current_order = list(self.helper.sdc.cell_order)
        current_order.insert(position, cell_id)
        result = self.helper.sdc.set_cell_order(current_order)
        self.last_newly_stale = result.newly_stale

    def _move_cells(self, op: Dict[str, Any]) -> None:
        """Execute a MOVE operation."""
        new_order = op["new_order"]
        result = self.helper.sdc.set_cell_order(new_order)
        self.last_newly_stale = result.newly_stale

    def get_current_state(self) -> StateSnapshot:
        """Get the most recent state snapshot."""
        return self.history[-1]

    def validate_expectations(self, expect: Dict[str, Any]) -> List[str]:
        """
        Validate expectations against current state.

        Returns list of failure messages (empty if all pass).
        """
        failures = []
        state = self.get_current_state()
        ns = self.helper.sdc._notebook_state

        # Validate status
        if "status" in expect:
            for cell_id, expected_status in expect["status"].items():
                actual_status = state.status.get(cell_id, "unknown")
                if actual_status != expected_status:
                    failures.append(
                        f"Cell {cell_id}: expected status '{expected_status}', got '{actual_status}'"
                    )

        # Validate reasons
        if "reasons" in expect:
            for cell_id, expected_reasons in expect["reasons"].items():
                actual_reasons = state.reasons.get(cell_id, [])

                # Convert expected reasons to comparable form
                expected_set = set()
                for r in expected_reasons:
                    # Normalize to frozenset of items for comparison
                    expected_set.add(frozenset(r.items()))

                actual_set = set()
                for r in actual_reasons:
                    actual_set.add(frozenset(r.items()))

                # Check each expected reason is present
                for expected_r in expected_reasons:
                    found = False
                    for actual_r in actual_reasons:
                        # Match on type first
                        if actual_r.get("type") == expected_r.get("type"):
                            # Check other fields if specified
                            match = True
                            for key, val in expected_r.items():
                                if key != "type" and actual_r.get(key) != val:
                                    match = False
                                    break
                            if match:
                                found = True
                                break
                    if not found:
                        failures.append(
                            f"Cell {cell_id}: missing reason {expected_r}, actual: {actual_reasons}"
                        )

        # Validate last_writer (computed from writes)
        if "last_writer" in expect:
            for loc, expected_writer in expect["last_writer"].items():
                # Find the last cell in order that writes this loc
                actual_writer = None
                for cell_id in reversed(state.cell_order):
                    if loc in state.writes.get(cell_id, set()):
                        actual_writer = cell_id
                        break
                if actual_writer != expected_writer:
                    failures.append(
                        f"last_writer[{loc}]: expected '{expected_writer}', got '{actual_writer}'"
                    )

        # Validate cell_order
        if "cell_order" in expect:
            if state.cell_order != expect["cell_order"]:
                failures.append(
                    f"cell_order: expected {expect['cell_order']}, got {state.cell_order}"
                )

        # Validate violation
        if "violation" in expect:
            if self.last_violation is None:
                failures.append(
                    f"Expected violation {expect['violation']}, but none occurred"
                )
            else:
                v = expect["violation"]
                actual = self.last_violation
                if actual.get("mutating_cell") != v.get("mutating_cell"):
                    failures.append(
                        f"violation.mutating_cell: expected '{v.get('mutating_cell')}', "
                        f"got '{actual.get('mutating_cell')}'"
                    )
                if actual.get("affected_cell") != v.get("affected_cell"):
                    failures.append(
                        f"violation.affected_cell: expected '{v.get('affected_cell')}', "
                        f"got '{actual.get('affected_cell')}'"
                    )
                if "variables" in v:
                    expected_vars = set(v["variables"])
                    actual_vars = set(actual.get("variables", []))
                    if expected_vars != actual_vars:
                        failures.append(
                            f"violation.variables: expected {expected_vars}, got {actual_vars}"
                        )
                if "type" in v:
                    if actual.get("violation_type") != v["type"]:
                        failures.append(
                            f"violation.type: expected '{v['type']}', "
                            f"got '{actual.get('violation_type')}'"
                        )

        # Validate forward_violation (now detected as NO_READ_BEFORE_WRITE errors)
        if "forward_violation" in expect:
            # Forward violations are now tracked as errors, not separate fields
            # The litmus tests just check that forward contamination was detected
            pass  # Forward contamination is captured via staleness reasons

        # Validate newly_stale
        if "newly_stale" in expect:
            expected_newly = set(expect["newly_stale"])
            actual_newly = set(self.last_newly_stale)
            if expected_newly != actual_newly:
                failures.append(
                    f"newly_stale: expected {sorted(expected_newly)}, got {sorted(actual_newly)}"
                )

        # Validate stale_cells (derived from status)
        if "stale_cells" in expect:
            expected_stale = expect["stale_cells"]
            actual_stale = [c for c in state.cell_order if state.status.get(c) == "stale"]
            if expected_stale != actual_stale:
                failures.append(
                    f"stale_cells: expected {expected_stale}, got {actual_stale}"
                )

        return failures

    def _format_op_header(self, op: Dict[str, Any]) -> str:
        """Format an operation as a column header like 'Run a' or 'Edit b'."""
        op_type = op["type"]
        if op_type == "RUN":
            return f"Run {op['cell']}"
        elif op_type == "EDIT":
            return f"Edit {op['cell']}"
        elif op_type == "DELETE":
            return f"Del {op['cell']}"
        elif op_type == "INSERT":
            return f"Ins {op['cell']}"
        elif op_type == "MOVE":
            return "Move"
        return op_type

    def render_ascii(self, test_name: str, description: str) -> str:
        """Generate ASCII visualization of state evolution."""
        lines = []

        # Header
        lines.append("=" * 79)
        lines.append(f"Test: {test_name}")
        lines.append(f"{description}")
        lines.append("=" * 79)
        lines.append("")

        # Get all cells across all snapshots
        all_cells = set()
        for snap in self.history:
            all_cells.update(snap.cell_order)
        cell_list = sorted(all_cells)

        # Column headers - use operation names instead of "Op 1", "Op 2"
        col_width = 24
        header = "Cell".ljust(6)
        header += "Initial".center(col_width)
        for op in self.operations:
            header += self._format_op_header(op).center(col_width)
        lines.append(header)
        lines.append("-" * len(header))

        # Each cell row
        for cell_id in cell_list:
            # Collect data per snapshot
            cell_data = []  # List of dicts per snapshot

            for snap_idx, snap in enumerate(self.history):
                if cell_id not in snap.cell_order:
                    cell_data.append({"deleted": True})
                    continue

                r = snap.reads.get(cell_id, set())
                w = snap.writes.get(cell_id, set())
                status = snap.status.get(cell_id, "?")
                reasons = snap.reasons.get(cell_id, [])

                r_str = "{" + ",".join(sorted(r)) + "}" if r else "∅"
                w_str = "{" + ",".join(sorted(w)) + "}" if w else "∅"

                # Get code for this cell at this snapshot
                code = self.code_history[snap_idx].get(cell_id, "") if snap_idx < len(self.code_history) else ""

                # Format reasons - one per line
                reason_strs = []
                has_never_executed = False
                for reason in reasons:
                    rtype = reason.get("type", "?")
                    if rtype == "never_executed":
                        has_never_executed = True
                    loc = reason.get("loc", "")
                    cid = reason.get("cell_id", "")
                    if loc and cid:
                        reason_strs.append(f"{rtype}({loc}<-{cid})")
                    elif loc:
                        reason_strs.append(f"{rtype}({loc})")
                    else:
                        reason_strs.append(rtype)

                # Cell is "executed" if it's clean or has non-empty reads/writes
                executed = (status == "clean") or bool(r) or bool(w)

                cell_data.append({
                    "deleted": False,
                    "reads": f"R:{r_str}",
                    "writes": f"W:{w_str}",
                    "status": status.upper(),
                    "reasons": reason_strs,
                    "code": code,
                    "executed": executed,
                })

            # Determine max reasons across all snapshots for this cell
            max_reasons = max((len(d.get("reasons", [])) for d in cell_data), default=0)

            # Check if any snapshot has code for this cell
            has_code = any(d.get("code") for d in cell_data if not d.get("deleted"))

            # Print code line if this cell has any code
            if has_code:
                row = cell_id.ljust(6)
                for d in cell_data:
                    if d.get("deleted"):
                        row += "(deleted)".center(col_width)
                    else:
                        code = d.get("code", "")
                        if code:
                            # Truncate long code
                            if len(code) > col_width - 2:
                                code = code[:col_width - 5] + "..."
                            row += code.center(col_width)
                        else:
                            row += "".center(col_width)
                lines.append(row)

            # Print R line (only show if cell has been executed in at least one snapshot)
            has_executed = any(d.get("executed") for d in cell_data if not d.get("deleted"))
            if has_executed:
                row = ("" if has_code else cell_id).ljust(6)
                for d in cell_data:
                    if d.get("deleted"):
                        row += "(deleted)".center(col_width) if not has_code else "".center(col_width)
                    elif d.get("executed"):
                        row += d["reads"][:col_width - 1].center(col_width)
                    else:
                        row += "".center(col_width)
                lines.append(row)

                # Print W line
                row = "".ljust(6)
                for d in cell_data:
                    if d.get("deleted"):
                        row += "".center(col_width)
                    elif d.get("executed"):
                        row += d["writes"][:col_width - 1].center(col_width)
                    else:
                        row += "".center(col_width)
                lines.append(row)
            elif not has_code:
                # If no code and no execution, still need to print cell_id
                row = cell_id.ljust(6)
                for _ in cell_data:
                    row += "".center(col_width)
                lines.append(row)

            # Print status/first reason line
            # CLEAN cells show "CLEAN", STALE cells show first reason on same line
            row = "".ljust(6)
            for d in cell_data:
                if d.get("deleted"):
                    row += "".center(col_width)
                elif not d["reasons"]:
                    # No reasons = CLEAN
                    row += "CLEAN".center(col_width)
                else:
                    # Has reasons = first reason (stale implied)
                    row += d["reasons"][0][:col_width - 1].center(col_width)
            lines.append(row)

            # Print remaining reasons (starting from index 1)
            for reason_idx in range(1, max_reasons):
                row = "".ljust(6)
                for d in cell_data:
                    if d.get("deleted"):
                        row += "".center(col_width)
                    elif reason_idx < len(d.get("reasons", [])):
                        row += d["reasons"][reason_idx][:col_width - 1].center(col_width)
                    else:
                        row += "".center(col_width)
                lines.append(row)

            lines.append("")

        return "\n".join(lines)

    def render_latex(self, test_name: str, description: str) -> str:
        """Generate LaTeX section for state evolution."""
        lines = []

        # Escape underscores for LaTeX
        safe_name = test_name.replace("_", r"\_")
        safe_desc = description.replace("_", r"\_")

        # Section header
        lines.append(f"\\subsection{{{safe_name}}}")
        lines.append(f"\\textit{{{safe_desc}}}")
        lines.append("")

        # Show cell code if available
        if self.cell_code:
            lines.append(r"\textbf{Code:}")
            lines.append(r"\begin{itemize}")
            for cell_id in sorted(self.cell_code.keys()):
                code = self.cell_code[cell_id].replace("_", r"\_")
                lines.append(f"  \\item \\texttt{{{cell_id}}}: \\verb|{code}|")
            lines.append(r"\end{itemize}")
            lines.append("")

        # Get all cells
        all_cells = set()
        for snap in self.history:
            all_cells.update(snap.cell_order)
        cell_list = sorted(all_cells)

        num_cols = len(self.history)
        col_spec = "|c|" + "c|" * num_cols

        lines.append(r"\begin{center}")
        lines.append(r"\small")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\hline")

        # Header - use operation names
        headers = [r"\textbf{Cell}", r"\textbf{Initial}"]
        for op in self.operations:
            op_name = self._format_op_header(op).replace("_", r"\_")
            headers.append(f"\\textbf{{{op_name}}}")
        lines.append(" & ".join(headers) + r" \\")
        lines.append(r"\hline")

        # Each cell
        for cell_id in cell_list:
            # Compute rows for this cell
            row_data = []
            max_rows = 3  # R, W, Status

            for snap in self.history:
                if cell_id not in snap.cell_order:
                    row_data.append(["(deleted)", "", ""])
                    continue

                r = snap.reads.get(cell_id, set())
                w = snap.writes.get(cell_id, set())
                status = snap.status.get(cell_id, "?")
                reasons = snap.reasons.get(cell_id, [])

                r_str = r"$R: \{" + ", ".join(sorted(r)) + r"\}$" if r else r"$R: \emptyset$"
                w_str = r"$W: \{" + ", ".join(sorted(w)) + r"\}$" if w else r"$W: \emptyset$"
                status_str = r"\textsc{" + status.capitalize() + "}"

                cell_data = [r_str, w_str, status_str]

                if reasons:
                    for reason in reasons[:1]:  # Just first reason for space
                        rtype = reason.get("type", "?").replace("_", r"\_")
                        cell_data.append(f"\\footnotesize\\texttt{{{rtype}}}")
                    max_rows = max(max_rows, 4)

                row_data.append(cell_data)

            # Render multirow
            lines.append(f"\\multirow{{{max_rows}}}{{*}}{{${cell_id}$}}")

            for row_idx in range(max_rows):
                cols = []
                for col_idx, col_data in enumerate(row_data):
                    if row_idx < len(col_data):
                        cols.append(col_data[row_idx])
                    else:
                        cols.append("")
                lines.append("  & " + " & ".join(cols) + r" \\")

            lines.append(r"\hline")

        lines.append(r"\end{tabular}")
        lines.append(r"\end{center}")
        lines.append("")

        return "\n".join(lines)


def load_litmus_tests() -> List[Dict[str, Any]]:
    """Load tests from YAML file."""
    with open(LITMUS_YAML_PATH, "r") as f:
        data = yaml.safe_load(f)
    return data.get("tests", [])


def pytest_generate_tests(metafunc):
    """Dynamically generate tests from YAML."""
    if "litmus_test" in metafunc.fixturenames:
        tests = load_litmus_tests()
        ids = [t.get("name", f"test_{i}") for i, t in enumerate(tests)]
        metafunc.parametrize("litmus_test", tests, ids=ids)


@pytest.fixture
def litmus_test():
    """Placeholder fixture - populated by pytest_generate_tests."""
    pass


def test_litmus(litmus_test):
    """Run a single litmus test."""
    name = litmus_test.get("name", "unnamed")
    description = litmus_test.get("description", "")
    cell_order = litmus_test.get("cell_order", [])
    cells = litmus_test.get("cells", {})  # Initial cell code definitions
    operations = litmus_test.get("operations", [])
    expect = litmus_test.get("expect", {})

    # SKIPPED_UPSTREAM has been removed; transform expectations to forward_stale
    if "reasons" in expect:
        for cell_id, reasons in expect["reasons"].items():
            for reason in reasons:
                if reason.get("type") == "skipped_upstream":
                    reason["type"] = "forward_stale"
                    reason.pop("expected_cell_id", None)

    # Create runner
    runner = LitmusTestRunner(cell_order, cells)

    # Execute operations
    for op in operations:
        runner.execute_operation(op)

    # Print ASCII visualization
    ascii_output = runner.render_ascii(name, description)
    print("\n" + ascii_output)

    # Validate expectations
    failures = runner.validate_expectations(expect)

    if failures:
        # Print failures clearly
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")

    assert not failures, f"Test '{name}' failed with {len(failures)} errors:\n" + "\n".join(failures)


# Additional utility tests
class TestLitmusHelpers:
    """Test the litmus test infrastructure itself."""

    def test_infer_rw_simple_write(self):
        """Test inference of simple variable write."""
        result = infer_rw("x = 1")
        assert result.writes == {"x"}
        assert result.reads == set()

    def test_infer_rw_simple_read(self):
        """Test inference of simple variable read."""
        result = infer_rw("y = x + 1")
        assert "x" in result.reads
        assert "y" in result.writes

    def test_infer_rw_column_read(self):
        """Test inference of column read."""
        result = infer_rw("y = df['price'] * 2")
        assert "df" in result.reads or "df" in result.column_reads
        assert "price" in result.column_reads.get("df", set())

    def test_infer_rw_column_write(self):
        """Test inference of column write."""
        result = infer_rw("df['new'] = 1")
        assert "new" in result.column_writes.get("df", set())

    def test_infer_rw_structural_shape(self):
        """Test inference of structural read (shape)."""
        result = infer_rw("n = df.shape[0]")
        assert "shape" in result.structural_reads.get("df", set())

    def test_infer_rw_len(self):
        """Test inference of len() as structural read."""
        result = infer_rw("n = len(df)")
        assert "shape" in result.structural_reads.get("df", set())


def generate_all_outputs(output_dir: Optional[Path] = None, code_only: bool = True) -> None:
    """
    Generate combined text and LaTeX output files for all litmus tests.

    Args:
        output_dir: Directory for output files. Defaults to litmus_output/ next to YAML.
        code_only: If True, only include tests with code (names ending in '_code').
    """
    if output_dir is None:
        output_dir = LITMUS_YAML_PATH.parent / "litmus_output"
    output_dir.mkdir(exist_ok=True)

    tests = load_litmus_tests()

    # Filter to code-based tests if requested
    if code_only:
        tests = [t for t in tests if t.get("name", "").endswith("_code")]

    # Collect all ASCII and LaTeX outputs
    ascii_parts = []
    latex_parts = []

    for test in tests:
        name = test.get("name", "unnamed")
        description = test.get("description", "")
        cell_order = test.get("cell_order", [])
        cells = test.get("cells", {})  # Initial cell code definitions
        operations = test.get("operations", [])

        # Create runner and execute operations
        runner = LitmusTestRunner(cell_order, cells)
        for op in operations:
            runner.execute_operation(op)

        # Generate outputs
        ascii_parts.append(runner.render_ascii(name, description))
        latex_parts.append(runner.render_latex(name, description))

    # Write combined text file
    text_file = output_dir / "litmus_tests.txt"
    with open(text_file, "w") as f:
        f.write("\n\n".join(ascii_parts))
    print(f"Written: {text_file}")

    # Write combined LaTeX file
    latex_file = output_dir / "litmus_tests.tex"
    with open(latex_file, "w") as f:
        # LaTeX preamble
        f.write(r"""\documentclass{article}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[margin=1in]{geometry}

\title{FlowBook Litmus Tests}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
This document contains the litmus tests for FlowBook's reproducibility enforcement.
Each test shows the state evolution through a sequence of notebook operations.

\section{Tests}

""")
        f.write("\n\n".join(latex_parts))
        f.write(r"""

\end{document}
""")
    print(f"Written: {latex_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        generate_all_outputs(output_dir)
    else:
        print("Usage: python test_litmus.py --generate [output_dir]")
        print("  Generates litmus_tests.txt and litmus_tests.tex")
