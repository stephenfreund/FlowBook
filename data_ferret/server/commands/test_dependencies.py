"""
Test dependencies command implementation.

This module implements the TestDependenciesCommand which validates static
dependency analysis by comparing it against runtime behavior captured through
dynamic tracking.

Purpose:
--------
The command helps verify the accuracy and completeness of static dependency
analysis used throughout DataFerret (for optimization, validation, etc.).
It identifies:
- Soundness violations (static analysis missed dependencies)
- Conservative over-approximations (expected behavior)
- Edge cases where static analysis has limitations

Key Components:
--------------
1. compare_dependencies(): Compares static vs dynamic for a single cell
2. print_comparison_result(): Outputs colored terminal comparison
3. create_comparison_output(): Generates HTML notebook output
4. TestDependenciesCommand: Main command implementation

Static vs Dynamic Analysis:
---------------------------
Static Analysis (AST-based):
- Parses code without executing it
- Over-approximates to be conservative
- Includes all possible code paths
- Warns on dynamic code patterns (eval, getattr, reflection)
- Treats function arguments as potential callbacks

Dynamic Tracking (Runtime):
- Captures actual variable access during execution
- Precise to the specific execution path taken
- Uses TrackingDict to intercept reads/writes
- Only includes what actually executed

Expected Discrepancies:
----------------------
The static analyzer is intentionally conservative, so "static > dynamic"
is expected and acceptable. Common causes:

1. Function arguments treated as callbacks:
   Code: dists = pairwise_distances(pts)
   Static: Adds 'pts' to functions_called (might be callback)
   Dynamic: Only uses 'pts' as data
   Result: Over-approximation with warning (PASS)

2. Conditional code paths:
   Code: if condition: x = 1
   Static: Assumes all paths might execute
   Dynamic: Only one path executes
   Result: Over-approximation (INFO)

3. Method dispatch uncertainty:
   Code: obj.save()
   Static: Includes all methods named 'save'
   Dynamic: Only the actual method called
   Result: Over-approximation (INFO)

Actual Issues:
-------------
"Dynamic > static" indicates the static analyzer missed something:
- CRITICAL: Missed reads (soundness violation)
- WARNING: Missed writes

Usage:
------
Command line:
    data_ferret test_dependencies notebook.ipynb -o output.ipynb

Programmatic:
    from data_ferret.server.commands.test_dependencies import (
        TestDependenciesCommand,
        compare_dependencies
    )
"""

import copy
import sys
import traceback
from typing import Any, Dict, List, Optional, Set

from data_ferret.server.base import NotebookCommand, ProcessingResult
from data_ferret.server.kernel_helper import KernelHelper
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.util.dependencies import analyze_notebook, CellDependencies
from data_ferret.util.ferret_metadata import FerretMetadata, DynamicDependencies
from data_ferret.util.metadata_extractor import extract_and_set_metadata
from data_ferret.util.output import log, timer


# Color scheme for different severity levels
SEVERITY_STYLES = {
    "critical": {
        "border": "#f44336",  # Red
        "bg": "#ffebee",
        "text": "#c62828",
        "symbol": "✗"
    },
    "warning": {
        "border": "#ff9800",  # Orange
        "bg": "#fff3e0",
        "text": "#e65100",
        "symbol": "⚠"
    },
    "info": {
        "border": "#2196f3",  # Blue
        "bg": "#e3f2fd",
        "text": "#1565c0",
        "symbol": "ℹ"
    },
    "pass": {
        "border": "#4caf50",  # Green
        "bg": "#f1f8f4",
        "text": "#2e7d32",
        "symbol": "✓"
    }
}


def compare_dependencies(
    cell_id: str,
    static_deps: CellDependencies,
    dynamic_deps: DynamicDependencies
) -> Dict[str, Any]:
    """
    Compare static vs dynamic dependencies for a single cell.

    This function implements the lenient comparison philosophy where static
    over-approximation (static > dynamic) is expected and acceptable, while
    dynamic finding dependencies that static missed (dynamic > static) indicates
    a soundness violation.

    Args:
        cell_id: The cell identifier
        static_deps: Static analysis results from analyze_notebook()
                    Contains: globals_read, globals_written, warnings, etc.
        dynamic_deps: Dynamic tracking results from kernel execution
                     Contains: reads_before_writes, writes

    Returns:
        Dictionary with comparison results and verdict:
        {
            "cell_id": str,
            "has_mismatch": bool,  # True if any differences found
            "severity": str,  # "critical" | "warning" | "info" | "pass"
            "verdict": str,  # "FAIL" | "WARNING" | "PASS"
            "comparison": {
                "reads": {
                    "static": List[str],  # Variables in static analysis
                    "dynamic": List[str],  # Variables actually read
                    "static_only": List[str],  # Over-approximated
                    "dynamic_only": List[str],  # Missed by static (BAD)
                    "match": bool
                },
                "writes": {...}  # Same structure as reads
            },
            "static_warnings": List[str],  # Warnings from static analysis
            "warnings": List[str],  # Comparison warnings
            "explanation": str  # Human-readable explanation
        }

    Severity Rules:
        - critical: dynamic_only_reads present (soundness violation)
        - warning: dynamic_only_writes present
        - info: static_only present with static warnings (expected)
        - pass: perfect match or acceptable over-approximation
    """
    static_reads = set(static_deps.globals_read)
    static_writes = set(static_deps.globals_written)
    dynamic_reads = set(dynamic_deps.reads_before_writes)
    dynamic_writes = set(dynamic_deps.writes)

    # Find differences
    static_only_reads = static_reads - dynamic_reads
    dynamic_only_reads = dynamic_reads - static_reads
    static_only_writes = static_writes - dynamic_writes
    dynamic_only_writes = dynamic_writes - static_writes

    # Determine severity and verdict
    warnings = []
    severity = "pass"

    # CRITICAL: Dynamic found something static missed
    if dynamic_only_reads:
        warnings.append(
            f"SOUNDNESS VIOLATION: Dynamic reads {sorted(dynamic_only_reads)} "
            f"not detected by static analysis"
        )
        severity = "critical"

    if dynamic_only_writes:
        warnings.append(
            f"Dynamic writes {sorted(dynamic_only_writes)} "
            f"not detected by static analysis"
        )
        if severity != "critical":
            severity = "warning"

    # EXPECTED: Static over-approximated
    explanation_parts = []
    if static_only_reads:
        if static_deps.warnings:
            explanation_parts.append(
                f"Static over-approximated reads {sorted(static_only_reads)} "
                f"(expected due to {len(static_deps.warnings)} static analysis warnings)"
            )
            if severity == "pass":
                severity = "info"
        else:
            explanation_parts.append(
                f"Static over-approximated reads {sorted(static_only_reads)} "
                f"(may be due to conservative method dispatch or conditional writes)"
            )
            if severity == "pass":
                severity = "info"

    if static_only_writes:
        explanation_parts.append(
            f"Static over-approximated writes {sorted(static_only_writes)}"
        )

    # Determine verdict
    if severity == "critical":
        verdict = "FAIL"
    elif severity == "warning":
        verdict = "WARNING"
    else:
        verdict = "PASS"

    explanation = "; ".join(explanation_parts) if explanation_parts else "Perfect match"

    has_mismatch = bool(static_only_reads or dynamic_only_reads or
                        static_only_writes or dynamic_only_writes)

    return {
        "cell_id": cell_id,
        "has_mismatch": has_mismatch,
        "severity": severity,
        "verdict": verdict,
        "comparison": {
            "reads": {
                "static": sorted(static_reads),
                "dynamic": sorted(dynamic_reads),
                "static_only": sorted(static_only_reads),
                "dynamic_only": sorted(dynamic_only_reads),
                "match": len(static_only_reads) == 0 and len(dynamic_only_reads) == 0
            },
            "writes": {
                "static": sorted(static_writes),
                "dynamic": sorted(dynamic_writes),
                "static_only": sorted(static_only_writes),
                "dynamic_only": sorted(dynamic_only_writes),
                "match": len(static_only_writes) == 0 and len(dynamic_only_writes) == 0
            }
        },
        "static_warnings": static_deps.warnings.copy(),
        "warnings": warnings,
        "explanation": explanation
    }


def print_comparison_result(idx: int, comparison: Dict[str, Any]) -> None:
    """
    Print comparison results to terminal with color-coded output.

    Outputs a formatted comparison report showing:
    - Cell index and ID
    - Verdict status (PASS/WARNING/FAIL) with color
    - Static vs dynamic reads comparison
    - Static vs dynamic writes comparison
    - Static analysis warnings (if any)
    - Comparison warnings (if any)
    - Human-readable explanation

    Args:
        idx: Cell index in notebook (0-based)
        comparison: Comparison result dictionary from compare_dependencies()

    Terminal Colors:
        - Green (PASS): Perfect match or acceptable over-approximation
        - Blue (INFO): Expected over-approximation with static warnings
        - Yellow (WARNING): Static missed writes
        - Red (CRITICAL): Soundness violation

    Example Output:
        ======================================================================
        Cell [5] (ID: usak)
        ======================================================================
        Status: PASS

        Static reads:  ['pairwise_distances', 'pts']
        Dynamic reads: ['pairwise_distances', 'pts']
        → Perfect match ✓

        Static writes:  ['_diff', '_end', '_start', 'dists']
        Dynamic writes: ['dists']
        → Static over-approximated: ['_diff', '_end', '_start']

        Explanation: Static over-approximated writes ['_diff', '_end', '_start']
        ======================================================================
    """
    print()
    print("=" * 70)
    print(f"Cell [{idx}] (ID: {comparison['cell_id']})")
    print("=" * 70)

    # Color codes for terminal
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

    severity_colors = {
        "critical": RED,
        "warning": YELLOW,
        "info": BLUE,
        "pass": GREEN
    }

    color = severity_colors.get(comparison['severity'], RESET)
    print(f"Status: {color}{comparison['verdict']}{RESET}")

    # Print reads comparison
    reads = comparison['comparison']['reads']
    print(f"\nStatic reads:  {reads['static']}")
    print(f"Dynamic reads: {reads['dynamic']}")

    if reads['static_only']:
        print(f"{BLUE}→ Static over-approximated: {reads['static_only']}{RESET}")
    if reads['dynamic_only']:
        print(f"{RED}→ Dynamic found extra: {reads['dynamic_only']}{RESET}")
    if reads['match']:
        print(f"{GREEN}→ Perfect match ✓{RESET}")

    # Print writes comparison
    writes = comparison['comparison']['writes']
    print(f"\nStatic writes:  {writes['static']}")
    print(f"Dynamic writes: {writes['dynamic']}")

    if writes['static_only']:
        print(f"{BLUE}→ Static over-approximated: {writes['static_only']}{RESET}")
    if writes['dynamic_only']:
        print(f"{RED}→ Dynamic found extra: {writes['dynamic_only']}{RESET}")
    if writes['match']:
        print(f"{GREEN}→ Perfect match ✓{RESET}")

    # Print static warnings
    if comparison['static_warnings']:
        print(f"\n{YELLOW}Static analysis warnings:{RESET}")
        for warning in comparison['static_warnings']:
            print(f"  - {warning}")

    # Print comparison warnings
    if comparison['warnings']:
        print(f"\n{RED}Comparison warnings:{RESET}")
        for warning in comparison['warnings']:
            print(f"  - {warning}")

    # Print explanation
    print(f"\n{color}Explanation: {comparison['explanation']}{RESET}")
    print("=" * 70)


def create_comparison_output(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a Jupyter notebook output displaying dependency comparison results.

    Generates an HTML display_data output that appears in the notebook cell
    after execution in JupyterLab. The output is styled with color-coded
    borders and backgrounds based on severity.

    Args:
        comparison: Comparison result dictionary from compare_dependencies()

    Returns:
        Jupyter notebook output dictionary:
        {
            "output_type": "display_data",
            "data": {
                "text/html": str,  # Styled HTML with comparison details
                "text/plain": str  # Plain text fallback
            },
            "metadata": {}
        }

    Visual Styling:
        - PASS (green): border=#4caf50, bg=#f1f8f4, symbol=✓
        - INFO (blue): border=#2196f3, bg=#e3f2fd, symbol=ℹ
        - WARNING (orange): border=#ff9800, bg=#fff3e0, symbol=⚠
        - CRITICAL (red): border=#f44336, bg=#ffebee, symbol=✗

    HTML Structure:
        <div> with colored left border and background
          <div> Bold verdict with symbol
          <div> Comparison details
            - Reads: static vs dynamic
            - Writes: static vs dynamic
            - Over-approximations highlighted
            - Missing dependencies highlighted in severity color
            - Explanation in italics

    Example HTML Output:
        For a PASS with over-approximation:
        <div style="border-left: 4px solid #4caf50; padding: 8px; ...">
          <div style="font-weight: bold; color: #2e7d32;">
            ✓ Dependency Test: PASS
          </div>
          <div style="font-size: 12px; margin-top: 4px;">
            <b>Reads:</b> ✓ Match (2 vars)<br>
            <b>Writes:</b> Static: [...], Dynamic: [...]<br>
            → Static over-approximated: ['_diff', '_end', '_start']<br>
            <i>Explanation: Static over-approximated writes [...]</i>
          </div>
        </div>

    This output is appended to cell["outputs"] and appears directly in
    JupyterLab below the cell's execution results.
    """
    severity = comparison.get("severity", "pass")
    styles = SEVERITY_STYLES[severity]

    # Build HTML content
    html_parts = [
        f'<div style="border-left: 4px solid {styles["border"]}; '
        f'padding: 8px; margin: 8px 0; background: {styles["bg"]};">',
        f'<div style="font-weight: bold; color: {styles["text"]};">'
        f'{styles["symbol"]} Dependency Test: {comparison["verdict"]}</div>',
        f'<div style="font-size: 12px; margin-top: 4px;">',
    ]

    # Add reads comparison
    reads = comparison["comparison"]["reads"]
    if reads["match"]:
        html_parts.append(f'<b>Reads:</b> ✓ Match ({len(reads["static"])} vars)<br>')
    else:
        html_parts.append(f'<b>Reads:</b> Static: {reads["static"]}, Dynamic: {reads["dynamic"]}<br>')
        if reads["static_only"]:
            html_parts.append(f'→ Static over-approximated: {reads["static_only"]}<br>')
        if reads["dynamic_only"]:
            html_parts.append(f'→ <span style="color: {styles["text"]};">Dynamic found extra: {reads["dynamic_only"]}</span><br>')

    # Add writes comparison
    writes = comparison["comparison"]["writes"]
    if writes["match"]:
        html_parts.append(f'<b>Writes:</b> ✓ Match ({len(writes["static"])} vars)<br>')
    else:
        html_parts.append(f'<b>Writes:</b> Static: {writes["static"]}, Dynamic: {writes["dynamic"]}<br>')
        if writes["static_only"]:
            html_parts.append(f'→ Static over-approximated: {writes["static_only"]}<br>')
        if writes["dynamic_only"]:
            html_parts.append(f'→ <span style="color: {styles["text"]};">Dynamic found extra: {writes["dynamic_only"]}</span><br>')

    # Add explanation
    if comparison.get("explanation"):
        html_parts.append(f'<i>{comparison["explanation"]}</i>')

    html_parts.extend(['</div>', '</div>'])
    html = ''.join(html_parts)

    # Create plain text fallback
    plain = f'{styles["symbol"]} Dependency Test: {comparison["verdict"]}\n{comparison.get("explanation", "")}'

    return {
        "output_type": "display_data",
        "data": {
            "text/html": html,
            "text/plain": plain
        },
        "metadata": {}
    }


class TestDependenciesCommand(NotebookCommand):
    """
    Tests static dependency analysis against dynamic execution tracking.

    This command validates the accuracy of static dependency analysis by comparing
    it against actual runtime behavior captured via dynamic tracking. It helps
    identify cases where static analysis may be incomplete (soundness violations)
    or overly conservative (expected over-approximations).

    Execution Flow:
    ---------------
    1. Computes static dependencies for all cells using AST analysis
    2. Enables dynamic tracking in the kernel (TrackingDict)
    3. Executes all cells sequentially (like ExecuteAllCommand)
    4. After each cell, compares static vs dynamic dependencies
    5. Reports results to terminal, notebook outputs, and metadata
    6. Provides summary verdict (PASS/FAIL)

    Comparison Philosophy - Lenient Approach:
    -----------------------------------------
    Static analysis intentionally OVER-APPROXIMATES to be conservative:
    - Includes ALL possible dependencies (may include some that don't execute)
    - Accounts for conditional code paths, method dispatch, transitive calls
    - Warns when encountering dynamic code patterns (eval, getattr, etc.)

    Dynamic tracking is PRECISE:
    - Only captures variables actually read/written during execution
    - Reflects the specific code path taken at runtime

    Expected Behaviors (NOT treated as errors):
    ------------------------------------------
    1. Static > Dynamic when static has analysis warnings
       Example: getattr() usage, eval(), reflection APIs
       Reason: Static analysis can't fully analyze dynamic code

    2. Static > Dynamic due to conservative analysis
       Example: Variables passed as function arguments
       Reason: Static analysis treats arguments as potential callbacks

       Code example:
           dists = pairwise_distances(pts)  # pts passed as argument

       Static analysis adds 'pts' to functions_called because it might be
       a callback (like: apply_func(my_callback, data)). This generates
       a warning: "call to 'pts' which is not a function definition - may
       be aliased function, dependencies may be incomplete"

       This is EXPECTED and CORRECT conservative behavior. The test will
       show this as an over-approximation with verdict PASS.

    3. Static > Dynamic for method dispatch
       Reason: Static analysis includes all methods with the same name
       Example: obj.save() might match User.save() AND File.save()

    Actual Issues (Generate warnings):
    ----------------------------------
    1. CRITICAL: Dynamic > Static for reads
       Severity: critical
       Meaning: Soundness violation - static analysis missed dependencies
       Example: Dynamic read variable 'x' but static didn't detect it
       Verdict: FAIL

    2. WARNING: Dynamic > Static for writes
       Severity: warning
       Meaning: Static analysis missed variable writes
       Example: Dynamic wrote 'y' but static didn't detect it
       Verdict: WARNING

    Severity Levels:
    ---------------
    - critical: Soundness violation (dynamic > static for reads)
    - warning:  Static missed writes (dynamic > static for writes)
    - info:     Expected over-approximation (static > dynamic with warnings)
    - pass:     Perfect match or acceptable over-approximation

    Output Formats:
    --------------
    1. Terminal Output (colored):
       - Detailed comparison for each cell
       - Shows static vs dynamic deps side-by-side
       - Color-coded by severity (red/yellow/blue/green)
       - Summary statistics at end

    2. Notebook Cell Outputs (HTML):
       - Visual comparison appended to each executed cell
       - Color-coded div with border (green/blue/yellow/red)
       - Appears directly in JupyterLab notebook UI
       - Includes explanation of mismatches

    3. Metadata (JSON):
       - Per-cell comparison results
       - Summary statistics
       - Overall verdict
       - Returned in ProcessingResult

    Usage:
    ------
    Command line:
        data_ferret test_dependencies notebook.ipynb -o output.ipynb

    The command requires a kernel and will:
    - Execute all code cells
    - Compare dependencies
    - Save results to output notebook
    - Write metadata to metadata.json

    Example Output:
    --------------
    Cell [5] (ID: usak)
    Status: PASS

    Static reads:  ['pairwise_distances', 'pts']
    Dynamic reads: ['pairwise_distances', 'pts']
    → Perfect match ✓

    Static writes:  ['_diff', '_end', '_start', 'dists']
    Dynamic writes: ['dists']
    → Static over-approximated: ['_diff', '_end', '_start']

    Static analysis warnings:
      - call to 'pts' which is not a function definition - may be aliased function

    Explanation: Static over-approximated writes ['_diff', '_end', '_start']

    Summary:
    Total cells executed: 7
    Perfect matches: 4
    Expected over-approximations: 1
    Soundness violations: 0
    Overall verdict: PASS
    """

    @property
    def command_name(self) -> str:
        return "test_dependencies"

    @property
    def display_name(self) -> str:
        return "Test Dependencies"

    @property
    def icon_name(self) -> str:
        return "ui-components:run"

    @property
    def tooltip(self) -> str:
        return "Execute cells and compare static vs dynamic dependency analysis"

    @property
    def requires_kernel(self) -> bool:
        return True

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FerretKernelClient] = None,
        selected_cell_ids: Optional[list] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> ProcessingResult:
        """Execute all code cells and test dependency analysis."""
        if kernel_client is None:
            return ProcessingResult(
                notebook=notebook_content,
                metadata={
                    "status": "error",
                    "command": self.command_name,
                    "error": "Kernel client required but not provided",
                },
                total_cost=0.0,
                total_time=0.0
            )

        with self.timing_context() as get_elapsed:
            # Step 1: Compute static dependencies for the entire notebook
            with timer(key="static_analysis", message="Computing static dependencies"):
                static_dependencies = analyze_notebook(notebook_content)

            new_notebook = copy.deepcopy(notebook_content)
            cells = new_notebook.get("cells", [])

            execution_results = []
            total_executed = 0
            status = "success"

            # Summary counters
            perfect_matches = 0
            expected_over_approximations = 0
            soundness_violations = 0
            cells_with_critical_issues = []

            # Step 2: Enable dynamic tracking BEFORE execution
            with timer(key="enable_tracking", message="Enabling dynamic tracking"):
                kernel_client.execute("%enable_global_tracking", store_history=False)

            # Step 3: Enable Scalene (for consistency with ExecuteAllCommand)
            kernel_client.execute("%enable_scalene", store_history=False)

            with timer(key="execute_and_test", message="Executing cells and testing dependencies"):
                for idx, cell in enumerate(cells):
                    if cell.get("cell_type") == "code":
                        cell_id = cell.get("id")

                        if selected_cell_ids and cell_id not in selected_cell_ids:
                            continue

                        with timer(key="execute_cell", message=f"Executing cell {idx}:{cell_id}"):
                            source = cell.get("source", "")
                            if isinstance(source, list):
                                source = "".join(source)

                            metadata = cell.get("metadata", {}).copy()
                            metadata['cell_id'] = cell_id

                            if source.strip():
                                try:
                                    # Execute the cell
                                    with self.timing_context() as cell_get_elapsed:
                                        result = KernelHelper.execute_code(
                                            kernel_client,
                                            source,
                                            timeout=30 * 60,  # 30 minutes
                                            cell_id=cell_id,
                                            cell_metadata=metadata,
                                        )

                                    cell["execution_count"] = result["execution_count"]
                                    cell["outputs"] = result["outputs"]

                                    # Extract metadata (including dynamic dependencies)
                                    extract_and_set_metadata(cell, result["outputs"])

                                    # Get static and dynamic dependencies
                                    static_deps = static_dependencies.get(cell_id)
                                    ferret_meta = FerretMetadata.from_cell(cell)
                                    dynamic_deps = ferret_meta.get_dynamic_dependencies()

                                    # Compare dependencies
                                    comparison = None
                                    if static_deps and dynamic_deps:
                                        comparison = compare_dependencies(
                                            cell_id, static_deps, dynamic_deps
                                        )

                                        # Print to terminal
                                        print_comparison_result(idx, comparison)

                                        # Append to cell outputs for JupyterLab display
                                        comparison_output = create_comparison_output(comparison)
                                        cell["outputs"].append(comparison_output)

                                        # Update summary counters
                                        if comparison['severity'] == 'critical':
                                            soundness_violations += 1
                                            cells_with_critical_issues.append(cell_id)
                                        elif comparison['severity'] == 'info':
                                            expected_over_approximations += 1
                                        elif comparison['severity'] == 'pass' and not comparison['has_mismatch']:
                                            perfect_matches += 1
                                    elif not static_deps:
                                        print(f"WARNING: No static dependencies for cell {cell_id}")
                                    elif not dynamic_deps:
                                        print(f"WARNING: No dynamic dependencies for cell {cell_id}")

                                    # Handle execution errors
                                    if result["status"] == "error":
                                        status = "error"
                                        error_message = result["error_message"]
                                        print()
                                        print(f"--------------------------------")
                                        print(f"{error_message}")
                                        print(f"--------------------------------")

                                        execution_results.append({
                                            "cell_index": idx,
                                            "cell_id": cell_id,
                                            "status": "error",
                                            "execution_count": result["execution_count"],
                                            "error_message": error_message,
                                            "execution_time": cell_get_elapsed() * 1000,
                                            "dependency_comparison": comparison if comparison else {
                                                "incomplete": True,
                                                "explanation": "Cell execution failed"
                                            }
                                        })
                                        break  # Stop on first error

                                    execution_results.append({
                                        "cell_index": idx,
                                        "cell_id": cell_id,
                                        "status": result["status"],
                                        "execution_count": result["execution_count"],
                                        "execution_time": cell_get_elapsed() * 1000,
                                        "dependency_comparison": comparison
                                    })
                                    log(f"[{result['execution_count']}]")

                                    total_executed += 1

                                except Exception as e:
                                    # Catch any Python exceptions during execution
                                    cell["outputs"] = [{
                                        "output_type": "error",
                                        "ename": e.__class__.__name__,
                                        "evalue": str(e),
                                        "traceback": traceback.format_exception(type(e), e, e.__traceback__),
                                    }]

                                    execution_results.append({
                                        "cell_index": idx,
                                        "cell_id": cell_id,
                                        "status": "error",
                                        "execution_time": cell_get_elapsed() * 1000,
                                    })
                                    status = "error"
                                    break  # Stop on exception

            # Step 4: Disable dynamic tracking AFTER execution
            with timer(key="disable_tracking", message="Disabling dynamic tracking"):
                kernel_client.execute("%disable_global_tracking", store_history=False)

            # Determine overall verdict
            overall_verdict = "FAIL" if soundness_violations > 0 else "PASS"

            # Print summary
            print()
            print("=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"Total cells executed: {total_executed}")
            print(f"Perfect matches: {perfect_matches}")
            print(f"Expected over-approximations: {expected_over_approximations}")
            print(f"Soundness violations: {soundness_violations}")
            if cells_with_critical_issues:
                print(f"Cells with critical issues: {cells_with_critical_issues}")
            print(f"\nOverall verdict: {overall_verdict}")
            print("=" * 70)

            metadata = {
                "status": status,
                "command": self.command_name,
                "execution": {
                    "total_executed": total_executed,
                    "results": execution_results,
                },
                "summary": {
                    "total_cells": total_executed,
                    "perfect_matches": perfect_matches,
                    "expected_over_approximations": expected_over_approximations,
                    "soundness_violations": soundness_violations,
                    "cells_with_critical_issues": cells_with_critical_issues,
                    "overall_verdict": overall_verdict
                }
            }

            total_time = get_elapsed()

        return ProcessingResult(
            notebook=new_notebook,
            metadata=metadata,
            total_cost=0.0,
            total_time=total_time
        )
