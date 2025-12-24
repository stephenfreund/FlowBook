"""
Test Leq command implementation.

This command executes all cells while validating that cells don't modify
variables in their read-before-write set (dependencies they read).

The "leq" (less-than-or-equal) semantics allow:
- New variables to be created by the cell
- DataFrames to have additional columns added
But flag as errors:
- Modifications to variables that were read before being written
"""

import copy
import traceback
from typing import Any, Dict, List, Optional, Set

from data_ferret.server.base import NotebookCommand, ProcessingResult
from data_ferret.server.kernel_helper import KernelHelper
from data_ferret.server.kernel_manager import FerretKernelClient
from data_ferret.kernel.kernel_command_client import KernelCommandClient
from data_ferret.kernel.types import DiffResult, format_diff_as_markdown
from data_ferret.util.ferret_metadata import FerretMetadata
from data_ferret.util.metadata_extractor import extract_and_set_metadata
from data_ferret.util.output import log, timer, error as log_error


# Severity styles for HTML output
SEVERITY_STYLES = {
    "error": {
        "border": "#f44336",  # Red
        "bg": "#ffebee",
        "text": "#c62828",
        "symbol": "✗"
    },
    "pass": {
        "border": "#4caf50",  # Green
        "bg": "#f1f8f4",
        "text": "#2e7d32",
        "symbol": "✓"
    }
}


def format_diff_detail(var_name: str, node: Any, path: str = "") -> List[str]:
    """
    Recursively format a diff node into human-readable lines.

    Args:
        var_name: The variable name
        node: The diff node (ValueComparison or dict)
        path: The current path within the variable

    Returns:
        List of formatted strings describing the differences
    """
    from data_ferret.kernel.types import ValueComparison

    lines = []
    full_path = f"{var_name}{path}" if path else var_name

    if isinstance(node, ValueComparison):
        # Leaf node - actual difference
        lines.append(f"  {full_path}: {node.message}")
    elif isinstance(node, dict):
        # Compound structure - recurse
        for key, child_node in node.items():
            if key == "_truncated":
                if isinstance(child_node, ValueComparison):
                    lines.append(f"  {full_path}: (truncated) {child_node.message}")
                continue
            new_path = f"{path}{key}"
            lines.extend(format_diff_detail(var_name, child_node, new_path))

    return lines


def create_leq_error_output(
    cell_id: str,
    diff: DiffResult,
    rbw_set: Set[str]
) -> Dict[str, Any]:
    """
    Create a Jupyter notebook output displaying leq test failure.

    Args:
        cell_id: The cell identifier
        diff: The DiffResult showing what changed
        rbw_set: The read-before-write set that was tested

    Returns:
        Jupyter notebook output dictionary with HTML display
    """
    styles = SEVERITY_STYLES["error"]

    # Format the differences as markdown for the expandable section
    diff_markdown = format_diff_as_markdown(diff)

    # Build HTML content
    html_parts = [
        f'<div style="border-left: 4px solid {styles["border"]}; '
        f'padding: 8px; margin: 8px 0; background: {styles["bg"]};">',
        f'<div style="font-weight: bold; color: {styles["text"]};">'
        f'{styles["symbol"]} Leq Test: FAILED</div>',
        f'<div style="font-size: 12px; margin-top: 4px;">',
        f'<b>Cell:</b> {cell_id}<br>',
        f'<b>Read-before-write variables tested:</b> {sorted(rbw_set)}<br>',
        '<br>',
        f'<b>Variables modified (should not have changed):</b><br>',
    ]

    # Add detailed information for each modified variable
    for var_name, node in diff.differences.items():
        html_parts.append(f'<div style="margin-left: 12px; margin-top: 4px;">')
        html_parts.append(f'<span style="color: {styles["text"]}; font-weight: bold;">- {var_name}</span><br>')

        # Get detailed diff info for this variable
        detail_lines = format_diff_detail(var_name, node)
        for line in detail_lines[:5]:  # Show first 5 differences inline
            # Escape HTML
            escaped_line = line.replace('<', '&lt;').replace('>', '&gt;')
            html_parts.append(f'<span style="font-family: monospace; font-size: 11px; color: #666;">{escaped_line}</span><br>')

        if len(detail_lines) > 5:
            html_parts.append(f'<span style="font-style: italic; color: #999;">... and {len(detail_lines) - 5} more differences</span><br>')

        html_parts.append('</div>')

    html_parts.append('<br>')
    html_parts.append('<details><summary style="cursor: pointer; color: #666;">Click for full details</summary>')
    # Escape HTML in the markdown
    escaped_markdown = diff_markdown.replace('<', '&lt;').replace('>', '&gt;')
    html_parts.append(f'<pre style="font-size: 11px; background: white; padding: 8px; margin-top: 4px; overflow-x: auto;">{escaped_markdown}</pre>')
    html_parts.append('</details>')

    html_parts.extend(['</div>', '</div>'])
    html = ''.join(html_parts)

    # Create plain text fallback with details
    plain_lines = [f'{styles["symbol"]} Leq Test: FAILED']
    plain_lines.append(f'Modified variables: {list(diff.differences.keys())}')
    for var_name, node in diff.differences.items():
        detail_lines = format_diff_detail(var_name, node)
        plain_lines.extend(detail_lines[:3])  # First 3 details per var
        if len(detail_lines) > 3:
            plain_lines.append(f'  ... and {len(detail_lines) - 3} more')

    plain = '\n'.join(plain_lines)

    return {
        "output_type": "display_data",
        "data": {
            "text/html": html,
            "text/plain": plain
        },
        "metadata": {}
    }


def create_leq_pass_output(cell_id: str, rbw_set: Set[str]) -> Dict[str, Any]:
    """
    Create a Jupyter notebook output displaying leq test success.

    Args:
        cell_id: The cell identifier
        rbw_set: The read-before-write set that was tested

    Returns:
        Jupyter notebook output dictionary with HTML display
    """
    styles = SEVERITY_STYLES["pass"]

    html_parts = [
        f'<div style="border-left: 4px solid {styles["border"]}; '
        f'padding: 8px; margin: 8px 0; background: {styles["bg"]};">',
        f'<div style="font-weight: bold; color: {styles["text"]};">'
        f'{styles["symbol"]} Leq Test: PASSED</div>',
        f'<div style="font-size: 12px; margin-top: 4px;">',
        f'<b>Read-before-write variables verified:</b> {sorted(rbw_set) if rbw_set else "(none)"}<br>',
        '</div>',
        '</div>'
    ]
    html = ''.join(html_parts)

    plain = f'{styles["symbol"]} Leq Test: PASSED'

    return {
        "output_type": "display_data",
        "data": {
            "text/html": html,
            "text/plain": plain
        },
        "metadata": {}
    }


class TestLeqCommand(NotebookCommand):
    """
    Executes all cells while testing that read-before-write variables aren't modified.

    This command validates the correctness of cell execution by ensuring that
    variables a cell reads (but doesn't write) are not modified by that cell.
    This is the "leq" property: post-execution state should be >= pre-execution
    state for the read-before-write variables.

    Execution Flow:
    ---------------
    1. Enables dynamic tracking to capture read-before-write sets
    2. For each code cell:
       a. Saves a checkpoint of the kernel state (pre_{cell_id})
       b. Executes the cell
       c. Extracts the read-before-write set from dynamic tracking
       d. Saves a post-execution checkpoint (post_{cell_id})
       e. Compares pre vs post using leq semantics, restricted to RBW vars
       f. Reports any differences as errors
       g. Deletes temporary checkpoints to save memory
    3. Reports summary of all failures

    Leq Semantics:
    --------------
    - Extra variables in post (new variables) are allowed
    - Extra DataFrame columns in post are allowed
    - Only changes to read-before-write variables are flagged

    Usage:
    ------
    Command line:
        data_ferret test_leq notebook.ipynb -o output.ipynb
    """

    @property
    def command_name(self) -> str:
        return "test_leq"

    @property
    def display_name(self) -> str:
        return "Test Leq"

    @property
    def icon_name(self) -> str:
        return "ui-components:run"

    @property
    def tooltip(self) -> str:
        return "Execute cells and verify read-before-write variables aren't modified"

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
        """Execute all code cells and test leq property."""
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
            new_notebook = copy.deepcopy(notebook_content)
            cells = new_notebook.get("cells", [])

            execution_results = []
            leq_failures: List[Dict[str, Any]] = []
            total_executed = 0
            total_tested = 0
            total_passed = 0
            status = "success"

            # Create kernel command client for checkpoint operations
            kernel_command_client = KernelCommandClient(kernel_client, timeout=60, retries=3)

            # Step 1: Enable dynamic tracking
            with timer(key="enable_tracking", message="Enabling dynamic tracking"):
                kernel_client.execute("%enable_global_tracking", store_history=False)

            # Enable Scalene for profiling
            kernel_client.execute("%enable_scalene", store_history=False)

            with timer(key="execute_and_test_leq", message="Executing cells and testing leq"):
                for idx, cell in enumerate(cells):
                    if cell.get("cell_type") != "code":
                        continue

                    cell_id = cell.get("id")

                    if selected_cell_ids and cell_id not in selected_cell_ids:
                        continue

                    source = cell.get("source", "")
                    if isinstance(source, list):
                        source = "".join(source)

                    if not source.strip():
                        continue

                    with timer(key="test_leq_cell", message=f"Testing leq for cell {idx}:{cell_id}"):
                        metadata = cell.get("metadata", {}).copy()
                        metadata['cell_id'] = cell_id

                        try:
                            # Step 2a: Save pre-execution checkpoint
                            pre_checkpoint_name = f"pre_{cell_id}"
                            with timer(key="save_pre_checkpoint", message=f"Saving pre checkpoint {pre_checkpoint_name}"):
                                kernel_command_client.checkpoint_save(pre_checkpoint_name)

                            # Step 2b: Execute the cell
                            with self.timing_context() as cell_get_elapsed:
                                result = KernelHelper.execute_code(
                                    kernel_client,
                                    source,
                                    self.timeout,
                                    cell_id=cell_id,
                                    cell_metadata=metadata,
                                )

                            cell["execution_count"] = result["execution_count"]
                            cell["outputs"] = result["outputs"]

                            # Step 2c: Extract metadata including dynamic dependencies
                            extract_and_set_metadata(cell, result["outputs"])

                            # Handle execution errors
                            if result["status"] == "error":
                                status = "error"
                                error_message = result["error_message"]
                                print()
                                print(f"--------------------------------")
                                print(f"Cell {cell_id} execution error:")
                                print(f"{error_message}")
                                print(f"--------------------------------")

                                execution_results.append({
                                    "cell_index": idx,
                                    "cell_id": cell_id,
                                    "status": "error",
                                    "execution_count": result["execution_count"],
                                    "error_message": error_message,
                                    "execution_time": cell_get_elapsed() * 1000,
                                })

                                # Cleanup pre checkpoint before breaking
                                try:
                                    kernel_command_client.checkpoint_delete(pre_checkpoint_name)
                                except Exception:
                                    pass

                                break  # Stop on first error

                            # Get read-before-write set from dynamic deps
                            ferret_meta = FerretMetadata.from_cell(cell)
                            dynamic_deps = ferret_meta.get_dynamic_dependencies()
                            rbw_set: Set[str] = set()
                            column_rbw: Dict[str, Set[str]] = {}
                            if dynamic_deps:
                                rbw_set = set(dynamic_deps.reads_before_writes)
                                # Extract column-level RBW for DataFrames
                                if dynamic_deps.column_reads_before_writes:
                                    column_rbw = {
                                        k: set(v) for k, v in dynamic_deps.column_reads_before_writes.items()
                                    }

                            # Step 2d: Save post-execution checkpoint
                            post_checkpoint_name = f"post_{cell_id}"
                            with timer(key="save_post_checkpoint", message=f"Saving post checkpoint {post_checkpoint_name}"):
                                kernel_command_client.checkpoint_save(post_checkpoint_name)

                            # Step 2e: Compare using leq if we have RBW variables
                            leq_passed = True
                            diff_result = None

                            if rbw_set:
                                total_tested += 1
                                with timer(key="compare_leq", message=f"Comparing leq for {cell_id}"):
                                    compare_response = kernel_command_client.checkpoint_compare_leq(
                                        pre_checkpoint_name,
                                        post_checkpoint_name,
                                        keys_to_include=rbw_set,
                                        column_rbw=column_rbw if column_rbw else None
                                    )

                                leq_passed = compare_response.is_leq
                                diff_result = compare_response.diff

                                if leq_passed:
                                    total_passed += 1
                                    # Add pass output to cell
                                    pass_output = create_leq_pass_output(cell_id, rbw_set)
                                    cell["outputs"].append(pass_output)
                                    print(f"Cell {cell_id}: Leq test PASSED (tested {len(rbw_set)} vars)")
                                else:
                                    # Step 2f: Report error
                                    error_output = create_leq_error_output(cell_id, diff_result, rbw_set)
                                    cell["outputs"].append(error_output)

                                    # Collect detailed failure info for metadata
                                    failure_details = {}
                                    for var_name, node in diff_result.differences.items():
                                        detail_lines = format_diff_detail(var_name, node)
                                        failure_details[var_name] = detail_lines

                                    leq_failures.append({
                                        "cell_id": cell_id,
                                        "cell_index": idx,
                                        "rbw_set": sorted(rbw_set),
                                        "modified_vars": list(diff_result.differences.keys()),
                                        "details": failure_details
                                    })

                                    # Print detailed failure info to terminal
                                    print()
                                    print("=" * 60)
                                    print(f"LEQ TEST FAILED: Cell {cell_id}")
                                    print("=" * 60)
                                    print(f"Read-before-write variables tested: {sorted(rbw_set)}")
                                    print(f"Variables that were modified:")
                                    for var_name, node in diff_result.differences.items():
                                        print(f"\n  {var_name}:")
                                        detail_lines = format_diff_detail(var_name, node)
                                        for line in detail_lines[:10]:  # Show up to 10 details
                                            print(f"  {line}")
                                        if len(detail_lines) > 10:
                                            print(f"    ... and {len(detail_lines) - 10} more differences")
                                    print("=" * 60)
                            else:
                                # No RBW vars, nothing to test
                                pass_output = create_leq_pass_output(cell_id, rbw_set)
                                cell["outputs"].append(pass_output)

                            # Step 2g: Cleanup checkpoints
                            with timer(key="cleanup_checkpoints", message=f"Cleaning up checkpoints for {cell_id}"):
                                try:
                                    kernel_command_client.checkpoint_delete(pre_checkpoint_name)
                                except Exception as e:
                                    log_error(f"Failed to delete pre checkpoint: {e}")
                                try:
                                    kernel_command_client.checkpoint_delete(post_checkpoint_name)
                                except Exception as e:
                                    log_error(f"Failed to delete post checkpoint: {e}")

                            execution_results.append({
                                "cell_index": idx,
                                "cell_id": cell_id,
                                "status": result["status"],
                                "execution_count": result["execution_count"],
                                "execution_time": cell_get_elapsed() * 1000,
                                "leq_test": {
                                    "tested": bool(rbw_set),
                                    "passed": leq_passed,
                                    "rbw_vars": sorted(rbw_set) if rbw_set else [],
                                    "modified_vars": list(diff_result.differences.keys()) if diff_result else []
                                }
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
                                "execution_time": 0,
                            })
                            status = "error"
                            break  # Stop on exception

            # Step 3: Disable dynamic tracking
            with timer(key="disable_tracking", message="Disabling dynamic tracking"):
                kernel_client.execute("%disable_global_tracking", store_history=False)

            # Determine overall verdict
            overall_verdict = "PASS" if len(leq_failures) == 0 else "FAIL"

            # Print summary
            print()
            print("=" * 70)
            print("LEQ TEST SUMMARY")
            print("=" * 70)
            print(f"Total cells executed: {total_executed}")
            print(f"Cells with RBW vars tested: {total_tested}")
            print(f"Cells passed: {total_passed}")
            print(f"Cells failed: {len(leq_failures)}")
            if leq_failures:
                print(f"\nFailed cells:")
                for failure in leq_failures:
                    print(f"\n  Cell {failure['cell_id']}:")
                    print(f"    RBW vars tested: {failure['rbw_set']}")
                    print(f"    Modified vars: {failure['modified_vars']}")
                    # Show brief details
                    for var_name, details in failure.get('details', {}).items():
                        if details:
                            print(f"      {var_name}: {details[0].strip()}")
                            if len(details) > 1:
                                print(f"        ... and {len(details) - 1} more differences")
            print(f"\nOverall verdict: {overall_verdict}")
            print("=" * 70)

            result_metadata = {
                "status": status,
                "command": self.command_name,
                "execution": {
                    "total_executed": total_executed,
                    "results": execution_results,
                },
                "leq_test_summary": {
                    "total_tested": total_tested,
                    "total_passed": total_passed,
                    "total_failed": len(leq_failures),
                    "failures": leq_failures,
                    "overall_verdict": overall_verdict
                }
            }

            total_time = get_elapsed()

        return ProcessingResult(
            notebook=new_notebook,
            metadata=result_metadata,
            total_cost=0.0,
            total_time=total_time
        )
