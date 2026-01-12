"""
Correctness testing framework for SDC kernel.

Verifies that re-executing cells produces the same state changes as the
original execution.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from flowbook.kernel.checkpoint import Checkpoint
from flowbook.util.output import log, timer

from .runner import SDCSimulator, CellRecord
from .notebook_loader import Cell, load_notebook


@dataclass
class CorrectnessResult:
    """Result of a single correctness test iteration."""

    cell_id: str
    iteration: int
    passed: bool
    expected_changes: List[str]  # Variables that should change
    actual_changes: List[str]  # Variables that did change
    unexpected_diffs: Dict[str, str]  # Variable -> description of difference
    missing_changes: List[str]  # Variables that should have changed but didn't
    extra_changes: List[str]  # Variables that changed but shouldn't have
    execution_time_ms: float
    re_execution_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


def _compare_checkpoints(
    expected_post: Checkpoint,
    actual_post: Checkpoint,
    expected_changes: Set[str],
) -> tuple:
    """
    Compare two post-execution checkpoints.

    Args:
        expected_post: The original post-checkpoint
        actual_post: The post-checkpoint after re-execution
        expected_changes: Variables that were expected to change

    Returns:
        Tuple of (passed, unexpected_diffs, missing_changes, extra_changes)
    """
    # Get the diff between expected and actual post states
    diff_result = Checkpoint.diff(expected_post, actual_post, use_leq=False)

    unexpected_diffs: Dict[str, str] = {}
    if diff_result.differences:
        for var_name, diff_node in diff_result.differences.items():
            # Format the difference for reporting
            unexpected_diffs[var_name] = str(diff_node)

    # If there are any differences, the test failed
    passed = len(unexpected_diffs) == 0

    # Track which variables changed when they shouldn't have (or vice versa)
    # Note: This is a simplified check - in practice, we mainly care about
    # whether the final states match, not whether the same variables changed
    missing_changes: List[str] = []
    extra_changes: List[str] = []

    return passed, unexpected_diffs, missing_changes, extra_changes


def run_correctness_test(
    simulator: SDCSimulator,
    n_iterations: int = 10,
    seed: Optional[int] = None,
) -> List[CorrectnessResult]:
    """
    Run correctness tests on an already-executed simulator.

    For each iteration:
    1. Pick a random cell
    2. Restore its pre-checkpoint
    3. Re-execute the cell
    4. Compare the resulting state to the original post-checkpoint

    Args:
        simulator: SDCSimulator that has already executed a notebook
        n_iterations: Number of test iterations
        seed: Random seed for reproducibility

    Returns:
        List of CorrectnessResult objects
    """
    if seed is not None:
        random.seed(seed)

    results: List[CorrectnessResult] = []
    cell_ids = simulator.get_cell_ids()

    if not cell_ids:
        raise ValueError("No cells to test - execute a notebook first")

    for i in range(n_iterations):
        # Pick a random cell
        cell_id = random.choice(cell_ids)
        cell = simulator.get_cell(cell_id)

        if cell is None:
            continue

        log(f"Iteration {i + 1}/{n_iterations}: Testing cell {cell_id}")

        # Get original record and post-checkpoint
        original_record = simulator.cell_records.get(cell_id)
        if original_record is None:
            continue

        expected_post = simulator.get_post_checkpoint(cell_id)
        expected_changes = set(original_record.sdc_result.changed_variables)

        # Restore pre-checkpoint and re-execute
        try:
            with timer(key="correct:re_execute", message=f"Re-execute {cell_id}") as t:
                simulator.restore_pre_checkpoint(cell_id)
                new_record = simulator.execute_cell(cell)
            re_exec_time = t.duration()

            # Get the actual post state
            actual_post = simulator.get_post_checkpoint(cell_id)
            actual_changes = set(new_record.sdc_result.changed_variables)

            # Compare checkpoints
            with timer(key="correct:compare_checkpoints"):
                passed, unexpected_diffs, missing_changes, extra_changes = _compare_checkpoints(
                    expected_post, actual_post, expected_changes
                )

            # Also check for execution errors
            if new_record.error and not original_record.error:
                passed = False
                unexpected_diffs["__execution_error__"] = new_record.error

            result = CorrectnessResult(
                cell_id=cell_id,
                iteration=i + 1,
                passed=passed,
                expected_changes=list(expected_changes),
                actual_changes=list(actual_changes),
                unexpected_diffs=unexpected_diffs,
                missing_changes=missing_changes,
                extra_changes=extra_changes,
                execution_time_ms=original_record.execution_time_ms,
                re_execution_time_ms=re_exec_time,
            )

        except Exception as e:
            result = CorrectnessResult(
                cell_id=cell_id,
                iteration=i + 1,
                passed=False,
                expected_changes=list(expected_changes),
                actual_changes=[],
                unexpected_diffs={},
                missing_changes=[],
                extra_changes=[],
                execution_time_ms=original_record.execution_time_ms if original_record else 0,
                re_execution_time_ms=0,
                error=str(e),
            )

        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        log(f"  [{status}] {len(result.unexpected_diffs)} diffs")

    return results


def run_correctness_test_from_notebook(
    notebook_path: str,
    n_iterations: int = 10,
    seed: Optional[int] = None,
) -> tuple:
    """
    Run correctness tests on a notebook file.

    This is a convenience function that:
    1. Loads the notebook
    2. Executes it with the simulator
    3. Runs correctness tests

    Args:
        notebook_path: Path to .ipynb file
        n_iterations: Number of test iterations
        seed: Random seed for reproducibility

    Returns:
        Tuple of (simulator, results)
    """
    with timer(key="correct:load_notebook", message=f"Loading notebook {notebook_path}"):
        cells = load_notebook(notebook_path)

    log(f"Found {len(cells)} code cells")

    with timer(key="correct:execute_notebook", message="Executing notebook"):
        simulator = SDCSimulator()
        simulator.execute_notebook(cells)

    log(f"Running {n_iterations} correctness tests")

    with timer(key="correct:run_tests", message=f"Running {n_iterations} correctness tests"):
        results = run_correctness_test(
            simulator,
            n_iterations=n_iterations,
            seed=seed,
        )

    return simulator, results
