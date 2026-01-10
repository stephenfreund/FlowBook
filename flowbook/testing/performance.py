"""
Performance testing framework for SDC kernel.

Measures SDC checking overhead in various scenarios:
- Clean: No modifications (best case)
- Modified: Variables randomly modified to trigger checking
"""

import copy
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from flowbook.kernel.checkpoint import Checkpoint, Checkpoints
from flowbook.kernel.models import TrackingData
from flowbook.sdc_kernel.sdc_enforcer import (
    SDCEnforcer,
    PRE_CHECKPOINT_PREFIX,
    POST_CHECKPOINT_PREFIX,
)

from .runner import SDCSimulator, CellRecord
from .notebook_loader import Cell, load_notebook


@dataclass
class PerformanceResult:
    """Result of a single performance test iteration."""

    cell_id: str
    iteration: int
    scenario: str  # 'clean' or 'modified'
    check_time_ms: float
    total_time_ms: float  # Including checkpoint restore
    num_variables_in_namespace: int
    num_variables_checked: int  # Variables that were diffed
    num_modifications: int  # For 'modified' scenario
    modified_variables: List[str]
    reads: List[str]  # Variables read by this cell
    writes: List[str]  # Variables written by this cell
    changed_variables: List[str]  # Variables that actually changed
    stale_cells: List[str]  # Cells marked stale after check
    has_violation: bool
    timestamp: datetime = field(default_factory=datetime.now)


def _randomly_modify_namespace(
    namespace: Dict[str, Any],
    num_modifications: int,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    """
    Randomly modify variables in a namespace.

    Args:
        namespace: The namespace to modify
        num_modifications: Number of variables to modify
        exclude: Variable names to exclude from modification

    Returns:
        List of variable names that were modified
    """
    exclude = exclude or set()

    # Get modifiable variables (exclude private and special names)
    modifiable = [
        name for name in namespace.keys()
        if not name.startswith("_")
        and name not in exclude
        and not callable(namespace[name])
        and not isinstance(namespace[name], type)
    ]

    if not modifiable:
        return []

    # Pick random variables to modify
    num_to_modify = min(num_modifications, len(modifiable))
    to_modify = random.sample(modifiable, num_to_modify)

    for name in to_modify:
        value = namespace[name]
        # Modify based on type
        if isinstance(value, (int, float)):
            namespace[name] = value + random.random() * 100
        elif isinstance(value, str):
            namespace[name] = value + "_modified"
        elif isinstance(value, list):
            namespace[name] = value + ["__modified__"]
        elif isinstance(value, dict):
            namespace[name] = {**value, "__modified__": True}
        elif hasattr(value, "copy"):
            # For pandas DataFrames, Series, etc.
            try:
                new_val = value.copy()
                if hasattr(new_val, "__setitem__"):
                    # Add a marker column or modify first column
                    if hasattr(new_val, "columns"):  # DataFrame
                        new_val["__test_modified__"] = 1
                    elif hasattr(new_val, "iloc"):  # Series
                        if len(new_val) > 0:
                            new_val.iloc[0] = new_val.iloc[0] + 1 if isinstance(new_val.iloc[0], (int, float)) else "__mod__"
                namespace[name] = new_val
            except Exception:
                pass

    return to_modify


def _measure_sdc_check(
    enforcer: SDCEnforcer,
    checkpoints: Checkpoints,
    cell_id: str,
    namespace: Dict[str, Any],
    tracking: TrackingData,
) -> tuple:
    """
    Measure the time for a single SDC check.

    Returns:
        Tuple of (check_time_ms, sdc_result)
    """
    # Create fresh checkpoints for this measurement
    pre_name = f"_perf_pre_{cell_id}"
    post_name = f"_perf_post_{cell_id}"

    # We need to simulate the state as if we just ran the cell
    # Pre-checkpoint is the "before" state, post is "after"
    # For measurement, we use the same namespace for both (no change scenario)
    # or modified namespace (change scenario)

    checkpoints.save(pre_name, namespace, max_size_mb=None)
    pre_checkpoint = checkpoints.saved[pre_name]

    # For post, we just use the same namespace (this simulates re-running)
    checkpoints.save(post_name, namespace, max_size_mb=None)
    post_checkpoint = checkpoints.saved[post_name]

    start = time.perf_counter()
    result = enforcer.check(
        cell_id=cell_id,
        pre_checkpoint=pre_checkpoint,
        post_checkpoint=post_checkpoint,
        tracking=tracking,
        continue_on_violation=True,
        namespace=namespace,
    )
    check_time = (time.perf_counter() - start) * 1000

    # Clean up temporary checkpoints
    checkpoints.delete(pre_name)
    checkpoints.delete(post_name)

    return check_time, result


def run_performance_test(
    simulator: SDCSimulator,
    n_iterations: int = 10,
    seed: Optional[int] = None,
    modifications_per_test: int = 3,
    verbose: bool = False,
) -> List[PerformanceResult]:
    """
    Run performance tests on an already-executed simulator.

    For each iteration:
    1. Pick a random cell
    2. CLEAN scenario: Restore pre-checkpoint, measure check time with no changes
    3. MODIFIED scenario: Copy state, randomly modify variables, measure check time

    Args:
        simulator: SDCSimulator that has already executed a notebook
        n_iterations: Number of test iterations
        seed: Random seed for reproducibility
        modifications_per_test: Number of variables to modify in modified scenario
        verbose: If True, print progress

    Returns:
        List of PerformanceResult objects
    """
    if seed is not None:
        random.seed(seed)

    results: List[PerformanceResult] = []
    cell_ids = simulator.get_cell_ids()

    if not cell_ids:
        raise ValueError("No cells to test - execute a notebook first")

    for i in range(n_iterations):
        # Pick a random cell
        cell_id = random.choice(cell_ids)

        if verbose:
            print(f"  Iteration {i + 1}/{n_iterations}: Testing cell {cell_id}")

        # Get original record
        original_record = simulator.cell_records.get(cell_id)
        if original_record is None:
            continue

        tracking = original_record.tracking

        # ===== CLEAN SCENARIO =====
        # Restore pre-checkpoint and measure check time
        start_total = time.perf_counter()
        simulator.restore_pre_checkpoint(cell_id)
        namespace_copy = simulator.get_current_namespace()

        check_time, sdc_result = _measure_sdc_check(
            simulator.enforcer,
            simulator.checkpoints,
            cell_id,
            namespace_copy,
            tracking,
        )
        total_time = (time.perf_counter() - start_total) * 1000

        clean_result = PerformanceResult(
            cell_id=cell_id,
            iteration=i + 1,
            scenario="clean",
            check_time_ms=check_time,
            total_time_ms=total_time,
            num_variables_in_namespace=len(namespace_copy),
            num_variables_checked=len(tracking.reads_before_writes) + len(tracking.writes),
            num_modifications=0,
            modified_variables=[],
            reads=list(tracking.reads_before_writes),
            writes=list(tracking.writes),
            changed_variables=sdc_result.changed_variables,
            stale_cells=sdc_result.stale_cells,
            has_violation=sdc_result.violation is not None,
        )
        results.append(clean_result)

        if verbose:
            print(f"    [CLEAN] check: {check_time:.3f}ms, total: {total_time:.3f}ms")

        # ===== MODIFIED SCENARIO =====
        # Copy post state and randomly modify variables
        start_total = time.perf_counter()
        simulator.restore_post_checkpoint(cell_id)
        namespace_copy = simulator.get_current_namespace()

        # Modify some variables
        modified_vars = _randomly_modify_namespace(
            namespace_copy,
            modifications_per_test,
            exclude={"__builtins__", "__name__", "__doc__"},
        )

        check_time, sdc_result = _measure_sdc_check(
            simulator.enforcer,
            simulator.checkpoints,
            cell_id,
            namespace_copy,
            tracking,
        )
        total_time = (time.perf_counter() - start_total) * 1000

        modified_result = PerformanceResult(
            cell_id=cell_id,
            iteration=i + 1,
            scenario="modified",
            check_time_ms=check_time,
            total_time_ms=total_time,
            num_variables_in_namespace=len(namespace_copy),
            num_variables_checked=len(tracking.reads_before_writes) + len(tracking.writes),
            num_modifications=len(modified_vars),
            modified_variables=modified_vars,
            reads=list(tracking.reads_before_writes),
            writes=list(tracking.writes),
            changed_variables=sdc_result.changed_variables,
            stale_cells=sdc_result.stale_cells,
            has_violation=sdc_result.violation is not None,
        )
        results.append(modified_result)

        if verbose:
            print(f"    [MODIFIED] check: {check_time:.3f}ms, modified: {modified_vars}")

    return results


def run_performance_test_from_notebook(
    notebook_path: str,
    n_iterations: int = 10,
    seed: Optional[int] = None,
    modifications_per_test: int = 3,
    verbose: bool = False,
) -> tuple:
    """
    Run performance tests on a notebook file.

    This is a convenience function that:
    1. Loads the notebook
    2. Executes it with the simulator
    3. Runs performance tests

    Args:
        notebook_path: Path to .ipynb file
        n_iterations: Number of test iterations
        seed: Random seed for reproducibility
        modifications_per_test: Number of variables to modify per test
        verbose: If True, print progress

    Returns:
        Tuple of (simulator, results)
    """
    if verbose:
        print(f"Loading notebook: {notebook_path}")

    cells = load_notebook(notebook_path)

    if verbose:
        print(f"Found {len(cells)} code cells")
        print("Executing notebook...")

    simulator = SDCSimulator(verbose=verbose)
    simulator.execute_notebook(cells)

    if verbose:
        print(f"\nRunning {n_iterations} performance tests...")

    results = run_performance_test(
        simulator,
        n_iterations=n_iterations,
        seed=seed,
        modifications_per_test=modifications_per_test,
        verbose=verbose,
    )

    return simulator, results


def summarize_performance_results(results: List[PerformanceResult]) -> Dict[str, Any]:
    """
    Compute summary statistics from performance results.

    Args:
        results: List of PerformanceResult objects

    Returns:
        Dict with summary statistics
    """
    clean_results = [r for r in results if r.scenario == "clean"]
    modified_results = [r for r in results if r.scenario == "modified"]

    def stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "median": 0}
        sorted_vals = sorted(values)
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": sorted_vals[len(sorted_vals) // 2],
        }

    return {
        "total_tests": len(results),
        "clean": {
            "count": len(clean_results),
            "check_time_ms": stats([r.check_time_ms for r in clean_results]),
            "total_time_ms": stats([r.total_time_ms for r in clean_results]),
        },
        "modified": {
            "count": len(modified_results),
            "check_time_ms": stats([r.check_time_ms for r in modified_results]),
            "total_time_ms": stats([r.total_time_ms for r in modified_results]),
            "avg_modifications": sum(r.num_modifications for r in modified_results) / len(modified_results) if modified_results else 0,
        },
    }
