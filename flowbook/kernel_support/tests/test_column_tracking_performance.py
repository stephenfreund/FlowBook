"""
Performance benchmark for column and structural tracking install/uninstall.

Measures the overhead of:
1. Installing/uninstalling column tracking patches
2. Installing/uninstalling structural tracking patches
3. Walking namespaces to find DataFrames
4. Full start_column_tracking/stop_column_tracking cycle
"""

import time
import statistics
import pandas as pd
import numpy as np
from typing import Callable, List, Tuple

from flowbook.kernel_support.column_tracking import ColumnAccessTracker, walk_dataframes, walk_pandas_objects
from flowbook.kernel_support.structural_tracking import StructuralAccessTracker, StructuralTrackingMode
from flowbook.kernel_support.tracking import TrackingDict


def benchmark(func: Callable, iterations: int = 100, warmup: int = 5) -> Tuple[float, float]:
    """
    Run a function multiple times and return mean and std dev of execution times.

    Returns:
        Tuple of (mean_time_us, std_dev_us) in microseconds
    """
    # Warmup
    for _ in range(warmup):
        func()

    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000)  # Convert to microseconds

    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0


def create_test_namespace(n_dataframes: int, n_series: int, rows: int = 100, cols: int = 10) -> dict:
    """Create a test namespace with DataFrames and Series."""
    ns = {}
    for i in range(n_dataframes):
        ns[f'df_{i}'] = pd.DataFrame(
            np.random.randn(rows, cols),
            columns=[f'col_{j}' for j in range(cols)]
        )
    for i in range(n_series):
        ns[f's_{i}'] = pd.Series(np.random.randn(rows), name=f'series_{i}')
    return ns


def test_column_tracking_install_uninstall():
    """Benchmark column tracking install/uninstall cycles."""
    print("\n=== Column Tracking Install/Uninstall ===")

    def install_uninstall():
        tracker = ColumnAccessTracker()
        tracker.install()
        tracker.uninstall()

    mean, std = benchmark(install_uninstall, iterations=100)
    print(f"Install + Uninstall:    {mean:.1f} µs (±{std:.1f} µs)")

    # Measure install alone
    def install_only():
        tracker = ColumnAccessTracker()
        tracker.install()
        # Must uninstall after to reset state
        tracker.uninstall()

    # Breakdown: install vs uninstall
    trackers = []

    def install_step():
        t = ColumnAccessTracker()
        t.install()
        trackers.append(t)

    def uninstall_step():
        if trackers:
            t = trackers.pop()
            t.uninstall()

    # Pre-create trackers for uninstall test
    for _ in range(110):
        t = ColumnAccessTracker()
        t.install()
        trackers.append(t)

    uninstall_mean, uninstall_std = benchmark(uninstall_step, iterations=100)
    print(f"Uninstall only:         {uninstall_mean:.1f} µs (±{uninstall_std:.1f} µs)")

    # Now test install
    install_mean, install_std = benchmark(install_step, iterations=100)
    print(f"Install only:           {install_mean:.1f} µs (±{install_std:.1f} µs)")

    # Cleanup remaining trackers
    while trackers:
        trackers.pop().uninstall()

    return {
        'install_uninstall': mean,
        'install': install_mean,
        'uninstall': uninstall_mean,
    }


def test_structural_tracking_install_uninstall():
    """Benchmark structural tracking install/uninstall cycles."""
    print("\n=== Structural Tracking Install/Uninstall ===")

    def install_uninstall():
        tracker = StructuralAccessTracker(mode=StructuralTrackingMode.WARN)
        tracker.install()
        tracker.uninstall()

    mean, std = benchmark(install_uninstall, iterations=100)
    print(f"Install + Uninstall:    {mean:.1f} µs (±{std:.1f} µs)")

    # Breakdown
    trackers = []

    def uninstall_step():
        if trackers:
            t = trackers.pop()
            t.uninstall()

    # Pre-create trackers for uninstall test
    for _ in range(110):
        t = StructuralAccessTracker(mode=StructuralTrackingMode.WARN)
        t.install()
        trackers.append(t)

    uninstall_mean, uninstall_std = benchmark(uninstall_step, iterations=100)
    print(f"Uninstall only:         {uninstall_mean:.1f} µs (±{uninstall_std:.1f} µs)")

    def install_step():
        t = StructuralAccessTracker(mode=StructuralTrackingMode.WARN)
        t.install()
        trackers.append(t)

    install_mean, install_std = benchmark(install_step, iterations=100)
    print(f"Install only:           {install_mean:.1f} µs (±{install_std:.1f} µs)")

    # Cleanup
    while trackers:
        trackers.pop().uninstall()

    return {
        'install_uninstall': mean,
        'install': install_mean,
        'uninstall': uninstall_mean,
    }


def test_walk_dataframes_overhead():
    """Benchmark the walk_dataframes and walk_pandas_objects functions."""
    print("\n=== Namespace Walking Overhead ===")

    scenarios = [
        ("Empty namespace", 0, 0),
        ("5 DataFrames", 5, 0),
        ("20 DataFrames", 20, 0),
        ("5 DFs + 5 Series", 5, 5),
        ("20 DFs + 20 Series", 20, 20),
    ]

    for name, n_df, n_series in scenarios:
        ns = create_test_namespace(n_df, n_series)

        def walk_df():
            list(walk_dataframes(ns))

        def walk_pandas():
            list(walk_pandas_objects(ns))

        df_mean, df_std = benchmark(walk_df, iterations=200)
        pd_mean, pd_std = benchmark(walk_pandas, iterations=200)

        print(f"{name:25s}: walk_dataframes={df_mean:.1f}µs, walk_pandas_objects={pd_mean:.1f}µs")


def test_full_tracking_cycle():
    """Benchmark the full start_column_tracking/stop_column_tracking cycle."""
    print("\n=== Full Tracking Cycle (start + stop) ===")

    scenarios = [
        ("Empty namespace", 0, 0),
        ("5 DataFrames", 5, 0),
        ("20 DataFrames", 20, 0),
        ("5 DFs + 5 Series", 5, 5),
        ("20 DFs + 20 Series", 20, 20),
    ]

    for name, n_df, n_series in scenarios:
        ns = create_test_namespace(n_df, n_series)
        td = TrackingDict(ns)

        def full_cycle():
            td.start_column_tracking()
            td.stop_column_tracking()

        mean, std = benchmark(full_cycle, iterations=100)
        print(f"{name:25s}: {mean:.1f} µs (±{std:.1f} µs)")


def test_patched_vs_unpatched_operations():
    """Measure overhead of patched DataFrame operations."""
    print("\n=== Patched vs Unpatched DataFrame Operations ===")

    df = pd.DataFrame({
        'a': np.random.randn(1000),
        'b': np.random.randn(1000),
        'c': np.random.randn(1000),
    })

    n_ops = 10000

    # Test __getitem__ without patches
    def unpatched_getitem():
        for _ in range(n_ops):
            _ = df['a']

    unpatched_mean, _ = benchmark(unpatched_getitem, iterations=50)
    unpatched_per_op = unpatched_mean * 1000 / n_ops  # Convert to ns

    # Test __getitem__ with patches installed
    tracker = ColumnAccessTracker()
    tracker.register_df(df, 'df')
    tracker.install()

    def patched_getitem():
        for _ in range(n_ops):
            _ = df['a']

    patched_mean, _ = benchmark(patched_getitem, iterations=50)
    patched_per_op = patched_mean * 1000 / n_ops  # Convert to ns

    tracker.uninstall()

    print(f"Unpatched __getitem__:  {unpatched_per_op:.1f} ns/op")
    print(f"Patched __getitem__:    {patched_per_op:.1f} ns/op")
    print(f"Overhead per operation: {patched_per_op - unpatched_per_op:.1f} ns ({patched_per_op/unpatched_per_op:.2f}x)")


def test_always_on_approach_simulation():
    """
    Simulate the 'always-on patches with global flag' approach.

    This tests what would happen if we kept patches installed permanently
    and just used a flag to enable/disable actual tracking.
    """
    print("\n=== Always-On Approach Simulation ===")

    # Create namespace
    ns = create_test_namespace(10, 5)
    td = TrackingDict(ns)

    # Current approach: install/uninstall each time
    def current_approach():
        td.reset_tracking()
        td.start_column_tracking()
        # Simulate some operations
        for key in list(ns.keys())[:5]:
            if isinstance(ns[key], pd.DataFrame):
                _ = ns[key]['col_0']
        td.stop_column_tracking()

    current_mean, current_std = benchmark(current_approach, iterations=50)

    # Simulated always-on approach: just toggle tracking flag
    # First, install patches once
    col_tracker = ColumnAccessTracker()
    col_tracker.install()
    struct_tracker = StructuralAccessTracker(mode=StructuralTrackingMode.WARN)
    struct_tracker.install()

    # Register DataFrames
    for path, df in walk_dataframes(ns):
        col_tracker.register_df(df, path)
    for path, obj in walk_pandas_objects(ns):
        struct_tracker.register(obj, path)

    def always_on_approach():
        # Just reset tracking state and do operations
        col_tracker.reset()
        struct_tracker.reset()
        # Re-register (still needed to map IDs to paths)
        for path, df in walk_dataframes(ns):
            col_tracker.register_df(df, path)
        for path, obj in walk_pandas_objects(ns):
            struct_tracker.register(obj, path)
        # Simulate some operations
        for key in list(ns.keys())[:5]:
            if isinstance(ns[key], pd.DataFrame):
                _ = ns[key]['col_0']

    always_on_mean, always_on_std = benchmark(always_on_approach, iterations=50)

    # Cleanup
    col_tracker.uninstall()
    struct_tracker.uninstall()

    print(f"Current (install/uninstall): {current_mean:.1f} µs (±{current_std:.1f} µs)")
    print(f"Always-on (flag toggle):     {always_on_mean:.1f} µs (±{always_on_std:.1f} µs)")
    print(f"Savings:                     {current_mean - always_on_mean:.1f} µs ({(1 - always_on_mean/current_mean)*100:.1f}%)")


def count_patched_methods():
    """Count how many methods are being patched."""
    print("\n=== Number of Patched Methods ===")

    from flowbook.kernel_support.structural_tracking import STRUCTURE_USING_METHODS, PANDAS_FUNCTIONS_TO_WRAP

    col_tracker = ColumnAccessTracker()
    col_tracker.install()
    col_patches = len(col_tracker._original_methods)
    col_tracker.uninstall()

    struct_tracker = StructuralAccessTracker(mode=StructuralTrackingMode.WARN)
    struct_tracker.install()
    struct_patches = len(struct_tracker._original_methods)
    struct_tracker.uninstall()

    print(f"Column tracking patches:     {col_patches} methods")
    print(f"Structural tracking patches: {struct_patches} methods")
    print(f"  - Structure-using methods: {len(STRUCTURE_USING_METHODS)}")
    print(f"  - Pandas functions:        {len(PANDAS_FUNCTIONS_TO_WRAP)}")
    print(f"Total patches per cycle:     {col_patches + struct_patches} methods")


if __name__ == '__main__':
    print("=" * 60)
    print("Column/Structural Tracking Performance Benchmark")
    print("=" * 60)

    count_patched_methods()
    test_column_tracking_install_uninstall()
    test_structural_tracking_install_uninstall()
    test_walk_dataframes_overhead()
    test_full_tracking_cycle()
    test_patched_vs_unpatched_operations()
    test_always_on_approach_simulation()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key findings:
1. Install/uninstall cycles are expensive due to many method patches
2. The 'always-on' approach could save significant overhead
3. Main costs: patching methods + walking namespace to register DataFrames
    """)
