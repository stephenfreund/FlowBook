"""
Performance benchmark for TrackingDict vs standard dict.

Measures the overhead of __getitem__ and __setitem__ operations.
"""

import time
import statistics
from typing import Callable, List, Tuple

from flowbook.kernel_support.tracking import TrackingDict


def benchmark(func: Callable, iterations: int = 100) -> Tuple[float, float]:
    """
    Run a function multiple times and return mean and std dev of execution times.

    Returns:
        Tuple of (mean_time_ns, std_dev_ns) per operation
    """
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times.append(end - start)

    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0


def test_getitem_performance():
    """Benchmark __getitem__ performance."""
    n_ops = 100_000

    # Setup: populate both dicts with same data
    regular_dict = {f'var_{i}': i for i in range(1000)}
    tracking_dict = TrackingDict(dict(regular_dict))

    # Keys to access (cycle through existing keys)
    keys = [f'var_{i % 1000}' for i in range(n_ops)]

    def regular_getitem():
        for key in keys:
            _ = regular_dict[key]

    def tracking_getitem():
        tracking_dict.reset_tracking()
        for key in keys:
            _ = tracking_dict[key]

    def tracking_getitem_disabled():
        tracking_dict._tracking_enabled = False
        for key in keys:
            _ = tracking_dict[key]
        tracking_dict._tracking_enabled = True

    # Warmup
    regular_getitem()
    tracking_getitem()
    tracking_getitem_disabled()

    # Benchmark
    reg_mean, reg_std = benchmark(regular_getitem, iterations=50)
    track_mean, track_std = benchmark(tracking_getitem, iterations=50)
    track_disabled_mean, track_disabled_std = benchmark(tracking_getitem_disabled, iterations=50)

    # Per-operation times
    reg_per_op = reg_mean / n_ops
    track_per_op = track_mean / n_ops
    track_disabled_per_op = track_disabled_mean / n_ops

    print("\n=== __getitem__ Performance ===")
    print(f"Operations: {n_ops:,}")
    print(f"Regular dict:           {reg_per_op:.1f} ns/op (±{reg_std/n_ops:.1f} ns)")
    print(f"TrackingDict (enabled): {track_per_op:.1f} ns/op (±{track_std/n_ops:.1f} ns)")
    print(f"TrackingDict (disabled):{track_disabled_per_op:.1f} ns/op (±{track_disabled_std/n_ops:.1f} ns)")
    print(f"Overhead (enabled):     {track_per_op - reg_per_op:.1f} ns/op ({track_per_op/reg_per_op:.2f}x)")
    print(f"Overhead (disabled):    {track_disabled_per_op - reg_per_op:.1f} ns/op ({track_disabled_per_op/reg_per_op:.2f}x)")

    return {
        'regular': reg_per_op,
        'tracking_enabled': track_per_op,
        'tracking_disabled': track_disabled_per_op,
        'overhead_enabled': track_per_op / reg_per_op,
        'overhead_disabled': track_disabled_per_op / reg_per_op,
    }


def test_setitem_performance():
    """Benchmark __setitem__ performance."""
    n_ops = 100_000

    # Keys and values to set
    keys = [f'var_{i}' for i in range(n_ops)]
    values = list(range(n_ops))

    def regular_setitem():
        d = {}
        for key, val in zip(keys, values):
            d[key] = val

    def tracking_setitem():
        real_ns = {}
        td = TrackingDict(real_ns)
        for key, val in zip(keys, values):
            td[key] = val

    def tracking_setitem_disabled():
        real_ns = {}
        td = TrackingDict(real_ns)
        td._tracking_enabled = False
        for key, val in zip(keys, values):
            td[key] = val

    # Warmup
    regular_setitem()
    tracking_setitem()
    tracking_setitem_disabled()

    # Benchmark
    reg_mean, reg_std = benchmark(regular_setitem, iterations=50)
    track_mean, track_std = benchmark(tracking_setitem, iterations=50)
    track_disabled_mean, track_disabled_std = benchmark(tracking_setitem_disabled, iterations=50)

    # Per-operation times
    reg_per_op = reg_mean / n_ops
    track_per_op = track_mean / n_ops
    track_disabled_per_op = track_disabled_mean / n_ops

    print("\n=== __setitem__ Performance ===")
    print(f"Operations: {n_ops:,}")
    print(f"Regular dict:           {reg_per_op:.1f} ns/op (±{reg_std/n_ops:.1f} ns)")
    print(f"TrackingDict (enabled): {track_per_op:.1f} ns/op (±{track_std/n_ops:.1f} ns)")
    print(f"TrackingDict (disabled):{track_disabled_per_op:.1f} ns/op (±{track_disabled_std/n_ops:.1f} ns)")
    print(f"Overhead (enabled):     {track_per_op - reg_per_op:.1f} ns/op ({track_per_op/reg_per_op:.2f}x)")
    print(f"Overhead (disabled):    {track_disabled_per_op - reg_per_op:.1f} ns/op ({track_disabled_per_op/reg_per_op:.2f}x)")

    return {
        'regular': reg_per_op,
        'tracking_enabled': track_per_op,
        'tracking_disabled': track_disabled_per_op,
        'overhead_enabled': track_per_op / reg_per_op,
        'overhead_disabled': track_disabled_per_op / reg_per_op,
    }


def test_mixed_workload():
    """Benchmark a realistic mixed read/write workload."""
    n_ops = 100_000

    # Mix of reads and writes (80% reads, 20% writes - typical workload)
    regular_dict = {f'var_{i}': i for i in range(1000)}
    tracking_dict = TrackingDict(dict(regular_dict))

    # Prepare operations: 80% reads, 20% writes
    import random
    random.seed(42)
    ops = []
    for i in range(n_ops):
        if random.random() < 0.8:
            ops.append(('get', f'var_{random.randint(0, 999)}'))
        else:
            ops.append(('set', f'var_{random.randint(0, 999)}', random.randint(0, 10000)))

    def regular_mixed():
        d = dict(regular_dict)
        for op in ops:
            if op[0] == 'get':
                _ = d[op[1]]
            else:
                d[op[1]] = op[2]

    def tracking_mixed():
        td = TrackingDict(dict(regular_dict))
        for op in ops:
            if op[0] == 'get':
                _ = td[op[1]]
            else:
                td[op[1]] = op[2]

    # Warmup
    regular_mixed()
    tracking_mixed()

    # Benchmark
    reg_mean, reg_std = benchmark(regular_mixed, iterations=50)
    track_mean, track_std = benchmark(tracking_mixed, iterations=50)

    # Per-operation times
    reg_per_op = reg_mean / n_ops
    track_per_op = track_mean / n_ops

    print("\n=== Mixed Workload (80% reads, 20% writes) ===")
    print(f"Operations: {n_ops:,}")
    print(f"Regular dict:           {reg_per_op:.1f} ns/op (±{reg_std/n_ops:.1f} ns)")
    print(f"TrackingDict (enabled): {track_per_op:.1f} ns/op (±{track_std/n_ops:.1f} ns)")
    print(f"Overhead:               {track_per_op - reg_per_op:.1f} ns/op ({track_per_op/reg_per_op:.2f}x)")

    return {
        'regular': reg_per_op,
        'tracking_enabled': track_per_op,
        'overhead': track_per_op / reg_per_op,
    }


def test_attribute_access_overhead():
    """Measure the cost of attribute lookups in TrackingDict.__getitem__."""
    n_ops = 100_000

    td = TrackingDict({f'var_{i}': i for i in range(1000)})
    keys = [f'var_{i % 1000}' for i in range(n_ops)]

    # Direct attribute access (simulating what __getitem__ does internally)
    real_ns = td._real_ns
    tracking_enabled = True
    writes = td._writes
    reads_before_writes = td._reads_before_writes

    def cached_attrs():
        """Access with pre-cached attribute references."""
        for key in keys:
            value = real_ns[key]
            if tracking_enabled and key not in writes:
                reads_before_writes.add(key)

    def normal_attrs():
        """Access through self (normal TrackingDict behavior)."""
        td.reset_tracking()
        for key in keys:
            _ = td[key]

    # Warmup
    cached_attrs()
    normal_attrs()

    # Benchmark
    cached_mean, cached_std = benchmark(cached_attrs, iterations=50)
    normal_mean, normal_std = benchmark(normal_attrs, iterations=50)

    cached_per_op = cached_mean / n_ops
    normal_per_op = normal_mean / n_ops

    print("\n=== Attribute Access Overhead ===")
    print(f"Operations: {n_ops:,}")
    print(f"Cached attributes:      {cached_per_op:.1f} ns/op (±{cached_std/n_ops:.1f} ns)")
    print(f"Normal (via self):      {normal_per_op:.1f} ns/op (±{normal_std/n_ops:.1f} ns)")
    print(f"Attribute lookup cost:  {normal_per_op - cached_per_op:.1f} ns/op")


if __name__ == '__main__':
    print("=" * 60)
    print("TrackingDict Performance Benchmark")
    print("=" * 60)

    getitem_results = test_getitem_performance()
    setitem_results = test_setitem_performance()
    mixed_results = test_mixed_workload()
    test_attribute_access_overhead()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"__getitem__ overhead: {getitem_results['overhead_enabled']:.2f}x slower than regular dict")
    print(f"__setitem__ overhead: {setitem_results['overhead_enabled']:.2f}x slower than regular dict")
    print(f"Mixed workload overhead: {mixed_results['overhead']:.2f}x slower than regular dict")
