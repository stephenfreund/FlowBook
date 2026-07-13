"""Performance test for DataFrame subset detection.

Simulates realistic Kaggle notebook conditions with:
- Large DataFrames (500K+ rows)
- String-based indices (slow for get_indexer)
- Many columns with various dtypes
- Multiple potential subset relationships
"""

import time
import numpy as np
import pandas as pd
import gc
import sys

from flowbook.kernel_support.df_subset_detector import DataFrameSubsetDetector


def create_large_string_indexed_dataframes(n_rows=500000, n_cols=50):
    """Create DataFrames with string indices - worst case for get_indexer."""
    print(f"Creating {n_rows:,} row DataFrames with string indices...")

    # Create column data
    data = {}
    for i in range(n_cols // 2):
        data[f"numeric_{i}"] = np.random.randn(n_rows).astype(np.float64)
        data[f"int_{i}"] = np.random.randint(0, 10000, n_rows, dtype=np.int64)

    # String columns (object dtype)
    for i in range(n_cols // 4):
        data[f"str_{i}"] = [f"value_{j % 1000}" for j in range(n_rows)]

    # Create base DataFrame with STRING index (slow for lookups!)
    train = pd.DataFrame(data)
    train.index = pd.Index([f"row_{i:08d}" for i in range(n_rows)], dtype=str)

    dfs = {"train": train}

    # Create subsets
    dfs["subset_80pct"] = train.iloc[::5 * 4].copy()  # 20% of rows
    dfs["subset_50pct"] = train.iloc[::2].copy()  # 50% of rows
    dfs["subset_30pct"] = train.iloc[::3].copy()  # 33% of rows

    # Create non-subset (different values)
    shuffled = train.sample(frac=0.4, random_state=42)
    shuffled.index = pd.Index([f"other_{i:08d}" for i in range(len(shuffled))], dtype=str)
    dfs["not_subset"] = shuffled

    return dfs


def create_datetime_indexed_dataframes(n_rows=500000, n_cols=50):
    """Create DataFrames with datetime indices."""
    print(f"Creating {n_rows:,} row DataFrames with datetime indices...")

    data = {}
    for i in range(n_cols // 2):
        data[f"numeric_{i}"] = np.random.randn(n_rows).astype(np.float64)
        data[f"int_{i}"] = np.random.randint(0, 10000, n_rows, dtype=np.int64)

    train = pd.DataFrame(data)
    train.index = pd.date_range("2020-01-01", periods=n_rows, freq="s")

    dfs = {"train": train}
    dfs["subset_50pct"] = train.iloc[::2].copy()
    dfs["subset_30pct"] = train.iloc[::3].copy()

    return dfs


def create_integer_indexed_dataframes(n_rows=500000, n_cols=50):
    """Create DataFrames with integer indices (fastest case)."""
    print(f"Creating {n_rows:,} row DataFrames with integer indices...")

    data = {}
    for i in range(n_cols // 2):
        data[f"numeric_{i}"] = np.random.randn(n_rows).astype(np.float64)
        data[f"int_{i}"] = np.random.randint(0, 10000, n_rows, dtype=np.int64)

    train = pd.DataFrame(data)
    # Default RangeIndex is fastest

    dfs = {"train": train}
    dfs["subset_50pct"] = train.iloc[::2].copy()
    dfs["subset_30pct"] = train.iloc[::3].copy()

    return dfs


def benchmark_index_types():
    """Compare detection performance across different index types."""
    print("=" * 70)
    print("Benchmarking different index types")
    print("=" * 70)

    detector = DataFrameSubsetDetector(
        min_rows=100,
        min_savings_bytes=1024,
        max_dataframes=20,
        timeout_ms=120000,  # 2 minutes
    )

    # Test each index type
    for name, creator in [
        ("Integer (RangeIndex)", create_integer_indexed_dataframes),
        ("Datetime", create_datetime_indexed_dataframes),
        ("String", create_large_string_indexed_dataframes),
    ]:
        print(f"\n{name}:")
        dfs = creator(n_rows=200000, n_cols=30)

        for df_name, df in dfs.items():
            mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"  {df_name}: {len(df):,} rows, {mem_mb:.1f}MB, index={type(df.index).__name__}")

        gc.collect()

        start = time.perf_counter()
        result = detector.detect(dfs)
        elapsed = time.perf_counter() - start

        print(f"  Detection: {elapsed*1000:.1f}ms, {len(result.relations)} relations found")


def benchmark_many_dataframes():
    """Test with many DataFrames (like a complex notebook)."""
    print("\n" + "=" * 70)
    print("Benchmarking with many DataFrames")
    print("=" * 70)

    n_rows = 200000
    n_cols = 30

    # Create base data
    data = {}
    for i in range(n_cols // 2):
        data[f"numeric_{i}"] = np.random.randn(n_rows).astype(np.float64)
        data[f"int_{i}"] = np.random.randint(0, 10000, n_rows, dtype=np.int64)

    train = pd.DataFrame(data)

    # Create many filtered/subset DataFrames (like typical EDA)
    dfs = {"train": train}

    # Add many subsets
    for i in range(19):  # Total 20 DataFrames (max_dataframes limit)
        frac = 0.9 - i * 0.04
        if frac > 0.1:
            subset = train.sample(frac=frac, random_state=i).sort_index()
            dfs[f"df_{i}"] = subset

    print(f"Created {len(dfs)} DataFrames")

    detector = DataFrameSubsetDetector(
        min_rows=100,
        min_savings_bytes=1024,
        max_dataframes=20,
        timeout_ms=120000,
    )

    gc.collect()

    start = time.perf_counter()
    result = detector.detect(dfs)
    elapsed = time.perf_counter() - start

    print(f"Detection: {elapsed*1000:.1f}ms")
    print(f"Relations found: {len(result.relations)}")
    print(f"Number of _check_subset calls expected: {len(dfs) * (len(dfs)-1) // 2}")


def benchmark_caching():
    """Test caching performance across repeated detections."""
    print("\n" + "=" * 70)
    print("Benchmarking caching (simulates repeated cell executions)")
    print("=" * 70)

    n_rows = 200000
    n_cols = 30

    # Create base data
    data = {}
    for i in range(n_cols // 2):
        data[f"numeric_{i}"] = np.random.randn(n_rows).astype(np.float64)
        data[f"int_{i}"] = np.random.randint(0, 10000, n_rows, dtype=np.int64)

    train = pd.DataFrame(data)

    # Create subsets
    dfs = {
        "train": train,
        "subset1": train[train["int_0"] > 5000].copy(),
        "subset2": train[train["int_0"] > 7000].copy(),
        "subset3": train.sample(frac=0.6, random_state=42).sort_index(),
    }

    print(f"Created {len(dfs)} DataFrames")

    # Single detector instance (caching enabled)
    detector = DataFrameSubsetDetector(
        min_rows=100,
        min_savings_bytes=1024,
        max_dataframes=20,
        timeout_ms=120000,
    )

    # First run - cache miss
    gc.collect()
    start = time.perf_counter()
    result = detector.detect(dfs)
    first_elapsed = (time.perf_counter() - start) * 1000

    # Subsequent runs - cache hit
    times = []
    for i in range(10):
        start = time.perf_counter()
        result = detector.detect(dfs)
        times.append((time.perf_counter() - start) * 1000)

    avg_cached = sum(times) / len(times)

    print(f"First run (cold cache): {first_elapsed:.2f}ms")
    print(f"Subsequent runs (warm cache): {avg_cached:.2f}ms avg")
    print(f"Speedup: {first_elapsed / avg_cached:.1f}x")
    print(f"Relations found: {len(result.relations)}")


def print_timer_breakdown():
    """Print timing breakdown from flowbook-times.json."""
    try:
        import json
        with open("flowbook-times.json") as f:
            data = json.load(f)

        if not data:
            print("No timer data found")
            return

        # Aggregate times by key
        times = {}
        for entry in data:
            if isinstance(entry, dict) and "key" in entry and "duration" in entry:
                key = entry["key"]
                if key not in times:
                    times[key] = {"count": 0, "total_ms": 0}
                times[key]["count"] += 1
                times[key]["total_ms"] += entry["duration"]  # Already in ms

        print("\n" + "=" * 70)
        print("Timer breakdown (subset-related)")
        print("=" * 70)

        for k, v in sorted(times.items(), key=lambda x: x[1]["total_ms"], reverse=True):
            if "subset" in k.lower():
                avg = v["total_ms"] / v["count"] if v["count"] > 0 else 0
                print(f"  {k}: {v['total_ms']:.2f}ms total, {avg:.3f}ms avg ({v['count']} calls)")

    except Exception as e:
        print(f"Could not print timer breakdown: {e}")


def benchmark_52_cell_simulation():
    """Simulate 52-cell notebook with caching."""
    print("\n" + "=" * 70)
    print("Simulating 52-cell notebook execution WITH caching")
    print("=" * 70)

    n_rows = 200000
    n_cols = 30

    # Create base data
    data = {}
    for i in range(n_cols // 2):
        data[f"numeric_{i}"] = np.random.randn(n_rows).astype(np.float64)
        data[f"int_{i}"] = np.random.randint(0, 10000, n_rows, dtype=np.int64)

    train = pd.DataFrame(data)

    # Create many subsets (simulating EDA operations)
    dfs = {"train": train}
    for i in range(15):
        frac = 0.9 - i * 0.05
        if frac > 0.1:
            subset = train.sample(frac=frac, random_state=i).sort_index()
            dfs[f"df_{i}"] = subset

    print(f"Created {len(dfs)} DataFrames")

    # Single detector with persistent cache
    detector = DataFrameSubsetDetector(
        min_rows=100,
        min_savings_bytes=1024,
        max_dataframes=20,
        timeout_ms=120000,
    )

    gc.collect()

    # Simulate 52 cell executions
    times = []
    for i in range(52):
        start = time.perf_counter()
        result = detector.detect(dfs)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    print(f"First cell: {times[0]:.2f}ms")
    print(f"Average (all 52): {sum(times)/len(times):.2f}ms")
    print(f"Average (cells 2-52): {sum(times[1:])/len(times[1:]):.2f}ms")
    print(f"Total time: {sum(times):.2f}ms")
    print(f"Relations found: {len(result.relations)}")


if __name__ == "__main__":
    # Clear old timer data
    import os
    if os.path.exists("flowbook-times.json"):
        os.remove("flowbook-times.json")

    # Run benchmarks
    benchmark_index_types()
    benchmark_many_dataframes()
    benchmark_caching()
    benchmark_52_cell_simulation()

    # Print timer breakdown
    print_timer_breakdown()
