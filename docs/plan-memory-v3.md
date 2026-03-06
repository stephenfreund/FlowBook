# Plan: v3 Memory Measurement for Plots 4 & 6

**Status:** Implemented
**Date:** 2026-03-06
**Scope:** `compare_baseline.py`, `compare_overhead.py`, new `compare_overhead_v3.py`

## Problem

Plots 4 (Memory Overhead stacked area) and 6 (Checkpoint Overhead Ratio per cell) produce
incorrect results. The current approach measures memory only from the FlowBook run and
derives overhead internally. This conflates FlowBook's own memory impact with the user
program's memory, making the overhead numbers unreliable.

## Solution Overview

1. The baseline run (Phase 3) records namespace size and GPU memory **before** each cell.
2. The FlowBook run (Phase 4) records namespace size, GPU memory, cumulative checkpoint
   size, and enforcer overhead **before** each cell.
3. Cross-run subtraction gives the true overhead at each step.
4. Per-cell checkpoint cost is the marginal change in overhead between consecutive steps.
5. A new `compare_overhead_v3.py` module handles all v3 plotting and statistics, leaving
   v2 code untouched.

### Definitions

```
Base_i    = pre_namespace_mb[i] + pre_gpu_mb[i]          (from baseline run, before step i)
Flow_i    = pre_namespace_mb[i] + pre_gpu_mb[i]
            + pre_checkpoint_cumulative_mb[i]
            + pre_enforcer_overhead_mb[i]                 (from FlowBook run, before step i)

MemoryOverhead_i = Flow_i - Base_i                        (overhead of FlowBook at step i)
Checkpoint_i     = MemoryOverhead_{i+1} - MemoryOverhead_i  (marginal checkpoint cost of step i)
```

Plot 4 uses the FlowBook run's own `namespace_mb`, `gpu_mb`, and `checkpoint_cumulative_mb`
(self-referential; documented below). Plot 6 uses `Checkpoint_i / Base_i`.

---

## 0. Methodology Soundness

### Valid

- **Separate kernel processes** for baseline and FlowBook are the standard A/B comparison.
  Each process has its own heap, so HeapSizer measurements are isolated.
- **Pre-cell measurement** captures the memory state the cell will execute against, which
  is the correct denominator for a per-step ratio.
- **HeapSizer via `user_expressions` with empty code and `silent=True`** does not create
  checkpoints in the FlowBook kernel. Already implemented and verified.
- **Marginal overhead (`Overhead_{i+1} - Overhead_i`)** is valid because enforcer metadata
  is small (~KB) and grows slowly. Checkpoint deepcopy dominates. Any enforcer growth
  is captured in `pre_enforcer_overhead_mb` and cancels out in the subtraction.

### Assumptions to Document

| Assumption | Risk | Mitigation |
|---|---|---|
| Deterministic namespace: both runs produce the same user objects at each step. | Non-deterministic notebooks (random seeds, timestamps, caching) may diverge. | Document in output. Add sanity check: warn if `abs(Base_i - FlowBookNamespace_i) / Base_i > 0.10` for any cell. |
| GPU memory via pynvml reflects actual usage. | Pooled allocators (RMM, PyTorch caching) may not release memory when objects are freed. | Document. GPU overhead comparison is best-effort. |
| HeapSizer traversal is consistent across runs. | Object graph differences (e.g., module caches) could cause small discrepancies. | HeapSizer filters to user namespace (no `_`-prefixed, no callables, no modules). Discrepancies should be < 1 MB. |

### Not Prone to Bad Results

The main failure mode is namespace divergence between runs. The sanity check above catches
this. For deterministic notebooks (the common case), the methodology is sound.

---

## 1. Changes to `compare_baseline.py`

### 1.1 `MemoryCellMetrics` Dataclass

Add four new fields. Existing fields are unchanged for backward compatibility with Plot 4.

```python
@dataclass
class MemoryCellMetrics:
    cell_id: str
    cell_index: int

    # --- Pre-execution (NEW, measured BEFORE the cell runs) ---
    pre_namespace_mb: float              # User namespace before this cell
    pre_gpu_mb: float                    # GPU memory before this cell

    # --- Post-execution (EXISTING, measured AFTER the cell runs) ---
    namespace_mb: float                  # User namespace after this cell
    gpu_mb: float                        # GPU memory after this cell

    # --- Checkpoint overhead (FlowBook only; 0 for baseline) ---
    checkpoint_delta_mb: float           # This cell's checkpoint contribution
    checkpoint_cumulative_mb: float      # Total checkpoint overhead after this cell

    # --- Pre-execution FlowBook overhead (NEW; 0 for baseline) ---
    pre_checkpoint_cumulative_mb: float = 0.0   # Cumulative checkpoints before this cell
    pre_enforcer_overhead_mb: float = 0.0       # Enforcer metadata before this cell

    # --- Per-variable detail (FlowBook only) ---
    checkpoint_by_var: Optional[Dict[str, float]] = None
    checkpoint_var_costs: Optional[Dict[str, Any]] = None

    status: str = "ok"
    error: Optional[str] = None
    is_rerun: bool = False
```

### 1.2 `run_baseline_memory` (Phase 3)

Current code measures namespace only after each cell. Change to measure before AND after.

```
warmup(kernel_client)

for i, cell in enumerate(code_cells):
    pre_ns   = get_namespace_size(kernel_client)         # BEFORE
    pre_gpu  = get_kernel_gpu_memory_mb(kernel_client)   # BEFORE

    execute_cell_baseline(kernel_client, source, timeout)

    post_ns  = get_namespace_size(kernel_client)         # AFTER
    post_gpu = get_kernel_gpu_memory_mb(kernel_client)   # AFTER

    results.cells.append(MemoryCellMetrics(
        pre_namespace_mb      = pre_ns["total_mb"],
        pre_gpu_mb            = pre_gpu,
        namespace_mb          = post_ns["total_mb"],
        gpu_mb                = post_gpu,
        checkpoint_delta_mb   = 0.0,
        checkpoint_cumulative_mb = 0.0,
        # pre_checkpoint_cumulative_mb and pre_enforcer_overhead_mb default to 0
    ))

# Final measurement (used as Overhead_{N} in the cross-run formula)
final_ns  = get_namespace_size(kernel_client)
final_gpu = get_kernel_gpu_memory_mb(kernel_client)
results.totals = {
    "final_namespace_mb": final_ns["total_mb"],
    "final_gpu_mb": final_gpu,
    "max_namespace_mb": max_footprint_mb,
}
```

### 1.3 `run_flowbook_memory` (Phase 4)

Add pre-cell measurements of namespace, GPU, checkpoint cumulative, and enforcer overhead.

```
warmup(kernel_client)

prev_cell_id = None

for i, cell in enumerate(code_cells):
    pre_ns  = get_namespace_size(kernel_client)
    pre_gpu = get_kernel_gpu_memory_mb(kernel_client)

    if prev_cell_id is not None:
        pre_ckpt_overhead = get_checkpoint_overhead(kernel_client, prev_cell_id)
        pre_ckpt_mb = pre_ckpt_overhead.get("total_mb", 0.0)
        breakdown = get_flowbook_overhead_breakdown(kernel_client)
        pre_enforcer_mb = (breakdown.get("execution_records_mb", 0.0)
                           + breakdown.get("tracking_metadata_mb", 0.0)
                           + breakdown.get("other_mb", 0.0))
    else:
        pre_ckpt_mb = 0.0
        pre_enforcer_mb = 0.0

    execute_cell_flowbook(kernel_client, source, cell_id, cell_order, timeout)

    post_ns  = get_namespace_size(kernel_client)
    post_gpu = get_kernel_gpu_memory_mb(kernel_client)
    overhead = get_checkpoint_overhead(kernel_client, cell_id)
    var_costs = get_flowbook_checkpoint_var_costs(kernel_client, cell_id)

    results.cells.append(MemoryCellMetrics(
        pre_namespace_mb             = pre_ns["total_mb"],
        pre_gpu_mb                   = pre_gpu,
        namespace_mb                 = post_ns["total_mb"],
        gpu_mb                       = post_gpu,
        checkpoint_delta_mb          = <computed from overhead as today>,
        checkpoint_cumulative_mb     = overhead.get("total_mb", 0.0),
        pre_checkpoint_cumulative_mb = pre_ckpt_mb,
        pre_enforcer_overhead_mb     = pre_enforcer_mb,
        checkpoint_by_var            = overhead.get("by_variable") or None,
        checkpoint_var_costs         = var_costs or None,
    ))

    prev_cell_id = cell_id

# Final measurement
final_ns  = get_namespace_size(kernel_client)
final_gpu = get_kernel_gpu_memory_mb(kernel_client)
final_overhead = get_checkpoint_overhead(kernel_client, last_cell_id)
final_breakdown = get_flowbook_overhead_breakdown(kernel_client)
results.totals = {
    "final_namespace_mb": final_ns["total_mb"],
    "final_gpu_mb": final_gpu,
    "final_checkpoint_cumulative_mb": final_overhead.get("total_mb", 0.0),
    "final_enforcer_overhead_mb": (final_breakdown.get("execution_records_mb", 0.0)
                                   + final_breakdown.get("tracking_metadata_mb", 0.0)
                                   + final_breakdown.get("other_mb", 0.0)),
    "max_namespace_mb": max_footprint_mb,
    "memory_overhead_ratio": <computed as today>,
}
```

### 1.4 JSON Output Schema (version 3.0)

Bump `ComparisonResult.version` to `"3.0"`. The top-level structure is unchanged.
`MemoryCellMetrics` gains the four new fields listed in 1.1. Baseline cells have the
FlowBook-only fields set to 0.

```json
{
  "version": "3.0",
  "notebook_path": "...",
  "timestamp": "...",
  "scalene_available": true,
  "metadata": { "num_cells": 12, "staleness_mode": "semantic", ... },
  "kernels": {
    "baseline": {
      "kernel_name": "baseline_kernel",
      "timing": { ... },
      "memory": {
        "kernel_name": "baseline_kernel",
        "cells": [
          {
            "cell_id": "abcd",
            "cell_index": 0,
            "pre_namespace_mb": 0.5,
            "pre_gpu_mb": 0.0,
            "namespace_mb": 5.2,
            "gpu_mb": 0.0,
            "checkpoint_delta_mb": 0.0,
            "checkpoint_cumulative_mb": 0.0,
            "pre_checkpoint_cumulative_mb": 0.0,
            "pre_enforcer_overhead_mb": 0.0,
            "checkpoint_by_var": null,
            "checkpoint_var_costs": null,
            "status": "ok",
            "error": null,
            "is_rerun": false
          }
        ],
        "rerun_cells": [],
        "totals": {
          "final_namespace_mb": 10.2,
          "final_gpu_mb": 0.0,
          "max_namespace_mb": 10.2
        }
      }
    },
    "flowbook": {
      "kernel_name": "flowbook_kernel",
      "timing": { ... },
      "memory": {
        "kernel_name": "flowbook_kernel",
        "cells": [
          {
            "cell_id": "abcd",
            "cell_index": 0,
            "pre_namespace_mb": 0.5,
            "pre_gpu_mb": 0.0,
            "namespace_mb": 5.3,
            "gpu_mb": 0.0,
            "checkpoint_delta_mb": 0.3,
            "checkpoint_cumulative_mb": 0.3,
            "pre_checkpoint_cumulative_mb": 0.0,
            "pre_enforcer_overhead_mb": 0.0,
            "checkpoint_by_var": { "df": 0.3 },
            "checkpoint_var_costs": { "df": { "bytes": 314572, "deepcopy_ms": 12.3 } },
            "status": "ok",
            "error": null,
            "is_rerun": false
          }
        ],
        "rerun_cells": [],
        "totals": {
          "final_namespace_mb": 10.4,
          "final_gpu_mb": 0.0,
          "final_checkpoint_cumulative_mb": 2.1,
          "final_enforcer_overhead_mb": 0.05,
          "max_namespace_mb": 10.4,
          "memory_overhead_ratio": 1.21
        }
      }
    }
  }
}
```

---

## 2. Changes to `compare_overhead.py` (Minimal — Version Dispatch Only)

### 2.1 Add `is_v3_format`

```python
def is_v3_format(data: Dict[str, Any]) -> bool:
    version = data.get("_version") or data.get("version", "1.0")
    return str(version).startswith("3")
```

### 2.2 Dispatch in `main()`

After the file-loading loop (around line 2926), before sort/aggregate/output:

```python
# Check if any file is v3 format
has_v3 = any(is_v3_format(d) for d in file_data.values())
has_v2 = any(not is_v3_format(d) for d in file_data.values())

if has_v3 and has_v2:
    print("Error: Cannot mix v2 and v3 comparison files. "
          "Re-run older notebooks with current compare-baseline to produce v3 output.",
          file=sys.stderr)
    sys.exit(1)

if has_v3:
    from flowbook.cli.compare_overhead_v3 import process_v3
    process_v3(file_data, args)
    return

# ... existing v2 code continues unchanged ...
```

### 2.3 No Other Changes to `compare_overhead.py`

All v2 logic (stats, plots, CDFs, histograms) remains untouched. The `is_v3_format`
function and the dispatch block are the only additions.

---

## 3. New File: `flowbook/cli/compare_overhead_v3.py`

### 3.1 Imports from `compare_overhead`

Reuse shared utilities. Do not duplicate them.

```python
from flowbook.cli.compare_overhead import (
    # Data loading
    load_comparison_json, extract_warnings,
    # Trial averaging
    group_trial_files, average_trial_data,
    # Formatting
    FileStats, AggregateStats,
    format_table, format_json_output, format_csv,
    # Checkpoint data extraction (timing vars, memory vars)
    extract_checkpoint_timing_var_data, extract_checkpoint_var_data,
)
```

### 3.2 `compute_file_stats_v3(data, file_path) -> FileStats`

Computes timing stats identically to v2 (timing data is unchanged).

For memory, computes the cross-run metrics:

```python
baseline_mem_cells = baseline.get("memory", {}).get("cells", [])
flowbook_mem_cells = flowbook.get("memory", {}).get("cells", [])
has_baseline_memory = bool(baseline_mem_cells)

if has_baseline_memory:
    # Cross-run overhead sequence
    overhead = []
    for i in range(num_cells):
        bc = baseline_mem_cells[i]
        fc = flowbook_mem_cells[i]
        base_i = bc["pre_namespace_mb"] + bc["pre_gpu_mb"]
        flow_i = (fc["pre_namespace_mb"] + fc["pre_gpu_mb"]
                  + fc["pre_checkpoint_cumulative_mb"]
                  + fc["pre_enforcer_overhead_mb"])
        overhead.append(flow_i - base_i)

    # Final overhead (after last cell)
    bt = baseline.get("memory", {}).get("totals", {})
    ft = flowbook.get("memory", {}).get("totals", {})
    base_final = bt.get("final_namespace_mb", 0) + bt.get("final_gpu_mb", 0)
    flow_final = (ft.get("final_namespace_mb", 0) + ft.get("final_gpu_mb", 0)
                  + ft.get("final_checkpoint_cumulative_mb", 0)
                  + ft.get("final_enforcer_overhead_mb", 0))
    overhead.append(flow_final - base_final)

    # Marginal checkpoint cost per cell
    checkpoint = [overhead[i+1] - overhead[i] for i in range(num_cells)]

    # Per-cell ratio: Checkpoint_i / Base_i
    per_cell_memory_overhead = []
    for i in range(num_cells):
        bc = baseline_mem_cells[i]
        base_i = bc["pre_namespace_mb"] + bc["pre_gpu_mb"]
        if base_i >= 1.0:  # Minimum meaningful base
            per_cell_memory_overhead.append(checkpoint[i] / base_i)
        else:
            per_cell_memory_overhead.append(0.0)
else:
    # Fallback: FlowBook-only (same as v2 logic)
    ...
```

Populates `FileStats` with the same fields as v2, using the new ratios for
`memory_overhead_ratio`, `memory_overhead_pct`, and `per_cell_memory_overhead_mb`.

### 3.3 `plot_combined_v3(data, ...) -> Figure`

6-panel layout identical to v2's `plot_combined_v2`. Panels differ as follows:

| Panel | Content | Change from v2 |
|---|---|---|
| 1 (top-left) | Timing Comparison | No change |
| 2 (top-right) | Checkpoint Time by Variable | No change |
| 3 (mid-left) | Memory Overhead (Plot 3) | **New**: When baseline available, shows baseline namespace as bottom layer and FlowBook overhead (cross-run difference) on top. Falls back to FlowBook self-referential + checkpoint_var_costs when no baseline. |
| 4 (mid-right) | Checkpoint Memory by Variable | No change |
| 5 (bottom-left) | Overhead Time per Cell | No change |
| 6 (bottom-right) | Checkpoint Overhead Ratio (Plot 6) | **New**: uses `Checkpoint_i / Base_i` from cross-run data. Falls back to checkpoint_var_costs deltas when no baseline. |

### 3.4 Panel 3 Detail (Plot 3)

**With baseline memory (cross-run):**

- Layer 1 (gray): `namespace_mb` from baseline run — true user namespace
- Layer 2 (blue): FlowBook overhead = FlowBook total - baseline total (cross-run subtraction)
- Title: "Memory Overhead (Cross-Run)"

**Without baseline memory (fallback):**

- Layer 1 (gray): `namespace_mb` from FlowBook run
- Layer 2 (orange): `gpu_mb` from FlowBook run
- Layer 3 (blue): Checkpoint overhead derived from `checkpoint_var_costs` bytes (logical
  checkpoint size, cumulative max). Uses `checkpoint_cumulative_mb` when non-zero.
- Title: "Memory Overhead (FlowBook Only)"
- Annotation: "Namespace from FlowBook run"

### 3.5 Panel 6 Detail (Plot 6)

**With baseline memory (cross-run):**

Bar chart where bar height at cell i = `Checkpoint_i / Base_i`.

- `Checkpoint_i = MemoryOverhead_{i+1} - MemoryOverhead_i`
- `Base_i = baseline_pre_namespace_mb[i] + baseline_pre_gpu_mb[i]`
- Y-axis label: "Checkpoint / Base Memory"
- Title: "Checkpoint Overhead Ratio (Cross-Run)"

**Without baseline memory (fallback):**

Same as v2: `checkpoint_delta_mb / (prev_namespace_mb + prev_gpu_mb)` from FlowBook data.

- Title: "Checkpoint Overhead Ratio (FlowBook Only)"

### 3.6 `plot_overhead_histograms_v3` and `plot_overhead_cdfs_v3`

Identical structure to v2 versions. The only difference is the data: `per_cell_memory_overhead_mb`
now contains cross-run `Checkpoint_i / Base_i` ratios instead of FlowBook-only ratios.

- Histogram x-axis label: "Checkpoint / Base Memory Ratio"
- CDF x-axis label: "Checkpoint / Base Memory Ratio"

### 3.7 `process_v3(file_data, args)`

Main entry point called from `compare_overhead.main()`. Replicates the flow of the v2
`main()` function after file loading:

```python
def process_v3(file_data: Dict[str, Dict], args) -> None:
    # 1. Compute stats for each file
    stats_list = []
    for file_path, data in file_data.items():
        stats = compute_file_stats_v3(data, file_path)
        stats_list.append(stats)

    # 2. Sort
    sort_key = {
        "slowdown": lambda s: s.slowdown,
        "memory":   lambda s: s.memory_overhead_pct,
        "runtime":  lambda s: s.baseline_runtime_ms,
        "name":     lambda s: s.notebook_name,
    }[args.sort_by]
    stats_list.sort(key=sort_key, reverse=(args.sort_by != "name"))

    # 3. Aggregate
    aggregate = compute_aggregate_stats_v3(stats_list)

    # 4. Format output
    if args.format == "table":
        print(format_table(stats_list, aggregate))
    elif args.format == "json":
        print(format_json_output(stats_list, aggregate))
    elif args.format == "csv":
        print(format_csv(stats_list, aggregate))

    # 5. Generate plots
    if args.plot:
        # Per-notebook combined plots + aggregate histograms + CDFs
        # Same structure as v2 main(), using v3 plot functions
        ...
```

---

## 4. Mode Handling

### 4.1 CLI Flags

| Flag | Description |
|---|---|
| (default) | Runs FlowBook timing, baseline memory, and FlowBook memory |
| `--skip-baseline` | Skip baseline memory run (FlowBook-only mode) |
| `--baseline-timing` | Also run baseline timing phase (skipped by default since it's not used in plots) |
| `--skip-memory` | Skip all memory measurement phases (timing only) |

### 4.2 Phase Execution by Mode

| Flag | Phase 1 (FB Timing) | Phase 2 (Base Timing) | Phase 3 (Base Memory) | Phase 4 (FB Memory) | JSON Version |
|---|---|---|---|---|---|
| (default) | Yes | Skip | Yes | Yes | 3.0 |
| `--skip-baseline` | Yes | Skip | Skip | Yes | 3.0 |
| `--baseline-timing` | Yes | Yes | Yes | Yes | 3.0 |
| `--skip-memory` | Yes | Skip | Skip | Skip | 3.0 |
| `--skip-baseline --skip-memory` | Yes | Skip | Skip | Skip | 3.0 |

### 4.3 Plot Availability by Mode

| Mode | Plot 3 (Panel 3) | Plot 6 (Panel 6) | CDFs / Histograms |
|---|---|---|---|
| Default | Cross-run (baseline ns + overhead) | Cross-run `Checkpoint_i / Base_i` | Cross-run ratios |
| `--skip-baseline` | FlowBook self-referential + checkpoint_var_costs | Fallback from `checkpoint_var_costs` | Fallback ratios |
| `--skip-memory` | "No data" placeholder | "No data" placeholder | "No data" |

### 4.4 Null Handling

The JSON always contains both `"baseline"` and `"flowbook"` keys. When a phase is skipped,
the corresponding `timing` or `memory` field is `null`. This is the same as v2.

`compute_file_stats_v3` checks:
- `baseline_memory is None` → use fallback for Plot 6 and CDF/histogram
- `flowbook_memory is None` → skip all memory plots
- `baseline_timing is None` → derive slowdown from FlowBook code time (same as v2)

### 4.5 Mixed Version Rejection

If `compare_overhead` receives a mix of v2 and v3 files, it prints an error and exits.
Users must re-run older notebooks to produce v3 output. This avoids complex compatibility
logic in the v3 module.

---

## 5. Test Cases

### 5.1 Unit Tests: `compare_baseline.py`

| # | Test | Validates |
|---|---|---|
| 1 | Pre-cell measurement ordering | Mock `get_namespace_size` to return `[1, 5, 10]` for three calls (pre-cell-0, post-cell-0, pre-cell-1...). Verify `pre_namespace_mb` for cell 0 is 1, `namespace_mb` is 5. |
| 2 | First cell pre-checkpoint is zero | Cell 0 must have `pre_checkpoint_cumulative_mb = 0.0` and `pre_enforcer_overhead_mb = 0.0`. |
| 3 | Pre-checkpoint tracks previous cell | Cell i's `pre_checkpoint_cumulative_mb` equals cell (i-1)'s `checkpoint_cumulative_mb`. |
| 4 | Baseline cells have zero FlowBook fields | All of `checkpoint_delta_mb`, `pre_checkpoint_cumulative_mb`, `pre_enforcer_overhead_mb` are 0.0. |
| 5 | JSON v3 roundtrip | Serialize `ComparisonResult`, reload from JSON, verify `version == "3.0"` and all new fields present with correct values. |
| 6 | Totals include new fields | FlowBook totals contain `final_checkpoint_cumulative_mb` and `final_enforcer_overhead_mb`. Baseline totals contain `final_namespace_mb` and `final_gpu_mb`. |

### 5.2 Unit Tests: `compare_overhead.py` (Dispatch)

| # | Test | Validates |
|---|---|---|
| 7 | `is_v3_format` detection | Returns True for `{"version": "3.0"}`, False for `{"version": "2.0"}` and `{"version": "1.0"}`. |
| 8 | Dispatch triggers on v3 | Mock `process_v3`. Load a v3 JSON file. Verify `process_v3` is called. |
| 9 | v2 files run existing code | Load v2 JSON files. Verify `process_v3` is NOT called and existing v2 output is produced. |
| 10 | Mixed v2+v3 rejected | Provide one v2 and one v3 file. Verify error message and `sys.exit(1)`. |

### 5.3 Unit Tests: `compare_overhead_v3.py`

| # | Test | Validates |
|---|---|---|
| 11 | `Checkpoint_i` computation | Given baseline pre-values `[10, 15, 20, 25]` and FlowBook pre-values with overhead `[0, 5, 8, 12]` and final overhead 15, verify `Checkpoint = [5, 3, 4, 3]`. |
| 12 | `Checkpoint_i / Base_i` ratio | From test 11, verify ratios are `[5/10, 3/15, 4/20, 3/25] = [0.5, 0.2, 0.2, 0.12]`. |
| 13 | Small base clamped to zero | When `Base_i < 1.0 MB`, ratio is 0.0 (not division-by-near-zero). |
| 14 | Fallback when no baseline memory | Baseline memory is `null`. Verify Plot 6 uses FlowBook-only `delta / (prev_ns + prev_gpu)`. |
| 15 | Aggregate stats use v3 ratios | `per_cell_memory_overhead_mb` in `FileStats` contains cross-run ratios. `compute_aggregate_stats_v3` produces correct percentiles. |
| 16 | Plot 4 uses FlowBook self-referential data | Verify `namespace_mb`, `gpu_mb`, `checkpoint_cumulative_mb` come from FlowBook memory cells, not baseline. |
| 17 | Plot 6 title changes with mode | With baseline: "Cross-Run". Without: "FlowBook Only". |

### 5.4 Integration Tests

| # | Test | Validates |
|---|---|---|
| 18 | End-to-end with `--run-baseline` | Run a 3-cell test notebook through all 4 phases. Verify JSON output is v3 with all new fields. Generate PDF with 6 panels + histograms + CDFs. No exceptions. |
| 19 | End-to-end FlowBook-only | Run without `--run-baseline`. Verify Plot 4 works, Plot 6 uses fallback, JSON has null baseline memory. |
| 20 | `--skip-memory` mode | Verify no memory phases run, JSON has null memory for both kernels, plots show "No data". |
| 21 | Sanity: `MemoryOverhead_i >= 0` | After warmup, overhead should be non-negative for most cells. Warn (do not fail) if systematically negative—indicates namespace divergence. |
| 22 | Sanity: `Checkpoint_i >= 0` during initial run | Marginal checkpoint cost should be non-negative for the initial top-to-bottom pass. Allow negative during reruns (old checkpoints replaced). |
| 23 | Backward compat: v2 files unchanged | Run `compare_overhead` on existing v2 JSON files. Output must be identical to before this change. |
| 24 | Trial averaging with v3 | Run 2 trials of the same notebook. Verify `average_trial_data` correctly averages the new fields. |

---

## 6. File Change Summary

| File | Type | Changes |
|---|---|---|
| `flowbook/server/commands/compare_baseline.py` | Modify | `MemoryCellMetrics` gains 4 fields. `run_baseline_memory` adds pre-cell measurements. `run_flowbook_memory` adds pre-cell measurements. Version bumped to `"3.0"`. |
| `flowbook/cli/compare_overhead.py` | Modify (minimal) | Add `is_v3_format()`. Add dispatch block in `main()` (~15 lines). No other changes. |
| `flowbook/cli/compare_overhead_v3.py` | **New file** | `compute_file_stats_v3`, `compute_aggregate_stats_v3`, `plot_combined_v3`, `plot_overhead_histograms_v3`, `plot_overhead_cdfs_v3`, `process_v3`. |
| `flowbook/cli/tests/test_compare_overhead_v3.py` | **New file** | Tests 7-17 from Section 5. |
| `flowbook/server/commands/tests/test_compare_baseline_v3.py` | **New file** | Tests 1-6 from Section 5. |

---

## 7. Migration

No migration needed. Old v2 JSON files continue to work with the existing v2 code path.
New runs automatically produce v3 JSON with baseline memory by default (use `--skip-baseline`
to opt out). Users who want v3 plots for old notebooks must re-run `compare-baseline`.
