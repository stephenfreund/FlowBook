# Per-Variable and Per-Type Timing Breakdown for Compare-Baseline

## Overview

Add per-variable and per-type timing breakdowns to FlowBook's compare-baseline command, mirroring the existing memory breakdown infrastructure. This allows users to see which variables and types consume the most time during checkpoint creation and reproducibility checking.

---

## Requirements

1. **Per-cell timing breakdown** (ALREADY EXISTS): execution time, checkpoint time, check time
2. **Per-variable timing for checkpoint creation** (deepcopy)
3. **Per-type timing for checkpoint creation**
4. **Per-variable timing for diff/checking operations**
5. **Per-type timing for diff/checking operations**
6. **New plots**: Timing breakdown by variable and type (similar to memory plots 3 and 4)

---

## Phase 1: Core Timing Collection (memory_checkpoint.py)

### 1.1 Add Timing Cost Storage

**File**: `flowbook/kernel_support/memory_checkpoint.py`

Add to `MemoryCheckpoints.__init__()` (around line 1818):

```python
# Per-variable timing costs from last deepcopy
self._last_var_timing_costs: dict[str, dict] = {}

# Per-checkpoint timing costs (keyed by checkpoint name)
self._var_timing_costs_by_checkpoint: dict[str, dict[str, dict]] = {}
```

**Data structure per variable:**
```python
{
    'var_name': {
        'deepcopy_ms': float,  # Time to deepcopy this variable
        'type': str,           # Type name (e.g., 'DataFrame')
        'module': str,         # Module (e.g., 'pandas.core.frame')
    }
}
```

### 1.2 Modify _deep_copy_user_ns() to Collect Timing

**Location**: Lines 1865-1950

The per-variable loop already measures timing but only logs it. Store it instead:

```python
var_timing_costs = {}  # NEW

for k, v in variables.items():
    start_time = time.time()
    # ... existing deepcopy code ...
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000

    # NEW: Store per-variable timing
    var_timing_costs[k] = {
        'deepcopy_ms': duration_ms,
        'type': type(v).__name__,
        'module': type(v).__module__,
    }

# Store timing costs for later retrieval
self._last_var_timing_costs = var_timing_costs
```

### 1.3 Store Timing Costs by Checkpoint Name

In `save()` method (around line 2111), after storing memory costs:

```python
# NEW: Store timing costs keyed by checkpoint name
if self._last_var_timing_costs:
    self._var_timing_costs_by_checkpoint[name] = self._last_var_timing_costs.copy()
```

### 1.4 Add API Methods

Add methods following the pattern of `get_cell_checkpoint_costs()`:

```python
def get_cell_checkpoint_timing_costs(self, cell_id: str) -> dict[str, dict]:
    """
    Get combined per-variable timing costs for a cell's checkpoints.
    Combines pre and post checkpoint timing.

    Returns:
        {
            'var_name': {
                'deepcopy_ms': float,   # Combined pre + post
                'type': str,
                'module': str,
                'pre_deepcopy_ms': float,
                'post_deepcopy_ms': float,
            }
        }
    """
```

---

## Phase 2: Compare-Baseline Integration (compare_baseline.py)

### 2.1 Update MemoryCellMetrics

**Location**: Lines 88-100

Add new field:

```python
@dataclass
class MemoryCellMetrics:
    # ... existing fields ...
    checkpoint_var_costs: Optional[Dict[str, Any]] = None  # Existing
    checkpoint_timing_costs: Optional[Dict[str, Any]] = None  # NEW
```

### 2.2 Add Retrieval Function

Add `get_flowbook_checkpoint_timing_costs()` following the pattern of `get_flowbook_checkpoint_var_costs()` (lines 567-641):

```python
def get_flowbook_checkpoint_timing_costs(
    kernel_client, cell_id: str, timeout: float = 30.0
) -> Dict[str, Any]:
    """Get per-variable checkpoint timing costs from FlowBook kernel."""
    expr = (
        f"(__import__('flowbook.kernel_support.memory_checkpoint', fromlist=['MemoryCheckpoints'])"
        f".MemoryCheckpoints._instance.get_cell_checkpoint_timing_costs('{cell_id}') "
        # ... same pattern as memory costs ...
    )
    # ... same implementation pattern ...
```

### 2.3 Update run_flowbook_memory()

**Location**: Lines 1037-1182

After collecting var_costs, also collect timing_costs:

```python
var_costs = get_flowbook_checkpoint_var_costs(kernel_client, cell_id)
timing_costs = get_flowbook_checkpoint_timing_costs(kernel_client, cell_id)  # NEW

# Accumulate cumulative timing (like memory)
if timing_costs:
    cell_timing_ms = sum(v.get('deepcopy_ms', 0) for v in timing_costs.values())
    cumulative_timing_ms += cell_timing_ms

# Build cumulative timing overhead breakdown
timing_overhead_breakdown = {
    'checkpoint_ms': cumulative_timing_ms,
    # ... other timing categories ...
}

results.cells.append(MemoryCellMetrics(
    # ... existing fields ...
    checkpoint_timing_costs=timing_costs if timing_costs else None,  # NEW
))
```

---

## Phase 3: Plot Generation (compare_overhead.py)

### 3.1 Add Extraction Functions

Following the pattern of `extract_checkpoint_type_data_v2()` and `extract_checkpoint_var_data()`:

```python
def extract_checkpoint_timing_type_data(data: Dict[str, Any], top_n: int = 10) -> Optional[Dict[str, Any]]:
    """
    Extract CUMULATIVE checkpoint timing breakdown by type.
    Uses checkpoint_timing_costs field.

    Returns dict with:
        cells: list of cell indices
        by_type: dict mapping type name to list of CUMULATIVE ms per cell
        types_ordered: list ordered by total time descending
        total_ms: list of cumulative total ms per cell
    """

def extract_checkpoint_timing_var_data(data: Dict[str, Any], top_n: int = 10) -> Optional[Dict[str, Any]]:
    """
    Extract CUMULATIVE per-variable checkpoint timing.

    Returns dict with:
        cells: list of cell indices
        by_var: dict mapping var name to list of CUMULATIVE ms per cell
        vars_ordered: list ordered by total time descending
    """
```

### 3.2 Add Timing Panels to plot_combined_v2()

**Location**: Lines 1244-1538

Update to 6-panel layout (2x3) when timing data available:

```
| Panel 1: Timing Comparison     | Panel 2: Memory Overhead Breakdown |
| Panel 3: Checkpoint by Type    | Panel 4: Checkpoint by Variable    |
| Panel 5: Checkpoint Time/Type  | Panel 6: Checkpoint Time/Variable  |
```

Or alternatively, add timing breakdown as a new stacked area in Panel 1 showing the breakdown of state_duration_ms by type/variable.

### 3.3 New Plot Panels

**Panel 5: Checkpoint Time by Type**
- Stacked area chart
- Y-axis: Cumulative time (seconds)
- X-axis: Cell number
- Stacked colors by type (DataFrame, ndarray, list, etc.)
- Shows which types take the most checkpoint time

**Panel 6: Checkpoint Time by Variable**
- Stacked area chart
- Y-axis: Cumulative time (seconds)
- X-axis: Cell number
- Stacked colors by variable name
- Shows which variables take the most checkpoint time

---

## Phase 4: Diff/Check Timing

For per-variable diff timing (the "check" phase), this requires instrumenting diff.py:

1. Track timing per-variable during comparison loop in `diff.py`
2. Store in similar structure: `_var_diff_timing_costs_by_checkpoint`
3. Add `get_cell_diff_timing_costs()` API
4. Add extraction functions and plots for diff timing

This is more complex because diffing involves:
- Variable existence checks
- Type comparison
- Deep value comparison (for DataFrames, column-by-column)

---

## JSON Format Changes

The comparison JSON gains new fields in each cell:

```json
{
  "kernels": {
    "flowbook": {
      "memory": {
        "cells": [
          {
            "cell_id": "abc1",
            "checkpoint_var_costs": { ... },      // existing memory
            "checkpoint_timing_costs": {          // NEW
              "df": {
                "deepcopy_ms": 45.2,
                "type": "DataFrame",
                "module": "pandas.core.frame",
                "pre_deepcopy_ms": 22.1,
                "post_deepcopy_ms": 23.1
              },
              "arr": {
                "deepcopy_ms": 12.5,
                "type": "ndarray",
                "module": "numpy",
                "pre_deepcopy_ms": 6.2,
                "post_deepcopy_ms": 6.3
              }
            },
            "overhead_breakdown": { ... }
          }
        ]
      }
    }
  }
}
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `flowbook/kernel_support/memory_checkpoint.py` | Add timing storage, collection in `_deep_copy_user_ns()`, API methods |
| `flowbook/server/commands/compare_baseline.py` | Add `checkpoint_timing_costs` to MemoryCellMetrics, add retrieval function |
| `flowbook/cli/compare_overhead.py` | Add timing extraction functions, update plot_combined_v2() for 6 panels |

---

## Testing Strategy

### Unit Tests (memory_checkpoint.py)
- Test timing costs recorded during deepcopy
- Test timing costs stored per checkpoint
- Test `get_cell_checkpoint_timing_costs()` combines pre/post correctly

### Integration Tests (compare_baseline.py)
- Test timing costs appear in comparison JSON
- Test timing extraction functions work correctly

### Plot Tests (compare_overhead.py)
- Test timing plots generated when data available
- Test graceful handling when timing data missing (old files)

---

## Implementation Order

1. **Phase 1**: Add timing collection to memory_checkpoint.py
2. **Phase 2**: Add timing retrieval to compare_baseline.py
3. **Phase 3**: Add timing plots to compare_overhead.py
4. **Phase 4**: Testing and validation
5. **Phase 5**: Add diff/check timing breakdown

---

## Design Notes

- **Overhead**: Minimal - just `time.time()` calls around existing deepcopy operations
- **Consistency**: Uses same variable/type as memory for easy comparison
- **Backwards Compatible**: New fields are optional; old comparison files work with new code
- **Units**: Milliseconds (ms) to match existing timing fields
- **Cumulative**: Plots show cumulative timing like memory plots
