# SDC Testing Framework

A standalone testing framework for validating the correctness and measuring the performance of the SDC (Sequential Dataflow Consistency) kernel.

## Overview

This framework provides two types of tests:

1. **Correctness Testing** - Verifies that re-executing cells produces identical state changes
2. **Performance Testing** - Measures SDC checking overhead in various scenarios

Both frameworks use simulated execution (Python's `exec()`) with full SDC infrastructure (checkpoints, read/write tracking, enforcement) without requiring a running Jupyter kernel.

## Command Line Usage

### Correctness Testing

```bash
# Basic usage
python -m flowbook.testing.scripts.run_correctness <notebook.ipynb>

# Full options
python -m flowbook.testing.scripts.run_correctness <notebook.ipynb> \
    -n 20           # Number of iterations (default: 10)
    -s 42           # Random seed for reproducibility
    -o ./results    # Output directory (default: ./test_results)
    -v              # Verbose output
```

### Performance Testing

```bash
# Basic usage
python -m flowbook.testing.scripts.run_performance <notebook.ipynb>

# Full options
python -m flowbook.testing.scripts.run_performance <notebook.ipynb> \
    -n 20           # Number of iterations (default: 10)
    -m 5            # Variables to modify per test (default: 3)
    -s 42           # Random seed for reproducibility
    -o ./results    # Output directory (default: ./test_results)
    -v              # Verbose output
```

### Examples

```bash
# Quick correctness check
python -m flowbook.testing.scripts.run_correctness flowbook/testing/notebooks/multi_dataframe.ipynb

# Thorough performance test with verbose output
python -m flowbook.testing.scripts.run_performance flowbook/testing/notebooks/ml_workflow.ipynb -n 50 -m 5 -s 123 -v

# Test your own notebook
python -m flowbook.testing.scripts.run_correctness examples/Example.ipynb -n 10 -s 42
```

## Test Notebooks

Located in `flowbook/testing/notebooks/`:

| Notebook | Cells | Description | Key Features |
|----------|-------|-------------|--------------|
| `deterministic.ipynb` | 6 | Simple deterministic operations | Scalars, lists, dicts, strings |
| `nondeterministic.ipynb` | 5 | Random operations | Tests seeding fix |
| `dependencies.ipynb` | 8 | Variable dependency chains | Tests dependency tracking |
| `pandas_heavy.ipynb` | 8 | Single DataFrame operations | Filter, aggregate, groupby |
| `multi_dataframe.ipynb` | 10 | E-commerce data model | 3 DataFrames (customers, products, orders), joins, aggregations |
| `data_pipeline.ipynb` | 10 | Sensor data ETL pipeline | Raw data, cleaning, rolling stats, pivots, correlations |
| `ml_workflow.ipynb` | 10 | ML classification pipeline | Feature engineering, encoding, train/test split, sklearn model |

## Output

Results are saved in both JSON (detailed) and CSV (summary) formats:

```
test_results/
├── correctness_<notebook>_<timestamp>.json
├── correctness_<notebook>_<timestamp>.csv
├── performance_<notebook>_<timestamp>.json
└── performance_<notebook>_<timestamp>.csv
```

### JSON Format (Correctness)

```json
{
  "test_type": "correctness",
  "notebook": "path/to/notebook.ipynb",
  "timestamp": "2026-01-06T12:00:00",
  "config": {
    "n_iterations": 10,
    "seed": 42,
    "notebook": "path/to/notebook.ipynb"
  },
  "summary": {
    "total_tests": 10,
    "passed": 10,
    "failed": 0
  },
  "results": [
    {
      "cell_id": "aaaa",
      "iteration": 1,
      "passed": true,
      "expected_changes": ["x", "y"],
      "actual_changes": ["x", "y"],
      "unexpected_diffs": {},
      "execution_time_ms": 1.23,
      "re_execution_time_ms": 1.45
    }
  ]
}
```

### JSON Format (Performance)

```json
{
  "test_type": "performance",
  "notebook": "path/to/notebook.ipynb",
  "config": {
    "n_iterations": 10,
    "seed": 42,
    "modifications_per_test": 3
  },
  "results": [
    {
      "cell_id": "aaaa",
      "iteration": 1,
      "scenario": "clean",
      "check_time_ms": 0.25,
      "total_time_ms": 0.45,
      "num_variables_in_namespace": 10,
      "num_variables_checked": 3,
      "reads": ["x", "y"],
      "writes": ["z"],
      "has_violation": false
    }
  ]
}
```

---

## Design & Implementation

### Architecture

```
flowbook/testing/
├── __init__.py            # Package exports
├── notebook_loader.py     # Parse notebooks, extract cells
├── runner.py              # SDCSimulator - core execution engine
├── correctness.py         # Correctness testing framework
├── performance.py         # Performance testing framework
├── results.py             # JSON/CSV result logging
├── notebooks/             # Test notebooks
│   ├── deterministic.ipynb
│   ├── nondeterministic.ipynb
│   ├── dependencies.ipynb
│   ├── pandas_heavy.ipynb
│   ├── multi_dataframe.ipynb
│   ├── data_pipeline.ipynb
│   └── ml_workflow.ipynb
└── scripts/
    ├── __init__.py
    ├── run_correctness.py # CLI for correctness tests
    └── run_performance.py # CLI for performance tests
```

### Core Components

#### 1. `notebook_loader.py` - Notebook Parser

Loads Jupyter notebooks and extracts code cells:

```python
@dataclass
class Cell:
    cell_id: str      # Unique cell identifier
    source: str       # Cell source code
    cell_type: str    # 'code' or 'markdown'
    index: int        # Position in notebook (used for seeding)

def load_notebook(path: str) -> List[Cell]
```

#### 2. `runner.py` - SDC Simulator

The core execution engine that simulates SDC kernel behavior:

```python
class SDCSimulator:
    def __init__(self, verbose: bool = False):
        self.checkpoints = Checkpoints(...)  # Checkpoint manager
        self.enforcer = SDCEnforcer(...)     # SDC enforcement
        self.namespace = {}                   # Execution namespace
        self.cell_records = {}                # Execution records

    def execute_notebook(self, cells: List[Cell]) -> None:
        """Execute all cells in order with full SDC tracking."""

    def execute_cell(self, cell: Cell) -> CellRecord:
        """Execute single cell with checkpoints and tracking."""

    def restore_pre_checkpoint(self, cell_id: str) -> None:
        """Restore namespace to pre-execution state."""

    def restore_post_checkpoint(self, cell_id: str) -> None:
        """Restore namespace to post-execution state."""
```

**Execution Flow for Each Cell:**

1. Save pre-checkpoint (deep copy of namespace)
2. Set random seeds based on cell index (for determinism)
3. Execute code with `TrackingDict` (captures reads/writes)
4. Save post-checkpoint
5. Run `SDCEnforcer.check()` (backward mutation, staleness)
6. Record timing and results

**Deterministic Execution:**

To ensure reproducibility, random seeds are set before each cell execution:

```python
def _set_random_seeds(self, cell_index: int) -> None:
    random.seed(cell_index)
    np.random.seed(cell_index)
```

This means cells using `random.random()` or `np.random.rand()` will produce identical results on re-execution.

#### 3. `correctness.py` - Correctness Testing

Verifies that re-executing cells produces identical state:

```python
@dataclass
class CorrectnessResult:
    cell_id: str
    iteration: int
    passed: bool
    expected_changes: List[str]   # Variables that should change
    actual_changes: List[str]     # Variables that did change
    unexpected_diffs: Dict[str, str]  # Differences found
    execution_time_ms: float
    re_execution_time_ms: float

def run_correctness_test(
    simulator: SDCSimulator,
    n_iterations: int = 10,
    seed: Optional[int] = None,
) -> List[CorrectnessResult]
```

**Algorithm:**

```
For N iterations:
    1. Pick a random cell
    2. Get the original post-checkpoint (expected state)
    3. Restore the pre-checkpoint
    4. Re-execute the cell
    5. Compare new post-checkpoint to expected
    6. Record pass/fail and any differences
```

**What Constitutes Correctness:**

- The final namespace state after re-execution must match the original post-checkpoint
- Comparison uses `Checkpoint.diff()` which handles:
  - Scalar values (int, float, str, bool)
  - Collections (list, dict, set)
  - NumPy arrays (element-wise)
  - Pandas DataFrames/Series (structure and values)
  - Nested structures

#### 4. `performance.py` - Performance Testing

Measures SDC checking overhead:

```python
@dataclass
class PerformanceResult:
    cell_id: str
    iteration: int
    scenario: str              # 'clean' or 'modified'
    check_time_ms: float       # SDCEnforcer.check() time
    total_time_ms: float       # Including checkpoint restore
    num_variables_in_namespace: int
    num_variables_checked: int
    num_modifications: int     # For 'modified' scenario
    modified_variables: List[str]
    has_violation: bool

def run_performance_test(
    simulator: SDCSimulator,
    n_iterations: int = 10,
    modifications_per_test: int = 3,
) -> List[PerformanceResult]
```

**Two Scenarios:**

1. **CLEAN** - Best case, no modifications:
   - Restore pre-checkpoint
   - Measure `SDCEnforcer.check()` time
   - No variables modified externally

2. **MODIFIED** - Simulated changes:
   - Restore post-checkpoint
   - Randomly modify N variables in namespace
   - Measure `SDCEnforcer.check()` time
   - Tests checking overhead when differences exist

**Variable Modification Strategy:**

```python
def _randomly_modify_namespace(namespace, num_modifications):
    # Modifies based on type:
    # - int/float: add random value
    # - str: append "_modified"
    # - list: append element
    # - dict: add key
    # - DataFrame: add column or modify values
```

#### 5. `results.py` - Result Logging

Handles output in JSON and CSV formats:

```python
class ResultLogger:
    def __init__(self, output_dir: str, test_name: str, test_type: str):
        ...

    def log(self, result: Any) -> None:
        """Log a single test result."""

    def save_json(self) -> str:
        """Save detailed JSON with all results and metadata."""

    def save_csv(self) -> str:
        """Save summary CSV for spreadsheet analysis."""

    def save_all(self) -> Tuple[str, str]:
        """Save both formats, return paths."""
```

### Key Dependencies

The framework uses existing SDC infrastructure:

- `flowbook.kernel.checkpoint.Checkpoints` - State snapshots
- `flowbook.kernel.checkpoint.Checkpoint.diff()` - State comparison
- `flowbook.kernel.tracking.TrackingDict` - Read/write tracking
- `flowbook.sdc_kernel.sdc_enforcer.SDCEnforcer` - SDC enforcement

### Isolation Strategy

The framework is completely isolated:

- All code in separate `flowbook/testing/` directory
- No modifications to existing SDC kernel or checkpoint code
- Only imports from existing modules, no monkey-patching
- Standalone scripts, not integrated into main CLI

### Typical Performance Metrics

Based on test notebooks:

| Notebook Type | Check Time (ms) | Namespace Size |
|---------------|-----------------|----------------|
| Simple scalars | 0.2 - 0.3 | 5 - 10 vars |
| Single DataFrame | 0.7 - 2.0 | 10 - 15 vars |
| Multiple DataFrames | 0.8 - 2.5 | 10 - 15 vars |
| ML Pipeline | 1.0 - 18.0 | 15 - 30 vars |

Factors affecting check time:
- Number of variables in namespace
- Number of variables accessed by cell (reads + writes)
- Complexity of data structures (DataFrames vs scalars)
- Deep alias relationships between variables

---

## Extending the Framework

### Adding New Test Notebooks

1. Create a new `.ipynb` file in `notebooks/`
2. Use 4-character lowercase cell IDs (e.g., "aaaa", "bbbb")
3. Ensure cells are deterministic or use seeded random
4. Run correctness test to verify: `python -m flowbook.testing.scripts.run_correctness notebooks/your_notebook.ipynb`

### Programmatic Usage

```python
from flowbook.testing import (
    load_notebook,
    SDCSimulator,
    run_correctness_test,
    run_performance_test,
    ResultLogger,
)

# Load and execute notebook
cells = load_notebook("path/to/notebook.ipynb")
simulator = SDCSimulator(verbose=True)
simulator.execute_notebook(cells)

# Run tests
correctness_results = run_correctness_test(simulator, n_iterations=20, seed=42)
performance_results = run_performance_test(simulator, n_iterations=20, seed=42)

# Access internals
pre_checkpoint = simulator.get_pre_checkpoint("cell_id")
post_checkpoint = simulator.get_post_checkpoint("cell_id")
record = simulator.cell_records["cell_id"]
```
