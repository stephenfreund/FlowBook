# Validate Change Command

## Overview

The Validate Change command validates selected cells by comparing their code with the next cell's code, using the current cell's output variables. It uses the kernel's `test_code` comm handler to verify that modifications produce equivalent results.

## Files Created/Modified

### 1. Command Implementation: `data_ferret/server/commands/validate_change.py`

The `ValidateChangeCommand` class:
- Processes **selected cells only** - if no cells are selected, no work is done
- For each selected cell:
  - Gets the cell's source code as `original_code`
  - Gets the next cell's source code as `modified_code`
  - Gets the cell's output variables from dependency analysis (`globals_written`)
  - Sends a `test_code` comm message to the kernel
  - Stores the per-cell result
- Returns results as a map from `cell_id` to result metadata

**Command Properties:**
- **Name:** `validate_change`
- **Display Name:** Validate Change
- **Icon:** `ui-components:check`
- **Requires Kernel:** Yes

**Key Methods:**
- `_send_test_code_comm()`: Sends comm message to kernel and receives response
- `_get_next_cell_source()`: Safely gets the next cell's code
- `_get_cell_output_variables()`: Filters output variables from dependency analysis

### 2. Updated: `data_ferret/server/kernel_manager.py`

Retained the `TestCodeData` dataclass for type safety:
```python
@dataclass
class TestCodeData:
    ok: bool
    result: str
```

The `test_code()` method was removed - its logic now lives in the `ValidateChangeCommand` class.

### 3. Updated: `data_ferret/server/commands/__init__.py`

Added `ValidateChangeCommand` to the exports:
```python
from data_ferret.server.commands.validate_change import ValidateChangeCommand
```

### 4. Frontend Integration

#### `src/types.ts`
Added Validate Change command to `FERRET_COMMANDS`:
```typescript
{
  id: 'validate_change',
  label: 'Validate Change',
  icon: 'ui-components:check',
  tooltip: 'Validate selected cells with next cell comparison',
  requires_kernel: true
}
```

#### `src/toolbar.ts`
Updated to pass selected cell IDs to commands:
- Added `getSelectedCellIds()` method to extract selected cells
- Updated button click handlers to pass selection to manager

#### `src/manager.ts`
Updated `executeCommand()` signature to accept:
- Single `cellId` string (from cell toolbar)
- Array of `cellIds` (from notebook toolbar with selection)
- `undefined` (for commands that don't need selection)

## Usage

### Via JupyterLab UI

1. **Notebook Toolbar**: Click the "Validate Change" button after selecting cells in the notebook
2. **Cell Toolbar**: Click the "Validate Change" button on any individual cell
3. **Command Palette**: Search for "Ferret: Validate Change" and execute

The command only processes selected cells. If no cells are selected, no work is done.

### Via CLI

```bash
python -m data_ferret.server.cli validate_change <notebook.ipynb>
```

Note: CLI version processes all cells. For selected cells, use the JupyterLab UI.

### Via Python API

```python
from data_ferret.server.registry import CommandRegistry
from data_ferret.server.kernel_manager import FerretKernelClient

# Get the command
registry = CommandRegistry()
validate_cmd = registry.get_command('validate_change')

# Execute with kernel client and selected cells
result = await validate_cmd.process(
    notebook_content=notebook,
    kernel_client=kernel_client,
    selected_cell_ids=['cell1', 'cell2']  # Optional - if None, no work is done
)

# Access results
metadata = result['metadata']
print(f"Status: {metadata['status']}")
print(f"Total processed: {metadata['total_processed']}")

# Per-cell results
for cell_id, cell_result in metadata['results'].items():
    print(f"Cell {cell_id}:")
    print(f"  OK: {cell_result['ok']}")
    if cell_result['ok']:
        print(f"  Result: {cell_result['result']}")
    else:
        print(f"  Error: {cell_result['error']}")
```

## How It Works

1. **Selection**: User selects cells in the notebook or clicks a cell toolbar button

2. **Dependency Analysis**: The command analyzes the entire notebook once using `analyze_notebook()` to compute each cell's `globals_written` (output variables)

3. **Cell Processing**: For each selected cell:
   - Extracts the cell's source code as `original_code`
   - Gets the next cell's source code as `modified_code`
   - Filters the cell's output variables (excluding system/private variables)
   - Sends a `test_code` comm message to the kernel

4. **Comm Message**: The `_send_test_code_comm()` method:
   - Generates a unique comm_id
   - Sends a `comm_open` message to the kernel with:
     - `target_name`: "test_code"
     - `data`: Contains `original_code`, `modified_code`, and `output_variables`
   - Waits for the `comm_msg` response from the kernel
   - Returns a `TestCodeData` object with `ok` and `result` fields

5. **Kernel Processing**: The kernel's `FerretKernel` has a registered comm handler (`_test_code_comm_open`) that:
   - Executes the original code
   - Captures output variables
   - Executes the modified code
   - Compares the output variables
   - Sends back a `comm_msg` response with the comparison result

6. **Result Collection**: The command:
   - Collects per-cell results in a dictionary
   - Logs each result (✓ for pass, ✗ for fail/error)
   - Returns results mapped by cell_id in the metadata

## Example Output

When executed with 2 selected cells, the command logs:

```
[Analyzing notebook dependencies... 5 ms]
[Validating 2 selected cell(s)...]
[Validating cells...]
  [Validating cell 0:cell1...]
    [✓] Cell 0: Variables match: x, y
  [12 ms]
  [Validating cell 2:cell3...]
    [✗] Cell 2: Variables differ: b (expected 10, got 20)
  [10 ms]
[25 ms]
[Completed: 2 cell(s) validated]
```

And returns:

```json
{
  "notebook": <original notebook>,
  "metadata": {
    "status": "success",
    "command": "validate_change",
    "results": {
      "cell1": {
        "ok": true,
        "result": "Variables match: x, y",
        "error": null
      },
      "cell3": {
        "ok": false,
        "result": null,
        "error": "Variables differ: b (expected 10, got 20)"
      }
    },
    "total_processed": 2
  }
}
```

## Testing

Run the test script to verify the command works correctly:

```bash
python test_validate_change_command.py
```

The test script verifies:
- Command registration and properties
- Handling of no selection / empty selection
- Error handling when kernel client is missing
- Per-cell result structure
- Next cell extraction logic
- Output variable filtering
- Integration with dependency analysis

## Key Features

- **Selection-based processing**: Only processes selected cells, no work if nothing is selected
- **Per-cell results**: Returns a map from cell_id to result, making it easy to track which cells passed/failed
- **Dependency analysis**: Uses `analyze_notebook()` to accurately determine each cell's output variables
- **System variable filtering**: Excludes IPython internals (`_`, `In`, `Out`, etc.) from comparison
- **Next cell comparison**: Validates if the next cell's code produces equivalent results to the current cell
- **Robust error handling**: Catches and reports errors per cell without stopping the entire process

## Implementation Details

### Cell Selection Patterns

The command supports three selection patterns:

1. **No selection** (`selected_cell_ids=None` or `[]`): Returns immediately with no work
2. **Single cell** (from cell toolbar): `selected_cell_ids=['cell1']`
3. **Multiple cells** (from notebook toolbar): `selected_cell_ids=['cell1', 'cell2', 'cell3']`

### Variable Filtering

Output variables are filtered to exclude:
- Variables starting with `_` (private/internal)
- IPython system variables: `get_ipython`, `In`, `Out`, `exit`, `quit`, `_`, `__`, `___`, `_i`, `_ii`, `_iii`, `_dh`

### Next Cell Logic

- If the next cell doesn't exist, `modified_code` is empty string
- Only code cells are used - markdown cells are skipped
- Last cell always has empty `modified_code`

## Notes

- The command requires a running `ferret_kernel`
- The comm mechanism is implemented in the kernel (`data_ferret/kernel/ferret_kernel.py`)
- The command follows the same pattern as `ProfileCommand` for selected cell processing
- Frontend integration uses the same approach as other commands with selection awareness
