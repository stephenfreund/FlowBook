# Categorize and Fix Reproducibility Errors

Analyze reproducibility errors from a FlowBook error report or directly from a processed notebook, categorize each error, and optionally fix them.

## Usage

```
/categorize-repro-errors ERROR_REPORT_FILE [NOTEBOOKS_DIR] [--fix]
/categorize-repro-errors NOTEBOOK_PATH [--fix]
```

**Mode 1: Error Report Mode**

- `ERROR_REPORT_FILE`: Path to the error report file (e.g., `errors.txt`)
- `NOTEBOOKS_DIR`: Optional directory containing the notebook files
- `--fix`: Optional flag to apply fixes after categorization

**Mode 2: Single Notebook Mode** (when only a `.ipynb` path is provided)

- `NOTEBOOK_PATH`: Path to a processed notebook (must have been run through FlowBook kernel)
- `--fix`: Optional flag to apply fixes after categorization

## Task

### For Error Report Mode:

1. Parse the error report file using `flowbook/scripts/parse_repro_errors.py`
2. For each notebook with errors, launch a parallel agent to analyze and categorize errors

### For Single Notebook Mode:

1. Extract errors directly from the notebook's cell metadata (see "Extracting Errors from Notebook" below)
2. Analyze and categorize errors for that single notebook

3. Each error should be categorized into exactly ONE of these categories:

### Error Categories

| Category | Description | Example | Fix Strategy |
|----------|-------------|---------|--------------|
| **In-place variable reassignment** | Cell reads and overwrites same variable | `train = pd.concat([train, extra])` | Deep-copy + alpha-rename |
| **Sequential transformation chain** | Downstream depends on upstream transformation | Imputation then feature engineering | Deep-copy + alpha-rename |
| **Diagnostic inspection before mutation** | Read-only cell captures pre-transformation state | `df.info()` before `df["col"] = ...` | Cell split + `%diagnostic` |
| **Visualization before mutation** | Plot accesses all columns before column added | `sns.heatmap(df.corr())` before new col | Cell split + `%diagnostic` |
| **Reusing variable for different purposes** | Variable reused for different purposes in disjoint regions of the code | `model` reused for different model | Alpha-rename downstream |
| **Unrecoverable in-place mutation** | Cell mutates object without rebinding | `model.fit()`, `df.drop(inplace=True)` | See sub-types below |

### Unrecoverable Mutation Sub-types

When the predicate is `"unrecoverable_mutation"`, identify the sub-type from the cell source:

| Sub-type                  | Detection Pattern                                          | Fix Type           | Example                      |
| ------------------------- | ---------------------------------------------------------- | ------------------ | ---------------------------- |
| **ML model mutation**     | `.fit()`, `.fit_transform()`, `.predict()` on model/scaler | `model-copy`       | `model.fit(X, y)`            |
| **DataFrame inplace**     | `inplace=True` argument                                    | `inplace-to-copy`  | `df.drop(col, inplace=True)` |
| **Structural assignment** | `.columns = ...`, `.index = ...`                           | `struct-copy`      | `df.columns = ['a', 'b']`    |
| **Container mutation**    | `.append()`, `[i] = ...` on list/dict/array                | `inplace-reassign` | `arr[5] = 99`                |

**Why these are unrecoverable:** Re-executing the cell cannot restore the full value of the variable. For example, `model.fit()` only trains the model — it cannot "un-train" changes from a deleted cell. Similarly, `arr[5] = 99` sets one element but cannot restore what a deleted cell wrote to `arr[3]`.

## Important Notes

- **Always use code cell indices (`@N`)** to reference cells, NOT cell IDs. Cell IDs in processed notebooks may not match the original notebooks. The error report provides both cell IDs and code cell indices — always use the code cell index with `@` prefix (e.g., `@5`).
- Cell indices in error reports are **CODE cell indices** (not including markdown cells)
- Write results to `error_categories.tsv` as you go
- TSV format: `NOTEBOOK_NAME<TAB>ERROR_NUMBER<TAB>CELL_ID<TAB>CELL_CODE_INDEX<TAB>CATEGORY<TAB>VARIABLE<TAB>FIX_TYPE<TAB>EXPLANATION`
- The VARIABLE column should contain the primary variable involved in the error
- The EXPLANATION column should contain the rationale for the categorization

## Fix Script Usage

After categorization, apply fixes using `flowbook/scripts/fix_repro_errors.py`.

**IMPORTANT:** Always use `@N` notation (code cell index) when calling the fix script. Never use cell IDs.

### High-level fix types (single command):

```bash
# For in-place reassignment or sequential chain:
python flowbook/scripts/fix_repro_errors.py NOTEBOOK @CODE_INDEX --fix-type inplace-reassign --variable VAR

# For variable reuse:
python flowbook/scripts/fix_repro_errors.py NOTEBOOK @CODE_INDEX --fix-type variable-reuse --variable VAR

# For ML model mutation (unrecoverable):
python flowbook/scripts/fix_repro_errors.py NOTEBOOK @CODE_INDEX --fix-type model-copy --variable VAR

# For DataFrame inplace=True (unrecoverable):
python flowbook/scripts/fix_repro_errors.py NOTEBOOK @CODE_INDEX --fix-type inplace-to-copy --variable VAR

# For structural assignment (unrecoverable):
python flowbook/scripts/fix_repro_errors.py NOTEBOOK @CODE_INDEX --fix-type struct-copy --variable VAR
```

### Diagnostic/Visualization fixes (agent-driven cell splitting):

For diagnostic and visualization errors, the agent must analyze the cell and split it using primitive operations. **Do NOT blindly add `%diagnostic` to cells that contain mutations.**

The fix script provides three primitive operations:

```bash
# Replace a cell's source code:
python flowbook/scripts/fix_repro_errors.py NOTEBOOK @N --fix-type set-source --source-file PATH

# Insert a new code cell after @N:
python flowbook/scripts/fix_repro_errors.py NOTEBOOK @N --fix-type insert-cell-after --source-file PATH

# Prepend %diagnostic magic to a cell:
python flowbook/scripts/fix_repro_errors.py NOTEBOOK @N --fix-type add-diagnostic
```

**Index safety:** `insert-cell-after` shifts all subsequent code cell indices by 1. When applying multiple fixes to the same notebook, process diagnostic/visualization fixes from **highest code cell index to lowest** to avoid index invalidation. Other fix types (inplace-reassign, model-copy, etc.) don't insert cells and are safe in any order.

#### How to fix diagnostic/visualization errors:

1. **Read the cell source** at the error's code cell index
2. **Classify each line** as either mutation (writes variables, assigns, calls .fit(), etc.) or diagnostic (print, display, .info(), .head(), plotting, etc.)
3. **Decide the fix**:
   - **If the cell is purely diagnostic** (no mutations at all): Just use `add-diagnostic`
   - **If the cell mixes mutation and diagnostic code**: Split it:
     a. Write the mutation lines to a temp file
     b. Write the diagnostic lines to a temp file
     c. Use `set-source` to replace the original cell with mutation-only code
     d. Use `insert-cell-after` to add the diagnostic code as a new cell
     e. Use `add-diagnostic` on the new cell (which is now at @N+1)
   - **If the cell is purely mutation** (no diagnostic code): This was miscategorized. Do NOT add `%diagnostic`. Instead, recategorize as `inplace-reassign` or `sequential-chain` and apply the appropriate deep-copy fix.
4. **Add `# [FLOWBOOK FIX]` comments** to both the mutation and diagnostic cells explaining what was done

The script creates `<notebook>-fixed.ipynb` with:

- Comments marked `# [FLOWBOOK FIX]` explaining the original error and fix
- Deep copies with `_flow_XXXX` suffix for renamed variables
- Split cells with `%diagnostic` magic on the read-only part
- For `model-copy`: Uses `safe_model_copy()` which handles sklearn, PyTorch, XGBoost, etc.
- For `inplace-to-copy`: Converts `df.method(inplace=True)` to `df = df.method()`

## Extracting Errors from Notebook

When using Single Notebook Mode, extract errors from the notebook's cell outputs. FlowBook stores violation information in `display_data` outputs with special metadata keys.

**Python code to extract errors from a notebook:**

```python
import json
from pathlib import Path

def extract_errors_from_notebook(notebook_path: str) -> dict:
    """Extract reproducibility errors from a processed FlowBook notebook.

    Returns dict in same format as parse_repro_errors.py output:
    {
        "notebook.ipynb": {
            "notebook_path": "/path/to/notebook.ipynb",
            "error_count": N,
            "errors": [
                {
                    "error_num": 1,
                    "cell_id": "abcd",
                    "cell_index": 5,  # CODE cell index
                    "summary": "Cell @X reads and writes the same locations: var",
                    "predicate": "no_read_and_write",
                    "locations": ["var"],
                    "accepted": True
                },
                ...
            ]
        }
    }
    """
    path = Path(notebook_path)
    with open(path) as f:
        nb = json.load(f)

    errors = []
    code_cell_index = 0
    error_num = 0

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue

        cell_id = cell.get('id', '')

        # Check outputs for predicate_violation metadata
        for output in cell.get('outputs', []):
            if output.get('output_type') != 'display_data':
                continue

            metadata = output.get('metadata', {})

            # Check for predicate_violation (NoReadAndWrite, BackwardStale, etc.)
            if 'predicate_violation' in metadata:
                violation = metadata['predicate_violation']
                error_num += 1
                errors.append({
                    'error_num': error_num,
                    'cell_id': violation.get('cell_id', cell_id),
                    'cell_index': code_cell_index,
                    'summary': violation.get('message', ''),
                    'predicate': violation.get('predicate', ''),
                    'locations': violation.get('locations', []),
                    'accepted': violation.get('accepted', True)
                })

            # Also check flowbook metadata for stale cells info
            if 'flowbook' in metadata:
                fb = metadata['flowbook']
                # Stale cells indicate forward contamination (not an error per se,
                # but useful context for categorization)

        code_cell_index += 1

    notebook_name = path.name
    return {
        notebook_name: {
            'notebook_path': str(path.absolute()),
            'error_count': len(errors),
            'errors': errors
        }
    }
```

**Key metadata locations in notebook JSON:**

- `cell.outputs[].metadata.predicate_violation` - Violation details:
  - `predicate`: Type of violation (`"no_read_and_write"`, `"backward_stale"`, etc.)
  - `cell_id`: Cell that triggered the violation
  - `locations`: List of variable names involved
  - `message`: Human-readable error message
  - `accepted`: Whether execution continued despite violation
- `cell.outputs[].metadata.flowbook` - Execution metadata:
  - `reads`: Variables read by the cell
  - `writes`: Variables written by the cell
  - `stale_cells`: List of cell IDs that became stale
  - `staleness_reasons`: Dict mapping cell_id to reason objects

## Instructions

When the user invokes this command:

### Detect Mode

First, determine which mode to use:

- If the first argument ends with `.ipynb`, use **Single Notebook Mode**
- Otherwise, use **Error Report Mode**

### Error Report Mode

1. Run the parsing script to get structured error data:

   ```bash
   python flowbook/scripts/parse_repro_errors.py $ERROR_REPORT_FILE $NOTEBOOKS_DIR --json
   ```

2. Create/initialize the output file `error_categories.tsv` with header

3. For each notebook with errors, launch a parallel **opus** agent to:
   - Read the relevant section from the error report (use line_range from parsed data)
   - Read the notebook to understand context
   - Categorize each error according to the taxonomy
   - Identify the primary variable involved
   - **Always reference cells by code cell index (`@N`), never by cell ID**
   - For diagnostic/visualization errors: read the cell source, determine which lines are mutation vs diagnostic, and note the split plan
   - Produce a short, coherent explanation for the categorization
   - Output TSV lines

### Single Notebook Mode

1. Read the notebook JSON and extract errors using the logic above (check each cell's outputs for `predicate_violation` metadata)

2. If no errors found, report that the notebook has no reproducibility violations

3. For each error found:
   - Read the cell source code to understand context
   - Look at surrounding cells for the full picture
   - Categorize the error according to the taxonomy above
   - Identify the primary variable involved (from `locations` or by analyzing the code)
   - **Always reference cells by code cell index (`@N`)**

### Final Steps (both modes)

4. Collect results and write to `error_categories.tsv` (for single notebook mode, also print to console)

5. Print a summary of categories found

6. If `--fix` flag is provided:
   - For each notebook, initialize the fixed copy with `--init --force`
   - Apply non-inserting fixes (inplace-reassign, sequential-chain, model-copy, inplace-to-copy, struct-copy, variable-reuse) in any order
   - Apply diagnostic/visualization splits from **highest code cell index to lowest** (to avoid index shifts from cell insertions):
     - Read the cell source
     - Determine mutation vs diagnostic lines
     - If purely diagnostic: use `add-diagnostic`
     - If mixed: write temp files and use `set-source` + `insert-cell-after` + `add-diagnostic`
     - If purely mutation: skip %diagnostic, apply appropriate rename fix instead
   - Report which notebooks were fixed and where the `-fixed.ipynb` files are

## Progress Reporting

**Report progress as you work.** For each notebook being processed, output status like:

```
[1/23] backpack-pred-baseline-ensemble-eda.ipynb (4 errors)
  - Error 1 (@3): Diagnostic inspection before mutation → train_data [split: 4 mutation + 2 diagnostic lines]
  - Error 2 (@5): Sequential transformation chain → test_data
  - Error 3 (@8): Visualization before mutation → train_data [pure diagnostic, add %diagnostic]
  - Error 4 (@12): Sequential transformation chain → train_data
```

When applying fixes (with `--fix`):

```
[1/23] backpack-pred-baseline-ensemble-eda.ipynb
  Initializing: backpack-pred-baseline-ensemble-eda-fixed.ipynb
  - Fixing @5: sequential-chain --variable test_data
  - Fixing @12: sequential-chain --variable train_data
  - Fixing @8: add-diagnostic (pure diagnostic cell)
  - Fixing @3: set-source + insert-cell-after + add-diagnostic (split mixed cell)
  ✓ Fixed 4 errors → backpack-pred-baseline-ensemble-eda-fixed.ipynb
```

At the end, print a summary:

```
=== Summary ===
Notebooks processed: 23
Total errors categorized: 116
  - In-place variable reassignment: 41
  - Sequential transformation chain: 51
  - Diagnostic inspection before mutation: 17
  - Visualization before mutation: 2
  - Reusing variable for different purposes: 5
  - Unrecoverable mutation (ML model): 12
  - Unrecoverable mutation (inplace): 8
  - Unrecoverable mutation (structural): 3

Fixed notebooks saved to:
  - .../backpack-pred-baseline-ensemble-eda-fixed.ipynb
  - .../forecasting-sticker-sales-fixed.ipynb
  - ...
```
