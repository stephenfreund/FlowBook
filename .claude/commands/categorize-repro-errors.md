# Categorize and Fix Reproducibility Errors

Analyze reproducibility errors from a FlowBook error report or directly from a processed notebook, categorize each error, and optionally fix them.

## Usage

```
/categorize-repro-errors ERROR_REPORT_FILE NOTEBOOKS_DIR [--fix]
/categorize-repro-errors NOTEBOOK_PATH [--fix]
```

**Mode 1: Error Report Mode**
- `ERROR_REPORT_FILE`: Path to the error report file (e.g., `errors.txt`)
- `NOTEBOOKS_DIR`: Directory containing the notebook files
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
| **Diagnostic inspection before mutation** | Read-only cell captures pre-transformation state | `df.info()` before `df["col"] = ...` | Add `%diagnostic` magic |
| **Visualization before mutation** | Plot accesses all columns before column added | `sns.heatmap(df.corr())` before new col | Add `%diagnostic` magic |
| **Reusing variable for different purposes** | Variable reused for different purposes | `model` reused for different model | Alpha-rename downstream |

## Important Notes

- Cell indices in error reports are **CODE cell indices** (not including markdown cells)
- Write results to `error_categories.tsv` as you go
- TSV format: `NOTEBOOK_NAME<TAB>ERROR_NUMBER<TAB>CELL_ID<TAB>CATEGORY<TAB>VARIABLE`
- The VARIABLE column should contain the primary variable involved in the error

## Fix Script Usage

After categorization, apply fixes using `flowbook/scripts/fix_repro_errors.py`:

```bash
# For in-place reassignment or sequential chain:
python flowbook/scripts/fix_repro_errors.py NOTEBOOK CELL_ID --fix-type inplace-reassign --variable VAR

# For diagnostic/visualization:
python flowbook/scripts/fix_repro_errors.py NOTEBOOK CELL_ID --fix-type diagnostic-split

# For variable reuse:
python flowbook/scripts/fix_repro_errors.py NOTEBOOK CELL_ID --fix-type variable-reuse --variable VAR
```

The script creates `<notebook>-fixed.ipynb` with:
- Comments marked `# [FLOWBOOK FIX]` explaining the original error and fix
- Deep copies with `_flow_XXXX` suffix for renamed variables
- `%diagnostic` magic for inspection cells (tells kernel to skip reproducibility checks)

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

3. For each notebook with errors, launch a background agent to:
   - Read the relevant section from the error report (use line_range from parsed data)
   - Read the notebook to understand context
   - Categorize each error according to the taxonomy
   - Identify the primary variable involved
   - Output TSV lines

### Single Notebook Mode

1. Read the notebook JSON and extract errors using the logic above (check each cell's outputs for `predicate_violation` metadata)

2. If no errors found, report that the notebook has no reproducibility violations

3. For each error found:
   - Read the cell source code to understand context
   - Look at surrounding cells for the full picture
   - Categorize the error according to the taxonomy above
   - Identify the primary variable involved (from `locations` or by analyzing the code)

### Final Steps (both modes)

4. Collect results and write to `error_categories.tsv` (for single notebook mode, also print to console)

5. Print a summary of categories found

6. If `--fix` flag is provided:
   - For each error, apply fixes using the fix script
   - Report which notebooks were fixed and where the `-fixed.ipynb` files are

## Progress Reporting

**Report progress as you work.** For each notebook being processed, output status like:

```
[1/23] backpack-pred-baseline-ensemble-eda.ipynb (4 errors)
  - Error 1 (cell tpje): Diagnostic inspection before mutation → train_data
  - Error 2 (cell fdke): Sequential transformation chain → test_data
  - Error 3 (cell gdch): Visualization before mutation → train_data
  - Error 4 (cell sozj): Sequential transformation chain → train_data
```

When applying fixes (with `--fix`):

```
[1/23] backpack-pred-baseline-ensemble-eda.ipynb
  Initializing: backpack-pred-baseline-ensemble-eda-fixed.ipynb
  - Fixing cell tpje: diagnostic-split
  - Fixing cell fdke: sequential-chain --variable test_data
  - Fixing cell gdch: visualization-split
  - Fixing cell sozj: sequential-chain --variable train_data
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

Fixed notebooks saved to:
  - .../backpack-pred-baseline-ensemble-eda-fixed.ipynb
  - .../forecasting-sticker-sales-fixed.ipynb
  - ...
```
