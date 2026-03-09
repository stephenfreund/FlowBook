# Categorize and Fix Reproducibility Errors

Analyze reproducibility errors from a FlowBook error report, categorize each error, and optionally fix them.

## Usage

```
/categorize-repro-errors ERROR_REPORT_FILE NOTEBOOKS_DIR [--fix]
```

Arguments:
- `ERROR_REPORT_FILE`: Path to the error report file (e.g., `errors.txt`)
- `NOTEBOOKS_DIR`: Directory containing the notebook files
- `--fix`: Optional flag to apply fixes after categorization

## Task

1. Parse the error report file using `flowbook/scripts/parse_repro_errors.py`
2. For each notebook with errors, launch a parallel agent to analyze and categorize errors
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

## Instructions

When the user invokes this command:

1. First, run the parsing script to get structured error data:
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

4. Collect all results and write to `error_categories.tsv`

5. Print a summary of categories found

6. If `--fix` flag is provided:
   - For each notebook, apply fixes using the fix script
   - Report which notebooks were fixed and where the -fixed.ipynb files are

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
