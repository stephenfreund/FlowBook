---
description: "Run a Jupyter notebook through FlowBook, executing all cells in order. Stops on the first error and reports it. Shows the reproducibility status at the end."
---

# Basic Run

Run a notebook through FlowBook using the MCP server. Execute all cells in order, report any errors or reproducibility violations.

**Input**: $ARGUMENTS (path to a .ipynb file)

## Steps

1. Load the notebook:
   ```
   load_notebook("$ARGUMENTS")
   ```

2. Enable continue-after-violation so all cells run even if there are violations:
   ```
   continue_after_violation(true)
   ```

3. Run all cells:
   ```
   run_all_cells()
   ```

4. If `run_all_cells` reports an error, get the failing cell to show the user what went wrong:
   ```
   get_next_actionable_cell()
   ```

5. Show the reproducibility status:
   ```
   get_status()
   ```

6. Print the session log for review:
   ```
   print_log()
   ```

7. Save the log:
   ```
   save_log()
   ```

## Output

Report to the user:
- Whether the notebook ran successfully or which cell errored
- Number of reproducibility violations found (if any), with a brief description of each
- Number of stale cells
- Path to the saved log file

Keep the report concise. If there are violations, list them briefly (cell ID, type, variable). Do not attempt to fix anything — just report.
