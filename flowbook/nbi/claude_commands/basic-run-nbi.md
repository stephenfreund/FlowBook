---
description: 'Run the currently open notebook through FlowBook, executing all cells. Stops on the first error and reports reproducibility status. Works on the active JupyterLab notebook.'
---

# Basic Run (NBI)

Run the currently open notebook through FlowBook using NBI tools. Execute all cells, report any errors or reproducibility violations.

**Input**: $ARGUMENTS (optional — ignored, works on active notebook)

## Steps

1. Enable continue-after-violation so all cells run even if there are violations:

   ```
   continue_after_violation(true)
   ```

2. Run all actionable cells:

   ```
   run_actionable_cells()
   ```

3. Show the reproducibility status:

   ```
   get_status()
   ```

4. If there were errors or violations, show the first problem:

   ```
   get_next_actionable_cell()
   ```

5. Print the session log for review:

   ```
   print_log()
   ```

## Output

Report to the user:

- Whether the notebook ran successfully or which cell errored (use @A notation)
- Number of reproducibility violations found (if any), with a brief description of each
- Number of stale cells
- Whether the notebook is reproducible

Keep the report concise. If there are violations, list them briefly (cell @-label, type, variable). Do not attempt to fix anything — just report.
