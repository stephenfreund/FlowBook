## Fix algorithm

IMPORTANT: Before each tool call, print a one-line status message explaining what you're doing
(e.g., "Running all actionable cells...", "Checkpointing before fix...", "Renaming 'df' to 'df_clean' from @C...").

To read the full notebook, call read_cell() with no arguments — returns all cells in one call.
To read a single cell, pass its @-label (e.g., read_cell("@C")).

**Run every change immediately.** After every `add_cell`, `edit_cell_source`,
`alpha_rename`, `insert_deepcopy`, `remove_inplace`, `merge_cells`, `move_cell`, or
`mark_diagnostic`, call `run_actionable_cells()` before making the next change. Do not
batch edits. Running after each change surfaces errors and violations at their true
source; batching hides which edit caused the failure and wastes checkpoint rollbacks.

Run-and-fix loop:
1. run_actionable_cells() — runs all stale/unexecuted cells, stops on first error or violation.
2. If violations found:
   a. checkpoint() — save state before attempting fix.
   b. get_next_actionable_cell() — find the problem cell.
   c. read_cell("@X") — read the problematic cell.
   d. Apply the appropriate fix from the taxonomy above.
   e. run_actionable_cells() — re-run to check the fix worked.
   f. If worse, restore(checkpoint_id) and try a different strategy.
3. save_notebook() when done.

Use checkpoint() before making changes, restore() if needed.
Always use FlowBook tools for cell operations — they preserve cell identity and track reproducibility.
Never use indices or cell IDs directly — always use @A notation.
