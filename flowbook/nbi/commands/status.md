---
description: 'Show the reproducibility status of the current notebook'
---

# Notebook Reproducibility Status

Summarize the reproducibility state of the currently open notebook.

1. Call `get_status` once to get the aggregate counts (total, executed, clean, stale,
   violations, reproducible).
2. Call `read_cell()` with no arguments to get every code cell with its @-label,
   FlowBook status, and any violation reason.
3. Respond with exactly two sections:

   **Summary**: one line in the format
   `N/M executed | K violations | S stale | reproducible ✓/✗`.

   **Cells**: a markdown table with columns `| Cell | Status | Note |`.
   - `Cell` is the @-label (`@A`, `@B`, ...).
   - `Status` is one of `clean`, `stale`, `violation`, or `unrun`.
   - `Note` is the violation predicate + location when present, the staleness cause
     when available, or empty.

Do not call refactor, checkpoint, or execution tools. Do not run cells. Do not save.
This command is read-only.
