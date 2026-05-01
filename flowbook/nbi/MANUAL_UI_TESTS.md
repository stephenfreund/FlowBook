# Manual UI Testing Guide — FlowBook + NBI Integration

## Prerequisites

1. FlowBook installed in dev mode: `pip install -e .`
2. Notebook Intelligence installed with the NBI PR applied
3. JupyterLab running: `jupyter lab`
4. NBI chat panel open (Claude Code or other participant)

## Test 1: Frontend Bridge Commands (Browser Console)

Open a notebook with `flowbook_kernel`, then open the browser dev console (F12).

### 1a. Check FlowBook is active

```js
await app.commands.execute('flowbook:is-active');
// Expected: {active: true, kernel: "flowbook_kernel"}
```

### 1b. Get cell count

```js
await app.commands.execute('flowbook:get-cell-count');
// Expected: {total: N, code_cells: M, markdown_cells: K}
```

### 1c. Get cell data (code-cell index 0 = first code cell)

```js
await app.commands.execute('flowbook:get-cell', { cellIndex: 0 });
// Expected: {label: "@A", cell_id: "xxxx", cell_type: "code", source: "...", ...}
```

### 1d. Get status before running anything

```js
await app.commands.execute('flowbook:get-status');
// Expected: {total_code_cells: M, executed: 0, stale: 0, ...}
```

### 1e. Run a cell

```js
await app.commands.execute('flowbook:run-cell', { cellIndex: 0 });
// Expected: {label: "@A", status: "ok", outputs_text: "...", flowbook_meta: {...}}
// Also verify: cell output is visible in JupyterLab
```

### 1f. Edit a cell source (identity-safe)

```js
// Note the cell ID before edit
let before = await app.commands.execute('flowbook:get-cell', { cellIndex: 0 });
console.log('Before ID:', before.cell_id);

await app.commands.execute('flowbook:edit-cell-source', {
  cellIndex: 0,
  source: 'x = 42'
});

let after = await app.commands.execute('flowbook:get-cell', { cellIndex: 0 });
console.log('After ID:', after.cell_id);
// CRITICAL: cell_id must be the SAME before and after
// Also: cell should show as stale in the FlowBook UI (if previously executed)
```

### 1g. Get next actionable

```js
await app.commands.execute('flowbook:get-next-actionable');
// Expected: {index: 0, label: "@A", cell_id: "...", reason: "stale"} or {done: true}
```

### 1h. Get stale cells

```js
await app.commands.execute('flowbook:get-stale-cells');
// Expected: [{index: 0, label: "@A", cell_id: "...", reason: "..."}]
```

### 1i. Get metadata after execution

```js
await app.commands.execute('flowbook:get-metadata', { cellIndex: 0 });
// Expected: includes read_locs, write_locs, execution_seq, etc.
```

## Test 2: Toolbar Button — Run All Actionable

### Setup

1. Open a notebook with 5+ code cells (none executed)
2. Verify the step-into icon button is visible in the toolbar

### 2a. Basic run-all

1. Click the toolbar button
2. **Expected**: All cells execute in order, one by one
3. **Expected**: Each cell scrolls into view as it runs
4. **Expected**: After completion, all cells show execution counts

### 2b. Stop on error

1. Edit cell @C to contain `1/0`
2. Restart kernel, click toolbar button
3. **Expected**: Cells @A, @B execute. @C errors. Loop stops.
4. **Expected**: Cells @D, @E remain unexecuted

### 2c. Stop on violation

1. Create a notebook with a NoReadAndWrite violation (e.g., `train = pd.concat([train, extra])`)
2. Ensure `continue_after_violation` is False (default)
3. Click toolbar button
4. **Expected**: Loop stops at the violating cell

### 2d. Cancel via kernel interrupt

1. Add a cell with `import time; time.sleep(60)`
2. Click toolbar button
3. While it's running that cell, click the kernel interrupt button (stop icon)
4. **Expected**: Current cell shows KeyboardInterrupt, loop stops

## Test 3: Code-Cell-Only Indexing

### Setup

Create a notebook with this structure:

- Markdown cell: "# Title"
- Code cell: `x = 1`
- Markdown cell: "## Section"
- Code cell: `y = x + 1`
- Code cell: `z = y + 2`

### 3a. Verify @-labels skip markdown

```js
// @A should be "x = 1" (first code cell, not "# Title")
let a = await app.commands.execute('flowbook:get-cell', { cellIndex: 0 });
console.log(a.label, a.source); // @A, "x = 1"

// @B should be "y = x + 1"
let b = await app.commands.execute('flowbook:get-cell', { cellIndex: 1 });
console.log(b.label, b.source); // @B, "y = x + 1"

// @C should be "z = y + 2"
let c = await app.commands.execute('flowbook:get-cell', { cellIndex: 2 });
console.log(c.label, c.source); // @C, "z = y + 2"
```

### 3b. Verify run-cell uses code-cell index

```js
// Run @B (second code cell) — should run "y = x + 1", not the markdown
await app.commands.execute('flowbook:run-cell', { cellIndex: 1 });
// Verify "y = x + 1" cell has output, not the markdown cell
```

## Test 4: NBI Extension (requires NBI PR)

### 4a. Extension discovery

```bash
# After pip install -e ., verify extension.json is installed
ls $(python -c "import sys; print(sys.prefix)")/share/jupyter/nbi_extensions/flowbook/
# Expected: extension.json
```

### 4b. Extension activation

1. Launch JupyterLab with NBI
2. Check NBI server logs for: `Activated NBI extension 'flowbook.nbi.extension.FlowBookNBIExtension'`

### 4c. Tool disabling

1. Open NBI chat
2. Ask Claude: "What tools do you have?"
3. **Expected**: FlowBook tools are listed (get_flowbook_metadata, alpha_rename, etc.)
4. **Expected**: NBI's `set_cell_type_and_source` is NOT listed (disabled)

### 4d. Basic workflow via Claude

1. Open a notebook with `flowbook_kernel`
2. In NBI chat, type: "What is the reproducibility status of this notebook?"
3. **Expected**: Claude calls `get_flowbook_status` and reports cell counts
4. Type: "Run all the cells"
5. **Expected**: Claude calls `run_actionable_cells`, cells execute visibly
6. Type: "What does cell @A read and write?"
7. **Expected**: Claude calls `get_flowbook_metadata` with cell="@A"

### 4e. Refactoring via Claude

1. Open a notebook with a reproducibility violation
2. Ask: "Fix the reproducibility violations"
3. **Expected**: Claude uses `checkpoint`, `get_next_actionable_cell`, `read_cell`, then appropriate refactoring tools (`alpha_rename`, `remove_inplace`, etc.)
4. **Expected**: Cell edits are visible in JupyterLab immediately (identity-safe)
5. **Expected**: After fixes, `run_actionable_cells` shows notebook is reproducible

### 4f. Identity preservation

1. Open a notebook, note a cell ID (via browser console: `app.shell.currentWidget.content.widgets[0].model.id`)
2. Ask Claude to edit that cell's source
3. Verify the cell ID is unchanged after the edit
4. Verify FlowBook's staleness highlighting updates correctly

## Test 5: Standalone MCP Server

### 5a. New tools

```bash
# Start the MCP server and test new tools
# (Normally done via Claude Code with FlowBook MCP configured)
```

Via Claude Code with FlowBook MCP:

1. Load a notebook: "Load examples/demos/01_Basic_Tracking.ipynb"
2. Run actionable cells: "Run all actionable cells"
3. **Expected**: `run_actionable_cells` loops, shows @-labels in output
4. Get metadata: "Show me the metadata for @A"
5. **Expected**: `get_flowbook_metadata` returns read/write locs with @-labels

### 5b. Output visibility

1. Load a notebook via MCP
2. Have JupyterLab open with the same notebook (shared kernel)
3. Run a cell via MCP
4. **Expected**: Cell outputs appear in JupyterLab (via Contents API push)

### 5c. Tool renames

1. Try using `read_cell` (should work)
2. Try using `edit_cell_source` (should work)
3. The old names `get_cell` and `edit_cell` should no longer be available

## Test 6: Skills

### 6a. /basic-run

```
/basic-run examples/demos/01_Basic_Tracking.ipynb
```

- **Expected**: Uses `run_actionable_cells` instead of `run_all_cells`
- **Expected**: Output shows @-labels

### 6b. /fix-notebook

```
/fix-notebook examples/demos/05_Column_Conflict.ipynb
```

- **Expected**: Uses `read_cell` (not `get_cell`), `edit_cell_source` (not `edit_cell`)
- **Expected**: Cell references use @-labels in output table
