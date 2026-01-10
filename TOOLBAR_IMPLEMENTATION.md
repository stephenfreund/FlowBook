# Toolbar Implementation with ToolbarRegistry

## Overview

The FlowBook extension now uses the modern JupyterLab 4.0+ `IToolbarWidgetRegistry` approach for managing toolbar buttons. Commands appear in two locations:

1. **Notebook Toolbar** - Buttons at the top of each notebook
2. **Cell Toolbar** - Buttons in the cell toolbar (shown when cells are selected)

## Architecture

### Dual Registration Strategy

The extension uses a hybrid approach to ensure buttons appear by default while also being customizable:

1. **ToolbarRegistry** - Registers factories for toolbar customization
2. **DocumentRegistry** - Direct insertion ensures buttons appear by default

### Components

#### 1. Notebook Toolbar (`src/toolbar.ts`)

**Registration Function:**

```typescript
registerNotebookToolbarItems(manager, toolbarRegistry);
```

- Registers button factories with `IToolbarWidgetRegistry`
- Makes buttons available in JupyterLab's toolbar customizer
- Buttons can be added/removed/reordered by users

**Direct Insertion Class:**

```typescript
NotebookToolbarExtension implements DocumentRegistry.IWidgetExtension
```

- Adds buttons directly to new notebook panels
- Ensures buttons appear by default without user configuration
- Uses `CommandToolbarButton` for consistency

#### 2. Cell Toolbar (`src/celltoolbar.ts`)

**Registration Function:**

```typescript
registerCellToolbarItems(manager, toolbarRegistry);
```

- Registers button factories for the cell toolbar
- Cell toolbar = notebook's toolbar shown when a cell is selected
- **Not** individual toolbars on each cell (JupyterLab 4.0 doesn't support this)

**Placeholder Class:**

```typescript
CellToolbarExtension;
```

- Kept for compatibility
- Cell toolbars fully managed via ToolbarRegistry
- No additional setup needed

### Integration (`src/index.ts`)

```typescript
// Register with ToolbarRegistry (for customization)
registerNotebookToolbarItems(manager, toolbarRegistry);
registerCellToolbarItems(manager, toolbarRegistry);

// Add to notebooks by default via DocumentRegistry
const notebookExtension = new NotebookToolbarExtension(manager);
app.docRegistry.addWidgetExtension('Notebook', notebookExtension);

// Initialize cell toolbar
new CellToolbarExtension(manager, tracker);
```

## How It Works

### Notebook Toolbar Flow

1. **Extension Activation**
   - Plugin loads and creates `FlowbookCommandsManager`
   - Commands loaded from server via `/flowbook/list`

2. **Registration**
   - `registerNotebookToolbarItems()` registers factories with `IToolbarWidgetRegistry`
   - Each command gets a factory: `Notebook` → `flowbook-{command_id}`

3. **Direct Insertion**
   - `NotebookToolbarExtension.createNew()` called for each new notebook
   - Buttons created using `CommandToolbarButton`
   - Inserted at rank 100+ to appear after default buttons

4. **Button Click**
   - `CommandToolbarButton` executes JupyterLab command
   - Command ID format: `ferret:{command_id}`
   - Manager handles execution via `executeCommand()`

### Cell Toolbar Flow

1. **Registration**
   - `registerCellToolbarItems()` registers factories with type `'Cell'`
   - Factories create `CommandToolbarButton` instances

2. **Display**
   - JupyterLab shows cell toolbar when user selects a cell
   - Registered buttons appear in the cell toolbar
   - Same commands as notebook toolbar, just different location

3. **Customization**
   - Users can customize cell toolbar via settings
   - Buttons can be added/removed/reordered
   - Configuration persists across sessions

## Button Creation

All buttons use `CommandToolbarButton`:

```typescript
new CommandToolbarButton({
  commands: app.commands,
  id: commandId, // e.g., 'ferret:analyze'
  label: cmdInfo.label // e.g., 'Analyze Notebook'
});
```

### Benefits

- **Consistent appearance** with other JupyterLab buttons
- **Automatic icon handling** based on command registration
- **Tooltip support** from command caption
- **Execution through command system** for proper state management

## JupyterLab 4.0 Changes

### vs. JupyterLab 3.x

**Old Approach (3.x):**

```typescript
// Manual button creation
const button = new ToolbarButton({
  label: 'My Command',
  onClick: () => {
    /* handler */
  }
});
panel.toolbar.addItem('my-button', button);
```

**New Approach (4.0+):**

```typescript
// Factory registration
toolbarRegistry.addFactory('Notebook', 'my-button', panel => {
  return new CommandToolbarButton({
    commands: app.commands,
    id: 'my:command'
  });
});

// Still supported for default appearance:
panel.toolbar.insertItem(rank, 'my-button', button);
```

### Key Differences

1. **Factory Pattern** - Buttons created by factories, not directly
2. **ToolbarRegistry** - Central registration for customization
3. **CommandToolbarButton** - Preferred over manual ToolbarButton
4. **Cell Toolbar** - New in 4.0, shows when cells selected
5. **Customization UI** - Built-in UI for toolbar management

## Customization

### User Customization

Users can customize toolbars via:

1. **View → Customize Toolbar**
2. **Right-click toolbar → Customize**
3. Settings → Advanced Settings → Notebook

### Available Actions

- **Add** buttons from available items
- **Remove** buttons from toolbar
- **Reorder** buttons by dragging
- **Reset** to default configuration

### Programmatic Customization

Extensions can provide default configurations:

```json
{
  "toolbar": [
    { "name": "flowbook-analyze", "rank": 100 },
    { "name": "flowbook-validate", "rank": 101 }
  ]
}
```

## Command Registration

Commands registered in `FlowbookCommandsManager.registerCommands()`:

```typescript
this.app.commands.addCommand(`ferret:${cmdInfo.id}`, {
  label: cmdInfo.label,
  caption: cmdInfo.tooltip,
  execute: async () => {
    const current = this.tracker.currentWidget;
    if (current) {
      await this.executeCommand(cmdInfo.id, current);
    }
  }
});
```

### Command Execution

1. User clicks button
2. `CommandToolbarButton` executes command via `app.commands`
3. Command handler gets current notebook from tracker
4. `executeCommand()` sends request to `/flowbook/execute`
5. Server processes command and returns results
6. Notebook updated with results

## Rank System

Buttons positioned using rank (lower = left/earlier):

| Range | Purpose                    |
| ----- | -------------------------- |
| 0-99  | JupyterLab default buttons |
| 100+  | Ferret extension buttons   |

**Ferret button ranks:**

- First command: 100
- Second command: 101
- Third command: 102
- etc.

## Debugging

### Check Registration

**Console Commands:**

```javascript
// Check if factories registered
window.jupyterapp.commands.listCommands().filter(c => c.startsWith('ferret:'));

// Check toolbar registry
// (ToolbarRegistry not directly accessible from console)
```

**Python Server:**

```bash
# Check available commands
curl http://localhost:8888/flowbook/list
```

### Common Issues

**Buttons not appearing:**

1. Check extension loaded: `jupyter labextension list`
2. Verify commands loaded from server
3. Check browser console for errors
4. Ensure notebook toolbar not customized to hide buttons

**Buttons in wrong position:**

1. Check rank values (should be 100+)
2. Verify insertion index in `NotebookToolbarExtension`
3. Check for rank conflicts with other extensions

**Commands not executing:**

1. Verify command registered: check `app.commands`
2. Check server `/flowbook/execute` endpoint
3. Look for errors in browser/server console
4. Verify notebook tracker has current widget

## Benefits of This Approach

### For Users

✅ **Buttons appear by default** - No configuration needed
✅ **Full customization** - Can add/remove/reorder via UI
✅ **Consistent experience** - Matches JupyterLab standards
✅ **Per-cell actions** - Cell toolbar for cell-specific operations

### For Developers

✅ **Modern API** - Uses JupyterLab 4.0+ best practices
✅ **Future-proof** - Compatible with JupyterLab evolution
✅ **Maintainable** - Clear separation of concerns
✅ **Extensible** - Easy to add new commands

## Testing

### Manual Testing

1. **Notebook Toolbar:**
   - Open any notebook
   - Check toolbar for Ferret buttons
   - Click button, verify command executes

2. **Cell Toolbar:**
   - Select a code cell
   - Check cell toolbar appears
   - Verify Ferret buttons present

3. **Customization:**
   - Right-click toolbar → Customize
   - Try removing a Ferret button
   - Verify button removed
   - Reset to defaults

### Automated Testing

Currently no automated tests. Future additions:

- Test factory registration
- Test button creation
- Test command execution
- Test toolbar customization

## Future Enhancements

Possible improvements:

1. **Icons** - Custom icons for each command
2. **Context Sensitivity** - Hide/disable buttons based on cell type
3. **Keyboard Shortcuts** - Add shortcuts to commands
4. **Status Indicators** - Show command execution state
5. **Dropdown Menus** - Group related commands
6. **Cell-Specific Commands** - Commands that operate on selected cell only

## References

- [JupyterLab 4.0 Extension Migration Guide](https://jupyterlab.readthedocs.io/en/latest/extension/extension_migration.html)
- [IToolbarWidgetRegistry API](https://jupyterlab.readthedocs.io/en/latest/api/interfaces/apputils.IToolbarWidgetRegistry.html)
- [DocumentRegistry API](https://jupyterlab.readthedocs.io/en/latest/api/classes/docregistry.DocumentRegistry.html)
