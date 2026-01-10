# MessagePanel Component

## Overview

The `MessagePanel` is a JupyterLab panel that displays real-time messages from the FlowBook server's message broadcaster. It appears on the right side of the JupyterLab interface and shows output from commands as they execute.

## Architecture

### Frontend (TypeScript)

**File**: `src/panel.tsx`

The `MessagePanel` class extends Lumino's `Widget` and uses React for rendering. It:

1. Connects to the server's SSE (Server-Sent Events) stream at `/flowbook/stream`
2. Receives messages in real-time as commands execute
3. Displays messages in a scrollable, auto-updating panel
4. Automatically reconnects if the connection is lost

**Key Features**:

- Auto-scrolling to show latest messages
- Reconnection logic with 5-second retry
- Four message types: `APPEND`, `NEWLINE`, `END`, `CLEAR`
- Proper cleanup on disposal

### Backend (Python)

**Files**:

- `flowbook/server/message_broadcaster.py` - Singleton broadcaster
- `flowbook/server/handlers.py` - SSE handler at `/flowbook/stream`

The `MessageBroadcaster` uses async queues to broadcast messages to all connected clients via Server-Sent Events.

## Usage

### Displaying Messages from Commands

Commands can use the broadcaster to send real-time updates:

```python
from flowbook.server.base import NotebookCommand
from flowbook.server.message_broadcaster import get_broadcaster

class MyCommand(NotebookCommand):
    def process(self, notebook_content: dict, kernel_client=None, **kwargs) -> dict:
        broadcaster = get_broadcaster()

        # Append text to current line (no newline)
        broadcaster.append("Processing... ")

        # Do some work...

        broadcaster.append("done")
        broadcaster.newline()

        # Start a new line and append more text
        broadcaster.append("Analyzing cells...")
        broadcaster.newline()

        # Signal completion
        broadcaster.end()

        return {
            "notebook": notebook_content,
            "metadata": {"status": "success"}
        }
```

### Message Types

1. **APPEND**: Add text to the current line without a newline

   ```python
   broadcaster.append("Processing... ")
   ```

2. **NEWLINE**: Start a new line

   ```python
   broadcaster.newline()
   ```

3. **END**: Signal command completion (adds "--- Complete ---" marker)

   ```python
   broadcaster.end()
   ```

4. **CLEAR**: Clear the entire panel
   ```python
   broadcaster.clear()
   ```

## Testing

### 1. Build the Extension

```bash
jlpm build
```

### 2. Restart JupyterLab

```bash
jupyter lab
```

### 3. Test with Example Command

The extension includes an example command that demonstrates the message panel:

1. Open a notebook
2. Click the "Example Message Stream" button in the toolbar
3. Watch the MessagePanel on the right side update in real-time

The example command is defined in `flowbook/server/example_message_command.py`.

## Panel Integration

The panel is automatically added to JupyterLab in `src/index.ts`:

```typescript
const messagePanel = new MessagePanel();
app.shell.add(messagePanel, 'right', { rank: 500 });
```

This adds the panel to the right sidebar with a rank of 500 (controls stacking order).

## Styling

The panel is styled in `style/base.css` with classes:

- `.flowbook-message-panel` - The panel container
- `.flowbook-message-display` - The scrollable message display area
- `.flowbook-message-content` - The pre-formatted text content

The styles use JupyterLab's CSS variables for theming:

- `--jp-layout-color0/1` - Background colors
- `--jp-ui-font-color1` - Text color
- `--jp-code-font-family` - Monospace font
- `--jp-border-color2/3` - Scrollbar colors

## Connection Lifecycle

1. **Initialization**: Panel connects to SSE stream on creation
2. **Receiving Messages**: EventSource streams messages from server
3. **Display Update**: React component re-renders with new messages
4. **Disconnection**: If connection fails, panel schedules reconnect in 5 seconds
5. **Cleanup**: On disposal, closes SSE connection and unmounts React component

## API Reference

### MessagePanel Methods

- `constructor()` - Creates panel and connects to stream
- `clear()` - Manually clear all messages
- `isConnected` - Boolean property indicating connection status
- `dispose()` - Clean up resources and close connection

### MessageBroadcaster Methods

- `append(text: str)` - Append text to current line
- `newline()` - Start a new line
- `end()` - Signal completion
- `clear()` - Clear the panel
- `send_message(message: Message, client_id: Optional[str])` - Send custom message

## Example Output

When running the example command, you'll see:

```
Starting example command...
Processing step 1/5... done
Processing step 2/5... done
Processing step 3/5... done
Processing step 4/5... done
Processing step 5/5... done

Example command completed successfully!
--- Complete ---
```

## Troubleshooting

### Panel Not Showing

1. Check that the extension is built: `jlpm build`
2. Verify JupyterLab was restarted after building
3. Look for the "Ferret Output" tab in the right sidebar
4. Check browser console for errors

### No Messages Appearing

1. Verify SSE connection in browser dev tools (Network tab, filter for "stream")
2. Check server logs for errors
3. Ensure command is using `get_broadcaster()` correctly
4. Test with the example command first

### Connection Issues

If the panel shows disconnection errors:

1. Check that the Jupyter server is running
2. Verify `/flowbook/stream` endpoint is registered
3. Look for CORS or authentication issues in browser console
4. Check firewall/proxy settings that might block SSE
