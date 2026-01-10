# Message Panel Usage Guide

The FlowBook extension now includes a real-time message streaming system that allows server-side commands to send messages to a client-side panel displayed on the right side of JupyterLab.

## Overview

The messaging system consists of three main components:

1. **Server-side Message Broadcaster** - Manages message queues and broadcasts to connected clients
2. **SSE (Server-Sent Events) Handler** - Streams messages from server to client over HTTP
3. **Client-side Message Panel** - Displays messages in a right-side panel in JupyterLab

## Architecture

### Server Side

- **`message_broadcaster.py`** - Contains the `MessageBroadcaster` singleton class
- **`handlers.py`** - Includes the `MessageStreamHandler` for SSE streaming
- Messages are sent asynchronously and queued per client

### Client Side

- **`panel.ts`** - Contains the `MessagePanel` widget
- Connects to `/flowbook/stream` endpoint via EventSource API
- Automatically reconnects on connection loss
- Auto-scrolls to show latest messages

## Using the Message Broadcaster

### Basic Usage

In any command that extends `NotebookCommand`, you can send messages to the client panel:

```python
from flowbook.server.message_broadcaster import get_broadcaster

class MyCommand(NotebookCommand):
    def process(self, notebook_content, kernel_client=None, **kwargs):
        broadcaster = get_broadcaster()

        # Append text to current line
        broadcaster.append("Processing notebook... ")

        # Do some work...

        # Append more text to the same line
        broadcaster.append("done")

        # Start a new line
        broadcaster.newline()

        # More messages
        broadcaster.append("Analyzing cells...")
        broadcaster.newline()

        # Signal completion
        broadcaster.end()

        return {"notebook": notebook_content, "metadata": {...}}
```

### Available Methods

#### `broadcaster.append(text: str, client_id: Optional[str] = None)`
Appends text to the current line in the panel.

```python
broadcaster.append("Loading data")
broadcaster.append(".")
broadcaster.append(".")
broadcaster.append(".")
# Displays: "Loading data..."
```

#### `broadcaster.newline(client_id: Optional[str] = None)`
Starts a new line in the panel.

```python
broadcaster.append("First line")
broadcaster.newline()
broadcaster.append("Second line")
# Displays:
# First line
# Second line
```

#### `broadcaster.end(client_id: Optional[str] = None)`
Signals that the message/command is complete. This also starts a new line.

```python
broadcaster.append("Command completed!")
broadcaster.end()
```

#### `broadcaster.clear(client_id: Optional[str] = None)`
Clears all content from the panel.

```python
broadcaster.clear()  # Panel is now empty
```

### Broadcasting vs. Targeted Messages

By default, messages are broadcast to all connected clients. You can send messages to a specific client by providing their `client_id`:

```python
# Broadcast to all clients
broadcaster.append("Global message")

# Send to specific client only
broadcaster.append("Private message", client_id="some-client-id")
```

Note: Client IDs are automatically generated and sent when clients connect to the stream.

### Advanced Usage with Message Objects

For more control, you can send custom `Message` objects:

```python
from flowbook.server.message_broadcaster import (
    get_broadcaster,
    Message,
    MessageType
)

broadcaster = get_broadcaster()

# Send a custom message with metadata
msg = Message(
    type=MessageType.APPEND,
    content="Processing...",
    metadata={"progress": 50, "step": "validation"}
)
broadcaster.send_message(msg)
```

### Message Types

- `MessageType.APPEND` - Append text to current line
- `MessageType.NEWLINE` - Start a new line
- `MessageType.END` - Signal completion (also starts new line)
- `MessageType.CLEAR` - Clear the panel
- `MessageType.CONNECTED` - Internal: Sent when client connects

## Example Command

See `flowbook/server/example_message_command.py` for a complete working example:

```python
from flowbook.server.base import NotebookCommand
from flowbook.server.message_broadcaster import get_broadcaster

class ExampleMessageCommand(NotebookCommand):
    @property
    def command_name(self) -> str:
        return "example_message"

    @property
    def display_name(self) -> str:
        return "Example Message Stream"

    @property
    def requires_kernel(self) -> bool:
        return False

    def process(self, notebook_content, kernel_client=None, **kwargs):
        broadcaster = get_broadcaster()

        broadcaster.append("Starting command...")
        broadcaster.newline()

        for i in range(5):
            broadcaster.append(f"Step {i+1}/5... ")
            # Do work here
            broadcaster.append("done")
            broadcaster.newline()

        broadcaster.append("Complete!")
        broadcaster.end()

        return {"notebook": notebook_content, "metadata": {"status": "success"}}
```

## Client Panel

The message panel is automatically added to the right sidebar when the extension loads. Users can:

- **Open/Close** the panel via the right sidebar
- **Resize** the panel by dragging the divider
- **Auto-scroll** - The panel automatically scrolls to show new messages
- **View history** - All messages remain visible until cleared

### Panel Features

- Monospace font for better readability
- Auto-reconnection if connection is lost
- Keepalive messages every 30 seconds to maintain connection
- Clean, minimalist design matching JupyterLab theme

## Technical Details

### Server-Sent Events (SSE)

The system uses SSE instead of WebSockets because:
- Simpler implementation (HTTP-based)
- One-directional (server to client) is sufficient
- Automatic reconnection built into EventSource API
- Works through most proxies and firewalls

### Message Queue

- Each connected client has their own message queue
- Queues are managed by the singleton `MessageBroadcaster`
- Messages are non-blocking (uses `put_nowait`)
- Clients are automatically unregistered on disconnect

### Connection Management

- Clients connect to `/flowbook/stream`
- Connection authenticated via Jupyter's authentication
- Keepalive comments sent every 30 seconds
- Automatic reconnection after 5 seconds on error

## Troubleshooting

### Messages not appearing

1. Check that the message panel is visible in the right sidebar
2. Check browser console for connection errors
3. Verify the `/flowbook/stream` endpoint is accessible
4. Check server logs for SSE handler errors

### Connection issues

- The panel will automatically attempt to reconnect after 5 seconds
- Check network tab in browser dev tools for failed SSE connections
- Verify Jupyter server is running and accessible

### Performance

- Messages are queued and sent asynchronously
- If queue fills up, new messages are dropped (not blocked)
- Consider batching messages if sending many updates rapidly

## Best Practices

1. **Use descriptive messages** - Help users understand what's happening
2. **Signal completion** - Always call `broadcaster.end()` when done
3. **Handle errors** - Wrap broadcaster calls in try/except if needed
4. **Don't spam** - Avoid sending hundreds of messages per second
5. **Clear when appropriate** - Use `broadcaster.clear()` to start fresh

## Future Enhancements

Potential improvements:
- Message filtering/search
- Export message history
- Color-coded message types
- Progress bars
- Collapsible message groups
