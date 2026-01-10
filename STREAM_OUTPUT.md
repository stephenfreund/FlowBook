# StreamOutputContext Implementation

## Overview

The `StreamOutputContext` captures all output from the FlowBook output system and broadcasts it to connected clients in real-time through the MessagePanel.

## Components

### 1. StreamOutputContext (`flowbook/util/output.py`)

A new output context class that redirects output to any file-like object with `write()` and `flush()` methods.

**Key Features:**

- Integrates with the existing output system's context chain
- Strips ANSI color codes before sending to stream
- Automatically flushes on every write
- Thread-safe using the existing output lock

**Usage:**

```python
from flowbook.util.output import stream_output

# Any file-like object with write() and flush()
with stream_output(my_stream):
    log("This will be streamed")
    print("This too!")
```

### 2. BroadcastStream (`flowbook/server/message_broadcaster.py`)

A file-like wrapper around `MessageBroadcaster` that provides the standard `write()/flush()` interface.

**Key Features:**

- Converts write() calls to append()/newline() messages
- Handles newlines intelligently by splitting text
- Buffers current line to track state
- Can be used as a context manager

**Example:**

```python
from flowbook.server.message_broadcaster import get_broadcast_stream

stream = get_broadcast_stream()
stream.write("Hello")
stream.write(" World")
stream.write("\n")  # Sends: APPEND "Hello", APPEND " World", NEWLINE
```

### 3. Handler Integration (`flowbook/server/handlers.py`)

The `FlowbookCommandHandler` now wraps command execution with `stream_output()`:

```python
# Execute command with output streaming to clients
with stream_output(get_broadcast_stream()):
    result = command.process(
        notebook_content, kernel_client=kernel_client, **params
    )
```

This means **all output from commands is automatically broadcast to the MessagePanel**, including:

- `log()` statements
- `print()` statements
- `error()` statements
- Timer contexts
- Any other output generated during command execution

## How It Works

### Flow

1. **Command Execution Starts**
   - Handler creates a `BroadcastStream` wrapper around the global broadcaster
   - Handler enters `stream_output()` context with the stream
   - `StreamOutputContext` adds itself to the output contexts chain

2. **Command Generates Output**
   - Command calls `log("Processing...")` or similar
   - Output system writes to stdout AND all registered contexts
   - `StreamOutputContext.write()` receives the text
   - Text is stripped of ANSI codes
   - Text is forwarded to `BroadcastStream.write()`

3. **BroadcastStream Processes Text**
   - Splits text by newlines
   - Sends `APPEND` messages for text content
   - Sends `NEWLINE` messages for line breaks
   - Messages go to the global `MessageBroadcaster`

4. **MessageBroadcaster Distributes**
   - Queues messages for all connected SSE clients
   - `MessageStreamHandler` sends messages via Server-Sent Events
   - MessagePanel receives and displays messages in real-time

5. **Command Execution Completes**
   - `StreamOutputContext` exits and flushes remaining buffer
   - Context removes itself from the chain
   - Normal execution continues

### Example Output Flow

```python
# Command code:
with stream_output(get_broadcast_stream()):
    log("Starting analysis")
    log("Found 10 cells")
    print("Processing complete")

# Messages sent to clients:
APPEND: "[06:24:40] [Starting analysis]"
NEWLINE
APPEND: "[06:24:40] [Found 10 cells]"
NEWLINE
APPEND: "[06:24:40] Processing complete"
NEWLINE
```

## Testing

A test file `test_stream_output.py` validates the implementation:

```bash
python test_stream_output.py
```

**Tests:**

1. Direct BroadcastStream write/flush operations
2. StreamOutputContext integration with output system
3. Multiline text handling

All tests pass ✓

## Benefits

### For Command Authors

**Before:**

```python
def process(self, notebook_content, kernel_client=None, **kwargs):
    broadcaster = get_broadcaster()
    broadcaster.append("Processing... ")
    # Do work
    broadcaster.append("done")
    broadcaster.newline()
    return result
```

**After:**

```python
def process(self, notebook_content, kernel_client=None, **kwargs):
    log("Processing...")
    # Do work
    log("Done")
    return result
```

Commands now use familiar `log()`, `print()`, `error()` functions instead of manually managing broadcaster calls. Output automatically appears in the MessagePanel.

### For Users

- Real-time feedback for all command operations
- Consistent output formatting
- Timestamped messages from the timer contexts
- Color-free output (ANSI codes stripped)

## API Reference

### stream_output(stream)

Creates an output context that redirects to a file-like stream.

**Parameters:**

- `stream`: Object with `write(text: str)` and `flush()` methods

**Returns:**

- `StreamOutputContext` instance

**Example:**

```python
from flowbook.util.output import stream_output, log

class MyStream:
    def write(self, text):
        print(f"STREAM: {text}")

    def flush(self):
        pass

with stream_output(MyStream()):
    log("This goes to MyStream")
```

### BroadcastStream(broadcaster=None)

File-like wrapper for MessageBroadcaster.

**Parameters:**

- `broadcaster`: Optional `MessageBroadcaster` instance (defaults to global)

**Methods:**

- `write(text: str) -> int`: Write text, handling newlines
- `flush()`: No-op (messages sent immediately)

**Example:**

```python
from flowbook.server.message_broadcaster import BroadcastStream

stream = BroadcastStream()
stream.write("Hello\nWorld\n")
# Sends: APPEND "Hello", NEWLINE, APPEND "World", NEWLINE
```

### get_broadcast_stream()

Convenience function to get a BroadcastStream connected to the global broadcaster.

**Returns:**

- `BroadcastStream` instance

**Example:**

```python
from flowbook.server.message_broadcaster import get_broadcast_stream
from flowbook.util.output import stream_output, log

with stream_output(get_broadcast_stream()):
    log("Broadcast to all clients")
```

## Implementation Details

### ANSI Code Stripping

All output is stripped of ANSI color codes before broadcasting to ensure clean display in the MessagePanel:

```python
message = strip_ansi(message)  # Removes terminal color codes
```

### Newline Handling

The `BroadcastStream` intelligently splits text by newlines:

```python
text = "Line 1\nLine 2\nLine 3"
# Results in:
# APPEND "Line 1"
# NEWLINE
# APPEND "Line 2"
# NEWLINE
# APPEND "Line 3"
```

### Thread Safety

The `StreamOutputContext` uses the existing output system's lock to ensure thread-safe operation:

```python
with self.outer.lock:
    self.outer.output_contexts.append(self)
```

## Future Enhancements

Possible improvements:

1. **Buffering**: Add optional buffering to reduce message frequency
2. **Filtering**: Allow filtering by log level or message type
3. **Formatting**: Support for structured metadata in messages
4. **Per-client streams**: Direct output to specific clients
5. **Error highlighting**: Special formatting for error messages in the panel

## Troubleshooting

### Output Not Appearing in Panel

1. Verify MessagePanel is connected (check browser console)
2. Ensure command is using output functions (`log`, `print`, etc.)
3. Check SSE stream in browser dev tools (Network tab)

### Duplicate Messages

If you see duplicate messages:

- Don't manually call broadcaster methods AND use output functions
- Choose one approach: either use output functions (recommended) or broadcaster directly

### ANSI Codes in Output

If color codes appear:

- Check that `strip_ansi()` is being called
- Verify the text utility is imported correctly
