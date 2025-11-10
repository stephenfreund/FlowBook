# ANSI Color Support in MessagePanel

## Overview

The MessagePanel now supports ANSI color codes from the server output. Instead of stripping colors completely, the system parses ANSI escape sequences and sends color metadata to the client, where it's rendered using CSS variables.

## How It Works

### Server-Side (Python)

1. **ANSI Parsing** (`data_ferret/util/text.py`)
   - `parse_ansi_text(text)` - Extracts ANSI codes and returns stripped text + metadata
   - Recognizes 16 standard ANSI colors (black, red, green, yellow, blue, magenta, cyan, white + bright variants)
   - Detects bold formatting (`\x1B[1m`)

2. **BroadcastStream** (`data_ferret/server/message_broadcaster.py`)
   - `write()` method parses ANSI codes from incoming text
   - Creates `Message` objects with color/bold metadata
   - Sends styled messages to all connected clients

### Client-Side (TypeScript)

1. **Message Structure** (`src/panel.tsx`)
   - Messages include optional `metadata` with `color` and `bold` properties
   - Panel stores message segments with their styling

2. **React Rendering**
   - Each segment rendered as `<span>` with inline styles
   - Colors mapped to CSS variables: `var(--ferret-color-{color})`
   - Bold applied via `fontWeight: 'bold'`

3. **CSS Variables** (`style/base.css`)
   - 16 ANSI colors defined as CSS variables on `.ferret-message-panel`
   - Colors chosen for good visibility in JupyterLab themes

## Supported Colors

### Standard Colors

- **black** (#000000)
- **red** (#cd3131)
- **green** (#0dbc79)
- **yellow** (#e5e510)
- **blue** (#2472c8)
- **magenta** (#bc3fbc)
- **cyan** (#11a8cd)
- **white** (#e5e5e5)

### Bright Colors

- **bright-black** (#666666)
- **bright-red** (#f14c4c)
- **bright-green** (#23d18b)
- **bright-yellow** (#f5f543)
- **bright-blue** (#3b8eea)
- **bright-magenta** (#d670d6)
- **bright-cyan** (#29b8db)
- **bright-white** (#e5e5e5)

### Bold

Bold text is rendered with `font-weight: bold`.

## Usage Examples

### Python (Server)

```python
from data_ferret.util.output import log
import termcolor

# Using termcolor (recommended)
log(termcolor.colored("Processing...", "cyan"))
log(termcolor.colored("Success!", "green"))
log(termcolor.colored("Error!", "red"))
log(termcolor.colored("Warning", "yellow", attrs=["bold"]))

# Manual ANSI codes (if needed)
log("\x1B[31mError message\x1B[0m")  # Red text
log("\x1B[1;32mBold green\x1B[0m")   # Bold green text
```

### Output System Integration

The color support works automatically with the existing output system:

```python
from data_ferret.util.output import log, error, print as out_print
from data_ferret.server.message_broadcaster import get_broadcast_stream
from data_ferret.util.output import stream_output

with stream_output(get_broadcast_stream()):
    # These automatically preserve ANSI colors
    log(termcolor.colored("Starting analysis", "cyan"))
    log(termcolor.colored("Found 10 items", "green"))
    error(termcolor.colored("Failed to process item 5", "red"))
```

### In Commands

```python
from data_ferret.server.base import NotebookCommand
from data_ferret.util.output import log
import termcolor

class MyCommand(NotebookCommand):
    def process(self, notebook_content, kernel_client=None, **kwargs):
        # Colors automatically broadcast to MessagePanel
        log(termcolor.colored("Analyzing notebook...", "cyan"))

        # Do work...

        log(termcolor.colored("✓ Analysis complete", "green"))

        return {"notebook": notebook_content, "metadata": {"status": "success"}}
```

## Message Flow

1. **Command outputs colored text**

   ```python
   log(termcolor.colored("Processing", "cyan"))
   # Produces: "\x1B[36mProcessing\x1B[0m"
   ```

2. **StreamOutputContext receives it**
   - Preserves ANSI codes (doesn't strip them)
   - Passes to BroadcastStream

3. **BroadcastStream parses ANSI**

   ```python
   parse_ansi_text("\x1B[36mProcessing\x1B[0m")
   # Returns: ("Processing", {"color": "cyan"})
   ```

4. **Message created with metadata**

   ```python
   Message(
       type=MessageType.APPEND,
       content="Processing",
       metadata={"color": "cyan"}
   )
   ```

5. **JSON sent via SSE**

   ```json
   {
     "type": "append",
     "content": "Processing",
     "metadata": { "color": "cyan" }
   }
   ```

6. **Panel renders colored span**
   ```tsx
   <span style={{ color: 'var(--ferret-color-cyan)' }}>Processing</span>
   ```

## ANSI Code Mapping

| ANSI Code  | Color Name     | CSS Variable                    |
| ---------- | -------------- | ------------------------------- |
| `\x1B[30m` | black          | `--ferret-color-black`          |
| `\x1B[31m` | red            | `--ferret-color-red`            |
| `\x1B[32m` | green          | `--ferret-color-green`          |
| `\x1B[33m` | yellow         | `--ferret-color-yellow`         |
| `\x1B[34m` | blue           | `--ferret-color-blue`           |
| `\x1B[35m` | magenta        | `--ferret-color-magenta`        |
| `\x1B[36m` | cyan           | `--ferret-color-cyan`           |
| `\x1B[37m` | white          | `--ferret-color-white`          |
| `\x1B[90m` | bright-black   | `--ferret-color-bright-black`   |
| `\x1B[91m` | bright-red     | `--ferret-color-bright-red`     |
| `\x1B[92m` | bright-green   | `--ferret-color-bright-green`   |
| `\x1B[93m` | bright-yellow  | `--ferret-color-bright-yellow`  |
| `\x1B[94m` | bright-blue    | `--ferret-color-bright-blue`    |
| `\x1B[95m` | bright-magenta | `--ferret-color-bright-magenta` |
| `\x1B[96m` | bright-cyan    | `--ferret-color-bright-cyan`    |
| `\x1B[97m` | bright-white   | `--ferret-color-bright-white`   |
| `\x1B[1m`  | bold           | `fontWeight: bold`              |
| `\x1B[0m`  | reset          | (clears styling)                |

## Implementation Details

### Segment Merging

The panel intelligently merges consecutive segments with the same styling:

```python
# Input:
broadcaster.append("Hello ")  # cyan
broadcaster.append("World")   # cyan

# Result: Single segment
{content: "Hello World", color: "cyan"}

# But different colors create separate segments:
broadcaster.append("Hello ")  # cyan
broadcaster.append("World")   # red

# Result: Two segments
[
  {content: "Hello ", color: "cyan"},
  {content: "World", color: "red"}
]
```

### Multiple ANSI Codes

If text contains multiple ANSI codes, only the **first** color/style is extracted:

```python
text = "\x1B[31m\x1B[1mBold Red\x1B[0m"
# Extracts: color='red', bold=True
```

### Reset Codes

The reset code `\x1B[0m` is recognized and returns `None` for metadata:

```python
parse_ansi_text("\x1B[0mNormal text")
# Returns: ("Normal text", None)
```

## Testing

Run the test suite to verify ANSI color handling:

```bash
python test_stream_output.py
```

**Test coverage:**

- Test 1: Basic BroadcastStream write/flush
- Test 2: StreamOutputContext integration
- Test 3: Multiline text handling
- Test 4: ANSI color parsing and metadata transmission

All tests verify that:

- ANSI codes are correctly parsed
- Color metadata is extracted
- Messages are properly formatted
- Colors are preserved through the entire pipeline

## Customization

### Changing Colors

Edit the CSS variables in `style/base.css`:

```css
.ferret-message-panel {
  --ferret-color-red: #ff0000; /* Bright red */
  --ferret-color-green: #00ff00; /* Bright green */
  /* ... etc ... */
}
```

### Adding New Colors

1. Add to ANSI_COLORS mapping in `data_ferret/util/text.py`:

   ```python
   ANSI_COLORS = {
       # ...
       '38;5;208': 'orange',  # 256-color code
   }
   ```

2. Add CSS variable in `style/base.css`:

   ```css
   .ferret-message-panel {
     --ferret-color-orange: #ffa500;
   }
   ```

3. Rebuild: `jlpm build`

## Limitations

1. **Single style per segment**: Currently extracts only the first ANSI code found
2. **No background colors**: Only foreground colors supported
3. **No underline/italic**: Only color and bold supported
4. **Standard colors only**: 256-color and RGB ANSI codes not yet supported

## Future Enhancements

Possible improvements:

1. **Full ANSI support**: 256-color palette, RGB colors
2. **Background colors**: Support for `\x1B[4Xm` codes
3. **Text decorations**: Underline, italic, strikethrough
4. **Multiple styles**: Apply multiple ANSI codes to single segment
5. **Theme integration**: Adapt colors to JupyterLab light/dark theme

## Troubleshooting

### Colors not appearing

1. Check that output contains ANSI codes:

   ```python
   import termcolor
   text = termcolor.colored("Test", "red")
   print(repr(text))  # Should show: '\x1B[31mTest\x1B[0m'
   ```

2. Verify termcolor is installed:

   ```bash
   pip install termcolor
   ```

3. Check browser console for errors

4. Verify CSS variables are defined (inspect element in browser dev tools)

### Wrong colors

1. Check ANSI code mapping in `data_ferret/util/text.py`
2. Verify CSS variable values in browser dev tools
3. Ensure CSS variables are scoped to `.ferret-message-panel`

### Colors stripped instead of displayed

1. Verify BroadcastStream is being used (not direct append() calls)
2. Check that StreamOutputContext is preserving ANSI codes
3. Ensure parse_ansi_text() is returning correct metadata
