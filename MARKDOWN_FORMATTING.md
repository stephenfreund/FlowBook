# Markdown Formatting for DiffResult

## Overview

The `format_diff_as_markdown()` function converts a `DiffResult` into a human-readable markdown list, making it easy to present differences in documentation, reports, or user interfaces.

## Usage

```python
from flowbook.kernel.diff import Diff
from flowbook.kernel.types import format_diff_as_markdown

# Create a diff
differ = Diff()
result = differ.diff(
    {'x': 1, 'config': {'timeout': 30}},
    {'x': 2, 'config': {'timeout': 60}}
)

# Format as markdown
markdown = format_diff_as_markdown(result)
print(markdown)
```

**Output:**
```markdown
## Differences Found

- **config['timeout']**: Integer mismatch at config['timeout']: 30 vs 60
- **x**: Integer mismatch at x: 1 vs 2
```

## Features

### 1. **Simple Variable Differences**
Variables are shown with their full path and description:
```markdown
- **variable_name**: Description of the difference
```

### 2. **Close Float Indicator**
Floats within tolerance get a special `(close)` marker:
```markdown
- **pi** *(close)*: Float close at pi: 3.14159265358979 vs 3.14159265358980 (within tolerance)
```

### 3. **Nested Structure Support**
Full paths are shown for nested differences:
```markdown
- **config['timeout']**: Integer mismatch at config['timeout']: 30 vs 60
- **user.address['city']**: String mismatch: 'NYC' vs 'Boston'
- **data.items[0].value**: Integer mismatch: 100 vs 200
```

### 4. **List Indexing**
List differences show the specific index:
```markdown
- **scores[1]**: Integer mismatch at scores[1]: 90 vs 92
- **items[3]**: Integer mismatch at items[3]: 4 vs 88
```

### 5. **Alphabetical Sorting**
Variables are automatically sorted alphabetically for easy scanning.

### 6. **Truncation Messages**
When hitting `max_diffs_per_container`, a truncation message appears:
```markdown
- **values[0]**: Integer mismatch at values[0]: 0 vs 10
- **values[1]**: Integer mismatch at values[1]: 1 vs 11
  - *values: Truncated after 5 differences (max_diffs_per_container=5)*
```

### 7. **Empty Diff Handling**
When there are no differences:
```markdown
## No Differences Found

All variables are equal.
```

## API Reference

### `format_diff_as_markdown(diff_result: DiffResult) -> str`

Convert a DiffResult to a human-readable markdown list.

**Parameters:**
- `diff_result`: A DiffResult object (Dict[str, DiffNode])

**Returns:**
- A markdown-formatted string with:
  - A heading ("## Differences Found" or "## No Differences Found")
  - Bulleted list of all differences
  - Full paths for nested structures
  - Status indicators for close floats
  - Truncation messages when applicable

**Example:**
```python
from flowbook.kernel.types import format_diff_as_markdown

markdown = format_diff_as_markdown(result)
```

## Use Cases

1. **Notebook Output**: Display differences in Jupyter notebook cells
2. **Test Reports**: Show what changed between test runs
3. **Documentation**: Include diff summaries in generated docs
4. **CLI Output**: Pretty-print differences for command-line tools
5. **Logging**: Human-readable diff logs for debugging

## Demo

Run the demo script to see all features in action:
```bash
python demo_markdown.py
```

See `test_markdown_format.py` for additional examples and test cases.
