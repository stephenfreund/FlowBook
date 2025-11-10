# Using the CLI with Kernel Connection Files

The DataFerret CLI automatically detects whether you're providing a notebook file or a kernel connection file.

## Overview

You can run commands on notebooks using either:
1. A new kernel (started automatically)
2. An existing kernel (via connection file)

The CLI intelligently detects which files you provide based on their names and contents.

## Usage

### Option 1: Notebook Only (Starts New Kernel)

```bash
python -m data_ferret.server.cli <command> <notebook.ipynb>
```

This will:
- Start a new ferret_kernel
- Execute the command
- Shutdown the kernel when done

### Option 2: Notebook + Connection File (Use Existing Kernel)

```bash
# Order doesn't matter - CLI auto-detects which is which
python -m data_ferret.server.cli <command> notebook.ipynb kernel-abc123.json
# or
python -m data_ferret.server.cli <command> kernel-abc123.json notebook.ipynb
```

This will:
- Connect to the running kernel specified in the connection file
- Execute the command on the notebook
- Leave the kernel running when done

## Getting the Connection File Path

### From JupyterLab UI

1. Open your notebook in JupyterLab
2. Make sure a kernel is running
3. Click the "Copy Connection" button in the notebook toolbar
4. Paste the path into your command

### Example Workflow

```bash
# 1. In JupyterLab, click "Copy Connection" button
#    (copies something like: /Users/user/Library/Jupyter/runtime/kernel-abc123.json)

# 2. Run CLI command with both files (order doesn't matter)
python -m data_ferret.server.cli inspect \
  examples/Example.ipynb \
  /Users/user/Library/Jupyter/runtime/kernel-abc123.json
```

## How Auto-Detection Works

The CLI detects file types by:
1. **Extension**: `.ipynb` files are notebooks
2. **Naming pattern**: `kernel-*.json` files are connection files
3. **Content**: Checks JSON structure (connection files have transport/ip/ports, notebooks have cells)

## Arguments

**Positional:**
- `command` - Required. The ferret command to execute (e.g., inspect, optimize, validate)
- `paths` - Required. One or more paths (notebook and/or connection file)

**Options:**
- `--kernel-name` - Optional. Kernel name if starting new kernel (default: ferret_kernel)
- `--output`, `-o` - Optional. Output file path (default: adds _processed suffix)
- `--model` - Optional. AI model to use (default: gpt-4o)
- `--fast-model` - Optional. Fast AI model (default: gpt-4o-mini)
- `--cell-ids` - Optional. Specific cell IDs to process. Supports `#N` notation for 1-based code cell indices (e.g., `#1 #3`) or actual cell ID UUIDs. Can mix both formats.

## Notes

- You must provide at least a notebook file
- Connection file is optional - if not provided, a new kernel is started
- When using a connection file, the `--kernel-name` option is ignored
- The CLI will NOT shutdown the kernel when using a connection file
- This allows you to run multiple commands on the same kernel instance
- The connection file must be valid JSON and accessible from the CLI process

## Cell ID Notation

The `--cell-ids` option supports two formats:

### 1. Index Notation (`#N`)
Use `#N` where N is a 1-based index of code cells only (markdown cells are skipped).

```bash
# Process first code cell
python -m data_ferret.server.cli inspect notebook.ipynb --cell-ids #1

# Process first and third code cells
python -m data_ferret.server.cli validate notebook.ipynb --cell-ids #1 #3

# Process cells 2 through 4
python -m data_ferret.server.cli optimize notebook.ipynb --cell-ids #2 #3 #4
```

**Important:** The index refers to code cells only. If your notebook has:
- Cell 0: Markdown
- Cell 1: Code (this is #1)
- Cell 2: Markdown
- Cell 3: Code (this is #2)
- Cell 4: Code (this is #3)

### 2. UUID Notation
Use the actual cell ID (UUID) from the notebook JSON.

```bash
python -m data_ferret.server.cli inspect notebook.ipynb \
  --cell-ids 6268e917-94eb-4f4f-9f09-3c988ae84e96
```

### 3. Mixed Notation
You can mix both formats in the same command.

```bash
python -m data_ferret.server.cli validate notebook.ipynb \
  --cell-ids #1 6268e917-94eb-4f4f-9f09-3c988ae84e96 #3
```

## Examples

```bash
# Just notebook (starts new kernel)
python -m data_ferret.server.cli inspect notebook.ipynb

# Notebook + connection file (uses existing kernel)
python -m data_ferret.server.cli inspect notebook.ipynb kernel-abc.json

# Connection file first (order doesn't matter)
python -m data_ferret.server.cli inspect kernel-abc.json notebook.ipynb

# With output file
python -m data_ferret.server.cli optimize notebook.ipynb kernel-abc.json -o optimized.ipynb

# With specific cells using index notation
python -m data_ferret.server.cli validate notebook.ipynb --cell-ids #1 #2

# With specific cells using UUID notation
python -m data_ferret.server.cli validate notebook.ipynb --cell-ids cell-id-1 cell-id-2

# Mixed cell ID formats
python -m data_ferret.server.cli inspect notebook.ipynb \
  --cell-ids #1 actual-uuid-here #3
```
