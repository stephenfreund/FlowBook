# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataFerret is a JupyterLab 4.0+ extension that combines a TypeScript frontend with a Python server extension and a custom IPython kernel. The extension provides notebook analysis, validation, execution, and AI-powered capabilities through a command-based architecture.

## Development Commands

### Initial Setup

```bash
# Install package in development mode
pip install -e "."

# Link development version with JupyterLab
jupyter labextension develop . --overwrite

# Enable server extension
jupyter server extension enable data_ferret
```

### Building

```bash
# Development build (with source maps)
jlpm build

# Production build
jlpm build:prod

# Clean build artifacts
jlpm clean:all
```

### Development Workflow

```bash
# Terminal 1: Auto-rebuild TypeScript on changes
jlpm watch

# Terminal 2: Run JupyterLab
jupyter lab
```

After making changes, refresh JupyterLab in the browser. The `jlpm watch` command automatically rebuilds TypeScript and the labextension.

### Linting

```bash
# Run all linters with auto-fix
jlpm lint

# Individual linters
jlpm eslint          # TypeScript/JavaScript
jlpm prettier        # Code formatting
jlpm stylelint       # CSS

# Check without fixing
jlpm lint:check
```

### Testing

```bash
# Run Python tests
pytest data_ferret/

# Run specific test file
pytest data_ferret/kernel/test_diff.py
```

### Verification

```bash
# Check server extension is enabled
jupyter server extension list

# Check frontend extension is installed
jupyter labextension list
```

## Architecture

### Three-Tier Structure

1. **Frontend (TypeScript)**: `src/` - JupyterLab UI components with two kernel-specific plugins
2. **Server Extension (Python)**: `data_ferret/server/` - HTTP handlers and command processing
3. **Custom Kernels**: Two enhanced IPython kernels for different use cases
   - `data_ferret/kernel/` - Full-featured kernel with AI commands, profiling, checkpointing
   - `data_ferret/sdc_kernel/` - SDC-focused kernel with always-on dataflow tracking

### Frontend Components (`src/`)

The frontend exports **two JupyterLab plugins** that activate based on the kernel in use:

```
src/
├── index.ts                 # Exports [ferretPlugin, sdcPlugin]
├── shared/                  # Shared utilities
│   ├── kerneldetection.ts   # KernelDetector class for kernel type detection
│   └── types.ts             # Shared type definitions
├── ferret/                  # Ferret kernel plugin (AI commands)
│   ├── plugin.ts            # Plugin activation with kernel gating
│   ├── manager.ts           # FerretCommandsManager orchestrates commands
│   ├── types.ts             # TypeScript interfaces
│   ├── toolbar.ts           # Notebook toolbar buttons
│   ├── celltoolbar.ts       # Cell-level toolbar buttons
│   ├── metadatapanel.tsx    # Metadata panel (profile, dependencies, etc.)
│   ├── cellhighlighter.ts   # Visual indicators for optimization potential
│   ├── executionhook.ts     # Auto-generation and metadata extraction
│   ├── history.ts           # Undo/redo history manager
│   ├── historypanel.tsx     # History panel UI
│   ├── unittestpanel.tsx    # Unit test panel
│   └── unittesttracker.ts   # Unit test cell tracking
├── sdc/                     # SDC kernel plugin (staleness tracking)
│   ├── plugin.ts            # Plugin activation with kernel gating
│   ├── types.ts             # ISDCMetadata, ISDCViolation interfaces
│   ├── stalenessmanager.ts  # Tracks stale cells per notebook
│   ├── metadatapanel.tsx    # SDC metadata panel (reads, writes, stale cells)
│   ├── cellhighlighter.ts   # Red highlighting for stale cells
│   └── executionhook.ts     # Extract ferret_sdc metadata from outputs
├── api.ts                   # Shared FerretAPI for HTTP communication
├── kernel.ts                # Shared KernelUtils
└── [other shared files]     # panel.tsx, executiondialog.tsx, etc.
```

**Plugin Activation**:
- `data_ferret:plugin` - Activates UI only when kernel is `ferret_kernel`
- `data_ferret:sdc` - Activates UI only when kernel is `ferret_sdc_kernel`

**Data Flow**: User clicks button → `executeCommand()` → `FerretAPI.executeCommand()` → POST to `/ferret/execute` → Backend processes → Notebook updated

### Server Extension (`data_ferret/server/`)

The server uses the modern **ExtensionApp** pattern (not legacy extension points).

- `__init__.py` - `DataFerretExtension(ExtensionApp)` class with `initialize_handlers()` method
- `handlers.py` - HTTP request handlers:
  - `POST /ferret/execute` - Execute a command (FerretCommandHandler)
  - `GET /ferret/list` - List available commands (CommandListHandler)
- `base.py` - `NotebookCommand` abstract base class
- `registry.py` - `CommandRegistry` singleton managing available commands
- `commands.py` - Built-in command implementations:
  - `AnalyzeNotebookCommand` - Notebook structure analysis
  - `ValidateNotebookCommand` - Syntax validation
  - `ExecuteAllCommand` - Run all cells (requires kernel)
  - `InspectVariablesCommand` - Kernel namespace inspection
- `kernel_manager.py` - `KernelConnectionManager` and `FerretKernelClient` for kernel communication
- `cli.py` - Command-line interface entry point

### Ferret Kernel (`data_ferret/kernel/`)

Full-featured kernel extending IPython with advanced features:

- `ferret_kernel.py` - Main kernel implementation
- `ferret_client.py` - Enhanced `BlockingKernelClient` that includes `cell_id` and `cell_metadata` in execution messages
- `checkpoint.py` - State snapshots (save/restore kernel state)
- `diff.py` - Namespace diffing to track variable changes between executions
- `equality.py` - Deep equality checking for various Python types
- `tracking.py` - `TrackingDict` for optional variable access tracking
- `magics.py` - IPython magic commands (`%enable_scalene`, `%checkpoint`, etc.)
- `ferret_pdb.py` - Debugger integration

**Features** (all optional, toggled via magic commands):
- Scalene profiling for CPU/memory analysis
- Checkpointing for save/restore kernel state
- Variable tracking for read-before-write analysis
- Monotonicity enforcement

### SDC Kernel (`data_ferret/sdc_kernel/`)

Simplified kernel focused on Sequential Dataflow Consistency (SDC):

- `ferret_sdc_kernel.py` - SDC-focused kernel implementation
- `ferret_sdc_client.py` - Client with `cell_order` injection for SDC checks
- `sdc_enforcer.py` - Implements SDC rules (staleness propagation, backward mutation detection)
- `models.py` - `SDCMetadata`, `SDCViolation`, `SDCResult` data classes

**Features** (always enabled):
- Variable tracking for all executions
- Staleness computation (which cells need re-execution)
- Backward mutation detection (Rule 3 violations)
- Automatic rollback on SDC violations

**SDC Metadata Format** (sent via `display_data` output):
```python
{
  "ferret_sdc": {
    "cell_id": str,
    "execution_seq": int,
    "reads": List[str],
    "writes": List[str],
    "changed_variables": List[str],
    "stale_cells": List[str],
    "violation": Optional[dict],
    "cell_order": List[str]
  }
}
```

**Key Feature**: Both `FerretKernelClient` and `FerretSDCKernelClient` inject `cell_id` and metadata into kernel messages, enabling cell-level tracking.

### Command Pattern

Commands follow a registry pattern:

```python
class SomeCommand(NotebookCommand):
    @property
    def command_name(self) -> str:
        return "command_id"

    @property
    def display_name(self) -> str:
        return "Human Readable Name"

    @property
    def requires_kernel(self) -> bool:
        return True  # If kernel communication needed

    def process(self, notebook_content: dict, kernel_client=None, **kwargs) -> dict:
        # Process notebook, optionally execute code via kernel_client
        return {"notebook": modified_notebook, "metadata": {...}}
```

Register in `data_ferret/server/__init__.py` or `commands.py`.

### Agent Integration (`data_ferret/agent/`)

- `agent.py` - `FerretAgent` uses `openai-agents` framework with litellm backend
- `llm_cost.py` - Tracks API costs for LLM usage
- Configured with OpenAI API for AI-powered notebook analysis

## Cell ID Normalization

All notebooks entering the system (via CLI or server) are automatically normalized to ensure consistent cell identification:

- **4-character lowercase IDs**: All cells receive unique 4-character lowercase letter IDs (e.g., "abcd", "xyzw")
- **Automatic ID generation**: Cells without IDs are assigned new unique IDs
- **ID replacement**: Non-4-character IDs (like UUIDs or custom IDs) are replaced with new 4-character IDs
- **Duplicate handling**: Duplicate IDs are automatically regenerated to ensure uniqueness
- **Source normalization**: Cell sources are converted from list to string format

This normalization happens transparently at entry points:
- **CLI**: `load_notebook()` in `data_ferret/cli/helpers.py`
- **Server**: `FerretCommandHandler.post()` in `data_ferret/server/handlers.py`
- **Core function**: `normalize_notebook()` in `data_ferret/util/cell_ids.py`

### Why 4-character IDs?

- **Readability**: Short IDs are easy to read in logs and debugging (26^4 = 456,976 possible IDs)
- **Consistency**: All notebooks use the same ID format regardless of source
- **Simplicity**: Easier to reference and track cells in development and testing

## Code Style

### TypeScript

- **Interfaces**: Must start with `I` and use PascalCase (e.g., `ICommandInfo`)
- **Quotes**: Single quotes, avoid template literals unless necessary
- **Equality**: Use strict equality (`===`)
- **Callbacks**: Prefer arrow functions
- **Curly braces**: Always use for control structures

### Python

- Follow standard Python conventions
- Use type hints where applicable
- Abstract base classes for extensibility (e.g., `NotebookCommand`)

## Important Files

### Configuration

- `pyproject.toml` - Python package config, dependencies, build system (hatchling)
- `package.json` - NPM package config, scripts, linting rules
- `tsconfig.json` - TypeScript compiler settings (ES2020, strict mode)
- `jupyter-config/server-config/data_ferret.json` - Jupyter server extension registration

### Build Artifacts

- `lib/` - Compiled TypeScript output (gitignored)
- `data_ferret/labextension/` - JupyterLab extension bundle (auto-generated)
- `data_ferret/_version.py` - Auto-generated from package.json version

## Extension Points

### Adding a New Command

1. Create class in `data_ferret/server/commands.py` inheriting `NotebookCommand`
2. Implement required properties: `command_name`, `display_name`, `icon_name`, `requires_kernel`
3. Implement `process()` method
4. Register in `CommandRegistry` (typically auto-registered via import)
5. Frontend automatically discovers via `GET /ferret/list`

### Modifying Kernel Behavior

**Ferret Kernel**:
- Kernel spec: `data_ferret/kernel/kernelspec/`
- Main kernel class: `data_ferret/kernel/ferret_kernel.py`

**SDC Kernel**:
- Kernel spec: `data_ferret/sdc_kernel/kernelspec/`
- Main kernel class: `data_ferret/sdc_kernel/ferret_sdc_kernel.py`
- SDC logic: `data_ferret/sdc_kernel/sdc_enforcer.py`

**Frontend**:
- Shared kernel utilities: `src/kernel.ts`
- Kernel detection: `src/shared/kerneldetection.ts`

## Dependencies

### Python (Key)

- `jupyter_server>=2.4.0` - Server extension base
- `jupyterlab>=4.0.0` - Lab integration
- `openai` + `openai-agents[litellm]` - AI capabilities
- `scalene` - Memory profiling (custom git version)
- Data science stack: `pandas`, `numpy`, `scikit-learn`, `scipy`, `seaborn`, `matplotlib`
- Testing: `pytest`, `hypothesis`
- LSP: `jedi-language-server`, `python-lsp-jsonrpc`

### TypeScript (Key)

- `@jupyterlab/application`, `@jupyterlab/notebook`, `@jupyterlab/cells` - JupyterLab APIs
- `@jupyterlab/services` - Kernel and server communication
- Build: `@jupyterlab/builder`, TypeScript ~5.4
- Linting: ESLint, Prettier, Stylelint

## Troubleshooting

### Extension Not Loading

```bash
# Check server extension
jupyter server extension list
# Should show "data_ferret" as enabled

# Check frontend extension
jupyter labextension list
# Should show "data_ferret" in enabled extensions
```

### Build Issues

```bash
# Clean everything and rebuild
jlpm clean:all
jlpm build
pip install -e "." --force-reinstall
jupyter labextension develop . --overwrite
```

### Development Mode Not Updating

- Ensure `jlpm watch` is running
- Hard refresh browser (Cmd+Shift+R / Ctrl+Shift+F5)
- Check browser console for errors
- Verify `jupyter lab` is running from repo root

## Notes

- The extension uses **modern ExtensionApp** architecture (not legacy `_load_jupyter_server_extension`)
- Kernel installation happens automatically on import via `make_kernels()` in `__init__.py`
- Timer utilities (`data_ferret/util/output.py`) provide performance instrumentation throughout
- Cell metadata tracking requires `FerretKernelClient` for proper `cell_id` propagation
