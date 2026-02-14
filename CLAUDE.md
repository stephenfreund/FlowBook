# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlowBook is a JupyterLab 4.0+ extension that combines a TypeScript frontend with a Python server extension and custom IPython kernels. The extension provides notebook analysis, validation, execution, reproducibility enforcement, and AI-powered capabilities through a command-based architecture.

## Development Commands

### Initial Setup

```bash
# Install package in development mode
pip install -e "."

# Link development version with JupyterLab
jupyter labextension develop . --overwrite

# Enable server extension
jupyter server extension enable flowbook
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
pytest flowbook/

# Run specific test file
pytest flowbook/kernel/tests/test_reproducibility_enforcer.py
```

Test files (`test_*.py`) must be placed in a `tests/` subdirectory of the package they test. For example, tests for `flowbook/kernel/` go in `flowbook/kernel/tests/`. Each `tests/` directory must contain an `__init__.py` file.

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
2. **Server Extension (Python)**: `flowbook/server/` - HTTP handlers and command processing
3. **Custom Kernels**: Three IPython kernels for different use cases
   - `flowbook/kernel/` - FlowBook kernel with always-on reproducibility tracking
   - `flowbook/kernel_support/` - Experimental kernel with AI commands, profiling, checkpointing
   - `flowbook/checkpoint_kernel/` - Checkpoint kernel for timing/benchmarking

### Frontend Components (`src/`)

The frontend exports **two JupyterLab plugins** that activate based on the kernel in use:

```
src/
├── index.ts                 # Exports [flowbookPlugin, experimentalPlugin]
├── api.ts                   # Shared FlowbookAPI for HTTP communication
├── kernel.ts                # Shared KernelUtils
├── handler.ts               # Request handler
├── cellindex.ts             # Cell index management
├── cellindexutils.ts        # Cell index utilities
├── executiondialog.tsx       # Execution dialog component
├── logpanel.tsx             # Log panel UI component
├── messagecomponents.tsx    # Message UI components
├── shared/                  # Shared utilities
│   ├── kerneldetection.ts   # KernelDetector class for kernel type detection
│   └── types.ts             # Shared type definitions
├── experimental/            # Experimental kernel plugin (AI commands)
│   ├── plugin.ts            # Plugin activation with kernel gating
│   ├── manager.ts           # FlowbookCommandsManager orchestrates commands
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
└── flowbook/                # FlowBook kernel plugin (reproducibility tracking)
    ├── plugin.ts            # Plugin activation with kernel gating
    ├── types.ts             # IReproducibilityMetadata, IReproducibilityViolation, IReproducibilityCellState
    ├── stalenessmanager.ts  # Tracks stale cells per notebook
    ├── metadatapanel.tsx    # Reproducibility metadata panel (reads, writes, stale cells)
    ├── cellhighlighter.ts   # Red highlighting for stale cells
    └── executionhook.ts     # Extract flowbook metadata from outputs + cell edit detection
```

**Plugin Activation**:

- `flowbook:plugin` - Activates UI only when kernel is `flowbook_kernel`
- `flowbook:experimental` - Activates UI only when kernel is `experimental_kernel`

**Data Flow**: User clicks button → `executeCommand()` → `FlowbookAPI.executeCommand()` → POST to `/flowbook/execute` → Backend processes → Notebook updated

### Server Extension (`flowbook/server/`)

The server uses the modern **ExtensionApp** pattern (not legacy extension points).

- `__init__.py` - `FlowBookExtension(ExtensionApp)` class with `initialize_handlers()` method
- `handlers.py` - HTTP request handlers:
  - `POST /flowbook/execute` - Execute a command (FlowbookCommandHandler)
  - `GET /flowbook/list` - List available commands (CommandListHandler)
- `base.py` - `NotebookCommand` abstract base class
- `registry.py` - `CommandRegistry` singleton managing available commands
- `commands.py` - Built-in command implementations:
  - `AnalyzeNotebookCommand` - Notebook structure analysis
  - `ValidateNotebookCommand` - Syntax validation
  - `ExecuteBaseCommand` - Run all cells (requires kernel)
  - `InspectVariablesCommand` - Kernel namespace inspection
- `kernel_manager.py` - `KernelConnectionManager` and `FlowbookKernelClient` for kernel communication
- `cli.py` - Command-line interface entry point

### FlowBook Kernel (`flowbook/kernel/`)

The primary kernel with always-on reproducibility enforcement:

- `flowbook_kernel.py` - Main `FlowbookKernel` implementation with magic commands
- `flowbook_client.py` - `FlowbookKernelClient` with `cell_order` injection for reproducibility checks
- `reproducibility_enforcer.py` - `ReproducibilityEnforcer` implements formal transition rules (see below)
- `models.py` - `ReproducibilityMetadata`, `ReproducibilityViolation`, `ReproducibilityResult`, `ReproducibilityExecutionRecord` data classes
- `changes.py` - Typed records of what changed between checkpoints (`ValueChanged`, `ColumnAdded`, etc.)
- `access_events.py` - Typed records of variable/column/structural access during cell execution
- `change_detector.py` - Converts `MemoryCheckpointDiffResult` to typed `Change` list
- `conflict_resolver.py` - Matches `Change`s against `AccessEvent`s using declarative rules
- `conflict_rules.py` - Declarative specification of reproducibility conflict detection

**Formal Transition Rules** (from `FORMAL_DEVELOPMENT.md`):

The enforcer implements four transition rules from the formal specification:

| Rule | Condition | Cell status | Effect |
|---|---|---|---|
| EXEC-ACCEPT | No backward conflict, not forward-contaminated | fresh | StaleFwd |
| EXEC-CONTAMINATED | No backward conflict, forward-contaminated | stale | StaleFwd |
| EXEC-REJECT | Backward conflict (BackConflict) | unchanged | rollback |
| EXEC-RESTORE | All predecessors fresh, execute from prefix checkpoint | fresh | StaleFwd |

And three runtime checks:

- **BackConflict** (Def 1.8.2): Cell wrote to location read by earlier *fresh* cell → reject
- **FwdContaminated** (Def 1.8.3): Cell read location written by later executed cell → accept as stale
- **StaleFwd** (Def 1.8.1): Cell wrote to location read by later fresh cell → mark later cell stale

**Magic Commands**:

| Command | Description |
|---|---|
| `%notebook_structure <ids...>` | Set notebook cell order (sent by frontend before each execution) |
| `%cell_edited <cell_id>` | Mark edited cell stale (EDIT transition §2.3, sent by frontend) |
| `%flowbook_status` | Display current reproducibility state |
| `%flowbook_stale` | Show stale cells |
| `%continue_after_violation <on/off>` | Control whether backward violations reject or warn |
| `%structural_tracking <off/warn/enforce>` | Set DataFrame structural attribute tracking mode |

**Features** (always enabled):

- Variable tracking for all executions
- Staleness computation (which cells need re-execution)
- Backward mutation detection with column-aware conflict resolution
- Forward contamination detection (cell marked stale, not rejected)
- EXEC-RESTORE from prefix checkpoint when all predecessors are fresh
- Edit-triggered staleness via frontend notification
- Automatic rollback on backward reproducibility violations

**Metadata Format** (sent via `display_data` output):

```python
{
  "flowbook": {
    "cell_id": str,
    "execution_seq": int,
    "reads": List[str],
    "writes": List[str],
    "changed_variables": List[str],
    "stale_cells": List[str],
    "violation": Optional[dict],
    "cell_order": List[str],
    "column_reads": Dict[str, List[str]],
    "column_writes": Dict[str, List[str]],
    "column_changed": Dict[str, List[str]],
    "structural_reads": Dict[str, List[str]],
    "structural_warnings": List[str],
    "file_reads": List[str],
    "file_writes": List[str],
    "run_duration_ms": float,
    "state_duration_ms": float,
    "check_duration_ms": float,
    "cell_is_contaminated": bool,
    "exec_mode": "live" | "restore"
  }
}
```

### Experimental Kernel (`flowbook/kernel_support/`)

Full-featured kernel extending IPython with advanced features:

- `experimental_kernel.py` - Main `ExperimentalKernel` implementation
- `experimental_client.py` - Enhanced `BlockingKernelClient` that includes `cell_id` and `cell_metadata` in execution messages
- `checkpoint.py` - State snapshots (save/restore kernel state)
- `diff.py` - Namespace diffing to track variable changes between executions
- `tracking.py` - `TrackingDict` for optional variable access tracking
- `magics.py` - IPython magic commands (`%enable_scalene`, `%checkpoint`, etc.)
- `flowbook_pdb.py` - Debugger integration

**Features** (all optional, toggled via magic commands):

- Scalene profiling for CPU/memory analysis
- Checkpointing for save/restore kernel state
- Variable tracking for read-before-write analysis
- Monotonicity enforcement

### Checkpoint Kernel (`flowbook/checkpoint_kernel/`)

- `checkpoint_kernel.py` - `CheckpointKernel` for timing and benchmarking
- `checkpoint_client.py` - `CheckpointKernelClient`

**Key Feature**: Both `FlowbookKernelClient` and `ExperimentalKernelClient` inject `cell_id` and metadata into kernel messages, enabling cell-level tracking.

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

Register in `flowbook/server/__init__.py` or `commands.py`.

### Agent Integration (`flowbook/agent/`)

- `agent.py` - `FlowbookAgent` uses `openai-agents` framework with litellm backend
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

- **CLI**: `load_notebook()` in `flowbook/cli/helpers.py`
- **Server**: `FlowbookCommandHandler.post()` in `flowbook/server/handlers.py`
- **Core function**: `normalize_notebook()` in `flowbook/util/cell_ids.py`

### Why 4-character IDs?

- **Readability**: Short IDs are easy to read in logs and debugging (26^4 = 456,976 possible IDs)
- **Consistency**: All notebooks use the same ID format regardless of source
- **Simplicity**: Easier to reference and track cells in development and testing

## Code Style

### Python

- Follow standard Python conventions
- Use type hints where applicable
- Abstract base classes for extensibility (e.g., `NotebookCommand`)
- **No relative imports**: All imports must use absolute paths (e.g., `from flowbook.kernel.models import ...`, not `from .models import ...`)
- Test files go in `tests/` subdirectories with `__init__.py` files

### TypeScript

- **Interfaces**: Must start with `I` and use PascalCase (e.g., `ICommandInfo`)
- **Quotes**: Single quotes, avoid template literals unless necessary
- **Equality**: Use strict equality (`===`)
- **Callbacks**: Prefer arrow functions
- **Curly braces**: Always use for control structures

## Important Files

### Configuration

- `pyproject.toml` - Python package config, dependencies, build system (hatchling)
- `package.json` - NPM package config, scripts, linting rules
- `tsconfig.json` - TypeScript compiler settings (ES2020, strict mode)
- `jupyter-config/server-config/flowbook.json` - Jupyter server extension registration

### Build Artifacts

- `lib/` - Compiled TypeScript output (gitignored)
- `flowbook/labextension/` - JupyterLab extension bundle (auto-generated)
- `flowbook/_version.py` - Auto-generated from package.json version

## Extension Points

### Adding a New Command

1. Create class in `flowbook/server/commands.py` inheriting `NotebookCommand`
2. Implement required properties: `command_name`, `display_name`, `icon_name`, `requires_kernel`
3. Implement `process()` method
4. Register in `CommandRegistry` (typically auto-registered via import)
5. Frontend automatically discovers via `GET /flowbook/list`

### Modifying Kernel Behavior

**FlowBook Kernel** (reproducibility):

- Kernel spec: `flowbook/kernel/kernelspec/`
- Main kernel class: `flowbook/kernel/flowbook_kernel.py`
- Reproducibility logic: `flowbook/kernel/reproducibility_enforcer.py`
- Conflict detection pipeline: `access_events.py` → `change_detector.py` → `conflict_rules.py` → `conflict_resolver.py`
- Formal specification: `FORMAL_DEVELOPMENT.md`

**Experimental Kernel** (AI commands, profiling):

- Kernel spec: `flowbook/kernel_support/kernelspec/`
- Main kernel class: `flowbook/kernel_support/experimental_kernel.py`

**Checkpoint Kernel** (benchmarking):

- Kernel spec: `flowbook/checkpoint_kernel/kernelspec/`
- Main kernel class: `flowbook/checkpoint_kernel/checkpoint_kernel.py`

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
# Should show "flowbook" as enabled

# Check frontend extension
jupyter labextension list
# Should show "flowbook" in enabled extensions
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
- Timer utilities (`flowbook/util/output.py`) provide performance instrumentation throughout
- Cell metadata tracking requires `FlowbookKernelClient` for proper `cell_id` propagation
- Formal specification of reproducibility rules is in `FORMAL_DEVELOPMENT.md` with an Implementation Map linking formal definitions to code locations
- The frontend `executionhook.ts` sends `%notebook_structure` before each cell execution and `%cell_edited` (debounced 1s) when a previously-executed cell's source changes


## Formal Specification Sync

This project maintains a formal specification in `FORMAL_DEVELOPMENT.md` that maps formal concepts to their source code implementations. The spec and the code must always be kept in sync — **changes flow in both directions:**

- **Spec → Code:** When a formal concept in `FORMAL_DEVELOPMENT.md` is added, modified, or removed, the corresponding source code MUST be updated to reflect the change. The spec is the source of truth for *what* the system should do.
- **Code → Spec:** When source code implementing a formal concept is created, modified, renamed, or deleted, the mapping in `FORMAL_DEVELOPMENT.md` MUST be updated to reflect the change.

Before completing any task, verify that `FORMAL_DEVELOPMENT.md` and the source code it references are consistent with each other.
