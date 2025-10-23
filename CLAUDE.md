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

1. **Frontend (TypeScript)**: `src/` - JupyterLab UI components and command palette integration
2. **Server Extension (Python)**: `data_ferret/server/` - HTTP handlers and command processing
3. **Custom Kernel**: `data_ferret/kernel/` - Enhanced IPython kernel with state management

### Frontend Components (`src/`)

- `index.ts` - JupyterLab plugin entry point, activates extension
- `manager.ts` - `FerretCommandsManager` orchestrates command loading and execution
- `api.ts` - `FerretAPI` handles HTTP communication with `/ferret/*` endpoints
- `kernel.ts` - `KernelUtils` manages kernel sessions and initialization
- `toolbar.ts` / `celltoolbar.ts` - Add command buttons to notebook UI
- `types.ts` - TypeScript interfaces for API contracts

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

### Custom Kernel (`data_ferret/kernel/`)

Extends IPython kernel with advanced features:

- `ferret_kernel.py` - Main kernel implementation (21 KB)
- `ferret_client.py` - Enhanced `BlockingKernelClient` that includes `cell_id` and `cell_metadata` in execution messages
- `checkpoint.py` - State snapshots (save/restore kernel state)
- `diff.py` - Namespace diffing to track variable changes between executions
- `equality.py` - Deep equality checking for various Python types
- `extended_types.py` - Type introspection utilities
- `magics.py` - IPython magic commands
- `ferret_pdb.py` - Debugger integration

**Key Feature**: `FerretKernelClient.execute()` overrides parent to inject `cell_id` and `cell_metadata` into kernel messages, enabling cell-level tracking.

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

- Kernel spec: `data_ferret/kernel/kernelspec/`
- Main kernel class: `data_ferret/kernel/ferret_kernel.py`
- Client-side kernel utilities: `src/kernel.ts`

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
