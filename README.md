![](https://github.com/stephenfreund/FlowBook/raw/main/media/flowbook-small.png)

# FlowBook

by
[Emery Berger](https://emeryberger.com),
[Cormac Flanaga](https://users.soe.ucsc.edu/~cormac/),
[Stephen Freund](https://www.cs.williams.edu/~freund/),
[Eunice Jun](http://eunicemjun.com/)
(ordered alphabetically)

**Automatic reproducibility tracking for Jupyter notebooks.**

FlowBook is a JupyterLab 4.0+ extension that tracks how data flows
between cells and tells you exactly which cells are stale after an
edit or out-of-order execution. When every cell is clean, your
notebook's outputs are guaranteed to match a fresh top-to-bottom run.

### Key Features

- **Always-on staleness tracking** — cells that need to be (re-)run
  are highlighted in yellow, automatically, as you work.
- **Violation detection** — cells whose execution would break
  reproducibility (e.g., overwriting a variable read by an earlier
  cell) are flagged in red and optionally rejected.
- **Variable- and column-level precision** — FlowBook tracks
  individual DataFrame columns, row sets, and file accesses, not just
  top-level variable names.
- **Metadata panel** — a sidebar showing read/write sets, staleness
  reasons, dependency graphs, and timing information.

## Requirements

- JupyterLab >= 4.0.0

## Quick Start

Install FlowBook using `pip`:

```bash
python3 -m pip install flowbook-python
```

Then launch jupyter lab

```bash
jupyter lab .
```

Once JupyterLab opens, create or open a notebook and select the
**FlowBook** kernel from the kernel picker.

## Troubleshoot

If you are seeing the frontend extension, but it is not working, check
that the server extension is enabled:

```bash
jupyter server extension list
```

If the server extension is installed and enabled, but you are not seeing
the frontend extension, check the frontend extension is installed:

```bash
jupyter labextension list
```

## Uninstall

To remove the extension, execute:

```bash
pip uninstall flowbook
```

## Command Line Tools

FlowBook provides several command line tools for notebook processing,
optimization, and analysis. See [CLI.md](CLI.md) for complete
documentation.

- `flowbook` - Main CLI for notebook processing commands

## Source Installation

Clone this repository and then install it as an editable package

```bash
python3 -m pip install -e .
jupyter lab examples/
```

Once JupyterLab opens, create or open a notebook and select the
**FlowBook** kernel from the kernel picker. Start with
`GettingStarted.ipynb`, then explore the `demos/` and `litmus/`
directories.

Note: You will need NodeJS to build the extension package.

The `jlpm` command is JupyterLab's pinned version of
[yarn](https://yarnpkg.com/) that is installed with JupyterLab. You may use
`yarn` or `npm` in lieu of `jlpm` below.

```bash
# Clone the repo to your local environment
# Change directory to the flowbook directory
# Install package in development mode
pip install -e "."
# Link your development version of the extension with JupyterLab
jupyter labextension develop . --overwrite
# Server extension must be manually installed in develop mode
jupyter server extension enable flowbook
# Rebuild extension Typescript source after making changes
jlpm build
```

You can watch the source directory and run JupyterLab at the same time in different terminals to watch for changes in the extension's source and automatically rebuild the extension.

```bash
# Watch the source directory in one terminal, automatically rebuilding when needed
jlpm watch
# Run JupyterLab in another terminal
jupyter lab
```

With the watch command running, every saved change will immediately be built locally and available in your running JupyterLab. Refresh JupyterLab to load the change in your browser (you may need to wait several seconds for the extension to be rebuilt).

By default, the `jlpm build` command generates the source maps for this extension to make it easier to debug using the browser dev tools. To also generate source maps for the JupyterLab core extensions, you can run the following command:

```bash
jupyter lab build --minimize=False
```

### Development uninstall

```bash
# Server extension must be manually disabled in develop mode
jupyter server extension disable flowbook
pip uninstall flowbook
```

In development mode, you will also need to remove the symlink created by `jupyter labextension develop`
command. To find its location, you can run `jupyter labextension list` to figure out where the `labextensions`
folder is located. Then you can remove the symlink named `flowbook` within that folder.
