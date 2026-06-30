![](https://github.com/stephenfreund/FlowBook/raw/main/media/flowbook-small.png)

---

[Emery Berger](https://emeryberger.com),
[Cormac Flanagan](https://users.soe.ucsc.edu/~cormac/),
[Stephen Freund](https://www.cs.williams.edu/~freund/),
[Eunice Jun](http://eunicemjun.com/)

**Output consistency for Jupyter notebooks.**

FlowBook is a JupyterLab extension that enforces _rerun consistency_:
re-executing any cell from the current state would produce a result
consistent with a top-to-bottom execution of the notebook,
regardless of which cells have been run, modified, and rerun.
Cells whose inputs may have changed are marked _stale_,
and operations that would break rerun consistency (e.g., a later cell
overwriting a value read by an earlier one) disallowed.

When every cell is _clean_ — executed and rerun consistent —
the notebook is guaranteed _output consistent_: running it top-to-bottom
from an empty store yields exactly the outputs currently recorded.

For technical details and a complete evaluation, see our arXiv paper: [FlowBook: Enforcing Reproducibility in Computational Notebooks](https://arxiv.org/abs/2605.01560).

(Note: we sometimes use reproducibility as a synonym for output consistency in the documentation and code.)

## Quick Start

1. Install from Pypi:

    ```bash
    python3 -m pip install flowbook-python
    ```
  
    or from this repository:
  
    ```bash
    python3 -m pip install -e .
    ```

2. You will need Python and node.  More details on installation from source below.

    Then launch jupyter lab
  
    ```bash
    jupyter lab examples/
    ```

Once JupyterLab opens, create or open a notebook and select the
**FlowBook** kernel from the kernel picker.

To walk through FlowBook's features interactively, download the
[Getting Started demo notebook](https://github.com/stephenfreund/FlowBook/raw/main/examples/GettingStarted.ipynb),
open it in JupyterLab. Be sure to use the **FlowBook Kernel**.

For a longer, self-contained tutorial, download our
[FlowBook tutorial](https://github.com/stephenfreund/FlowBook/raw/main/examples/FlowBookTutorial.ipynb)

## AI Fix Suggestions

FlowBook can ask an LLM to diagnose any violation it raises and propose
one-click fixes. The suggestion appears directly inside the violation notice
underneath the cell, with buttons that apply the fix automatically.

To use this feature, set the `ANTHROPIC_API_KEY` environment variable to a
valid Anthropic API key before launching JupyterLab. If no provider key is
set, the suggestion UI is silently disabled — violations still appear, just
without diagnoses and fix buttons.

For other providers, model selection, costs, safety constraints, and the
full feature description, see **[AI.md](AI.md)**.

## Troubleshoot

If FlowBook does not appear to be working, work through these steps:

**1. Confirm the server extension is enabled.**

```bash
jupyter server extension list
```

Look for `flowbook` marked as `enabled`. If it is missing or disabled,
enable it:

```bash
jupyter server extension enable flowbook
```

**2. Confirm the frontend extension is installed.**

```bash
jupyter labextension list
```

Look for `flowbook` in the list of enabled extensions. If it is not
there, reinstall the package:

```bash
python3 -m pip install --force-reinstall flowbook-python
```

**3. Confirm the FlowBook kernel is registered.**

```bash
jupyter kernelspec list
```

You should see `flowbook_kernel` in the output. If it is missing,
reinstall the package (step 2) — the kernelspec is registered at
install time.

**4. Pick the FlowBook kernel in your notebook.**

FlowBook only tracks notebooks running under the **FlowBook Kernel**.
Use JupyterLab's kernel picker (top-right of the notebook) to switch
away from the default Python kernel if you are not seeing
staleness/violation markers.

**5. Hard-refresh the browser.**

After installing or upgrading, JupyterLab may cache older frontend
assets. Do a hard refresh (`Cmd+Shift+R` on macOS, `Ctrl+Shift+F5` on
Linux/Windows) and reopen the notebook.

**6. Check the browser console and the JupyterLab server log.**

Open the browser's developer tools (`Cmd+Option+I` / `Ctrl+Shift+I`)
and look for errors in the Console tab. Also look at the terminal
where you launched `jupyter lab` for server-side errors. These often
point directly at the underlying problem (missing dependency, version
mismatch, etc.).

**7. AI fix suggestions not appearing.**

If violations show up correctly but you don't see a diagnosis or fix
buttons underneath them:

- Confirm `litellm` is installed (`pip show litellm`).
- Confirm the provider API key for your configured `fix_model` is set in
  the same shell where you launched `jupyter lab` (run `echo
$ANTHROPIC_API_KEY` etc.).
- Check the Jupyter server log for `503` responses from
  `/flowbook/suggest-fix` — those mean the feature self-disabled because
  no provider key was found.
- Hard-refresh the browser; previously cached HTML for older violation
  notices won't have the new diagnosis/buttons placeholders.

**8. Still stuck?**

Please file an issue at
[github.com/stephenfreund/FlowBook/issues](https://github.com/stephenfreund/FlowBook/issues)
with the outputs of the commands above, your OS and Python version,
and a minimal notebook that reproduces the problem.

## Uninstall

To remove the extension, execute:

```bash
pip uninstall flowbook-python
```

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

### Running tests

Run the full Python test suite with `pytest`:

```bash
pytest flowbook/
```

To run the tests for a specific subpackage, point `pytest` at its `tests/` directory, e.g.:

```bash
pytest flowbook/kernel/tests/
pytest flowbook/mcp/tests/
```

### Development uninstall

```bash
# Server extension must be manually disabled in develop mode
jupyter server extension disable flowbook
pip uninstall flowbook-python
```

In development mode, you will also need to remove the symlink created by `jupyter labextension develop`
command. To find its location, you can run `jupyter labextension list` to figure out where the `labextensions`
folder is located. Then you can remove the symlink named `flowbook` within that folder.
