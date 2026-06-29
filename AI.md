# AI Fix Suggestions

FlowBook can ask an LLM to diagnose any violation it raises and propose
one-click fixes. The suggestion appears directly inside the violation notice
underneath the cell, with buttons that apply the fix automatically.

To use this feature set the `ANTHROPIC_API_KEY` environment variable to a
valid Anthropic API key. See [Setup](#setup) below for other providers.

## What you see

When a cell fails a reproducibility predicate (e.g.
`train = pd.concat([train, extra])` trips _NoReadAndWrite_), the violation
notice shows:

1. The usual red error message describing the predicate violation.
2. A short, italicized **diagnosis** that streams in as the model thinks.
3. One to three **fix buttons** like _"Rename train → train_combined"_ or
   _"Replace inplace=True with assignment"_.
4. An **Other Fix…** button — always present — that lets you describe a
   fix in your own words and have the AI carry it out.

Click a built-in fix button and FlowBook applies the corresponding AST
transformation (`alpha_rename`, `remove_inplace`, `insert_deepcopy`,
`mark_diagnostic`, `merge_cells`, or `move_cell`) to the affected cells.
An **Undo fix** button appears for 30 seconds afterwards, which restores
the pre-fix state — including cells that were removed, moved, or added.

For built-in fixes the model never writes free-form code: every suggestion
is validated server-side against an allowlist of those six tools before it
can be applied.

## Agentic inspection

The diagnosis call is **agentic**: while the model reasons it can read any
cell's source, outputs, traceback, or flowbook metadata through a fixed set
of read-only tools. The LLM uses these to learn the shape of a DataFrame,
the type of an object, or what a distant cell did — context that often
materially changes which fix it picks. These read-only tools have no
ability to mutate the notebook, run code, or access the filesystem.

## "Other Fix…"

When the built-in suggestions don't match what you want, click **Other Fix…**
to open an inline textarea. Describe the change you want, e.g. _"split this
cell so the plot is its own step"_ or _"replace the dropna with fillna(0)"_,
then Submit. The AI applies the change using a small set of mutator tools
(`edit_cell_source`, `insert_cell_after`, `delete_cell`, `merge_cells`,
`move_cell`, `mark_diagnostic`) operating directly on the notebook JSON on
the server. Safety constraints enforced server-side:

- Code cells must still parse as Python after every edit; malformed code is
  rejected and the model can retry on its next tool call.
- Mutators never run cell code, never touch the kernel, never read or write
  files.
- The user's instruction is part of the prompt and is logged server-side.

The same Undo button works for custom fixes: it restores the original
sources and flowbook metadata of modified cells, deletes any cells the AI
added, and re-inserts any cells the AI deleted.

## Setup

`litellm` is included with FlowBook, so no extra install is needed. Just set
an API key for whichever provider you want to use:

```bash
# Anthropic (default — Claude Opus)
export ANTHROPIC_API_KEY=sk-ant-...

# Or OpenAI
export OPENAI_API_KEY=sk-...

# Or any other litellm-supported provider:
# GEMINI_API_KEY, AZURE_API_KEY, COHERE_API_KEY, MISTRAL_API_KEY, GROQ_API_KEY
```

Start JupyterLab as usual:

```bash
jupyter lab
```

If no provider key is set, FlowBook silently disables the suggestion UI —
violation notices still appear normally, just without the diagnosis and
fix buttons.

## Choosing a model

The default model is `anthropic/claude-opus-4-7`. Override it with the
`--FlowBookExtension.fix_model` flag (or set it in your Jupyter config):

```bash
# Use GPT-4o instead
jupyter lab --FlowBookExtension.fix_model=openai/gpt-4o

# Use Gemini
jupyter lab --FlowBookExtension.fix_model=gemini/gemini-2.0-flash

# Use Claude Haiku (faster + cheaper, less accurate on hard cases)
jupyter lab --FlowBookExtension.fix_model=anthropic/claude-haiku-4-5-20251001
```

Any model identifier `litellm` accepts will work — see
[the litellm provider list](https://docs.litellm.ai/docs/providers) for the
full set. The corresponding provider API key must be present in the
environment.

To persist the model choice, add it to `~/.jupyter/jupyter_server_config.py`:

```python
c.FlowBookExtension.fix_model = "openai/gpt-4o"
```

## Cost and behaviour notes

- **Auto-trigger with cancellation.** When a violation lands on a cell,
  FlowBook fires a suggestion request in the background. If you edit the
  cell, re-run it, or otherwise resolve the violation before the request
  finishes, the in-flight call is aborted — you don't pay for diagnoses
  the user didn't need.
- **Per-call cost.** A typical request is ~1.5KB of input + ~300 tokens of
  output. On Claude Opus that's roughly $0.04 per violation; on Haiku or
  GPT-4o it's a few tenths of a cent.
- **API key stays on the server.** The key is read from the Jupyter
  server's environment (or its config) — the browser never sees it. All
  LLM calls happen in the Jupyter server extension.
- **Validation before apply.** Every LLM-produced fix plan is validated
  against an allowlist of six tools, with the args shape, referenced
  cell_ids, and named variables all checked before any code is touched.
  An invalid suggestion is dropped; you'll see the diagnosis text but no
  buttons.
