"""End-to-end integration tests: suggest-fix → apply-fix → verify transform.

Each test simulates one violation taxonomy entry by:
  1. Constructing a synthetic notebook with the violating source pattern
     and the matching flowbook metadata.
  2. Mocking FixSuggester to return a canned diagnosis + plan.
  3. Driving SuggestFixHandler through SSE; capturing the streamed plan.
  4. Calling ApplyFixHandler with the selected fix.
  5. Asserting the resulting notebook matches the expected source.

This is the "fixture notebook" test promised in the design plan. No kernel,
no real LLM. The goal is to verify the data flow + dispatcher correctness
for each violation type the LLM might encounter in the wild.
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from flowbook.server.fix_models import FixPlan, FixSuggestion
from flowbook.server.fix_suggester import PlanEvent, TextEvent
from flowbook.server.handlers import ApplyFixHandler, SuggestFixHandler


# ---------------------------------------------------------------------------
# Reuse the handler mocking harness from test_fix_handlers
# ---------------------------------------------------------------------------

from flowbook.server.tests.test_fix_handlers import _make_handler


def _aio(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _code(id_: str, src: str, flowbook_meta: dict = None) -> dict:
    cell = {
        "cell_type": "code",
        "id": id_,
        "source": src,
        "metadata": {},
        "outputs": [],
        "execution_count": None,
    }
    if flowbook_meta is not None:
        cell["metadata"]["flowbook"] = flowbook_meta
    return cell


def _nb(*cells) -> dict:
    return {
        "cells": list(cells),
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def _scripted_suggester(diagnosis: str, fix: FixSuggestion):
    """Build a FakeSuggester class that yields the given canned events."""

    class FakeSuggester:
        def __init__(self, *args, **kwargs):
            pass

        async def stream(self, context, notebook=None):
            yield TextEvent(text=diagnosis)
            yield PlanEvent(plan=FixPlan(fixes=[fix]))

    return FakeSuggester


def _parse_sse_plan(written_chunks: list) -> dict:
    """Extract the 'plan' event payload from SSE write() chunks."""
    text = "".join(written_chunks)
    for raw in text.split("\n\n"):
        if not raw.strip():
            continue
        lines = raw.split("\n")
        event_type = None
        data = None
        for line in lines:
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        if event_type == "plan" and data:
            return json.loads(data)
    raise AssertionError("No plan event in SSE stream")


def _run_suggest_then_apply(
    notebook: dict,
    violating_cell_id: str,
    fix: FixSuggestion,
    monkeypatch,
    diagnosis: str = "Diagnosis sentence.",
) -> tuple[dict, dict]:
    """Drive both handlers in sequence; return (apply_result, modified_nb)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    FakeCls = _scripted_suggester(diagnosis, fix)

    # 1. SuggestFix
    with patch("flowbook.server.handlers.FixSuggester", side_effect=FakeCls):
        suggest_handler = _make_handler(
            SuggestFixHandler,
            settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
            body={"notebook": notebook, "cell_id": violating_cell_id},
        )
        _aio(suggest_handler.post())
        plan = _parse_sse_plan(suggest_handler._written)

    # The diagnosis frame should also be present.
    assert "event: diagnosis" in "".join(suggest_handler._written)

    # 2. ApplyFix using the first proposed fix
    chosen = plan["fixes"][0]
    apply_handler = _make_handler(
        ApplyFixHandler,
        settings={},
        body={
            "notebook": notebook,
            "tool": chosen["tool"],
            "args": chosen["args"],
        },
    )
    _aio(apply_handler.post())
    assert apply_handler._status == 200, apply_handler._finished[-1]
    payload = json.loads(apply_handler._finished[-1])
    return payload["result"], payload["notebook"]


# ---------------------------------------------------------------------------
# Violation taxonomy coverage
# ---------------------------------------------------------------------------


class TestNoReadAndWrite:
    """train = pd.concat([train, extra]) → alpha_rename to train_combined."""

    def test_full_flow(self, monkeypatch):
        nb = _nb(
            _code("a000", "import pandas as pd\ntrain = pd.read_csv('x.csv')"),
            _code(
                "b000",
                "train = pd.concat([train, extra])",
                flowbook_meta={"errors": [{
                    "error_type": "no_read_and_write",
                    "locations": ["train"],
                }]},
            ),
            _code("c000", "y = train.head()"),
        )
        fix = FixSuggestion(
            label="Rename train → train_combined",
            rationale="Cell reads and writes 'train'.",
            tool="alpha_rename",
            args={
                "cell_id": "b000",
                "old_name": "train",
                "new_name": "train_combined",
            },
        )
        result, new_nb = _run_suggest_then_apply(nb, "b000", fix, monkeypatch)
        assert result["ok"]
        assert "train_combined" in _src(new_nb["cells"][1])
        assert "train_combined" in _src(new_nb["cells"][2])
        # Earlier cell unaffected
        assert "pd.read_csv" in _src(new_nb["cells"][0])


class TestUnrecoverableMutationInplace:
    """df.drop(inplace=True) → remove_inplace."""

    def test_full_flow(self, monkeypatch):
        nb = _nb(
            _code("a000", "import pandas as pd\ndf = pd.read_csv('x.csv')"),
            _code(
                "b000",
                "df.drop(columns=['unused'], inplace=True)",
                flowbook_meta={"errors": [{
                    "error_type": "unrecoverable_mutation",
                    "locations": ["df"],
                }]},
            ),
        )
        fix = FixSuggestion(
            label="Replace inplace=True with assignment",
            rationale="In-place mutation breaks rerun consistency.",
            tool="remove_inplace",
            args={"cell_id": "b000", "variable": "df"},
        )
        result, new_nb = _run_suggest_then_apply(nb, "b000", fix, monkeypatch)
        assert result["ok"]
        new_source = _src(new_nb["cells"][1])
        assert "inplace" not in new_source
        assert "df = df.drop" in new_source


class TestNoWriteAfterReadDiagnostic:
    """Inspection cell sitting above mutation → mark_diagnostic."""

    def test_full_flow(self, monkeypatch):
        nb = _nb(
            _code("a000", "df = pd.read_csv('x.csv')"),
            _code(
                "b000",
                "df.info()",
                flowbook_meta={"errors": [{
                    "error_type": "no_write_after_read",
                    "locations": ["df"],
                    "causer_cell": "@a000",
                }]},
            ),
            _code("c000", "df = df.dropna()"),
        )
        fix = FixSuggestion(
            label="Mark inspection cell as diagnostic",
            rationale="df.info() is read-only; mark it so it doesn't trip rerun consistency.",
            tool="mark_diagnostic",
            args={"cell_id": "b000"},
        )
        result, new_nb = _run_suggest_then_apply(nb, "b000", fix, monkeypatch)
        assert result["ok"]
        assert _src(new_nb["cells"][1]).startswith("%diagnostic")


class TestSequentialTransform:
    """Sequential df = ... cells → merge_cells."""

    def test_full_flow(self, monkeypatch):
        nb = _nb(
            _code("a000", "import pandas as pd\ndf = pd.read_csv('x.csv')"),
            _code("b000", "df = df.fillna(0)"),
            _code(
                "c000",
                "df = df.assign(feature=df['a'] * 2)",
                flowbook_meta={"errors": [{
                    "error_type": "no_write_after_read",
                    "locations": ["df"],
                    "causer_cell": "@b000",
                }]},
            ),
        )
        fix = FixSuggestion(
            label="Merge sequential df transforms",
            rationale="Adjacent cells transform the same df — merge keeps the chain explicit.",
            tool="merge_cells",
            args={"cell_ids": ["b000", "c000"]},
        )
        result, new_nb = _run_suggest_then_apply(nb, "c000", fix, monkeypatch)
        assert result["ok"]
        # Two code cells remain
        code_cells = [c for c in new_nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) == 2
        merged = _src(code_cells[1])
        assert "fillna" in merged
        assert "assign" in merged


class TestInsertDeepcopyOnModelFit:
    """model.fit() → insert_deepcopy."""

    def test_full_flow(self, monkeypatch):
        nb = _nb(
            _code("a000", "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression()"),
            _code(
                "b000",
                "model.fit(X, y)",
                flowbook_meta={"errors": [{
                    "error_type": "unrecoverable_mutation",
                    "locations": ["model"],
                }]},
            ),
            _code("c000", "pred = model.predict(X)"),
        )
        fix = FixSuggestion(
            label="Fit a deepcopy of the model",
            rationale="model.fit() mutates the estimator in place — fit a copy instead.",
            tool="insert_deepcopy",
            args={"cell_id": "b000", "variable": "model"},
        )
        result, new_nb = _run_suggest_then_apply(nb, "b000", fix, monkeypatch)
        assert result["ok"]
        b_src = _src(new_nb["cells"][1])
        assert "copy.deepcopy(model)" in b_src
        assert "model_b000" in b_src
        # Downstream renamed
        assert "model_b000" in _src(new_nb["cells"][2])


class TestBadPlanRollback:
    """If the suggested fix can't apply (e.g. cell missing), notebook is unchanged."""

    def test_apply_rejects_unknown_cell(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        nb = _nb(_code("a000", "x = 1", flowbook_meta={"errors": [{
            "error_type": "no_read_and_write",
            "locations": ["x"],
        }]}))
        original = _src(nb["cells"][0])

        handler = _make_handler(
            ApplyFixHandler,
            settings={},
            body={
                "notebook": nb,
                "tool": "alpha_rename",
                "args": {"cell_id": "zzzz", "old_name": "x", "new_name": "y"},
            },
        )
        _aio(handler.post())
        assert handler._status == 400
        # Notebook content not corrupted
        assert _src(nb["cells"][0]) == original
