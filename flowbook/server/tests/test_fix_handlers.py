"""Tests for SuggestFixHandler and ApplyFixHandler.

These bypass Tornado's authentication and run the handlers in-process with a
mocked LLM, so no network access is required.
"""

import asyncio
import json
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from flowbook.server.fix_suggester import ErrorEvent, PlanEvent, TextEvent
from flowbook.server.handlers import ApplyFixHandler, SuggestFixHandler


# ---------------------------------------------------------------------------
# Helpers: build a handler instance with the minimum harness Tornado needs
# ---------------------------------------------------------------------------

def _make_handler(cls, settings: dict, body: dict):
    """Construct a handler without going through Tornado's request pipeline."""
    handler = cls.__new__(cls)
    app = MagicMock()
    app.settings = settings
    handler.application = app
    handler._jupyter_current_user = "test-user"
    handler.current_user = "test-user"

    # Capture writes/finishes for assertions.
    handler._written: List[str] = []
    handler._finished: List[str] = []
    handler._status = 200
    handler._headers = {}

    def write(payload):
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        handler._written.append(payload)

    def finish(payload=None):
        if payload is not None:
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            handler._finished.append(payload)

    def set_status(s):
        handler._status = s

    def set_header(k, v):
        handler._headers[k] = v

    def flush():
        pass

    handler.write = write
    handler.finish = finish
    handler.set_status = set_status
    handler.set_header = set_header
    handler.flush = flush
    handler.get_json_body = lambda: body
    # APIHandler.settings is a property that returns application.settings,
    # which we already set on the mock above. No assignment needed.
    return handler


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if asyncio.get_event_loop().is_running() is False else asyncio.run(coro)


# Try to use a fresh loop each time to avoid contamination.
def _aio(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# SuggestFixHandler
# ---------------------------------------------------------------------------

def _violation_nb(cell_id="bbbb"):
    return {
        "cells": [
            {"cell_type": "code", "id": "aaaa", "source": "train = pd.read_csv('x.csv')", "metadata": {}},
            {
                "cell_type": "code", "id": cell_id,
                "source": "train = pd.concat([train, extra])",
                "metadata": {"flowbook": {"errors": [{
                    "error_type": "no_read_and_write",
                    "locations": ["train"],
                }]}},
            },
        ]
    }


class TestSuggestFixHandler:
    def test_503_when_no_provider_key(self, monkeypatch):
        for var in (
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
            "AZURE_API_KEY", "COHERE_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        handler = _make_handler(
            SuggestFixHandler,
            settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
            body={"notebook": _violation_nb(), "cell_id": "bbbb"},
        )
        _aio(handler.post())
        assert handler._status == 503
        body = json.loads(handler._finished[-1])
        assert body["feature_disabled"] is True

    def test_400_missing_notebook(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        handler = _make_handler(
            SuggestFixHandler,
            settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
            body={"cell_id": "bbbb"},
        )
        _aio(handler.post())
        assert handler._status == 400

    def test_404_when_cell_has_no_violation(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        nb = {
            "cells": [
                {"cell_type": "code", "id": "aaaa", "source": "x = 1", "metadata": {}},
            ]
        }
        handler = _make_handler(
            SuggestFixHandler,
            settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
            body={"notebook": nb, "cell_id": "aaaa"},
        )
        _aio(handler.post())
        assert handler._status == 404

    def test_sse_frames_emitted_for_text_and_plan(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        # Scripted suggester: two text chunks, then a plan.
        async def fake_stream(self, context):
            yield TextEvent(text="Diagnosis part 1. ")
            yield TextEvent(text="Part 2.")
            from flowbook.server.fix_models import FixPlan, FixSuggestion
            yield PlanEvent(plan=FixPlan(fixes=[FixSuggestion(
                label="Rename train",
                rationale="...",
                tool="alpha_rename",
                args={"cell_id": "bbbb", "old_name": "train", "new_name": "train_combined"},
            )]))

        with patch("flowbook.server.handlers.FixSuggester") as fake_cls:
            fake_cls.return_value.stream = lambda ctx, _s=fake_stream, _i=None: fake_stream(_i, ctx)
            # The above is awkward — easier path: subclass with bound method.
            class FakeSuggester:
                def __init__(self, *a, **k): pass
                async def stream(self_, context, notebook=None):
                    yield TextEvent(text="Diagnosis part 1. ")
                    yield TextEvent(text="Part 2.")
                    from flowbook.server.fix_models import FixPlan, FixSuggestion
                    yield PlanEvent(plan=FixPlan(fixes=[FixSuggestion(
                        label="Rename train",
                        rationale="...",
                        tool="alpha_rename",
                        args={"cell_id": "bbbb", "old_name": "train", "new_name": "train_combined"},
                    )]))
            fake_cls.side_effect = FakeSuggester

            handler = _make_handler(
                SuggestFixHandler,
                settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
                body={"notebook": _violation_nb(), "cell_id": "bbbb"},
            )
            _aio(handler.post())

        # Collect SSE frames from .write() calls
        all_writes = "".join(handler._written)
        assert "event: diagnosis" in all_writes
        assert "Diagnosis part 1." in all_writes
        assert "Part 2." in all_writes
        assert "event: plan" in all_writes
        assert "alpha_rename" in all_writes
        assert "event: done" in all_writes

    def test_sse_error_event_when_suggester_fails(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        class FakeSuggester:
            def __init__(self, *a, **k): pass
            async def stream(self_, context, notebook=None):
                yield ErrorEvent(message="oops")

        with patch("flowbook.server.handlers.FixSuggester", side_effect=FakeSuggester):
            handler = _make_handler(
                SuggestFixHandler,
                settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
                body={"notebook": _violation_nb(), "cell_id": "bbbb"},
            )
            _aio(handler.post())

        all_writes = "".join(handler._written)
        assert "event: error" in all_writes
        assert "oops" in all_writes


# ---------------------------------------------------------------------------
# ApplyFixHandler
# ---------------------------------------------------------------------------

class TestApplyFixHandler:
    def test_rejects_unknown_tool(self):
        nb = {"cells": [{"cell_type": "code", "id": "aaaa", "source": "x = 1", "metadata": {}}]}
        handler = _make_handler(
            ApplyFixHandler,
            settings={},
            body={"notebook": nb, "tool": "rm_rf", "args": {}},
        )
        _aio(handler.post())
        assert handler._status == 400
        assert "Unknown" in handler._finished[-1] or "unsupported" in handler._finished[-1]

    def test_rejects_arg_shape_mismatch(self):
        nb = {"cells": [{"cell_type": "code", "id": "aaaa", "source": "x = 1", "metadata": {}}]}
        handler = _make_handler(
            ApplyFixHandler,
            settings={},
            body={
                "notebook": nb,
                "tool": "alpha_rename",
                "args": {"cell_id": "aaaa", "old_name": "x"},  # missing new_name
            },
        )
        _aio(handler.post())
        assert handler._status == 400
        assert "mismatch" in handler._finished[-1]

    def test_happy_path_returns_modified_notebook(self):
        nb = {"cells": [
            {"cell_type": "code", "id": "aaaa", "source": "x = 1", "metadata": {}},
            {"cell_type": "code", "id": "bbbb", "source": "y = x + 1", "metadata": {}},
        ]}
        handler = _make_handler(
            ApplyFixHandler,
            settings={},
            body={
                "notebook": nb,
                "tool": "alpha_rename",
                "args": {"cell_id": "aaaa", "old_name": "x", "new_name": "x_renamed"},
            },
        )
        _aio(handler.post())
        assert handler._status == 200
        payload = json.loads(handler._finished[-1])
        assert payload["result"]["ok"] is True
        # Both cells should have the rename applied
        out_nb = payload["notebook"]
        assert "x_renamed" in "".join(out_nb["cells"][0]["source"])
        assert "x_renamed" in "".join(out_nb["cells"][1]["source"])

    def test_dispatcher_error_returns_400(self):
        nb = {"cells": [{"cell_type": "code", "id": "aaaa", "source": "y = 1", "metadata": {}}]}
        handler = _make_handler(
            ApplyFixHandler,
            settings={},
            body={
                "notebook": nb,
                "tool": "alpha_rename",
                "args": {"cell_id": "aaaa", "old_name": "missing", "new_name": "x"},
            },
        )
        _aio(handler.post())
        assert handler._status == 400
