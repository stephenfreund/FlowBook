"""Tests for CustomFixHandler.

The agentic loop is mocked at the FixSuggester level — we substitute a fake
suggester that yields a scripted sequence of TextEvents, CustomDoneEvent.
The handler should drive the SSE stream and emit a 'done' frame whose data
payload matches what _build_custom_fix_response would produce from the
mutation log.
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from flowbook.server.fix_suggester import (
    CustomDoneEvent,
    ErrorEvent,
    TextEvent,
)
from flowbook.server.fix_tools_mutator import MutationEntry, MutationLog
from flowbook.server.handlers import CustomFixHandler

from flowbook.server.tests.test_fix_handlers import _make_handler


def _aio(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _code(id_, src, **kw):
    return {
        "cell_type": "code",
        "id": id_,
        "source": src,
        "metadata": kw.pop("metadata", {}),
        "outputs": kw.pop("outputs", []),
        "execution_count": kw.pop("execution_count", None),
        **kw,
    }


def _nb(*cells):
    return {
        "cells": list(cells),
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _parse_sse(written_chunks: list, event_type: str) -> dict:
    text = "".join(written_chunks)
    for raw in text.split("\n\n"):
        lines = raw.split("\n")
        et = None
        data = None
        for line in lines:
            if line.startswith("event:"):
                et = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        if et == event_type and data:
            return json.loads(data)
    raise AssertionError(f"No '{event_type}' event in SSE stream")


# ---------------------------------------------------------------------------
# Scripted suggester factories
# ---------------------------------------------------------------------------

def _scripted_custom_suggester(
    text_chunks,
    mutator_calls,
    final_summary,
):
    """Build a fake FixSuggester whose custom_stream yields the scripted
    events AND applies the given mutator_calls to the notebook so the
    handler sees real mutations.
    """
    from flowbook.server.fix_tools_mutator import dispatch

    class FakeSuggester:
        def __init__(self, *a, **k):
            pass

        async def custom_stream(
            self, notebook, cell_id, cell_alpha, instruction
        ):
            log = MutationLog()
            for text in text_chunks:
                yield TextEvent(text=text)
            # Apply each scripted mutation so the notebook ends up in the
            # expected state and the log gets populated.
            for tool, args in mutator_calls:
                dispatch(notebook, log, tool, args)
            yield CustomDoneEvent(summary=final_summary, log=log)

    return FakeSuggester


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRequestValidation:
    def test_503_when_no_provider_key(self, monkeypatch):
        for var in (
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
            "AZURE_API_KEY", "COHERE_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        handler = _make_handler(
            CustomFixHandler,
            settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
            body={
                "notebook": _nb(_code("a000", "x = 1")),
                "cell_id": "a000",
                "instruction": "do something",
            },
        )
        _aio(handler.post())
        assert handler._status == 503

    def test_400_when_instruction_empty(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        handler = _make_handler(
            CustomFixHandler,
            settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
            body={
                "notebook": _nb(_code("a000", "x = 1")),
                "cell_id": "a000",
                "instruction": "   ",
            },
        )
        _aio(handler.post())
        assert handler._status == 400

    def test_404_for_unknown_cell(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        handler = _make_handler(
            CustomFixHandler,
            settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
            body={
                "notebook": _nb(_code("a000", "x = 1")),
                "cell_id": "zzzz",
                "instruction": "do something",
            },
        )
        _aio(handler.post())
        assert handler._status == 404


class TestCustomFixHappyPath:
    def test_edit_cell_source(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        nb = _nb(_code("a000", "x = 1"), _code("b000", "y = 2"))
        FakeCls = _scripted_custom_suggester(
            text_chunks=["Editing the cell."],
            mutator_calls=[
                ("edit_cell_source", {"cell_id": "b000", "new_source": "y = 99"}),
            ],
            final_summary="Changed y from 2 to 99.",
        )
        with patch("flowbook.server.handlers.FixSuggester", side_effect=FakeCls):
            handler = _make_handler(
                CustomFixHandler,
                settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
                body={"notebook": nb, "cell_id": "b000", "instruction": "change y to 99"},
            )
            _aio(handler.post())

        done = _parse_sse(handler._written, "done")
        assert done["ok"] is True
        assert done["modified_cells"] == ["b000"]
        assert done["cells_added"] == []
        assert done["cells_removed"] == []
        assert done["pre_fix_sources"]["b000"] == "y = 2"
        assert done["post_fix_sources"]["b000"] == "y = 99"
        assert done["summary"] == "Changed y from 2 to 99."

    def test_insert_then_edit(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        nb = _nb(_code("a000", "x = 1"))
        FakeCls = _scripted_custom_suggester(
            text_chunks=[],
            mutator_calls=[
                ("insert_cell_after", {"after_cell_id": "a000", "source": "y = 2"}),
            ],
            final_summary="Added a new cell.",
        )
        with patch("flowbook.server.handlers.FixSuggester", side_effect=FakeCls):
            handler = _make_handler(
                CustomFixHandler,
                settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
                body={"notebook": nb, "cell_id": "a000", "instruction": "add a y=2 cell after this"},
            )
            _aio(handler.post())

        done = _parse_sse(handler._written, "done")
        assert done["ok"] is True
        assert len(done["cells_added"]) == 1
        new_id = done["cells_added"][0]
        assert done["post_fix_sources"][new_id] == "y = 2"
        # Order shows the new cell after a000
        assert done["new_cell_order"] == ["a000", new_id]

    def test_delete_cell(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        nb = _nb(_code("a000", "x = 1"), _code("b000", "y = 2"))
        FakeCls = _scripted_custom_suggester(
            text_chunks=[],
            mutator_calls=[("delete_cell", {"cell_id": "b000"})],
            final_summary="Removed cell b000.",
        )
        with patch("flowbook.server.handlers.FixSuggester", side_effect=FakeCls):
            handler = _make_handler(
                CustomFixHandler,
                settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
                body={"notebook": nb, "cell_id": "a000", "instruction": "delete the second cell"},
            )
            _aio(handler.post())

        done = _parse_sse(handler._written, "done")
        assert done["cells_removed"] == ["b000"]
        # pre_fix_sources captured b000's original source so undo can restore
        assert done["pre_fix_sources"]["b000"] == "y = 2"
        assert done["new_cell_order"] == ["a000"]

    def test_text_chunks_streamed_as_diagnosis(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        nb = _nb(_code("a000", "x = 1"))
        FakeCls = _scripted_custom_suggester(
            text_chunks=["Looking at the notebook. ", "Now editing."],
            mutator_calls=[("edit_cell_source", {"cell_id": "a000", "new_source": "x = 2"})],
            final_summary="Done.",
        )
        with patch("flowbook.server.handlers.FixSuggester", side_effect=FakeCls):
            handler = _make_handler(
                CustomFixHandler,
                settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
                body={"notebook": nb, "cell_id": "a000", "instruction": "change x"},
            )
            _aio(handler.post())

        all_writes = "".join(handler._written)
        assert "event: diagnosis" in all_writes
        assert "Looking at the notebook." in all_writes
        assert "Now editing." in all_writes


class TestCustomFixErrorPath:
    def test_suggester_error_event_emitted(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        nb = _nb(_code("a000", "x = 1"))

        class FakeSuggester:
            def __init__(self, *a, **k):
                pass
            async def custom_stream(self, notebook, cell_id, cell_alpha, instruction):
                yield ErrorEvent(message="LLM unavailable")

        with patch("flowbook.server.handlers.FixSuggester", side_effect=FakeSuggester):
            handler = _make_handler(
                CustomFixHandler,
                settings={"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}},
                body={"notebook": nb, "cell_id": "a000", "instruction": "anything"},
            )
            _aio(handler.post())

        err = _parse_sse(handler._written, "error")
        assert "LLM unavailable" in err["message"]
