"""Tests for fix_suggester: prompt assembly, primer loading, parse + validate."""

import os
from unittest.mock import patch

import pytest

from flowbook.server.fix_models import PlanValidationError
from flowbook.server.fix_suggester import (
    DEFAULT_MODEL,
    FIX_PLAN_CLOSE,
    FIX_PLAN_OPEN,
    _extract_and_parse_plan,
    _provider_prefix,
    build_context_from_notebook,
    build_system_prompt,
    build_user_message,
    feature_enabled,
    get_model,
    load_primer,
)


class TestPrimer:
    def test_primer_loads_and_is_nonempty(self):
        primer = load_primer()
        assert len(primer) > 500
        assert "Rerun consistency" in primer

    def test_system_prompt_includes_primer(self):
        prompt = build_system_prompt()
        # All four predicate names should appear via the primer.
        assert "NoReadAndWrite" in prompt
        assert "WriteBeforeRead" in prompt
        assert "NoReadBeforeWrite" in prompt
        assert "NoWriteAfterRead" in prompt

    def test_system_prompt_lists_all_six_tools(self):
        prompt = build_system_prompt()
        for tool in (
            "alpha_rename",
            "remove_inplace",
            "insert_deepcopy",
            "mark_diagnostic",
            "merge_cells",
            "move_cell",
        ):
            assert tool in prompt, f"Tool {tool!r} missing from system prompt"


class TestUserMessage:
    def test_includes_cell_alpha_and_source(self):
        from flowbook.server.fix_models import ViolationContext
        ctx = ViolationContext(
            cell_id="abcd",
            cell_alpha="C",
            cell_source="train = pd.concat([train, extra])\n",
            error_type="no_read_and_write",
            locations=["train"],
            causer_cells=[],
            cell_order=["aaaa", "bbbb", "abcd"],
            surrounding_sources={"bbbb": "x = 1\n"},
        )
        msg = build_user_message(ctx)
        assert "@C" in msg
        assert "abcd" in msg
        assert "no_read_and_write" in msg
        assert "pd.concat" in msg
        assert "bbbb" in msg  # surrounding cell shown


class TestParseFixPlan:
    def test_well_formed_plan_parses(self):
        text = f"""
        The diagnosis text comes first.

        {FIX_PLAN_OPEN}{{"fixes": [
          {{"label": "Rename train", "rationale": "...", "tool": "alpha_rename",
            "args": {{"cell_id": "B", "old_name": "train", "new_name": "train_combined"}}}}
        ]}}{FIX_PLAN_CLOSE}
        """
        plan = _extract_and_parse_plan(text)
        assert len(plan.fixes) == 1
        assert plan.fixes[0].tool == "alpha_rename"
        assert plan.fixes[0].args["new_name"] == "train_combined"

    def test_missing_block_raises(self):
        with pytest.raises(PlanValidationError, match="no FIX_PLAN"):
            _extract_and_parse_plan("just diagnosis, no block")

    def test_bad_json_raises(self):
        text = f"{FIX_PLAN_OPEN}{{not valid json{FIX_PLAN_CLOSE}"
        with pytest.raises(PlanValidationError, match="valid JSON"):
            _extract_and_parse_plan(text)

    def test_missing_fixes_key_raises(self):
        text = f'{FIX_PLAN_OPEN}{{"something": []}}{FIX_PLAN_CLOSE}'
        with pytest.raises(PlanValidationError, match="fixes"):
            _extract_and_parse_plan(text)


class TestProviderResolution:
    def test_explicit_prefix(self):
        assert _provider_prefix("anthropic/claude-opus-4-7") == "anthropic"
        assert _provider_prefix("openai/gpt-4o") == "openai"

    def test_unprefixed_openai_names(self):
        assert _provider_prefix("gpt-4o") == "openai"
        assert _provider_prefix("o1-preview") == "openai"

    def test_unprefixed_claude_names(self):
        assert _provider_prefix("claude-opus-4-7") == "anthropic"


class TestFeatureEnabled:
    def test_disabled_with_no_keys(self, monkeypatch):
        for var in (
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
            "AZURE_API_KEY", "COHERE_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        assert feature_enabled({"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}}) is False

    def test_enabled_when_provider_key_set(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert feature_enabled({"flowbook": {"fix_model": "anthropic/claude-opus-4-7"}}) is True

    def test_default_model_when_unset(self):
        assert get_model({"flowbook": {}}) == DEFAULT_MODEL
        assert get_model(None) == DEFAULT_MODEL


class TestExtensionShapedSettings:
    """Production: web_app.settings['flowbook'] is the FlowBookExtension
    instance (jupyter_server's ExtensionApp publishes self there). Earlier
    versions of get_model() called .get() on it and crashed. This regression
    test pins down the attribute-style access path.
    """

    def test_get_model_reads_attribute_from_instance(self):
        class FakeExt:
            fix_model = "openai/gpt-4o"
        assert get_model({"flowbook": FakeExt()}) == "openai/gpt-4o"

    def test_get_model_falls_back_to_default_when_attr_missing(self):
        class FakeExt:
            pass  # no fix_model attribute
        assert get_model({"flowbook": FakeExt()}) == DEFAULT_MODEL

    def test_feature_enabled_with_instance_style_settings(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        class FakeExt:
            fix_model = "anthropic/claude-opus-4-7"
        assert feature_enabled({"flowbook": FakeExt()}) is True

    def test_handles_arbitrary_object_without_get_method(self):
        """Reproduces the original 500: object lacks .get(), must not crash."""
        class FakeExt:
            fix_model = "anthropic/claude-opus-4-7"
            # deliberately no `.get` attribute
        # Should return the model from the attribute, not raise AttributeError.
        assert get_model({"flowbook": FakeExt()}) == "anthropic/claude-opus-4-7"


class TestBuildContextFromNotebook:
    def _nb(self, cells):
        return {"cells": cells}

    def test_returns_none_when_no_violation(self):
        nb = self._nb([
            {"cell_type": "code", "id": "aaaa", "source": "x = 1", "metadata": {}},
        ])
        assert build_context_from_notebook(nb, "aaaa") is None

    def test_returns_none_when_cell_missing(self):
        nb = self._nb([
            {"cell_type": "code", "id": "aaaa", "source": "x = 1", "metadata": {}},
        ])
        assert build_context_from_notebook(nb, "zzzz") is None

    def test_extracts_first_violation(self):
        nb = self._nb([
            {"cell_type": "code", "id": "aaaa", "source": "x = 1", "metadata": {}},
            {
                "cell_type": "code", "id": "bbbb",
                "source": "train = pd.concat([train, extra])",
                "metadata": {"flowbook": {"errors": [{
                    "error_type": "no_read_and_write",
                    "locations": ["train"],
                    "causer_cell": "@aaaa",
                }]}},
            },
            {"cell_type": "code", "id": "cccc", "source": "y = train", "metadata": {}},
        ])
        ctx = build_context_from_notebook(nb, "bbbb")
        assert ctx is not None
        assert ctx.error_type == "no_read_and_write"
        assert ctx.locations == ["train"]
        assert ctx.causer_cells == ["A"]  # aaaa is the first code cell → @A
        assert ctx.cell_alpha == "B"
        assert "aaaa" in ctx.surrounding_sources
        assert "cccc" in ctx.surrounding_sources

    def test_indices_to_alpha_wraps(self):
        # 27 cells, the last is index 26 → 'AA'
        cells = [
            {"cell_type": "code", "id": f"c{i:03d}", "source": "", "metadata": {}}
            for i in range(26)
        ]
        cells.append({
            "cell_type": "code", "id": "last", "source": "x = 1",
            "metadata": {"flowbook": {"errors": [{
                "error_type": "no_read_and_write",
                "locations": ["x"],
            }]}},
        })
        ctx = build_context_from_notebook(self._nb(cells), "last")
        assert ctx is not None
        assert ctx.cell_alpha == "AA"
