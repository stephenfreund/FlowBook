"""Drift guards for the unified prompt/catalog (Phase 2) and the validation
allowlist (Phase 3).

These keep the single sources honest: the generated doc, the LLM prompt, and
the fix-plan validation contract must all stay in lockstep with REGISTRY.
"""

import typing
from pathlib import Path

from flowbook.tools import names
from flowbook.tools.prompt import render_tool_catalog, render_tool_catalog_doc


_DOC = Path(__file__).resolve().parents[2] / "docs" / "_generated" / "tool_catalog.md"


def test_catalog_doc_in_sync_with_registry():
    """flowbook/docs/_generated/tool_catalog.md must match the registry.

    If this fails, regenerate it:
        python -c "from flowbook.tools.prompt import render_tool_catalog_doc as r; \
            open('flowbook/docs/_generated/tool_catalog.md','w').write(r())"
    """
    assert _DOC.exists(), f"generated catalog missing at {_DOC}"
    assert _DOC.read_text(encoding="utf-8") == render_tool_catalog_doc()


def test_catalog_lists_every_tool():
    catalog = render_tool_catalog()
    for name in names():
        assert f"`{name}(" in catalog


def test_mutator_overlapping_schemas_come_from_registry():
    """The custom-fix mutator tools that overlap the registry (merge_cells,
    move_cell, mark_diagnostic) must expose the registry's arg schema, since
    they are applied via apply_fix()."""
    from flowbook.server.fix_tools_mutator import TOOL_SCHEMAS
    from flowbook.tools import get

    by_name = {s["function"]["name"]: s["function"] for s in TOOL_SCHEMAS}
    for name in ("merge_cells", "move_cell", "mark_diagnostic"):
        assert by_name[name]["parameters"] == get(name).parameters
        assert by_name[name]["description"] == get(name).description


def test_fixtoolname_matches_registry():
    """fix_models.FixToolName (a static Literal) must equal the registry names."""
    from flowbook.server.fix_models import FixToolName, TOOL_ARG_SCHEMAS

    literal_names = set(typing.get_args(FixToolName))
    assert literal_names == set(names())
    # And the derived arg-schemas cover exactly those tools.
    assert set(TOOL_ARG_SCHEMAS) == set(names())
