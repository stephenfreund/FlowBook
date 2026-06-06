"""
Tests for the meaningful-edit refinement of [Inst-Edit].

A source edit only marks a cell stale when it changes the cell's meaning. Edits
that touch only whitespace, blank lines, indentation, or comments are cosmetic
and leave the cell's status untouched; an edit that brings the source back to an
AST identical to the last execution clears the CODE_CHANGED reason.

Covers:
- FlowbookKernel._source_fingerprint (AST canonicalization, magics, syntax errors)
- NotebookState fingerprint storage (set/get/clear/delete)
- ReproducibilityEnforcer.clear_code_changed (symmetric clear)
- FlowbookKernel._process_cell_edit (end-to-end edit classification)
"""

import pytest

from flowbook.kernel.flowbook_kernel import FlowbookKernel
from flowbook.kernel.notebook_state import NotebookState
from flowbook.kernel.models import Reason, ReasonType
from flowbook.kernel.tests.conftest import ReproducibilityTestHelper


def _make_kernel(enforcer=None):
    """Build a FlowbookKernel shell with just enough wiring to test edits.

    Uses __new__ to skip the heavy IPython kernel init, then attaches a real
    IPython InteractiveShell (its input_transformer_manager makes magics
    fingerprint correctly) and captures outgoing flowbook messages.
    """
    from IPython.core.interactiveshell import InteractiveShell

    kernel = FlowbookKernel.__new__(FlowbookKernel)
    kernel.shell = InteractiveShell.instance()

    sent = []
    kernel._send_flowbook_message = lambda msg: sent.append(msg)
    kernel._sent_messages = sent
    if enforcer is not None:
        kernel._enforcer = enforcer
    return kernel


# ---------------------------------------------------------------------------
# _source_fingerprint
# ---------------------------------------------------------------------------


class TestSourceFingerprint:
    def setup_method(self):
        self.kernel = _make_kernel()

    def fp(self, src):
        return self.kernel._source_fingerprint(src)

    def test_comment_only_edit_is_equal(self):
        assert self.fp("x = 1  # original") == self.fp("x = 1  # changed comment")

    def test_added_comment_line_is_equal(self):
        assert self.fp("x = 1") == self.fp("# leading note\nx = 1")

    def test_whitespace_and_blank_lines_equal(self):
        assert self.fp("x=1") == self.fp("x = 1   ")
        assert self.fp("x = 1") == self.fp("\n\nx = 1\n\n")

    def test_reindented_block_is_equal(self):
        a = "def f():\n    return 1"
        b = "def f():\n        return 1"  # different indent width, same structure
        assert self.fp(a) == self.fp(b)

    def test_string_literal_change_differs(self):
        assert self.fp('x = "hello"') != self.fp('x = "world"')

    def test_real_code_change_differs(self):
        assert self.fp("x = 1") != self.fp("x = 2")
        assert self.fp("x = 1") != self.fp("y = 1")

    def test_underscore_numeric_literal_is_equal(self):
        assert self.fp("x = 1000") == self.fp("x = 1_000")

    def test_int_vs_float_literal_differs(self):
        assert self.fp("x = 1") != self.fp("x = 1.0")

    def test_magics_are_parseable_and_stable(self):
        a = "%timeout 5\nx = 1  # a"
        b = "%timeout 5\nx = 1  # b"
        assert self.fp(a) is not None
        assert self.fp(a) == self.fp(b)

    def test_shell_command_parseable(self):
        assert self.fp("!ls -la") is not None

    def test_syntax_error_returns_none(self):
        assert self.fp("x = ") is None
        assert self.fp("def f(:\n    pass") is None


# ---------------------------------------------------------------------------
# NotebookState fingerprint storage
# ---------------------------------------------------------------------------


class TestNotebookStateFingerprints:
    def test_set_and_get(self):
        ns = NotebookState()
        ns.set_fingerprint("a", "FP")
        assert ns.get_fingerprint("a") == "FP"

    def test_get_unknown_is_none(self):
        ns = NotebookState()
        assert ns.get_fingerprint("missing") is None

    def test_set_none_clears(self):
        ns = NotebookState()
        ns.set_fingerprint("a", "FP")
        ns.set_fingerprint("a", None)
        assert ns.get_fingerprint("a") is None

    def test_delete_pops_fingerprint(self):
        ns = NotebookState()
        ns.cell_order = ["a"]
        ns.set_fingerprint("a", "FP")
        ns.handle_delete("a")
        assert ns.get_fingerprint("a") is None

    def test_clear_drops_fingerprints(self):
        ns = NotebookState()
        ns.set_fingerprint("a", "FP")
        ns.clear()
        assert ns.get_fingerprint("a") is None


# ---------------------------------------------------------------------------
# Enforcer.clear_code_changed
# ---------------------------------------------------------------------------


class TestClearCodeChanged:
    def test_clear_only_code_changed_goes_clean(self):
        helper = ReproducibilityTestHelper()
        helper.set_cell_order(["a"])
        helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        sdc = helper.sdc
        sdc._notebook_state.set_stale("a", {Reason(ReasonType.CODE_CHANGED)})
        assert "a" in sdc.get_stale_cells()

        stale = sdc.clear_code_changed("a")
        assert "a" not in stale
        assert sdc._notebook_state.is_clean("a")

    def test_clear_preserves_other_reason(self):
        helper = ReproducibilityTestHelper()
        helper.set_cell_order(["a"])
        helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        sdc = helper.sdc
        sdc._notebook_state.set_stale(
            "a",
            {
                Reason(ReasonType.CODE_CHANGED),
                Reason(ReasonType.FORWARD_STALE, loc="x", cell_id="z"),
            },
        )

        stale = sdc.clear_code_changed("a")
        assert "a" in stale  # FORWARD_STALE keeps it stale
        reasons = {r.type for r in sdc._notebook_state.get_reasons("a")}
        assert ReasonType.CODE_CHANGED not in reasons
        assert ReasonType.FORWARD_STALE in reasons


# ---------------------------------------------------------------------------
# _process_cell_edit end-to-end
# ---------------------------------------------------------------------------


def _setup_executed_cell(source="x = 1"):
    """Return (kernel, enforcer) with cell 'a' executed and fingerprinted."""
    helper = ReproducibilityTestHelper()
    helper.set_cell_order(["a"])
    helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
    sdc = helper.sdc
    kernel = _make_kernel(enforcer=sdc)
    sdc.set_fingerprint("a", kernel._source_fingerprint(source))
    return kernel, sdc


class TestProcessCellEdit:
    def test_cosmetic_edit_stays_clean_no_message(self):
        kernel, sdc = _setup_executed_cell("x = 1")
        kernel._process_cell_edit("a", "x = 1   # just a comment")
        assert sdc._notebook_state.is_clean("a")
        assert kernel._sent_messages == []

    def test_whitespace_edit_stays_clean(self):
        kernel, sdc = _setup_executed_cell("x = 1")
        kernel._process_cell_edit("a", "\n\nx=1\n")
        assert sdc._notebook_state.is_clean("a")
        assert kernel._sent_messages == []

    def test_meaningful_edit_marks_stale_and_emits(self):
        kernel, sdc = _setup_executed_cell("x = 1")
        kernel._process_cell_edit("a", "x = 2")
        assert "a" in sdc.get_stale_cells()
        assert any(m.get("type") == "metadata" for m in kernel._sent_messages)

    def test_syntax_error_marks_stale(self):
        kernel, sdc = _setup_executed_cell("x = 1")
        kernel._process_cell_edit("a", "x = ")
        assert "a" in sdc.get_stale_cells()

    def test_revert_to_last_run_clears_staleness(self):
        kernel, sdc = _setup_executed_cell("x = 1")
        # First, a meaningful edit marks it stale.
        kernel._process_cell_edit("a", "x = 2")
        assert "a" in sdc.get_stale_cells()
        kernel._sent_messages.clear()
        # Editing back to source with the last-run AST clears CODE_CHANGED.
        kernel._process_cell_edit("a", "x = 1  # back to original")
        assert sdc._notebook_state.is_clean("a")
        assert any(m.get("type") == "metadata" for m in kernel._sent_messages)

    def test_cosmetic_edit_preserves_other_staleness(self):
        kernel, sdc = _setup_executed_cell("x = 1")
        sdc._notebook_state.set_stale(
            "a", {Reason(ReasonType.FORWARD_STALE, loc="x", cell_id="z")}
        )
        kernel._process_cell_edit("a", "x = 1  # comment only")
        # Still stale for the upstream reason; no spurious status flash.
        assert "a" in sdc.get_stale_cells()
        reasons = {r.type for r in sdc._notebook_state.get_reasons("a")}
        assert ReasonType.FORWARD_STALE in reasons
        assert kernel._sent_messages == []

    def test_unexecuted_cell_without_fingerprint_marks_stale(self):
        # No record / no fingerprint -> falls through to conservative mark.
        helper = ReproducibilityTestHelper()
        helper.set_cell_order(["a"])
        helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        sdc = helper.sdc
        kernel = _make_kernel(enforcer=sdc)
        # No set_fingerprint -> get_fingerprint is None -> mark stale on any edit.
        kernel._process_cell_edit("a", "x = 1  # comment")
        assert "a" in sdc.get_stale_cells()
