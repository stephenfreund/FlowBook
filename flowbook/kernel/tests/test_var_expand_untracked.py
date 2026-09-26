"""
Regression tests: `!cmd` lines and magic arguments run IPython's var_expand untracked.

var_expand copies the caller's frame locals (``ns.update(frame.f_locals)``) to
expand ``{x}``/``$x``. At cell level those locals are the TrackingDict, so the
copy read every variable: each was recorded as a read of the cell, and any
read-blocked uncopyable variable (an open file ``f``, a figure ``fig``) raised
UncopyableReadError, failing a cell as plain as ``!head submission.csv``.
FlowbookKernel._patch_var_expand suspends tracking around the expansion.
"""

import os
from types import SimpleNamespace

import pytest

from flowbook.kernel.flowbook_kernel import FlowbookKernel
from flowbook.kernel_support.tracking import TrackingDict, UncopyableReadError


@pytest.fixture
def shell_ns():
    """An InteractiveShell whose user_ns is a TrackingDict with 'f' read-blocked.

    Returns (shell, tracking_dict, patch) where patch() applies the kernel's
    var_expand patch. The shared shell instance is restored afterwards.
    """
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    old_user_ns = shell.user_ns
    td = TrackingDict({"prefix": "abc", "x": 1, "f": object(), "__shell": shell})
    shell.user_ns = td
    td.block_variable("f")

    def patch():
        FlowbookKernel._patch_var_expand(SimpleNamespace(shell=shell), td)

    try:
        yield shell, td, patch
    finally:
        shell.__dict__.pop("var_expand", None)  # drop the instance-level patch
        shell.user_ns = old_user_ns


def run_cell(td, source):
    """Execute source the way the kernel runs a cell: the TrackingDict is both
    globals and locals, so a frame's f_locals is the TrackingDict itself."""
    td.reset_tracking()
    exec(source, td)


class TestVarExpandWithoutPatch:
    """Documents the bug the patch fixes."""

    def test_shell_command_raises_on_blocked_variable(self, shell_ns):
        shell, td, _ = shell_ns
        with pytest.raises(UncopyableReadError, match="'f'"):
            run_cell(td, "__shell.var_expand('head {prefix}.csv')")


class TestVarExpandUntracked:

    def test_shell_command_with_blocked_variable_runs(self, shell_ns):
        """The pgs501 failure: `!head submission.csv` after `fig` was blocked."""
        shell, td, patch = shell_ns
        patch()
        run_cell(td, "__shell.system('true')")
        assert td._real_ns["_exit_code"] == 0

    def test_expansion_still_substitutes_variables(self, shell_ns):
        shell, td, patch = shell_ns
        patch()
        run_cell(td, "__out = __shell.var_expand('head {prefix}.csv $x')")
        assert td._real_ns["__out"] == "head abc.csv 1"

    def test_expansion_records_no_reads(self, shell_ns):
        """The namespace copy is infrastructure, not a read of every variable."""
        shell, td, patch = shell_ns
        patch()
        run_cell(td, "__shell.var_expand('head {prefix}.csv')")
        assert not {"prefix", "x", "f"} & td.reads_before_writes

    def test_expansion_uses_callers_locals(self, shell_ns):
        """The wrapper frame is skipped: a `!cmd` inside a function sees its locals."""
        shell, td, patch = shell_ns
        patch()
        run_cell(td, "def g():\n    prefix = 'local'\n    return __shell.var_expand('{prefix}')\n__out = g()")
        assert td._real_ns["__out"] == "local"

    def test_line_magic_with_blocked_variable_runs(self, shell_ns, monkeypatch):
        """Magic arguments go through var_expand too (run_line_magic)."""
        shell, td, patch = shell_ns
        patch()
        monkeypatch.delenv("FLOWBOOK_TEST_VAR", raising=False)  # restores the environment after %env
        run_cell(td, "__shell.run_line_magic('env', 'FLOWBOOK_TEST_VAR={prefix}')")
        assert os.environ["FLOWBOOK_TEST_VAR"] == "abc"
        assert not {"prefix", "x", "f"} & td.reads_before_writes

    def test_blocked_variable_still_blocked_for_user_code(self, shell_ns):
        shell, td, patch = shell_ns
        patch()
        run_cell(td, "__shell.var_expand('echo')")
        with pytest.raises(UncopyableReadError, match="'f'"):
            run_cell(td, "f")

    def test_tracking_resumes_after_expansion(self, shell_ns):
        shell, td, patch = shell_ns
        patch()
        run_cell(td, "__shell.var_expand('echo {prefix}')\ny = x")
        assert "x" in td.reads_before_writes
