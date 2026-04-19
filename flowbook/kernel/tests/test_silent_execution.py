"""Tests for silent-execution fast path in FlowbookKernel._do_execute_impl.

silent=True must bypass the entire reproducibility pipeline — no checkpoint, no
R/W tracking, no enforcer check, no staleness update, no flowbook_update IOPub
messages — while still forwarding the request (including user_expressions) to
IPython's do_execute. Protocol messages in cell_meta ("flowbook", "cell_order")
are still honored because they carry out-of-band configuration, not user code.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from flowbook.kernel.flowbook_kernel import FlowbookKernel


def _bare_kernel():
    """FlowbookKernel instance with no __init__ run — enough fields to drive
    _do_execute_impl's silent branch."""
    kernel = FlowbookKernel.__new__(FlowbookKernel)
    kernel._cell_id = "test-cell"
    kernel._enforcer = MagicMock(name="enforcer")
    kernel._checkpoints = MagicMock(name="checkpoints")
    kernel._display = MagicMock(name="display")
    kernel._handle_flowbook_message = MagicMock(name="_handle_flowbook_message")
    kernel._take_checkpoint = MagicMock(
        name="_take_checkpoint",
        side_effect=AssertionError("checkpoint must not be taken on silent path"),
    )
    kernel._restore_checkpoint = MagicMock(
        name="_restore_checkpoint",
        side_effect=AssertionError("restore must not run on silent path"),
    )
    kernel._display_execution_result = MagicMock(
        name="_display_execution_result",
        side_effect=AssertionError("metadata broadcast must not run on silent path"),
    )
    kernel._send_flowbook_message = MagicMock(
        name="_send_flowbook_message",
        side_effect=AssertionError("flowbook_update must not be sent on silent path"),
    )
    kernel._send_predicate_violation = MagicMock(
        name="_send_predicate_violation",
        side_effect=AssertionError("no violations on silent path"),
    )
    # _ipython_do_execute is async; record calls and return a canned reply.
    kernel._ipython_reply = {
        "status": "ok",
        "execution_count": 1,
        "payload": [],
        "user_expressions": {"_mem": {"status": "ok", "data": {"text/plain": "42"}}},
    }
    kernel._ipython_calls = []

    async def fake_ipython(code, silent, store_history=True,
                           user_expressions=None, allow_stdin=False, *,
                           cell_meta=None, cell_id=None):
        kernel._ipython_calls.append({
            "code": code, "silent": silent, "store_history": store_history,
            "user_expressions": user_expressions, "allow_stdin": allow_stdin,
            "cell_meta": cell_meta, "cell_id": cell_id,
        })
        return kernel._ipython_reply

    kernel._ipython_do_execute = fake_ipython
    return kernel


@pytest.mark.asyncio
async def test_silent_returns_ipython_result_unchanged():
    k = _bare_kernel()
    result = await k._do_execute_impl(
        "x = 1", silent=True, store_history=True,
        user_expressions=None, allow_stdin=False, cell_meta=None,
    )
    assert result is k._ipython_reply


@pytest.mark.asyncio
async def test_silent_forwards_user_expressions():
    k = _bare_kernel()
    await k._do_execute_impl(
        "", silent=True, store_history=False,
        user_expressions={"_mem": "_flowbook_measure_memory()"},
        allow_stdin=False, cell_meta=None,
    )
    assert len(k._ipython_calls) == 1
    call = k._ipython_calls[0]
    assert call["user_expressions"] == {"_mem": "_flowbook_measure_memory()"}
    assert call["silent"] is True
    assert call["cell_id"] == "test-cell"


@pytest.mark.asyncio
async def test_silent_forwards_non_empty_code():
    """Silent path should forward any code, not just the empty-code benchmark pattern."""
    k = _bare_kernel()
    await k._do_execute_impl(
        "y = 1 + 1", silent=True, store_history=True,
        user_expressions=None, allow_stdin=False, cell_meta=None,
    )
    assert k._ipython_calls[0]["code"] == "y = 1 + 1"


@pytest.mark.asyncio
async def test_silent_skips_checkpoint_and_enforcer():
    """The AssertionError side_effects on the mock methods fire if they're touched."""
    k = _bare_kernel()
    await k._do_execute_impl(
        "x = 1", silent=True, store_history=True,
        user_expressions=None, allow_stdin=False, cell_meta=None,
    )
    k._enforcer.check.assert_not_called()
    k._take_checkpoint.assert_not_called()
    k._restore_checkpoint.assert_not_called()
    k._display_execution_result.assert_not_called()
    k._send_flowbook_message.assert_not_called()


@pytest.mark.asyncio
async def test_silent_honors_flowbook_cell_meta():
    """Protocol messages are out-of-band config; silent must still apply them."""
    k = _bare_kernel()
    meta = {"flowbook": {"type": "sync"}}
    await k._do_execute_impl(
        "", silent=True, store_history=True,
        user_expressions=None, allow_stdin=False, cell_meta=meta,
    )
    k._handle_flowbook_message.assert_called_once_with({"type": "sync"})


@pytest.mark.asyncio
async def test_silent_honors_cell_order_cell_meta():
    k = _bare_kernel()
    meta = {"cell_order": ["a", "b", "c"]}
    await k._do_execute_impl(
        "", silent=True, store_history=True,
        user_expressions=None, allow_stdin=False, cell_meta=meta,
    )
    k._enforcer.set_cell_order.assert_called_once_with(["a", "b", "c"])


@pytest.mark.asyncio
async def test_silent_with_no_cell_id():
    """Benchmark probes often have no cell_id — must not blow up."""
    k = _bare_kernel()
    k._cell_id = None
    result = await k._do_execute_impl(
        "", silent=True, store_history=True,
        user_expressions={"_m": "1+1"}, allow_stdin=False, cell_meta=None,
    )
    assert result is k._ipython_reply
    assert k._ipython_calls[0]["cell_id"] is None


@pytest.mark.asyncio
async def test_silent_does_not_advance_enforcer_seq():
    """The seq_counter is an attribute on the enforcer; silent must not touch it."""
    k = _bare_kernel()
    # Track access to the counter — reads are fine, increments are not.
    k._enforcer.seq_counter = 7
    await k._do_execute_impl(
        "x = 1", silent=True, store_history=True,
        user_expressions=None, allow_stdin=False, cell_meta=None,
    )
    # Counter was not incremented by our code path.
    assert k._enforcer.seq_counter == 7
    # And no rollback happened on the enforcer.
    k._enforcer.rollback_last_check.assert_not_called()


@pytest.mark.asyncio
async def test_silent_without_isolate_does_not_checkpoint():
    """Without the isolate flag, silent stays on the cheap direct-passthrough
    path — no checkpoint (important for the _flowbook_measure_memory probe)."""
    k = _bare_kernel()
    await k._do_execute_impl(
        "x = 1", silent=True, store_history=True,
        user_expressions=None, allow_stdin=False, cell_meta=None,
    )
    k._take_checkpoint.assert_not_called()
    k._restore_checkpoint.assert_not_called()


@pytest.mark.asyncio
async def test_silent_with_isolate_checkpoints_and_restores():
    """cell_meta['flowbook_isolate']=True: checkpoint → run → restore."""
    k = _bare_kernel()
    # Replace the asserting mocks with plain mocks (the flag path USES them).
    k._take_checkpoint = MagicMock(name="_take_checkpoint")
    k._restore_checkpoint = MagicMock(name="_restore_checkpoint")
    k._apply_restore_memo = MagicMock(name="_apply_restore_memo")
    k._checkpoints.delete = MagicMock(name="delete")

    calls = []
    k._take_checkpoint.side_effect = lambda name: calls.append(("ckpt", name))
    k._restore_checkpoint.side_effect = lambda name: calls.append(("restore", name))
    k._checkpoints.delete.side_effect = lambda name: calls.append(("delete", name))

    original_ipython = k._ipython_do_execute
    async def tracked_ipython(*args, **kwargs):
        calls.append(("run",))
        return await original_ipython(*args, **kwargs)
    k._ipython_do_execute = tracked_ipython

    result = await k._do_execute_impl(
        "x = 1", silent=True, store_history=True,
        user_expressions=None, allow_stdin=False,
        cell_meta={"flowbook_isolate": True},
    )

    # Ordering: checkpoint → run → restore → delete
    steps = [c[0] for c in calls]
    assert steps == ["ckpt", "run", "restore", "delete"]

    # Same ckpt_id used for checkpoint, restore, and delete
    ckpt_name = calls[0][1]
    assert calls[2][1] == ckpt_name
    assert calls[3][1] == ckpt_name
    assert ckpt_name.startswith("__scratch__:")

    assert result is k._ipython_reply
    k._apply_restore_memo.assert_called_once()


@pytest.mark.asyncio
async def test_silent_with_isolate_restores_even_on_error():
    """If the inner execution raises, restore MUST still run."""
    k = _bare_kernel()
    k._take_checkpoint = MagicMock(name="_take_checkpoint")
    k._restore_checkpoint = MagicMock(name="_restore_checkpoint")
    k._apply_restore_memo = MagicMock(name="_apply_restore_memo")
    k._checkpoints.delete = MagicMock(name="delete")

    async def boom(*args, **kwargs):
        raise RuntimeError("kernel blew up")
    k._ipython_do_execute = boom

    with pytest.raises(RuntimeError, match="kernel blew up"):
        await k._do_execute_impl(
            "raise RuntimeError('boom')", silent=True, store_history=True,
            user_expressions=None, allow_stdin=False,
            cell_meta={"flowbook_isolate": True},
        )

    k._take_checkpoint.assert_called_once()
    k._restore_checkpoint.assert_called_once()
    k._apply_restore_memo.assert_called_once()
    k._checkpoints.delete.assert_called_once()


@pytest.mark.asyncio
async def test_silent_with_isolate_false_is_no_op():
    """flowbook_isolate=False (explicit) takes the plain silent path."""
    k = _bare_kernel()
    await k._do_execute_impl(
        "x = 1", silent=True, store_history=True,
        user_expressions=None, allow_stdin=False,
        cell_meta={"flowbook_isolate": False},
    )
    k._take_checkpoint.assert_not_called()
    k._restore_checkpoint.assert_not_called()


@pytest.mark.asyncio
async def test_silent_matches_benchmark_mem_pattern():
    """End-to-end shape of the benchmark_checkpoint.measure_memory() call:
    empty code + user_expressions + silent=True. Must return the user_expressions
    result untouched."""
    k = _bare_kernel()
    k._ipython_reply = {
        "status": "ok",
        "execution_count": 0,
        "user_expressions": {
            "_mem": {"status": "ok", "data": {"text/plain": "(123, 456, {}, 0.001)"}},
        },
    }
    result = await k._do_execute_impl(
        "", silent=True, store_history=False,
        user_expressions={"_mem": "_flowbook_measure_memory()"},
        allow_stdin=False, cell_meta=None,
    )
    assert result["user_expressions"]["_mem"]["status"] == "ok"
    assert "(123, 456" in result["user_expressions"]["_mem"]["data"]["text/plain"]
