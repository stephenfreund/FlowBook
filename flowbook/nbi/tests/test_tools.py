"""Tests for FlowBook NBI tool implementations."""

import ast
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

pytest.importorskip(
    "notebook_intelligence",
    reason="notebook_intelligence not installed; skipping NBI tool tests",
)

from flowbook.nbi import tools
from flowbook.nbi.session import FlowBookSession


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_response():
    """Create a mock ChatResponse with a configurable run_ui_command."""
    response = AsyncMock()
    response.run_ui_command = AsyncMock()
    return response


@pytest.fixture
def mock_request():
    return MagicMock()


@pytest.fixture
def session():
    """Create a real FlowBookSession and install it as the module-level _session."""
    s = FlowBookSession()
    tools._session = s
    yield s
    tools._session = None


def _make_cell(source, cell_type='code'):
    """Helper to create a cell_data dict like the bridge returns."""
    return {'source': source, 'cell_type': cell_type, 'cell_id': 'abcd'}


async def _call(tool_obj, mock_response, mock_request, **kwargs):
    """Call a SimpleTool's underlying function with mock response/request."""
    return await tool_obj._tool_function(
        **kwargs, response=mock_response, request=mock_request
    )


# ==================================================================
# Category 1: Metadata & Status
# ==================================================================

class TestGetFlowbookMetadata:
    @pytest.mark.asyncio
    async def test_calls_bridge_with_parsed_index(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {'reads': [], 'writes': []}
        result = await _call(
            tools.get_flowbook_metadata, mock_response, mock_request, cell='@C'
        )
        mock_response.run_ui_command.assert_awaited_once_with(
            'flowbook:get-metadata', {'cellIndex': 2}
        )
        assert 'reads' in result

    @pytest.mark.asyncio
    async def test_numeric_cell_ref(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {'status': 'ok'}
        await _call(
            tools.get_flowbook_metadata, mock_response, mock_request, cell='0'
        )
        mock_response.run_ui_command.assert_awaited_once_with(
            'flowbook:get-metadata', {'cellIndex': 0}
        )


class TestGetNextActionableCell:
    @pytest.mark.asyncio
    async def test_calls_bridge(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {'index': 0, 'reason': 'stale'}
        result = await _call(
            tools.get_next_actionable_cell, mock_response, mock_request
        )
        mock_response.run_ui_command.assert_awaited_once_with(
            'flowbook:get-next-actionable', {}
        )
        assert 'stale' in result


class TestGetFlowbookStatus:
    @pytest.mark.asyncio
    async def test_calls_bridge(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {
            'total': 5, 'executed': 3, 'stale': 1, 'clean': 2
        }
        result = await _call(
            tools.get_flowbook_status, mock_response, mock_request
        )
        mock_response.run_ui_command.assert_awaited_once_with(
            'flowbook:get-status', {}
        )
        assert 'total' in result


# ==================================================================
# Category 2: Cell Operations
# ==================================================================

class TestReadCell:
    @pytest.mark.asyncio
    async def test_calls_bridge(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = _make_cell('x = 1')
        result = await _call(
            tools.read_cell, mock_response, mock_request, cell='@A'
        )
        mock_response.run_ui_command.assert_awaited_once_with(
            'flowbook:get-cell', {'cellIndex': 0}
        )
        assert 'x = 1' in result


class TestReadCellOutput:
    @pytest.mark.asyncio
    async def test_calls_bridge(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {'output': '42'}
        result = await _call(
            tools.read_cell_output, mock_response, mock_request, cell='@B'
        )
        mock_response.run_ui_command.assert_awaited_once_with(
            'flowbook:get-cell-output', {'cellIndex': 1}
        )
        assert '42' in result


class TestEditCellSource:
    @pytest.mark.asyncio
    async def test_calls_bridge(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {'ok': True}
        result = await _call(
            tools.edit_cell_source, mock_response, mock_request,
            cell='@C', source='y = 2'
        )
        mock_response.run_ui_command.assert_awaited_once_with(
            'flowbook:edit-cell-source', {'cellIndex': 2, 'source': 'y = 2'}
        )
        assert 'ok' in result


class TestAddCodeCell:
    @pytest.mark.asyncio
    async def test_calls_nbi_and_notifies(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {}
        result = await _call(
            tools.add_code_cell, mock_response, mock_request, source='z = 3'
        )
        assert mock_response.run_ui_command.await_count == 2
        mock_response.run_ui_command.assert_any_await(
            'notebook-intelligence:add-code-cell-to-active-notebook',
            {'source': 'z = 3'}
        )
        mock_response.run_ui_command.assert_any_await(
            'flowbook:notify-structure', {}
        )
        assert 'Added code cell' in result


class TestAddMarkdownCell:
    @pytest.mark.asyncio
    async def test_calls_nbi_and_notifies(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {}
        result = await _call(
            tools.add_markdown_cell, mock_response, mock_request,
            source='# Title'
        )
        assert mock_response.run_ui_command.await_count == 2
        mock_response.run_ui_command.assert_any_await(
            'notebook-intelligence:add-markdown-cell-to-active-notebook',
            {'source': '# Title'}
        )
        mock_response.run_ui_command.assert_any_await(
            'flowbook:notify-structure', {}
        )
        assert 'Added markdown cell' in result


class TestDeleteCell:
    @pytest.mark.asyncio
    async def test_calls_bridge(self, mock_response, mock_request):
        mock_response.run_ui_command.side_effect = [
            _make_cell('x = 1'),     # flowbook:get-cell
            {'code_cells': 5},       # flowbook:get-cell-count
            {},                      # delete-cell-at-index
            {},                      # flowbook:notify-structure
        ]
        result = await _call(
            tools.delete_cell, mock_response, mock_request, cell='@B'
        )
        assert 'Deleted cell @B' in result
        mock_response.run_ui_command.assert_any_await(
            'notebook-intelligence:delete-cell-at-index', {'cellIndex': 1}
        )


class TestGetCellCount:
    @pytest.mark.asyncio
    async def test_calls_bridge(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {
            'code_cells': 5, 'markdown_cells': 2, 'total': 7
        }
        result = await _call(
            tools.get_cell_count, mock_response, mock_request
        )
        mock_response.run_ui_command.assert_awaited_once_with(
            'flowbook:get-cell-count', {}
        )
        assert 'code_cells' in result


# ==================================================================
# Category 3: Execution
# ==================================================================

class TestRunCell:
    @pytest.mark.asyncio
    async def test_calls_bridge(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {'output': 'hello'}
        result = await _call(
            tools.run_cell, mock_response, mock_request, cell='@A'
        )
        mock_response.run_ui_command.assert_awaited_once_with(
            'flowbook:run-cell', {'cellIndex': 0}
        )
        assert 'hello' in result


class TestRunActionableCell:
    @pytest.mark.asyncio
    async def test_runs_next_actionable(self, mock_response, mock_request):
        mock_response.run_ui_command.side_effect = [
            {'index': 2, 'reason': 'stale'},  # get-next-actionable
            {'output': 'result'},               # run-cell
        ]
        result = await _call(
            tools.run_actionable_cell, mock_response, mock_request
        )
        mock_response.run_ui_command.assert_any_await(
            'flowbook:get-next-actionable', {}
        )
        mock_response.run_ui_command.assert_any_await(
            'flowbook:run-cell', {'cellIndex': 2}
        )

    @pytest.mark.asyncio
    async def test_returns_done_when_all_clean(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {'done': True}
        result = await _call(
            tools.run_actionable_cell, mock_response, mock_request
        )
        assert 'All cells are clean' in result
        # Should NOT have called run-cell
        assert mock_response.run_ui_command.await_count == 1


class TestRunActionableCells:
    @pytest.mark.asyncio
    async def test_calls_bridge(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {'ran': 3, 'errors': 0}
        result = await _call(
            tools.run_actionable_cells, mock_response, mock_request
        )
        mock_response.run_ui_command.assert_awaited_once_with(
            'flowbook:run-actionable-cells', {}
        )


class TestContinueAfterViolation:
    @pytest.mark.asyncio
    async def test_calls_bridge_true(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {}
        result = await _call(
            tools.continue_after_violation, mock_response, mock_request,
            enabled=True
        )
        mock_response.run_ui_command.assert_awaited_once_with(
            'flowbook:set-continue-after-violation', {'enabled': True}
        )
        assert 'True' in result

    @pytest.mark.asyncio
    async def test_calls_bridge_false(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {}
        result = await _call(
            tools.continue_after_violation, mock_response, mock_request,
            enabled=False
        )
        mock_response.run_ui_command.assert_awaited_once_with(
            'flowbook:set-continue-after-violation', {'enabled': False}
        )
        assert 'False' in result


# ==================================================================
# Category 4: Source Refactoring Tools
# ==================================================================

def _bridge(sources):
    """A fake `run_ui_command` backed by an in-memory list of code-cell sources.

    The unified refactoring tools snapshot the whole notebook (get-cell-count +
    get-cell for every cell), run a shared handler, then replay edit-cell-source
    by index. This responder serves those commands from `state`, so tests can
    assert on the resulting cell sources rather than a brittle call sequence.
    Returns ``(async_run_fn, state_list)``.
    """
    state = list(sources)

    async def run(cmd, params=None):
        params = params or {}
        if cmd == 'flowbook:get-cell-count':
            return {'total': len(state), 'code_cells': len(state), 'markdown_cells': 0}
        if cmd == 'flowbook:get-cell':
            i = params['cellIndex']
            return {
                'cell_id': f'c{i:02d}',
                'source': state[i],
                'cell_type': 'code',
                'label': index_to_alpha(i),
            }
        if cmd == 'flowbook:edit-cell-source':
            state[params['cellIndex']] = params['source']
            return {'ok': True, 'cell_id': f"c{params['cellIndex']:02d}"}
        return {}

    return run, state


def _install_bridge(mock_response, sources):
    run, state = _bridge(sources)
    mock_response.run_ui_command = AsyncMock(side_effect=run)
    return state


from flowbook.util.cell_index import index_to_alpha  # noqa: E402


class TestAlphaRename:
    @pytest.mark.asyncio
    async def test_renames_in_multiple_cells(self, mock_response, mock_request):
        """Rename 'x' to 'y' from @A; @C (no x) is untouched."""
        state = _install_bridge(mock_response, ['x = 1', 'print(x)', 'z = 42'])
        result = await _call(
            tools.alpha_rename, mock_response, mock_request,
            cell='@A', old_name='x', new_name='y'
        )
        assert '@A' in result and '@B' in result and '@C' not in result
        assert "'x' -> 'y'" in result and '2 cells' in result
        assert state == ['y = 1', 'print(y)', 'z = 42']

    @pytest.mark.asyncio
    async def test_no_occurrences(self, mock_response, mock_request):
        _install_bridge(mock_response, ['a = 1', 'b = 2'])
        result = await _call(
            tools.alpha_rename, mock_response, mock_request,
            cell='@A', old_name='x', new_name='y'
        )
        assert 'No occurrences' in result

    @pytest.mark.asyncio
    async def test_skips_empty_cells(self, mock_response, mock_request):
        state = _install_bridge(mock_response, ['', 'x = 1'])
        result = await _call(
            tools.alpha_rename, mock_response, mock_request,
            cell='@A', old_name='x', new_name='y'
        )
        assert '@B' in result and '1 cells' in result
        assert state == ['', 'y = 1']

    @pytest.mark.asyncio
    async def test_starts_from_specified_cell(self, mock_response, mock_request):
        """Starting from @B leaves @A's use of x alone."""
        state = _install_bridge(mock_response, ['x = 1', 'x = x + 1', 'print(x)'])
        await _call(
            tools.alpha_rename, mock_response, mock_request,
            cell='@B', old_name='x', new_name='y'
        )
        # @A untouched; @B and @C fully renamed (every x from @B onward).
        assert state == ['x = 1', 'y = y + 1', 'print(y)']


class TestRemoveInplace:
    @pytest.mark.asyncio
    async def test_removes_inplace(self, mock_response, mock_request):
        state = _install_bridge(mock_response, ["df.drop(columns=['x'], inplace=True)"])
        result = await _call(
            tools.remove_inplace, mock_response, mock_request,
            cell='@A', variable='df'
        )
        assert 'drop' in result and 'Removed inplace=True' in result
        assert 'inplace' not in state[0] and 'df = df.drop' in state[0]

    @pytest.mark.asyncio
    async def test_no_inplace_found(self, mock_response, mock_request):
        _install_bridge(mock_response, ['x = df.head()'])
        result = await _call(
            tools.remove_inplace, mock_response, mock_request,
            cell='@A', variable='df'
        )
        assert 'No inplace=True found' in result

    @pytest.mark.asyncio
    async def test_syntax_error_reports_no_effect(self, mock_response, mock_request):
        # Unparseable source: the shared handler tries a regex fallback, finds no
        # inplace call, and reports a no-op (rather than a parse error).
        _install_bridge(mock_response, ['def f(:\n  pass'])
        result = await _call(
            tools.remove_inplace, mock_response, mock_request,
            cell='@A', variable='df'
        )
        assert 'No inplace=True found' in result

    @pytest.mark.asyncio
    async def test_multiple_inplace_calls(self, mock_response, mock_request):
        source = "df.drop(columns=['a'], inplace=True)\ndf.fillna(0, inplace=True)"
        _install_bridge(mock_response, [source])
        result = await _call(
            tools.remove_inplace, mock_response, mock_request,
            cell='@A', variable='df'
        )
        assert 'drop' in result and 'fillna' in result


class TestInsertDeepcopy:
    @pytest.mark.asyncio
    async def test_inserts_deepcopy_and_renames_downstream(
        self, mock_response, mock_request
    ):
        state = _install_bridge(mock_response, [
            'df = load()',            # @A
            'print(df.head())',       # @B (start)
            'result = df.describe()', # @C uses df -> renamed
            'x = 42',                 # @D no df
        ])
        result = await _call(
            tools.insert_deepcopy, mock_response, mock_request,
            cell='@B', variable='df'
        )
        assert 'deepcopy' in result and '@B' in result and '@C' in result
        # New name is collision-safe: {var}_{cell_id} (cell_id 'c01' for @B).
        assert 'df_c01' in result
        assert 'import copy' in state[1] and 'copy.deepcopy(df)' in state[1]
        assert 'df_c01' in state[2]      # downstream renamed
        assert state[0] == 'df = load()' and state[3] == 'x = 42'  # untouched

    @pytest.mark.asyncio
    async def test_no_downstream_renames(self, mock_response, mock_request):
        state = _install_bridge(mock_response, ['a = 1', 'b = 2', 'print(df)', 'x = 1'])
        result = await _call(
            tools.insert_deepcopy, mock_response, mock_request,
            cell='@C', variable='df'
        )
        assert 'deepcopy' in result and '@D' not in result
        assert 'copy.deepcopy(df)' in state[2] and state[3] == 'x = 1'


class TestMarkDiagnostic:
    @pytest.mark.asyncio
    async def test_prepends_diagnostic(self, mock_response, mock_request):
        mock_response.run_ui_command.side_effect = [
            _make_cell('df.info()'),   # get-cell
            {'ok': True},              # edit-cell-source
        ]
        result = await _call(
            tools.mark_diagnostic, mock_response, mock_request, cell='@D'
        )
        assert 'Marked cell @D as diagnostic' in result
        edit_call = [
            c for c in mock_response.run_ui_command.call_args_list
            if c[0][0] == 'flowbook:edit-cell-source'
        ][0]
        new_source = edit_call[0][1]['source']
        assert new_source.startswith('%diagnostic')

    @pytest.mark.asyncio
    async def test_idempotent_already_marked(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = _make_cell('%diagnostic\ndf.info()')
        result = await _call(
            tools.mark_diagnostic, mock_response, mock_request, cell='@A'
        )
        assert 'already marked' in result
        # Should have only one call (get-cell), no edit
        assert mock_response.run_ui_command.await_count == 1

    @pytest.mark.asyncio
    async def test_idempotent_leading_whitespace(self, mock_response, mock_request):
        """Cells with leading whitespace before %diagnostic are NOT already marked
        (the code checks source.lstrip().startswith)."""
        mock_response.run_ui_command.side_effect = [
            _make_cell('  %diagnostic\ndf.info()'),
            {'ok': True},
        ]
        result = await _call(
            tools.mark_diagnostic, mock_response, mock_request, cell='@A'
        )
        assert 'already marked' in result


class TestMergeCells:
    @pytest.mark.asyncio
    async def test_merges_three_cells(self, mock_response, mock_request):
        mock_response.run_ui_command.side_effect = [
            # get-cell for each index (sorted: 0, 1, 2)
            _make_cell('x = 1'),
            _make_cell('y = 2'),
            _make_cell('z = 3'),
            # edit-cell-source for merged content into first cell
            {'ok': True},
            # delete-cell-at-index for index 2 (reversed)
            {},
            # delete-cell-at-index for index 1 (reversed)
            {},
            # flowbook:notify-structure
            {},
        ]
        result = await _call(
            tools.merge_cells, mock_response, mock_request,
            cells='@A,@B,@C'
        )
        assert 'Merged cells @A, @B, @C into @A' in result
        # Verify merged source
        edit_call = [
            c for c in mock_response.run_ui_command.call_args_list
            if c[0][0] == 'flowbook:edit-cell-source'
        ][0]
        merged = edit_call[0][1]['source']
        assert 'x = 1' in merged
        assert 'y = 2' in merged
        assert 'z = 3' in merged

    @pytest.mark.asyncio
    async def test_needs_at_least_two_cells(self, mock_response, mock_request):
        result = await _call(
            tools.merge_cells, mock_response, mock_request, cells='@A'
        )
        assert 'Need at least 2 cells' in result

    @pytest.mark.asyncio
    async def test_sorts_indices(self, mock_response, mock_request):
        """Even if refs are given out of order, cells are sorted."""
        mock_response.run_ui_command.side_effect = [
            _make_cell('y = 2'),     # sorted index 1
            _make_cell('z = 3'),     # sorted index 2
            {'ok': True},            # edit into index 1
            {},                      # delete index 2
            {},                      # notify-structure
        ]
        result = await _call(
            tools.merge_cells, mock_response, mock_request, cells='@C,@B'
        )
        # Should merge into the lower-indexed cell (@B)
        assert '@B' in result

    @pytest.mark.asyncio
    async def test_deletes_in_reverse_order(self, mock_response, mock_request):
        mock_response.run_ui_command.side_effect = [
            _make_cell('a'),
            _make_cell('b'),
            _make_cell('c'),
            {'ok': True},   # edit
            {},              # delete index 2
            {},              # delete index 1
            {},              # notify
        ]
        await _call(
            tools.merge_cells, mock_response, mock_request, cells='@A,@B,@C'
        )
        delete_calls = [
            c for c in mock_response.run_ui_command.call_args_list
            if c[0][0] == 'notebook-intelligence:delete-cell-at-index'
        ]
        # Should delete index 2 first, then index 1
        assert delete_calls[0][0][1]['cellIndex'] == 2
        assert delete_calls[1][0][1]['cellIndex'] == 1


class TestMoveCell:
    @pytest.mark.asyncio
    async def test_calls_bridge(self, mock_response, mock_request):
        mock_response.run_ui_command.side_effect = [
            {},  # flowbook:move-cell
            {},  # flowbook:notify-structure
        ]
        result = await _call(
            tools.move_cell, mock_response, mock_request,
            cell='@C', after_cell='@A'
        )
        mock_response.run_ui_command.assert_any_await(
            'flowbook:move-cell', {'fromIndex': 2, 'toIndex': 0}
        )
        mock_response.run_ui_command.assert_any_await(
            'flowbook:notify-structure', {}
        )
        assert 'Moved @C' in result
        assert 'after @A' in result


# ==================================================================
# Category 5: Notebook Lifecycle
# ==================================================================

class TestCreateNotebook:
    @pytest.mark.asyncio
    async def test_calls_bridge(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {'path': 'Untitled.ipynb'}
        result = await _call(
            tools.create_notebook, mock_response, mock_request
        )
        mock_response.run_ui_command.assert_awaited_once_with(
            'notebook-intelligence:create-new-notebook-from-py', {'code': ''}
        )
        assert 'Untitled.ipynb' in result


class TestSaveNotebook:
    @pytest.mark.asyncio
    async def test_calls_bridge(self, mock_response, mock_request):
        mock_response.run_ui_command.return_value = {}
        result = await _call(
            tools.save_notebook, mock_response, mock_request
        )
        mock_response.run_ui_command.assert_awaited_once_with('docmanager:save')
        assert 'Saved' in result


# ==================================================================
# Category 6: Checkpoint & Logging
# ==================================================================

class TestCheckpoint:
    @pytest.mark.asyncio
    async def test_saves_all_cells(self, mock_response, mock_request, session):
        mock_response.run_ui_command.side_effect = [
            {'code_cells': 2},
            _make_cell('x = 1'),
            _make_cell('y = 2'),
        ]
        result = await _call(
            tools.checkpoint, mock_response, mock_request
        )
        assert 'ckpt_0' in result
        assert '2 code cells' in result
        # Verify checkpoint was stored in session
        cells = session.get_checkpoint('ckpt_0')
        assert len(cells) == 2
        assert cells[0]['source'] == 'x = 1'
        assert cells[1]['source'] == 'y = 2'

    @pytest.mark.asyncio
    async def test_incremental_ids(self, mock_response, mock_request, session):
        for _ in range(2):
            mock_response.run_ui_command.side_effect = [
                {'code_cells': 1},
                _make_cell('x = 1'),
            ]
            await _call(tools.checkpoint, mock_response, mock_request)
        assert session.list_checkpoints()[0]['id'] == 'ckpt_0'
        assert session.list_checkpoints()[1]['id'] == 'ckpt_1'


class TestRestore:
    @pytest.mark.asyncio
    async def test_restores_cells(self, mock_response, mock_request, session):
        # Save a checkpoint first
        session.save_checkpoint([
            {'label': '@A', 'cell_type': 'code', 'source': 'original_a'},
            {'label': '@B', 'cell_type': 'code', 'source': 'original_b'},
        ])
        mock_response.run_ui_command.return_value = {'ok': True}
        result = await _call(
            tools.restore, mock_response, mock_request,
            checkpoint_id='ckpt_0'
        )
        assert 'Restored 2 cells' in result
        assert 'ckpt_0' in result
        # Verify edit-cell-source was called for each cell
        edit_calls = [
            c for c in mock_response.run_ui_command.call_args_list
            if c[0][0] == 'flowbook:edit-cell-source'
        ]
        assert len(edit_calls) == 2
        assert edit_calls[0][0][1] == {'cellIndex': 0, 'source': 'original_a'}
        assert edit_calls[1][0][1] == {'cellIndex': 1, 'source': 'original_b'}

    @pytest.mark.asyncio
    async def test_missing_checkpoint_raises(self, mock_response, mock_request, session):
        with pytest.raises(KeyError, match='not found'):
            await _call(
                tools.restore, mock_response, mock_request,
                checkpoint_id='nonexistent'
            )


class TestListCheckpoints:
    @pytest.mark.asyncio
    async def test_empty(self, mock_response, mock_request, session):
        result = await _call(
            tools.list_checkpoints, mock_response, mock_request
        )
        assert 'No checkpoints' in result

    @pytest.mark.asyncio
    async def test_with_checkpoints(self, mock_response, mock_request, session):
        session.save_checkpoint([
            {'label': '@A', 'cell_type': 'code', 'source': 'x = 1'},
        ])
        session.save_checkpoint([
            {'label': '@A', 'cell_type': 'code', 'source': 'a'},
            {'label': '@B', 'cell_type': 'code', 'source': 'b'},
        ])
        result = await _call(
            tools.list_checkpoints, mock_response, mock_request
        )
        assert 'ckpt_0' in result
        assert '1 cells' in result
        assert 'ckpt_1' in result
        assert '2 cells' in result


class TestGetLog:
    @pytest.mark.asyncio
    async def test_empty_log(self, mock_response, mock_request, session):
        result = await _call(tools.get_log, mock_response, mock_request)
        assert 'No events' in result

    @pytest.mark.asyncio
    async def test_with_events(self, mock_response, mock_request, session):
        session.log_event('run_cell', {'cell': '@A'}, 'ok', 100.0)
        result = await _call(tools.get_log, mock_response, mock_request)
        assert 'run_cell' in result


class TestPrintLog:
    @pytest.mark.asyncio
    async def test_returns_formatted_log(self, mock_response, mock_request, session):
        session.log_event('edit_cell', {'cell': '@B'}, 'done', 50.0)
        result = await _call(tools.print_log, mock_response, mock_request)
        assert 'edit_cell' in result


class TestInsertCell:
    @pytest.mark.asyncio
    async def test_calls_bridge(self, mock_response, mock_request):
        mock_response.run_ui_command.side_effect = [
            _make_cell('x = 1'),  # get-cell for @A
            {},                   # add-code-cell
            {},                   # notify-structure
        ]
        result = await _call(
            tools.insert_cell, mock_response, mock_request,
            after_cell='@A', cell_type='code', source='y = 2'
        )
        assert 'Inserted code cell after @A' in result
