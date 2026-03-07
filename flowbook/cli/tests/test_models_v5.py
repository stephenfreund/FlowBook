"""Tests for V5 simplified memory models."""

import pytest

from flowbook.cli.models import V5CellMemory, V5MemoryResult


class TestV5CellMemory:
    """Tests for V5CellMemory dataclass."""

    def test_from_dict_basic(self):
        """V5CellMemory parses JSON correctly."""
        data = {
            'cell_id': 'abc1',
            'cell_index': 0,
            'user_ns_mb': 42.1,
            'gpu_mb': 0.0,
            'checkpoint_mb': 5.3,
            'checkpoint_vars': {'arr': 3.2, 'df': 2.1}
        }
        cell = V5CellMemory.from_dict(data)

        assert cell.cell_id == 'abc1'
        assert cell.cell_index == 0
        assert cell.user_ns_mb == 42.1
        assert cell.gpu_mb == 0.0
        assert cell.checkpoint_mb == 5.3
        assert cell.checkpoint_vars['arr'] == 3.2
        assert cell.checkpoint_vars['df'] == 2.1

    def test_base_mb_property(self):
        """base_mb = user_ns_mb + gpu_mb."""
        cell = V5CellMemory(
            cell_id='a', cell_index=0,
            user_ns_mb=10.0, gpu_mb=5.0,
            checkpoint_mb=2.0
        )

        assert cell.base_mb == 15.0  # 10 + 5

    def test_total_mb_property(self):
        """total_mb = base_mb + checkpoint_mb."""
        cell = V5CellMemory(
            cell_id='a', cell_index=0,
            user_ns_mb=10.0, gpu_mb=5.0,
            checkpoint_mb=2.0
        )

        assert cell.total_mb == 17.0  # 10 + 5 + 2

    def test_to_dict_roundtrip(self):
        """to_dict and from_dict are inverses."""
        original = V5CellMemory(
            cell_id='xyz',
            cell_index=5,
            user_ns_mb=100.0,
            gpu_mb=50.0,
            checkpoint_mb=25.0,
            checkpoint_vars={'big_array': 20.0, 'small_df': 5.0}
        )

        d = original.to_dict()
        restored = V5CellMemory.from_dict(d)

        assert restored.cell_id == original.cell_id
        assert restored.cell_index == original.cell_index
        assert restored.user_ns_mb == original.user_ns_mb
        assert restored.gpu_mb == original.gpu_mb
        assert restored.checkpoint_mb == original.checkpoint_mb
        assert restored.checkpoint_vars == original.checkpoint_vars

    def test_from_dict_defaults(self):
        """Missing fields get default values."""
        data = {'cell_id': 'a', 'cell_index': 0}
        cell = V5CellMemory.from_dict(data)

        assert cell.user_ns_mb == 0.0
        assert cell.gpu_mb == 0.0
        assert cell.checkpoint_mb == 0.0
        assert cell.checkpoint_vars == {}


class TestV5MemoryResult:
    """Tests for V5MemoryResult dataclass."""

    def test_from_dict_multiple_cells(self):
        """V5MemoryResult parses JSON with multiple cells."""
        data = {
            'cells': [
                {'cell_id': 'a', 'cell_index': 0, 'user_ns_mb': 1.0, 'gpu_mb': 0.0,
                 'checkpoint_mb': 0.5, 'checkpoint_vars': {}},
                {'cell_id': 'b', 'cell_index': 1, 'user_ns_mb': 10.0, 'gpu_mb': 0.0,
                 'checkpoint_mb': 2.0, 'checkpoint_vars': {'x': 2.0}},
            ]
        }
        result = V5MemoryResult.from_dict(data)

        assert len(result.cells) == 2
        assert result.cells[0].cell_id == 'a'
        assert result.cells[1].checkpoint_mb == 2.0
        assert result.cells[1].checkpoint_vars == {'x': 2.0}

    def test_all_cells_includes_reruns(self):
        """all_cells combines cells and rerun_cells."""
        result = V5MemoryResult(
            cells=[
                V5CellMemory('a', 0, 1.0, 0.0, 0.5),
                V5CellMemory('b', 1, 2.0, 0.0, 1.0),
            ],
            rerun_cells=[
                V5CellMemory('a', 0, 1.5, 0.0, 0.6),
            ]
        )

        assert len(result.all_cells) == 3

    def test_final_checkpoint_mb(self):
        """final_checkpoint_mb returns last cell's checkpoint_mb."""
        result = V5MemoryResult(
            cells=[
                V5CellMemory('a', 0, 1.0, 0.0, 0.5),
                V5CellMemory('b', 1, 10.0, 0.0, 5.0),
            ]
        )

        assert result.final_checkpoint_mb == 5.0

    def test_final_user_ns_mb(self):
        """final_user_ns_mb returns last cell's user_ns_mb."""
        result = V5MemoryResult(
            cells=[
                V5CellMemory('a', 0, 1.0, 0.0, 0.5),
                V5CellMemory('b', 1, 10.0, 0.0, 5.0),
            ]
        )

        assert result.final_user_ns_mb == 10.0

    def test_peak_checkpoint_mb(self):
        """peak_checkpoint_mb returns max checkpoint_mb across all cells."""
        result = V5MemoryResult(
            cells=[
                V5CellMemory('a', 0, 1.0, 0.0, 2.0),
                V5CellMemory('b', 1, 10.0, 0.0, 8.0),
                V5CellMemory('c', 2, 5.0, 0.0, 3.0),
            ]
        )

        assert result.peak_checkpoint_mb == 8.0

    def test_to_dict_roundtrip(self):
        """to_dict and from_dict are inverses."""
        original = V5MemoryResult(
            cells=[
                V5CellMemory('a', 0, 1.0, 0.0, 0.5, {'x': 0.5}),
            ],
            rerun_cells=[
                V5CellMemory('a', 0, 1.5, 0.0, 0.6, {'x': 0.6}),
            ]
        )

        d = original.to_dict()
        restored = V5MemoryResult.from_dict(d)

        assert len(restored.cells) == 1
        assert len(restored.rerun_cells) == 1
        assert restored.cells[0].checkpoint_vars == {'x': 0.5}
        assert restored.rerun_cells[0].checkpoint_vars == {'x': 0.6}

    def test_empty_result(self):
        """Empty result has sane defaults."""
        result = V5MemoryResult()

        assert result.cells == []
        assert result.rerun_cells == []
        assert result.all_cells == []
        assert result.final_checkpoint_mb == 0.0
        assert result.final_user_ns_mb == 0.0
        assert result.peak_checkpoint_mb == 0.0
