"""Tests for compare_baseline rerun overhead measurement."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict

from flowbook.server.commands.compare_baseline import (
    RerunOverheadMeasurement,
    RerunOverheadResult,
    measure_rerun_overhead,
)


class TestRerunOverheadDataclasses:
    """Test the rerun overhead dataclasses."""

    def test_rerun_overhead_measurement_defaults(self):
        """RerunOverheadMeasurement has correct defaults."""
        m = RerunOverheadMeasurement(
            iteration=0,
            cell_id="a",
            cell_index=0,
            checkpoint_ms=10.0,
            diff_ms=5.0,
            check_ms=2.0,
            total_overhead_ms=17.0,
        )
        assert m.iteration == 0
        assert m.cell_id == "a"
        assert m.cell_index == 0
        assert m.checkpoint_ms == 10.0
        assert m.diff_ms == 5.0
        assert m.check_ms == 2.0
        assert m.total_overhead_ms == 17.0
        assert m.checkpoint_by_var == {}
        assert m.checkpoint_var_costs == {}

    def test_rerun_overhead_result_defaults(self):
        """RerunOverheadResult has correct defaults."""
        r = RerunOverheadResult(
            rerun_n=3,
            quartile_indices=[0, 2, 4],
        )
        assert r.rerun_n == 3
        assert r.quartile_indices == [0, 2, 4]
        assert r.measurements == []

    def test_rerun_overhead_result_with_measurements(self):
        """RerunOverheadResult stores measurements correctly."""
        m1 = RerunOverheadMeasurement(
            iteration=0, cell_id="a", cell_index=0,
            checkpoint_ms=10.0, diff_ms=5.0, check_ms=2.0, total_overhead_ms=17.0,
        )
        m2 = RerunOverheadMeasurement(
            iteration=1, cell_id="a", cell_index=0,
            checkpoint_ms=12.0, diff_ms=4.0, check_ms=3.0, total_overhead_ms=19.0,
        )
        r = RerunOverheadResult(
            rerun_n=2,
            quartile_indices=[0],
            measurements=[m1, m2],
        )
        assert len(r.measurements) == 2
        assert r.measurements[0].checkpoint_ms == 10.0
        assert r.measurements[1].checkpoint_ms == 12.0


class TestQuartileIndexCalculation:
    """Test quartile index calculation logic."""

    def test_quartile_indices_single_cell(self):
        """With 1 cell, quartile index is [0]."""
        # Mock kernel client
        mock_client = MagicMock()
        code_cells = [{"id": "a"}]

        result = measure_rerun_overhead(mock_client, code_cells, rerun_n=0)
        # With rerun_n=0, should return empty result but with correct quartile_indices
        assert result.quartile_indices == [0]
        assert result.measurements == []

    def test_quartile_indices_empty(self):
        """With 0 cells, quartile indices is []."""
        mock_client = MagicMock()
        code_cells = []

        result = measure_rerun_overhead(mock_client, code_cells, rerun_n=1)
        assert result.quartile_indices == []
        assert result.measurements == []

    def test_quartile_indices_five_cells(self):
        """With 5 cells (0,1,2,3,4), quartile indices are [0,1,2,3,4]."""
        mock_client = MagicMock()
        code_cells = [{"id": f"cell_{i}"} for i in range(5)]

        result = measure_rerun_overhead(mock_client, code_cells, rerun_n=0)
        # K=5, (K-1)=4
        # 0, 4//4=1, 4//2=2, 3*4//4=3, 4
        assert result.quartile_indices == [0, 1, 2, 3, 4]

    def test_quartile_indices_ten_cells(self):
        """With 10 cells, quartile indices are [0, 2, 4, 6, 9]."""
        mock_client = MagicMock()
        code_cells = [{"id": f"cell_{i}"} for i in range(10)]

        result = measure_rerun_overhead(mock_client, code_cells, rerun_n=0)
        # K=10, (K-1)=9
        # 0, 9//4=2, 9//2=4, 3*9//4=6, 9
        assert result.quartile_indices == [0, 2, 4, 6, 9]

    def test_quartile_indices_two_cells(self):
        """With 2 cells, quartile indices are [0, 1]."""
        mock_client = MagicMock()
        code_cells = [{"id": "a"}, {"id": "b"}]

        result = measure_rerun_overhead(mock_client, code_cells, rerun_n=0)
        # K=2, (K-1)=1
        # 0, 1//4=0, 1//2=0, 3*1//4=0, 1 -> unique: {0, 1}
        assert result.quartile_indices == [0, 1]


class TestMeasureRerunOverheadFunction:
    """Test the measure_rerun_overhead function behavior."""

    def test_no_measurements_when_rerun_n_zero(self):
        """No measurements when rerun_n is 0."""
        mock_client = MagicMock()
        code_cells = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

        result = measure_rerun_overhead(mock_client, code_cells, rerun_n=0)
        assert result.rerun_n == 0
        assert result.measurements == []

    def test_measurement_count_matches_iterations_times_quartiles(self):
        """Number of measurements = rerun_n * len(quartile_indices)."""
        mock_client = MagicMock()

        # Setup mock to return overhead data
        def mock_get_iopub_msg(timeout=1.0):
            return {
                'parent_header': {'msg_id': mock_client.execute.return_value},
                'header': {'msg_type': 'display_data'},
                'content': {
                    'metadata': {
                        'flowbook': {
                            'rerun_overhead': {
                                'cell_id': 'test',
                                'checkpoint_ms': 1.0,
                                'diff_ms': 0.5,
                                'check_ms': 0.2,
                                'total_overhead_ms': 1.7,
                                'checkpoint_by_var': {},
                                'checkpoint_var_costs': {},
                            }
                        }
                    }
                },
            }

        mock_client.execute.return_value = "msg_1"
        mock_client.get_iopub_msg.side_effect = mock_get_iopub_msg

        code_cells = [{"id": f"cell_{i}"} for i in range(5)]
        result = measure_rerun_overhead(mock_client, code_cells, rerun_n=2, timeout=1.0)

        # 5 cells -> 5 quartile indices [0,1,2,3,4]
        # 2 iterations * 5 quartiles = 10 measurements
        expected_count = 2 * 5
        assert len(result.measurements) == expected_count
