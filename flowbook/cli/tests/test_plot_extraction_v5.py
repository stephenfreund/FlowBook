"""Tests for v5 plot extraction functions."""

import pytest
from flowbook.cli.models import V5CellMemory, V5MemoryResult
from flowbook.cli.plot_extraction import (
    extract_plot2_data_v5,
    extract_plot3_data_v5,
    extract_plot4_data_v5,
    extract_plot6_data_v5,
    extract_v5_memory_result,
)


class TestExtractPlot2DataV5:
    """Tests for extract_plot2_data_v5 (Checkpoint Time by Variable)."""

    def test_extracts_timing_series(self):
        """Extracts per-variable deepcopy timing as time series."""
        cells = [
            V5CellMemory('a', 0, 1.0, 0.0, 0.0, {}, {'x': 10.0, 'y': 5.0}),  # ms
            V5CellMemory('b', 1, 10.0, 0.0, 0.5, {'x': 0.5}, {'x': 15.0, 'y': 8.0}),
        ]
        result = extract_plot2_data_v5(cells)

        assert result is not None
        assert 'x' in result.var_series
        assert 'y' in result.var_series
        # Converted to seconds
        assert result.var_series['x'] == pytest.approx([0.010, 0.015])
        assert result.var_series['y'] == pytest.approx([0.005, 0.008])

    def test_orders_by_total_time(self):
        """Variables ordered by total time descending."""
        cells = [
            V5CellMemory('a', 0, 1.0, 0.0, 0.0, {}, {'slow': 100.0, 'fast': 10.0}),
        ]
        result = extract_plot2_data_v5(cells)

        assert result.vars_ordered[0] == 'slow'
        assert result.vars_ordered[1] == 'fast'

    def test_aggregates_other_beyond_top_n(self):
        """Variables beyond top_n aggregated as 'other'."""
        cells = [
            V5CellMemory('a', 0, 1.0, 0.0, 0.0, {},
                         {'a': 50.0, 'b': 40.0, 'c': 30.0, 'd': 20.0, 'e': 10.0}),
        ]
        result = extract_plot2_data_v5(cells, top_n=3)

        assert 'a' in result.var_series
        assert 'b' in result.var_series
        assert 'c' in result.var_series
        assert 'other' in result.var_series
        assert 'd' not in result.var_series
        assert 'e' not in result.var_series
        # other = d + e = 20 + 10 = 30ms = 0.030s
        assert result.var_series['other'] == pytest.approx([0.030])

    def test_empty_timing_returns_none(self):
        """Returns None when no timing data."""
        cells = [V5CellMemory('a', 0, 1.0, 0.0, 0.0, {}, {})]
        result = extract_plot2_data_v5(cells)
        assert result is None

    def test_empty_cells_returns_none(self):
        """Returns None for empty input."""
        result = extract_plot2_data_v5([])
        assert result is None


class TestExtractPlot3DataV5:
    """Tests for extract_plot3_data_v5."""

    def test_basic_extraction(self):
        """Extracts base_mb and overhead_mb correctly (FlowBook-only mode)."""
        cells = [
            V5CellMemory('a', 0, user_ns_mb=1.0, gpu_mb=0.0, checkpoint_mb=0.0),
            V5CellMemory('b', 1, user_ns_mb=10.0, gpu_mb=0.0, checkpoint_mb=0.5),
            V5CellMemory('c', 2, user_ns_mb=50.0, gpu_mb=0.0, checkpoint_mb=5.0),
        ]
        result = extract_plot3_data_v5(cells)

        assert result is not None
        assert result.cells == [1, 2, 3]
        assert result.base_mb == [1.0, 10.0, 50.0]
        assert result.overhead_mb == [0.0, 0.5, 5.0]  # Direct from checkpoint_mb
        assert result.has_baseline is False

    def test_cross_run_comparison(self):
        """Cross-run comparison: overhead = flowbook.total - baseline.total."""
        # FlowBook cells with checkpoint overhead
        fb_cells = [
            V5CellMemory('a', 0, user_ns_mb=10.0, gpu_mb=0.0, checkpoint_mb=5.0),  # total=15
            V5CellMemory('b', 1, user_ns_mb=20.0, gpu_mb=0.0, checkpoint_mb=10.0),  # total=30
        ]

        # Baseline cells (simulated - just need post.total_mb)
        class MockBaselineCell:
            def __init__(self, idx, total):
                self.cell_index = idx
                self.post = type('Post', (), {'total_mb': total})()

        baseline_cells = [
            MockBaselineCell(0, 10.0),  # baseline total=10
            MockBaselineCell(1, 18.0),  # baseline total=18
        ]

        result = extract_plot3_data_v5(fb_cells, baseline_cells)

        assert result is not None
        assert result.has_baseline is True
        # user_ns_mb and gpu_mb come from FlowBook cells
        assert result.user_ns_mb == [10.0, 20.0]
        assert result.gpu_mb == [0.0, 0.0]
        # base_mb = user_ns_mb + gpu_mb (FlowBook values)
        assert result.base_mb == [10.0, 20.0]
        # overhead = flowbook.total - baseline.total
        # Cell 0: 15 - 10 = 5
        # Cell 1: 30 - 18 = 12
        assert result.overhead_mb == [5.0, 12.0]

    def test_peak_overhead(self):
        """Finds peak overhead correctly."""
        cells = [
            V5CellMemory('a', 0, user_ns_mb=10.0, gpu_mb=0.0, checkpoint_mb=1.0),
            V5CellMemory('b', 1, user_ns_mb=20.0, gpu_mb=0.0, checkpoint_mb=5.0),
            V5CellMemory('c', 2, user_ns_mb=30.0, gpu_mb=0.0, checkpoint_mb=3.0),
        ]
        result = extract_plot3_data_v5(cells)

        assert result.peak_overhead_mb == 5.0
        assert result.peak_cell == 1  # Cell index 1 (0-indexed)
        # Peak % = 5.0 / 20.0 * 100 = 25%
        assert result.peak_overhead_pct == pytest.approx(25.0)

    def test_empty_cells_returns_none(self):
        """Returns None for empty input."""
        result = extract_plot3_data_v5([])
        assert result is None


class TestExtractPlot4DataV5:
    """Tests for extract_plot4_data_v5."""

    def test_extracts_per_variable_series(self):
        """Extracts per-variable checkpoint sizes."""
        cells = [
            V5CellMemory('a', 0, user_ns_mb=1.0, gpu_mb=0.0, checkpoint_mb=0.0, checkpoint_vars={}),
            V5CellMemory('b', 1, user_ns_mb=10.0, gpu_mb=0.0, checkpoint_mb=2.0, checkpoint_vars={'arr': 2.0}),
            V5CellMemory('c', 2, user_ns_mb=50.0, gpu_mb=0.0, checkpoint_mb=5.0, checkpoint_vars={'arr': 2.0, 'df': 3.0}),
        ]
        result = extract_plot4_data_v5(cells)

        assert result is not None
        assert 'arr' in result.var_series
        assert 'df' in result.var_series
        assert result.var_series['arr'] == [0.0, 2.0, 2.0]
        assert result.var_series['df'] == [0.0, 0.0, 3.0]

    def test_namespace_and_gpu_extracted(self):
        """Extracts namespace and GPU memory correctly."""
        cells = [
            V5CellMemory('a', 0, user_ns_mb=10.0, gpu_mb=5.0, checkpoint_mb=1.0, checkpoint_vars={'x': 1.0}),
            V5CellMemory('b', 1, user_ns_mb=20.0, gpu_mb=10.0, checkpoint_mb=2.0, checkpoint_vars={'x': 2.0}),
        ]
        result = extract_plot4_data_v5(cells)

        assert result.namespace_mb == [10.0, 20.0]
        assert result.gpu_mb == [5.0, 10.0]

    def test_top_n_aggregation(self):
        """Aggregates variables beyond top_n into 'other'."""
        # Create cells with 5 variables
        vars1 = {'a': 5.0, 'b': 4.0, 'c': 3.0, 'd': 2.0, 'e': 1.0}
        cells = [V5CellMemory('x', 0, user_ns_mb=1.0, gpu_mb=0.0, checkpoint_mb=15.0, checkpoint_vars=vars1)]

        result = extract_plot4_data_v5(cells, top_n=3)

        # Should have a, b, c, and "other"
        assert 'a' in result.var_series
        assert 'b' in result.var_series
        assert 'c' in result.var_series
        assert 'other' in result.var_series
        assert 'd' not in result.var_series
        assert 'e' not in result.var_series
        # other = d + e = 2 + 1 = 3
        assert result.var_series['other'] == [3.0]

    def test_no_checkpoint_vars_returns_none(self):
        """Returns None if no checkpoint variables."""
        cells = [V5CellMemory('a', 0, user_ns_mb=1.0, gpu_mb=0.0, checkpoint_mb=0.0)]
        result = extract_plot4_data_v5(cells)
        assert result is None


class TestExtractPlot6DataV5:
    """Tests for extract_plot6_data_v5."""

    def test_computes_checkpoint_delta_ratios(self):
        """Computes ratios correctly."""
        cells = [
            V5CellMemory('a', 0, user_ns_mb=10.0, gpu_mb=0.0, checkpoint_mb=1.0),
            V5CellMemory('b', 1, user_ns_mb=20.0, gpu_mb=0.0, checkpoint_mb=3.0),
        ]
        result = extract_plot6_data_v5(cells)

        assert result is not None
        assert result.cells == [1, 2]
        # Cell 0: delta=1.0, base=0 -> ratio=0 (base too small)
        # Cell 1: delta=3.0-1.0=2.0, base=10.0 -> ratio=0.2
        assert result.ratios[1] == pytest.approx(0.2)

    def test_small_base_gives_zero_ratio(self):
        """Ratio is 0 when prev base is below threshold."""
        cells = [
            V5CellMemory('a', 0, user_ns_mb=0.05, gpu_mb=0.0, checkpoint_mb=1.0),  # 50KB < 100KB threshold
            V5CellMemory('b', 1, user_ns_mb=10.0, gpu_mb=0.0, checkpoint_mb=3.0),
        ]
        result = extract_plot6_data_v5(cells)

        # prev_base_mb = 0.05 < MIN_BASE_MB (0.1), so ratio = 0
        assert result.ratios[1] == 0.0


class TestExtractV5MemoryResult:
    """Tests for extract_v5_memory_result."""

    def test_parses_v5_format(self):
        """Parses native v5 JSON correctly."""
        data = {
            "version": "5.0",
            "kernels": {
                "flowbook": {
                    "memory": {
                        "cells": [
                            {"cell_id": "a", "cell_index": 0, "user_ns_mb": 10.0,
                             "gpu_mb": 0.0, "checkpoint_mb": 5.0, "checkpoint_vars": {"arr": 5.0}},
                        ],
                        "rerun_cells": []
                    }
                }
            }
        }
        result = extract_v5_memory_result(data)

        assert result is not None
        assert len(result.cells) == 1
        assert result.cells[0].user_ns_mb == 10.0
        assert result.cells[0].checkpoint_mb == 5.0
        assert result.cells[0].checkpoint_vars == {"arr": 5.0}

    def test_converts_v4_format(self):
        """Converts v4 format to v5."""
        data = {
            "version": "4.0",
            "kernels": {
                "flowbook": {
                    "memory": {
                        "cells": [
                            {
                                "cell_id": "a", "cell_index": 0,
                                "post_user_ns_mb": 10.0, "post_gpu_mb": 0.0,
                                "post_overhead_mb": 5.0,
                                "post_checkpoint_vars": {
                                    "_post_a": {"arr": {"size_mb": 3.0}, "df": {"size_mb": 2.0}}
                                }
                            },
                        ],
                        "rerun_cells": []
                    }
                }
            }
        }
        result = extract_v5_memory_result(data)

        assert result is not None
        assert len(result.cells) == 1
        assert result.cells[0].user_ns_mb == 10.0
        assert result.cells[0].checkpoint_mb == 5.0
        # Variables aggregated across checkpoints
        assert result.cells[0].checkpoint_vars.get("arr") == pytest.approx(3.0)
        assert result.cells[0].checkpoint_vars.get("df") == pytest.approx(2.0)

    def test_no_memory_data_returns_none(self):
        """Returns None when no memory data present."""
        data = {"version": "5.0", "kernels": {"flowbook": {}}}
        result = extract_v5_memory_result(data)
        assert result is None
