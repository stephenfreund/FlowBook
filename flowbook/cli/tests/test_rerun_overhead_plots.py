"""Tests for rerun overhead extraction and plotting functions."""

import pytest
from typing import Dict, List

from flowbook.cli.compare_overhead import (
    extract_rerun_overhead_data,
    RerunOverheadCDFData,
)


class TestExtractRerunOverheadData:
    """Tests for extract_rerun_overhead_data function."""

    def test_returns_none_when_no_data(self):
        """Returns None when no raw data provided."""
        assert extract_rerun_overhead_data([]) is None

    def test_returns_none_when_no_rerun_overhead(self):
        """Returns None when no rerun_overhead section in data."""
        raw_data = [
            {"kernels": {"flowbook": {"timing": {}}}},
            {"kernels": {"baseline": {"timing": {}}}},
        ]
        assert extract_rerun_overhead_data(raw_data) is None

    def test_extracts_single_measurement(self):
        """Extracts data from a single measurement."""
        raw_data = [
            {
                "rerun_overhead": {
                    "rerun_n": 1,
                    "quartile_indices": [0],
                    "measurements": [
                        {
                            "iteration": 0,
                            "cell_id": "a",
                            "cell_index": 0,
                            "checkpoint_ms": 10.0,
                            "diff_ms": 5.0,
                            "check_ms": 2.0,
                            "total_overhead_ms": 17.0,
                        }
                    ],
                }
            }
        ]
        result = extract_rerun_overhead_data(raw_data)

        assert result is not None
        assert result.total_overhead_ms == [17.0]
        assert result.checkpoint_ms == [10.0]
        assert result.diff_ms == [5.0]
        assert result.check_ms == [2.0]

    def test_extracts_multiple_measurements(self):
        """Extracts data from multiple measurements."""
        raw_data = [
            {
                "rerun_overhead": {
                    "rerun_n": 2,
                    "quartile_indices": [0, 4],
                    "measurements": [
                        {"checkpoint_ms": 10.0, "diff_ms": 5.0, "check_ms": 2.0, "total_overhead_ms": 17.0},
                        {"checkpoint_ms": 12.0, "diff_ms": 6.0, "check_ms": 3.0, "total_overhead_ms": 21.0},
                        {"checkpoint_ms": 8.0, "diff_ms": 4.0, "check_ms": 1.0, "total_overhead_ms": 13.0},
                        {"checkpoint_ms": 15.0, "diff_ms": 7.0, "check_ms": 4.0, "total_overhead_ms": 26.0},
                    ],
                }
            }
        ]
        result = extract_rerun_overhead_data(raw_data)

        assert result is not None
        assert len(result.total_overhead_ms) == 4
        assert result.total_overhead_ms == [17.0, 21.0, 13.0, 26.0]

    def test_aggregates_across_multiple_files(self):
        """Aggregates rerun overhead data from multiple comparison files."""
        raw_data = [
            {
                "rerun_overhead": {
                    "rerun_n": 1,
                    "quartile_indices": [0],
                    "measurements": [
                        {"checkpoint_ms": 10.0, "diff_ms": 5.0, "check_ms": 2.0, "total_overhead_ms": 17.0},
                    ],
                }
            },
            {
                "rerun_overhead": {
                    "rerun_n": 1,
                    "quartile_indices": [0],
                    "measurements": [
                        {"checkpoint_ms": 20.0, "diff_ms": 10.0, "check_ms": 5.0, "total_overhead_ms": 35.0},
                    ],
                }
            },
        ]
        result = extract_rerun_overhead_data(raw_data)

        assert result is not None
        assert len(result.total_overhead_ms) == 2
        assert 17.0 in result.total_overhead_ms
        assert 35.0 in result.total_overhead_ms

    def test_cdf_data_is_sorted(self):
        """total_sorted is properly sorted for CDF."""
        raw_data = [
            {
                "rerun_overhead": {
                    "rerun_n": 1,
                    "quartile_indices": [0, 1],
                    "measurements": [
                        {"total_overhead_ms": 30.0, "checkpoint_ms": 0, "diff_ms": 0, "check_ms": 0},
                        {"total_overhead_ms": 10.0, "checkpoint_ms": 0, "diff_ms": 0, "check_ms": 0},
                        {"total_overhead_ms": 20.0, "checkpoint_ms": 0, "diff_ms": 0, "check_ms": 0},
                    ],
                }
            }
        ]
        result = extract_rerun_overhead_data(raw_data)

        assert result is not None
        assert result.total_sorted == [10.0, 20.0, 30.0]
        assert result.total_percentiles == [1/3, 2/3, 1.0]

    def test_skips_files_without_rerun_overhead(self):
        """Skips files that don't have rerun_overhead section."""
        raw_data = [
            {"kernels": {"flowbook": {}}},  # No rerun_overhead
            {
                "rerun_overhead": {
                    "rerun_n": 1,
                    "quartile_indices": [0],
                    "measurements": [
                        {"checkpoint_ms": 10.0, "diff_ms": 5.0, "check_ms": 2.0, "total_overhead_ms": 17.0},
                    ],
                }
            },
        ]
        result = extract_rerun_overhead_data(raw_data)

        assert result is not None
        assert len(result.total_overhead_ms) == 1
        assert result.total_overhead_ms == [17.0]


class TestRerunOverheadCDFData:
    """Tests for RerunOverheadCDFData dataclass."""

    def test_dataclass_fields(self):
        """RerunOverheadCDFData has expected fields."""
        data = RerunOverheadCDFData(
            total_overhead_ms=[10.0, 20.0],
            total_sorted=[10.0, 20.0],
            total_percentiles=[0.5, 1.0],
            checkpoint_ms=[5.0, 10.0],
            diff_ms=[3.0, 6.0],
            check_ms=[2.0, 4.0],
        )
        assert data.total_overhead_ms == [10.0, 20.0]
        assert data.total_sorted == [10.0, 20.0]
        assert data.total_percentiles == [0.5, 1.0]
        assert data.checkpoint_ms == [5.0, 10.0]
        assert data.diff_ms == [3.0, 6.0]
        assert data.check_ms == [2.0, 4.0]
