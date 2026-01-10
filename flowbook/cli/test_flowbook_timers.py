"""Unit tests for flowbook_timers CLI tool."""

import json
import tempfile
from pathlib import Path

import pytest

from flowbook.cli.flowbook_timers import (
    load_timings,
    load_multiple_timings,
    group_by_key,
    calculate_stats,
    build_stats_table,
    combine_timings,
    format_time,
    format_table,
    format_json_single,
    format_csv_single,
    TimerStats,
)


class TestLoadTimings:
    """Test loading timing data from files."""

    def test_load_valid_json(self, tmp_path):
        """Test loading a valid JSON file."""
        file_path = tmp_path / "timings.json"
        data = [
            {"key": "timer1", "duration": 0.1},
            {"key": "timer2", "duration": 0.2},
        ]
        file_path.write_text(json.dumps(data))

        result = load_timings(str(file_path))
        assert len(result) == 2
        assert result[0]["key"] == "timer1"
        assert result[1]["duration"] == 0.2

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        result = load_timings("/nonexistent/file.json")
        assert result == []

    def test_load_invalid_json(self, tmp_path):
        """Test loading a file with invalid JSON."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("{invalid json")

        result = load_timings(str(file_path))
        assert result == []

    def test_load_non_list_json(self, tmp_path):
        """Test loading JSON that's not a list."""
        file_path = tmp_path / "notlist.json"
        file_path.write_text('{"key": "value"}')

        result = load_timings(str(file_path))
        assert result == []

    def test_load_missing_fields(self, tmp_path):
        """Test loading records with missing fields."""
        file_path = tmp_path / "missing.json"
        data = [
            {"key": "timer1", "duration": 0.1},
            {"key": "timer2"},  # Missing duration
            {"duration": 0.3},  # Missing key
            {"key": "timer3", "duration": 0.4},
        ]
        file_path.write_text(json.dumps(data))

        result = load_timings(str(file_path))
        assert len(result) == 2  # Only valid records
        assert result[0]["key"] == "timer1"
        assert result[1]["key"] == "timer3"


class TestLoadMultipleTimings:
    """Test loading from multiple files."""

    def test_load_multiple_valid_files(self, tmp_path):
        """Test loading multiple valid files."""
        file1 = tmp_path / "file1.json"
        file2 = tmp_path / "file2.json"

        file1.write_text(json.dumps([{"key": "a", "duration": 0.1}]))
        file2.write_text(json.dumps([{"key": "b", "duration": 0.2}]))

        result = load_multiple_timings([str(file1), str(file2)])

        assert len(result) == 2
        assert str(file1) in result
        assert str(file2) in result
        assert len(result[str(file1)]) == 1
        assert len(result[str(file2)]) == 1

    def test_load_with_some_invalid_files(self, tmp_path):
        """Test loading when some files are invalid."""
        file1 = tmp_path / "valid.json"
        file2 = tmp_path / "invalid.json"

        file1.write_text(json.dumps([{"key": "a", "duration": 0.1}]))
        file2.write_text("{invalid")

        result = load_multiple_timings([str(file1), str(file2)])

        assert len(result) == 1
        assert str(file1) in result
        assert str(file2) not in result


class TestGroupByKey:
    """Test grouping timings by key."""

    def test_group_single_key(self):
        """Test grouping with single timer key."""
        timings = [
            {"key": "timer1", "duration": 0.1},
            {"key": "timer1", "duration": 0.2},
            {"key": "timer1", "duration": 0.3},
        ]

        result = group_by_key(timings)

        assert len(result) == 1
        assert "timer1" in result
        assert result["timer1"] == [0.1, 0.2, 0.3]

    def test_group_multiple_keys(self):
        """Test grouping with multiple timer keys."""
        timings = [
            {"key": "timer1", "duration": 0.1},
            {"key": "timer2", "duration": 0.2},
            {"key": "timer1", "duration": 0.3},
            {"key": "timer2", "duration": 0.4},
        ]

        result = group_by_key(timings)

        assert len(result) == 2
        assert result["timer1"] == [0.1, 0.3]
        assert result["timer2"] == [0.2, 0.4]

    def test_group_empty_list(self):
        """Test grouping empty list."""
        result = group_by_key([])
        assert result == {}


class TestCalculateStats:
    """Test statistics calculation."""

    def test_calculate_stats_basic(self):
        """Test basic statistics calculation."""
        durations = [0.1, 0.2, 0.3, 0.4, 0.5]

        result = calculate_stats(durations)

        assert result["count"] == 5
        assert result["total"] == pytest.approx(1.5)
        assert result["mean"] == pytest.approx(0.3)
        assert result["median"] == pytest.approx(0.3)
        assert result["min"] == pytest.approx(0.1)
        assert result["max"] == pytest.approx(0.5)

    def test_calculate_stats_empty(self):
        """Test statistics on empty list."""
        result = calculate_stats([])

        assert result["count"] == 0
        assert result["total"] == 0.0
        assert result["mean"] == 0.0

    def test_calculate_stats_single_value(self):
        """Test statistics with single value."""
        result = calculate_stats([0.5])

        assert result["count"] == 1
        assert result["total"] == 0.5
        assert result["mean"] == 0.5
        assert result["median"] == 0.5
        assert result["min"] == 0.5
        assert result["max"] == 0.5
        assert result["std"] == 0.0


class TestBuildStatsTable:
    """Test building statistics table."""

    def test_build_stats_table(self):
        """Test building stats table from timings."""
        timings = [
            {"key": "timer1", "duration": 0.1},
            {"key": "timer1", "duration": 0.2},
            {"key": "timer2", "duration": 0.5},
        ]

        result = build_stats_table(timings)

        assert len(result) == 2
        assert all(isinstance(s, TimerStats) for s in result)

        timer1_stats = next(s for s in result if s.key == "timer1")
        assert timer1_stats.count == 2
        assert timer1_stats.total == pytest.approx(0.3)

        timer2_stats = next(s for s in result if s.key == "timer2")
        assert timer2_stats.count == 1
        assert timer2_stats.total == pytest.approx(0.5)

    def test_build_stats_table_empty(self):
        """Test building stats table from empty list."""
        result = build_stats_table([])
        assert result == []


class TestCombineTimings:
    """Test combining timings from multiple files."""

    def test_combine_timings(self):
        """Test combining timings from multiple files."""
        timings_by_file = {
            "file1.json": [
                {"key": "timer1", "duration": 0.1},
                {"key": "timer2", "duration": 0.2},
            ],
            "file2.json": [
                {"key": "timer1", "duration": 0.3},
            ],
        }

        result = combine_timings(timings_by_file)

        assert len(result) == 3
        assert {"key": "timer1", "duration": 0.1} in result
        assert {"key": "timer2", "duration": 0.2} in result
        assert {"key": "timer1", "duration": 0.3} in result

    def test_combine_empty(self):
        """Test combining from empty dict."""
        result = combine_timings({})
        assert result == []


class TestFormatTime:
    """Test time formatting."""

    def test_format_small_values(self):
        """Test formatting small time values in ms."""
        assert "0.50" == format_time(0.5)
        assert "0.10" == format_time(0.1)

    def test_format_milliseconds(self):
        """Test formatting times in milliseconds."""
        assert "5.00" == format_time(5.0)
        assert "500.00" == format_time(500.0)

    def test_format_large_values(self):
        """Test formatting large time values in ms."""
        assert "1000.00" == format_time(1000.0)
        assert "5500.00" == format_time(5500.0)


class TestFormatTable:
    """Test table formatting."""

    def test_format_table_basic(self):
        """Test basic table formatting."""
        stats = [
            TimerStats(
                key="timer1",
                count=2,
                total=0.3,
                mean=0.15,
                median=0.15,
                min=0.1,
                max=0.2,
                std=0.05,
                p95=0.195,
            ),
        ]

        result = format_table(stats)

        assert "Timer Statistics (all times in ms)" in result
        assert "timer1" in result
        assert "2" in result  # count
        assert "TOTAL" in result

    def test_format_table_with_title(self):
        """Test table formatting with custom title."""
        stats = [
            TimerStats(
                key="timer1",
                count=1,
                total=0.1,
                mean=0.1,
                median=0.1,
                min=0.1,
                max=0.1,
                std=0.0,
                p95=0.1,
            ),
        ]

        result = format_table(stats, title="Custom Title")

        assert "Custom Title" in result
        assert "Timer Statistics (all times in ms)" not in result

    def test_format_table_empty(self):
        """Test formatting empty stats."""
        result = format_table([])
        assert "No timing data available" in result

    def test_format_table_with_top_limit(self):
        """Test table with top N limit."""
        stats = [
            TimerStats("timer1", 1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0, 0.3),
            TimerStats("timer2", 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.2),
            TimerStats("timer3", 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1),
        ]

        result = format_table(stats, top=2)

        # Should only show top 2 by total
        assert "timer1" in result
        assert "timer2" in result
        assert "timer3" not in result


class TestFormatJson:
    """Test JSON formatting."""

    def test_format_json_single(self):
        """Test JSON formatting for single file."""
        stats = [
            TimerStats("timer1", 2, 0.3, 0.15, 0.15, 0.1, 0.2, 0.05, 0.195),
        ]
        timings = [
            {"key": "timer1", "duration": 0.1},
            {"key": "timer1", "duration": 0.2},
        ]

        result = format_json_single(stats, timings)

        data = json.loads(result)
        assert "summary" in data
        assert data["summary"]["total_records"] == 2
        assert data["summary"]["unique_timers"] == 1
        assert len(data["timers"]) == 1
        assert data["timers"][0]["key"] == "timer1"


class TestFormatCsv:
    """Test CSV formatting."""

    def test_format_csv_single(self):
        """Test CSV formatting for single file."""
        stats = [
            TimerStats("timer1", 2, 0.3, 0.15, 0.15, 0.1, 0.2, 0.05, 0.195),
            TimerStats("timer2", 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5),
        ]

        result = format_csv_single(stats)

        lines = result.split("\n")
        assert lines[0] == "key,count,total,mean,median,min,max,std,p95"
        assert "timer1" in lines[1]
        assert "timer2" in lines[2]
        assert len(lines) == 3


class TestIntegration:
    """Integration tests with real workflow."""

    def test_full_workflow_single_file(self, tmp_path):
        """Test complete workflow with single file."""
        file_path = tmp_path / "timings.json"
        data = [
            {"key": "operation1", "duration": 0.1},
            {"key": "operation1", "duration": 0.2},
            {"key": "operation2", "duration": 0.5},
        ]
        file_path.write_text(json.dumps(data))

        # Load
        timings = load_timings(str(file_path))
        assert len(timings) == 3

        # Build stats
        stats = build_stats_table(timings)
        assert len(stats) == 2

        # Format table
        table = format_table(stats)
        assert "operation1" in table
        assert "operation2" in table

        # Format JSON
        json_output = format_json_single(stats, timings)
        data = json.loads(json_output)
        assert data["summary"]["total_records"] == 3

        # Format CSV
        csv_output = format_csv_single(stats)
        assert "operation1" in csv_output

    def test_full_workflow_multiple_files(self, tmp_path):
        """Test complete workflow with multiple files."""
        file1 = tmp_path / "file1.json"
        file2 = tmp_path / "file2.json"

        file1.write_text(json.dumps([
            {"key": "timer_a", "duration": 0.1},
            {"key": "timer_b", "duration": 0.2},
        ]))

        file2.write_text(json.dumps([
            {"key": "timer_a", "duration": 0.15},
            {"key": "timer_c", "duration": 0.3},
        ]))

        # Load multiple
        timings_by_file = load_multiple_timings([str(file1), str(file2)])
        assert len(timings_by_file) == 2

        # Combine
        combined = combine_timings(timings_by_file)
        assert len(combined) == 4

        # Build combined stats
        stats = build_stats_table(combined)
        assert len(stats) == 3  # timer_a, timer_b, timer_c

        timer_a = next(s for s in stats if s.key == "timer_a")
        assert timer_a.count == 2  # From both files


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
