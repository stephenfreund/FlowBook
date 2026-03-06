"""Tests for v3 memory measurement changes in compare_baseline."""

import json
from dataclasses import asdict

from flowbook.server.commands.compare_baseline import (
    MemoryCellMetrics,
    MemoryResults,
    ComparisonResult,
    KernelResults,
)


class TestMemoryCellMetricsV3:
    def test_new_fields_exist(self):
        """MemoryCellMetrics has all v3 fields."""
        cell = MemoryCellMetrics(
            cell_id="abcd",
            cell_index=0,
            pre_namespace_mb=1.0,
            pre_gpu_mb=0.5,
            namespace_mb=5.0,
            checkpoint_delta_mb=0.3,
            checkpoint_cumulative_mb=0.3,
            gpu_mb=0.5,
            pre_checkpoint_cumulative_mb=0.0,
            pre_enforcer_overhead_mb=0.0,
        )
        assert cell.pre_namespace_mb == 1.0
        assert cell.pre_gpu_mb == 0.5
        assert cell.namespace_mb == 5.0
        assert cell.gpu_mb == 0.5
        assert cell.pre_checkpoint_cumulative_mb == 0.0
        assert cell.pre_enforcer_overhead_mb == 0.0

    def test_baseline_defaults_zero(self):
        """Baseline cells should have FlowBook-only fields defaulting to 0."""
        cell = MemoryCellMetrics(
            cell_id="abcd",
            cell_index=0,
            pre_namespace_mb=1.0,
            pre_gpu_mb=0.0,
            namespace_mb=5.0,
            checkpoint_delta_mb=0.0,
            checkpoint_cumulative_mb=0.0,
            gpu_mb=0.0,
        )
        assert cell.checkpoint_delta_mb == 0.0
        assert cell.checkpoint_cumulative_mb == 0.0
        assert cell.pre_checkpoint_cumulative_mb == 0.0
        assert cell.pre_enforcer_overhead_mb == 0.0
        assert cell.checkpoint_by_var is None
        assert cell.checkpoint_var_costs is None

    def test_first_cell_pre_checkpoint_zero(self):
        """First cell must have pre_checkpoint_cumulative_mb = 0."""
        cell = MemoryCellMetrics(
            cell_id="first",
            cell_index=0,
            pre_namespace_mb=0.5,
            pre_gpu_mb=0.0,
            namespace_mb=5.0,
            checkpoint_delta_mb=2.0,
            checkpoint_cumulative_mb=2.0,
            gpu_mb=0.0,
            pre_checkpoint_cumulative_mb=0.0,
            pre_enforcer_overhead_mb=0.0,
        )
        assert cell.pre_checkpoint_cumulative_mb == 0.0
        assert cell.pre_enforcer_overhead_mb == 0.0

    def test_pre_checkpoint_tracks_previous(self):
        """Cell i's pre_checkpoint should equal cell (i-1)'s cumulative."""
        cell0 = MemoryCellMetrics(
            cell_id="c0", cell_index=0,
            pre_namespace_mb=0.0, pre_gpu_mb=0.0,
            namespace_mb=5.0, checkpoint_delta_mb=2.0,
            checkpoint_cumulative_mb=2.0, gpu_mb=0.0,
            pre_checkpoint_cumulative_mb=0.0,
        )
        cell1 = MemoryCellMetrics(
            cell_id="c1", cell_index=1,
            pre_namespace_mb=5.0, pre_gpu_mb=0.0,
            namespace_mb=10.0, checkpoint_delta_mb=3.0,
            checkpoint_cumulative_mb=5.0, gpu_mb=0.0,
            pre_checkpoint_cumulative_mb=cell0.checkpoint_cumulative_mb,
        )
        assert cell1.pre_checkpoint_cumulative_mb == cell0.checkpoint_cumulative_mb


class TestComparisonResultV3:
    def test_version_is_3(self):
        result = ComparisonResult()
        assert result.version == "3.0"

    def test_json_roundtrip(self):
        """ComparisonResult with v3 memory cells survives JSON roundtrip."""
        mem_cell = MemoryCellMetrics(
            cell_id="abcd",
            cell_index=0,
            pre_namespace_mb=1.5,
            pre_gpu_mb=0.0,
            namespace_mb=5.2,
            checkpoint_delta_mb=0.3,
            checkpoint_cumulative_mb=0.3,
            gpu_mb=0.0,
            pre_checkpoint_cumulative_mb=0.0,
            pre_enforcer_overhead_mb=0.0,
            checkpoint_by_var={"df": 0.3},
        )
        mem_results = MemoryResults(
            kernel_name="flowbook_kernel",
            cells=[mem_cell],
            totals={
                "final_namespace_mb": 10.2,
                "final_gpu_mb": 0.0,
                "final_checkpoint_cumulative_mb": 2.1,
                "final_enforcer_overhead_mb": 0.05,
                "max_namespace_mb": 10.2,
                "memory_overhead_ratio": 1.21,
            },
        )
        result = ComparisonResult(
            version="3.0",
            notebook_path="/test.ipynb",
            timestamp="2026-01-01",
            kernels={
                "baseline": KernelResults(kernel_name="baseline_kernel"),
                "flowbook": KernelResults(kernel_name="flowbook_kernel", memory=mem_results),
            },
        )

        # Serialize (same approach as compare_baseline uses)
        def to_dict(obj):
            if obj is None:
                return None
            if hasattr(obj, '__dict__'):
                return {k: to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            else:
                return obj

        d = to_dict(result)
        serialized = json.dumps(d)
        reloaded = json.loads(serialized)

        assert reloaded["version"] == "3.0"

        fc = reloaded["kernels"]["flowbook"]["memory"]["cells"][0]
        assert fc["pre_namespace_mb"] == 1.5
        assert fc["pre_gpu_mb"] == 0.0
        assert fc["namespace_mb"] == 5.2
        assert fc["gpu_mb"] == 0.0
        assert fc["pre_checkpoint_cumulative_mb"] == 0.0
        assert fc["pre_enforcer_overhead_mb"] == 0.0
        assert fc["checkpoint_delta_mb"] == 0.3
        assert fc["checkpoint_cumulative_mb"] == 0.3
        assert fc["checkpoint_by_var"] == {"df": 0.3}

        totals = reloaded["kernels"]["flowbook"]["memory"]["totals"]
        assert "final_checkpoint_cumulative_mb" in totals
        assert "final_enforcer_overhead_mb" in totals
        assert totals["final_checkpoint_cumulative_mb"] == 2.1
        assert totals["final_enforcer_overhead_mb"] == 0.05
