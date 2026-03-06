"""Tests for compare_overhead_v3 module."""

import json
import copy
import pytest

from flowbook.cli.compare_overhead import is_v3_format, is_v2_format
from flowbook.cli.compare_overhead_v3 import (
    _compute_cross_run_overhead,
    _compute_fallback_ratios,
    compute_file_stats_v3,
    MIN_MEANINGFUL_BASE_MB,
)


def _make_v3_data(
    num_cells=4,
    baseline_pre_ns=None,
    baseline_pre_gpu=None,
    baseline_post_ns=None,
    baseline_post_gpu=None,
    flowbook_pre_ns=None,
    flowbook_pre_gpu=None,
    flowbook_post_ns=None,
    flowbook_post_gpu=None,
    flowbook_pre_ckpt=None,
    flowbook_pre_enforcer=None,
    flowbook_ckpt_delta=None,
    flowbook_ckpt_cumulative=None,
    baseline_final_ns=None,
    baseline_final_gpu=None,
    flowbook_final_ns=None,
    flowbook_final_gpu=None,
    flowbook_final_ckpt=None,
    flowbook_final_enforcer=None,
    include_baseline_memory=True,
):
    """Build a minimal v3 comparison data dict for testing."""
    if baseline_pre_ns is None:
        baseline_pre_ns = [10.0, 15.0, 20.0, 25.0][:num_cells]
    if baseline_pre_gpu is None:
        baseline_pre_gpu = [0.0] * num_cells
    if baseline_post_ns is None:
        baseline_post_ns = [15.0, 20.0, 25.0, 30.0][:num_cells]
    if baseline_post_gpu is None:
        baseline_post_gpu = [0.0] * num_cells
    if flowbook_pre_ns is None:
        flowbook_pre_ns = [10.0, 15.0, 20.0, 25.0][:num_cells]
    if flowbook_pre_gpu is None:
        flowbook_pre_gpu = [0.0] * num_cells
    if flowbook_post_ns is None:
        flowbook_post_ns = [15.0, 20.0, 25.0, 30.0][:num_cells]
    if flowbook_post_gpu is None:
        flowbook_post_gpu = [0.0] * num_cells
    if flowbook_pre_ckpt is None:
        flowbook_pre_ckpt = [0.0, 2.0, 5.0, 8.0][:num_cells]
    if flowbook_pre_enforcer is None:
        flowbook_pre_enforcer = [0.0, 0.01, 0.02, 0.03][:num_cells]
    if flowbook_ckpt_delta is None:
        flowbook_ckpt_delta = [2.0, 3.0, 3.0, 4.0][:num_cells]
    if flowbook_ckpt_cumulative is None:
        flowbook_ckpt_cumulative = [2.0, 5.0, 8.0, 12.0][:num_cells]

    if baseline_final_ns is None:
        baseline_final_ns = baseline_post_ns[-1]
    if baseline_final_gpu is None:
        baseline_final_gpu = 0.0
    if flowbook_final_ns is None:
        flowbook_final_ns = flowbook_post_ns[-1]
    if flowbook_final_gpu is None:
        flowbook_final_gpu = 0.0
    if flowbook_final_ckpt is None:
        flowbook_final_ckpt = flowbook_ckpt_cumulative[-1]
    if flowbook_final_enforcer is None:
        flowbook_final_enforcer = flowbook_pre_enforcer[-1] + 0.01

    def _make_mem_cells(pre_ns, pre_gpu, post_ns, post_gpu, pre_ckpt, pre_enf, ckpt_delta, ckpt_cum):
        cells = []
        for i in range(num_cells):
            cells.append({
                "cell_id": f"c{i}",
                "cell_index": i,
                "pre_namespace_mb": pre_ns[i],
                "pre_gpu_mb": pre_gpu[i],
                "namespace_mb": post_ns[i],
                "gpu_mb": post_gpu[i],
                "checkpoint_delta_mb": ckpt_delta[i] if ckpt_delta else 0.0,
                "checkpoint_cumulative_mb": ckpt_cum[i] if ckpt_cum else 0.0,
                "pre_checkpoint_cumulative_mb": pre_ckpt[i] if pre_ckpt else 0.0,
                "pre_enforcer_overhead_mb": pre_enf[i] if pre_enf else 0.0,
                "status": "ok",
            })
        return cells

    baseline_mem_cells = _make_mem_cells(
        baseline_pre_ns, baseline_pre_gpu, baseline_post_ns, baseline_post_gpu,
        [0.0] * num_cells, [0.0] * num_cells, [0.0] * num_cells, [0.0] * num_cells,
    )
    flowbook_mem_cells = _make_mem_cells(
        flowbook_pre_ns, flowbook_pre_gpu, flowbook_post_ns, flowbook_post_gpu,
        flowbook_pre_ckpt, flowbook_pre_enforcer, flowbook_ckpt_delta, flowbook_ckpt_cumulative,
    )

    timing_cells = []
    for i in range(num_cells):
        timing_cells.append({
            "cell_id": f"c{i}",
            "cell_index": i,
            "execute_duration_ms": 100.0 + i * 10,
            "code_duration_ms": 90.0 + i * 10,
            "state_duration_ms": 5.0,
            "check_duration_ms": 2.0,
            "status": "ok",
        })

    data = {
        "version": "3.0",
        "_version": "3.0",
        "notebook_path": "/test/notebook.ipynb",
        "timestamp": "2026-01-01T00:00:00",
        "scalene_available": True,
        "metadata": {"num_cells": num_cells, "staleness_mode": "semantic"},
        "kernels": {
            "baseline": {
                "kernel_name": "baseline_kernel",
                "timing": {
                    "kernel_name": "baseline_kernel",
                    "cells": copy.deepcopy(timing_cells),
                    "rerun_cells": [],
                    "totals": {
                        "execute_duration_ms": sum(c["execute_duration_ms"] for c in timing_cells),
                        "code_duration_ms": sum(c["execute_duration_ms"] for c in timing_cells),
                    },
                },
                "memory": {
                    "kernel_name": "baseline_kernel",
                    "cells": baseline_mem_cells,
                    "rerun_cells": [],
                    "totals": {
                        "final_namespace_mb": baseline_final_ns,
                        "final_gpu_mb": baseline_final_gpu,
                        "max_namespace_mb": max(baseline_post_ns),
                    },
                } if include_baseline_memory else None,
            },
            "flowbook": {
                "kernel_name": "flowbook_kernel",
                "timing": {
                    "kernel_name": "flowbook_kernel",
                    "cells": timing_cells,
                    "rerun_cells": [],
                    "totals": {
                        "execute_duration_ms": sum(c["execute_duration_ms"] for c in timing_cells),
                        "code_duration_ms": sum(c["code_duration_ms"] for c in timing_cells),
                        "state_duration_ms": sum(c["state_duration_ms"] for c in timing_cells),
                        "check_duration_ms": sum(c["check_duration_ms"] for c in timing_cells),
                    },
                },
                "memory": {
                    "kernel_name": "flowbook_kernel",
                    "cells": flowbook_mem_cells,
                    "rerun_cells": [],
                    "totals": {
                        "final_namespace_mb": flowbook_final_ns,
                        "final_gpu_mb": flowbook_final_gpu,
                        "final_checkpoint_cumulative_mb": flowbook_final_ckpt,
                        "final_enforcer_overhead_mb": flowbook_final_enforcer,
                        "max_namespace_mb": max(flowbook_post_ns),
                        "memory_overhead_ratio": (flowbook_final_ns + flowbook_final_ckpt) / flowbook_final_ns if flowbook_final_ns > 0 else 1.0,
                    },
                },
            },
        },
    }
    return data


class TestVersionDetection:
    def test_is_v3_format_true(self):
        assert is_v3_format({"version": "3.0"}) is True

    def test_is_v3_format_false_v2(self):
        assert is_v3_format({"version": "2.0"}) is False

    def test_is_v3_format_false_v1(self):
        assert is_v3_format({"version": "1.0"}) is False

    def test_is_v3_format_false_missing(self):
        assert is_v3_format({}) is False

    def test_is_v2_format_still_works(self):
        assert is_v2_format({"version": "2.0"}) is True
        assert is_v2_format({"version": "3.0"}) is False


class TestCrossRunOverhead:
    def test_basic_computation(self):
        """Given known pre-values, verify Checkpoint_i and ratios."""
        data = _make_v3_data(
            num_cells=4,
            baseline_pre_ns=[10.0, 15.0, 20.0, 25.0],
            baseline_pre_gpu=[0.0, 0.0, 0.0, 0.0],
            flowbook_pre_ns=[10.0, 15.0, 20.0, 25.0],
            flowbook_pre_gpu=[0.0, 0.0, 0.0, 0.0],
            flowbook_pre_ckpt=[0.0, 5.0, 8.0, 12.0],
            flowbook_pre_enforcer=[0.0, 0.0, 0.0, 0.0],
            # Final: baseline ns=30, flowbook ns=30 + ckpt=15
            baseline_final_ns=30.0,
            baseline_final_gpu=0.0,
            flowbook_final_ns=30.0,
            flowbook_final_gpu=0.0,
            flowbook_final_ckpt=15.0,
            flowbook_final_enforcer=0.0,
        )

        baseline_cells = data["kernels"]["baseline"]["memory"]["cells"]
        flowbook_cells = data["kernels"]["flowbook"]["memory"]["cells"]
        baseline_totals = data["kernels"]["baseline"]["memory"]["totals"]
        flowbook_totals = data["kernels"]["flowbook"]["memory"]["totals"]

        base_values, checkpoint_costs, ratios = _compute_cross_run_overhead(
            baseline_cells, flowbook_cells, baseline_totals, flowbook_totals,
        )

        # Overhead = [0, 5, 8, 12] (Flow_i - Base_i = pre_ckpt + pre_enforcer)
        # Final overhead = (30+15) - 30 = 15
        # Checkpoint = [5-0, 8-5, 12-8, 15-12] = [5, 3, 4, 3]
        assert checkpoint_costs == pytest.approx([5.0, 3.0, 4.0, 3.0])

        # Ratios = [5/10, 3/15, 4/20, 3/25] = [0.5, 0.2, 0.2, 0.12]
        assert ratios == pytest.approx([0.5, 0.2, 0.2, 0.12])

        # Base values
        assert base_values == pytest.approx([10.0, 15.0, 20.0, 25.0])

    def test_small_base_clamped(self):
        """When Base_i < MIN_MEANINGFUL_BASE_MB, ratio should be 0."""
        data = _make_v3_data(
            num_cells=2,
            baseline_pre_ns=[0.1, 10.0],
            baseline_pre_gpu=[0.0, 0.0],
            flowbook_pre_ns=[0.1, 10.0],
            flowbook_pre_gpu=[0.0, 0.0],
            flowbook_pre_ckpt=[0.0, 2.0],
            flowbook_pre_enforcer=[0.0, 0.0],
            baseline_final_ns=15.0,
            flowbook_final_ns=15.0,
            flowbook_final_ckpt=5.0,
            flowbook_final_enforcer=0.0,
        )

        baseline_cells = data["kernels"]["baseline"]["memory"]["cells"]
        flowbook_cells = data["kernels"]["flowbook"]["memory"]["cells"]
        baseline_totals = data["kernels"]["baseline"]["memory"]["totals"]
        flowbook_totals = data["kernels"]["flowbook"]["memory"]["totals"]

        _, _, ratios = _compute_cross_run_overhead(
            baseline_cells, flowbook_cells, baseline_totals, flowbook_totals,
        )

        # First cell: base = 0.1 < 1.0, so ratio = 0.0
        assert ratios[0] == 0.0
        # Second cell: base = 10.0 >= 1.0, non-zero ratio
        assert ratios[1] != 0.0

    def test_gpu_included_in_base(self):
        """GPU memory should be included in Base_i and Flow_i."""
        data = _make_v3_data(
            num_cells=2,
            baseline_pre_ns=[5.0, 10.0],
            baseline_pre_gpu=[5.0, 5.0],
            flowbook_pre_ns=[5.0, 10.0],
            flowbook_pre_gpu=[5.0, 5.0],
            flowbook_pre_ckpt=[0.0, 3.0],
            flowbook_pre_enforcer=[0.0, 0.0],
            baseline_final_ns=15.0,
            baseline_final_gpu=5.0,
            flowbook_final_ns=15.0,
            flowbook_final_gpu=5.0,
            flowbook_final_ckpt=6.0,
            flowbook_final_enforcer=0.0,
        )

        baseline_cells = data["kernels"]["baseline"]["memory"]["cells"]
        flowbook_cells = data["kernels"]["flowbook"]["memory"]["cells"]
        baseline_totals = data["kernels"]["baseline"]["memory"]["totals"]
        flowbook_totals = data["kernels"]["flowbook"]["memory"]["totals"]

        base_values, _, _ = _compute_cross_run_overhead(
            baseline_cells, flowbook_cells, baseline_totals, flowbook_totals,
        )

        # Base_0 = pre_ns(5) + pre_gpu(5) = 10
        assert base_values[0] == pytest.approx(10.0)
        # Base_1 = pre_ns(10) + pre_gpu(5) = 15
        assert base_values[1] == pytest.approx(15.0)


class TestFallbackRatios:
    def test_basic_fallback(self):
        """FlowBook-only fallback uses delta / (prev_ns + prev_gpu)."""
        cells = [
            {"namespace_mb": 10.0, "gpu_mb": 0.0, "checkpoint_delta_mb": 2.0},
            {"namespace_mb": 15.0, "gpu_mb": 0.0, "checkpoint_delta_mb": 3.0},
            {"namespace_mb": 20.0, "gpu_mb": 0.0, "checkpoint_delta_mb": 4.0},
        ]
        ratios = _compute_fallback_ratios(cells)

        # Cell 0: no prev, ratio = 0
        assert ratios[0] == 0.0
        # Cell 1: 3 / 10 = 0.3
        assert ratios[1] == pytest.approx(0.3)
        # Cell 2: 4 / 15 ≈ 0.267
        assert ratios[2] == pytest.approx(4.0 / 15.0)

    def test_fallback_small_base(self):
        cells = [
            {"namespace_mb": 0.1, "gpu_mb": 0.0, "checkpoint_delta_mb": 1.0},
            {"namespace_mb": 10.0, "gpu_mb": 0.0, "checkpoint_delta_mb": 2.0},
        ]
        ratios = _compute_fallback_ratios(cells)
        assert ratios[0] == 0.0  # No prev
        assert ratios[1] == 0.0  # prev base = 0.1 < 1.0


class TestComputeFileStatsV3:
    def test_cross_run_stats(self):
        """Full stats computation with baseline memory."""
        data = _make_v3_data(num_cells=3)
        stats = compute_file_stats_v3(data, "/test.json")

        assert stats.num_cells == 3
        assert stats.notebook_name == "notebook.ipynb"
        assert len(stats.per_cell_memory_overhead_mb) == 3
        assert stats.memory_overhead_ratio > 0

    def test_flowbook_only_fallback(self):
        """Stats computation without baseline memory falls back gracefully."""
        data = _make_v3_data(num_cells=3, include_baseline_memory=False)
        stats = compute_file_stats_v3(data, "/test.json")

        assert stats.num_cells == 3
        assert len(stats.per_cell_memory_overhead_mb) == 3
        # Should use self-referential ratio from totals
        assert stats.memory_overhead_ratio > 1.0

    def test_timing_stats_match(self):
        """Timing stats should be computed correctly."""
        data = _make_v3_data(num_cells=2)
        stats = compute_file_stats_v3(data, "/test.json")

        # state_duration_ms per cell = 5.0, check = 2.0
        assert stats.state_overhead_ms == pytest.approx(10.0)
        assert stats.check_overhead_ms == pytest.approx(4.0)
        assert len(stats.per_cell_checkpoint_overhead_ms) == 2
        assert all(v == pytest.approx(5.0) for v in stats.per_cell_checkpoint_overhead_ms)

    def test_json_roundtrip(self):
        """v3 data should survive JSON serialization."""
        data = _make_v3_data(num_cells=2)
        serialized = json.dumps(data)
        reloaded = json.loads(serialized)

        assert reloaded["version"] == "3.0"
        fc = reloaded["kernels"]["flowbook"]["memory"]["cells"][0]
        assert "pre_namespace_mb" in fc
        assert "pre_gpu_mb" in fc
        assert "pre_checkpoint_cumulative_mb" in fc
        assert "pre_enforcer_overhead_mb" in fc

        bc = reloaded["kernels"]["baseline"]["memory"]["cells"][0]
        assert bc["pre_checkpoint_cumulative_mb"] == 0.0
        assert bc["pre_enforcer_overhead_mb"] == 0.0
