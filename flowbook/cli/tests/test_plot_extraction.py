"""Tests for flowbook.cli.plot_extraction functions."""

import pytest
from flowbook.cli.models import (
    ComparisonResult,
    ComparisonMetadata,
    BaselineMemoryResult,
    BaselineCellMemory,
    BaselineMemorySnapshot,
    FlowBookMemoryResult,
    FlowBookCellMemory,
    FlowBookMemorySnapshot,
    CheckpointVarSizes,
    CheckpointVarInfo,
)
from flowbook.cli.plot_extraction import (
    extract_plot3_data,
    extract_plot4_data,
    extract_plot6_data,
    extract_cdf_data,
)


def make_baseline_cell(cell_id: str, idx: int, pre_ns: float, post_ns: float) -> BaselineCellMemory:
    """Helper to create a baseline cell."""
    return BaselineCellMemory(
        cell_id=cell_id,
        cell_index=idx,
        pre=BaselineMemorySnapshot(user_ns_mb=pre_ns, gpu_mb=0.0),
        post=BaselineMemorySnapshot(user_ns_mb=post_ns, gpu_mb=0.0),
    )


def make_flowbook_cell(
    cell_id: str,
    idx: int,
    post_ns: float,
    post_overhead: float,
    checkpoint_vars: dict = None,
) -> FlowBookCellMemory:
    """Helper to create a FlowBook cell."""
    if checkpoint_vars is None:
        checkpoint_vars = {}

    cv = {
        name: CheckpointVarSizes(vars={
            v: CheckpointVarInfo(size_mb=mb) for v, mb in vars.items()
        })
        for name, vars in checkpoint_vars.items()
    }

    return FlowBookCellMemory(
        cell_id=cell_id,
        cell_index=idx,
        pre=FlowBookMemorySnapshot(user_ns_mb=0.0, gpu_mb=0.0, overhead_mb=0.0),
        post=FlowBookMemorySnapshot(
            user_ns_mb=post_ns,
            gpu_mb=0.0,
            overhead_mb=post_overhead,
            checkpoint_vars=cv,
        ),
    )


class TestExtractPlot3Data:
    """Tests for extract_plot3_data (Memory Overhead)."""

    def test_returns_none_without_flowbook(self):
        """Returns None if no FlowBook data."""
        result = ComparisonResult()
        assert extract_plot3_data(result) is None

    def test_with_baseline_computes_overhead(self):
        """With baseline: overhead = flow_total - base_total."""
        baseline = BaselineMemoryResult(cells=[
            make_baseline_cell("a", 0, 0, 100),
            make_baseline_cell("b", 1, 100, 200),
        ])
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 50),  # total=150
            make_flowbook_cell("b", 1, 200, 100),  # total=300
        ])
        result = ComparisonResult(baseline=baseline, flowbook=flowbook)

        p3 = extract_plot3_data(result)

        assert p3 is not None
        assert p3.has_baseline is True
        assert p3.base_mb == [100, 200]
        assert p3.overhead_mb == [50, 100]  # 150-100, 300-200

    def test_without_baseline_uses_flowbook_ns(self):
        """Without baseline: base = flowbook namespace, overhead = overhead_mb."""
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 50),
            make_flowbook_cell("b", 1, 200, 100),
        ])
        result = ComparisonResult(flowbook=flowbook)

        p3 = extract_plot3_data(result)

        assert p3 is not None
        assert p3.has_baseline is False
        assert p3.base_mb == [100, 200]
        assert p3.overhead_mb == [50, 100]

    def test_peak_overhead_computed(self):
        """Peak overhead is found correctly."""
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 10),
            make_flowbook_cell("b", 1, 100, 50),  # Peak
            make_flowbook_cell("c", 2, 100, 30),
        ])
        result = ComparisonResult(flowbook=flowbook)

        p3 = extract_plot3_data(result)

        assert p3.peak_overhead_mb == 50
        assert p3.peak_cell == 1
        assert p3.peak_overhead_pct == 50.0  # 50/100 * 100

    def test_cells_are_1_indexed(self):
        """Cell numbers are 1-indexed for display."""
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 10),
            make_flowbook_cell("b", 1, 100, 20),
        ])
        result = ComparisonResult(flowbook=flowbook)

        p3 = extract_plot3_data(result)

        assert p3.cells == [1, 2]

    def test_includes_rerun_cells(self):
        """Rerun cells are included."""
        flowbook = FlowBookMemoryResult(
            cells=[make_flowbook_cell("a", 0, 100, 10)],
            rerun_cells=[make_flowbook_cell("a", 0, 100, 20)],
        )
        result = ComparisonResult(flowbook=flowbook)

        p3 = extract_plot3_data(result)

        assert len(p3.cells) == 2
        assert p3.initial_count == 1


class TestExtractPlot4Data:
    """Tests for extract_plot4_data (Checkpoint Memory by Variable)."""

    def test_returns_none_without_flowbook(self):
        """Returns None if no FlowBook data."""
        result = ComparisonResult()
        assert extract_plot4_data(result) is None

    def test_returns_none_without_checkpoint_vars(self):
        """Returns None if no checkpoint variables."""
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 0),  # No checkpoint_vars
        ])
        result = ComparisonResult(flowbook=flowbook)

        assert extract_plot4_data(result) is None

    def test_extracts_var_series(self):
        """Variable sizes extracted correctly."""
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 30, {"_pre_a": {"df": 20, "X": 10}}),
            make_flowbook_cell("b", 1, 150, 45, {"_pre_b": {"df": 25, "X": 15, "Y": 5}}),
        ])
        result = ComparisonResult(flowbook=flowbook)

        p4 = extract_plot4_data(result)

        assert p4 is not None
        assert "df" in p4.var_series
        assert p4.var_series["df"] == [20, 25]
        assert p4.var_series["X"] == [10, 15]

    def test_vars_ordered_by_max_size(self):
        """Variables ordered by max size descending."""
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 30, {"_pre_a": {"small": 1, "big": 100, "medium": 50}}),
        ])
        result = ComparisonResult(flowbook=flowbook)

        p4 = extract_plot4_data(result, top_n=10)

        assert p4.vars_ordered[0] == "big"
        assert p4.vars_ordered[1] == "medium"
        assert p4.vars_ordered[2] == "small"

    def test_aggregates_other_vars(self):
        """Variables beyond top_n are aggregated as 'other'."""
        checkpoint_vars = {"_pre_a": {f"var{i}": i for i in range(20)}}
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 100, checkpoint_vars),
        ])
        result = ComparisonResult(flowbook=flowbook)

        p4 = extract_plot4_data(result, top_n=5)

        assert len(p4.vars_ordered) == 6  # 5 + "other"
        assert "other" in p4.vars_ordered

    def test_sums_across_checkpoints(self):
        """Multiple checkpoints per cell are summed."""
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 60, {
                "_pre_a": {"df": 20},
                "_post_a": {"df": 20},
                "_pre_b": {"df": 20},
            }),
        ])
        result = ComparisonResult(flowbook=flowbook)

        p4 = extract_plot4_data(result)

        # df appears in 3 checkpoints, each 20 MB
        assert p4.var_series["df"] == [60]


class TestExtractPlot6Data:
    """Tests for extract_plot6_data (Checkpoint Overhead Ratio CDF)."""

    def test_returns_none_without_flowbook(self):
        """Returns None if no FlowBook data."""
        result = ComparisonResult()
        assert extract_plot6_data(result) is None

    def test_computes_ratios_correctly(self):
        """ratio = checkpoint_mb / user_ns_mb."""
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 50, {"_pre_a": {"df": 50}}),
            make_flowbook_cell("b", 1, 200, 100, {"_pre_b": {"df": 100}}),
        ])
        result = ComparisonResult(flowbook=flowbook)

        p6 = extract_plot6_data(result)

        assert p6 is not None
        assert len(p6.ratios) == 2
        assert p6.ratios[0] == 0.5  # 50/100
        assert p6.ratios[1] == 0.5  # 100/200

    def test_excludes_small_namespace(self):
        """Cells with user_ns < 1MB are excluded."""
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 0.5, 10, {"_pre_a": {"df": 10}}),  # Excluded
            make_flowbook_cell("b", 1, 100, 50, {"_pre_b": {"df": 50}}),
        ])
        result = ComparisonResult(flowbook=flowbook)

        p6 = extract_plot6_data(result)

        assert len(p6.ratios) == 1

    def test_sorted_ratios_ascending(self):
        """sorted_ratios in ascending order."""
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 80, {"_pre_a": {"df": 80}}),
            make_flowbook_cell("b", 1, 100, 20, {"_pre_b": {"df": 20}}),
            make_flowbook_cell("c", 2, 100, 50, {"_pre_c": {"df": 50}}),
        ])
        result = ComparisonResult(flowbook=flowbook)

        p6 = extract_plot6_data(result)

        assert p6.sorted_ratios == sorted(p6.sorted_ratios)

    def test_percentiles_0_to_1(self):
        """Percentiles range from > 0 to 1.0."""
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell(f"c{i}", i, 100, i * 10, {f"_pre_{i}": {"df": i * 10}})
            for i in range(10)
        ])
        result = ComparisonResult(flowbook=flowbook)

        p6 = extract_plot6_data(result)

        assert p6.percentiles[0] > 0
        assert p6.percentiles[-1] == 1.0

    def test_median_computed(self):
        """Median is approximately 50th percentile."""
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell(f"c{i}", i, 100, (i + 1) * 10, {f"_pre_{i}": {"df": (i + 1) * 10}})
            for i in range(10)
        ])
        result = ComparisonResult(flowbook=flowbook)

        p6 = extract_plot6_data(result)

        # Ratios: 0.1, 0.2, ..., 1.0
        # Median should be around 0.55
        assert 0.5 <= p6.median_ratio <= 0.6


class TestExtractCDFData:
    """Tests for extract_cdf_data (aggregate CDFs)."""

    def test_returns_none_without_data(self):
        """Returns None if no results."""
        assert extract_cdf_data([]) is None

    def test_aggregates_memory_ratios(self):
        """Memory ratios collected from all results."""
        results = [
            ComparisonResult(flowbook=FlowBookMemoryResult(cells=[
                make_flowbook_cell("a", 0, 100, 50, {"_pre_a": {"df": 50}}),
            ])),
            ComparisonResult(flowbook=FlowBookMemoryResult(cells=[
                make_flowbook_cell("b", 0, 100, 100, {"_pre_b": {"df": 100}}),
            ])),
        ]

        cdf = extract_cdf_data(results)

        assert cdf is not None
        assert len(cdf.memory_ratios) == 2
        assert 0.5 in cdf.memory_ratios
        assert 1.0 in cdf.memory_ratios

    def test_peak_memory_per_notebook(self):
        """Peak memory overhead collected per notebook."""
        results = [
            ComparisonResult(flowbook=FlowBookMemoryResult(cells=[
                make_flowbook_cell("a", 0, 100, 10, {"_pre_a": {"df": 10}}),
                make_flowbook_cell("b", 1, 100, 50, {"_pre_b": {"df": 50}}),  # Peak: 50%
            ])),
            ComparisonResult(flowbook=FlowBookMemoryResult(cells=[
                make_flowbook_cell("c", 0, 100, 30, {"_pre_c": {"df": 30}}),  # Peak: 30%
            ])),
        ]

        cdf = extract_cdf_data(results)

        assert len(cdf.peak_memory_pct) == 2
        assert 50.0 in cdf.peak_memory_pct
        assert 30.0 in cdf.peak_memory_pct


class TestSyntacticVsSemanticExtraction:
    """Test extraction works correctly for both modes."""

    def test_syntactic_single_checkpoint_per_cell(self):
        """Syntactic mode: one checkpoint shows reasonable overhead."""
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 50, {"_pre_a": {"df": 50}}),
            make_flowbook_cell("b", 1, 100, 50, {"_pre_b": {"df": 50}}),
        ])
        result = ComparisonResult(
            metadata=ComparisonMetadata("syntactic", 2, 300),
            flowbook=flowbook,
        )

        p3 = extract_plot3_data(result)
        p6 = extract_plot6_data(result)

        # Each cell has 50MB overhead
        assert p3.overhead_mb == [50, 50]
        assert p6.median_ratio == 0.5

    def test_semantic_checkpoints_accumulate(self):
        """Semantic mode: checkpoints accumulate, overhead grows."""
        # Cell 0: 1 checkpoint, Cell 1: 2 checkpoints
        flowbook = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 50, {"_pre_a": {"df": 50}}),
            make_flowbook_cell("b", 1, 100, 100, {
                "_pre_a": {"df": 50},
                "_pre_b": {"df": 50},
            }),
        ])
        result = ComparisonResult(
            metadata=ComparisonMetadata("semantic", 2, 300),
            flowbook=flowbook,
        )

        p3 = extract_plot3_data(result)
        p6 = extract_plot6_data(result)

        # Cell 1 has double overhead
        assert p3.overhead_mb == [50, 100]
        assert p6.ratios == [0.5, 1.0]

    def test_semantic_larger_than_syntactic(self):
        """Semantic mode shows larger overhead than syntactic for same notebook."""
        # Same 2-cell notebook
        syntactic = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 50, {"_pre_a": {"df": 50}}),
            make_flowbook_cell("b", 1, 100, 50, {"_pre_b": {"df": 50}}),
        ])

        semantic = FlowBookMemoryResult(cells=[
            make_flowbook_cell("a", 0, 100, 50, {"_pre_a": {"df": 50}}),
            make_flowbook_cell("b", 1, 100, 100, {"_pre_a": {"df": 50}, "_pre_b": {"df": 50}}),
        ])

        syn_result = ComparisonResult(flowbook=syntactic)
        sem_result = ComparisonResult(flowbook=semantic)

        syn_p3 = extract_plot3_data(syn_result)
        sem_p3 = extract_plot3_data(sem_result)

        # Semantic final overhead > syntactic
        assert sem_p3.overhead_mb[-1] > syn_p3.overhead_mb[-1]
