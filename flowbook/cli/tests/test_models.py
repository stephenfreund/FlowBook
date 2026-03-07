"""Tests for flowbook.cli.models data structures."""

import pytest
from flowbook.cli.models import (
    # Baseline models
    BaselineMemorySnapshot,
    BaselineCellMemory,
    BaselineMemoryResult,
    # FlowBook models
    CheckpointVarInfo,
    CheckpointVarSizes,
    FlowBookMemorySnapshot,
    FlowBookCellMemory,
    FlowBookMemoryResult,
    # Result models
    ComparisonMetadata,
    ComparisonResult,
    # Plot models
    Plot3Data,
    Plot4Data,
    Plot6Data,
)


class TestBaselineMemorySnapshot:
    """Tests for BaselineMemorySnapshot."""

    def test_total_mb(self):
        """total_mb = user_ns_mb + gpu_mb."""
        s = BaselineMemorySnapshot(user_ns_mb=100.0, gpu_mb=20.0)
        assert s.total_mb == 120.0

    def test_total_mb_zero_gpu(self):
        """total_mb with no GPU."""
        s = BaselineMemorySnapshot(user_ns_mb=50.0, gpu_mb=0.0)
        assert s.total_mb == 50.0

    def test_to_dict(self):
        """Serialization to dict."""
        s = BaselineMemorySnapshot(user_ns_mb=100.0, gpu_mb=20.0)
        d = s.to_dict()
        assert d == {"user_ns_mb": 100.0, "gpu_mb": 20.0}

    def test_from_dict(self):
        """Deserialization from dict."""
        d = {"user_ns_mb": 100.0, "gpu_mb": 20.0}
        s = BaselineMemorySnapshot.from_dict(d)
        assert s.user_ns_mb == 100.0
        assert s.gpu_mb == 20.0

    def test_from_dict_missing_fields(self):
        """Deserialization with missing fields defaults to 0."""
        s = BaselineMemorySnapshot.from_dict({})
        assert s.user_ns_mb == 0.0
        assert s.gpu_mb == 0.0


class TestBaselineCellMemory:
    """Tests for BaselineCellMemory."""

    def test_basic_structure(self):
        """Cell has pre and post snapshots."""
        pre = BaselineMemorySnapshot(user_ns_mb=50.0, gpu_mb=0.0)
        post = BaselineMemorySnapshot(user_ns_mb=100.0, gpu_mb=0.0)
        cell = BaselineCellMemory(cell_id="abc1", cell_index=0, pre=pre, post=post)

        assert cell.cell_id == "abc1"
        assert cell.cell_index == 0
        assert cell.pre.user_ns_mb == 50.0
        assert cell.post.user_ns_mb == 100.0
        assert cell.status == "ok"

    def test_error_cell(self):
        """Cell with error status."""
        pre = BaselineMemorySnapshot(user_ns_mb=50.0, gpu_mb=0.0)
        post = BaselineMemorySnapshot(user_ns_mb=50.0, gpu_mb=0.0)
        cell = BaselineCellMemory(
            cell_id="abc1", cell_index=0, pre=pre, post=post,
            status="error", error="NameError: x not defined"
        )
        assert cell.status == "error"
        assert "NameError" in cell.error

    def test_round_trip(self):
        """Serialization and deserialization."""
        pre = BaselineMemorySnapshot(user_ns_mb=50.0, gpu_mb=10.0)
        post = BaselineMemorySnapshot(user_ns_mb=100.0, gpu_mb=10.0)
        cell = BaselineCellMemory(cell_id="xyz2", cell_index=5, pre=pre, post=post)

        d = cell.to_dict()
        cell2 = BaselineCellMemory.from_dict(d)

        assert cell2.cell_id == "xyz2"
        assert cell2.cell_index == 5
        assert cell2.pre.user_ns_mb == 50.0
        assert cell2.post.user_ns_mb == 100.0


class TestBaselineMemoryResult:
    """Tests for BaselineMemoryResult."""

    def test_empty_result(self):
        """Empty result has 0 values."""
        r = BaselineMemoryResult()
        assert r.final_user_ns_mb == 0.0
        assert r.max_user_ns_mb == 0.0
        assert r.all_cells == []

    def test_final_user_ns_mb(self):
        """final_user_ns_mb is last cell's post value."""
        cells = [
            BaselineCellMemory(
                cell_id="a", cell_index=0,
                pre=BaselineMemorySnapshot(0.0, 0.0),
                post=BaselineMemorySnapshot(50.0, 0.0),
            ),
            BaselineCellMemory(
                cell_id="b", cell_index=1,
                pre=BaselineMemorySnapshot(50.0, 0.0),
                post=BaselineMemorySnapshot(100.0, 0.0),
            ),
        ]
        r = BaselineMemoryResult(cells=cells)
        assert r.final_user_ns_mb == 100.0

    def test_max_user_ns_mb(self):
        """max_user_ns_mb is peak across all cells."""
        cells = [
            BaselineCellMemory(
                cell_id="a", cell_index=0,
                pre=BaselineMemorySnapshot(0.0, 0.0),
                post=BaselineMemorySnapshot(200.0, 0.0),  # Peak
            ),
            BaselineCellMemory(
                cell_id="b", cell_index=1,
                pre=BaselineMemorySnapshot(200.0, 0.0),
                post=BaselineMemorySnapshot(100.0, 0.0),  # Decreased
            ),
        ]
        r = BaselineMemoryResult(cells=cells)
        assert r.max_user_ns_mb == 200.0

    def test_all_cells_includes_reruns(self):
        """all_cells includes both initial and rerun cells."""
        cells = [
            BaselineCellMemory(
                cell_id="a", cell_index=0,
                pre=BaselineMemorySnapshot(0.0, 0.0),
                post=BaselineMemorySnapshot(50.0, 0.0),
            ),
        ]
        rerun_cells = [
            BaselineCellMemory(
                cell_id="a", cell_index=0,
                pre=BaselineMemorySnapshot(50.0, 0.0),
                post=BaselineMemorySnapshot(60.0, 0.0),
            ),
        ]
        r = BaselineMemoryResult(cells=cells, rerun_cells=rerun_cells)
        assert len(r.all_cells) == 2


class TestCheckpointVarSizes:
    """Tests for CheckpointVarSizes."""

    def test_total_mb(self):
        """total_mb sums all variable sizes."""
        cvs = CheckpointVarSizes(vars={
            "df": CheckpointVarInfo(size_mb=10.0),
            "X": CheckpointVarInfo(size_mb=5.0),
        })
        assert cvs.total_mb == 15.0

    def test_getitem(self):
        """Indexing returns variable size."""
        cvs = CheckpointVarSizes(vars={
            "df": CheckpointVarInfo(size_mb=10.0),
        })
        assert cvs["df"] == 10.0
        assert cvs["missing"] == 0.0

    def test_empty(self):
        """Empty checkpoint has 0 total."""
        cvs = CheckpointVarSizes()
        assert cvs.total_mb == 0.0


class TestFlowBookMemorySnapshot:
    """Tests for FlowBookMemorySnapshot."""

    def test_total_mb(self):
        """total_mb = user_ns + gpu + overhead."""
        s = FlowBookMemorySnapshot(user_ns_mb=100.0, gpu_mb=20.0, overhead_mb=50.0)
        assert s.total_mb == 170.0

    def test_checkpoint_count(self):
        """checkpoint_count is number of checkpoints."""
        s = FlowBookMemorySnapshot(
            user_ns_mb=100.0, gpu_mb=0.0, overhead_mb=30.0,
            checkpoint_vars={
                "_pre_a": CheckpointVarSizes(),
                "_pre_b": CheckpointVarSizes(),
            }
        )
        assert s.checkpoint_count == 2

    def test_total_checkpoint_mb(self):
        """total_checkpoint_mb sums all checkpoint sizes."""
        s = FlowBookMemorySnapshot(
            user_ns_mb=100.0, gpu_mb=0.0, overhead_mb=30.0,
            checkpoint_vars={
                "_pre_a": CheckpointVarSizes(vars={"df": CheckpointVarInfo(10.0)}),
                "_pre_b": CheckpointVarSizes(vars={"df": CheckpointVarInfo(10.0), "X": CheckpointVarInfo(5.0)}),
            }
        )
        assert s.total_checkpoint_mb == 25.0

    def test_var_totals(self):
        """var_totals sums each variable across checkpoints."""
        s = FlowBookMemorySnapshot(
            user_ns_mb=100.0, gpu_mb=0.0, overhead_mb=30.0,
            checkpoint_vars={
                "_pre_a": CheckpointVarSizes(vars={
                    "df": CheckpointVarInfo(10.0),
                    "X": CheckpointVarInfo(5.0),
                }),
                "_pre_b": CheckpointVarSizes(vars={
                    "df": CheckpointVarInfo(10.0),
                    "Y": CheckpointVarInfo(3.0),
                }),
            }
        )
        totals = s.var_totals()
        assert totals["df"] == 20.0  # 10 + 10
        assert totals["X"] == 5.0
        assert totals["Y"] == 3.0

    def test_var_types(self):
        """var_types collects type names from all checkpoints."""
        s = FlowBookMemorySnapshot(
            user_ns_mb=100.0, gpu_mb=0.0, overhead_mb=30.0,
            checkpoint_vars={
                "_pre_a": CheckpointVarSizes(vars={
                    "df": CheckpointVarInfo(10.0, type_name="DataFrame"),
                }),
            }
        )
        types = s.var_types()
        assert types["df"] == "DataFrame"


class TestFlowBookCellMemory:
    """Tests for FlowBookCellMemory."""

    def test_basic_structure(self):
        """Cell has pre and post snapshots with overhead."""
        pre = FlowBookMemorySnapshot(user_ns_mb=50.0, gpu_mb=0.0, overhead_mb=10.0)
        post = FlowBookMemorySnapshot(user_ns_mb=100.0, gpu_mb=0.0, overhead_mb=30.0)
        cell = FlowBookCellMemory(cell_id="abc1", cell_index=0, pre=pre, post=post)

        assert cell.cell_id == "abc1"
        assert cell.pre.overhead_mb == 10.0
        assert cell.post.overhead_mb == 30.0
        assert cell.post.total_mb == 130.0

    def test_round_trip(self):
        """Serialization and deserialization."""
        pre = FlowBookMemorySnapshot(
            user_ns_mb=50.0, gpu_mb=0.0, overhead_mb=10.0,
            checkpoint_vars={"_pre_a": CheckpointVarSizes(vars={"df": CheckpointVarInfo(5.0)})}
        )
        post = FlowBookMemorySnapshot(
            user_ns_mb=100.0, gpu_mb=0.0, overhead_mb=30.0,
            checkpoint_vars={"_pre_a": CheckpointVarSizes(vars={"df": CheckpointVarInfo(15.0)})}
        )
        cell = FlowBookCellMemory(cell_id="xyz2", cell_index=5, pre=pre, post=post)

        d = cell.to_dict()
        cell2 = FlowBookCellMemory.from_dict(d)

        assert cell2.cell_id == "xyz2"
        assert cell2.post.overhead_mb == 30.0
        assert cell2.post.checkpoint_vars["_pre_a"].total_mb == 15.0


class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_empty_result(self):
        """Empty result serializes correctly."""
        r = ComparisonResult()
        d = r.to_dict()
        assert d["version"] == "4.0"

    def test_round_trip(self):
        """Full result round-trips through dict."""
        metadata = ComparisonMetadata(
            staleness_mode="semantic",
            num_cells=10,
            timeout_seconds=300.0,
        )

        baseline = BaselineMemoryResult(cells=[
            BaselineCellMemory(
                cell_id="a", cell_index=0,
                pre=BaselineMemorySnapshot(0.0, 0.0),
                post=BaselineMemorySnapshot(100.0, 0.0),
            )
        ])

        flowbook = FlowBookMemoryResult(cells=[
            FlowBookCellMemory(
                cell_id="a", cell_index=0,
                pre=FlowBookMemorySnapshot(0.0, 0.0, 0.0),
                post=FlowBookMemorySnapshot(100.0, 0.0, 50.0),
            )
        ])

        r = ComparisonResult(
            metadata=metadata,
            baseline=baseline,
            flowbook=flowbook,
        )

        d = r.to_dict()
        r2 = ComparisonResult.from_dict(d)

        assert r2.metadata.staleness_mode == "semantic"
        assert r2.baseline.final_user_ns_mb == 100.0
        assert r2.flowbook.final_overhead_mb == 50.0


class TestSyntacticVsSemanticModels:
    """Test that models work correctly for both modes."""

    def test_syntactic_single_checkpoint(self):
        """Syntactic mode: one checkpoint per cell."""
        post = FlowBookMemorySnapshot(
            user_ns_mb=100.0, gpu_mb=0.0, overhead_mb=50.0,
            checkpoint_vars={
                "_pre_abc1": CheckpointVarSizes(vars={
                    "df": CheckpointVarInfo(50.0),
                }),
            }
        )
        assert post.checkpoint_count == 1
        assert post.total_checkpoint_mb == 50.0

    def test_semantic_multiple_checkpoints(self):
        """Semantic mode: multiple checkpoints accumulate."""
        post = FlowBookMemorySnapshot(
            user_ns_mb=100.0, gpu_mb=0.0, overhead_mb=150.0,
            checkpoint_vars={
                "_pre_abc1": CheckpointVarSizes(vars={"df": CheckpointVarInfo(50.0)}),
                "_pre_xyz2": CheckpointVarSizes(vars={"df": CheckpointVarInfo(50.0)}),
                "_pre_def3": CheckpointVarSizes(vars={"df": CheckpointVarInfo(50.0)}),
            }
        )
        assert post.checkpoint_count == 3
        assert post.total_checkpoint_mb == 150.0
        assert post.var_totals()["df"] == 150.0

    def test_semantic_larger_than_syntactic(self):
        """Semantic overhead > syntactic for same namespace size."""
        syntactic = FlowBookMemorySnapshot(
            user_ns_mb=100.0, gpu_mb=0.0, overhead_mb=50.0,
            checkpoint_vars={"_pre_a": CheckpointVarSizes(vars={"df": CheckpointVarInfo(50.0)})}
        )

        semantic = FlowBookMemorySnapshot(
            user_ns_mb=100.0, gpu_mb=0.0, overhead_mb=500.0,
            checkpoint_vars={
                f"_pre_{i}": CheckpointVarSizes(vars={"df": CheckpointVarInfo(50.0)})
                for i in range(10)
            }
        )

        assert semantic.overhead_mb > syntactic.overhead_mb
        assert semantic.total_checkpoint_mb > syntactic.total_checkpoint_mb
