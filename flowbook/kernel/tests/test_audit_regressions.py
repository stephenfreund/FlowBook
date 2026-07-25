"""
Regression tests for enforcer bugs found in the 2026-07-12 audit
(see AUDIT-2026-07-12.md, findings H1, H2, H3, H5).

H1: A check() that returned before STEP 3 (e.g. the truncation early-return)
    left the PREVIOUS cell's _pending_snapshot in place, so the kernel's
    rollback_last_check() restored the previous cell's committed state.
H2: Forward staleness dropped Var/Rows/Cols write locs: Var(x) was suppressed
    for rebound variables with column-level changes, and typed changes
    (RowsAdded etc.) never entered the conflict set.
H3: The main check() path stored read/write locs without LocRef qualifiers,
    degrading the ▷ relation to variable-name equality.
H5: `del x` was invisible to tracking (covered at the TrackingDict level in
    flowbook/kernel_support/tests/test_tracking.py; covered here at the
    enforcer level: a deletion write propagates staleness to readers).
"""

import pandas as pd

from flowbook.kernel.loc_ids import LocRef
from flowbook.kernel.locations import ReadLocType
from flowbook.kernel.tests.conftest import ReproducibilityTestHelper, make_tracking


class TestPendingSnapshotNotStale:
    """H1: a stale _pending_snapshot must not survive into the next check()."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b"])

    def test_rollback_after_early_return_does_not_corrupt_previous_cell(self):
        # Cell a executes successfully and commits its record.
        result_a = self.helper.execute_cell(
            "a", {}, {"x": 1}, writes={"x"}
        )
        assert not result_a.has_errors()
        state = self.helper.sdc._notebook_state
        assert state.has_record("a")
        assert state.is_clean("a")

        # Cell z is not in the cell order: check() returns before STEP 3
        # ever runs for it (no snapshot is taken for z).
        self.helper.execute_cell("z", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        # The kernel calls rollback_last_check() whenever it rejects a result.
        # With the stale snapshot bug, this restored cell a's PRE-execution
        # state (empty), wiping a's committed reads/writes/status.
        self.helper.sdc.rollback_last_check()

        assert state.has_record("a"), (
            "rollback after an early-return check() must not erase the "
            "previous cell's committed record"
        )
        assert state.is_clean("a")

    def test_snapshot_cleared_at_start_of_next_check(self):
        self.helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        # After a successful check() the snapshot legitimately remains set
        # (the kernel may still decide to roll back this result).
        assert self.helper.sdc._pending_snapshot is not None
        # b's check() commits a and takes a fresh snapshot for b only.
        self.helper.execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        snap = self.helper.sdc._pending_snapshot
        assert snap is not None and snap.cell_id == "b"


class TestForwardStalenessTypedLocs:
    """H2: Rows/Var write locs must reach the forward-staleness conflict set."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b"])

    def test_row_count_change_stales_len_reader(self):
        # Cell a creates df; cell b reads len(df) (Rows read) only.
        df = pd.DataFrame({"x": [1, 2]})
        self.helper.execute_cell("a", {}, {"df": df}, writes={"df"})
        self.helper.execute_cell(
            "b",
            {"df": df},
            {"df": df, "n": 2},
            reads={"df"},
            writes={"n"},
            structural_reads={"df": {"len"}},
        )

        # Re-run a: rows appended IN PLACE (df not rebound). The diff yields
        # RowsAdded; before the fix only Col locs entered the conflict set and
        # Col ▷ Rows = false, so b was never marked stale.
        self.helper.save_pre_checkpoint("a", {"df": df})
        df.loc[2] = [3]
        result = self.helper.sdc.check(
            cell_id="a",
            pre_checkpoint=self.helper.get_pre_checkpoint("a"),
            namespace={"df": df},
            tracking=make_tracking(
                row_mutations={"df"},
                column_writes={"df": {"x"}},
            ),
            continue_on_violation=True,
        )

        assert "b" in result.stale_cells, (
            "a row-count change must invalidate len(df) readers "
            "(Rows(df) ▷ Rows(df))"
        )

    def test_rebind_with_column_changes_stales_var_reader(self):
        # Cell a creates df; cell b binds z = df (Var(df) read only).
        df1 = pd.DataFrame({"x": [1, 2]})
        self.helper.execute_cell("a", {}, {"df": df1}, writes={"df"})
        self.helper.execute_cell(
            "b", {"df": df1}, {"df": df1, "z": df1}, reads={"df"}, writes={"z"}
        )

        # Re-run a REBINDING df to a frame with different column values.
        # The diff decomposes this into column-level changes, which used to
        # suppress Var(df) — leaving the binding-only reader b clean.
        # Per FORMAL_DEVELOPMENT.md's ▷ matrix, Var write ▷ Var read = Yes.
        df2 = pd.DataFrame({"x": [10, 20]})
        result = self.helper.execute_cell(
            "a", {"df": df1}, {"df": df2}, writes={"df"},
            column_writes={"df": {"x"}},
        )

        assert "b" in result.stale_cells, (
            "rebinding df must invalidate binding-only (Var) readers even "
            "when the diff shows only column-level changes"
        )


class TestStoredLocsCarryLocRefQualifiers:
    """H3: the main check() path must store R/W with LocRef qualifiers."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b"])

    def test_column_read_stored_with_locref(self):
        df = pd.DataFrame({"price": [1, 2]})
        self.helper.execute_cell("a", {}, {"df": df}, writes={"df"})
        self.helper.execute_cell(
            "b",
            {"df": df},
            {"df": df, "y": 3},
            reads={"df"},
            writes={"y"},
            column_reads={"df": {"price"}},
        )

        stored_reads = self.helper.sdc._notebook_state.reads.get("b", frozenset())
        col_locs = [r for r in stored_reads if r.type == ReadLocType.COLUMN]
        assert col_locs, "expected a stored Col read loc for df['price']"
        assert all(isinstance(r.qualifier, LocRef) for r in col_locs), (
            "main-path stored Col locs must carry LocRef qualifiers "
            "(object-identity-based conflict detection), not bare strings"
        )


class TestDeletePropagatesStaleness:
    """H5: a deletion write (del x) must invalidate downstream readers of x."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "c", "b"])

    def test_del_marks_downstream_reader_stale(self):
        # a: x = 1;  c: del x;  b (below c): reads x.
        self.helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        self.helper.execute_cell(
            "b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"}
        )

        # c deletes x. TrackingDict.__delitem__ records this as a write
        # (tested in kernel_support), so the enforcer sees writes={"x"} with
        # x absent from the post-execution namespace.
        result = self.helper.execute_cell("c", {"x": 1, "y": 2}, {"y": 2}, writes={"x"})

        assert not result.has_errors(), (
            "a deletion rebinds nothing it read and is recoverable by rerun; "
            f"got errors: {result.errors}"
        )
        assert self.helper.sdc._notebook_state.is_clean("c")
        assert "b" in result.stale_cells, (
            "deleting x must mark downstream readers of x stale"
        )


class TestCanonicalWriteSet:
    """Audit item 6: one canonical Wᵢ builder (compute_cell_write_locs) for
    all predicates and staleness. These cover the M3/M4 gaps the divergent
    builders caused: tracked writes checked even with an empty diff, and
    structural/file writes visible to NoReadAndWrite/NoWriteAfterRead."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_file_write_after_read_is_violation(self):
        # a reads data.csv; c (below) writes data.csv → NoWriteAfterRead.
        from flowbook.kernel_support.models import TrackingData

        self.helper.save_pre_checkpoint("a", {})
        self.helper.sdc.check(
            cell_id="a",
            pre_checkpoint=self.helper.get_pre_checkpoint("a"),
            namespace={"d": 1},
            tracking=TrackingData(
                reads_before_writes=set(), writes={"d"},
                file_reads_before_writes={"data.csv"},
            ),
        )
        self.helper.save_pre_checkpoint("c", {"d": 1})
        result = self.helper.sdc.check(
            cell_id="c",
            pre_checkpoint=self.helper.get_pre_checkpoint("c"),
            namespace={"d": 1},
            tracking=TrackingData(
                reads_before_writes=set(), writes=set(),
                file_writes={"data.csv"},
            ),
        )
        assert any(
            e.error_type.value == "no_write_after_read" for e in result.errors
        ), f"file write below a file reader must violate NoWriteAfterRead; got {result.errors}"

    def test_idempotent_column_rewrite_still_conflicts(self):
        # b reads df['x']; c rewrites df['x'] with IDENTICAL values (empty
        # diff). The tracked Col write must still conflict with b's read —
        # previously the check was skipped entirely when the diff was empty.
        df = pd.DataFrame({"x": [1, 2]})
        self.helper.execute_cell("a", {}, {"df": df}, writes={"df"})
        self.helper.execute_cell(
            "b", {"df": df}, {"df": df, "y": 3},
            reads={"df"}, writes={"y"}, column_reads={"df": {"x"}},
        )
        self.helper.save_pre_checkpoint("c", {"df": df, "y": 3})
        df["x"] = [1, 2]  # in place, values unchanged → empty diff
        result = self.helper.sdc.check(
            cell_id="c",
            pre_checkpoint=self.helper.get_pre_checkpoint("c"),
            namespace={"df": df, "y": 3},
            tracking=make_tracking(reads={"df"}, column_writes={"df": {"x"}}),
        )
        assert any(
            e.error_type.value == "no_write_after_read" for e in result.errors
        ), f"tracked column write with empty diff must still conflict; got {result.errors}"

    def test_read_len_then_mutate_rows_is_read_and_write(self):
        # A cell that reads len(df) and then mutates rows in place reads and
        # writes the same location (Rows ▷ Rows) — the paper's
        # "diagnostic inspection before mutation" category. Previously
        # structural writes were invisible to NoReadAndWrite.
        df = pd.DataFrame({"x": [1, 2, 3]})
        self.helper.execute_cell("a", {}, {"df": df}, writes={"df"})
        self.helper.save_pre_checkpoint("b", {"df": df})
        df.drop(index=[0], inplace=True)
        result = self.helper.sdc.check(
            cell_id="b",
            pre_checkpoint=self.helper.get_pre_checkpoint("b"),
            namespace={"df": df},
            tracking=make_tracking(
                reads={"df"},
                structural_reads={"df": {"len"}},
                row_mutations={"df"},
                column_writes={"df": {"x"}},
            ),
            continue_on_violation=True,
        )
        assert any(
            e.error_type.value == "no_read_and_write" for e in result.errors
        ), f"len read + row mutation must violate NoReadAndWrite; got {result.errors}"


class TestNonStringColumnLabels:
    """Audit M5 (enforcer level): an in-place write to an int-labeled column
    is recoverable — previously it recorded no column write and landed in
    the UNRECOVERABLE_MUTATION catch-all."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b"])

    def test_int_column_write_is_recoverable(self):
        df = pd.DataFrame({0: [1, 2], "y": [3, 4]})
        self.helper.execute_cell("a", {}, {"df": df}, writes={"df"})

        self.helper.save_pre_checkpoint("b", {"df": df})
        df[0] = [10, 20]  # in place, int label
        result = self.helper.sdc.check(
            cell_id="b",
            pre_checkpoint=self.helper.get_pre_checkpoint("b"),
            namespace={"df": df},
            tracking=make_tracking(
                reads={"df"},
                column_writes={"df": {"0"}},  # tracked via str(label)
            ),
        )
        assert not any(
            e.error_type.value == "unrecoverable_mutation" for e in result.errors
        ), f"int-labeled column write must be recoverable; got {result.errors}"


class TestDroppedColumnWrites:
    """Audit M6: dropped writes are now detected at LOC granularity.
    Previously name-level: a cell that wrote df["x"] before and df["z"] now
    showed no removed write (both are name "df")."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def _run_inplace(self, cell_id, ns, mutate, **tracking):
        self.helper.save_pre_checkpoint(cell_id, ns)
        mutate()
        return self.helper.sdc.check(
            cell_id=cell_id,
            pre_checkpoint=self.helper.get_pre_checkpoint(cell_id),
            namespace=ns,
            tracking=make_tracking(**tracking),
        )

    def test_dropped_col_write_stales_downstream_reader(self):
        # a writes df["x"] in place; b (below) reads df["x"].
        # a is edited to write df["z"] instead → dropped Col(df, x) must
        # invalidate b (Wᵢ ∪ W'ᵢ at loc level).
        df = pd.DataFrame({"x": [0, 0], "z": [0, 0]})
        ns = {"df": df}

        self._run_inplace(
            "a", ns, lambda: df.__setitem__("x", [1, 2]),
            reads={"df"}, column_writes={"df": {"x"}},
        )
        self.helper.execute_cell(
            "b", ns, {"df": df, "y": 3},
            reads={"df"}, writes={"y"}, column_reads={"df": {"x"}},
        )

        self.helper.sdc._notebook_state.handle_edit("a")
        result = self._run_inplace(
            "a", {"df": df}, lambda: df.__setitem__("z", [9, 9]),
            reads={"df"}, column_writes={"df": {"z"}},
        )

        assert "b" in result.stale_cells, (
            "dropping the Col(df, x) write must invalidate x readers below"
        )

    def test_dropped_col_write_exposes_last_writer_above(self):
        # a writes df["x"]; c (below) also wrote df["x"]. c is edited to
        # write df["z"] → a is again the last writer of x → BACKWARD_STALE.
        df = pd.DataFrame({"x": [0, 0], "z": [0, 0]})
        ns = {"df": df}

        self._run_inplace(
            "a", ns, lambda: df.__setitem__("x", [1, 2]),
            reads={"df"}, column_writes={"df": {"x"}},
        )
        self._run_inplace(
            "c", ns, lambda: df.__setitem__("x", [5, 6]),
            reads={"df"}, column_writes={"df": {"x"}},
        )

        self.helper.sdc._notebook_state.handle_edit("c")
        result = self._run_inplace(
            "c", ns, lambda: df.__setitem__("z", [9, 9]),
            reads={"df"}, column_writes={"df": {"z"}},
        )

        assert "a" in result.stale_cells, (
            "dropping c's Col(df, x) write must re-expose a, the last "
            "writer of x above c"
        )


class TestStructuredChangeParsing:
    """Audit M11: change classification uses structured diff payloads, not
    regexes over human-readable messages — column names containing quotes,
    commas, or brackets must classify correctly."""

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b"])

    def test_awkward_column_name_is_recoverable(self):
        name = "it's, a ['weird'] col"
        df = pd.DataFrame({name: [1, 2]})
        self.helper.execute_cell("a", {}, {"df": df}, writes={"df"})

        self.helper.save_pre_checkpoint("b", {"df": df})
        df[name] = [10, 20]
        result = self.helper.sdc.check(
            cell_id="b",
            pre_checkpoint=self.helper.get_pre_checkpoint("b"),
            namespace={"df": df},
            tracking=make_tracking(
                reads={"df"}, column_writes={"df": {name}},
            ),
        )
        assert not any(
            e.error_type.value == "unrecoverable_mutation" for e in result.errors
        ), f"quote/comma/bracket column name must classify correctly; got {result.errors}"

    def test_awkward_column_name_stales_reader(self):
        name = "it's, a ['weird'] col"
        df = pd.DataFrame({name: [1, 2], "y": [3, 4]})
        ns = {"df": df}
        self.helper.save_pre_checkpoint("a", ns)
        df[name] = [0, 0]
        self.helper.sdc.check(
            cell_id="a",
            pre_checkpoint=self.helper.get_pre_checkpoint("a"),
            namespace=ns,
            tracking=make_tracking(reads={"df"}, column_writes={"df": {name}}),
        )
        self.helper.execute_cell(
            "b", ns, {"df": df, "s": 1},
            reads={"df"}, writes={"s"}, column_reads={"df": {name}},
        )

        self.helper.sdc._notebook_state.handle_edit("a")
        self.helper.save_pre_checkpoint("a", ns)
        df[name] = [7, 8]
        result = self.helper.sdc.check(
            cell_id="a",
            pre_checkpoint=self.helper.get_pre_checkpoint("a"),
            namespace=ns,
            tracking=make_tracking(reads={"df"}, column_writes={"df": {name}}),
        )
        assert "b" in result.stale_cells
