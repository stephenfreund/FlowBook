"""
Tests derived from examples in the FlowBook paper (main.tex).

Each test class corresponds to a specific figure or example from the paper,
with test methods that verify FlowBook's behavior matches exactly what the
paper describes. References use the format: Figure N / Section N.M.

Paper structure:
  - Section 2 (Motivating Example): healthcare.ipynb scenario
  - Section 3 (Semantics): formal definitions of notebook operations
  - Section 4 (Dynamic Analysis):
      - Figure 5: Four predicate violations (litmus tests)
      - Figure 6: Four staleness scenarios
      - Definition 4.1: Rerun Consistent Accesses predicates
  - Section 5 (Implementation):
      - Figure 4: Element-wise DataFrame mutation example
  - Related Work:
      - IPyflow irreproducible behavior (add_column.ipynb)
      - Marimo irreproducible behavior (mutate.ipynb)
"""

import pytest
import numpy as np
import pandas as pd

from flowbook.kernel.tests.conftest import make_tracking, ReproducibilityTestHelper
from flowbook.kernel.models import ErrorType, ReasonType
from flowbook.kernel.locations import ReadLoc, WriteLoc, writelocset_var_names, readlocset_var_names


# =============================================================================
# Helper
# =============================================================================


def _find_error(result, error_type):
    """Find the first error of a given type in the result."""
    for e in result.errors:
        if e.error_type == error_type:
            return e
    return None


def _has_error(result, error_type):
    """Check if result contains an error of a given type."""
    return _find_error(result, error_type) is not None


def _stale_reasons(result, cell_id):
    """Get all reason types for a stale cell."""
    reasons = result.staleness_reasons.get(cell_id, [])
    return {r["type"] for r in reasons}


# =============================================================================
# Figure 5: Four Predicate Violations (Litmus Tests)
# Section 4.2, Definition 4.1
# =============================================================================


class TestFigure5_PredicateViolations:
    """
    Figure 5: Scenarios violating each predicate from Definition 4.1.

    These are the four litmus tests showing one violation per predicate:
      1. NoReadAndWrite:  R_i ∩ W_i ≠ ∅
      2. WriteBeforeRead: R_i ⊄ W_{1..i-1}
      3. NoReadBeforeWrite: R_i ∩ W_{i+1..n} ≠ ∅
      4. NoWriteAfterRead: W_i ∩ R_{1..i-1} ≠ ∅
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    # ------------------------------------------------------------------
    # Test 1: NoReadAndWrite violation (Figure 5, panel 1)
    #
    # @A: x = 0
    # @B: x = x + 1    ← Both reads and writes x
    #
    # Run A; Run B → B rejected: "Both reads and writes x"
    # ------------------------------------------------------------------

    def test_no_read_and_write_violation(self):
        """Figure 5.1: Cell B reads and writes x → NoReadAndWrite violation."""
        self.helper.set_cell_order(["a", "b"])

        # Run A: x = 0
        result_a = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 0},
            writes={"x"},
        )
        assert not result_a.has_errors()

        # Run B: x = x + 1 (reads x, writes x)
        result_b = self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"x": 0},
            post_namespace={"x": 1},
            reads={"x"},
            writes={"x"},
        )
        assert result_b.has_errors()
        err = _find_error(result_b, ErrorType.NO_READ_AND_WRITE)
        assert err is not None, f"Expected NO_READ_AND_WRITE, got {[e.error_type for e in result_b.errors]}"
        assert "x" in err.locations

    # ------------------------------------------------------------------
    # Test 2: WriteBeforeRead violation (Figure 5, panel 2)
    #
    # @C: df = pd.read_csv("data.csv")   [not yet executed]
    # @D: print(df.head())
    #
    # Run D (skipping C) → D rejected: "df not written by cell above"
    # ------------------------------------------------------------------

    def test_write_before_read_violation(self):
        """
        Figure 5.2: Cell D reads df, but C above hasn't been run → WriteBeforeRead violation.

        The implementation only flags WriteBeforeRead for variables not in the
        namespace (undefined reads). To match the paper, df must not exist.
        """
        self.helper.set_cell_order(["c", "d"])

        # Run D without running C first.
        # D reads df, but no cell above has written df.
        # df is NOT in the namespace (C never ran), so this is undefined.
        result_d = self.helper.execute_cell(
            cell_id="d",
            pre_namespace={},
            post_namespace={},
            reads={"df"},
        )
        assert result_d.has_errors()
        err = _find_error(result_d, ErrorType.WRITE_BEFORE_READ)
        assert err is not None, f"Expected WRITE_BEFORE_READ, got {[e.error_type for e in result_d.errors]}"
        assert "df" in err.locations

    # ------------------------------------------------------------------
    # Test 3: NoReadBeforeWrite violation (Figure 5, panel 3)
    #
    # @E: df["price"].mean()
    # @F: df["price"] = [100, 200, 300]
    #
    # Run F; Run E → E rejected:
    #   "Reads df["price"] already written by F below"
    # ------------------------------------------------------------------

    def test_no_read_before_write_violation(self):
        """Figure 5.3: Cell E reads df["price"] written by F below → NoReadBeforeWrite violation."""
        self.helper.set_cell_order(["e", "f"])

        # First, a setup cell that creates df (implicit: some cell wrote df)
        # For this test, we need F to have been executed first.

        df = pd.DataFrame({"price": [10, 20, 30]})

        # Run F first: df["price"] = [100, 200, 300]
        df_after_f = df.copy()
        df_after_f["price"] = [100, 200, 300]
        result_f = self.helper.execute_cell(
            cell_id="f",
            pre_namespace={"df": df},
            post_namespace={"df": df_after_f},
            reads={"df"},
            writes={"df"},
            column_reads={"df": set()},
            column_writes={"df": {"price"}},
            continue_on_violation=True,  # F reads and writes df
        )

        # Run E: df["price"].mean() — reads df["price"] which was written by F below
        result_e = self.helper.execute_cell(
            cell_id="e",
            pre_namespace={"df": df_after_f},
            post_namespace={"df": df_after_f, "mean_price": 200.0},
            reads={"df"},
            writes={"mean_price"},
            column_reads={"df": {"price"}},
            continue_on_violation=True,
        )
        # E should be flagged: reads df["price"] written by F below
        assert _has_error(result_e, ErrorType.NO_READ_BEFORE_WRITE), \
            f"Expected NO_READ_BEFORE_WRITE, got {[e.error_type for e in result_e.errors]}"

    # ------------------------------------------------------------------
    # Test 4: NoWriteAfterRead violation (Figure 5, panel 4)
    #
    # @G: data = np.array([1, 2, 3])
    # @H: np.mean(data)
    # @I: data = np.array([10, 20, 30])
    #
    # Run G; Run H; Run I → I rejected:
    #   "Writes data already read by H above"
    # ------------------------------------------------------------------

    def test_no_write_after_read_violation(self):
        """Figure 5.4: Cell I writes data already read by H above → NoWriteAfterRead violation."""
        self.helper.set_cell_order(["g", "h", "i"])

        data_1 = np.array([1, 2, 3])

        # Run G: data = np.array([1, 2, 3])
        result_g = self.helper.execute_cell(
            cell_id="g",
            pre_namespace={},
            post_namespace={"data": data_1},
            writes={"data"},
        )
        assert not result_g.has_errors()

        # Run H: np.mean(data)
        result_h = self.helper.execute_cell(
            cell_id="h",
            pre_namespace={"data": data_1},
            post_namespace={"data": data_1},
            reads={"data"},
        )
        assert not result_h.has_errors()

        # Run I: data = np.array([10, 20, 30])
        data_2 = np.array([10, 20, 30])
        result_i = self.helper.execute_cell(
            cell_id="i",
            pre_namespace={"data": data_1},
            post_namespace={"data": data_2},
            reads=set(),
            writes={"data"},
        )
        assert result_i.has_errors()
        err = _find_error(result_i, ErrorType.NO_WRITE_AFTER_READ)
        assert err is not None, f"Expected NO_WRITE_AFTER_READ, got {[e.error_type for e in result_i.errors]}"
        assert "data" in err.locations
        assert err.causer_cell == "h"


# =============================================================================
# Figure 6: Staleness Scenarios
# Section 4.3
# =============================================================================


class TestFigure6_StalenessScenarios:
    """
    Figure 6: Scenarios demonstrating when FlowBook marks cells stale.

      1. ForwardStale (write→read): edit+rerun a cell, downstream reader stale
      2. ForwardStale (write→write): run cell that writes same var as cell above
      3. ForwardStale (delete): delete a cell, downstream reader stale
      4. BackwardStale (removed write): edit cell to stop writing a var
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    # ------------------------------------------------------------------
    # Test 1: ForwardStale write→read (Figure 6, panel 1)
    #
    # (a) Run J; Run K:
    #   @J: x = 100
    #   @K: print(x)
    #
    # (b) Edit J to "x = 9999"; Rerun J → K marked stale:
    #   "x modified by J"
    # ------------------------------------------------------------------

    def test_forward_stale_write_read(self):
        """Figure 6.1: Edit and rerun J, K becomes stale because it reads x."""
        self.helper.set_cell_order(["j", "k"])

        # (a) Run J: x = 100
        result_j = self.helper.execute_cell(
            cell_id="j",
            pre_namespace={},
            post_namespace={"x": 100},
            writes={"x"},
        )
        assert not result_j.has_errors()

        # (a) Run K: print(x)
        result_k = self.helper.execute_cell(
            cell_id="k",
            pre_namespace={"x": 100},
            post_namespace={"x": 100},
            reads={"x"},
        )
        assert not result_k.has_errors()
        assert "k" not in result_k.stale_cells

        # (b) Edit J (marks J stale, clears its R/W)
        self.helper.sdc._notebook_state.handle_edit("j")

        # (b) Rerun J: x = 9999
        result_j2 = self.helper.execute_cell(
            cell_id="j",
            pre_namespace={},
            post_namespace={"x": 9999},
            writes={"x"},
        )
        assert not result_j2.has_errors()
        # K should be stale: x modified by J
        assert "k" in result_j2.stale_cells

    # ------------------------------------------------------------------
    # Test 2: ForwardStale write→write (Figure 6, panel 2)
    #
    # @L: y = 10
    # @M: y = 20
    #
    # Run M; Run L → M stale: "y modified by L"
    #
    # Note: run M first (out of order), then L. L writes y, which M
    # also writes, so M is forward-stale.
    # ------------------------------------------------------------------

    def test_forward_stale_write_write(self):
        """Figure 6.2: Run M then L. L writes y, M also writes y → M stale."""
        self.helper.set_cell_order(["l", "m"])

        # Run M first (out of order): y = 20
        result_m = self.helper.execute_cell(
            cell_id="m",
            pre_namespace={},
            post_namespace={"y": 20},
            writes={"y"},
        )
        # M reads no variables not written by cells above, but "y" is not
        # written by any cell above M. However, M only writes y, doesn't
        # read it, so WriteBeforeRead doesn't apply (reads is empty).
        assert not _has_error(result_m, ErrorType.NO_WRITE_AFTER_READ)

        # Run L: y = 10
        result_l = self.helper.execute_cell(
            cell_id="l",
            pre_namespace={},
            post_namespace={"y": 10},
            writes={"y"},
        )
        assert not result_l.has_errors()
        # M should be stale: y modified by L (ForwardStale: W'_L ∩ W_M ≠ ∅)
        assert "m" in result_l.stale_cells

    # ------------------------------------------------------------------
    # Test 3: ForwardStale delete (Figure 6, panel 3)
    #
    # (a) Run N; Run O:
    #   @N: df = load_data()
    #   @O: df.describe()
    #
    # (b) Delete N → O stale: "df modified by a deleted cell"
    # ------------------------------------------------------------------

    def test_forward_stale_delete(self):
        """Figure 6.3: Delete N, O becomes stale because N wrote df."""
        self.helper.set_cell_order(["n", "o"])

        df = pd.DataFrame({"x": [1, 2, 3]})

        # (a) Run N: df = load_data()
        result_n = self.helper.execute_cell(
            cell_id="n",
            pre_namespace={},
            post_namespace={"df": df},
            writes={"df"},
        )
        assert not result_n.has_errors()

        # (a) Run O: df.describe()
        result_o = self.helper.execute_cell(
            cell_id="o",
            pre_namespace={"df": df},
            post_namespace={"df": df},
            reads={"df"},
        )
        assert not result_o.has_errors()
        assert "o" not in result_o.stale_cells

        # (b) Delete N
        self.helper.sdc._notebook_state.handle_delete("n")

        # O should now be stale
        status_o = self.helper.sdc._notebook_state.status.get("o")
        assert status_o is not None
        assert not status_o.is_clean, "O should be stale after deleting N"
        # Check reason mentions df and forward_stale
        reason_types = {r.type for r in status_o.reasons}
        assert ReasonType.FORWARD_STALE in reason_types, \
            f"Expected FORWARD_STALE, got {reason_types}"

    # ------------------------------------------------------------------
    # Test 4: BackwardStale removed write (Figure 6, panel 4)
    #
    # (a) Run P; Run Q; Run R:
    #   @P: z = 0
    #   @Q: z = 99
    #   @R: print(z)
    #
    # (b) Edit Q to "other = 10" (stops writing z); Rerun Q:
    #   → P stale (BackwardStale: Q stopped writing z, P is LastWriter)
    #   → R stale (ForwardStale: z no longer written by Q)
    # ------------------------------------------------------------------

    def test_backward_stale_removed_write(self):
        """Figure 6.4: Edit Q to remove z write → P and R both become stale."""
        self.helper.set_cell_order(["p", "q", "r"])

        # (a) Run P: z = 0
        result_p = self.helper.execute_cell(
            cell_id="p",
            pre_namespace={},
            post_namespace={"z": 0},
            writes={"z"},
        )
        assert not result_p.has_errors()

        # (a) Run Q: z = 99
        result_q = self.helper.execute_cell(
            cell_id="q",
            pre_namespace={"z": 0},
            post_namespace={"z": 99},
            writes={"z"},
        )
        assert not _has_error(result_q, ErrorType.NO_WRITE_AFTER_READ)

        # (a) Run R: print(z)
        result_r = self.helper.execute_cell(
            cell_id="r",
            pre_namespace={"z": 99},
            post_namespace={"z": 99},
            reads={"z"},
        )
        assert not result_r.has_errors()

        # (b) Edit Q (marks Q stale, clears R/W)
        self.helper.sdc._notebook_state.handle_edit("q")

        # (b) Rerun Q: other = 10 (no longer writes z)
        result_q2 = self.helper.execute_cell(
            cell_id="q",
            pre_namespace={"z": 0},
            post_namespace={"z": 0, "other": 10},
            writes={"other"},
        )

        # P should be stale (BackwardStale: Q stopped writing z, P is LastWriter)
        assert "p" in result_q2.stale_cells, \
            f"P should be backward-stale, stale cells: {result_q2.stale_cells}"

        # R should be stale (ForwardStale: z was written by old Q, now removed)
        assert "r" in result_q2.stale_cells, \
            f"R should be forward-stale, stale cells: {result_q2.stale_cells}"


# =============================================================================
# Section 2: Motivating Example (healthcare.ipynb)
# Figure 1 & surrounding text
# =============================================================================


class TestMotivatingExample_Healthcare:
    """
    Section 2: Motivating Example (healthcare.ipynb).

    Notebook structure:
      @A: import pandas as pd; from scipy.stats import pearsonr, ttest_ind
      @B: df = load_data()
         ... (other cells elided)
      @C: pearsonr(df["age"], df["spending"])
         ...
      @E: ttest_ind(df[df["state"]=="TX"]["spending"], ...)
         ...
      @F: df["income"].hist(bins=20)

    Scenarios:
      1. Run all cells → all clean
      2. Edit B to filter income>0 (B') → C, E, F stale
      3. Rerun C, E, F → all clean again
      4. Insert D after C: df["age"] = normalize(df["age"]) → rejected
      5. Edit B again to income==0 (B'') → C, E, F stale again
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c", "e", "f"])

    def _run_initial_notebook(self):
        """Run all cells top to bottom: A, B, C, E, F."""
        df = pd.DataFrame({
            "age": [25, 35, 45, 55, 65],
            "spending": [100, 200, 300, 400, 500],
            "income": [0, 30000, 50000, 70000, 0],
            "state": ["TX", "NY", "TX", "NY", "TX"],
        })

        # A: imports (no variables written to namespace beyond modules)
        result_a = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"pd": pd},
            writes={"pd"},
        )
        assert not result_a.has_errors()

        # B: df = load_data()
        result_b = self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"pd": pd},
            post_namespace={"pd": pd, "df": df},
            writes={"df"},
        )
        assert not result_b.has_errors()

        # C: pearsonr(df["age"], df["spending"])
        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"pd": pd, "df": df},
            post_namespace={"pd": pd, "df": df},
            reads={"df"},
            column_reads={"df": {"age", "spending"}},
        )
        assert not result_c.has_errors()

        # E: ttest_ind(df[...]["spending"], ...)
        result_e = self.helper.execute_cell(
            cell_id="e",
            pre_namespace={"pd": pd, "df": df},
            post_namespace={"pd": pd, "df": df},
            reads={"df"},
            column_reads={"df": {"state", "spending"}},
        )
        assert not result_e.has_errors()

        # F: df["income"].hist(bins=20)
        result_f = self.helper.execute_cell(
            cell_id="f",
            pre_namespace={"pd": pd, "df": df},
            post_namespace={"pd": pd, "df": df},
            reads={"df"},
            column_reads={"df": {"income"}},
        )
        assert not result_f.has_errors()
        assert "c" not in result_f.stale_cells
        assert "e" not in result_f.stale_cells

        return df

    def test_initial_run_all_clean(self):
        """Section 2: After running all cells top-to-bottom, all are clean."""
        self._run_initial_notebook()
        state = self.helper.sdc._notebook_state
        for cell_id in ["a", "b", "c", "e", "f"]:
            assert state.is_clean(cell_id), f"Cell {cell_id} should be clean"

    def test_edit_B_marks_CEF_stale(self):
        """Section 2: Edit B to add income filter → C, E, F stale."""
        df = self._run_initial_notebook()

        # Edit B to B': df = load_data(); df = df[df["income"] > 0]
        self.helper.sdc._notebook_state.handle_edit("b")

        # Rerun B' with filtered data
        df_filtered = df[df["income"] > 0].copy()
        result_b2 = self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"pd": pd},
            post_namespace={"pd": pd, "df": df_filtered},
            writes={"df"},
        )
        assert not result_b2.has_errors()

        # C, E, F should all be stale (they all read df)
        assert "c" in result_b2.stale_cells, "C should be stale after B edit"
        assert "e" in result_b2.stale_cells, "E should be stale after B edit"
        assert "f" in result_b2.stale_cells, "F should be stale after B edit"

    def test_rerun_stale_cells_all_clean(self):
        """Section 2: After rerunning stale C, E, F → all clean."""
        df = self._run_initial_notebook()

        # Edit and rerun B'
        self.helper.sdc._notebook_state.handle_edit("b")
        df_filtered = df[df["income"] > 0].copy()
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"pd": pd},
            post_namespace={"pd": pd, "df": df_filtered},
            writes={"df"},
        )

        # Rerun C
        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"pd": pd, "df": df_filtered},
            post_namespace={"pd": pd, "df": df_filtered},
            reads={"df"},
            column_reads={"df": {"age", "spending"}},
        )
        assert not result_c.has_errors()

        # Rerun E
        result_e = self.helper.execute_cell(
            cell_id="e",
            pre_namespace={"pd": pd, "df": df_filtered},
            post_namespace={"pd": pd, "df": df_filtered},
            reads={"df"},
            column_reads={"df": {"state", "spending"}},
        )
        assert not result_e.has_errors()

        # Rerun F
        result_f = self.helper.execute_cell(
            cell_id="f",
            pre_namespace={"pd": pd, "df": df_filtered},
            post_namespace={"pd": pd, "df": df_filtered},
            reads={"df"},
            column_reads={"df": {"income"}},
        )
        assert not result_f.has_errors()

        # All should be clean
        state = self.helper.sdc._notebook_state
        for cell_id in ["a", "b", "c", "e", "f"]:
            assert state.is_clean(cell_id), f"Cell {cell_id} should be clean"

    def test_insert_D_normalize_age_rejected(self):
        """
        Section 2: Insert D after C: df["age"] = normalize(df["age"]).

        D both reads and writes df["age"] → NoReadAndWrite violation.
        Also D writes df["age"] already read by C above → NoWriteAfterRead violation.

        Paper says FlowBook reports both violations:
          "Both reads and writes df["age"]"
          "Writes df["age"] already read by C above"
        """
        df = self._run_initial_notebook()

        # Insert D after C: order becomes [a, b, c, d, e, f]
        self.helper.sdc._notebook_state.handle_insert("d", 3)
        self.helper.set_cell_order(["a", "b", "c", "d", "e", "f"])

        # Run D: df["age"] = (df["age"] - df["age"].mean()) / df["age"].std()
        df_normalized = df.copy()
        df_normalized["age"] = (df["age"] - df["age"].mean()) / df["age"].std()

        result_d = self.helper.execute_cell(
            cell_id="d",
            pre_namespace={"pd": pd, "df": df},
            post_namespace={"pd": pd, "df": df_normalized},
            reads={"df"},
            writes={"df"},
            column_reads={"df": {"age"}},
            column_writes={"df": {"age"}},
            continue_on_violation=True,
        )

        # Paper says two errors:
        # 1. NoReadAndWrite: both reads and writes df["age"] (or df)
        assert _has_error(result_d, ErrorType.NO_READ_AND_WRITE), \
            f"Expected NO_READ_AND_WRITE error, got {[e.error_type for e in result_d.errors]}"

        # 2. NoWriteAfterRead: writes df["age"] already read by C above
        assert _has_error(result_d, ErrorType.NO_WRITE_AFTER_READ), \
            f"Expected NO_WRITE_AFTER_READ error, got {[e.error_type for e in result_d.errors]}"

    def test_edit_B_to_zero_income_marks_stale(self):
        """Section 2: Edit B to income==0 filter → C, E, F stale again."""
        df = self._run_initial_notebook()

        # First edit: filter income > 0
        self.helper.sdc._notebook_state.handle_edit("b")
        df_filtered = df[df["income"] > 0].copy()
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"pd": pd},
            post_namespace={"pd": pd, "df": df_filtered},
            writes={"df"},
        )
        # Rerun stale cells
        for cell_id in ["c", "e", "f"]:
            col_reads = {"df": {"age", "spending"}} if cell_id == "c" \
                else {"df": {"state", "spending"}} if cell_id == "e" \
                else {"df": {"income"}}
            self.helper.execute_cell(
                cell_id=cell_id,
                pre_namespace={"pd": pd, "df": df_filtered},
                post_namespace={"pd": pd, "df": df_filtered},
                reads={"df"},
                column_reads=col_reads,
            )

        # Second edit: B'' filter income == 0
        self.helper.sdc._notebook_state.handle_edit("b")
        df_zero = df[df["income"] == 0].copy()
        result_b3 = self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"pd": pd},
            post_namespace={"pd": pd, "df": df_zero},
            writes={"df"},
        )
        assert not result_b3.has_errors()
        assert "c" in result_b3.stale_cells
        assert "e" in result_b3.stale_cells
        assert "f" in result_b3.stale_cells


# =============================================================================
# Figure 4: Element-wise DataFrame Mutation
# Section 5 (Implementation)
# =============================================================================


class TestFigure4_ElementWiseMutation:
    """
    Figure 4: Element-wise DataFrame mutations.

    (a) Run A; Run B; Run C; Run D:
      @A: df = pd.DataFrame({"x": [0, 1]})
      @B: df.loc[0, "x"] = 99
      @C: df.loc[1, "x"] = 88
      @D: print(df["x"].tolist())  → [99, 88]

    (b) Delete C; Run D → output is still [99, 88] but
      top-to-bottom would produce [99, 1] — irreproducible!

    (c) FlowBook requires full-column assignment:
      Run A; Run B → B rejected: "Use df['x'] = ... for full-column assignment"

    Note: In the FlowBook model, element-wise mutations like df.loc[0, "x"] = 99
    are detected as in-place mutations of df (the diff catches that df changed
    without df being in the write set via proper column assignment).
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_element_wise_mutation_detected(self):
        """
        Figure 4(c): Cell B uses df.loc[0, "x"] = 99 which mutates df in-place.

        This is detected because:
        - B reads df (to access it)
        - The diff detects df changed (df.loc mutation)
        - But B didn't write df via proper column assignment
        → Unrecoverable mutation detected
        """
        self.helper.set_cell_order(["a", "b"])

        df = pd.DataFrame({"x": [0, 1]})

        # Run A: df = pd.DataFrame({"x": [0, 1]})
        result_a = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df.copy()},
            writes={"df"},
            column_writes={"df": {"x"}},
        )
        assert not result_a.has_errors()

        # Run B: df.loc[0, "x"] = 99
        # B reads df but mutates it in-place without column assignment.
        # The pre_namespace has original df, post has mutated df.
        df_original = df.copy()
        df_mutated = df.copy()
        df_mutated.loc[0, "x"] = 99

        # B reads df (to access it for .loc) and the diff will catch
        # that df was modified. Since this is an in-place mutation
        # rather than column assignment, it should be flagged.
        result_b = self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df_original},
            post_namespace={"df": df_mutated},
            reads={"df"},
            writes=set(),  # B doesn't write df via column assignment
            column_reads={"df": {"x"}},
        )
        # The system should detect this as a problem.
        # Either: UNRECOVERABLE_MUTATION (diff detects change not in writes)
        # or: NO_READ_AND_WRITE won't fire since writes is empty.
        # The diff detects df changed → flagged.
        assert result_b.has_errors(), \
            "Element-wise mutation should be detected as an error"

    def test_delete_cell_causes_staleness(self):
        """
        Figure 4(b): After deleting C, D's output is inconsistent.

        Run A, B (full-column), C (full-column), D → all clean.
        Delete C → D becomes stale.
        """
        self.helper.set_cell_order(["a", "b", "c", "d"])

        df = pd.DataFrame({"x": [0, 1]})

        # Run A: df = pd.DataFrame({"x": [0, 1]})
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df.copy()},
            writes={"df"},
            column_writes={"df": {"x"}},
        )

        # Run B: df["x"] = [99, df["x"][1]] (full-column assignment)
        df_b = df.copy()
        df_b["x"] = [99, 1]
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df.copy()},
            post_namespace={"df": df_b},
            reads={"df"},
            writes={"df"},
            column_reads={"df": {"x"}},
            column_writes={"df": {"x"}},
            continue_on_violation=True,
        )

        # Run C: df["x"] = [99, 88] (full-column assignment)
        df_c = df_b.copy()
        df_c["x"] = [99, 88]
        self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df_b},
            post_namespace={"df": df_c},
            reads={"df"},
            writes={"df"},
            column_reads={"df": {"x"}},
            column_writes={"df": {"x"}},
            continue_on_violation=True,
        )

        # Run D: print(df["x"].tolist())
        self.helper.execute_cell(
            cell_id="d",
            pre_namespace={"df": df_c},
            post_namespace={"df": df_c},
            reads={"df"},
            column_reads={"df": {"x"}},
        )

        # Delete C
        self.helper.sdc._notebook_state.handle_delete("c")

        # D should be stale (it read df["x"] which C wrote)
        status_d = self.helper.sdc._notebook_state.status.get("d")
        assert not status_d.is_clean, "D should be stale after deleting C"


# =============================================================================
# Related Work: IPyflow Irreproducible Behavior (add_column.ipynb)
# =============================================================================


class TestIPyflowComparison:
    """
    Related work comparison: add_column.ipynb.

    IPyflow permits this irreproducible sequence:
      @A: import pandas as pd; df = pd.DataFrame({'a': [1, 2, 3]})
      @B: df.sum()     [execution order: 3]
      @C: df['b'] = [4, 5, 6]  [execution order: 2]

    After running A, C, B in that order, B's output shows both columns
    (a: 6, b: 15) because B was run after C added column 'b'.
    But top-to-bottom would show only (a: 6) since C hasn't run yet.

    FlowBook detects this: when B is run after C, B reads df['b']
    which was written by C below → NoReadBeforeWrite violation.
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["a", "b", "c"])

    def test_add_column_forward_contamination(self):
        """IPyflow comparison: df.sum() reads column 'b' written by C below."""
        df_initial = pd.DataFrame({"a": [1, 2, 3]})

        # Run A: create df
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"pd": pd, "df": df_initial.copy()},
            writes={"pd", "df"},
            column_writes={"df": {"a"}},
        )

        # Run C (out of order): df['b'] = [4, 5, 6]
        df_with_b = df_initial.copy()
        df_with_b["b"] = [4, 5, 6]
        self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"pd": pd, "df": df_initial.copy()},
            post_namespace={"pd": pd, "df": df_with_b},
            reads={"df"},
            writes={"df"},
            column_reads={"df": set()},
            column_writes={"df": {"b"}},
            continue_on_violation=True,
        )

        # Run B (after C): df.sum()
        # df.sum() aggregates across all columns, which is a structural read
        # on the column structure (Attr(df, columns)). C below added column 'b'
        # (ColAdd(df, b)), which conflicts with Attr(df, columns).
        result_b = self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"pd": pd, "df": df_with_b},
            post_namespace={"pd": pd, "df": df_with_b},
            reads={"df"},
            structural_reads={"df": {"columns"}},  # df.sum() depends on column structure
            continue_on_violation=True,
        )
        # FlowBook detects forward contamination: B reads Attr(df, columns),
        # C below wrote ColAdd(df, b) which conflicts with it
        assert _has_error(result_b, ErrorType.NO_READ_BEFORE_WRITE), \
            f"Expected NO_READ_BEFORE_WRITE, got {[e.error_type for e in result_b.errors]}"


# =============================================================================
# Related Work: Marimo Irreproducible Behavior (mutate.ipynb)
# =============================================================================


class TestMarimoComparison:
    """
    Related work comparison: mutate.ipynb.

    Marimo permits this irreproducible sequence when E is run twice:
      @D: import pandas as pd; df = pd.DataFrame({'a': [1, 2, 3]})
      @E: df['a'] = df['a'] * 10
      @F: df['a'].sum()

    Running D, E, F gives sum=60. Running E again gives sum=600 (not 60).
    FlowBook catches this: E both reads and writes df['a'] → NoReadAndWrite.
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()
        self.helper.set_cell_order(["d", "e", "f"])

    def test_marimo_mutate_no_read_and_write(self):
        """Marimo comparison: df['a'] = df['a'] * 10 reads and writes df['a']."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        # Run D: create df
        self.helper.execute_cell(
            cell_id="d",
            pre_namespace={},
            post_namespace={"pd": pd, "df": df.copy()},
            writes={"pd", "df"},
            column_writes={"df": {"a"}},
        )

        # Run E: df['a'] = df['a'] * 10
        df_mutated = df.copy()
        df_mutated["a"] = df["a"] * 10
        result_e = self.helper.execute_cell(
            cell_id="e",
            pre_namespace={"pd": pd, "df": df.copy()},
            post_namespace={"pd": pd, "df": df_mutated},
            reads={"df"},
            writes={"df"},
            column_reads={"df": {"a"}},
            column_writes={"df": {"a"}},
        )
        # E both reads and writes df (and df['a']) → NoReadAndWrite
        assert _has_error(result_e, ErrorType.NO_READ_AND_WRITE), \
            f"Expected NO_READ_AND_WRITE, got {[e.error_type for e in result_e.errors]}"


# =============================================================================
# Definition 4.1: Predicate Combinations
# Additional tests for predicate interactions
# =============================================================================


class TestPredicateCombinations:
    """
    Tests for scenarios where multiple predicates interact, derived from
    the formal definitions in Section 4.

    Definition 4.1 gives four independent predicates. These tests verify
    that the system correctly handles scenarios involving multiple
    predicates simultaneously.
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_all_predicates_pass_clean_notebook(self):
        """
        All four predicates satisfied → cell is clean.

        @A: x = 1
        @B: y = x + 1
        @C: z = y + 1

        Linear dependency chain, all predicates pass for each cell.
        """
        self.helper.set_cell_order(["a", "b", "c"])

        result_a = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 1},
            writes={"x"},
        )
        assert not result_a.has_errors()

        result_b = self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"x": 1},
            post_namespace={"x": 1, "y": 2},
            reads={"x"},
            writes={"y"},
        )
        assert not result_b.has_errors()

        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2, "z": 3},
            reads={"y"},
            writes={"z"},
        )
        assert not result_c.has_errors()

        state = self.helper.sdc._notebook_state
        assert state.is_clean("a")
        assert state.is_clean("b")
        assert state.is_clean("c")

    def test_write_before_read_undefined_variable(self):
        """
        WriteBeforeRead fires when variable is not in namespace.

        @A: (not yet executed, would write x)
        @B: y = 1
        @C: z = x + y   ← x not in namespace (WriteBeforeRead)

        Scenario: Run B, then C (skipping A). x doesn't exist.
        """
        self.helper.set_cell_order(["a", "b", "c"])

        # Run B: y = 1
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={},
            post_namespace={"y": 1},
            writes={"y"},
        )

        # Run C: reads x (not in namespace at all) and y
        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"y": 1},  # x not in namespace
            post_namespace={"y": 1, "z": 43},
            reads={"x", "y"},
            writes={"z"},
            continue_on_violation=True,
        )
        # x not in namespace and not written by any cell above → WriteBeforeRead
        assert _has_error(result_c, ErrorType.WRITE_BEFORE_READ), \
            f"Expected WRITE_BEFORE_READ, got {[e.error_type for e in result_c.errors]}"

    def test_read_and_write_same_var_multiple_cells(self):
        """
        NoReadAndWrite is per-cell, not per-variable.

        @A: x = 1
        @B: y = x      ← reads x, writes y → OK (different vars)
        @C: x = x + 1  ← reads x, writes x → NoReadAndWrite

        Only C violates NoReadAndWrite.
        """
        self.helper.set_cell_order(["a", "b", "c"])

        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 1},
            writes={"x"},
        )

        result_b = self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"x": 1},
            post_namespace={"x": 1, "y": 1},
            reads={"x"},
            writes={"y"},
        )
        assert not _has_error(result_b, ErrorType.NO_READ_AND_WRITE)

        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"x": 1, "y": 1},
            post_namespace={"x": 2, "y": 1},
            reads={"x"},
            writes={"x"},
        )
        assert _has_error(result_c, ErrorType.NO_READ_AND_WRITE)

    def test_out_of_order_forward_contamination_marked_stale(self):
        """
        Section 4: Forward contamination marks cell stale (not rejected).

        @A: x = 1
        @B: print(x)

        Run B first (prints x from store), then run A.
        B read x which was written by... nobody at the time.
        Then A runs and writes x → B should become stale.
        """
        self.helper.set_cell_order(["a", "b"])

        # Run B first: reads x (already in namespace somehow)
        result_b = self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"x": 0},
            post_namespace={"x": 0},
            reads={"x"},
            continue_on_violation=True,
        )

        # Run A: x = 1
        result_a = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 1},
            writes={"x"},
        )
        assert not result_a.has_errors()
        # B should be stale: ForwardStale(W'_A ∩ R_B = {x} ≠ ∅)
        assert "b" in result_a.stale_cells


# =============================================================================
# Section 3: Formal Semantics — Notebook Operations
# Tests for Edit, Insert, Delete, Move operations
# =============================================================================


class TestNotebookOperations:
    """
    Section 3: Notebook operations from the formal semantics.

    Tests for:
      - Edit(i, c): marks cell stale
      - Insert(i, c): new cell is stale
      - Delete(i): propagates staleness
      - Move(i, j): propagates staleness
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_edit_marks_cell_stale(self):
        """[Inst-Edit]: T' = T[i := stale]"""
        self.helper.set_cell_order(["a", "b"])

        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 1},
            writes={"x"},
        )
        state = self.helper.sdc._notebook_state
        assert state.is_clean("a")

        # Edit A
        state.handle_edit("a")
        assert not state.is_clean("a"), "Edited cell should be stale"

    def test_insert_new_cell_is_stale(self):
        """[Inst-Insert]: T' = T_{1..i-1} · stale · T_{i..n}"""
        self.helper.set_cell_order(["a", "c"])

        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"x": 1},
            post_namespace={"x": 1, "y": 2},
            reads={"x"},
            writes={"y"},
        )

        state = self.helper.sdc._notebook_state
        assert state.is_clean("a")
        assert state.is_clean("c")

        # Insert B between A and C
        state.handle_insert("b", 1)

        # B should be stale (never executed)
        assert not state.is_clean("b"), "Inserted cell should be stale"
        # A and C should still be clean
        assert state.is_clean("a"), "A should remain clean"
        assert state.is_clean("c"), "C should remain clean"

    def test_delete_propagates_forward_staleness(self):
        """
        [Inst-Delete]: ForwardStale for cells that read deleted cell's writes.
        """
        self.helper.set_cell_order(["a", "b"])

        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"x": 1},
            post_namespace={"x": 1, "y": 2},
            reads={"x"},
            writes={"y"},
        )

        state = self.helper.sdc._notebook_state
        assert state.is_clean("b")

        # Delete A (which wrote x that B reads)
        state.handle_delete("a")

        # B should be stale
        assert not state.is_clean("b"), "B should be stale after deleting A"

    def test_delete_propagates_backward_staleness(self):
        """
        [Inst-Delete]: BackwardStale for LastWriter of deleted cell's writes.

        @P: z = 0
        @Q: z = 99
        Delete Q → P should become backward-stale (P is LastWriter of z before Q).
        """
        self.helper.set_cell_order(["p", "q", "r"])

        self.helper.execute_cell(
            cell_id="p",
            pre_namespace={},
            post_namespace={"z": 0},
            writes={"z"},
        )
        self.helper.execute_cell(
            cell_id="q",
            pre_namespace={"z": 0},
            post_namespace={"z": 99},
            writes={"z"},
        )
        self.helper.execute_cell(
            cell_id="r",
            pre_namespace={"z": 99},
            post_namespace={"z": 99},
            reads={"z"},
        )

        state = self.helper.sdc._notebook_state
        assert state.is_clean("p")
        assert state.is_clean("r")

        # Delete Q
        state.handle_delete("q")

        # P should be backward-stale (P is LastWriter of z before Q)
        assert not state.is_clean("p"), "P should be backward-stale"
        # R should be forward-stale (it read z which Q wrote)
        assert not state.is_clean("r"), "R should be forward-stale"


# =============================================================================
# Section 4: ForwardStale and BackwardStale Formal Definitions
# =============================================================================


class TestStalenessDefinitions:
    """
    Tests for the formal staleness definitions from Section 4.3:

    ForwardStale(R, W, W', i, j) ≡ j > i ∧ (W_i ∪ W'_i) ∩ (R_j ∪ W_j) ≠ ∅
    BackwardStale(W, W', i, j) ≡ j < i ∧ j = LastWriter(W, i, ℓ) for some ℓ ∈ W_i \ W'_i
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_forward_stale_reads_overlap(self):
        """ForwardStale: W'_i ∩ R_j ≠ ∅ → j stale (write→read)."""
        self.helper.set_cell_order(["a", "b"])

        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 1},
            writes={"x"},
        )
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"x": 1},
            post_namespace={"x": 1},
            reads={"x"},
        )

        # Edit and rerun A with different value
        self.helper.sdc._notebook_state.handle_edit("a")
        result = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 999},
            writes={"x"},
        )
        assert "b" in result.stale_cells

    def test_forward_stale_writes_overlap(self):
        """ForwardStale: W'_i ∩ W_j ≠ ∅ → j stale (write→write)."""
        self.helper.set_cell_order(["a", "b"])

        # B writes x too
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={},
            post_namespace={"x": 2},
            writes={"x"},
        )

        # Run A (writes x) → B stale because W'_A ∩ W_B = {x}
        result = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 1},
            writes={"x"},
        )
        assert "b" in result.stale_cells

    def test_forward_stale_old_writes_overlap(self):
        """ForwardStale: W_i (old writes) also contributes. (W_i ∪ W'_i) ∩ R_j."""
        self.helper.set_cell_order(["a", "b"])

        # A writes x and y
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 1, "y": 2},
            writes={"x", "y"},
        )
        # B reads y
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"x": 1, "y": 2},
            post_namespace={"x": 1, "y": 2},
            reads={"y"},
        )

        # Edit A, now only writes x (not y anymore)
        self.helper.sdc._notebook_state.handle_edit("a")
        result = self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 10},
            writes={"x"},
        )
        # B should still be stale because W_A (old) included y, and B reads y
        # ForwardStale: (W_A ∪ W'_A) ∩ R_B = ({x,y} ∪ {x}) ∩ {y} = {y} ≠ ∅
        assert "b" in result.stale_cells

    def test_backward_stale_last_writer(self):
        """BackwardStale: ℓ ∈ W_i \ W'_i and j = LastWriter(W, i, ℓ)."""
        self.helper.set_cell_order(["a", "b"])

        # A writes x
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"x": 1},
            writes={"x"},
        )
        # B writes x (overrides A's x)
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"x": 1},
            post_namespace={"x": 99},
            writes={"x"},
        )

        state = self.helper.sdc._notebook_state
        assert state.is_clean("a")

        # Edit B, now only writes y (not x)
        state.handle_edit("b")
        result = self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"x": 1},
            post_namespace={"x": 1, "y": 5},
            writes={"y"},
        )

        # A is LastWriter of x before B, and x ∈ W_B \ W'_B
        # → A should be backward-stale
        assert "a" in result.stale_cells

    def test_no_backward_stale_when_no_prior_writer(self):
        """BackwardStale: if no LastWriter exists, no cell goes backward-stale."""
        self.helper.set_cell_order(["a", "b"])

        # Only B writes x (no cell above writes x)
        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"y": 1},
            writes={"y"},
        )
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"y": 1},
            post_namespace={"y": 1, "x": 99},
            writes={"x"},
        )

        # Edit B, stop writing x
        self.helper.sdc._notebook_state.handle_edit("b")
        result = self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"y": 1},
            post_namespace={"y": 1, "z": 5},
            writes={"z"},
        )
        # No LastWriter for x above B → A should NOT be backward-stale
        assert "a" not in result.stale_cells


# =============================================================================
# Section 4: Column-Level Tracking
# DataFrame column granularity examples
# =============================================================================


class TestColumnGranularity:
    """
    Tests for column-level tracking from the formal model.

    The paper defines locations as:
      ℓ ∈ Loc ::= x | d.c

    where d.c is a DataFrame column. Column-level tracking avoids
    false positives when cells operate on disjoint columns.
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_disjoint_columns_no_conflict(self):
        """
        Cells operating on disjoint columns should not conflict.

        @A: df = pd.DataFrame({"x": [1], "y": [2]})
        @B: result = df["x"].sum()   (reads df.x)
        @C: df["y"] = [99]           (writes df.y)

        C writes df.y, B reads df.x → no conflict at column level.
        """
        self.helper.set_cell_order(["a", "b", "c"])

        df = pd.DataFrame({"x": [1], "y": [2]})

        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df.copy()},
            writes={"df"},
            column_writes={"df": {"x", "y"}},
        )

        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df.copy()},
            post_namespace={"df": df.copy(), "result": 1},
            reads={"df"},
            writes={"result"},
            column_reads={"df": {"x"}},
        )

        # C writes df["y"], B reads df["x"] → disjoint, no backward conflict
        df_modified = df.copy()
        df_modified["y"] = [99]
        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df.copy(), "result": 1},
            post_namespace={"df": df_modified, "result": 1},
            reads={"df"},
            writes={"df"},
            column_reads={"df": set()},
            column_writes={"df": {"y"}},
            continue_on_violation=True,
        )
        # No backward mutation on B (disjoint columns)
        assert not _has_error(result_c, ErrorType.NO_WRITE_AFTER_READ), \
            f"Should not have backward error for disjoint columns, got {result_c.errors}"

    def test_overlapping_columns_conflict(self):
        """
        Cells operating on overlapping columns should conflict.

        @A: df = pd.DataFrame({"x": [1], "y": [2]})
        @B: result = df["x"].sum()   (reads df.x)
        @C: df["x"] = [99]           (writes df.x)

        C writes df.x, B reads df.x → conflict.
        """
        self.helper.set_cell_order(["a", "b", "c"])

        df = pd.DataFrame({"x": [1], "y": [2]})

        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df.copy()},
            writes={"df"},
            column_writes={"df": {"x", "y"}},
        )

        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df.copy()},
            post_namespace={"df": df.copy(), "result": 1},
            reads={"df"},
            writes={"result"},
            column_reads={"df": {"x"}},
        )

        df_modified = df.copy()
        df_modified["x"] = [99]
        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df.copy(), "result": 1},
            post_namespace={"df": df_modified, "result": 1},
            reads={"df"},
            writes={"df"},
            column_reads={"df": set()},
            column_writes={"df": {"x"}},
            continue_on_violation=True,
        )
        # Backward mutation: C writes df.x, B reads df.x
        assert _has_error(result_c, ErrorType.NO_WRITE_AFTER_READ), \
            f"Should have backward error for overlapping columns, got {result_c.errors}"

    def test_adding_new_column_no_stale(self):
        """
        Adding a new column should not mark readers of other columns stale.

        @A: df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        @B: print(df["x"].sum())   (reads df.x only)
        @C: df["z"] = [5, 6]      (writes df.z only)

        C adds df.z, B only reads df.x → B should NOT be stale.
        """
        self.helper.set_cell_order(["a", "b", "c"])

        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

        self.helper.execute_cell(
            cell_id="a",
            pre_namespace={},
            post_namespace={"df": df.copy()},
            writes={"df"},
            column_writes={"df": {"x", "y"}},
        )
        self.helper.execute_cell(
            cell_id="b",
            pre_namespace={"df": df.copy()},
            post_namespace={"df": df.copy()},
            reads={"df"},
            column_reads={"df": {"x"}},
        )

        df_with_z = df.copy()
        df_with_z["z"] = [5, 6]
        result_c = self.helper.execute_cell(
            cell_id="c",
            pre_namespace={"df": df.copy()},
            post_namespace={"df": df_with_z},
            reads={"df"},
            writes={"df"},
            column_reads={"df": set()},
            column_writes={"df": {"z"}},
            continue_on_violation=True,
        )

        # B should NOT be stale (reads df.x, C writes df.z — disjoint)
        assert "b" not in result_c.stale_cells, \
            f"B should not be stale when C adds unrelated column, stale: {result_c.stale_cells}"


# =============================================================================
# Section 4: Theorem 3 (Progress) — Strategy of running first stale cell
# =============================================================================


class TestProgress:
    """
    Theorem 3 (Progress): Running stale cells top-to-bottom terminates
    in either all-clean state or a stuck cell.

    These tests verify that the natural strategy works correctly.
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_linear_chain_reaches_all_clean(self):
        """
        Linear dependency chain: run each stale cell top-to-bottom.

        @A: x = 1
        @B: y = x + 1
        @C: z = y + 1

        Edit A → B, C stale. Rerun A, B, C → all clean.
        """
        self.helper.set_cell_order(["a", "b", "c"])

        # Initial run
        self.helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        self.helper.execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})
        self.helper.execute_cell("c", {"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3}, reads={"y"}, writes={"z"})

        state = self.helper.sdc._notebook_state
        assert all(state.is_clean(c) for c in ["a", "b", "c"])

        # Edit A
        state.handle_edit("a")
        assert not state.is_clean("a")

        # Rerun A → marks B stale (B reads x which A writes)
        # C is NOT stale yet: C reads y, A writes x — no overlap.
        self.helper.execute_cell("a", {}, {"x": 10}, writes={"x"})
        assert not state.is_clean("b"), "B should be stale (reads x written by A)"
        # C stays clean until B reruns (C depends on y, not x)

        # Rerun B → B becomes clean, marks C stale (C reads y which B writes)
        self.helper.execute_cell("b", {"x": 10}, {"x": 10, "y": 11}, reads={"x"}, writes={"y"})
        assert state.is_clean("b")
        assert not state.is_clean("c"), "C should be stale (reads y written by B)"

        # Rerun C → all clean
        self.helper.execute_cell("c", {"x": 10, "y": 11}, {"x": 10, "y": 11, "z": 12}, reads={"y"}, writes={"z"})
        assert all(state.is_clean(c) for c in ["a", "b", "c"])

    def test_stuck_cell_no_read_and_write(self):
        """
        Progress terminates at stuck cell.

        @A: x = 1
        @B: x = x + 1   ← stuck: NoReadAndWrite fails

        Run A → clean. Run B → rejected (stuck).
        """
        self.helper.set_cell_order(["a", "b"])

        self.helper.execute_cell("a", {}, {"x": 1}, writes={"x"})

        result_b = self.helper.execute_cell(
            "b", {"x": 1}, {"x": 2}, reads={"x"}, writes={"x"},
        )
        assert result_b.has_errors()
        # B is stuck — cannot proceed past it


# =============================================================================
# Section 4.3: Well-Formedness (Definition 4.2)
# =============================================================================


class TestWellFormedness:
    """
    Definition 4.2 (Well-Formed State): For every clean cell i,
    R and W are rerun consistent for i.

    These tests verify the well-formedness invariant is maintained
    across various operations.
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_initial_state_is_well_formed(self):
        """All cells stale initially → trivially well-formed."""
        self.helper.set_cell_order(["a", "b", "c"])
        state = self.helper.sdc._notebook_state
        # All cells should be stale (never executed)
        for cell_id in ["a", "b", "c"]:
            assert not state.is_clean(cell_id)

    def test_well_formedness_preserved_after_run(self):
        """After a valid run, well-formedness is preserved."""
        self.helper.set_cell_order(["a", "b"])

        self.helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        self.helper.execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        state = self.helper.sdc._notebook_state
        # Both clean → rerun consistency holds
        assert state.is_clean("a")
        assert state.is_clean("b")

        # Verify R and W are correctly recorded
        assert "x" in writelocset_var_names(state.writes.get("a", frozenset()))
        assert "x" in readlocset_var_names(state.reads.get("b", frozenset()))
        assert "y" in writelocset_var_names(state.writes.get("b", frozenset()))

    def test_well_formedness_preserved_after_edit(self):
        """After edit, edited cell is stale → well-formedness trivially holds for it."""
        self.helper.set_cell_order(["a", "b"])

        self.helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        self.helper.execute_cell("b", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        state = self.helper.sdc._notebook_state
        state.handle_edit("a")

        # A is stale → no consistency requirement
        assert not state.is_clean("a")
        # B should still be clean (edit doesn't propagate until rerun)
        # Actually B depends on A, but edit alone doesn't mark downstream stale
        # (that happens when A is re-executed)

    def test_well_formedness_preserved_after_insert(self):
        """Insertion preserves well-formedness for existing cells."""
        self.helper.set_cell_order(["a", "c"])

        self.helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        self.helper.execute_cell("c", {"x": 1}, {"x": 1, "y": 2}, reads={"x"}, writes={"y"})

        state = self.helper.sdc._notebook_state
        state.handle_insert("b", 1)

        # A and C should still be clean
        assert state.is_clean("a")
        assert state.is_clean("c")
        # B is stale (never executed)
        assert not state.is_clean("b")


# =============================================================================
# Section 5: Implementation — internal cell reads/writes (x = 0; x = x + 1)
# =============================================================================


class TestInternalCellReadsWrites:
    """
    Section 4.2: A cell like "x = 0; x = x + 1" does NOT read x from
    the store — it reads x within its own execution. So R = ∅, W = {x}.

    This is important: cells that only use variables they define themselves
    don't create dependencies.
    """

    def setup_method(self):
        self.helper = ReproducibilityTestHelper()

    def test_self_contained_cell_no_reads(self):
        """
        Cell "x = 0; x = x + 1" has R = ∅, W = {x}.
        It doesn't read x from the store, only from its own write.
        """
        self.helper.set_cell_order(["a", "b"])

        # A: x = 10
        self.helper.execute_cell("a", {}, {"x": 10}, writes={"x"})

        # B: x = 0; x = x + 1 → reads=∅ (self-contained), writes={x}
        # This should trigger NO_WRITE_AFTER_READ since A hasn't read x,
        # but B writes x. Actually, the predicate is W_B ∩ R_{above} = ∅.
        # A doesn't read x (it only writes x). So no violation.
        result_b = self.helper.execute_cell(
            "b", {"x": 10}, {"x": 1},
            reads=set(),  # x = 0; x = x + 1 doesn't read from store
            writes={"x"},
        )
        # No violation: W_B ∩ R_{above} = {x} ∩ ∅ = ∅
        assert not _has_error(result_b, ErrorType.NO_WRITE_AFTER_READ)

    def test_self_contained_cell_doesnt_create_dependency(self):
        """
        If A writes x, and B defines x internally (reads=∅, writes={x}),
        then B doesn't depend on A's x — it creates its own.
        Editing A should NOT mark B stale for x dependency.
        But A and B both write x, so ForwardStale might still mark B.
        """
        self.helper.set_cell_order(["a", "b"])

        self.helper.execute_cell("a", {}, {"x": 1}, writes={"x"})
        self.helper.execute_cell("b", {"x": 1}, {"x": 42}, reads=set(), writes={"x"})

        # B reads nothing from A, so R_B = ∅
        state = self.helper.sdc._notebook_state
        assert state.reads.get("b", frozenset()) == frozenset()
