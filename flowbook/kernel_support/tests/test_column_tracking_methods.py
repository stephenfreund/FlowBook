"""
Tests for DataFrame aggregation method column tracking.

Verifies that .sum(), .mean(), .describe(), etc. record column reads
so that column-level staleness works correctly with Var(x) = binding-only.
"""

import pytest
import pandas as pd
from packaging.version import Version

from flowbook.kernel_support.column_tracking import ColumnAccessTracker

# Pandas 2.2.x with future.infer_string=True routes string-column reductions
# through pyarrow, but lacks a fallback when pyarrow's compute kernel doesn't
# accept the input (e.g. `sum` on `large_string` raises ArrowNotImplementedError).
# Pandas 2.3 added that fallback. Skip the affected case until the floor moves.
PANDAS_LT_2_3 = Version(pd.__version__) < Version("2.3")


@pytest.fixture
def tracker():
    """Create and activate a column tracker."""
    t = ColumnAccessTracker()
    t.activate()
    yield t
    t.deactivate()


@pytest.fixture
def df():
    """Sample DataFrame with mixed types."""
    return pd.DataFrame({
        "price": [10, 20, 30],
        "qty": [1, 2, 3],
        "name": ["a", "b", "c"],
    })


def _get_reads(tracker, df):
    """Register df and return its read set."""
    df_id = id(df)
    tracker.register_df(df, "df")
    return tracker._reads_by_id[df_id]


class TestAggregationMethodTracking:
    """Tests that aggregation methods record column reads."""

    def test_sum_records_all_columns(self, tracker, df):
        reads = _get_reads(tracker, df)
        df.sum(numeric_only=True)
        assert "price" in reads
        assert "qty" in reads
        assert "name" in reads  # conservative: records all columns

    def test_mean_records_all_columns(self, tracker, df):
        reads = _get_reads(tracker, df)
        df.mean(numeric_only=True)
        assert "price" in reads
        assert "qty" in reads

    def test_describe_records_all_columns(self, tracker, df):
        reads = _get_reads(tracker, df)
        df.describe()
        assert "price" in reads
        assert "qty" in reads

    def test_std_records_columns(self, tracker, df):
        reads = _get_reads(tracker, df)
        df.std(numeric_only=True)
        assert "price" in reads

    @pytest.mark.skipif(
        PANDAS_LT_2_3,
        reason=(
            "pandas 2.2.x bug: df.apply(lambda x: x.sum()) on a string column "
            "raises ArrowNotImplementedError instead of falling back. "
            "Fixed in pandas 2.3."
        ),
    )
    def test_apply_records_all_columns(self, tracker, df):
        reads = _get_reads(tracker, df)
        df.apply(lambda x: x.sum())
        assert "price" in reads
        assert "qty" in reads
        assert "name" in reads

    def test_to_numpy_records_all_columns(self, tracker, df):
        reads = _get_reads(tracker, df)
        df.to_numpy()
        assert "price" in reads
        assert "qty" in reads
        assert "name" in reads

    def test_values_records_all_columns(self, tracker, df):
        reads = _get_reads(tracker, df)
        _ = df.values
        assert "price" in reads
        assert "qty" in reads
        assert "name" in reads

    def test_min_max_records_columns(self, tracker, df):
        reads = _get_reads(tracker, df)
        df.min()
        df.max()
        assert "price" in reads
        assert "qty" in reads

    def test_nunique_records_columns(self, tracker, df):
        reads = _get_reads(tracker, df)
        df.nunique()
        assert "price" in reads

    def test_median_records_columns(self, tracker, df):
        reads = _get_reads(tracker, df)
        df.median(numeric_only=True)
        assert "price" in reads

    def test_to_dict_records_columns(self, tracker, df):
        reads = _get_reads(tracker, df)
        df.to_dict()
        assert "price" in reads
        assert "name" in reads


class TestNoTrackingWhenInactive:
    """Tests that methods don't track when tracker is inactive."""

    def test_sum_no_tracking_when_inactive(self, df):
        """No tracker active — sum should still work normally."""
        result = df.sum(numeric_only=True)
        assert "price" in result.index

    def test_unregistered_df_no_tracking(self, tracker, df):
        """Tracker active but df not registered — no reads recorded."""
        df.sum(numeric_only=True)
        reads = tracker._reads_by_id.get(id(df), set())
        assert len(reads) == 0

    def test_values_no_crash_when_inactive(self, df):
        """No tracker active — values should still work."""
        vals = df.values
        assert vals.shape == (3, 3)
