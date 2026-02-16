"""Tests for tracking.py - Targeting uncovered dict protocol methods.

Coverage gaps include:
- __setitem__ with Series tracking (line 121)
- __iter__, __len__, __repr__ (lines 131, 134, 137)
- update with non-dict iterable (lines 163-164, 166)
- setdefault (lines 169-171)
- pop with default (lines 174-181)
- popitem (lines 184-185)
- clear, copy (lines 188, 191)
"""

import pytest
import pandas as pd
import numpy as np

from flowbook.kernel_support.tracking import TrackingDict


class TestTrackingDictProtocol:
    """Tests for TrackingDict dict protocol methods."""

    def test_iter(self):
        """__iter__ iterates over keys."""
        ns = {"a": 1, "b": 2, "c": 3}
        td = TrackingDict(ns)
        assert set(td) == {"a", "b", "c"}

    def test_len(self):
        """__len__ returns number of keys."""
        ns = {"a": 1, "b": 2}
        td = TrackingDict(ns)
        assert len(td) == 2

    def test_repr(self):
        """__repr__ shows TrackingDict wrapper."""
        ns = {"x": 1}
        td = TrackingDict(ns)
        r = repr(td)
        assert "TrackingDict" in r
        assert "'x': 1" in r

    def test_update_with_dict(self):
        """update with a dict uses __setitem__ for tracking."""
        ns = {}
        td = TrackingDict(ns)
        td.update({"x": 1, "y": 2})
        assert ns["x"] == 1
        assert ns["y"] == 2
        assert "x" in td.writes
        assert "y" in td.writes

    def test_update_with_iterable(self):
        """update with key-value iterable."""
        ns = {}
        td = TrackingDict(ns)
        td.update([("a", 10), ("b", 20)])
        assert ns["a"] == 10
        assert ns["b"] == 20

    def test_update_with_kwargs(self):
        """update with keyword arguments."""
        ns = {}
        td = TrackingDict(ns)
        td.update(x=1, y=2)
        assert ns["x"] == 1
        assert ns["y"] == 2

    def test_setdefault_new_key(self):
        """setdefault sets and returns default for new key."""
        ns = {}
        td = TrackingDict(ns)
        result = td.setdefault("x", 42)
        assert result == 42
        assert ns["x"] == 42

    def test_setdefault_existing_key(self):
        """setdefault returns existing value for existing key."""
        ns = {"x": 10}
        td = TrackingDict(ns)
        result = td.setdefault("x", 42)
        assert result == 10
        assert ns["x"] == 10

    def test_pop_existing(self):
        """pop removes and returns value."""
        ns = {"x": 42}
        td = TrackingDict(ns)
        result = td.pop("x")
        assert result == 42
        assert "x" not in ns

    def test_pop_with_default(self):
        """pop returns default for missing key."""
        ns = {}
        td = TrackingDict(ns)
        result = td.pop("x", "default")
        assert result == "default"

    def test_pop_missing_no_default(self):
        """pop raises KeyError for missing key without default."""
        ns = {}
        td = TrackingDict(ns)
        with pytest.raises(KeyError):
            td.pop("x")

    def test_popitem(self):
        """popitem removes and returns an arbitrary item."""
        ns = {"x": 42}
        td = TrackingDict(ns)
        key, value = td.popitem()
        assert key == "x"
        assert value == 42
        assert len(ns) == 0

    def test_clear(self):
        """clear empties the namespace."""
        ns = {"a": 1, "b": 2}
        td = TrackingDict(ns)
        td.clear()
        assert len(ns) == 0

    def test_copy(self):
        """copy returns a plain dict copy."""
        ns = {"x": 1, "y": 2}
        td = TrackingDict(ns)
        result = td.copy()
        assert result == {"x": 1, "y": 2}
        assert isinstance(result, dict)
        assert not isinstance(result, TrackingDict)


class TestTrackingDictSeriesTracking:
    """Tests for Series tracking in TrackingDict."""

    def test_setitem_series_tracked(self):
        """Setting a Series triggers structural tracker registration."""
        ns = {}
        td = TrackingDict(ns)
        s = pd.Series([1, 2, 3])
        td["my_series"] = s
        assert ns["my_series"] is s
        assert "my_series" in td.writes

    def test_getitem_series_tracked(self):
        """Getting a Series triggers structural tracker registration."""
        s = pd.Series([1, 2, 3])
        ns = {"my_series": s}
        td = TrackingDict(ns)
        result = td["my_series"]
        assert result is s
        assert "my_series" in td.reads_before_writes

    def test_setitem_dataframe_tracked(self):
        """Setting a DataFrame triggers column and structural registration."""
        ns = {}
        td = TrackingDict(ns)
        df = pd.DataFrame({"a": [1, 2]})
        td["my_df"] = df
        assert ns["my_df"] is df
        assert "my_df" in td.writes


class TestTrackingDictContextManagers:
    """Tests for TrackingDict context managers."""

    def test_track_execution_context(self):
        """track_execution enables and disables tracking."""
        ns = {"x": 1}
        td = TrackingDict(ns)
        with td.track_execution():
            _ = td["x"]
            td["y"] = 2
        assert "x" in td.reads_before_writes
        assert "y" in td.writes
        # Tracking should be disabled after context
        assert not td._tracking_enabled

    def test_suspended_context(self):
        """suspended context prevents tracking."""
        ns = {"x": 1}
        td = TrackingDict(ns)
        td._tracking_enabled = True
        with td.suspended():
            assert not td._tracking_enabled
            td["y"] = 2
        assert td._tracking_enabled
        assert "y" not in td.writes

    def test_suspended_restores_previous_state(self):
        """suspended restores the previous tracking state."""
        ns = {}
        td = TrackingDict(ns)
        td._tracking_enabled = False
        with td.suspended():
            pass
        assert not td._tracking_enabled


class TestTrackingDictNoNamespace:
    """Tests for TrackingDict without a namespace (standalone)."""

    def test_no_namespace(self):
        """TrackingDict works without a namespace argument."""
        td = TrackingDict()
        td["x"] = 42
        assert td["x"] == 42
        assert "x" in td
        assert len(td) == 1
