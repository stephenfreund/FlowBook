"""
Tests for StableIdMap, LocRef, and LocRef-aware conflict detection.

Tests the core innovation: weakref-based stable object identity that
survives checkpoint deep copy and detects id() reuse after GC.
"""

import gc
import weakref

import pandas as pd
import pytest

from flowbook.kernel.loc_ids import StableIdMap, LocRef, get_qualifier, build_loc_context
from flowbook.kernel.locations import (
    ReadLoc, WriteLoc,
    write_conflicts_read, has_conflict,
    tracking_to_readlocset, tracking_to_writelocset,
    _same_dataframe,
)
from flowbook.kernel_support.models import TrackingData


class TestStableIdMap:
    """Test StableIdMap — the weakref-based identity map."""

    def test_same_object_same_id(self):
        """Same object always returns the same stable_id."""
        m = StableIdMap()
        df = pd.DataFrame({"a": [1]})
        id1 = m.get_stable(df)
        id2 = m.get_stable(df)
        assert id1 == id2

    def test_different_objects_different_ids(self):
        """Different objects get different stable_ids."""
        m = StableIdMap()
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [1]})
        assert m.get_stable(df1) != m.get_stable(df2)

    def test_alias_same_id(self):
        """df2 = df (alias) returns the same stable_id."""
        m = StableIdMap()
        df = pd.DataFrame({"a": [1]})
        df2 = df  # alias, not copy
        assert m.get_stable(df) == m.get_stable(df2)

    def test_user_copy_different_id(self):
        """df.copy() gets a different stable_id (independent object)."""
        m = StableIdMap()
        df = pd.DataFrame({"a": [1]})
        df_copy = df.copy()
        assert m.get_stable(df) != m.get_stable(df_copy)

    def test_apply_memo_transfers_id(self):
        """apply_memo transfers stable_ids from originals to copies."""
        m = StableIdMap()
        df = pd.DataFrame({"a": [1]})
        original_id = m.get_stable(df)

        # Simulate deep copy with memo
        import copy
        memo = {}
        df_copy = copy.deepcopy(df, memo)

        # Apply the memo (maps id(original) → copy)
        m.apply_memo(memo)

        # Copy should have the same stable_id
        assert m.get_stable(df_copy) == original_id

    def test_id_reuse_detection(self):
        """Detect id() reuse after GC via weakref."""
        m = StableIdMap()

        # Create and register an object
        df1 = pd.DataFrame({"a": [1]})
        id1 = m.get_stable(df1)

        # Remember the python id
        python_id = id(df1)

        # Delete the object
        del df1
        gc.collect()

        # Create a new object that MAY reuse the same python id()
        # We can't guarantee id reuse, but we test that if it happens,
        # a new stable_id is assigned
        df2 = pd.DataFrame({"b": [2]})
        id2 = m.get_stable(df2)

        if id(df2) == python_id:
            # id() was reused — stable_id should be DIFFERENT
            assert id2 != id1, "Should detect id reuse via weakref"
        # If id() wasn't reused, the test is vacuous but still valid

    def test_lookup_exists(self):
        """lookup() returns stable_id for known objects."""
        m = StableIdMap()
        df = pd.DataFrame({"a": [1]})
        sid = m.get_stable(df)
        assert m.lookup(df) == sid

    def test_lookup_missing(self):
        """lookup() returns None for unknown objects."""
        m = StableIdMap()
        df = pd.DataFrame({"a": [1]})
        assert m.lookup(df) is None

    def test_clear(self):
        """clear() resets all state."""
        m = StableIdMap()
        df = pd.DataFrame({"a": [1]})
        m.get_stable(df)
        m.clear()
        assert m.lookup(df) is None
        assert len(m) == 0

    def test_ids_are_monotonic(self):
        """Stable IDs increment monotonically."""
        m = StableIdMap()
        ids = [m.get_stable(pd.DataFrame({"a": [i]})) for i in range(5)]
        assert ids == [0, 1, 2, 3, 4]


class TestLocRef:
    """Test LocRef frozen dataclass."""

    def test_equality(self):
        """LocRefs with same fields are equal."""
        a = LocRef(42, "df")
        b = LocRef(42, "df")
        assert a == b

    def test_different_loc_id(self):
        """LocRefs with different loc_ids are not equal."""
        a = LocRef(42, "df")
        b = LocRef(43, "df")
        assert a != b

    def test_different_var_name(self):
        """LocRefs with different var_names are not equal."""
        a = LocRef(42, "df")
        b = LocRef(42, "df2")
        assert a != b

    def test_hashable(self):
        """LocRef is hashable (for use in sets and as dict keys)."""
        a = LocRef(42, "df")
        b = LocRef(42, "df")
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_frozen(self):
        """LocRef is immutable."""
        a = LocRef(42, "df")
        with pytest.raises(AttributeError):
            a.loc_id = 99


class TestGetQualifier:
    """Test the get_qualifier bridge function."""

    def test_dataframe_gets_locref(self):
        """DataFrame variables get LocRef qualifiers."""
        m = StableIdMap()
        df = pd.DataFrame({"a": [1]})
        ns = {"df": df}
        q = get_qualifier("df", ns, m)
        assert isinstance(q, LocRef)
        assert q.var_name == "df"

    def test_non_dataframe_gets_string(self):
        """Non-DataFrame variables get string qualifiers."""
        m = StableIdMap()
        ns = {"x": 42}
        q = get_qualifier("x", ns, m)
        assert q == "x"
        assert isinstance(q, str)

    def test_no_namespace_gets_string(self):
        """Without namespace, always returns string."""
        q = get_qualifier("df", None, None)
        assert q == "df"
        assert isinstance(q, str)

    def test_series_gets_locref(self):
        """Series variables also get LocRef qualifiers."""
        m = StableIdMap()
        s = pd.Series([1, 2, 3])
        ns = {"s": s}
        q = get_qualifier("s", ns, m)
        assert isinstance(q, LocRef)

    def test_missing_var_gets_string(self):
        """Variable not in namespace gets string qualifier."""
        m = StableIdMap()
        q = get_qualifier("missing", {"x": 1}, m)
        assert q == "missing"


class TestSameDataframe:
    """Test _same_dataframe qualifier comparison helper."""

    def test_both_string_equal(self):
        assert _same_dataframe("df", "df") is True

    def test_both_string_different(self):
        assert _same_dataframe("df", "df2") is False

    def test_both_locref_same_id(self):
        """Same loc_id, different var_names → same DataFrame."""
        a = LocRef(42, "df")
        b = LocRef(42, "df2")
        assert _same_dataframe(a, b) is True

    def test_both_locref_different_id(self):
        """Different loc_ids → different DataFrames."""
        a = LocRef(42, "df")
        b = LocRef(43, "df")
        assert _same_dataframe(a, b) is False

    def test_mixed_locref_string_match(self):
        """LocRef var_name matches string qualifier."""
        a = LocRef(42, "df")
        assert _same_dataframe(a, "df") is True
        assert _same_dataframe("df", a) is True

    def test_mixed_locref_string_no_match(self):
        """LocRef var_name doesn't match string qualifier."""
        a = LocRef(42, "df")
        assert _same_dataframe(a, "other") is False

    def test_none_handling(self):
        assert _same_dataframe(None, None) is True
        assert _same_dataframe("df", None) is False
        assert _same_dataframe(None, "df") is False



class TestConflictRelationWithLocRef:
    """Test ▷ relation with LocRef qualifiers."""

    def test_col_col_same_locref(self):
        """Col(d1, c) ▷ Col(d1, c) = True when same loc_id."""
        ref = LocRef(42, "df")
        w = WriteLoc.col(ref, "price")
        r = ReadLoc.col(ref, "price")
        assert write_conflicts_read(w, r) is True

    def test_col_col_alias_different_var_name(self):
        """Col with same loc_id but different var_name → STILL conflicts (same object)."""
        w = WriteLoc.col(LocRef(42, "df"), "price")
        r = ReadLoc.col(LocRef(42, "df2"), "price")
        assert write_conflicts_read(w, r) is True

    def test_col_col_different_locref(self):
        """Col with different loc_ids → no conflict (different objects)."""
        w = WriteLoc.col(LocRef(42, "df"), "price")
        r = ReadLoc.col(LocRef(43, "df"), "price")
        assert write_conflicts_read(w, r) is False

    def test_var_col_same_var_name(self):
        """Var(x) ▷ Col(LocRef(42, x), c) = False — Var only conflicts with Var."""
        w = WriteLoc.var("df")
        r = ReadLoc.col(LocRef(42, "df"), "price")
        assert write_conflicts_read(w, r) is False

    def test_var_col_different_var_name(self):
        """Var(x) ▷ Col(LocRef(42, y), c) = False — different variable name."""
        w = WriteLoc.var("df")
        r = ReadLoc.col(LocRef(42, "df2"), "price")
        assert write_conflicts_read(w, r) is False

    def test_col_attr_same_locref(self):
        """Col with same loc_id conflicts with structural attr."""
        w = WriteLoc.col(LocRef(42, "df"), "new_col")
        r = ReadLoc.attr(LocRef(42, "df"), "columns")
        assert write_conflicts_read(w, r) is True

    def test_rows_col_same_locref(self):
        """Rows with same loc_id conflicts with column reads."""
        w = WriteLoc.rows("df", qualifier=LocRef(42, "df"))
        r = ReadLoc.col(LocRef(42, "df"), "price")
        assert write_conflicts_read(w, r) is True

    def test_rows_col_alias_different_var_name(self):
        """Rows with same loc_id but different var names → conflicts."""
        w = WriteLoc.rows("df", qualifier=LocRef(42, "df"))
        r = ReadLoc.col(LocRef(42, "df2"), "price")
        assert write_conflicts_read(w, r) is True


class TestLocRefInSets:
    """Test LocRef qualifiers in ReadLocSet/WriteLocSet operations."""

    def test_readloc_with_locref_in_set(self):
        """ReadLocs with LocRef can be stored in frozensets."""
        r1 = ReadLoc.col(LocRef(42, "df"), "price")
        r2 = ReadLoc.col(LocRef(42, "df"), "quantity")
        r3 = ReadLoc.col(LocRef(42, "df"), "price")  # duplicate
        s = frozenset({r1, r2, r3})
        assert len(s) == 2  # r1 and r3 are equal

    def test_has_conflict_with_locref(self):
        """has_conflict works with LocRef qualifiers."""
        writes = frozenset({WriteLoc.col(LocRef(42, "df"), "price")})
        reads = frozenset({ReadLoc.col(LocRef(42, "df"), "price")})
        assert has_conflict(writes, reads) is True

    def test_no_conflict_different_locref(self):
        """No conflict when loc_ids differ (independent DataFrames)."""
        writes = frozenset({WriteLoc.col(LocRef(42, "df"), "price")})
        reads = frozenset({ReadLoc.col(LocRef(43, "df"), "price")})
        assert has_conflict(writes, reads) is False


class TestTrackingToLocsetWithStableMap:
    """Test tracking_to_readlocset/writelocset with StableIdMap."""

    def test_read_locset_gets_locref_qualifiers(self):
        """Column reads get LocRef qualifiers when stable_map is provided."""
        m = StableIdMap()
        df = pd.DataFrame({"price": [1], "qty": [2]})
        ns = {"df": df}
        tracking = TrackingData(
            reads_before_writes={"df"},
            writes=set(),
            column_reads_before_writes={"df": {"price"}},
            column_writes={},
        )

        locs = tracking_to_readlocset(tracking, ns, m)
        col_locs = [l for l in locs if l.type.value == "col"]
        assert len(col_locs) == 1
        assert isinstance(col_locs[0].qualifier, LocRef)
        assert col_locs[0].qualifier.var_name == "df"
        assert col_locs[0].name == "price"

    def test_write_locset_gets_locref_qualifiers(self):
        """Column writes get LocRef qualifiers when stable_map is provided."""
        m = StableIdMap()
        df = pd.DataFrame({"price": [1]})
        ns = {"df": df}
        tracking = TrackingData(
            reads_before_writes=set(),
            writes={"df"},
            column_reads_before_writes={},
            column_writes={"df": {"price"}},
        )

        locs = tracking_to_writelocset(tracking, ns, m)
        col_locs = [l for l in locs if l.type.value == "col"]
        assert len(col_locs) == 1
        assert isinstance(col_locs[0].qualifier, LocRef)

    def test_non_df_vars_get_string_qualifiers(self):
        """Non-DataFrame variables get Var locs (no qualifier)."""
        m = StableIdMap()
        ns = {"x": 42}
        tracking = TrackingData(
            reads_before_writes={"x"},
            writes=set(),
            column_reads_before_writes={},
            column_writes={},
        )

        locs = tracking_to_readlocset(tracking, ns, m)
        var_locs = [l for l in locs if l.type.value == "var"]
        assert len(var_locs) == 1
        assert var_locs[0].name == "x"
        assert var_locs[0].qualifier is None

    def test_aliased_dfs_share_locref(self):
        """Two variable names for the same DataFrame share the same loc_id."""
        m = StableIdMap()
        df = pd.DataFrame({"price": [1]})
        df2 = df  # alias
        ns = {"df": df, "df2": df2}

        tracking1 = TrackingData(
            reads_before_writes={"df"},
            writes=set(),
            column_reads_before_writes={"df": {"price"}},
            column_writes={},
        )
        tracking2 = TrackingData(
            reads_before_writes={"df2"},
            writes=set(),
            column_reads_before_writes={"df2": {"price"}},
            column_writes={},
        )

        locs1 = tracking_to_readlocset(tracking1, ns, m)
        locs2 = tracking_to_readlocset(tracking2, ns, m)

        col1 = [l for l in locs1 if l.type.value == "col"][0]
        col2 = [l for l in locs2 if l.type.value == "col"][0]

        # Same loc_id (same object), different var_name
        assert col1.qualifier.loc_id == col2.qualifier.loc_id
        assert col1.qualifier.var_name == "df"
        assert col2.qualifier.var_name == "df2"

        # And they match via _same_dataframe
        assert _same_dataframe(col1.qualifier, col2.qualifier) is True


class TestLocRefSerialization:
    """Test LocRef serialization for frontend."""

    def test_readloc_to_dict_with_locref(self):
        """ReadLoc.to_dict() serializes LocRef as loc_id + var_name."""
        loc = ReadLoc.col(LocRef(42, "df"), "price")
        d = loc.to_dict()
        assert d["qualifier"] == 42  # loc_id as int
        assert d["var_name"] == "df"
        assert d["type"] == "col"
        assert d["name"] == "price"

    def test_writeloc_to_dict_with_locref(self):
        """WriteLoc.to_dict() serializes LocRef similarly."""
        loc = WriteLoc.col(LocRef(42, "df"), "price")
        d = loc.to_dict()
        assert d["qualifier"] == 42
        assert d["var_name"] == "df"

    def test_readloc_to_dict_without_locref(self):
        """ReadLoc.to_dict() with string qualifier is unchanged."""
        loc = ReadLoc.col("df", "price")
        d = loc.to_dict()
        assert d["qualifier"] == "df"
        assert "var_name" not in d

    def test_display_name_with_locref(self):
        """display_name uses var_name for display."""
        loc = ReadLoc.col(LocRef(42, "df"), "price")
        assert loc.display_name() == "df['price']"

    def test_var_name_with_locref(self):
        """var_name() extracts the variable name from LocRef."""
        loc = ReadLoc.col(LocRef(42, "df"), "price")
        assert loc.var_name() == "df"


class TestBuildLocContext:
    """Test build_loc_context for frontend display."""

    def test_single_df(self):
        m = StableIdMap()
        df = pd.DataFrame({"a": [1]})
        ns = {"df": df}
        ctx = build_loc_context(ns, m)
        assert len(ctx) == 1
        loc_id = m.get_stable(df)
        assert ctx[loc_id] == {"df"}

    def test_aliased_dfs(self):
        m = StableIdMap()
        df = pd.DataFrame({"a": [1]})
        ns = {"df": df, "df2": df}
        ctx = build_loc_context(ns, m)
        loc_id = m.get_stable(df)
        assert ctx[loc_id] == {"df", "df2"}

    def test_independent_dfs(self):
        m = StableIdMap()
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [2]})
        ns = {"df1": df1, "df2": df2}
        ctx = build_loc_context(ns, m)
        assert len(ctx) == 2
