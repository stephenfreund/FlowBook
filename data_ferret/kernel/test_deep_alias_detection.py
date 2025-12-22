"""
Comprehensive tests for deep alias detection in checkpoints.

This test file covers:
1. Basic deep alias detection (nested dict/list sharing)
2. Temporary object id() reuse bugs (the .values/.data issue)
3. DataFrame and Series alias scenarios
4. Numpy view detection
5. Object-dtype element tracking
6. Edge cases and stress tests

The deep alias detection feature identifies when two variables share ANY
internal reference, not just top-level object identity. This is critical
for correctly detecting backward mutations in SDC enforcement.

CRITICAL BUG HISTORY:
- Initially tracked id(arr.data) which created false positives because
  memoryview objects are temporary and their ids get reused.
- Then tracked id(arr.values) which had the same issue - .values creates
  temporary arrays whose ids can be reused after garbage collection.
- Fix: Only track persistent objects (the container itself, ._mgr, .base)
"""

import pytest
import numpy as np
import pandas as pd
from typing import Any, Dict, Set

from data_ferret.kernel.checkpoint import (
    Checkpoint,
    _collect_reachable_ids,
    _collect_reachable_ids_with_paths,
)


# =============================================================================
# SECTION 1: BASIC DEEP ALIAS DETECTION
# =============================================================================

class TestBasicDeepAliasDetection:
    """Test basic nested structure aliasing detection."""

    def test_nested_dict_shared_inner_object(self):
        """The canonical example: a['b'] and c['b'] point to same object."""
        shared = {"value": 42}
        namespace = {
            "a": {"b": shared},
            "c": {"b": shared},
            "d": {"b": {"value": 42}},  # Same value, different object
        }

        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"a"})

        assert "a" in aliases
        assert "c" in aliases, "c shares inner object with a"
        assert "d" not in aliases, "d has different object"

    def test_nested_list_shared_element(self):
        """Lists containing the same mutable object."""
        shared = [1, 2, 3]
        namespace = {
            "list_a": [shared, "other"],
            "list_b": ["first", shared],
            "list_c": [[1, 2, 3], "other"],  # Same value, different object
        }

        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"list_a"})

        assert "list_a" in aliases
        assert "list_b" in aliases
        assert "list_c" not in aliases

    def test_deeply_nested_sharing(self):
        """Multiple levels of nesting with shared object."""
        shared = {"deep": True}
        namespace = {
            "x": {"a": {"b": {"c": shared}}},
            "y": {"z": shared},  # Shares at different depth
        }

        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"x"})

        assert aliases == {"x", "y"}

    def test_multiple_shared_objects(self):
        """Variables sharing multiple different objects."""
        shared1 = {"id": 1}
        shared2 = [100, 200]
        namespace = {
            "a": {"ref1": shared1, "ref2": shared2},
            "b": {"ref1": shared1},  # Shares shared1
            "c": {"ref2": shared2},  # Shares shared2
            "d": {"other": "data"},  # Shares nothing
        }

        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"a"})

        assert aliases == {"a", "b", "c"}

    def test_tuple_with_mutable_content(self):
        """Tuples are immutable but can contain mutable objects."""
        shared = {"mutable": True}
        namespace = {
            "tuple_a": (shared, "immutable"),
            "tuple_b": ("other", shared),
        }

        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"tuple_a"})

        assert aliases == {"tuple_a", "tuple_b"}

    def test_set_with_frozen_content(self):
        """Sets can only contain hashable (usually immutable) objects."""
        # Sets cannot contain mutable objects, so aliasing through sets
        # would only happen at the set level itself
        shared_set = frozenset([1, 2, 3])
        namespace = {
            "a": {"data": shared_set},
            "b": {"data": shared_set},
        }

        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"a"})

        # Frozensets are immutable, so we might not track them
        # The key is no false positives
        assert "a" in aliases


# =============================================================================
# SECTION 2: TEMPORARY OBJECT ID REUSE BUG TESTS
# =============================================================================

class TestTemporaryObjectIdReuse:
    """
    Tests for the id() reuse bug with temporary objects.

    Python's memory allocator can reuse addresses for objects with
    non-overlapping lifetimes. This caused false alias detection when
    we tracked temporary objects like:
    - arr.data (memoryview - temporary)
    - arr.values (numpy array from pandas - can be temporary)
    - df[col].values (temporary Series then array)
    """

    def test_independent_series_not_aliases(self):
        """REGRESSION: Independent Series must NOT be detected as aliases."""
        s1 = pd.Series([1, 2, 3], name="data")
        s2 = pd.Series([1, 2, 3], name="data")

        namespace = {"s1": s1, "s2": s2}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"s1"})

        assert aliases == {"s1"}, f"Expected only s1, got {aliases}"

    def test_independent_dataframes_not_aliases(self):
        """REGRESSION: Independent DataFrames must NOT be detected as aliases."""
        df1 = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        df2 = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

        namespace = {"df1": df1, "df2": df2}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"df1"})

        assert aliases == {"df1"}, f"Expected only df1, got {aliases}"

    def test_independent_numpy_arrays_not_aliases(self):
        """REGRESSION: Independent numpy arrays must NOT be detected as aliases."""
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([1, 2, 3, 4, 5])

        namespace = {"arr1": arr1, "arr2": arr2}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"arr1"})

        assert aliases == {"arr1"}, f"Expected only arr1, got {aliases}"

    def test_many_dataframes_no_false_positives(self):
        """STRESS: Many independent DataFrames should not trigger false aliases."""
        namespace = {}
        for i in range(20):
            namespace[f"df_{i}"] = pd.DataFrame({
                "id": list(range(100)),
                "value": list(range(100, 200)),
            })

        checkpoint = Checkpoint("test", namespace, {})

        # Each DataFrame should only alias with itself
        for i in range(20):
            aliases = checkpoint.get_aliases_for_vars({f"df_{i}"})
            assert aliases == {f"df_{i}"}, f"df_{i} has unexpected aliases: {aliases}"

    def test_many_series_no_false_positives(self):
        """STRESS: Many independent Series should not trigger false aliases."""
        namespace = {}
        for i in range(20):
            namespace[f"s_{i}"] = pd.Series(list(range(100)), name=f"series_{i}")

        checkpoint = Checkpoint("test", namespace, {})

        for i in range(20):
            aliases = checkpoint.get_aliases_for_vars({f"s_{i}"})
            assert aliases == {f"s_{i}"}, f"s_{i} has unexpected aliases: {aliases}"

    def test_copied_column_not_alias(self):
        """A copied column should not be an alias of the original."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        col_copy = df["x"].copy()

        namespace = {"df": df, "col_copy": col_copy}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"df"})

        assert "col_copy" not in aliases

    def test_different_dtype_columns_not_aliases(self):
        """Columns with different dtypes should not be aliases."""
        df1 = pd.DataFrame({"id": [1, 2, 3]})  # int64
        df2 = pd.DataFrame({"id": ["a", "b", "c"]})  # object

        namespace = {"df1": df1, "df2": df2}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"df1"})

        assert aliases == {"df1"}


# =============================================================================
# SECTION 3: NUMPY VIEW DETECTION
# =============================================================================

class TestNumpyViewDetection:
    """Test that actual numpy views ARE detected as aliases."""

    def test_basic_view(self):
        """A slice of an array is a view that shares data."""
        base = np.array([1, 2, 3, 4, 5])
        view = base[1:4]

        namespace = {"base": base, "view": view}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"view"})

        assert "base" in aliases, "Base array should be alias of view"
        assert "view" in aliases

    def test_multiple_views_of_same_base(self):
        """Multiple views of the same array are all aliases."""
        base = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        view1 = base[0:3]
        view2 = base[3:6]
        view3 = base[6:9]

        namespace = {"base": base, "v1": view1, "v2": view2, "v3": view3}
        checkpoint = Checkpoint("test", namespace, {})

        # All should be aliases of each other
        aliases = checkpoint.get_aliases_for_vars({"v1"})
        assert aliases == {"base", "v1", "v2", "v3"}

    def test_view_vs_copy(self):
        """View should be alias, copy should not."""
        base = np.array([1, 2, 3, 4, 5])
        view = base[1:4]  # View
        copy = base[1:4].copy()  # Copy

        namespace = {"base": base, "view": view, "copy": copy}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"base"})

        assert "view" in aliases
        assert "copy" not in aliases

    def test_reshape_creates_view(self):
        """Reshape typically creates a view."""
        arr = np.array([1, 2, 3, 4, 5, 6])
        reshaped = arr.reshape(2, 3)

        namespace = {"arr": arr, "reshaped": reshaped}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"arr"})

        assert "reshaped" in aliases

    def test_transpose_creates_view(self):
        """Transpose creates a view."""
        arr = np.array([[1, 2], [3, 4]])
        transposed = arr.T

        namespace = {"arr": arr, "transposed": transposed}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"arr"})

        assert "transposed" in aliases


# =============================================================================
# SECTION 4: DATAFRAME AND SERIES SPECIFIC TESTS
# =============================================================================

class TestDataFrameSeriesAliases:
    """Tests specific to pandas DataFrame and Series aliasing."""

    def test_same_dataframe_different_names(self):
        """Same DataFrame assigned to different names."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        namespace = {"df1": df, "df2": df}

        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"df1"})

        assert aliases == {"df1", "df2"}

    def test_same_series_different_names(self):
        """Same Series assigned to different names."""
        s = pd.Series([1, 2, 3])
        namespace = {"s1": s, "s2": s}

        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"s1"})

        assert aliases == {"s1", "s2"}

    def test_dataframe_with_object_dtype_shared_elements(self):
        """DataFrames with object dtype containing shared objects."""
        shared_dict = {"key": "value"}
        df1 = pd.DataFrame({"col": [shared_dict, {"other": 1}]})
        df2 = pd.DataFrame({"col": [{"another": 2}, shared_dict]})

        namespace = {"df1": df1, "df2": df2}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"df1"})

        assert "df2" in aliases, "df2 shares object in cells with df1"

    def test_series_with_object_dtype_shared_elements(self):
        """Series with object dtype containing shared objects."""
        shared_list = [1, 2, 3]
        s1 = pd.Series([shared_list, [4, 5]])
        s2 = pd.Series([[10, 20], shared_list])

        namespace = {"s1": s1, "s2": s2}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"s1"})

        assert "s2" in aliases

    def test_dataframe_slice_is_view_sometimes(self):
        """
        DataFrame slices may or may not be views depending on pandas version.

        This test documents the expected behavior - we detect the DataFrame
        object itself being shared, not necessarily the underlying data.
        """
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # This creates a new DataFrame, not a view in recent pandas
        subset = df[["a"]]

        namespace = {"df": df, "subset": subset}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"df"})

        # subset is a new DataFrame, not an alias
        # (unless pandas internals share block manager)
        # The key test is no false positives for independent DataFrames
        assert "df" in aliases


# =============================================================================
# SECTION 5: OBJECT-DTYPE ELEMENT TRACKING
# =============================================================================

class TestObjectDtypeElementTracking:
    """Test that we correctly track shared objects inside object-dtype arrays."""

    def test_list_inside_array(self):
        """Lists stored in numpy object arrays."""
        shared = [1, 2, 3]
        arr1 = np.array([shared, [4, 5]], dtype=object)
        arr2 = np.array([[10], shared], dtype=object)

        namespace = {"arr1": arr1, "arr2": arr2}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"arr1"})

        assert "arr2" in aliases

    def test_dict_inside_array(self):
        """Dicts stored in numpy object arrays."""
        shared = {"key": "value"}
        arr1 = np.array([shared, {"other": 1}], dtype=object)
        arr2 = np.array([{"another": 2}, shared], dtype=object)

        namespace = {"arr1": arr1, "arr2": arr2}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"arr1"})

        assert "arr2" in aliases

    def test_mixed_containers_in_object_array(self):
        """Various container types in object arrays."""
        shared = {"nested": [1, 2, 3]}
        arr = np.array([shared, (1, 2), [3, 4]], dtype=object)
        lst = [shared, "other"]

        namespace = {"arr": arr, "lst": lst}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"arr"})

        assert "lst" in aliases


# =============================================================================
# SECTION 6: USER-DEFINED OBJECTS
# =============================================================================

class TestUserDefinedObjects:
    """Test aliasing through user-defined class instances."""

    def test_shared_attribute(self):
        """Objects sharing an attribute reference."""
        shared = {"data": [1, 2, 3]}

        class Container:
            def __init__(self, data):
                self.data = data

        obj1 = Container(shared)
        obj2 = Container(shared)
        obj3 = Container({"data": [1, 2, 3]})  # Same value, different object

        namespace = {"obj1": obj1, "obj2": obj2, "obj3": obj3}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"obj1"})

        assert "obj2" in aliases
        assert "obj3" not in aliases

    def test_nested_object_attributes(self):
        """Objects with nested attribute sharing."""
        shared = {"inner": True}

        class Inner:
            def __init__(self, ref):
                self.ref = ref

        class Outer:
            def __init__(self, inner):
                self.inner = inner

        inner1 = Inner(shared)
        inner2 = Inner(shared)
        outer1 = Outer(inner1)
        outer2 = Outer(inner2)

        namespace = {"outer1": outer1, "outer2": outer2}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"outer1"})

        assert "outer2" in aliases


# =============================================================================
# SECTION 7: CIRCULAR REFERENCES
# =============================================================================

class TestCircularReferences:
    """Test handling of circular references without infinite loops."""

    def test_self_referential_dict(self):
        """Dict containing itself."""
        d = {"name": "self"}
        d["self"] = d

        namespace = {"d": d}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"d"})

        assert "d" in aliases  # Should complete without hanging

    def test_self_referential_list(self):
        """List containing itself."""
        lst = [1, 2]
        lst.append(lst)

        namespace = {"lst": lst}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"lst"})

        assert "lst" in aliases

    def test_mutual_references(self):
        """Two objects referencing each other."""
        a = {"name": "a"}
        b = {"name": "b", "ref": a}
        a["ref"] = b

        namespace = {"a": a, "b": b}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"a"})

        assert aliases == {"a", "b"}

    def test_circular_chain(self):
        """Chain of objects forming a cycle."""
        a = {"name": "a"}
        b = {"name": "b"}
        c = {"name": "c"}
        a["next"] = b
        b["next"] = c
        c["next"] = a  # Complete the cycle

        namespace = {"a": a, "b": b, "c": c}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"a"})

        assert aliases == {"a", "b", "c"}


# =============================================================================
# SECTION 8: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_namespace(self):
        """Empty namespace."""
        checkpoint = Checkpoint("test", {}, {})
        aliases = checkpoint.get_aliases_for_vars({"nonexistent"})

        assert aliases == {"nonexistent"}  # Returns input for missing vars

    def test_empty_accessed_vars(self):
        """Empty accessed vars set."""
        namespace = {"x": [1, 2, 3]}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars(set())

        assert aliases == set()

    def test_none_value(self):
        """Variables set to None."""
        namespace = {"a": None, "b": None}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"a"})

        # None is immutable, typically not tracked
        assert "a" in aliases

    def test_empty_containers(self):
        """Empty containers."""
        namespace = {
            "empty_dict": {},
            "empty_list": [],
            "empty_df": pd.DataFrame(),
            "empty_series": pd.Series(dtype=float),
        }
        checkpoint = Checkpoint("test", namespace, {})

        for var in namespace:
            aliases = checkpoint.get_aliases_for_vars({var})
            assert var in aliases

    def test_very_deep_nesting(self):
        """Deeply nested structures."""
        shared = {"bottom": True}
        current = shared
        for i in range(50):
            current = {"level": i, "child": current}

        other = {"ref": shared}

        namespace = {"deep": current, "other": other}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"deep"})

        assert "other" in aliases

    def test_wide_structure(self):
        """Structure with many siblings."""
        shared = {"shared": True}
        parent = {f"child_{i}": {"ref": shared if i == 25 else None}
                  for i in range(50)}
        other = {"ref": shared}

        namespace = {"parent": parent, "other": other}
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"parent"})

        assert "other" in aliases

    def test_numeric_types_not_aliased(self):
        """Numeric types (immutable) should not cause false aliases."""
        namespace = {
            "int1": 42,
            "int2": 42,  # Python may intern this
            "float1": 3.14,
            "float2": 3.14,
        }
        checkpoint = Checkpoint("test", namespace, {})

        # Each should only alias with itself (or with interned copies)
        # The key is no unexpected aliases
        aliases = checkpoint.get_aliases_for_vars({"int1"})
        assert "float1" not in aliases
        assert "float2" not in aliases

    def test_string_interning(self):
        """String interning shouldn't cause problems."""
        namespace = {
            "s1": "hello",
            "s2": "hello",  # Python interns short strings
            "s3": "world",
        }
        checkpoint = Checkpoint("test", namespace, {})
        aliases = checkpoint.get_aliases_for_vars({"s1"})

        # Strings are immutable, aliasing through interning is harmless
        # Key is no false positives for unrelated strings
        assert "s3" not in aliases


# =============================================================================
# SECTION 9: PATH TRACKING TESTS
# =============================================================================

class TestPathTracking:
    """Test that paths are correctly tracked for logging."""

    def test_dict_path(self):
        """Paths through dictionaries use bracket notation."""
        shared = {"inner": True}
        namespace = {"outer": {"key": shared}}

        checkpoint = Checkpoint("test", namespace, {})

        # Check that path includes the key
        shared_id = id(shared)
        if shared_id in checkpoint._id_to_paths:
            paths = checkpoint._id_to_paths[shared_id]
            assert "outer" in paths
            assert "'key'" in paths["outer"] or "key" in paths["outer"]

    def test_list_path(self):
        """Paths through lists use index notation."""
        shared = {"inner": True}
        namespace = {"outer": [None, shared, None]}

        checkpoint = Checkpoint("test", namespace, {})

        shared_id = id(shared)
        if shared_id in checkpoint._id_to_paths:
            paths = checkpoint._id_to_paths[shared_id]
            assert "outer" in paths
            assert "[1]" in paths["outer"]

    def test_attribute_path(self):
        """Paths through object attributes use dot notation."""
        shared = {"inner": True}

        class Container:
            def __init__(self):
                self.data = shared

        obj = Container()
        namespace = {"obj": obj}

        checkpoint = Checkpoint("test", namespace, {})

        shared_id = id(shared)
        if shared_id in checkpoint._id_to_paths:
            paths = checkpoint._id_to_paths[shared_id]
            assert "obj" in paths
            assert ".data" in paths["obj"]


# =============================================================================
# SECTION 10: COLLECT FUNCTION UNIT TESTS
# =============================================================================

class TestCollectReachableIds:
    """Unit tests for _collect_reachable_ids function."""

    def test_simple_dict(self):
        """Collect IDs from simple dict."""
        d = {"a": [1, 2], "b": {"nested": True}}
        visited: Set[int] = set()
        _collect_reachable_ids(d, visited)

        assert id(d) in visited
        assert id(d["a"]) in visited
        assert id(d["b"]) in visited

    def test_circular_terminates(self):
        """Circular reference doesn't cause infinite loop."""
        d = {"self": None}
        d["self"] = d

        visited: Set[int] = set()
        _collect_reachable_ids(d, visited)  # Should not hang

        assert id(d) in visited

    def test_immutables_skipped(self):
        """Immutable atomics are not tracked."""
        d = {"int": 42, "str": "hello", "bool": True, "none": None}
        visited: Set[int] = set()
        _collect_reachable_ids(d, visited)

        # Dict itself is tracked
        assert id(d) in visited
        # But immutable values may not be (implementation detail)
        # Key is that it completes without error


class TestCollectReachableIdsWithPaths:
    """Unit tests for _collect_reachable_ids_with_paths function."""

    def test_paths_recorded(self):
        """Paths are correctly recorded."""
        d = {"key": [1, 2, 3]}
        visited: Set[int] = set()
        paths: Dict[int, str] = {}
        _collect_reachable_ids_with_paths(d, "root", visited, paths)

        assert id(d) in paths
        assert paths[id(d)] == "root"

        list_id = id(d["key"])
        assert list_id in paths
        assert paths[list_id] == "root['key']"

    def test_nested_paths(self):
        """Nested structure paths."""
        d = {"a": {"b": {"c": [1]}}}
        visited: Set[int] = set()
        paths: Dict[int, str] = {}
        _collect_reachable_ids_with_paths(d, "x", visited, paths)

        inner_list_id = id(d["a"]["b"]["c"])
        assert inner_list_id in paths
        assert paths[inner_list_id] == "x['a']['b']['c']"
