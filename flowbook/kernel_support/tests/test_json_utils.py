"""Tests for json_utils.py - JSON serialization with numpy/float handling."""

import numpy as np
import pytest

from flowbook.kernel_support.json_utils import make_json_safe


class TestMakeJsonSafe:
    """Tests for make_json_safe function."""

    def test_plain_dict(self):
        """Plain dict is returned with values recursively processed."""
        result = make_json_safe({"a": 1, "b": "hello"})
        assert result == {"a": 1, "b": "hello"}

    def test_nested_dict(self):
        """Nested dicts are recursively processed."""
        result = make_json_safe({"outer": {"inner": 42}})
        assert result == {"outer": {"inner": 42}}

    def test_list(self):
        """Lists are recursively processed."""
        result = make_json_safe([1, 2, 3])
        assert result == [1, 2, 3]

    def test_tuple(self):
        """Tuples are recursively processed."""
        result = make_json_safe((1, 2, 3))
        assert result == [1, 2, 3]

    def test_nested_list_in_dict(self):
        """Nested list inside a dict is processed."""
        result = make_json_safe({"data": [1, 2, 3]})
        assert result == {"data": [1, 2, 3]}

    def test_small_ndarray(self):
        """Small numpy arrays (size <= 100) are converted to lists."""
        arr = np.array([1, 2, 3])
        result = make_json_safe(arr)
        assert result == [1, 2, 3]

    def test_large_ndarray(self):
        """Large numpy arrays (size > 100) become summary dicts."""
        arr = np.zeros((200,))
        result = make_json_safe(arr)
        assert result["_type"] == "ndarray"
        assert result["shape"] == (200,)
        assert result["dtype"] == "float64"
        assert result["size"] == 200
        assert "summary" in result

    def test_2d_ndarray_large(self):
        """Large 2D numpy array gives proper shape info."""
        arr = np.ones((20, 20))
        result = make_json_safe(arr)
        assert result["_type"] == "ndarray"
        assert result["shape"] == (20, 20)
        assert result["size"] == 400

    def test_small_ndarray_with_nested_values(self):
        """Small numpy array with nested values gets recursively processed."""
        arr = np.array([1.0, 2.0, float("nan")])
        result = make_json_safe(arr)
        # NaN values in the list should be handled
        assert result[0] == 1.0
        assert result[1] == 2.0
        # NaN converted by make_json_safe through recursion
        assert result[2] is None

    def test_numpy_integer(self):
        """numpy integer types are converted to Python int."""
        result = make_json_safe(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_floating(self):
        """numpy floating types are converted to Python float."""
        result = make_json_safe(np.float64(3.14))
        assert result == 3.14

    def test_numpy_nan(self):
        """numpy NaN is converted to None."""
        result = make_json_safe(np.float64(float("nan")))
        assert result is None

    def test_numpy_inf(self):
        """numpy positive infinity is converted to 'Infinity' string."""
        result = make_json_safe(np.float64(float("inf")))
        assert result == "Infinity"

    def test_numpy_neg_inf(self):
        """numpy negative infinity is converted to '-Infinity' string."""
        result = make_json_safe(np.float64(float("-inf")))
        assert result == "-Infinity"

    def test_python_float_nan(self):
        """Python float NaN is converted to None."""
        result = make_json_safe(float("nan"))
        assert result is None

    def test_python_float_inf(self):
        """Python float positive infinity is converted to 'Infinity' string."""
        result = make_json_safe(float("inf"))
        assert result == "Infinity"

    def test_python_float_neg_inf(self):
        """Python float negative infinity is converted to '-Infinity' string."""
        result = make_json_safe(float("-inf"))
        assert result == "-Infinity"

    def test_python_normal_float(self):
        """Normal Python float is passed through."""
        result = make_json_safe(3.14)
        assert result == 3.14

    def test_string_passthrough(self):
        """Strings are passed through unchanged."""
        result = make_json_safe("hello")
        assert result == "hello"

    def test_none_passthrough(self):
        """None is passed through unchanged."""
        result = make_json_safe(None)
        assert result is None

    def test_bool_passthrough(self):
        """Booleans are passed through unchanged."""
        assert make_json_safe(True) is True
        assert make_json_safe(False) is False

    def test_int_passthrough(self):
        """Integers are passed through unchanged."""
        result = make_json_safe(42)
        assert result == 42

    def test_complex_nested_structure(self):
        """Complex nested structure with mixed numpy and Python types."""
        data = {
            "values": [np.int64(1), np.float64(2.5), float("nan")],
            "array": np.array([10, 20]),
            "nested": {"x": np.float64(float("inf"))},
        }
        result = make_json_safe(data)
        assert result["values"][0] == 1
        assert result["values"][1] == 2.5
        assert result["values"][2] is None
        assert result["array"] == [10, 20]
        assert result["nested"]["x"] == "Infinity"

    def test_numpy_integer_nan_via_float_cast(self):
        """numpy integer types that are NaN (edge case via float cast)."""
        # np.integer cannot be NaN directly, but np.floating can
        val = np.float32(float("nan"))
        result = make_json_safe(val)
        assert result is None

    def test_numpy_integer_inf(self):
        """numpy floating infinity through the np.integer/floating branch."""
        val = np.float32(float("inf"))
        result = make_json_safe(val)
        assert result == "Infinity"

        val = np.float32(float("-inf"))
        result = make_json_safe(val)
        assert result == "-Infinity"
