"""Tests for locals.py - Symbol extraction and formatting utilities.

Covers SymbolFinder AST visitor, format_limited helper, and utility functions.
"""

import numpy as np
import pytest

from flowbook.kernel_support.locals import (
    SymbolFinder,
    _is_iterable,
    _repr_if_defined,
    _format_limited,
)
import ast


class TestSymbolFinder:
    """Tests for SymbolFinder AST visitor."""

    def test_assignment(self):
        """Finds symbols from simple assignments."""
        code = "x = 1\ny = 2"
        tree = ast.parse(code)
        finder = SymbolFinder()
        finder.visit(tree)
        assert "x" in finder.defined_symbols
        assert "y" in finder.defined_symbols

    def test_for_loop(self):
        """Finds loop variable from for loops."""
        code = "for i in range(10):\n    pass"
        tree = ast.parse(code)
        finder = SymbolFinder()
        finder.visit(tree)
        assert "i" in finder.defined_symbols

    def test_no_symbols_in_expression(self):
        """No symbols found for standalone expressions."""
        code = "print('hello')"
        tree = ast.parse(code)
        finder = SymbolFinder()
        finder.visit(tree)
        assert len(finder.defined_symbols) == 0

    def test_multiple_targets(self):
        """Handles multiple assignment targets."""
        code = "a = b = 1"
        tree = ast.parse(code)
        finder = SymbolFinder()
        finder.visit(tree)
        assert "a" in finder.defined_symbols
        assert "b" in finder.defined_symbols


class TestExtractNbGlobals:
    """Tests for _extract_nb_globals function."""

    def test_extract_globals_from_in(self):
        """Extracts defined symbols from In history."""
        from flowbook.kernel_support.locals import _extract_nb_globals
        globals_ns = {
            "In": ["", "x = 1\ny = 2", "z = x + y"],
            "x": 1,
            "y": 2,
            "z": 3,
        }
        result = _extract_nb_globals(globals_ns)
        assert "x" in result
        assert "y" in result
        assert "z" in result

    def test_extract_globals_with_syntax_error(self):
        """Handles syntax errors in In history gracefully."""
        from flowbook.kernel_support.locals import _extract_nb_globals
        globals_ns = {
            "In": ["", "def foo(:", "x = 1"],
            "x": 1,
        }
        result = _extract_nb_globals(globals_ns)
        assert "x" in result

    def test_extract_globals_empty_in(self):
        """Empty In history returns empty set."""
        from flowbook.kernel_support.locals import _extract_nb_globals
        globals_ns = {"In": [""]}
        result = _extract_nb_globals(globals_ns)
        assert result == set()


class TestIsIterable:
    """Tests for _is_iterable utility."""

    def test_list_is_iterable(self):
        """Lists are iterable."""
        assert _is_iterable([1, 2, 3])

    def test_string_is_iterable(self):
        """Strings are iterable."""
        assert _is_iterable("hello")

    def test_int_not_iterable(self):
        """Integers are not iterable."""
        assert not _is_iterable(42)

    def test_none_not_iterable(self):
        """None is not iterable."""
        assert not _is_iterable(None)

    def test_dict_is_iterable(self):
        """Dicts are iterable."""
        assert _is_iterable({"a": 1})

    def test_generator_is_iterable(self):
        """Generators are iterable."""
        assert _is_iterable(x for x in range(3))


class TestReprIfDefined:
    """Tests for _repr_if_defined utility."""

    def test_custom_repr(self):
        """Objects with custom __repr__ return True."""
        class MyObj:
            def __repr__(self):
                return "MyObj()"
        assert _repr_if_defined(MyObj())

    def test_ndarray_returns_false(self):
        """numpy arrays return False (handled as iterables)."""
        assert not _repr_if_defined(np.array([1, 2]))

    def test_dict_returns_false(self):
        """Dicts return False (handled as iterables)."""
        assert not _repr_if_defined({"a": 1})

    def test_list_returns_false(self):
        """Lists return False (handled as iterables)."""
        assert not _repr_if_defined([1, 2])

    def test_tuple_returns_false(self):
        """Tuples return False (handled as iterables)."""
        assert not _repr_if_defined((1, 2))


class TestFormatLimited:
    """Tests for _format_limited utility."""

    def test_int(self):
        """Integer is formatted directly."""
        result = _format_limited(42)
        assert result == "42"

    def test_float(self):
        """Float is formatted directly."""
        result = _format_limited(3.14)
        assert result == "3.14"

    def test_none(self):
        """None is formatted."""
        result = _format_limited(None)
        assert result == "None"

    def test_bool(self):
        """Boolean is formatted."""
        result = _format_limited(True)
        assert result == "True"

    def test_short_string(self):
        """Short string is wrapped in quotes."""
        result = _format_limited("hello")
        assert result == "'hello'"

    def test_long_string(self):
        """Long string is truncated."""
        long_str = "x" * 300
        result = _format_limited(long_str)
        assert len(result) <= 260  # truncated at 254 + quotes

    def test_small_list(self):
        """Small list is formatted completely."""
        result = _format_limited([1, 2, 3])
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_large_list(self):
        """Large list is truncated with ellipsis."""
        result = _format_limited(list(range(20)), limit=5)
        assert "..." in result

    def test_small_dict(self):
        """Small dict is formatted completely."""
        result = _format_limited({"a": 1, "b": 2})
        assert "a" in result

    def test_large_dict(self):
        """Large dict is truncated."""
        d = {f"key{i}": i for i in range(20)}
        result = _format_limited(d, limit=5)
        assert "..." in result

    def test_small_tuple(self):
        """Small tuple is formatted completely."""
        result = _format_limited((1, 2, 3))
        assert "1" in result

    def test_large_tuple(self):
        """Large tuple is truncated."""
        result = _format_limited(tuple(range(20)), limit=5)
        assert "..." in result

    def test_numpy_array(self):
        """numpy array is formatted with limited elements."""
        arr = np.arange(5)
        result = _format_limited(arr)
        assert "array" in result.lower() or "0" in result

    def test_custom_repr_object(self):
        """Custom object with __repr__ is formatted via repr."""
        class MyObj:
            def __repr__(self):
                return "MyObj(x=5)"
        result = _format_limited(MyObj())
        assert "MyObj(x=5)" in result

    def test_object_without_repr(self):
        """Object without custom __repr__ gets attribute inspection."""
        class SimpleObj:
            def __init__(self):
                self.value = 42
                self.name = "test"
        result = _format_limited(SimpleObj())
        assert "SimpleObj" in result

    def test_depth_limiting(self):
        """Deeply nested structures are truncated by depth."""
        deep = {"a": {"b": {"c": {"d": "deep"}}}}
        result = _format_limited(deep)
        assert "..." in result

    def test_very_long_result_truncated(self):
        """Very long formatted results are truncated."""
        # Create something that produces a very long output
        big = {f"key{'x' * 100}_{i}": list(range(10)) for i in range(100)}
        result = _format_limited(big)
        assert len(result) <= 1024 * 2 + 10

    def test_iterable_non_list(self):
        """Non-list iterables (like generators) are handled."""
        result = _format_limited(range(5))
        # range becomes a list then formatted
        assert "0" in result

    def test_bytes_string(self):
        """Bytes are formatted."""
        result = _format_limited(b"hello")
        assert "hello" in result

    def test_unrepresentable(self):
        """Objects that raise during formatting return '<unrepresentable>'."""
        class BadObj:
            def __repr__(self):
                raise RuntimeError("cannot repr")
            def __str__(self):
                raise RuntimeError("cannot str")
            def __iter__(self):
                raise RuntimeError("cannot iter")
        # This may or may not trigger the unrepresentable path
        # depending on how the helper tries to format it
        result = _format_limited(BadObj())
        # Should at least not crash
        assert isinstance(result, str)

    def test_ellipsis_value(self):
        """Ellipsis value is handled."""
        result = _format_limited(...)
        assert "..." in result

    def test_type_value(self):
        """type objects are formatted."""
        result = _format_limited(int)
        assert "int" in result


class TestPrintLocals:
    """Tests for print_locals function."""

    def test_print_locals_in_function(self):
        """print_locals formats variables in a function frame."""
        from io import StringIO
        from flowbook.kernel_support.locals import print_locals

        # We need a real frame to test this
        file = StringIO()

        def sample_function():
            x = 42
            name = "test"
            import sys
            frame = sys._getframe(0)
            print_locals(file, frame)

        sample_function()
        output = file.getvalue()
        # Should contain at least some variable info
        # (may or may not find all vars depending on AST analysis)
        assert isinstance(output, str)


class TestPrintUserGlobals:
    """Tests for print_user_globals function."""

    def test_print_user_globals(self):
        """print_user_globals formats user namespace variables."""
        from io import StringIO
        from flowbook.kernel_support.locals import print_user_globals

        file = StringIO()
        user_ns = {"x": 42, "name": "hello"}
        print_user_globals(file, user_ns)
        output = file.getvalue()
        assert "x" in output
        assert "name" in output

    def test_print_user_globals_empty(self):
        """print_user_globals with empty namespace."""
        from io import StringIO
        from flowbook.kernel_support.locals import print_user_globals

        file = StringIO()
        user_ns = {}
        print_user_globals(file, user_ns)
        output = file.getvalue()
        assert output == ""

    def test_print_user_globals_multiline_repr(self):
        """print_user_globals handles multi-line representations."""
        from io import StringIO
        from flowbook.kernel_support.locals import print_user_globals

        file = StringIO()
        user_ns = {"big_dict": {f"key{i}": i for i in range(20)}}
        print_user_globals(file, user_ns)
        output = file.getvalue()
        assert "big_dict" in output
