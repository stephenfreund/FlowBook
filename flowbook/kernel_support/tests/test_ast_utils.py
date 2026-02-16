"""Tests for ast_utils.py - AST code transformation utilities."""

import pytest

from flowbook.kernel_support.ast_utils import wrap_last_expr_with_print_repr


class TestWrapLastExprWithPrintRepr:
    """Tests for wrap_last_expr_with_print_repr function."""

    def test_single_expression(self):
        """Single expression gets wrapped with print(repr(...))."""
        code = "x + 1"
        result = wrap_last_expr_with_print_repr(code)
        assert "__val" in result
        assert "print" in result
        assert "repr" in result

    def test_assignment_not_wrapped(self):
        """Assignment statement is returned unchanged."""
        code = "x = 42"
        result = wrap_last_expr_with_print_repr(code)
        assert result == code

    def test_multiple_statements_last_is_expr(self):
        """Multiple statements: only last expression is wrapped if it's an expr."""
        code = "x = 1\nx + 1"
        result = wrap_last_expr_with_print_repr(code)
        assert "__val" in result
        assert "x = 1" in result

    def test_multiple_statements_last_is_assignment(self):
        """Multiple statements: nothing wrapped if last is assignment."""
        code = "x = 1\ny = 2"
        result = wrap_last_expr_with_print_repr(code)
        assert result == code

    def test_empty_code(self):
        """Empty code is returned unchanged."""
        code = ""
        result = wrap_last_expr_with_print_repr(code)
        assert result == code

    def test_syntax_error_returns_original(self):
        """Code with syntax errors is returned unchanged."""
        code = "def foo(:"
        result = wrap_last_expr_with_print_repr(code)
        assert result == code

    def test_function_call_expression(self):
        """Function call as last expression gets wrapped."""
        code = "print('hello')"
        result = wrap_last_expr_with_print_repr(code)
        assert "__val" in result

    def test_none_check_in_wrapped_code(self):
        """Wrapped code includes a None check (if __val is not None)."""
        code = "42"
        result = wrap_last_expr_with_print_repr(code)
        assert "is not None" in result

    def test_for_loop_not_wrapped(self):
        """For loop is not an expression, should not be wrapped."""
        code = "for i in range(10):\n    pass"
        result = wrap_last_expr_with_print_repr(code)
        assert result == code

    def test_if_statement_not_wrapped(self):
        """If statement is not an expression, should not be wrapped."""
        code = "if True:\n    pass"
        result = wrap_last_expr_with_print_repr(code)
        assert result == code

    def test_class_definition_not_wrapped(self):
        """Class definition is not an expression, should not be wrapped."""
        code = "class Foo:\n    pass"
        result = wrap_last_expr_with_print_repr(code)
        assert result == code

    def test_wrapped_code_executes_correctly(self):
        """Wrapped code executes and captures the expression value."""
        code = "2 + 3"
        result = wrap_last_expr_with_print_repr(code)
        # Execute the wrapped code and capture output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        exec(result)
        sys.stdout = old_stdout
        output = buffer.getvalue().strip()
        assert output == "5"

    def test_wrapped_none_expression_no_output(self):
        """Wrapped code with None expression produces no output."""
        code = "None"
        result = wrap_last_expr_with_print_repr(code)
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        exec(result)
        sys.stdout = old_stdout
        output = buffer.getvalue().strip()
        assert output == ""
