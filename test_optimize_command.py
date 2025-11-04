"""
Tests for the optimize command functionality.
"""

import pytest
from data_ferret.server.commands.optimize import extract_function, replace_function
from data_ferret.util.ferret_metadata import CodeSnippet, OptimizedCodeResponse


class TestExtractFunction:
    """Tests for extract_function helper."""

    def test_extract_simple_function(self):
        """Test extracting a simple function."""
        source = """
def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
"""
        result = extract_function(source, "hello")
        assert result is not None
        assert "def hello():" in result
        assert 'print("Hello, World!")' in result
        assert "goodbye" not in result

    def test_extract_function_with_args(self):
        """Test extracting a function with arguments."""
        source = """
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y
"""
        result = extract_function(source, "add")
        assert result is not None
        assert "def add(a, b):" in result
        assert "return a + b" in result
        assert "multiply" not in result

    def test_extract_function_with_decorators(self):
        """Test extracting a function with decorators."""
        source = """
@property
def name(self):
    return self._name

def other():
    pass
"""
        result = extract_function(source, "name")
        assert result is not None
        assert "@property" in result
        assert "def name(self):" in result

    def test_extract_multiline_function(self):
        """Test extracting a multi-line function."""
        source = """
def complex_function(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""
        result = extract_function(source, "complex_function")
        assert result is not None
        assert "def complex_function(data):" in result
        assert "result = []" in result
        assert "return result" in result

    def test_extract_nested_function(self):
        """Test extracting outer function (nested functions included)."""
        source = """
def outer():
    def inner():
        return 42
    return inner()
"""
        result = extract_function(source, "outer")
        assert result is not None
        assert "def outer():" in result
        assert "def inner():" in result
        assert "return inner()" in result

    def test_extract_nonexistent_function(self):
        """Test extracting a function that doesn't exist."""
        source = """
def hello():
    print("Hello!")
"""
        result = extract_function(source, "nonexistent")
        assert result is None

    def test_extract_function_invalid_syntax(self):
        """Test extracting from invalid Python code."""
        source = "this is not valid python code {"
        result = extract_function(source, "anything")
        assert result is None

    def test_extract_function_from_class(self):
        """Test extracting a top-level function when class methods exist."""
        source = """
class MyClass:
    def method(self):
        pass

def standalone():
    return "I'm standalone"
"""
        result = extract_function(source, "standalone")
        assert result is not None
        assert "def standalone():" in result
        assert "class MyClass" not in result


class TestReplaceFunction:
    """Tests for replace_function helper."""

    def test_replace_simple_function(self):
        """Test replacing a simple function."""
        source = """
def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
"""
        new_function = """def hello():
    print("Hello, Universe!")"""

        result = replace_function(source, "hello", new_function)
        assert "Hello, Universe!" in result
        assert "Hello, World!" not in result
        assert "def goodbye():" in result

    def test_replace_first_function(self):
        """Test replacing the first function in a file."""
        source = """def first():
    return 1

def second():
    return 2
"""
        new_function = """def first():
    return 100"""

        result = replace_function(source, "first", new_function)
        assert "return 100" in result
        # Check the exact pattern is replaced (not just substring)
        assert "    return 1\n" not in result
        assert "def second():" in result

    def test_replace_last_function(self):
        """Test replacing the last function in a file."""
        source = """def first():
    return 1

def second():
    return 2
"""
        new_function = """def second():
    return 200"""

        result = replace_function(source, "second", new_function)
        assert "return 200" in result
        # Check the exact pattern is replaced (not just substring)
        assert "    return 2\n" not in result
        assert "def first():" in result

    def test_replace_middle_function(self):
        """Test replacing a function in the middle."""
        source = """def first():
    return 1

def middle():
    return 2

def last():
    return 3
"""
        new_function = """def middle():
    return 222"""

        result = replace_function(source, "middle", new_function)
        assert "return 222" in result
        assert "def first():" in result
        assert "def last():" in result
        lines = result.split("\n")
        # Check order is preserved
        first_idx = next(i for i, line in enumerate(lines) if "def first()" in line)
        middle_idx = next(i for i, line in enumerate(lines) if "def middle()" in line)
        last_idx = next(i for i, line in enumerate(lines) if "def last()" in line)
        assert first_idx < middle_idx < last_idx

    def test_replace_function_preserves_surrounding_code(self):
        """Test that replacing a function preserves code around it."""
        source = """# Header comment
import os

def target():
    return 1

# Footer comment
x = 10
"""
        new_function = """def target():
    return 999"""

        result = replace_function(source, "target", new_function)
        assert "# Header comment" in result
        assert "import os" in result
        assert "# Footer comment" in result
        assert "x = 10" in result
        assert "return 999" in result

    def test_replace_nonexistent_function(self):
        """Test replacing a function that doesn't exist returns original."""
        source = """def hello():
    print("Hello!")
"""
        new_function = """def nonexistent():
    pass"""

        result = replace_function(source, "nonexistent", new_function)
        assert result == source

    def test_replace_function_invalid_syntax(self):
        """Test replacing in invalid Python code returns original."""
        source = "this is not valid python code {"
        new_function = "def anything(): pass"

        result = replace_function(source, "anything", new_function)
        assert result == source

    def test_replace_function_multiline_replacement(self):
        """Test replacing with a multi-line function."""
        source = """def simple():
    return 1

def other():
    return 2
"""
        new_function = """def simple():
    # This is a longer function
    x = 10
    y = 20
    return x + y"""

        result = replace_function(source, "simple", new_function)
        assert "x = 10" in result
        assert "y = 20" in result
        assert "return x + y" in result
        assert "return 1" not in result

    def test_replace_preserves_indentation_context(self):
        """Test that replacement works with proper context."""
        source = """def func1():
    pass

def func2():
    return 42

def func3():
    pass
"""
        new_function = """def func2():
    return 100"""

        result = replace_function(source, "func2", new_function)
        # Verify structure is maintained
        assert "def func1():" in result
        assert "def func2():" in result
        assert "def func3():" in result
        assert "return 100" in result
        assert "return 42" not in result


class TestCodeSnippet:
    """Tests for CodeSnippet model."""

    def test_code_snippet_creation(self):
        """Test creating a CodeSnippet."""
        snippet = CodeSnippet(
            cell_id="cell-123",
            function_name="my_function",
            source="def my_function():\n    pass"
        )
        assert snippet.cell_id == "cell-123"
        assert snippet.function_name == "my_function"
        assert "def my_function()" in snippet.source
        assert snippet.optimizations_applied is None

    def test_code_snippet_whole_cell(self):
        """Test CodeSnippet for whole cell (no function name)."""
        snippet = CodeSnippet(
            cell_id="cell-456",
            function_name=None,
            source="x = 10\ny = 20\nprint(x + y)"
        )
        assert snippet.cell_id == "cell-456"
        assert snippet.function_name is None
        assert "x = 10" in snippet.source
        assert snippet.optimizations_applied is None

    def test_code_snippet_with_optimizations(self):
        """Test CodeSnippet with optimizations applied."""
        snippet = CodeSnippet(
            cell_id="cell-789",
            function_name="optimized_func",
            source="def optimized_func():\n    return sum(range(100))",
            optimizations_applied=["Replaced loop with sum()", "Used built-in range()"]
        )
        assert snippet.cell_id == "cell-789"
        assert snippet.function_name == "optimized_func"
        assert snippet.optimizations_applied is not None
        assert len(snippet.optimizations_applied) == 2
        assert "Replaced loop with sum()" in snippet.optimizations_applied

    def test_code_snippet_serialization(self):
        """Test that CodeSnippet can be serialized/deserialized."""
        snippet = CodeSnippet(
            cell_id="cell-789",
            function_name="test_func",
            source="def test_func():\n    return True",
            optimizations_applied=["Added type hints"]
        )
        # Convert to dict and back
        data = snippet.model_dump()
        restored = CodeSnippet.model_validate(data)
        assert restored.cell_id == snippet.cell_id
        assert restored.function_name == snippet.function_name
        assert restored.source == snippet.source
        assert restored.optimizations_applied == snippet.optimizations_applied


class TestOptimizedCodeResponse:
    """Tests for OptimizedCodeResponse model."""

    def test_optimized_code_response_creation(self):
        """Test creating an OptimizedCodeResponse."""
        response = OptimizedCodeResponse(
            optimized_code="def foo():\n    return sum(range(10))",
            optimizations_applied=["Replaced loop with sum()", "Used built-in range()"]
        )
        assert "sum(range(10))" in response.optimized_code
        assert len(response.optimizations_applied) == 2
        assert "Replaced loop" in response.optimizations_applied[0]

    def test_optimized_code_response_empty_optimizations(self):
        """Test OptimizedCodeResponse with no optimizations (code already optimal)."""
        response = OptimizedCodeResponse(
            optimized_code="def bar():\n    return 42",
            optimizations_applied=[]
        )
        assert response.optimized_code == "def bar():\n    return 42"
        assert len(response.optimizations_applied) == 0

    def test_optimized_code_response_serialization(self):
        """Test that OptimizedCodeResponse can be serialized/deserialized."""
        response = OptimizedCodeResponse(
            optimized_code="x = [i**2 for i in range(100)]",
            optimizations_applied=["Replaced loop with list comprehension"]
        )
        # Convert to dict and back
        data = response.model_dump()
        restored = OptimizedCodeResponse.model_validate(data)
        assert restored.optimized_code == response.optimized_code
        assert restored.optimizations_applied == response.optimizations_applied


class TestOptimizeIntegration:
    """Integration tests for the optimize command flow."""

    def test_extract_and_replace_roundtrip(self):
        """Test extracting a function and replacing it."""
        source = """def calculate(x, y):
    return x + y

def process(data):
    return [x * 2 for x in data]
"""
        # Extract the calculate function
        extracted = extract_function(source, "calculate")
        assert extracted is not None

        # Create an optimized version
        optimized = """def calculate(x, y):
    # Optimized version
    return x + y"""

        # Replace it
        result = replace_function(source, "calculate", optimized)

        # Verify
        assert "# Optimized version" in result
        assert "def process(data):" in result
        assert "def calculate(x, y):" in result

    def test_multiple_replacements(self):
        """Test replacing multiple functions sequentially."""
        source = """def func_a():
    return 'a'

def func_b():
    return 'b'

def func_c():
    return 'c'
"""
        # Replace func_a
        source = replace_function(source, "func_a", "def func_a():\n    return 'A'")
        # Replace func_c
        source = replace_function(source, "func_c", "def func_c():\n    return 'C'")

        # Verify both replacements
        assert "return 'A'" in source
        assert "return 'b'" in source  # func_b unchanged
        assert "return 'C'" in source
        assert "return 'a'" not in source
        assert "return 'c'" not in source

    def test_extract_modify_replace_workflow(self):
        """Test the complete workflow of extract -> modify -> replace."""
        original_source = """import numpy as np

def slow_sum(arr):
    total = 0
    for item in arr:
        total += item
    return total

def main():
    data = [1, 2, 3, 4, 5]
    result = slow_sum(data)
    print(result)
"""
        # Extract the slow function
        slow_func = extract_function(original_source, "slow_sum")
        assert slow_func is not None
        assert "for item in arr:" in slow_func

        # Simulate optimization (in real code, this would come from LLM)
        optimized_func = """def slow_sum(arr):
    return sum(arr)"""

        # Replace with optimized version
        optimized_source = replace_function(original_source, "slow_sum", optimized_func)

        # Verify
        assert "return sum(arr)" in optimized_source
        assert "for item in arr:" not in optimized_source
        assert "def main():" in optimized_source
        assert "import numpy as np" in optimized_source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
