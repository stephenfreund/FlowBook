"""
Test validation in optimize command.
"""

import nbformat
from data_ferret.server.commands.optimize import OptimizeCommand
from data_ferret.util.ferret_metadata import CodeSnippet
from data_ferret.util.dependencies import analyze_notebook, CellDependencies


def test_get_modified_globals_for_cell():
    """Test extracting modified globals from dependencies."""
    # Create a simple notebook
    nb = nbformat.v4.new_notebook()

    cell1 = nbformat.v4.new_code_cell(source="x = 10\ny = 20\nresult = x + y")
    cell1["id"] = "cell-1"

    cell2 = nbformat.v4.new_code_cell(source="z = result * 2")
    cell2["id"] = "cell-2"

    nb["cells"] = [cell1, cell2]

    # Analyze dependencies
    dependencies_dict = analyze_notebook(nb)

    # Test the helper function
    from data_ferret.server.commands.optimize import ValidationHelper

    # Cell 1 should write x, y, result
    globals_cell1 = ValidationHelper.get_modified_globals_for_cell("cell-1", dependencies_dict)
    assert "x" in globals_cell1
    assert "y" in globals_cell1
    assert "result" in globals_cell1

    # Cell 2 should write z
    globals_cell2 = ValidationHelper.get_modified_globals_for_cell("cell-2", dependencies_dict)
    assert "z" in globals_cell2

    # System variables should be filtered out
    assert "_" not in globals_cell1
    assert "get_ipython" not in globals_cell1

    print("✓ get_modified_globals_for_cell works correctly")


def test_build_optimized_code_for_validation():
    """Test building optimized code for validation."""
    from data_ferret.server.commands.optimize import ValidationHelper

    # Create snippets
    original_snippets = [
        CodeSnippet(
            cell_id="cell-1",
            function_name=None,
            source="def slow_sum(arr):\n    total = 0\n    for x in arr:\n        total += x\n    return total"
        )
    ]

    optimized_snippets = [
        CodeSnippet(
            cell_id="cell-1",
            function_name=None,
            source="def slow_sum(arr):\n    return sum(arr)",
            optimizations_applied=["Replaced loop with built-in sum()"]
        )
    ]

    # Create cell map
    cell_map = {
        "cell-1": {
            "id": "cell-1",
            "source": "def slow_sum(arr):\n    total = 0\n    for x in arr:\n        total += x\n    return total"
        }
    }

    # Build optimized code
    optimized_code = ValidationHelper.build_optimized_code_for_validation(
        original_snippets,
        optimized_snippets,
        cell_map,
        "cell-1"
    )

    # Verify structure
    assert "def slow_sum(arr):" in optimized_code
    assert "return sum(arr)" in optimized_code
    assert "# Optimized cell cell-1" in optimized_code

    print("✓ build_optimized_code_for_validation works correctly")
    print("\nGenerated code:")
    print(optimized_code)


def test_build_optimized_code_with_dependencies():
    """Test building optimized code when dependencies are modified."""
    from data_ferret.server.commands.optimize import ValidationHelper

    # Scenario: Cell 2 calls a function from Cell 1
    # Optimization modifies the function in Cell 1 and uses it in Cell 2

    original_snippets = [
        CodeSnippet(
            cell_id="cell-1",
            function_name="helper",
            source="def helper(x):\n    result = 0\n    for i in range(x):\n        result += i\n    return result"
        ),
        CodeSnippet(
            cell_id="cell-2",
            function_name=None,
            source="data = [1, 2, 3]\noutput = [helper(x) for x in data]"
        )
    ]

    optimized_snippets = [
        CodeSnippet(
            cell_id="cell-1",
            function_name="helper",
            source="def helper(x):\n    return sum(range(x))",
            optimizations_applied=["Vectorized loop"]
        ),
        CodeSnippet(
            cell_id="cell-2",
            function_name=None,
            source="data = [1, 2, 3]\noutput = [helper(x) for x in data]",
            optimizations_applied=[]
        )
    ]

    cell_map = {
        "cell-1": {"id": "cell-1", "source": "def helper(x):\n    ..."},
        "cell-2": {"id": "cell-2", "source": "data = [1, 2, 3]\n..."}
    }

    # Build code for validating cell-2
    optimized_code = ValidationHelper.build_optimized_code_for_validation(
        original_snippets,
        optimized_snippets,
        cell_map,
        "cell-2"  # Triggering cell
    )

    # Verify that dependency (cell-1) comes before triggering cell (cell-2)
    helper_pos = optimized_code.find("def helper(x):")
    data_pos = optimized_code.find("data = [1, 2, 3]")

    assert helper_pos < data_pos, "Dependencies should come before triggering cell"
    assert "Modified from cell cell-1" in optimized_code
    assert "Optimized cell cell-2" in optimized_code

    print("✓ build_optimized_code_for_validation handles dependencies correctly")
    print("\nGenerated code with dependencies:")
    print(optimized_code)


def test_dependencies_analysis():
    """Test that dependency analysis works for optimization validation."""
    # Create a notebook with dependencies
    nb = nbformat.v4.new_notebook()

    cell1 = nbformat.v4.new_code_cell(source="""
def slow_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total
""")
    cell1["id"] = "cell-1"

    cell2 = nbformat.v4.new_code_cell(source="""
data = [1, 2, 3, 4, 5]
result = slow_sum(data)
print(result)
""")
    cell2["id"] = "cell-2"

    nb["cells"] = [cell1, cell2]

    # Analyze dependencies
    dependencies_dict = analyze_notebook(nb)

    # Verify cell-1 writes slow_sum
    assert "slow_sum" in dependencies_dict["cell-1"].globals_written

    # Verify cell-2 reads slow_sum and writes result, data
    assert "slow_sum" in dependencies_dict["cell-2"].globals_read
    assert "result" in dependencies_dict["cell-2"].globals_written
    assert "data" in dependencies_dict["cell-2"].globals_written

    print("✓ Dependency analysis works correctly")
    print(f"\nCell 1 writes: {dependencies_dict['cell-1'].globals_written}")
    print(f"Cell 2 reads: {dependencies_dict['cell-2'].globals_read}")
    print(f"Cell 2 writes: {dependencies_dict['cell-2'].globals_written}")


if __name__ == "__main__":
    test_get_modified_globals_for_cell()
    print()
    test_build_optimized_code_for_validation()
    print()
    test_build_optimized_code_with_dependencies()
    print()
    test_dependencies_analysis()
    print("\n✅ All validation tests passed!")
