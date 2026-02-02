"""
Test script to validate diff.py refactoring works correctly.
"""

from flowbook.kernel.diff import Diff
from flowbook.kernel.types import ValueComparison, DiffResult

def test_basic_types():
    """Test basic type comparisons return ValueComparison."""
    differ = Diff()

    # Test equal namespaces
    result = differ.diff({"x": 1}, {"x": 1})
    assert result == {}, f"Expected empty diff, got: {result}"
    print("✓ Equal integers: no diff")

    # Test different integers
    result = differ.diff({"x": 1}, {"x": 2})
    assert "x" in result
    assert isinstance(result["x"], ValueComparison)
    assert result["x"].status == "different"
    print(f"✓ Different integers: {result['x'].message}")

    # Test float close
    result = differ.diff({"x": 1.0}, {"x": 1.0 + 1e-6})
    assert "x" in result
    assert isinstance(result["x"], ValueComparison)
    assert result["x"].status == "close"
    print(f"✓ Close floats: {result['x'].message}")

    # Test float different
    result = differ.diff({"x": 1.0}, {"x": 2.0})
    assert "x" in result
    assert isinstance(result["x"], ValueComparison)
    assert result["x"].status == "different"
    print(f"✓ Different floats: {result['x'].message}")


def test_list_all_diffs():
    """Test that list comparison finds ALL differences."""
    differ = Diff()

    # List with multiple differences
    result = differ.diff(
        {"x": [1, 2, 3, 4, 5]},
        {"x": [1, 99, 3, 88, 5]}
    )

    assert "x" in result
    assert isinstance(result["x"], dict)  # DiffNode is a dict

    # Should have differences at indices 1 and 3
    assert "[1]" in result["x"]
    assert "[3]" in result["x"]
    assert "[0]" not in result["x"]  # Index 0 is equal
    assert "[2]" not in result["x"]  # Index 2 is equal
    assert "[4]" not in result["x"]  # Index 4 is equal

    print(f"✓ List with multiple diffs: found {len(result['x'])} differences")
    print(f"  [1]: {result['x']['[1]'].message}")
    print(f"  [3]: {result['x']['[3]'].message}")


def test_dict_all_diffs():
    """Test that dict comparison finds ALL differences."""
    differ = Diff()

    # Dict with multiple differences
    result = differ.diff(
        {"x": {"a": 1, "b": 2, "c": 3, "d": 4}},
        {"x": {"a": 1, "b": 99, "c": 3, "d": 88}}
    )

    assert "x" in result
    assert isinstance(result["x"], dict)

    # Should have differences at keys 'b' and 'd'
    assert "['b']" in result["x"]
    assert "['d']" in result["x"]
    assert "['a']" not in result["x"]  # 'a' is equal
    assert "['c']" not in result["x"]  # 'c' is equal

    print(f"✓ Dict with multiple diffs: found {len(result['x'])} differences")
    b_key = "['b']"
    d_key = "['d']"
    print(f"  ['b']: {result['x'][b_key].message}")
    print(f"  ['d']: {result['x'][d_key].message}")


def test_nested_structures():
    """Test nested structures with multiple diffs."""
    differ = Diff()

    result = differ.diff(
        {"x": {"nested": [1, 2, 3]}},
        {"x": {"nested": [1, 99, 3]}}
    )

    assert "x" in result
    assert isinstance(result["x"], dict)
    assert "['nested']" in result["x"]
    assert "[1]" in result["x"]["['nested']"]

    print(f"✓ Nested structure: found diff in x['nested'][1]")


def test_only_differences():
    """Test that only differences are included in results."""
    differ = Diff()

    # Many equal items, few different
    result = differ.diff(
        {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
        {"a": 1, "b": 99, "c": 3, "d": 4, "e": 5}
    )

    # Should only have 'b' in result
    assert len(result) == 1
    assert "b" in result
    assert "a" not in result
    assert "c" not in result

    print("✓ Only differences included: 1 out of 5 variables")


def test_limits():
    """Test that diff limit is respected."""
    differ = Diff(max_diffs_per_container=5)

    # Create list with many differences
    list_a = list(range(100))
    list_b = [x + 1 for x in list_a]  # All elements different

    result = differ.diff({"x": list_a}, {"x": list_b})

    assert "x" in result
    # Should have at most 6 entries: 5 diffs + 1 truncated message
    assert len(result["x"]) <= 6
    assert "_truncated" in result["x"]

    print(f"✓ Limit respected: {len(result['x'])} entries (including truncation message)")


if __name__ == "__main__":
    print("Testing diff.py refactoring...\n")

    try:
        test_basic_types()
        print()
        test_list_all_diffs()
        print()
        test_dict_all_diffs()
        print()
        test_nested_structures()
        print()
        test_only_differences()
        print()
        test_limits()
        print()
        print("All tests passed! ✓")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
