"""Test the format_diff_as_markdown function."""

from flowbook.kernel.types import ValueComparison, format_diff_as_markdown
from flowbook.kernel.diff import Diff


def test_empty_diff():
    """Test formatting when there are no differences."""
    result = {}
    markdown = format_diff_as_markdown(result)
    print("Empty diff:")
    print(markdown)
    print()
    assert "No Differences Found" in markdown


def test_simple_difference():
    """Test formatting a simple variable difference."""
    result = {
        'x': ValueComparison(
            status='different',
            value1=1,
            value2=2,
            message='Int mismatch at x: 1 vs 2'
        )
    }
    markdown = format_diff_as_markdown(result)
    print("Simple difference:")
    print(markdown)
    print()
    assert "**x**:" in markdown
    assert "Int mismatch" in markdown


def test_close_float():
    """Test formatting a close float with status indicator."""
    result = {
        'y': ValueComparison(
            status='close',
            value1=1.0000001,
            value2=1.0000002,
            message='Float close at y: 1.0000001 vs 1.0000002 (within tolerance)'
        )
    }
    markdown = format_diff_as_markdown(result)
    print("Close float:")
    print(markdown)
    print()
    assert "**y** *(close)*:" in markdown
    assert "Float close" in markdown


def test_nested_list_diffs():
    """Test formatting nested list differences."""
    result = {
        'nums': {
            '[0]': ValueComparison(
                status='different',
                value1=1,
                value2=10,
                message='Int mismatch at nums[0]: 1 vs 10'
            ),
            '[2]': ValueComparison(
                status='different',
                value1=3,
                value2=30,
                message='Int mismatch at nums[2]: 3 vs 30'
            )
        }
    }
    markdown = format_diff_as_markdown(result)
    print("Nested list diffs:")
    print(markdown)
    print()
    assert "**nums[0]**:" in markdown
    assert "**nums[2]**:" in markdown
    assert "1 vs 10" in markdown
    assert "3 vs 30" in markdown


def test_nested_dict_diffs():
    """Test formatting nested dict differences."""
    result = {
        'config': {
            "['timeout']": ValueComparison(
                status='different',
                value1=30,
                value2=60,
                message="Int mismatch at config['timeout']: 30 vs 60"
            ),
            "['debug']": ValueComparison(
                status='different',
                value1=True,
                value2=False,
                message="Bool mismatch at config['debug']: True vs False"
            )
        }
    }
    markdown = format_diff_as_markdown(result)
    print("Nested dict diffs:")
    print(markdown)
    print()
    assert "**config['timeout']**:" in markdown
    assert "**config['debug']**:" in markdown


def test_deeply_nested_structure():
    """Test formatting deeply nested structure."""
    result = {
        'data': {
            '.items': {
                '[0]': {
                    '.value': ValueComparison(
                        status='different',
                        value1=100,
                        value2=200,
                        message='Int mismatch at data.items[0].value: 100 vs 200'
                    )
                }
            }
        }
    }
    markdown = format_diff_as_markdown(result)
    print("Deeply nested structure:")
    print(markdown)
    print()
    assert "**data.items[0].value**:" in markdown
    assert "100 vs 200" in markdown


def test_truncation_marker():
    """Test formatting with truncation marker."""
    result = {
        'big_list': {
            '[0]': ValueComparison(status='different', value1=1, value2=10, message='Diff at [0]'),
            '[1]': ValueComparison(status='different', value1=2, value2=20, message='Diff at [1]'),
            '_truncated': ValueComparison(
                status='different',
                value1=None,
                value2=None,
                message='Truncated after 2 differences'
            )
        }
    }
    markdown = format_diff_as_markdown(result)
    print("With truncation:")
    print(markdown)
    print()
    assert "Truncated after 2 differences" in markdown


def test_multiple_variables():
    """Test formatting multiple variables (should be sorted)."""
    result = {
        'z': ValueComparison(status='different', value1=3, value2=4, message='z diff'),
        'a': ValueComparison(status='different', value1=1, value2=2, message='a diff'),
        'm': ValueComparison(status='different', value1=5, value2=6, message='m diff'),
    }
    markdown = format_diff_as_markdown(result)
    print("Multiple variables (sorted):")
    print(markdown)
    print()

    # Variables should appear in sorted order
    lines = markdown.split('\n')
    var_lines = [l for l in lines if l.startswith('- **')]
    assert '**a**' in var_lines[0]
    assert '**m**' in var_lines[1]
    assert '**z**' in var_lines[2]


def test_real_diff_result():
    """Test with an actual Diff object result."""
    differ = Diff()

    # Create two namespaces with multiple differences
    ns1 = {
        'x': 1,
        'y': 1.0000001,
        'data': {'a': 1, 'b': 2, 'c': 3},
        'items': [10, 20, 30]
    }
    ns2 = {
        'x': 2,
        'y': 1.0000002,  # Close enough
        'data': {'a': 1, 'b': 99, 'c': 3},  # b is different
        'items': [10, 99, 30]  # items[1] is different
    }

    result = differ.diff(ns1, ns2)
    markdown = format_diff_as_markdown(result)

    print("Real diff result:")
    print(markdown)
    print()

    assert "**x**:" in markdown
    assert "**y** *(close)*:" in markdown  # Should show close indicator
    assert "**data['b']**:" in markdown
    assert "**items[1]**:" in markdown


if __name__ == '__main__':
    test_empty_diff()
    test_simple_difference()
    test_close_float()
    test_nested_list_diffs()
    test_nested_dict_diffs()
    test_deeply_nested_structure()
    test_truncation_marker()
    test_multiple_variables()
    test_real_diff_result()

    print("✓ All markdown formatting tests passed!")
