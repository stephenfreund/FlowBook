"""
Demo of the format_diff_as_markdown function.

This script demonstrates how the markdown formatter produces human-readable
output for various types of differences.
"""

from flowbook.kernel.diff import Diff
from flowbook.kernel.types import format_diff_as_markdown


def demo():
    """Run a comprehensive demo of markdown formatting."""
    differ = Diff()

    print("=" * 70)
    print("DEMO: format_diff_as_markdown()")
    print("=" * 70)
    print()

    # Example 1: Simple variable differences
    print("Example 1: Simple variable differences")
    print("-" * 70)
    ns1 = {'x': 1, 'name': 'Alice', 'active': True}
    ns2 = {'x': 2, 'name': 'Bob', 'active': False}
    result = differ.diff(ns1, ns2)
    print(format_diff_as_markdown(result))
    print()

    # Example 2: Close floats (with tolerance)
    print("Example 2: Close floats (within tolerance)")
    print("-" * 70)
    ns1 = {'pi': 3.141592653589793, 'e': 2.718281828459045}
    ns2 = {'pi': 3.141592653589794, 'e': 2.718281828459046}
    result = differ.diff(ns1, ns2)
    print(format_diff_as_markdown(result))
    print()

    # Example 3: Nested data structures
    print("Example 3: Nested data structures")
    print("-" * 70)
    ns1 = {
        'config': {
            'timeout': 30,
            'retries': 3,
            'debug': False
        },
        'data': [1, 2, 3, 4, 5]
    }
    ns2 = {
        'config': {
            'timeout': 60,
            'retries': 3,
            'debug': True
        },
        'data': [1, 99, 3, 88, 5]
    }
    result = differ.diff(ns1, ns2)
    print(format_diff_as_markdown(result))
    print()

    # Example 4: Complex nested objects
    print("Example 4: Complex nested structures")
    print("-" * 70)

    class Person:
        def __init__(self, name, age, address):
            self.name = name
            self.age = age
            self.address = address

    ns1 = {
        'user': Person('Alice', 30, {'city': 'NYC', 'zip': '10001'}),
        'scores': [85, 90, 88]
    }
    ns2 = {
        'user': Person('Alice', 31, {'city': 'Boston', 'zip': '02101'}),
        'scores': [85, 92, 88]
    }
    result = differ.diff(ns1, ns2)
    print(format_diff_as_markdown(result))
    print()

    # Example 5: Large list with truncation
    print("Example 5: Large list (with truncation)")
    print("-" * 70)
    differ_limited = Diff(max_diffs_per_container=5)
    ns1 = {'values': list(range(100))}
    ns2 = {'values': [x + 10 for x in range(100)]}
    result = differ_limited.diff(ns1, ns2)
    print(format_diff_as_markdown(result))
    print()

    # Example 6: No differences
    print("Example 6: No differences")
    print("-" * 70)
    ns1 = {'x': 1, 'y': 2, 'z': 3}
    ns2 = {'x': 1, 'y': 2, 'z': 3}
    result = differ.diff(ns1, ns2)
    print(format_diff_as_markdown(result))
    print()

    # Example 7: Added and removed variables
    print("Example 7: Added and removed variables")
    print("-" * 70)
    ns1 = {'old_var': 123, 'common': 'hello'}
    ns2 = {'new_var': 456, 'common': 'hello'}
    result = differ.diff(ns1, ns2)
    print(format_diff_as_markdown(result))
    print()

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo()
