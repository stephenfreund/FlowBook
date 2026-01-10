"""
Demo of DiffResult as a Pydantic BaseModel.

This script demonstrates the serialization/deserialization capabilities
of the new DiffResult BaseModel.
"""

import json
from flowbook.kernel.diff import Diff
from flowbook.kernel.types import DiffResult, format_diff_as_markdown


def demo():
    """Run a comprehensive demo of DiffResult BaseModel features."""
    print("=" * 70)
    print("DEMO: DiffResult as Pydantic BaseModel")
    print("=" * 70)
    print()

    # Create a diff
    differ = Diff()
    ns1 = {
        'x': 1,
        'y': 3.14159265358979,
        'config': {'timeout': 30, 'retries': 3},
        'items': [10, 20, 30]
    }
    ns2 = {
        'x': 2,
        'y': 3.14159265358980,  # Close
        'config': {'timeout': 60, 'retries': 3},
        'items': [10, 99, 30]
    }

    result = differ.diff(ns1, ns2)

    print("1. Original DiffResult")
    print("-" * 70)
    print(f"Type: {type(result)}")
    print(f"Variables with differences: {list(result.keys())}")
    print()

    # Demo 1: Serialize to dict
    print("2. Serialize to Dict (model_dump)")
    print("-" * 70)
    data = result.model_dump()
    print(json.dumps(data, indent=2))
    print()

    # Demo 2: Serialize to JSON
    print("3. Serialize to JSON (model_dump_json)")
    print("-" * 70)
    json_str = result.model_dump_json(indent=2)
    print(json_str)
    print(f"\nJSON size: {len(json_str)} bytes")
    print()

    # Demo 3: Deserialize from JSON
    print("4. Deserialize from JSON (model_validate_json)")
    print("-" * 70)
    restored = DiffResult.model_validate_json(json_str)
    print(f"Type: {type(restored)}")
    print(f"Variables restored: {list(restored.keys())}")
    print(f"Matches original? {restored == result}")
    print()

    # Demo 4: Dict-like interface
    print("5. Dict-Like Interface")
    print("-" * 70)
    print(f"'x' in result: {'x' in result}")
    print(f"len(result): {len(result)}")
    print(f"result['x']: {result['x']}")
    print(f"result.get('z', 'not found'): {result.get('z', 'not found')}")
    print()

    # Demo 5: Iteration
    print("6. Iteration")
    print("-" * 70)
    for var, diff in result.items():
        print(f"  - {var}: {type(diff).__name__}")
    print()

    # Demo 6: Comparison
    print("7. Comparison with Dict")
    print("-" * 70)
    empty_result = DiffResult(differences={})
    print(f"empty_result == {{}}: {empty_result == {}}")
    print(f"result == {{}}: {result == {}}")
    print(f"result == result: {result == result}")
    print()

    # Demo 7: Save to file and restore
    print("8. Save to File and Restore")
    print("-" * 70)
    filename = "/tmp/diff_result.json"
    with open(filename, 'w') as f:
        f.write(result.model_dump_json(indent=2))
    print(f"Saved to: {filename}")

    with open(filename, 'r') as f:
        json_data = f.read()
    restored_from_file = DiffResult.model_validate_json(json_data)
    print(f"Restored from file: {len(restored_from_file)} variables")
    print(f"Matches original? {restored_from_file == result}")
    print()

    # Demo 8: Format as markdown
    print("9. Format as Markdown")
    print("-" * 70)
    markdown = format_diff_as_markdown(result)
    print(markdown)
    print()

    # Demo 9: API simulation
    print("10. Simulated API Usage")
    print("-" * 70)

    # Simulate sending via API
    api_payload = result.model_dump()
    print("API Request payload:")
    print(json.dumps(api_payload, indent=2)[:200] + "...")
    print()

    # Simulate receiving from API
    received = DiffResult.model_validate(api_payload)
    print(f"API Response processed: {len(received)} differences")
    print()

    # Demo 10: Complex nested structure
    print("11. Complex Nested Structure")
    print("-" * 70)

    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    complex_ns1 = {
        'user': Person('Alice', 30),
        'scores': [85, 90, 88],
        'metadata': {'version': '1.0', 'tags': ['prod', 'active']}
    }
    complex_ns2 = {
        'user': Person('Alice', 31),
        'scores': [85, 92, 88],
        'metadata': {'version': '2.0', 'tags': ['prod', 'active']}
    }

    complex_result = differ.diff(complex_ns1, complex_ns2)

    # Serialize and deserialize
    json_complex = complex_result.model_dump_json(indent=2)
    restored_complex = DiffResult.model_validate_json(json_complex)

    print(f"Complex diff: {len(restored_complex)} variables")
    print(f"Successfully serialized and deserialized: {restored_complex == complex_result}")
    print()

    # Show the structure
    for var in restored_complex:
        print(f"  - {var}")
    print()

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("- DiffResult is a full Pydantic BaseModel")
    print("- Seamless JSON serialization/deserialization")
    print("- Maintains dict-like interface for backward compatibility")
    print("- Perfect for APIs, databases, and file storage")
    print("- All nested ValueComparison objects are properly restored")


if __name__ == '__main__':
    demo()
