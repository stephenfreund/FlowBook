"""Test DiffResult serialization and deserialization."""

import json
from flowbook.kernel.diff import Diff
from flowbook.kernel.types import DiffResult, ValueComparison


def test_basic_serialization():
    """Test basic model_dump and model_dump_json."""
    result = DiffResult(differences={
        'x': ValueComparison(
            status='different',
            value1=1,
            value2=2,
            message='Int mismatch at x: 1 vs 2'
        )
    })

    # Test model_dump (to dict)
    data = result.model_dump()
    print("model_dump():")
    print(json.dumps(data, indent=2))
    print()

    assert 'differences' in data
    assert 'x' in data['differences']
    assert data['differences']['x']['status'] == 'different'
    assert data['differences']['x']['message'] == 'Int mismatch at x: 1 vs 2'

    # Test model_dump_json (to JSON string)
    json_str = result.model_dump_json(indent=2)
    print("model_dump_json():")
    print(json_str)
    print()

    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed['differences']['x']['status'] == 'different'


def test_deserialization():
    """Test model_validate and model_validate_json."""
    # Create a DiffResult
    original = DiffResult(differences={
        'x': ValueComparison(
            status='different',
            value1=1,
            value2=2,
            message='Int mismatch'
        ),
        'y': ValueComparison(
            status='close',
            value1=1.0000001,
            value2=1.0000002,
            message='Float close'
        )
    })

    # Serialize to JSON
    json_str = original.model_dump_json()

    # Deserialize from JSON
    restored = DiffResult.model_validate_json(json_str)

    print("Original:")
    print(original)
    print()
    print("Restored:")
    print(restored)
    print()

    # Verify the restoration
    assert isinstance(restored, DiffResult)
    assert 'x' in restored
    assert 'y' in restored
    assert isinstance(restored['x'], ValueComparison)
    assert isinstance(restored['y'], ValueComparison)
    assert restored['x'].status == 'different'
    assert restored['y'].status == 'close'
    assert restored['x'].message == 'Int mismatch'
    assert restored['y'].message == 'Float close'


def test_nested_structure_serialization():
    """Test serialization of nested structures."""
    differ = Diff()
    result = differ.diff(
        {'data': {'a': 1, 'b': 2, 'c': 3}},
        {'data': {'a': 1, 'b': 99, 'c': 3}}
    )

    print("Nested structure:")
    print(result)
    print()

    # Serialize
    json_str = result.model_dump_json(indent=2)
    print("JSON:")
    print(json_str)
    print()

    # Deserialize
    restored = DiffResult.model_validate_json(json_str)

    print("Restored:")
    print(restored)
    print()

    # Verify structure
    assert isinstance(restored, DiffResult)
    assert 'data' in restored
    assert isinstance(restored['data'], dict)
    assert "['b']" in restored['data']
    assert isinstance(restored['data']["['b']"], ValueComparison)


def test_real_world_diff():
    """Test with a real diff result."""
    differ = Diff()

    ns1 = {
        'x': 1,
        'y': 1.0000001,
        'items': [10, 20, 30],
        'config': {'timeout': 30, 'debug': False}
    }
    ns2 = {
        'x': 2,
        'y': 1.0000002,  # Close enough
        'items': [10, 99, 30],
        'config': {'timeout': 60, 'debug': False}
    }

    result = differ.diff(ns1, ns2)

    print("Real-world diff:")
    print(f"Found {len(result)} variables with differences")
    print()

    # Serialize
    json_str = result.model_dump_json(indent=2)
    print("JSON length:", len(json_str), "bytes")
    print()

    # Deserialize
    restored = DiffResult.model_validate_json(json_str)

    print("Restored successfully")
    print(f"Restored {len(restored)} variables")
    print()

    # Verify all variables are present
    for var in result.keys():
        assert var in restored
        print(f"✓ Variable '{var}' restored")

    print()
    print("All variables restored correctly!")


def test_empty_diff():
    """Test serialization of empty diff."""
    result = DiffResult(differences={})

    json_str = result.model_dump_json()
    print("Empty diff JSON:")
    print(json_str)
    print()

    restored = DiffResult.model_validate_json(json_str)
    assert len(restored) == 0
    assert not restored  # Should be falsy


def test_dict_compatibility():
    """Test that DiffResult works like a dict."""
    result = DiffResult(differences={
        'x': ValueComparison(
            status='different',
            value1=1,
            value2=2,
            message='test'
        )
    })

    # Test dict-like operations
    assert 'x' in result
    assert 'y' not in result
    assert len(result) == 1
    assert list(result.keys()) == ['x']
    assert isinstance(result['x'], ValueComparison)

    # Test comparison with dict
    assert result != {}
    assert result == {'x': result['x']}

    print("✓ Dict-like operations work correctly")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing DiffResult Serialization/Deserialization")
    print("=" * 70)
    print()

    test_basic_serialization()
    test_deserialization()
    test_nested_structure_serialization()
    test_real_world_diff()
    test_empty_diff()
    test_dict_compatibility()

    print("=" * 70)
    print("✓ All serialization tests passed!")
    print("=" * 70)
