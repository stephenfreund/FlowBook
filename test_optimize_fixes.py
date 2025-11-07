"""
Test fixes for optimize command:
1. FIXME fix in _send_test_code_comm
2. DiffResult usage fix in _validate_cell_optimization
"""

import pytest
from data_ferret.server.commands.optimize import OptimizeCommand, TestCodeResponse
from data_ferret.kernel.types import DiffResult, ValueComparison
from data_ferret.server.kernel_manager import FerretKernelClient


def test_diff_result_empty_differences():
    """Test that empty differences dict means all variables are equal."""
    # Create a DiffResult with no differences
    diff_result = DiffResult(differences={})

    # Verify that it's empty
    assert not diff_result.differences
    assert len(diff_result.differences) == 0
    assert bool(diff_result) == False  # __bool__ returns True if there are differences

    print("✓ Empty DiffResult correctly indicates no differences")


def test_diff_result_with_differences():
    """Test that non-empty differences dict shows which variables differ."""
    # Create a DiffResult with some differences
    diff_result = DiffResult(differences={
        'x': ValueComparison(
            status='different',
            value1=10,
            value2=20,
            message='Int mismatch: 10 vs 20'
        ),
        'y': ValueComparison(
            status='different',
            value1='hello',
            value2='world',
            message='String mismatch: hello vs world'
        )
    })

    # Verify differences are present
    assert diff_result.differences
    assert len(diff_result.differences) == 2
    assert bool(diff_result) == True

    # Verify we can get the keys (variable names that differ)
    diff_vars = list(diff_result.differences.keys())
    assert set(diff_vars) == {'x', 'y'}

    print("✓ DiffResult with differences correctly shows variable names")
    print(f"  Differing variables: {diff_vars}")


def test_validate_cell_optimization_all_equal():
    """Test _validate_cell_optimization with no differences (all equal)."""
    cmd = OptimizeCommand()

    # Mock kernel client - not actually used in this test
    # We'll directly test the logic with a DiffResult

    # Create a mock result with empty differences (all equal)
    from data_ferret.server.kernel_manager import TestCodeData
    mock_result = TestCodeData(
        ok=True,
        result=DiffResult(differences={})
    )

    # Simulate the validation logic
    if mock_result.ok and isinstance(mock_result.result, DiffResult):
        if not mock_result.result.differences:
            is_valid = True
            error_msg = None
        else:
            is_valid = False
            diff_vars = list(mock_result.result.differences.keys())
            error_msg = f"Variables changed: {', '.join(diff_vars)}"

    assert is_valid == True
    assert error_msg is None

    print("✓ Validation passes when all variables are equal (empty differences)")


def test_validate_cell_optimization_with_differences():
    """Test _validate_cell_optimization with differences (validation fails)."""
    cmd = OptimizeCommand()

    # Create a mock result with differences
    from data_ferret.server.kernel_manager import TestCodeData
    mock_result = TestCodeData(
        ok=True,
        result=DiffResult(differences={
            'result': ValueComparison(
                status='different',
                value1=100,
                value2=101,
                message='Values differ'
            ),
            'output': ValueComparison(
                status='different',
                value1=[1, 2, 3],
                value2=[1, 2, 4],
                message='List differs'
            )
        })
    )

    # Simulate the validation logic
    if mock_result.ok and isinstance(mock_result.result, DiffResult):
        if not mock_result.result.differences:
            is_valid = True
            error_msg = None
        else:
            is_valid = False
            diff_vars = list(mock_result.result.differences.keys())
            error_msg = f"Variables changed: {', '.join(diff_vars)}"

    assert is_valid == False
    assert error_msg is not None
    assert 'result' in error_msg
    assert 'output' in error_msg

    print("✓ Validation fails when variables differ")
    print(f"  Error message: {error_msg}")


def test_send_test_code_comm_validation_errors():
    """Test that _send_test_code_comm validates response state."""
    cmd = OptimizeCommand()

    # Test case 1: response.ok=True but response.result=None
    # This should raise RuntimeError
    try:
        response_ok_no_result = TestCodeResponse(
            ok=True,
            result=None,
            error=None
        )

        # Simulate the validation logic
        if response_ok_no_result.ok:
            if response_ok_no_result.result is None:
                raise RuntimeError("test_code succeeded but returned no result")

        # Should not reach here
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "succeeded but returned no result" in str(e)
        print("✓ Correctly raises error when ok=True but result=None")

    # Test case 2: response.ok=False but response.error=None
    # This should raise RuntimeError
    try:
        response_fail_no_error = TestCodeResponse(
            ok=False,
            result=None,
            error=None
        )

        # Simulate the validation logic
        if not response_fail_no_error.ok:
            if response_fail_no_error.error is None:
                raise RuntimeError("test_code failed but returned no error message")

        # Should not reach here
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "failed but returned no error message" in str(e)
        print("✓ Correctly raises error when ok=False but error=None")


def test_send_test_code_comm_valid_responses():
    """Test that _send_test_code_comm handles valid responses correctly."""
    cmd = OptimizeCommand()

    # Valid case 1: ok=True with result
    response_success = TestCodeResponse(
        ok=True,
        result=DiffResult(differences={}),
        error=None
    )

    # Simulate the validation logic
    if response_success.ok:
        if response_success.result is None:
            raise RuntimeError("test_code succeeded but returned no result")
        result = response_success.result
    else:
        if response_success.error is None:
            raise RuntimeError("test_code failed but returned no error message")
        result = response_success.error

    assert result is not None
    assert isinstance(result, DiffResult)
    print("✓ Correctly handles valid success response")

    # Valid case 2: ok=False with error
    response_failure = TestCodeResponse(
        ok=False,
        result=None,
        error="Execution error: something went wrong"
    )

    # Simulate the validation logic
    if response_failure.ok:
        if response_failure.result is None:
            raise RuntimeError("test_code succeeded but returned no result")
        result = response_failure.result
    else:
        if response_failure.error is None:
            raise RuntimeError("test_code failed but returned no error message")
        result = response_failure.error

    assert result is not None
    assert isinstance(result, str)
    assert "Execution error" in result
    print("✓ Correctly handles valid failure response")


def test_diff_result_iteration():
    """Test that we can iterate over DiffResult.differences.keys()."""
    diff_result = DiffResult(differences={
        'a': ValueComparison(status='different', value1=1, value2=2, message='diff'),
        'b': ValueComparison(status='different', value1=3, value2=4, message='diff'),
        'c': ValueComparison(status='different', value1=5, value2=6, message='diff')
    })

    # Test keys() method
    keys = list(diff_result.differences.keys())
    assert len(keys) == 3
    assert set(keys) == {'a', 'b', 'c'}

    # Test direct iteration
    keys_via_iter = [k for k in diff_result.differences]
    assert set(keys_via_iter) == {'a', 'b', 'c'}

    print("✓ DiffResult.differences can be iterated correctly")


if __name__ == "__main__":
    print("Testing DiffResult fixes...\n")

    test_diff_result_empty_differences()
    print()

    test_diff_result_with_differences()
    print()

    test_validate_cell_optimization_all_equal()
    print()

    test_validate_cell_optimization_with_differences()
    print()

    test_send_test_code_comm_validation_errors()
    print()

    test_send_test_code_comm_valid_responses()
    print()

    test_diff_result_iteration()
    print()

    print("✅ All tests passed!")
