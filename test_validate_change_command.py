#!/usr/bin/env python3
"""
Test script for the ValidateChangeCommand.

This script creates a simple notebook with test cases and verifies
the ValidateChangeCommand processes them correctly using a mock kernel client.
"""

import asyncio
from unittest.mock import MagicMock
from data_ferret.server.registry import CommandRegistry
from data_ferret.server.kernel_manager import TestCodeData


async def main():
    print("=" * 70)
    print("ValidateChangeCommand Verification")
    print("=" * 70)

    # Get the command from registry
    print("\n[1] Getting validate_change command from registry...")
    registry = CommandRegistry()
    validate_cmd = registry.get_command('validate_change')

    print(f"    Command name: {validate_cmd.command_name}")
    print(f"    Display name: {validate_cmd.display_name}")
    print(f"    Icon: {validate_cmd.icon_name}")
    print(f"    Tooltip: {validate_cmd.tooltip}")
    print(f"    Requires kernel: {validate_cmd.requires_kernel}")

    # Create a mock kernel client
    print("\n[2] Creating mock kernel client...")
    mock_kernel_client = MagicMock()

    # We need to mock the _send_test_code_comm method's response
    # The actual method sends comm messages and waits for replies
    # For testing, we'll mock the entire comm exchange

    # Create a simple notebook with test cases
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "x = 1\ny = 2",
                "metadata": {},
                "outputs": []
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "x = 1\ny = 2",  # Same as cell1 - should pass
                "metadata": {},
                "outputs": []
            },
            {
                "id": "cell3",
                "cell_type": "code",
                "source": "a = 5\nb = 10",
                "metadata": {},
                "outputs": []
            },
            {
                "id": "cell4",
                "cell_type": "code",
                "source": "a = 5\nb = 20",  # Different from cell3 - should fail
                "metadata": {},
                "outputs": []
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # Test Case 1: No cells selected
    print("\n[3] Test Case 1: No cells selected...")
    result = await validate_cmd.process(
        notebook_content=notebook,
        kernel_client=mock_kernel_client,
        selected_cell_ids=None
    )

    metadata = result.get('metadata', {})
    print(f"    Status: {metadata.get('status')}")
    print(f"    Total processed: {metadata.get('total_processed')}")
    print(f"    Results: {metadata.get('results')}")

    assert metadata.get('status') == 'success', "Expected success status"
    assert metadata.get('total_processed') == 0, "Expected 0 cells processed"
    assert metadata.get('results') == {}, "Expected empty results"
    print("    ✓ Passed")

    # Test Case 2: Empty selection
    print("\n[4] Test Case 2: Empty selection list...")
    result = await validate_cmd.process(
        notebook_content=notebook,
        kernel_client=mock_kernel_client,
        selected_cell_ids=[]
    )

    metadata = result.get('metadata', {})
    print(f"    Status: {metadata.get('status')}")
    print(f"    Total processed: {metadata.get('total_processed')}")

    assert metadata.get('status') == 'success', "Expected success status"
    assert metadata.get('total_processed') == 0, "Expected 0 cells processed"
    print("    ✓ Passed")

    # Test Case 3: No kernel client
    print("\n[5] Test Case 3: No kernel client...")
    result = await validate_cmd.process(
        notebook_content=notebook,
        kernel_client=None,
        selected_cell_ids=["cell1"]
    )

    metadata = result.get('metadata', {})
    print(f"    Status: {metadata.get('status')}")
    print(f"    Error: {metadata.get('error')}")

    assert metadata.get('status') == 'error', "Expected error status"
    assert 'kernel' in metadata.get('error', '').lower(), "Expected kernel error"
    print("    ✓ Passed")

    # Test Case 4: Structure verification with mock comm handler
    print("\n[6] Test Case 4: Verify command structure...")

    # Mock the comm exchange
    # We need to mock the internal _send_test_code_comm method
    original_send = validate_cmd._send_test_code_comm

    call_count = 0
    def mock_send(kernel_client, original_code, modified_code, output_variables):
        nonlocal call_count
        call_count += 1
        # Return mock success
        return TestCodeData(
            ok=True,
            result=f"Test passed for call {call_count}"
        )

    validate_cmd._send_test_code_comm = mock_send

    try:
        result = await validate_cmd.process(
            notebook_content=notebook,
            kernel_client=mock_kernel_client,
            selected_cell_ids=["cell1", "cell3"]
        )

        metadata = result.get('metadata', {})
        print(f"    Status: {metadata.get('status')}")
        print(f"    Total processed: {metadata.get('total_processed')}")
        print(f"    Results keys: {list(metadata.get('results', {}).keys())}")

        assert metadata.get('status') == 'success', "Expected success status"
        assert metadata.get('total_processed') == 2, f"Expected 2 cells processed, got {metadata.get('total_processed')}"
        assert 'cell1' in metadata.get('results', {}), "Expected cell1 in results"
        assert 'cell3' in metadata.get('results', {}), "Expected cell3 in results"

        # Verify result structure
        cell1_result = metadata['results']['cell1']
        print(f"\n    Cell1 result structure:")
        print(f"      ok: {cell1_result.get('ok')}")
        print(f"      result: {cell1_result.get('result')}")
        print(f"      error: {cell1_result.get('error')}")

        assert 'ok' in cell1_result, "Expected 'ok' field in result"
        assert 'result' in cell1_result, "Expected 'result' field in result"
        assert 'error' in cell1_result, "Expected 'error' field in result"

        print("    ✓ Passed")

    finally:
        # Restore original method
        validate_cmd._send_test_code_comm = original_send

    # Test Case 5: Verify next cell extraction
    print("\n[7] Test Case 5: Verify next cell code extraction...")

    next_code = validate_cmd._get_next_cell_source(notebook['cells'], 'cell1')
    print(f"    Next cell for cell1: {repr(next_code)}")
    assert next_code == "x = 1\ny = 2", "Expected cell2's code"

    next_code = validate_cmd._get_next_cell_source(notebook['cells'], 'cell4')
    print(f"    Next cell for cell4 (last): {repr(next_code)}")
    assert next_code == "", "Expected empty string for last cell"

    print("    ✓ Passed")

    # Test Case 6: Verify output variables extraction
    print("\n[8] Test Case 6: Verify output variables filtering...")

    # We need to analyze the notebook first to get dependencies
    from data_ferret.util.dependencies import analyze_notebook

    deps_dict = analyze_notebook(notebook)

    out_vars = validate_cmd._get_cell_output_variables(deps_dict, 'cell1')
    print(f"    Output variables for cell1: {out_vars}")
    assert isinstance(out_vars, list), "Expected list of variables"
    assert 'x' in out_vars, "Expected 'x' in output variables"
    assert 'y' in out_vars, "Expected 'y' in output variables"

    # Verify system variables are filtered
    assert not any(v.startswith('_') for v in out_vars), "System variables should be filtered"

    print("    ✓ Passed")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
