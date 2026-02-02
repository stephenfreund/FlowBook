#!/usr/bin/env python3
"""
Demonstration that the test_comm command is properly structured.
This doesn't actually run the kernel but shows the command is registered and callable.
"""

from unittest.mock import Mock, MagicMock
import asyncio
from flowbook.server.registry import CommandRegistry
from flowbook.server.kernel_manager import TestCodeData


async def main():
    print("=" * 70)
    print("Test Comm Command Structure Demonstration")
    print("=" * 70)

    # Get the registry and command
    print("\n[1] Getting test_comm command from registry...")
    registry = CommandRegistry()
    test_comm_cmd = registry.get_command('test_comm')

    print(f"    Command name: {test_comm_cmd.command_name}")
    print(f"    Display name: {test_comm_cmd.display_name}")
    print(f"    Icon: {test_comm_cmd.icon_name}")
    print(f"    Tooltip: {test_comm_cmd.tooltip}")
    print(f"    Requires kernel: {test_comm_cmd.requires_kernel}")

    # Create a mock kernel client
    print("\n[2] Creating mock kernel client...")
    mock_kernel_client = MagicMock()

    # Mock the test_code method to return a success response
    mock_result = TestCodeData(
        ok=True,
        result="Mock test result: Successfully processed test code!"
    )
    mock_kernel_client.test_code.return_value = mock_result

    # Create a simple notebook
    notebook = {
        "cells": [{"id": "cell1", "cell_type": "code", "source": "x = 1"}],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # Execute the command with the mock client
    print("[3] Executing test_comm command with mock kernel client...")
    result = await test_comm_cmd.process(
        notebook_content=notebook,
        kernel_client=mock_kernel_client
    )

    # Display the results
    print("\n" + "=" * 70)
    print("RESULTS (from mock)")
    print("=" * 70)

    metadata = result.get('metadata', {})
    print(f"\nStatus: {metadata.get('status')}")
    print(f"Command: {metadata.get('command')}")

    test_result = metadata.get('test_comm_result', {})
    print(f"\nTest Comm Result:")
    print(f"  OK: {test_result.get('ok')}")
    print(f"  Result: {test_result.get('result')}")

    # Verify the kernel client method was called
    print("\n[4] Verifying kernel client interaction...")
    print(f"    test_code() was called: {mock_kernel_client.test_code.called}")
    print(f"    test_code() call count: {mock_kernel_client.test_code.call_count}")

    print("\n" + "=" * 70)
    print("✓ Command structure is correct!")
    print("  When used with a real kernel client, it will:")
    print("  1. Call kernel_client.test_code()")
    print("  2. Receive TestCodeData response")
    print("  3. Return the result in metadata")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
