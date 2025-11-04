#!/usr/bin/env python3
"""
Test script for the test_comm command.
"""

import asyncio
import json
from jupyter_client import KernelManager
from data_ferret.server.registry import CommandRegistry
from data_ferret.server.kernel_manager import FerretKernelClient


async def main():
    # Create a simple notebook
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "x = 1"
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    print("=" * 60)
    print("Testing Test Comm Command")
    print("=" * 60)

    # Start a kernel
    print("\n[1] Starting kernel...")
    km = KernelManager(kernel_name='ferret_kernel')
    km.start_kernel()

    try:
        # Create a FerretKernelClient
        print("[2] Creating kernel client...")
        ferret_client = FerretKernelClient(kernel_id="test_kernel")
        ferret_client.load_connection_info(km.get_connection_info())
        ferret_client.start_channels()
        ferret_client.wait_for_ready(timeout=30)

        # Get the test_comm command
        print("[3] Getting test_comm command...")
        registry = CommandRegistry()
        test_comm_cmd = registry.get_command('test_comm')

        # Execute the command
        print("[4] Executing test_comm command...")
        result = await test_comm_cmd.process(
            notebook_content=notebook,
            kernel_client=ferret_client
        )

        # Print the results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        metadata = result.get('metadata', {})
        print(f"\nStatus: {metadata.get('status')}")
        print(f"Command: {metadata.get('command')}")

        test_result = metadata.get('test_comm_result', {})
        print(f"\nTest Comm Result:")
        print(f"  OK: {test_result.get('ok')}")
        print(f"  Result: {test_result.get('result')}")

        print("\n" + "=" * 60)

    finally:
        # Clean up
        print("\n[5] Cleaning up...")
        ferret_client.stop_channels()
        km.shutdown_kernel()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
