#!/usr/bin/env python3
"""
Test script to verify progress messages from kernel.

This creates a real kernel connection and tests the validate_change command
to ensure progress messages are received and logged.
"""

import asyncio
import sys
from data_ferret.server.registry import CommandRegistry
from data_ferret.server.kernel_manager import KernelConnectionManager
from jupyter_server.serverapp import ServerApp


async def main():
    print("=" * 70)
    print("Progress Messages Test")
    print("=" * 70)

    # Initialize Jupyter server app
    print("\n[1] Initializing Jupyter server...")
    app = ServerApp()
    app.initialize(argv=[])

    # Get the command
    print("\n[2] Getting validate_change command from registry...")
    registry = CommandRegistry()
    validate_cmd = registry.get_command('validate_change')

    print(f"    Command name: {validate_cmd.command_name}")

    # Create kernel connection manager
    print("\n[3] Setting up kernel connection...")
    kernel_manager = KernelConnectionManager(app)

    # Start a ferret_kernel specifically
    kernel_id = await app.kernel_manager.start_kernel(kernel_name='ferret_kernel')
    print(f"    Started kernel: {kernel_id}")

    # Get kernel client
    kernel_client = kernel_manager.get_kernel_client(kernel_id)
    print(f"    Connected to kernel")

    # Create a test notebook
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "x = 1\ny = 2\nz = x + y",
                "metadata": {},
                "outputs": []
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "x = 1\ny = 2\nz = x + y",  # Same code - should validate
                "metadata": {},
                "outputs": []
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    print("\n[4] Executing validate_change command...")
    print("    Expected to see progress messages from kernel:")
    print()

    try:
        result = await validate_cmd.process(
            notebook_content=notebook,
            kernel_client=kernel_client,
            selected_cell_ids=["cell1"]
        )

        print()
        print("[5] Results:")
        metadata = result.get('metadata', {})
        print(f"    Status: {metadata.get('status')}")
        print(f"    Total processed: {metadata.get('total_processed')}")

        if metadata.get('results'):
            for cell_id, cell_result in metadata['results'].items():
                print(f"\n    Cell {cell_id}:")
                print(f"      OK: {cell_result.get('ok')}")
                if cell_result.get('ok'):
                    print(f"      Result: {cell_result.get('result')}")
                else:
                    print(f"      Error: {cell_result.get('error')}")

        print("\n" + "=" * 70)
        print("✓ Test completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup
        print("\n[6] Cleaning up...")
        kernel_manager.cleanup_client(kernel_id)
        await app.kernel_manager.shutdown_kernel(kernel_id)
        print("    Kernel stopped")


if __name__ == "__main__":
    asyncio.run(main())
