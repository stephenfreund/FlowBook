"""
Command-line interface for ferret notebook processing.
"""
import argparse
import json
import sys
import asyncio
from jupyter_client import KernelManager

from .registry import CommandRegistry
from .kernel_manager import FerretKernelClient
from .config import FerretConfig


def cli_main():
    """Command-line interface for the ferret command processor."""
    parser = argparse.ArgumentParser(
        description="Process Jupyter notebooks with ferret commands"
    )

    registry = CommandRegistry()

    parser.add_argument(
        "command",
        choices=registry.list_commands(),
        help="Command to execute"
    )

    parser.add_argument(
        "notebook",
        help="Path to the Jupyter notebook file"
    )

    parser.add_argument(
        "--kernel-id",
        "-k",
        help="ID of running kernel to connect to"
    )

    parser.add_argument(
        "--kernel-name",
        default="ferret_kernel",
        help="Kernel name for new kernel (default: ferret_kernel)"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output file for the new notebook (default: adds _processed suffix)"
    )

    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="AI model to use for commands (default: gpt-4o)"
    )

    parser.add_argument(
        "--fast-model",
        default="gpt-4o-mini",
        help="Fast AI model to use for lightweight operations (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--cell-ids",
        "-c",
        nargs="+",
        help="Optional list of cell IDs to process (default: process all cells)"
    )

    args = parser.parse_args()

    # Create config from CLI arguments with same defaults as Jupyter
    config = FerretConfig(
        model=args.model,
        fast_model=args.fast_model
    )

    kernel_manager = None
    kernel_client = None

    try:
        with open(args.notebook, 'r', encoding='utf-8') as f:
            notebook_content = json.load(f)

        command = registry.get_command(args.command)

        if command.requires_kernel:
            if args.kernel_id:
                print(f"Connecting to kernel: {args.kernel_id}")
                raise NotImplementedError("Connecting to existing kernel by ID not yet implemented in CLI")
            else:
                print(f"Starting new kernel: {args.kernel_name}")
                # Start kernel manager and create our custom FerretKernelClient
                kernel_manager = KernelManager(kernel_name=args.kernel_name)
                try:
                    kernel_manager.start_kernel()
                except Exception as e:
                    print(f"Error starting kernel: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
                    return 1

                kernel_client = FerretKernelClient(kernel_id=kernel_manager.kernel_id)
                kernel_client.load_connection_info(kernel_manager.get_connection_info())
                kernel_client.start_channels()
                try:
                    kernel_client.wait_for_ready(timeout=30)
                except Exception as e:
                    print(f"Error waiting for kernel to be ready: {e}", file=sys.stderr)
                    # Try to read kernel stderr/stdout for more details
                    if kernel_manager.is_alive():
                        print("Kernel is still running but not responding", file=sys.stderr)
                    else:
                        print("Kernel has died", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
                    return 1

                assert isinstance(kernel_client, FerretKernelClient)
                print(f"Kernel started successfully")

        # Run async command.process() in event loop
        result = asyncio.run(command.process(
            notebook_content,
            kernel_client=kernel_client,
            selected_cell_ids=args.cell_ids,
            config=config
        ))

        if args.output:
            notebook_output = args.output
        else:
            base_name = args.notebook.rsplit('.', 1)[0]
            notebook_output = f"{base_name}_processed.ipynb"

        with open(notebook_output, 'w', encoding='utf-8') as f:
            json.dump(result["notebook"], f, indent=2)
        print(f"Processed notebook written to {notebook_output}")

        metadata_output = json.dumps(result["metadata"], indent=2)

        print("\nMetadata:")
        print(metadata_output)

        return 0

    except FileNotFoundError:
        print(f"Error: Notebook file not found: {args.notebook}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in notebook: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if kernel_client:
            kernel_client.stop_channels()
        if kernel_manager:
            kernel_manager.shutdown_kernel()


if __name__ == "__main__":
    sys.exit(cli_main())
