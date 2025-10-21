"""
Command-line interface for ferret notebook processing.
"""
import argparse
import json
import sys
from jupyter_client import KernelManager

from .registry import CommandRegistry
from .kernel_manager import FerretKernelClient


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
        default="python3",
        help="Kernel name for new kernel (default: python3)"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output file for the new notebook (default: adds _processed suffix)"
    )

    parser.add_argument(
        "--metadata-output",
        "-m",
        help="Output file for metadata JSON (default: stdout)"
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )

    args = parser.parse_args()

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
                kernel_manager.start_kernel()
                
                kernel_client = FerretKernelClient(kernel_id=kernel_manager.kernel_id)
                kernel_client.load_connection_info(kernel_manager.get_connection_info())
                kernel_client.start_channels()
                kernel_client.wait_for_ready(timeout=30)
                
                assert isinstance(kernel_client, FerretKernelClient)
                print(f"Kernel started successfully")

        result = command.process(notebook_content, kernel_client=kernel_client)

        if args.output:
            notebook_output = args.output
        else:
            base_name = args.notebook.rsplit('.', 1)[0]
            notebook_output = f"{base_name}_processed.ipynb"

        with open(notebook_output, 'w', encoding='utf-8') as f:
            json.dump(result["notebook"], f, indent=2)
        print(f"Processed notebook written to {notebook_output}")

        if args.pretty:
            metadata_output = json.dumps(result["metadata"], indent=2)
        else:
            metadata_output = json.dumps(result["metadata"])

        if args.metadata_output:
            with open(args.metadata_output, 'w', encoding='utf-8') as f:
                f.write(metadata_output)
            print(f"Metadata written to {args.metadata_output}")
        else:
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
