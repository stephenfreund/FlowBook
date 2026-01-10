"""
Command-line interface for analyzing metadata from flowbook commands.

This tool reads one or more metadata JSON files and produces a report showing:
- Success/failure status for each file
- Command details for failures
- Summary statistics
"""

import argparse
import json
import sys
from typing import Dict, Any, List


def metadata_cli_main() -> int:
    """
    Command-line interface for analyzing metadata from flowbook commands.

    Reads one or more metadata JSON files and produces a report showing:
    - Success/failure status for each file
    - Command details for failures
    - Summary statistics

    Returns:
        0 if all files have status="success"
        1 if any file has status="error" or other non-success status
    """
    parser = argparse.ArgumentParser(
        description="Analyze metadata from flowbook commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single metadata file
  flowbook_metadata metadata.json

  # Analyze multiple metadata files
  flowbook_metadata run1_metadata.json run2_metadata.json run3_metadata.json

  # Analyze all metadata files in current directory
  flowbook_metadata *.json
        """,
    )

    parser.add_argument(
        "metadata_files",
        nargs="+",
        help="One or more metadata JSON files to analyze"
    )

    args = parser.parse_args()

    # Process each file
    results = []
    for filepath in args.metadata_files:
        result = process_metadata_file(filepath)
        results.append(result)

    # Display human-readable report
    display_report(results)

    # Exit with failure if any failed
    failed_count = sum(1 for r in results if r["status"] != "success")
    return 1 if failed_count > 0 else 0


def process_metadata_file(filepath: str) -> Dict[str, Any]:
    """
    Process a single metadata file and return result.

    Args:
        filepath: Path to the metadata JSON file

    Returns:
        Dictionary containing extracted metadata and status
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Extract key fields
        status = determine_status(metadata)
        return {
            "filepath": filepath,
            "status": status,
            "command": metadata.get("command", "unknown"),
            "notebook": metadata.get("notebook", "unknown"),
            "total_time": metadata.get("total_time", 0.0),
            "total_cost": metadata.get("total_cost", 0.0),
            "timestamp": metadata.get("timestamp", ""),
            "raw_metadata": metadata
        }
    except FileNotFoundError:
        return {
            "filepath": filepath,
            "status": "error",
            "error": "File not found"
        }
    except json.JSONDecodeError as e:
        return {
            "filepath": filepath,
            "status": "error",
            "error": f"Invalid JSON: {e}"
        }
    except Exception as e:
        return {
            "filepath": filepath,
            "status": "error",
            "error": str(e)
        }


def determine_status(metadata: Dict[str, Any]) -> str:
    """
    Determine success/failure from status field.

    Args:
        metadata: The loaded metadata dictionary

    Returns:
        Status string: 'success', 'failure', or 'unknown'

    Logic:
    1. Check top-level 'status' field first
    2. Fall back to command_metadata['status'] if available
    3. Success values: 'success', 'ok', 'completed' (case-insensitive)
    4. Failure values: 'error', 'failed', 'failure' (case-insensitive)
    5. Unknown status is treated as 'unknown'
    """
    # Check top-level status first
    status = metadata.get("status", "").lower()

    # Success values
    if status in ("success", "ok", "completed"):
        return "success"

    # Failure values
    if status in ("error", "failed", "failure"):
        return "failure"

    # Check nested command_metadata
    if "command_metadata" in metadata:
        cmd_status = metadata["command_metadata"].get("status", "").lower()
        if cmd_status in ("success", "ok", "completed"):
            return "success"
        if cmd_status in ("error", "failed", "failure"):
            return "failure"

    return "unknown"


def display_report(results: List[Dict[str, Any]]) -> None:
    """
    Display human-readable report of results.

    Args:
        results: List of result dictionaries from process_metadata_file
    """
    # Summary statistics
    total = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] in ("failure", "error"))
    unknown = sum(1 for r in results if r["status"] == "unknown")

    print("=" * 60)
    print("METADATA ANALYSIS REPORT")
    print("=" * 60)
    print(f"Files Processed: {total}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Unknown: {unknown}")
    print()

    # Overall status
    if failed > 0 or unknown > 0:
        print("Overall Status: FAILED")
    else:
        print("Overall Status: SUCCESS")
    print()
    print("=" * 60)
    print("FILE RESULTS")
    print("=" * 60)
    print()

    # Individual file results
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['filepath']}")
        print(f"   Status: {result['status'].upper()}")

        if result["status"] == "success":
            if result.get('notebook'):
                print(f"   Notebook: {result['notebook']}")
            print(f"   Command: {result.get('command', 'unknown')}")
            print(f"   Time: {result.get('total_time', 0):.2f}s")
            print(f"   Cost: ${result.get('total_cost', 0):.4f}")
            if result.get('timestamp'):
                print(f"   Timestamp: {result['timestamp']}")
        else:
            # Show details for failures
            if result.get('notebook'):
                print(f"   Notebook: {result['notebook']}")
            print(f"   Error: {result.get('error', 'Unknown error')}")

            # If we have command_metadata with details, show them
            raw_metadata = result.get("raw_metadata", {})
            if "command_metadata" in raw_metadata:
                cmd_metadata = raw_metadata["command_metadata"]

                # Show error details if present
                if "error" in cmd_metadata:
                    print(f"   Details: {cmd_metadata['error']}")
                elif "message" in cmd_metadata:
                    print(f"   Details: {cmd_metadata['message']}")
                # For validation failures, show validation details
                elif "validation" in cmd_metadata:
                    validation = cmd_metadata["validation"]
                    if "issues" in validation and validation["issues"]:
                        print(f"   Issues: {len(validation['issues'])} validation issue(s)")
                        for issue in validation["issues"][:3]:  # Show first 3
                            print(f"     - {issue}")

        print()

    print("=" * 60)


if __name__ == "__main__":
    sys.exit(metadata_cli_main())
