"""
Backward compatibility wrapper for CLI.

DEPRECATED: This module has been moved to flowbook.cli
The functionality has been refactored into a proper CLI package.

For new code, please import from:
- flowbook.cli.cli (for main CLI)
- flowbook.cli.optimize_cli (for optimization pipeline)
- flowbook.cli.helpers (for helper functions)
"""

import sys
import warnings

# Import from new location
from flowbook.cli.cli import cli_main
from flowbook.cli.helpers import (
    convert_all_source_to_strings,
    detect_file_type,
    convert_cell_indices_to_ids,
)

# Show deprecation warning when running as main module
if __name__ == "__main__":
    warnings.warn(
        "Running 'python -m flowbook.server.cli' is deprecated. "
        "Please use 'flowbook' command or 'python -m flowbook.cli.cli' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    sys.exit(cli_main())
