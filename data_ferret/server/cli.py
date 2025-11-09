"""
Backward compatibility wrapper for CLI.

DEPRECATED: This module has been moved to data_ferret.cli
The functionality has been refactored into a proper CLI package.

For new code, please import from:
- data_ferret.cli.cli (for main CLI)
- data_ferret.cli.optimize_cli (for optimization pipeline)
- data_ferret.cli.helpers (for helper functions)
"""

import sys
import warnings

# Import from new location
from data_ferret.cli.cli import cli_main
from data_ferret.cli.helpers import (
    convert_all_source_to_strings,
    detect_file_type,
    convert_cell_indices_to_ids,
)

# Show deprecation warning when running as main module
if __name__ == "__main__":
    warnings.warn(
        "Running 'python -m data_ferret.server.cli' is deprecated. "
        "Please use 'data_ferret' command or 'python -m data_ferret.cli.cli' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    sys.exit(cli_main())
