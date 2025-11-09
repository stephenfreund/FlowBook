"""
Command-line interface package for DataFerret.

This package provides CLI tools for processing Jupyter notebooks:
- cli.py: General-purpose command execution
- optimize_cli.py: Cell-by-cell optimization pipeline
"""

from .cli import cli_main
from .optimize_cli import optimize_cli_main

__all__ = ['cli_main', 'optimize_cli_main']
