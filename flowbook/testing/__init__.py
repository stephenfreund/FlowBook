"""
Reproducibility Testing Framework.

Provides standalone tools for testing reproducibility kernel correctness and performance.
"""

from flowbook.testing.notebook_loader import Cell, load_notebook
from flowbook.testing.runner import ReproducibilitySimulator, CellRecord
from flowbook.testing.correctness import run_correctness_test, CorrectnessResult
from flowbook.testing.performance import run_performance_test, PerformanceResult
from flowbook.testing.results import ResultLogger

__all__ = [
    "Cell",
    "load_notebook",
    "ReproducibilitySimulator",
    "CellRecord",
    "run_correctness_test",
    "CorrectnessResult",
    "run_performance_test",
    "PerformanceResult",
    "ResultLogger",
]
