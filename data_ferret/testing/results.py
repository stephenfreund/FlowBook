"""
Result logging utilities for SDC testing framework.

Provides JSON and CSV output for test results.
"""

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class TestConfig:
    """Configuration used for a test run."""

    n_iterations: int
    seed: Optional[int]
    notebook: str
    modifications_per_test: int = 0  # For performance tests


@dataclass
class TestSummary:
    """Summary statistics for a test run."""

    total_tests: int
    passed: int = 0
    failed: int = 0


class ResultLogger:
    """
    Logs test results to JSON and CSV files.

    Usage:
        logger = ResultLogger("./output", "correctness_test")
        for result in results:
            logger.log(result)
        json_path, csv_path = logger.save_all()
    """

    def __init__(self, output_dir: str, test_name: str, test_type: str = "correctness"):
        """
        Initialize the result logger.

        Args:
            output_dir: Directory to save output files
            test_name: Base name for output files
            test_type: Type of test ('correctness' or 'performance')
        """
        self.output_dir = Path(output_dir)
        self.test_name = test_name
        self.test_type = test_type
        self.results: List[Any] = []
        self.config: Optional[TestConfig] = None
        self.start_time: datetime = datetime.now()

    def set_config(self, config: TestConfig) -> None:
        """Set the test configuration."""
        self.config = config

    def log(self, result: Any) -> None:
        """Log a test result."""
        self.results.append(result)

    def _get_summary(self) -> TestSummary:
        """Compute summary statistics."""
        total = len(self.results)
        if self.test_type == "correctness":
            passed = sum(1 for r in self.results if getattr(r, "passed", True))
            failed = total - passed
        else:
            passed = total
            failed = 0
        return TestSummary(total_tests=total, passed=passed, failed=failed)

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        """Convert a result to a JSON-serializable dict."""
        d = asdict(result) if hasattr(result, "__dataclass_fields__") else vars(result)
        # Convert datetime objects
        for key, value in d.items():
            if isinstance(value, datetime):
                d[key] = value.isoformat()
            elif isinstance(value, set):
                d[key] = list(value)
        return d

    def save_json(self) -> str:
        """
        Save detailed JSON output.

        Returns:
            Path to saved JSON file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.test_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        data = {
            "test_type": self.test_type,
            "notebook": self.config.notebook if self.config else "unknown",
            "timestamp": self.start_time.isoformat(),
            "config": asdict(self.config) if self.config else {},
            "summary": asdict(self._get_summary()),
            "results": [self._serialize_result(r) for r in self.results],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return str(filepath)

    def save_csv(self) -> str:
        """
        Save summary CSV output.

        Returns:
            Path to saved CSV file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.test_name}_{timestamp}.csv"
        filepath = self.output_dir / filename

        if not self.results:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("# No results\n")
            return str(filepath)

        # Get field names from first result
        first_result = self._serialize_result(self.results[0])
        fieldnames = list(first_result.keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                row = self._serialize_result(result)
                # Convert lists to comma-separated strings for CSV
                for key, value in row.items():
                    if isinstance(value, list):
                        row[key] = ",".join(str(v) for v in value)
                writer.writerow(row)

        return str(filepath)

    def save_all(self) -> tuple:
        """
        Save both JSON and CSV files.

        Returns:
            Tuple of (json_path, csv_path)
        """
        return self.save_json(), self.save_csv()

    def print_summary(self) -> None:
        """Print a summary of results to stdout."""
        summary = self._get_summary()
        print(f"\n{'=' * 50}")
        print(f"Test Summary: {self.test_name}")
        print(f"{'=' * 50}")
        print(f"Test Type: {self.test_type}")
        print(f"Notebook: {self.config.notebook if self.config else 'unknown'}")
        print(f"Total Tests: {summary.total_tests}")
        if self.test_type == "correctness":
            print(f"Passed: {summary.passed}")
            print(f"Failed: {summary.failed}")
            if summary.total_tests > 0:
                pass_rate = (summary.passed / summary.total_tests) * 100
                print(f"Pass Rate: {pass_rate:.1f}%")
        print(f"{'=' * 50}\n")
