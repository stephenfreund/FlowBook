"""
Test the --downsample-csv option for execute_all and execute_sdc commands.

Tests both:
1. The patch logic in isolation (unit tests)
2. The patch applied in a real kernel (integration tests)
"""

import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

from flowbook.server.kernel_helper import CSV_DOWNSAMPLE_PATCH_TEMPLATE


def create_test_notebook(csv_path: str) -> dict:
    """Create a minimal notebook that reads a CSV and prints its shape."""
    return {
        "cells": [
            {
                "id": "aaaa",
                "cell_type": "code",
                "source": dedent(f'''
                    import pandas as pd
                    df = pd.read_csv("{csv_path}")
                    print(f"Shape: {{df.shape}}")
                    print(f"Rows: {{len(df)}}")
                    len(df)
                ''').strip(),
                "metadata": {},
                "outputs": [],
                "execution_count": None
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def create_csv_file(path: Path, num_rows: int = 100) -> None:
    """Create a simple CSV file with the given number of rows."""
    with open(path, "w") as f:
        f.write("a,b,c\n")
        for i in range(num_rows):
            f.write(f"{i},{i*2},{i*3}\n")


class TestDownsampleCSVPatch:
    """Test the downsampling logic in isolation (without kernel)."""

    def test_patch_code_generation(self):
        """Test that the patch code template is generated correctly."""
        proportion = 0.1
        patch_code = CSV_DOWNSAMPLE_PATCH_TEMPLATE.format(proportion=proportion)

        # Verify the patch code contains key elements
        assert "_original_pd_read_csv = pd.read_csv" in patch_code
        assert "0.1" in patch_code
        assert "df.head(n_rows)" in patch_code
        # Verify cuDF support is included
        assert "import cudf" in patch_code
        assert "_original_cudf_read_csv" in patch_code

    def test_downsample_logic_directly(self):
        """Test the downsampling calculation directly."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            create_csv_file(csv_path, num_rows=100)

            # Read full file
            df_full = pd.read_csv(csv_path)
            assert len(df_full) == 100

            # Simulate the downsampling logic
            downsample_csv = 0.1
            n_rows = int(len(df_full) * downsample_csv)
            df_downsampled = df_full.head(n_rows)

            assert len(df_downsampled) == 10
            assert n_rows == 10

    def test_downsample_edge_cases(self):
        """Test edge cases for downsampling."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            create_csv_file(csv_path, num_rows=100)

            df = pd.read_csv(csv_path)

            # Test 50%
            n_rows = int(len(df) * 0.5)
            assert n_rows == 50

            # Test 1% (should give 1 row)
            n_rows = int(len(df) * 0.01)
            assert n_rows == 1

            # Test 0% (should give 0 rows)
            n_rows = int(len(df) * 0.0)
            assert n_rows == 0

            # Test 100%
            n_rows = int(len(df) * 1.0)
            assert n_rows == 100

    def test_patch_applies_correctly(self):
        """Test that the monkey-patch actually works when applied."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            create_csv_file(csv_path, num_rows=100)

            # Store original
            original_read_csv = pd.read_csv

            try:
                # Apply the patch
                downsample_proportion = 0.1
                _original_read_csv = pd.read_csv

                def _downsampled_read_csv(*args, **kwargs):
                    df = _original_read_csv(*args, **kwargs)
                    n_rows = int(len(df) * downsample_proportion)
                    return df.head(n_rows)

                pd.read_csv = _downsampled_read_csv

                # Now read the CSV - should be downsampled
                df = pd.read_csv(str(csv_path))
                assert len(df) == 10, f"Expected 10 rows (10% of 100), got {len(df)}"

                # Verify it's the first 10 rows
                assert df['a'].tolist() == list(range(10))

            finally:
                # Restore original
                pd.read_csv = original_read_csv

                # Verify restoration
                df = pd.read_csv(str(csv_path))
                assert len(df) == 100


class TestDownsampleCSVKernel:
    """Integration tests that verify the patch works in a real kernel."""

    def test_patch_works_in_kernel(self):
        """Test that the downsample patch works when applied in the kernel."""
        from flowbook.cli.helpers import setup_kernel, cleanup_kernel
        from flowbook.server.kernel_helper import KernelHelper

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            create_csv_file(csv_path, num_rows=100)

            kernel_manager, kernel_client = setup_kernel(kernel_name="flowbook_kernel")
            try:
                # Step 1: Read CSV without patch - should get 100 rows
                result = KernelHelper.execute_code(
                    kernel_client,
                    f"""
import pandas as pd
df = pd.read_csv("{csv_path}")
len(df)
""",
                )
                # Find the execute_result
                for output in result["outputs"]:
                    if output.get("output_type") == "execute_result":
                        assert output["data"]["text/plain"] == "100", \
                            f"Before patch: expected 100 rows, got {output['data']['text/plain']}"
                        break
                else:
                    pytest.fail("No execute_result found for before patch check")

                # Step 2: Apply the patch
                patch_code = CSV_DOWNSAMPLE_PATCH_TEMPLATE.format(proportion=0.1)
                result = KernelHelper.execute_code(kernel_client, patch_code, store_history=False)

                # Verify patch was applied
                stream_text = ""
                for output in result["outputs"]:
                    if output.get("output_type") == "stream":
                        stream_text += output.get("text", "")

                assert "downsampling enabled" in stream_text.lower(), \
                    f"Expected confirmation message, got: {stream_text}"

                # Step 3: Read CSV after patch - should get 10 rows
                result = KernelHelper.execute_code(
                    kernel_client,
                    f"""
df2 = pd.read_csv("{csv_path}")
len(df2)
""",
                )
                # Find the execute_result
                for output in result["outputs"]:
                    if output.get("output_type") == "execute_result":
                        assert output["data"]["text/plain"] == "10", \
                            f"After patch: expected 10 rows, got {output['data']['text/plain']}"
                        break
                else:
                    pytest.fail("No execute_result found for after patch check")

            finally:
                cleanup_kernel(kernel_client, kernel_manager)

    def test_downsample_different_proportions(self):
        """Test different downsampling proportions."""
        from flowbook.cli.helpers import setup_kernel, cleanup_kernel
        from flowbook.server.kernel_helper import KernelHelper

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            create_csv_file(csv_path, num_rows=1000)

            kernel_manager, kernel_client = setup_kernel(kernel_name="flowbook_kernel")
            try:
                # Apply 5% patch
                patch_code = CSV_DOWNSAMPLE_PATCH_TEMPLATE.format(proportion=0.05)
                KernelHelper.execute_code(kernel_client, patch_code, store_history=False)

                result = KernelHelper.execute_code(
                    kernel_client,
                    f"""
import pandas as pd
df = pd.read_csv("{csv_path}")
len(df)
""",
                )
                for output in result["outputs"]:
                    if output.get("output_type") == "execute_result":
                        assert output["data"]["text/plain"] == "50", \
                            f"Expected 50 rows (5% of 1000), got {output['data']['text/plain']}"
                        break
            finally:
                cleanup_kernel(kernel_client, kernel_manager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
