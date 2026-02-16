"""Tests for install.py - Kernel spec installation utility."""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, mock_open, call

from flowbook.kernel_support.install import install_kernel


class TestInstallKernel:
    """Tests for install_kernel function."""

    @patch("flowbook.kernel_support.install.KernelSpecManager")
    @patch("builtins.open", new_callable=mock_open)
    @patch("flowbook.kernel_support.install.json")
    def test_install_kernel_basic(self, mock_json, mock_file, mock_ksm_class):
        """install_kernel installs kernel spec and updates argv."""
        # Set up mock KernelSpecManager
        mock_ksm = MagicMock()
        mock_ksm.install_kernel_spec.return_value = "/fake/dest/kernel_name"
        mock_ksm_class.return_value = mock_ksm

        # Set up mock json.load to return a kernel.json with argv
        mock_json.load.return_value = {"argv": ["/old/python", "-m", "some_module"]}

        result = install_kernel("/fake/package", "my_kernel")

        # Verify kernel spec was installed
        mock_ksm.install_kernel_spec.assert_called_once_with(
            "/fake/package/kernelspec",
            kernel_name="my_kernel",
            user=True,
            replace=True,
        )

        # Verify kernel.json was read and written
        assert mock_file.call_count == 2  # open for read, open for write

        # Verify argv[0] was updated to current python
        written_spec = mock_json.dump.call_args[0][0]
        assert written_spec["argv"][0] == sys.executable

        # Verify return value
        assert result == "/fake/dest/kernel_name"

    @patch("flowbook.kernel_support.install.KernelSpecManager")
    @patch("builtins.open", new_callable=mock_open)
    @patch("flowbook.kernel_support.install.json")
    def test_install_kernel_updates_python_path(self, mock_json, mock_file, mock_ksm_class):
        """install_kernel replaces argv[0] with the current sys.executable."""
        mock_ksm = MagicMock()
        mock_ksm.install_kernel_spec.return_value = "/dest"
        mock_ksm_class.return_value = mock_ksm

        mock_json.load.return_value = {"argv": ["/usr/bin/python3", "-m", "kernel"]}

        install_kernel("/pkg", "test_kernel")

        written_spec = mock_json.dump.call_args[0][0]
        assert written_spec["argv"][0] == sys.executable
        assert written_spec["argv"][1] == "-m"
        assert written_spec["argv"][2] == "kernel"
