"""Tests for kernel discovery module."""

import json
import os

import pytest

from flowbook.kernel_discovery import (
    _discovery_path,
    _is_pid_alive,
    read_discovery,
    remove_discovery,
    write_discovery,
)


@pytest.fixture
def dummy_connection_file(tmp_path):
    """Create a dummy kernel connection file."""
    conn_file = tmp_path / "kernel-test123.json"
    conn_file.write_text(json.dumps({"transport": "tcp", "ip": "127.0.0.1"}))
    return str(conn_file)


@pytest.fixture
def notebook_path():
    return "/tmp/test_discovery_notebook.ipynb"


@pytest.fixture(autouse=True)
def cleanup_discovery(notebook_path):
    """Remove discovery file after each test."""
    yield
    remove_discovery(notebook_path)


class TestDiscoveryPath:
    def test_deterministic(self):
        p1 = _discovery_path("/foo/bar.ipynb")
        p2 = _discovery_path("/foo/bar.ipynb")
        assert p1 == p2

    def test_different_paths_different_hashes(self):
        p1 = _discovery_path("/foo/bar.ipynb")
        p2 = _discovery_path("/foo/baz.ipynb")
        assert p1 != p2

    def test_in_jupyter_runtime_dir(self):
        from jupyter_core.paths import jupyter_runtime_dir

        p = _discovery_path("/foo/bar.ipynb")
        assert p.startswith(jupyter_runtime_dir())

    def test_filename_format(self):
        p = _discovery_path("/foo/bar.ipynb")
        basename = os.path.basename(p)
        assert basename.startswith("flowbook-")
        assert basename.endswith(".json")


class TestWriteAndRead:
    def test_write_and_read_back(self, notebook_path, dummy_connection_file):
        write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", os.getpid(), "mcp"
        )
        result = read_discovery(notebook_path)
        assert result is not None
        assert result["notebook_path"] == notebook_path
        assert result["connection_file"] == dummy_connection_file
        assert result["kernel_name"] == "flowbook_kernel"
        assert result["pid"] == os.getpid()
        assert result["started_by"] == "mcp"

    def test_started_at_is_set(self, notebook_path, dummy_connection_file):
        write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", os.getpid(), "mcp"
        )
        result = read_discovery(notebook_path)
        assert "started_at" in result
        assert isinstance(result["started_at"], float)

    def test_overwrite(self, notebook_path, dummy_connection_file):
        write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", os.getpid(), "mcp"
        )
        write_discovery(
            notebook_path,
            dummy_connection_file,
            "flowbook_kernel",
            os.getpid(),
            "jupyterlab",
        )
        result = read_discovery(notebook_path)
        assert result["started_by"] == "jupyterlab"


class TestRemove:
    def test_remove_existing(self, notebook_path, dummy_connection_file):
        write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", os.getpid(), "mcp"
        )
        remove_discovery(notebook_path)
        assert read_discovery(notebook_path) is None

    def test_remove_nonexistent(self, notebook_path):
        # Should not raise
        remove_discovery(notebook_path)


class TestStalenessValidation:
    def test_stale_pid(self, notebook_path, dummy_connection_file):
        """Dead PID should be auto-cleaned."""
        write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", 99999999, "mcp"
        )
        result = read_discovery(notebook_path)
        assert result is None
        # File should have been cleaned up
        assert not os.path.exists(_discovery_path(notebook_path))

    def test_missing_connection_file(self, notebook_path):
        """Missing connection file should be auto-cleaned."""
        write_discovery(
            notebook_path,
            "/nonexistent/kernel.json",
            "flowbook_kernel",
            os.getpid(),
            "mcp",
        )
        result = read_discovery(notebook_path)
        assert result is None

    def test_no_file(self, notebook_path):
        """No discovery file should return None."""
        result = read_discovery(notebook_path)
        assert result is None

    def test_corrupt_json(self, notebook_path):
        """Corrupt JSON should be auto-cleaned."""
        path = _discovery_path(notebook_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("not valid json{{{")
        result = read_discovery(notebook_path)
        assert result is None


class TestIsPidAlive:
    def test_current_pid(self):
        assert _is_pid_alive(os.getpid()) is True

    def test_nonexistent_pid(self):
        assert _is_pid_alive(99999999) is False
