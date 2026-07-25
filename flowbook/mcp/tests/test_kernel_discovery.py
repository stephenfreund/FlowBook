"""Tests for kernel discovery module."""

import json
import os
import stat

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


class TestWriteRefusal:
    """write_discovery must refuse writes that would break sharing."""

    def test_refuses_pid_zero(self, notebook_path, dummy_connection_file):
        result = write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", 0, "jupyterlab"
        )
        assert result is False
        assert not os.path.exists(_discovery_path(notebook_path))

    def test_refuses_negative_pid(self, notebook_path, dummy_connection_file):
        result = write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", -1, "mcp"
        )
        assert result is False
        assert not os.path.exists(_discovery_path(notebook_path))

    def test_returns_true_on_success(self, notebook_path, dummy_connection_file):
        result = write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", os.getpid(), "mcp"
        )
        assert result is True
        assert read_discovery(notebook_path) is not None


class TestNoClobber:
    """A live entry for a DIFFERENT kernel must not be overwritten."""

    def _second_connection_file(self, tmp_path):
        conn = tmp_path / "kernel-other456.json"
        conn.write_text(json.dumps({"transport": "tcp", "ip": "127.0.0.1"}))
        return str(conn)

    def test_live_different_kernel_not_clobbered(
        self, notebook_path, dummy_connection_file, tmp_path, monkeypatch
    ):
        other_conn = self._second_connection_file(tmp_path)
        # Treat the entry's pid as alive regardless of its actual value
        monkeypatch.setattr(
            "flowbook.kernel_discovery._is_pid_alive", lambda pid: True
        )
        assert write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", 12345, "mcp"
        ) is True

        # Attempt to clobber with a different connection file → refused
        result = write_discovery(
            notebook_path, other_conn, "flowbook_kernel", 67890, "jupyterlab"
        )
        assert result is False

        disc = read_discovery(notebook_path)
        assert disc is not None
        assert disc["connection_file"] == dummy_connection_file
        assert disc["started_by"] == "mcp"

    def test_same_connection_file_overwrites(
        self, notebook_path, dummy_connection_file, monkeypatch
    ):
        monkeypatch.setattr(
            "flowbook.kernel_discovery._is_pid_alive", lambda pid: True
        )
        assert write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", 12345, "mcp"
        ) is True
        # Restart/refresh: same connection file, new writer → allowed
        assert write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", 12346, "jupyterlab"
        ) is True
        disc = read_discovery(notebook_path)
        assert disc["started_by"] == "jupyterlab"
        assert disc["pid"] == 12346

    def test_dead_entry_overwritten(
        self, notebook_path, dummy_connection_file, tmp_path, monkeypatch
    ):
        other_conn = self._second_connection_file(tmp_path)
        monkeypatch.setattr(
            "flowbook.kernel_discovery._is_pid_alive", lambda pid: False
        )
        assert write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", 12345, "mcp"
        ) is True
        # Existing entry's pid is dead → overwrite allowed even with a
        # different connection file
        assert write_discovery(
            notebook_path, other_conn, "flowbook_kernel", 67890, "jupyterlab"
        ) is True

        # Read back raw (read_discovery would reject the dead pid)
        with open(_discovery_path(notebook_path)) as f:
            disc = json.load(f)
        assert disc["connection_file"] == other_conn

    def test_corrupt_existing_file_overwritten(
        self, notebook_path, dummy_connection_file
    ):
        path = _discovery_path(notebook_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("not valid json{{{")
        assert write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", os.getpid(), "mcp"
        ) is True
        assert read_discovery(notebook_path)["pid"] == os.getpid()


class TestAtomicWrite:
    def test_no_temp_files_left_behind(self, notebook_path, dummy_connection_file):
        path = _discovery_path(notebook_path)
        runtime_dir = os.path.dirname(path)
        basename = os.path.basename(path)
        before = {
            f for f in os.listdir(runtime_dir) if f.startswith(basename)
        } if os.path.isdir(runtime_dir) else set()

        assert write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", os.getpid(), "mcp"
        ) is True

        after = {f for f in os.listdir(runtime_dir) if f.startswith(basename)}
        # Only the discovery file itself — no leftover tempfiles
        assert after - before <= {basename}
        assert basename in after

    def test_written_file_is_complete_json(self, notebook_path, dummy_connection_file):
        write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", os.getpid(), "mcp"
        )
        with open(_discovery_path(notebook_path)) as f:
            doc = json.load(f)  # would raise if partially written
        assert doc["pid"] == os.getpid()


class TestFilePermissions:
    """Audit S4: discovery files must be owner-only (0o600)."""

    def test_written_file_is_0o600(self, notebook_path, dummy_connection_file):
        assert write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", os.getpid(), "mcp"
        ) is True
        mode = os.stat(_discovery_path(notebook_path)).st_mode
        assert stat.S_IMODE(mode) == 0o600


class TestIsPidAlive:
    def test_current_pid(self):
        assert _is_pid_alive(os.getpid()) is True

    def test_nonexistent_pid(self):
        assert _is_pid_alive(99999999) is False

    def test_permission_error_means_alive(self, monkeypatch):
        """Audit C10: a live process owned by another user raises
        PermissionError from os.kill(pid, 0) — that is ALIVE, not dead."""

        def raise_permission(pid, sig):
            raise PermissionError("Operation not permitted")

        monkeypatch.setattr(os, "kill", raise_permission)
        assert _is_pid_alive(12345) is True

    def test_process_lookup_error_means_dead(self, monkeypatch):
        def raise_lookup(pid, sig):
            raise ProcessLookupError("No such process")

        monkeypatch.setattr(os, "kill", raise_lookup)
        assert _is_pid_alive(12345) is False

    def test_read_discovery_keeps_other_users_kernel(
        self, notebook_path, dummy_connection_file, monkeypatch
    ):
        """A discovery file for a kernel owned by another user must not be
        auto-deleted (audit C10)."""
        assert write_discovery(
            notebook_path, dummy_connection_file, "flowbook_kernel", 12345, "mcp"
        ) is True

        def raise_permission(pid, sig):
            raise PermissionError("Operation not permitted")

        monkeypatch.setattr(os, "kill", raise_permission)
        disc = read_discovery(notebook_path)
        assert disc is not None
        assert os.path.exists(_discovery_path(notebook_path))
