"""Tests for VirtualFileSystem."""

import os
import shutil
import tempfile

import pytest

from flowbook.kernel_support.virtual_fs import VirtualFileSystem, FileTrackingData


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp(prefix="test_vfs_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def vfs():
    v = VirtualFileSystem()
    yield v
    v.disable()


class TestVFSBasics:
    def test_default_state(self, vfs):
        assert not vfs.enabled
        assert not vfs.tracking_only

    def test_enable_disable(self, vfs):
        vfs.enable()
        assert vfs.enabled
        assert not vfs.tracking_only
        vfs.disable()
        assert not vfs.enabled

    def test_enable_tracking_only(self, vfs):
        vfs.enable_tracking_only()
        assert not vfs.enabled
        assert vfs.tracking_only
        vfs.disable()
        assert not vfs.tracking_only


class TestVFSOverlay:
    def test_write_goes_to_overlay(self, vfs, tmpdir):
        vfs.enable()
        real_file = os.path.join(tmpdir, "test.txt")

        # Write via patched open
        with open(real_file, "w") as f:
            f.write("hello overlay")

        # Real file should NOT exist
        orig_exists = vfs._originals["os.path.exists"]
        assert not orig_exists(real_file)

        # But patched exists should find it
        assert os.path.exists(real_file)

        # Read should resolve from overlay
        with open(real_file, "r") as f:
            assert f.read() == "hello overlay"

    def test_read_from_real_fs(self, vfs, tmpdir):
        real_file = os.path.join(tmpdir, "existing.txt")
        # Create real file BEFORE enabling VFS
        with open(real_file, "w") as f:
            f.write("real content")

        vfs.enable()

        # Reading should get real content
        with open(real_file, "r") as f:
            assert f.read() == "real content"

    def test_commit(self, vfs, tmpdir):
        vfs.enable()
        real_file = os.path.join(tmpdir, "commit_test.txt")

        with open(real_file, "w") as f:
            f.write("to commit")

        orig_exists = vfs._originals["os.path.exists"]
        assert not orig_exists(real_file)

        vfs.commit()

        # After commit, real file should exist
        assert orig_exists(real_file)
        orig_open = vfs._originals["builtins.open"]
        with orig_open(real_file, "r") as f:
            assert f.read() == "to commit"

    def test_rollback(self, vfs, tmpdir):
        vfs.enable()
        real_file = os.path.join(tmpdir, "rollback_test.txt")

        with open(real_file, "w") as f:
            f.write("will rollback")

        assert os.path.exists(real_file)
        vfs.rollback()
        # After rollback, overlay is cleared
        assert not os.path.exists(real_file)

    def test_delete_in_overlay(self, vfs, tmpdir):
        real_file = os.path.join(tmpdir, "delete_me.txt")
        with open(real_file, "w") as f:
            f.write("original")

        vfs.enable()
        os.remove(real_file)

        # Should appear deleted
        assert not os.path.exists(real_file)

        # Real file still exists
        orig_exists = vfs._originals["os.path.exists"]
        assert orig_exists(real_file)

    def test_listdir_merges(self, vfs, tmpdir):
        # Create real file
        with open(os.path.join(tmpdir, "real.txt"), "w") as f:
            f.write("real")

        vfs.enable()

        # Create overlay file
        with open(os.path.join(tmpdir, "overlay.txt"), "w") as f:
            f.write("overlay")

        entries = os.listdir(tmpdir)
        assert "real.txt" in entries
        assert "overlay.txt" in entries


class TestVFSTracking:
    def test_cumulative_tracking(self, vfs, tmpdir):
        real_file = os.path.join(tmpdir, "track.txt")
        with open(real_file, "w") as f:
            f.write("initial")

        vfs.enable()

        # Read
        with open(real_file, "r") as f:
            f.read()

        reads = vfs.get_read_paths()
        assert os.path.abspath(real_file) in reads

        # Write
        out_file = os.path.join(tmpdir, "output.txt")
        with open(out_file, "w") as f:
            f.write("output")

        writes = vfs.get_write_paths()
        assert os.path.abspath(out_file) in writes

    def test_cell_tracking_reads_before_writes(self, vfs, tmpdir):
        vfs.enable()

        file_a = os.path.join(tmpdir, "a.txt")
        file_b = os.path.join(tmpdir, "b.txt")

        # Create files first via overlay
        with open(file_a, "w") as f:
            f.write("a")
        with open(file_b, "w") as f:
            f.write("b")

        # Reset cell tracking
        vfs.reset_cell_tracking()

        # Read file_a then write file_b (a is rbw, b is write-only)
        with open(file_a, "r") as f:
            f.read()
        with open(file_b, "w") as f:
            f.write("new b")

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(file_a) in tracking.file_reads_before_writes
        assert os.path.abspath(file_b) in tracking.file_writes
        assert os.path.abspath(file_b) not in tracking.file_reads_before_writes

    def test_cell_tracking_write_then_read(self, vfs, tmpdir):
        vfs.enable()
        vfs.reset_cell_tracking()

        file_c = os.path.join(tmpdir, "c.txt")

        # Write first, then read — should NOT be in reads_before_writes
        with open(file_c, "w") as f:
            f.write("written first")
        with open(file_c, "r") as f:
            f.read()

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(file_c) in tracking.file_writes
        assert os.path.abspath(file_c) not in tracking.file_reads_before_writes

    def test_reset_cell_tracking_clears_per_cell(self, vfs, tmpdir):
        vfs.enable()

        file_d = os.path.join(tmpdir, "d.txt")
        with open(file_d, "w") as f:
            f.write("d")

        tracking1 = vfs.get_cell_file_tracking()
        assert os.path.abspath(file_d) in tracking1.file_writes

        vfs.reset_cell_tracking()
        tracking2 = vfs.get_cell_file_tracking()
        assert len(tracking2.file_writes) == 0
        assert len(tracking2.file_reads_before_writes) == 0

        # But cumulative tracking still has it
        assert os.path.abspath(file_d) in vfs.get_write_paths()


class TestTrackingOnlyMode:
    def test_tracking_only_records_paths(self, vfs, tmpdir):
        real_file = os.path.join(tmpdir, "track_only.txt")
        with open(real_file, "w") as f:
            f.write("initial")

        vfs.enable_tracking_only()

        # Read
        with open(real_file, "r") as f:
            f.read()
        assert os.path.abspath(real_file) in vfs.get_read_paths()

        # Write goes to real FS (no overlay)
        out_file = os.path.join(tmpdir, "track_only_out.txt")
        with open(out_file, "w") as f:
            f.write("real write")
        assert os.path.abspath(out_file) in vfs.get_write_paths()
        assert os.path.exists(out_file)  # Actually written to real FS

    def test_tracking_only_cell_tracking(self, vfs, tmpdir):
        real_file = os.path.join(tmpdir, "cell_track.txt")
        with open(real_file, "w") as f:
            f.write("data")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        with open(real_file, "r") as f:
            f.read()

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(real_file) in tracking.file_reads_before_writes
