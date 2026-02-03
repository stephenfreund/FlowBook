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


class TestSocketFiltering:
    """Tests for internal socket file filtering."""

    def test_flowbook_socket_not_tracked(self, vfs, tmpdir):
        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        # Simulate access to flowbook internal socket
        socket_path = os.path.join(tmpdir, "flowbook_cli_1234.sock")
        abs_socket = os.path.abspath(socket_path)

        # Directly call tracking methods (socket doesn't need to exist)
        vfs._track_read(socket_path)
        vfs._track_write(socket_path)

        # Should not be tracked
        assert abs_socket not in vfs.get_read_paths()
        assert abs_socket not in vfs.get_write_paths()

    def test_flowlab_socket_not_tracked(self, vfs, tmpdir):
        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        socket_path = os.path.join(tmpdir, "flowlab_5678.sock")
        abs_socket = os.path.abspath(socket_path)

        vfs._track_read(socket_path)
        vfs._track_write(socket_path)

        assert abs_socket not in vfs.get_read_paths()
        assert abs_socket not in vfs.get_write_paths()

    def test_regular_sock_file_tracked(self, vfs, tmpdir):
        """Regular .sock files (not flowbook/flowlab) should be tracked."""
        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        socket_path = os.path.join(tmpdir, "myapp.sock")
        abs_socket = os.path.abspath(socket_path)

        vfs._track_read(socket_path)
        vfs._track_write(socket_path)

        # Regular sock files should be tracked
        assert abs_socket in vfs.get_read_paths()
        assert abs_socket in vfs.get_write_paths()

    def test_regular_files_still_tracked(self, vfs, tmpdir):
        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        regular_file = os.path.join(tmpdir, "data.txt")
        abs_file = os.path.abspath(regular_file)

        vfs._track_read(regular_file)
        vfs._track_write(regular_file)

        assert abs_file in vfs.get_read_paths()
        assert abs_file in vfs.get_write_paths()


class TestNamespacePatching:
    """Tests for namespace patching functionality."""

    def test_patch_namespace_adds_open(self, vfs):
        vfs.enable_tracking_only()

        # Create a namespace without 'open'
        namespace = {"x": 1, "y": 2}
        vfs.patch_namespace(namespace)

        assert "open" in namespace
        assert namespace["open"] is vfs._patched_open

    def test_patch_namespace_replaces_open(self, vfs):
        vfs.enable_tracking_only()

        original_open = lambda: None
        namespace = {"open": original_open, "x": 1}
        vfs.patch_namespace(namespace)

        assert namespace["open"] is vfs._patched_open
        assert namespace["open"] is not original_open

    def test_patch_namespace_idempotent(self, vfs):
        vfs.enable_tracking_only()

        namespace = {"x": 1}
        vfs.patch_namespace(namespace)
        patched = namespace["open"]

        # Patch again
        vfs.patch_namespace(namespace)

        # Should be the same patched function
        assert namespace["open"] is patched

    def test_unpatch_namespace_restores_original(self, vfs):
        vfs.enable_tracking_only()

        original_open = lambda: None
        namespace = {"open": original_open}
        vfs.patch_namespace(namespace)

        assert namespace["open"] is not original_open

        vfs._unpatch_namespaces()

        assert namespace["open"] is original_open

    def test_unpatch_namespace_removes_if_not_present(self, vfs):
        vfs.enable_tracking_only()

        namespace = {"x": 1}  # No 'open' originally
        vfs.patch_namespace(namespace)

        assert "open" in namespace

        vfs._unpatch_namespaces()

        assert "open" not in namespace

    def test_disable_unpatches_namespaces(self, vfs):
        vfs.enable_tracking_only()

        original_open = lambda: None
        namespace = {"open": original_open}
        vfs.patch_namespace(namespace)

        vfs.disable()

        assert namespace["open"] is original_open

    def test_patch_namespace_noop_when_disabled(self, vfs):
        namespace = {"x": 1}
        vfs.patch_namespace(namespace)

        # Should not add 'open' when VFS is disabled
        assert "open" not in namespace


class TestTrackingOnlyExtendedOps:
    """Tests for extended file operation tracking in tracking-only mode."""

    def test_os_remove_tracked_as_write(self, vfs, tmpdir):
        real_file = os.path.join(tmpdir, "to_delete.txt")
        with open(real_file, "w") as f:
            f.write("delete me")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.remove(real_file)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(real_file) in tracking.file_writes

    def test_os_unlink_tracked_as_write(self, vfs, tmpdir):
        real_file = os.path.join(tmpdir, "to_unlink.txt")
        with open(real_file, "w") as f:
            f.write("unlink me")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.unlink(real_file)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(real_file) in tracking.file_writes

    def test_os_mkdir_tracked_as_write(self, vfs, tmpdir):
        new_dir = os.path.join(tmpdir, "newdir")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.mkdir(new_dir)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(new_dir) in tracking.file_writes

    def test_os_makedirs_tracked_as_write(self, vfs, tmpdir):
        new_dirs = os.path.join(tmpdir, "a", "b", "c")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.makedirs(new_dirs)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(new_dirs) in tracking.file_writes

    def test_os_rmdir_tracked_as_write(self, vfs, tmpdir):
        dir_to_remove = os.path.join(tmpdir, "remove_me")
        os.mkdir(dir_to_remove)

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.rmdir(dir_to_remove)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(dir_to_remove) in tracking.file_writes

    def test_shutil_rmtree_tracked_as_write(self, vfs, tmpdir):
        dir_to_remove = os.path.join(tmpdir, "tree_to_remove")
        os.makedirs(os.path.join(dir_to_remove, "subdir"))

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        shutil.rmtree(dir_to_remove)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(dir_to_remove) in tracking.file_writes

    def test_os_rename_tracked_as_read_and_write(self, vfs, tmpdir):
        src_file = os.path.join(tmpdir, "source.txt")
        dst_file = os.path.join(tmpdir, "dest.txt")
        with open(src_file, "w") as f:
            f.write("content")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.rename(src_file, dst_file)

        tracking = vfs.get_cell_file_tracking()
        # Source is read (dependency)
        assert os.path.abspath(src_file) in tracking.file_reads_before_writes
        # Destination is write
        assert os.path.abspath(dst_file) in tracking.file_writes

    def test_shutil_copy_tracked_as_read_and_write(self, vfs, tmpdir):
        src_file = os.path.join(tmpdir, "copy_src.txt")
        dst_file = os.path.join(tmpdir, "copy_dst.txt")
        with open(src_file, "w") as f:
            f.write("copy me")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        shutil.copy(src_file, dst_file)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(src_file) in tracking.file_reads_before_writes
        assert os.path.abspath(dst_file) in tracking.file_writes

    def test_shutil_copy2_tracked_as_read_and_write(self, vfs, tmpdir):
        src_file = os.path.join(tmpdir, "copy2_src.txt")
        dst_file = os.path.join(tmpdir, "copy2_dst.txt")
        with open(src_file, "w") as f:
            f.write("copy2 me")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        shutil.copy2(src_file, dst_file)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(src_file) in tracking.file_reads_before_writes
        assert os.path.abspath(dst_file) in tracking.file_writes

    def test_shutil_move_tracked_as_read_and_write(self, vfs, tmpdir):
        src_file = os.path.join(tmpdir, "move_src.txt")
        dst_file = os.path.join(tmpdir, "move_dst.txt")
        with open(src_file, "w") as f:
            f.write("move me")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        shutil.move(src_file, dst_file)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(src_file) in tracking.file_reads_before_writes
        assert os.path.abspath(dst_file) in tracking.file_writes


class TestTrackingOnlyReadOps:
    """Tests for file read operation tracking in tracking-only mode."""

    def test_os_path_exists_tracked_as_read(self, vfs, tmpdir):
        real_file = os.path.join(tmpdir, "exists_check.txt")
        with open(real_file, "w") as f:
            f.write("exists")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.path.exists(real_file)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(real_file) in tracking.file_reads_before_writes

    def test_os_listdir_tracked_as_read(self, vfs, tmpdir):
        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.listdir(tmpdir)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(tmpdir) in tracking.file_reads_before_writes

    def test_os_stat_tracked_as_read(self, vfs, tmpdir):
        real_file = os.path.join(tmpdir, "stat_check.txt")
        with open(real_file, "w") as f:
            f.write("stat me")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.stat(real_file)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(real_file) in tracking.file_reads_before_writes

    def test_os_path_isfile_tracked_as_read(self, vfs, tmpdir):
        real_file = os.path.join(tmpdir, "isfile_check.txt")
        with open(real_file, "w") as f:
            f.write("check")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.path.isfile(real_file)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(real_file) in tracking.file_reads_before_writes

    def test_os_path_isdir_tracked_as_read(self, vfs, tmpdir):
        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.path.isdir(tmpdir)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(tmpdir) in tracking.file_reads_before_writes

    def test_os_path_getsize_tracked_as_read(self, vfs, tmpdir):
        real_file = os.path.join(tmpdir, "getsize_check.txt")
        with open(real_file, "w") as f:
            f.write("size")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.path.getsize(real_file)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(real_file) in tracking.file_reads_before_writes

    def test_os_path_getmtime_tracked_as_read(self, vfs, tmpdir):
        real_file = os.path.join(tmpdir, "getmtime_check.txt")
        with open(real_file, "w") as f:
            f.write("mtime")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.path.getmtime(real_file)

        tracking = vfs.get_cell_file_tracking()
        assert os.path.abspath(real_file) in tracking.file_reads_before_writes


class TestCellTrackingExtended:
    """Tests for per-cell tracking with extended operations."""

    def test_read_op_before_write_op_on_same_path(self, vfs, tmpdir):
        """If we check existence then delete, file is in reads_before_writes."""
        real_file = os.path.join(tmpdir, "check_then_delete.txt")
        with open(real_file, "w") as f:
            f.write("data")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        # Check existence (read) then delete (write)
        if os.path.exists(real_file):
            os.remove(real_file)

        tracking = vfs.get_cell_file_tracking()
        abs_file = os.path.abspath(real_file)
        # Should be in reads_before_writes because we read first
        assert abs_file in tracking.file_reads_before_writes
        # Also in writes
        assert abs_file in tracking.file_writes

    def test_write_op_before_read_op_on_same_path(self, vfs, tmpdir):
        """If we create then check, file is NOT in reads_before_writes."""
        new_dir = os.path.join(tmpdir, "create_then_check")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        # Create (write) then check (read)
        os.mkdir(new_dir)
        os.path.isdir(new_dir)

        tracking = vfs.get_cell_file_tracking()
        abs_dir = os.path.abspath(new_dir)
        # Should NOT be in reads_before_writes because we wrote first
        assert abs_dir not in tracking.file_reads_before_writes
        # Should be in writes
        assert abs_dir in tracking.file_writes

    def test_reset_clears_extended_tracking(self, vfs, tmpdir):
        real_file = os.path.join(tmpdir, "to_clear.txt")
        with open(real_file, "w") as f:
            f.write("data")

        vfs.enable_tracking_only()
        vfs.reset_cell_tracking()

        os.path.exists(real_file)
        os.remove(real_file)

        tracking1 = vfs.get_cell_file_tracking()
        assert len(tracking1.file_reads_before_writes) > 0
        assert len(tracking1.file_writes) > 0

        vfs.reset_cell_tracking()

        tracking2 = vfs.get_cell_file_tracking()
        assert len(tracking2.file_reads_before_writes) == 0
        assert len(tracking2.file_writes) == 0

    def test_cumulative_includes_all_operations(self, vfs, tmpdir):
        file1 = os.path.join(tmpdir, "f1.txt")
        file2 = os.path.join(tmpdir, "f2.txt")
        with open(file1, "w") as f:
            f.write("data")

        vfs.enable_tracking_only()

        # Cell 1: read file1
        vfs.reset_cell_tracking()
        with open(file1, "r") as f:
            f.read()

        # Cell 2: write file2, delete file1
        vfs.reset_cell_tracking()
        with open(file2, "w") as f:
            f.write("new")
        os.remove(file1)

        # Cumulative should have both reads and writes
        reads = vfs.get_read_paths()
        writes = vfs.get_write_paths()
        assert os.path.abspath(file1) in reads
        assert os.path.abspath(file1) in writes  # deleted
        assert os.path.abspath(file2) in writes
