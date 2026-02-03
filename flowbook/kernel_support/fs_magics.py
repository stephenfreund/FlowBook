"""
FileSystemMagics - Magic commands for VFS and file checkpointing.

Registered as a separate @magics_class to avoid MRO issues with kernels
that already inherit from Magics.
"""

from IPython.core.magic import Magics, line_magic, magics_class


@magics_class
class FileSystemMagics(Magics):
    """Magic commands for virtual filesystem and file checkpointing."""

    def __init__(self, shell, kernel):
        super().__init__(shell)
        self._kernel = kernel

    @line_magic
    def virtual_fs(self, line):
        """
        Toggle virtual filesystem.

        Usage:
            %virtual_fs         - Show current status
            %virtual_fs on      - Enable VFS (writes go to overlay)
            %virtual_fs off     - Disable VFS (discard overlay)
            %virtual_fs ?       - Show current status
        """
        arg = line.strip().lower()
        vfs = self._kernel._vfs
        display = self._kernel._display

        if not arg or arg == "?":
            if vfs.enabled:
                status = "ON (full VFS mode)"
            elif vfs.tracking_only:
                status = "ON (tracking-only mode)"
            else:
                status = "OFF"
            display.display_icon_and_text("info", f"Virtual filesystem: {status}")
            return

        if arg in ("on", "true", "1", "enable"):
            vfs.enable()
            display.display_icon_and_text("ok", "Virtual filesystem enabled")
        elif arg in ("off", "false", "0", "disable"):
            vfs.disable()
            display.display_icon_and_text("ok", "Virtual filesystem disabled")
        else:
            display.display_icon_and_text("error", f"Invalid: '{arg}'. Use 'on', 'off', or '?'")

    @line_magic
    def file_checkpoints(self, line):
        """
        Toggle file checkpointing.

        Usage:
            %file_checkpoints         - Show current status
            %file_checkpoints on      - Enable file checkpointing
            %file_checkpoints off     - Disable file checkpointing
            %file_checkpoints ?       - Show current status
        """
        arg = line.strip().lower()
        file_cp = self._kernel._checkpoints.file
        display = self._kernel._display

        if not arg or arg == "?":
            status = "ON" if file_cp._enabled else "OFF"
            display.display_icon_and_text("info", f"File checkpoints: {status}")
            return

        if arg in ("on", "true", "1", "enable"):
            file_cp.enable()
            # Also enable tracking-only VFS if VFS is not already enabled
            vfs = self._kernel._vfs
            if not vfs.enabled and not vfs.tracking_only:
                vfs.enable_tracking_only()
            display.display_icon_and_text("ok", "File checkpoints enabled")
        elif arg in ("off", "false", "0", "disable"):
            file_cp.disable()
            display.display_icon_and_text("ok", "File checkpoints disabled")
        else:
            display.display_icon_and_text("error", f"Invalid: '{arg}'. Use 'on', 'off', or '?'")

    @line_magic
    def commit_files(self, line):
        """
        Apply VFS overlay to real filesystem.

        Usage:
            %commit_files  - Write overlay files to real FS and clear overlay
        """
        vfs = self._kernel._vfs
        display = self._kernel._display

        if not vfs.enabled:
            display.display_icon_and_text("error", "Virtual filesystem is not enabled")
            return

        write_count = len(vfs.get_write_paths())
        vfs.commit()
        display.display_icon_and_text("ok", f"Committed {write_count} file(s) to real filesystem")
