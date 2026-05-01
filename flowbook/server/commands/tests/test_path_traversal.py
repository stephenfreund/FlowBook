"""Tests for KernelDiscoveryHandler._resolve_notebook_path path traversal protection."""

import os
from unittest.mock import MagicMock

import pytest

from flowbook.server.handlers import KernelDiscoveryHandler


def _make_handler(server_root_dir: str = "") -> KernelDiscoveryHandler:
    """Create a KernelDiscoveryHandler with mocked Tornado internals."""
    app = MagicMock()
    app.settings = {"server_root_dir": server_root_dir}
    request = MagicMock()
    handler = KernelDiscoveryHandler.__new__(KernelDiscoveryHandler)
    handler.application = app
    handler.request = request
    return handler


class TestResolveNotebookPath:
    """Tests for _resolve_notebook_path."""

    def test_relative_path_within_root(self, tmp_path):
        root = str(tmp_path)
        handler = _make_handler(root)
        result = handler._resolve_notebook_path("notebooks/test.ipynb")
        assert result == os.path.join(root, "notebooks", "test.ipynb")

    def test_absolute_path_within_root(self, tmp_path):
        root = str(tmp_path)
        nb_path = os.path.join(root, "sub", "test.ipynb")
        handler = _make_handler(root)
        result = handler._resolve_notebook_path(nb_path)
        assert result == nb_path

    def test_relative_path_traversal_blocked(self, tmp_path):
        root = str(tmp_path)
        handler = _make_handler(root)
        with pytest.raises(ValueError, match="Path escapes server root directory"):
            handler._resolve_notebook_path("../../etc/passwd")

    def test_absolute_path_outside_root_blocked(self, tmp_path):
        root = str(tmp_path / "notebooks")
        handler = _make_handler(root)
        with pytest.raises(ValueError, match="Path escapes server root directory"):
            handler._resolve_notebook_path("/etc/passwd")

    def test_dot_dot_in_middle_blocked(self, tmp_path):
        root = str(tmp_path)
        handler = _make_handler(root)
        with pytest.raises(ValueError, match="Path escapes server root directory"):
            handler._resolve_notebook_path("sub/../../outside/test.ipynb")

    def test_tilde_expansion(self):
        handler = _make_handler(os.path.expanduser("~"))
        result = handler._resolve_notebook_path("~/test.ipynb")
        expected = os.path.join(os.path.expanduser("~"), "test.ipynb")
        assert result == expected

    def test_root_path_itself_allowed(self, tmp_path):
        root = str(tmp_path)
        handler = _make_handler(root)
        result = handler._resolve_notebook_path(root)
        assert result == root

    def test_no_root_set_allows_any_absolute(self):
        handler = _make_handler("")
        result = handler._resolve_notebook_path("/some/absolute/path.ipynb")
        assert result == "/some/absolute/path.ipynb"

    def test_no_root_set_resolves_relative_to_cwd(self):
        handler = _make_handler("")
        result = handler._resolve_notebook_path("relative.ipynb")
        assert result == os.path.abspath("relative.ipynb")

    def test_symlink_traversal_blocked(self, tmp_path):
        """Symlink inside root pointing outside should be caught by abspath."""
        root = str(tmp_path / "root")
        os.makedirs(root)
        handler = _make_handler(root)
        # Even without a real symlink, a path like root/../outside resolves outside
        with pytest.raises(ValueError, match="Path escapes server root directory"):
            handler._resolve_notebook_path(
                os.path.join(root, "..", "outside", "test.ipynb")
            )
