"""Tests for cell ID validation in NotebookSession refactoring tools."""

import pytest

from flowbook.mcp.session import NotebookSession


def _make_session():
    """Create a session with a simple two-cell notebook (no kernel)."""
    session = NotebookSession()
    session.notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "id": "aaaa",
                "cell_type": "code",
                "source": "x = 1",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            },
            {
                "id": "bbbb",
                "cell_type": "code",
                "source": "y = x + 1",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            },
        ],
    }
    session.notebook_path = "/tmp/test.ipynb"
    return session


class TestCellIdValidation:
    """Verify helpful errors when cell_id is not found."""

    def test_alpha_rename_invalid_cell_id(self):
        session = _make_session()
        with pytest.raises(ValueError, match="not found in notebook"):
            session.alpha_rename("zzzz", "x", "x_renamed")

    def test_alpha_rename_lists_available_cells(self):
        session = _make_session()
        with pytest.raises(ValueError, match="aaaa.*bbbb"):
            session.alpha_rename("zzzz", "x", "x_renamed")

    def test_alpha_rename_valid_cell_id(self):
        session = _make_session()
        # Should not raise — cell exists (may not rename anything, but no ValueError)
        result = session.alpha_rename("aaaa", "x", "x_renamed")
        assert "modified_cells" in result

    def test_insert_deepcopy_invalid_cell_id(self):
        session = _make_session()
        with pytest.raises(ValueError, match="not found"):
            session.insert_deepcopy("zzzz", "x")
