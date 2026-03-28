"""Tests for display_helpers.py - Kernel output formatting helpers."""

import pytest
from unittest.mock import MagicMock, patch, call

from flowbook.kernel_support.display_helpers import DisplayHelper, DEFAULT_DIV_STYLE


class TestDisplayHelper:
    """Tests for DisplayHelper class."""

    def test_default_div_style(self):
        """DisplayHelper uses default div style."""
        helper = DisplayHelper()
        assert helper.div_style == DEFAULT_DIV_STYLE

    def test_custom_div_style(self):
        """DisplayHelper can use a custom div style."""
        style = "color: red;"
        helper = DisplayHelper(div_style=style)
        assert helper.div_style == style

    @patch("flowbook.kernel_support.display_helpers.display")
    def test_display_cell_id(self, mock_display):
        """display_cell_id outputs markdown with cell ID."""
        helper = DisplayHelper()
        helper.display_cell_id("abcd")
        mock_display.assert_called_once()
        # Check the Markdown object was passed
        args = mock_display.call_args
        md_obj = args[0][0]
        assert "abcd" in md_obj.data

    @patch("flowbook.kernel_support.display_helpers.display")
    def test_display_icon_and_text_simple(self, mock_display):
        """display_icon_and_text with no contents shows simple div."""
        helper = DisplayHelper()
        helper.display_icon_and_text("info", "Hello world")
        mock_display.assert_called_once()
        args, kwargs = mock_display.call_args
        data = args[0]
        assert "text/markdown" in data
        assert "text/plain" in data
        assert "Hello world" in data["text/plain"]
        assert "Hello world" in data["text/markdown"]
        assert kwargs.get("raw") is True

    @patch("flowbook.kernel_support.display_helpers.display")
    def test_display_icon_and_text_with_contents(self, mock_display):
        """display_icon_and_text with contents creates expandable details."""
        helper = DisplayHelper()
        helper.display_icon_and_text("check", "Details", contents="some details here")
        mock_display.assert_called_once()
        args, kwargs = mock_display.call_args
        data = args[0]
        assert "<details" in data["text/markdown"]
        assert "some details here" in data["text/markdown"]

    @patch("flowbook.kernel_support.display_helpers.display")
    def test_display_icon_and_text_no_metadata_param(self, mock_display):
        """display_icon_and_text does not pass metadata (protocol uses comm/IOPub)."""
        helper = DisplayHelper()
        helper.display_icon_and_text("ok", "Done")
        mock_display.assert_called_once()
        _, kwargs = mock_display.call_args
        assert "metadata" not in kwargs
