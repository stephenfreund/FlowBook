"""
Tests for metadata formatting.
"""

import pytest
from data_ferret.cli.helpers import format_metadata


def test_format_simple_metadata():
    """Test formatting simple metadata."""
    metadata = {
        "status": "success",
        "command": "test",
        "message": "Test message"
    }

    result = format_metadata(metadata)

    assert "Status: success" in result
    assert "Command: test" in result
    assert "Message: Test message" in result


def test_format_metadata_with_nested_dict():
    """Test formatting metadata with nested dictionaries."""
    metadata = {
        "status": "success",
        "command": "prepare_code",
        "transformations": {
            "cells_modified": 2,
            "total_transformations": 3
        }
    }

    result = format_metadata(metadata)

    assert "Status: success" in result
    assert "Transformations:" in result
    assert "Cells Modified: 2" in result
    assert "Total Transformations: 3" in result


def test_format_metadata_with_list():
    """Test formatting metadata with lists."""
    metadata = {
        "status": "success",
        "command": "validate",
        "validation": {
            "issues": ["Issue 1", "Issue 2"],
            "warnings": []
        }
    }

    result = format_metadata(metadata)

    assert "Issues:" in result
    assert "- Issue 1" in result
    assert "- Issue 2" in result
    assert "Warnings:" in result
    assert "(none)" in result


def test_format_metadata_with_list_of_dicts():
    """Test formatting metadata with list of dictionaries."""
    metadata = {
        "status": "success",
        "command": "prepare_code",
        "transformations": {
            "summary": [
                {
                    "cell_id": "abc",
                    "count": 1
                },
                {
                    "cell_id": "def",
                    "count": 2
                }
            ]
        }
    }

    result = format_metadata(metadata)

    assert "Summary:" in result
    # YAML-style formatting with dash
    assert "- Cell Id: abc" in result
    assert "- Cell Id: def" in result
    assert "Count: 1" in result
    assert "Count: 2" in result


def test_format_metadata_underscore_to_title_case():
    """Test that underscores are converted to title case."""
    metadata = {
        "status": "success",
        "command": "test",
        "some_field_name": "value",
        "nested": {
            "another_field": 123
        }
    }

    result = format_metadata(metadata)

    assert "Some Field Name:" in result
    assert "Another Field: 123" in result
