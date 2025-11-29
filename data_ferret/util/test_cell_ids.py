"""Unit tests for cell ID generation and notebook normalization."""

import pytest
from data_ferret.util.cell_ids import generate_cell_id, normalize_notebook


class TestGenerateCellId:
    """Tests for generate_cell_id function."""

    def test_generates_four_character_id(self):
        """Test that generated ID is 4 characters."""
        cell_id = generate_cell_id(set())
        assert len(cell_id) == 4
        assert cell_id.isalpha()
        assert cell_id.islower()

    def test_generates_unique_id(self):
        """Test that generated ID is not in existing set."""
        existing = {"abcdef", "ghijkl", "mnopqr"}
        cell_id = generate_cell_id(existing)
        assert cell_id not in existing

    def test_generates_different_ids(self):
        """Test that multiple calls generate different IDs."""
        existing = set()
        ids = []
        for _ in range(100):
            cell_id = generate_cell_id(existing)
            ids.append(cell_id)
            existing.add(cell_id)

        # All should be unique
        assert len(ids) == len(set(ids))

    def test_avoids_existing_ids(self):
        """Test that generated IDs avoid a large set of existing IDs."""
        # Create 1000 existing IDs
        existing = set()
        for _ in range(1000):
            existing.add(generate_cell_id(existing))

        # Generate 100 more - should all be unique
        for _ in range(100):
            cell_id = generate_cell_id(existing)
            assert cell_id not in existing
            existing.add(cell_id)


class TestNormalizeNotebook:
    """Tests for normalize_notebook function."""

    def test_adds_ids_to_cells_without_them(self):
        """Test that cells without IDs receive new IDs."""
        notebook = {
            "cells": [
                {"cell_type": "code", "source": "print('hello')"},
                {"cell_type": "markdown", "source": "# Title"},
            ]
        }

        result = normalize_notebook(notebook)

        assert result["cells"][0]["id"] is not None
        assert result["cells"][1]["id"] is not None
        assert len(result["cells"][0]["id"]) == 4
        assert len(result["cells"][1]["id"]) == 4

    def test_preserves_existing_four_char_ids(self):
        """Test that cells with valid 4-char IDs keep their IDs."""
        notebook = {
            "cells": [
                {"id": "abcd", "cell_type": "code", "source": "x = 1"},
                {"id": "efgh", "cell_type": "code", "source": "y = 2"},
            ]
        }

        result = normalize_notebook(notebook)

        assert result["cells"][0]["id"] == "abcd"
        assert result["cells"][1]["id"] == "efgh"

    def test_replaces_non_four_char_ids(self):
        """Test that non-4-character IDs are replaced."""
        notebook = {
            "cells": [
                {"id": "cell01", "cell_type": "code", "source": "x = 1"},  # 6 chars
                {"id": "ab", "cell_type": "code", "source": "y = 2"},     # 2 chars
                {"id": "toolong123", "cell_type": "code", "source": "z = 3"},  # 10 chars
                {"id": "ABC1", "cell_type": "code", "source": "a = 4"},  # has uppercase and number
            ]
        }

        result = normalize_notebook(notebook)

        # All should have 4-character lowercase IDs now
        for cell in result["cells"]:
            assert len(cell["id"]) == 4
            assert cell["id"].isalpha()
            assert cell["id"].islower()

        # All IDs should be unique
        ids = [cell["id"] for cell in result["cells"]]
        assert len(ids) == len(set(ids))

    def test_handles_duplicate_ids(self):
        """Test that duplicate IDs are regenerated."""
        notebook = {
            "cells": [
                {"id": "same", "cell_type": "code", "source": "x = 1"},
                {"id": "same", "cell_type": "code", "source": "y = 2"},
                {"id": "same", "cell_type": "code", "source": "z = 3"},
            ]
        }

        result = normalize_notebook(notebook)

        # All IDs should be unique now
        ids = [cell["id"] for cell in result["cells"]]
        assert len(ids) == len(set(ids))

    def test_converts_list_source_to_string(self):
        """Test that list sources are converted to strings."""
        notebook = {
            "cells": [
                {
                    "id": "cell01",
                    "cell_type": "code",
                    "source": ["line 1\n", "line 2\n", "line 3"],
                }
            ]
        }

        result = normalize_notebook(notebook)

        assert isinstance(result["cells"][0]["source"], str)
        assert result["cells"][0]["source"] == "line 1\nline 2\nline 3"

    def test_preserves_string_source(self):
        """Test that string sources are unchanged."""
        notebook = {
            "cells": [
                {"id": "cell01", "cell_type": "code", "source": "print('hello')"}
            ]
        }

        result = normalize_notebook(notebook)

        assert result["cells"][0]["source"] == "print('hello')"

    def test_returns_original_if_no_changes_needed(self):
        """Test that original notebook is returned if already normalized."""
        notebook = {
            "cells": [
                {"id": "abcd", "cell_type": "code", "source": "x = 1"},
                {"id": "efgh", "cell_type": "code", "source": "y = 2"},
            ]
        }

        result = normalize_notebook(notebook)

        # Should be the exact same object
        assert result is notebook

    def test_does_not_mutate_input(self):
        """Test that input notebook is not modified."""
        notebook = {
            "cells": [
                {"cell_type": "code", "source": ["line 1\n", "line 2"]},
                {"id": "cell02", "cell_type": "code", "source": "y = 2"},
            ]
        }

        # Keep reference to original cell
        original_cell = notebook["cells"][0]

        result = normalize_notebook(notebook)

        # Original cell should still not have ID and still have list source
        assert "id" not in original_cell
        assert isinstance(original_cell["source"], list)

    def test_handles_empty_notebook(self):
        """Test that empty notebooks are handled correctly."""
        notebook = {"cells": []}

        result = normalize_notebook(notebook)

        assert result == notebook
        assert result is notebook

    def test_handles_notebook_without_cells_key(self):
        """Test that notebooks without cells key are handled."""
        notebook = {"metadata": {}}

        result = normalize_notebook(notebook)

        assert result == notebook
        assert result is notebook

    def test_mixed_cells_with_and_without_ids(self):
        """Test handling of mixed cells (some with IDs, some without)."""
        notebook = {
            "cells": [
                {"id": "abcd", "cell_type": "code", "source": "x = 1"},
                {"cell_type": "code", "source": "y = 2"},
                {"id": "efgh", "cell_type": "markdown", "source": "# Title"},
                {"cell_type": "code", "source": "z = 3"},
            ]
        }

        result = normalize_notebook(notebook)

        # First and third cells keep their valid 4-char IDs
        assert result["cells"][0]["id"] == "abcd"
        assert result["cells"][2]["id"] == "efgh"

        # Second and fourth cells get new IDs
        assert result["cells"][1]["id"] is not None
        assert len(result["cells"][1]["id"]) == 4
        assert result["cells"][3]["id"] is not None
        assert len(result["cells"][3]["id"]) == 4

        # All IDs are unique
        ids = [cell["id"] for cell in result["cells"]]
        assert len(ids) == len(set(ids))

    def test_complex_normalization(self):
        """Test complex case with duplicates, missing IDs, and list sources."""
        notebook = {
            "cells": [
                {
                    "id": "abcd",  # Valid 4-char ID but will be duplicate
                    "cell_type": "code",
                    "source": ["x = 1\n", "y = 2"],
                },
                {"id": "abcd", "cell_type": "code", "source": "z = 3"},  # Duplicate
                {"cell_type": "markdown", "source": ["# Title\n", "## Subtitle"]},  # No ID
                {"id": "efgh", "cell_type": "code", "source": "a = 4"},  # Valid unique 4-char ID
            ]
        }

        result = normalize_notebook(notebook)

        # All IDs should be unique and 4 characters
        ids = [cell["id"] for cell in result["cells"]]
        assert len(ids) == len(set(ids))
        assert len(ids) == 4
        for cell_id in ids:
            assert len(cell_id) == 4
            assert cell_id.isalpha()
            assert cell_id.islower()

        # All sources should be strings
        for cell in result["cells"]:
            assert isinstance(cell["source"], str)

        # The unique valid ID should be preserved
        assert result["cells"][3]["id"] == "efgh"

        # Original notebook should be unchanged
        assert isinstance(notebook["cells"][0]["source"], list)
        assert notebook["cells"][0]["id"] == "abcd"
        assert notebook["cells"][1]["id"] == "abcd"

    def test_preserves_other_cell_properties(self):
        """Test that other cell properties are preserved."""
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "x = 1",
                    "execution_count": 5,
                    "outputs": [],
                    "metadata": {"custom": "value"},
                }
            ]
        }

        result = normalize_notebook(notebook)

        cell = result["cells"][0]
        assert cell["cell_type"] == "code"
        assert cell["execution_count"] == 5
        assert cell["outputs"] == []
        assert cell["metadata"] == {"custom": "value"}

    def test_preserves_notebook_metadata(self):
        """Test that notebook-level metadata is preserved."""
        notebook = {
            "metadata": {
                "kernelspec": {"name": "python3"},
                "language_info": {"name": "python"},
            },
            "cells": [{"id": "cell01", "cell_type": "code", "source": "x = 1"}],
        }

        result = normalize_notebook(notebook)

        assert result["metadata"] == notebook["metadata"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
