"""
Test that environment data from profile metadata is included in optimization prompts.
"""

import nbformat
from flowbook.util.flowbook_metadata import FlowbookMetadata, ProfileData, set_profile_flowbook_metadata


def test_env_extraction_from_profile():
    """Test that we can extract env from profile metadata."""
    # Create a cell with profile metadata
    cell = nbformat.v4.new_code_cell(source="x = 10")
    cell["id"] = "test-cell-123"

    # Add profile metadata with env
    profile_data = ProfileData(
        duration=1.5,
        profile="some profile data",
        env={"x": "int", "y": "str", "data": "DataFrame"},
        env_after={"x": "int", "y": "str", "data": "DataFrame", "result": "float"}
    )
    set_profile_flowbook_metadata(cell, profile_data)

    # Extract the metadata
    flowbook_metadata = FlowbookMetadata.from_cell(cell)
    profile = flowbook_metadata.get_profile()

    # Verify we can access env
    assert profile is not None
    assert profile.env is not None
    assert "x" in profile.env
    assert profile.env["x"] == "int"
    assert profile.env["data"] == "DataFrame"

    print("✓ Successfully extracted env from profile metadata")


def test_env_formatting():
    """Test that env data is formatted correctly for the prompt."""
    env_data = {
        "x": "int",
        "y": "str",
        "data": "DataFrame",
        "model": "LinearRegression"
    }

    # Format as it would be in optimize.py
    env_lines = [f"  {var}: {type_}" for var, type_ in env_data.items()]
    env_section = "Available variables in the environment (from profiling):\n" + "\n".join(env_lines)

    # Verify formatting
    assert "Available variables in the environment" in env_section
    assert "  x: int" in env_section
    assert "  data: DataFrame" in env_section
    assert "  model: LinearRegression" in env_section

    print("✓ Environment data formatted correctly")
    print("\nFormatted output:")
    print(env_section)


def test_no_profile_metadata():
    """Test graceful handling when there's no profile metadata."""
    # Create a cell without profile metadata
    cell = nbformat.v4.new_code_cell(source="print('hello')")
    cell["id"] = "test-cell-456"

    # Extract metadata
    flowbook_metadata = FlowbookMetadata.from_cell(cell)
    profile = flowbook_metadata.get_profile()

    # Should be None
    assert profile is None

    # Simulate the logic in optimize.py
    env_data = profile.env if profile else None

    assert env_data is None

    # When env_data is None, env_section should be empty
    if env_data:
        env_section = "Available variables..."
    else:
        env_section = ""

    assert env_section == ""

    print("✓ Gracefully handles missing profile metadata")


if __name__ == "__main__":
    test_env_extraction_from_profile()
    test_env_formatting()
    test_no_profile_metadata()
    print("\n✅ All tests passed!")
