"""
Test that environment data from profile metadata is included in generate prompts.
"""

import nbformat
from data_ferret.util.ferret_metadata import FerretMetadata, ProfileData, set_profile_ferret_metadata
from data_ferret.util.prompts import get_prompt


def test_generate_prompt_with_env():
    """Test that generate_input prompt includes env_section."""

    # Simulate the parameters passed in generate.py
    prefix = "import pandas as pd\nimport numpy as np\n\ndf = pd.read_csv('data.csv')"
    specification = "Create a function to calculate the mean of column 'price'"

    # Simulate env_section with profile data (from env_after of previous cell)
    env_data = {
        "pd": "module",
        "np": "module",
        "df": "DataFrame"
    }
    env_lines = [f"  {var}: {type_}" for var, type_ in env_data.items()]
    env_section = "Available variables in the environment (from profiling):\n" + "\n".join(env_lines)

    # Get the formatted prompt
    prompt = get_prompt(
        "generate_input",
        prefix=prefix,
        specification=specification,
        env_section=env_section,
    )

    # Verify all parts are included
    assert "import pandas as pd" in prompt
    assert "Create a function to calculate" in prompt
    assert "Available variables in the environment (from profiling):" in prompt
    assert "  pd: module" in prompt
    assert "  df: DataFrame" in prompt

    print("✓ Generate prompt correctly includes env section")
    print("\n" + "="*60)
    print("Generated prompt:")
    print("="*60)
    print(prompt)
    print("="*60)


def test_generate_prompt_without_env():
    """Test that generate_input works with empty env_section."""

    prefix = "x = 10"
    specification = "Create a function that doubles a number"
    env_section = ""  # No profile data available

    # Get the formatted prompt
    prompt = get_prompt(
        "generate_input",
        prefix=prefix,
        specification=specification,
        env_section=env_section,
    )

    # Verify it still works with empty env_section
    assert "x = 10" in prompt
    assert "Create a function that doubles" in prompt
    # Should not have the env header since env_section is empty
    assert "Available variables in the environment" not in prompt

    print("✓ Generate prompt works with empty env section")


def test_env_tracking_in_notebook():
    """Test tracking env_after through multiple cells."""

    # Create a notebook with multiple cells
    nb = nbformat.v4.new_notebook()

    # Cell 1: Has profile data
    cell1 = nbformat.v4.new_code_cell(source="import pandas as pd\ndf = pd.DataFrame()")
    cell1["id"] = "cell-1"
    profile1 = ProfileData(
        duration=0.5,
        profile="profile1",
        env={},
        env_after={"pd": "module", "df": "DataFrame"}
    )
    set_profile_ferret_metadata(cell1, profile1)

    # Cell 2: Has profile data (environment grows)
    cell2 = nbformat.v4.new_code_cell(source="result = df.shape[0]")
    cell2["id"] = "cell-2"
    profile2 = ProfileData(
        duration=0.3,
        profile="profile2",
        env={"pd": "module", "df": "DataFrame"},
        env_after={"pd": "module", "df": "DataFrame", "result": "int"}
    )
    set_profile_ferret_metadata(cell2, profile2)

    # Cell 3: String spec cell (would be processed by generate)
    cell3 = nbformat.v4.new_code_cell(source='"Filter the dataframe to only positive results"')
    cell3["id"] = "cell-3"

    nb["cells"] = [cell1, cell2, cell3]

    # Simulate the env tracking logic from generate.py process method
    current_env = None
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and cell["source"].strip():
            ferret_metadata = FerretMetadata.from_cell(cell)
            profile = ferret_metadata.get_profile()
            if profile and profile.env_after:
                current_env = profile.env_after

    # After processing all cells before cell3, current_env should have all variables
    assert current_env is not None
    assert "pd" in current_env
    assert "df" in current_env
    assert "result" in current_env
    assert current_env["result"] == "int"

    print("✓ Environment tracking works correctly through multiple cells")
    print(f"  Final environment: {current_env}")


if __name__ == "__main__":
    test_generate_prompt_with_env()
    print()
    test_generate_prompt_without_env()
    print()
    test_env_tracking_in_notebook()
    print("\n✅ All generate command tests passed!")
