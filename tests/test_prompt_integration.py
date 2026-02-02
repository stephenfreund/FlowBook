"""
Test that the optimization prompt correctly includes env section.
"""

from flowbook.util.prompts import get_prompt


def test_optimization_prompt_with_env():
    """Test that optimization_input prompt includes env_section."""

    # Simulate the parameters passed in optimize.py
    prefix = "import pandas as pd\nimport numpy as np"
    kind = "function"
    cell_source = "def slow_function(data):\n    return [x * 2 for x in data]"
    optimization_descriptions = "Optimizations to apply:\n- Vectorize the operation"

    # Simulate env_section with profile data
    env_data = {
        "pd": "module",
        "np": "module",
        "data": "DataFrame"
    }
    env_lines = [f"  {var}: {type_}" for var, type_ in env_data.items()]
    env_section = "Available variables in the environment (from profiling):\n" + "\n".join(env_lines)

    # Get the formatted prompt
    prompt = get_prompt(
        "optimization_input",
        prefix=prefix,
        kind=kind,
        cell_source=cell_source,
        env_section=env_section,
        optimization_descriptions=optimization_descriptions,
    )

    # Verify all parts are included
    assert "import pandas as pd" in prompt
    assert "def slow_function(data):" in prompt
    assert "Available variables in the environment (from profiling):" in prompt
    assert "  pd: module" in prompt
    assert "  data: DataFrame" in prompt
    assert "Optimizations to apply:" in prompt
    assert "Vectorize the operation" in prompt

    print("✓ Optimization prompt correctly includes env section")
    print("\n" + "="*60)
    print("Generated prompt:")
    print("="*60)
    print(prompt)
    print("="*60)


def test_optimization_prompt_without_env():
    """Test that optimization_input works with empty env_section."""

    prefix = "import pandas as pd"
    kind = "cell"
    cell_source = "result = pd.DataFrame({'a': [1, 2, 3]})"
    optimization_descriptions = "Optimizations to apply:\n- Use more efficient data structure"
    env_section = ""  # No profile data available

    # Get the formatted prompt
    prompt = get_prompt(
        "optimization_input",
        prefix=prefix,
        kind=kind,
        cell_source=cell_source,
        env_section=env_section,
        optimization_descriptions=optimization_descriptions,
    )

    # Verify it still works with empty env_section
    assert "import pandas as pd" in prompt
    assert "result = pd.DataFrame" in prompt
    assert "Optimizations to apply:" in prompt
    # Should not have the env header since env_section is empty
    assert "Available variables in the environment" not in prompt

    print("✓ Optimization prompt works with empty env section")


if __name__ == "__main__":
    test_optimization_prompt_with_env()
    print()
    test_optimization_prompt_without_env()
    print("\n✅ All prompt integration tests passed!")
