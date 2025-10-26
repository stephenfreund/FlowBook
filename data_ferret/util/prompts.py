"""
Prompt management for DataFerret.

This module provides functionality to load prompts from YAML files
and format them with provided substitutions.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Default prompts that will be used if no YAML file is found
DEFAULT_PROMPTS = {
    "cell_inspection": """\
You are an expert at optimizing code. You will be asked about a code cell.
Estimate the performance gains from further optimizing this code using the scale low (0) to high (5).
Code that does not take much time to run should get a low score.
You cannot change the hardware or the environment. The effects of the code must be the same.
"""
}

class PromptManager:
    """Manages loading and formatting prompts from YAML files."""
    
    def __init__(self, yaml_path: Optional[str] = None):
        """Initialize the PromptManager.
        
        Args:
            yaml_path: Path to a YAML file containing prompts. If not provided,
                     will look for 'prompts.yaml' in the package directory.
        """
        self.prompts = DEFAULT_PROMPTS.copy()
        
        if yaml_path is None:
            # Look for prompts.yaml in the package directory
            package_dir = Path(__file__).parent
            yaml_path = package_dir / "prompts.yaml"
            
            if not yaml_path.exists():
                return
        
        self._load_prompts(yaml_path)
    
    def _load_prompts(self, yaml_path: str) -> None:
        """Load prompts from a YAML file.
        
        Args:
            yaml_path: Path to the YAML file.
            
        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
            yaml.YAMLError: If the YAML file is malformed.
        """
        try:
            with open(yaml_path, 'r') as f:
                yaml_prompts = yaml.safe_load(f) or {}
                self.prompts.update(yaml_prompts)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {yaml_path}")
    
    def get_prompt(self, key: str, **kwargs: Any) -> str:
        """Get a prompt by key and format it with the provided substitutions.
        
        Args:
            key: The key of the prompt to retrieve.
            **kwargs: Key-value pairs for string formatting.
            
        Returns:
            The formatted prompt string.
            
        Raises:
            KeyError: If the prompt key doesn't exist.
        """
        if key not in self.prompts:
            raise KeyError(f"No prompt found for key: {key}")
            
        prompt = self.prompts[key]
        
        try:
            return prompt.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Missing required substitution key for prompt '{key}': {e}")

# Create a default instance for easy importing
default_prompt_manager = PromptManager()

def get_prompt(key: str, **kwargs: Any) -> str:
    """Convenience function to get a prompt from the default prompt manager.
    
    Args:
        key: The key of the prompt to retrieve.
        **kwargs: Key-value pairs for string formatting.
        
    Returns:
        The formatted prompt string.
    """
    return default_prompt_manager.get_prompt(key, **kwargs)