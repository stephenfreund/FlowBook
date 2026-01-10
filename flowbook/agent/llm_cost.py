from agents import Usage
import json

import os

with open(os.path.join(os.path.dirname(__file__), "llm_costs.json")) as f:
    _costs = json.load(f)


def provider(model: str) -> str:
    return _costs.get(model, {}).get("litellm_provider")


def full_name(model: str) -> str:
    return f"{provider(model)}/{model}"


def cost(model: str, response_usage: Usage) -> float:
    model_cost = _costs.get(model)
    if model_cost is None:
        raise ValueError(f"Unknown model: {model}")

    cached_tokens = response_usage.input_tokens_details.cached_tokens
    input_tokens = response_usage.input_tokens - cached_tokens
    output_tokens = response_usage.output_tokens
    return (
        cached_tokens * model_cost["input_cost_per_token"]
        + input_tokens * model_cost["input_cost_per_token"]
        + output_tokens * model_cost["output_cost_per_token"]
    )
