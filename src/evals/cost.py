"""
cost.py
=======
Dollar cost of model calls from token counts, using the per-model prices in
config evals.prices_per_mtok (USD per million tokens). Used to report what a
judge or robustness run spent and to check it against the budget.
"""

from typing import Any


def call_cost(
    config: dict[str, Any], model: str, input_tokens: int, output_tokens: int
) -> float:
    prices = config["evals"]["prices_per_mtok"].get(model)
    if prices is None:
        raise KeyError(f"No price for {model} in evals.prices_per_mtok")
    return round(
        input_tokens / 1e6 * prices["input"] + output_tokens / 1e6 * prices["output"],
        4,
    )
