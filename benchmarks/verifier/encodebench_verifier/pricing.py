"""Published per-token prices, loaded from ``benchmarks/verifier/pricing.json``.

Never quote a price from memory: every entry in the JSON names its source.
Models without an entry render cost as blank on the board.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_PRICING_FILE = Path(__file__).resolve().parent.parent / "pricing.json"


@dataclass(frozen=True)
class Price:
    model: str
    input_usd_per_million: float
    output_usd_per_million: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_usd_per_million": self.input_usd_per_million,
            "output_usd_per_million": self.output_usd_per_million,
            "source": self.source,
        }


def load_pricing(path: Path = DEFAULT_PRICING_FILE) -> dict[str, Price]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read pricing file {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise ValueError(f"pricing file {path} must carry a 'models' object")
    prices: dict[str, Price] = {}
    for model, entry in payload["models"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"pricing entry for {model!r} is not an object")
        try:
            price = Price(
                model=model,
                input_usd_per_million=float(entry["input_usd_per_million"]),
                output_usd_per_million=float(entry["output_usd_per_million"]),
                source=str(entry["source"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"pricing entry for {model!r} needs numeric input/output prices "
                f"and a source: {exc}"
            ) from exc
        if price.input_usd_per_million < 0 or price.output_usd_per_million < 0:
            raise ValueError(f"pricing entry for {model!r} has a negative price")
        if not price.source.strip():
            raise ValueError(f"pricing entry for {model!r} has an empty source")
        prices[model] = price
    return prices


def price_for(model: str, prices: Optional[dict[str, Price]] = None) -> Optional[Price]:
    prices = load_pricing() if prices is None else prices
    return prices.get(model)


def cost_usd(
    price: Optional[Price], tokens_input: Optional[int], tokens_output: Optional[int]
) -> Optional[float]:
    """Cost from a published price and *reported* usage; unknown usage is blank."""

    if price is None or tokens_input is None or tokens_output is None:
        return None
    return round(
        tokens_input * price.input_usd_per_million / 1_000_000
        + tokens_output * price.output_usd_per_million / 1_000_000,
        8,
    )
