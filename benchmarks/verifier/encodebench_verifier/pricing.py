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
    payload = json.loads(Path(path).read_text())
    prices: dict[str, Price] = {}
    for model, entry in (payload.get("models") or {}).items():
        prices[model] = Price(
            model=model,
            input_usd_per_million=float(entry["input_usd_per_million"]),
            output_usd_per_million=float(entry["output_usd_per_million"]),
            source=str(entry["source"]),
        )
    return prices


def price_for(model: str, prices: Optional[dict[str, Price]] = None) -> Optional[Price]:
    prices = load_pricing() if prices is None else prices
    return prices.get(model)


def cost_usd(
    price: Optional[Price], tokens_input: int, tokens_output: int
) -> Optional[float]:
    if price is None:
        return None
    return round(
        tokens_input * price.input_usd_per_million / 1_000_000
        + tokens_output * price.output_usd_per_million / 1_000_000,
        8,
    )
