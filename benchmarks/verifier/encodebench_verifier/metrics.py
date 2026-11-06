"""Detection metrics for the verifier board.

All scores are "probability the artifact is defective" in [0, 1]; higher
means the judge is more suspicious. Nothing here assumes a distribution.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Optional, Sequence


def auc(positives: Sequence[float], negatives: Sequence[float]) -> Optional[float]:
    """Mann-Whitney AUC: P(score(defective) > score(control)), ties count 0.5.

    Pooled, not paired: every scored defective case is compared with every
    scored control case of the same kind. ``None`` when either side is empty.
    """

    if not positives or not negatives:
        return None
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return round(wins / (len(positives) * len(negatives)), 6)


def paired_rise_rate(pairs: Sequence[tuple[float, float]]) -> Optional[float]:
    """Fraction of (defective, control) pairs where the defective scored higher."""

    if not pairs:
        return None
    return round(sum(1 for pos, neg in pairs if pos > neg) / len(pairs), 6)


def mean_paired_delta(pairs: Sequence[tuple[float, float]]) -> Optional[float]:
    if not pairs:
        return None
    return round(sum(pos - neg for pos, neg in pairs) / len(pairs), 6)


def detection_at_false_alarm_ceiling(
    positives: Sequence[float], negatives: Sequence[float], ceiling: float
) -> Optional[float]:
    """Detection rate when the threshold lets at most ``ceiling`` of controls through.

    The threshold is the lowest score that admits at most
    ``floor(ceiling * n_controls)`` controls strictly above it (the (k+1)-th
    largest control score); detection is the share of defective cases
    strictly above that threshold. With a 0.0 ceiling this is "the share of
    defective cases scoring above every control".
    """

    if not positives or not negatives:
        return None
    if not 0.0 <= ceiling <= 1.0:
        raise ValueError("ceiling must be within [0, 1]")
    # Float-safe floor: 0.29 * 100 is 28.999999999999996 in binary floating point.
    allowed = math.floor(ceiling * len(negatives) + 1e-9)
    ordered = sorted(negatives, reverse=True)
    threshold = ordered[allowed] if allowed < len(ordered) else float("-inf")
    return round(sum(1 for pos in positives if pos > threshold) / len(positives), 6)


def rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def median_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(float(median(values)), 3)


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 6)
