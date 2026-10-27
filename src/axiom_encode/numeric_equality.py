"""Shared numeric equality for RuleSpec runtime and expected values."""

import math
from decimal import Decimal

_DECIMAL_RESIDUE_TOLERANCE = Decimal("1e-18")
_BINARY64_EXACT_INTEGER_LIMIT = Decimal(2**53)


def rulespec_numeric_values_equal(
    actual: Decimal,
    expected: Decimal,
    *,
    actual_kind: str,
    expected_kind: str,
) -> bool:
    """Accept exact decimal equality or one safe binary64 representation step."""

    if abs(actual - expected) <= _DECIMAL_RESIDUE_TOLERANCE:
        return True
    if actual_kind == expected_kind == "integer":
        return False
    if (
        abs(actual) >= _BINARY64_EXACT_INTEGER_LIMIT
        or abs(expected) >= _BINARY64_EXACT_INTEGER_LIMIT
    ):
        return False
    actual_float = float(actual)
    expected_float = float(expected)
    if not (math.isfinite(actual_float) and math.isfinite(expected_float)):
        return False
    return (
        actual_float == expected_float
        or math.nextafter(actual_float, expected_float) == expected_float
    )
