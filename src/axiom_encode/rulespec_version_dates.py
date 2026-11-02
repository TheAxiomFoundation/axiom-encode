"""Locate documented RuleSpec version dates without erasing their proof paths."""

from collections.abc import Mapping
from typing import Any


def version_date_fields(version: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return actual date keys and refuse duplicate canonical/alias spellings.

    Leave each caller's value and type checks intact. The engine refuses both
    spellings of one field even when they contain the same date.
    """
    fields = []
    for canonical, alias in (("effective_from", "from"), ("effective_to", "to")):
        if canonical in version and alias in version:
            return None
        fields.append(canonical if canonical in version else alias)
    return fields[0], fields[1]
