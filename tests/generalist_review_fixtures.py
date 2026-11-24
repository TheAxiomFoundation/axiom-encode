"""Review evidence for tests of the generalist review record."""

from __future__ import annotations

import hashlib


def whole_text(raw: bytes) -> dict[str, object]:
    """Coverage of one text that a single reviewer call showed whole."""

    digest = hashlib.sha256(raw).hexdigest()
    return {
        "sha256": digest,
        "size": len(raw),
        "chunks": [{"start": 0, "end": len(raw), "sha256": digest}],
    }


def whole_review_coverage(
    rulespec: bytes,
    test: bytes | None,
    source: bytes | None = None,
    *,
    calls: int = 1,
) -> dict[str, object]:
    """The coverage block of a review that showed every text whole."""

    return {
        "calls_planned": calls,
        "calls_made": calls,
        "rulespec": whole_text(rulespec),
        "test": None if test is None else whole_text(test),
        "source": None if source is None else whole_text(source),
    }
