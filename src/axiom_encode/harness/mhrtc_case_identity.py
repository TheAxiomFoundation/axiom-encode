"""Frozen companion selection for private MHRtC capture, not source authority.

The ordinary loader's float view establishes caller identity only. Requests and
expected arithmetic must use the returned exact Decimal view. No mutable parsed
case is retained here, and this public value cannot authorize evidence itself.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import yaml

from .mhrtc_request import parse_mhrtc_companion


def _same_caller_tree(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (
            all(type(key) is str for key in left)
            and all(type(key) is str for key in right)
            and left.keys() == right.keys()
            and all(_same_caller_tree(left[key], right[key]) for key in left)
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _same_caller_tree(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is float:
        return (
            math.isfinite(left) and math.isfinite(right) and left.hex() == right.hex()
        )
    if type(left) in (str, bool, int, type(None)):
        return left == right
    return False


@dataclass(frozen=True)
class FrozenMhrtcCase:
    """Exact source-backed case data; a public locator, never an authority token."""

    companion_sha256: str
    ordinal: int
    case: dict


def frozen_mhrtc_case(raw: bytes, ordinal: int, caller_case: object) -> FrozenMhrtcCase:
    """Validate raw bytes before selecting an exact, independently parsed case."""
    if type(raw) is not bytes or len(raw) > 4 * 1024 * 1024:
        raise ValueError("bounded frozen companion bytes required")
    if type(ordinal) is not int or ordinal < 1:
        raise ValueError("positive exact integer case ordinal required")
    text = raw.decode("utf-8")
    exact = parse_mhrtc_companion(text)
    if ordinal > len(exact):
        raise ValueError("case ordinal outside frozen companion")
    # Strict parsing above rejects aliases, duplicates and unsupported facts
    # before the ordinary caller representation is examined.
    ordinary = yaml.safe_load(text)
    if not _same_caller_tree(caller_case, ordinary[ordinal - 1]):
        raise ValueError("caller case differs from frozen companion")
    return FrozenMhrtcCase(hashlib.sha256(raw).hexdigest(), ordinal, exact[ordinal - 1])
