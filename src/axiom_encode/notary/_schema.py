"""Small closed-schema predicates shared by public lineage verification."""

from __future__ import annotations

import base64
import binascii
import re

from .canonical import is_canonical, strict_parse


def digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def lane_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+", value) is not None
    )


def fields(value: object, names: set[str]) -> bool:
    return isinstance(value, dict) and value.keys() == names


def canonical_object(raw: bytes) -> dict | None:
    if not is_canonical(raw):
        return None
    value = strict_parse(raw)
    return value if isinstance(value, dict) else None


def decode_base64(value: object) -> bytes | None:
    if not isinstance(value, str):
        return None
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        return None
    return decoded if base64.b64encode(decoded).decode("ascii") == value else None


def relative_path(value: object) -> bool:
    return (
        nonempty(value)
        and "\0" not in value
        and all(part not in {"", ".", ".."} for part in value.split("/"))
    )


def ordered_strings(values: object, *, unique: bool = True) -> bool:
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        return False
    encoded = [v.encode("utf-8") for v in values]
    return all(
        left < right if unique else left <= right
        for left, right in zip(encoded, encoded[1:])
    )
