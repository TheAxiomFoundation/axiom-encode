"""Digest and YAML helpers shared by every module in the verifier track."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import yaml


def canonical_json(payload: Any) -> str:
    """Stable JSON text: sorted keys, no whitespace variance, UTF-8 kept."""

    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _BlockDumper(yaml.SafeDumper):
    """Multi-line strings render as literal blocks so formulas stay readable."""


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    if "\n" in value:
        # PyYAML picks the chomping indicator (``|`` vs ``|-``) from the value
        # and falls back to a quoted scalar when block style is not allowed,
        # so the text round-trips unchanged either way.
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", value)


_BlockDumper.add_representer(str, _represent_str)


def load_yaml_document(text: str) -> Any:
    """Parse one YAML document; raises ``yaml.YAMLError`` on malformed input."""

    return yaml.safe_load(text)


def dump_yaml_document(document: Any) -> str:
    """Canonical re-serialisation used for both members of a synthetic pair.

    Both the control and the defective artifact are dumped through this one
    function, so the only difference between them is the planted edit — not
    quoting style, indentation, or line folding inherited from the encoder's
    original output.
    """

    return yaml.dump(
        document,
        Dumper=_BlockDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=100_000,
    )
