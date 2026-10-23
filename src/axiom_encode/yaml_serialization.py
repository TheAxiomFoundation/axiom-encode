"""Safe YAML serialization that preserves strings for YAML 1.2 consumers.

PyYAML's YAML 1.1 resolver does not recognize all YAML 1.2 numbers, so its
default dumper may emit strings such as ``1e5`` or ``-0o17`` without quotes.
Keep PyYAML's safe representations, and quote strings a YAML 1.2 core-schema
consumer could otherwise resolve as numbers, booleans, or null.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

_YAML12_NON_STRING = re.compile(
    r"(?:null|Null|NULL|~|true|True|TRUE|false|False|FALSE)"
    r"|[-+]?(?:[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)"
    r"|[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?\.(?:inf|Inf|INF|nan|NaN|NAN)"
)


class Yaml12SafeDumper(yaml.SafeDumper):
    """SafeDumper with string typing preserved across YAML schema versions."""


def represent_yaml12_string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    """Quote a string whenever a YAML 1.2 core resolver could change its type."""

    # PyYAML emits raw NEL as a YAML line break when Unicode is allowed;
    # loading that scalar normalizes NEL into a space. Double quotes escape it.
    style = (
        '"'
        if not value or _YAML12_NON_STRING.fullmatch(value) or "\x85" in value
        else None
    )
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


Yaml12SafeDumper.add_representer(str, represent_yaml12_string)


def safe_dump_yaml(data: Any, stream: Any = None, **kwargs: Any) -> Any:
    """Like ``yaml.safe_dump``, preserving string types for YAML 1.2 readers.

    This uses a local dumper subclass and never mutates PyYAML's global loader,
    dumper, or resolver configuration. Existing safe_dump keyword arguments and
    stream behavior are forwarded unchanged.
    """

    return yaml.dump(data, stream=stream, Dumper=Yaml12SafeDumper, **kwargs)
