"""Select the generated property table matching Python's Unicode database."""

from __future__ import annotations

import importlib
import unicodedata

_SUPPORTED_VERSIONS = ("15.1.0", "16.0.0", "17.0.0")
if unicodedata.unidata_version not in _SUPPORTED_VERSIONS:
    raise RuntimeError(
        f"Unicode {unicodedata.unidata_version} needs its default-ignorable table; "
        "run scripts/generate_default_ignorable_table.py --download and add the version "
        "to _unicode_default_ignorables._SUPPORTED_VERSIONS"
    )
_data = importlib.import_module(
    f"._unicode_default_ignorables_{unicodedata.unidata_version.replace('.', '_')}",
    __package__,
)
UNICODE_VERSION: str = _data.UNICODE_VERSION
DERIVED_CORE_PROPERTIES_SHA256: str = _data.DERIVED_CORE_PROPERTIES_SHA256
DEFAULT_IGNORABLE_RANGES: tuple[tuple[int, int], ...] = _data.DEFAULT_IGNORABLE_RANGES
