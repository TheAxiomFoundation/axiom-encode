"""The wall's gate for tests that call the Hebrew readers directly (#1882).

`analyze_complete_source_unit` opens the gate once per candidate, from its
source and deferral reasons (`source_completeness._hebrew_reading`), and every
Hebrew read sits behind it. A test that calls a reader such as
`recognize_source_structure` or `_reason_dependency_is_source_bound` outside an
analysis opens it here, as the analysis of a Hebrew candidate would.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from axiom_encode.harness import source_completeness


@contextlib.contextmanager
def open_hebrew_gate() -> Iterator[None]:
    """Run the enclosed direct calls behind an open gate."""

    token = source_completeness._HEBREW_SCRIPT.set(True)
    try:
        yield
    finally:
        source_completeness._HEBREW_SCRIPT.reset(token)
