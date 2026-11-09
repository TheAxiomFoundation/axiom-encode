"""Source-conflict compare step.

Two encoded documents that decide the same question (a statute and its
regulation, a regulation and a manual) can give different results for the
same household. This package finds those households from the encodings:

- :mod:`.link` reads the link file that says which rules decide the same
  question and maps the shared facts to each module's inputs;
- :mod:`.gates` checks the link against the compiled modules;
- :mod:`.values` builds the household grid from the modules' thresholds;
- :mod:`.compare` runs both modules in the rules engine and writes the
  conflict record;
- :mod:`.engine` is the only module that depends on the rest of axiom-encode.

See ``docs/source-conflict-compare.md``.
"""

from .compare import RECORD_FORMAT, REVIEW_STATES, CompareError, compare, render_record
from .gates import LinkGateError
from .link import LINK_FORMAT, Link, LinkError, load_link, parse_link

__all__ = [
    "LINK_FORMAT",
    "RECORD_FORMAT",
    "REVIEW_STATES",
    "CompareError",
    "Link",
    "LinkError",
    "LinkGateError",
    "compare",
    "load_link",
    "parse_link",
    "render_record",
]
