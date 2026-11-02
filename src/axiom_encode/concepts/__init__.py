"""Canonical-concept registry for axiom-encode.

Prevents the name-drift class bug where the same legal concept gets different
variable names across encoder runs. The registry maps a canonical concept id
to one approved variable name plus its accepted producer anchors; the
validator refuses to apply generated RuleSpec that uses a blocked synonym or
conflicts with a registered canonical.

Cross-jurisdiction semantics
----------------------------
Anchors carry a jurisdiction prefix: `us:` for federal, `us-co:`, `us-ny:`,
etc. for state RuleSpec. A registered canonical's `producer_anchors` are the
only approved locations of its producer rule, regardless of where consumers
live. Most concepts have exactly one. So a state policy in
`us-co:policies/...` that consumes a federal concept must reference it as
`us:regulations/7-cfr/.../#snap_total_gross_income` — anchoring it under
`us-co:` triggers `anchored_ref_miss`. State-only concepts (no federal
counterpart) get a `us-co:` / `us-ny:` producer anchor and are
state-canonical. Concepts marked `producer_missing: true` allow the
canonical name in consumers while the producer is encoded in a follow-up.

Vintaged producers
------------------
A concept whose value is republished per period lists one accepted producer
per vintage in `producer_anchors` (the SNAP cost-of-living allotments have
one module per fiscal year), optionally with each vintage's dates in
`producer_periods`. A module may define the name, and a non-input reference
may target it, at any listed anchor and nowhere else; test auto-repair never
moves a reference between vintages. `producer_anchor` remains as a
back-compat label and, when set, must be one of `producer_anchors`.
"""

from .audit import (
    DriftFinding,
    audit_corpus,
)
from .auto_repair import auto_repair_test_yaml_canonical_violations
from .registry import (
    Concept,
    ConceptRegistry,
    ProducerPeriod,
    load_concept_registry,
)
from .validator import (
    CanonicalNameViolation,
    validate_generated_against_registry,
)

__all__ = [
    "Concept",
    "ConceptRegistry",
    "DriftFinding",
    "CanonicalNameViolation",
    "ProducerPeriod",
    "audit_corpus",
    "auto_repair_test_yaml_canonical_violations",
    "load_concept_registry",
    "validate_generated_against_registry",
]
