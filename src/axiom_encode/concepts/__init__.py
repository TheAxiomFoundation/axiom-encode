"""Canonical-concept registry for axiom-encode.

Prevents the name-drift class bug where the same legal concept gets different
variable names across encoder runs. The registry maps a canonical concept id
to one approved variable name plus a producer anchor; the validator refuses
to apply generated RuleSpec that uses a blocked synonym or conflicts with a
registered canonical.

Cross-jurisdiction semantics
----------------------------
Anchors carry a jurisdiction prefix: `us:` for federal, `us-co:`, `us-ny:`,
etc. for state RuleSpec. A registered canonical's `producer_anchor` is the
single approved location of its producer rule, regardless of where consumers
live. So a state policy in `us-co:policies/...` that consumes a federal
concept must reference it as `us:regulations/7-cfr/.../#snap_total_gross_income`
— anchoring it under `us-co:` triggers `anchored_ref_miss`. State-only
concepts (no federal counterpart) get a `us-co:` / `us-ny:` `producer_anchor`
and are state-canonical. Concepts marked `producer_missing: true` allow the
canonical name in consumers while the producer is encoded in a follow-up.
"""

from .audit import (
    DriftFinding,
    audit_corpus,
)
from .auto_repair import auto_repair_test_yaml_canonical_violations
from .registry import (
    Concept,
    ConceptRegistry,
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
    "audit_corpus",
    "auto_repair_test_yaml_canonical_violations",
    "load_concept_registry",
    "validate_generated_against_registry",
]
