"""Canonical-concept registry for axiom-encode.

Prevents the name-drift class bug where the same legal concept gets different
variable names across encoder runs. The registry maps a canonical concept id
to one approved variable name plus a producer anchor; the validator refuses
to apply generated RuleSpec that uses a blocked synonym or conflicts with a
registered canonical.
"""

from .audit import (
    DriftFinding,
    audit_corpus,
)
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
    "load_concept_registry",
    "validate_generated_against_registry",
]
