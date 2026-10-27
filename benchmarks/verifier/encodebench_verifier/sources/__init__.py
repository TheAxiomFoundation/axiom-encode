"""Case sources: where known-good artifacts and real defect pairs come from."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnownGoodArtifact:
    """A (provision, artifact) believed correct, ready for the mutator."""

    key: str
    citation: str
    provision_text: str
    artifact_text: str
    origin: dict[str, Any] = field(default_factory=dict)
