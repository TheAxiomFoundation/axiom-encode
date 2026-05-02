"""Legal-ID keyed PolicyEngine oracle mapping registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_MAPPING_TYPES = {
    "direct_variable",
    "derived_expression",
    "many_to_one",
    "one_to_many",
    "not_comparable",
}


@dataclass(frozen=True)
class PolicyEngineMapping:
    """Mapping from an Axiom legal output ID to a PolicyEngine comparison target."""

    legal_id: str
    country: str
    mapping_type: str
    policyengine_variable: str | None = None
    program: str | None = None
    entity: str | None = None
    period: str | None = None
    unit: str | None = None
    comparison: str | None = None
    expression: str | None = None
    rationale: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def comparable(self) -> bool:
        return self.mapping_type != "not_comparable" and bool(
            self.policyengine_variable or self.expression
        )


@dataclass
class PolicyEngineOracleCoverage:
    """Coverage counters for a PolicyEngine oracle run."""

    total_outputs: int = 0
    skipped: int = 0
    comparable: int = 0
    passed: int = 0
    failed: int = 0
    unmapped: int = 0
    unsupported: int = 0
    adapter_errors: int = 0
    setup_errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "total_outputs": self.total_outputs,
            "skipped": self.skipped,
            "comparable": self.comparable,
            "passed": self.passed,
            "failed": self.failed,
            "unmapped": self.unmapped,
            "unsupported": self.unsupported,
            "adapter_errors": self.adapter_errors,
            "setup_errors": self.setup_errors,
        }


@dataclass(frozen=True)
class PolicyEngineOracleRegistry:
    """Resolved PolicyEngine mapping registry."""

    mappings_by_legal_id: dict[str, PolicyEngineMapping] = field(default_factory=dict)

    def mapping_for_legal_id(
        self, legal_id: str | None, *, country: str | None = None
    ) -> PolicyEngineMapping | None:
        if not legal_id:
            return None
        mapping = self.mappings_by_legal_id.get(str(legal_id))
        if mapping is None:
            return None
        if country and mapping.country != country:
            return None
        return mapping

    def legal_ids_for_policyengine_variable(
        self, policyengine_variable: str, *, country: str | None = None
    ) -> list[str]:
        return [
            legal_id
            for legal_id, mapping in self.mappings_by_legal_id.items()
            if mapping.policyengine_variable == policyengine_variable
            and (country is None or mapping.country == country)
        ]

    def validate(self) -> list[str]:
        issues: list[str] = []
        seen: set[str] = set()
        for legal_id, mapping in self.mappings_by_legal_id.items():
            if legal_id in seen:
                issues.append(f"Duplicate PolicyEngine mapping legal_id: {legal_id}")
            seen.add(legal_id)
            if "#" not in legal_id or ":" not in legal_id:
                issues.append(
                    f"PolicyEngine mapping key must be a canonical legal ID: {legal_id}"
                )
            if mapping.mapping_type not in SUPPORTED_MAPPING_TYPES:
                issues.append(
                    f"Unsupported PolicyEngine mapping_type for {legal_id}: "
                    f"{mapping.mapping_type}"
                )
            if mapping.mapping_type == "direct_variable" and not (
                mapping.policyengine_variable
            ):
                issues.append(
                    f"Direct PolicyEngine mapping missing policyengine_variable: {legal_id}"
                )
            if mapping.mapping_type == "not_comparable" and not mapping.rationale:
                issues.append(
                    f"PolicyEngine not_comparable mapping missing rationale: {legal_id}"
                )
        return issues


def load_policyengine_registry() -> PolicyEngineOracleRegistry:
    """Load packaged PolicyEngine mappings."""
    mappings: dict[str, PolicyEngineMapping] = {}
    mapping_dir = Path(__file__).with_name("mappings")
    for mapping_path in sorted(mapping_dir.glob("*.yaml")):
        payload = yaml.safe_load(mapping_path.read_text()) or {}
        raw_mappings = payload.get("mappings", [])
        if not isinstance(raw_mappings, list):
            raise ValueError(f"{mapping_path} mappings must be a list")
        for raw_mapping in raw_mappings:
            if not isinstance(raw_mapping, dict):
                raise ValueError(f"{mapping_path} contains a non-object mapping")
            mapping = _mapping_from_payload(raw_mapping)
            if mapping.legal_id in mappings:
                raise ValueError(
                    f"Duplicate PolicyEngine mapping legal_id: {mapping.legal_id}"
                )
            mappings[mapping.legal_id] = mapping
    registry = PolicyEngineOracleRegistry(mappings)
    issues = registry.validate()
    if issues:
        raise ValueError("Invalid PolicyEngine registry: " + "; ".join(issues))
    return registry


def _mapping_from_payload(payload: dict[str, Any]) -> PolicyEngineMapping:
    aliases = payload.get("aliases", ())
    if aliases is None:
        aliases = ()
    if isinstance(aliases, str):
        aliases = (aliases,)
    return PolicyEngineMapping(
        legal_id=str(payload["legal_id"]),
        country=str(payload.get("country", "us")),
        mapping_type=str(payload.get("mapping_type", "direct_variable")),
        policyengine_variable=payload.get("policyengine_variable"),
        program=payload.get("program"),
        entity=payload.get("entity"),
        period=payload.get("period"),
        unit=payload.get("unit"),
        comparison=payload.get("comparison"),
        expression=payload.get("expression"),
        rationale=payload.get("rationale"),
        aliases=tuple(str(alias) for alias in aliases),
    )
