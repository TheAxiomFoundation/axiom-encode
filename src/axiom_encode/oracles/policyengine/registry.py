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
    "parameter_value",
    "not_comparable",
}
SUPPORTED_MATCH_TYPES = {"exact", "prefix"}


@dataclass(frozen=True)
class PolicyEngineMapping:
    """Mapping from an Axiom legal output ID to a PolicyEngine comparison target."""

    legal_id: str
    country: str
    mapping_type: str
    match_type: str = "exact"
    policyengine_variable: str | None = None
    policyengine_parameter: str | None = None
    parameter_key: str | None = None
    parameter_keys: tuple[str, ...] = ()
    parameter_key_input: str | None = None
    parameter_key_map: dict[str, str] = field(default_factory=dict)
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
            self.policyengine_variable or self.policyengine_parameter or self.expression
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
    prefix_mappings: tuple[PolicyEngineMapping, ...] = ()

    def mapping_for_legal_id(
        self, legal_id: str | None, *, country: str | None = None
    ) -> PolicyEngineMapping | None:
        if not legal_id:
            return None
        legal_id_text = str(legal_id)
        mapping = self.mappings_by_legal_id.get(legal_id_text)
        if mapping is not None and (country is None or mapping.country == country):
            return mapping
        for prefix_mapping in self.prefix_mappings:
            if country and prefix_mapping.country != country:
                continue
            if legal_id_text.startswith(prefix_mapping.legal_id):
                return prefix_mapping
        return None

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
        for legal_id, mapping in [
            *self.mappings_by_legal_id.items(),
            *((mapping.legal_id, mapping) for mapping in self.prefix_mappings),
        ]:
            if legal_id in seen:
                issues.append(f"Duplicate PolicyEngine mapping legal_id: {legal_id}")
            seen.add(legal_id)
            if mapping.match_type not in SUPPORTED_MATCH_TYPES:
                issues.append(
                    f"Unsupported PolicyEngine match_type for {legal_id}: "
                    f"{mapping.match_type}"
                )
            if mapping.match_type == "exact" and (
                "#" not in legal_id or ":" not in legal_id
            ):
                issues.append(
                    f"PolicyEngine mapping key must be a canonical legal ID: {legal_id}"
                )
            if mapping.match_type == "prefix":
                if ":" not in legal_id:
                    issues.append(
                        "PolicyEngine prefix mapping key must be a canonical "
                        f"legal ID prefix: {legal_id}"
                    )
                if mapping.mapping_type != "not_comparable":
                    issues.append(
                        "PolicyEngine prefix mappings may only classify "
                        f"not_comparable outputs: {legal_id}"
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
            if (
                mapping.mapping_type == "parameter_value"
                and not mapping.policyengine_parameter
            ):
                issues.append(
                    f"PolicyEngine parameter mapping missing policyengine_parameter: {legal_id}"
                )
            key_selectors = [
                bool(mapping.parameter_key),
                bool(mapping.parameter_keys),
                bool(mapping.parameter_key_input),
            ]
            if sum(key_selectors) > 1:
                issues.append(
                    "PolicyEngine parameter mapping must use only one of "
                    f"parameter_key, parameter_keys, or parameter_key_input: {legal_id}"
                )
            if mapping.parameter_key_map and not mapping.parameter_key_input:
                issues.append(
                    "PolicyEngine parameter_key_map requires "
                    f"parameter_key_input: {legal_id}"
                )
            if mapping.mapping_type == "not_comparable" and not mapping.rationale:
                issues.append(
                    f"PolicyEngine not_comparable mapping missing rationale: {legal_id}"
                )
        return issues


def load_policyengine_registry() -> PolicyEngineOracleRegistry:
    """Load packaged PolicyEngine mappings."""
    mappings: dict[str, PolicyEngineMapping] = {}
    prefix_mappings: list[PolicyEngineMapping] = []
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
        raw_prefixes = payload.get("prefixes", [])
        if not isinstance(raw_prefixes, list):
            raise ValueError(f"{mapping_path} prefixes must be a list")
        for raw_prefix in raw_prefixes:
            if not isinstance(raw_prefix, dict):
                raise ValueError(f"{mapping_path} contains a non-object prefix")
            prefix_payload = {**raw_prefix, "match_type": "prefix"}
            prefix_mappings.append(_mapping_from_payload(prefix_payload))
    registry = PolicyEngineOracleRegistry(
        mappings,
        tuple(
            sorted(
                prefix_mappings,
                key=lambda mapping: len(mapping.legal_id),
                reverse=True,
            )
        ),
    )
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
    parameter_keys = payload.get("parameter_keys", ())
    if parameter_keys is None:
        parameter_keys = ()
    if isinstance(parameter_keys, str):
        parameter_keys = (parameter_keys,)
    legal_id = payload.get("legal_id", payload.get("legal_id_prefix"))
    if legal_id is None:
        raise ValueError("PolicyEngine mapping missing legal_id")
    return PolicyEngineMapping(
        legal_id=str(legal_id),
        country=str(payload.get("country", "us")),
        mapping_type=str(payload.get("mapping_type", "direct_variable")),
        match_type=str(payload.get("match_type", "exact")),
        policyengine_variable=payload.get("policyengine_variable"),
        policyengine_parameter=payload.get("policyengine_parameter"),
        parameter_key=(
            str(payload["parameter_key"])
            if payload.get("parameter_key") is not None
            else None
        ),
        parameter_keys=tuple(str(key) for key in parameter_keys),
        parameter_key_input=payload.get("parameter_key_input"),
        parameter_key_map={
            str(key): str(value)
            for key, value in (payload.get("parameter_key_map") or {}).items()
        },
        program=payload.get("program"),
        entity=payload.get("entity"),
        period=payload.get("period"),
        unit=payload.get("unit"),
        comparison=payload.get("comparison"),
        expression=payload.get("expression"),
        rationale=payload.get("rationale"),
        aliases=tuple(str(alias) for alias in aliases),
    )
