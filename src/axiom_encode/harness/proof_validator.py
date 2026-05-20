"""RuleSpec proof-tree validation.

This module intentionally validates proof structure without knowing how to
encode policy. It checks that executable RuleSpec atoms point to direct corpus
source text, accepted claim references declared by the module, or explicit
RuleSpec import exports. `proof-validate` can also run the cross-repo
source-claim validator so claim IDs must resolve to accepted, corpus-backed
claim records.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(frozen=True)
class ProofValidationResult:
    """Result for standalone RuleSpec proof validation."""

    passed: bool
    issues: list[str]
    atoms_checked: int
    proof_required: bool


PROOF_ATOM_KINDS = frozenset(
    {
        "amount",
        "condition",
        "definition",
        "default",
        "effective_period",
        "exception",
        "formula",
        "import",
        "ordering",
        "parameter",
        "parameter_table",
        "predicate",
        "table_cell",
        "unit",
    }
)
POLICY_RULE_KINDS = frozenset({"derived", "derived_relation", "parameter"})


def find_rulespec_proof_issues(content: str) -> list[str]:
    """Return proof-validation issues for a RuleSpec YAML document."""
    return validate_rulespec_proofs(content).issues


def validate_rulespec_proofs(
    content: str,
    *,
    validate_claim_records: bool = False,
    require_policy_proofs: bool = False,
) -> ProofValidationResult:
    """Validate explicit proof trees in a RuleSpec YAML document.

    Strict proof validation is enabled with:

    ```yaml
    module:
      proof_validation:
        required: true
    ```

    When strict mode is off, any proof blocks that are present are still checked.
    """
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError) as exc:
        return ProofValidationResult(
            passed=False,
            issues=[f"RuleSpec proof YAML parse failed: {exc}"],
            atoms_checked=0,
            proof_required=False,
        )

    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return ProofValidationResult(
            passed=True,
            issues=[],
            atoms_checked=0,
            proof_required=False,
        )

    module = payload.get("module")
    proof_required = require_policy_proofs or _module_requires_proofs(module)
    source_paths = _module_source_paths(module, payload)
    source_value_keys = _module_source_value_keys(module, payload)
    declared_claims = _module_source_claim_ids(module)

    rules = payload.get("rules")
    if not isinstance(rules, list):
        issues = _claim_record_issues(content, validate_claim_records)
        return ProofValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            atoms_checked=0,
            proof_required=proof_required,
        )

    issues: list[str] = []
    atoms_checked = 0
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        rule_name = str(rule.get("name") or f"rules[{index}]").strip() or (
            f"rules[{index}]"
        )
        proof = _rule_proof(rule)
        if proof is None:
            if proof_required and _is_policy_bearing_rule(rule):
                issues.append(
                    "Proof missing: "
                    f"rule `{rule_name}` is policy-bearing and must declare "
                    "`metadata.proof.atoms`."
                )
            continue
        rule_issues, rule_atom_count = _validate_rule_proof(
            rule_name=rule_name,
            proof=proof,
            source_paths=source_paths,
            source_value_keys=source_value_keys,
            declared_claims=declared_claims,
        )
        issues.extend(rule_issues)
        atoms_checked += rule_atom_count

    issues.extend(_claim_record_issues(content, validate_claim_records))
    return ProofValidationResult(
        passed=len(issues) == 0,
        issues=issues,
        atoms_checked=atoms_checked,
        proof_required=proof_required,
    )


def _claim_record_issues(content: str, enabled: bool) -> list[str]:
    if not enabled:
        return []
    from .validator_pipeline import find_source_claim_reference_issues

    return find_source_claim_reference_issues(content)


def _module_requires_proofs(module: Any) -> bool:
    if not isinstance(module, dict):
        return False
    proof_validation = module.get("proof_validation")
    if isinstance(proof_validation, dict):
        return proof_validation.get("required") is True
    return False


def _module_source_paths(module: Any, payload: dict[str, Any]) -> set[str]:
    source_verification = _source_verification_block(module, payload)
    if not isinstance(source_verification, dict):
        return set()

    paths: set[str] = set()
    single = str(source_verification.get("corpus_citation_path") or "").strip()
    if single:
        paths.add(single)
    raw_many = source_verification.get("corpus_citation_paths")
    if isinstance(raw_many, list):
        paths.update(str(item).strip() for item in raw_many if str(item).strip())
    return paths


def _module_source_value_keys(module: Any, payload: dict[str, Any]) -> set[str]:
    source_verification = _source_verification_block(module, payload)
    if not isinstance(source_verification, dict):
        return set()
    values = source_verification.get("values")
    if not isinstance(values, dict):
        return set()
    return {str(key) for key in values}


def _source_verification_block(module: Any, payload: dict[str, Any]) -> Any:
    if isinstance(module, dict) and isinstance(module.get("source_verification"), dict):
        return module["source_verification"]
    return payload.get("source_verification")


def _module_source_claim_ids(module: Any) -> set[str]:
    if not isinstance(module, dict):
        return set()
    raw_claims = module.get("source_claims")
    if not isinstance(raw_claims, list):
        return set()

    claim_ids: set[str] = set()
    for raw_claim in raw_claims:
        if isinstance(raw_claim, str):
            claim_id = raw_claim.strip()
        elif isinstance(raw_claim, dict):
            claim_id = str(raw_claim.get("id") or "").strip()
        else:
            claim_id = ""
        if claim_id:
            claim_ids.add(claim_id)
    return claim_ids


def _rule_proof(rule: dict[str, Any]) -> dict[str, Any] | None:
    metadata = rule.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("proof"), dict):
        return metadata["proof"]
    proof = rule.get("proof")
    if isinstance(proof, dict):
        return proof
    return None


def _is_policy_bearing_rule(rule: dict[str, Any]) -> bool:
    kind = str(rule.get("kind") or "").strip()
    if kind in POLICY_RULE_KINDS:
        return True
    if kind in {"data_relation", "source_relation"}:
        return False
    return bool(rule.get("versions"))


def _validate_rule_proof(
    *,
    rule_name: str,
    proof: dict[str, Any],
    source_paths: set[str],
    source_value_keys: set[str],
    declared_claims: set[str],
) -> tuple[list[str], int]:
    atoms = proof.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        return (
            [
                "Proof malformed: "
                f"rule `{rule_name}` must declare non-empty "
                "`metadata.proof.atoms`."
            ],
            0,
        )

    issues: list[str] = []
    atoms_checked = 0
    for index, atom in enumerate(atoms):
        label = f"rule `{rule_name}` proof atom {index}"
        if not isinstance(atom, dict):
            issues.append(f"Proof atom malformed: {label} must be a mapping.")
            continue
        atoms_checked += 1
        issues.extend(
            _validate_proof_atom(
                atom=atom,
                label=label,
                source_paths=source_paths,
                source_value_keys=source_value_keys,
                declared_claims=declared_claims,
            )
        )
    return issues, atoms_checked


def _validate_proof_atom(
    *,
    atom: dict[str, Any],
    label: str,
    source_paths: set[str],
    source_value_keys: set[str],
    declared_claims: set[str],
) -> list[str]:
    issues: list[str] = []
    path = str(atom.get("path") or "").strip()
    if not path:
        issues.append(f"Proof atom missing path: {label} must declare `path`.")

    kind = str(atom.get("kind") or "").strip()
    if kind not in PROOF_ATOM_KINDS:
        allowed = ", ".join(sorted(PROOF_ATOM_KINDS))
        issues.append(
            "Proof atom kind invalid: "
            f"{label} has kind `{kind or '<missing>'}`; allowed kinds are {allowed}."
        )

    proof_source_count = sum(
        1 for key in ("source", "claim", "import") if atom.get(key) is not None
    )
    if proof_source_count == 0:
        issues.append(
            "Proof atom missing evidence: "
            f"{label} must declare at least one of `source`, `claim`, or `import`."
        )

    if "source" in atom:
        issues.extend(
            _validate_source_proof_atom(
                source=atom.get("source"),
                label=label,
                kind=kind,
                source_paths=source_paths,
                source_value_keys=source_value_keys,
            )
        )
    if "claim" in atom:
        issues.extend(
            _validate_claim_proof_atom(
                claim=atom.get("claim"),
                label=label,
                declared_claims=declared_claims,
            )
        )
    if "import" in atom:
        issues.extend(_validate_import_proof_atom(atom.get("import"), label))

    return issues


def _validate_source_proof_atom(
    *,
    source: Any,
    label: str,
    kind: str,
    source_paths: set[str],
    source_value_keys: set[str],
) -> list[str]:
    if not isinstance(source, dict):
        return [f"Proof source malformed: {label} `source` must be a mapping."]

    issues: list[str] = []
    citation_path = str(source.get("corpus_citation_path") or "").strip()
    if not citation_path:
        issues.append(
            "Proof source missing corpus path: "
            f"{label} `source.corpus_citation_path` is required."
        )
    elif source_paths and citation_path not in source_paths:
        allowed = ", ".join(f"`{path}`" for path in sorted(source_paths))
        issues.append(
            "Proof source outside RuleSpec source: "
            f"{label} cites `{citation_path}`, but the module verifies against "
            f"{allowed}."
        )

    value_key = str(source.get("value_key") or "").strip()
    if value_key and source_value_keys and value_key not in source_value_keys:
        allowed = ", ".join(f"`{key}`" for key in sorted(source_value_keys))
        issues.append(
            "Proof source value key missing: "
            f"{label} cites `source.value_key: {value_key}`, but "
            f"`module.source_verification.values` only declares {allowed}."
        )

    table = source.get("table")
    if kind == "table_cell":
        if not isinstance(table, dict):
            issues.append(
                "Proof table provenance missing: "
                f"{label} is `table_cell` and must declare `source.table`."
            )
        else:
            missing = [
                field
                for field in ("header", "row", "column")
                if not str(table.get(field) or "").strip()
            ]
            if missing:
                issues.append(
                    "Proof table cell provenance incomplete: "
                    f"{label} `source.table` must declare "
                    + ", ".join(f"`{field}`" for field in missing)
                    + "."
                )
    elif kind == "parameter_table" and isinstance(table, dict):
        if not str(table.get("header") or "").strip():
            issues.append(
                "Proof table provenance incomplete: "
                f"{label} `source.table.header` is required for parameter tables."
            )
        has_row_key = bool(str(table.get("row_key") or table.get("row") or "").strip())
        has_column_key = bool(
            str(table.get("column_key") or table.get("column") or "").strip()
        )
        if not has_row_key or not has_column_key:
            issues.append(
                "Proof table provenance incomplete: "
                f"{label} parameter table proof should declare row and column keys."
            )

    return issues


def _validate_claim_proof_atom(
    *,
    claim: Any,
    label: str,
    declared_claims: set[str],
) -> list[str]:
    claim_id = ""
    if isinstance(claim, str):
        claim_id = claim.strip()
    elif isinstance(claim, dict):
        claim_id = str(claim.get("id") or "").strip()
    if not claim_id:
        return [
            "Proof claim malformed: "
            f"{label} `claim` must be a claim ID string or `{{id: ...}}` mapping."
        ]

    if not declared_claims:
        return [
            "Proof claim undeclared: "
            f"{label} references `{claim_id}`, but `module.source_claims` is empty."
        ]
    if claim_id not in declared_claims:
        return [
            "Proof claim outside declared claims: "
            f"{label} references `{claim_id}`, but it is not listed in "
            "`module.source_claims`."
        ]
    return []


def _validate_import_proof_atom(raw_import: Any, label: str) -> list[str]:
    if not isinstance(raw_import, dict):
        return [f"Proof import malformed: {label} `import` must be a mapping."]

    missing = [
        field
        for field in ("target", "output", "hash")
        if not str(raw_import.get(field) or "").strip()
    ]
    if missing:
        return [
            "Proof import contract incomplete: "
            f"{label} `import` must declare "
            + ", ".join(f"`{field}`" for field in missing)
            + "."
        ]

    hash_value = str(raw_import.get("hash") or "")
    if not hash_value.startswith("sha256:"):
        return [
            "Proof import hash invalid: "
            f"{label} `import.hash` must start with `sha256:`."
        ]
    return []


def source_proof_paths(content: str) -> list[str]:
    """Return explicit proof atom paths, useful for debugging and tests."""
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if not isinstance(payload, dict):
            return []
        rules = payload.get("rules")
        if not isinstance(rules, list):
            return []
        paths: list[str] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            proof = _rule_proof(rule)
            if not isinstance(proof, dict):
                continue
            atoms = proof.get("atoms")
            if not isinstance(atoms, list):
                continue
            for atom in atoms:
                if isinstance(atom, dict) and str(atom.get("path") or "").strip():
                    paths.append(str(atom["path"]))
        return paths
    return []
