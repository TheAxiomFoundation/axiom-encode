"""RuleSpec proof-tree validation.

This module intentionally validates proof structure without knowing how to
encode policy. It checks that executable RuleSpec atoms point to direct corpus
source text, accepted claim references declared by the module, or explicit
RuleSpec import exports. `proof-validate` can also run the cross-repo
source-claim validator so claim IDs must resolve to accepted, corpus-backed
claim records.

It also derives *money proof obligations* (see
``find_missing_money_proof_atoms``). Where explicit proof validation is
per-file opt-in via ``module.proof_validation.required: true``, the money-atom
derivation is unconditional: every policy-bearing monetary value (currency
parameters, currency parameter-table cells, and currency literals inside
derived-rule formulas) is expected to carry a proof atom that cites a
provision. This closes the enforcement gap where a repository can ship
monetary parameters with no proof atoms at all because no file opts in.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Money proof obligations
#
# Strict proof validation (``validate_rulespec_proofs``) is per-file opt-in via
# ``module.proof_validation.required: true``. That leaves a gap: a repository
# with zero opted-in files can ship monetary parameters carrying no proof atom
# at all. ``find_missing_money_proof_atoms`` closes the money-scoped part of
# that gap by deriving obligations directly from the compiled RuleSpec: every
# policy-bearing monetary value must have a proof atom at a matching path whose
# source cites a provision (direct corpus source, accepted claim, or import).
# ---------------------------------------------------------------------------

# Currency units that mark a value as monetary. Matched case-insensitively.
MONEY_UNITS = frozenset({"eur", "usd", "gbp", "cad", "aud", "nzd", "chf", "jpy"})
# dtypes that mark a rule as monetary regardless of unit.
MONEY_DTYPES = frozenset({"money", "currency"})


@dataclass(frozen=True)
class MoneyProofObligation:
    """A single derived money proof obligation for one RuleSpec location."""

    rule_name: str
    path: str
    kind: str
    reason: str
    satisfied: bool


@dataclass
class MoneyAtomReport:
    """Per-module result of money proof-obligation derivation."""

    obligations: list[MoneyProofObligation] = field(default_factory=list)

    @property
    def missing(self) -> list[MoneyProofObligation]:
        return [
            obligation for obligation in self.obligations if not obligation.satisfied
        ]

    @property
    def missing_count(self) -> int:
        return len(self.missing)

    @property
    def obligation_count(self) -> int:
        return len(self.obligations)


def find_missing_money_proof_atoms(content: str) -> MoneyAtomReport:
    """Derive money proof obligations for a RuleSpec YAML document.

    An obligation is *satisfied* when the rule declares a proof atom whose
    ``path`` matches the money-bearing location (``versions[i].formula`` or
    ``versions[i].values``) and whose evidence cites a provision (a ``source``
    with a ``corpus_citation_path``, a declared ``claim``, or an ``import``).

    Non-monetary values, and monetary formulas whose only numeric literals are
    structural sentinels (``{-1, 0, 1, 2, 3}`` and half-up ``0.5``), create no
    obligation. This mirrors the encoder's own numeric-grounding exclusions so
    the money-atom surface never diverges from what grounding treats as a
    policy number.
    """
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return MoneyAtomReport()

    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return MoneyAtomReport()

    rules = payload.get("rules")
    if not isinstance(rules, list):
        return MoneyAtomReport()

    selector_table_keys = _selector_table_keys(rules)

    report = MoneyAtomReport()
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        if not _is_policy_bearing_rule(rule):
            continue
        if not _rule_is_monetary(rule):
            continue

        rule_name = str(rule.get("name") or f"rules[{index}]").strip() or (
            f"rules[{index}]"
        )
        satisfied_paths = _cited_proof_atom_paths(rule)
        selector_keys = (
            selector_table_keys.get(rule_name)
            if _rule_is_structural_selector(rule)
            else None
        )

        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        for version_index, version in enumerate(versions):
            if not isinstance(version, dict):
                continue
            for path, kind, reason in _money_locations_for_version(
                version,
                version_index=version_index,
                selector_keys=selector_keys,
            ):
                report.obligations.append(
                    MoneyProofObligation(
                        rule_name=rule_name,
                        path=path,
                        kind=kind,
                        reason=reason,
                        satisfied=_normalize_atom_path(path) in satisfied_paths,
                    )
                )
    return report


def _rule_is_monetary(rule: dict[str, Any]) -> bool:
    dtype = str(rule.get("dtype") or "").strip().lower()
    if dtype in MONEY_DTYPES:
        return True
    unit = str(rule.get("unit") or "").strip().lower()
    return unit in MONEY_UNITS


def _rule_is_structural_selector(rule: dict[str, Any]) -> bool:
    """Reuse the encoder's structural-selector test for formula extraction."""
    from .validator_pipeline import _is_structural_selector_rule

    return bool(_is_structural_selector_rule(rule))


def _selector_table_keys(rules: list[Any]) -> dict[str, set[str]]:
    """Reuse the encoder's index-selector key map for structural exclusions."""
    from .validator_pipeline import _rulespec_index_selector_keys

    return _rulespec_index_selector_keys(rules)


def _money_locations_for_version(
    version: dict[str, Any],
    *,
    version_index: int,
    selector_keys: set[str] | None,
) -> list[tuple[str, str, str]]:
    """Return (path, atom_kind, reason) for each money-bearing spot in a version.

    ``atom_kind`` is the proof-atom ``kind`` the location is expected to use so
    error messages can point the encoder at the right shape.
    """
    from .validator_pipeline import (
        GROUNDING_ALLOWED_VALUES,
        _extract_formula_grounding_values,
        _numeric_rule_value,
    )

    locations: list[tuple[str, str, str]] = []

    formula = version.get("formula")
    formula_has_money_literal = False
    if isinstance(formula, (int, float)) and not isinstance(formula, bool):
        formula_has_money_literal = float(formula) not in GROUNDING_ALLOWED_VALUES
    elif isinstance(formula, str):
        formula_has_money_literal = bool(
            _extract_formula_grounding_values(
                1,
                formula,
                structural_selector_keys=selector_keys,
            )
        )
    if formula_has_money_literal:
        locations.append(
            (
                f"versions[{version_index}].formula",
                "parameter",
                "monetary formula carries a policy numeric literal",
            )
        )

    table_values = version.get("values")
    if isinstance(table_values, dict):
        for cell_value in table_values.values():
            extracted = _numeric_rule_value(cell_value)
            if extracted is None:
                continue
            _, value = extracted
            if value in GROUNDING_ALLOWED_VALUES:
                continue
            locations.append(
                (
                    f"versions[{version_index}].values",
                    "parameter_table",
                    "monetary parameter table carries policy cell values",
                )
            )
            break

    return locations


def _cited_proof_atom_paths(rule: dict[str, Any]) -> set[str]:
    """Return normalized proof-atom paths that cite a provision.

    A proof atom counts only when it names a path AND carries at least one of
    ``source`` (with a ``corpus_citation_path``), ``claim``, or ``import``. An
    atom with a path but no evidence does not satisfy a money obligation.
    """
    proof = _rule_proof(rule)
    if not isinstance(proof, dict):
        return set()
    atoms = proof.get("atoms")
    if not isinstance(atoms, list):
        return set()

    paths: set[str] = set()
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        path = str(atom.get("path") or "").strip()
        if not path:
            continue
        if _atom_cites_provision(atom):
            paths.add(_normalize_atom_path(path))
    return paths


def _atom_cites_provision(atom: dict[str, Any]) -> bool:
    source = atom.get("source")
    if (
        isinstance(source, dict)
        and str(source.get("corpus_citation_path") or "").strip()
    ):
        return True
    claim = atom.get("claim")
    if isinstance(claim, str) and claim.strip():
        return True
    if isinstance(claim, dict) and str(claim.get("id") or "").strip():
        return True
    raw_import = atom.get("import")
    if isinstance(raw_import, dict) and str(raw_import.get("target") or "").strip():
        return True
    return False


_ATOM_PATH_INDEX_PATTERN = re.compile(r"\[\s*(\d+)\s*\]")


def _normalize_atom_path(path: str) -> str:
    """Normalize a proof-atom path for matching.

    A ``versions[0].values`` atom covers the whole table, so cell suffixes such
    as ``versions[0].values.household_size_1`` are folded to the table path. A
    bare ``versions.formula`` (no index) is treated as ``versions[0].formula``
    so authors are not forced to index a single-version rule.
    """
    normalized = re.sub(r"\s+", "", str(path)).strip().rstrip(".")
    normalized = _ATOM_PATH_INDEX_PATTERN.sub(r"[\1]", normalized)
    # Fold a bare `versions.<field>` to `versions[0].<field>`.
    normalized = re.sub(r"^versions\.", "versions[0].", normalized)
    # Fold table-cell suffixes onto the table path.
    match = re.match(r"^(versions\[\d+\]\.values)\b", normalized)
    if match:
        return match.group(1)
    match = re.match(r"^(versions\[\d+\]\.formula)\b", normalized)
    if match:
        return match.group(1)
    return normalized


# ---------------------------------------------------------------------------
# Ratchet file support
#
# A repository burns its money-atom debt down over time. The ratchet file
# records the allowance so CI can fail on any *new* atom-less monetary value
# while tolerating the known backlog. Two shapes are accepted:
#
#   total_allowed: 217          # a single repo-wide budget, OR
#   paths:                      # a per-file budget (path relative to repo root)
#     be/statutes/x.yaml: 12
#     be/statutes/y.yaml: 3
#
# When both are present, per-path budgets take precedence for listed files and
# ``total_allowed`` covers everything else. An absent ratchet file means a
# strict zero allowance.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoneyAtomRatchet:
    """Parsed money-atom ratchet allowances."""

    total_allowed: int | None
    per_path: dict[str, int]

    @classmethod
    def empty(cls) -> MoneyAtomRatchet:
        return cls(total_allowed=None, per_path={})


def load_money_atom_ratchet(text: str) -> MoneyAtomRatchet:
    """Parse a money-atom ratchet YAML document.

    Raises ``ValueError`` for a malformed ratchet so CI fails loudly rather
    than silently treating a typo as a zero allowance.
    """
    payload = yaml.safe_load(text)
    if payload is None:
        return MoneyAtomRatchet.empty()
    if not isinstance(payload, dict):
        raise ValueError("money-atom ratchet must be a mapping")

    total_allowed: int | None = None
    if "total_allowed" in payload:
        raw_total = payload.get("total_allowed")
        if isinstance(raw_total, bool) or not isinstance(raw_total, int):
            raise ValueError("`total_allowed` must be an integer")
        if raw_total < 0:
            raise ValueError("`total_allowed` must not be negative")
        total_allowed = raw_total

    per_path: dict[str, int] = {}
    raw_paths = payload.get("paths")
    if raw_paths is not None:
        if not isinstance(raw_paths, dict):
            raise ValueError("`paths` must be a mapping of file path to integer")
        for key, value in raw_paths.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"`paths.{key}` must be an integer")
            if value < 0:
                raise ValueError(f"`paths.{key}` must not be negative")
            per_path[str(key)] = value

    return MoneyAtomRatchet(total_allowed=total_allowed, per_path=per_path)


@dataclass
class MoneyAtomFileResult:
    """Money-atom result for a single file, keyed by its display path."""

    path: str
    report: MoneyAtomReport

    @property
    def missing_count(self) -> int:
        return self.report.missing_count

    @property
    def obligation_count(self) -> int:
        return self.report.obligation_count


@dataclass
class MoneyAtomRun:
    """Aggregate money-atom result across a set of files, with ratchet verdict."""

    files: list[MoneyAtomFileResult]
    ratchet: MoneyAtomRatchet
    over_budget_paths: dict[str, tuple[int, int]] = field(default_factory=dict)
    total_over_budget: tuple[int, int] | None = None

    @property
    def total_missing(self) -> int:
        return sum(file.missing_count for file in self.files)

    @property
    def total_obligations(self) -> int:
        return sum(file.obligation_count for file in self.files)

    @property
    def passed(self) -> bool:
        return not self.over_budget_paths and self.total_over_budget is None


def evaluate_money_atoms(
    files: list[tuple[str, str]],
    ratchet: MoneyAtomRatchet | None = None,
) -> MoneyAtomRun:
    """Run the money-atom check across ``(display_path, content)`` pairs.

    The ratchet verdict works as follows. Any file with an explicit per-path
    budget is checked against that budget individually. All remaining files'
    missing counts are summed and checked against ``total_allowed`` (0 when the
    ratchet omits it). This keeps a single repo-wide budget simple while still
    letting a repo pin per-file allowances where that is clearer.
    """
    ratchet = ratchet or MoneyAtomRatchet.empty()

    file_results = [
        MoneyAtomFileResult(
            path=display_path, report=find_missing_money_proof_atoms(content)
        )
        for display_path, content in files
    ]

    over_budget_paths: dict[str, tuple[int, int]] = {}
    untracked_missing = 0
    for result in file_results:
        if result.path in ratchet.per_path:
            allowed = ratchet.per_path[result.path]
            if result.missing_count > allowed:
                over_budget_paths[result.path] = (result.missing_count, allowed)
        else:
            untracked_missing += result.missing_count

    total_over_budget: tuple[int, int] | None = None
    total_budget = ratchet.total_allowed if ratchet.total_allowed is not None else 0
    if untracked_missing > total_budget:
        total_over_budget = (untracked_missing, total_budget)

    return MoneyAtomRun(
        files=file_results,
        ratchet=ratchet,
        over_budget_paths=over_budget_paths,
        total_over_budget=total_over_budget,
    )


def emit_money_atom_ratchet(files: list[tuple[str, str]]) -> str:
    """Return a seed ratchet YAML capturing the current missing-atom backlog.

    The seed uses a single ``total_allowed`` equal to the current repo-wide
    missing count, plus a commented per-path breakdown so the backlog is
    visible and can be burned down file by file.
    """
    file_results = [
        (display_path, find_missing_money_proof_atoms(content).missing_count)
        for display_path, content in files
    ]
    per_path = {display_path: count for display_path, count in file_results if count}
    total = sum(per_path.values())

    lines = [
        "# Money-atom proof-obligation ratchet.",
        "#",
        "# Each count is the number of monetary values (currency parameters,",
        "# currency parameter-table cells, and currency literals in derived",
        "# formulas) that still lack a proof atom citing a provision.",
        "# `axiom-encode proof-validate --require-money-atoms` fails when the",
        "# untracked missing count exceeds `total_allowed`. Burn this down; do",
        "# not raise it. Regenerate with `--emit-ratchet`.",
        f"total_allowed: {total}",
    ]
    if per_path:
        lines.append("# Current per-file backlog (informational):")
        for display_path in sorted(per_path):
            lines.append(f"#   {display_path}: {per_path[display_path]}")
    return "\n".join(lines) + "\n"
