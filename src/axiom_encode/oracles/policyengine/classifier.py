"""Pattern-driven PolicyEngine oracle classifier for RuleSpec encodings.

The classifier walks a `rulespec-us-<state>` repository, looks up each
output's name in the requested program's adapter catalog in `adapters.py`,
and emits a `us.yaml` mapping entry per output.

Single source of truth: each program's pattern catalog is the adapter list
used by its oracle/replay lane. Anything that adapter list maps to a
PolicyEngine variable, the classifier promotes to a `direct_variable` entry.
Anything else gets a per-rule `not_comparable` entry whose rationale
references the rule's RuleSpec `source:` field rather than a chapter-level
fig leaf.

Strict matching: a rule's name must equal one of the adapter's
`rule_names` (case-sensitive, no fragment match). A dtype guard prevents
a Judgment-typed predicate from being promoted to a Money-typed variable
even when names collide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from axiom_encode.oracles.policyengine.adapters import (
    PE_US_PROGRAM_VAR_ADAPTERS,
    PE_US_VAR_ADAPTERS,
    PolicyEngineUSVarAdapter,
)

# RuleSpec dtypes that align with money-amount PolicyEngine variables.
_MONEY_DTYPES = frozenset({"Money", "USD", "Integer", "Decimal", "Number"})
# RuleSpec dtypes that align with boolean-decision PolicyEngine variables.
_DECISION_DTYPES = frozenset({"Judgment", "Bool", "Boolean", "Decision"})
# RuleSpec dtypes that align with rate/ratio PolicyEngine variables.
_RATE_DTYPES = frozenset({"Rate", "Decimal", "Number", "Float"})

# Heuristic: PolicyEngine variable names ending in these tokens are typically
# monetary. Used only when the adapter does not record an explicit comparison.
_MONEY_VAR_SUFFIXES = (
    "_allotment",
    "_amount",
    "_contribution",
    "_deduction",
    "_allowance",
    "_income",
    "_income_pre_shelter",
)

_HEALTH_PROGRAMS = frozenset({"medicaid", "chip", "aca_ptc"})

_PROGRAM_TOKEN_RE = {
    token: re.compile(rf"(^|[^a-z0-9]){token}([^a-z0-9]|$)")
    for token in ("medicaid", "chip", "aca")
}


@dataclass(frozen=True)
class Classification:
    """One classification entry for `us.yaml` emission."""

    legal_id: str
    mapping_type: str  # "direct_variable" or "not_comparable"
    policyengine_variable: str | None
    entity: str | None
    period: str | None
    unit: str | None
    comparison: str | None
    rationale: str
    matched_adapter: str | None  # e.g., the rule_name that matched


def _build_rule_name_index(
    adapters: Iterable[PolicyEngineUSVarAdapter],
) -> dict[str, PolicyEngineUSVarAdapter]:
    """Build a {rule_name: adapter} lookup from the adapter catalog.

    When multiple adapters declare the same rule_name (rare), the first one
    in iteration order wins, matching Populace comparison precedence.
    """
    index: dict[str, PolicyEngineUSVarAdapter] = {}
    for adapter in adapters:
        for rule_name in adapter.rule_names:
            index.setdefault(rule_name, adapter)
    return index


def _adapter_unit_comparison(
    adapter: PolicyEngineUSVarAdapter,
) -> tuple[str | None, str]:
    """Infer (unit, comparison) for the adapter's PE variable.

    Heuristic until adapters carry explicit comparison metadata. Variables
    ending in money-like suffixes are treated as `comparison: money` with
    `unit: USD`; everything else is `comparison: decision`.
    """
    if adapter.unit is not None or adapter.comparison is not None:
        return (adapter.unit, adapter.comparison or "decision")
    pe_var = adapter.pe_var
    if pe_var.endswith(_MONEY_VAR_SUFFIXES):
        return ("USD", "money")
    return (None, "decision")


def _adapter_entity_period(
    adapter: PolicyEngineUSVarAdapter,
) -> tuple[str, str | None]:
    """Infer RuleSpec entity and period metadata for a PE adapter."""
    if adapter.entity:
        return (adapter.entity, adapter.period)
    entity = "spm_unit" if adapter.spm else "household"
    period = "month" if adapter.monthly else None
    return (entity, period)


def _classify_one(
    *,
    rule_name: str,
    legal_id: str,
    dtype: str | None,
    source_text: str | None,
    rule_index: dict[str, PolicyEngineUSVarAdapter],
) -> Classification:
    adapter = rule_index.get(rule_name)
    if adapter is None:
        return _not_comparable(legal_id, rule_name, source_text)

    unit, comparison = _adapter_unit_comparison(adapter)

    # dtype guard: protect against promoting a boolean predicate to a money
    # variable (or vice versa) when names happen to overlap.
    if dtype:
        if comparison == "money" and dtype not in _MONEY_DTYPES:
            return _not_comparable(legal_id, rule_name, source_text)
        if comparison in {"decision", "boolean"} and dtype not in _DECISION_DTYPES:
            return _not_comparable(legal_id, rule_name, source_text)
        if comparison == "rate" and dtype not in _RATE_DTYPES:
            return _not_comparable(legal_id, rule_name, source_text)

    entity, period = _adapter_entity_period(adapter)

    return Classification(
        legal_id=legal_id,
        mapping_type="direct_variable",
        policyengine_variable=adapter.pe_var,
        entity=entity,
        period=period,
        unit=unit,
        comparison=comparison,
        rationale=(
            f"State program output `{rule_name}` matches the PolicyEngine variable "
            f"`{adapter.pe_var}` via the PolicyEngine adapter catalog (rule_names "
            f"includes `{rule_name}`)."
        ),
        matched_adapter=rule_name,
    )


def _not_comparable(
    legal_id: str, rule_name: str, source_text: str | None
) -> Classification:
    excerpt = ""
    if source_text:
        excerpt = f" grounded against `{source_text.strip()[:160]}`"
    return Classification(
        legal_id=legal_id,
        mapping_type="not_comparable",
        policyengine_variable=None,
        entity=None,
        period=None,
        unit=None,
        comparison=None,
        rationale=(
            f"Source-specific intermediate output (rule `{rule_name}`,{excerpt}). "
            "Not present in the requested PolicyEngine adapter catalog. "
            "Promote to `direct_variable` by adding an "
            "adapter entry once a one-to-one PE target is identified."
        ),
        matched_adapter=None,
    )


def iter_rules_in_rulespec_file(
    path: Path,
) -> Iterable[tuple[str, str | None, str | None]]:
    """Yield (rule_name, source_text, dtype) for every executable rule."""
    try:
        payload = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return
    if not isinstance(payload, dict):
        return
    for rule in payload.get("rules", []) or []:
        if not isinstance(rule, dict):
            continue
        name = rule.get("name")
        if not name or rule.get("kind") not in {"derived", "parameter"}:
            continue
        source = rule.get("source")
        if isinstance(source, dict):
            source_text = source.get("text") or source.get("excerpt")
        elif isinstance(source, str):
            source_text = source
        else:
            source_text = None
        yield (name, source_text, rule.get("dtype"))


def _infer_program_from_legal_id(legal_id: str, *, rule_name: str = "") -> str:
    """Infer the program namespace for classifier filtering.

    This intentionally stays lightweight: it prevents a program-specific
    classify run from emitting not_comparable entries for unrelated outputs in
    a mixed RuleSpec repo. The coverage reporter has the fuller registry-aware
    program analysis.
    """
    lowered = legal_id.lower()
    rule = rule_name.lower()
    if _is_combined_health_source(lowered):
        return _infer_health_program_from_rule_name(rule)
    if _PROGRAM_TOKEN_RE["medicaid"].search(lowered):
        return "medicaid"
    if _PROGRAM_TOKEN_RE["chip"].search(lowered):
        return "chip"
    if (
        _PROGRAM_TOKEN_RE["aca"].search(lowered)
        or "premium_tax_credit" in lowered
        or lowered.startswith("us:statutes/26/36b")
    ):
        return "aca_ptc"
    if "hcpf" in lowered or "health" in lowered:
        return "health"
    if "snap" in lowered:
        return "snap"
    if lowered.startswith(("us:statutes/42/1382", "us:statutes/42/1382f")):
        return "ssi"
    if lowered.startswith("us:statutes/26/") or lowered.startswith(
        "us-co:statutes/39/"
    ):
        return "tax"
    return "unknown"


def _is_combined_health_source(lowered_legal_id: str) -> bool:
    return bool(
        "medicaid-chip" in lowered_legal_id
        or "medicaid_chip" in lowered_legal_id
        or (
            _PROGRAM_TOKEN_RE["medicaid"].search(lowered_legal_id)
            and _PROGRAM_TOKEN_RE["chip"].search(lowered_legal_id)
            and ("bhp" in lowered_legal_id or "eligibility" in lowered_legal_id)
        )
    )


def _infer_health_program_from_rule_name(rule_name: str) -> str:
    if _PROGRAM_TOKEN_RE["chip"].search(rule_name):
        return "chip"
    if _PROGRAM_TOKEN_RE["medicaid"].search(rule_name):
        return "medicaid"
    if _PROGRAM_TOKEN_RE["aca"].search(rule_name) or "premium_tax_credit" in rule_name:
        return "aca_ptc"
    return "health"


def _program_matches_classify_run(
    *,
    requested_program: str,
    legal_id: str,
    rule_name: str,
    rule_index: dict[str, PolicyEngineUSVarAdapter],
) -> bool:
    """Return whether a RuleSpec output belongs in this program classify run."""
    if requested_program == "snap":
        # Preserve the historical SNAP classifier behavior for existing users
        # and tests. New multi-program catalogs are filtered below.
        return True

    inferred = _infer_program_from_legal_id(legal_id, rule_name=rule_name)
    if requested_program == "health":
        return inferred in _HEALTH_PROGRAMS | {"health"} or rule_name in rule_index
    if inferred == requested_program:
        return True
    return inferred == "health" and rule_name in rule_index


def classify_rulespec_repo(
    *,
    repo_root: Path,
    jurisdiction: str,
    program: str = "snap",
    adapters: Iterable[PolicyEngineUSVarAdapter] | None = None,
) -> list[Classification]:
    """Walk a `rulespec-us-<state>` repo and classify every executable output."""
    # Resolve the adapter catalog at call time (not at function-def time) so
    # callers and tests can inject alternatives via either the explicit
    # argument or by patching `PE_US_VAR_ADAPTERS` on this module.
    if adapters is None:
        if program == "snap":
            adapters = PE_US_VAR_ADAPTERS
        else:
            try:
                adapters = PE_US_PROGRAM_VAR_ADAPTERS[program]
            except KeyError as exc:
                supported = ", ".join(sorted(PE_US_PROGRAM_VAR_ADAPTERS))
                raise NotImplementedError(
                    f"Program {program!r} has no adapter catalog. "
                    f"Supported programs: {supported}."
                ) from exc
    rule_index = _build_rule_name_index(adapters)

    classifications: list[Classification] = []
    for path in sorted(repo_root.rglob("*.yaml")):
        if path.name.endswith(".test.yaml"):
            continue
        rel = path.relative_to(repo_root)
        rel_no_ext = str(rel).removesuffix(".yaml")
        file_legal_prefix = f"{jurisdiction}:{rel_no_ext}"
        for rule_name, source_text, dtype in iter_rules_in_rulespec_file(path):
            legal_id = f"{file_legal_prefix}#{rule_name}"
            if not _program_matches_classify_run(
                requested_program=program,
                legal_id=legal_id,
                rule_name=rule_name,
                rule_index=rule_index,
            ):
                continue
            classifications.append(
                _classify_one(
                    rule_name=rule_name,
                    legal_id=legal_id,
                    dtype=dtype,
                    source_text=source_text,
                    rule_index=rule_index,
                )
            )
    return classifications


def classification_to_yaml_block(c: Classification, *, program: str = "snap") -> str:
    """Render a Classification as a `us.yaml`-shaped mapping entry."""
    lines = [f"  - legal_id: {c.legal_id}"]
    lines.append("    country: us")
    lines.append(f"    program: {program}")
    lines.append(f"    mapping_type: {c.mapping_type}")
    if c.policyengine_variable is not None:
        lines.append(f"    policyengine_variable: {c.policyengine_variable}")
    if c.entity is not None:
        lines.append(f"    entity: {c.entity}")
    if c.period is not None:
        lines.append(f"    period: {c.period}")
    if c.unit is not None:
        lines.append(f"    unit: {c.unit}")
    if c.comparison is not None:
        lines.append(f"    comparison: {c.comparison}")
    if c.rationale:
        rendered = yaml.safe_dump(
            c.rationale.replace("\n", " ").strip(),
            default_style='"',
            width=10_000,
        ).strip()
        lines.append(f"    rationale: {rendered}")
    return "\n".join(lines)


def emit_us_yaml_block(
    classifications: Iterable[Classification], *, program: str = "snap"
) -> str:
    """Emit a YAML fragment ready to splice into `us.yaml` under `mappings:`."""
    return "\n\n".join(
        classification_to_yaml_block(c, program=program) for c in classifications
    )


__all__ = [
    "Classification",
    "classify_rulespec_repo",
    "classification_to_yaml_block",
    "emit_us_yaml_block",
    "iter_rules_in_rulespec_file",
]
