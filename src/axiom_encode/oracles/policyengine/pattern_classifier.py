"""Pattern-based oracle coverage classifier for state SNAP RuleSpec encodings.

The classifier walks a rulespec-us-<state> repository, looks at each output's
name + source text, and proposes a PolicyEngine oracle coverage entry. The
result is a YAML block suitable for appending to
`src/axiom_encode/oracles/policyengine/mappings/us.yaml`.

The pattern list captures the high-confidence federal-aligned SNAP outputs
(standard deduction, shelter deduction, etc.). Anything that does not match a
pattern is classified as `not_comparable` with a per-rule rationale pulled from
the rule's `source:` field — i.e., the rationale is the actual source text the
encoder grounded against, not a chapter-level fig leaf.

This lets future state encodings get the comparable mappings for free, and
makes the residual `not_comparable` entries informative enough that a later
Codex/agent pass can promote them to direct_variable / parameter_value based on
the rule-specific rationale, without having to re-read the source corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class Pattern:
    """One rule-name pattern → PolicyEngine mapping."""

    rule_pattern: re.Pattern[str]
    mapping_type: str  # direct_variable | parameter_value
    policyengine_variable: str | None = None
    policyengine_parameter: str | None = None
    entity: str = "spm_unit"
    period: str = "month"
    unit: str | None = None
    comparison: str | None = None
    rationale: str = ""


# High-confidence SNAP patterns.
#
# Patterns are intentionally strict: the rule name must BE the canonical name,
# not contain it as a fragment. A loose match would falsely classify predicates
# like `arrearage_payments_allowed_in_child_support_deduction` as if they were
# the deduction itself.
#
# A pattern also requires the rule's `dtype` to align with the PolicyEngine
# variable's natural type (Money for $ amounts, Judgment for booleans), since
# state encodings frequently use the same word in both a predicate name and an
# amount name. The `dtype` check is performed by `classify_output`, not the
# regex.
#
# Order matters — the first match wins. More specific patterns precede more
# general ones.
SNAP_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        rule_pattern=re.compile(r"^snap_standard_deduction$|^standard_deduction$|^standard_deduction_amount$"),
        mapping_type="direct_variable",
        policyengine_variable="snap_standard_deduction",
        unit="USD",
        comparison="money",
        rationale="State SNAP standard deduction is the same household-level standard deduction PolicyEngine exposes as snap_standard_deduction.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^snap_earned_income_deduction$|^earned_income_deduction$|^earned_income_deduction_amount$"),
        mapping_type="direct_variable",
        policyengine_variable="snap_earned_income_deduction",
        unit="USD",
        comparison="money",
        rationale="State SNAP earned-income deduction is the same household-level deduction PolicyEngine exposes as snap_earned_income_deduction.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^snap_excess_medical_expense_deduction$|^excess_medical_expense_deduction$|^medical_expense_deduction$"),
        mapping_type="direct_variable",
        policyengine_variable="snap_excess_medical_expense_deduction",
        unit="USD",
        comparison="money",
        rationale="State SNAP excess medical expense deduction is the same household-level deduction PolicyEngine exposes as snap_excess_medical_expense_deduction.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^snap_excess_shelter_expense_deduction$|^excess_shelter_expense_deduction$|^excess_shelter_deduction$|^shelter_deduction$"),
        mapping_type="direct_variable",
        policyengine_variable="snap_excess_shelter_expense_deduction",
        unit="USD",
        comparison="money",
        rationale="State SNAP excess shelter expense deduction is the same household-level deduction PolicyEngine exposes as snap_excess_shelter_expense_deduction.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^snap_dependent_care_deduction$|^dependent_care_deduction$"),
        mapping_type="direct_variable",
        policyengine_variable="snap_dependent_care_deduction",
        unit="USD",
        comparison="money",
        rationale="State SNAP dependent care deduction matches the household-level deduction PolicyEngine exposes as snap_dependent_care_deduction.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^snap_child_support_gross_income_deduction$|^child_support_deduction$|^child_support_gross_income_deduction$"),
        mapping_type="direct_variable",
        policyengine_variable="snap_child_support_gross_income_deduction",
        unit="USD",
        comparison="money",
        rationale="State SNAP child-support gross-income deduction matches the household-level deduction PolicyEngine exposes as snap_child_support_gross_income_deduction.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^snap_standard_utility_allowance$|^standard_utility_allowance$|^sua$"),
        mapping_type="direct_variable",
        policyengine_variable="snap_standard_utility_allowance",
        unit="USD",
        comparison="money",
        rationale="State SNAP standard utility allowance is the same household-level allowance PolicyEngine exposes as snap_standard_utility_allowance.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^snap_limited_utility_allowance$|^limited_utility_allowance$|^lua$"),
        mapping_type="direct_variable",
        policyengine_variable="snap_limited_utility_allowance",
        unit="USD",
        comparison="money",
        rationale="State SNAP limited utility allowance is the same household-level allowance PolicyEngine exposes as snap_limited_utility_allowance.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^snap_individual_utility_allowance$|^individual_utility_allowance$"),
        mapping_type="direct_variable",
        policyengine_variable="snap_individual_utility_allowance",
        unit="USD",
        comparison="money",
        rationale="State SNAP individual utility allowance is the same household-level allowance PolicyEngine exposes as snap_individual_utility_allowance.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^meets_snap_gross_income_test$|^meets_gross_income_test$"),
        mapping_type="direct_variable",
        policyengine_variable="meets_snap_gross_income_test",
        unit=None,
        comparison="decision",
        rationale="State SNAP gross-income eligibility test matches the household-level test PolicyEngine exposes as meets_snap_gross_income_test.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^meets_snap_net_income_test$|^meets_net_income_test$"),
        mapping_type="direct_variable",
        policyengine_variable="meets_snap_net_income_test",
        unit=None,
        comparison="decision",
        rationale="State SNAP net-income eligibility test matches the household-level test PolicyEngine exposes as meets_snap_net_income_test.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^meets_snap_asset_test$|^meets_asset_test$"),
        mapping_type="direct_variable",
        policyengine_variable="meets_snap_asset_test",
        unit=None,
        comparison="decision",
        rationale="State SNAP asset eligibility test matches the household-level test PolicyEngine exposes as meets_snap_asset_test.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^snap_normal_allotment$|^normal_allotment$|^monthly_allotment$|^monthly_snap_allotment$"),
        mapping_type="direct_variable",
        policyengine_variable="snap_normal_allotment",
        unit="USD",
        comparison="money",
        rationale="State SNAP normal monthly allotment matches the household-level allotment PolicyEngine exposes as snap_normal_allotment.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^snap_max_allotment$|^maximum_allotment$|^max_allotment$"),
        mapping_type="direct_variable",
        policyengine_variable="snap_max_allotment",
        unit="USD",
        comparison="money",
        rationale="State SNAP maximum allotment matches the household-level cap PolicyEngine exposes as snap_max_allotment.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^snap_min_allotment$|^minimum_allotment$|^min_allotment$"),
        mapping_type="direct_variable",
        policyengine_variable="snap_min_allotment",
        unit="USD",
        comparison="money",
        rationale="State SNAP minimum allotment matches the household-level floor PolicyEngine exposes as snap_min_allotment.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^is_snap_eligible$|^snap_eligible$|^household_snap_eligible$"),
        mapping_type="direct_variable",
        policyengine_variable="is_snap_eligible",
        unit=None,
        comparison="decision",
        rationale="State SNAP eligibility decision matches the household-level decision PolicyEngine exposes as is_snap_eligible.",
    ),
    Pattern(
        rule_pattern=re.compile(r"^snap_expected_contribution$|^expected_contribution$"),
        mapping_type="direct_variable",
        policyengine_variable="snap_expected_contribution",
        unit="USD",
        comparison="money",
        rationale="State SNAP expected contribution matches the household-level expected contribution PolicyEngine exposes as snap_expected_contribution.",
    ),
)


# When a pattern would map to a Money/USD comparison, the rule's dtype must
# align (Money or Integer). Boolean predicates with names like
# `arrearage_payments_allowed_in_child_support_deduction` should not be matched
# to the dollar-amount variable.
_MONEY_DTYPES = {"Money", "USD", "Integer", "Decimal", "Number"}
_DECISION_DTYPES = {"Judgment", "Bool", "Boolean", "Decision"}


@dataclass(frozen=True)
class Classification:
    """One classification entry for us.yaml emission."""

    legal_id: str
    mapping_type: str
    policyengine_variable: str | None
    policyengine_parameter: str | None
    entity: str | None
    period: str | None
    unit: str | None
    comparison: str | None
    rationale: str
    matched_pattern: str | None


def classify_output(
    *,
    rule_name: str,
    legal_id: str,
    source_text: str | None,
    dtype: str | None = None,
    patterns: Iterable[Pattern] = SNAP_PATTERNS,
) -> Classification:
    """Classify one output: return Classification with patterns applied.

    `dtype` is the rule's RuleSpec dtype (e.g., "Money", "Judgment"). When a
    pattern's `comparison` is "money", the rule's dtype must be a money-like
    type; when "decision", the rule's dtype must be a boolean-like type. This
    prevents predicates and amounts from matching the same pattern.
    """
    for pattern in patterns:
        if not pattern.rule_pattern.match(rule_name):
            continue
        # dtype guard: reject pattern match when the rule's dtype clearly
        # disagrees with the pattern's natural comparison kind.
        if dtype:
            if pattern.comparison == "money" and dtype not in _MONEY_DTYPES:
                continue
            if pattern.comparison == "decision" and dtype not in _DECISION_DTYPES:
                continue
        return Classification(
            legal_id=legal_id,
            mapping_type=pattern.mapping_type,
            policyengine_variable=pattern.policyengine_variable,
            policyengine_parameter=pattern.policyengine_parameter,
            entity=pattern.entity,
            period=pattern.period,
            unit=pattern.unit,
            comparison=pattern.comparison,
            rationale=pattern.rationale,
            matched_pattern=pattern.rule_pattern.pattern,
        )
    # Fallback: not_comparable with per-rule rationale.
    rationale = (
        f"Source-specific intermediate output (rule `{rule_name}`); "
        + (
            f"grounded against {source_text}. "
            if source_text
            else ""
        )
        + "PolicyEngine does not expose an aligned variable; promote to direct_variable / parameter_value if a one-to-one PE target is identified."
    )
    return Classification(
        legal_id=legal_id,
        mapping_type="not_comparable",
        policyengine_variable=None,
        policyengine_parameter=None,
        entity=None,
        period=None,
        unit=None,
        comparison=None,
        rationale=rationale,
        matched_pattern=None,
    )


def classification_to_yaml_block(c: Classification, *, program: str = "snap") -> str:
    """Render a Classification as a us.yaml entry."""
    lines = [f"  - legal_id: {c.legal_id}"]
    lines.append(f"    country: us")
    lines.append(f"    program: {program}")
    lines.append(f"    mapping_type: {c.mapping_type}")
    if c.policyengine_variable is not None:
        lines.append(f"    policyengine_variable: {c.policyengine_variable}")
    if c.policyengine_parameter is not None:
        lines.append(f"    policyengine_parameter: {c.policyengine_parameter}")
    if c.entity is not None:
        lines.append(f"    entity: {c.entity}")
    if c.period is not None:
        lines.append(f"    period: {c.period}")
    if c.unit is not None:
        lines.append(f"    unit: {c.unit}")
    if c.comparison is not None:
        lines.append(f"    comparison: {c.comparison}")
    if c.rationale:
        # YAML scalar safety: always emit as a quoted scalar via yaml.safe_dump
        # to escape colons, hashes, brackets, and other reserved characters.
        rationale_one_line = c.rationale.replace("\n", " ").strip()
        # `yaml.safe_dump` emits "value\n"; strip the trailing newline.
        rendered = yaml.safe_dump(
            rationale_one_line, default_style='"', width=10_000
        ).strip()
        lines.append(f"    rationale: {rendered}")
    return "\n".join(lines)


def iter_rules_in_rulespec_file(path: Path) -> Iterable[tuple[str, str | None, str | None]]:
    """Yield (rule_name, source_text, dtype) for every executable rule in a RuleSpec file."""
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        return
    for rule in payload.get("rules", []) or []:
        if not isinstance(rule, dict):
            continue
        name = rule.get("name")
        if not name:
            continue
        kind = rule.get("kind")
        # Executable outputs: derived, parameter.
        if kind not in {"derived", "parameter"}:
            continue
        source = rule.get("source")
        if isinstance(source, dict):
            source_text = source.get("text") or source.get("excerpt")
        else:
            source_text = source if isinstance(source, str) else None
        dtype = rule.get("dtype")
        yield (name, source_text, dtype)


def classify_rulespec_repo(
    *,
    repo_root: Path,
    jurisdiction: str,
    program: str = "snap",
) -> list[Classification]:
    """Walk a rulespec-us-<state> repository and classify every output."""
    classifications: list[Classification] = []
    for path in sorted(repo_root.rglob("*.yaml")):
        if path.name.endswith(".test.yaml"):
            continue
        # Build legal-id prefix from path relative to repo root.
        rel = path.relative_to(repo_root)
        # rel: regulations/106-cmr/364/400/block-1.yaml
        rel_no_ext = str(rel).removesuffix(".yaml")
        file_legal_prefix = f"{jurisdiction}:{rel_no_ext}"
        for rule_name, source_text, dtype in iter_rules_in_rulespec_file(path):
            legal_id = f"{file_legal_prefix}#{rule_name}"
            classifications.append(
                classify_output(
                    rule_name=rule_name,
                    legal_id=legal_id,
                    source_text=source_text,
                    dtype=dtype,
                )
            )
    return classifications


def emit_us_yaml_block(classifications: Iterable[Classification], *, program: str = "snap") -> str:
    """Emit a YAML fragment ready to splice into us.yaml under mappings:."""
    return "\n\n".join(classification_to_yaml_block(c, program=program) for c in classifications)


__all__ = [
    "Classification",
    "Pattern",
    "SNAP_PATTERNS",
    "classify_output",
    "classify_rulespec_repo",
    "classification_to_yaml_block",
    "emit_us_yaml_block",
]
