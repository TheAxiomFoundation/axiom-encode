"""Pattern-driven PolicyEngine oracle classifier for RuleSpec encodings.

The classifier walks a `rulespec-us-<state>` repository, looks up each
output's name in `PE_US_VAR_ADAPTERS` (the canonical comparison-time
adapter list in `adapters.py`), and emits a `us.yaml` mapping entry per
output.

Single source of truth: the pattern catalog is `PE_US_VAR_ADAPTERS` —
the same list used by ECPS comparison at test time. Anything that adapter
list maps to a PolicyEngine variable, the classifier promotes to a
`direct_variable` entry. Anything else gets a per-rule `not_comparable`
entry whose rationale references the rule's RuleSpec `source:` field
rather than a chapter-level fig leaf.

Strict matching: a rule's name must equal one of the adapter's
`rule_names` (case-sensitive, no fragment match). A dtype guard prevents
a Judgment-typed predicate from being promoted to a Money-typed variable
even when names collide.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from axiom_encode.oracles.policyengine.adapters import (
    PE_US_VAR_ADAPTERS,
    PolicyEngineUSVarAdapter,
)


# RuleSpec dtypes that align with money-amount PolicyEngine variables.
_MONEY_DTYPES = frozenset({"Money", "USD", "Integer", "Decimal", "Number"})
# RuleSpec dtypes that align with boolean-decision PolicyEngine variables.
_DECISION_DTYPES = frozenset({"Judgment", "Bool", "Boolean", "Decision"})

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
    in iteration order wins, matching ECPS comparison precedence.
    """
    index: dict[str, PolicyEngineUSVarAdapter] = {}
    for adapter in adapters:
        for rule_name in adapter.rule_names:
            index.setdefault(rule_name, adapter)
    return index


def _adapter_unit_comparison(adapter: PolicyEngineUSVarAdapter) -> tuple[str | None, str]:
    """Infer (unit, comparison) for the adapter's PE variable.

    Heuristic until adapters carry explicit comparison metadata. Variables
    ending in money-like suffixes are treated as `comparison: money` with
    `unit: USD`; everything else is `comparison: decision`.
    """
    pe_var = adapter.pe_var
    if pe_var.endswith(_MONEY_VAR_SUFFIXES):
        return ("USD", "money")
    return (None, "decision")


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
        if comparison == "decision" and dtype not in _DECISION_DTYPES:
            return _not_comparable(legal_id, rule_name, source_text)

    entity = "spm_unit" if adapter.spm else "household"
    period = "month" if adapter.monthly else None

    return Classification(
        legal_id=legal_id,
        mapping_type="direct_variable",
        policyengine_variable=adapter.pe_var,
        entity=entity,
        period=period,
        unit=unit,
        comparison=comparison,
        rationale=(
            f"State SNAP output `{rule_name}` matches the PolicyEngine variable "
            f"`{adapter.pe_var}` via the ECPS adapter catalog (rule_names "
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
            "Not present in the PolicyEngine ECPS adapter catalog "
            "(`PE_US_VAR_ADAPTERS`). Promote to `direct_variable` by adding an "
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


def classify_rulespec_repo(
    *,
    repo_root: Path,
    jurisdiction: str,
    program: str = "snap",
    adapters: Iterable[PolicyEngineUSVarAdapter] | None = None,
) -> list[Classification]:
    """Walk a `rulespec-us-<state>` repo and classify every executable output."""
    if program != "snap":
        # PE_US_VAR_ADAPTERS today is SNAP-shaped; other programs will need
        # their own adapter catalogs before this command can serve them.
        raise NotImplementedError(
            f"Program {program!r} has no adapter catalog yet; only `snap` is supported."
        )
    # Resolve the adapter catalog at call time (not at function-def time) so
    # callers and tests can inject alternatives via either the explicit
    # argument or by patching `PE_US_VAR_ADAPTERS` on this module.
    if adapters is None:
        adapters = PE_US_VAR_ADAPTERS
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


def classification_to_yaml_block(
    c: Classification, *, program: str = "snap"
) -> str:
    """Render a Classification as a `us.yaml`-shaped mapping entry."""
    lines = [f"  - legal_id: {c.legal_id}"]
    lines.append(f"    country: us")
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
