"""Deterministic boundary-case synthesis for complete-source test demands.

The complete-source validator rejects candidates whose companion tests do not
put an input at a source-stated boundary. That gap is mechanical when an
existing model-written case already exercises the covering rule and a formula
compares one of its inputs directly with the boundary value: the missing case
is that case with the input set to the boundary.

This module proposes exactly that clone and nothing else. It never supplies
an expected value: the caller executes each proposal (the rules engine
supplies every asserted value) and re-runs the complete-source analysis. The
boundary witness itself checks the comparison's operator and polarity against
the source, so a wrong `<=` for "less than" or an inverted comparison keeps
its demand. A proposal is kept only when

* the engine reproduces every value the model asserted in the source case on
  the covering rule's entity (a clone never takes the encoding's side of a
  conflict with the model's stated expectation),
* the clone asserts only the covering rule and the local dependencies it reads,
* the analysis removes the targeted boundary demand, removes nothing but
  boundary demands, and adds no finding.

Demands whose purpose is a model-stated value are never synthesized: formula
branches, nonnegative clamps, unasserted principal outputs and paired
exception evidence all stay with the model.
"""

from __future__ import annotations

import copy
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from axiom_encode.harness.source_completeness import (
    MISSING_BOUNDARY_DETAIL_MARKER,
    CompanionTestObligation,
)

SYNTHESIZED_CASE_PREFIX = "auto_source_"
_TESTS_TAG = "[complete-source-unit:tests]"
_PROPOSALS_PER_OBLIGATION = 6
_CASES_PER_RULE = 6
_SYNTHESIZED_KINDS = frozenset({"boundary"})

SYNTHESIZABLE_TEST_ISSUE_MARKERS = (
    "Companion tests do not exercise every source-stated boundary input",
)

EvaluateCase = Callable[[dict[str, Any], tuple[str, ...]], Mapping[str, Any] | None]
# Returns the findings for a case list, or an analysis object carrying
# ``issues`` and ``companion_test_obligations``.
AnalyzeCases = Callable[[list[dict[str, Any]]], object]


@dataclass(frozen=True)
class CompanionCaseProposal:
    """One cloned case to execute; ``case`` has no ``output`` yet.

    ``source_output_keys`` are the outputs the model asserted in
    ``source_case`` on the covering rule's entity; the engine must reproduce
    every one of them before any clone of that case counts.
    """

    obligation: CompanionTestObligation
    case: dict[str, Any]
    output_keys: tuple[str, ...]
    source_case: dict[str, Any] | None = None
    source_output_keys: tuple[str, ...] = ()


def has_synthesizable_test_issue(issues: Iterable[object]) -> bool:
    """Return whether any finding is a demand this module can try to meet."""

    return any(
        _TESTS_TAG in str(issue)
        and any(marker in str(issue) for marker in SYNTHESIZABLE_TEST_ISSUE_MARKERS)
        for issue in issues
    )


def obligation_key(obligation: CompanionTestObligation) -> tuple[str, str]:
    return (obligation.kind, obligation.label)


def companion_obligation_keys(
    issues: Iterable[object],
    obligations: Iterable[CompanionTestObligation] = (),
) -> frozenset[tuple[str, str]]:
    """Split findings into individually creditable keys.

    Grouped test findings (boundary lists, clamp lists) contribute one key per
    listed demand, and per-branch findings are keyed by branch identity rather
    than by the observed-period detail that changes when a case is added.
    Every other finding is keyed by its full text. The clamp finding shows a
    bounded list, so its keys come from the analysis' complete clamp
    obligations whenever the caller supplies them.
    """

    clamp_labels = [
        obligation.label for obligation in obligations if obligation.kind == "clamp"
    ]
    keys: set[tuple[str, str]] = set()
    for issue in issues:
        text = str(issue)
        if _TESTS_TAG in text:
            if "source-stated boundary input; missing:" in text:
                listed = text.split("missing:", 1)[1]
                listed = listed.split(MISSING_BOUNDARY_DETAIL_MARKER, 1)[0]
                keys.update(
                    ("boundary", item.strip())
                    for item in listed.strip().rstrip(".").split(",")
                    if item.strip()
                )
                continue
            branch = re.search(
                r"do not demonstrate formula branch (.+?)\. Exact source computation",
                text,
            )
            if branch is not None:
                keys.add(("formula-branch", branch.group(1)))
                continue
            if "Direct nonnegative clamp companion evidence is missing for" in text:
                if clamp_labels:
                    keys.update(("clamp", label) for label in clamp_labels)
                    continue
                clamps = re.findall(r"`max\(0, [^`]+\)` in `[^`]+`", text)
                keys.update(("clamp", clamp) for clamp in clamps)
                omitted = re.search(r"\((\d+) additional clamps omitted\)", text)
                if omitted is not None or not clamps:
                    keys.add(("issue", text))
                continue
            output = re.search(
                r"Principal output `([^`]+)` is never asserted by a companion test",
                text,
            )
            if output is not None:
                keys.add(("principal-output", output.group(1)))
                continue
            if "require paired positive/blocking cases" in text and "missing:" in text:
                head, listed = text.split("missing:", 1)
                listed = listed.split(". The evaluator recognized", 1)[0]
                conditions = [
                    item.strip()
                    for item in re.split(
                        r";\s+(?=[a-z]{2}(?:-[a-z0-9_]+)*/)", listed.strip()
                    )
                    if item.strip()
                ]
                keys.update(("paired-condition", head + item) for item in conditions)
                if not conditions:
                    keys.add(("issue", text))
                continue
        keys.add(("issue", text))
    return frozenset(keys)


def _local_name(key: object) -> str:
    fragment = str(key).strip().rsplit("#", 1)[-1]
    return fragment.rsplit(".", 1)[-1]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def _yaml_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _slug(text: str) -> str:
    text = re.sub(r"(?<![A-Za-z0-9])-(?=\d)", "minus_", text)
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()[:80] or "case"


def _usable_case(case: object) -> bool:
    if not isinstance(case, dict) or not isinstance(case.get("input"), dict):
        return False
    outputs = case.get("output")
    return (
        isinstance(outputs, dict)
        and bool(outputs)
        and "tables" not in case
        and not any(isinstance(value, list) for value in outputs.values())
    )


def _unique_name(base: str, used: set[str]) -> str:
    name = base
    suffix = 2
    while name in used:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def propose_companion_cases(
    obligations: Sequence[CompanionTestObligation],
    *,
    rules_payload: Mapping[str, Any],
    test_cases: Sequence[object],
) -> list[CompanionCaseProposal]:
    """Clone existing cases onto each missing boundary, round-robin."""

    rules = rules_payload.get("rules") if isinstance(rules_payload, Mapping) else None
    principal_rules = {
        str(rule.get("name")): rule
        for rule in (rules if isinstance(rules, list) else [])
        if isinstance(rule, dict)
        and str(rule.get("kind") or "").strip().lower()
        in {"derived", "derived_relation"}
        and rule.get("name")
    }
    cases = [case for case in test_cases if _usable_case(case)]
    asserted: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        for key in case["output"]:
            asserted.setdefault(_local_name(key), []).append(case)

    def entity(rule_name: str) -> str:
        return str((principal_rules.get(rule_name) or {}).get("entity") or "")

    def dependency_closure(rule_name: str) -> set[str]:
        closure: set[str] = set()
        pending = [rule_name]
        while pending and len(closure) < 64:
            name = pending.pop()
            if name in closure or name not in principal_rules:
                continue
            closure.add(name)
            versions = principal_rules[name].get("versions")
            formulas = " ".join(
                str(version.get("formula"))
                for version in (versions if isinstance(versions, list) else [])
                if isinstance(version, dict)
            )
            pending.extend(
                set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formulas))
                & set(principal_rules)
            )
        return closure

    per_obligation: list[list[CompanionCaseProposal]] = []
    for obligation in obligations:
        if obligation.kind not in _SYNTHESIZED_KINDS:
            continue
        proposals: list[CompanionCaseProposal] = []
        seen: set[str] = set()
        targets = set(obligation.input_names)
        for rule_name in obligation.rule_names:
            target_entity = entity(rule_name)
            reached = dependency_closure(rule_name)
            for case in asserted.get(rule_name, [])[:_CASES_PER_RULE]:
                same_entity = tuple(
                    str(key)
                    for key in case["output"]
                    if _local_name(key) in principal_rules
                    and entity(_local_name(key)) == target_entity
                )
                # The clone re-asserts only the covering rule and what it reads.
                output_keys = tuple(
                    key for key in same_entity if _local_name(key) in reached
                )
                for key, value in case["input"].items():
                    if _local_name(key) not in targets or not _is_number(value):
                        continue
                    for target in obligation.values:
                        if float(value) == float(target):
                            continue
                        inputs = dict(case["input"])
                        inputs[key] = _yaml_number(target)
                        clone = {
                            "name": (
                                f"{SYNTHESIZED_CASE_PREFIX}boundary_"
                                f"{_slug(f'{_local_name(key)}_{target:g}')}"
                            ),
                            **{
                                field: copy.deepcopy(item)
                                for field, item in case.items()
                                if field not in {"name", "output"}
                            },
                            "input": inputs,
                        }
                        identity = repr((sorted(inputs.items(), key=str), output_keys))
                        if (
                            identity in seen
                            or len(proposals) >= _PROPOSALS_PER_OBLIGATION
                        ):
                            continue
                        seen.add(identity)
                        proposals.append(
                            CompanionCaseProposal(
                                obligation,
                                clone,
                                output_keys,
                                source_case=case,
                                source_output_keys=same_entity,
                            )
                        )
        if proposals:
            per_obligation.append(proposals)

    ordered: list[CompanionCaseProposal] = []
    for index in range(_PROPOSALS_PER_OBLIGATION):
        ordered.extend(
            proposals[index] for proposals in per_obligation if index < len(proposals)
        )
    return ordered


def _analysis_keys(result: object) -> frozenset[tuple[str, str]]:
    """Key an analysis result, or a bare finding list, for credit checks."""

    if hasattr(result, "issues"):
        return companion_obligation_keys(
            result.issues,
            getattr(result, "companion_test_obligations", ()),
        )
    return companion_obligation_keys(result)  # type: ignore[arg-type]


def _same_expected_value(engine_value: object, asserted: object) -> bool:
    """Compare an engine-rendered value with a model-asserted expectation."""

    if isinstance(engine_value, str) or isinstance(asserted, str):
        return str(engine_value).strip().lower().replace("-", "_") == str(
            asserted
        ).strip().lower().replace("-", "_")
    if isinstance(engine_value, bool) or isinstance(asserted, bool):
        return engine_value is asserted
    try:
        return Decimal(str(engine_value)) == Decimal(str(asserted))
    except (InvalidOperation, ValueError):
        return False


def _source_case_agrees_with_engine(
    proposal: CompanionCaseProposal,
    evaluate_case: EvaluateCase,
) -> bool:
    """Clone only cases whose own stated expectations the engine reproduces.

    Every output the model asserted in the source case on the covering
    rule's entity must execute to the asserted value; a case with no such
    assertion fails closed. This catches an encoding that conflicts with the
    model's stated values. It cannot catch a model case and an encoding that
    are wrong in the same way.
    """

    source = proposal.source_case
    outputs = source.get("output") if isinstance(source, dict) else None
    keys = proposal.source_output_keys
    if not isinstance(outputs, dict) or not keys:
        return False
    unasserted = {k: v for k, v in source.items() if k not in {"name", "output"}}
    try:
        values = evaluate_case(copy.deepcopy(unasserted), keys)
    except Exception:
        return False
    return bool(values) and all(
        key in values and _same_expected_value(values[key], outputs[key])
        for key in keys
    )


def _credits_target(
    proposal: CompanionCaseProposal,
    current_keys: frozenset[tuple[str, str]],
    trial_keys: frozenset[tuple[str, str]],
) -> bool:
    """Credit only the targeted boundary (plus other boundary demands)."""

    removed = current_keys - trial_keys
    return (
        not (trial_keys - current_keys)
        and obligation_key(proposal.obligation) in removed
        and all(kind in _SYNTHESIZED_KINDS for kind, _label in removed)
    )


def synthesize_companion_cases(
    *,
    test_cases: Sequence[object],
    proposals: Sequence[CompanionCaseProposal],
    evaluate_case: EvaluateCase,
    analyze_cases: AnalyzeCases,
    baseline: object | None = None,
    max_analyses: int = 32,
    time_budget_seconds: float = 120.0,
) -> list[dict[str, Any]]:
    """Keep executed proposals that the complete-source analysis credits.

    ``analyze_cases`` returns the findings (or an analysis carrying
    ``issues`` and ``companion_test_obligations``) for a full case list; the
    caller should include every companion-test finding its validator raises,
    so a proposal that trips any other check is never kept. ``baseline`` is
    that result for ``test_cases`` when the caller already has it.
    """

    deadline = time.monotonic() + time_budget_seconds
    current_cases = [case for case in test_cases if isinstance(case, dict)]
    used_names = {str(case.get("name") or "") for case in current_cases}
    current_keys = _analysis_keys(
        baseline if baseline is not None else analyze_cases(list(current_cases))
    )
    accepted: list[dict[str, Any]] = []
    analyses = 0
    evaluations = 0
    source_agreement: dict[tuple[int, tuple[str, ...]], bool] = {}
    for proposal in proposals:
        if (
            analyses >= max_analyses
            or evaluations >= 4 * max_analyses
            or time.monotonic() > deadline
        ):
            break
        if obligation_key(proposal.obligation) not in current_keys:
            continue
        source_key = (id(proposal.source_case), proposal.source_output_keys)
        if source_key not in source_agreement:
            evaluations += 1
            source_agreement[source_key] = _source_case_agrees_with_engine(
                proposal, evaluate_case
            )
        if not source_agreement[source_key]:
            continue
        evaluations += 1
        try:
            outputs = evaluate_case(copy.deepcopy(proposal.case), proposal.output_keys)
        except Exception:
            continue
        if not outputs or set(outputs) != set(proposal.output_keys):
            continue
        case = {
            **copy.deepcopy(proposal.case),
            "name": _unique_name(str(proposal.case["name"]), used_names),
            "output": dict(outputs),
        }
        analyses += 1
        try:
            trial_keys = _analysis_keys(analyze_cases([*current_cases, case]))
        except Exception:
            continue
        if _credits_target(proposal, current_keys, trial_keys):
            accepted.append(case)
            current_cases.append(case)
            used_names.add(case["name"])
            current_keys = trial_keys
    return accepted


def expected_value_from_engine_output(actual_output: object) -> Any | None:
    """Render one engine output as a companion-test expected value."""

    if not isinstance(actual_output, dict):
        return None
    if actual_output.get("kind") == "judgment":
        outcome = str(actual_output.get("outcome"))
        return outcome if outcome in {"holds", "not_holds"} else None
    scalar = (
        actual_output.get("value")
        if actual_output.get("kind") == "scalar"
        else actual_output
    )
    if not isinstance(scalar, dict):
        return None
    kind = scalar.get("kind")
    value = scalar.get("value")
    if kind == "bool" and isinstance(value, bool):
        return value
    if kind not in {"integer", "decimal"} or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    if number == number.to_integral_value():
        return int(number)
    as_float = float(number)
    return as_float if Decimal(repr(as_float)) == number else None
