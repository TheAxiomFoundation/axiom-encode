"""Deterministic companion-case synthesis for complete-source test demands.

The complete-source validator rejects candidates whose companion tests do not
exercise a source boundary, a source formula branch, a nonnegative clamp, or a
principal output. Many of these gaps are mechanical: an existing case already
exercises the right rule, but no case puts the compared input at the
source-stated value, takes the other path of a boolean selector, runs in the
period of another formula version, feeds a negative value to a clamp, or
asserts the unasserted output.

This module proposes clones of existing cases that change exactly one thing
(one input, or the period) to a value taken from the validator's structured
demand. It never invents an expected value: the caller executes each proposal
(the rules engine supplies every asserted value) and re-runs the complete-source
analysis. A proposal is kept only when the analysis then credits it, that is,
when the remaining demands strictly shrink and no other finding appears.
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
_CASES_PER_RULE = 3
_ISO_PERIOD = re.compile(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?")

SYNTHESIZABLE_TEST_ISSUE_MARKERS = (
    "Companion tests do not exercise every source-stated boundary input",
    "Companion tests do not demonstrate formula branch",
    "Direct nonnegative clamp companion evidence is missing",
    "is never asserted by a companion test",
)

EvaluateCase = Callable[[dict[str, Any], tuple[str, ...]], Mapping[str, Any] | None]
# Returns the findings for a case list, or an analysis object carrying
# ``issues`` and ``companion_test_obligations``.
AnalyzeCases = Callable[[list[dict[str, Any]]], object]


@dataclass(frozen=True)
class CompanionCaseProposal:
    """One cloned case to execute; ``case`` has no ``output`` yet."""

    obligation: CompanionTestObligation
    case: dict[str, Any]
    output_keys: tuple[str, ...]


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


def _period_like(period: object, iso_date: str) -> object | None:
    """Render a version start date in the same shape as an existing period."""

    match = _ISO_PERIOD.fullmatch(str(period).strip()) if period is not None else None
    target = _ISO_PERIOD.fullmatch(iso_date)
    if match is None or target is None or isinstance(period, dict):
        return None
    if match.group(3) is not None:
        return iso_date
    if match.group(2) is not None:
        return iso_date[:7]
    if isinstance(period, int):
        return int(iso_date[:4])
    return iso_date[:4]


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


def _output_key_like(case: Mapping[str, Any], rule_name: str) -> str:
    for key in case.get("output") or {}:
        text = str(key)
        if "#" in text:
            return f"{text.rsplit('#', 1)[0]}#{rule_name}"
    return rule_name


def propose_companion_cases(
    obligations: Sequence[CompanionTestObligation],
    *,
    rules_payload: Mapping[str, Any],
    test_cases: Sequence[object],
) -> list[CompanionCaseProposal]:
    """Clone existing cases toward each structured demand, round-robin."""

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
    used_names = {
        str(case.get("name") or "") for case in test_cases if isinstance(case, dict)
    }

    def entity(rule_name: str) -> str:
        return str((principal_rules.get(rule_name) or {}).get("entity") or "")

    def clone(
        obligation: CompanionTestObligation,
        case: dict[str, Any],
        detail: str,
        *,
        inputs: dict[str, Any] | None = None,
        period: object | None = None,
        output_keys: Sequence[str] | None = None,
    ) -> CompanionCaseProposal:
        name = f"{SYNTHESIZED_CASE_PREFIX}{obligation.kind}_{_slug(detail)}"
        candidate = name
        suffix = 2
        while candidate in used_names:
            candidate = f"{name}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        proposed = {
            key: copy.deepcopy(value)
            for key, value in case.items()
            if key not in {"name", "output"}
        }
        proposed = {"name": candidate, **proposed}
        if inputs is not None:
            proposed["input"] = inputs
        if period is not None:
            proposed["period"] = period
        keys = tuple(output_keys) if output_keys is not None else tuple(case["output"])
        return CompanionCaseProposal(obligation, proposed, tuple(map(str, keys)))

    def scoped_output_keys(case: dict[str, Any], rule_name: str) -> list[str]:
        """Keep the case's local outputs on the rule's entity, plus the rule.

        One executed case queries one entity, so outputs of other entities
        (or parameters) in a model-written case are dropped from the clone.
        """

        target = entity(rule_name)
        keys = [
            str(key)
            for key in case["output"]
            if _local_name(key) in principal_rules
            and entity(_local_name(key)) == target
        ]
        if not any(_local_name(key) == rule_name for key in keys):
            keys.append(_output_key_like(case, rule_name))
        return keys

    per_obligation: list[list[CompanionCaseProposal]] = []
    for obligation in obligations:
        proposals: list[CompanionCaseProposal] = []
        seen: set[str] = set()

        def add(proposal: CompanionCaseProposal) -> None:
            identity = repr(
                (
                    sorted(proposal.case.get("input", {}).items(), key=str),
                    proposal.case.get("period"),
                    proposal.output_keys,
                )
            )
            if identity not in seen and len(proposals) < _PROPOSALS_PER_OBLIGATION:
                seen.add(identity)
                proposals.append(proposal)

        rule_cases = [
            (rule_name, case)
            for rule_name in obligation.rule_names
            for case in asserted.get(rule_name, [])[:_CASES_PER_RULE]
        ]
        if obligation.kind in {"boundary", "formula-branch", "clamp"}:
            targets = set(obligation.input_names)
            candidates = list(rule_cases)
            if obligation.kind == "clamp" and obligation.rule_names:
                owner = obligation.rule_names[0]
                candidates += [
                    (owner, case)
                    for case in cases
                    if all(case is not other for _name, other in rule_cases)
                    and any(_local_name(key) in targets for key in case["input"])
                ]
            for rule_name, case in candidates:
                for key, value in case["input"].items():
                    if _local_name(key) not in targets or not _is_number(value):
                        continue
                    for target in obligation.values:
                        if float(value) == float(target):
                            continue
                        inputs = dict(case["input"])
                        inputs[key] = _yaml_number(target)
                        add(
                            clone(
                                obligation,
                                case,
                                f"{_local_name(key)}_{target:g}",
                                inputs=inputs,
                                output_keys=scoped_output_keys(case, rule_name),
                            )
                        )
        if obligation.kind == "formula-branch":
            toggles = set(obligation.toggle_input_names)
            for rule_name, case in rule_cases:
                for key, value in case["input"].items():
                    if _local_name(key) in toggles and isinstance(value, bool):
                        inputs = dict(case["input"])
                        inputs[key] = not value
                        add(
                            clone(
                                obligation,
                                case,
                                f"{_local_name(key)}_{not value}",
                                inputs=inputs,
                                output_keys=scoped_output_keys(case, rule_name),
                            )
                        )
            for rule_name, case in rule_cases:
                versions = (principal_rules.get(rule_name) or {}).get("versions")
                for version in versions if isinstance(versions, list) else []:
                    start = str((version or {}).get("effective_from") or "")
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", start):
                        continue
                    period = _period_like(case.get("period"), start)
                    if period is None or str(period) == str(case.get("period")):
                        continue
                    add(
                        clone(
                            obligation,
                            case,
                            f"{rule_name}_{start}",
                            period=period,
                            output_keys=scoped_output_keys(case, rule_name),
                        )
                    )
        if obligation.kind == "principal-output" and obligation.rule_names:
            output_name = obligation.rule_names[0]
            for case in cases:
                if entity(output_name) not in {
                    entity(_local_name(key)) for key in case["output"]
                }:
                    continue
                add(
                    clone(
                        obligation,
                        case,
                        output_name,
                        output_keys=(_output_key_like(case, output_name),),
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


def synthesize_companion_cases(
    *,
    test_cases: Sequence[object],
    proposals: Sequence[CompanionCaseProposal],
    evaluate_case: EvaluateCase,
    analyze_cases: AnalyzeCases,
    baseline: object | None = None,
    max_analyses: int = 32,
    time_budget_seconds: float = 180.0,
) -> list[dict[str, Any]]:
    """Keep executed proposals that the complete-source analysis credits.

    ``analyze_cases`` returns the findings (or an analysis carrying
    ``issues`` and ``companion_test_obligations``) for a full case list; the
    caller should include every companion-test finding its validator raises,
    so a proposal that trips any other check is never kept. ``baseline`` is
    that result for ``test_cases`` when the caller already has it.
    """

    current_cases = [case for case in test_cases if isinstance(case, dict)]
    current_keys = _analysis_keys(
        baseline if baseline is not None else analyze_cases(list(current_cases))
    )
    accepted: list[dict[str, Any]] = []
    analyses = 0
    evaluations = 0
    deadline = time.monotonic() + time_budget_seconds
    for proposal in proposals:
        if (
            analyses >= max_analyses
            or evaluations >= 4 * max_analyses
            or time.monotonic() > deadline
        ):
            break
        if obligation_key(proposal.obligation) not in current_keys:
            continue
        evaluations += 1
        try:
            outputs = evaluate_case(copy.deepcopy(proposal.case), proposal.output_keys)
        except Exception:
            continue
        if not outputs or set(outputs) != set(proposal.output_keys):
            continue
        case = {**copy.deepcopy(proposal.case), "output": dict(outputs)}
        analyses += 1
        try:
            trial_keys = _analysis_keys(analyze_cases([*current_cases, case]))
        except Exception:
            continue
        if trial_keys < current_keys:
            accepted.append(case)
            current_cases.append(case)
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
