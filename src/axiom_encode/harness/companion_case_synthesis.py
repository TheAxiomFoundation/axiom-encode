"""Deterministic companion-case synthesis for complete-source test demands.

The complete-source validator rejects candidates whose companion tests do not
exercise a source boundary, a source formula branch or a nonnegative clamp.
Some of these gaps are mechanical: an existing case already exercises the right
rule, but no case puts the compared input at the source-stated value, takes the
other path of a boolean selector, runs in the period of another formula
version, or feeds a negative value to a clamp on the path the case executes.

This module proposes clones of existing model-written cases that change
exactly one thing (one input, or the period) to a value taken from the
validator's structured demand. It never supplies an expected value: the caller
executes each proposal (the rules engine supplies every asserted value) and
re-runs the complete-source analysis. A proposal is kept only when

* the engine returns a value for every asserted output,
* for a clamp, the clamped input is live: the same clone with a positive input
  gives the clamped output a different engine value (a clone whose path never
  reaches ``max(0, x)`` therefore never closes the clamp demand), and
* the analysis credits exactly that: the targeted demand disappears, only
  boundary, formula-branch or clamp demands disappear, and no finding appears.

Demands whose purpose is a model-stated expectation are never synthesized:
an unasserted principal output, and paired exception evidence.
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
_ISO_PERIOD = re.compile(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?")
_SYNTHESIZED_KINDS = frozenset({"boundary", "formula-branch", "clamp"})
# A clamp guards a quantity; a limit, rate or standard is a parameter-like
# input whose negative value would be an implausible fixture.
_LIMIT_LIKE_INPUT = re.compile(
    r"(?:^|_)(?:limit|limits|threshold|thresholds|ceiling|cap|maximum|minimum|"
    r"rate|rates|percent|percentage|standard|standards)(?:_|$)"
)
_LIVENESS_PROBE = 100

SYNTHESIZABLE_TEST_ISSUE_MARKERS = (
    "Companion tests do not exercise every source-stated boundary input",
    "Companion tests do not demonstrate formula branch",
    "Direct nonnegative clamp companion evidence is missing",
)

EvaluateCase = Callable[[dict[str, Any], tuple[str, ...]], Mapping[str, Any] | None]
# Returns the findings for a case list, or an analysis object carrying
# ``issues`` and ``companion_test_obligations``.
AnalyzeCases = Callable[[list[dict[str, Any]]], object]


@dataclass(frozen=True)
class CompanionCaseProposal:
    """One cloned case to execute; ``case`` has no ``output`` yet.

    ``liveness_case`` is the same clone with the changed input positive; the
    proposal counts only if ``liveness_output_key`` executes differently.
    """

    obligation: CompanionTestObligation
    case: dict[str, Any]
    output_keys: tuple[str, ...]
    liveness_case: dict[str, Any] | None = None
    liveness_output_key: str | None = None
    source_case: dict[str, Any] | None = None


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

    def entity(rule_name: str) -> str:
        return str((principal_rules.get(rule_name) or {}).get("entity") or "")

    def local_output_prefix(case: dict[str, Any]) -> str | None:
        """The module's own output-key prefix, taken from a local output."""

        prefixes = {
            str(key).rsplit("#", 1)[0] if "#" in str(key) else ""
            for key in case["output"]
            if _local_name(key) in principal_rules
        }
        return prefixes.pop() if len(prefixes) == 1 else None

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

    def scoped_output_keys(case: dict[str, Any], rule_name: str) -> list[str] | None:
        """Keep the case's asserted outputs that the rule reaches, plus the rule.

        One executed case queries one entity, and a clone asserts only the
        targeted rule and the local dependencies it reads: other outputs of
        a model-written case are dropped rather than re-asserted.
        """

        target = entity(rule_name)
        reached = dependency_closure(rule_name)
        keys = [
            str(key)
            for key in case["output"]
            if _local_name(key) in reached and entity(_local_name(key)) == target
        ]
        if not any(_local_name(key) == rule_name for key in keys):
            prefix = local_output_prefix(case)
            if prefix is None:
                return None
            keys.append(f"{prefix}#{rule_name}" if prefix else rule_name)
        return keys

    def clone(
        case: dict[str, Any],
        *,
        inputs: dict[str, Any] | None = None,
        period: object | None = None,
    ) -> dict[str, Any]:
        proposed = {
            key: copy.deepcopy(value)
            for key, value in case.items()
            if key not in {"name", "output"}
        }
        if inputs is not None:
            proposed["input"] = inputs
        if period is not None:
            proposed["period"] = period
        return proposed

    def named(
        obligation: CompanionTestObligation, detail: str, case: dict[str, Any]
    ) -> dict[str, Any]:
        base = f"{SYNTHESIZED_CASE_PREFIX}{_slug(obligation.kind)}_{_slug(detail)}"
        return {"name": base, **case}

    per_obligation: list[list[CompanionCaseProposal]] = []
    for obligation in obligations:
        if obligation.kind not in _SYNTHESIZED_KINDS:
            continue
        proposals: list[CompanionCaseProposal] = []
        seen: set[str] = set()

        def add(
            proposal: CompanionCaseProposal,
            seen: set[str] = seen,
            proposals: list[CompanionCaseProposal] = proposals,
        ) -> None:
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
            if obligation.kind == "clamp":
                targets = {
                    name for name in targets if not _LIMIT_LIKE_INPUT.search(name)
                }
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
                output_keys = scoped_output_keys(case, rule_name)
                if output_keys is None:
                    continue
                for key, value in case["input"].items():
                    if _local_name(key) not in targets or not _is_number(value):
                        continue
                    for target in obligation.values:
                        if float(value) == float(target):
                            continue
                        inputs = dict(case["input"])
                        inputs[key] = _yaml_number(target)
                        liveness_case = None
                        liveness_key = None
                        if obligation.kind == "clamp":
                            positive = dict(case["input"])
                            positive[key] = (
                                value if float(value) > 0 else _LIVENESS_PROBE
                            )
                            liveness_case = clone(case, inputs=positive)
                            liveness_key = next(
                                output
                                for output in output_keys
                                if _local_name(output) == rule_name
                            )
                        add(
                            CompanionCaseProposal(
                                obligation,
                                named(
                                    obligation,
                                    f"{_local_name(key)}_{target:g}",
                                    clone(case, inputs=inputs),
                                ),
                                tuple(output_keys),
                                liveness_case=liveness_case,
                                liveness_output_key=liveness_key,
                                source_case=case,
                            )
                        )
        if obligation.kind == "formula-branch":
            toggles = set(obligation.toggle_input_names)
            for rule_name, case in rule_cases:
                output_keys = scoped_output_keys(case, rule_name)
                if output_keys is None:
                    continue
                for key, value in case["input"].items():
                    if _local_name(key) in toggles and isinstance(value, bool):
                        inputs = dict(case["input"])
                        inputs[key] = not value
                        add(
                            CompanionCaseProposal(
                                obligation,
                                named(
                                    obligation,
                                    f"{_local_name(key)}_{not value}",
                                    clone(case, inputs=inputs),
                                ),
                                tuple(output_keys),
                                source_case=case,
                            )
                        )
            for rule_name, case in rule_cases:
                output_keys = scoped_output_keys(case, rule_name)
                if output_keys is None:
                    continue
                versions = (principal_rules.get(rule_name) or {}).get("versions")
                for version in versions if isinstance(versions, list) else []:
                    start = str((version or {}).get("effective_from") or "")
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", start):
                        continue
                    period = _period_like(case.get("period"), start)
                    if period is None or str(period) == str(case.get("period")):
                        continue
                    add(
                        CompanionCaseProposal(
                            obligation,
                            named(
                                obligation,
                                f"{rule_name}_{start}",
                                clone(case, period=period),
                            ),
                            tuple(output_keys),
                            source_case=case,
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

    A clone inherits its structure from a model-written case. If the engine
    disagrees with that case's asserted values, the model's expectation and
    the encoding conflict, and no clone may pin the encoding's side.
    """

    source = proposal.source_case
    if source is None:
        return True
    outputs = source.get("output") if isinstance(source, dict) else None
    if not isinstance(outputs, dict):
        return False
    keys = tuple(key for key in proposal.output_keys if key in outputs)
    if not keys:
        return True
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
    """Credit only the targeted demand (plus other synthesizable demands)."""

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
    time_budget_seconds: float = 180.0,
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
        source_key = (id(proposal.source_case), proposal.output_keys)
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
        if proposal.liveness_case is not None:
            liveness_key = str(proposal.liveness_output_key)
            evaluations += 1
            try:
                twin = evaluate_case(
                    copy.deepcopy(proposal.liveness_case), (liveness_key,)
                )
            except Exception:
                continue
            if not twin or liveness_key not in twin:
                continue
            if twin[liveness_key] == outputs.get(liveness_key):
                # The clamped input does not change the executed result: the
                # clone's path never reaches max(0, x).
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
