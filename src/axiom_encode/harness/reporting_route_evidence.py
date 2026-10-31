"""Executed evidence for three closed student-loan reporting instructions.

These are distinct sparse source-route worlds, not an eligibility blocking pair.
No inactive factual input is invented to manufacture a counterfactual.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from axiom_encode.form_source_structure import form_display_spans
from axiom_encode.harness.measurement_role_evidence import (
    BoundMeasurementRoles,
    ExecutedMeasurementWitness,
)
from axiom_encode.rulespec_version_dates import version_date_fields


@dataclass(frozen=True)
class ReportingRouteCoverage:
    source: str
    clauses: tuple[tuple[int, int, str], ...]
    display_end: int
    year: int

    def owns(self, branch: Any) -> bool:
        if branch.required_principal_output is not None:
            return False
        if self.source[branch.start : branch.end] != branch.text:
            return False
        for start, end, kind in self.clauses:
            if branch.end != end:
                continue
            if branch.start == start:
                return True
            if kind != "current" or not (0 <= branch.start < self.display_end < start):
                continue
            # The generic recognizer's first condition includes a suffix of the
            # independently correlated printed form code and repeated caption.
            # Admit that exact display fragment, not arbitrary leading prose.
            prefix = " ".join(self.source[branch.start : start].split())
            if (
                prefix
                == f"M -V ({self.year}-12) T Interest Paid on a Student Loan SCHEDULE M"
            ):
                return True
        return False

    def unmatched(self, branches: Any) -> tuple[Any, ...]:
        used: set[int] = set()
        missing = []
        for branch in branches:
            # Exact clauses have unique ends; one physical resource cannot
            # discharge a duplicated obligation, even at the same location.
            if not self.owns(branch) or branch.end in used:
                missing.append(branch)
            else:
                used.add(branch.end)
        return tuple(missing)


def certify_reporting_routes(
    context: BoundMeasurementRoles | None,
    records: set[ExecutedMeasurementWitness],
    *,
    principal_rules: dict[str, dict[str, Any]],
    formula_environment: dict[str, Any],
) -> ReportingRouteCoverage | None:
    from axiom_encode.harness import source_completeness as s

    if context is None or len(context.current) != 2:
        return None
    from axiom_encode.harness.evals import _resolve_eval_output_path

    try:
        path = _resolve_eval_output_path(context.citation)
    except (ValueError, TypeError):
        return None
    jurisdiction, separator, _ = context.citation.partition("/")
    if (
        not separator
        or path.is_absolute()
        or ".." in path.parts
        or path.suffix != ".yaml"
    ):
        return None
    module = f"{jurisdiction}:{path.with_suffix('').as_posix()}"
    primitive_names = {
        context.selector,
        context.prior,
        context.paid,
        context.used,
        f"schedule_m_student_loan_interest_paid_{context.year}_line_48",
        "schedule_m_student_loan_interest_claimed_line_60",
    }
    input_keys = {f"{module}#input.{name}" for name in primitive_names}
    owner = context.owner
    available = "schedule_m_student_loan_interest_available_line_52"
    claim = "schedule_m_student_loan_interest_claim_line_385"
    carry = "schedule_m_student_loan_interest_carryforward_line_62"
    required = {owner, available, claim, carry}
    if not required <= principal_rules.keys():
        return None
    carry_rule = principal_rules[carry]
    if any(
        carry_rule.get(k) != v
        for k, v in {
            "kind": "derived",
            "entity": "Person",
            "dtype": "Money",
            "unit": "CAD",
            "period": "Year",
        }.items()
    ):
        return None
    versions = carry_rule.get("versions")
    if (
        not isinstance(versions, list)
        or len(versions) != 1
        or not isinstance(versions[0], dict)
    ):
        return None
    version = versions[0]
    date_fields = version_date_fields(version)
    if (
        date_fields is None
        or version.get(date_fields[0]) != f"{context.year}-01-01"
        or version.get(date_fields[1]) != f"{context.year}-12-31"
    ):
        return None
    expression = s._parse_formula_expression(str(version.get("formula", "")))
    expected = s._parse_formula_expression(
        f"{available} - schedule_m_student_loan_interest_claimed_line_60"
    )
    if (
        expression is None
        or expected is None
        or ast.dump(expression) != ast.dump(expected)
    ):
        return None
    atoms = tuple(s._rule_source_excerpt_atoms(carry_rule))
    flat_source = " ".join(context.source.split())
    for path, quote in (
        ("versions[0].formula", "Subtract line 60 from line 52."),
        (f"versions[0].{date_fields[0]}", context.header),
        (f"versions[0].{date_fields[1]}", context.header),
    ):
        if not any(
            p == path
            and c == context.citation
            and quote in " ".join(e.split())
            and " ".join(e.split()) in flat_source
            for p, c, e in atoms
        ):
            return None

    grouped: dict[str, dict[str, set[str]]] = {}
    for record in records:
        if not record.evidence or any(
            e.owner != record.owner or e.version != 0 for e in record.evidence
        ):
            continue
        grouped.setdefault(record.case_json, {}).setdefault(record.owner, set()).update(
            e.role for e in record.evidence
        )
    valid: list[tuple[dict[str, Any], dict[str, Any], set[str]]] = []
    for case_json, roles in grouped.items():
        if not {owner, available, claim} <= roles.keys():
            continue
        if (
            not {"current", "current_amount"} <= roles[available]
            or "current_claim" not in roles[claim]
        ):
            continue
        case = json.loads(case_json)
        # Certificate facts and assertions must belong to the source-bound
        # module before existing suffix-based formula evaluation is consulted.
        inputs, outputs = case.get("input"), case.get("output")
        if (
            not isinstance(inputs, dict)
            or not set(inputs) <= input_keys
            or any(isinstance(value, (dict, list)) for value in inputs.values())
            or not isinstance(outputs, dict)
            or set(outputs) != {f"{module}#{name}" for name in required}
        ):
            continue
        environment = s._case_input_formula_environment(case)
        if environment is None:
            continue
        money_names = {
            context.prior,
            context.paid,
            context.used,
            f"schedule_m_student_loan_interest_paid_{context.year}_line_48",
            "schedule_m_student_loan_interest_claimed_line_60",
        }
        if any(
            isinstance(environment[name], bool)
            or not isinstance(environment[name], (int, float, Decimal))
            or not Decimal(str(environment[name])).is_finite()
            for name in money_names & environment.keys()
        ):
            continue
        raw_inputs = case.get("input")
        if not isinstance(raw_inputs, dict) or any(
            isinstance(value, bool)
            for key, value in raw_inputs.items()
            if str(key).rsplit("#", 1)[-1].removeprefix("input.") in money_names
        ):
            continue
        dependency_environment = s._case_asserted_dependency_environment(
            principal_rules,
            case,
            formula_environment=formula_environment,
        )
        if not required <= dependency_environment.keys():
            continue
        # All four actual asserted outputs must replay, including the final
        # carryforward. Existing evaluator and corroborated dependencies only.
        if not all(
            (
                execution := s._case_formula_execution(
                    principal_rules[name],
                    case,
                    formula_environment=formula_environment,
                    dependency_environment=dependency_environment,
                )
            )
            is not None
            and s._asserted_formula_runtime_values_equal(
                principal_rules[name],
                s._formula_execution_runtime_value(execution),
                s._test_case_asserted_output_value(case, name),
            )
            for name in required
        ):
            continue
        valid.append((case, environment, roles[owner]))

    def positive(value: Any) -> bool:
        return (
            isinstance(value, (int, float, Decimal))
            and not isinstance(value, bool)
            and Decimal(str(value)).is_finite()
            and Decimal(str(value)) > 0
        )

    # Prior amount is an actually supplied positive unused amount, with the full
    # calculated-and-carry source clause already bound by measurement evidence.
    # Historical world must omit the inactive prior ScheduleM/assessment fact;
    # prior world must omit inactive historical facts. No name-only decision.
    prior = [
        item
        for item in valid
        if {"prior", "prior_amount"} <= item[2]
        and item[1].get(context.selector) is True
        and positive(item[1].get(context.prior))
        and context.paid not in item[1]
        and context.used not in item[1]
    ]
    historical = [
        item
        for item in valid
        if "history" in item[2]
        and item[1].get(context.selector) is False
        and positive(item[1].get(context.paid))
        and context.prior not in item[1]
    ]
    if not prior or not historical:
        return None
    current_name = f"schedule_m_student_loan_interest_paid_{context.year}_line_48"
    current_pair = False
    for zero_case, zero, _ in valid:
        value = zero.get(current_name)
        if isinstance(value, bool) or value != 0:
            continue
        for positive_case, paid, _ in valid:
            if not positive(paid.get(current_name)) or zero_case.get(
                "period"
            ) != positive_case.get("period"):
                continue
            # Compare actual declared fact maps, not dependency values or
            # fabricated inactive assignments. Both cases are already executed.
            left, right = zero_case.get("input", {}), positive_case.get("input", {})
            if set(left) != set(right):
                continue
            changed = [key for key in left if left[key] != right[key]]
            if len(changed) == 1 and changed[0].endswith("#input." + current_name):
                current_pair = True
    if not current_pair:
        return None
    clauses = []
    all_roles = (*context.roles, *(r for b in context.current for r in b.roles))
    for name in ("current", "prior", "history"):
        roles = [role for role in all_roles if role.name == name]
        if len(roles) != 1 or context.source.count(roles[0].text) != 1:
            return None
        start = context.source.index(roles[0].text)
        clauses.append((start, start + len(roles[0].text), name))
    display = form_display_spans(context.source)
    if len(display) < 3:
        return None
    current_start = next(start for start, _end, kind in clauses if kind == "current")
    if (
        " ".join(context.source[display[2][1] : current_start].split())
        != "T Interest Paid on a Student Loan SCHEDULE M"
    ):
        return None
    return ReportingRouteCoverage(
        context.source, tuple(clauses), display[2][1], context.year
    )
