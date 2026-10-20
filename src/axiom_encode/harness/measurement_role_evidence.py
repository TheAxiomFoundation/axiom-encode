"""Closed, source-owned student-loan measurement periods, not numeric operands.

This evidence is provisional: the ordinary source-branch witness must succeed
before callers may consume its exact source spans. It does not establish an
eligibility condition, a changed-outcome pair, or a tax-year exception.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from axiom_encode.form_source_structure import form_display_spans


@dataclass(frozen=True)
class MeasurementSourceEvidence:
    start: int
    end: int
    year: int
    role: str
    owner: str
    version: int


@dataclass(frozen=True)
class _Role:
    name: str
    text: str
    spans: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _CurrentBinding:
    owner: str
    rule_json: str
    formula_key: str
    operation: str
    operation_span: tuple[int, int]
    roles: tuple[_Role, ...]


@dataclass(frozen=True)
class BoundMeasurementRoles:
    source: str
    citation: str
    year: int
    header: str
    owner: str
    selector: str
    prior: str
    paid: str
    used: str
    rule_json: str
    roles: tuple[_Role, ...]
    current: tuple[_CurrentBinding, ...]

    def evidence(
        self, *, rule: dict[str, Any], case: dict[str, Any], execution: Any, branch: Any
    ) -> tuple[MeasurementSourceEvidence, ...]:
        # Use the existing evaluator/AST parser; never interpret formulas here.
        from axiom_encode.harness import source_completeness as s

        if execution is None:
            return ()
        rule_json = json.dumps(rule, sort_keys=True)
        if rule_json != self.rule_json:
            return self._current_evidence(rule, case, execution, branch, rule_json)
        if case.get("period") != {
            "period_kind": "tax_year",
            "start": f"{self.year}-01-01",
            "end": f"{self.year}-12-31",
        }:
            return ()
        if not s._asserted_formula_runtime_values_equal(
            rule,
            s._formula_execution_runtime_value(execution),
            s._test_case_asserted_output_value(case, self.owner),
        ):
            return ()
        trace = execution.trace
        if (
            len(trace) != 1
            or trace[0].kind != "if"
            or trace[0].selectors != (self.selector,)
        ):
            return ()
        role_names = (
            {"prior", "prior_amount"}
            if trace[0].choice == 0
            else {"history"}
            if trace[0].choice == 1
            else set()
        )
        leaf = s._parse_formula_expression(execution.leaf)
        if trace[0].choice == 0:
            if not isinstance(leaf, ast.Name) or leaf.id != self.prior:
                return ()
        elif not _subtraction(leaf, self.paid, self.used):
            return ()
        # A reached prior branch cannot consume the historical clause (or vice
        # versa). Global related-role spans are committed only with this witness.
        roles = tuple(role for role in self.roles if role.name in role_names)
        if not any(
            start < branch.end and branch.start < end
            for role in roles
            for start, end, _ in role.spans
        ):
            return ()
        return tuple(
            MeasurementSourceEvidence(start, end, year, role.name, self.owner, 0)
            for role in roles
            for start, end, year in role.spans
        )

    def _current_evidence(
        self,
        rule: dict[str, Any],
        case: dict[str, Any],
        execution: Any,
        branch: Any,
        rule_json: str,
    ) -> tuple[MeasurementSourceEvidence, ...]:
        from axiom_encode.harness import source_completeness as s

        for binding in self.current:
            if rule_json != binding.rule_json or execution.trace:
                continue
            expression = s._parse_formula_expression(execution.leaf)
            if expression is None or ast.dump(expression) != binding.formula_key:
                continue
            if case.get("period") != {
                "period_kind": "tax_year",
                "start": f"{self.year}-01-01",
                "end": f"{self.year}-12-31",
            }:
                continue
            if " ".join(branch.text.split()) != binding.operation:
                continue
            if not s._asserted_formula_runtime_values_equal(
                rule,
                s._formula_execution_runtime_value(execution),
                s._test_case_asserted_output_value(case, binding.owner),
            ):
                continue
            return tuple(
                MeasurementSourceEvidence(start, end, year, role.name, binding.owner, 0)
                for role in binding.roles
                for start, end, year in role.spans
            )
        return ()


def _subtraction(node: Any, left: str, right: str) -> bool:
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and isinstance(node.left, ast.Name)
        and node.left.id == left
        and isinstance(node.right, ast.Name)
        and node.right.id == right
    )


def _match(pattern: str, source: str) -> re.Match[str] | None:
    matches = list(re.finditer(pattern.replace(" ", r"\s+"), source))
    return matches[0] if len(matches) == 1 else None


def bind_measurement_roles(
    payload: dict[str, Any], source: str, citation: str
) -> BoundMeasurementRoles | None:
    """Bind raw declarations before any name-keyed projection can hide duplicates."""
    from axiom_encode.harness import source_completeness as s

    quoted = s._annual_source_quoted_spans(source)
    if quoted is None or not form_display_spans(source):
        return None
    if payload.get("imports") or payload.get("using"):
        return None
    patterns = {
        "current": r"Complete this schedule if interest was paid in (?P<year>(?:19|20)\d{2}) on a student loan granted to you\.",
        "prior": r"Also complete this schedule if, for (?P<year>(?:19|20)\d{2}), you calculated an amount for interest paid on a student loan and decided to carry the amount forward to a future year\.",
        "history": r"If interest was paid on your student loan after (?P<after>(?:19|20)\d{2}), but in (?P<prior>(?:19|20)\d{2}) you did not calculate the amount that could be carried forward, enter the result of the following calculation on line 46: interest paid for (?P<first>(?:19|20)\d{2}) through (?P<last>(?:19|20)\d{2}) minus the interest already used to calculate a credit for interest paid on a student loan for previous years\.",
        "prior_amount": r"Unused portion of the interest paid on a student loan \(line 62 of Schedule M for (?P<year>(?:19|20)\d{2}) or the amount shown in this respect on your (?P<notice>(?:19|20)\d{2}) notice of assessment\)",
    }
    matches = {name: _match(pattern, source) for name, pattern in patterns.items()}
    if any(match is None for match in matches.values()):
        return None
    if any(
        start < match.end() and match.start() < end
        for match in matches.values()
        if match is not None
        for start, end in quoted
    ):
        return None
    current, prior, history, amount = (matches[name] for name in patterns)
    assert (
        current is not None
        and prior is not None
        and history is not None
        and amount is not None
    )
    year = int(current["year"])
    if not (
        year - 1
        == int(prior["year"])
        == int(history["prior"])
        == int(history["last"])
        == int(amount["year"])
        == int(amount["notice"])
        and int(history["first"]) == int(history["after"]) + 1
    ):
        return None
    header = _match(
        rf"Revenu Quebec Schedule M {year} - Interest Paid on a Student Loan", source
    )
    if (
        header is None
        or header.start() != 0
        or source[header.end() : header.end() + 1] != "\n"
    ):
        return None
    selector = f"schedule_m_carryforward_calculated_for_{year - 1}"
    prior_name = "schedule_m_unused_student_loan_interest_line_46"
    paid = (
        f"schedule_m_student_loan_interest_paid_{history['first']}_through_{year - 1}"
    )
    used = "schedule_m_student_loan_interest_used_for_previous_year_credits"
    names = [
        selector,
        prior_name,
        paid,
        used,
        f"schedule_m_student_loan_interest_paid_{year}_line_48",
        "schedule_m_student_loan_interest_claimed_line_60",
    ]
    expected = [
        {
            "name": name,
            "entity": "Person",
            "dtype": "Boolean" if name == selector else "Money",
            "period": "Year",
            **({} if name == selector else {"unit": "CAD"}),
        }
        for name in names
    ]
    inputs = payload.get("inputs")
    rules = payload.get("rules")
    if (
        not isinstance(inputs, list)
        or not isinstance(rules, list)
        or any(
            not isinstance(item, dict) or not isinstance(item.get("name"), str)
            for item in inputs + rules
        )
    ):
        return None
    all_names = [item["name"] for item in inputs + rules]
    if len(all_names) != len(set(all_names)) or sorted(
        inputs, key=lambda x: x["name"]
    ) != sorted(expected, key=lambda x: x["name"]):
        return None
    if payload.get("units") != [{"name": "CAD", "kind": "currency", "minor_units": 2}]:
        return None
    owner = f"schedule_m_unused_student_loan_interest_for_{year}_line_46"
    owners = [rule for rule in rules if rule["name"] == owner]
    if len(owners) != 1:
        return None
    rule = owners[0]
    if any(
        rule.get(key) != value
        for key, value in {
            "kind": "derived",
            "entity": "Person",
            "dtype": "Money",
            "unit": "CAD",
            "period": "Year",
        }.items()
    ):
        return None
    versions = rule.get("versions")
    if (
        not isinstance(versions, list)
        or len(versions) != 1
        or not isinstance(versions[0], dict)
    ):
        return None
    version = versions[0]
    if (
        version.get("effective_from") != f"{year}-01-01"
        or version.get("effective_to") != f"{year}-12-31"
    ):
        return None
    formula = version.get("formula")
    if not isinstance(formula, str):
        return None
    node = s._parse_first_formula_branch_node(formula)
    if (
        node is None
        or node.start != 0
        or node.end != len(formula)
        or node.kind != "if"
        or node.selectors != (selector,)
        or len(node.choices) != 2
    ):
        return None
    true = s._parse_formula_expression(node.choices[0])
    if (
        not isinstance(true, ast.Name)
        or true.id != prior_name
        or not _subtraction(s._parse_formula_expression(node.choices[1]), paid, used)
    ):
        return None
    flat = " ".join(source.split())
    atoms = tuple(s._rule_source_excerpt_atoms(rule))

    def proves(path: str, quote: str) -> bool:
        return any(
            atom_path == path
            and atom_citation == citation
            and " ".join(quote.split()) in " ".join(excerpt.split())
            and " ".join(excerpt.split()) in flat
            for atom_path, atom_citation, excerpt in atoms
        )

    if not all(
        proves(f"versions[0].{field}", header[0])
        for field in ("effective_from", "effective_to")
    ):
        return None
    roles = []
    for name in ("prior", "history", "prior_amount"):
        match = matches[name]
        assert match is not None
        if not proves("versions[0].formula", match[0]):
            return None
        roles.append(
            _Role(
                name,
                match[0],
                tuple(
                    (*match.span(group), int(match[group]))
                    for group in match.groupdict()
                ),
            )
        )
    current_bindings = []
    current_specs = (
        (
            "schedule_m_student_loan_interest_available_line_52",
            f"{owner} + schedule_m_student_loan_interest_paid_{year}_line_48",
            "Add lines 46 and 48.",
            (
                ("current", patterns["current"]),
                (
                    "current_amount",
                    rf"Interest paid in (?P<year>{year}) on a student loan",
                ),
            ),
        ),
        (
            "schedule_m_student_loan_interest_claim_line_385",
            "schedule_m_student_loan_interest_claimed_line_60",
            "Carry all or part of this amount to line 385 of your return.",
            (
                (
                    "current_claim",
                    rf"Amount carried to line 385 of your (?P<year>{year}) return",
                ),
            ),
        ),
    )
    for current_owner, expected_formula, operation, role_patterns in current_specs:
        candidates = [r for r in rules if r["name"] == current_owner]
        if len(candidates) != 1:
            continue
        candidate = candidates[0]
        if any(
            candidate.get(k) != v
            for k, v in {
                "kind": "derived",
                "entity": "Person",
                "dtype": "Money",
                "unit": "CAD",
                "period": "Year",
            }.items()
        ):
            continue
        current_versions = candidate.get("versions")
        if (
            not isinstance(current_versions, list)
            or len(current_versions) != 1
            or not isinstance(current_versions[0], dict)
        ):
            continue
        current_version = current_versions[0]
        if (
            current_version.get("effective_from") != f"{year}-01-01"
            or current_version.get("effective_to") != f"{year}-12-31"
        ):
            continue
        expr = s._parse_formula_expression(str(current_version.get("formula", "")))
        expected_expr = s._parse_formula_expression(expected_formula)
        if (
            expr is None
            or expected_expr is None
            or ast.dump(expr) != ast.dump(expected_expr)
        ):
            continue
        operation_match = _match(re.escape(operation).replace(r"\ ", " "), source)
        if operation_match is None or any(
            a < operation_match.end() and operation_match.start() < b for a, b in quoted
        ):
            continue
        current_atoms = tuple(s._rule_source_excerpt_atoms(candidate))

        def current_proves(path: str, quote: str) -> bool:
            return any(
                a == path
                and c == citation
                and " ".join(quote.split()) in " ".join(e.split())
                and " ".join(e.split()) in flat
                for a, c, e in current_atoms
            )

        if not all(
            current_proves(f"versions[0].{field}", header[0])
            for field in ("effective_from", "effective_to")
        ) or not current_proves("versions[0].formula", operation):
            continue
        current_roles = []
        for role_name, pattern in role_patterns:
            match = _match(pattern, source)
            if (
                match is None
                or any(a < match.end() and match.start() < b for a, b in quoted)
                or not current_proves("versions[0].formula", match[0])
            ):
                break
            current_roles.append(
                _Role(
                    role_name,
                    match[0],
                    tuple(
                        (*match.span(group), int(match[group]))
                        for group in match.groupdict()
                    ),
                )
            )
        else:
            current_bindings.append(
                _CurrentBinding(
                    current_owner,
                    json.dumps(candidate, sort_keys=True),
                    ast.dump(expr),
                    operation,
                    operation_match.span(),
                    tuple(current_roles),
                )
            )
    return BoundMeasurementRoles(
        source,
        citation,
        year,
        header[0],
        owner,
        selector,
        prior_name,
        paid,
        used,
        json.dumps(rule, sort_keys=True),
        tuple(roles),
        tuple(current_bindings),
    )
