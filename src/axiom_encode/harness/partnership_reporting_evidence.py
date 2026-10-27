"""Closed, executed Schedule L partnership reporting evidence.

Distinct sparse reporting worlds are not eligibility counterfactuals. This
certificate owns only its physical condition, never numeric or annual evidence.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import yaml

from axiom_encode.form_source_structure import form_display_spans
from axiom_encode.source_hash import _UniqueKeySafeLoader

PREFIX = "schedule_l_"
SELECTOR = PREFIX + "is_member_of_partnership"
GROSS = PREFIX + "reported_cryptoassets_gross_income_line_16_1"
NET = PREFIX + "reported_cryptoassets_net_income_line_26_1"
TOTAL = PREFIX + "primary_net_business_income_line_27"
RETURN = PREFIX + "net_business_income_line_164"
LM53 = PREFIX + "family_resource_qpip_qpp_earnings_line_40"
PARTNER = (PREFIX + "rl15_box_14_7_amount", PREFIX + "rl15_box_1_12_amount")
DIRECT = (
    PREFIX + "cryptoassets_gross_income_line_16_1",
    PREFIX + "cryptoassets_net_income_line_26_1",
)
OTHER_NET = tuple(
    PREFIX + n
    for n in (
        "business_other_net_income_line_22",
        "farming_net_income_line_23",
        "fishing_net_income_line_24",
        "profession_net_income_line_25",
        "commission_net_income_line_26",
    )
)
SPECIAL = tuple(
    PREFIX + n
    for n in (
        "retiring_partner_or_physician_allowance_line_28",
        "specified_member_partnership_income_line_29",
        "post_operation_commissions_line_30",
    )
)
LM53_INPUT = PREFIX + "lm53_line_8_family_resource_qpip_qpp_earnings"
HEADER = "Revenu Quebec Schedule L 2025 - Business Income"
CONDITION = "If you are a member of a partnership:"
GROSS_QUOTE = "Enter the amount from box 14-7 of your RL-15 slip on line 16.1."
NET_QUOTE = "Enter the amount from box 1-12 of the RL-15 slip on line 26.1."
QUOTES = {
    GROSS: GROSS_QUOTE,
    NET: NET_QUOTE,
    TOTAL: "Add lines 22 through 26.1.",
    RETURN: "Add lines 27 through 30.",
    LM53: "Insurable earnings (QPIP) and pensionable earnings (QPP) of a person responsible for a family-type resource\nor an intermediate resource (line 8 of form LM-53-V)",
}
FORMULAS = {
    GROSS: f"if {SELECTOR}: {PARTNER[0]} else: {DIRECT[0]}",
    NET: f"if {SELECTOR}: {PARTNER[1]} else: {DIRECT[1]}",
    TOTAL: " + ".join((*OTHER_NET, NET)),
    RETURN: " + ".join((TOTAL, *SPECIAL)),
    LM53: LM53_INPUT,
}


@dataclass(frozen=True)
class ExecutedPartnershipRoute:
    case_name: str
    partner: bool
    selected_inputs: tuple[str, str]


@dataclass(frozen=True)
class PartnershipReportingCoverage:
    source: str
    start: int
    end: int
    witnesses: tuple[ExecutedPartnershipRoute, ...]

    def unmatched(self, branches: Any) -> tuple[Any, ...]:
        used = False
        missing = []
        for branch in branches:
            if (
                not used
                and branch.required_principal_output is None
                and branch.start == self.start
                and branch.end == self.end
                and branch.text == self.source[self.start : self.end]
            ):
                used = True
            else:
                missing.append(branch)
        return tuple(missing)


def certify_partnership_reporting(
    content: str | None,
    source: str,
    citation: str,
    cases: list[dict[str, Any]],
    *,
    principal_rules: dict[str, dict[str, Any]],
    formula_environment: dict[str, Any],
) -> PartnershipReportingCoverage | None:
    from axiom_encode.harness import source_completeness as s
    from axiom_encode.harness.evals import _resolve_eval_output_path

    if not isinstance(content, str):
        return None
    try:
        payload = yaml.load(content, Loader=_UniqueKeySafeLoader)
    except (yaml.YAMLError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("imports") or payload.get("using"):
        return None
    quoted = s._annual_source_quoted_spans(source)
    if (
        quoted is None
        or not source.startswith(HEADER + "\n")
        or not form_display_spans(source)
    ):
        return None
    required_quotes = (
        HEADER,
        CONDITION,
        *QUOTES.values(),
        "Carry the result to line 164 of your return.",
    )
    for quote in required_quotes:
        if source.count(quote) != 1:
            return None
        start = source.index(quote)
        if any(a < start + len(quote) and start < b for a, b in quoted):
            return None
    # The complete physical cryptoasset instruction cannot be borrowed from an
    # unrelated paragraph or joined to an additional eligibility proposition.
    block = CONDITION + "\n• " + GROSS_QUOTE + "\n• " + NET_QUOTE
    if source.count(block) != 1:
        return None
    if "T P -1. D. L-V (2025-12)" not in source:
        return None
    try:
        path = _resolve_eval_output_path(citation)
    except (ValueError, TypeError):
        return None
    jurisdiction, separator, _ = citation.partition("/")
    if (
        not separator
        or path.is_absolute()
        or ".." in path.parts
        or path.suffix != ".yaml"
    ):
        return None
    module = f"{jurisdiction}:{path.with_suffix('').as_posix()}"
    expected_module = "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income"
    metadata = payload.get("module")
    verification = (
        metadata.get("source_verification") if isinstance(metadata, dict) else None
    )
    if (
        module != expected_module
        or not isinstance(verification, dict)
        or verification.get("corpus_citation_path") != citation
    ):
        return None
    names = {SELECTOR, *PARTNER, *DIRECT, *OTHER_NET, *SPECIAL, LM53_INPUT}
    inputs, rules = payload.get("inputs"), payload.get("rules")
    if not isinstance(inputs, list) or not isinstance(rules, list):
        return None
    if any(
        not isinstance(item, dict) or not isinstance(item.get("name"), str)
        for item in inputs + rules
    ):
        return None
    all_names = [item["name"] for item in inputs + rules]
    if len(all_names) != len(set(all_names)):
        return None
    expected = [
        {
            "name": n,
            "entity": "Person",
            "dtype": "Boolean" if n == SELECTOR else "Money",
            "period": "Year",
            **({} if n == SELECTOR else {"unit": "CAD"}),
        }
        for n in sorted(names)
    ]
    if sorted(inputs, key=lambda x: x["name"]) != expected:
        return None
    if payload.get("units") != [{"name": "CAD", "kind": "currency", "minor_units": 2}]:
        return None
    indexed = {rule["name"]: rule for rule in rules}
    if set(indexed) != set(FORMULAS) or not set(FORMULAS) <= principal_rules.keys():
        return None
    flat = " ".join(source.split())
    for name, expected_formula in FORMULAS.items():
        rule = indexed[name]
        if (
            set(rule)
            != {
                "name",
                "kind",
                "entity",
                "dtype",
                "unit",
                "period",
                "versions",
                "metadata",
            }
            or principal_rules[name] != rule
        ):
            return None
        if any(
            rule.get(k) != v
            for k, v in {
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
            set(version) != {"effective_from", "effective_to", "formula"}
            or version["effective_from"] != "2025-01-01"
            or version["effective_to"] != "2025-12-31"
        ):
            return None
        formula = version["formula"]
        if not isinstance(formula, str):
            return None
        if name in (GROSS, NET):
            node = s._parse_first_formula_branch_node(formula)
            index = (GROSS, NET).index(name)
            if (
                node is None
                or node.start != 0
                or node.end != len(formula)
                or node.kind != "if"
                or node.selectors != (SELECTOR,)
                or len(node.choices) != 2
            ):
                return None
            for text, operand in zip(
                node.choices, (PARTNER[index], DIRECT[index]), strict=True
            ):
                expr = s._parse_formula_expression(text)
                if not isinstance(expr, ast.Name) or expr.id != operand:
                    return None
        else:
            expr, expected_expr = (
                s._parse_formula_expression(formula),
                s._parse_formula_expression(expected_formula),
            )
            if (
                expr is None
                or expected_expr is None
                or ast.dump(expr) != ast.dump(expected_expr)
            ):
                return None
        atoms = tuple(s._rule_source_excerpt_atoms(rule))
        obligations = [
            ("formula", QUOTES[name]),
            ("effective_from", HEADER),
            ("effective_to", HEADER),
        ]
        if name in (GROSS, NET):
            obligations.append(("formula", CONDITION))
        if not all(
            any(
                p == "versions[0]." + field
                and c == citation
                and " ".join(quote.split()) in " ".join(excerpt.split())
                and " ".join(excerpt.split()) in flat
                for p, c, excerpt in atoms
            )
            for field, quote in obligations
        ):
            return None
    input_keys = {f"{module}#input.{name}" for name in names}
    output_keys = {f"{module}#{name}" for name in FORMULAS}
    records = []
    executed_cases = []
    for case in cases:
        inputs, outputs = case.get("input"), case.get("output")
        if (
            not isinstance(inputs, dict)
            or not set(inputs) <= input_keys
            or not isinstance(outputs, dict)
            or set(outputs) != output_keys
            or case.get("tables")
            or case.get("period")
            != {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"}
        ):
            continue
        selector = inputs.get(f"{module}#input.{SELECTOR}")
        if not isinstance(selector, bool):
            continue
        selected, inactive = (PARTNER, DIRECT) if selector else (DIRECT, PARTNER)
        if any(f"{module}#input.{name}" in inputs for name in inactive):
            continue
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float, Decimal))
            or not Decimal(str(v)).is_finite()
            for k, v in inputs.items()
            if k != f"{module}#input.{SELECTOR}"
        ):
            continue
        environment = s._case_asserted_dependency_environment(
            principal_rules, case, formula_environment=formula_environment
        )
        if not set(FORMULAS) <= environment.keys():
            continue
        executions = {}
        for name in FORMULAS:
            execution = s._case_formula_execution(
                principal_rules[name],
                case,
                formula_environment=formula_environment,
                dependency_environment=environment,
            )
            if execution is None or not s._asserted_formula_runtime_values_equal(
                principal_rules[name],
                s._formula_execution_runtime_value(execution),
                s._test_case_asserted_output_value(case, name),
            ):
                break
            executions[name] = execution
        if len(executions) != 5:
            continue
        if not all(
            len(executions[name].trace) == 1
            and executions[name].trace[0].kind == "if"
            and executions[name].trace[0].selectors == (SELECTOR,)
            and executions[name].trace[0].choice == (0 if selector else 1)
            and executions[name].leaf.strip() == selected[index]
            and f"{module}#input.{selected[index]}" in inputs
            for index, name in enumerate((GROSS, NET))
        ):
            continue
        records.append(
            ExecutedPartnershipRoute(str(case.get("name", "")), selector, selected)
        )
        executed_cases.append((case, selector))
    if {record.partner for record in records} != {False, True}:
        return None
    # A distinct executed partner world must vary only the gross box. This
    # proves gross reporting changes while all four net/downstream outputs stay
    # unchanged; the AST contract alone is not a replacement for this witness.
    gross_input = f"{module}#input.{PARTNER[0]}"
    gross_output = f"{module}#{GROSS}"
    gross_pair = False
    for left, left_partner in executed_cases:
        if not left_partner:
            continue
        for right, right_partner in executed_cases:
            if not right_partner or set(left["input"]) != set(right["input"]):
                continue
            changed = {
                key
                for key in left["input"]
                if left["input"][key] != right["input"][key]
            }
            if changed != {gross_input}:
                continue
            changed_outputs = {
                key
                for key in left["output"]
                if left["output"][key] != right["output"][key]
            }
            if changed_outputs == {gross_output}:
                gross_pair = True
    if not gross_pair:
        return None
    start = source.index(CONDITION)
    return PartnershipReportingCoverage(
        source, start, start + len(CONDITION + "\n• " + GROSS_QUOTE), tuple(records)
    )
