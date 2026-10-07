"""Closed worksheet operations and annual evidence, independent of candidate ASTs.

Unknown source layouts, source contexts, paths or selected dependencies remain
unresolved. Transfer/aggregation is deliberately outside this certificate.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import source_path_shadow as paths
from axiom_encode.harness.proof_validator import validate_rulespec_proofs
from axiom_encode.harness.source_path_evidence import (
    CertifiedPathCoverage,
    _same_world,
    path_input_identity,
)

# Closed authenticated operation grammar. These literals describe supported
# source operations; matching proof metadata alone never certifies computation.
_SOURCE_OPERATIONS = {
    "line1": (
        (
            "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
            "Enter the amount from line 1 of Form T2209.",
        ),
    ),
    "line3": (
        (
            "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
            "Line 1 minus line 2 = 3",
        ),
    ),
    "line4": (
        (
            "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
            "Net foreign\nnon-business income (2) Provincial or territorial\n× = 4\nNet income (3) tax otherwise payable (4)",
        ),
        (
            "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
            "Enter the amount reported as net foreign non-business income from line 2 of Form T2209.",
        ),
    ),
    "line5": (
        (
            "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
            "Enter the amount from line 1 of Form T2209. 1\nEnter the amount from line 3 of Form T2209, unless you have to pay minimum tax.(1) – 2\nLine 1 minus line 2 = 3\nNet foreign\nnon-business income (2) Provincial or territorial\n× = 4\nNet income (3) tax otherwise payable (4)\nEnter whichever amount is less: line 3 or line 4.\nThe amount on line 5 should not be more than the amount entered Provincial or territorial\non the line for provincial or territorial tax otherwise payable. foreign tax credit 5",
        ),
        (
            "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
            "Before you complete this form, calculate your federal foreign tax credit by using Form T2209, Federal Foreign Tax Credits. If\nthe amount of the federal foreign non-business income tax credit you are entitled to deduct is equal to the foreign non-business\ntax you paid, your provincial or territorial foreign tax credit would be zero. As a result, you do not have to complete this form.",
        ),
        (
            "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
            "This form does not apply to residents of Quebec.",
        ),
        (
            "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
            "If you must pay minimum tax and you were a resident of Manitoba, you cannot claim a provincial foreign tax credit.",
        ),
    ),
    "net_income": (
        (
            "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
            "Enter the amount reported as net income from line 2 of Form T2209.",
        ),
        (
            "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
            "If you were a resident of Canada for part of the year, include the income for the part of the year you were a resident of\nCanada plus any income and losses referred to in paragraphs 115(1)(a) to (c) of the Income Tax Act as reported on your\nCanadian tax return, for the part of the year you were not a resident of Canada.",
        ),
        (
            "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
            'If you paid tax to more than one jurisdiction in 2025, calculate this amount according to note (3) of Form T2209. However,\ninstead of using "line 23600 of your return plus the amount on line 68360 of Form T1206" as stated in that note, use the\namount allocated to your province or territory of residence in column 4 of Part 1 of Form T2203 to do the calculation.',
        ),
        (
            "ca/policy/cra/t1-2025/federal-foreign-tax-credits",
            '(3) Net income\nAmount from line 23600 of your return (or the amount you would have entered if the instructions said "if negative, show in\nbrackets") plus the amount on line 68360 of Form T1206, Total split income, minus any:\n(cid:129) amount deductible as a Canadian Forces personnel and police deduction (line 24400 of your return)\n(cid:129) amount deductible as security options deductions (line 24900 of your return)\n(cid:129) amount deductible as an other payments deduction (line 25000 of your return)\n(cid:129) net capital losses of other years you claimed (line 25300 of your return)\n(cid:129) capital gains deduction for qualifying business transfer or qualifying cooperative conversion you claimed (line 25395 of\nyour return)\n(cid:129) capital gains deduction you claimed (line 25400 of your return)\n(cid:129) amounts deductible as net employment income from a prescribed international organization, as foreign income exempt\nunder a tax treaty, or as adult basic education tuition assistance (included on line 25600 of your return)',
        ),
    ),
}

_PRIMARY = "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit"
_SECONDARY = "ca/policy/cra/t1-2025/federal-foreign-tax-credits"
_LINE1 = "provincial_territorial_2025_t2036_foreign_non_business_tax_line_1"
_LINE2 = "provincial_territorial_2025_t2036_federal_credit_line_2"
_LINE3 = "provincial_territorial_2025_t2036_excess_foreign_tax_line_3"
_LINE4 = "provincial_territorial_2025_t2036_tax_otherwise_payable_line_4"
_LINE5 = "provincial_territorial_foreign_tax_credit_line_5"
_TAX = "provincial_or_territorial_tax_otherwise_payable"
_OUTPUTS = (_LINE1, _LINE2, _LINE3, _LINE4, _LINE5)
_PAID = "t2209_non_business_income_tax_paid_to_foreign_country_line_1"
_CREDIT = "t2209_federal_non_business_foreign_tax_credit_line_3"
_RESIDENCE = "province_or_territory_of_residence_at_year_end"
_DEDUCTIONS = (
    "deductible_canadian_forces_police_24400",
    "deductible_security_options_24900",
    "deductible_other_payments_25000",
    "claimed_net_capital_losses_25300",
    "claimed_qualifying_transfer_capital_gains_25395",
    "claimed_capital_gains_25400",
    "deductible_prescribed_organization_employment_25600",
    "deductible_treaty_exempt_foreign_income_25600",
    "deductible_adult_basic_education_tuition_25600",
)


@dataclass(frozen=True)
class WorksheetOperationCoverage:
    annual_spans: tuple[tuple[int, int], ...] = ()
    cap_span: tuple[int, int] | None = None
    outputs: tuple[str, ...] = ()
    case_links: tuple[tuple[str, int], ...] = ()
    unresolved: tuple[str, ...] = ()
    source_sha256: str = ""

    def remaining_exception(
        self, branch, *, source_text, principal_rules, corpus_citation_path
    ):
        """Discharge only this owned numeric cap; retain every transfer byte."""
        if (
            not self.cap_span
            or self.unresolved
            or corpus_citation_path != _PRIMARY
            or hashlib.sha256(source_text.encode()).hexdigest() != self.source_sha256
        ):
            return branch
        a, b = self.cap_span
        if branch.start != a or branch.end < b:
            return branch
        claimed = False
        for name, rule in principal_rules.items():
            for path, citation, excerpt in sc._rule_source_excerpt_atoms(rule):
                index = sc._formula_proof_version_index(path)
                if (
                    index is None
                    or citation != _PRIMARY
                    or source_text[a:b] not in excerpt
                ):
                    continue
                if name != _LINE5 or not any(
                    i == index and start == "2025-01-01" and end == "2025-12-31"
                    for i, _, start, end in sc._effective_formula_version_intervals(
                        rule
                    )
                ):
                    return branch
                claimed = True
        if not claimed:
            return branch
        if branch.end == b:
            return None
        remaining_start = b
        while remaining_start < branch.end and source_text[remaining_start].isspace():
            remaining_start += 1
        if not source_text[remaining_start : branch.end].startswith(
            "Enter the total from line 5"
        ):
            return branch
        return replace(
            branch,
            start=remaining_start,
            text=source_text[remaining_start : branch.end],
        )


def _formula_ast(formula: str, depth: int = 0) -> ast.expr | None:
    if depth > 32:
        return None
    node = sc._first_formula_branch_node(formula.strip())
    if node is None:
        return paths._expression(formula)
    if (
        node.kind != "if"
        or node.start != 0
        or formula.strip()[node.end :].strip()
        or len(node.selectors) != 1
        or len(node.choices) != 2
    ):
        return None
    condition = paths._expression(node.selectors[0])
    a, b = (_formula_ast(x, depth + 1) for x in node.choices)
    if condition is None or a is None or b is None:
        return None
    return ast.IfExp(test=condition, body=a, orelse=b)


def _annual_intro(source: str) -> re.Match[str] | None:
    pattern = re.compile(
        r"^Use this form to calculate the foreign non-business income tax credit\s+"
        r"for (?P<year>2025) that you can deduct from the income tax\s+"
        r"payable to the province or territory you resided in at the end of the tax year\.",
        re.MULTILINE,
    )
    matches = tuple(pattern.finditer(source))
    quoted = sc._annual_source_quoted_spans(source)
    if len(matches) != 1 or quoted is None:
        return None
    intro = matches[0]
    header = re.fullmatch(
        r"(?:[A-Z][A-Za-z]* )?Form [A-Z]*[1-9]\d* "
        + intro["year"]
        + r" (?P<title>[A-Za-z -]{1,120})\n\s*(?:Protected [A-Z] when completed\s+)?(?P=title)\s*",
        source[: intro.start()],
    )
    if header is None or any(a < intro.end() and b > 0 for a, b in quoted):
        return None
    return intro


def _residence_definition_bound(rule, selected, source, citation, inputs):
    if paths._residence_selector_is_source_bound(
        rule, selected, source, citation, inputs
    ):
        return True
    intro = _annual_intro(source)
    if intro is None:
        return False
    name = rule.get("name")
    special = {
        "resident_of_quebec_at_year_end": (
            "Quebec",
            "This form does not apply to residents of Quebec.",
        ),
        "resident_of_manitoba_at_year_end": (
            "Manitoba",
            "If you must pay minimum tax and you were a resident of Manitoba, you cannot claim a provincial foreign tax credit.",
        ),
    }
    if name not in special:
        return False
    place, predicate = special[name]
    if source.count(predicate) != 1:
        return False
    quoted = sc._annual_source_quoted_spans(source)
    start = source.index(predicate)
    if quoted is None or any(a <= start < b for a, b in quoted):
        return False
    return (
        sc._source_owned_residence_definition(
            rule,
            selected=selected,
            place=place,
            year=intro["year"],
            source_text=source,
            predicate_text=predicate,
            annual_header=sc._collapse_text(
                intro.group().split(" that you can deduct", 1)[0]
            ),
            intro_text=sc._collapse_text(intro.group()),
            citation=citation,
            input_declarations=inputs,
        )
        is not None
    )


class _OperationResolver(paths._Resolver):
    """Typed selected expansion; certified monetary results remain atomic."""

    def expand(self, node: ast.expr, stack: tuple[str, ...] = ()) -> ast.expr | None:
        if len(stack) > 128:
            self.errors.add("operation dependency depth")
            return None
        if isinstance(node, ast.Name) and node.id in (self.rules or {}):
            if node.id in stack:
                self.errors.add("operation dependency cycle")
                return None
            selected = self.select(node.id)
            if selected is None:
                return None
            rule, formula = selected
            declared = paths._declared_type(rule)
            if declared is None:
                self.errors.add("unknown declaration type")
                return None
            if node.id in {_LINE2, _TAX}:
                if declared != paths._ValueType("money", "CAD"):
                    self.errors.add("certified result type changed")
                    return None
                return self.mark(node, declared)
            if rule.get("dtype") == "Judgment" and "resident_of_" in node.id:
                if not _residence_definition_bound(
                    rule,
                    self.dependencies[node.id].version,
                    self.source,
                    self.citation,
                    self.inputs or {},
                ):
                    self.errors.add("unbound residence definition")
                    return None
            parsed = _formula_ast(formula)
            result = self.expand(parsed, (*stack, node.id)) if parsed else None
            if result is None or not paths._assignable(self.types[result], declared):
                self.errors.add("operation dependency type mismatch: " + node.id)
                return None
            return self.mark(result, declared)
        if isinstance(node, ast.IfExp):
            condition = self.expand(node.test, stack)
            a, b = self.expand(node.body, stack), self.expand(node.orelse, stack)
            if condition is not None and a is not None and b is not None:
                combined = paths._common_type(self.types[a], self.types[b])
                if self.types[condition].kind == "boolean" and combined is not None:
                    return self.mark(
                        ast.IfExp(test=condition, body=a, orelse=b), combined
                    )
            self.errors.add("conditional operation type mismatch")
            return None
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [self.expand(x, stack) for x in node.values]
            if all(x is not None and self.types[x].kind == "boolean" for x in values):
                return self.mark(
                    ast.BoolOp(op=node.op, values=values), paths._ValueType("boolean")
                )
            self.errors.add("Boolean operation type mismatch")
            return None
        return super().expand(node, stack)


def _expected_operations() -> dict[str, ast.expr]:
    # Independent source operands, never expanded through candidate formulas.
    deductions = " - ".join(
        (
            "amount_allocated_to_province_or_territory_of_residence_t2203_part_1_column_4",
            *_DEDUCTIONS,
        )
    )
    # Local helper's inactive zero is outside its dispatch branch: preserve its
    # exact form to avoid silently inventing simplification of unknown paths.
    selected_net = (
        f"(({deductions} if jurisdictions_tax_paid_count > 1 else 0) "
        "if jurisdictions_tax_paid_count > 1 else "
        "((income_for_part_of_year_resident_in_canada + "
        "income_and_losses_under_paragraphs_115_1_a_to_c_for_nonresident_part "
        "if resident_of_canada_for_part_of_year else 0) "
        "if resident_of_canada_for_part_of_year else t2209_net_income_line_2))"
    )
    excess = f"({_PAID} - {_LINE2})"
    ratio = f"(t2209_net_foreign_non_business_income_line_2 / {selected_net}) * {_TAX}"
    principal = (
        f"0 if {_RESIDENCE} == 'Quebec' else "
        f"(0 if minimum_tax_is_payable and {_RESIDENCE} == 'Manitoba' else "
        f"(0 if {_CREDIT} == {_PAID} else min({excess}, {ratio}, {_TAX})))"
    )
    return {
        name: ast.parse(text, mode="eval").body
        for name, text in {
            _LINE1: _PAID,
            _LINE3: excess,
            _LINE4: ratio,
            _LINE5: principal,
        }.items()
    }


def _same_expression(a: ast.expr, b: ast.expr) -> bool:
    class EqualityOrder(ast.NodeTransformer):
        def visit_Compare(self, node: ast.Compare) -> ast.AST:
            node = self.generic_visit(node)
            if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
                left, right = sorted((node.left, node.comparators[0]), key=ast.dump)
                node.left, node.comparators = left, [right]
            return node

    return ast.dump(EqualityOrder().visit(a)) == ast.dump(EqualityOrder().visit(b))


def _zero_reason(environment: Mapping[str, Any]) -> str | None:
    place = environment.get(_RESIDENCE)
    if place not in sc._CANADIAN_RESIDENCE_PLACE_NAMES:
        return None
    if place == "Quebec":
        return "quebec"
    amt = environment.get("minimum_tax_is_payable")
    if type(amt) is not bool:
        return None
    if amt and place == "Manitoba":
        return "manitoba"
    if _CREDIT not in environment or _PAID not in environment:
        return None
    return "equality" if environment[_CREDIT] == environment[_PAID] else "calculation"


def _zero_and_computation_witnesses(cases, links):
    principal = {i for owner, i in links if owner == _LINE5}
    seen = set()
    for i in principal:
        a = cases[i]
        env_a = sc._case_input_formula_environment(a) or {}
        reason_a = _zero_reason(env_a)
        if reason_a != "calculation":
            continue
        a_value = sc._rulespec_runtime_decimal(
            sc._test_case_asserted_output_value(a, _LINE5)
        )
        if a_value is None or a_value == 0:
            continue
        for j in principal:
            b = cases[j]
            env_b = sc._case_input_formula_environment(b) or {}
            reason = _zero_reason(env_b)
            allowed = {
                "quebec": (_RESIDENCE,),
                "manitoba": (_RESIDENCE, "minimum_tax_is_payable"),
                "equality": (_CREDIT, _PAID),
            }.get(reason, ())
            if sc._rulespec_runtime_decimal(
                sc._test_case_asserted_output_value(b, _LINE5)
            ) == 0 and any(_same_world(a, b, key) for key in allowed):
                seen.add(reason)
    net_paths = set()
    cap = False
    for owner, i in links:
        case = cases[i]
        env = sc._case_input_formula_environment(case) or {}
        if owner == _LINE4:
            count = env.get("jurisdictions_tax_paid_count")
            if type(count) is int:
                if count > 1:
                    net_paths.add("multiple")
                elif env.get("resident_of_canada_for_part_of_year") is True:
                    net_paths.add("part_year")
                elif env.get("resident_of_canada_for_part_of_year") is False:
                    net_paths.add("ordinary")
        if owner == _LINE5 and _zero_reason(env) == "calculation":
            values = [
                sc._rulespec_runtime_decimal(
                    sc._test_case_asserted_output_value(case, n)
                )
                for n in (_LINE3, _LINE4, _TAX, _LINE5)
            ]
            if all(x is not None for x in values):
                excess, ratio, upper, result = values
                cap = cap or (upper < excess and upper < ratio and result == upper)
    missing = [
        "missing source zero pair: " + x
        for x in {"quebec", "manitoba", "equality"} - seen
    ]
    missing += [
        "unexecuted net-income path: " + x
        for x in {"multiple", "part_year", "ordinary"} - net_paths
    ]
    if not cap:
        missing.append("no actual active cap execution")
    return missing


def certify_worksheet_operations(
    payload: Mapping[str, Any],
    *,
    source_text: str,
    corpus_citation_path: str,
    source_context: Mapping[str, str | None] | None,
    test_cases: Sequence[object] | None,
    paths_evidence: CertifiedPathCoverage,
) -> WorksheetOperationCoverage:
    """One universal certificate; failures never add annual spans."""
    if corpus_citation_path != _PRIMARY or not source_context or not test_cases:
        return WorksheetOperationCoverage()
    raw_primary = source_context.get(_PRIMARY)
    secondary = source_context.get(_SECONDARY)
    if (
        not isinstance(raw_primary, str)
        or raw_primary.strip() != source_text
        or not isinstance(secondary, str)
        or not secondary
    ):
        return WorksheetOperationCoverage(
            unresolved=("missing or inconsistent source context",)
        )
    rules = paths._index(payload.get("rules"))
    inputs = paths._index(payload.get("inputs"))
    if rules is None or inputs is None or not set(_OUTPUTS) <= rules.keys():
        return WorksheetOperationCoverage(
            unresolved=("missing five-output declarations",)
        )
    verification = (payload.get("module") or {}).get("source_verification") or {}
    if (
        verification.get("source_sha256")
        != hashlib.sha256(raw_primary.encode()).hexdigest()
    ):
        return WorksheetOperationCoverage(unresolved=("raw source identity mismatch",))
    if paths_evidence.input_identity != path_input_identity(payload, test_cases):
        return WorksheetOperationCoverage(
            unresolved=("stale candidate/case path certificate",)
        )
    if paths_evidence.unresolved or not paths_evidence.condition_spans:
        return WorksheetOperationCoverage(
            unresolved=("incomplete monetary path certificate",)
        )
    owners = {x.owner for x in paths_evidence.obligations if x.certified}
    if not {_LINE2, _TAX} <= owners:
        return WorksheetOperationCoverage(
            unresolved=("missing monetary owner certificate",)
        )
    quoted = sc._annual_source_quoted_spans(source_text)
    if quoted is None:
        return WorksheetOperationCoverage(unresolved=("malformed source quotes",))
    # Exact closed operation texts are independently source-defined. Proofs
    # must additionally bind each selected computation below.
    operation_spans = {}
    for group, operations in _SOURCE_OPERATIONS.items():
        spans = []
        for citation, text in operations:
            body = source_text if citation == _PRIMARY else secondary
            if body.count(text) != 1:
                return WorksheetOperationCoverage(
                    unresolved=("unknown operation grammar: " + group,)
                )
            start = body.index(text)
            body_quotes = sc._annual_source_quoted_spans(body)
            if body_quotes is None or any(a <= start < b for a, b in body_quotes):
                return WorksheetOperationCoverage(unresolved=("quoted operation",))
            spans.append((citation, start, start + len(text), text))
        operation_spans[group] = spans
    secondary_operations = [
        (start, end)
        for citation, start, end, _ in operation_spans["net_income"]
        if citation == _SECONDARY
    ]
    if len(secondary_operations) != 1:
        return WorksheetOperationCoverage(
            unresolved=("secondary note owner unresolved",)
        )
    note_start, note_end = secondary_operations[0]
    # The complete supported note ends at its publisher footer and next note.
    # An exact operation substring cannot borrow ownership across new restrictions.
    if (note_start and secondary[note_start - 1] != "\n") or not secondary[
        note_end:
    ].startswith("\nT2209 E (25) Page 4 of 5\n\n(4) Basic federal tax\n"):
        return WorksheetOperationCoverage(
            unresolved=("secondary note boundary unresolved",)
        )
    intro = _annual_intro(source_text)
    if intro is None:
        return WorksheetOperationCoverage(unresolved=("annual form header missing",))
    prefix = sc._collapse_text(intro.group().split(" that you can deduct", 1)[0])
    cases = sc._typed_numeric_expected_cases(test_cases, rules) or ()
    constants = sc._constant_rule_environment(payload)
    expected = _expected_operations()
    declared_names = {
        node.id
        for tree in expected.values()
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    } - {_LINE2, _TAX, "min"}
    for name in declared_names:
        item = inputs.get(name, {})
        dtype, unit = (
            ("Text", None)
            if name == _RESIDENCE
            else ("Integer", None)
            if name == "jurisdictions_tax_paid_count"
            else ("Boolean", None)
            if name in {"minimum_tax_is_payable", "resident_of_canada_for_part_of_year"}
            else ("Money", "CAD")
        )
        if (
            item.get("dtype"),
            item.get("unit"),
            item.get("entity"),
            item.get("period"),
        ) != (dtype, unit, "Person", "Year"):
            return WorksheetOperationCoverage(
                unresolved=("wrong source input contract: " + name,)
            )
    dependencies = {
        d.name: d
        for r in paths_evidence.obligations
        if r.certified
        for d in r.selected_dependencies
    }
    errors = []
    for owner, descriptor in expected.items():
        resolver = _OperationResolver(
            payload, owner, "2025-01-01", "2025-12-31", source_text, _PRIMARY
        )
        actual = resolver.expand(ast.Name(id=owner))
        if (
            actual is None
            or resolver.errors
            or not _same_expression(actual, descriptor)
        ):
            errors.extend(resolver.errors or {"operation differs: " + owner})
        dependencies.update(resolver.dependencies)
    if errors:
        return WorksheetOperationCoverage(unresolved=tuple(sorted(set(errors))))
    for name, rule in rules.items():
        if name in dependencies:
            continue
        for path, cite, excerpt in sc._rule_source_excerpt_atoms(rule):
            if sc._formula_proof_version_index(path) is None:
                continue
            if any(
                cite == c and text in excerpt
                for spans in operation_spans.values()
                for c, _, _, text in spans
            ):
                errors.append("ambiguous additional operation owner: " + name)
    for name, dep in dependencies.items():
        if (dep.start, dep.end) != ("2025-01-01", "2025-12-31"):
            errors.append("partial selected annual interval: " + name)
        for field in ("effective_from", "effective_to"):
            if not any(
                path == f"versions[{dep.version}].{field}"
                and cite == _PRIMARY
                and prefix in sc._collapse_text(ex)
                and ex in source_text
                for path, cite, ex in sc._rule_source_excerpt_atoms(rules[name])
            ):
                errors.append("unowned annual date proof: " + name)
    subset = {**payload, "rules": [rules[n] for n in dependencies]}
    errors.extend(
        validate_rulespec_proofs(
            yaml.safe_dump(subset),
            require_policy_proofs=True,
            source_texts=source_context,
        ).issues
    )
    # Require exact independently specified operation text under its actual
    # owner. Secondary net-income operations belong to selected net helpers.
    for group, owner in {
        "line1": _LINE1,
        "line3": _LINE3,
        "line4": _LINE4,
        "line5": _LINE5,
    }.items():
        atoms = tuple(sc._rule_source_excerpt_atoms(rules[owner]))
        version = dependencies[owner].version
        for cite, _, _, text in operation_spans[group]:
            if not any(
                path == f"versions[{version}].formula" and c == cite and text in ex
                for path, c, ex in atoms
            ):
                errors.append("missing owned operation proof: " + owner)
    net_owners = (
        "t2036_net_income_for_credit",
        "net_income_for_part_year_resident",
        "net_income_for_multiple_jurisdictions",
    )
    for cite, _, _, text in operation_spans["net_income"]:
        required_owner = (
            "net_income_for_multiple_jurisdictions"
            if cite == _SECONDARY or text.startswith("If you paid tax")
            else "net_income_for_part_year_resident"
            if text.startswith("If you were a resident of Canada")
            else "t2036_net_income_for_credit"
        )
        dep = dependencies.get(required_owner)
        if (
            required_owner not in net_owners
            or dep is None
            or not any(
                path == f"versions[{dep.version}].formula" and cite == c and text in ex
                for path, c, ex in sc._rule_source_excerpt_atoms(rules[required_owner])
            )
        ):
            errors.append("missing owned net-income operation proof: " + required_owner)
    links = []
    for owner in _OUTPUTS:
        count = 0
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                continue
            asserted = sc._test_case_asserted_output_value(case, owner)
            if asserted is sc._UNRESOLVED_CONDITION_VALUE:
                continue
            if case.get("period") != {
                "period_kind": "tax_year",
                "start": "2025-01-01",
                "end": "2025-12-31",
            }:
                errors.append("nonannual asserted case: " + owner)
                continue
            env = sc._case_asserted_dependency_environment(
                rules, case, formula_environment=constants
            )
            execution = sc._case_formula_execution(
                rules[owner],
                case,
                formula_environment=constants,
                dependency_environment=env,
            )
            actual = (
                sc._formula_execution_runtime_value(execution)
                if execution
                else sc._UNRESOLVED_CONDITION_VALUE
            )
            if (
                actual is sc._UNRESOLVED_CONDITION_VALUE
                or not sc._asserted_formula_runtime_values_equal(
                    rules[owner], actual, asserted
                )
            ):
                errors.append("failed owner execution: " + owner)
            else:
                count += 1
                links.append((owner, index))
        if not count:
            errors.append("unexecuted source output: " + owner)
    errors.extend(_zero_and_computation_witnesses(cases, links))
    if errors:
        return WorksheetOperationCoverage(unresolved=tuple(sorted(set(errors))))
    frames = paths.cap_transfer_frames(source_text)
    caps = [frame for frame in frames if frame.kind == "numeric_upper_bound"]
    if len(caps) != 1:
        return WorksheetOperationCoverage(
            unresolved=("cap operation context unresolved",)
        )
    annual = {intro.span("year")}
    # Each additional occurrence belongs to an independently certified operation,
    # not to all occurrences of this numeric value in the document.
    role_patterns = (
        ("ontario_large", r"total foreign taxes paid for (?P<year>2025)"),
        ("other_amt", r"foreign country for (?P<year>2025)"),
        (
            "ontario_multi",
            r"paid\s+tax\s+to\s+more\s+than\s+one\s+jurisdiction\s+in\s+(?P<year>2025)",
        ),
    )
    for identity, pattern in role_patterns:
        records = [
            r
            for r in paths_evidence.obligations
            if r.identity == identity and r.certified
        ]
        for record in records:
            for span in record.source_spans:
                for match in re.finditer(pattern, span.text):
                    annual.add(
                        (
                            span.start + match.start("year"),
                            span.start + match.end("year"),
                        )
                    )
    for citation, start, _, text in operation_spans["net_income"]:
        if citation != _PRIMARY or not text.startswith("If you paid tax"):
            continue
        match = re.search(
            r"paid\s+tax\s+to\s+more\s+than\s+one\s+jurisdiction\s+in\s+(?P<year>2025)",
            text,
        )
        if match:
            annual.add((start + match.start("year"), start + match.end("year")))
    return WorksheetOperationCoverage(
        tuple(sorted(annual)),
        (caps[0].body.start, caps[0].body.end),
        _OUTPUTS,
        tuple(links),
        (),
        hashlib.sha256(source_text.encode()).hexdigest(),
    )
