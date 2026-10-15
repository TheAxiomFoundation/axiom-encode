"""Retain source obligations and compare known witness limits with the baseline.

The procedure exemption must preserve every nonexempt analyzer verdict.
Absolute rejection of the historical witness limitations is tracked in #1920.
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from decimal import Decimal

import pytest
from hypothesis import Phase, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.manual_procedure_differential import (
    analyze_with_manual_procedure_differential,
    manual_procedure_requires_paired_witness,
)


def _analyze(source: str, expression: str, cases: list[dict]):
    citation = "de/statute/estg/32a"
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation}
rules:
  - name: result
    kind: derived
    source: {citation}(1)
    versions:
      - formula: '{expression}'
"""
    return analyze_with_manual_procedure_differential(
        content,
        f"(1) {source}",
        corpus_citation_path=citation,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _paired_issue(result) -> bool:
    return any("paired positive/blocking" in issue for issue in result.issues)


def _assert_empty_exemption_issues(result, source, expression, cases):
    # The analyzer helper already ran this exact fixture with the list emptied.
    assert result.issues == result.empty_list_issues


def _case(name: object, inputs: dict, output: bool) -> dict:
    return {
        "name": str(name),
        "period": "2026",
        "input": inputs,
        "output": {"result": output},
    }


def test_round4_both_conjuncts_projected_keeps_the_obligation():
    source = "If both income and assets are greater than or equal to the income threshold, the applicant is ineligible."
    cases = [
        _case(
            assets,
            {"income": 99, "assets": assets, "income_threshold": 100},
            assets < 100,
        )
        for assets in (99, 100)
    ]
    result = _analyze(source, "if assets >= income_threshold: false else: true", cases)
    assert _paired_issue(result)


@pytest.mark.parametrize("qualification", ("If", "Unless"))
def test_round4_qualified_proviso_keeps_the_independent_obligation(qualification):
    source = f"{qualification} the applicant is disabled, the credit is allowed, provided income does not exceed the income threshold."
    cases = [
        _case(income, {"income": income, "income_threshold": 100}, income <= 100)
        for income in (99, 101)
    ]
    result = _analyze(source, "if income > income_threshold: false else: true", cases)
    assert _paired_issue(result)


@pytest.mark.parametrize("mirrored", (False, True))
@pytest.mark.parametrize("group_first", (False, True))
def test_round4_nested_parenthesized_source_is_satisfiable(mirrored, group_first):
    income = "income is greater than or equal to the income threshold"
    assets = "assets are greater than or equal to the asset threshold"
    income_formula = "income >= income_threshold"
    asset_formula = "assets >= asset_threshold"
    if mirrored:
        income = "the income threshold is less than or equal to income"
        assets = "the asset threshold is less than or equal to assets"
        income_formula = "income_threshold <= income"
        asset_formula = "asset_threshold <= assets"
    if group_first:
        condition = f"({income} and the applicant is a resident) or {assets}"
        formula = f"({income_formula} and is_resident) or {asset_formula}"
    else:
        condition = f"{income} or ({assets} and the applicant is a resident)"
        formula = f"{income_formula} or ({asset_formula} and is_resident)"
    cases = [
        _case(
            (income_value, assets_value, resident),
            {
                "income": income_value,
                "income_threshold": 100,
                "assets": assets_value,
                "asset_threshold": 100,
                "is_resident": resident,
            },
            not (
                (income_value >= 100 and resident) or assets_value >= 100
                if group_first
                else income_value >= 100 or (assets_value >= 100 and resident)
            ),
        )
        for income_value, assets_value, resident in itertools.product(
            (99, 100, 101), (99, 100, 101), (False, True)
        )
    ]
    source = f"If {condition}, the applicant is ineligible."
    rule = {
        "name": "result",
        "kind": "derived",
        "versions": [{"formula": f"if {formula}: false else: true"}],
    }
    assert manual_procedure_requires_paired_witness(source)
    assert all(
        sc._formula_execution_runtime_value(sc._case_formula_execution(rule, case))
        == case["output"]["result"]
        for case in cases
    )
    result = _analyze(source, f"if {formula}: false else: true", cases)
    assert not result.issues


@pytest.mark.parametrize("operand_side", ("subject", "object"))
def test_round4_compound_operand_projection_keeps_the_obligation(operand_side):
    if operand_side == "subject":
        source = "If total earned and unearned income is greater than or equal to the income threshold, the applicant is ineligible."
        formula = "if unearned_income >= income_threshold: false else: true"
        cases = [
            _case(
                value,
                {
                    "total_earned_and_unearned_income": 99,
                    "unearned_income": value,
                    "income_threshold": 100,
                },
                value < 100,
            )
            for value in (99, 100)
        ]
    else:
        source = "If income is greater than or equal to the earned and unearned income threshold, the applicant is ineligible."
        formula = "if income >= earned: false else: true"
        cases = [
            _case(
                value,
                {
                    "income": value,
                    "earned": 100,
                    "earned_and_unearned_income_threshold": 200,
                },
                value < 100,
            )
            for value in (99, 100)
        ]
    result = _analyze(source, formula, cases)
    assert manual_procedure_requires_paired_witness(source)
    # #1920 restores absolute rejection when the historical witness is fixed.
    _assert_empty_exemption_issues(result, source, formula, cases)


@pytest.mark.parametrize("operand_side", ("subject", "object"))
def test_shared_quantity_head_cannot_be_projected_from_enumerated_operands(
    operand_side,
):
    if operand_side == "subject":
        source = "If both earned and unearned income are greater than or equal to the income threshold, the applicant is ineligible."
        formula = "if earned >= income_threshold and unearned_income >= income_threshold: false else: true"
        cases = [
            _case(
                (earned, unearned),
                {
                    "earned": earned,
                    "earned_income": 99,
                    "unearned_income": unearned,
                    "income_threshold": 100,
                },
                not (earned >= 100 and unearned >= 100),
            )
            for earned, unearned in itertools.product((99, 100, 101), repeat=2)
        ]
    else:
        source = "If income is greater than or equal to both the earned and unearned income threshold, the applicant is ineligible."
        formula = "if income >= earned and income >= unearned_income_threshold: false else: true"
        cases = [
            _case(
                (earned, unearned),
                {
                    "income": 100,
                    "earned": earned,
                    "earned_income_threshold": 101,
                    "unearned_income_threshold": unearned,
                },
                not (100 >= earned and 100 >= unearned),
            )
            for earned, unearned in itertools.product((99, 100, 101), repeat=2)
        ]
    result = _analyze(source, formula, cases)
    assert manual_procedure_requires_paired_witness(source)
    # #1920 restores absolute rejection for object projection after the witness fix.
    if operand_side == "object":
        _assert_empty_exemption_issues(result, source, formula, cases)
    else:
        assert _paired_issue(result)


@pytest.mark.parametrize("attribute", ("resident", "disabled", "qualifies"))
def test_distinct_source_subjects_keep_independent_flag_obligations(attribute):
    predicate = "qualifies" if attribute == "qualifies" else f"is_{attribute}"
    phrase = {
        "qualifies": "qualifies",
        "resident": "is a resident",
        "disabled": "is disabled",
    }[attribute]
    source = (
        f"If the applicant {phrase} and the household {phrase}, the credit is allowed."
    )
    cases = [
        _case(
            value,
            {predicate: value, f"household_{predicate}": False},
            value,
        )
        for value in (False, True)
    ]
    result = _analyze(source, f"if {predicate}: true else: false", cases)
    assert manual_procedure_requires_paired_witness(source)
    # #1920 restores absolute rejection when the historical witness is fixed.
    _assert_empty_exemption_issues(
        result, source, f"if {predicate}: true else: false", cases
    )


@dataclass(frozen=True)
class _Predicate:
    name: str
    bound: str | None = None
    inclusive: bool = True
    mirrored: bool = False
    negated: bool = False
    subject: str = "applicant"

    @property
    def formula(self) -> str:
        if self.bound is None:
            return f"not {self.name}" if self.negated else self.name
        operator = ">=" if self.inclusive else ">"
        if self.mirrored:
            return f"{self.bound} {'<=' if self.inclusive else '<'} {self.name}"
        return f"{self.name} {operator} {self.bound}"

    @property
    def source(self) -> str:
        if self.bound is None:
            if self.name.endswith("qualifies"):
                predicate = "does not qualify" if self.negated else "qualifies"
                return f"the {self.subject} {predicate}"
            attribute = "disabled" if self.name.endswith("is_disabled") else "resident"
            predicate = "a resident" if attribute == "resident" else "disabled"
            return f"the {self.subject} is {'not ' if self.negated else ''}{predicate}"
        name = self.name.replace("_", " ")
        bound = self.bound.replace("_", " ")
        if self.mirrored:
            phrase = "is less than or equal to" if self.inclusive else "is less than"
            return f"the {bound} {phrase} {name}"
        phrase = "is greater than or equal to" if self.inclusive else "is greater than"
        return f"{name} {phrase} the {bound}"

    def evaluate(self, inputs: dict) -> bool:
        if self.bound is None:
            value = inputs[self.name]
            return not value if self.negated else value
        left, right = inputs[self.name], inputs[self.bound]
        return left >= right if self.inclusive else left > right


@dataclass(frozen=True)
class _Expression:
    connector: str
    left: _Predicate | _Expression
    right: _Predicate | _Expression

    @property
    def source(self) -> str:
        return f"({self.left.source} {self.connector} {self.right.source})"

    @property
    def formula(self) -> str:
        return f"({self.left.formula} {self.connector} {self.right.formula})"

    def evaluate(self, inputs: dict) -> bool:
        if self.connector == "and":
            return self.left.evaluate(inputs) and self.right.evaluate(inputs)
        return self.left.evaluate(inputs) or self.right.evaluate(inputs)


def _predicates(expression: _Predicate | _Expression) -> tuple[_Predicate, ...]:
    if isinstance(expression, _Predicate):
        return (expression,)
    return _predicates(expression.left) + _predicates(expression.right)


def _replace(expression, selected: _Predicate, replacement: _Predicate):
    if isinstance(expression, _Predicate):
        return replacement if expression == selected else expression
    return _Expression(
        expression.connector,
        _replace(expression.left, selected, replacement),
        _replace(expression.right, selected, replacement),
    )


def _drop(expression: _Expression, selected: _Predicate):
    if expression.left == selected:
        return expression.right
    if expression.right == selected:
        return expression.left
    return _Expression(
        expression.connector,
        _drop(expression.left, selected)
        if isinstance(expression.left, _Expression)
        else expression.left,
        _drop(expression.right, selected)
        if isinstance(expression.right, _Expression)
        else expression.right,
    )


def _generated_cases(expression, threshold: int, *, permission: bool):
    predicates = _predicates(expression)
    names = tuple(dict.fromkeys(predicate.name for predicate in predicates))
    bounds = {predicate.bound: threshold for predicate in predicates if predicate.bound}
    # Enumerated objects need distinct boundary combinations; two equal fixed
    # thresholds cannot demonstrate that either conjunct is independent.
    shared_names = {
        name
        for name in names
        if len({predicate.bound for predicate in predicates if predicate.name == name})
        > 1
    }
    if shared_names:
        names = tuple(name for name in names if name not in shared_names) + tuple(
            bounds
        )
        bounds = {name: threshold for name in shared_names}
    domains = [
        (threshold - 1, threshold, threshold + 1)
        if name in shared_names
        or any(
            predicate.bound and name in (predicate.name, predicate.bound)
            for predicate in predicates
        )
        else (False, True)
        for name in names
    ]
    return [
        _case(
            values,
            {
                **bounds,
                **dict(zip(names, values, strict=True)),
                "other_income": threshold + 2,
                "other_income_threshold": threshold + 2,
            },
            expression.evaluate({**bounds, **dict(zip(names, values, strict=True))})
            is permission,
        )
        for values in itertools.product(*domains)
    ]


def _formula(expression, permission: bool):
    return (
        f"if {expression.formula}: {'true' if permission else 'false'} "
        f"else: {'false' if permission else 'true'}"
    )


_SHAPES = (
    "flat",
    "nested_left",
    "nested_right",
    "compound_subject",
    "compound_object",
    "both_subjects",
    "both_objects",
    "shared_subjects",
    "shared_objects",
    "independent_subjects",
    "qualified_if",
    "qualified_unless",
    "qualified_provided",
)


@st.composite
def _condition_trees(draw, shape):
    first = _Predicate(
        "total_earned_and_unearned_income" if shape == "compound_subject" else "income",
        "earned_and_unearned_income_threshold"
        if shape == "compound_object"
        else "income_threshold",
        inclusive=draw(st.booleans()),
        mirrored=draw(st.booleans()),
    )
    second = _Predicate(
        "assets",
        "asset_threshold",
        inclusive=not first.inclusive
        if shape.startswith("nested_")
        else draw(st.booleans()),
        mirrored=draw(st.booleans()),
    )
    resident = _Predicate("is_resident")
    connector = draw(st.sampled_from(("and", "or")))
    permission = draw(st.booleans())
    proposition = "the credit is allowed" if permission else "the credit is not allowed"
    if shape.startswith("qualified_"):
        marker = shape.removeprefix("qualified_")
        flag = _Predicate("is_disabled", negated=marker == "unless")
        # A does-not-exceed proviso is the complement of a strict comparison.
        bound = _Predicate("income_threshold", "income", inclusive=True)
        expression = _Expression("and", flag, bound)
        source = (
            f"{'Provided that' if marker == 'provided' else marker.capitalize()} "
            f"the applicant is disabled, {proposition}, "
            "provided income does not exceed the income threshold."
        )
    elif shape in {"both_subjects", "shared_subjects"}:
        shared = shape == "shared_subjects"
        modifiers = draw(st.sampled_from((("earned", "unearned"), ("gross", "net"))))
        quantity = draw(
            st.sampled_from(("income", "asset", "benefit amount", "tax credit"))
        )
        second = _Predicate(
            f"{modifiers[1]} {quantity}".replace(" ", "_") if shared else "assets",
            "income_threshold",
            first.inclusive,
        )
        first = _Predicate(
            f"{modifiers[0]} {quantity}".replace(" ", "_") if shared else "income",
            "income_threshold",
            first.inclusive,
        )
        expression = _Expression("and", first, second)
        relation = "greater than or equal to" if first.inclusive else "greater than"
        subjects = (
            f"{modifiers[0]} and {modifiers[1]} {quantity}"
            if shared
            else "income and assets"
        )
        source = (
            f"If both {subjects} are {relation} the income threshold, {proposition}."
        )
    elif shape in {"both_objects", "shared_objects"}:
        shared = shape == "shared_objects"
        modifiers = draw(st.sampled_from((("earned", "unearned"), ("gross", "net"))))
        quantity = draw(
            st.sampled_from(("income", "asset", "benefit amount", "tax credit"))
        )
        second = _Predicate(
            "income",
            f"{modifiers[1]} {quantity} threshold".replace(" ", "_")
            if shared
            else "asset_threshold",
            first.inclusive,
        )
        first = _Predicate(
            "income",
            f"{modifiers[0]} {quantity} threshold".replace(" ", "_")
            if shared
            else "income_threshold",
            first.inclusive,
        )
        expression = _Expression("and", first, second)
        relation = "greater than or equal to" if first.inclusive else "greater than"
        objects = (
            f"the {modifiers[0]} and {modifiers[1]} {quantity} threshold"
            if shared
            else "the income threshold and the asset threshold"
        )
        source = f"If income is {relation} both {objects}, {proposition}."
    else:
        if shape == "independent_subjects":
            subject = draw(
                st.sampled_from(
                    ("household", "person", "individual", "assistance group")
                )
            )
            attribute = draw(st.sampled_from(("resident", "disabled", "qualifies")))
            predicate = "qualifies" if attribute == "qualifies" else f"is_{attribute}"
            expression = _Expression(
                "and",
                first,
                _Expression(
                    connector,
                    _Predicate(predicate),
                    _Predicate(
                        f"{subject.replace(' ', '_')}_{predicate}", subject=subject
                    ),
                ),
            )
        elif shape == "nested_left":
            expression = _Expression(
                connector,
                _Expression("and" if connector == "or" else "or", first, resident),
                second,
            )
        elif shape == "nested_right":
            expression = _Expression(
                connector,
                first,
                _Expression("and" if connector == "or" else "or", second, resident),
            )
        elif shape == "flat":
            expression = _Expression(connector, first, resident)
        else:
            expression = first
        source = f"If {expression.source}, {proposition}."
    return source, expression, permission, shape


@pytest.mark.parametrize("shape", _SHAPES)
@settings(max_examples=8, deadline=None, phases=(Phase.generate,), database=None)
@given(data=st.data(), threshold=st.integers(min_value=2, max_value=1000))
def test_generated_source_conditions_keep_their_paired_obligation(
    shape, data, threshold
):
    scenario = data.draw(_condition_trees(shape))
    source, expression, permission, shape = scenario
    cases = _generated_cases(expression, threshold, permission=permission)
    assert manual_procedure_requires_paired_witness(source)

    numeric = next(
        predicate for predicate in _predicates(expression) if predicate.bound
    )
    dishonest = [
        _replace(
            expression,
            numeric,
            _Predicate(
                numeric.name, numeric.bound, not numeric.inclusive, numeric.mirrored
            ),
        ),
        _replace(
            expression,
            numeric,
            _Predicate(
                "other_income", numeric.bound, numeric.inclusive, numeric.mirrored
            ),
        ),
        _replace(
            expression,
            numeric,
            _Predicate(
                numeric.name,
                "other_income_threshold",
                numeric.inclusive,
                numeric.mirrored,
            ),
        ),
    ]
    if isinstance(expression, _Expression):
        dishonest.extend(
            _drop(expression, predicate) for predicate in _predicates(expression)
        )
    for defect in dishonest:
        # Preserve the truthfully asserted defect grids from the earlier
        # review. The source clause remains an obligation for every grid.
        defect_cases = [
            {**case, "output": {"result": defect.evaluate(case["input"]) is permission}}
            for case in cases
        ]
        assert manual_procedure_requires_paired_witness(source), (
            shape,
            source,
            defect.formula,
            defect_cases,
        )

    # Preserve the sparse companion grid without introducing a new way for
    # procedure classification to remove its source condition.
    first_inputs = cases[0]["input"]
    if isinstance(expression, _Expression):
        independent_name = next(
            name
            for name in first_inputs
            if any(case["input"][name] != first_inputs[name] for case in cases)
        )
        incomplete = [
            case
            for case in cases
            if case["input"][independent_name] == first_inputs[independent_name]
        ]
        assert manual_procedure_requires_paired_witness(source), (
            shape,
            source,
            independent_name,
            incomplete,
        )


def test_existing_numeric_runtime_keeps_decimal_equality_representation():
    source = "If income is greater than or equal to the income threshold, the applicant is ineligible."
    cases = [
        _case(
            value,
            {"income": value, "income_threshold": 0.1},
            value < Decimal("0.1"),
        )
        for value in (Decimal("0.0"), Decimal("0.1"), Decimal("0.2"))
    ]
    rule = {
        "name": "result",
        "kind": "derived",
        "versions": [{"formula": "if income >= income_threshold: false else: true"}],
    }
    assert manual_procedure_requires_paired_witness(source)
    assert all(
        sc._formula_execution_runtime_value(sc._case_formula_execution(rule, case))
        == case["output"]["result"]
        for case in cases
    )


def test_generated_provided_example_keeps_the_inclusive_obligation():
    source = "Provided that the applicant is disabled, the credit is not allowed, provided income does not exceed the income threshold."
    expression = _Expression(
        "and", _Predicate("is_disabled"), _Predicate("income_threshold", "income")
    )
    strict = _Expression(
        "and",
        _Predicate("is_disabled"),
        _Predicate("income_threshold", "income", False),
    )
    cases = [
        {**case, "output": {"result": not strict.evaluate(case["input"])}}
        for case in _generated_cases(expression, 2, permission=False)
    ]
    result = _analyze(source, _formula(strict, False), cases)
    assert manual_procedure_requires_paired_witness(source)
    # #1920 restores absolute rejection when the historical witness is fixed.
    _assert_empty_exemption_issues(result, source, _formula(strict, False), cases)


def test_literal_inclusive_comparison_preserves_existing_numeric_evidence():
    source = "If income is greater than or equal to 100, the applicant is ineligible."
    cases = [
        _case(income, {"income": income}, income < 100) for income in (99, 100, 101)
    ]
    result = _analyze(source, "if income >= 100: false else: true", cases)
    # The existing numeric-representation gate is independent of this paired
    # source-condition obligation and remains visible in the analyzer result.
    assert not _paired_issue(result)


# Keep the absolute checks that passed for every original baseline example.
# Mixed or failing witness checks retain differential coverage for #1920.
_BASELINE_REJECTED_DEFECTS = {
    "flat": {4},
    "nested_left": {4},
    "nested_right": {5},
    "compound_subject": {0, 1, 2},
    "compound_object": {0, 1, 2},
    "both_subjects": {0, 1, 2, 3, 4},
    "both_objects": {0, 1, 2, 3, 4},
    "shared_subjects": {0, 1, 2, 3, 4},
    "shared_objects": {0, 1, 2, 3, 4},
    "independent_subjects": {2},
    "qualified_if": {2, 3},
    "qualified_unless": {2, 3},
    "qualified_provided": set(),
}
_BASELINE_REJECTED_SPARSE_SHAPES = {
    "both_subjects",
    "both_objects",
    "shared_subjects",
    "shared_objects",
    "independent_subjects",
    "qualified_if",
    "qualified_unless",
}


@pytest.mark.parametrize("shape", _SHAPES)
@settings(
    max_examples=8,
    deadline=None,
    phases=(Phase.generate,),
    database=None,
    derandomize=True,
)
@given(data=st.data(), threshold=st.integers(min_value=2, max_value=1000))
def test_generated_source_conditions_require_semantic_equivalence(
    shape, data, threshold
):
    scenario = data.draw(_condition_trees(shape))
    source, expression, permission, shape = scenario
    cases = _generated_cases(expression, threshold, permission=permission)
    # #1920 restores absolute acceptance/rejection after the historical witness fix.
    honest = _analyze(source, _formula(expression, permission), cases)
    _assert_empty_exemption_issues(
        honest, source, _formula(expression, permission), cases
    )
    if shape not in {
        "compound_subject",
        "compound_object",
        "both_subjects",
        "both_objects",
        "shared_subjects",
        "shared_objects",
        "qualified_unless",
    }:
        assert not honest.issues

    numeric = next(
        predicate for predicate in _predicates(expression) if predicate.bound
    )
    dishonest = [
        _replace(
            expression,
            numeric,
            _Predicate(
                numeric.name, numeric.bound, not numeric.inclusive, numeric.mirrored
            ),
        ),
        _replace(
            expression,
            numeric,
            _Predicate(
                "other_income", numeric.bound, numeric.inclusive, numeric.mirrored
            ),
        ),
        _replace(
            expression,
            numeric,
            _Predicate(
                numeric.name,
                "other_income_threshold",
                numeric.inclusive,
                numeric.mirrored,
            ),
        ),
    ]
    if isinstance(expression, _Expression):
        dishonest.extend(
            _drop(expression, predicate) for predicate in _predicates(expression)
        )
    for defect_index, defect in enumerate(dishonest):
        # Assert what the dishonest formula actually does, so forged outputs
        # cannot distort the analyzer's comparison with the baseline.
        defect_cases = [
            {**case, "output": {"result": defect.evaluate(case["input"]) is permission}}
            for case in cases
        ]
        result = _analyze(source, _formula(defect, permission), defect_cases)
        _assert_empty_exemption_issues(
            result, source, _formula(defect, permission), defect_cases
        )
        if defect_index in _BASELINE_REJECTED_DEFECTS[shape]:
            assert _paired_issue(result), (shape, source, defect.formula, result.issues)

    # Preserve the sparse grids that fix an independent condition on one side
    # of its boundary and compare their full analyzer verdict with baseline.
    first_inputs = cases[0]["input"]
    if isinstance(expression, _Expression):
        independent_name = next(
            name
            for name in first_inputs
            if any(case["input"][name] != first_inputs[name] for case in cases)
        )
        incomplete = [
            case
            for case in cases
            if case["input"][independent_name] == first_inputs[independent_name]
        ]
        result = _analyze(source, _formula(expression, permission), incomplete)
        _assert_empty_exemption_issues(
            result, source, _formula(expression, permission), incomplete
        )
        if shape in _BASELINE_REJECTED_SPARSE_SHAPES:
            assert _paired_issue(result), (
                shape,
                source,
                independent_name,
                result.issues,
            )
