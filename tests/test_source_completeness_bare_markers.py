"""Bare outlines must keep branch and condition evidence separate."""

import functools
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

FIXTURES = Path(__file__).parent / "fixtures" / "nz_bare_labels"
CITATION = "nz/statute/act/public/2018/0032/section/71"


SOURCE = """In this subpart, unless the context otherwise requires,—
qualifying benefit means—
a
a main benefit under this Act;
b
a benefit specified in this section.
"""


def test_bare_paragraphs_do_not_join_context_exception_to_first_branch():
    branches = sc.recognize_source_structure(SOURCE)
    paragraphs = [branch for branch in branches if branch.path]
    assert [branch.path for branch in paragraphs] == [("a",), ("b",)]
    assert paragraphs[0].text.strip() == "a main benefit under this Act;"
    conditions = sc._source_exception_branches(
        SOURCE,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    assert len(conditions) == 1
    assert "unless the context otherwise requires" in conditions[0].text
    assert "main benefit" not in conditions[0].text
    assert "a a main" not in sc._collapse_text(conditions[0].text)


def test_ambiguous_bare_sequence_keeps_legacy_source_obligation():
    ambiguous = SOURCE.replace("\nb\n", "\nd\n")
    branches = sc.recognize_source_structure(ambiguous)
    conditions = sc._source_exception_branches(
        ambiguous,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    assert conditions
    assert "unless the context otherwise requires" in conditions[0].text


def test_bare_chapeau_retains_numeric_boundary_and_condition_obligations():
    source = (
        "A person qualifies only if the person is at least 65 years old:—\n"
        "a\nthe person has a main benefit; or\n"
        "b\nthe person receives a pension.\n"
    )
    branches = sc.recognize_source_structure(source)
    chapeau = next(branch for branch in branches if branch.kind == "source-unit")
    assert chapeau.start == 0
    assert chapeau.text == source[: chapeau.end]
    assert "65 years old" in chapeau.text
    boundaries = sc._source_boundary_obligations(
        branches,
        narrative_formula_branches=(),
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
    )
    assert any(branch == chapeau and value.value == 65 for branch, value in boundaries)
    conditions = sc._source_exception_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )
    assert any("at least 65" in branch.text for branch in conditions)


def test_bare_numeric_branch_never_remasks_operative_formula_amount():
    prefix = (
        "1\nThe following benefit conditions apply:—\n"
        "a\nA person must receive a benefit; or\n"
        "b\nA person must receive a pension.\n"
    )
    opening = "2\nThe benefit must be calculated as "
    # Put the amount at the subsection marker's absolute source coordinate,
    # where incorrectly applying that coordinate to local text would erase it.
    padding = " " * (len(prefix) - len(opening))
    source = f"{prefix}{opening}{padding}50 dollars.\n"
    branch = next(
        branch
        for branch in sc.recognize_source_structure(source)
        if branch.path == ("2",)
    )
    assert not branch.structural_numeric_spans
    occurrences = sc._formula_branch_computation_occurrences(
        branch,
        interval=None,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
    )
    assert [occurrence.value for occurrence in occurrences] == [50]


def test_verbatim_s16_computation_chapeau_retains_first_child_operand():
    source = (FIXTURES / "property_relationships_s16.txt").read_text()
    branches = sc.recognize_source_structure(source)
    formulas = sc._source_formula_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )
    computation = next(branch for branch in formulas if branch.path == ("2", "d"))
    assert "relationship property is to be divided" in computation.text
    assert "other spouse or partner ( party B ) is included" in computation.text
    assert "\ni\n" not in computation.text
    assert (
        computation.text
        == sc.mask_bare_labels(source)[computation.start : computation.end]
    )
    assert {("2", "d", "i"), ("2", "d", "ii")} <= {branch.path for branch in branches}


def test_root_bare_computation_chapeau_retains_first_child_operand():
    source = (
        "The payment must be calculated as:—\n"
        "a\n50 dollars for an eligible person; or\n"
        "b\n25 dollars otherwise.\n"
    )
    branches = sc.recognize_source_structure(source)
    formulas = sc._source_formula_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )
    assert len(formulas) == 2
    assert formulas[0].path == ()
    assert "50 dollars for an eligible person" in formulas[0].text
    assert "25 dollars" not in formulas[0].text
    assert formulas[1].path == ("b",)
    assert "25 dollars" in formulas[1].text
    assert formulas[1].computation_chapeau is not None


@pytest.mark.parametrize("dishonest", [False, True])
def test_bare_computation_operand_requires_execution_of_its_amount(dishonest):
    source = (
        "1\nThe payment must be calculated as:—\n"
        "a\n50 dollars for an eligible person; or\n"
        "b\n25 dollars otherwise.\n"
    )
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [{"name": "eligible", "dtype": "Bool"}],
        "rules": [
            _source_rule(
                "payment",
                "0" if dishonest else "if eligible: eligible_amount else: other_amount",
                "The payment must be calculated as:—",
                f"{CITATION}(1)",
                dtype="Money",
            )
        ],
    }
    for name, amount, excerpt, label in (
        ("eligible_amount", 50, "50 dollars for an eligible person;", "a"),
        ("other_amount", 25, "25 dollars otherwise.", "b"),
    ):
        rule = _source_rule(
            name,
            str(amount),
            excerpt,
            f"{CITATION}(1)({label})",
            dtype="Money",
        )
        rule["kind"] = "parameter"
        payload["rules"].append(rule)
    cases = [
        {
            "period": "2026-07-01",
            "input": {"eligible": eligible},
            "output": {"payment": 0 if dishonest else 50 if eligible else 25},
        }
        for eligible in (False, True)
    ]
    result = _analyze(source, payload, cases)
    formula_issues = [issue for issue in result.issues if "formula branch" in issue]
    assert bool(formula_issues) == dishonest


def test_inherited_parent_operation_does_not_require_duplicate_child_witness():
    source = (
        "1\nThe payment must be calculated as:—\n"
        "a\nthe income multiplied by 2; or\n"
        "b\nthe income multiplied by 3.\n"
    )
    branches = sc.recognize_source_structure(source)
    formulas = sc._source_formula_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )
    assert len(formulas) == 2
    assert [branch.path for branch in formulas] == [("1",), ("1", "b")]
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {"name": "eligible", "dtype": "Bool"},
            {"name": "income", "dtype": "Money"},
        ],
        "rules": [
            _source_rule(
                "payment",
                "if eligible: income * 2 else: income * 3",
                source,
                f"{CITATION}(1)",
                dtype="Money",
            )
        ],
    }
    for factor in (2, 3):
        rule = _source_rule(
            f"payment_factor_{factor}",
            str(factor),
            f"the income multiplied by {factor}",
            f"{CITATION}(1)",
            dtype="Int32",
        )
        rule["kind"] = "parameter"
        payload["rules"].append(rule)
    cases = [
        {
            "period": "2026-07-01",
            "input": {"eligible": eligible, "income": 10},
            "output": {"payment": 20 if eligible else 30},
        }
        for eligible in (True, False)
    ]
    result = _analyze(source, payload, cases)
    assert not any("formula branch" in issue for issue in result.issues)


def test_inherited_computation_stops_before_later_deferred_child():
    source = (
        "1\nThe payment must be calculated as:—\n"
        "a\n50 dollars for an eligible person\n"
        "b\nthe amount stated in the federal return copy that must be filed.\n"
    )
    branches = sc.recognize_source_structure(source)
    deferred = next(branch for branch in branches if branch.path == ("1", "b"))
    formulas = sc._source_formula_branches(
        source,
        branches=branches,
        active_branches=[branch for branch in branches if branch != deferred],
        deferred_paths={("1", "b")},
    )
    assert len(formulas) == 1
    assert "50 dollars" in formulas[0].text
    assert "federal return" not in formulas[0].text
    assert formulas[0].end <= deferred.start


def test_inherited_computation_retains_active_blocks_after_deferred_child():
    source = (
        "1\nThe payment must be calculated as:—\n"
        "a\n50 dollars for the first payment; and\n"
        "b\nthe amount stated in the federal return copy that must be filed; and\n"
        "c\n25 dollars for the final payment.\n"
    )
    branches = sc.recognize_source_structure(source)
    deferred = next(branch for branch in branches if branch.path == ("1", "b"))
    formulas = sc._source_formula_branches(
        source,
        branches=branches,
        active_branches=[branch for branch in branches if branch != deferred],
        deferred_paths={("1", "b")},
    )
    assert len(formulas) == 2
    assert "50 dollars" in formulas[0].text
    assert "25 dollars" in formulas[1].text
    assert formulas[1].path == ("1", "c")
    assert formulas[1].computation_chapeau is not None
    assert all("federal return" not in branch.text for branch in formulas)
    for branch in formulas:
        assert branch.text == sc.mask_bare_labels(source)[branch.start : branch.end]


@pytest.mark.parametrize("proof_style", ["whole-source", "chapeau"])
@pytest.mark.parametrize("operand_style", ["scalar", "multiplication"])
@pytest.mark.parametrize("dishonest", [False, True])
def test_deferred_middle_child_keeps_separate_active_computation_witnesses(
    proof_style, operand_style, dishonest
):
    chapeau = "The payment must be calculated as:—"
    operands = (
        ("50 dollars for the first payment", "25 dollars for the other payment")
        if operand_style == "scalar"
        else ("the income multiplied by 2", "the income multiplied by 3")
    )
    source = (
        f"1\n{chapeau}\n"
        f"a\n{operands[0]}; or\n"
        "b\nthe amount stated in the federal return copy that must be filed; or\n"
        f"c\n{operands[1]}.\n"
    )
    first, second = (50, 25) if operand_style == "scalar" else (2, 3)
    formula = (
        "if eligible: first_amount else: "
        + ("first_amount" if dishonest else "other_amount")
        if operand_style == "scalar"
        else "if eligible: income * 2 else: income * " + ("2" if dishonest else "3")
    )
    payload = {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {"corpus_citation_path": CITATION},
            "deferred_outputs": [
                {
                    "output": "nz:statutes/act/public/2018/0032/section/71/1/b#return_amount",
                    "reason": (
                        f"{CITATION}(1)(b) cannot be encoded until the runtime has a "
                        "document-filing capability for the required federal return copy."
                    ),
                }
            ],
        },
        "inputs": [
            {"name": "eligible", "dtype": "Bool"},
            {"name": "income", "dtype": "Money"},
        ],
        "rules": [
            _source_rule(
                "payment",
                formula,
                source if proof_style == "whole-source" else chapeau,
                f"{CITATION}(1)",
                dtype="Money",
            )
        ],
    }
    for name, value, excerpt, label in (
        ("first_amount", first, operands[0], "a"),
        ("other_amount", second, operands[1], "c"),
    ):
        rule = _source_rule(
            name, str(value), excerpt, f"{CITATION}(1)({label})", dtype="Money"
        )
        rule["kind"] = "parameter"
        payload["rules"].append(rule)
    branches = sc.recognize_source_structure(source)
    covered, issues = sc._deferred_coverage(
        payload, corpus_citation_path=CITATION, source_text=source, branches=branches
    )
    assert covered == {("1", "b")}
    assert not issues
    formulas = sc._source_formula_branches(
        source,
        branches=branches,
        active_branches=[branch for branch in branches if branch.path not in covered],
        deferred_paths=covered,
    )
    assert [branch.path for branch in formulas] == [("1",), ("1", "c")]
    assert all("federal return" not in branch.text for branch in formulas)
    cases = [
        {
            "period": "2026-07-01",
            "input": {"eligible": eligible, "income": 10},
            "output": {
                "payment": (first if eligible or dishonest else second)
                * (1 if operand_style == "scalar" else 10)
            },
        }
        for eligible in (True, False)
    ]
    result = _analyze(source, payload, cases)
    assert any("formula branch" in issue for issue in result.issues) == dishonest


@pytest.mark.parametrize("proof_style", ["whole-source", "chapeau"])
@pytest.mark.parametrize("cue_style", ["multiplier", "percentage", "percentage-child"])
@pytest.mark.parametrize("dishonest", [False, True])
def test_common_chapeau_factor_is_demanded_on_later_active_children(
    proof_style, cue_style, dishonest
):
    factor = 2 if cue_style == "multiplier" else 0.5
    chapeau = (
        "The payment must be calculated as the amount multiplied by 2 for:—"
        if cue_style == "multiplier"
        else "The payment must be calculated as 50% of:—"
    )
    second = (
        "the person's savings multiplied by 2."
        if cue_style == "percentage-child"
        else "the person's savings for the second payment."
    )
    source = (
        f"1\n{chapeau}\na\nthe person's income for the first payment; or\nb\n{second}\n"
    )
    wrong_factor = 3 if factor == 2 else 0.25
    actual_factor = wrong_factor if dishonest else factor
    child_factor = 2 if cue_style == "percentage-child" else 1
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {"name": "eligible", "dtype": "Bool"},
            {"name": "income", "dtype": "Money"},
            {"name": "savings", "dtype": "Money"},
        ],
        "rules": [
            _source_rule(
                "payment",
                f"if eligible: income * {factor} else: savings * {actual_factor}"
                + (" * 2" if child_factor == 2 else ""),
                source if proof_style == "whole-source" else chapeau,
                f"{CITATION}(1)",
                dtype="Money",
            )
        ],
    }
    for name, value, excerpt, path in (
        ("common_factor", factor, chapeau, f"{CITATION}(1)"),
        ("child_factor", child_factor, second, f"{CITATION}(1)(b)"),
    ):
        if name == "child_factor" and child_factor == 1:
            continue
        rule = _source_rule(name, str(value), excerpt, path, dtype="Decimal")
        rule["kind"] = "parameter"
        payload["rules"].append(rule)
    cases = [
        {
            "period": "2026-07-01",
            "input": {"eligible": eligible, "income": 10, "savings": 20},
            "output": {
                "payment": 10 * factor
                if eligible
                else 20 * actual_factor * child_factor
            },
        }
        for eligible in (True, False)
    ]
    result = _analyze(source, payload, cases)
    assert any("formula branch" in issue for issue in result.issues) == dishonest


def test_common_factor_preserves_original_coordinates_and_excludes_cue_threshold():
    source = (
        "1\nIf the income is at least 1000, the payment must be calculated as "
        "the income multiplied by 2 for:—\n"
        "a\nthe first payment; or\n"
        "b\nIf the period is at least 2 days, the person's income.\n"
    )
    extract = functools.partial(
        extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
    )
    branches = sc.recognize_source_structure(source)
    formulas = sc._source_formula_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )
    later = next(branch for branch in formulas if branch.path == ("1", "b"))
    assert later.text == source[later.start : later.end]
    occurrences = sc._formula_branch_computation_occurrences(
        later,
        interval=sc._formula_branch_interval(
            later, extract_numeric_occurrences=extract
        ),
        extract_numeric_occurrences=extract,
    )
    assert [occurrence.value for occurrence in occurrences] == [2]
    factor = occurrences[0]
    assert factor.start < 0
    assert source[later.start + factor.start : later.start + factor.end] == "2"
    assert "multiplied by 2" in source[: later.start + factor.end]


@pytest.mark.parametrize("dishonest", [False, True])
def test_nested_computation_chapeaux_retain_outer_factor_and_inner_operation(dishonest):
    source = (
        "1\nThe payment must be calculated as 50% of:—\n"
        "a\nthe sum of:—\n"
        "i\nthe person's income; and\n"
        "ii\nthe person's savings.\n"
    )
    branches = sc.recognize_source_structure(source)
    formulas = sc._source_formula_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )
    later = next(branch for branch in formulas if branch.path == ("1", "a", "ii"))
    assert [context[2] for context in later.computation_chapeaux] == [
        "The payment must be calculated as 50% of:—",
        "the sum of:—",
    ]
    assert later.text == source[later.start : later.end]
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {"name": "income", "dtype": "Money"},
            {"name": "savings", "dtype": "Money"},
        ],
        "rules": [
            _source_rule(
                "payment",
                "(income + savings) * " + ("1" if dishonest else "0.5"),
                source,
                f"{CITATION}(1)",
                dtype="Money",
            )
        ],
    }
    rule = _source_rule(
        "common_factor", "0.5", "50% of", f"{CITATION}(1)", dtype="Decimal"
    )
    rule["kind"] = "parameter"
    payload["rules"].append(rule)
    cases = [
        {
            "period": "2026-07-01",
            "input": {"income": income, "savings": 20},
            "output": {"payment": (income + 20) * (1 if dishonest else 0.5)},
        }
        for income in (10, 30)
    ]
    result = _analyze(source, payload, cases)
    assert any("formula branch" in issue for issue in result.issues) == dishonest


@pytest.mark.parametrize("dishonest", [False, True])
def test_deferred_first_child_keeps_active_amount_execution_obligation(dishonest):
    source = (
        "1\nThe payment must be calculated as:—\n"
        "a\nthe amount stated in the federal return copy that must be filed; or\n"
        "b\n25 dollars for the active payment.\n"
    )
    payload = {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {"corpus_citation_path": CITATION},
            "deferred_outputs": [
                {
                    "output": "nz:statutes/act/public/2018/0032/section/71/1/a#return_amount",
                    "reason": (
                        f"{CITATION}(1)(a) cannot be encoded until the runtime has a "
                        "document-filing capability for the required federal return copy."
                    ),
                }
            ],
        },
        "rules": [
            _source_rule(
                "active_payment",
                "0" if dishonest else "active_amount",
                "The payment must be calculated as:—",
                f"{CITATION}(1)",
                dtype="Money",
            )
        ],
    }
    amount = _source_rule(
        "active_amount",
        "25",
        "25 dollars for the active payment.",
        f"{CITATION}(1)(b)",
        dtype="Money",
    )
    amount["kind"] = "parameter"
    payload["rules"].append(amount)
    branches = sc.recognize_source_structure(source)
    covered, issues = sc._deferred_coverage(
        payload, corpus_citation_path=CITATION, source_text=source, branches=branches
    )
    assert covered == {("1", "a")}
    assert not issues
    formulas = sc._source_formula_branches(
        source,
        branches=branches,
        active_branches=[branch for branch in branches if branch.path not in covered],
        deferred_paths=covered,
    )
    assert len(formulas) == 1
    assert "25 dollars" in formulas[0].text
    assert "federal return" not in formulas[0].text
    cases = [
        {
            "period": "2026-07-01",
            "input": {},
            "output": {"active_payment": 0 if dishonest else 25},
        }
    ]
    result = _analyze(source, payload, cases)
    assert any("formula branch" in issue for issue in result.issues) == dishonest


def test_verbatim_s71_retains_three_definitions_and_three_paragraphs():
    source = (FIXTURES / "s71.txt").read_text()
    branches = sc.recognize_source_structure(source)
    assert [branch.label for branch in branches if branch.kind == "definition"] == [
        "qualifying benefit",
        "relevant payment date",
        "winter period",
    ]
    paragraphs = [branch for branch in branches if branch.path]
    assert [branch.path for branch in paragraphs] == [("a",), ("b",), ("c",)]
    assert paragraphs[-1].text.strip() == "veteran’s pension"
    clauses = tuple(sc._source_clause_spans(source, branches=branches))
    assert any(text.startswith("relevant payment date") for _, _, text in clauses)
    assert any(text.startswith("winter period") for _, _, text in clauses)
    conditions = sc._source_exception_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )
    context = next(branch for branch in conditions if "unless" in branch.text)
    assert "qualifying benefit" not in context.text
    assert all("\na\n" not in branch.text for branch in branches)
    assert "a a main benefit" not in sc._collapse_text(context.text)


def _analyze(source, payload, cases):
    return sc.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _source_rule(name, formula, excerpt, source_path=CITATION, *, dtype="Bool"):
    return {
        "name": name,
        "kind": "derived",
        "dtype": dtype,
        "source": source_path,
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": excerpt,
                        },
                    }
                ]
            }
        },
        "versions": [{"formula": formula}],
    }


def _qualifying_benefit_candidate(source):
    """An executable candidate for the verbatim qualifying-benefit definition."""

    inputs = [
        "context_otherwise_requires",
        "context_qualifying_benefit",
        "main_benefit_under_this_act",
        "receives_new_zealand_superannuation",
        "receives_veterans_pension",
    ]
    formula = (
        "if context_otherwise_requires: context_qualifying_benefit else: "
        "main_benefit_under_this_act or receives_new_zealand_superannuation "
        "or receives_veterans_pension"
    )
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [{"name": name, "dtype": "Bool"} for name in inputs],
        "rules": [
            _source_rule("qualifying_benefit", formula, source),
            _source_rule(
                "qualifying_main_benefit",
                "main_benefit_under_this_act",
                "a main benefit under this Act; or",
                f"{CITATION}(a)",
            ),
            _source_rule(
                "qualifying_superannuation",
                "receives_new_zealand_superannuation",
                "New Zealand superannuation; or",
                f"{CITATION}(b)",
            ),
            _source_rule(
                "qualifying_veterans_pension",
                "receives_veterans_pension",
                "veteran’s pension",
                f"{CITATION}(c)",
            ),
        ],
    }
    cases = []
    for overrides in (
        {},
        {"main_benefit_under_this_act": True},
        {"receives_new_zealand_superannuation": True},
        {"receives_veterans_pension": True},
        {"main_benefit_under_this_act": True, "context_otherwise_requires": True},
        {"context_qualifying_benefit": True, "context_otherwise_requires": True},
    ):
        values = dict.fromkeys(inputs, False) | overrides
        expected = (
            values["context_qualifying_benefit"]
            if values["context_otherwise_requires"]
            else any(values[name] for name in inputs[2:])
        )
        cases.append(
            {
                "period": "2026-07-01",
                "input": values,
                "output": {
                    "qualifying_benefit": expected,
                    "qualifying_main_benefit": values[inputs[2]],
                    "qualifying_superannuation": values[inputs[3]],
                    "qualifying_veterans_pension": values[inputs[4]],
                },
            }
        )
    return payload, cases


def _s71_candidate(source):
    """Represent every definition, with source-stated dates as local facts."""

    qualifying, remainder = source.split("relevant payment date", 1)
    relevant, winter = ("relevant payment date" + remainder).split("winter period", 1)
    winter = "winter period" + winter
    payload, cases = _qualifying_benefit_candidate(qualifying)
    payload["inputs"].extend(
        [
            {"name": "instalment_payment_date", "dtype": "Date"},
            {"name": "calendar_year_may_first", "dtype": "Date"},
            {"name": "context_relevant_payment_date", "dtype": "Date"},
            {"name": "context_winter_period_start_date", "dtype": "Date"},
            {"name": "context_winter_period_duration_weeks", "dtype": "Int32"},
        ]
    )
    payload["rules"].extend(
        [
            _source_rule(
                "relevant_payment_date",
                "if context_otherwise_requires: context_relevant_payment_date "
                "else: instalment_payment_date",
                relevant,
                dtype="Date",
            ),
            _source_rule(
                "winter_period_start_date",
                "if context_otherwise_requires: context_winter_period_start_date "
                "else: calendar_year_may_first",
                winter,
                dtype="Date",
            ),
            _source_rule(
                "winter_period_weeks",
                "if context_otherwise_requires: context_winter_period_duration_weeks "
                "else: winter_period_duration_weeks",
                winter,
                dtype="Int32",
            ),
        ]
    )
    for name, value, excerpt in (
        (
            "minimum_received_winter_days",
            1,
            "for 1 or more days during the winter period",
        ),
        ("winter_period_duration_weeks", 22, "the 22-week period starting on 1 May"),
        ("winter_period_start_day", 1, "starting on 1 May"),
    ):
        payload["rules"].append(
            {
                "name": name,
                "kind": "parameter",
                "dtype": "Int32",
                "source": CITATION,
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "kind": "formula",
                                "source": {
                                    "corpus_citation_path": CITATION,
                                    "excerpt": excerpt,
                                },
                            }
                        ]
                    }
                },
                "versions": [{"formula": str(value)}],
            }
        )
    for case in cases:
        case["input"].update(
            instalment_payment_date="2026-07-03",
            calendar_year_may_first="2026-05-01",
            context_relevant_payment_date="2026-07-04",
            context_winter_period_start_date="2026-06-01",
            context_winter_period_duration_weeks=20,
        )
        context = case["input"]["context_otherwise_requires"]
        case["output"].update(
            relevant_payment_date="2026-07-04" if context else "2026-07-03",
            winter_period_start_date="2026-06-01" if context else "2026-05-01",
            winter_period_weeks=20 if context else 22,
        )
    return payload, cases


def test_verbatim_s71_honest_context_candidate_passes_complete_source_gates():
    source = (FIXTURES / "s71.txt").read_text()
    payload, cases = _s71_candidate(source)
    result = _analyze(source, payload, cases)
    assert not result.issues


def test_verbatim_s71_dropped_context_exception_keeps_tests_obligation():
    source = (FIXTURES / "s71.txt").read_text()
    payload, cases = _s71_candidate(source)
    payload["rules"][0]["versions"][0]["formula"] = (
        "main_benefit_under_this_act or receives_new_zealand_superannuation "
        "or receives_veterans_pension"
    )
    for rule in payload["rules"]:
        formula = rule["versions"][0].get("formula", "")
        if formula.startswith("if context_otherwise_requires:"):
            rule["versions"][0]["formula"] = formula.split(" else: ", 1)[1]
    for case in cases:
        case["output"]["qualifying_benefit"] = any(
            case["input"][name]
            for name in (
                "main_benefit_under_this_act",
                "receives_new_zealand_superannuation",
                "receives_veterans_pension",
            )
        )
        case["output"].update(
            relevant_payment_date="2026-07-03",
            winter_period_start_date="2026-05-01",
            winter_period_weeks=22,
        )
    result = _analyze(source, payload, cases)
    assert any("[complete-source-unit:tests]" in issue for issue in result.issues)
    assert any("context otherwise requires" in issue for issue in result.issues)


def test_exact_bare_branch_accepts_existing_runtime_gap_wording():
    source = (
        "The secretary shall perform the following actions:—\n"
        "a\nThe secretary shall file an annual administration report;\n"
        "b\nThe secretary may require a copy of the federal return to be filed.\n"
    )
    branches = sc.recognize_source_structure(source)
    payload = {
        "module": {
            "deferred_outputs": [
                {
                    "output": "nz:statutes/act/public/2018/0032/section/71/b#return_copy",
                    "reason": (
                        f"{CITATION}(b) cannot be encoded until the runtime has a "
                        "document-filing capability for the required federal return copy."
                    ),
                }
            ]
        }
    }
    covered, issues = sc._deferred_coverage(
        payload, corpus_citation_path=CITATION, source_text=source, branches=branches
    )
    assert covered == {("b",)}
    assert not issues


@pytest.mark.parametrize(
    "reason",
    [
        "The runtime has no document-filing capability for the federal return copy.",
        f"{CITATION}(b) cannot be encoded until the runtime has more capabilities.",
        f"{CITATION}(a) cannot be encoded until the runtime has a "
        "document-filing capability for the required federal return copy.",
    ],
)
def test_bare_branch_keeps_imprecise_runtime_gap_deferrals_rejected(reason):
    source = (
        "The secretary shall perform the following actions:—\n"
        "a\nThe secretary shall file an annual administration report;\n"
        "b\nThe secretary may require a copy of the federal return to be filed.\n"
    )
    payload = {
        "module": {
            "deferred_outputs": [
                {
                    "output": "nz:statutes/act/public/2018/0032/section/71/b#return_copy",
                    "reason": reason,
                }
            ]
        }
    }
    covered, issues = sc._deferred_coverage(
        payload,
        corpus_citation_path=CITATION,
        source_text=source,
        branches=sc.recognize_source_structure(source),
    )
    assert not covered
    assert any("[complete-source-unit:deferral]" in issue for issue in issues)


def test_inserted_bare_subsection_uses_normalized_gate_branch_paths():
    source = "".join(
        f"{label}\nThe secretary shall perform the following actions:—\n"
        "a\nThe secretary shall file an annual administration report;\n"
        "b\nThe secretary may require a copy of the federal return to be filed.\n"
        for label in ("1", "2", "2A", "3")
    )
    branches = sc.recognize_source_structure(source)
    assert ("2a", "b") in {branch.path for branch in branches}
    payload = {
        "module": {
            "deferred_outputs": [
                {
                    "output": "nz:statutes/act/public/2018/0032/section/71/2A/b#return_copy",
                    "reason": (
                        f"{CITATION}(2A)(b) cannot be encoded until the runtime has a "
                        "document-filing capability for the required federal return copy."
                    ),
                }
            ]
        }
    }
    covered, issues = sc._deferred_coverage(
        payload, corpus_citation_path=CITATION, source_text=source, branches=branches
    )
    assert covered == {("2a", "b")}
    assert not issues


def test_labelled_definition_text_does_not_invent_source_unit_definition_branch():
    source = (
        "The following definitions apply:—\n"
        "a\nqualifying benefit means a main benefit under this Act;\n"
        "b\nwinter period means the 22-week period starting on 1 May.\n"
    )
    branches = sc.recognize_source_structure(source)
    assert [branch.path for branch in branches if branch.path] == [("a",), ("b",)]
    assert all(branch.kind != "definition" for branch in branches)


def test_bare_definition_excerpts_preserve_source_offsets():
    source = (
        "The following definitions apply:—\n"
        "  qualifying benefit means—\n"
        "a\na main benefit under this Act; or\n"
        "b\na pension.\n"
        "  winter period means the 22-week period.\n"
    )
    branches = sc.recognize_source_structure(source)
    definitions = [branch for branch in branches if branch.kind == "definition"]
    assert len(definitions) == 2
    masked = sc.mask_bare_labels(source)
    for definition in definitions:
        assert definition.text == masked[definition.start : definition.end]


def test_detached_bare_definition_retains_numeric_boundaries():
    source = (
        "qualifying benefit means—\n"
        "a\na main benefit under this Act; or\n"
        "b\na pension.\n"
        "minimum period means a period of at least 22 weeks.\n"
    )
    branches = sc.recognize_source_structure(source)
    boundaries = sc._source_boundary_obligations(
        branches,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
    )
    assert any(
        branch.kind == "definition"
        and branch.label == "minimum period"
        and value.value == 22
        for branch, value in boundaries
    )


def test_nested_bare_definition_stays_inside_its_subsection():
    source = (
        "1\nThe following definitions apply:—\n"
        "qualifying period means a period of at least 22 weeks, comprising:—\n"
        "a\nbenefit weeks; or\n"
        "b\npension weeks.\n"
        "2\nAnother subsection applies.\n"
    )
    branches = sc.recognize_source_structure(source)
    definition = next(branch for branch in branches if branch.kind == "definition")
    sibling = next(branch for branch in branches if branch.path == ("2",))
    assert definition.path == ("1",)
    assert definition.end == sibling.start
    assert "Another subsection" not in definition.text
    boundaries = sc._source_boundary_obligations(
        branches,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
    )
    assert sum(value.value == 22 for _branch, value in boundaries) == 1


def test_numeric_outline_definition_after_children_belongs_to_subsection():
    source = (
        "1\nThe following definitions apply:—\n"
        "qualifying benefit means—\n"
        "a\na main benefit under this Act; or\n"
        "b\na pension.\n"
        "minimum period means a period of at least 22 weeks.\n"
    )
    branches = sc.recognize_source_structure(source)
    paragraph = next(branch for branch in branches if branch.path == ("1", "b"))
    definition = next(branch for branch in branches if branch.label == "minimum period")
    assert "minimum period" not in paragraph.text
    assert definition.path == ("1",)
    assert definition.start == paragraph.end
    boundaries = sc._source_boundary_obligations(
        branches,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
    )
    assert [(branch.path, value.value) for branch, value in boundaries] == [
        (("1",), 22)
    ]


def test_equal_thresholds_in_distinct_bare_definitions_keep_both_obligations():
    source = (
        "qualifying benefit means—\n"
        "a\na main benefit under this Act; or\n"
        "b\na pension.\n"
        "minimum period means a period of at least 22 weeks.\n"
        "maximum period means a period of at least 22 weeks.\n"
    )
    branches = sc.recognize_source_structure(source)
    boundaries = sc._source_boundary_obligations(
        branches,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
    )
    assert [(branch.label, value.value) for branch, value in boundaries] == [
        ("minimum period", 22),
        ("maximum period", 22),
    ]


@pytest.mark.parametrize(
    "term", ["ACC", "COVID-19 benefit", "person's income", "person’s income"]
)
def test_detached_definition_spelling_cannot_hide_numeric_recall_in_deferral(term):
    source = (
        "benefit means—\n"
        "a\na main benefit; or\n"
        "b\nthe federal return copy that must be filed.\n"
        f"{term} means a payment of 22 dollars.\n"
    )
    payload = {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {"corpus_citation_path": CITATION},
            "deferred_outputs": [
                {
                    "output": "nz:statutes/act/public/2018/0032/section/71/b#return_copy",
                    "reason": (
                        f"{CITATION}(b) cannot be encoded until the runtime has a "
                        "document-filing capability for the required federal return copy."
                    ),
                }
            ],
        },
        "rules": [],
    }
    branches = sc.recognize_source_structure(source)
    paragraph = next(branch for branch in branches if branch.path == ("b",))
    definition = next(branch for branch in branches if branch.label == term)
    assert term not in paragraph.text
    assert definition.path == ()
    assert definition.start == paragraph.end
    covered, issues = sc._deferred_coverage(
        payload, corpus_citation_path=CITATION, source_text=source, branches=branches
    )
    assert covered == {("b",)}
    assert not issues
    result = _analyze(source, payload, [])
    assert any("numeric value 22" in issue for issue in result.issues)


def test_s71_calendar_definition_deferral_keeps_existing_gate_rejection():
    source = (FIXTURES / "s71.txt").read_text()
    payload = {
        "module": {
            "deferred_outputs": [
                {
                    "output": "nz:statutes/act/public/2018/0032/section/71#winter_period",
                    "reason": (
                        f"{CITATION} cannot be encoded until the runtime has a "
                        "date-arithmetic capability for the winter period "
                        "and relevant payment date."
                    ),
                }
            ]
        }
    }
    covered, issues = sc._deferred_coverage(
        payload,
        corpus_citation_path=CITATION,
        source_text=source,
        branches=sc.recognize_source_structure(source),
    )
    assert not covered
    assert any("[complete-source-unit:deferral]" in issue for issue in issues)


@pytest.mark.parametrize("path", [(), ("a",), ("b",), ("c",)])
def test_s71_calendar_runtime_gap_is_not_authorized_by_structure(path):
    source = (FIXTURES / "s71.txt").read_text()
    suffix = "".join(f"({part})" for part in path)
    assert not sc._reason_names_source_bound_runtime_gap(
        f"{CITATION}{suffix} cannot be encoded until the runtime has a "
        "date-arithmetic capability for the winter period and relevant payment date.",
        source,
        path=path,
        corpus_citation_path=CITATION,
    )
