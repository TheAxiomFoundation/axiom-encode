"""Age coverage must follow executed dates and a controlling birthday relation."""

import functools
from copy import deepcopy
from datetime import date

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

CITATION = "de/guidance/bzst-dakg-2025/a-19-2/document-1"
CONDITION = "soweit das Kind das 25. Lebensjahr vollendet hat"
SOURCE = "(1) Beginn der Behinderung, " + CONDITION + "."
BIRTHDAY = (
    "if date_add_years(date_add_years(child_birth_date, age_limit), -age_limit) != child_birth_date: "
    "date_add_days(date_add_years(child_birth_date, age_limit), 1) "
    "else: date_add_years(child_birth_date, age_limit)"
)


def _fixture():
    def rule(name, dtype, formula, kind="derived"):
        return {
            "name": name,
            "kind": kind,
            "dtype": dtype,
            "entity": "Child",
            "period": "Day",
            "source": CITATION + "(1)",
            "versions": [{"effective_from": "2025-01-01", "formula": formula}],
        }

    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {"name": "child_birth_date", "dtype": "Date"},
            {"name": "onset_recorded", "dtype": "Boolean"},
            {"name": "other", "dtype": "Boolean"},
        ],
        "rules": [
            rule("age_limit", "Count", "25", "parameter"),
            rule("birthday", "Date", BIRTHDAY),
            rule(
                "content_sufficient",
                "Judgment",
                "period_start < birthday or onset_recorded",
            ),
        ],
    }
    period = {
        "period_kind": "custom",
        "name": "day",
        "start": "2025-03-01",
        "end": "2025-03-01",
    }
    cases = [
        {
            "period": deepcopy(period),
            "input": {
                "child_birth_date": birth,
                "onset_recorded": False,
                "other": False,
            },
            "output": {"content_sufficient": sufficient, "birthday": birthday},
        }
        for birth, birthday, sufficient in [
            ("2000-03-02", "2025-03-02", True),
            ("2000-03-01", "2025-03-01", False),
        ]
    ]
    return payload, cases


def _analyze(payload, cases, source=SOURCE):
    return sc.analyze_complete_source_unit(
        yaml.safe_dump(payload, sort_keys=False),
        source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="de-DE"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _missing_age_test(result):
    return any("[complete-source-unit:tests]" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "formula",
    [
        "period_start < birthday or onset_recorded",
        "birthday > period_start or onset_recorded",
        "not (period_start >= birthday) or onset_recorded",
        "not (birthday <= period_start) or onset_recorded",
    ],
)
def test_executed_birthday_relation_covers_completed_age_condition(formula):
    payload, cases = _fixture()
    payload["rules"][-1]["versions"][0]["formula"] = formula
    assert not _missing_age_test(_analyze(payload, cases))


@pytest.mark.parametrize(
    "mutation",
    [
        "no_pair",
        "same_birth_date",
        "two_inputs",
        "different_period_end",
        "missing_period",
        "wrong_assertion",
        "wrong_birthday_assertion",
        "different_output_keys",
        "wrong_path",
        "string_input",
        "unrelated_date",
        "age_as_input",
        "wrong_age",
        "wrong_calendar_function",
        "uncorrected_leap_day",
        "constant_output",
        "unreachable_age_relation",
        "unrelated_varying_dependency",
        "wrong_inclusivity",
        "extra_birthday_use",
    ],
)
def test_calendar_witness_rejects_invalid_or_unrelated_pairs(mutation):
    payload, cases = _fixture()
    source = SOURCE
    birthday = payload["rules"][1]["versions"][0]
    result = payload["rules"][-1]["versions"][0]
    if mutation == "no_pair":
        cases = cases[:1]
    elif mutation == "same_birth_date":
        cases[1]["input"]["child_birth_date"] = cases[0]["input"]["child_birth_date"]
    elif mutation == "two_inputs":
        cases[1]["input"]["other"] = True
    elif mutation == "different_period_end":
        cases[1]["period"]["end"] = "2025-03-02"
    elif mutation == "missing_period":
        for case in cases:
            del case["period"]
            case["input"]["period_start"] = date(2025, 3, 1)
    elif mutation == "wrong_assertion":
        cases[1]["output"]["content_sufficient"] = True
    elif mutation == "wrong_birthday_assertion":
        cases[1]["output"]["birthday"] = "2025-03-02"
    elif mutation == "different_output_keys":
        del cases[1]["output"]["birthday"]
    elif mutation == "wrong_path":
        payload["rules"][-1]["source"] = CITATION + "(2)"
    elif mutation == "string_input":
        payload["inputs"][0]["dtype"] = "String"
    elif mutation == "unrelated_date":
        payload["inputs"][0]["name"] = "document_issue_date"
        birthday["formula"] = BIRTHDAY.replace(
            "child_birth_date", "document_issue_date"
        )
        for case in cases:
            case["input"]["document_issue_date"] = case["input"].pop("child_birth_date")
    elif mutation == "age_as_input":
        payload["rules"].pop(0)
        payload["inputs"].append({"name": "age_limit", "dtype": "Count"})
        for case in cases:
            case["input"]["age_limit"] = 25
    elif mutation == "wrong_age":
        source = SOURCE.replace("25.", "26.")
    elif mutation == "wrong_calendar_function":
        birthday["formula"] = BIRTHDAY.replace("date_add_years", "date_add_months")
    elif mutation == "uncorrected_leap_day":
        birthday["formula"] = "date_add_years(child_birth_date, age_limit)"
    elif mutation == "constant_output":
        result["formula"] = "(period_start < birthday or onset_recorded) or true"
        for case in cases:
            case["output"]["content_sufficient"] = True
    elif mutation == "unreachable_age_relation":
        result["formula"] = (
            "(false and period_start < birthday) or child_birth_date > other_date"
        )
        payload["inputs"].append({"name": "other_date", "dtype": "Date"})
        for case in cases:
            case["input"]["other_date"] = "2000-03-01"
    elif mutation == "unrelated_varying_dependency":
        extra = deepcopy(payload["rules"][1])
        extra["name"] = "other_date"
        payload["rules"].append(extra)
        result["formula"] = (
            "(false and period_start < birthday) or period_start < other_date"
        )
    elif mutation == "wrong_inclusivity":
        result["formula"] = "period_start <= birthday or onset_recorded"
        cases[1]["input"]["child_birth_date"] = "2000-02-28"
        cases[1]["output"]["birthday"] = "2025-02-28"
    elif mutation == "extra_birthday_use":
        result["formula"] = (
            "(false and period_start < birthday) or period_start < birthday"
        )
    assert _missing_age_test(_analyze(payload, cases, source))


def test_leap_day_attainment_is_executed_on_first_march():
    payload, cases = _fixture()
    cases[0]["input"]["child_birth_date"] = "2000-03-02"
    cases[1]["input"]["child_birth_date"] = "2000-02-29"
    cases[1]["output"]["birthday"] = "2025-03-01"
    assert not _missing_age_test(_analyze(payload, cases))


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace("vollendet hat", "nicht vollendet hat"),
        SOURCE.replace("Kind", "Antragsteller"),
        SOURCE.replace("vollendet hat", "vollendet hat und der Antrag gestellt wurde"),
    ],
)
def test_calendar_condition_does_not_drop_other_source_requirements(source):
    payload, cases = _fixture()
    assert _missing_age_test(_analyze(payload, cases, source))


def test_pathological_age_text_is_not_a_calendar_condition():
    witness = sc._ExceptionWitness(
        "result",
        "child_birth_date",
        True,
        True,
        True,
        False,
        None,
        calendar_attainment_age=25,
    )
    text = "soweit das Kind das " + "9" * 5000 + ". Lebensjahr vollendet hat"
    assert not sc._calendar_age_witness_matches_source(text, witness)


@pytest.mark.parametrize(
    "name,accepted",
    [
        ("child_birth_date", True),
        ("child_date_of_birth", True),
        ("subject_child_recorded_birth_date", True),
        ("current_child_registered_birth_date", True),
        ("kind_geburtsdatum", True),
        ("parent_birth_date", False),
        ("applicant_date_of_birth", False),
        ("spouse_geburtsdatum", False),
        ("unrelated_birth_date", False),
        ("not_child_birth_date", False),
        ("other_child_birth_date", False),
        ("parent_child_birth_date", False),
        ("child_not_birth_date", False),
    ],
)
def test_calendar_witness_requires_positive_child_subject(name, accepted):
    payload, cases = _fixture()
    payload["inputs"][0]["name"] = name
    payload["rules"][1]["versions"][0]["formula"] = BIRTHDAY.replace(
        "child_birth_date", name
    )
    for case in cases:
        case["input"][name] = case["input"].pop("child_birth_date")
    assert _missing_age_test(_analyze(payload, cases)) is not accepted


@pytest.mark.parametrize("pretyped", [False, True])
@pytest.mark.parametrize(
    "mutation",
    ["string", "undeclared", "duplicate", "conflicting", "rule_collision", "valid"],
)
def test_calendar_witness_requires_unambiguous_declared_date_input(pretyped, mutation):
    payload, cases = _fixture()
    if pretyped:
        for case in cases:
            case["input"]["child_birth_date"] = date.fromisoformat(
                case["input"]["child_birth_date"]
            )
    if mutation == "string":
        payload["inputs"][0]["dtype"] = "String"
    elif mutation == "undeclared":
        payload["inputs"].pop(0)
    elif mutation == "duplicate":
        payload["inputs"].append(deepcopy(payload["inputs"][0]))
    elif mutation == "conflicting":
        payload["inputs"].append({"name": "child_birth_date", "dtype": "String"})
    elif mutation == "rule_collision":
        payload["rules"].append({"name": "child_birth_date", "dtype": "Date"})
    assert _missing_age_test(_analyze(payload, cases)) is (mutation != "valid")
