"""A `min(amount, cap)` cap is witnessed by a case at the cap.

Arizona DES FAA5 medical block 8 (local run f61921aa) says "Up to $40 per day
for lodging is allowed when the participant must travel over 50 miles". The
candidate gates on `travel_distance_miles > threshold` and caps the cost with
`min(max(0, daily_lodging_cost), medical_lodging_daily_cap)`. Its companion
tests set `daily_lodging_cost` to 30, 40 and 60 under qualifying conditions,
but the boundary check still refused it with `missing: source unit=40`: the
comparison gate made the capped cost drop out of the rule's selector names, so
no case could reach the cap reader.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    _numeric_profile_for_citation_path,
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

FIXTURE = Path(__file__).parent / "fixtures" / "az_batch3_medical_block_8"
PROVENANCE = json.loads((FIXTURE / "provenance.json").read_text())
CITATION = PROVENANCE["citation"]
PREFIX = "us-az:policies/des/faa5/na-medical-expenses-and-deduction/block-8"
OUTPUT = f"{PREFIX}#allowable_daily_medical_lodging_cost"
COST = f"{PREFIX}#input.daily_lodging_cost"
CAP_CASE = "qualifying_lodging_exactly_at_daily_cap"
BOUNDARY_FINDING = "missing: source unit=40"


def _read(name: str, key: str) -> str:
    text = (FIXTURE / name).read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == PROVENANCE[key]["sha256"]
    return text


CANDIDATE = _read("candidate.yaml", "candidate")
CASES = yaml.safe_load(_read("candidate.test.yaml", "companion"))
SOURCE = _read("source.txt", "source")


def _issues(candidate: str = CANDIDATE, cases: list | None = None) -> list[str]:
    profile = _numeric_profile_for_citation_path(CITATION)
    bindings = sc.collect_artifact_numeric_bindings(
        candidate,
        extract_named_scalars=extract_named_scalar_occurrences,
    )
    analysis = sc.analyze_complete_source_unit(
        candidate,
        SOURCE,
        corpus_citation_path=CITATION,
        test_cases=CASES if cases is None else cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
        artifact_numeric_values=tuple(value for _name, value in bindings),
        artifact_numeric_bindings=bindings,
    )
    return list(analysis.issues)


def _boundary_refused(issues: list[str]) -> bool:
    return any(BOUNDARY_FINDING in issue for issue in issues)


def _cap_case() -> dict:
    return next(case for case in CASES if case["name"] == CAP_CASE)


def _without_cap_case() -> list[dict]:
    return [case for case in CASES if case["name"] != CAP_CASE]


def test_fixture_is_the_recorded_refusal():
    assert "$40 per day" in SOURCE
    assert BOUNDARY_FINDING in PROVENANCE["recorded_issue"]
    assert _cap_case()["input"][COST] == 40
    assert _cap_case()["output"][OUTPUT] == 40


def test_case_at_the_cap_witnesses_the_source_boundary():
    issues = _issues()
    assert not _boundary_refused(issues)
    assert issues == []


@pytest.mark.parametrize(
    "cap_formula",
    [
        "min(medical_lodging_daily_cap, max(0, daily_lodging_cost))",
        "min(daily_lodging_cost, medical_lodging_daily_cap)",
    ],
)
def test_operand_order_and_clamp_do_not_matter(cap_formula):
    candidate = CANDIDATE.replace(
        "min(max(0, daily_lodging_cost), medical_lodging_daily_cap)", cap_formula
    )
    assert candidate != CANDIDATE
    assert not _boundary_refused(_issues(candidate))


def test_cases_below_and_above_the_cap_alone_are_still_refused():
    assert _boundary_refused(_issues(cases=_without_cap_case()))


def test_a_blocked_case_at_the_cap_is_still_refused():
    blocked = json.loads(json.dumps(_cap_case()))
    blocked["input"][f"{PREFIX}#input.participant_has_lodging_cost"] = False
    blocked["output"][OUTPUT] = 0
    assert _boundary_refused(_issues(cases=[*_without_cap_case(), blocked]))


def test_a_cap_on_another_value_is_still_refused():
    candidate = CANDIDATE.replace(
        "min(max(0, daily_lodging_cost), medical_lodging_daily_cap)",
        "min(max(0, daily_lodging_cost), medical_lodging_daily_cap + 5)",
    )
    assert candidate != CANDIDATE
    assert _boundary_refused(_issues(candidate))


def test_minimum_operand_names_skip_called_functions():
    rule = {
        "versions": [
            {
                "formula": (
                    "if miles > threshold:\n"
                    "  min(max(0, cost), cap)\n"
                    "else: min(a, b, c)"
                )
            }
        ]
    }
    assert sc._rule_minimum_operand_names(rule) == {"cost", "cap"}
