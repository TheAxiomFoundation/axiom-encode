import copy
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.student_loan_schedule_binding import (
    bind_student_loan_schedule,
)
from axiom_encode.student_loan_schedule_source import recognize_student_loan_schedule

ROOT = Path(__file__).parent / "fixtures/student_loan_schedule"
SOURCE = (ROOT / "source.txt").read_text()
CITATION = "ca/policy/revenu-quebec/tp1-2025/schedule-m-student-loan-interest"
DATA = yaml.safe_load((ROOT / "diagnostic.yaml").read_text())


def bind(data):
    source = recognize_student_loan_schedule(SOURCE, CITATION)
    return bind_student_loan_schedule(source, yaml.safe_dump(data, sort_keys=False))


def test_complete_graph_binds_without_claiming_execution():
    bound = bind(DATA)
    assert bound is not None
    assert len(bound.input_roles) == 9
    assert len(bound.rule_roles) == 7
    assert dict(bound.rule_roles)["guide"].endswith("consultation_required")


@pytest.mark.parametrize("index", range(7))
def test_unknown_executable_rule_key_refuses(index):
    data = copy.deepcopy(DATA)
    data["rules"][index]["guard"] = "false"
    assert bind(data) is None


@pytest.mark.parametrize("index", range(9))
def test_input_metadata_cannot_create_authority(index):
    data = copy.deepcopy(DATA)
    data["inputs"][index]["metadata"] = {"proof": {"atoms": []}}
    assert bind(data) is None


@pytest.mark.parametrize(
    "kind",
    [
        "extra_rule",
        "extra_input",
        "duplicate",
        "imports",
        "units",
        "wrong_entity",
        "open_year",
        "proof_false",
        "proof_integer",
        "foreign_source",
        "nonexistent_proof",
        "wrong_proof_kind",
        "wrong_date_path",
        "free_year",
        "input_alias",
    ],
)
def test_closed_partition_and_proof_controls(kind):
    d = copy.deepcopy(DATA)
    if kind == "extra_rule":
        r = copy.deepcopy(d["rules"][0])
        r["name"] += "_extra"
        d["rules"].append(r)
    elif kind == "extra_input":
        i = copy.deepcopy(d["inputs"][0])
        i["name"] += "_extra"
        d["inputs"].append(i)
    elif kind == "duplicate":
        d["rules"][1]["name"] = d["rules"][0]["name"]
    elif kind == "imports":
        d["imports"] = []
    elif kind == "units":
        d["units"][0]["minor_units"] = 3
    elif kind == "wrong_entity":
        d["inputs"][0]["entity"] = "Family"
    elif kind == "open_year":
        del d["rules"][0]["versions"][0]["effective_to"]
    elif kind in {"proof_false", "proof_integer"}:
        d["module"]["proof_validation"]["required"] = (
            False if kind == "proof_false" else 1
        )
    elif kind == "foreign_source":
        d["rules"][0]["metadata"]["proof"]["atoms"][0]["source"][
            "corpus_citation_path"
        ] += "/foreign"
    elif kind == "nonexistent_proof":
        d["rules"][0]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
            "This sentence does not exist."
        )
    elif kind == "wrong_proof_kind":
        d["rules"][4]["metadata"]["proof"]["atoms"][0]["kind"] = "unit"
    elif kind == "wrong_date_path":
        d["rules"][6]["metadata"]["proof"]["atoms"][-1]["path"] = (
            "versions[1].effective_to"
        )
    elif kind == "free_year":
        d["rules"][6]["metadata"]["proof"]["atoms"][-1]["source"]["excerpt"] = "2025"
    elif kind == "input_alias":
        d["inputs"][0]["name"] = "opaque_fact"
    assert bind(d) is None


@pytest.mark.parametrize(
    "index,before,after",
    [
        (
            0,
            "if schedule_m_carryforward_calculated_for_2024:",
            "if not schedule_m_carryforward_calculated_for_2024:",
        ),
        (
            0,
            "schedule_m_student_loan_interest_paid_1998_through_2024 - schedule_m_student_loan_interest_used_for_previous_year_credits",
            "schedule_m_student_loan_interest_used_for_previous_year_credits - schedule_m_student_loan_interest_paid_1998_through_2024",
        ),
        (1, " + ", " - "),
        (2, "max(0,", "max(1,"),
        (2, "min(", "max("),
        (3, " - ", " + "),
        (4, " and ", " or "),
        (4, " > 0", " >= 0"),
        (
            4,
            "schedule_m_decided_to_carry_2024_student_loan_interest_to_future_year",
            "is_completing_student_loan_interest_schedule",
        ),
        (
            5,
            "is_completing_student_loan_interest_schedule",
            "wants_to_keep_student_loan_interest_schedule_copy",
        ),
        (
            6,
            "wants_to_keep_student_loan_interest_schedule_copy",
            "is_completing_student_loan_interest_schedule",
        ),
    ],
)
def test_ordered_source_graph_not_just_names(index, before, after):
    d = copy.deepcopy(DATA)
    formula = d["rules"][index]["versions"][0]["formula"]
    assert before in formula
    d["rules"][index]["versions"][0]["formula"] = formula.replace(before, after)
    assert bind(d) is None


def test_copied_description_does_not_rescue_wrong_role():
    d = copy.deepcopy(DATA)
    d["inputs"][7]["name"] = "is_not_completing_student_loan_interest_schedule"
    d["rules"][5]["versions"][0]["formula"] = d["inputs"][7]["name"]
    assert bind(d) is None


def test_comment_else_cannot_create_executable_branch():
    d = copy.deepcopy(DATA)
    d["rules"][0]["versions"][0]["formula"] = d["rules"][0]["versions"][0][
        "formula"
    ].replace(" else:", " # else:")
    assert bind(d) is None


@pytest.mark.parametrize("value", [[], {}, 1, None])
@pytest.mark.parametrize("field", ["kind", "path"])
def test_malformed_atom_scalar_refuses_without_exception(field, value):
    d = copy.deepcopy(DATA)
    d["rules"][0]["metadata"]["proof"]["atoms"][0][field] = value
    assert bind(d) is None


def test_currency_contract_is_type_exact():
    d = copy.deepcopy(DATA)
    d["units"][0]["minor_units"] = 2.0
    assert bind(d) is None


@pytest.mark.parametrize(
    "index,description",
    [
        (4, "Historical interest paid1998 through2024 excluding2025"),
        (6, "Explicit claimant form-use context"),
        (4, ""),
    ],
)
def test_contradictory_or_empty_prose_does_not_override_role(index, description):
    d = copy.deepcopy(DATA)
    d["inputs"][index]["description"] = description
    assert bind(d) is None


@pytest.mark.parametrize(
    "excerpt",
    [
        "2025",
        "Interest paid in 2025 on a student loan",
        "Amount carried to line 385 of your 2025 return",
    ],
)
def test_supplemental_year_atom_cannot_replace_endpoint_anchor(excerpt):
    d = copy.deepcopy(DATA)
    for atom in d["rules"][6]["metadata"]["proof"]["atoms"]:
        if atom["path"].startswith("versions[0].effective_"):
            atom["source"]["excerpt"] = excerpt
    assert bind(d) is None


@pytest.mark.parametrize(
    "index,text",
    [
        (6, "Not a prior carry decision"),
        (8, "No wish to keep a copy"),
        (7, "Not form-use"),
        (6, "Explicit prior carry decision; use only if eligible."),
    ],
)
def test_unknown_or_negated_defining_description_refuses(index, text):
    d = copy.deepcopy(DATA)
    d["inputs"][index]["description"] = text
    assert bind(d) is None


def test_unwrapped_source_requires_physical_dated_header_anchor():
    import hashlib

    source = SOURCE.split("\n\n", 1)[1]
    record = recognize_student_loan_schedule(source, CITATION)
    d = copy.deepcopy(DATA)
    d["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    for rule in d["rules"]:
        for atom in rule["metadata"]["proof"]["atoms"]:
            if atom["source"]["excerpt"].startswith("Revenu Quebec Schedule M 2025 -"):
                atom["source"]["excerpt"] = "T P -1. D. M -V (2025-12)"
    assert (
        bind_student_loan_schedule(record, yaml.safe_dump(d, sort_keys=False))
        is not None
    )
