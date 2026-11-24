"""Static source/AST binding, not execution, factual truth or native admission."""

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from axiom_encode.bound_retirement_transfer_form import bind_retirement_transfer_form

FIXTURE = Path(__file__).parent / "fixtures/retirement_transfer"
CITATION = "ca/policy/revenu-quebec/tp1-2025/schedule-q-retirement-income-transfer"


def data():
    return yaml.safe_load((FIXTURE / "policy.yaml").read_text())


def source():
    return (FIXTURE / "source.txt").read_text()


def bind(payload=None, text=None):
    payload = data() if payload is None else payload
    text = source() if text is None else text
    return bind_retirement_transfer_form(
        yaml.safe_dump(payload, sort_keys=False), text, CITATION
    )


def rule(payload, name):
    return next(r for r in payload["rules"] if r["name"] == name)


def rebind_source(payload, old, new):
    payload["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        new.encode()
    ).hexdigest()
    for r in payload["rules"]:
        for a in r["metadata"]["proof"]["atoms"]:
            a["source"]["excerpt"] = a["source"]["excerpt"].replace(old, new)


def test_exact_original_complete21_rules16_facts62_proofs():
    p = (FIXTURE / "policy.yaml").read_text()
    b = bind_retirement_transfer_form(p, source(), CITATION)
    assert b is not None and b.bound(p, source(), CITATION)
    assert len(b.rule_roles) == 21 and len(b.input_names) == 16
    assert b.relation_arguments == ("Person", "Person")
    assert b.source.birth_cutoff.isoformat() == "1961-01-01"
    assert b.source.spouse_reference_date.isoformat() == "2025-12-31"
    assert sum(len(r["metadata"]["proof"]["atoms"]) for r in data()["rules"]) == 62


def test_order_and_optional_descriptions_do_not_assert_input_truth():
    p = data()
    p["rules"].reverse()
    p["inputs"].reverse()
    for item in p["inputs"]:
        item.pop("description")
    assert bind(p) is not None


@pytest.mark.parametrize("index", range(16))
@pytest.mark.parametrize(
    "defect", ["opposite_description", "copied_role", "wrong_entity", "default"]
)
def test_every_fact_declared_role_and_type_is_closed(index, defect):
    p = data()
    item = p["inputs"][index]
    if defect == "opposite_description":
        item["description"] = "Not " + item["description"]
    elif defect == "copied_role":
        item["description"] = p["inputs"][(index + 1) % 16]["description"]
    elif defect == "wrong_entity":
        item["entity"] = "Family"
    else:
        item["default"] = 0
    assert bind(p) is None


@pytest.mark.parametrize("index", range(21))
def test_each_declaration_requires_its_own_exact_source_proof(index):
    p = data()
    p["rules"][index]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = "2025"
    assert bind(p) is None


@pytest.mark.parametrize("value", [None, False, 1, "true", [], {"required": True}])
def test_strict_proof_requirement_has_boolean_type(value):
    p = data()
    p["module"]["proof_validation"]["required"] = value
    assert bind(p) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", []),
        ("path", {}),
        ("kind", "unsupported"),
        ("path", "versions[9].formula"),
    ],
)
def test_malformed_atom_fields_refuse_without_exception(field, value):
    p = data()
    p["rules"][0]["metadata"]["proof"]["atoms"][0][field] = value
    assert bind(p) is None


@pytest.mark.parametrize(
    "defect",
    [
        "extra_rule",
        "extra_input",
        "duplicate_rule",
        "duplicate_input",
        "extra_unit",
        "minor_float",
        "minor_bool",
        "imports",
        "rule_extra",
        "module_extra",
    ],
)
def test_full_declaration_partition(defect):
    p = data()
    if defect == "extra_rule":
        x = deepcopy(p["rules"][1])
        x["name"] = "unowned_amount"
        p["rules"].append(x)
    elif defect == "extra_input":
        x = deepcopy(p["inputs"][0])
        x["name"] = "unowned_fact"
        p["inputs"].append(x)
    elif defect == "duplicate_rule":
        p["rules"][1] = deepcopy(p["rules"][0])
    elif defect == "duplicate_input":
        p["inputs"][1] = deepcopy(p["inputs"][0])
    elif defect == "extra_unit":
        p["units"].append({"name": "USD", "kind": "currency", "minor_units": 2})
    elif defect == "minor_float":
        p["units"][0]["minor_units"] = 2.0
    elif defect == "minor_bool":
        p["units"][0]["minor_units"] = True
    elif defect == "imports":
        p["imports"] = ["ca:foreign"]
    elif defect == "rule_extra":
        p["rules"][1]["requires"] = "foreign"
    else:
        p["module"]["unknown_execution_mode"] = True
    assert bind(p) is None


FORMULA_MUTANTS = [
    ("schedule_q_retirement_income_deductions_line_18", " + ", " - "),
    (
        "schedule_q_eligible_retirement_income_line_20",
        "schedule_q_retirement_income_line_122_line_10 - schedule_q_retirement_income_deductions_line_18",
        "schedule_q_retirement_income_deductions_line_18 - schedule_q_retirement_income_line_122_line_10",
    ),
    (
        "normalized_requested_retirement_income_transfer",
        "min(max(0, schedule_q_requested_retirement_income_transfer_line_22),",
        "min(schedule_q_requested_retirement_income_transfer_line_22,",
    ),
    ("retirement_income_transfer_permission_applies", " and ", " or "),
    ("retirement_income_transfer_permission_applies", " < ", " <= "),
    (
        "retirement_income_transfer_permission_applies",
        "[schedule_q_source_year]",
        "[period_start]",
    ),
    (
        "schedule_q_retirement_income_transferred_line_245",
        "if retirement_income_transfer_permission_applies:",
        "if True:",
    ),
    (
        "entered_quebec_withholding_line50",
        "if compensation_included_in_line20:",
        "if not compensation_included_in_line20:",
    ),
    (
        "schedule_q_quebec_withholding_transferred_line_451_1",
        "if schedule_q_retirement_income_transferred_line_245 > 0:",
        "if retirement_income_transfer_permission_applies:",
    ),
    (
        "schedule_q_quebec_withholding_transferred_line_451_1",
        " * schedule_q_retirement_income_transferred_line_245 / schedule_q_eligible_retirement_income_line_20",
        " * schedule_q_eligible_retirement_income_line_20 / schedule_q_retirement_income_transferred_line_245",
    ),
    (
        "retirement_income_received_line123",
        "sum(retirement_income_transferred_to.schedule_q_retirement_income_transferred_line_245)",
        "schedule_q_requested_retirement_income_transfer_line_22",
    ),
    (
        "quebec_withholding_received_line451_3",
        "schedule_q_quebec_withholding_transferred_line_451_1",
        "entered_quebec_withholding_line50",
    ),
    ("corresponding_withholding_transfer_required", " > 0", " >= 0"),
    ("retirement_transfer_schedule_enclosure_required", " and ", " or "),
    (
        "spouse_identification_reporting_required",
        "is_completing_retirement_transfer_schedule",
        "has_selected_spouse_at_reference_date",
    ),
    (
        "compensation_eligible_income_guide_consultation_required",
        "received_retirement_compensation_amounts",
        "compensation_included_in_line20",
    ),
    (
        "retirement_transfer_copy_guide_consultation_required",
        "wants_retirement_transfer_schedule_copy",
        "is_completing_retirement_transfer_schedule",
    ),
    (
        "recipient_withholding_reporting_required",
        "retirement_income_received_line123",
        "quebec_withholding_received_line451_3",
    ),
]


@pytest.mark.parametrize("name,old,new", FORMULA_MUTANTS)
def test_ordered_source_ast_and_distinct_principal_owners(name, old, new):
    p = data()
    v = rule(p, name)["versions"][0]
    assert old in v["formula"]
    v["formula"] = v["formula"].replace(old, new)
    assert bind(p) is None


@pytest.mark.parametrize(
    "defect",
    [
        "string_key",
        "float_key",
        "bool_key",
        "wrong_date",
        "date_bool",
        "date_numeric",
        "extra_index",
        "query_year",
        "integer_float",
        "integer_bool",
        "rate_bool",
        "outside_version",
        "relation_extra",
        "relation_kind",
    ],
)
def test_fixed_date_parameter_index_types_and_relation_contract(defect):
    p = data()
    v = rule(p, "retirement_transfer_birth_cutoff")["versions"][0]
    if defect == "string_key":
        v["values"] = {"2025": "1961-01-01"}
    elif defect == "float_key":
        v["values"] = {2025.0: "1961-01-01"}
    elif defect == "bool_key":
        v["values"] = {True: "1961-01-01"}
    elif defect == "wrong_date":
        v["values"][2025] = "1961-01-02"
    elif defect == "date_bool":
        v["values"][2025] = True
    elif defect == "date_numeric":
        v["values"][2025] = 1961.0
    elif defect == "extra_index":
        v["values"][2026] = "1961-01-01"
    elif defect == "query_year":
        rule(p, "retirement_transfer_birth_cutoff")["indexed_by"] = "tax_year"
    elif defect == "integer_float":
        rule(p, "schedule_q_source_year")["versions"][0]["formula"] = 2025.0
    elif defect == "integer_bool":
        rule(p, "schedule_q_source_year")["versions"][0]["formula"] = True
    elif defect == "rate_bool":
        p["rules"][0]["versions"][0]["formula"] = True
    elif defect == "outside_version":
        v["effective_to"] = "2026-12-31"
    elif defect == "relation_extra":
        rule(p, "retirement_income_transferred_to")["data_relation"]["cardinality"] = 1
    else:
        rule(p, "retirement_income_transferred_to")["data_relation"]["arguments"] = [
            "Family",
            "Person",
        ]
    assert bind(p) is None


def test_duplicate_keys_and_policy_aliases_refuse():
    p = (FIXTURE / "policy.yaml").read_text()
    assert (
        bind_retirement_transfer_form(p + "\ninputs: []\n", source(), CITATION) is None
    )
    assert (
        bind_retirement_transfer_form(
            p.replace(
                "required: true", "required: &enabled true\n    copied: *enabled"
            ),
            source(),
            CITATION,
        )
        is None
    )


@pytest.mark.parametrize(
    "mutation", ["negation", "qualification", "duplicate", "header", "rate"]
)
def test_coherent_source_hash_changes_do_not_create_unproved_owners(mutation):
    p = data()
    s = source()
    if mutation == "negation":
        t = s.replace("If you were born", "Do not transfer. If you were born")
    elif mutation == "qualification":
        t = s.replace(
            "Enclose this schedule with your return.",
            "Enclose this schedule with your return unless exempt.",
        )
    elif mutation == "duplicate":
        t = s + s
    elif mutation == "header":
        t = s.replace("T P -1. D.Q-V (2025-12)", "T P -1. D.X-V (2025-12)")
    else:
        t = s.replace("maximum: 50%", "maximum: 60%")
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        t.encode()
    ).hexdigest()
    assert bind(p, t) is None


def test_physical_header_anchor_without_resolver_wrapper():
    p = data()
    s = source()
    a = s.index("T P -1.")
    old = s[:a].rstrip("\r\n")
    t = s[a:]
    header = t[: t.index("\n\nIf you were born")]
    for r in p["rules"]:
        for atom in r["metadata"]["proof"]["atoms"]:
            if atom["source"]["excerpt"] == old:
                atom["source"]["excerpt"] = header
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        t.encode()
    ).hexdigest()
    assert bind(p, t) is not None


def test_full_crlf_source_and_proof_offsets_remain_raw():
    p = data()
    t = source().replace("\n", "\r\n")
    for r in p["rules"]:
        for a in r["metadata"]["proof"]["atoms"]:
            a["source"]["excerpt"] = a["source"]["excerpt"].replace("\n", "\r\n")
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        t.encode()
    ).hexdigest()
    assert bind(p, t) is not None


@pytest.mark.parametrize(
    "defect",
    [
        "hash",
        "citation",
        "extra_atom",
        "omitted_atom",
        "origin_type",
        "wrong_source_role",
    ],
)
def test_exact_source_hash_and_complete_proof_inventory(defect):
    p = data()
    atom = p["rules"][0]["metadata"]["proof"]["atoms"][0]
    if defect == "hash":
        p["module"]["source_verification"]["source_sha256"] = "0" * 64
    elif defect == "citation":
        atom["source"]["corpus_citation_path"] = "ca/policy/foreign"
    elif defect == "extra_atom":
        p["rules"][0]["metadata"]["proof"]["atoms"].append(deepcopy(atom))
    elif defect == "omitted_atom":
        p["rules"][0]["metadata"]["proof"]["atoms"].pop()
    elif defect == "origin_type":
        atom["source"] = []
    else:
        atom["source"]["excerpt"] = "Enclose this schedule with your return."
    assert bind(p) is None


def test_coherently_reordered_addends_do_not_reassign_printed_roles():
    p = data()
    v = rule(p, "schedule_q_retirement_income_deductions_line_18")["versions"][0]
    operands = v["formula"].split(" + ")
    operands[0], operands[1] = operands[1], operands[0]
    v["formula"] = " + ".join(operands)
    assert bind(p) is None


@pytest.mark.parametrize(
    "replacement",
    [
        "schedule_q_requested_retirement_income_transfer_line_22",
        "normalized_requested_retirement_income_transfer",
    ],
)
def test_withholding_guard_and_numerator_require_actual_authorized_transfer(
    replacement,
):
    p = data()
    v = rule(p, "schedule_q_quebec_withholding_transferred_line_451_1")["versions"][0]
    v["formula"] = v["formula"].replace(
        "schedule_q_retirement_income_transferred_line_245", replacement
    )
    assert bind(p) is None


@pytest.mark.parametrize(
    "formula",
    [
        "sum(retirement_income_transferred_to.retirement_income_received_line123)",
        "sum(other_relation.schedule_q_retirement_income_transferred_line_245)",
        "sum(retirement_income_transferred_to.schedule_q_quebec_withholding_transferred_line_451_1)",
    ],
)
def test_recipient_provider_self_reference_foreign_relation_and_cross_role_refuse(
    formula,
):
    p = data()
    rule(p, "retirement_income_received_line123")["versions"][0]["formula"] = formula
    assert bind(p) is None
