"""Source-conformance forms must not infer arithmetic or missing factual inputs."""

import ast
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import worksheet_operation_evidence as operations


def expression(text):
    return ast.parse(text, mode="eval").body


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("(x if p else 0) if p else y", "x if p else y", True),
        ("x if p else (0 if p else y)", "x if p else y", True),
        ("(x if count > 1 else 0) if count > 1 else y", "x if count > 1 else y", True),
        ("(x if count > 2 else 0) if count > 1 else y", "x if count > 1 else y", False),
        ("(x if p else 0) if p else y", "y if p else x", False),
        ("(x+1 if p else 0) if p else y", "x if p else y", False),
        ("x if p else 0", "x", False),
        ("(x if q else 0) if p else y", "x if p else y", False),
        ("x if p and q else y", "x if q and p else y", False),
        ("x if p or q else y", "x if p else x if q else y", False),
        ("(x if a/b>1 else 0) if a/b>1 else y", "x if a/b>1 else y", False),
        ("x if a == b else y", "x if b == a else y", True),
    ],
)
def test_only_known_enclosing_predicates_are_removed(left, right, expected):
    a, b = expression(left), expression(right)
    before = (ast.dump(a), ast.dump(b))
    assert operations._same_expression(a, b) is expected
    assert (ast.dump(a), ast.dump(b)) == before


def principal():
    # Independent test spelling of the native form, using the source ratio
    # descriptor only for the already separately tested net-income computation.
    ratio = ast.unparse(operations._expected_operations()[operations._LINE4])
    return (
        "0 if province_or_territory_of_residence_at_year_end == 'Quebec' or "
        "(province_or_territory_of_residence_at_year_end == 'Manitoba' "
        "and minimum_tax_is_payable) else "
        "0 if t2209_federal_non_business_foreign_tax_credit_line_3 == "
        "t2209_non_business_income_tax_paid_to_foreign_country_line_1 else "
        f"min(t2209_non_business_income_tax_paid_to_foreign_country_line_1 - "
        f"{operations._LINE2}, {ratio}, {operations._TAX})"
    )


def test_fixed_alternative_is_only_for_source_principal_output():
    actual = expression(principal())
    expected = operations._expected_operations()[operations._LINE5]
    assert not operations._same_expression(actual, expected)
    assert operations._matches_source_operation(operations._LINE5, actual, expected)
    assert not operations._matches_source_operation(operations._LINE4, actual, expected)


@pytest.mark.parametrize(
    "old,new",
    [
        ("and minimum_tax_is_payable", "or minimum_tax_is_payable"),
        ("and minimum_tax_is_payable", "and not minimum_tax_is_payable"),
        ("'Quebec'", "'Ontario'"),
        ("'Manitoba'", "'Alberta'"),
        ("and minimum_tax_is_payable", "and True"),
        ("0 if province", "1 if province"),
        ("min(", "max("),
        (f", {operations._TAX})", ", 999999)"),
        (f"- {operations._LINE2}", f"+ {operations._LINE2}"),
        ("and minimum_tax_is_payable", "and minimum_tax_is_payable and unknown_fact"),
    ],
)
def test_alternative_rejects_changed_source_gates_and_arithmetic(old, new):
    assert old in principal()
    actual = expression(principal().replace(old, new))
    assert not operations._matches_source_operation(
        operations._LINE5, actual, operations._expected_operations()[operations._LINE5]
    )


def test_source_descriptor_acceptance_does_not_supply_missing_inputs():
    # Preserve the two actual lazy orders, including their different failures.
    native = compile(ast.Expression(expression(principal())), "<native>", "eval")
    expected = compile(
        ast.Expression(operations._expected_operations()[operations._LINE5]),
        "<source>",
        "eval",
    )
    facts = {
        "province_or_territory_of_residence_at_year_end": "Ontario",
        "t2209_federal_non_business_foreign_tax_credit_line_3": 20,
        "t2209_non_business_income_tax_paid_to_foreign_country_line_1": 20,
    }
    assert eval(native, {"__builtins__": {}}, facts) == 0
    with pytest.raises(NameError, match="minimum_tax_is_payable"):
        eval(expected, {"__builtins__": {}}, facts)
    facts["province_or_territory_of_residence_at_year_end"] = "Manitoba"
    with pytest.raises(NameError, match="minimum_tax_is_payable"):
        eval(native, {"__builtins__": {}}, facts)


def test_closed_operation_gate_still_rejects_invalid_dependency_before_normalizing():
    # The existing resolver must retain an invalid helper even if an enclosing
    # condition would make one of its arms unreachable in the compared form.
    fixture = Path(__file__).parent / "fixtures/worksheet_operations"
    payload = yaml.safe_load((fixture / "candidate.yaml").read_text())
    helper = next(
        r
        for r in payload["rules"]
        if r["name"] == "net_income_for_multiple_jurisdictions"
    )
    helper["unit"] = "USD"
    source = (fixture.parent / "source_path_shadow/source.txt").read_text()
    resolver = operations._OperationResolver(
        payload,
        operations._LINE4,
        "2025-01-01",
        "2025-12-31",
        source,
        operations._PRIMARY,
    )
    assert resolver.expand(ast.Name(id=operations._LINE4)) is None
    assert resolver.errors


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "wrong_gate",
        "wrong_arm",
        "wrong_sum",
        "missing_term",
        "donor_helper",
        "wrong_version",
    ],
)
def test_inline_proof_requires_direct_sum_in_exact_selected_arm(change):
    formula = (
        "if jurisdictions_tax_paid_count > 1:\n  net_income_for_multiple_jurisdictions\n"
        "else:\n  if resident_of_canada_for_part_of_year:\n"
        "    income_for_part_of_year_resident_in_canada + income_and_losses_under_paragraphs_115_1_a_to_c_for_nonresident_part\n"
        "  else:\n    t2209_net_income_line_2"
    )
    if change == "wrong_gate":
        formula = formula.replace("resident_of_canada_for_part_of_year", "unknown_fact")
    elif change == "wrong_arm":
        formula = formula.replace(
            "if resident_of_canada_for_part_of_year:",
            "if not resident_of_canada_for_part_of_year:",
        )
    elif change == "wrong_sum":
        formula = formula.replace(" + ", " - ")
    elif change == "missing_term":
        formula = formula.replace(
            " + income_and_losses_under_paragraphs_115_1_a_to_c_for_nonresident_part",
            "",
        )
    elif change == "donor_helper":
        formula = formula.replace(
            "income_for_part_of_year_resident_in_canada + income_and_losses_under_paragraphs_115_1_a_to_c_for_nonresident_part",
            "unrelated_helper",
        )
    rule = {"versions": [{"formula": formula}, {"formula": "unrelated_helper"}]}
    assert operations._owns_inline_part_year_sum(
        rule, 1 if change == "wrong_version" else 0
    ) is (change == "none")


@pytest.mark.parametrize("proof_location", ["owner", "missing", "decoy"])
def test_inline_source_certificate_requires_actual_owner_proof(proof_location):
    import copy

    from tests.test_worksheet_operation_evidence import certify, inputs, named

    payload, cases = inputs()
    helper_name = "net_income_for_part_year_resident"
    helper = named(payload, helper_name)
    root = named(payload, "t2036_net_income_for_credit")
    direct_sum = (
        "income_for_part_of_year_resident_in_canada + "
        "income_and_losses_under_paragraphs_115_1_a_to_c_for_nonresident_part"
    )
    assert helper_name in root["versions"][0]["formula"]
    root["versions"][0]["formula"] = root["versions"][0]["formula"].replace(
        helper_name, direct_sum
    )
    # The native inline form spells the already source-proved threshold directly.
    assert (
        named(payload, "multiple_foreign_country_count_threshold")["versions"][0][
            "formula"
        ]
        == 1
    )
    root["versions"][0]["formula"] = root["versions"][0]["formula"].replace(
        "multiple_foreign_country_count_threshold", "1"
    )
    atoms = [
        copy.deepcopy(a)
        for a in helper["metadata"]["proof"]["atoms"]
        if a["path"] == "versions[0].formula"
    ]
    part_year_source = operations._SOURCE_OPERATIONS["net_income"][1][1]
    root["metadata"]["proof"]["atoms"] = [
        atom
        for atom in root["metadata"]["proof"]["atoms"]
        if not (
            atom["path"] == "versions[0].formula"
            and part_year_source in atom["source"]["excerpt"]
        )
    ]
    payload["rules"] = [r for r in payload["rules"] if r["name"] != helper_name]
    # This synthetic fixture models an inline calculation without a public helper.
    # Principal calculation inputs and expected values remain unchanged.
    for case in cases:
        for key in list(case.get("output", {})):
            if key.rsplit("#", 1)[-1] == helper_name:
                del case["output"][key]
    if proof_location == "owner":
        root["metadata"]["proof"]["atoms"].extend(atoms)
    elif proof_location == "decoy":
        decoy = copy.deepcopy(helper)
        decoy["name"] = "unrelated_part_year_proof_donor"
        payload["rules"].append(decoy)
    result = certify(payload, cases)
    if proof_location == "owner":
        assert not result.unresolved
        assert result.outputs == operations._OUTPUTS
    else:
        assert result.unresolved
        assert not result.outputs


def test_selected_helper_missing_proof_cannot_borrow_parent_proof():
    from tests.test_worksheet_operation_evidence import certify, inputs, named

    payload, cases = inputs()
    helper = named(payload, "net_income_for_part_year_resident")
    parent = named(payload, "t2036_net_income_for_credit")
    excerpt = operations._SOURCE_OPERATIONS["net_income"][1][1]
    assert any(
        excerpt in atom["source"]["excerpt"]
        for atom in parent["metadata"]["proof"]["atoms"]
    )
    helper["metadata"]["proof"]["atoms"] = [
        atom
        for atom in helper["metadata"]["proof"]["atoms"]
        if atom["path"] != "versions[0].formula"
    ]
    result = certify(payload, cases)
    assert result.unresolved
    assert not result.outputs
