from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_path_shadow as shadow

FIXTURE = Path(__file__).parent / "fixtures" / "source_path_shadow"
SOURCE = (FIXTURE / "source.txt").read_text()
L2 = "provincial_territorial_2025_t2036_federal_credit_line_2"
TP = "provincial_or_territorial_tax_otherwise_payable"
TEXT = "province_or_territory_of_residence_at_year_end"
TAX = "total_non_business_income_taxes_paid_to_all_foreign_countries"
SMALL = "ontario_minimum_tax_single_country_special_credit"
X = "t2209_non_business_income_tax_paid_to_foreign_country_line_1"
Y = "t691_foreign_taxes_line_8_part_4"
Z = "t691_special_foreign_tax_credit_line_11_part_4"
CITATION = "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit"


@pytest.fixture
def candidate():
    return yaml.safe_load((FIXTURE / "candidate.yaml").read_text())


def rule(payload, name):
    return next(x for x in payload["rules"] if x["name"] == name)


def pred(name, op, value, kind):
    return shadow.Predicate(name, "Person", "Year", op, value, kind)


def obligation(source=SOURCE):
    frame = shadow.closed_amt_frame(source)
    assert frame is not None
    fields = dict(frame.fields)
    # Explicit test contract, not an asserted automatic candidate/source match.
    # The independent operation descriptor uses source operands and does not
    # read any candidate formula or helper while constructing its expectation.
    assert (fields["base"], fields["tax_line"], fields["credit_line"]) == (
        "1",
        "8",
        "11",
    )
    return shadow.SourceObligation(
        "ontario_small",
        L2,
        hashlib.sha256(source.encode()).hexdigest(),
        CITATION,
        (frame.owner,),
        (
            pred("minimum_tax_is_payable", "==", True, "boolean"),
            pred(TEXT, "==", "Ontario", "text"),
            pred(TAX, "<=", Decimal(fields["threshold"]), "numeric"),
        ),
        f"({X} / ({X} + {Y})) * {Z}",
        "Person",
        "2025-01-01",
        "2025-12-31",
        input_contracts=tuple(
            shadow.InputContract(name, dtype, "Person", "Year", unit, (frame.owner,))
            for name, dtype, unit in (
                ("minimum_tax_is_payable", "Boolean", None),
                (TEXT, "Text", None),
                (TAX, "Money", "CAD"),
                (X, "Money", "CAD"),
                (Y, "Money", "CAD"),
                (Z, "Money", "CAD"),
            )
        ),
    )


def inspect(payload, source=SOURCE):
    return shadow.inspect_source_paths(
        payload, source_text=source, owner=L2, obligations=[obligation(source)]
    )


def compatible(result):
    return [r for r in result.rows if r["status"] != "excluded"]


def test_real_source_frames_keep_identification_and_closed_note():
    frames = shadow.closed_result_frames(SOURCE)
    assert len(frames) == 4
    on = frames[1]
    assert on.body.valid(SOURCE)
    assert "The result from line 81" in on.body.text
    assert "The amount from line 58" in on.body.text
    assert dict(on.fields)["zero_a"] == "70"
    assert frames[2].kind == "applicable_form_alternative"
    amt = shadow.closed_amt_frame(SOURCE)
    assert amt and amt.owner.valid(SOURCE)
    assert dict(amt.fields)["threshold"] == "200"
    assert "another" in amt.body.text


@pytest.mark.parametrize(
    "old,new",
    [
        (
            "The result from line 81 is your provincial or territorial tax otherwise payable.",
            "The result from line 81 is a different amount.",
        ),
        (
            "The amount from line 58 is your provincial or territorial tax otherwise payable.",
            "The amount from line 58 may be used.",
        ),
        ("However, if", "Separately, if"),
        (
            "calculate this amount by entering",
            "calculate an unrelated amount by entering",
        ),
        (
            "If you were a resident of Alberta,",
            "If you were a resident of Alberta and disabled,",
        ),
        (
            "See the privacy notice",
            "Only eligible claimants may claim. See the privacy notice",
        ),
    ],
)
def test_source_result_frames_fail_closed(old, new):
    assert not shadow.closed_result_frames(SOURCE.replace(old, new))


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace(
            "resident of another province", "resident of another eligible province"
        ),
        lambda s: s.replace("more than $200, you must", "more than $201, you must"),
        lambda s: s.replace(
            "(cid:129) If you were a resident of Ontario",
            "(2) If you were a resident of Ontario",
        ),
        lambda s: s.replace("on\nline 2.\n(2)", "on\nline 3.\n(2)"),
        lambda s: '"' + s + '"',
    ],
)
def test_amt_complete_sibling_contract_is_not_adjacent_if_inference(change):
    changed = change(SOURCE)
    assert changed != SOURCE
    assert shadow.closed_amt_frame(changed) is None


def test_actual_candidate_paths_and_unscoped_parameters(candidate):
    result = inspect(candidate)
    assert len(result.paths) == 5
    assert not result.unresolved
    assert len(compatible(result)) == 1
    row = compatible(result)[0]
    assert row["predicates_entailed"] and row["independent_expression_matches"]
    assert row["acceptance"] is False
    assert "unresolved" in row["operation_proof"]
    dep = next(
        d
        for p in result.paths
        for d in p.dependencies
        if d.name == "t2036_foreign_tax_threshold"
    )
    assert dep.entity == dep.period == ""
    assert not result.acceptance_activated


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_dispatch",
        "dead_helper",
        "threshold201",
        "missing_amt",
        "wrong_arithmetic",
        "wrong_slot",
        "wrong_entity",
        "wrong_period",
        "cycle",
        "unknown_import",
    ],
)
def test_real_candidate_mutations_do_not_match_positive_contract(candidate, mutation):
    owner = rule(candidate, L2)["versions"][0]
    if mutation == "wrong_dispatch":
        owner["formula"] = owner["formula"].replace(
            SMALL, "t2209_federal_non_business_foreign_tax_credit_line_3"
        )
    elif mutation == "dead_helper":
        owner["formula"] = "t2209_federal_non_business_foreign_tax_credit_line_3"
    elif mutation == "threshold201":
        rule(candidate, "t2036_foreign_tax_threshold")["versions"][0]["formula"] = 201
    elif mutation == "missing_amt":
        owner["formula"] = owner["formula"].replace(
            "not minimum_tax_is_payable", "minimum_tax_is_payable"
        )
    elif mutation == "wrong_arithmetic":
        r = rule(candidate, SMALL)["versions"][0]
        r["formula"] = r["formula"].replace("*", "+")
    elif mutation == "wrong_slot":
        r = rule(candidate, "resident_of_ontario_at_year_end")["versions"][0]
        r["formula"] = r["formula"].replace(TEXT, "province_of_birth")
    elif mutation in {"wrong_entity", "wrong_period"}:
        rule(candidate, SMALL)["entity" if mutation == "wrong_entity" else "period"] = (
            "Household" if mutation == "wrong_entity" else "Month"
        )
    elif mutation == "cycle":
        rule(candidate, SMALL)["versions"][0]["formula"] = SMALL
    elif mutation == "unknown_import":
        rule(candidate, SMALL)["versions"][0]["formula"] = "unavailable_import"
    result = inspect(candidate)
    assert (
        not compatible(result)
        or any(r["status"] == "unresolved" for r in compatible(result))
        or result.unresolved
    )


def test_selected_helper_supersession_cannot_inherit_owner_year(candidate):
    helper = rule(candidate, SMALL)
    late = copy.deepcopy(helper["versions"][0])
    late["effective_from"] = "2025-07-01"
    late["formula"] = late["formula"].replace("*", "+")
    helper["versions"].append(late)
    result = inspect(candidate)
    assert {p.start for p in result.paths} == {"2025-01-01", "2025-07-01"}
    rows = compatible(result)
    assert len(rows) == 2
    assert rows[0]["independent_expression_matches"]
    assert not rows[1]["independent_expression_matches"]
    assert any("selected proof" in e for p in result.paths for e in p.unresolved)


def test_selected_constant_supersession_changes_real_paths(candidate):
    parameter = rule(candidate, "t2036_foreign_tax_threshold")
    late = copy.deepcopy(parameter["versions"][0])
    late.update(effective_from="2025-07-01", formula=201)
    parameter["versions"].append(late)
    result = inspect(candidate)
    values = {
        p.value
        for path in result.paths
        if path.start == "2025-07-01"
        for p in path.predicates
        if p.name == TAX
    }
    assert values == {Decimal(201)}
    assert any(row["status"] == "unresolved" for row in compatible(result))


def test_no_source_or_no_paths_cannot_be_positive(candidate):
    o = obligation()
    bad = shadow.SourceObligation(**{**o.__dict__, "source_sha256": "0" * 64})
    result = shadow.inspect_source_paths(
        candidate, source_text=SOURCE, owner=L2, obligations=[bad]
    )
    assert result.unresolved and not result.paths
    assert not result.acceptance_activated


def test_duplicate_declarations_rejected(candidate):
    candidate["inputs"].append(copy.deepcopy(candidate["inputs"][0]))
    assert inspect(candidate).unresolved == (
        "invalid/duplicate candidate declarations",
    )


@pytest.mark.parametrize(
    "field,value",
    [("dtype", "Text"), ("unit", "USD"), ("entity", "Household"), ("period", "Month")],
)
def test_independent_monetary_input_contract_rejects_actual_candidate_mutation(
    candidate, field, value
):
    next(x for x in candidate["inputs"] if x["name"] == Z)[field] = value
    result = inspect(candidate)
    assert result.unresolved
    assert all(row["status"] == "unresolved" for row in result.rows)


def test_source_owner_entity_cannot_be_borrowed(candidate):
    o = replace(obligation(), entity="Household")
    result = shadow.inspect_source_paths(
        candidate, source_text=SOURCE, owner=L2, obligations=[o]
    )
    assert "source owner entity/period mismatch" in result.unresolved
    assert all(row["status"] == "unresolved" for row in result.rows)


def test_actual_missing_amt_guard_not_just_inverted_polarity(candidate):
    r = rule(candidate, L2)["versions"][0]
    # The correct helper is live, but its owning AMT guard is actually absent.
    r["formula"] = (
        f"if resident_of_ontario_at_year_end: {SMALL} else: t2209_federal_non_business_foreign_tax_credit_line_3"
    )
    result = inspect(candidate)
    assert any(
        row["independent_expression_matches"] and not row["predicates_entailed"]
        for row in compatible(result)
    )
    assert all(row["status"] == "unresolved" for row in compatible(result))


@pytest.mark.parametrize(
    "name", ["t2036_foreign_tax_threshold", "resident_of_british_columbia_at_year_end"]
)
@pytest.mark.parametrize(
    "fault", ["missing_formula_proof", "wrong_citation", "missing_date_proof"]
)
def test_unproved_selected_dependency_cannot_exclude_an_apparent_contradiction(
    candidate, name, fault
):
    r = rule(candidate, name)
    atoms = r["metadata"]["proof"]["atoms"]
    if fault == "wrong_citation":
        for atom in atoms:
            atom["source"]["corpus_citation_path"] = "ca/foreign/source"
    elif fault == "missing_formula_proof":
        atoms[:] = [a for a in atoms if not a["path"].endswith("formula")]
    else:
        atoms[:] = [a for a in atoms if not a["path"].endswith("effective_to")]
    result = inspect(candidate)
    affected = {
        i
        for i, path in enumerate(result.paths)
        if any(d.name == name for d in path.dependencies)
    }
    assert affected
    assert all(
        row["status"] == "unresolved" for row in result.rows if row["path"] in affected
    )


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_helper_name",
        "wrong_text_identity",
        "negative_name",
        "wrong_formula_proof",
        "bare_date_atom",
    ],
)
def test_structural_derived_equality_does_not_borrow_source_residence_identity(
    candidate, fault
):
    name = "resident_of_ontario_at_year_end"
    helper = rule(candidate, name)
    if fault in {"wrong_helper_name", "negative_name"}:
        replacement = (
            "opaque_eligibility" if fault == "wrong_helper_name" else "non_" + name
        )
        helper["name"] = replacement
        owner = rule(candidate, L2)["versions"][0]
        owner["formula"] = owner["formula"].replace(name, replacement)
    elif fault == "wrong_text_identity":
        declaration = next(x for x in candidate["inputs"] if x["name"] == TEXT)
        declaration["name"] = "province_of_birth"
        helper["versions"][0]["formula"] = helper["versions"][0]["formula"].replace(
            TEXT, "province_of_birth"
        )
    else:
        for atom in helper["metadata"]["proof"]["atoms"]:
            if fault == "wrong_formula_proof" and atom["path"].endswith("formula"):
                atom["source"]["excerpt"] = "See the privacy notice on your return."
            if fault == "bare_date_atom" and atom["path"].endswith(
                ("effective_from", "effective_to")
            ):
                atom["source"]["excerpt"] = "for 2025"
    result = inspect(candidate)
    assert any(
        "unproved derived selector" in e for p in result.paths for e in p.unresolved
    )
    assert any(row["status"] == "unresolved" for row in result.rows)


@pytest.mark.parametrize(
    "name,field,value",
    [
        (L2, "unit", "USD"),
        (L2, "dtype", "Text"),
        (SMALL, "unit", "USD"),
        (SMALL, "dtype", "Text"),
        ("t2036_foreign_tax_threshold", "unit", "USD"),
        ("t2036_foreign_tax_threshold", "dtype", "Text"),
        ("ontario_minimum_tax_single_country_ratio", "dtype", "Text"),
    ],
)
def test_typed_closure_checks_owner_helpers_and_comparison_constants(
    candidate, name, field, value
):
    rule(candidate, name)[field] = value
    result = inspect(candidate)
    assert not any(row["status"] == "diagnostic_match" for row in compatible(result))
    assert result.unresolved or any(path.unresolved for path in result.paths)


def test_direct_numeric_literal_threshold_retains_supported_context_coercion(candidate):
    owner = rule(candidate, L2)["versions"][0]
    owner["formula"] = owner["formula"].replace("t2036_foreign_tax_threshold", "200")
    result = inspect(candidate)
    assert len(compatible(result)) == 1
    assert compatible(result)[0]["status"] == "diagnostic_match"


def test_scalar_parameter_cannot_claim_money_by_candidate_unit_mutation(candidate):
    parameter = rule(candidate, "t691_non_business_foreign_tax_rate")
    parameter["unit"] = "USD"
    result = inspect(candidate)
    affected = [
        p
        for p in result.paths
        if any(d.name == parameter["name"] for d in p.dependencies)
    ]
    assert affected and all(p.unresolved for p in affected)


def bridge(source=SOURCE):
    return shadow.worksheet_shadow_contract(
        source, citation=CITATION, line2_owner=L2, tax_otherwise_owner=TP
    )


def both_owners(candidate, source=SOURCE):
    contract = bridge(source)
    return {
        owner: shadow.inspect_source_paths(
            candidate, source_text=source, owner=owner, obligations=contract.obligations
        )
        for owner in (L2, TP)
    }


def test_source_fields_independently_construct_all_twelve_operation_descriptors(
    candidate,
):
    contract = bridge()
    assert not contract.unresolved and len(contract.obligations) == 12
    reports = both_owners(candidate)
    assert [len(r.paths) for r in reports.values()] == [5, 7]
    assert sum(len(r.rows) for r in reports.values()) == 74
    for result in reports.values():
        assert not result.unresolved
        assert all(row["status"] == "diagnostic_match" for row in compatible(result))
        assert all(
            row["operation_proof"].startswith("unresolved")
            for row in compatible(result)
        )
        assert not result.acceptance_activated
    ordinary = next(o for o in contract.obligations if o.identity == "ontario_single")
    multiple = next(o for o in contract.obligations if o.identity == "ontario_multi")
    assert any("result from line 81" in s.text for s in ordinary.spans)
    assert any("amount from line 58" in s.text for s in multiple.spans)


@pytest.mark.parametrize(
    "mutation",
    [
        "unrelated_choice",
        "swapped_choice",
        "omitted_branch",
        "wrong_line",
        "extra_eligibility",
        "wrong_amount_type",
        "wrong_amount_unit",
    ],
)
def test_actual_alberta_candidate_mutations_do_not_match_source_operation_set(
    candidate, mutation
):
    owner = rule(candidate, TP)["versions"][0]
    if mutation == "unrelated_choice":
        owner["formula"] = owner["formula"].replace(
            "alberta_applicable_form_is_428mj", "minimum_tax_is_payable"
        )
    elif mutation == "swapped_choice":
        owner["formula"] = owner["formula"].replace(
            "if alberta_applicable_form_is_428mj:",
            "if not alberta_applicable_form_is_428mj:",
        )
    elif mutation == "omitted_branch":
        owner["formula"] = owner["formula"].replace(
            "ab428_line_63 + ab428_line_68", "ab428mj_line_20 + ab428mj_line_36"
        )
    elif mutation == "wrong_line":
        owner["formula"] = owner["formula"].replace("ab428mj_line_36", "ab428_line_68")
    elif mutation == "extra_eligibility":
        owner["formula"] = owner["formula"].replace(
            "ab428mj_line_20 + ab428mj_line_36",
            "if minimum_tax_is_payable: ab428mj_line_20 + ab428mj_line_36 else: 0",
        )
    else:
        declaration = next(
            x for x in candidate["inputs"] if x["name"] == "ab428mj_line_36"
        )
        declaration["dtype" if mutation == "wrong_amount_type" else "unit"] = (
            "Text" if mutation == "wrong_amount_type" else "USD"
        )
    result = both_owners(candidate)[TP]
    relevant = [r for r in compatible(result) if r["obligation"].startswith("alberta_")]
    assert result.unresolved or any(r["status"] == "unresolved" for r in relevant)


def test_source_changed_line_is_not_justified_by_candidate_name(candidate):
    source = SOURCE.replace("The amount from line 58 is", "The amount from line 59 is")
    candidate["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    contract = bridge(source)
    multiple = next(o for o in contract.obligations if o.identity == "ontario_multi")
    assert "line_59" in multiple.expected_expression
    result = both_owners(candidate, source)[TP]
    assert any(
        r["status"] == "unresolved"
        for r in compatible(result)
        if r["obligation"] == "ontario_multi"
    )


def test_each_obligation_citation_must_match_not_first_only(candidate):
    obligations = list(bridge().obligations)
    i = next(i for i, o in enumerate(obligations) if o.identity == "ontario_large")
    obligations[i] = replace(obligations[i], citation="ca/foreign/source")
    result = shadow.inspect_source_paths(
        candidate, source_text=SOURCE, owner=L2, obligations=obligations
    )
    assert result.unresolved == (
        "candidate/obligation citation or source identity mismatch",
    )
    assert not result.paths


def test_actual_existing_cases_are_linked_by_selected_path_and_corroborated(candidate):
    cases = yaml.safe_load((FIXTURE / "cases.yaml").read_text())
    reports = both_owners(candidate)
    links = [
        x
        for result in reports.values()
        for x in shadow.corroborated_case_links(candidate, result, cases)
    ]
    assert len(links) == 82
    # The actual analyzer normalizes declared numeric expected strings before
    # invoking the strict dependency matcher. Preserve the original fixture.
    assert all(x["corroborated"] for x in links)
    original = next(c for c in cases if c["name"] == "ontario_threshold_pair_a")
    ratio = "ontario_minimum_tax_single_country_ratio"
    assert (
        next(v for k, v in original["output"].items() if k.endswith("#" + ratio))
        == "0.5"
    )
    for owner, result in reports.items():
        linked_paths = {
            x["path"] for x in shadow.corroborated_case_links(candidate, result, cases)
        }
        assert linked_paths == set(range(len(result.paths)))


def test_wrong_principal_assertion_is_not_a_corroborated_case_link(candidate):
    cases = yaml.safe_load((FIXTURE / "cases.yaml").read_text())
    original = cases[0]
    key = next(k for k in original["output"] if k.endswith("#" + L2))
    original["output"][key] += 1
    links = shadow.corroborated_case_links(
        candidate, both_owners(candidate)[L2], cases[:1]
    )
    assert links and not links[0]["corroborated"]


def test_missing_reached_helper_assertion_cannot_be_replaced_by_formula_only(candidate):
    cases = yaml.safe_load((FIXTURE / "cases.yaml").read_text())
    selected = next(
        c for c in cases if any(k.endswith("#" + SMALL) for k in c["output"])
    )
    for key in tuple(selected["output"]):
        if key.endswith("#" + SMALL):
            del selected["output"][key]
    links = shadow.corroborated_case_links(
        candidate, both_owners(candidate)[L2], [selected]
    )
    assert not links or not all(x["corroborated"] for x in links)


def test_cap_is_direct_min_operation_while_transfer_is_not_value_change(candidate):
    principal = "provincial_territorial_foreign_tax_credit_line_5"
    result = shadow.inspect_cap_dependency(
        candidate, source_text=SOURCE, principal=principal, bound_owner=TP
    )
    assert result["source_roles"] == ("numeric_upper_bound", "output_transfer")
    assert (
        sum(x.get("direct_min_bound_operand", False) for x in result["observations"])
        == 1
    )
    formula = rule(candidate, principal)["versions"][0]
    original = formula["formula"]
    formula["formula"] = original.replace(
        ",\n        provincial_or_territorial_tax_otherwise_payable", ""
    )
    assert formula["formula"] != original
    changed = shadow.inspect_cap_dependency(
        candidate, source_text=SOURCE, principal=principal, bound_owner=TP
    )
    assert not any(
        x.get("direct_min_bound_operand", False) for x in changed["observations"]
    )
    assert not changed["acceptance"]


@pytest.mark.parametrize(
    "mutation",
    [
        "container",
        "on_part",
        "ab_part",
        "footnote",
        "quoted_line1",
        "quoted_line2",
        "embedded_line1",
        "embedded_line2",
    ],
)
def test_source_coordinate_and_ordinary_alias_identity_cannot_borrow_existing_contract(
    candidate, mutation
):
    source = SOURCE
    line1 = "Enter the amount from line 1 of Form T2209. 1"
    line2 = "Enter the amount from line 3 of Form T2209, unless you have to pay minimum tax.(1) – 2"
    if mutation == "container":
        source = source.replace("T2203", "T9999")
    elif mutation == "on_part":
        source = source.replace(
            "Part 4 of Section ON428MJ", "Part 5 of Section ON428MJ"
        )
    elif mutation == "ab_part":
        source = source.replace(
            "in Part 4 of\nSection AB428MJ", "in Part 5 of\nSection AB428MJ"
        )
    elif mutation == "footnote":
        source = source.replace(line2, line2.replace("tax.(1)", "tax.(2)"))
    else:
        line = line1 if mutation.endswith("line1") else line2
        replacement = (
            '"' + line + '"' if mutation.startswith("quoted") else "Example: " + line
        )
        source = source.replace(line, replacement)
    assert source != SOURCE
    # Even matching mutated proof bytes/source metadata cannot rescue these
    # aliases: the source-contract builder itself rejects wrong ownership.
    candidate["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    contract = bridge(source)
    assert contract.unresolved and not contract.obligations
    for owner in (L2, TP):
        result = shadow.inspect_source_paths(
            candidate, source_text=source, owner=owner, obligations=contract.obligations
        )
        assert result.unresolved and not result.rows


def test_case_boundary_preserves_text_expected_values(candidate, monkeypatch):
    cases = yaml.safe_load((FIXTURE / "cases.yaml").read_text())
    candidate["rules"].append({"name": "text_label", "dtype": "Text"})
    cases[0]["output"]["fixture#text_label"] = "0.50"
    actual_adapter = shadow.sc._typed_numeric_expected_cases
    seen = []

    def checked_adapter(case_values, rules):
        normalized = actual_adapter(case_values, rules)
        assert normalized[0]["output"]["fixture#text_label"] == "0.50"
        assert isinstance(normalized[0]["output"]["fixture#text_label"], str)
        seen.append(True)
        return normalized

    monkeypatch.setattr(shadow.sc, "_typed_numeric_expected_cases", checked_adapter)
    shadow.corroborated_case_links(candidate, both_owners(candidate)[L2], cases)
    assert seen == [True]
