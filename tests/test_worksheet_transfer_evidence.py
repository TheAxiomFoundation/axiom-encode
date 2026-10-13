from __future__ import annotations

import copy
import functools
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import validator_pipeline as v
from axiom_encode.harness import worksheet_operation_evidence as operations
from axiom_encode.harness import worksheet_transfer_evidence as transfer

ROOT = Path(__file__).parent / "fixtures"
SOURCE = (ROOT / "source_path_shadow/source.txt").read_text()
CONTEXT = {
    operations._PRIMARY: SOURCE,
    operations._SECONDARY: (ROOT / "worksheet_operations/secondary.txt").read_text(),
}


def inputs():
    return yaml.safe_load(
        (ROOT / "worksheet_transfers/candidate.yaml").read_text()
    ), yaml.safe_load((ROOT / "worksheet_transfers/cases.yaml").read_text())


def certify(payload, cases):
    return transfer.certify_worksheet_transfers(
        payload,
        source_text=SOURCE,
        corpus_citation_path=operations._PRIMARY,
        source_context=CONTEXT,
        test_cases=cases,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
    )


def rule(p, name):
    return next(r for r in p["rules"] if r["name"] == name)


def test_actual_four_cases_compute_children_before_sum_and_routes():
    p, c = inputs()
    result = certify(p, c)
    assert not result.unresolved
    assert len(result.cases) == 4
    assert [x.children for x in result.cases] == [
        (150, 150),
        (150, 150),
        (150, 80),
        (150, 80),
    ]
    assert [x.total for x in result.cases] == [300, 300, 230, 230]
    assert result.routing_pairs == ((50, 51), (52, 53))
    assert result.source_spans == ((1669, 1807), (1808, 2097))


@pytest.mark.parametrize(
    "mutation",
    [
        "wrapper",
        "count",
        "reverse_route",
        "wrong_threshold",
        "paid_selector",
        "wrong_unit",
        "wrong_entity",
        "partial_year",
        "missing_proof",
        "unrelated_owner",
        "missing_child",
        "empty",
        "wrong_parent",
        "wrong_child_kind",
        "cross_year",
        "not_same_world",
        "wrong_sum_value",
    ],
)
def test_actual_candidate_and_case_mutations_remain_unresolved(mutation):
    p, c = inputs()
    root = c[-1]
    relation = next(k for k, value in root["input"].items() if isinstance(value, list))
    row = root["input"][relation][0]
    if mutation == "wrapper":
        rule(p, transfer.UNIT)["versions"][0]["formula"] = operations._LINE4
    elif mutation == "count":
        rule(p, transfer.TOTAL)["versions"][0]["formula"] = (
            f"count({transfer.RELATION})"
        )
    elif mutation in {"reverse_route", "wrong_threshold", "paid_selector"}:
        r = rule(p, transfer.MULTI)["versions"][0]
        r["formula"] = (
            r["formula"].replace(
                "> 1", "<= 1" if mutation == "reverse_route" else "> 2"
            )
            if mutation != "paid_selector"
            else r["formula"].replace(transfer.PAYABLE, "jurisdictions_tax_paid_count")
        )
    elif mutation == "wrong_unit":
        rule(p, transfer.UNIT)["unit"] = "USD"
    elif mutation == "wrong_entity":
        rule(p, transfer.UNIT)["entity"] = "Person"
    elif mutation == "partial_year":
        rule(p, transfer.TOTAL)["versions"][0]["effective_to"] = "2025-06-30"
    elif mutation == "missing_proof":
        rule(p, transfer.MULTI).pop("metadata")
    elif mutation == "unrelated_owner":
        r = copy.deepcopy(rule(p, transfer.MULTI))
        r["name"] = "unrelated_total"
        p["rules"].append(r)
    elif mutation == "missing_child":
        row.pop(
            next(
                k
                for k in row
                if k.endswith(
                    "#input.amount_allocated_to_province_or_territory_of_residence_t2203_part_1_column_4"
                )
            )
        )
    elif mutation == "empty":
        root["input"][relation] = []
    elif mutation == "wrong_parent":
        root["output"][
            next(k for k in root["output"] if k.endswith("#" + transfer.TOTAL))
        ] = 999
    elif mutation == "wrong_child_kind":
        row[next(k for k in row if k.endswith("#input.minimum_tax_is_payable"))] = (
            "true"
        )
    elif mutation == "cross_year":
        root["period"]["end"] = "2026-12-31"
    elif mutation == "not_same_world":
        c = c[:50] + [c[-1]]
    elif mutation == "wrong_sum_value":
        rule(p, transfer.TOTAL)["versions"][0]["formula"] += " + 1"
    result = certify(p, c)
    assert result.unresolved and not result.source_spans and not result.cases


@pytest.mark.parametrize("where", ["root_relation", "root_count", "child", "output"])
def test_foreign_qualified_aliases_never_bind_local_contract(where):
    p, c = inputs()
    case = c[-1]
    mapping = case["output"] if where == "output" else case["input"]
    if where == "child":
        mapping = next(v for v in case["input"].values() if isinstance(v, list))[0]
    key = next(
        k
        for k in mapping
        if (
            "#relation." in k
            if where == "root_relation"
            else "#input." in k
            if where == "root_count"
            else True
        )
    )
    mapping["ca:policies/unrelated/module#" + key.split("#", 1)[1]] = mapping.pop(key)
    result = certify(p, c)
    assert result.unresolved and not result.source_spans


def test_reordering_unequal_children_does_not_change_decimal_sum():
    p, c = inputs()
    for case in c[-2:]:
        rows = next(v for v in case["input"].values() if isinstance(v, list))
        rows.reverse()
    result = certify(p, c)
    assert not result.unresolved
    assert result.cases[-1].children == (80, 150)
    assert result.cases[-1].total == 230


@pytest.mark.parametrize(
    "field,value",
    [("dtype", "Text"), ("entity", "Household"), ("period", "Month"), ("unit", "USD")],
)
def test_selected_child_helper_contract_cannot_borrow_parent_certificate(field, value):
    p, c = inputs()
    rule(p, "net_income_for_multiple_jurisdictions")[field] = value
    result = certify(p, c)
    assert result.unresolved and not result.source_spans


@pytest.mark.parametrize("field,value", [("unit", "USD"), ("date", "2025-06-30")])
def test_selected_child_constant_retains_source_type_and_interval(field, value):
    p, c = inputs()
    r = rule(p, "t2036_foreign_tax_threshold")
    if field == "date":
        r["versions"][0]["effective_to"] = value
    else:
        r[field] = value
    result = certify(p, c)
    assert result.unresolved and not result.source_spans


def test_unknown_post_transfer_restriction_is_not_ignored():
    import hashlib

    p, c = inputs()
    source = SOURCE.replace(
        transfer._MULTI_TEXT,
        transfer._MULTI_TEXT + " Only disabled claimants may transfer this amount.",
    )
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    result = transfer.certify_worksheet_transfers(
        p,
        source_text=source,
        corpus_citation_path=operations._PRIMARY,
        source_context={**CONTEXT, operations._PRIMARY: source},
        test_cases=c,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
    )
    assert result.unresolved == ("unknown transfer tail ownership",)
    assert not result.source_spans


def test_eleven_child_generated_id_order_is_not_assumed_to_be_row_order():
    from decimal import Decimal

    values = [Decimal(0)] * 11
    values[2] = values[3] = Decimal("0.4")
    values[9] = Decimal("79228162514264337593543950330")
    # Both caller orders are individually representable, but lose fractions at
    # different points. The component must not select whichever matches a test.
    assert transfer._companion_sum(values) is None
    assert transfer._companion_sum([Decimal(i) for i in range(11)]) == 55


def test_generated_id_order_sum_overflow_stays_unresolved():
    from decimal import Decimal

    assert (
        transfer._companion_sum([Decimal("79228162514264337593543950335"), Decimal(1)])
        is None
    )


def _analyze(payload, cases, *, source=SOURCE, context=CONTEXT):
    from axiom_encode.harness import source_completeness as sc

    # Mirror the real ValidatorPipeline boundary: grounding is not inventory,
    # and bound artifact constants accompany the raw source occurrences.
    profile = v._numeric_profile_for_citation_path(operations._PRIMARY)
    content = yaml.safe_dump(payload)
    bindings = v.collect_artifact_numeric_bindings(
        content,
        extract_named_scalars=v.extract_named_scalar_occurrences,
        imported_symbol_contents=(),
    )
    return sc.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=operations._PRIMARY,
        source_context=context,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            v.extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=v.extract_named_scalar_occurrences,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        artifact_numeric_bindings=bindings,
        artifact_numeric_values=tuple(value for _, value in bindings),
    )


def _extra_claimant(payload, excerpt):
    extra = copy.deepcopy(rule(payload, transfer.TOTAL))
    extra["name"] = "unrelated_partial_transfer_claimant"
    extra["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = excerpt
    payload["rules"].append(extra)


@pytest.mark.parametrize(
    "excerpt",
    [
        "the provincial or territorial foreign tax credit of\nForm 428.",
        "enter the amount from line 5 on the applicable line in Part 4,\nSection 428MJ of Form T2203",
        operations._SOURCE_OPERATIONS["line5"][0][1],
    ],
)
def test_partial_and_cap_proofs_on_other_owners_do_not_borrow_transfer(excerpt):
    p, c = inputs()
    _extra_claimant(p, excerpt)
    result = certify(p, c)
    assert result.unresolved
    assert not result.source_spans and result.scalar_operations is None


def test_bundle_preserves_master_cases_and_rejects_stale_identity():
    p, c = inputs()
    original = copy.deepcopy(c)
    result = certify(p, c)
    assert result.matches(p, c, SOURCE, CONTEXT)
    assert c == original and len(c) == 54
    assert result.flat_case_indices == tuple(range(50))
    assert result.scalar_paths.input_identity != result.input_identity
    for altered in (c[:-1], c[::-1]):
        assert not result.matches(p, altered, SOURCE, CONTEXT)
    changed = copy.deepcopy(p)
    rule(changed, transfer.UNIT)["versions"][0]["effective_to"] = "2025-06-30"
    assert not result.matches(changed, c, SOURCE, CONTEXT)
    assert not result.owns_output(transfer.UNIT, rule(changed, transfer.UNIT))
    assert not result.matches(p, c, SOURCE + " ", CONTEXT)
    assert not result.matches(p, c, SOURCE, {**CONTEXT, operations._SECONDARY: ""})


def test_actual_analyzer_consumes_transfer_without_removing_master_cases():
    p, c = inputs()
    original = copy.deepcopy(c)
    result = _analyze(p, c)
    assert c == original and len(c) == 54
    assert not result.issues


@pytest.mark.parametrize(
    "mutation",
    [
        "partial_owner",
        "partial_date",
        "wrong_sum",
        "missing_child",
        "missing_pair",
        "source_tail",
    ],
)
def test_actual_analyzer_failed_bundle_cannot_grant_transfer_or_annual(mutation):
    p, c = inputs()
    source, context = SOURCE, CONTEXT
    if mutation == "partial_owner":
        _extra_claimant(
            p, "the provincial or territorial foreign tax credit of\nForm 428."
        )
    elif mutation == "partial_date":
        rule(p, transfer.UNIT)["versions"][0]["effective_to"] = "2025-06-30"
    elif mutation == "wrong_sum":
        rule(p, transfer.TOTAL)["versions"][0]["formula"] += " + 1"
    elif mutation == "missing_child":
        rows = next(
            value for value in c[-1]["input"].values() if isinstance(value, list)
        )
        del rows[0][
            next(key for key in rows[0] if key.endswith("input." + operations._CREDIT))
        ]
    elif mutation == "missing_pair":
        c = c[:-3]  # One genuine transfer execution is not a routing pair.
    else:
        source = SOURCE.replace(
            "\nT2036 E (25)",
            "\nOnly eligible disabled claimants may transfer.\nT2036 E (25)",
            1,
        )
        # Update declared raw identity so this exercises ownership, not a stale hash.
        import hashlib

        p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
            source.encode()
        ).hexdigest()
        context = {**CONTEXT, operations._PRIMARY: source}
    issues = _analyze(p, c, source=source, context=context).issues
    assert any("2025 has no named scalar" in issue for issue in issues)
    assert any(
        "Enter the total from line 5" in issue
        or "If you have to pay tax to more than one jurisdiction" in issue
        for issue in issues
    )


def test_actual_pipeline_uses_full_master_cases_and_optional_context(tmp_path):
    p, c = inputs()
    pipeline = v.ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-ca",
        axiom_rules_path=tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    content = yaml.safe_dump(p)
    assert (
        pipeline._complete_source_unit_issues(
            content,
            validation_source_texts={operations._PRIMARY: SOURCE},
            proof_source_texts=CONTEXT,
            test_cases=c,
        )
        == []
    )
    missing = pipeline._complete_source_unit_issues(
        content,
        validation_source_texts={operations._PRIMARY: SOURCE},
        proof_source_texts=None,
        test_cases=c,
    )
    assert any("2025 has no named scalar" in issue for issue in missing)


def test_exact_consumer_rejects_partial_spans_and_stale_selected_owners():
    from types import SimpleNamespace

    p, c = inputs()
    result = certify(p, c)
    principals = {r["name"]: r for r in p["rules"] if r.get("kind") == "derived"}
    for a, b in result.source_spans:
        branch = SimpleNamespace(start=a, end=b)
        assert result.owns_exception(
            branch,
            source_text=SOURCE,
            principal_rules=principals,
            corpus_citation_path=operations._PRIMARY,
        )
        assert not result.owns_exception(
            SimpleNamespace(start=a + 1, end=b),
            source_text=SOURCE,
            principal_rules=principals,
            corpus_citation_path=operations._PRIMARY,
        )
    selected = principals[transfer.ORDINARY]
    assert result.owns_condition(
        transfer.ORDINARY,
        selected,
        1492,
        1807,
        "2025-01-01",
        "2025-12-31",
        source_text=SOURCE,
    )
    assert not result.owns_condition(
        transfer.ORDINARY,
        selected,
        1500,
        1807,
        "2025-01-01",
        "2025-12-31",
        source_text=SOURCE,
    )
    assert not result.owns_condition(
        transfer.ORDINARY,
        selected,
        1492,
        1807,
        "2025-01-01",
        "2025-06-30",
        source_text=SOURCE,
    )
    stale = copy.deepcopy(principals)
    stale[operations._LINE5]["versions"][0]["formula"] = "0"
    assert not result.owns_exception(
        SimpleNamespace(start=1669, end=1807),
        source_text=SOURCE,
        principal_rules=stale,
        corpus_citation_path=operations._PRIMARY,
    )


@pytest.mark.parametrize(
    "value, accepted",
    [
        (-(2**63), True),
        (2**63 - 1, True),
        (-(2**63) - 1, False),
        (2**63, False),
        (True, False),
        (False, False),
        (1.0, False),
    ],
)
def test_child_integer_wire_bounds_do_not_invent_legal_positivity(value, accepted):
    declaration = {"entity": "Person", "period": "Year", "dtype": "Integer"}
    assert transfer._input_value_matches(declaration, value) is accepted


@pytest.mark.parametrize("location", ["root", "child"])
@pytest.mark.parametrize("value", [2**63, -(2**63) - 1])
def test_extra_nested_case_outside_i64_cannot_borrow_valid_case_evidence(
    location, value
):
    p, c = inputs()
    extra = copy.deepcopy(c[-1])
    extra["name"] = "out_of_wire_range_control"
    if location == "root":
        key = next(
            k for k in extra["input"] if k.endswith("#input." + transfer.PAYABLE)
        )
        extra["input"][key] = value
    else:
        rows = next(v for v in extra["input"].values() if isinstance(v, list))
        key = next(
            k for k in rows[0] if k.endswith("#input.jurisdictions_tax_paid_count")
        )
        rows[0][key] = value
    result = certify(p, [*c, extra])
    assert result.unresolved
    assert not result.cases and result.scalar_operations is None


def test_root_count_wire_maximum_retains_existing_positive_domain():
    p, c = inputs()
    extra = copy.deepcopy(c[-1])
    extra["name"] = "integer_wire_boundary_not_a_new_legal_scenario"
    key = next(k for k in extra["input"] if k.endswith("#input." + transfer.PAYABLE))
    extra["input"][key] = 2**63 - 1
    result = certify(p, [*c, extra])
    assert not result.unresolved and len(result.cases) == 5
    assert result.cases[-1].multi == 230
    # This verifies the existing typed boundary only, not real jurisdiction
    # counts or new legal eligibility facts.
