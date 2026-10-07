from __future__ import annotations

import copy
import functools
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import source_path_evidence as evidence
from axiom_encode.harness import validator_pipeline as v

FIXTURE = Path(__file__).parent / "fixtures/source_path_shadow"
SOURCE = (FIXTURE / "source.txt").read_text()
CITATION = "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit"
EXTRACT = functools.partial(
    v.extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
)


def inputs():
    return yaml.safe_load((FIXTURE / "candidate.yaml").read_text()), yaml.safe_load(
        (FIXTURE / "cases.yaml").read_text()
    )


def certify(payload, cases, source=SOURCE):
    return evidence.certify_worksheet_paths(
        payload,
        source_text=source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        extract_numeric_occurrences=EXTRACT,
    )


def analyze(payload, cases):
    return sc.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        SOURCE,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=EXTRACT,
        extract_numeric_grounding_occurrences=EXTRACT,
        extract_named_scalars=v.extract_named_scalar_occurrences,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
    )


def rule(p, name):
    return next(r for r in p["rules"] if r["name"] == name)


def test_original47_certifies_only_four_note4_frames_not_missing_amt_pairs():
    p, c = inputs()
    result = certify(p, c)
    assert len(result.condition_spans) == 4
    assert all(name == evidence._TAX for name, *_ in result.condition_spans)
    assert all(
        not o.certified for o in result.obligations if o.owner == evidence._LINE2
    )
    assert all(o.certified for o in result.obligations if o.owner == evidence._TAX)
    assert not result.owns_exception(1493, 1807)  # cap/transfer untouched
    assert not result.owns_exception(1808, 2097)


def test_actual_analyzer_keeps_cap_transfer_year_and_amt_findings():
    p, c = inputs()
    issues = analyze(p, c).issues
    condition = next(x for x in issues if "source-explicit-conditions" in x)
    assert evidence._LINE2 in condition
    assert evidence._TAX not in condition
    tests = next(x for x in issues if "Source-stated exceptions" in x)
    assert "amount on line 5 should not be more" in tests
    assert 'entering "0" on lines 43 and 45' not in tests
    assert "resident of Alberta" not in tests
    assert any("2025 has no named scalar" in x for x in issues)


@pytest.mark.parametrize(
    "mutation",
    [
        "unit",
        "dtype",
        "entity",
        "wrong_result",
        "dead_result",
        "missing_proof",
        "bad_date",
        "partial_interval",
        "wrong_case",
    ],
)
def test_actual_analyzer_bad_evidence_retains_monetary_findings(mutation):
    p, c = inputs()
    r = rule(p, evidence._TAX)
    if mutation in {"unit", "dtype", "entity"}:
        r[mutation] = {"unit": "USD", "dtype": "Text", "entity": "Household"}[mutation]
    elif mutation == "wrong_result":
        r["versions"][0]["formula"] = r["versions"][0]["formula"].replace(
            "on428_line_81_after_entering_zero_on_lines_70_and_72",
            "on428mj_line_58_after_entering_zero_on_lines_43_and_45",
        )
    elif mutation == "dead_result":
        r["versions"][0]["formula"] = "0"
    elif mutation == "missing_proof":
        r["metadata"]["proof"]["atoms"] = [
            a
            for a in r["metadata"]["proof"]["atoms"]
            if a["path"] != "versions[0].formula"
        ]
    elif mutation == "bad_date":
        for a in r["metadata"]["proof"]["atoms"]:
            if a["path"].endswith("effective_to"):
                a["source"]["excerpt"] = "for 2025"
    elif mutation == "partial_interval":
        r["versions"][0]["effective_to"] = "2025-06-30"
    else:
        for case in c:
            for key in case["output"]:
                if key.endswith("#" + evidence._TAX):
                    case["output"][key] += 1
    result = certify(p, c)
    assert not any(name == evidence._TAX for name, *_ in result.condition_spans)
    issues = analyze(p, c).issues
    assert any("resident of Alberta" in issue for issue in issues)
    assert any("2025 has no named scalar" in issue for issue in issues)


def test_no_numeric_certification_or_no_cases_never_grants_coverage():
    p, c = inputs()
    assert not evidence.certify_worksheet_paths(
        p, source_text=SOURCE, corpus_citation_path=CITATION, test_cases=c
    ).condition_spans
    assert not certify(p, []).condition_spans


def test_original_case_objects_and_source_are_not_mutated():
    p, c = inputs()
    original = copy.deepcopy((p, c))
    source = SOURCE
    certify(p, c)
    assert (p, c) == original and SOURCE == source


def test_counterpart_output_key_mismatch_cannot_supply_parent_pair():
    p, c = inputs()
    for case in c:
        if (
            case["name"].endswith("_residence_pair_False")
            or case["name"] == "ontario_nonamt_single"
        ):
            case["output"]["unrelated#extra_assertion_" + case["name"]] = 1
    result = certify(p, c)
    assert not any(a <= 5071 and 5206 <= b for _, a, b, _, _ in result.condition_spans)
    assert any("resident of Ontario" in issue for issue in analyze(p, c).issues)


@pytest.mark.parametrize("version", [False, True])
def test_other_claiming_owner_or_interval_cannot_borrow_certified_exception(version):
    p, c = inputs()
    extra = copy.deepcopy(rule(p, evidence._TAX))
    extra["name"] = "other_claimed_tax_result"
    extra["versions"][0]["formula"] = "0"
    if version:
        extra["versions"][0]["effective_from"] = "2026-01-01"
        extra["versions"][0]["effective_to"] = "2026-12-31"
    p["rules"].append(extra)
    result = certify(p, c)
    assert result.condition_spans  # legitimate original-owner paths remain
    assert not result.owns_exception(
        5554,
        5832,
        source_text=SOURCE,
        principal_rules={r["name"]: r for r in p["rules"]},
    )
    assert any("resident of Alberta" in issue for issue in analyze(p, c).issues)


def narrow_row_parts(source):
    start = source.index(" 1\nEnter the amount from line 3")
    end = source.index("\nThe amount on line 5")
    clause = sc._SourceConditionClause((), start, end, source[start:end])
    return sc._partition_condition_clause_at_rows(
        clause, source_text=source, row_ends=sc._corroborated_unless_row_ends(source)
    )


def test_source_global_complete_context_partitions_only_exact_owned_early_operations():
    before = SOURCE
    parts = narrow_row_parts(SOURCE)
    assert [(x.start, x.end) for x in parts] == [(1220, 1309), (1309, 1492)]
    assert "".join(SOURCE[x.start : x.end] for x in parts) == SOURCE[1220:1492]
    assert "unless you have to pay minimum tax" in parts[0].text
    assert "Line 1 minus line 2" in parts[1].text
    assert "Enter whichever amount is less" in parts[1].text
    assert SOURCE == before


@pytest.mark.parametrize(
    "old,new",
    [
        ("minimum tax.(1) – 2", "minimum tax only for residents.(1) – 2"),
        (
            "Net foreign\nnon-business income",
            "Only resident claimants\nnon-business income",
        ),
        ("should not be more than", "should be more than"),
        ("Enter the total from line 5", "Enter the total from line 4"),
        (
            "on the line for the provincial or territorial foreign tax credit of",
            "on the line for the credit for disabled claimants of",
        ),
        ("of\nForm 428.", "of\nForm 428 if you are eligible."),
        (
            "\nLine 1 minus line 2 = 3",
            "\nIf the claimant is disabled, Line 1 minus line 2 = 3",
        ),
    ],
)
def test_narrow_clause_still_requires_unchanged_complete_layout_and_transfer(old, new):
    source = SOURCE.replace(old, new)
    assert source != SOURCE
    assert len(narrow_row_parts(source)) == 1


def test_narrow_clause_rejects_missing_transfer_or_quote_wrapped_context():
    assert (
        len(narrow_row_parts(SOURCE[: SOURCE.index("Enter the total from line 5")]))
        == 1
    )
    assert len(narrow_row_parts('"' + SOURCE + '"')) == 1


@pytest.mark.parametrize("end", [1320, 1410, 1491])
def test_narrow_clause_cannot_end_inside_an_operand_or_instruction(end):
    clause = sc._SourceConditionClause((), 1220, end, SOURCE[1220:end])
    assert sc._partition_condition_clause_at_rows(
        clause, source_text=SOURCE, row_ends=sc._corroborated_unless_row_ends(SOURCE)
    ) == (clause,)


def test_narrow_clause_does_not_discard_unowned_preceding_conditions():
    clause = sc._SourceConditionClause((), 0, 1492, SOURCE[:1492])
    assert sc._partition_condition_clause_at_rows(
        clause, source_text=SOURCE, row_ends=sc._corroborated_unless_row_ends(SOURCE)
    ) == (clause,)


@pytest.mark.parametrize("start", [1230, 1260])
def test_narrow_clause_cannot_start_inside_the_owned_row(start):
    clause = sc._SourceConditionClause((), start, 1492, SOURCE[start:1492])
    assert sc._partition_condition_clause_at_rows(
        clause, source_text=SOURCE, row_ends=sc._corroborated_unless_row_ends(SOURCE)
    ) == (clause,)


def test_narrow_clause_requires_unique_complete_context():
    assert len(narrow_row_parts(SOURCE + "\n" + SOURCE)) == 1


@pytest.mark.parametrize("start", [1220, 1221, 1222, 1223])
def test_narrow_clause_accepts_only_coordinate_or_instruction_whitespace_boundaries(
    start,
):
    clause = sc._SourceConditionClause((), start, 1492, SOURCE[start:1492])
    parts = sc._partition_condition_clause_at_rows(
        clause, source_text=SOURCE, row_ends=sc._corroborated_unless_row_ends(SOURCE)
    )
    assert len(parts) == 2
    assert parts[0].start == start and parts[-1].end == 1492


def test_post_transfer_restriction_is_not_owned_or_erased_by_early_partition():
    end = SOURCE.index("Form 428.") + len("Form 428.")
    restriction = (
        " Only claimants who are residents and whose spouses are disabled may claim."
    )
    source = SOURCE[:end] + restriction + SOURCE[end:]
    # The early proof ends before the cap and transfer: partitioning those exact
    # owned bytes cannot certify or discard a later, separately owned restriction.
    early = narrow_row_parts(source)
    assert [(p.start, p.end) for p in early] == [(1220, 1309), (1309, 1492)]
    assert source[end : end + len(restriction)] == restriction
    whole = sc._SourceConditionClause(
        (), 1220, end + len(restriction), source[1220 : end + len(restriction)]
    )
    assert sc._partition_condition_clause_at_rows(
        whole, source_text=source, row_ends=sc._corroborated_unless_row_ends(source)
    ) == (whole,)
    assert sc._source_conjunctive_fact_gates(whole.text)
