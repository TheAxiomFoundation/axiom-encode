"""Copied-return categories retain physical source occurrence custody."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as s

F = Path(__file__).parent / "fixtures"
BODY = (F / "detached_schedule2_source.txt").read_text()
PAYLOAD = yaml.safe_load((F / "same_page_floor_candidate.yaml").read_text())
CITATION = "ca/policy/cra/t1-2025/federal-spouse-transfer-amounts"


def bind(payload=PAYLOAD, body=BODY):
    return s._bound_copied_amount_categories(payload, body, CITATION, {CITATION: body})


def test_actual_direct_copy_binds_only_category18():
    result = bind()
    assert len(result) == 1
    assert BODY[slice(*result[0].span)] == "18"
    assert result[0].value == 18
    mapping = s._NumericRecallSourceMap(list(range(len(BODY))))
    recalled = s.authoritative_numeric_recall_text(
        BODY, corpus_citation_path=CITATION, _source_map=mapping
    )
    start = recalled.index("under 18") + len("under ")
    assert mapping.source_span(start, start + 2) == result[0].span


@pytest.mark.parametrize(
    "mutation",
    [
        "formula",
        "wrong_input",
        "unit",
        "dates",
        "proof",
        "default",
        "duplicate",
        "hash",
    ],
)
def test_wrong_projection_or_custody_refuses(mutation):
    p = deepcopy(PAYLOAD)
    r = next(r for r in p["rules"] if r["name"] == "schedule_2_caregiver_amount_line_2")
    i = next(i for i in p["inputs"] if i["name"] == r["versions"][0]["formula"])
    if mutation == "formula":
        r["versions"][0]["formula"] += " + 0"
    elif mutation == "wrong_input":
        r["versions"][0]["formula"] = "spouse_taxable_income_line_26000"
    elif mutation == "unit":
        r["unit"] = "USD"
    elif mutation == "dates":
        r["versions"][0]["effective_to"] = "2026-12-31"
    elif mutation == "proof":
        r["metadata"]["proof"]["atoms"] = []
    elif mutation == "default":
        i["default"] = 0
    elif mutation == "duplicate":
        other = deepcopy(r)
        other["name"] += "_duplicate"
        p["rules"].append(other)
    else:
        p["module"]["source_verification"]["source_sha256"] = "bad"
    assert not bind(p)


def test_untrusted_or_changed_source_refuses():
    assert not s._bound_copied_amount_categories(PAYLOAD, BODY, CITATION, {})
    assert not bind(body=BODY.replace("under 18", "under 19"))
    assert not bind(body=BODY + "18")


@pytest.mark.parametrize(
    "key,value", [("rounding", "floor"), ("resolved_currency_rounding", {})]
)
def test_executable_projection_transform_refuses(key, value):
    p = deepcopy(PAYLOAD)
    r = next(r for r in p["rules"] if r["name"] == "schedule_2_caregiver_amount_line_2")
    r[key] = value
    assert not bind(p)


def numeric_issues(payload=PAYLOAD):
    from functools import partial

    from axiom_encode.harness.validator_pipeline import (
        extract_named_scalar_occurrences,
        extract_typed_numeric_inventory_occurrences_from_text,
        numeric_value_is_grounded,
    )

    cases = yaml.safe_load((F / "same_page_floor_candidate.test.yaml").read_text())
    result = s.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        BODY,
        corpus_citation_path=CITATION,
        test_cases=cases,
        source_context={CITATION: BODY},
        extract_numeric_occurrences=partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return {
        str(i)
        for i in result.issues
        if "[complete-source-unit:numeric-recall]" in str(i)
    }


def test_actual_numeric_loop_projection_delta():
    original = numeric_issues()
    assert not any("value 18 " in i for i in original)
    p = deepcopy(PAYLOAD)
    r = next(r for r in p["rules"] if r["name"] == "schedule_2_caregiver_amount_line_2")
    r["versions"][0]["formula"] += " + 0"
    mutated = numeric_issues(p)
    assert len(mutated - original) == 1
    assert all("value 18 " in i for i in mutated - original)
    assert original <= mutated


@pytest.mark.parametrize(
    "mask", ["represented_annual_spans", "represented_measurement_spans"]
)
def test_actual_numeric_loop_synthetic_mask_path(monkeypatch, mask):
    # Synthetic routing control, not certification of an annual/measurement fact.
    original = s._companion_test_issues
    span = (BODY.index("1960"), BODY.index("1960") + 4)

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        kwargs[mask].add(span)
        return result

    monkeypatch.setattr(s, "_companion_test_issues", wrapped)
    assert not any("value 18 " in i for i in numeric_issues())


@pytest.mark.parametrize("replacement", [None, 0])
def test_actual_numeric_loop_corrupt_origin_refuses(monkeypatch, replacement):
    # Preserve the recalled literal/value, corrupt only its physical custody.
    original = s.authoritative_numeric_recall_text

    def wrapped(*args, **kwargs):
        text = original(*args, **kwargs)
        mapping = kwargs.get("_source_map")
        if mapping is not None and "under 18" in text:
            start = text.index("under 18") + len("under ")
            mapping.offsets[start : start + 2] = [replacement, replacement]
        return text

    monkeypatch.setattr(s, "authoritative_numeric_recall_text", wrapped)
    assert any("value 18 " in i for i in numeric_issues())
