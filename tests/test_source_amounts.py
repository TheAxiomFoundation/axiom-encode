"""Numeric outputs in the source-conflict compare step (``combine: value``)."""

from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from axiom_encode.source_conflicts.artifact import Artifact
from axiom_encode.source_conflicts.compare import CompareError, compare, render_record
from axiom_encode.source_conflicts.engine import Engine
from axiom_encode.source_conflicts.gates import LinkGateError
from axiom_encode.source_conflicts.link import LinkError, load_link, parse_link
from axiom_encode.source_conflicts.question import compare_question, parse_question
from tests.test_source_conflicts import (
    FIXTURES,
    LINK,
    REPOSITORY,
    RULESPEC,
    FakeEngine,
    artifact_payload,
    run_compare,
)

AMOUNT_LINK = FIXTURES / "income_threshold.link.yaml"
SA = "us:statutes/999/2/a#"
RB = "us:regulations/99-cfr/1/3#"
MODULES = {
    "us/statutes/999/2/a.yaml": "amount_a",
    "us/regulations/99-cfr/1/3.yaml": "amount_b",
    "us/statutes/999/1/a.yaml": "a",
    "us/regulations/99-cfr/1/2.yaml": "b",
}


def document() -> dict:
    return yaml.safe_load(AMOUNT_LINK.read_text())


def engine() -> FakeEngine:
    return FakeEngine({path: artifact_payload(name) for path, name in MODULES.items()})


def run(
    tmp_path: Path, doc: dict | None = None, fake: FakeEngine | None = None
) -> dict:
    return compare(
        parse_link(doc if doc is not None else document()),
        rulespec_root=RULESPEC,
        engine=fake or engine(),
        workdir=tmp_path,
        repository=REPOSITORY,
    )


def test_two_amounts_that_differ_are_a_witness_with_both_values(tmp_path):
    record = run(tmp_path)
    assert record["review_state"] == "unreviewed"
    assert record["conflict"] is True
    assert record["comparison"] == {"kind": "value", "tolerance": "0", "unit": "USD"}
    # 7.25 is the fixed federal wage inside max(federal, state): a threshold of the state wage.
    assert record["facts"]["state_wage"]["values"] == ["0", "6.25", "7.25", "8.25"]
    assert record["counts"] == {
        "grid": 8,
        "combinations": 8,
        "removed_by_constraints": 0,
        "removed_that_differ": 0,
        "run": 8,
        "skipped_by_guard": 0,
        "differ": 1,
        "witness_groups": 1,
    }
    (group,) = record["witness_groups"]
    assert group["direction"] == {"a": "lower", "b": "higher"}
    assert group["deciding_rules"] == {
        "a": [SA + "monthly_income_threshold"],
        "b": [RB + "income_threshold_for_month"],
    }
    assert group["conditions"] == [
        {"fact": "state_wage", "read_by": "b", "value": "nonzero"},
        {"fact": "state_elects_own_wage", "read_by": "b", "value": True},
    ]
    assert group["difference"] == {"min": "80", "max": "80"}
    assert group["witness"] == {
        "facts": {"state_wage": "8.25", "state_elects_own_wage": True},
        "a": {"result": "580", "outputs": {SA + "monthly_income_threshold": "580"}},
        "b": {"result": "660", "outputs": {RB + "income_threshold_for_month": "660"}},
        "difference": "80",
    }
    assert RB + "hours_for_threshold" in record["sides"]["b"]["rules"]


def test_tolerance_is_the_largest_difference_that_is_not_a_conflict(tmp_path):
    doc = document()
    doc["facts"]["state_wage"]["values"] = ["7.25", "7.26", "8.25", "15"]
    record = run(tmp_path, doc)
    (group,) = record["witness_groups"]
    assert (record["counts"]["differ"], group["count"]) == (3, 3)
    assert group["difference"] == {"min": "0.8", "max": "620"}
    assert group["witness"]["difference"] == "0.8"

    for tolerance, differ in (
        ("0.79", 3),
        ("0.8", 2),
        (80, 1),
        ("619.99", 1),
        (620, 0),
    ):
        doc["tolerance"] = tolerance
        record = run(tmp_path, doc)
        assert record["counts"]["differ"] == differ, tolerance
        assert record["conflict"] is (differ > 0)
        assert record["comparison"]["tolerance"] == str(tolerance)


def test_both_directions_are_separate_groups(tmp_path):
    doc = document()
    doc["fixed"][SA + "input.federal_minimum_wage"]["value"] = "8"
    doc["facts"]["state_wage"]["values"] = ["0", "15"]
    record = run(tmp_path, doc)
    directions = [group["direction"] for group in record["witness_groups"]]
    assert {"a": "higher", "b": "lower"} in directions
    assert {"a": "lower", "b": "higher"} in directions
    higher = [g for g in record["witness_groups"] if g["direction"]["a"] == "higher"]
    assert all(Decimal(g["difference"]["max"]) < 0 for g in higher)
    assert (
        sum(g["count"] for g in record["witness_groups"])
        == record["counts"]["differ"]
        == 4
    )


def test_value_records_are_byte_identical_and_yes_no_records_have_no_new_key(tmp_path):
    assert render_record(run(tmp_path)) == render_record(run(tmp_path))
    yes_no = run_compare(tmp_path=tmp_path)
    assert "comparison" not in yes_no
    assert all("difference" not in group for group in yes_no["witness_groups"])
    assert yes_no["witness_groups"][0]["direction"] == {"a": False, "b": True}


def test_the_hash_of_a_yes_no_link_does_not_change_with_this_feature():
    # Pinned on the branch before numeric outputs: a v1 yes/no link keeps its hash.
    assert (
        load_link(LINK).sha256
        == "beab81a533aae1fc9d8daf691be3a4039d6890143ec073727e3811084fc8dc67"
    )
    doc = document()
    first = parse_link(doc).sha256
    doc["tolerance"] = "0.00"
    assert parse_link(doc).sha256 == first
    doc["tolerance"] = "1"
    assert parse_link(doc).sha256 != first


@pytest.mark.parametrize(
    "change,message",
    [
        (
            lambda d: d["b"].update(combine="any"),
            "must both be 'value', or both be 'any' or 'all'",
        ),
        (
            lambda d: d["a"]["outputs"].append(SA + "x"),
            "a.outputs must name exactly one rule",
        ),
        (
            lambda d: d.update(tolerance=-1),
            "link.tolerance must be a number that is not negative",
        ),
        (lambda d: d.update(tolerance="about 1"), "link.tolerance must be a number"),
    ],
)
def test_invalid_value_links_are_refused(change, message):
    doc = document()
    change(doc)
    with pytest.raises(LinkError, match=message):
        parse_link(doc)


def test_tolerance_on_a_yes_no_link_is_refused():
    doc = yaml.safe_load(LINK.read_text())
    doc["tolerance"] = 1
    with pytest.raises(
        LinkError, match="applies only when both sides have combine 'value'"
    ):
        parse_link(doc)


def test_gates_match_the_output_kind_to_combine(tmp_path):
    doc = yaml.safe_load(LINK.read_text())
    for side in ("a", "b"):
        doc[side]["combine"] = "value"
    with pytest.raises(
        LinkGateError, match="has dtype judgment; combine 'value' compares a number"
    ):
        run(tmp_path, doc)
    doc = document()
    for side in ("a", "b"):
        doc[side]["combine"] = "any"
    with pytest.raises(LinkGateError, match="use combine 'value' to compare a number"):
        run(tmp_path, doc)


def test_gate_units_of_the_two_numbers_must_match(tmp_path):
    payloads = {path: artifact_payload(name) for path, name in MODULES.items()}
    payloads["us/regulations/99-cfr/1/3.yaml"]["program"]["derived"][0]["unit"] = "GBP"
    with pytest.raises(
        LinkGateError, match="different units: USD on side a, GBP on side b"
    ):
        run(tmp_path, fake=FakeEngine(payloads))


def test_an_engine_result_that_is_no_number_is_an_error(tmp_path):
    class Broken(FakeEngine):
        def run(self, artifact, request):
            response = super().run(artifact, request)
            for result in response["results"]:
                for output in result["outputs"].values():
                    output["value"] = {"kind": "text", "value": "n/a"}
            return response

    fake = Broken({path: artifact_payload(name) for path, name in MODULES.items()})
    with pytest.raises(CompareError, match="is not a number"):
        run(tmp_path, fake=fake)


def test_a_constant_in_min_or_max_is_a_threshold():
    artifact = Artifact(artifact_payload("amount_b"), "2027-01-01")
    derived, _, _ = artifact.closure([RB + "income_threshold_for_month"])
    assert artifact.thresholds(derived, {}) == []
    (threshold,) = artifact.thresholds(derived, {"federal_wage_rate": "7.25"})
    assert (threshold.value, threshold.inputs) == (
        Decimal("7.25"),
        frozenset({"state_wage_rate"}),
    )


def test_a_question_passes_its_tolerance_to_pairs_of_numbers(tmp_path):
    link = document()
    question = {
        "format": "axiom-source-question/v1",
        "id": "us:fixture/monthly-income-threshold",
        "question": link["question"],
        "entity": link["entity"],
        "period": link["period"],
        "facts": {name: {"kind": fact["kind"]} for name, fact in link["facts"].items()},
        "tolerance": "79.99",
        "answers": [
            {
                "id": "statute",
                "label": link["a"]["label"],
                "document_type": "statute",
                "jurisdiction": "us",
                "corpus_citation_path": "us/statute/999/2/a",
                "module": link["a"]["module"],
                "outputs": link["a"]["outputs"],
                "combine": "value",
                "fixed": {
                    SA + "input.federal_minimum_wage": {
                        "kind": "decimal",
                        "value": "7.25",
                    }
                },
            },
            {
                "id": "regulation",
                "label": link["b"]["label"],
                "document_type": "regulation",
                "jurisdiction": "us",
                "corpus_citation_path": "us/regulation/99/1/3",
                "module": link["b"]["module"],
                "outputs": link["b"]["outputs"],
                "combine": "value",
                "inputs": {name: fact["b"] for name, fact in link["facts"].items()},
                "fixed": {
                    RB + "input.federal_wage_rate": {"kind": "decimal", "value": "7.25"}
                },
            },
        ],
    }
    summary, records = compare_question(
        parse_question(question),
        rulespec_root=RULESPEC,
        engine=engine(),
        workdir=tmp_path,
        repository=REPOSITORY,
    )
    (pair,) = summary["pairs"]
    assert (pair["run"], pair["differ"]) == (8, 1)
    assert pair["witness_groups"] == [
        {
            "direction": {"a": "lower", "b": "higher"},
            "count": 1,
            "conditions": [
                {"fact": "state_wage", "read_by": "b", "value": "nonzero"},
                {"fact": "state_elects_own_wage", "read_by": "b", "value": True},
            ],
            "difference": {"min": "80", "max": "80"},
        }
    ]
    assert records["statute~regulation.json"]["comparison"]["tolerance"] == "79.99"
    question["tolerance"] = 80
    summary, _ = compare_question(
        parse_question(question),
        rulespec_root=RULESPEC,
        engine=engine(),
        workdir=tmp_path,
        repository=REPOSITORY,
    )
    assert summary["counts"]["pairs_that_differ"] == 0
    question["tolerance"] = -1
    with pytest.raises(
        LinkError, match="tolerance must be a number that is not negative"
    ):
        parse_question(question)


def test_real_engine_gives_the_amount_record_the_fixture_artifacts_give(
    real_engine, tmp_path
):
    checkout = tmp_path / "rulespec-us"
    shutil.copytree(RULESPEC, checkout)
    workdir = tmp_path / "work"
    workdir.mkdir()
    record = compare(
        load_link(AMOUNT_LINK),
        rulespec_root=checkout,
        engine=Engine(real_engine.binary, "f" * 40),
        workdir=workdir,
        repository=REPOSITORY,
    )
    expected = run(tmp_path / "fake")
    assert record["counts"]["differ"] == 1
    for volatile in (record, expected):
        volatile["engine"]["version"] = None
    assert json.loads(render_record(record)) == json.loads(render_record(expected))
