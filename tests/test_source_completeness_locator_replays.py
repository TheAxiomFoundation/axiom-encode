"""Replay committed candidate fixtures without demanding citation locators.

The three U.S. fixtures retain source, candidate, companion tests, and metadata.
The tests authenticate these bytes against the committed hashes; the fixture
README describes the evidence that is available from this checkout.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.source_completeness import (
    analyze_complete_source_unit,
    authoritative_numeric_recall_text,
)
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

FIXTURES = Path(__file__).parent / "fixtures/source_completeness"
RECALL_VALUE = re.compile(
    r"\[complete-source-unit:numeric-recall\] Authoritative corpus numeric "
    r"value (\S+) has no named scalar"
)


def _numeric_demands(issues: tuple[str, ...] | list[str]) -> set[str]:
    return {match.group(1) for issue in issues if (match := RECALL_VALUE.search(issue))}


def _inventory(source: str, citation: str) -> set[float]:
    cleaned = authoritative_numeric_recall_text(source, corpus_citation_path=citation)
    return {
        occurrence.value
        for occurrence in extract_typed_numeric_inventory_occurrences_from_text(
            cleaned, profile="en-US"
        )
    }


@functools.lru_cache(maxsize=None)
def _replay(run_id: str):
    fixture = FIXTURES / f"signed_reencode_{run_id}"
    recorded = json.loads((fixture / "issues.json").read_text())
    content = (fixture / "candidate.yaml").read_text()
    tests = (fixture / "candidate.test.yaml").read_text()
    assert hashlib.sha256(content.encode()).hexdigest() == recorded["rulespec_sha256"]
    assert hashlib.sha256(tests.encode()).hexdigest() == recorded["tests_sha256"]
    source = (fixture / "source.txt").read_text()
    metadata = json.loads((fixture / "source-metadata.json").read_text())
    assert (
        hashlib.sha256(source.encode()).hexdigest()
        == metadata["source_attestation"]["resolved_text_sha256"]
    )
    analysis = analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=recorded["citation"],
        test_cases=yaml.safe_load(tests),
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return recorded, source, list(analysis.issues)


@pytest.mark.parametrize(
    ("run_id", "locator_demands", "operative_values", "operative_excerpts"),
    (
        (
            "37936331759",
            {"162", "323", "106"},
            {12000},
            ("sixty-five (65) years of age or older", "only one claim per year"),
        ),
        (
            "37936338132",
            {"49", "620"},
            {2500, 3401, 502},
            ("$ 0-2,500 $502", "3,401-3,550 345"),
        ),
        (
            "37936341740",
            {"444", "117", "118", "552", "648", "217", "615", "658", "725"},
            {4000, 70, 500, 3000, 6000, 930},
            (),
        ),
    ),
)
def test_real_candidates_no_longer_demand_locator_scalars(
    run_id, locator_demands, operative_values, operative_excerpts
):
    recorded, source, issues = _replay(run_id)

    assert not _numeric_demands(issues) & locator_demands
    assert operative_values <= _inventory(source, recorded["citation"])
    cleaned = authoritative_numeric_recall_text(
        source, corpus_citation_path=recorded["citation"]
    )
    assert all(excerpt in cleaned for excerpt in operative_excerpts)


@pytest.mark.parametrize(
    ("run_id", "current_count"),
    (("37936331759", 0), ("37936338132", 51), ("37936341740", 117)),
)
def test_replay_diagnostic_totals(run_id, current_count):
    _recorded, _source, issues = _replay(run_id)
    assert len(issues) == current_count


def test_real_arizona_omissions_remain_rejected():
    _recorded, _source, issues = _replay("37936338132")

    assert {"1750", "1751", "1851", "1951"} <= _numeric_demands(issues)
    assert any("[complete-source-unit:formula-output]" in issue for issue in issues)


def test_real_virginia_omissions_remain_rejected():
    _recorded, _source, issues = _replay("37936341740")

    assert {"3000", "6000", "930", "500"} <= _numeric_demands(issues)
    assert any("[complete-source-unit:formula-output]" in issue for issue in issues)


def test_arizona_replay_only_excludes_statutes_at_large_numbers():
    recorded, source, issues = _replay("37936338132")
    cleaned = authoritative_numeric_recall_text(
        source, corpus_citation_path=recorded["citation"]
    )
    assert source.count("(49 Stat. 620)") == 1
    assert "(49 Stat. 620)" not in cleaned
    assert "(   Stat.    )" in cleaned
    assert cleaned.startswith(source.split("\n\n", 1)[0] + "\n\n")
    assert {"43", "1072"} <= _numeric_demands(issues)
    assert not {"49", "620"} & _numeric_demands(issues)
    assert any(
        "[complete-source-unit:formula-output]" in issue
        and "43-1072 - Earned credit for property taxes;" in issue
        for issue in issues
    )
