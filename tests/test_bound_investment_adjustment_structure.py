"""Complete-source recognition tests, not production source-credit tests."""

from pathlib import Path

import pytest

from axiom_encode.harness.bound_investment_adjustment_structure import (
    CITATION,
    recognize_investment_adjustment_source,
)

SOURCE = (
    Path(__file__).parent / "fixtures/schedule-n-authenticated-body.txt"
).read_text()


def test_complete_raw_source_partition():
    bound = recognize_investment_adjustment_source(SOURCE)
    assert bound is not None
    assert len(bound.records) == 36
    assert len(bound.separators) == 35
    spans = sorted([r.span for r in bound.records] + list(bound.separators))
    assert spans[0][0] == 0 and spans[-1][1] == len(SOURCE)
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
    assert bound.bound(SOURCE, CITATION)
    assert not bound.bound(SOURCE, CITATION + "/other")
    assert (
        bound.source_sha256
        == "06295f18048c53e225bf0e8ba0c2581b4ac500f4702358de380a79f62acc812f"
    )


@pytest.mark.parametrize("index", range(36))
def test_each_physical_record_mutation_rejected(index):
    bound = recognize_investment_adjustment_source(SOURCE)
    start, end = bound.records[index].span
    changed = SOURCE[:start] + "UNRECOGNIZED" + SOURCE[end:]
    assert recognize_investment_adjustment_source(changed) is None


@pytest.mark.parametrize(
    "old,new",
    [
        ("multiplied by 50%", "multiplied by 40%"),
        ("+ 12", "+ 10"),
        ("– 58", "– 56"),
        ("only if", "unless"),
        ("if code “RO”", "if code “RB”"),
        ("2025-12", "2026-12"),
        ("Y4N1 ZZ 89527849", "Y4N1 ZZ 89527840"),
        ("\b", ""),
        ("\n\n", "\n"),
    ],
)
def test_operative_or_physical_change_rejected(old, new):
    assert old in SOURCE
    assert recognize_investment_adjustment_source(SOURCE.replace(old, new, 1)) is None


@pytest.mark.parametrize(
    "changed", [SOURCE + "\n", " " + SOURCE, SOURCE + SOURCE, SOURCE.upper(), ""]
)
def test_no_prefix_suffix_normalization_or_duplicate_acceptance(changed):
    assert recognize_investment_adjustment_source(changed) is None
