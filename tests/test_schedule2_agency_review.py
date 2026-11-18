"""Embedded agency-review questions retain source coverage, not false gates."""

from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as s

BODY = (Path(__file__).parent / "fixtures/detached_schedule2_source.txt").read_text()
TEXT = BODY[2647:2818]


@pytest.mark.parametrize("text", [TEXT, " ".join(TEXT.split())])
def test_review_question_is_not_a_paired_claimant_condition(text):
    assert s._source_is_agency_review_interrogative(text)
    assert not s._source_exception_requires_paired_witness(text)


@pytest.mark.parametrize(
    "text",
    [
        "If the CRA finds your spouse ineligible, you cannot claim the credit.",
        "You may claim the credit only if the CRA approves your claim.",
        "The CRA will review your claim if you request a review.",
        "Before assessing your return, the CRA will review your claim only if your spouse is eligible.",
        TEXT + " If the CRA asks you to, send a new Form T2201.",
        TEXT + " You must attach a completed form if filing on paper.",
        TEXT[:-1] + "; if approval ends, send a new form.",
        "The CRA will review your claim to see if your spouse is eligible and will deny your claim if not.",
    ],
)
def test_real_conditions_and_adjacent_duties_remain_paired(text):
    assert not s._source_is_agency_review_interrogative(text)
    assert s._source_exception_requires_paired_witness(text)


def test_source_extraction_preserves_question_and_adjacent_obligations():
    branches = s.recognize_source_structure(
        BODY,
        corpus_citation_path="ca/policy/cra/t1-2025/federal-spouse-transfer-amounts",
    )
    clauses = s._source_exception_branches(
        BODY,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        corpus_citation_path="ca/policy/cra/t1-2025/federal-spouse-transfer-amounts",
    )
    question = next(b for b in clauses if (b.start, b.end) == (2647, 2818))
    assert question.text == TEXT
    assert not s._source_exception_requires_paired_witness(question.text)
    assert any(
        "if the CRA asks you to" in b.text
        and s._source_exception_requires_paired_witness(b.text)
        for b in clauses
    )
    assert s._source_exception_requires_paired_witness("if the CRA asks you to.")
