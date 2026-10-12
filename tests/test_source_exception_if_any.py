from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc

NM_SPOUSE_SOURCE = """B. Any resident may claim a credit if the resident:
(3) compensates a caregiver for child day care for a qualifying dependent to enable such resident together with the resident's spouse, if any and if not disabled, to be gainfully employed;
"""


def _exception_obligations(source: str) -> tuple[sc.SourceStructureBranch, ...]:
    branches = tuple(
        sc.recognize_source_structure(
            source, corpus_citation_path="us-nm/statute/7-2-18.1"
        )
    )
    return sc._source_exception_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )


def test_nm_spouse_parenthetical_is_not_a_bare_exception_obligation():
    obligations = _exception_obligations(NM_SPOUSE_SOURCE)

    assert not any(ob.text.strip() == "if any and" for ob in obligations)
    spouse_clauses = [ob for ob in obligations if "if not disabled" in ob.text]
    assert len(spouse_clauses) == 1
    spouse_clause = spouse_clauses[0]
    assert "the resident's spouse, if any and if not disabled" in spouse_clause.text
    assert (
        spouse_clause.text == NM_SPOUSE_SOURCE[spouse_clause.start : spouse_clause.end]
    )
    assert sc._source_exception_selector_is_relevant(
        spouse_clause.text, "resident_has_spouse"
    )
    assert sc._source_exception_selector_is_relevant(
        spouse_clause.text, "spouse_is_disabled"
    )


@pytest.mark.parametrize("prefix", ["The credit applies ", "The credit applies, "])
@pytest.mark.parametrize(
    "condition",
    [
        "if any dependent is disabled",
        "if any and all dependents are disabled",
        "if any, but not all, dependents are disabled",
    ],
)
def test_quantified_any_conditions_remain_independent_obligations(prefix, condition):
    source = f"(1) {prefix}{condition} and if the resident is employed;"

    assert len(sc._source_exception_or_applicability_matches(source)) == 2
    obligations = _exception_obligations(source)
    assert len(obligations) == 2
    assert any(condition in ob.text for ob in obligations)
    assert all(
        sc._source_exception_requires_paired_witness(ob.text) for ob in obligations
    )


def test_standalone_parenthetical_presence_condition_is_preserved():
    source = "(1) The claimant's spouse, if any, must be gainfully employed;"

    assert len(sc._source_exception_or_applicability_matches(source)) == 1
    assert [ob.text for ob in _exception_obligations(source)] == [source]


@settings(max_examples=80, deadline=None)
@given(
    subject=st.sampled_from(["spouse", "partner", "caregiver"]),
    condition=st.sampled_from(["not disabled", "gainfully employed", "eligible"]),
    conjunction=st.sampled_from(["and", "or"]),
    cue=st.sampled_from(["if", "when", "unless"]),
    attachment=st.sampled_from([(", ", ""), (" (", ")"), (" [", "]")]),
    whitespace=st.sampled_from([" ", "\t", "\n", "  "]),
    uppercase=st.booleans(),
)
def test_parenthetical_cue_removal_preserves_the_full_real_condition(
    subject, condition, conjunction, cue, attachment, whitespace, uppercase
):
    opening, closing = attachment
    source = (
        f"(1) The claimant's {subject}{opening}if{whitespace}any{closing}"
        f"{whitespace}{conjunction}{whitespace}{cue}{whitespace}{condition}, "
        "must be gainfully employed;"
    )
    if uppercase:
        source = source.upper()

    cues = sc._source_exception_or_applicability_matches(source)
    assert len(cues) == 1
    assert cues[0].group(0).lower() == cue
    obligations = _exception_obligations(source)
    assert len(obligations) == 1
    assert obligations[0].text == source
    assert obligations[0].text == source[obligations[0].start : obligations[0].end]
    assert sc._source_exception_requires_paired_witness(obligations[0].text)


@settings(max_examples=80, deadline=None)
@given(
    subject=st.sampled_from(["dependent", "spouse", "caregiver", "child", "resident"]),
    predicate=st.sampled_from(["disabled", "not disabled", "employed", "eligible"]),
    whitespace=st.sampled_from([" ", "\t", "\n", "  "]),
    uppercase=st.booleans(),
)
def test_any_with_a_subject_predicate_always_retains_its_condition_cue(
    subject, predicate, whitespace, uppercase
):
    source = (
        f"(1) Credit applies, if{whitespace}any{whitespace}{subject} "
        f"is {predicate} and if the claimant has income;"
    )
    if uppercase:
        source = source.upper()

    cues = sc._source_exception_or_applicability_matches(source)
    assert len(cues) == 2
    obligations = _exception_obligations(source)
    assert len(obligations) == 2
    first = obligations[0]
    assert f"ANY{whitespace}{subject.upper()}" in first.text.upper()
    assert f"IS {predicate.upper()}" in first.text.upper()
    assert first.text == source[first.start : first.end]
    assert sc._source_exception_requires_paired_witness(first.text)
