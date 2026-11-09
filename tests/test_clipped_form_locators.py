"""Complete-body locator evidence survives clause clipping, never amounts."""

from pathlib import Path

import pytest

from axiom_encode.form_numeric_structure import form_numeric_locator_spans
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_occurrences_from_text,
)

P = Path(__file__).parent / "fixtures/form-numeric-locators/step-2.txt"


def branches(text):
    return s._source_formula_branches(
        text,
        corpus_citation_path="ca/policy/test",
        branches=(),
        active_branches=(),
        deferred_paths=set(),
    )


def occurrences(branch):
    return s._formula_branch_computation_occurrences(
        branch,
        interval=None,
        extract_numeric_occurrences=extract_typed_numeric_occurrences_from_text,
    )


def test_complete_body_referral_reaches_clipped_computation():
    text = P.read_text()
    branch = next(b for b in branches(text) if b.text.startswith("Line 32"))
    assert text[branch.start : branch.end] == branch.text
    position = branch.text.index("Step 4") + len("Step ")
    assert (position, position + 1) in branch.structural_numeric_spans
    assert not occurrences(branch)
    assert not form_numeric_locator_spans(branch.text)
    clipped = s.SourceStructureBranch(
        (), "formula-clause", "clipped", branch.text, 0, len(branch.text)
    )
    assert any(x.value == 4 for x in occurrences(clipped))


@pytest.mark.parametrize("value", ["4 dollars", "4 percent", "4 days", "4 + amount"])
def test_substantive_four_never_becomes_structural(value):
    text = P.read_text().replace(
        "Add lines 29 to 31 (see line 25000 in Step 4)", "Income + " + value
    )
    found = [b for b in branches(text) if b.text.startswith("Line 32")]
    assert found
    assert any(x.source_value == 4 for b in found for x in occurrences(b))
    for b in found:
        for a, z in b.structural_numeric_spans:
            assert b.text[a:z] != "4"


@pytest.mark.parametrize(
    "ref",
    [
        "(see line 25000 in Step 4",
        "((see line 25000 in Step 4))",
        "(discussion (see line 25000 in Step 4)",
        '"(see line 25000 in Step 4)"',
        "(see line 25000 in Step 4 days)",
    ],
)
def test_incomplete_or_quoted_referral_receives_no_new_span(ref):
    text = P.read_text().replace("(see line 25000 in Step 4)", ref)
    position = text.index(ref) + ref.index("Step 4") + len("Step ")
    assert not any(a <= position < z for a, z in form_numeric_locator_spans(text))
    for b in branches(text):
        if b.start <= position < b.end:
            assert not any(
                a <= position - b.start < z for a, z in b.structural_numeric_spans
            )


def test_other_body_cannot_reuse_complete_grid_spans():
    text = P.read_text().replace("Step 2 - Total income", "Unknown heading")
    position = text.index("Step 4") + len("Step ")
    assert not any(a <= position < z for a, z in form_numeric_locator_spans(text))
    for b in branches(text):
        if b.start <= position < b.end:
            assert not any(
                a <= position - b.start < z for a, z in b.structural_numeric_spans
            )


def test_retained_piece_projects_only_fully_contained_source_spans():
    text = P.read_text()
    labels = form_numeric_locator_spans(text)
    retained = s._retain_bare_punctuation_computations(
        text,
        branches=(),
        obligations=(),
        deferred_intervals=(),
        worksheet_labels=labels,
    )
    assert retained
    for b in retained:
        assert b.text == text[b.start : b.end]
        assert b.structural_numeric_spans == tuple(
            (a - b.start, z - b.start) for a, z in labels if b.start <= a < z <= b.end
        )
