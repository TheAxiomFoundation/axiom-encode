from __future__ import annotations

import pytest

from axiom_encode.harness.source_completeness import (
    _corroborated_form_output_label_spans,
    _mask_numeric_spans,
    _source_formula_branches,
    authoritative_numeric_recall_text,
)

# Exact contiguous T2036 publisher layout; external/form references are retained.
FORM = "Provincial or Territorial Foreign Tax Credit\nEnter the amount from line 1 of Form T2209. 1\nEnter the amount from line 3 of Form T2209, unless you have to pay minimum tax.(1) – 2\nLine 1 minus line 2 = 3\nNet foreign\nnon-business income (2) Provincial or territorial\n× = 4\nNet income (3) tax otherwise payable (4)\nEnter whichever amount is less: line 3 or line 4.\nThe amount on line 5 should not be more than the amount entered Provincial or territorial\non the line for provincial or territorial tax otherwise payable. foreign tax credit 5\nEnter the total from line 5 (for each country if applicable) on the line for the provincial or territorial foreign tax credit of\n"


def test_complete_form_masks_only_terminal_coordinates():
    spans = _corroborated_form_output_label_spans(FORM)
    assert [FORM[a:b] for a, b in spans] == ["1", "2", "3", "4", "5"]
    masked = _mask_numeric_spans(FORM, spans)
    assert len(masked) == len(FORM)
    assert "Line 1 minus line 2 =  " in masked
    assert "× =  " in masked
    assert "should not be more than" in masked
    assert "line 3 or line 4" in masked
    assert "Enter the total from line 5" in masked


@pytest.mark.parametrize(
    "prefix,suffix", [('"', '"'), ("“", "”"), ("'", "'"), ("«", "»")]
)
def test_quoted_form_is_not_admitted(prefix, suffix):
    assert not _corroborated_form_output_label_spans(prefix + "\n" + FORM + suffix)


@pytest.mark.parametrize(
    "old,new",
    [
        ("Enter the total from line 5", "Enter the total from line 6"),
        ("Line 1 minus line 2 = 3", "Line 1 minus line 2 = 2"),
        ("× = 4", "× = 3"),
        ("line 3 or line 4", "line 3 or line 6"),
        ("foreign tax credit 5", "foreign tax credit 6"),
        ("foreign tax credit 5", "foreign tax credit $5"),
        ("foreign tax credit 5", "foreign tax credit 5 CAD"),
        ("foreign tax credit 5", "foreign tax credit 5%"),
        ("foreign tax credit 5", "foreign tax credit at most 5"),
        ("foreign tax credit 5", "foreign tax credit CAD 5"),
        ("Provincial or Territorial Foreign Tax Credit\n", ""),
        ("× = 4", "× 4"),
        ("× = 4", "x = 4"),
        ("Net foreign\n", ""),
        ("Line 1 minus line 2 = 3\n", ""),
        ("\n", " "),
    ],
)
def test_incomplete_or_conflicting_layout_is_not_admitted(old, new):
    assert not _corroborated_form_output_label_spans(FORM.replace(old, new))


def test_substantive_values_remain_even_when_equal_to_labels():
    source = "Tax year 2025; rate 66.6666%; threshold $200.\n" + FORM.replace(
        "unless you have to pay minimum tax.",
        "unless you have to pay minimum tax or the amount exceeds 2, 3, 4 or 5.",
    )
    spans = _corroborated_form_output_label_spans(source)
    assert len(spans) == 5
    masked = _mask_numeric_spans(source, spans)
    assert "exceeds 2, 3, 4 or 5." in masked
    cleaned = authoritative_numeric_recall_text(source)
    assert "2025" in cleaned and "66.6666%" in cleaned and "$200" in cleaned
    assert "exceeds 2, 3, 4 or 5." in cleaned


@pytest.mark.parametrize(
    "source",
    [
        "Line 1 minus line 2 = 3",
        "Multiply by 3. Cap at 4. Pay 5 dollars.",
        "Enter the amount 2\n× = 4\ncredit 5",
        "Benefit is at most 5.",
    ],
)
def test_isolated_arithmetic_or_amounts_are_not_labels(source):
    assert not _corroborated_form_output_label_spans(source)


@pytest.mark.parametrize(
    "limit",
    ["up to", "limited to", "not exceeding", "capped at", "no more than", "at maximum"],
)
def test_matching_heading_does_not_make_limit_a_label(limit):
    source = FORM.replace(
        "Foreign Tax Credit\n", f"Foreign Tax Credit {limit}\n"
    ).replace("foreign tax credit 5", f"foreign tax credit {limit} 5")
    assert not _corroborated_form_output_label_spans(source)


def test_formula_clauses_retain_full_source_coordinate_corroboration():
    source = "Threshold $200; year 2025.\n" + FORM
    labels = set(_corroborated_form_output_label_spans(source))
    clauses = _source_formula_branches(
        source, branches=(), active_branches=(), deferred_paths=set()
    )
    attached = set()
    for clause in clauses:
        assert clause.text == source[clause.start : clause.end]
        for start, end in clause.structural_numeric_spans:
            absolute = (clause.start + start, clause.start + end)
            assert absolute in labels
            assert source[absolute[0] : absolute[1]] in {"1", "2", "3", "4", "5"}
            attached.add(absolute)
    expected = {
        (start, end)
        for start, end in labels
        if any(clause.start <= start < end <= clause.end for clause in clauses)
    }
    assert attached == expected
    assert {source[a:b] for a, b in attached} == {"1", "2", "3", "4"}
    assert "200" in source and "2025" in source


@pytest.mark.parametrize(
    "source",
    [
        FORM.replace("Enter the total from line 5", "Enter the total from line 6"),
        FORM.replace("Provincial or Territorial Foreign Tax Credit\n", ""),
        '"' + FORM + '"',
    ],
)
def test_formula_clauses_do_not_infer_labels_from_partial_layout(source):
    assert not _corroborated_form_output_label_spans(source)
    clauses = _source_formula_branches(
        source, branches=(), active_branches=(), deferred_paths=set()
    )
    assert clauses
    assert all(not clause.structural_numeric_spans for clause in clauses)
