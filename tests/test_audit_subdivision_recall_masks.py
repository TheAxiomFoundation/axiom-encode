"""The audit independently rejects operative numeric removals."""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness
from scripts import audit_subdivision_recall_masks as audit
from scripts import reproduce_subdivision_recall_audit as reproduce
from scripts.audit_subdivision_recall_masks import (
    entry_removals,
    independent_identifier_context,
    project_recall_removals,
    require_inventory_projections,
)


@pytest.fixture
def pinned_baseline(tmp_path):
    source = subprocess.run(
        ["git", "show", f"{reproduce.BASELINE_COMMIT}:{reproduce.SOURCE.as_posix()}"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout
    assert hashlib.sha256(source).hexdigest() == reproduce.BASELINE_SHA256
    baseline = tmp_path / "baseline.py"
    baseline.write_bytes(source)
    return baseline


def test_pinned_baseline_isolates_current_entry_masks(pinned_baseline):
    masks, statements = audit.isolate_entry_masks(pinned_baseline)
    source = (
        "Under subdivision 2 of this section, the credit is $300. "
        "The 48 contiguous states have 48 eligible children."
    )
    assert masks(source) == source.replace("subdivision 2", "subdivision  ").replace(
        "48 contiguous", "   contiguous"
    )
    assert len(statements) == 2
    assert all("_without_us_manual_locators" not in item for item in statements)


@pytest.mark.parametrize("change", ("helper", "prelude", "suffix"))
def test_pinned_baseline_rejects_unexplained_recall_changes(
    pinned_baseline, tmp_path, monkeypatch, change
):
    tree = ast.parse(Path(completeness.__file__).read_text())
    name = (
        "_mask_numeric_spans"
        if change == "helper"
        else "authoritative_numeric_recall_text"
    )
    function = next(node for node in tree.body if getattr(node, "name", None) == name)
    if change == "helper":
        # Both the inherited cleanup and the new entry masks reach this helper.
        function.body.insert(0, ast.Return(value=ast.Name(id="text", ctx=ast.Load())))
        expected = "existing definition outside recall cleanup changed"
    elif change == "prelude":
        function.body[1].value.args[0] = ast.Constant(value="")
        expected = "Baseline recall cleanup"
    else:
        function.body[-1] = ast.Return(value=ast.Constant(value=""))
        expected = "Baseline recall cleanup"
    mutated = tmp_path / "mutated_current.py"
    mutated.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    monkeypatch.setattr(completeness, "__file__", str(mutated))
    with pytest.raises(ValueError, match=expected):
        audit.isolate_entry_masks(pinned_baseline)


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 2 or 20 or 30 percent of income.",
        "Under subdivision 3 or more eligible children qualify.",
        "Under subdivision 20 of every 100 applicants qualify.",
        "Under subdivision 3–5 of the taxpayer’s children qualify.",
        "Under subdivision 20.5 percent of income.",
        "The benefit is $48 contiguous states.",
        "The benefit is $ 48 contiguous states.",
        "The benefit is $\n48 contiguous states.",
        "The benefit is $\r\n48 contiguous states.",
        "The benefit is ₩48 contiguous states.",
        "The benefit is USD 48 contiguous states.",
        "The benefit is usd 48 contiguous states.",
        "The benefit is yuan 48 contiguous states.",
        "A heading\n48 contiguous states.",
        "For 48 contiguous states.",
        "The benefit is 148 contiguous states.",
        "The benefit is a48 contiguous states.",
        "The benefit is 48 contiguous stateside offices.",
    ),
)
def test_audit_does_not_classify_operative_values_as_identifiers(source):
    for match in re.finditer(r"\d+", source):
        assert independent_identifier_context(source, *match.span()) is None


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 2, 20, 30 percent of income.",
        "Under subdivisions 2, 20 and 30 children qualify.",
    ),
)
def test_intermediate_commas_do_not_prove_coordinated_quantities_are_locators(source):
    for match in re.finditer(r"\d+", source):
        assert independent_identifier_context(source, *match.span()) is None


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 2, 20, or 30 percent of income, the credit is $300.",
        "A land subdivision 3, or more acres, qualifies.",
    ),
)
def test_comma_fallback_cannot_prove_an_ambiguous_quantity_is_a_locator(source):
    for match in re.finditer(r"\d+", source):
        assert independent_identifier_context(source, *match.span()) is None


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 2, 3 or more eligible children must qualify.",
        "Under subdivision 2, 20 of every 100 applicants qualify.",
        "Under subdivision 2, 3–5 of the taxpayer’s children must qualify.",
        "Under subdivision 2, the following applies: 20 percent of income.",
    ),
)
def test_a_complete_quantity_clause_independently_proves_the_locator_boundary(source):
    for match in re.finditer(r"\d+", source):
        if match.start() == source.index("2"):
            assert independent_identifier_context(source, *match.span()) is not None
        else:
            assert independent_identifier_context(source, *match.span()) is None


def test_a_quoted_term_definition_independently_proves_the_locator_boundary():
    source = (
        'For purposes of subdivision 15, "business interest" means the same as '
        "that term is defined under § 163(j) of the Internal Revenue Code."
    )
    start = source.index("15")
    assert independent_identifier_context(source, start, start + 2) is not None


@pytest.mark.parametrize("term", ('"20 percent"', "“20 percent”", "'20 percent'"))
def test_a_numeric_quoted_term_cannot_prove_a_locator_boundary(term):
    source = f"Under subdivision 2, {term} means the applicable rate."
    start = source.index("2")
    assert independent_identifier_context(source, start, start + 1) is None


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 20 of this section’s 100 applicants qualify.",
        "Under subdivision 20 of this section's 100 applicants qualify.",
        "Under subdivision 20 of section 100 applicants qualify.",
        "Under subdivision 3m, the credit applies.",
        "Under subdivision 3 m of this section, the credit applies.",
        "Under subdivision 20 is greater than the threshold.",
        "A land subdivision 3 m is required.",
        "A land subdivision 3m.",
        "Under subdivisions 2, 20 and 30 children qualify.",
        "Under subdivision 20 hereof’s 100 applicants qualify.",
        "Under subdivision 20 for the same taxable year’s 100 applicants qualify.",
        "Under subdivision 3–5.",
        "Under subdivisions 3–5.",
    ),
)
def test_quantity_contexts_cannot_complete_the_audit_locator_proof(source):
    for match in re.finditer(r"\d+", source):
        assert independent_identifier_context(source, *match.span()) is None


def test_possessive_object_does_not_prove_a_bare_coordinated_value_is_a_locator():
    source = "Under subdivision 2, 20 of this section's 100 applicants qualify."
    for match in re.finditer(r"\d+", source):
        if match.group() != "2":
            assert independent_identifier_context(source, *match.span()) is None


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 2, the credit is $300.",
        "Under subdivision B 3 of this section.",
        "Under subdivisions 2, 3, and 5 of this section.",
        "Under subdivision 1 or 2 for the same taxable year.",
        "Under subdivision 2(b) of this section.",
        "Under subdivision 2(b).",
        "Under subdivision B3 is required for qualifying taxpayers.",
        "Under subdivisions 3 through 5 of this section.",
        "The guidelines for the 48 contiguous states apply.",
        "48 contiguous states.",
        "\n 48 contiguous states.",
    ),
)
def test_audit_classifies_complete_identifiers(source):
    for match in re.finditer(r"\d+", source):
        if match.group() == "300":
            continue
        assert independent_identifier_context(source, *match.span()) is not None


def test_entry_removal_audit_preserves_offsets_and_nonnumeric_text():
    source = "Under subdivision 2, the following applies: 20 percent."
    masked = "Under subdivision  , the following applies: 20 percent."
    removals = entry_removals(source, masked)
    assert len(removals) == 1
    assert removals[0]["raw"] == "2"
    assert removals[0]["kind"] == "legal-locator"
    with pytest.raises(ValueError, match="offsets"):
        entry_removals(source, masked[:-1])
    with pytest.raises(ValueError, match="nonnumeric"):
        entry_removals(source, masked.replace("applies", "appliex"))


def test_entry_removal_audit_rejects_deletion_of_operative_rate():
    source = "Under subdivision 2 or 20.5 percent of income."
    with pytest.raises(ValueError, match="Unclassified removal"):
        entry_removals(source, source.replace("20.5", "  . "))


def _citation_projection_example():
    source = (
        "Under § 58.1-1, the credit under subdivision 8 of § 58.1-322.02; "
        "8 eligible children qualify for a $300 credit."
    )
    start = source.index("8 of")
    masked = source[:start] + " " + source[start + 1 :]

    def inherited_citation_cleanup(text, *, corpus_citation_path):
        return completeness._GERMAN_LEGAL_CITATION.sub("", text)

    baseline = SimpleNamespace(
        authoritative_numeric_recall_text=inherited_citation_cleanup
    )
    before = inherited_citation_cleanup(source, corpus_citation_path="")
    after = inherited_citation_cleanup(masked, corpus_citation_path="")
    return source, baseline, before, after, entry_removals(source, masked)


def test_removed_section_object_keeps_exact_original_identifier_proof():
    source, baseline, before, after, removals = _citation_projection_example()
    start = before.index("8 of")
    assert independent_identifier_context(before, start, start + 1) is None
    projections = project_recall_removals(baseline, source, "", before, after, removals)
    assert len(projections) == 1
    assert projections[0]["original_start"] == source.index("8 of")
    assert projections[0]["start"] == start
    assert projections[0]["original_start"] != projections[0]["start"]
    require_inventory_projections(
        [{"start": start, "end": start + 1, "raw": "8"}], projections
    )


def test_equal_valued_operative_occurrence_cannot_borrow_identifier_proof():
    source, baseline, before, after, removals = _citation_projection_example()
    projections = project_recall_removals(baseline, source, "", before, after, removals)
    start = before.index("8 eligible")
    with pytest.raises(ValueError, match="exact original-span proof"):
        require_inventory_projections(
            [{"start": start, "end": start + 1, "raw": "8"}], projections
        )


def test_erasing_an_equal_valued_quantity_cannot_pass_recall_projection():
    source, baseline, before, _after, removals = _citation_projection_example()
    start = before.index("8 eligible")
    wrong_recall = before[:start] + " " + before[start + 1 :]
    with pytest.raises(ValueError, match="changes another token"):
        project_recall_removals(baseline, source, "", before, wrong_recall, removals)


@pytest.mark.parametrize(
    "connector", ("through", "Through", "THROUGH", "tHrOuGh", "to", "To", "TO", "tO")
)
def test_case_variant_ranges_still_require_a_plural_statutory_object(connector):
    source = f"Land subdivision 3 {connector} 5: minimum acreage range."
    # The uppercase THROUGH instance is the review's input verbatim.
    for match in re.finditer(r"\d+", source):
        assert independent_identifier_context(source, *match.span()) is None
    with pytest.raises(ValueError, match="Unclassified removal"):
        entry_removals(source, re.sub(r"\d", " ", source))

    statutory = f"Under subdivisions 3 {connector} 5 of this section."
    removals = entry_removals(statutory, re.sub(r"\d", " ", statutory))
    assert [(item["raw"], item["kind"]) for item in removals] == [
        ("3", "legal-locator"),
        ("5", "legal-locator"),
    ]


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 2, 3 qualifies.",
        "Under subdivision 2, 3 Qualifies.",
        "Under subdivision 2, 3 QUALIFIES.",
        "Under subdivision 2, 3 qUaLiFiEs.",
    ),
)
def test_a_bare_numeric_subject_proves_a_finite_clause_boundary(source):
    masked = source.replace("2", " ")
    removals = entry_removals(source, masked)
    assert [(item["raw"], item["kind"]) for item in removals] == [
        ("2", "legal-locator")
    ]
    subject = source.index("3")
    assert independent_identifier_context(source, subject, subject + 1) is None


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 2, 3 and 4 qualify.",
        "Under subdivision 2, 3 AND 4 QUALIFY.",
        "Under subdivision 2, 3 or 4 qualifies.",
        "Under subdivision 2, 3 OR 4 QUALIFIES.",
    ),
)
def test_bare_numeric_subjects_do_not_prove_shared_quantity_coordination(source):
    for match in re.finditer(r"\d+", source):
        assert independent_identifier_context(source, *match.span()) is None


_AUDIT_VOCABULARY_CONTROLS = (
    ("Under subdivisions 3 through 5 of this section.", ("3", "5")),
    ("Under subdivisions 3 to 5 of this section.", ("3", "5")),
    ("Under subdivisions 3−5 of this section.", ("3", "5")),
    ("Under subdivisions 2, 3 and 5 of this section.", ("2", "3", "5")),
    ("Under subdivision 1 or 2 for the same taxable year.", ("1", "2")),
    ("Under subdivision B3 is required for qualifying taxpayers.", ("3",)),
    ("Under subdivision 2, 3 qualifies.", ("2",)),
    ("Under subdivision 2, 3 or more eligible children must qualify.", ("2",)),
    ("Under subdivision 2, 3 or fewer eligible children shall qualify.", ("2",)),
    ("Under subdivision 2, 3 or less eligible children may qualify.", ("2",)),
    ("Under subdivision 2, 3 or greater eligible children can qualify.", ("2",)),
    ("Under subdivision 2, 20 of every 100 applicants qualify.", ("2",)),
    ("Under subdivision 2, 3 through 5 children are eligible.", ("2",)),
    ('Under subdivision 2, "business interest" means the limit.', ("2",)),
    ("Under subdivision 2, 'business interest' means the limit.", ("2",)),
    ("The guidelines for the 48 contiguous states apply.", ("48",)),
    ("Land subdivision 3 through 5: minimum acreage range.", ()),
    ("Land subdivision 3 to 5: minimum acreage range.", ()),
    ("Land subdivision 3−5: minimum acreage range.", ()),
    ("Under subdivision 2 or 20 or 30 percent of income.", ()),
    ("Under subdivisions 2, 20 and 30 children qualify.", ()),
    ("Under subdivision 3 m of this section, the credit applies.", ()),
    ("Under subdivision 20 of this section's 100 applicants qualify.", ()),
    ("The benefit is USD 48 contiguous states.", ()),
    ("The benefit is yuan 48 contiguous states.", ()),
    ("The benefit is $ 48 contiguous states.", ()),
)


@st.composite
def _audit_case_and_typography_sources(draw):
    source, expected = draw(st.sampled_from(_AUDIT_VOCABULARY_CONTROLS))

    def keyword_case(match):
        word = match.group()
        variant = draw(st.sampled_from(("lower", "upper", "title", "mixed")))
        if variant == "mixed":
            return "".join(
                letter.upper() if index % 2 else letter.lower()
                for index, letter in enumerate(word)
            )
        return getattr(word, variant)()

    source = re.sub(r"[A-Za-z]+", keyword_case, source)
    space = draw(st.sampled_from((" ", "\u00a0", "\u202f")))
    source = source.replace(" ", space)
    if draw(st.booleans()):
        source = source.replace('"', "“", 1).replace('"', "”")
        source = source.replace("'", "’")
    return source, expected


@settings(max_examples=80, deadline=None, derandomize=True)
@given(_audit_case_and_typography_sources())
def test_audit_classification_is_invariant_to_keyword_case_and_typography(case):
    source, expected = case
    classified = []
    for match in re.finditer(r"\d+", source):
        proof = independent_identifier_context(source, *match.span())
        if proof is not None:
            classified.append(match.group())
    assert tuple(classified) == expected
    masked = source
    for match in reversed(list(re.finditer(r"\d+", source))):
        if independent_identifier_context(source, *match.span()) is not None:
            masked = (
                masked[: match.start()]
                + " " * len(match.group())
                + masked[match.end() :]
            )
    assert tuple(item["raw"] for item in entry_removals(source, masked)) == expected


_ORIGINAL_GLUED_MARKER = re.compile(
    r"(?<![\w])(?P<label>[1-9]\d?)"
    r"(?![A-ZÄÖÜ](?:\d+)?\b)"
    r"(?!(?i:st|nd|rd|th)\b)"
    r"(?=[A-ZÄÖÜ](?!:)(?![ \t]*[.:/\-\u2010-\u2015\u2212\ufe58\ufe63\uff0d]"
    r"[ \t]*\d))"
)


@pytest.mark.parametrize(
    "unit",
    ("PERCENT", "Percent", "PER CENT", "PER\u00a0CENT", "DOLLARS", "DAYS", "PEOPLE"),
)
def test_restored_glued_quantity_has_an_independent_exact_deletion_proof(unit):
    guarded = f"Under subdivision 2{unit} of income, the credit applies."
    original = _ORIGINAL_GLUED_MARKER.sub("", guarded)
    proofs = audit.prove_glued_quantity_restorations(
        original, guarded, _ORIGINAL_GLUED_MARKER
    )
    assert [(item["raw"], item["start"], item["end"]) for item in proofs] == [
        ("2", guarded.index("2"), guarded.index("2") + 1)
    ]
    assert all(item["unit_context"] in guarded for item in proofs)


@pytest.mark.parametrize(
    "guarded", ("2GADGETS qualify.", "2PERCENT extra 3.", "2PERCENT of income; 4")
)
def test_glued_quantity_proofs_reject_unrelated_text_or_numeric_changes(guarded):
    original = _ORIGINAL_GLUED_MARKER.sub("", guarded)
    if "PERCENT" in guarded:
        original = original.replace("3", "").replace("4", "")
    with pytest.raises(ValueError, match="independent glued-unit"):
        audit.prove_glued_quantity_restorations(
            original, guarded, _ORIGINAL_GLUED_MARKER
        )


def test_marker_assignment_exception_allows_only_the_exact_unit_guard():
    pattern = _ORIGINAL_GLUED_MARKER.pattern
    old = ast.parse(f"_GLUED_SENTENCE_MARKER = re.compile({pattern!r})").body[0]
    anchor = "(?=[A-ZÄÖÜ]"
    before, after = pattern.split(anchor)
    new = ast.parse(
        "_GLUED_SENTENCE_MARKER = re.compile("
        + repr(before)
        + " + rf'(?!{_RECALL_UNIT_WORD}\\b|(?i:per[ \\t\\u00a0\\u202f]+cent)\\b)' + "
        + repr(anchor + after)
        + ")"
    ).body[0]
    # Adjacent literals produce the exact JoinedStr used by production.
    new.value.args[0] = ast.JoinedStr(
        values=[
            ast.Constant(before + "(?!"),
            ast.FormattedValue(
                value=ast.Name(id="_RECALL_UNIT_WORD", ctx=ast.Load()), conversion=-1
            ),
            ast.Constant(r"\b|(?i:per[ \t\u00a0\u202f]+cent)\b)" + anchor + after),
        ]
    )
    assert audit.exact_glued_quantity_guard_assignment(
        old, new, completeness._RECALL_UNIT_WORD
    )
    for change in ("label", "units", "flags"):
        changed = ast.parse(ast.unparse(new)).body[0]
        if change == "label":
            changed.value.args[0].values[0].value = (
                before.replace("[1-9]", "[0-9]") + "(?!"
            )
        elif change == "units":
            changed.value.args[0].values[-1].value = (
                changed.value.args[0].values[-1].value.replace("cent", "cent|gadgets")
            )
        else:
            changed.value.keywords.append(
                ast.keyword(
                    arg="flags",
                    value=ast.Attribute(
                        value=ast.Name(id="re", ctx=ast.Load()),
                        attr="IGNORECASE",
                        ctx=ast.Load(),
                    ),
                )
            )
        assert not audit.exact_glued_quantity_guard_assignment(
            old, changed, completeness._RECALL_UNIT_WORD
        )


def test_added_inventory_requires_the_exact_retained_quantity_occurrence():
    guarded = "2PERCENT of income; 2 children qualify."
    original = _ORIGINAL_GLUED_MARKER.sub("", guarded)
    proofs = audit.prove_glued_quantity_restorations(
        original, guarded, _ORIGINAL_GLUED_MARKER
    )
    audit.require_quantity_addition_proofs([{"start": 0, "end": 1, "raw": "2"}], proofs)
    other = guarded.index("2 children")
    with pytest.raises(ValueError, match="exact retained quantity"):
        audit.require_quantity_addition_proofs(
            [{"start": other, "end": other + 1, "raw": "2"}], proofs
        )


@settings(max_examples=40, deadline=None, derandomize=True)
@given(
    number=st.integers(min_value=1, max_value=99),
    unit=st.sampled_from(("percent", "dollars", "days", "per cent", "per\u202fcent")),
    case_style=st.sampled_from(("lower", "upper", "title", "mixed")),
)
def test_glued_quantity_deletion_proofs_preserve_duplicate_quantities(
    number, unit, case_style
):
    if case_style == "mixed":
        unit = "".join(
            letter.upper() if index % 2 else letter.lower()
            for index, letter in enumerate(unit)
        )
    else:
        unit = getattr(unit, case_style)()
    guarded = f"{number}{unit} of income; {number} children qualify."
    original = _ORIGINAL_GLUED_MARKER.sub("", guarded)
    proofs = audit.prove_glued_quantity_restorations(
        original, guarded, _ORIGINAL_GLUED_MARKER
    )
    expected = int(original != guarded)
    assert len(proofs) == expected
    if proofs:
        assert (proofs[0]["start"], proofs[0]["end"], proofs[0]["raw"]) == (
            0,
            len(str(number)),
            str(number),
        )


def test_typed_delta_composition_handles_equal_valued_cancellations():
    quantity = {"value": 2.0, "raw": "2", "start": 0, "end": 1}
    locator = quantity | {"start": 20, "end": 21}
    empty = {"removed": [], "added": []}
    guard = {"removed": [], "added": [quantity]}
    entry = {"removed": [locator], "added": []}
    audit.require_delta_composition(empty, guard, entry)
    with pytest.raises(ValueError, match="independently proved deltas"):
        audit.require_delta_composition(guard, guard, entry)


def test_selected_row_bounds_are_explicit_and_do_not_change_signed_artifact_counts():
    artifact = SimpleNamespace(
        path="data/corpus/provisions/us/manual/input.jsonl", row_count=5
    )
    selected = audit.select_row_ranges(
        (artifact,), artifact_start=7, row_start=1, row_limit=2
    )
    assert selected == [
        {
            "artifact_index": 7,
            "path": artifact.path,
            "row_start": 1,
            "row_end": 3,
            "artifact_row_count": 5,
            "selected_row_count": 2,
        }
    ]
    whole = audit.select_row_ranges((artifact,), artifact_start=7)
    assert (
        whole[0]["row_start"],
        whole[0]["row_end"],
        whole[0]["selected_row_count"],
    ) == (0, 5, 5)


@pytest.mark.parametrize(
    "row_start,row_limit", ((-1, None), (5, None), (0, 0), (0, -1), (0, 6), (4, 2))
)
def test_row_selection_rejects_invalid_or_empty_bounds(row_start, row_limit):
    artifact = SimpleNamespace(path="input.jsonl", row_count=5)
    with pytest.raises(ValueError, match="row selection"):
        audit.select_row_ranges(
            (artifact,), artifact_start=0, row_start=row_start, row_limit=row_limit
        )


@pytest.mark.parametrize("row_start,row_limit", ((1, None), (0, 2)))
def test_row_selection_requires_exactly_one_artifact(row_start, row_limit):
    artifacts = (
        SimpleNamespace(path="first.jsonl", row_count=5),
        SimpleNamespace(path="second.jsonl", row_count=4),
    )
    with pytest.raises(ValueError, match="one artifact"):
        audit.select_row_ranges(
            artifacts, artifact_start=0, row_start=row_start, row_limit=row_limit
        )


def test_selected_slice_verifies_all_physical_artifact_rows(tmp_path):
    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text("{}\n{}\n{}\n")
    assert audit.verify_artifact_row_count(artifact, 3) == 3
    # Rows beyond a requested slice must still count toward the signed total.
    with pytest.raises(ValueError, match="row count"):
        audit.verify_artifact_row_count(artifact, 2)


def _coverage_range(index, start, end, total):
    return {
        "artifact_index": index,
        "row_start": start,
        "row_end": end,
        "artifact_row_count": total,
        "selected_row_count": end - start,
    }


def test_row_slices_cover_all_signed_rows_exactly_once():
    coverage = [
        _coverage_range(0, 0, 2, 5),
        _coverage_range(0, 2, 5, 5),
        _coverage_range(1, 0, 4, 4),
    ]
    audit.validate_row_range_coverage(coverage, [5, 4])


@pytest.mark.parametrize(
    "coverage",
    (
        [_coverage_range(0, 0, 2, 5), _coverage_range(0, 3, 5, 5)],
        [_coverage_range(0, 0, 3, 5), _coverage_range(0, 2, 5, 5)],
        [_coverage_range(0, 0, 4, 5)],
        [_coverage_range(0, 0, 6, 5)],
        [_coverage_range(1, 0, 5, 5)],
    ),
)
def test_combined_row_coverage_rejects_gaps_overlaps_or_incomplete_bounds(coverage):
    with pytest.raises(ValueError, match="coverage"):
        audit.validate_row_range_coverage(coverage, [5])
