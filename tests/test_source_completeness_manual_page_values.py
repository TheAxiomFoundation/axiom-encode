"""A page-shaped leading amount stays operative without complete locator proof."""

from __future__ import annotations

import functools

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

OREGON_ROOT = "us-or/manual/odhs/open"
REFERENCES = (
    "The payment is the number printed before the date.",
    "The household receives this amount.",
    "The figure above is the payment.",
    "Use the first number as the benefit amount.",
    "Eligibility depends on the value at the top of this page.",
)


def _issues(source, citation, profile, *, scalars=(7, 2026)):
    candidate = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {},
            "rules": [
                {"name": f"supplied_{index}", "versions": [{"formula": value}]}
                for index, value in enumerate(scalars)
            ],
        }
    )
    return sc.analyze_complete_source_unit(
        candidate,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    ).issues


def _requires(page, issues):
    return any(
        f"[complete-source-unit:numeric-recall] Authoritative corpus numeric "
        f"value {page} has no named scalar" in issue
        for issue in issues
    )


@pytest.mark.parametrize("profile", ("en-US", "legacy"))
@pytest.mark.parametrize("composed", (False, True), ids=("raw", "composed"))
@pytest.mark.parametrize("root", (False, True), ids=("page-citation", "root-citation"))
def test_initial_page_review_probe_demands_the_omitted_payment(profile, composed, root):
    source = "93 (07/2026) Chapter 2:Eligibility\n" + REFERENCES[0]
    if composed:
        source = "Page 93\n\n" + source
    citation = OREGON_ROOT if root else OREGON_ROOT + "/page-93"
    issues = _issues(source, citation, profile)
    assert len(issues) == 1
    assert _requires(93, issues)
    # Naming the operative amount resolves the public gate's diagnostic.
    assert _issues(source, citation, profile, scalars=(7, 2026, 93)) == ()


@pytest.mark.parametrize("profile", ("en-US", "legacy"))
@pytest.mark.parametrize("reference", REFERENCES[1:] + ("Applications are reviewed.",))
@pytest.mark.parametrize(
    "header", ("93 (07/2026) Chapter 2:Eligibility", "93 Chapter 2:Eligibility")
)
def test_initial_page_unknown_or_referential_body_fails_closed(
    profile, reference, header
):
    source = header + "\n" + reference
    assert _requires(93, _issues(source, OREGON_ROOT + "/page-93", profile))
    masked = sc._without_us_manual_locators(
        source, corpus_citation_path=OREGON_ROOT + "/page-93"
    )
    assert masked.startswith("93 ")
    assert masked.endswith(reference)
    assert len(masked) == len(source)
    assert (
        sc._without_us_manual_locators(
            masked, corpus_citation_path=OREGON_ROOT + "/page-93"
        )
        == masked
    )


@pytest.mark.parametrize("profile", ("en-US", "legacy", "da-DK"))
@pytest.mark.parametrize(
    "prefix",
    ("", '"', "\u00a0", "\u2028", "Unreviewed header\u2028", "Page 93\u2028\u2028"),
    ids=(
        "bare",
        "quoted",
        "nbsp",
        "unicode-line",
        "interior-line",
        "unicode-composition",
    ),
)
@pytest.mark.parametrize(
    "header",
    (
        "93\n(07/2026) Chapter 2:Eligibility",
        "93\nChapter 2:Eligibility",
        "93 CHAPTER 2",
    ),
)
def test_generic_structural_cleanup_cannot_hide_a_retained_initial_value(
    profile, prefix, header
):
    source = prefix + header + "\n" + REFERENCES[0]
    assert _requires(93, _issues(source, OREGON_ROOT + "/page-93", profile))
    amounts = [
        item
        for item in extract_typed_numeric_inventory_occurrences_from_text(
            source, profile=profile
        )
        if item.value == 93
    ]
    assert (len(prefix), len(prefix) + 2) in {
        (item.start, item.end) for item in amounts
    }


@pytest.mark.parametrize("quote", ('"', "'", "`", "“”"))
@pytest.mark.parametrize("profile", ("en-US", "legacy", "da-DK"))
@pytest.mark.parametrize("composed", (False, True), ids=("raw", "composed"))
@pytest.mark.parametrize("root", (False, True), ids=("page-citation", "root-citation"))
def test_referenced_paired_quoted_amount_is_required(quote, profile, composed, root):
    opening, closing = quote[0], quote[-1]
    source = f"{opening}93{closing}\n(07/2026) Chapter 2:Eligibility\n" + REFERENCES[0]
    if composed:
        source = "Page 93\n\n" + source
    citation = OREGON_ROOT if root else OREGON_ROOT + "/page-93"
    masked = sc._without_us_manual_locators(source, corpus_citation_path=citation)
    assert "93" in masked
    issues = _issues(source, citation, profile)
    assert _requires(93, issues), issues
    assert _issues(source, citation, profile, scalars=(7, 2026, 93)) == ()
    expected_start = source.index("93", source.index(opening))
    occurrences = extract_typed_numeric_inventory_occurrences_from_text(
        source, profile=profile
    )
    assert any(
        item.value == 93 and item.span == (expected_start, expected_start + 2)
        for item in occurrences
    )


@pytest.mark.parametrize("profile", ("en-US", "legacy", "da-DK"))
@settings(max_examples=80, deadline=None)
@given(
    opening=st.text(alphabet="\"'`“”‘’", min_size=1, max_size=4),
    closing=st.text(alphabet="\"'`“”‘’", min_size=1, max_size=4),
    page=st.integers(min_value=1, max_value=9999),
    composed=st.booleans(),
    root=st.booleans(),
    separator=st.sampled_from((" ", "\n", "\r\n", "\u2028")),
    stamp_first=st.booleans(),
    reference=st.sampled_from(REFERENCES),
)
@example(
    opening='"',
    closing='"',
    page=93,
    composed=False,
    root=False,
    separator="\n",
    stamp_first=True,
    reference=REFERENCES[0],
)
@example(
    opening="‘",
    closing="`",
    page=93,
    composed=True,
    root=True,
    separator="\r\n",
    stamp_first=False,
    reference=REFERENCES[0],
)
@example(
    opening="“'`",
    closing="`'”",
    page=93,
    composed=True,
    root=False,
    separator="\u2028",
    stamp_first=True,
    reference=REFERENCES[0],
)
def test_generated_quoted_referenced_initial_page_is_required(
    profile, opening, closing, page, composed, root, separator, stamp_first, reference
):
    month = 7 if page != 7 else 8
    year = 2026 if page != 2026 else 2027
    stamp = f"({month:02d}/{year})"
    header = (
        f"{stamp} Chapter 2:Eligibility"
        if stamp_first
        else f"Chapter 2:Eligibility {stamp}"
    )
    prefix = f"Page {page}\n\n" if composed else ""
    source = f"{prefix}{opening}{page}{closing}{separator}{header}\n{reference}"
    citation = OREGON_ROOT if root else f"{OREGON_ROOT}/page-{page}"
    expected_start = len(prefix) + len(opening)
    expected_span = (expected_start, expected_start + len(str(page)))
    masked = sc._without_us_manual_locators(source, corpus_citation_path=citation)
    assert len(masked) == len(source)
    assert masked[slice(*expected_span)] == str(page)
    assert _requires(page, _issues(source, citation, profile, scalars=(month, year)))
    assert _issues(source, citation, profile, scalars=(month, year, page)) == ()
    occurrences = extract_typed_numeric_inventory_occurrences_from_text(
        source, profile=profile
    )
    assert any(
        item.value == page and item.span == expected_span for item in occurrences
    )


@pytest.mark.parametrize("profile", ("en-US", "legacy"))
@settings(max_examples=60, deadline=None)
@given(
    page=st.integers(min_value=1, max_value=9999),
    reference=st.sampled_from(REFERENCES),
    composed=st.booleans(),
    root=st.booleans(),
    newline=st.sampled_from(("\n", "\r\n")),
    stamp_first=st.booleans(),
    separator=st.sampled_from((" ", "\n", "\r\n")),
    chapter=st.sampled_from(
        ("Chapter 2:Eligibility", "CHAPTER 2", "Chapter 2: Unknown")
    ),
    prefix=st.sampled_from(
        ("", "\t", "\u00a0", "\u2028", '"', "Unreviewed header\u2028")
    ),
)
@example(
    page=93,
    reference=REFERENCES[0],
    composed=False,
    root=False,
    newline="\n",
    stamp_first=True,
    separator=" ",
    chapter="Chapter 2:Eligibility",
    prefix="",
)
def test_generated_referenced_initial_page_is_required(
    profile,
    page,
    reference,
    composed,
    root,
    newline,
    stamp_first,
    separator,
    chapter,
    prefix,
):
    # Supply every other scalar while keeping the leading amount distinct.
    month = 7 if page != 7 else 8
    year = 2026 if page != 2026 else 2027
    stamp = f"({month:02d}/{year})"
    header = (
        f"{page}{separator}{stamp} {chapter}"
        if stamp_first
        else f"{page}{separator}{chapter} {stamp}"
    )
    source = prefix + header + newline + reference
    if composed:
        source = f"Page {page}{newline}{newline}" + source
    citation = OREGON_ROOT if root else f"{OREGON_ROOT}/page-{page}"
    assert _requires(page, _issues(source, citation, profile, scalars=(month, year)))


@pytest.mark.parametrize("profile", ("en-US", "legacy"))
@pytest.mark.parametrize("before", (False, True), ids=("later-page", "preceding-prose"))
def test_initial_page_authentication_uses_the_complete_composition(profile, before):
    header = "93 (07/2026) Chapter 2:Eligibility\nThe payment is $444."
    reference = "Use the number printed before the date as the additional payment."
    source = (
        reference + "\n\nPage 93\n\n" + header
        if before
        else header + "\n\nPage 94\n\n" + reference
    )
    assert _requires(93, _issues(source, OREGON_ROOT, profile, scalars=(7, 2026, 444)))


@pytest.mark.parametrize("profile", ("en-US", "legacy"))
@pytest.mark.parametrize(
    "source",
    (
        "93 (07/2026) Chapter 2:Eligibility",
        "93 (07/2026) Chapter 2:Eligibility\nThe payment is $444.",
        "26 Chapter 1: Introduction to the Oregon Programs Eligibility Notebook "
        "• Table of contents (07/2026) Introduction to the manual",
    ),
)
def test_initial_page_authenticated_complete_context_remains_excluded(profile, source):
    assert _issues(source, OREGON_ROOT, profile, scalars=(444,)) == ()
