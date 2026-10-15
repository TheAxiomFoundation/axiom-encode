"""Revision metadata cannot authorize hiding a date used by body prose.

The cover fixture is the complete pinned Oregon OPEN page-1 body. The
resolver adds its real ``Page 1`` heading when composing the bodyless book.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.corpus_resolver import resolve_local_corpus_source
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.manual_locator_metadata import OREGON_OPEN_REVISION_CAPTIONS
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.release_object_fixtures import bind_test_corpus_release

OREGON_ROOT = "us-or/manual/odhs/open"
OREGON_PAGE = f"{OREGON_ROOT}/page-93"
OREGON_CAPTION = (
    "93 Chapter 4:Working with families • Section 10: Support service payments"
)
OREGON_COVER_PREFIX = "OREGON PROGRAMS ELIGIBILITY NOTEBOOK (OPEN) 100-13450_DHS 2818"
WISCONSIN_PREFIX = (
    "WISCONSIN DEPARTMENT OF HEALTH SERVICES Division of Medicaid Services P-16001"
)
VIRGINIA_PREFIX = "032-03-0445-01"
EMPTY_RULES = "format: rulespec/v1\nmodule: {}\nrules: []\n"
RECALL_VALUE = re.compile(
    r"\[complete-source-unit:numeric-recall\] Authoritative corpus numeric "
    r"value (\S+) has no named scalar"
)


def _recalled_values(
    source: str, citation: str, *, profile: str = "en-US"
) -> set[float]:
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=citation
    )
    return {
        occurrence.value
        for occurrence in extract_typed_numeric_inventory_occurrences_from_text(
            cleaned, profile=profile
        )
    }


def _missing_values(
    source: str, citation: str, *, profile: str = "en-US"
) -> set[float]:
    result = sc.analyze_complete_source_unit(
        EMPTY_RULES,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return {
        float(match.group(1))
        for issue in result.issues
        if (match := RECALL_VALUE.search(issue)) is not None
    }


@pytest.mark.parametrize(
    ("source", "citation"),
    (
        (
            f"{OREGON_CAPTION}\n(10/2025) is the qualifying application month.",
            OREGON_PAGE,
        ),
        (
            f"{OREGON_CAPTION} (10/2025) is the qualifying application month.",
            OREGON_PAGE,
        ),
        (
            f"{OREGON_CAPTION}\n(10/2025)\nis the qualifying application month.",
            OREGON_PAGE,
        ),
        (
            f"{OREGON_CAPTION}\n(10/2025), applications qualify for assistance.",
            OREGON_PAGE,
        ),
        (
            f"{OREGON_CAPTION}\n(10/2025) provides the qualifying application month.",
            OREGON_PAGE,
        ),
        (
            f"{OREGON_COVER_PREFIX} (10/2025) is the qualifying application month.",
            f"{OREGON_ROOT}/page-1",
        ),
        (
            f"{WISCONSIN_PREFIX} (10/2025) is the qualifying application month.",
            "us-wi/manual/dhs/foodshare/page-1",
        ),
        (
            f"{WISCONSIN_PREFIX} (10/2025)\nif an application is received, pay it.",
            "us-wi/manual/dhs/foodshare/page-1",
        ),
        (
            f"{VIRGINIA_PREFIX} (10/2025)\nis the qualifying application month.",
            "us-va/manual/dss/snap/page-1",
        ),
        (
            f"{VIRGINIA_PREFIX} (10/2025)\nif an application is received, pay it.",
            "us-va/manual/dss/snap/page-1",
        ),
        (
            f"{VIRGINIA_PREFIX} (10/2025)\n"
            "Applicant Name must be provided for eligibility.",
            "us-va/manual/dss/snap/page-1",
        ),
        (
            f"{WISCONSIN_PREFIX} (10/2025)\n"
            "FoodShare Handbook Release determines the qualifying month.",
            "us-wi/manual/dhs/foodshare/page-1",
        ),
        (
            f"{OREGON_CAPTION}\n(10/2025) Introduction to the manual\n"
            "is required for eligibility.",
            OREGON_PAGE,
        ),
        (
            f"{VIRGINIA_PREFIX} (10/2025) Applicant Name\nis required for eligibility.",
            "us-va/manual/dss/snap/page-1",
        ),
        (
            f"Applications filed before ____________________ {VIRGINIA_PREFIX} "
            "(10/2025)",
            "us-va/manual/dss/snap/page-1",
        ),
        (
            "93 (10/2025) OREGON PROGRAMS ELIGIBILITY NOTEBOOK "
            "governs eligibility for assistance.",
            OREGON_PAGE,
        ),
        (
            "93 (10/2025) Chapter 4: Support Service Payments "
            "is the qualifying application month.",
            OREGON_PAGE,
        ),
        (
            f"Page 93\n\n{OREGON_CAPTION}\n(10/2025)\n\n"
            "Page 94\n\nis the qualifying application month.",
            OREGON_ROOT,
        ),
        (
            f"Page 1\n\n{OREGON_COVER_PREFIX} (10/2025)\n\n"
            "Page 2\n\nif an application is received, pay it.",
            OREGON_ROOT,
        ),
        (
            "93 (10/2025) Chapter 4: Working With Families Determines Eligibility",
            OREGON_PAGE,
        ),
        (
            f"Page 1\n\n{VIRGINIA_PREFIX} (10/2025)\n\n"
            f"Page 2\n\nis the qualifying application month.\n{VIRGINIA_PREFIX} (07/2026)",
            "us-va/manual/dss/snap/page-1",
        ),
        (
            f"Page 93\n\n{OREGON_CAPTION}\n(10/2025)\n\n"
            "Page 94\n\n94 (07/2026) Chapter 2:Eligibility\n"
            "Section 1: Application\nis the qualifying application month.",
            OREGON_ROOT,
        ),
        (
            f"Page 1\n\n{VIRGINIA_PREFIX} (10/2025)\n\n"
            "Page 2\n\n94 (07/2026) Chapter 2:Eligibility\n"
            "Section 1: Application\nis the qualifying application month.\n"
            f"{VIRGINIA_PREFIX} (08/2026)",
            "us-va/manual/dss/snap/page-1",
        ),
        (
            "93 (10/2025) Chapter 2:Eligibility\n"
            "Section 1: Application is the qualifying application month.",
            OREGON_PAGE,
        ),
        (
            "93 (10/2025) Chapter 2:Eligibility\n"
            "Section 1: Application\nis the qualifying application month.",
            OREGON_PAGE,
        ),
    ),
    ids=(
        "oregon-review-probe",
        "oregon-same-line-sentence",
        "oregon-line-broken-sentence",
        "oregon-condition",
        "oregon-unlisted-predicate",
        "oregon-cover-body-date",
        "wisconsin-body-date",
        "wisconsin-line-broken-condition",
        "virginia-line-broken-sentence",
        "virginia-line-broken-condition",
        "virginia-metadata-label-prefix-body-clause",
        "wisconsin-metadata-label-prefix-body-clause",
        "oregon-line-broken-metadata-label-body-clause",
        "virginia-line-broken-metadata-label-body-clause",
        "virginia-body-prefix-before-underscored-form-identifier",
        "oregon-initial-stamp-before-notebook-body-suffix",
        "oregon-initial-stamp-before-chapter-body-suffix",
        "oregon-caption-date-sentence-crosses-composed-page",
        "oregon-cover-date-condition-crosses-composed-page",
        "oregon-initial-stamp-before-titlecased-body-predicate",
        "virginia-successor-footer-cannot-authenticate-previous-body-date",
        "oregon-next-page-header-interrupts-body-date-sentence",
        "virginia-successor-header-and-footer-enclose-body-date-continuation",
        "oregon-section-prefix-before-same-line-body-sentence",
        "oregon-section-prefix-before-line-broken-body-sentence",
    ),
)
def test_metadata_prefix_cannot_hide_a_body_date_or_its_diagnostics(source, citation):
    assert {10, 2025} <= _recalled_values(source, citation)
    assert {10, 2025} <= _missing_values(source, citation)


@pytest.mark.parametrize(
    "later_pages",
    (
        "",
        "\n\nPage 94\n\nThe month shown above determines eligibility.",
        "\n\nPage 94\n\nApplications are reviewed.\n\n"
        "Page 95\n\nThe month shown above determines eligibility.",
    ),
    ids=("single-page-positive-control", "next-page-reference", "later-page-reference"),
)
def test_composed_amount_sentence_cannot_hide_a_revision_month_used_later(later_pages):
    source = (
        "Page 93\n\n93 (07/2026) Chapter 2: Eligibility\n"
        f"The payment is $93 and $2026.{later_pages}"
    )
    expected = {7, 93, 2026} if later_pages else {93, 2026}
    assert _recalled_values(source, OREGON_ROOT) == expected
    assert _missing_values(source, OREGON_ROOT) == expected


@pytest.mark.parametrize(
    "source",
    (
        "Page 93\n\n93 (07/2026) Chapter 2:Eligibility\n"
        "Section 1: Application\n1.1.1 Application Process\n\n"
        "Page 94\n\nThe month shown above determines eligibility.",
        f"Page 93\n\n{OREGON_CAPTION}\n(07/2026)\n\n"
        f"Page 94\n\n{OREGON_COVER_PREFIX} (08/2026)\n\n"
        "Page 95\n\nThe month shown above determines eligibility.",
        "Page 93\n\n93 (07/2026) Chapter 2:Eligibility\n"
        "Section 1: Application\n1.1.1 Application Process\n"
        "The month shown above determines eligibility.",
    ),
    ids=(
        "authenticated-section-heading",
        "metadata-only-successor-before-body",
        "same-page-section-heading-reference",
    ),
)
def test_composed_header_authentication_cannot_ignore_later_body_prose(source):
    assert {7, 2026} <= _recalled_values(source, OREGON_ROOT)
    assert {7, 2026} <= _missing_values(source, OREGON_ROOT)
    assert "(07/2026)" in sc._without_us_manual_locators(
        source, corpus_citation_path=OREGON_ROOT
    )


@pytest.mark.parametrize("citation", (OREGON_ROOT, OREGON_PAGE), ids=("root", "page"))
@pytest.mark.parametrize("composed", (False, True), ids=("raw", "composed"))
@pytest.mark.parametrize("profile", ("en-US", "legacy"))
def test_section_heading_cannot_hide_revision_date_used_by_same_page_prose(
    citation, composed, profile
):
    source = (
        "93 (07/2026) Chapter 2:Eligibility\n"
        "Section 1: Application\n1.1.1 Application Process\n"
        "The month shown above determines eligibility."
    )
    if composed:
        source = "Page 93\n\n" + source
    recalled = _recalled_values(source, citation, profile=profile)
    demanded = _missing_values(source, citation, profile=profile)
    assert {7, 2026} <= recalled
    assert {7, 2026} <= demanded
    assert sc._manual_revision_stamp_spans(source, corpus_citation_path=citation) == ()
    masked = sc._without_us_manual_locators(source, corpus_citation_path=citation)
    assert "(07/2026)" in masked
    assert len(masked) == len(source)
    assert (
        sc._without_us_manual_locators(masked, corpus_citation_path=citation) == masked
    )


@pytest.mark.parametrize("citation", (OREGON_ROOT, OREGON_PAGE), ids=("root", "page"))
@pytest.mark.parametrize("composed", (False, True), ids=("raw", "composed"))
@pytest.mark.parametrize("profile", ("en-US", "legacy"))
def test_independent_facts_before_reference_cannot_hide_a_revision_date(
    citation, composed, profile
):
    source = (
        "93 (07/2026) Chapter 2:Eligibility\n"
        "Section 1: Application\n1.1.1 Application Process\n"
        "Households receive $444 and must apply within 30 days.\n"
        "Applications filed in (11/2027) are eligible.\n"
        "The month shown above determines eligibility."
    )
    if composed:
        source = "Page 93\n\n" + source
    recalled = _recalled_values(source, citation, profile=profile)
    demanded = _missing_values(source, citation, profile=profile)
    assert {7, 2026, 444, 30, 11, 2027} <= recalled
    assert {7, 2026, 444, 30, 11, 2027} <= demanded
    assert "(07/2026)" in sc._without_us_manual_locators(
        source, corpus_citation_path=citation
    )


@settings(max_examples=60, deadline=None)
@given(
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    composed=st.booleans(),
    citation=st.sampled_from((OREGON_ROOT, OREGON_PAGE)),
    profile=st.sampled_from(("en-US", "legacy")),
    independent_facts=st.booleans(),
    newline=st.sampled_from(("\n", "\r\n")),
    reference=st.sampled_from(
        (
            "The month shown above determines eligibility.",
            "The month and year above establish the application period.",
            "Use the date above to determine the first month of assistance.",
        )
    ),
)
@example(
    month=7,
    year=2026,
    composed=False,
    citation=OREGON_ROOT,
    profile="en-US",
    independent_facts=False,
    newline="\n",
    reference="The month shown above determines eligibility.",
)
@example(
    month=11,
    year=2027,
    composed=False,
    citation=OREGON_PAGE,
    profile="legacy",
    independent_facts=True,
    newline="\n",
    reference="The month shown above determines eligibility.",
)
def test_generated_section_heading_cannot_authenticate_a_date_used_by_body_prose(
    month, year, composed, citation, profile, independent_facts, newline, reference
):
    """A heading cannot authenticate the later body use of a revision date."""
    stamp = f"({month:02d}/{year})"
    source = (
        f"93 {stamp} Chapter 2:Eligibility\n"
        "Section 1: Application\n1.1.1 Application Process\n"
    )
    if independent_facts:
        source += (
            "Households receive $444 and must apply within 30 days.\n"
            "Applications filed in (11/2027) are eligible.\n"
        )
    source += reference
    if composed:
        source = "Page 93\n\n" + source
    source = source.replace("\n", newline)
    recalled = _recalled_values(source, citation, profile=profile)
    demanded = _missing_values(source, citation, profile=profile)
    assert {month, year} <= recalled
    assert {month, year} <= demanded
    masked = sc._without_us_manual_locators(source, corpus_citation_path=citation)
    stamp_start = source.index(stamp)
    assert masked[stamp_start : stamp_start + len(stamp)] == stamp
    assert len(masked) == len(source)
    assert (
        sc._without_us_manual_locators(masked, corpus_citation_path=citation) == masked
    )


def test_earlier_composed_page_cannot_make_a_later_revision_stamp_operative():
    source = (
        "Page 92\n\nThe month shown below determines eligibility.\n\n"
        "Page 93\n\n93 (07/2026) Chapter 2: Eligibility\n"
        "The payment is $93 and $2026."
    )
    assert _recalled_values(source, OREGON_ROOT) == {7, 93, 2026}
    assert _missing_values(source, OREGON_ROOT) == {7, 93, 2026}


@settings(max_examples=50, deadline=None)
@given(
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    intermediate_page=st.booleans(),
    preceding_reference=st.booleans(),
    newline=st.sampled_from(("\n", "\r\n")),
)
@example(
    month=7, year=2026, intermediate_page=False, preceding_reference=False, newline="\n"
)
@example(
    month=7, year=2026, intermediate_page=False, preceding_reference=True, newline="\n"
)
def test_generated_composed_amount_header_authenticates_against_all_later_pages(
    month, year, intermediate_page, preceding_reference, newline
):
    """A body reference anywhere in the composed source keeps the month demand."""
    stamp = f"({month:02d}/{year})"
    source = (
        f"Page 93\n\n93 {stamp} Chapter 2: Eligibility\n"
        f"The payment is $93 and ${year}.\n\n"
    )
    if preceding_reference:
        source = "Page 92\n\nThe month shown below determines eligibility.\n\n" + source
    else:
        if intermediate_page:
            source += "Page 94\n\nApplications are reviewed.\n\n"
        source += (
            f"Page {95 if intermediate_page else 94}\n\n"
            "The month shown above determines eligibility."
        )
    source = source.replace("\n", newline)
    assert {month, 93, year} <= _recalled_values(source, OREGON_ROOT)
    assert {month, 93, year} <= _missing_values(source, OREGON_ROOT)
    assert stamp in sc._without_us_manual_locators(
        source, corpus_citation_path=OREGON_ROOT
    )


@pytest.fixture
def real_composed_oregon_cover(tmp_path: Path):
    """Resolve the actual bodyless root and cover rows from the pinned book."""
    version = "2026-07-16-or-programs-eligibility-notebook"
    shared = {
        "jurisdiction": "us-or",
        "document_class": "manual",
        "version": version,
        "source_as_of": "2026-07-16",
        "expression_date": "2026-07-01",
        "source_path": (
            "sources/us-or/manual/2026-07-16-or-programs-eligibility-notebook/"
            "official-documents/or-open-2818.pdf"
        ),
    }
    rows = (
        {
            **shared,
            "id": "3fef2437-4720-5b38-9b97-c8b2fd1f748d",
            "citation_path": OREGON_ROOT,
            "heading": "Oregon Programs Eligibility Notebook (OPEN)",
            "body": None,
            "level": 1,
            "ordinal": 1,
        },
        {
            **shared,
            "id": "04834956-7a7f-5aa4-aee0-5c842d669fe7",
            "citation_path": f"{OREGON_ROOT}/page-1",
            "heading": "Page 1",
            "body": f"{OREGON_COVER_PREFIX} (07/2026)",
            "level": 2,
            "ordinal": 1,
        },
    )
    artifact = (
        tmp_path
        / "data"
        / "corpus"
        / "provisions"
        / "us-or"
        / "manual"
        / f"{version}.jsonl"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("".join(json.dumps(row) + "\n" for row in rows))
    release = bind_test_corpus_release(
        tmp_path, "manual-stamp-review", [("us-or", "manual", version)]
    )
    resolved = resolve_local_corpus_source(OREGON_ROOT, release)
    assert resolved.body == f"Page 1\n\n{OREGON_COVER_PREFIX} (07/2026)"
    assert resolved.component_rows[0].citation_path == f"{OREGON_ROOT}/page-1"
    return resolved


@pytest.mark.parametrize("citation", (OREGON_ROOT, f"{OREGON_ROOT}/page-1"))
def test_real_resolver_composition_authenticates_cover_stamp(
    real_composed_oregon_cover, citation
):
    source = real_composed_oregon_cover.body
    spans = sc._manual_revision_stamp_spans(source, corpus_citation_path=citation)
    assert [source[start:end] for start, end in spans] == ["(07/2026)"]
    assert not ({7, 2026} & _recalled_values(source, citation))
    assert not ({7, 2026} & _missing_values(source, citation))


def test_resolver_composition_preserves_a_date_continued_on_the_next_page(
    real_composed_oregon_cover, tmp_path: Path
):
    """Build the attack through the real bodyless-root composer, not a prefix."""

    version = real_composed_oregon_cover.component_rows[0].version
    artifact = (
        tmp_path
        / "data"
        / "corpus"
        / "provisions"
        / "us-or"
        / "manual"
        / f"{version}.jsonl"
    )
    root, first = [json.loads(line) for line in artifact.read_text().splitlines()]
    first.update(
        citation_path=OREGON_PAGE,
        heading="Page 93",
        body=f"{OREGON_CAPTION}\n(10/2025)",
        ordinal=93,
    )
    second = {
        **first,
        "id": "de35b9fd-484b-42cb-bf18-e401f9d40b91",
        "citation_path": f"{OREGON_ROOT}/page-94",
        "heading": "Page 94",
        "body": "is the qualifying application month.",
        "ordinal": 94,
    }
    artifact.write_text(
        "".join(json.dumps(row) + "\n" for row in (root, first, second))
    )
    release = bind_test_corpus_release(
        tmp_path, "manual-cross-page-sentence-review", [("us-or", "manual", version)]
    )
    resolved = resolve_local_corpus_source(OREGON_ROOT, release)
    assert resolved.body == (
        f"Page 93\n\n{OREGON_CAPTION}\n(10/2025)\n\n"
        "Page 94\n\nis the qualifying application month."
    )
    assert {10, 2025} <= _recalled_values(resolved.body, OREGON_ROOT)
    assert {10, 2025} <= _missing_values(resolved.body, OREGON_ROOT)


def test_oregon_document_root_authenticates_its_raw_cover_stamp():
    source = f"{OREGON_COVER_PREFIX} (07/2026)"
    spans = sc._manual_revision_stamp_spans(source, corpus_citation_path=OREGON_ROOT)
    assert [source[start:end] for start, end in spans] == ["(07/2026)"]


@settings(max_examples=50, deadline=None)
@given(
    page=st.integers(min_value=1, max_value=999),
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
)
def test_generated_composed_pages_authenticate_each_stamp_at_original_offsets(
    page, month, year
):
    stamp = f"({month:02d}/{year})"
    source = "\n\n".join(
        f"Page {number}\n\n{OREGON_COVER_PREFIX} {stamp}" for number in (page, page + 1)
    )
    expected_spans = tuple(
        match.span() for match in re.finditer(re.escape(stamp), source)
    )
    assert (
        sc._manual_revision_stamp_spans(source, corpus_citation_path=OREGON_ROOT)
        == expected_spans
    )
    masked = sc._without_us_manual_locators(source, corpus_citation_path=OREGON_ROOT)
    assert len(masked) == len(source)
    assert all(
        not any(char.isdigit() for char in masked[a:b]) for a, b in expected_spans
    )


@pytest.mark.parametrize("composed", (False, True), ids=("raw-page", "composed-page"))
@settings(max_examples=30, deadline=None)
@given(
    replacement=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=16)
)
def test_generated_reviewed_page_authentication_requires_exact_body_bytes(
    composed, replacement
):
    stamp = "(10/2025)"
    original = f"{OREGON_CAPTION} {stamp} Applications qualify for assistance."
    stamp_start = original.index(stamp)
    digest = hashlib.sha256(original.encode()).hexdigest()
    prefix = "Page 93\n\n" if composed else ""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            sc,
            "REVIEWED_MANUAL_REVISION_STAMPS",
            {OREGON_ROOT: {digest: stamp_start}},
        )
        assert sc._manual_revision_stamp_spans(
            prefix + original, corpus_citation_path=OREGON_ROOT
        ) == ((len(prefix) + stamp_start, len(prefix) + stamp_start + len(stamp)),)
        changed = original.replace("qualify", replacement + "qualify")
        assert (
            sc._manual_revision_stamp_spans(
                prefix + changed, corpus_citation_path=OREGON_ROOT
            )
            == ()
        )
        assert stamp in sc._without_us_manual_locators(
            prefix + changed, corpus_citation_path=OREGON_ROOT
        )


BODY_CONTINUATIONS = (
    "is the qualifying application month.",
    "marks the first month of eligibility.",
    "provides the qualifying application month.",
    "Provides the qualifying application month.",
    "determines eligibility for assistance.",
    "establishes the application period.",
    "governs eligibility for assistance.",
    "denotes the qualifying application month.",
    ", applications qualify for assistance.",
    "if an application is received, pay it.",
    "is required before an application qualifies.",
    "Applicant Name must be provided for eligibility.",
    "FoodShare Handbook Release determines the qualifying month.",
)


@settings(max_examples=50, deadline=None)
@given(
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    metadata=st.sampled_from(
        (
            "OREGON PROGRAMS ELIGIBILITY NOTEBOOK",
            "Chapter 4: Support Service Payments",
            "Family Services procedure manual Chapter 4: Support Service Payments",
        )
    ),
    suffix_break=st.sampled_from((" ", "\n", "\n\n")),
    continuation=st.sampled_from(BODY_CONTINUATIONS),
)
@example(
    month=10,
    year=2025,
    metadata="Chapter 4: Support Service Payments",
    suffix_break=" ",
    continuation="is the qualifying application month.",
)
def test_generated_initial_stamp_cannot_hide_a_date_inside_a_header_like_sentence(
    month, year, metadata, suffix_break, continuation
):
    stamp = f"({month:02d}/{year})"
    source = f"93 {stamp} {metadata}{suffix_break}{continuation}"
    assert {month, year} <= _recalled_values(source, OREGON_PAGE)
    assert {month, year} <= _missing_values(source, OREGON_PAGE)
    assert stamp in sc._without_us_manual_locators(
        source, corpus_citation_path=OREGON_PAGE
    )


@settings(max_examples=50, deadline=None)
@given(
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    metadata=st.sampled_from((OREGON_CAPTION, OREGON_COVER_PREFIX)),
    continuation=st.sampled_from(BODY_CONTINUATIONS),
)
@example(
    month=10,
    year=2025,
    metadata=OREGON_CAPTION,
    continuation="is the qualifying application month.",
)
def test_generated_body_date_continuation_crosses_composed_page_boundaries(
    month, year, metadata, continuation
):
    stamp = f"({month:02d}/{year})"
    source = f"Page 93\n\n{metadata} {stamp}\n\nPage 94\n\n{continuation}"
    assert {month, year} <= _recalled_values(source, OREGON_ROOT)
    assert {month, year} <= _missing_values(source, OREGON_ROOT)
    assert stamp in sc._without_us_manual_locators(
        source, corpus_citation_path=OREGON_ROOT
    )


@settings(max_examples=50, deadline=None)
@given(
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    continuation=st.sampled_from(
        (
            "Determines Eligibility",
            "Marks The Qualifying Application Month",
            "Is The Qualifying Application Month",
            "Governs Eligibility For Assistance",
        )
    ),
    separator=st.sampled_from((" ", "\n")),
)
@example(month=10, year=2025, continuation="Determines Eligibility", separator=" ")
def test_generated_titlecase_predicate_cannot_turn_a_body_date_into_metadata(
    month, year, continuation, separator
):
    stamp = f"({month:02d}/{year})"
    source = f"93 {stamp} Chapter 4: Working With Families{separator}{continuation}"
    assert {month, year} <= _recalled_values(source, OREGON_PAGE)
    assert {month, year} <= _missing_values(source, OREGON_PAGE)


@settings(max_examples=50, deadline=None)
@given(
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    continuation=st.sampled_from(BODY_CONTINUATIONS),
)
@example(month=10, year=2025, continuation="is the qualifying application month.")
def test_generated_successor_footer_cannot_authenticate_a_previous_body_date(
    month, year, continuation
):
    stamp = f"({month:02d}/{year})"
    citation = "us-va/manual/dss/snap/page-1"
    source = (
        f"Page 1\n\n{VIRGINIA_PREFIX} {stamp}\n\n"
        f"Page 2\n\n{continuation}\n{VIRGINIA_PREFIX} (07/2026)"
    )
    assert {month, year} <= _recalled_values(source, citation)
    assert {month, year} <= _missing_values(source, citation)
    assert stamp in sc._without_us_manual_locators(
        source, corpus_citation_path=citation
    )


@settings(max_examples=50, deadline=None)
@given(
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    continuation=st.sampled_from(BODY_CONTINUATIONS),
)
@example(month=10, year=2025, continuation="is the qualifying application month.")
def test_generated_next_page_header_cannot_interrupt_an_operative_date_sentence(
    month, year, continuation
):
    stamp = f"({month:02d}/{year})"
    source = (
        f"Page 93\n\n{OREGON_CAPTION}\n{stamp}\n\n"
        "Page 94\n\n94 (07/2026) Chapter 2:Eligibility\n"
        f"Section 1: Application\n{continuation}"
    )
    assert {month, year} <= _recalled_values(source, OREGON_ROOT)
    assert {month, year} <= _missing_values(source, OREGON_ROOT)
    assert stamp in sc._without_us_manual_locators(
        source, corpus_citation_path=OREGON_ROOT
    )


@settings(max_examples=50, deadline=None)
@given(
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    continuation=st.sampled_from(BODY_CONTINUATIONS),
)
@example(month=10, year=2025, continuation="is the qualifying application month.")
def test_generated_successor_header_and_footer_cannot_hide_a_body_date(
    month, year, continuation
):
    stamp = f"({month:02d}/{year})"
    citation = "us-va/manual/dss/snap/page-1"
    source = (
        f"Page 1\n\n{VIRGINIA_PREFIX} {stamp}\n\n"
        "Page 2\n\n94 (07/2026) Chapter 2:Eligibility\n"
        f"Section 1: Application\n{continuation}\n{VIRGINIA_PREFIX} (08/2026)"
    )
    assert {month, year} <= _recalled_values(source, citation)
    assert {month, year} <= _missing_values(source, citation)
    assert stamp in sc._without_us_manual_locators(
        source, corpus_citation_path=citation
    )


@settings(max_examples=50, deadline=None)
@given(
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    separator=st.sampled_from((" ", "\n", "\n\n")),
    continuation=st.sampled_from(BODY_CONTINUATIONS),
)
@example(
    month=10,
    year=2025,
    separator=" ",
    continuation="is the qualifying application month.",
)
def test_generated_section_prefix_cannot_hide_a_body_date(
    month, year, separator, continuation
):
    stamp = f"({month:02d}/{year})"
    source = (
        f"93 {stamp} Chapter 2:Eligibility\n"
        f"Section 1: Application{separator}{continuation}"
    )
    assert {month, year} <= _recalled_values(source, OREGON_PAGE)
    assert {month, year} <= _missing_values(source, OREGON_PAGE)
    assert stamp in sc._without_us_manual_locators(
        source, corpus_citation_path=OREGON_PAGE
    )


@settings(max_examples=80, deadline=None)
@given(
    caption=st.sampled_from(sorted(OREGON_OPEN_REVISION_CAPTIONS)),
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    prefix_break=st.sampled_from((" ", "\n", "\n\n")),
    suffix_break=st.sampled_from((" ", "\n", "\n\n")),
    continuation=st.sampled_from(BODY_CONTINUATIONS),
)
def test_generated_registry_caption_cannot_authenticate_a_body_date(
    caption, month, year, prefix_break, suffix_break, continuation
):
    stamp = f"({month:02d}/{year})"
    source = f"93 {caption}{prefix_break}{stamp}{suffix_break}{continuation}"
    masked = sc._without_us_manual_locators(source, corpus_citation_path=OREGON_PAGE)
    assert stamp in masked
    assert {month, year} <= _recalled_values(source, OREGON_PAGE)
    assert len(masked) == len(source)


@settings(max_examples=80, deadline=None)
@given(
    metadata=st.sampled_from(
        (
            (OREGON_COVER_PREFIX, f"{OREGON_ROOT}/page-1"),
            (WISCONSIN_PREFIX, "us-wi/manual/dhs/foodshare/page-1"),
            (VIRGINIA_PREFIX, "us-va/manual/dss/snap/page-1"),
        )
    ),
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    suffix_break=st.sampled_from((" ", "\n", "\n\n")),
    continuation=st.sampled_from(BODY_CONTINUATIONS),
)
def test_generated_form_prefix_cannot_authenticate_a_body_date(
    metadata, month, year, suffix_break, continuation
):
    prefix, citation = metadata
    stamp = f"({month:02d}/{year})"
    source = f"{prefix} {stamp}{suffix_break}{continuation}"
    masked = sc._without_us_manual_locators(source, corpus_citation_path=citation)
    assert stamp in masked
    assert {month, year} <= _recalled_values(source, citation)
    assert len(masked) == len(source)


@dataclass(frozen=True)
class _ComposedVirginiaBook:
    citation: str
    body_sha256: str
    body: str = field(repr=False)
    roles: tuple[dict, ...] = field(repr=False)


@pytest.fixture(scope="module")
def real_composed_virginia_book():
    path = (
        Path(__file__).parent / "fixtures/manual_mask_review_composed_virginia.json.gz"
    )
    fixture = json.loads(gzip.decompress(path.read_bytes()))
    body = fixture["body"]
    assert len(body) == 1_431_563
    assert fixture["components"] == 675
    assert hashlib.sha256(body.encode()).hexdigest() == fixture["body_sha256"]
    return _ComposedVirginiaBook(
        fixture["citation"],
        fixture["body_sha256"],
        body,
        tuple(fixture["reviewed_revision_stamp_roles"]),
    )


@pytest.mark.parametrize("citation_suffix", ("", "/full-manual"), ids=("root", "book"))
def test_real_composed_virginia_book_keeps_all_authentic_revision_stamps(
    real_composed_virginia_book, citation_suffix
):
    source = real_composed_virginia_book
    expected = {tuple(role["span"]) for role in source.roles}
    assert len(expected) == 33
    for role in source.roles:
        start, end = role["span"]
        assert source.body[start:end] == role["stamp"]
    actual = sc._manual_revision_stamp_spans(
        source.body, corpus_citation_path=source.citation + citation_suffix
    )
    assert set(actual) == expected


@settings(max_examples=2, deadline=None)
@given(
    continuation=st.sampled_from(
        (
            "is the qualifying application month.",
            "must be the application month to receive assistance.",
        )
    )
)
def test_generated_real_virginia_book_body_date_crosses_the_next_page(
    real_composed_virginia_book, continuation
):
    source = real_composed_virginia_book
    role = next(
        item for item in source.roles if item["component"].endswith("/page-342")
    )
    start, end = role["span"]
    next_page = re.match(r"\n\nPage \d+\n\n", source.body[end:])
    assert next_page is not None
    insertion = end + next_page.end()
    edited = source.body[:insertion] + continuation + "\n" + source.body[insertion:]
    assert edited[:insertion] == source.body[:insertion]
    actual = sc._manual_revision_stamp_spans(
        edited, corpus_citation_path=source.citation + "/full-manual"
    )
    assert (start, end) not in actual
    assert edited[start:end] == role["stamp"]
