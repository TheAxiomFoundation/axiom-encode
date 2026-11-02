"""Flat-numbered regulation part headings compose into their parent (#1825).

Regulations numbered flat (``3.104``, ``3.104.1``, ...) print a part heading as
its own row, with no body and no rows under its path. As a leaf of a composed
regulation parent, such a row contributes its heading. Everything else keeps
main's contract: a bodyless target still refuses, a heading-only statute leaf
or a row with amendment or commencement metadata is still an uncovered branch,
and a row with no text at all is still refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.corpus_resolver import (
    InvalidActiveCorpusSourceError,
    resolve_local_corpus_source,
)
from tests.test_corpus_resolver import _release, _write_rows, _write_selector

ROOT = "us/regulation/8-ccr-1403-1"
STATUTE_ROOT = "us/statute/8-ccr-1403-1"


def _bind(
    tmp_path: Path,
    rows: list[dict[str, object]],
    version: str = "2026-09-27",
    document_class: str = "regulation",
):
    _write_selector(
        tmp_path,
        [
            {
                "jurisdiction": "us",
                "document_class": document_class,
                "version": version,
            }
        ],
    )
    _write_rows(tmp_path, version, rows, document_class=document_class)
    return _release(tmp_path)


def test_heading_only_target_still_refuses(tmp_path):
    """Requested on its own, a bodyless row refuses as on main."""

    release = _bind(
        tmp_path,
        [
            {
                "citation_path": f"{ROOT}/3.100",
                "body": "",
                "heading": "3.100 PART A: GENERAL PROVISIONS",
            }
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="no body-bearing"):
        resolve_local_corpus_source(f"{ROOT}/3.100", release)


def test_heading_only_statute_leaf_is_still_uncovered(tmp_path):
    release = _bind(
        tmp_path,
        [
            {"citation_path": STATUTE_ROOT, "body": None},
            {"citation_path": f"{STATUTE_ROOT}/1", "body": "covered"},
            {"citation_path": f"{STATUTE_ROOT}/2", "body": None, "heading": "PART"},
        ],
        document_class="statute",
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="uncovered bodyless"):
        resolve_local_corpus_source(STATUTE_ROOT, release)


@pytest.mark.parametrize(
    "metadata",
    [
        {"amendment_target": "us/regulation/other/1"},
        {"commencement_date": "2027-04-01"},
        {"commencement_citation_path": "us/regulation/other/2"},
        {"document_type": "amendment act"},
    ],
)
def test_heading_only_leaf_with_amendment_metadata_is_still_uncovered(
    tmp_path, metadata
):
    release = _bind(
        tmp_path,
        [
            {"citation_path": ROOT, "body": None},
            {"citation_path": f"{ROOT}/3.101", "body": "operative text"},
            {
                "citation_path": f"{ROOT}/3.102",
                "body": None,
                "heading": "3.102 Amended",
                "metadata": metadata,
            },
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="uncovered bodyless"):
        resolve_local_corpus_source(ROOT, release)


def test_root_over_flat_numbered_rows_composes_part_headings_in_order(tmp_path):
    release = _bind(
        tmp_path,
        [
            {"citation_path": ROOT, "body": None},
            {
                "citation_path": f"{ROOT}/3.100",
                "body": None,
                "heading": "3.100 PART A",
                "ordinal": 1,
            },
            {
                "citation_path": f"{ROOT}/3.101",
                "body": "A. Applicants must reside in Colorado.",
                "heading": "3.101 Residency",
                "ordinal": 2,
            },
            {
                "citation_path": f"{ROOT}/3.104",
                "body": "   ",
                "heading": "3.104 PART B",
                "ordinal": 3,
            },
            {
                "citation_path": f"{ROOT}/3.104.1",
                "body": "The limit is $500.",
                "ordinal": 4,
            },
        ],
    )

    resolved = resolve_local_corpus_source(ROOT, release)

    assert resolved.body == (
        "3.100 PART A\n\n"
        "3.101 Residency\n\nA. Applicants must reside in Colorado.\n\n"
        "3.104 PART B\n\n"
        "The limit is $500."
    )
    assert [row.citation_path for row in resolved.component_rows] == [
        f"{ROOT}/3.100",
        f"{ROOT}/3.101",
        f"{ROOT}/3.104",
        f"{ROOT}/3.104.1",
    ]


@pytest.mark.parametrize("heading", [None, "", "   "])
def test_section_with_no_text_is_still_refused(tmp_path, heading):
    row = {"citation_path": f"{ROOT}/3.100", "body": None}
    if heading is not None:
        row["heading"] = heading
    release = _bind(tmp_path, [row])

    with pytest.raises(InvalidActiveCorpusSourceError, match="no body-bearing"):
        resolve_local_corpus_source(f"{ROOT}/3.100", release)


@pytest.mark.parametrize("heading", [None, "  "])
def test_root_over_a_textless_descendant_is_still_refused(tmp_path, heading):
    empty = {"citation_path": f"{ROOT}/3.102", "body": None}
    if heading is not None:
        empty["heading"] = heading
    release = _bind(
        tmp_path,
        [
            {"citation_path": ROOT, "body": None},
            {"citation_path": f"{ROOT}/3.100", "body": None, "heading": "PART A"},
            {"citation_path": f"{ROOT}/3.101", "body": "operative text"},
            empty,
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="uncovered bodyless"):
        resolve_local_corpus_source(ROOT, release)


def test_heading_only_container_still_composes_from_its_children(tmp_path):
    release = _bind(
        tmp_path,
        [
            {"citation_path": ROOT, "body": None},
            {"citation_path": f"{ROOT}/3.100", "body": None, "heading": "PART A"},
            {"citation_path": f"{ROOT}/3.100/a", "body": "child text"},
        ],
    )

    assert resolve_local_corpus_source(ROOT, release).body == "child text"
    assert resolve_local_corpus_source(f"{ROOT}/3.100", release).body == "child text"


def test_repealed_heading_only_leaf_still_contributes_nothing(tmp_path):
    release = _bind(
        tmp_path,
        [
            {"citation_path": ROOT, "body": None},
            {"citation_path": f"{ROOT}/3.101", "body": "operative text"},
            {
                "citation_path": f"{ROOT}/3.102",
                "body": None,
                "heading": "3.102 (Repealed)",
                "metadata": {"status": "repealed"},
            },
        ],
    )

    assert resolve_local_corpus_source(ROOT, release).body == "operative text"


def test_root_of_only_heading_rows_resolves_to_the_headings(tmp_path):
    release = _bind(
        tmp_path,
        [
            {"citation_path": ROOT, "body": None},
            {"citation_path": f"{ROOT}/3.100", "body": None, "heading": "PART A"},
            {"citation_path": f"{ROOT}/3.200", "body": None, "heading": "PART B"},
        ],
    )

    assert resolve_local_corpus_source(ROOT, release).body == "PART A\n\nPART B"


_KINDS = st.sampled_from(["body", "body_and_heading", "heading_only", "empty"])


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(kinds=st.lists(_KINDS, min_size=1, max_size=6))
def test_property_flat_rows_resolve_iff_every_row_has_text(tmp_path_factory, kinds):
    """A flat root resolves exactly when no row is textless.

    When every row has a body (the inputs main already accepted), the composed
    text is main's: each body, prefixed by its heading when it has one. A
    heading-only row adds its heading at its position and nothing else.
    """

    tmp_path = tmp_path_factory.mktemp("flat")
    rows: list[dict[str, object]] = [{"citation_path": ROOT, "body": None}]
    expected_chunks: list[str] = []
    for index, kind in enumerate(kinds, start=1):
        path = f"{ROOT}/3.{100 + index}"
        heading = f"3.{100 + index} Heading {index}"
        body = f"Body {index} sets $ {index}0."
        row: dict[str, object] = {"citation_path": path, "ordinal": index}
        if kind == "body":
            row["body"] = body
            expected_chunks.append(body)
        elif kind == "body_and_heading":
            row["body"] = body
            row["heading"] = heading
            expected_chunks.append(f"{heading}\n\n{body}")
        elif kind == "heading_only":
            row["body"] = None
            row["heading"] = heading
            expected_chunks.append(heading)
        else:
            row["body"] = None
        rows.append(row)
    release = _bind(tmp_path, rows)

    if "empty" in kinds:
        with pytest.raises(InvalidActiveCorpusSourceError):
            resolve_local_corpus_source(ROOT, release)
        return
    resolved = resolve_local_corpus_source(ROOT, release)
    assert resolved.body == "\n\n".join(expected_chunks)
    assert len(resolved.component_rows) == len(kinds)
