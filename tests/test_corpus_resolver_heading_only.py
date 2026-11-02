"""Flat-numbered regulation part headings are refused at resolution (#1825).

Regulations numbered flat (``3.104``, ``3.104.1``, ...) print a part heading as
its own row, with a heading, no body and no rows under its path. Such a row has
no body digest, and every signed, applied or guarded source attestation needs
one for each component row. So the resolver refuses a composition that would
include one, before generation, and says why. Every other bodyless row keeps
main's refusal and message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import axiom_encode.cli as cli
from axiom_encode.corpus_resolver import (
    InvalidActiveCorpusSourceError,
    resolve_local_corpus_source,
)
from axiom_encode.harness.evals import resolve_corpus_source_unit
from tests.test_corpus_resolver import _release, _write_rows, _write_selector

ROOT = "us/regulation/8-ccr-1403-1"
STATUTE_ROOT = "us/statute/8-ccr-1403-1"
PART_HEADING = "heading-only descendant"


def _bind(
    tmp_path: Path,
    rows: list[dict[str, object]],
    version: str = "2026-09-27",
    document_class: str = "regulation",
):
    _write_selector(
        tmp_path,
        [{"jurisdiction": "us", "document_class": document_class, "version": version}],
    )
    _write_rows(tmp_path, version, rows, document_class=document_class)
    return _release(tmp_path)


def _flat_rows():
    return [
        {"citation_path": ROOT, "body": None},
        {
            "citation_path": f"{ROOT}/3.100",
            "body": None,
            "heading": "3.100 PART A",
            "ordinal": 1,
        },
        {
            "citation_path": f"{ROOT}/3.101",
            "body": "A. Applicants must reside in Colorado; the limit is $40.",
            "heading": "3.101 Residency",
            "ordinal": 2,
        },
    ]


def test_part_heading_leaf_is_refused_with_its_reason(tmp_path):
    release = _bind(tmp_path, _flat_rows())

    with pytest.raises(InvalidActiveCorpusSourceError, match=PART_HEADING) as raised:
        resolve_local_corpus_source(ROOT, release)

    message = str(raised.value)
    assert f"{ROOT}/3.100" in message
    assert "no body digest" in message
    assert "before generation" in message


def test_part_heading_is_refused_when_resolving_a_source_unit(tmp_path):
    """The refusal comes from the resolution encode runs before any model call.

    `_run_encode_attempt` calls `resolve_corpus_source_unit` before
    `run_model_eval`, so nothing is generated for such a source. Composing it
    instead would yield a component row without `body_sha256`, which
    `_source_attestation_structure_issues` refuses at sign, apply and guard
    time, after the model has run.
    """

    release = _bind(tmp_path, _flat_rows())

    with pytest.raises(InvalidActiveCorpusSourceError, match=PART_HEADING):
        resolve_corpus_source_unit(ROOT, release)


def test_a_body_digest_is_what_attestation_requires_of_each_component():
    """Pins the consumer contract the refusal exists for."""

    attestation = {
        "component_rows": [
            {
                "citation_path": f"{ROOT}/3.100",
                "provision_file": "data/corpus/provisions/us/regulation/x.jsonl",
                "provision_file_sha256": "0" * 64,
                "line_number": 2,
                "body_sha256": None,
                "source_path": "sources/x",
                "source_as_of": "2026-01-01",
                "expression_date": "2026-01-01",
            }
        ]
    }

    issues = cli._source_attestation_structure_issues(attestation)

    assert (
        "source_attestation.component_rows[0].body_sha256 must be a lowercase "
        "sha256 digest" in issues
    )


def test_root_of_only_part_headings_is_refused_with_its_reason(tmp_path):
    release = _bind(
        tmp_path,
        [
            {"citation_path": ROOT, "body": None},
            {"citation_path": f"{ROOT}/3.100", "body": None, "heading": "PART A"},
            {"citation_path": f"{ROOT}/3.200", "body": None, "heading": "PART B"},
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match=PART_HEADING):
        resolve_local_corpus_source(ROOT, release)


def test_part_heading_target_keeps_mains_refusal(tmp_path):
    release = _bind(
        tmp_path,
        [{"citation_path": f"{ROOT}/3.100", "body": "", "heading": "3.100 PART A"}],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="no body-bearing"):
        resolve_local_corpus_source(f"{ROOT}/3.100", release)


def test_heading_only_statute_leaf_keeps_mains_refusal(tmp_path):
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
def test_amendment_or_commencement_leaf_keeps_mains_refusal(tmp_path, metadata):
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


@pytest.mark.parametrize("heading", [None, "", "   "])
def test_textless_leaf_keeps_mains_refusal(tmp_path, heading):
    empty = {"citation_path": f"{ROOT}/3.102", "body": None}
    if heading is not None:
        empty["heading"] = heading
    release = _bind(
        tmp_path,
        [
            {"citation_path": ROOT, "body": None},
            {"citation_path": f"{ROOT}/3.101", "body": "operative text"},
            empty,
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="uncovered bodyless"):
        resolve_local_corpus_source(ROOT, release)


@pytest.mark.parametrize("root_heading", [None, "8 CCR 1403-1"])
def test_heading_container_over_bodyless_children_keeps_mains_refusal(
    tmp_path, root_heading
):
    """A heading row with a row under it is a container, not a part heading.

    In a tree with no bodies at all, main's message stands: the review's
    reproduction for the precheck (#2035 delta review, N1).
    """

    root = {"citation_path": ROOT, "body": None}
    if root_heading is not None:
        root["heading"] = root_heading
    release = _bind(
        tmp_path,
        [
            root,
            {
                "citation_path": f"{ROOT}/3.100",
                "body": None,
                "heading": "3.100 PART A",
                "ordinal": 1,
            },
            {"citation_path": f"{ROOT}/3.100/a", "body": None, "ordinal": 1},
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError) as raised:
        resolve_local_corpus_source(ROOT, release)

    assert str(raised.value) == (
        f"Active corpus source {ROOT!r} has no body-bearing descendants"
    )


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


def test_root_of_only_repealed_part_headings_keeps_mains_refusal(tmp_path):
    """Repealed rows are tombstones, not part headings, at every check."""

    release = _bind(
        tmp_path,
        [
            {"citation_path": ROOT, "body": None},
            {
                "citation_path": f"{ROOT}/3.100",
                "body": None,
                "heading": "3.100 (Repealed)",
                "metadata": {"status": "repealed"},
            },
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError) as raised:
        resolve_local_corpus_source(ROOT, release)

    assert "no body-bearing descendants" in str(raised.value)
    assert PART_HEADING not in str(raised.value)


_KINDS = st.sampled_from(["body", "body_and_heading", "part_heading", "empty"])


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(kinds=st.lists(_KINDS, min_size=1, max_size=6))
def test_property_flat_regulation_rows(tmp_path_factory, kinds):
    """A flat root composes as on main when every row has a body.

    With a part heading or a textless row it is refused; when the first
    bodyless row in order is a part heading (or every bodyless row is), the
    refusal names the heading-only descendant.
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
        elif kind == "part_heading":
            row["body"] = None
            row["heading"] = heading
        else:
            row["body"] = None
        rows.append(row)
    release = _bind(tmp_path, rows)

    bodyless = [kind for kind in kinds if kind in {"part_heading", "empty"}]
    if not bodyless:
        assert resolve_local_corpus_source(ROOT, release).body == "\n\n".join(
            expected_chunks
        )
        return
    with pytest.raises(InvalidActiveCorpusSourceError) as raised:
        resolve_local_corpus_source(ROOT, release)
    if bodyless[0] == "part_heading" or set(bodyless) == {"part_heading"}:
        assert PART_HEADING in str(raised.value)
