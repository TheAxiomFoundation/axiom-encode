"""Exact reviewed leaf scopes never waive their operative parent or siblings."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import coverage_exemptions as ce
from axiom_encode.harness import validator_pipeline as vp

ROOT = Path(__file__).parents[1]
# Each reviewed batch records the registry it extended. Batches are appended in
# order, so each baseline is a byte prefix of the current registry.
REVIEWS = [
    json.loads((ROOT / "docs/coverage-exemptions" / name).read_text())
    for name in (
        "four-editorial-leaves-review.json",
        "three-repealed-leaves-review.json",
    )
]
OCCURRENCES = [row for review in REVIEWS for row in review["occurrences"]]
TARGETS = [row["target"] for row in OCCURRENCES]
ENTRIES = tuple(
    e
    for e in ce.load_coverage_exemptions()
    if e.owner_evidence["citation_path"] in TARGETS
)


def source(entry):
    name = "a-9" if entry.citation_path.endswith("3121") else "b-12"
    return (
        ROOT / f"tests/fixtures/editorial_source_ancestry/{name}-source.txt"
    ).read_text()


def demands(entry, text):
    return [
        d
        for d in vp._high_signal_top_level_coverage_demands(text)
        if d.label == entry.label
    ]


def test_previous_registry_bytes_and_all_other_entries_preserved():
    lines = (
        (ROOT / "src/axiom_encode/harness/data/coverage_exemptions.jsonl")
        .read_bytes()
        .splitlines(keepends=True)
    )
    assert len(lines) == 352
    for review in REVIEWS:
        assert (
            hashlib.sha256(b"".join(lines[: review["baseline_entries"]])).hexdigest()
            == review["baseline_registry_sha256"]
        )
    # Each batch appends exactly its own occurrences after its baseline.
    for review, following in zip(REVIEWS, REVIEWS[1:]):
        assert (
            review["baseline_entries"] + len(review["occurrences"])
            == following["baseline_entries"]
        )
    last = REVIEWS[-1]
    assert last["baseline_entries"] + len(last["occurrences"]) == len(lines)
    appended = [
        json.loads(line)["owner_evidence"]["citation_path"]
        for line in lines[REVIEWS[0]["baseline_entries"] :]
    ]
    assert sorted(appended) == sorted(TARGETS)
    assert len(ENTRIES) == 7
    assert {e.owner_evidence["citation_path"] for e in ENTRIES} == set(TARGETS)
    assert {e.reason for e in ENTRIES} == {"repealed_tombstone"}


@pytest.mark.parametrize(
    "entry", ENTRIES, ids=lambda e: e.owner_evidence["citation_path"]
)
def test_actual_source_notice_and_exact_scoped_consumer(entry, monkeypatch):
    text = source(entry)
    row = next(
        r for r in OCCURRENCES if r["target"] == entry.owner_evidence["citation_path"]
    )
    start, end = row["notice_span"]
    assert text[start:end] == row["exact_notice"]
    assert text.count(row["exact_notice"]) == 1
    assert ce.text_sha256(text) == entry.source_text_sha256
    assert entry.owner_evidence["physical_start"] == start
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {
                "status": "deferred",
                "summary": row["exact_notice"],
                "source_verification": {"corpus_citation_path": entry.citation_path},
            },
            "rules": [],
        }
    )
    kwargs = dict(
        source_texts={entry.citation_path: text}, requested_source=row["target"]
    )
    # Actual shared coverage consumer, not merely the registry matcher.
    assert vp.find_source_subparagraph_coverage_issues(content, **kwargs) == []
    baseline = tuple(e for e in ce.load_coverage_exemptions() if e not in ENTRIES)
    monkeypatch.setattr(ce, "load_coverage_exemptions", lambda: baseline)
    assert vp.find_source_subparagraph_coverage_issues(content, **kwargs)


@pytest.mark.parametrize(
    "entry", ENTRIES, ids=lambda e: e.owner_evidence["citation_path"]
)
@pytest.mark.parametrize("scope_kind", ["parent", "unscoped", "sibling", "descendant"])
def test_no_parent_null_sibling_or_descendant_waiver(entry, scope_kind):
    text = source(entry)
    scope = {
        "parent": entry.requested_scope[:1],
        "unscoped": None,
        "sibling": (entry.label, str(int(entry.requested_scope[-1]) + 1)),
        "descendant": (*entry.requested_scope, "A"),
    }[scope_kind]
    original = demands(entry, text)
    assert original
    assert (
        ce.filter_coverage_demands(
            original,
            citation_path=entry.citation_path,
            requested_scope=scope,
            source_text=text,
        )
        == original
    )

    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {
                "source_verification": {"corpus_citation_path": entry.citation_path}
            },
            "rules": [],
        }
    )
    requested = entry.citation_path + ("/" + "/".join(scope) if scope else "")
    assert vp.find_source_subparagraph_coverage_issues(
        content, source_texts={entry.citation_path: text}, requested_source=requested
    )


@pytest.mark.parametrize(
    "entry", ENTRIES, ids=lambda e: e.owner_evidence["citation_path"]
)
@pytest.mark.parametrize(
    "mutation",
    ["source", "hash", "offset", "window", "window_hash", "citation", "scope", "label"],
)
def test_changed_binding_never_removes_demand(entry, mutation):
    text = source(entry)
    original = demands(entry, text)
    assert original
    altered = entry
    if mutation == "source":
        text += "\n"
    else:
        changes = {
            "hash": {"source_text_sha256": "0" * 64},
            "offset": {"physical_start": entry.physical_start + 1},
            "window": {"window": entry.window + "x"},
            "window_hash": {"window_sha256": "0" * 64},
            "citation": {"citation_path": entry.citation_path + "/other"},
            "scope": {"requested_scope": (entry.label,)},
            "label": {"label": "z"},
        }
        altered = replace(entry, **changes[mutation])
    assert (
        ce.filter_coverage_demands(
            original,
            citation_path=entry.citation_path,
            requested_scope=entry.requested_scope,
            source_text=text,
            exemptions=(altered,),
        )
        == original
    )
