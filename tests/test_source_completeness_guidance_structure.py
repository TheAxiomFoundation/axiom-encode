"""IRS guidance uses its printed outline while retaining exception obligations."""

from __future__ import annotations

import json
from pathlib import Path

from axiom_encode.harness import source_completeness as completeness

_FIXTURE = (
    Path(__file__).parent
    / "fixtures/source_completeness/irs_rev_proc_2025_32/page-14.json"
)
_RECORD = json.loads(_FIXTURE.read_text())
_SOURCE = _RECORD["body"]
_CITATION = _RECORD["citation_path"]


def test_irs_page14_preserves_subsections_and_their_numbered_paragraphs():
    branches = completeness.recognize_source_structure(
        _SOURCE, corpus_citation_path=_CITATION
    )
    by_path = {branch.path: branch for branch in branches}
    assert set(by_path) == {
        ("3",),
        ("05",),
        ("05", "1"),
        ("05", "2"),
        ("06",),
        ("06", "1"),
    }
    assert "$5,120" in by_path[("3",)].text
    assert "$2,200" in by_path[("05", "1")].text
    assert "$1,700" in by_path[("05", "2")].text
    assert "adjusted gross income" in by_path[("06", "1")].text
    assert "earned income" not in by_path[("3",)].text
    assert all(branch.kind != "sentence" for branch in branches)


def test_irs_page14_keeps_both_greater_income_witness_obligations():
    branches = completeness.recognize_source_structure(
        _SOURCE, corpus_citation_path=_CITATION
    )
    obligations = completeness._source_exception_branches(
        _SOURCE,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    assert len(obligations) == 2
    assert {branch.path for branch in obligations} == {("06", "1")}
    assert all("or, if greater, earned income" in branch.text for branch in obligations)
    assert all(
        "Absatz" not in completeness._branch_citation(_CITATION, branch)
        for branch in obligations
    )
    assert all(
        "paragraph" in completeness._branch_citation(_CITATION, branch).lower()
        for branch in obligations
    )


def test_irs_guidance_decimal_and_inline_statute_citations_are_not_branches():
    source = (
        ".06 Earned Income Credit. (1) In general. The rate is .25 per dollar "
        "under § 32(b)(2)(B). (2) Exception. If the taxpayer is excluded, no credit is allowed."
    )
    branches = completeness.recognize_source_structure(
        source, corpus_citation_path=_CITATION
    )
    assert {branch.path for branch in branches} == {("06",), ("06", "1"), ("06", "2")}
    assert "§ 32(b)(2)(B)" in next(
        branch.text for branch in branches if branch.path == ("06", "1")
    )


def test_default_german_structure_and_labels_are_preserved():
    branches = completeness.recognize_source_structure(
        "(3) Die Leistung beträgt 100 Euro."
    )
    assert len(branches) == 1
    assert branches[0].path == ("3",)
    assert completeness._branch_citation("de/statute/estg/32a", branches[0]).endswith(
        "[Absatz 3]"
    )


def test_irs_guidance_keeps_nested_list_structure_and_operative_chapeaux():
    source = (
        ".06 Earned Income Credit. Only residents qualify.\n"
        "(1) The taxpayer must satisfy:\n1. The age requirement;\n"
        "2. The residence requirement."
    )
    branches = completeness.recognize_source_structure(
        source, corpus_citation_path=_CITATION
    )
    assert {branch.path for branch in branches} == {
        ("06",),
        ("06", "1"),
        ("06", "1", "1"),
        ("06", "1", "2"),
    }
    parent = next(branch for branch in branches if branch.path == ("06",))
    assert not completeness._is_marker_only_container(
        parent, branches=branches, source_text=source
    )


def test_irs_guidance_preserves_nested_numeric_outline_ancestors():
    source = (
        ".06 Earned Income Credit.\n(1) General rule.\n(a) First category.\n"
        "(i) Included group.\n(A) First classification.\n"
        "(1) Nested amount is $100.\n(2) Nested amount is $200.\n"
        "(B) Second classification.\n(ii) Other group.\n(b) Second category.\n"
        "(2) General second amount is $300."
    )
    branches = completeness.recognize_source_structure(
        source, corpus_citation_path=_CITATION
    )
    paths = [branch.path for branch in branches]
    assert len(paths) == len(set(paths))
    assert ("06", "1", "a", "i", "a", "1") in paths
    assert ("06", "1", "a", "i", "a", "2") in paths
    by_path = {branch.path: branch for branch in branches}
    assert "$200" not in by_path[("06", "2")].text
    assert "$300" in by_path[("06", "2")].text
