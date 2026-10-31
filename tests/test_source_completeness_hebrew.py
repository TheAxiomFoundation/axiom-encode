"""Hebrew deferral precision and outline recognition (issue #1882).

The replay fixture is the final rejected candidate of signed re-encode run
38485394523 for Israel's National Health Insurance Law section 14. Every one of
its eleven Hebrew deferral reasons was rejected as imprecise, and the Hebrew
outline was read as German ``Absatz`` branches. These tests run that candidate
through the CI method, then hold the Hebrew rules to the English ones: precise
Hebrew reasons are accepted and every English negative control has a Hebrew
counterpart that still fails.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import re
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import hebrew_negators
from axiom_encode.harness import hebrew_source_text as hebrew
from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness.source_completeness import (
    analyze_complete_source_unit,
    recognize_source_structure,
)
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline,
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

FIXTURE = (
    Path(__file__).parent / "fixtures/source_completeness/signed_reencode_38485394523"
)
CITATION = "il/statute/national-health-insurance-law-1994/section-14"
ANCHOR = "il:statutes/national-health-insurance-law-1994/section-14"
DEFERRAL = "[complete-source-unit:deferral]"
STRUCTURE_BRANCH = re.compile(
    r"\[complete-source-unit:structure\] Source branch \S+ at "
    + re.escape(CITATION)
    + r"(?P<path>(?:\([a-z0-9]+\))+) \["
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@functools.lru_cache(maxsize=None)
def _fixture() -> tuple[dict, str, str, str]:
    recorded = json.loads((FIXTURE / "issues.json").read_text())
    content = (FIXTURE / "candidate.yaml").read_text()
    tests = (FIXTURE / "candidate.test.yaml").read_text()
    source = (FIXTURE / "source.txt").read_text()
    metadata = json.loads((FIXTURE / "source-metadata.json").read_text())
    assert recorded["citation"] == CITATION
    assert _sha256(content) == recorded["rulespec_sha256"]
    assert _sha256(tests) == recorded["tests_sha256"]
    payload = yaml.safe_load(content)
    assert _sha256(source) == payload["module"]["source_verification"]["source_sha256"]
    assert _sha256(source) == metadata["source_attestation"]["resolved_text_sha256"]
    return recorded, content, tests, source


def _ci_issues(content: str, source: str, test_cases=None) -> list[str]:
    """Run `ValidatorPipeline._complete_source_unit_issues`, as CI validation does."""

    with tempfile.TemporaryDirectory() as root:
        pipeline = ValidatorPipeline(
            policy_repo_path=Path(root),
            axiom_rules_path=Path(root),
            local_corpus_release=None,
            enable_oracles=False,
            source_citation_path=CITATION,
            require_complete_source_unit=True,
        )
        return pipeline._complete_source_unit_issues(
            content,
            validation_source_texts={CITATION: source},
            test_cases=test_cases,
        )


@functools.lru_cache(maxsize=None)
def _replayed_issues() -> tuple[str, ...]:
    _recorded, content, tests, source = _fixture()
    return tuple(_ci_issues(content, source, yaml.safe_load(tests)))


def _deferral_issue(issues, index: int) -> str | None:
    marker = f"`module.deferred_outputs[{index}]"
    matches = [
        issue for issue in issues if issue.startswith(DEFERRAL) and marker in issue
    ]
    assert len(matches) <= 1
    return matches[0] if matches else None


def _structure_paths(issues) -> set[tuple[str, ...]]:
    return {
        tuple(re.findall(r"\(([a-z0-9]+)\)", match.group("path")))
        for issue in issues
        if (match := STRUCTURE_BRANCH.match(issue))
    }


def _anchored_candidate(anchors: dict[int, str]) -> str:
    _recorded, content, _tests, _source = _fixture()
    payload = yaml.safe_load(content)
    for index, branch in anchors.items():
        record = payload["module"]["deferred_outputs"][index]
        base, _separator, fragment = record["output"].partition("#")
        record["output"] = f"{base}/{branch}#{fragment}"
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)


# --- The recorded run --------------------------------------------------------


def test_recorded_run_rejected_every_hebrew_deferral():
    recorded, content, _tests, _source = _fixture()
    deferrals = yaml.safe_load(content)["module"]["deferred_outputs"]

    rejected = [
        issue
        for issue in recorded["issues"]
        if "[complete-source-unit:deferral]" in issue
        and "identifies source branch (source unit) (`<source-unit>`)" in issue
    ]
    assert len(deferrals) == len(rejected) == 11
    assert sum("[Absatz " in issue for issue in recorded["issues"]) == 2


@pytest.mark.parametrize(
    ("index", "branch", "dependency"),
    ((3, "e1", "345ב"), (5, "f1", "341"), (7, "g2", "350א(א)")),
)
def test_hebrew_reason_naming_an_applied_section_is_precise(index, branch, dependency):
    """The false rejection: these reasons name a section their branch applies."""

    _recorded, content, _tests, _source = _fixture()
    reason = yaml.safe_load(content)["module"]["deferred_outputs"][index]["reason"]
    assert f"סעיף {dependency}" in reason

    issue = _deferral_issue(_replayed_issues(), index)

    assert issue is not None
    assert "otherwise precise reason cites exact source branch" in issue
    assert f"(`{branch}`)" in issue
    assert f"`{ANCHOR}/{branch}#" in issue
    assert "(source unit)" not in issue


EXPECTED_VERDICTS = {
    0: ("explicit-computation", "b/2"),
    1: ("explicit-computation", "b/1"),
    2: ("imprecise", "d/1"),
    3: ("precise-but-rooted", "e1"),
    4: ("imprecise", "f"),
    5: ("precise-but-rooted", "f1"),
    6: ("imprecise", "g1"),
    7: ("precise-but-rooted", "g2"),
    8: ("imprecise", "h/1"),
    9: ("imprecise", "i/1"),
    10: ("imprecise", "j/1"),
}


@pytest.mark.parametrize(("index", "expected"), sorted(EXPECTED_VERDICTS.items()))
def test_each_recorded_deferral_gets_the_english_rule_verdict(index, expected):
    verdict, branch = expected
    issue = _deferral_issue(_replayed_issues(), index)

    assert issue is not None
    assert f"(`{branch}`)" in issue
    if verdict == "explicit-computation":
        assert "states an explicit computation" in issue
        assert "local RuleSpec inputs" in issue
    elif verdict == "precise-but-rooted":
        assert "otherwise precise reason" in issue
    else:
        assert "is rooted at the source unit and its reason cites" in issue
        assert "does not name an exact missing dependency" in issue


def test_precise_deferrals_cover_their_branches_once_anchored():
    _recorded, _content, tests, source = _fixture()
    anchored = _anchored_candidate(
        {
            0: "b/2",
            1: "b/1",
            2: "d",
            3: "e1",
            4: "f",
            5: "f1",
            6: "g1",
            7: "g2",
            8: "h",
            9: "i",
            10: "j",
        }
    )

    issues = _ci_issues(anchored, source, yaml.safe_load(tests))
    rejected = {
        index for index in range(11) if _deferral_issue(issues, index) is not None
    }
    uncovered = _structure_paths(issues)

    assert rejected == {0, 1, 2, 4, 6, 8, 9, 10}
    assert not {("e1",), ("f1",), ("g2",)} & uncovered
    assert {("b", "2"), ("f",), ("g1",), ("h",), ("j", "2")} <= uncovered


def test_structure_issues_name_gematria_branches_with_native_citations():
    issues = _replayed_issues()

    assert not any("Absatz" in issue for issue in issues)
    assert any(
        f"{CITATION}(b)(2) [סעיף 14(ב)(2)] is neither encoded nor precisely" in issue
        for issue in issues
    )
    assert any(f"{CITATION}(i)(1)(a) [סעיף 14(ט)(1)(א)]" in issue for issue in issues)
    # Encoded through proof excerpts or a native `source:` citation.
    assert not {("a",), ("b", "1"), ("c", "1"), ("c", "2"), ("e", "1"), ("e", "2")} & (
        _structure_paths(issues)
    )
    # Marker-only containers such as `(ב) (1)` are not demanded on their own.
    assert not {("b",), ("c",), ("d",), ("d", "2"), ("e",)} & _structure_paths(issues)


SECTION_14_PATHS = [
    "a",
    "b",
    "b/1",
    "b/2",
    "c",
    "c/1",
    "c/2",
    "d",
    "d/1",
    "d/2",
    "d/2/a",
    "d/2/b",
    "d/2/c",
    "d/2/d",
    "d/2/e",
    "d/2/f",
    "d/3",
    "d/4",
    "e",
    "e/1",
    "e/2",
    "e/3",
    "e/4",
    "e1",
    "f",
    "f1",
    "g",
    "g1",
    "g2",
    "h",
    "h/1",
    "h/2",
    "i",
    "i/1",
    "i/1/a",
    "i/1/b",
    "i/2",
    "j",
    "j/1",
    "j/2",
]


def test_section_14_outline_is_read_in_gematria_order():
    _recorded, _content, _tests, source = _fixture()

    branches = recognize_source_structure(source, corpus_citation_path=CITATION)

    assert ["/".join(branch.path) for branch in branches] == SECTION_14_PATHS
    by_path = {"/".join(branch.path): branch for branch in branches}
    assert by_path["b/2"].text.startswith("(2) מבוטח שהוא עובד")
    assert by_path["e1"].label == "(ה1)"
    assert by_path["d/2/f"].text.startswith("(ו) תגמול המשתלם לעובד המשרת במילואים")


# --- Labels and outline -----------------------------------------------------


@pytest.mark.parametrize(
    ("marker", "path_label"),
    (
        ("א", "a"),
        ("ב", "b"),
        ("ט", "i"),
        ("י", "j"),
        ("יא", "k"),
        ("יב", "l"),
        ("טו", "o"),
        ("טז", "p"),
        ("כ", "t"),
        ("כא", "u"),
        ("כו", "z"),
        ("כז", "aa"),
        ("כח", "ab"),
        ("ל", "ad"),
        ("לד", "ah"),
        ("ה1", "e1"),
        ("ז2", "g2"),
        ("2", "2"),
        ("2א", "2a"),
    ),
)
def test_labels_follow_the_ratified_gematria_scheme(marker, path_label):
    """Values 1-26 map to a-z and 27 up continue in bijective base 26."""

    label = hebrew.parse_outline_label(marker)

    assert label is not None
    assert label.path_label == path_label
    assert hebrew.native_path_label(path_label) == marker


@pytest.mark.parametrize(
    ("source", "paths"),
    (
        ("(י) הוראה.\n(יא) הוראה.\n(יב) הוראה.", [("j",), ("k",), ("l",)]),
        ("(י) הוראה.\n(י״א) הוראה.\n(יב) הוראה.", [("j",), ("k",), ("l",)]),
        ('(י) הוראה.\n(י"א) הוראה.\n(י"ב) הוראה.', [("j",), ("k",), ("l",)]),
        ("(ה׳) הוראה.\n(ו׳) הוראה.", [("e",), ("f",)]),
        ("(ב) הוראה.\n(2א׳) הוראה.", [("b",), ("b", "2a")]),
        ('(ב) הוראה.\n(2א") הוראה.', [("b",), ("b", "2a")]),
    ),
)
def test_geresh_and_gershayim_labels_keep_their_branch(source, paths):
    branches = recognize_source_structure(source, corpus_citation_path=CITATION)

    assert [branch.path for branch in branches] == paths
    assert hebrew.native_source_reference_paths("סעיף 14(ב)(2א׳)", CITATION) == {
        ("b", "2a")
    }
    assert completeness_module._reason_cites_exact_current_statute_branch(
        "סעיף 14(י״א) קובע",
        corpus_citation_path=CITATION,
        path=("k",),
        strict_terminal=True,
    )


@pytest.mark.parametrize(
    "word", ("של", "חוק", "חל", "יה", "יו", "אב", "ה0", "02", "קק", "ך", "מן", "2ך")
)
def test_parenthesized_words_are_not_outline_labels(word):
    assert hebrew.parse_outline_label(word) is None


@settings(deadline=None)
@given(st.integers(min_value=1, max_value=100_000))
def test_ordinal_path_labels_are_a_bijection(value):
    label = hebrew.ordinal_path_label(value)

    assert re.fullmatch(r"[a-z]+", label)
    assert hebrew.ordinal_path_value(label) == value


@settings(deadline=None)
@given(st.integers(min_value=1, max_value=999))
def test_hebrew_numerals_round_trip(value):
    assert hebrew.hebrew_numeral_value(hebrew.hebrew_numeral(value)) == value


@pytest.mark.parametrize(
    ("numeral", "value"),
    (("טו", 15), ("טז", 16), ("קטו", 115), ("רטז", 216), ("תתקטו", 915)),
)
def test_fifteen_and_sixteen_are_read_after_hundreds(numeral, value):
    assert hebrew.hebrew_numeral_value(numeral) == value


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet="אבגדהוזחטיכלמנסעפצקרשתךםןףץ", min_size=1, max_size=5))
def test_only_the_canonical_numeral_is_read(letters):
    """Injective: an accepted numeral is exactly the one that value renders as."""

    value = hebrew.hebrew_numeral_value(letters)

    if value is not None:
        assert hebrew.hebrew_numeral(value) == letters


@settings(deadline=None)
@given(
    st.integers(min_value=1, max_value=99),
    st.integers(min_value=0, max_value=9),
)
def test_letter_labels_round_trip_through_paths(value, suffix):
    marker = hebrew.hebrew_numeral(value) + (str(suffix) if suffix else "")
    label = hebrew.parse_outline_label(marker)

    assert label is not None
    assert hebrew.native_path_label(label.path_label) == marker
    assert hebrew.parse_outline_label(hebrew.native_path_label(label.path_label)) == (
        label
    )


def _outline_tree(draw, depth: int, kind: str):
    count = draw(st.integers(min_value=1, max_value=4))
    nodes = []
    for value in range(1, count + 1):
        labels = [(value, 0)]
        if kind == "letter" and draw(st.booleans()):
            labels.append((value, 1))
        for label in labels:
            children = (
                _outline_tree(
                    draw, depth + 1, "paragraph" if kind == "letter" else "letter"
                )
                if depth < 2 and draw(st.booleans())
                else []
            )
            nodes.append((kind, label, children))
    return nodes


def _follows(label, previous) -> bool:
    (value, suffix), (previous_value, previous_suffix) = label, previous
    return (value == previous_value and suffix == previous_suffix + 1) or (
        value == previous_value + 1 and suffix == 0
    )


def _open_subparagraph(node):
    """The label left open after `node`'s subtree when it is a subparagraph."""

    depth = 0
    while node[2]:
        node, depth = node[2][-1], depth + 1
    return node[1] if depth == 2 else None


def _disambiguated(nodes):
    """Give a childless subsection a paragraph when its label is ambiguous.

    In `(א) (1) (א) ...` then `(ב) ...` the `(ב)` continues both the
    subsection and the subparagraph sequence. The outline reads a line that
    opens a paragraph, `(ב) (1)`, as the subsection and any other as the
    subparagraph; `test_ambiguous_childless_letter_continues_the_subparagraph`
    pins that reading. The generator renders only unambiguous outlines.
    """

    repaired = []
    for kind, label, children in nodes:
        open_label = _open_subparagraph(repaired[-1]) if repaired else None
        if not children and open_label is not None and _follows(label, open_label):
            children = [("paragraph", (1, 0), [])]
        repaired.append((kind, label, children))
    return repaired


@st.composite
def _outlines(draw):
    return _disambiguated(_outline_tree(draw, 0, "letter"))


def _render_outline(nodes, parent=()):
    lines, paths = [], []
    for kind, (value, suffix), children in nodes:
        if kind == "letter":
            marker = hebrew.hebrew_numeral(value) + (str(suffix) if suffix else "")
            path_label = hebrew.ordinal_path_label(value) + (
                str(suffix) if suffix else ""
            )
        else:
            marker = str(value)
            path_label = str(value)
        path = (*parent, path_label)
        paths.append(path)
        child_lines, child_paths = _render_outline(children, path)
        if child_lines:
            # A parent opens its first child on its own line: `(ה) (1) ...`.
            lines.append(f"({marker}) {child_lines[0]}")
            lines.extend(child_lines[1:])
        else:
            lines.append(f"({marker}) הוראה")
        paths.extend(child_paths)
    return lines, paths


@settings(max_examples=200, deadline=None)
@given(_outlines())
def test_rendered_outlines_are_recovered_exactly(nodes):
    """Differential: a generated outline, rendered as a statute, parses back."""

    lines, paths = _render_outline(nodes)
    source = "\n".join(lines)

    markers = hebrew.hebrew_outline_markers(source)

    assert [path for _start, path, _kind, _label in markers] == paths
    branches = recognize_source_structure(source, corpus_citation_path=CITATION)
    assert [branch.path for branch in branches] == paths


@settings(max_examples=200, deadline=None)
@given(st.text(alphabet=st.characters(max_codepoint=0x024F), max_size=400))
def test_non_hebrew_text_never_uses_the_hebrew_outline(text):
    assert hebrew.hebrew_outline_markers(text) == ()
    assert hebrew.strip_leading_outline_chain(text) is None


@pytest.mark.parametrize(
    ("source", "paths"),
    (
        ("(א) הוראה.\n(ב)המוסד חייב בתשלום.", [("a",), ("b",)]),
        ("(א) הוראה.\n(ב)(1)המוסד חייב בתשלום.", [("a",), ("b",), ("b", "1")]),
        ("(א) הוראה.\n(1)המוסד חייב בתשלום.", [("a",), ("a", "1")]),
    ),
)
def test_a_marker_touching_its_text_keeps_its_branch(source, paths):
    branches = recognize_source_structure(source, corpus_citation_path=CITATION)

    assert [branch.path for branch in branches] == paths


def test_ambiguous_childless_letter_continues_the_subparagraph():
    """The intended reading of the case `_disambiguated` keeps out of the property."""

    source = "(א) (1) (א) הוראה\n(ב) הוראה"

    paths = [
        path for _start, path, _kind, _label in hebrew.hebrew_outline_markers(source)
    ]

    assert paths == [("a",), ("a", "1"), ("a", "1", "a"), ("a", "1", "b")]


def test_ambiguous_letter_prefers_the_subsection_that_opens_a_paragraph():
    source = "(ד) (1) רישה –\n(א) פרט\n(ב) פרט\n(ג) פרט\n(ד) פרט\n(ה) (1) הוראה"

    paths = [
        path for _start, path, _kind, _label in hebrew.hebrew_outline_markers(source)
    ]

    assert paths[-2:] == [("e",), ("e", "1")]
    assert ("d", "1", "d") in paths


def test_inline_references_and_repeal_notes_are_not_branches():
    source = (
        "(א) מבוטח ישלם כאמור בסעיף קטן (ב)(1) ולפי פסקה (2).\n"
        "(ב) (1) מעביד ינכה.\n"
        "(2) עובד ישלם.\n"
        "(ג) (בוטל).\n"
        "(ד) הוראה."
    )

    branches = recognize_source_structure(source, corpus_citation_path=CITATION)

    assert [branch.path for branch in branches] == [
        ("a",),
        ("b",),
        ("b", "1"),
        ("b", "2"),
        ("d",),
    ]


# --- Native citations -----------------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    ("חסר סעיף 346(1000).", "חסר סעיף 346 (1000).", "חסר סעיף 346(x)."),
)
def test_an_unreadable_fragment_is_not_read_as_the_whole_section(reason):
    source = "(א) הוראה.\n(ה1) ישולם לפי סעיף 346(1)."

    assert not hebrew.SECTION_DEPENDENCY.search(reason)
    assert not hebrew.reason_names_source_bound_dependency(
        reason, source, corpus_citation_path=CITATION
    )


@pytest.mark.parametrize(
    ("reason", "path", "cited"),
    (
        ("סעיף 14(ב)(2) מחייב עובד", ("b", "2"), True),
        ("לפי סעיף 14(ה1), חסר", ("e1",), True),
        ("כאמור בסעיף 14(ד)(1)", ("d", "1"), True),
        ("שבסעיף 14 (ט) (1) (א)", ("i", "1", "a"), True),
        ("סעיף 14(ב)(2) לחוק זה", ("b", "2"), True),
        ("סעיף 14(ב)(2) מחייב", ("b",), False),
        ("סעיף 14(ב)(2)(א) מחייב", ("b", "2"), False),
        ("סעיף 14(ב)(2) לחוק הביטוח הלאומי", ("b", "2"), False),
        ("סעיף 141(ב)(2)", ("b", "2"), False),
        ("סעיף 14א(ב)(2)", ("b", "2"), False),
        ("סעיף קטן (ב)(2)", ("b", "2"), False),
        ("section 14(b)(2)", ("b", "2"), False),
        (f"{CITATION}(b)(2) is missing", ("b", "2"), True),
    ),
)
def test_native_branch_citations_resolve_to_exact_paths(reason, path, cited):
    assert (
        completeness_module._reason_cites_exact_current_statute_branch(
            reason,
            corpus_citation_path=CITATION,
            path=path,
            strict_terminal=True,
        )
        is cited
    )


def test_native_citation_of_a_suffixed_section():
    citation = "il/statute/income-tax-ordinance/section-121b"

    assert hebrew.native_section_label(citation) == "121ב"
    assert completeness_module._reason_cites_exact_current_statute_branch(
        "סעיף 121ב(ה) קובע",
        corpus_citation_path=citation,
        path=("e",),
        strict_terminal=True,
    )
    assert not completeness_module._reason_cites_exact_current_statute_branch(
        "סעיף 121(ה) קובע",
        corpus_citation_path=citation,
        path=("e",),
        strict_terminal=True,
    )


def test_native_range_resolves_like_its_canonical_form():
    """`14(ד)(1)–(4)` binds the same branch as `.../section-14(d)(1)–(4)`."""

    _recorded, _content, _tests, source = _fixture()
    branches = recognize_source_structure(source, corpus_citation_path=CITATION)

    def cited(reason):
        return {
            branch.path
            for branch in branches
            if completeness_module._reason_cites_exact_current_statute_branch(
                reason,
                corpus_citation_path=CITATION,
                path=branch.path,
                strict_terminal=True,
            )
        }

    assert cited("סעיף 14(ד)(1)–(4) מחייב") == cited(f"{CITATION}(d)(1)–(4) requires")
    assert cited("סעיף 14(ד)(1)–(4) מחייב") == {("d", "1")}


# --- Missing-dependency vocabulary -------------------------------------------


@pytest.mark.parametrize(
    "text",
    (
        "חסר",
        "חסרה הוראת החישוב",
        "וחסרים נתוני השכר",
        "שחסרות",
        "החסר",
        "דורש",
        "דורשת",
        "דורשים",
        "תלוי בסעיף 345ב",
        "תלויה בהוראות",
        "עד שיקודד",
        "עד אשר יקודד",
        "בלי הוראה",
        "ללא הוראה",
        "טרם קודד",
        "עדיין לא קודד",
        "בלתי זמין",
        "בלתי זמינה",
        "לא ניתן לחשב",
    ),
)
def test_hebrew_missing_dependency_language(text):
    assert completeness_module._MISSING_DEPENDENCY_LANGUAGE.search(text)


@pytest.mark.parametrize(
    "text",
    (
        "מחסור",
        "חסרון",
        "חסרי בית",
        "דורשי עבודה",
        "בלימה",
        "עדכון",
        "סעיפים קטנים (ב) עד (ה1)",
        "עד ליום",
        "תלוי ועומד",
        "זמין",
        "מקודד",
        # "Subtraction" and "subtractions", not "missing".
        "החסרה של הסכום",
        "והחסרה",
        "שהחסרה",
        "החסרות של סכומים",
    ),
)
def test_hebrew_word_boundaries_reject_lookalikes(text):
    assert not completeness_module._MISSING_DEPENDENCY_LANGUAGE.search(text)


@pytest.mark.parametrize(
    ("hebrew_text", "english_text"),
    (
        ("אינו מקודד", "is not encoded"),
        ("לא קודד", "was not encoded"),
        ("אינו זמין", "is not available"),
        ("אינן זמינות", "are not available"),
    ),
)
def test_plain_negated_states_are_no_missing_signal_in_either_language(
    hebrew_text, english_text
):
    """English signals "not yet encoded" and "unavailable", not these."""

    assert not completeness_module._MISSING_DEPENDENCY_LANGUAGE.search(english_text)
    assert not completeness_module._MISSING_DEPENDENCY_LANGUAGE.search(hebrew_text)


@pytest.mark.parametrize(
    "pattern",
    (
        hebrew.MISSING_DEPENDENCY_LANGUAGE,
        hebrew.ADVERSATIVE_LANGUAGE,
        hebrew.RUNTIME_GAP_LANGUAGE,
        hebrew.EDITORIAL_OMISSION_WORDS,
        hebrew._SIGNAL.pattern,
        hebrew._EXPLICIT_COMPUTATION.pattern,
    ),
)
def test_vocabulary_writes_final_letters_only_at_word_end(pattern):
    """`זמין(?:ה)?` can never match `זמינה`: a final letter followed by more letters."""

    assert not re.search(r"[ךםןףץ](?:\(\?:)?[א-ת]", pattern)


# --- Deferral precision: Hebrew mirrors of the English controls --------------

PRECISION_SOURCE = """\
(א) בסעיף זה – "סכום המינימום" – סכום של 47 שקלים חדשים לחודש.
(ב) (1) מעביד חייב לשלם למוסד דמי ביטוח בריאות בעד כל עובד, בשיעור של 5.17% מהכנסת העובד.
(2) מבוטח שהוא עובד חייב לשלם למוסד דמי ביטוח בריאות בשיעור של 5.17% מהכנסתו.
(ג) המוסד רשאי לדרוש מהמבוטח להגיש העתק של הדוח השנתי.
(ה1) מבוטח שמשתלמת לו פנסיה מוקדמת כהגדרתה בסעיף 345ב לחוק הביטוח הלאומי, חייב לשלם למוסד דמי ביטוח בריאות, ויחולו לענין זה ההוראות לפי סעיף 345ב לחוק האמור, בשינויים המחויבים.
(ו) על אף האמור בסעיף 360 לחוק הביטוח הלאומי, ישולם סכום המינימום."""


def _quoted_reason(reason: str) -> str:
    """Preserve visible Unicode and escape every raw YAML control character."""

    # JSON ensure_ascii=True emits surrogate pairs that YAML keeps as two
    # code units; their interpretation has a separate folding/offset property.
    quoted = json.dumps(reason, ensure_ascii=False)
    return "".join(
        f"\\u{ord(char):04x}" if unicodedata.category(char) == "Cc" else char
        for char in quoted
    )


def _deferral_content(branch: str | None, reason: str) -> str:
    output = (
        f"{ANCHOR}/{branch}#deferred_output" if branch else f"{ANCHOR}#deferred_output"
    )
    return (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {CITATION}\n"
        "  deferred_outputs:\n"
        f"    - output: {output}\n"
        f"      reason: {_quoted_reason(reason)}\n"
        "rules: []\n"
    )


def _analyze(content: str, source: str = PRECISION_SOURCE) -> tuple[str, ...]:
    return analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=CITATION,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    ).issues


def _accepted(branch: str, reason: str, source: str = PRECISION_SOURCE) -> bool:
    issues = _analyze(_deferral_content(branch, reason), source)
    path = tuple(branch.split("/"))
    return _deferral_issue(issues, 0) is None and path not in _structure_paths(issues)


@pytest.mark.parametrize(
    "reason",
    (
        "סעיף 14(ה1) מחייב תשלום לפי הוראות סעיף 345ב לחוק הביטוח הלאומי בשינויים "
        "המחויבים; חסרה הוראת החישוב המדויקת שבסעיף 345ב.",
        "חסרות הוראות סעיף 345ב לחוק הביטוח הלאומי.",
        "החישוב טרם קודד לפי סעיף 345ב.",
        "לא ניתן לחשב את התשלום ללא הוראות סעיף 345ב.",
    ),
)
def test_hebrew_external_dependency_applied_by_the_branch_is_precise(reason):
    assert _accepted("e1", reason)


@pytest.mark.parametrize(
    "reason",
    (
        # Generic claims, the counterparts of "Nicht umgesetzt." and "unavailable".
        "הענף אינו זמין.",
        "לא מיושם.",
        "סעיף 14(ה1) אינו זמין.",
        # The branch's own section is not an external dependency.
        "סעיף 14(ה1) אינו מקודד.",
        "חסר סעיף 14(ה1).",
        # A section the branch does not apply.
        "חסרה הוראת החישוב שבסעיף 346 לחוק הביטוח הלאומי.",
        # The same section number in another instrument or another fragment.
        "חסרה הוראת החישוב שבסעיף 345ב לפקודת מס הכנסה.",
        "חסרה הוראת החישוב שבסעיף 345ב(ג).",
        # Negated, reversed, or with no governing signal.
        "לא חסרה הוראת החישוב שבסעיף 345ב.",
        "הוראה שלא חסרה בסעיף 345ב.",
        "חסרה הוראת החישוב שבסעיף 345ב, אך היא קודדה.",
        "סעיף 345ב לחוק הביטוח הלאומי חסר.",
        # `requires` and `until` need the dependency's own missing state.
        "החישוב דורש את סעיף 345ב לחוק הביטוח הלאומי.",
        "לא ניתן לחשב עד שסעיף 345ב לחוק הביטוח הלאומי יפורש.",
        # As in English, those routes read only the bare section followed by
        # its state: no law qualifier and no relative `, which is not encoded`.
        "לא ניתן לחשב את התשלום עד שיקודד סעיף 345ב לחוק הביטוח הלאומי.",
        "לא ניתן לחשב את התשלום עד שסעיף 345ב לחוק הביטוח הלאומי יקודד.",
        "לא ניתן לחשב את התשלום כי סעיף 345ב לחוק הביטוח הלאומי טרם קודד.",
        "החישוב תלוי בסעיף 345ב, שאינו מקודד.",
        # Except, unless, and a reversal in the next clause.
        "חסרות הוראות, למעט סעיף 345ב.",
        "חסר מידע אף שסעיף 345ב קודד.",
        "חסרה הוראת סעיף 345ב, אלא אם היא קודדה.",
        "חסרה הוראת סעיף 345ב. אולם היא זמינה.",
        # An earlier citation in the clause leaves `missing` ambiguous.
        "סעיף 341 וחסרה הוראת סעיף 345ב.",
    ),
)
def test_imprecise_hebrew_deferral_does_not_cover_its_branch(reason):
    assert not _accepted("e1", reason)


def test_notwithstanding_reference_is_not_an_applied_dependency():
    """`על אף האמור בסעיף 360` displaces section 360; it does not apply it."""

    assert not _accepted("f", "חסרה הוראת סעיף 360 לחוק הביטוח הלאומי.")


def test_imprecise_hebrew_deferral_retry_names_native_citation_and_scheme():
    issues = _analyze(_deferral_content("e1", "הענף אינו זמין."))
    issue = _deferral_issue(issues, 0)

    assert issue is not None
    assert f"`{CITATION}(e1)` or the native citation `סעיף 14(ה1)`" in issue
    assert "gematria" in issue
    assert "(ט)→i" in issue


def test_hebrew_runtime_gap_is_refused_although_english_admits_its_translation():
    """Counterpart of `test_exact_canonical_state_branch_can_name_source_bound_runtime_gap`.

    The runtime-gap route needs two shared source terms of five letters or
    more, counted in English. No Hebrew word count can promise that, since a
    long Hebrew word may translate to a short English one (`נתונים`, "data"),
    so a Hebrew runtime-gap reason is refused.
    """

    assert not _accepted(
        "c",
        "סעיף 14(ג) לא ניתן לקודד: חסרה יכולת לייצג את הגשת העתק הדוח השנתי של המבוטח.",
    )
    assert _english_accepted(
        "c",
        "us-nj/statute/54a:8-6(c) cannot be encoded: missing a capability to "
        "represent the submission of a copy of the annual report of the insured.",
    )


@pytest.mark.parametrize(
    "term", ("נתונים", "טופס", "עובדה", "תאריך", "תכנית", "עותק", "עבודה", "חובה")
)
def test_a_term_with_a_short_translation_cannot_carry_a_runtime_gap(term):
    """Round-4 review: each of these is "data", "form", "fact", ... in English."""

    source = f"(א) הוראה.\n(ג) המוסד רשאי להגיש דוח על {term}."
    issues = _analyze(
        _deferral_content("c", f"סעיף 14(ג): חסר מידע על {term} של המוסד."), source
    )

    assert _deferral_issue(issues, 0) is not None


@pytest.mark.parametrize(
    ("branch", "reason"),
    (
        # The citation must be this exact branch of this law.
        (
            "c",
            "סעיף 14(ג) לחוק הביטוח הלאומי: חסרה יכולת לייצג את הגשת העתק הדוח השנתי.",
        ),
        ("c", "חסרה יכולת לייצג את הגשת העתק הדוח השנתי של המבוטח."),
        # Two shared source terms keep a vague capability claim out.
        ("c", "סעיף 14(ג) לא ניתן לקודד: חסר מידע."),
        # A branch that states a computation is never a runtime gap.
        ("b/2", "סעיף 14(ב)(2) לא ניתן לקודד: חסרים נתוני הכנסתו של המבוטח."),
    ),
)
def test_hebrew_runtime_gap_controls(branch, reason):
    assert not _accepted(branch, reason)


def test_root_hebrew_runtime_gap_on_computation_names_the_cited_branch():
    """Counterpart of `test_root_state_deferral_retry_rejects_explicit_computation_runtime_gap`."""

    issues = _analyze(
        _deferral_content(
            None, "סעיף 14(ב)(2) מחייב עובד לשלם; חסרים נתוני הכנסתו של המבוטח."
        )
    )
    issue = _deferral_issue(issues, 0)

    assert issue is not None
    assert "cites exact source branch (`b/2`)" in issue
    assert "states an explicit computation" in issue
    assert f"{ANCHOR}/b/2#" not in issue


def test_root_precise_hebrew_deferral_retry_preserves_the_branch():
    issues = _analyze(
        _deferral_content(
            None,
            "סעיף 14(ה1) מחייב תשלום לפי סעיף 345ב; חסרה הוראת החישוב שבסעיף 345ב.",
        )
    )
    issue = _deferral_issue(issues, 0)

    assert issue is not None
    assert "otherwise precise reason cites exact source branch (`e1`)" in issue
    assert f"`{ANCHOR}/e1#deferred_output`" in issue


def test_self_citation_in_the_missing_clause_is_ambiguous_as_in_english():
    """`<branch>: missing <section>` fails in English too: the clause has two citations."""

    issues = _analyze(
        _deferral_content(None, "סעיף 14(ה1): חסרה הוראת החישוב שבסעיף 345ב.")
    )
    issue = _deferral_issue(issues, 0)

    assert issue is not None
    assert "its reason cites exact source branch (`e1`)" in issue
    assert "does not name an exact missing dependency" in issue


def test_root_imprecise_deferral_names_the_branch_its_reason_cites():
    issues = _analyze(_deferral_content(None, "סעיף 14(ה1) אינו זמין."))
    issue = _deferral_issue(issues, 0)

    assert issue is not None
    assert (
        "is rooted at the source unit and its reason cites exact source branch" in issue
    )
    assert "(`e1`)" in issue
    assert f"`{ANCHOR}/e1#deferred_output`" in issue
    assert "(source unit) (`<source-unit>`)" not in issue


def test_root_imprecise_english_deferral_names_its_cited_branch_too():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-la/statute/47:295
  deferred_outputs:
    - output: us-la:statutes/47/295#federal_return_copy
      reason: R.S. 47:295(C) is unavailable.
rules: []
"""
    source = (
        "A. The secretary shall file an annual report.\n\n"
        "B. The secretary shall retain the audit record.\n\n"
        "C. The secretary may require that a copy of the federal return be filed."
    )

    issues = analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path="us-la/statute/47:295",
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    ).issues
    issue = _deferral_issue(issues, 0)

    assert issue is not None
    assert "its reason cites exact source branch (`c`)" in issue
    assert "`us-la:statutes/47/295/c#federal_return_copy`" in issue
    assert "gematria" not in issue


def test_the_signal_to_citation_bridge_is_bounded_for_translation():
    """A bridge is accepted up to `_BRIDGE_LIMIT` characters and refused past it."""

    verdicts = set()
    for words in range(8, 20):
        reason = "חסרות הוראות" + " החישוב" * words + " שבסעיף 345ב."
        bridge = reason.index("סעיף") - len("חסרות")
        accepted = _accepted("e1", reason)
        assert accepted is (bridge <= hebrew._BRIDGE_LIMIT), (words, bridge)
        verdicts.add(accepted)
    assert verdicts == {True, False}
    # Within English's 240 characters at twice the Hebrew length.
    assert hebrew._BRIDGE_LIMIT * 2 <= 240


_NEUTRAL_WORDS = (
    "ברור",
    "נכון",
    "ידוע",
    "הוכח",
    "נטען",
    "סביר",
    "להניח",
    "לומר",
    "לקבוע",
    "כל",
    "ראיה",
    "לכך",
    "מקום",
    "סיבה",
    "טענה",
    "במקרה",
    "הזה",
    # Words spelled with a negation's prefix letters that negate nothing:
    # "full", "prison", "surely", "before", "whence".
    "מלא",
    "כלא",
    "הלא",
    "בטרם",
    "מאין",
)
_PRECISE_MISSING_CLAIMS = (
    "חסרה הוראת סעיף 345ב.",
    "חסרות הוראות סעיף 345ב לחוק הביטוח הלאומי.",
    "חסרה הוראת החישוב המדויקת שבסעיף 345ב.",
)
_NEGATORS = ("לא", "אין", "אינו", "טרם", "מעולם לא", "לאו", "אי אפשר", "בלתי")
# The prefixes of ``_NEGATION_PREFIX``: ``ואין``, ``משלא``, ``וכשלא``, ``לכשלא``,
# and the Aramaic relative of ``דלא`` and ``ודלא`` (review round 13).
_NEGATION_PREFIXES = (
    "",
    "ו",
    "ש",
    "וש",
    "כש",
    "וכש",
    "מש",
    "ומש",
    "לכש",
    "ולכש",
    "בש",
    "ד",
    "וד",
)
# Points, cantillation, and direction and format marks: none changes a word.
# Every mark the fold drops, by Unicode property (review round 14):
# ``test_the_fold_drops_every_format_character_and_hebrew_mark`` pins the set.
# The surrogates fold too, but no YAML text holds one: they reach a reason
# only as YAML's reading of an escaped astral mark, which
# ``test_an_escaped_astral_mark_in_yaml_hides_no_negation`` covers.
_SURROGATES = frozenset(range(0xD800, 0xE000))
_INVISIBLE_MARKS = tuple(map(chr, sorted(hebrew.folded_marks() - _SURROGATES)))


@st.composite
def _marked(draw, text: str) -> str:
    """``text`` with any of its characters followed by an invisible mark."""

    marks = draw(
        st.lists(
            st.one_of(st.just(""), st.sampled_from(_INVISIBLE_MARKS)),
            min_size=len(text),
            max_size=len(text),
        )
    )
    return "".join(char + mark for char, mark in zip(text, marks, strict=True))


@st.composite
def _negations(draw) -> str:
    """A negation, under any of its prefixes, pointed or direction-marked."""

    negation = draw(st.sampled_from(_NEGATION_PREFIXES)) + draw(
        st.sampled_from(_NEGATORS)
    )
    return draw(_marked(negation))


@settings(max_examples=80, deadline=None)
@given(
    _negations(),
    st.lists(st.sampled_from(_NEUTRAL_WORDS), max_size=8),
    st.sampled_from(_PRECISE_MISSING_CLAIMS),
)
def test_a_negation_earlier_in_the_clause_governs_the_missing_claim(
    negation, words, claim
):
    """Polarity: a precise missing claim under any complement of a negation fails.

    The complement is worded freely, so no list of bridge words decides it; the
    same complement without the negation leaves the claim precise. The
    negation may take its prefixes (``משלא``, ``וכשלא``) and carry points or
    direction marks (review round 12), and a complement word that only
    starts with a prefix's letters (``מלא``, "full") negates nothing.
    """

    complement = " ".join(words)
    affirmed = f"{complement} ש{claim}".strip()
    negated = f"{negation} {affirmed}"

    assert _accepted("e1", affirmed)
    assert not _accepted("e1", negated)


# --- The English rule on the same text -----------------------------------------

ENGLISH_CITATION = "us-nj/statute/54a:8-6"
ENGLISH_ANCHOR = "us-nj:statutes/54a:8-6"
# `PRECISION_SOURCE` in English, with the recognizer's plain letters for
# (ה1) -> (d) and (ו) -> (e).
ENGLISH_PRECISION_SOURCE = """\
(a) In this section, "minimum amount" means 47 shekels a month.
(b)(1) An employer shall pay the Institute health insurance contributions for each employee, at a rate of 5.17% of the employee's income.
(2) An insured employee shall pay the Institute health insurance contributions at a rate of 5.17% of the employee's income.
(c) The Institute may require the insured to submit a copy of the annual report.
(d) An insured who receives an early pension as defined in section 345b of the National Insurance Law shall pay the Institute health insurance contributions, and the provisions under section 345b of the said law shall apply, with the necessary changes.
(e) Notwithstanding section 360 of the National Insurance Law, the minimum amount shall be paid."""
ENGLISH_BRANCH = {"b/2": "b/2", "c": "c", "e1": "d", "f": "e"}


def _english_accepted(
    branch: str, reason: str, source: str = ENGLISH_PRECISION_SOURCE
) -> bool:
    content = (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {ENGLISH_CITATION}\n"
        "  deferred_outputs:\n"
        f"    - output: {ENGLISH_ANCHOR}/{branch}#deferred_output\n"
        f"      reason: {_quoted_reason(reason)}\n"
        "rules: []\n"
    )
    issues = analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=ENGLISH_CITATION,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    ).issues
    uncovered = f"{ENGLISH_CITATION}" + "".join(
        f"({part})" for part in branch.split("/")
    )
    return _deferral_issue(issues, 0) is None and not any(
        issue.startswith("[complete-source-unit:structure]")
        and f" at {uncovered} [" in issue
        for issue in issues
    )


# (Hebrew branch, Hebrew reason, English reason, verdict). `hebrew-stricter`
# marks the intended cases where the Hebrew rule refuses what the English one
# admits. No case passes Hebrew alone.
DIFFERENTIAL_CASES = (
    (
        "e1",
        "סעיף 14(ה1) מחייב תשלום לפי הוראות סעיף 345ב לחוק הביטוח הלאומי בשינויים "
        "המחויבים; חסרה הוראת החישוב המדויקת שבסעיף 345ב.",
        "us-nj/statute/54a:8-6(d) requires payment under section 345b of the "
        "National Insurance Law with the necessary changes; missing the exact "
        "computation rule in section 345b.",
        "both",
    ),
    (
        "e1",
        "חסרות הוראות סעיף 345ב לחוק הביטוח הלאומי.",
        "missing the provisions of section 345b of the National Insurance Law.",
        "both",
    ),
    (
        "e1",
        "לא ניתן לחשב את התשלום עד שיקודד סעיף 345ב.",
        "the payment cannot be computed until section 345b is encoded.",
        "hebrew-stricter",
    ),
    (
        "e1",
        "לא ניתן לחשב את התשלום עד שיקודד סעיף 345ב לחוק הביטוח הלאומי.",
        "the payment cannot be computed until section 345b of the National "
        "Insurance Law is encoded.",
        "neither",
    ),
    (
        "e1",
        "החישוב תלוי בסעיף 345ב טרם קודד.",
        "The calculation depends on section 345b is not encoded.",
        "hebrew-stricter",
    ),
    (
        "e1",
        "החישוב תלוי בסעיף 345ב, שאינו מקודד.",
        "The calculation depends on section 345b, which is not encoded.",
        "neither",
    ),
    (
        "e1",
        "החישוב תלוי בסעיף 345ב, וסעיף 345ב טרם קודד.",
        "The calculation depends on section 345b and section 345b is not yet encoded.",
        "hebrew-stricter",
    ),
    (
        "e1",
        "חסרות הוראות, למעט סעיף 345ב.",
        "Missing provisions, except section 345b.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע אף שסעיף 345ב קודד.",
        "Missing information although section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "חסרה הוראת סעיף 345ב. אולם היא זמינה.",
        "Missing the rule in section 345b. However, it is available.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע אף כי סעיף 345ב קודד.",
        "Missing information although section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "חסרות הוראות מלבד סעיף 345ב.",
        "Missing provisions except section 345b.",
        "neither",
    ),
    (
        "e1",
        "החישוב שתלוי בסעיף 345ב טרם קודד.",
        "The calculation that depends on section 345b is not yet encoded.",
        "neither",
    ),
    (
        "e1",
        "החישוב התלוי בסעיף 345ב טרם קודד.",
        "The calculation that depends on section 345b is not yet encoded.",
        "neither",
    ),
    (
        "e1",
        "חסרות הוראות, להוציא סעיף 345ב.",
        "Missing provisions, except section 345b.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע, ואילו סעיף 345ב קודד.",
        "Missing information, whereas section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע, בעוד סעיף 345ב קודד.",
        "Missing information, while section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע, אף על פי כן סעיף 345ב קודד.",
        "Missing information, nevertheless section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "לא ניתן לחשב עד שיקודד החישוב החודשי הקבוע בסעיף 345ב.",
        "Cannot be computed until the monthly calculation set in section 345b is "
        "encoded.",
        "neither",
    ),
    (
        "e1",
        "לא ניתן לקבוע את מועד התשלום כי סעיף 345ב טרם קודד.",
        "The payment date cannot be determined because section 345b is not yet "
        "encoded.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע, ועם זאת סעיף 345ב קודד.",
        "Missing information, and nevertheless section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע, ועל אף זאת סעיף 345ב קודד.",
        "Missing information, and despite that section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע אף־על־פי שסעיף 345ב קודד.",
        "Missing information although section 345b is encoded.",
        "neither",
    ),
    ("e1", "Absatz 9: חסר סעיף 345ב.", "Absatz 9: missing section 345b.", "neither"),
    ("e1", "42 USC 9: חסר סעיף 345ב.", "42 USC 9: missing section 345b.", "neither"),
    (
        "e1",
        "il/statute/context. חסר סעיף 345ב.",
        "il/statute/context. Missing section 345b.",
        "both",
    ),
    (
        "e1",
        "לא ניתן לחשב עד שתקודד הקצבה הקבועה בסעיף 345ב.",
        "Cannot be computed until the allowance set in section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "לא היה חסר הכלל שבסעיף 345ב.",
        "The rule in section 345b was not missing.",
        "neither",
    ),
    (
        "e1",
        "חסרות הוראות סעיפים 345ב ו-345ג.",
        "Missing the provisions of sections 345b and 345c.",
        "neither",
    ),
    (
        "e1",
        "אינה יכולה להיות חסרה הוראת סעיף 345ב.",
        "The provision of section 345b cannot be missing.",
        "neither",
    ),
    (
        "e1",
        "אין זה נכון שחסרה הוראת סעיף 345ב.",
        "It is not true that the provision of section 345b is missing.",
        "neither",
    ),
    (
        "e1",
        "לא אמורה בכלל להיות חסרה הוראת סעיף 345ב.",
        "The provision of section 345b is not supposed to be missing at all.",
        "neither",
    ),
    (
        "e1",
        "לא־חסרה הוראת סעיף 345ב.",
        "The provision of section 345b is not missing.",
        "neither",
    ),
    (
        "e1",
        "חסרות הוראות סעיף 345b.",
        "Missing the provisions of section 345b.",
        "hebrew-stricter",
    ),
    (
        "e1",
        "לא אמורה להיות חסרה הוראת סעיף 345ב.",
        "The provision of section 345b is not supposed to be missing.",
        "neither",
    ),
    (
        "e1",
        "לא צפויים להיות חסרים כללי סעיף 345ב.",
        "The rules of section 345b are not expected to be missing.",
        "neither",
    ),
    (
        "e1",
        "לא" + " " * 60 + "היה חסר הכלל שבסעיף 345ב.",
        "The rule in section 345b was not missing.",
        "neither",
    ),
    (
        "e1",
        "לא בהכרח חסרה הוראת החישוב שבסעיף 345ב.",
        "The computation rule in section 345b is not necessarily missing.",
        "neither",
    ),
    (
        "e1",
        "לא יהיו חסרות הוראות סעיף 345ב.",
        "The provisions of section 345b will not be missing.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע, לעומת זאת סעיף 345ב קודד.",
        "Missing information, however section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע, ומנגד סעיף 345ב קודד.",
        "Missing information, and whereas section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "חסרות אך ורק הוראות סעיף 345ב.",
        "Missing only the provisions of section 345b.",
        "both",
    ),
    (
        "e1",
        "איננה חסרה הוראת החישוב שבסעיף 345ב.",
        "The computation rule in section 345b is not missing.",
        "neither",
    ),
    (
        "e1",
        "לא ניתן להכריע כי סעיף 345ב טרם קודד.",
        "Cannot be decided because section 345b is not yet encoded.",
        "neither",
    ),
    (
        "e1",
        "לא ניתן לחשב עד שיקודדו התנאים בהגשה הקבועים בסעיף 345ב.",
        "Cannot be computed until the conditions in filing set in section 345b "
        "are encoded.",
        "neither",
    ),
    (
        "e1",
        "לא ניתן לחשב עד שיקודדו התנאים לזכאות הקבועים בסעיף 345ב.",
        "Cannot be computed until the conditions for eligibility set in section "
        "345b are encoded.",
        "neither",
    ),
    (
        "e1",
        "42 U.S.C. 9(a): חסר סעיף 345ב.",
        "42 U.S.C. 9(a): missing section 345b.",
        "neither",
    ),
    (
        "e1",
        "42 USC sections 9: חסר סעיף 345ב.",
        "42 USC sections 9: missing section 345b.",
        "neither",
    ),
    (
        "e1",
        "לא ניתן לחשב עד שיקודד החישוב השנתי הקבוע בסעיף 345ב.",
        "Cannot be computed until the annual calculation set in section 345b is "
        "encoded.",
        "hebrew-stricter",
    ),
    (
        "e1",
        "החישוב אינו מקודד לפי סעיף 345ב.",
        "The calculation is not encoded under section 345b.",
        "neither",
    ),
    (
        "e1",
        "החישוב טרם קודד לפי סעיף 345ב.",
        "The calculation is not yet encoded under section 345b.",
        "both",
    ),
    (
        "e1",
        "לא ניתן לחשב את התשלום עד אשר יקודד סעיף 345ב.",
        "The payment cannot be computed until section 345b is encoded.",
        "hebrew-stricter",
    ),
    (
        "e1",
        "לא ניתן לחשב עד שיקודדו התנאים הקבועים בסעיף 345ב.",
        "Cannot be computed until the conditions set in section 345b are encoded.",
        "hebrew-stricter",
    ),
    (
        "e1",
        "לא ניתן לחשב עד שיקודדו הוראות החישוב המדויקות שבסעיף 345ב.",
        "Cannot be computed until the exact computation rules in section 345b "
        "are encoded.",
        "neither",
    ),
    (
        "e1",
        "לא ניתן לחשב את התשלום ללא הוראות סעיף 345ב.",
        "Cannot be computed without the provisions of section 345b.",
        "both",
    ),
    ("e1", "הענף אינו זמין.", "The branch is unavailable.", "neither"),
    (
        "e1",
        "סעיף 14(ה1) אינו מקודד.",
        "us-nj/statute/54a:8-6(d) is not yet encoded.",
        "neither",
    ),
    (
        "e1",
        "חסרה הוראת החישוב שבסעיף 346 לחוק הביטוח הלאומי.",
        "missing the computation rule in section 346 of the National Insurance Law.",
        "neither",
    ),
    (
        "e1",
        "לא חסרה הוראת החישוב שבסעיף 345ב.",
        "the computation rule in section 345b is not missing.",
        "neither",
    ),
    (
        "e1",
        "החישוב דורש את סעיף 345ב לחוק הביטוח הלאומי.",
        "the computation requires section 345b of the National Insurance Law.",
        "neither",
    ),
    (
        "e1",
        "סעיף 341 וחסרה הוראת סעיף 345ב.",
        "section 341 and missing the rule in section 345b.",
        "neither",
    ),
    (
        "e1",
        "סעיף 14(ה1): חסרה הוראת החישוב שבסעיף 345ב.",
        "us-nj/statute/54a:8-6(d): missing the computation rule in section 345b.",
        "neither",
    ),
    (
        "e1",
        "חסרה הוראת החישוב שבסעיף 345ב לפקודת מס הכנסה.",
        "missing the computation rule in section 345b of the Income Tax Ordinance.",
        "hebrew-stricter",
    ),
    (
        "e1",
        "חסרה הוראת החישוב שבסעיף 345ב, אך היא קודדה.",
        "missing the computation rule in section 345b, but it has been encoded.",
        "hebrew-stricter",
    ),
    (
        "f",
        "חסרה הוראת סעיף 360 לחוק הביטוח הלאומי.",
        "missing the rule in section 360 of the National Insurance Law.",
        "neither",
    ),
    (
        "b/2",
        "סעיף 14(ב)(2) לא ניתן לקודד: חסרים נתוני הכנסתו של המבוטח.",
        "us-nj/statute/54a:8-6(b)(2) cannot be encoded: missing data on the "
        "insured's income.",
        "neither",
    ),
    (
        "c",
        "סעיף 14(ג) לא ניתן לקודד: חסרה יכולת לייצג את הגשת העתק הדוח השנתי של המבוטח.",
        "us-nj/statute/54a:8-6(c) cannot be encoded: missing a capability to "
        "represent the submission of a copy of the annual report of the insured.",
        "hebrew-stricter",
    ),
    (
        "c",
        "סעיף 14(ג) לא ניתן לקודד: חסר מידע.",
        "us-nj/statute/54a:8-6(c) cannot be encoded: missing information.",
        "neither",
    ),
    (
        "c",
        "חסרה יכולת לייצג את הגשת העתק הדוח השנתי של המבוטח.",
        "missing a capability to represent the submission of a copy of the annual "
        "report of the insured.",
        "neither",
    ),
    # Two shared source terms: the same term twice, or `copy`/`העתק`, is not two.
    (
        "c",
        "סעיף 14(ג): חסר מידע מהמבוטח על המבוטח.",
        "us-nj/statute/54a:8-6(c): missing information from the insured about "
        "the insured.",
        "neither",
    ),
    (
        "c",
        "סעיף 14(ג): חסרה יכולת לייצג את העתק הדוח.",
        "us-nj/statute/54a:8-6(c): missing capability to represent the copy of "
        "the report.",
        "neither",
    ),
    (
        "c",
        "סעיף 14(ג): חסר מידע על מה שהמוסד רשאי לעשות.",
        "us-nj/statute/54a:8-6(c): missing information about what the Institute "
        "may do.",
        "neither",
    ),
    # Round 11: adversatives that are also ordinary words, with no punctuation.
    (
        "e1",
        "חסר מידע אולם סעיף 345ב קודד.",
        "Missing information however section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע אבל סעיף 345ב קודד.",
        "Missing information but section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "חסרות הוראות להוציא סעיף 345ב.",
        "Missing provisions except section 345b.",
        "neither",
    ),
    (
        "e1",
        "חסרות הוראות זולת סעיף 345ב.",
        "Missing provisions except section 345b.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע בעוד סעיף 345ב קודד.",
        "Missing information while section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע,   אולם סעיף 345ב קודד.",
        "Missing information,   however section 345b is encoded.",
        "neither",
    ),
    # The ordinary words where their form or governor fixes them.
    (
        "e1",
        "חסר מידע כדי להוציא רישיון לפי סעיף 345ב.",
        "Missing information to issue a license under section 345b.",
        "both",
    ),
    (
        "e1",
        "חסרה הוראת סעיף 345ב, שתחול בעוד שנה.",
        "missing the provision of section 345b, which will apply in a year.",
        "both",
    ),
    # Negated assertion, evidence and epistemic complements.
    (
        "e1",
        "אין סיבה להניח שחסרה הוראת סעיף 345ב.",
        "There is no reason to assume that the provision of section 345b is missing.",
        "neither",
    ),
    (
        "e1",
        "אין מקום לקבוע כי חסרה הוראת סעיף 345ב.",
        "There is no ground to hold that the provision of section 345b is missing.",
        "neither",
    ),
    (
        "e1",
        "לא ניתן לומר שחסרה הוראת סעיף 345ב.",
        "It cannot be said that the provision of section 345b is missing.",
        "neither",
    ),
    (
        "e1",
        "לא ברור שחסרה הוראת סעיף 345ב.",
        "It is unclear that the provision of section 345b is missing.",
        "neither",
    ),
    (
        "e1",
        "אין כל ראיה לכך שחסרה הוראת סעיף 345ב.",
        "There is no evidence that the provision of section 345b is missing.",
        "neither",
    ),
    (
        "e1",
        "אין טענה שחסרה הוראת סעיף 345ב.",
        "There is no claim that the provision of section 345b is missing.",
        "neither",
    ),
    (
        "e1",
        "לא הוכח שחסרה הוראת סעיף 345ב.",
        "It has not been shown that the provision of section 345b is missing.",
        "neither",
    ),
    (
        "e1",
        "מעולם לא נטען שחסרה הוראת סעיף 345ב.",
        "It was never claimed that the provision of section 345b is missing.",
        "neither",
    ),
    (
        "e1",
        "טרם הוכח שחסרה הוראת סעיף 345ב.",
        "It has not yet been shown that the provision of section 345b is missing.",
        "neither",
    ),
    (
        "e1",
        "לאו דווקא חסרה הוראת סעיף 345ב.",
        "The provision of section 345b is not necessarily missing.",
        "neither",
    ),
    # A negation between the signal and the citation: refused in Hebrew only.
    (
        "e1",
        "חסר מידע, לא ברור שחסרה הוראת סעיף 345ב.",
        "Missing information, it is unclear that the provision of section 345b is "
        "missing.",
        "hebrew-stricter",
    ),
    # A bridge that fits English's 240 characters only before translation.
    (
        "e1",
        "חסרות הוראות החישוב המפורטות לקביעת בסיס ההכנסה החייבת בדמי "
        "ביטוח בריאות לגבי מבוטח המשתלמת לו פנסיה מוקדמת, לרבות ניכויים, "
        "פטורים, תקרות, הכנסות נוספות, תקופות ביטוח וסכומי מינימום, "
        "בהתאם לסעיף 345ב.",
        "Missing the detailed computation provisions for determining the "
        "income base subject to health insurance contributions for an insured "
        "person receiving an early pension, including deductions, exemptions, "
        "ceilings, additional income, insurance periods and minimum amounts, "
        "in accordance with section 345b.",
        "neither",
    ),
    # `החסרה` with the article is "subtraction", not "missing".
    (
        "e1",
        "החסרה של הסכום לפי סעיף 345ב אינה נתמכת.",
        "Subtraction of the amount under section 345b is unsupported.",
        "neither",
    ),
    (
        "e1",
        "נדרשת החסרה של סכומים לפי סעיף 345ב.",
        "A subtraction of amounts under section 345b is required.",
        "neither",
    ),
    (
        "e1",
        "פעולת החסרה לפי סעיף 345ב.",
        "A subtraction operation under section 345b.",
        "neither",
    ),
    # Review round 12. Points, cantillation and direction marks hide no
    # negation or adversative: each reads as its unmarked word.
    ("e1", "ל\u05b9א חסר סעיף 345ב.", "Section 345b is not missing.", "neither"),
    ("e1", "ל\u200fא חסר סעיף 345ב.", "Section 345b is not missing.", "neither"),
    ("e1", "\u202bלא\u202c חסר סעיף 345ב.", "Section 345b is not missing.", "neither"),
    (
        "e1",
        "אי\u2067נ\u2069ו חסר סעיף 345ב.",
        "Section 345b is not missing.",
        "neither",
    ),
    (
        "e1",
        "א\u05b5ין חסרה הוראת סעיף 345ב.",
        "The provision of section 345b is not missing.",
        "neither",
    ),
    (
        "e1",
        "למרו\u05b9ת שחסר סעיף 345ב.",
        "Although section 345b is missing.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע, א\u05b2ב\u05b8ל סעיף 345ב קודד.",
        "Missing information, but section 345b is encoded.",
        "neither",
    ),
    (
        "e1",
        "חסר מידע, א\u200fולם סעיף 345ב קודד.",
        "Missing information, however section 345b is encoded.",
        "neither",
    ),
    # Nor do they hide a signal or a citation.
    ("e1", "ח\u05b8ס\u05b5ר סעיף 345ב.", "Missing section 345b.", "both"),
    ("e1", "חס\u200fר סעיף 345ב.", "Missing section 345b.", "both"),
    (
        "e1",
        "חסרות הוראות סעיף 345ב לחו\u05b9ק הביטוח הלאומי.",
        "missing the provisions of section 345b of the National Insurance Law.",
        "both",
    ),
    # Negations under their prefixes, and ``בלתי``.
    ("e1", "משלא חסר סעיף 345ב.", "Since section 345b is not missing.", "neither"),
    ("e1", "וכשלא חסר סעיף 345ב.", "And when section 345b is not missing.", "neither"),
    (
        "e1",
        "לכשלא חסרה הוראת סעיף 345ב.",
        "Once the provision of section 345b is not missing.",
        "neither",
    ),
    (
        "e1",
        "ומשאין חסרה הוראת סעיף 345ב.",
        "And since the provision of section 345b is not missing.",
        "neither",
    ),
    (
        "e1",
        "בלתי נכון שחסר סעיף 345ב.",
        "It is not true that section 345b is missing.",
        "neither",
    ),
    (
        "e1",
        "ובלתי סביר שחסרה הוראת סעיף 345ב.",
        "And it is implausible that the provision of section 345b is missing.",
        "neither",
    ),
    # ``בלתי`` between the signal and the citation leaves the claim unread,
    # as ``לא ברור`` does: refused in Hebrew, though English admits "unclear".
    (
        "e1",
        "חסרה הוראה בלתי ברורה של סעיף 345ב.",
        "Missing an unclear provision of section 345b.",
        "hebrew-stricter",
    ),
    # A word spelled with a negation's prefix letters negates nothing.
    (
        "e1",
        "חסר נוסח מלא של סעיף 345ב.",
        "Missing the full text of section 345b.",
        "both",
    ),
    # A reversal in English or German, read with English's own vocabulary.
    (
        "e1",
        "חסר סעיף 345ב. However it is available.",
        "Missing section 345b. However it is available.",
        "neither",
    ),
    (
        "e1",
        "חסר סעיף 345ב. Aber er ist verfügbar.",
        "Missing section 345b. Aber er ist verfügbar.",
        "neither",
    ),
    # The "yet" of "not yet" is no adversative, in English either.
    (
        "e1",
        "חסר סעיף 345ב, not yet encoded.",
        "Missing section 345b, not yet encoded.",
        "both",
    ),
    # An adversative later in the citation's own clause: refused in Hebrew,
    # as a Hebrew connective there is, though English admits it.
    (
        "e1",
        "חסר סעיף 345ב, although it is available.",
        "Missing section 345b, although it is available.",
        "hebrew-stricter",
    ),
)
_DIFFERENTIAL_VERDICTS = {
    "both": (True, True),
    "neither": (False, False),
    "hebrew-stricter": (False, True),
}


@pytest.mark.parametrize(
    ("branch", "hebrew_reason", "english_reason", "verdict"), DIFFERENTIAL_CASES
)
def test_hebrew_rule_agrees_with_the_english_rule(
    branch, hebrew_reason, english_reason, verdict
):
    """Differential: each Hebrew reason against its translation on the same text."""

    assert (
        _accepted(branch, hebrew_reason),
        _english_accepted(ENGLISH_BRANCH[branch], english_reason),
    ) == _DIFFERENTIAL_VERDICTS[verdict]


# --- Review round 12 ------------------------------------------------------------

# The review's minimized reproducers, verbatim: its source, its output branch
# `a` and its translations, Hebrew against English on each.
REVIEW_SOURCE = "(א) ישולם לפי סעיף 2."
REVIEW_ENGLISH_SOURCE = "(a) Payment shall be made under section 2."
REVIEW_NEGATED_SOURCE = "(a) It is not computed in accordance with section 2."
ROUND_12_REPRODUCERS = (
    (REVIEW_SOURCE, REVIEW_ENGLISH_SOURCE, "חסר סעיף 2", "Missing section 2.", "both"),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "לא חסר סעיף 2",
        "Section 2 is not missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "ל\u05b9א חסר סעיף 2",
        "Section 2 is not missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "ל\u200fא חסר סעיף 2",
        "Section 2 is not missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "למרו\u05b9ת שחסר סעיף 2",
        "Although section 2 is missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "משלא חסר סעיף 2",
        "Since section 2 is not missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "וכשלא חסר סעיף 2",
        "And when section 2 is not missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "בלתי נכון שחסר סעיף 2",
        "It is not true that section 2 is missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "חסר סעיף 2. However it is available.",
        "Missing section 2. However it is available.",
        "neither",
    ),
    # The source-negation guard: a pointed negation negates the link.
    (
        "(א) א\u05b5ינו מחושב בהתאם לסעיף 2.",
        REVIEW_NEGATED_SOURCE,
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א) אינו מחושב בהתאם לסעיף 2.",
        REVIEW_NEGATED_SOURCE,
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א) מחושב בהתאם לסעיף 2.",
        "(a) It is computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "both",
    ),
)


@pytest.mark.parametrize(
    ("hebrew_source", "english_source", "hebrew_reason", "english_reason", "verdict"),
    ROUND_12_REPRODUCERS,
)
def test_round_12_reproducers_get_the_english_verdict(
    hebrew_source, english_source, hebrew_reason, english_reason, verdict
):
    assert (
        _accepted("a", hebrew_reason, hebrew_source),
        _english_accepted("a", english_reason, english_source),
    ) == _DIFFERENTIAL_VERDICTS[verdict]


# A deferred branch whose link a negation governs, in Hebrew and in English,
# against the reason "missing the provisions of section 345b".
SOURCE_NEGATION_CASES = (
    (
        "החישוב מחושב בהתאם לסעיף 345ב.",
        "The payment is computed in accordance with section 345b.",
        "both",
    ),
    (
        "החישוב א\u05b5ינו מחושב בהתאם לסעיף 345ב.",
        "The payment is not computed in accordance with section 345b.",
        "neither",
    ),
    (
        "החישוב אי\u200fנו מחושב בהתאם לסעיף 345ב.",
        "The payment is not computed in accordance with section 345b.",
        "neither",
    ),
    (
        "החישוב ל\u05b9א יחושב בהתאם לסעיף 345ב.",
        "The payment shall not be computed in accordance with section 345b.",
        "neither",
    ),
    # Negations under the subordinators ``מש``, ``לכש`` and ``ומש``.
    (
        "משלא חושב התשלום בהתאם לסעיף 345ב, ישולם סכום המינימום.",
        "Since the payment was not computed in accordance with section 345b, the "
        "minimum amount shall be paid.",
        "neither",
    ),
    (
        "לכשלא יחושב התשלום בהתאם לסעיף 345ב, ישולם סכום המינימום.",
        "When the payment is not computed in accordance with section 345b, the "
        "minimum amount shall be paid.",
        "neither",
    ),
    (
        "ומשאין לחשב את התשלום בהתאם לסעיף 345ב, ישולם סכום המינימום.",
        "And since the payment is not to be computed in accordance with section "
        "345b, the minimum amount shall be paid.",
        "neither",
    ),
)


@pytest.mark.parametrize(
    ("hebrew_branch", "english_branch", "verdict"), SOURCE_NEGATION_CASES
)
def test_the_source_negation_guard_agrees_with_the_english_rule(
    hebrew_branch, english_branch, verdict
):
    assert (
        _accepted("e1", "חסרות הוראות סעיף 345ב.", f"(א) הוראה.\n(ה1) {hebrew_branch}"),
        _english_accepted(
            "d",
            "missing the provisions of section 345b.",
            f"(a) Provision.\n(d) {english_branch}",
        ),
    ) == _DIFFERENTIAL_VERDICTS[verdict]


@pytest.mark.parametrize(
    "adversative", ("אבל", "א\u05b2ב\u05b8ל", "א\u200fבל", "ו\u05b0או\u05bcל\u05b8ם")
)
def test_a_marked_hebrew_adversative_reverses_an_english_reason(adversative):
    """The shared checker reads a Hebrew adversative through its marks, as "But"."""

    assert _english_accepted("d", "Missing section 345b.")
    assert not _english_accepted("d", "Missing section 345b. But it is available.")
    assert not _english_accepted(
        "d", f"Missing section 345b. {adversative} it is available."
    )


@settings(max_examples=40, deadline=None)
@given(st.data())
def test_invisible_marks_never_change_a_hebrew_verdict(data):
    """Metamorphic: a reason reads as its unmarked words, through the CI method."""

    branch, reason, _english, _verdict = data.draw(st.sampled_from(DIFFERENTIAL_CASES))

    assert _accepted(branch, data.draw(_marked(reason))) is _accepted(branch, reason)


@settings(max_examples=100, deadline=None)
@given(st.data())
def test_the_hebrew_reader_folds_reason_and_scope_itself(data):
    """Metamorphic, on the reader alone: marks in reason or scope change nothing."""

    _branch, reason, _english, _verdict = data.draw(st.sampled_from(DIFFERENTIAL_CASES))
    scope = PRECISION_SOURCE.splitlines()[4]
    expected = hebrew.reason_names_source_bound_dependency(
        reason, scope, corpus_citation_path=CITATION
    )

    assert (
        hebrew.reason_names_source_bound_dependency(
            data.draw(_marked(reason)),
            data.draw(_marked(scope)),
            corpus_citation_path=CITATION,
        )
        is expected
    )


def test_text_without_hebrew_is_never_folded():
    """English and German text reaches every reader as the same object."""

    for text in ("Missing section\u200f 345b.", "Absatz\u00ad 2 Satz 1", "x\ufeff"):
        assert hebrew.fold_marks(text) is text


@pytest.mark.parametrize(
    ("marked", "plain"),
    (
        ("ל\u05b9א", "לא"),
        ("ל\u200fא", "לא"),
        ("א\u05b5ינ\ufb4b", "אינו"),
        ("\u202bחסר\u202c", "חסר"),
        ("ש\u05c1\u05b6ל\u05bc\u05b9א", "שלא"),
        ("\ufb2aלא", "שלא"),
        # Maqaf, geresh and gershayim are punctuation, and stay.
        ("על\u05beפי", "על\u05beפי"),
        ("סעיף 345י\u05f4א", "סעיף 345י\u05f4א"),
        ("סעיף 345ב\u05f3", "סעיף 345ב\u05f3"),
    ),
)
def test_marks_fold_to_the_word_itself(marked, plain):
    assert hebrew.fold_marks(marked) == plain


@settings(max_examples=300, deadline=None)
@given(
    st.lists(
        st.sampled_from(
            (
                *"אבגלתשמ סעיף 345.,()ab",
                *_INVISIBLE_MARKS,
                "\ufb4b",
                "\ufb2a",
                "\ufb4f",
                "\u05be",
                "\u05f4",
            )
        ),
        max_size=60,
    ).map("".join)
)
def test_folding_is_idempotent_and_leaves_no_mark(text):
    folded = hebrew.fold_marks(text)

    assert hebrew.fold_marks(folded) == folded
    if hebrew._HEBREW_SCRIPT.search(text):
        assert not hebrew._fold_rules()[1].search(folded)
    else:
        assert folded is text


@pytest.mark.parametrize(
    "negation",
    (
        "ולא",
        "שלא",
        "ושלא",
        "כשלא",
        "וכשלא",
        "משלא",
        "ומשלא",
        "לכשלא",
        "ולכשלא",
        "שאין",
        "ומשאין",
        "כשאינו",
        "משטרם",
        "בלתי",
        "ובלתי",
        "שבלתי",
        "משאי אפשר",
        "דלא",
        "ודלא",
        "דאין",
        "דאינו",
        "דטרם",
        "דלאו",
        "דבלתי",
        "דאי אפשר",
    ),
)
def test_a_negation_takes_its_prefixes(negation):
    assert hebrew._REASON_NEGATION.fullmatch(negation)


def test_prefixed_meolam_needs_negative_polarity_context():
    text = "שמעולם לא נטען שחסר סעיף 2"
    match = hebrew._REASON_NEGATION.search(text)
    assert match and match.group() == "שמעולם"
    assert not hebrew._REASON_NEGATION.search("מידע שמעולם העבודה")


@pytest.mark.parametrize(
    "word",
    (
        "מלא",
        "מלאה",
        "מלאים",
        "מלאכה",
        "כלא",
        "הלא",
        "הלאה",
        "בטרם",
        "מאין",
        "כאין",
        "לאין",
    ),
)
def test_a_word_spelled_with_prefix_letters_is_no_negation(word):
    """``מלא`` is "full", not ``מ`` + ``לא``: a bare preposition prefixes no negation."""

    assert not hebrew._REASON_NEGATION.search(word)
    assert not hebrew._NEGATED_SIGNAL_PREFIX.search(word)


def test_reason_adversatives_use_the_shared_vocabulary():
    assert (
        hebrew._ANY_ADVERSATIVE.pattern
        == completeness_module._ADVERSATIVE_LANGUAGE.pattern
    )
    assert hebrew._ANY_ADVERSATIVE.flags == (
        completeness_module._ADVERSATIVE_LANGUAGE.flags
    )


# --- Review round 13 ------------------------------------------------------------

# The review's reproducers and their siblings: the Aramaic relative ``ד`` before
# a negation, and an English or German negator before a Hebrew signal or link.
# Each Hebrew reason gets its translation's English verdict; `hebrew-stricter`
# marks a negator between the signal and its citation, which Hebrew refuses
# and English, reading no negation there, admits.
ROUND_13_REPRODUCERS = (
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "דלא חסר סעיף 2",
        "Section 2 is not missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "ודלא חסר סעיף 2",
        "And section 2 is not missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "not חסר סעיף 2",
        "Section 2 is not missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "It is not true that חסר סעיף 2.",
        "It is not true that section 2 is missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "nicht חסר סעיף 2",
        "Section 2 is not missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "It is not the case that חסר סעיף 2",
        "It is not the case that section 2 is missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "Es ist nicht so, dass חסר סעיף 2",
        "It is not the case that section 2 is missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "no חסר סעיף 2",
        "No section 2 is missing.",
        "neither",
    ),
    # Intended: a negator between the signal and the citation leaves the
    # missing claim unread in Hebrew, in any language, as ``לא`` there does.
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "חסר not סעיף 2",
        "Missing, not section 2.",
        "hebrew-stricter",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "חסר nicht סעיף 2",
        "Missing, not section 2.",
        "hebrew-stricter",
    ),
    # Controls: no negator, a word that only starts like one, a negator after
    # the citation, and one in an earlier clause.
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "It is true that חסר סעיף 2.",
        "It is true that missing section 2.",
        "both",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "Note: חסר סעיף 2",
        "Note: missing section 2.",
        "both",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "חסר סעיף 2, not yet encoded.",
        "Missing section 2, not yet encoded.",
        "both",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "nichts; חסר סעיף 2",
        "Nothing; missing section 2.",
        "both",
    ),
    # The same two classes in the source-link guard: ``דלא``, ``דאינו`` and an
    # English or German negator before ``בהתאם ל`` negate the link.
    (
        "(א) דלא מחושב בהתאם לסעיף 2.",
        "(a) That is not computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א) ודאינו מחושב בהתאם לסעיף 2.",
        "(a) And that is not computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א) not מחושב בהתאם לסעיף 2.",
        REVIEW_NEGATED_SOURCE,
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א) nicht מחושב בהתאם לסעיף 2.",
        REVIEW_NEGATED_SOURCE,
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    # English's own ``in accordance with`` negations, prohibitions and
    # exclusions (``_SOURCE_LINK_NEGATION_LANGUAGE``) negate a Hebrew link as
    # they negate its translation.
    (
        "(א) prohibited מחושב בהתאם לסעיף 2.",
        "(a) It is prohibited to compute it in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א) It is impossible that מחושב בהתאם לסעיף 2.",
        "(a) It is impossible that it is computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א) without being computed בהתאם לסעיף 2.",
        "(a) It is paid without being computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א) other than מחושב בהתאם לסעיף 2.",
        "(a) It is computed other than in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א) It fails to be מחושב בהתאם לסעיף 2.",
        "(a) It fails to be computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א) In no case מחושב בהתאם לסעיף 2.",
        "(a) In no case is it computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    # Intended: an English or German negator earlier in the clause refuses the
    # Hebrew signal even where English admits the word-for-word translation,
    # since English reads no negation before its own ``missing`` signal; the
    # ``not`` of the English signal "not yet encoded" counts too. And one
    # before ``לפי`` negates the link, which English does not check after
    # ``under``.
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "not חסר סעיף 2",
        "Not missing section 2.",
        "hebrew-stricter",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "Not yet encoded: חסר סעיף 2",
        "Not yet encoded: missing section 2.",
        "hebrew-stricter",
    ),
    (
        "(א) not ישולם לפי סעיף 2.",
        "(a) Payment shall not be made under section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "hebrew-stricter",
    ),
)


@pytest.mark.parametrize(
    ("hebrew_source", "english_source", "hebrew_reason", "english_reason", "verdict"),
    ROUND_13_REPRODUCERS,
)
def test_round_13_reproducers_get_the_english_verdict(
    hebrew_source, english_source, hebrew_reason, english_reason, verdict
):
    assert (
        _accepted("a", hebrew_reason, hebrew_source),
        _english_accepted("a", english_reason, english_source),
    ) == _DIFFERENTIAL_VERDICTS[verdict]


# Every word of the shared English and German negators (``_NEGATION_LANGUAGE``
# in source_completeness), and contractions of its ``n't``.
_OTHER_NEGATORS = (
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "nothing",
    "nowhere",
    "cannot",
    "false that",
    "untrue that",
    "isn't",
    "doesn’t",
    "can't",
    "won't",
    "nicht",
    "nichts",
    "kein",
    "keine",
    "keinen",
    "keinem",
    "keiner",
    "keines",
    "keinesfalls",
    "keineswegs",
    "nie",
    "niemals",
    "weder",
    "nirgends",
    "falsch, dass",
    "unwahr dass",
)
# Words that are no negation in any of the three languages: English and German
# fillers around a negator, and words that only start like one.
_OTHER_NEUTRAL_WORDS = (
    "it",
    "is",
    "true",
    "the",
    "case",
    "that",
    "es",
    "ist",
    "so",
    "dass",
    "wahr",
    "note",
    "knot",
    "nordic",
    "niece",
    "noted",
)


def test_reason_negations_use_the_shared_vocabulary():
    assert (
        hebrew._OTHER_NEGATION.pattern == completeness_module._NEGATION_LANGUAGE.pattern
    )
    assert hebrew._OTHER_NEGATION.flags == completeness_module._NEGATION_LANGUAGE.flags


def test_the_shared_vocabulary_holds_every_required_negator():
    """The review's list, ``n't`` in its contractions; ``without`` is a signal."""

    negation = completeness_module._NEGATION_LANGUAGE
    for word in _OTHER_NEGATORS:
        for form in (word, word.upper()):
            # A contraction is read at its ``n't``, the whole word otherwise.
            found = negation.search(form)
            assert found and found.end() == len(form), form
            assert found.start() == 0 or found.group(0).lower() in {"n't", "n’t"}
    for word in _OTHER_NEUTRAL_WORDS + ("without", "ohne", "Nottingham"):
        assert not negation.search(word), word


def test_the_shared_checker_passes_its_negators_to_the_hebrew_reader(monkeypatch):
    """The Hebrew route reads the shared checker's vocabulary, not only its copy."""

    assert _accepted("a", "nope חסר סעיף 2", REVIEW_SOURCE)
    monkeypatch.setattr(
        completeness_module,
        "_NEGATION_LANGUAGE",
        re.compile(r"\bnope\b", flags=re.IGNORECASE),
    )
    assert not _accepted("a", "nope חסר סעיף 2", REVIEW_SOURCE)
    # The source-link guard reads it too.
    assert not _accepted("a", "חסר סעיף 2", "(א) nope מחושב בהתאם לסעיף 2.")


@st.composite
def _other_negations(draw) -> str:
    """An English or German negator, in any letter case."""

    word = draw(st.sampled_from(_OTHER_NEGATORS))
    return draw(st.sampled_from((word, word.upper(), word.capitalize())))


@settings(max_examples=150, deadline=None)
@given(
    st.one_of(_negations(), _other_negations()),
    st.lists(st.sampled_from(_NEUTRAL_WORDS + _OTHER_NEUTRAL_WORDS), max_size=4),
    st.lists(st.sampled_from(_NEUTRAL_WORDS + _OTHER_NEUTRAL_WORDS), max_size=6),
    st.sampled_from(_PRECISE_MISSING_CLAIMS),
    st.booleans(),
)
@example("דלא", [], [], _PRECISE_MISSING_CLAIMS[0], False)
@example("not", ["It", "is"], ["true", "that"], _PRECISE_MISSING_CLAIMS[0], False)
@example("nicht", ["Es", "ist"], ["so,", "dass"], _PRECISE_MISSING_CLAIMS[1], False)
@example("not", [], [], _PRECISE_MISSING_CLAIMS[2], True)
def test_any_negator_before_an_accepted_signal_refuses_it(
    negation, lead, complement, claim, between
):
    """Polarity: a negator of any of the three languages flips an accepted claim.

    The negator is a Hebrew one under any of its prefixes, ``ד`` included, and
    pointed or direction-marked, or any word of the shared English and German
    vocabulary. Placed anywhere earlier in the clause than the signal, after
    words of any of the languages, or between the signal and its citation, it
    refuses the claim the same words accept without it.
    """

    signal, rest = claim.split(" ", 1)
    before = " ".join(lead)
    after = " ".join(complement)
    affirmed = f"{before} {after} ש{claim}".strip()
    if between:
        negated = f"{before} {after} ש{signal} {negation} {rest}".strip()
    else:
        negated = f"{before} {negation} {after} ש{claim}".strip()

    assert _accepted("e1", affirmed)
    assert not _accepted("e1", negated)


# A form of each alternative of English's ``in accordance with`` negations
# (``_SOURCE_LINK_NEGATION_LANGUAGE`` in source_completeness).
_OTHER_LINK_NEGATORS = (
    "not",
    "never",
    "neither",
    "nor",
    "cannot",
    "nowhere",
    "prohibited",
    "prohibits",
    "barred",
    "forbidden",
    "precluded",
    "prevented",
    "excluded",
    "denied",
    "except",
    "denies",
    "rejects",
    "disclaims",
    "false that",
    "untrue that",
    "impossible that",
    "no amount is",
    "and no tax shall",
    "isn't",
    "doesn’t",
    "can't",
    "mustn't",
    "fails to",
    "refuses to",
    "unable to",
    "without being computed",
    "without applying",
    "in no case",
    "in no event",
    "under no circumstances",
    "at no time",
    "by no means",
    "in no way",
    "on no account",
    "under no conditions",
    "in no manner",
    "in no respect",
    "rather than",
    "other than",
    # The German word for each (review round 14, finding 3).
    "nicht",
    "nie",
    "niemals",
    "weder",
    "nirgends",
    "nirgendwo",
    "kein",
    "keine",
    "keinem",
    "keinesfalls",
    "keineswegs",
    "außer",
    "ausser",
    "ausgenommen",
    "abgesehen von",
    "verbietet",
    "verboten",
    "untersagt",
    "verwehrt",
    "ausgeschlossen",
    "verhindert",
    "verweigert",
    "abgelehnt",
    "bestreitet",
    "verneint",
    "zurückgewiesen",
    "versäumt",
    "unterlässt",
    "unterlassen",
    "weigert",
    "außerstande",
    "ausserstande",
    "anstatt",
    "statt",
    "anders als",
    "falsch, dass",
    "unwahr dass",
    "unmöglich, dass",
    # ``without`` and ``ohne`` before a computation word of either language.
    "without Berechnung",
    "without being berechnet",
    "ohne computed",
    "ohne Berechnung",
    "ohne berechnet",
    "ohne zu berechnen",
    "ohne Anwendung",
    "ohne festgesetzt",
    "ohne ermittelt",
    "ohne erhoben",
    "ohne zu entstehen",
)
# The Hebrew negations the link guard reads, as ``_NEGATIONS`` reads them.
_LINK_NEGATORS = ("לא", "אין", "אינו", "בלתי")
# Words that negate no link in any of the three languages.
_LINK_NEUTRAL_WORDS = _OTHER_NEUTRAL_WORDS + ("החישוב", "התשלום", "ברור", "נכון")


def test_link_negations_use_the_shared_vocabulary():
    assert (
        hebrew._OTHER_LINK_NEGATION.pattern
        == completeness_module._SOURCE_LINK_NEGATION_LANGUAGE.pattern
    )
    assert (
        hebrew._OTHER_LINK_NEGATION.flags
        == completeness_module._SOURCE_LINK_NEGATION_LANGUAGE.flags
    )


def test_the_link_vocabulary_holds_every_listed_form():
    negation = completeness_module._SOURCE_LINK_NEGATION_LANGUAGE
    for word in _OTHER_LINK_NEGATORS:
        for form in (word, word.upper()):
            found = negation.search(form)
            assert found and found.span() == (0, len(form)), form
    for word in _LINK_NEUTRAL_WORDS + ("without delay", "no", "rather"):
        assert not negation.search(word), word


def test_the_shared_checker_passes_its_link_negators_to_the_hebrew_reader(monkeypatch):
    """The Hebrew link guard reads English's link vocabulary, not only its copy."""

    source = "(א) nope מחושב בהתאם לסעיף 2."
    assert _accepted("a", "חסר סעיף 2", source)
    monkeypatch.setattr(
        completeness_module,
        "_SOURCE_LINK_NEGATION_LANGUAGE",
        re.compile(r"\bnope\b", flags=re.IGNORECASE),
    )
    assert not _accepted("a", "חסר סעיף 2", source)


@st.composite
def _link_negations(draw) -> str:
    """A negation of a link, in any of the three languages.

    A Hebrew negation under any of its prefixes, ``ד`` included, pointed or
    direction-marked; a word of the shared English and German negators; or a
    form of English's ``in accordance with`` negations, in any letter case.
    """

    language = draw(st.sampled_from(("hebrew", "shared", "link")))
    if language == "hebrew":
        negation = draw(st.sampled_from(_NEGATION_PREFIXES)) + draw(
            st.sampled_from(_LINK_NEGATORS)
        )
        return draw(_marked(negation))
    word = draw(
        st.sampled_from(
            _OTHER_NEGATORS if language == "shared" else _OTHER_LINK_NEGATORS
        )
    )
    return draw(st.sampled_from((word, word.upper(), word.capitalize())))


@settings(max_examples=150, deadline=None)
@given(_link_negations(), st.lists(st.sampled_from(_LINK_NEUTRAL_WORDS), max_size=4))
@example("דלא", [])
@example("ודאינו", ["התשלום"])
@example("prohibited", [])
@example("impossible that", ["it", "is"])
@example("without being computed", [])
@example("nicht", ["es", "ist"])
def test_any_negator_before_an_accepted_link_unlinks_it(negation, words):
    """Polarity: a negator of any of the three languages negates a Hebrew link.

    Placed anywhere earlier in the deferred branch's clause than ``בהתאם ל``,
    after words of any of the languages, it leaves the cited section
    inoperative, so the precise missing claim the same words otherwise
    support is refused.
    """

    lead = " ".join(words)
    affirmed = f"(א) {lead} מחושב בהתאם לסעיף 2.".replace("  ", " ")
    negated = f"(א) {lead} {negation} מחושב בהתאם לסעיף 2.".replace("  ", " ")

    assert _accepted("a", "חסר סעיף 2", affirmed)
    assert not _accepted("a", "חסר סעיף 2", negated)


# `_EXPLICIT_COMPUTATION` as it was before its rate was anchored at the first
# digit of a run.
_UNANCHORED_COMPUTATION = re.compile(
    hebrew._EXPLICIT_COMPUTATION.pattern.replace("(?<!\\d)", "", 1)
)


@settings(max_examples=500, deadline=None)
@given(
    st.lists(
        st.sampled_from(
            ("1", "23", "5", ".", ",", "%", " ", "\t", "מ", "ו", "ן ", "ה", "כנס", "x")
        ),
        max_size=14,
    ).map("".join)
)
@example("5.17% מהכנסתו")
@example("1,234.5% מהכנסה")
@example("x12.5 % ומן ההכנסה")
@example("1" * 40)
def test_the_rate_anchor_changes_no_verdict(text):
    """Differential: a match from inside a run of digits implies one from its start."""

    assert _UNANCHORED_COMPUTATION.pattern != hebrew._EXPLICIT_COMPUTATION.pattern
    assert (hebrew._EXPLICIT_COMPUTATION.search(text) is None) is (
        _UNANCHORED_COMPUTATION.search(text) is None
    )


@pytest.mark.parametrize(
    "reason",
    (
        # The said law is the National Insurance Law, named earlier in the branch.
        "חסרה הוראת החישוב שבסעיף 345ב לחוק הביטוח הלאומי.",
        "חסרה הוראת החישוב שבסעיף 345ב.",
    ),
)
def test_the_said_law_is_the_law_named_before_it(reason):
    source = (
        "(א) הוראה.\n"
        "(ה1) מבוטח שמשתלמת לו קצבה כמשמעותה בחוק הביטוח הלאומי חייב לשלם, "
        "ויחולו לענין זה ההוראות לפי סעיף 345ב לחוק האמור."
    )

    issues = _analyze(_deferral_content("e1", reason), source)

    assert _deferral_issue(issues, 0) is None


@pytest.mark.parametrize(
    "reason",
    (
        "חסרה הוראת החישוב שבסעיף 345ב לפקודת מס הכנסה.",
        "חסרה הוראת החישוב שבסעיף 345ב לחוק הבטחת הכנסה.",
        "חסרה הוראת החישוב שבסעיף 345ב לחוק זה.",
        # A shortened name is another name: names compare whole.
        "חסרה הוראת החישוב שבסעיף 345ב לחוק הביטוח.",
    ),
)
def test_the_said_law_does_not_bind_another_instrument(reason):
    source = (
        "(א) הוראה.\n"
        "(ה1) מבוטח שמשתלמת לו קצבה כמשמעותה בחוק הביטוח הלאומי חייב לשלם, "
        "ויחולו לענין זה ההוראות לפי סעיף 345ב לחוק האמור."
    )

    issues = _analyze(_deferral_content("e1", reason), source)

    assert _deferral_issue(issues, 0) is not None


def test_the_said_law_with_no_antecedent_binds_no_named_instrument():
    source = "(א) הוראה.\n(ה1) ויחולו לענין זה ההוראות לפי סעיף 345ב לחוק האמור."

    named = _analyze(
        _deferral_content("e1", "חסרה הוראת סעיף 345ב לחוק הביטוח הלאומי."), source
    )
    unnamed = _analyze(_deferral_content("e1", "חסרה הוראת סעיף 345ב."), source)

    assert _deferral_issue(named, 0) is not None
    assert _deferral_issue(unnamed, 0) is None


@pytest.mark.parametrize(
    ("source", "linked"),
    (
        ("(א) הוראה.\n(ה1) יש לפעול בהתאם לסעיף 345ב.", True),
        ("(א) הוראה.\n(ה1) אין לפעול בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) לא יחולו ההוראות לפי סעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) החישוב איננו בהתאם לסעיף 345ב.", False),
        (
            "(א) הוראה.\n(ה1) אין לפעול בשום אופן ובשום נסיבות בהתאם לסעיף 345ב.",
            False,
        ),
        ("(א) הוראה.\n(ה1) תשלום שאינו בהתאם לסעיף 345ב פטור.", False),
        ("(א) הוראה.\n(ה1) תשלום, למעט תשלום לפי סעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) אין לחשב את שיעור התשלום בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) אין לחשב שכר בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) אין לחשב את התשלום שחל עליו המס בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) אסורה פעולה בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) אסורים מעשים בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) אסורות פעולות בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) אי אפשר לפעול בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) תשלום ללא ניכוי לפי סעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) בלא לחשב את התשלום בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) ושבלא לחשב את התשלום בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) המוסד מסרב לחשב את התשלום בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) המועצות מסרבות לחשב את התשלום בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) חישוב התשלום הוחרג בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) תשלומים מוחרגים בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) נמנע מן המוסד לחשב את התשלום בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) המוסד דוחה את הטענה בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) המוסד שולל את התשלום בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) התשלום נשלל בהתאם לסעיף 345ב.", False),
        # `דוחות` is also "reports": no negation.
        ("(א) הוראה.\n(ה1) הגשת דוחות לפי סעיף 345ב.", True),
        (
            "(א) הוראה.\n(ה1) המועצות דוחות את הטענה שיש לחשב את התשלום בהתאם לסעיף 345ב.",
            False,
        ),
        ("(א) הוראה.\n(ה1) החוק אוסר לחשב את התשלום בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) ייאסר לחשב את התשלום בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) המוסד מכחיש שיש לחשב את התשלום בהתאם לסעיף 345ב.", False),
        # English reads "in accordance with the provisions of" and `על  פי`.
        ("(א) הוראה.\n(ה1) יש לפעול בהתאם להוראות סעיף 345ב.", True),
        ("(א) הוראה.\n(ה1) יש לפעול בהתאם להוראת סעיף 345ב.", True),
        ("(א) הוראה.\n(ה1) יש לפעול על  פי סעיף 345ב.", True),
        # Plural `סעיפים` is not read, as English does not read "sections".
        ("(א) הוראה.\n(ה1) יש לפעול לפי סעיפים 345ב ו-345ג.", False),
        (
            "(א) הוראה.\n(ה1) אין לפעול "
            + "בכל עת ובכל מצב " * 20
            + "בהתאם לסעיף 345ב.",
            False,
        ),
        # A main-clause prohibition governs relative clauses after it.
        ("(א) הוראה.\n(ה1) אין לחשב את התשלום שנקבע בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) אין לחשב את התשלום הקבוע בסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) נאסר על המוסד לפעול בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) נאסרו פעולות בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) חל איסור לפעול בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) המוסד מנוע מלפעול בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) המועצות מנועות מלפעול בהתאם לסעיף 345ב.", False),
        # A cut word keeps its left boundary: `כלפי` ("against") is not `לפי`.
        ("(א) הוראה.\n(ה1) טענה כלפי" + " " * 237 + "סעיף 345ב.", False),
        (
            "(א) הוראה.\n(ה1) אין לחשב את התשלום שאינו עולה על הסכום שנקבע לפי סעיף 345ב.",
            False,
        ),
        # `בהתאם ל` takes no reset, as English binds negation to it.
        (
            "(א) הוראה.\n(ה1) תשלום שאינו כולל את הסכום שנקבע בהתאם לסעיף 345ב פטור.",
            False,
        ),
        # A relative negation ends where a relative clause carrying a `לפי` link
        # begins (section 14(ו1)).
        ("(א) הוראה.\n(ה1) החלק שאינו עולה על הסכום שנקבע לפי סעיף 345ב.", True),
        # `הקבוע ב` ("set in") is no linker, in English either.
        ("(א) הוראה.\n(ה1) החלק שאינו עולה על הסכום הקבוע בסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) תשלום כהגדרתו בסעיף 345ב.", False),
        # A relative clause's negation is read as negating, stricter than English.
        ("(א) הוראה.\n(ה1) מבוטח שלא עבד ישלם לפי סעיף 345ב.", False),
        # Inability, failure, refusal and prohibition, as English reads
        # "unable to", "fails to", "refuses to" and "prohibited".
        ("(א) הוראה.\n(ה1) נבצר מן המוסד לחשב את התשלום בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) נבצר ממנו לחשב את התשלום בהתאם לסעיף 345ב.", False),
        *(
            (f"(א) הוראה.\n(ה1) המוסד {verb} לחשב את התשלום בהתאם לסעיף 345ב.", False)
            for verb in ("יסרב", "תסרב", "יסרבו", "תסרבנה", "סירב", "סירבה")
        ),
        *(
            (f"(א) הוראה.\n(ה1) החישוב {verb} בהתאם לסעיף 345ב.", False)
            for verb in (
                "כושל",
                "כושלת",
                "כושלים",
                "כושלות",
                "ייכשל",
                "תיכשל",
                "ייכשלו",
                "תיכשלנה",
                "נכשל",
                "נכשלה",
            )
        ),
        *(
            (f"(א) הוראה.\n(ה1) {verb} את התשלום בהתאם לסעיף 345ב.", False)
            for verb in ("אל תחשב", "אל תחשבי", "אל תחשבו", "אל תחשבנה")
        ),
        # Ordinary words that the exclusion and prevention markers spell: the
        # infinitive "to issue" after a modal, "another person", "engine".
        ("(א) הוראה.\n(ה1) יש להוציא רישיון בהתאם לסעיף 345ב.", True),
        ("(א) הוראה.\n(ה1) יש להוציא דרכון בהתאם לסעיף 345ב.", True),
        ("(א) הוראה.\n(ה1) יש לפצות את הזולת בהתאם לסעיף 345ב.", True),
        ("(א) הוראה.\n(ה1) המס על מנוע ייקבע בהתאם לסעיף 345ב.", True),
        ("(א) הוראה.\n(ה1) מנועים חשמליים ימוסו בהתאם לסעיף 345ב.", True),
        # The same words as exclusion and prevention still negate.
        ("(א) הוראה.\n(ה1) כל התשלומים, להוציא תשלום בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) כל התשלומים, זולת תשלום בהתאם לסעיף 345ב.", False),
        ("(א) הוראה.\n(ה1) המוסד מנוע מלחשב את התשלום בהתאם לסעיף 345ב.", False),
    ),
)
def test_a_negated_source_link_does_not_apply_the_section(source, linked):
    issues = _analyze(_deferral_content("e1", "חסרות הוראות סעיף 345ב."), source)

    assert (_deferral_issue(issues, 0) is None) is linked


@pytest.mark.parametrize(
    "copula",
    (
        "אינו",
        "אינה",
        "אינם",
        "אינן",
        "איני",
        "אינני",
        "אינך",
        "איננו",
        "איננה",
        "אינכם",
        "אינכן",
    ),
)
def test_every_negative_copula_negates_a_link(copula):
    source = f"(א) הוראה.\n(ה1) החישוב {copula} בהתאם לסעיף 345ב."
    issues = _analyze(_deferral_content("e1", "חסרות הוראות סעיף 345ב."), source)

    assert _deferral_issue(issues, 0) is not None


@pytest.mark.parametrize(
    ("source_law", "reason_law", "linked"),
    (
        (
            "הפיקוח על שירותים פיננסיים (ביטוח)",
            "הפיקוח על שירותים פיננסיים (ביטוח)",
            True,
        ),
        (
            "הפיקוח על שירותים פיננסיים (ביטוח)",
            "הפיקוח על שירותים פיננסיים (קופות גמל)",
            False,
        ),
        ("הפיקוח על שירותים פיננסיים", "הפיקוח על הבנקים", False),
        ("הביטוח הלאומי בשינויים המחויבים", "הביטוח הלאומי", True),
        ("יסוד: כבוד האדם וחירותו", "יסוד: חופש העיסוק", False),
        ("יסוד: כבוד האדם וחירותו", "יסוד: כבוד האדם וחירותו", True),
        ("״הביטוח הלאומי״", "הביטוח הלאומי", True),
        ("״הביטוח הלאומי״", "״שירות המדינה״", False),
        # Two statutes, one name the other's first words: names compare whole.
        ("המקרקעין", "המקרקעין (חיזוק בתים משותפים מפני רעידות אדמה)", False),
        ("המקרקעין (חיזוק בתים משותפים מפני רעידות אדמה)", "המקרקעין", False),
        # A number continues a title; a glued Basic Law title is read too.
        ("התקציב לשנת הכספים 2025", "התקציב לשנת הכספים 2026", False),
        # An edition or enactment date continues a name too.
        (
            "הביטוח הלאומי [נוסח משולב], התשכ״ח–1968",
            "הביטוח הלאומי [נוסח משולב], התשנ״ה–1995",
            False,
        ),
        ("הביטוח הלאומי, התשכ״ח–1968", "הביטוח הלאומי, התשנ״ה–1995", False),
        ("יסוד:כבוד האדם וחירותו", "יסוד:חופש העיסוק", False),
        # Past six name words a name is uncertain and binds no named law.
        (
            "הרשות לפינוי שדות מוקשים ולטיפול בשטחי נפלים",
            "הרשות לפינוי שדות מוקשים ולטיפול בשטחי גבולות",
            False,
        ),
        (
            "שירות המדינה (" + "גמלאות " * 8 + ")",
            "שירות המדינה (" + "משמעת " * 8 + ")",
            False,
        ),
    ),
)
def test_a_law_name_is_read_to_its_end(source_law, reason_law, linked):
    source = f"(א) הוראה.\n(ה1) יש לפעול בהתאם לסעיף 42 לחוק {source_law}."
    reason = f"חסרות הוראות סעיף 42 לחוק {reason_law}."
    issues = _analyze(_deferral_content("e1", reason), source)

    assert (_deferral_issue(issues, 0) is None) is linked


@pytest.mark.parametrize("prefix", ("ש", "שה", "וש", "ושה", "כש", "כשה", "וכש", "וכשה"))
def test_the_said_law_follows_a_relative_mention(prefix):
    """`תשלום שחוק הביטוח הלאומי חל עליו` names the law "the said law" means."""

    source = (
        "(א) הוראה.\n(ה1) חוק שירות המדינה. "
        f"תשלום {prefix}חוק הביטוח הלאומי חל עליו, "
        "יש לפעול בהתאם לסעיף 42 לחוק האמור."
    )
    issues = _analyze(
        _deferral_content("e1", "חסרות הוראות סעיף 42 לחוק שירות המדינה."), source
    )

    assert _deferral_issue(issues, 0) is not None


@pytest.mark.parametrize("kind", ("חוק", "פקודה", "פקודת", "תקנות", "צו"))
@pytest.mark.parametrize("separator", (",", ":", "\n", ",\n", ":\n", " \n\n"))
@pytest.mark.parametrize(
    "tail", ("", ", התשנ״ה–1995", " [נוסח משולב], התשנ״ה–1995", " (נוסח משולב)")
)
def test_a_preposed_law_name_claims_no_branch_of_this_one(kind, separator, tail):
    """The law named first owns the citation, a line break between them included."""

    reference = f"{kind} הביטוח הלאומי{tail}{separator} סעיף 14(ב)"

    assert (
        completeness_module._paths_from_source_reference(
            reference, corpus_citation_path=CITATION
        )
        == set()
    )
    assert not hebrew.cites_current_branch(reference, CITATION, ("b",))
    assert not hebrew.cites_current_section(reference, CITATION)
    assert hebrew.cites_current_branch(
        f"חוק זה{separator} סעיף 14(ב)", CITATION, ("b",)
    )


def test_a_line_that_opens_a_branch_ends_a_laws_ownership():
    """A line break before an outline label is a structural boundary, not a wrap."""

    wrapped = "(ה) תשלום לפי חוק הביטוח הלאומי\nויחולו ההוראות לפי סעיף 345ב."
    branch = "(ה) תשלום לפי חוק הביטוח הלאומי\n(ה1) יחולו ההוראות לפי סעיף 345ב."

    assert [
        instrument for _match, instrument in hebrew._section_citations(wrapped)
    ] == [hebrew._UNRESOLVED_INSTRUMENT]
    assert [instrument for _match, instrument in hebrew._section_citations(branch)] == [
        None
    ]


@pytest.mark.parametrize(
    "reason",
    (
        "חוק הביטוח הלאומי: חסרות הוראות סעיף 42.",
        "פקודת מס הכנסה, התשכ״א–1961: חסרות הוראות סעיף 42.",
        "חסרות הוראות סעיף 42 לפקודה זו.",
        "חסרות הוראות סעיף 42 לתקנות אלה.",
        "חסרות הוראות סעיף 42 לצו זה.",
        "חסרות הוראות סעיף 42 לאותה פקודה.",
        "חסרות הוראות סעיף 42 לאותו חוק.",
    ),
)
def test_another_instrument_owns_a_reason_citation(reason):
    """A law named earlier in the clause, or "this ordinance" in a law, is not this law."""

    source = "(א) הוראה.\n(ה1) יש לפעול לפי סעיף 42 לחוק זה."
    issues = _analyze(_deferral_content("e1", reason), source)

    assert _deferral_issue(issues, 0) is not None


def test_this_law_binds_this_law():
    source = "(א) הוראה.\n(ה1) יש לפעול לפי סעיף 42 לחוק זה."
    issues = _analyze(_deferral_content("e1", "חסרות הוראות סעיף 42 לחוק זה."), source)

    assert _deferral_issue(issues, 0) is None


@pytest.mark.parametrize("qualifier", ("לפקודה זו", "לצו זה", "לתקנות אלה"))
def test_this_ordinance_in_a_law_claims_no_branch(qualifier):
    reference = f"סעיף 14(ב) {qualifier}"

    assert not hebrew.native_source_reference_paths(reference, CITATION)
    assert not hebrew.cites_current_branch(reference, CITATION, ("b",))
    assert hebrew.native_source_reference_paths("סעיף 14(ב) לחוק זה", CITATION) == {
        ("b",)
    }


def test_a_long_preposed_envelope_still_owns_the_citation():
    reference = "חוק הביטוח הלאומי, " + " " * 300 + "סעיף 14(ב)"

    assert hebrew.native_source_reference_paths(reference, CITATION) == set()
    assert not hebrew.cites_current_section(reference, CITATION)


@pytest.mark.parametrize(
    "component", ("1zzzzzzz", "1" + "z" * 40, "a" * 40_000, "zzz", "1aaa", "cv")
)
def test_unsupported_labels_are_refused_before_conversion(component):
    """A label is a numeral up to 99; nothing larger is converted (round 8)."""

    started = time.perf_counter()

    assert hebrew.native_path_label(component) is None
    assert hebrew.native_section_label(f"il/statute/x/section-1{component}") is None
    assert time.perf_counter() - started < 1


@pytest.mark.parametrize(
    ("leaf", "label"),
    (("section-14", "14"), ("section-121b", "121ב"), ("section-345k", "345יא")),
)
def test_supported_section_labels_render(leaf, label):
    assert hebrew.native_section_label(f"il/statute/x/{leaf}") == label


def test_hebrew_omission_labels_leave_english_omissions_alone():
    """Round-6 review: `(a) (1) omitted.` keeps its English reading."""

    for omission in ("omitted", "...", "weggefallen", "aufgehoben"):
        assert not completeness_module._EDITORIAL_OMISSION_ONLY.match(
            f"(a) (1) {omission}."
        )
    assert completeness_module._EDITORIAL_OMISSION_ONLY.match("(ג) (בוטל).")
    assert completeness_module._EDITORIAL_OMISSION_ONLY.match("(1) (ה1) בוטל.")


@pytest.mark.parametrize(
    ("reason", "source", "linked"),
    (
        ("חסרות הוראות סעיף 345י״א.", "(ה1) יש לפעול בהתאם לסעיף 345י.", False),
        ("חסרות הוראות סעיף 345י״א.", "(ה1) יש לפעול בהתאם לסעיף 345יא.", True),
        ('חסרות הוראות סעיף 345י"א.', "(ה1) יש לפעול בהתאם לסעיף 345יא.", True),
        ("חסרות הוראות סעיף 345יא.", "(ה1) יש לפעול בהתאם לסעיף 345י׳א.", True),
        # Typographic quotes a word processor puts for the gershayim.
        *(
            (f"חסרות הוראות סעיף 345י{mark}א.", source, linked)
            for mark in "“”‘’"
            for source, linked in (
                ("(ה1) יש לפעול בהתאם לסעיף 345י.", False),
                ("(ה1) יש לפעול בהתאם לסעיף 345יא.", True),
            )
        ),
        ("חסרות הוראות סעיף 345יא.", "(ה1) יש לפעול בהתאם לסעיף 345י”א.", True),
    ),
)
def test_a_marked_section_suffix_is_read_whole(reason, source, linked):
    issues = _analyze(_deferral_content("e1", reason), "(א) הוראה.\n" + source)

    assert (_deferral_issue(issues, 0) is None) is linked


@pytest.mark.parametrize("mark", "׳״'\"“”‘’")
def test_a_marked_suffix_is_another_section_than_the_current_one(mark):
    """`סעיף 14”א` is section 14יא, not section 14."""

    assert not hebrew.cites_current_section(f"סעיף 14{mark}א", CITATION)
    assert hebrew.cites_current_section("סעיף 14", CITATION)
    assert [
        hebrew._dependency_identity(match)
        for match, _instrument in hebrew._section_citations(f"סעיף 345י{mark}א")
    ] == [("345יא", ())]


@pytest.mark.parametrize(
    ("qualifier", "linked"),
    # Names compare whole, so the unqualified name is another law's too.
    (("(גמלאות)", True), ("", False), ("(משמעת)", False)),
)
def test_a_parenthetical_qualifier_distinguishes_laws_of_one_name(qualifier, linked):
    source = "(א) הוראה.\n(ה1) יש לפעול בהתאם לסעיף 42 לחוק שירות המדינה (גמלאות)."
    reason = f"חסרות הוראות סעיף 42 לחוק שירות המדינה {qualifier}".rstrip() + "."
    issues = _analyze(_deferral_content("e1", reason), source)

    assert (_deferral_issue(issues, 0) is None) is linked


@pytest.mark.parametrize(
    ("reason", "accepted"),
    (
        # Adversatives that are also ordinary words count wherever they stand,
        # punctuated or not and however much whitespace follows a comma.
        ("חסר מידע, אולם סעיף 345ב קודד.", False),
        ("חסר מידע. אולם סעיף 345ב קודד.", False),
        ("חסר מידע, להוציא סעיף 345ב.", False),
        ("חסר מידע, בעוד שסעיף 345ב קודד.", False),
        ("חסר מידע אולם סעיף 345ב קודד.", False),
        ("חסר מידע ואבל סעיף 345ב קודד.", False),
        ("חסרות הוראות זולת סעיף 345ב.", False),
        ("חסר מידע בעוד סעיף 345ב קודד.", False),
        ("חסר מידע,   אולם סעיף 345ב קודד.", False),
        ("חסר מידע,\t\t\tאבל סעיף 345ב קודד.", False),
        # A form or governor that fixes the ordinary word: an article or
        # preposition, a modal before the infinitive, a span of time after
        # ``בעוד``.
        ("חסרים נתוני האבל לפי סעיף 345ב.", True),
        ("חסרה הגדרת האולם שבסעיף 345ב.", True),
        ("חסר מידע כדי להוציא רישיון לפי סעיף 345ב.", True),
        ("חסר מידע, יש להוציא רישיון לפי סעיף 345ב.", True),
        ("חסרה הוראת סעיף 345ב, שתחול בעוד שנה.", True),
        ("חסרה הוראת סעיף 345ב, שתחול בעוד 30 יום.", True),
        # The bare noun, the ungoverned infinitive and a governor with a
        # prefix (``שיש``) read as the connective, stricter than English ("the
        # definition of a hall").
        ("חסרה הגדרת אולם שבסעיף 345ב.", False),
        ("חסרות ההוראות להוציא רישיון לפי סעיף 345ב.", False),
        ("חסר מידע שיש להוציא לפי סעיף 345ב.", False),
        ("חסר מידע ויש להוציא רישיון לפי סעיף 345ב.", False),
    ),
)
def test_ambiguous_adversatives_count_unless_their_form_fixes_the_word(
    reason, accepted
):
    source = "(א) הוראה.\n(ה1) רישיון העסק לאולם יינתן לפי סעיף 345ב."
    issues = _analyze(_deferral_content("e1", reason), source)

    assert (_deferral_issue(issues, 0) is None) is accepted


@pytest.mark.parametrize(
    "connective",
    (
        "אף־כי",
        "אם־כי",
        "אף־על־פי ש",
        "בכל־זאת",
        "יחד־עם־זאת",
        "הגם־ש",
        "גם־אם",
        "פרט־ל",
        "חוץ־מ",
        "לבד־מ",
        "אלא־אם",
        "אף־ש",
        "אלא־ש",
        "עם־זאת",
        "על־אף העובדה ש",
        "אף-על-פי ש",
        "ועם זאת",
        "ועל אף",
        "הגם כי",
        "בשעה ש",
        "בעת ש",
        "בזמן ש",
        "אע״פ ש",
        'אע"פ ש',
        "אעפ״י ש",
        'אעפ"י ש',
        "ובשעה ש",
        "עם כל זאת",
        "ועם כל זאת",
        "עם־כל־זאת",
        "עם כל זה",
        "ועם-כל-זה",
    ),
)
def test_compound_adversatives_take_any_separator(connective):
    glued = connective.endswith(("ש", "ל", "מ")) and connective[-2] in " ־-"
    text = f"חסר מידע {connective}{'' if glued else ' '}סעיף 345ב קודד."

    assert hebrew._ADVERSATIVE.search(text)
    assert not _accepted("e1", text)


def test_other_citations_use_the_english_citation_grammar():
    assert (
        hebrew._OTHER_CITATION.pattern
        == completeness_module._PRECISE_DEFERRAL_DEPENDENCY.pattern
    )
    assert hebrew._OTHER_CITATION.flags == (
        completeness_module._PRECISE_DEFERRAL_DEPENDENCY.flags
    )


_INSTRUMENTS = st.one_of(
    st.just(hebrew._CURRENT_INSTRUMENT),
    st.just(hebrew._UNRESOLVED_INSTRUMENT),
    st.tuples(
        st.sampled_from(("חוק", "פקודה", "תקנות")),
        st.lists(
            st.sampled_from(("הביטוח", "הלאומי", "שירות", "(גמלאות)")),
            min_size=1,
            max_size=3,
        ).map(tuple),
    ),
)


@settings(max_examples=300, deadline=None)
@given(st.lists(_INSTRUMENTS, max_size=6), _INSTRUMENTS)
def test_linked_instrument_index_agrees_with_the_pairwise_rule(cited, query):
    """Differential: the indexed lookup is `_same_instrument` against each entry."""

    linked = hebrew._LinkedInstruments()
    for instrument in cited:
        linked.add(instrument)

    assert linked.admits(query) == any(
        hebrew._same_instrument(query, instrument) for instrument in cited
    )


@pytest.mark.parametrize(
    ("hebrew_reason", "english_reason"),
    (
        (
            "החישוב תלוי בסעיף 345ב טרם קודד.",
            "The calculation depends on section 345b is not encoded.",
        ),
        (
            "לא ניתן לחשב את התשלום עד שיקודד סעיף 345ב.",
            "the payment cannot be computed until section 345b is encoded.",
        ),
        (
            "לא ניתן לחשב עד שיקודדו התנאים הקבועים בסעיף 345ב.",
            "Cannot be computed until the conditions set in section 345b are encoded.",
        ),
        (
            "לא ניתן לחשב את התשלום כי סעיף 345ב טרם קודד.",
            "the payment cannot be computed because section 345b is not yet encoded.",
        ),
    ),
)
def test_requires_until_and_cannot_routes_are_closed_to_hebrew(
    hebrew_reason, english_reason
):
    """Stricter than English, by design.

    These routes accept a reason only through English word lists (a bounded
    subject, "computed"/"encoded"), and a Hebrew word has more than one
    faithful translation, some outside those lists (`חישוב` is "calculation"
    and "computation"). A Hebrew reason names its missing section with
    `חסר`, `ללא` or `בלי` instead, as the retry hint shows.
    """

    assert not _accepted("e1", hebrew_reason)
    assert _english_accepted("d", english_reason)


# --- Rule `source:` citations ------------------------------------------------


@pytest.mark.parametrize(
    "reference",
    (
        "סעיף 14(ב)(2) לחוק הביטוח הלאומי",
        "סעיף 14(ב) (2) לחוק הביטוח הלאומי",
        "סעיף 14(ב)(2), לחוק הביטוח הלאומי",
        "סעיף 14(ב) שבחוק הביטוח הלאומי",
        "סעיף 14(ב) אשר בחוק הביטוח הלאומי",
        "סעיף 14(ב) מחוק הביטוח הלאומי",
        "סעיף 14(ב) מתוך חוק הביטוח הלאומי",
        "סעיף 14(ב) שמחוק הביטוח הלאומי",
        "סעיף 14(ב) שמן החוק הביטוח הלאומי",
        "סעיף 14(ב) שמתוך חוק הביטוח הלאומי",
        "סעיף 14(ב) אשר מחוק הביטוח הלאומי",
        "סעיף 14(ב) אשר מן החוק הביטוח הלאומי",
        "סעיף 14(ב) אשר מתוך חוק הביטוח הלאומי",
        "סעיף 14(ב) ל״חוק הביטוח הלאומי״",
        'סעיף 14(ב) ל"חוק הביטוח הלאומי"',
        "סעיף 14(ב) ל׳חוק הביטוח הלאומי׳",
        "סעיף 14(ב) ל'חוק הביטוח הלאומי'",
        "סעיף 14(ב) ל“חוק הביטוח הלאומי”",
        "סעיף 14(ב) ל‘חוק הביטוח הלאומי’",
        "סעיף 14(ב) ל«חוק הביטוח הלאומי»",
        "סעיף 14(ב) לאותה פקודה",
        "סעיף 14(ב) לאותו חוק",
        "סעיף 14(ב) באותן תקנות",
        "סעיף 14(ב) מאותו צו",
    ),
)
def test_another_laws_section_claims_no_branch_of_this_one(reference):
    """The qualifier binds the whole chain; no shorter prefix of it is claimed."""

    paths = completeness_module._paths_from_source_reference(
        reference, corpus_citation_path=CITATION
    )

    assert paths == set()
    assert not hebrew.cites_current_section(reference, CITATION)
    assert hebrew.cites_current_section("סעיף 14(ב)(2) לחוק זה", CITATION)


def _citation_scans(size: int):
    """The scans of ``test_native_citation_scans_stay_linear`` and their results.

    ``size`` counts quarters of the whole input, so 4 is the whole.
    """

    pattern = hebrew.current_branch_citation_pattern(CITATION, ("c",))
    names = ["צ" + hebrew.hebrew_numeral(value) for value in range(1, 800 * size + 1)]
    reads = functools.partial(
        hebrew.reason_names_source_bound_dependency, corpus_citation_path=CITATION
    )
    return (
        (
            "a long whitespace run after a citation",
            lambda: bool(pattern.search("סעיף 14(ג)" + " " * 16_000 * size + "x")),
            True,
        ),
        (
            "many citations in one reason clause",
            lambda: reads("חסר " + "סעיף 346 " * 2_000 * size, PRECISION_SOURCE),
            False,
        ),
        (
            "many reason clauses against many source citations of another section",
            lambda: reads(
                "חסר סעיף 346. " * 2_000 * size,
                "(ה1) " + "לפי סעיף 345ב. " * 2_000 * size,
            ),
            False,
        ),
        (
            "many reason fragments of one section, each a distinct query",
            lambda: reads(
                " ".join(f"חסר סעיף 346({index})." for index in range(2, 250 * size)),
                "(ה1) " + "לפי סעיף 346(1). " * 500 * size,
            ),
            False,
        ),
        (
            "many contextual references resolved against many named laws",
            lambda: len(
                hebrew._section_citations(
                    "חוק הביטוח הלאומי. " * 4_000 * size
                    + "סעיף 345ב לפקודה האמורה. " * 4_000 * size
                )
            ),
            4_000 * size,
        ),
        (
            "an outline that deepens without end",
            lambda: hebrew.hebrew_outline_markers(
                "(א) הוראה.\n(1) הוראה.\n" * 1_000 * size
            ),
            (),
        ),
        (
            "a citation chain thousands deep",
            lambda: bool(
                hebrew._linked_source_instruments(
                    "לפי סעיף 345ב" + "(1)" * 1_600 * size + "."
                )
            ),
            False,
        ),
        (
            "a long whitespace run inside an instrument name",
            lambda: len(
                hebrew._section_citations(
                    "סעיף 345ב לחוק" + " " * 16_000 * size + "הביטוח."
                )
            ),
            1,
        ),
        (
            "one long clause of citations",
            lambda: reads("חסר " + "סעיף 346 " * 16_000 * size, PRECISION_SOURCE),
            False,
        ),
        (
            "many punctuation-joined signals before one citation",
            lambda: reads("חסר," * 4_000 * size + "סעיף 346.", PRECISION_SOURCE),
            False,
        ),
        (
            "many citations after one negation",
            lambda: bool(
                hebrew._linked_source_instruments(
                    "אין לפעול " + "בהתאם לסעיף 345ב " * 2_000 * size + "."
                )
            ),
            False,
        ),
        (
            "many reason citations against many distinct named laws",
            lambda: reads(
                "חסר סעיף 345ב לחוק תתתתת. " * 800 * size,
                "(ה1) " + "".join(f"לפי סעיף 345ב לחוק {name}. " for name in names),
            ),
            False,
        ),
    )


def test_native_citation_scans_stay_linear():
    """A long whitespace run or many citations cannot make a scan quadratic.

    Each scan runs on a quarter of its input and on the whole. Four times the
    input costs about four times the CPU when the scan is linear and sixteen
    times when it is quadratic, so the bound is eight, midway between them on
    a log scale. The bound is on growth, not on the total, because a fixed
    total cannot do both jobs: GitHub's runners spent 5.5 to 6.2 CPU seconds
    on the linear scans, which take about 2.4 on a development machine, while
    a quadratic lookup (every bisection made a linear scan) costs only about
    4.6 times the linear scans at these sizes.
    """

    for (name, quarter_scan, quarter_result), (_, whole_scan, whole_result) in zip(
        _citation_scans(1), _citation_scans(4), strict=True
    ):
        # CPU time, not wall time, and the least of two interleaved runs at
        # each size, so neither a cold first run nor waiting on a busy host
        # counts.
        quarter = whole = float("inf")
        for _run in range(2):
            started = time.process_time()
            assert quarter_scan() == quarter_result, name
            quarter = min(quarter, time.process_time() - started)
            started = time.process_time()
            assert whole_scan() == whole_result, name
            whole = min(whole, time.process_time() - started)
        assert whole < 8 * quarter + 0.05, (name, quarter, whole)


def _least_cpu_seconds(call, *, runs: int = 3, repetitions: int = 1) -> float:
    """The least CPU time of ``runs`` runs of ``repetitions`` calls, per call.

    CPU time, not wall time, so a shared host's scheduling cannot fail a bound.
    """

    least = float("inf")
    for _run in range(runs):
        started = time.process_time()
        for _repetition in range(repetitions):
            call()
        least = min(least, time.process_time() - started)
    return least / repetitions


def test_an_equal_copy_of_a_text_is_scanned_in_linear_time():
    """Review round 12, finding 4: no index is looked up by an equal whole text.

    Caches keyed by the whole text compared it in full once per citation when
    an equal copy came back as another object, so the copy's scan grew
    quadratically: 0.05, 0.19 and 0.82 CPU seconds at 8,000, 16,000 and 32,000
    citations, against 0.01, 0.02 and 0.06 for the first scan.
    """

    first, copy = [], []
    for count in (8_000, 16_000, 32_000):
        text = "סעיף 1. " * count
        equal = text.encode().decode()
        assert equal == text and equal is not text
        first.append(
            _least_cpu_seconds(functools.partial(hebrew._section_citations, text))
        )
        copy.append(
            _least_cpu_seconds(functools.partial(hebrew._section_citations, equal))
        )
    # Four times the citations cost about four times the CPU when the scan is
    # linear, sixteen times when it is quadratic.
    assert first[-1] < 10 * first[0], first
    assert copy[-1] < 10 * copy[0], copy
    # The copy costs what the first scan costs, not a comparison per citation.
    assert copy[-1] < 2 * first[-1] + 0.01, (first, copy)


def test_a_reason_and_its_equal_copy_are_read_in_linear_time():
    """Review round 12, finding 4, through the shared checker.

    Before, a reason of 32,000 citations took 0.49 CPU seconds and its equal
    copy 3.2; both verdicts were, and are, refusals.
    """

    scope = "(ה1) לפי סעיף 345ב."

    def read(reason: str) -> bool:
        return completeness_module._reason_dependency_is_source_bound(
            reason, scope, corpus_citation_path=CITATION, path=("e1",)
        )

    first, copy = [], []
    for count in (8_000, 16_000, 32_000):
        reason = "missing חסר " + "סעיף 346. " * count
        equal = reason.encode().decode()
        if count == 8_000:
            assert not read(reason)
            assert not read(equal)
        first.append(_least_cpu_seconds(functools.partial(read, reason), runs=1))
        copy.append(_least_cpu_seconds(functools.partial(read, equal), runs=1))
    assert first[-1] < 10 * first[0], first
    assert copy[-1] < 10 * copy[0], copy
    assert copy[-1] < 2 * first[-1] + 0.05, (first, copy)


def test_a_long_run_of_digits_is_read_once():
    """Review round 12, finding 5: a rate is matched from the first digit of a run.

    Unanchored, the rate retried every suffix of a run of digits with no
    percent sign: 0.009 CPU seconds at 1,000 digits, 0.59 at 8,000.
    """

    per_call = []
    for size in (1_000, 2_000, 4_000, 8_000):
        digits = "1" * size
        assert not hebrew.states_explicit_computation(digits)
        per_call.append(
            _least_cpu_seconds(
                functools.partial(hebrew.states_explicit_computation, digits),
                repetitions=32_000 // size,
            )
        )
    # Eight times the digits cost about eight times the CPU when the scan is
    # linear, sixty-four times when it is quadratic.
    assert per_call[-1] < 24 * per_call[0], per_call
    assert hebrew.states_explicit_computation("1" * 8_000 + "% מהכנסתו")


@pytest.mark.parametrize(
    ("source_reference", "covers"),
    (
        ("סעיף 14(ג)", True),
        ('סעיף 14(ג), הגדרת "מבוטח"', True),
        ("סעיף 14(ג) לחוק הביטוח הלאומי", False),
        ("סעיף 15(ג)", False),
    ),
)
def test_native_rule_source_citation_claims_its_branch(source_reference, covers):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {CITATION}
rules:
  - name: copy_submission_required
    kind: derived
    dtype: Judgment
    period: Day
    source: {json.dumps(source_reference, ensure_ascii=False)}
    versions:
      - formula: 'true'
"""

    uncovered = _structure_paths(_analyze(content))

    assert (("c",) not in uncovered) is covers


# --- Review round 14 ------------------------------------------------------------
#
# Rounds 11 to 13 each closed the listed reproducers, and the next review found
# new members of the same families. Each test below pins a family: it
# enumerates every member, or samples all of them, rather than the reproducer.

# Finding 2: the fold drops marks by Unicode property, not from a list.


@functools.cache
def _unicode_format_characters() -> frozenset[int]:
    """Every ``Cf`` code point, read from the Unicode database one by one."""

    return frozenset(
        code
        for code in range(sys.maxunicode + 1)
        if unicodedata.category(chr(code)) == "Cf"
    )


_HEBREW_POINTS = frozenset(
    code for code in range(0x0591, 0x05C8) if unicodedata.category(chr(code)) == "Mn"
)


def test_the_fold_drops_every_format_character_and_hebrew_mark():
    """Every format character and Hebrew point folds, and no letter's mark."""

    folded = hebrew.folded_marks()
    format_characters = _unicode_format_characters()

    assert format_characters <= folded
    assert _HEBREW_POINTS <= folded
    # The review's omissions: the Mongolian vowel separator and the invisible
    # function application, times, separator and plus.
    assert {0x180E, 0x2061, 0x2062, 0x2063, 0x2064} <= folded
    # The surrogates YAML decodes an escaped astral mark into.
    assert _SURROGATES <= folded
    # Beyond them, only the invisible marks of no script.
    default_ignorables = {
        code
        for low, high in hebrew.DEFAULT_IGNORABLE_RANGES
        for code in range(low, high + 1)
    }
    assert default_ignorables <= folded
    white_space_controls = {0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x85}
    controls = {
        code
        for code in range(sys.maxunicode + 1)
        if unicodedata.category(chr(code)) == "Cc" and code not in white_space_controls
    }
    assert len(controls) == 59
    assert controls <= folded
    for code in (
        folded
        - format_characters
        - _HEBREW_POINTS
        - _SURROGATES
        - default_ignorables
        - controls
    ):
        name = unicodedata.name(chr(code))
        assert unicodedata.category(chr(code)) == "Mn", name
        assert "VARIATION SELECTOR" in name or name == "COMBINING GRAPHEME JOINER"
    # The maqaf, paseq, sof pasuq and inverted nun are punctuation.
    assert not {0x05BE, 0x05C0, 0x05C3, 0x05C6} & folded
    # Combining marks of other scripts are parts of letters.
    assert not (set(range(0x0300, 0x0370)) - {0x034F}) & folded


def test_combining_marks_outside_hebrew_keep_their_letters():
    """A decomposed German umlaut beside Hebrew is still the umlaut."""

    text = "für Absatz 2 und ká – חסר סעיף 2"

    assert hebrew.fold_marks(text) == text
    assert hebrew.fold_marks("ü⁣ חסר") == "ü חסר"


# The review's eight negators, four Hebrew and four English and German.
_FOLD_NEGATORS = ("לא", "אין", "אינו", "בלתי", "not", "never", "nicht", "keine")


@pytest.mark.parametrize(
    ("negator", "position"),
    [
        (negator, position)
        for negator in _FOLD_NEGATORS
        for position in range(1, len(negator))
    ],
)
def test_no_invisible_mark_inside_a_negator_hides_it(negator, position):
    """Finding 2's family, enumerated in full through the CI method.

    Each ``Cf`` code point in Unicode, each Hebrew point and each other folded
    mark, at each point inside each negator, before a signal and before a
    source link: the verdict is the unmarked negator's, a refusal.
    """

    def verdicts(word: str) -> tuple[bool, bool]:
        return (
            _accepted("a", f"{word} חסר סעיף 2", REVIEW_SOURCE),
            _accepted("a", "חסר סעיף 2", f"(א) {word} מחושב בהתאם לסעיף 2."),
        )

    expected = verdicts(negator)
    # Keep round 14's complete Cf/Mn family here. Round 15 separately
    # enumerates every Default_Ignorable_Code_Point through all four guards
    # and every token/position in test_source_completeness_hebrew_unicode.py;
    # repeating those reserved code points for each old negator adds no
    # coverage and multiplies the normal suite's analyzer work.
    marks = (
        _unicode_format_characters()
        | _HEBREW_POINTS
        | {
            code
            for code in hebrew.folded_marks()
            if unicodedata.category(chr(code)) in {"Cf", "Mn"}
        }
    ) - _SURROGATES

    assert expected == (False, False)
    differing = [
        f"U+{code:04X}"
        for code in sorted(marks)
        if verdicts(negator[:position] + chr(code) + negator[position:]) != expected
    ]
    assert not differing, differing


# The review's five marked/unmarked pairs: every letter of the body and reason
# carries a holam and a right-to-left mark.
_OFFSET_PAIRS = (
    ("a", "(א) ישולם לפי סעיף 2.\n(ב) הוראה נוספת.", "חסר סעיף 2"),
    ("a", "(א) ישולם לפי סעיף 2.\n(ב) הוראה נוספת.", "לא חסר סעיף 2"),
    ("a", "(א) אינו מחושב בהתאם לסעיף 2.\n(ב) הוראה נוספת.", "חסר סעיף 2"),
    (
        "a/1",
        "(א) (1) ישולם לפי סעיף 2.\n(2) הוראה נוספת.\n(ב) הוראה נוספת.",
        "חסר סעיף 2",
    ),
    ("a", "(א) ישולם לפי סעיף 2.\n(ב) הוראה נוספת.", "חסר סעיף 2. אולם הוא זמין."),
)


def _point_every_letter(text: str) -> str:
    return "".join(char + ("ֹ‏" if "א" <= char <= "ת" else "") for char in text)


@pytest.mark.parametrize(("branch", "source", "reason"), _OFFSET_PAIRS)
@pytest.mark.parametrize("label_marks", ("", "‏", "ֹ⁣"))
def test_marked_sources_keep_their_offsets_verdicts_and_issues(
    branch, source, reason, label_marks
):
    """Offsets: branches slice the marked text; verdicts and issues are the plain ones.

    The review marked bodies and reasons only; the labels are marked here too.
    """

    marked = "\n".join(
        line[: line.index(")")]
        + label_marks
        + line[line.index(")") : line.rindex(")") + 1]
        + _point_every_letter(line[line.rindex(")") + 1 :])
        for line in source.splitlines()
    )
    content = _deferral_content(branch, reason)
    marked_content = _deferral_content(branch, _point_every_letter(reason))
    plain_branches = recognize_source_structure(source, corpus_citation_path=CITATION)
    marked_branches = recognize_source_structure(marked, corpus_citation_path=CITATION)

    assert [item.path for item in marked_branches] == [
        item.path for item in plain_branches
    ]
    assert [item.label for item in marked_branches] == [
        item.label for item in plain_branches
    ]
    for item in marked_branches:
        assert marked[item.start : item.end].strip() == item.text
    assert _analyze(marked_content, marked) == _analyze(content, source)
    assert _accepted(branch, _point_every_letter(reason), marked) is _accepted(
        branch, reason, source
    )


# Finding 4: outline labels are read on the folded text.


_UNENCODED_CONTENT = (
    "format: rulespec/v1\n"
    "module:\n"
    "  source_verification:\n"
    f"    corpus_citation_path: {CITATION}\n"
    "rules: []\n"
)


@pytest.mark.parametrize("label", ("(א‏)", "(אֹ)", "(⁣א️)"))
def test_a_marked_label_still_opens_its_branch(label):
    """The review's table: a marked label hides no branch from the check."""

    source = f"{label} הוראה."
    branches = recognize_source_structure(source, corpus_citation_path=CITATION)
    issues = _analyze(_UNENCODED_CONTENT, source)

    assert [item.path for item in branches] == [("a",)]
    assert [item.label for item in branches] == ["(א)"]
    assert _structure_paths(issues) == {("a",)}
    assert issues == _analyze(_UNENCODED_CONTENT, "(א) הוראה.")


def test_a_marked_repeal_note_is_still_no_branch():
    plain = "(א) הוראה.\n(ב) (בוטל)"
    marked = "(א) הוראה.\n(ב‏) (בוטלָ)"

    assert [
        item.path
        for item in recognize_source_structure(marked, corpus_citation_path=CITATION)
    ] == [
        item.path
        for item in recognize_source_structure(plain, corpus_citation_path=CITATION)
    ]


@pytest.mark.parametrize(
    ("text", "body"),
    (
        ("(א‏) הוראה.", "הוראה."),
        ("(אֹ)(1⁣)‏ הוראה", "הוראה"),
        ("\n" * 300 + "(ב) (1) גוף", "גוף"),
        ("(" + "‏" * 500 + "א) גוף", "גוף"),
        ("(a) The rule", None),
        ("(1) The rule", None),
        ("(" + "ֹ" * 300, None),
    ),
)
def test_a_marked_leading_chain_is_stripped_from_the_original(text, body):
    assert hebrew.strip_leading_outline_chain(text) == body


@settings(max_examples=150, deadline=None)
@given(_outlines(), st.data())
def test_marks_inside_outline_labels_change_no_branch_or_issue(nodes, data):
    """Finding 4's family: any outline, any folded mark anywhere inside its labels.

    Paths, labels and every completeness issue are the unmarked outline's, and
    each branch slices its own text from the marked source.
    """

    lines, paths = _render_outline(nodes)
    plain = "\n".join(lines)
    pieces, inside = [], False
    for char in plain:
        if inside:
            pieces.append(
                data.draw(st.one_of(st.just(""), st.sampled_from(_INVISIBLE_MARKS)))
            )
        pieces.append(char)
        inside = (inside or char == "(") and char != ")"
    marked = "".join(pieces)
    # An imprecise deferral of the first branch, so a deferral message names
    # its label too.
    content = _deferral_content(paths[0][0], "חסר מידע")
    plain_branches = recognize_source_structure(plain, corpus_citation_path=CITATION)
    marked_branches = recognize_source_structure(marked, corpus_citation_path=CITATION)

    assert [item.path for item in marked_branches] == paths
    assert [item.path for item in plain_branches] == paths
    assert [item.label for item in marked_branches] == [
        item.label for item in plain_branches
    ]
    for marked_branch, plain_branch in zip(
        marked_branches, plain_branches, strict=True
    ):
        assert marked[marked_branch.start : marked_branch.end].strip() == (
            marked_branch.text
        )
        assert hebrew.fold_marks(marked_branch.text) == plain_branch.text
    assert _analyze(content, marked) == _analyze(content, plain)


# Finding 1: ``בלתי`` is a negation under any chain of prefix letters.

_BILTI_CHAINS = tuple(
    "".join(chain)
    for length in range(4)
    for chain in itertools.product(hebrew._PREFIX_LETTERS, repeat=length)
)


def test_bilti_is_a_negation_under_every_prefix_chain():
    """Each guard reads ``בלתי`` under each chain of up to three prefix letters."""

    assert len(_BILTI_CHAINS) == 1 + 8 + 8**2 + 8**3
    for chain in _BILTI_CHAINS:
        word = chain + "בלתי"
        assert hebrew._REASON_NEGATION.fullmatch(word), word
        assert hebrew._NEGATED_SIGNAL_PREFIX.search(word), word
        negation = hebrew._NEGATIONS.fullmatch(word)
        assert negation, word
        # A chain with the relative ``ש`` or ``ד`` opens a relative clause.
        assert hebrew._is_relative(negation) is bool(set(chain) & {"ש", "ד"}), word


@pytest.mark.parametrize("length", range(4))
def test_every_prefix_chain_before_bilti_refuses_as_the_bare_word_does(length):
    """Finding 1's family, enumerated in full through the CI method.

    Every chain of ``length`` prefix letters before ``בלתי``, in each place a
    negation governs: earlier in the clause through a complement (``הטענה
    הבלתי נכונה ש``), just before the signal (``לבלתי היות חסר``), between
    the signal and its citation, and before a source link. Without
    ``בלתי`` each text is accepted.
    """

    def verdicts(word: str) -> tuple[bool, ...]:
        return (
            _accepted("a", f"הטענה {word} נכונה שחסר סעיף 2", REVIEW_SOURCE),
            _accepted("a", f"{word} היות חסר סעיף 2", REVIEW_SOURCE),
            _accepted("a", f"חסר {word} סעיף 2", REVIEW_SOURCE),
            _accepted("a", "חסר סעיף 2", f"(א) {word} מחושב בהתאם לסעיף 2."),
        )

    assert verdicts("") == (True,) * 4
    assert verdicts("בלתי") == (False,) * 4
    accepted = [
        chain
        for chain in _BILTI_CHAINS
        if len(chain) == length and any(verdicts(chain + "בלתי"))
    ]
    assert not accepted, accepted


# Each single prefix letter before ``לא`` and ``אין``: read as a negation, or
# excluded because the letters make another word, with the reading that
# excludes it. Each excluded word gets its word-for-word English verdict.
_PREFIXED_NEGATION_READINGS = (
    ("ולא", True, "And not missing section 2."),
    ("ואין", True, "And there is no missing section 2."),
    ("שלא", True, "That not missing section 2."),
    ("שאין", True, "That there is no missing section 2."),
    ("דלא", True, "That not missing section 2."),
    ("דאין", True, "That there is no missing section 2."),
    # "Surely", a rhetorical affirmation, and "is there not", its question.
    ("הלא", False, "Surely missing section 2."),
    ("האין", False, "Is there not missing section 2?"),
    # "Without", a preposition: the link guard reads it, and a reason's
    # without-signals are ``בלי``, ``ללא`` and ``מבלי``.
    ("בלא", False, "Without missing section 2."),
    # Round 16: באין is lexical negative absence. The incomplete missing
    # fragment is explicitly Hebrew-stricter in ROUND_16_INTENDED_STRICTER_CONTEXTS.
    ("באין", True, "In the absence of missing section 2."),
    ("כלא", False, "Prison: missing section 2."),
    # "As nothing" (``כאין וכאפס``).
    ("כאין", False, "As nothing missing section 2."),
    # "Without", a missing-dependency signal of its own.
    ("ללא", False, "Without missing section 2."),
    # "To no end" (``לאין ספור``, "countless").
    ("לאין", False, "Countless missing section 2."),
    ("מלא", False, "Full: missing section 2."),
    ("מאין", False, "Whence missing section 2."),
)


def test_every_single_prefix_letter_is_listed():
    words = {word for word, _negation, _english in _PREFIXED_NEGATION_READINGS}

    assert words == {
        letter + negation
        for letter in hebrew._PREFIX_LETTERS
        for negation in ("לא", "אין")
    }


@pytest.mark.parametrize(("word", "negation", "english"), _PREFIXED_NEGATION_READINGS)
def test_each_prefix_of_a_negation_is_read_or_excluded(word, negation, english):
    """The narrow ``לא``/``אין`` grammar: each exclusion is another word."""

    lexical_negation = negation or word in {"בלא", "ללא"}
    assert bool(hebrew._REASON_NEGATION.fullmatch(word)) is lexical_negation
    assert bool(hebrew._NEGATED_SIGNAL_PREFIX.search(word)) is lexical_negation
    hebrew_verdict = _accepted("a", f"{word} חסר סעיף 2", REVIEW_SOURCE)
    if negation:
        assert not hebrew_verdict
    else:
        assert hebrew_verdict is _english_accepted("a", english, REVIEW_ENGLISH_SOURCE)


# Finding 3: exclusions and negations read across languages, symmetrically.
#
# A negation or exclusion of one language negates a computation word or link
# of another as its translation's negation does. The matrix below crosses a
# negator of each language with a computation word and a link of each, and
# compares each mixed source against its word-for-word English translation.


# A NamedTuple, not a dataclass: review scripts load this module unregistered.
class _Word(NamedTuple):
    text: str
    language: str
    english: str
    # A negator's kind ("negation", "without", "except"); for an object,
    # whether it is a computation word.
    kind: str


_CROSS_NEGATORS = (
    _Word("לא", "he", "not", "negation"),
    _Word("אינו", "he", "not", "negation"),
    _Word("אין", "he", "not", "negation"),
    _Word("בלתי", "he", "not", "negation"),
    _Word("בלי", "he", "without", "without"),
    _Word("ללא", "he", "without", "without"),
    _Word("מבלי", "he", "without", "without"),
    _Word("למעט", "he", "except", "except"),
    _Word("מלבד", "he", "except", "except"),
    _Word("not", "en", "not", "negation"),
    _Word("never", "en", "never", "negation"),
    _Word("prohibited", "en", "prohibited", "negation"),
    _Word("excluded", "en", "excluded", "negation"),
    _Word("without", "en", "without", "without"),
    _Word("except", "en", "except", "except"),
    _Word("other than", "en", "other than", "except"),
    _Word("nicht", "de", "not", "negation"),
    _Word("nie", "de", "never", "negation"),
    _Word("niemals", "de", "never", "negation"),
    _Word("verboten", "de", "prohibited", "negation"),
    _Word("ausgeschlossen", "de", "excluded", "negation"),
    _Word("ohne", "de", "without", "without"),
    _Word("außer", "de", "except", "except"),
    _Word("ausser", "de", "except", "except"),
    _Word("ausgenommen", "de", "except", "except"),
    _Word("anders als", "de", "other than", "except"),
    _Word("anstatt", "de", "rather than", "except"),
)
_CROSS_OBJECTS = (
    _Word("מחושב", "he", "computed", "computation"),
    _Word("חישוב", "he", "calculation", "computation"),
    _Word("דיחוי", "he", "delay", "other"),
    _Word("computed", "en", "computed", "computation"),
    _Word("calculation", "en", "calculation", "computation"),
    _Word("delay", "en", "delay", "other"),
    _Word("berechnet", "de", "computed", "computation"),
    _Word("Berechnung", "de", "calculation", "computation"),
    _Word("Verzögerung", "de", "delay", "other"),
)
# Each link with its citation. The German links take the English verdict of
# "under" and "pursuant to", which neither reader checks for negation.
_CROSS_LINKS = (
    _Word("בהתאם לסעיף 2", "he", "in accordance with section 2", "link"),
    _Word("לפי סעיף 2", "he", "under section 2", "link"),
    _Word("in accordance with section 2", "en", "in accordance with section 2", "link"),
    _Word("under section 2", "en", "under section 2", "link"),
    _Word("nach section 2", "de", "under section 2", "link"),
    _Word("gemäß section 2", "de", "pursuant to section 2", "link"),
)


@functools.cache
def _english_link_verdict(source: str) -> bool:
    return _english_accepted("a", "Missing section 2.", source)


def _mixed_link_verdicts(negator: str, obj: str, link: _Word) -> tuple[bool, ...]:
    """The mixed source's verdict on each corpus path that can read its link.

    A Hebrew link is read under the Israeli path. Any other is read by the
    shared English routes, under the English path and the Israeli one.
    """

    words = " ".join(part for part in (negator, obj, link.text) if part)
    if link.language == "he":
        return (_accepted("a", "חסר סעיף 2", f"(א) משולם {words}."),)
    return (
        _english_accepted("a", "Missing section 2.", f"(a) It is paid {words}."),
        _accepted("a", "Missing section 2.", f"(א) It is paid {words}."),
    )


def _intended_stricter(negator: _Word, obj: _Word, link: _Word) -> str | None:
    """Why the mixed source is refused where its translation is accepted, if it is."""

    # ``without delay`` and ``ohne Verzögerung`` exclude nothing in either
    # language; any other negator here negates in the mixed text.
    if (
        negator.kind == "without"
        and obj.kind != "computation"
        and "he" not in (negator.language, obj.language)
    ):
        return None
    if link.text.startswith("לפי"):
        return "a negated `לפי` link; English does not check `under` (round 13)"
    if (
        negator.kind == "without"
        and obj.kind != "computation"
        and (link.language == "he" or link.text.startswith("in accordance with"))
    ):
        # ``בלי דיחוי בהתאם ל`` is refused, and English "without delay in
        # accordance with" links; ``without`` and ``ohne`` before a Hebrew
        # word read as ``בלי`` does. Only a link read for negation shows it.
        return "a Hebrew exclusion excludes whatever follows it"
    return None


@pytest.mark.parametrize("negator", _CROSS_NEGATORS, ids=lambda word: word.text)
def test_every_cross_language_exclusion_gets_its_translations_verdict(negator):
    """Finding 3's family, enumerated: 944 mixed sources against their translations.

    For this negator and every computation word (and, after an exclusion,
    every other word) and link of the three languages, where the three are
    not all of one language, the mixed source gets its word-for-word English
    translation's verdict. It is refused where that translation is accepted
    only in the two labelled classes of ``_intended_stricter``, and never
    accepted where the translation is refused. Without the negator, every
    such source is accepted.
    """

    compared = 0
    for obj, link in itertools.product(_CROSS_OBJECTS, _CROSS_LINKS):
        if len({negator.language, obj.language, link.language}) == 1:
            continue
        if obj.kind != "computation" and negator.kind != "without":
            continue
        translation = f"(a) It is paid {negator.english} {obj.english} {link.english}."
        english = _english_link_verdict(translation)
        reason = _intended_stricter(negator, obj, link)
        expected = (False, True) if reason else (english, english)
        for mixed in _mixed_link_verdicts(negator.text, obj.text, link):
            assert (mixed, english) == expected, (negator.text, obj.text, link.text)
        assert all(_mixed_link_verdicts("", obj.text, link)), (obj.text, link.text)
        compared += 1
    assert compared


def test_the_cross_language_matrix_covers_every_language_pairing():
    """Each negator language meets each object and link language, all 27 ways."""

    triples = {
        (negator.language, obj.language, link.language)
        for negator, obj, link in itertools.product(
            _CROSS_NEGATORS, _CROSS_OBJECTS, _CROSS_LINKS
        )
    }
    assert triples == set(itertools.product(("he", "en", "de"), repeat=3))
    mixed = sum(
        1
        for negator, obj, link in itertools.product(
            _CROSS_NEGATORS, _CROSS_OBJECTS, _CROSS_LINKS
        )
        if len({negator.language, obj.language, link.language}) > 1
        and (obj.kind == "computation" or negator.kind == "without")
    )
    assert mixed == 944


# ``_SOURCE_LINK_NEGATION_LANGUAGE`` as main had it, inline in
# ``_source_link_scope_has_unreset_negation``: the English vocabulary alone.
_MAIN_LINK_NEGATION = re.compile(
    r"\b(?:not|never|neither|nor|cannot|nowhere|prohibit\w*|barred|"
    r"forbidden|precluded|prevented|excluded|denied|except)\b|"
    r"\b(?:denies?|rejects?|disclaims?)\b|"
    r"\b(?:false|untrue|impossible)\s+that\b|"
    r"(?:^|[,;:]\s*|\b(?:and|but|or|that|while|whereas)\s+)"
    r"no\s+(?:amount|tax|liability|computation|calculation|"
    r"determination|rate|base|statute|rule|provision|requirement)\b"
    r"(?:\s+[A-Za-z-]+){0,3}\s+(?:is|are|was|were|shall|may|must|will|"
    r"would|can|could|does|do|did|states?|provides?|requires?|"
    r"determines?|calculates?|computes?|applies?|arises?)\b|"
    r"\b(?:ca|do|does|did|is|are|was|were|"
    r"shall|will|would|could|should|must|need)n['’]t\b|"
    r"\b(?:fails?|refuses?|unable)\s+to\b|"
    r"\bwithout\s+(?:being\s+)?(?:determin\w*|comput\w*|calculat\w*|"
    r"appl\w*|impos\w*|aris\w*)\b|"
    r"\b(?:in\s+no\s+(?:case|event)|under\s+no\s+circumstances|"
    r"at\s+no\s+time|by\s+no\s+means|in\s+no\s+way|"
    r"on\s+no\s+account|under\s+no\s+conditions?|"
    r"in\s+no\s+(?:manner|respect))\b|"
    r"\b(?:rather|other)\s+than\b",
    flags=re.IGNORECASE,
)
# English words, the English link negators' own among them, and punctuation.
_ENGLISH_LINK_WORDS = (
    *(word for form in _OTHER_LINK_NEGATORS[:44] for word in form.split()),
    "being",
    "delay",
    "the",
    "amount",
    "is",
    "computed",
    "calculated",
    "applied",
    "determined",
    "imposed",
    "arising",
    "in",
    "accordance",
    "with",
    "under",
    "section",
    "2",
    "and",
    "that",
    "Statute",
    "zone",
    "nine",
    "never-ending",
)


@settings(max_examples=300, deadline=None)
@given(
    st.lists(st.sampled_from(_ENGLISH_LINK_WORDS), max_size=16),
    st.lists(st.sampled_from((" ", "  ", ", ", "; ", ": ", "\n")), max_size=16),
)
@example(["without", "being", "computed"], [])
@example(["and", "no", "amount", "is"], [])
@example(["other", "than"], [])
def test_the_link_vocabulary_reads_english_text_as_main_did(words, separators):
    """Differential: on English text the vocabulary matches exactly main's matches.

    Every alternative round 14 added needs a German word, ``zu`` or a Hebrew
    letter, so text of English words, the link negators' own included, is
    read with the same spans as main read it; and with no Hebrew script the
    shared checker has no cross-language reading of it.
    """

    text = "".join(
        word + (separators[index] if index < len(separators) else " ")
        for index, word in enumerate(words)
    )
    assert [
        match.span()
        for match in completeness_module._SOURCE_LINK_NEGATION_LANGUAGE.finditer(text)
    ] == [match.span() for match in _MAIN_LINK_NEGATION.finditer(text)]
    assert hebrew.cross_language_text(text) is None
    assert hebrew.english_reading_view(text) is None


def test_the_main_vocabulary_is_the_first_part_of_the_shared_one():
    assert completeness_module._SOURCE_LINK_NEGATION_LANGUAGE.pattern.startswith(
        _MAIN_LINK_NEGATION.pattern + "|"
    )
    assert _OTHER_LINK_NEGATORS[43] == "other than"


@pytest.mark.parametrize("exclusion", ("without", "ohne", "WITHOUT", "Ohne"))
@pytest.mark.parametrize("filler", ("", "being ", "zu "))
@pytest.mark.parametrize("word", ("חישוב", "מחושב", "דיחוי", "התשלום"))
def test_without_and_ohne_exclude_any_following_hebrew_word(exclusion, filler, word):
    """As ``בלי`` does; before an English or German word only a computation word."""

    negation = completeness_module._SOURCE_LINK_NEGATION_LANGUAGE
    text = f"{exclusion} {filler}{word}"
    found = negation.search(text)
    assert found and found.span() == (0, len(text) - len(word)), text
    for other in ("delay", "Verzögerung", "the", "Zahlung"):
        assert not negation.search(f"{exclusion} {filler}{other}"), other


# Hebrew negations before English signals and links: the converse direction.

_ENGLISH_PAID_SOURCE = "(a) It is paid in accordance with section 2."


def _english_routes_accept(reason: str, english_source: str) -> tuple[bool, bool]:
    """A mixed reason under the English path and the Israeli one."""

    return (
        _english_accepted("a", reason, english_source),
        _accepted("a", reason, english_source.replace("(a)", "(א)", 1)),
    )


@st.composite
def _hebrew_link_negations(draw) -> str:
    """A Hebrew link negation or exclusion, under its prefixes, marked."""

    negation = draw(st.sampled_from(_NEGATION_PREFIXES)) + draw(
        st.sampled_from(
            _LINK_NEGATORS + ("בלי", "ללא", "מבלי", "למעט", "מלבד", "נאסר", "אסור")
        )
    )
    return draw(_marked(negation))


@settings(max_examples=120, deadline=None)
@given(
    _negations(),
    st.lists(st.sampled_from(("it", "is", "true", "that", "the", "claim")), max_size=4),
    st.booleans(),
)
@example("לא", [], False)
@example("אין זה נכון ש", [], False)
@example("לא", [], True)
@example("ודלא", ["it", "is"], False)
@example("לא\u009b", [], False)
def test_a_hebrew_negation_before_an_english_signal_refuses_it(
    negation, words, between
):
    """Polarity: a Hebrew negation flips an accepted English reason.

    The mirror of ``test_any_negator_before_an_accepted_signal_refuses_it``.
    A Hebrew negation under any prefix, pointed or marked, anywhere earlier
    in the clause than ``missing``, or between it and its citation, refuses
    the reason the same words accept without it, under either corpus path. In
    a clause of its own before the reason, it refuses nothing.
    """

    lead = " ".join(words)
    affirmed = f"{lead} missing section 2.".strip()
    negated = (
        f"{lead} missing {negation} section 2."
        if between
        else f"{lead} {negation} missing section 2."
    ).strip()

    assert _english_routes_accept(affirmed, _ENGLISH_PAID_SOURCE) == (True, True)
    assert _english_routes_accept(negated, _ENGLISH_PAID_SOURCE) == (False, False)
    assert _english_routes_accept(
        f"{negation} ברור. {affirmed}", _ENGLISH_PAID_SOURCE
    ) == (True, True)


@settings(max_examples=120, deadline=None)
@given(
    _hebrew_link_negations(),
    st.lists(st.sampled_from(("the", "amount", "is", "החישוב")), max_size=3),
)
@example("לא", [])
@example("דלא", [])
@example("אינו", ["the", "amount", "is"])
@example("למעט", [])
@example("ולכשלמעט", [])
def test_a_hebrew_negation_before_an_english_link_unlinks_it(negation, words):
    """Polarity: a Hebrew link negation negates an ``in accordance with`` link.

    The mirror of ``test_any_negator_before_an_accepted_link_unlinks_it``,
    read by the English link guard under either corpus path.
    """

    lead = " ".join(words)
    affirmed = f"(a) It is paid {lead} computed in accordance with section 2."
    negated = f"(a) It is paid {lead} {negation} computed in accordance with section 2."

    assert _english_routes_accept("Missing section 2.", affirmed) == (True, True)
    assert _english_routes_accept("Missing section 2.", negated) == (False, False)


def test_an_english_reset_after_a_hebrew_negation_relinks_as_in_english():
    """The English reset reads after a Hebrew negation as after ``not``."""

    for negation in ("לא", "not"):
        source = (
            f"(a) {negation} computed, and the amount is computed in accordance "
            "with section 2."
        )
        assert _english_routes_accept("Missing section 2.", source) == (True, True)


def test_a_hebrew_negation_before_abweichend_von_is_nicht():
    source = "(a) Die Zahlung erfolgt {} abweichend von section 2."
    reason = "Missing section 2."
    affirmed = _english_accepted("a", reason, source.format(""))
    assert _english_accepted("a", reason, source.format("nicht")) is False
    assert _english_accepted("a", reason, source.format("לא")) is False
    assert hebrew.ends_with_negation("Die Zahlung erfolgt ל‏א ")
    assert not hebrew.ends_with_negation("Die Zahlung erfolgt nicht ")
    assert affirmed


# A script change is a word boundary for every negation and adversative.

_GLUED_LATIN_NEGATORS = tuple(
    word for word in _OTHER_NEGATORS + _OTHER_LINK_NEGATORS if " " not in word
)


@settings(max_examples=150, deadline=None)
@given(
    st.sampled_from(tuple(hebrew._PREFIX_LETTERS)),
    st.sampled_from(_GLUED_LATIN_NEGATORS),
    st.sampled_from(("he-reason", "he-link", "en-link")),
)
@example("ו", "not", "he-reason")
@example("ש", "nicht", "he-reason")
@example("ו", "not", "he-link")
@example("ש", "prohibited", "he-link")
@example("ו", "not", "en-link")
@example("ש", "ohne", "en-link")
def test_a_hebrew_prefix_glued_to_a_latin_negator_hides_nothing(prefix, negator, frame):
    """Metamorphic: ``וnot`` is read as ``ו־not``, in every frame.

    A Hebrew prefix glued to an English or German negator gets the verdict of
    the same prefix joined by a maqaf, where ``\\b`` already sees the word.
    """

    def verdict(word: str) -> tuple[bool, ...]:
        if frame == "he-reason":
            return (_accepted("a", f"{word} חסר סעיף 2", REVIEW_SOURCE),)
        if frame == "he-link":
            return (
                _accepted("a", "חסר סעיף 2", f"(א) ישולם {word} מחושב בהתאם לסעיף 2."),
            )
        return _english_routes_accept(
            "Missing section 2.",
            f"(a) It is paid {word} computed in accordance with section 2.",
        )

    assert verdict(prefix + negator) == verdict(f"{prefix}־{negator}")


@pytest.mark.parametrize(
    "adversative", ("however", "but", "although", "aber", "jedoch", "obwohl")
)
@pytest.mark.parametrize("prefix", ("ו", "ש"))
def test_a_hebrew_prefix_glued_to_a_latin_adversative_hides_nothing(
    adversative, prefix
):
    hebrew_reason = "חסר סעיף 2. {} הוא זמין."
    english_reason = "Missing section 2. {} it is available."
    for reason, accepts in (
        (hebrew_reason, lambda text: _accepted("a", text, REVIEW_SOURCE)),
        (
            english_reason,
            lambda text: _english_routes_accept(text, _ENGLISH_PAID_SOURCE),
        ),
    ):
        glued = accepts(reason.format(prefix + adversative))
        assert glued == accepts(reason.format(f"{prefix}־{adversative}"))
        assert not any(glued if isinstance(glued, tuple) else (glued,))


@pytest.mark.parametrize("negator", ("not", "never", "nicht", "ohne", "without"))
def test_a_latin_negator_glued_before_a_hebrew_word_is_still_read(negator):
    """``notמחושב`` is ``not מחושב``, in the Hebrew and the English link guard."""

    for words in (f"{negator}מחושב", f"{negator} מחושב"):
        assert not _accepted("a", "חסר סעיף 2", f"(א) ישולם {words} בהתאם לסעיף 2.")
        assert _english_routes_accept(
            "Missing section 2.",
            f"(a) It is paid {words} in accordance with section 2.",
        ) == (False, False)


@settings(max_examples=200, deadline=None)
@given(st.text(alphabet="אבגדlnotXY ־-.,", max_size=24))
def test_scripts_split_maps_every_start_back(text):
    """Every start read in the split copy maps to the same character of the text."""

    view, junctions = hebrew._split_scripts(text)
    assert view.replace(" ", "") == text.replace(" ", "")
    assert len(view) == len(text) + len(junctions)
    starts = hebrew._split_starts(text, (lambda copy: re.finditer(r"\S", copy),))
    assert starts == [index for index, char in enumerate(text) if not char.isspace()]


# Review round 14's reproducers, with their faithful translations. ``route``
# names the corpus path the mixed text is read under: ``il`` is the Israeli
# path (`_accepted`), ``us`` the English one (`_english_accepted`). The
# translation is always read under the English path.
ROUND_14_REPRODUCERS = (
    # Finding 1: ``בלתי`` under any chain of prefix letters.
    (
        "il",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "הטענה הבלתי נכונה שחסר סעיף 2",
        "The untrue claim that section 2 is missing.",
        "neither",
    ),
    (
        "il",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "לבלתי היות חסר סעיף 2",
        "So that section 2 is not missing.",
        "neither",
    ),
    # Finding 2: format characters the old list missed.
    (
        "il",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "ל⁣א חסר סעיף 2",
        "Section 2 is not missing.",
        "neither",
    ),
    (
        "il",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "ל᠎א חסר סעיף 2",
        "Section 2 is not missing.",
        "neither",
    ),
    (
        "il",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "אינ⁡ו חסר סעיף 2",
        "Section 2 is not missing.",
        "neither",
    ),
    (
        "il",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "בל⁢תי נכון שחסר סעיף 2",
        "It is untrue that section 2 is missing.",
        "neither",
    ),
    (
        "il",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "n⁤ot חסר סעיף 2",
        "Section 2 is not missing.",
        "neither",
    ),
    (
        "il",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "kei⁣ne חסר סעיף 2",
        "No section 2 is missing.",
        "neither",
    ),
    # Finding 3: ``without`` and ``ohne`` before a Hebrew computation.
    (
        "il",
        "(א) משולם without חישוב בהתאם לסעיף 2.",
        "(a) It is paid without calculation in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "il",
        "(א) משולם ohne חישוב בהתאם לסעיף 2.",
        "(a) It is paid without calculation in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "il",
        "(א) ohne מחושב בהתאם לסעיף 2.",
        "(a) It is paid without being computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    # Finding 3's reverse gaps, open on main too: a Hebrew negation before an
    # English signal, under either path, and before an English link.
    (
        "us",
        _ENGLISH_PAID_SOURCE,
        _ENGLISH_PAID_SOURCE,
        "לא missing section 2",
        "Section 2 is not missing.",
        "neither",
    ),
    (
        "il",
        "(א) It is paid in accordance with section 2.",
        _ENGLISH_PAID_SOURCE,
        "לא missing section 2",
        "Section 2 is not missing.",
        "neither",
    ),
    (
        "us",
        _ENGLISH_PAID_SOURCE,
        _ENGLISH_PAID_SOURCE,
        "אין זה נכון ש missing section 2.",
        "It is not true that section 2 is missing.",
        "neither",
    ),
    (
        "us",
        "(a) לא computed in accordance with section 2.",
        REVIEW_NEGATED_SOURCE,
        "Missing section 2.",
        "Missing section 2.",
        "neither",
    ),
    (
        "us",
        "(a) דלא computed in accordance with section 2.",
        REVIEW_NEGATED_SOURCE,
        "Missing section 2.",
        "Missing section 2.",
        "neither",
    ),
    (
        "us",
        "(a) אינו computed in accordance with section 2.",
        REVIEW_NEGATED_SOURCE,
        "Missing section 2.",
        "Missing section 2.",
        "neither",
    ),
    (
        "us",
        "(a) אין זה נכון ש computed in accordance with section 2.",
        REVIEW_NEGATED_SOURCE,
        "Missing section 2.",
        "Missing section 2.",
        "neither",
    ),
    # Glued scripts: a Hebrew prefix joined to an English or German word, and
    # an English word joined to a Hebrew one.
    (
        "il",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "וnot חסר סעיף 2",
        "And section 2 is not missing.",
        "neither",
    ),
    (
        "il",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "שnicht חסר סעיף 2",
        "That section 2 is not missing.",
        "neither",
    ),
    (
        "il",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "חסר סעיף 2. וhowever הוא זמין.",
        "Missing section 2. And however it is available.",
        "neither",
    ),
    (
        "il",
        "(א) ישולם וnot מחושב בהתאם לסעיף 2.",
        "(a) It is paid and not computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "il",
        "(א) ישולם שprohibited מחושב בהתאם לסעיף 2.",
        "(a) It is paid, which is prohibited to compute in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "us",
        _ENGLISH_PAID_SOURCE,
        _ENGLISH_PAID_SOURCE,
        "missing section 2. וhowever it is available.",
        "Missing section 2. And however it is available.",
        "neither",
    ),
    (
        "us",
        "(a) It is paid וnot computed in accordance with section 2.",
        "(a) It is paid and not computed in accordance with section 2.",
        "Missing section 2.",
        "Missing section 2.",
        "neither",
    ),
    (
        "us",
        "(a) It is paid notמחושב in accordance with section 2.",
        "(a) It is paid not computed in accordance with section 2.",
        "Missing section 2.",
        "Missing section 2.",
        "neither",
    ),
    # Controls accepted by both: an exclusion of no computation in German, a
    # Hebrew negation in a clause of its own or after the citation, and an
    # English reset after a Hebrew negation.
    (
        "il",
        "(א) משולם ohne Verzögerung בהתאם לסעיף 2.",
        "(a) It is paid without delay in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "both",
    ),
    (
        "us",
        _ENGLISH_PAID_SOURCE,
        _ENGLISH_PAID_SOURCE,
        "לא ברור. Missing section 2.",
        "Unclear. Missing section 2.",
        "both",
    ),
    (
        "us",
        _ENGLISH_PAID_SOURCE,
        _ENGLISH_PAID_SOURCE,
        "Missing section 2, שלא קודד.",
        "Missing section 2, which has not been encoded.",
        "both",
    ),
    (
        "us",
        "(a) לא computed, and the amount is computed in accordance with section 2.",
        "(a) not computed, and the amount is computed in accordance with section 2.",
        "Missing section 2.",
        "Missing section 2.",
        "both",
    ),
    # Intended: a Hebrew negation before an English signal refuses where
    # English, which reads no negation before its own ``missing``, accepts
    # the word-for-word translation; the mirror of round 13's ``not חסר
    # סעיף 2``. And ``without`` before a Hebrew word excludes whatever
    # follows, as ``בלי`` does, though English "without delay" links.
    (
        "us",
        _ENGLISH_PAID_SOURCE,
        _ENGLISH_PAID_SOURCE,
        "לא missing section 2",
        "Not missing section 2.",
        "hebrew-stricter",
    ),
    (
        "il",
        "(א) משולם without דיחוי בהתאם לסעיף 2.",
        "(a) It is paid without delay in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "hebrew-stricter",
    ),
)


@pytest.mark.parametrize(
    (
        "route",
        "mixed_source",
        "english_source",
        "mixed_reason",
        "english_reason",
        "verdict",
    ),
    ROUND_14_REPRODUCERS,
)
def test_round_14_reproducers_get_the_english_verdict(
    route, mixed_source, english_source, mixed_reason, english_reason, verdict
):
    mixed = (
        _english_accepted("a", mixed_reason, mixed_source)
        if route == "us"
        else _accepted("a", mixed_reason, mixed_source)
    )
    assert (mixed, _english_accepted("a", english_reason, english_source)) == (
        _DIFFERENTIAL_VERDICTS[verdict]
    )


def test_an_escaped_astral_mark_in_yaml_hides_no_negation():
    """YAML reads ``"\\ud804\\udcbd"`` as two surrogates; they fold as the mark does."""

    marked = "ל\U000110bdא missing section 2."
    escaped = json.dumps(marked)
    assert "\\ud804\\udcbd" in escaped
    reason = yaml.safe_load(escaped)
    assert [unicodedata.category(char) for char in reason[1:3]] == ["Cs", "Cs"]
    assert hebrew.fold_marks(reason) == "לא missing section 2."
    # ``_english_accepted`` writes the reason with ASCII escapes, as here.
    assert not _english_accepted("a", marked, _ENGLISH_PAID_SOURCE)
    content = _deferral_content("a", "חסר סעיף 2").replace(
        json.dumps("חסר סעיף 2", ensure_ascii=False),
        json.dumps(marked),
    )
    assert "\\ud804" in content
    source = "(א) It is paid in accordance with section 2."
    assert _deferral_issue(_analyze(content, source), 0)
    assert _accepted("a", "Missing section 2.", source)


# --- Review round 15 ------------------------------------------------------------

# Every new review reproducer has its faithful English translation. The two
# אל + ת-noun rows deliberately retain the source reader's conservative
# prohibitive reading in BOTH contexts (round 11's intended stricter class).
ROUND_15_REPRODUCERS = (
    *(
        (
            REVIEW_SOURCE,
            REVIEW_ENGLISH_SOURCE,
            f"ל{chr(code)}א חסר סעיף 2",
            "Section 2 is not missing.",
            "neither",
        )
        for code in (0x115F, 0x1160, 0x17B4, 0x17B5, 0x3164, 0xFFA0)
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "אל תטען שחסר סעיף 2",
        "Do not claim that section 2 is missing.",
        "neither",
    ),
    (
        "(א) טרם חושב בהתאם לסעיף 2.",
        "(a) It has not yet been calculated in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א) התשלום טרם חושב בהתאם לסעיף 2.",
        "(a) The payment has not yet been calculated in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א) תשלום מלא בהתאם לסעיף 2.",
        "(a) Full payment in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "both",
    ),
    (
        "(א) ישולם מס מלא בהתאם לסעיף 2.",
        "(a) Full tax shall be paid in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "both",
    ),
    *(
        (
            f"(א) כל התשלומים, חוץ מ{joint}{noun} בהתאם לסעיף 2.",
            f"(a) All payments, other than {english} in accordance with section 2.",
            "חסר סעיף 2",
            "Missing section 2.",
            "neither",
        )
        for joint in ("־", "-", "‐", "‑")
        for noun, english in (("תשלום", "a payment"), ("חישוב", "a calculation"))
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "אל תושב שטוען שחסר סעיף 2",
        "To a resident who claims that section 2 is missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "אל תושב: חסר סעיף 2",
        "To a resident: missing section 2.",
        "hebrew-stricter",
    ),
    (
        "(א) ישולם אל תושב בהתאם לסעיף 2.",
        "(a) It is paid to a resident in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "hebrew-stricter",
    ),
)


@pytest.mark.parametrize(
    ("hebrew_source", "english_source", "hebrew_reason", "english_reason", "verdict"),
    ROUND_15_REPRODUCERS,
)
def test_round_15_reproducers_get_the_english_verdict(
    hebrew_source, english_source, hebrew_reason, english_reason, verdict
):
    assert (
        _accepted("a", hebrew_reason, hebrew_source),
        _english_accepted("a", english_reason, english_source),
    ) == _DIFFERENTIAL_VERDICTS[verdict]


# Each inflected lexical form is pinned in both positions. The four without
# signals are allowed in reasons: their own negator is inside the signal.
# In sources they exclude the operative link. This intentional contextual
# asymmetry concerns signal ownership, never the vocabulary being read.
ROUND_15_NEGATOR_TRANSLATIONS = (
    (
        "לא",
        "לא נכון שחסר סעיף 2",
        "It is not true that section 2 is missing.",
        "(א) לא מחושב בהתאם לסעיף 2.",
        "(a) It is not computed in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אין",
        "אין בסיס לטענה שחסר סעיף 2",
        "There is no basis for claiming that section 2 is missing.",
        "(א) אין לחשב בהתאם לסעיף 2.",
        "(a) One must not calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אינו",
        "אינו טוען שחסר סעיף 2",
        "He is not claiming that section 2 is missing.",
        "(א) אינו מחשב בהתאם לסעיף 2.",
        "(a) He is not calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אינה",
        "אינה טוענת שחסר סעיף 2",
        "She is not claiming that section 2 is missing.",
        "(א) אינה מחשבת בהתאם לסעיף 2.",
        "(a) She is not calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אינם",
        "אינם טוענים שחסר סעיף 2",
        "They are not claiming that section 2 is missing.",
        "(א) אינם מחשבים בהתאם לסעיף 2.",
        "(a) They are not calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אינן",
        "אינן טוענות שחסר סעיף 2",
        "They are not claiming that section 2 is missing.",
        "(א) אינן מחשבות בהתאם לסעיף 2.",
        "(a) They are not calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "איני",
        "איני טוען שחסר סעיף 2",
        "I am not claiming that section 2 is missing.",
        "(א) איני מחשב בהתאם לסעיף 2.",
        "(a) I am not calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אינך",
        "אינך טוען שחסר סעיף 2",
        "You are not claiming that section 2 is missing.",
        "(א) אינך מחשב בהתאם לסעיף 2.",
        "(a) You are not calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אינכם",
        "אינכם טוענים שחסר סעיף 2",
        "You are not claiming that section 2 is missing.",
        "(א) אינכם מחשבים בהתאם לסעיף 2.",
        "(a) You are not calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אינכן",
        "אינכן טוענות שחסר סעיף 2",
        "You are not claiming that section 2 is missing.",
        "(א) אינכן מחשבות בהתאם לסעיף 2.",
        "(a) You are not calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "איננו",
        "איננו טוענים שחסר סעיף 2",
        "We are not claiming that section 2 is missing.",
        "(א) איננו מחשבים בהתאם לסעיף 2.",
        "(a) We are not calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "איננה",
        "איננה טוענת שחסר סעיף 2",
        "She is not claiming that section 2 is missing.",
        "(א) איננה מחשבת בהתאם לסעיף 2.",
        "(a) She is not calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אינני",
        "אינני טוען שחסר סעיף 2",
        "I am not claiming that section 2 is missing.",
        "(א) אינני מחשב בהתאם לסעיף 2.",
        "(a) I am not calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "טרם",
        "טרם הוכח שחסר סעיף 2",
        "It has not yet been proven that section 2 is missing.",
        "(א) טרם חושב בהתאם לסעיף 2.",
        "(a) It has not yet been computed in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מעולם",
        "מעולם לא נטען שחסר סעיף 2",
        "It has never been claimed that section 2 is missing.",
        "(א) מעולם לא חושב בהתאם לסעיף 2.",
        "(a) It has never been computed in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "לאו",
        "לאו דווקא חסר סעיף 2",
        "Section 2 is not necessarily missing.",
        "(א) לאו דווקא מחושב בהתאם לסעיף 2.",
        "(a) It is not necessarily computed in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "בלתי",
        "בלתי נכון שחסר סעיף 2",
        "It is untrue that section 2 is missing.",
        "(א) בלתי מחושב בהתאם לסעיף 2.",
        "(a) It is not computed in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אל",
        "אל תטען שחסר סעיף 2",
        "Do not claim that section 2 is missing.",
        "(א) אל תחשב בהתאם לסעיף 2.",
        "(a) Do not compute in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אסור",
        "אסור לטעון שחסר סעיף 2",
        "It is prohibited to claim that section 2 is missing.",
        "(א) אסור לחשב בהתאם לסעיף 2.",
        "(a) It is prohibited to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אסורה",
        "אסורה לטעון שחסר סעיף 2",
        "It is prohibited to claim that section 2 is missing.",
        "(א) אסורה לחשב בהתאם לסעיף 2.",
        "(a) It is prohibited to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אסורים",
        "אסורים לטעון שחסר סעיף 2",
        "It is prohibited to claim that section 2 is missing.",
        "(א) אסורים לחשב בהתאם לסעיף 2.",
        "(a) It is prohibited to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אסורות",
        "אסורות לטעון שחסר סעיף 2",
        "It is prohibited to claim that section 2 is missing.",
        "(א) אסורות לחשב בהתאם לסעיף 2.",
        "(a) It is prohibited to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נאסר",
        "נאסר לטעון שחסר סעיף 2",
        "It is prohibited to claim that section 2 is missing.",
        "(א) נאסר לחשב בהתאם לסעיף 2.",
        "(a) It is prohibited to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נאסרה",
        "נאסרה לטעון שחסר סעיף 2",
        "It is prohibited to claim that section 2 is missing.",
        "(א) נאסרה לחשב בהתאם לסעיף 2.",
        "(a) It is prohibited to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נאסרו",
        "נאסרו לטעון שחסר סעיף 2",
        "It is prohibited to claim that section 2 is missing.",
        "(א) נאסרו לחשב בהתאם לסעיף 2.",
        "(a) It is prohibited to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נאסרים",
        "נאסרים לטעון שחסר סעיף 2",
        "It is prohibited to claim that section 2 is missing.",
        "(א) נאסרים לחשב בהתאם לסעיף 2.",
        "(a) It is prohibited to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נאסרות",
        "נאסרות לטעון שחסר סעיף 2",
        "It is prohibited to claim that section 2 is missing.",
        "(א) נאסרות לחשב בהתאם לסעיף 2.",
        "(a) It is prohibited to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "בל",
        "בל לטעון שחסר סעיף 2",
        "It is prohibited to claim that section 2 is missing.",
        "(א) בל לחשב בהתאם לסעיף 2.",
        "(a) It is prohibited to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "לבל",
        "לבל לטעון שחסר סעיף 2",
        "It is prohibited to claim that section 2 is missing.",
        "(א) לבל לחשב בהתאם לסעיף 2.",
        "(a) It is prohibited to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "איסור",
        "איסור לטעון שחסר סעיף 2",
        "It is prohibited to claim that section 2 is missing.",
        "(א) איסור לחשב בהתאם לסעיף 2.",
        "(a) It is prohibited to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מסרב",
        "מסרב לטעון שחסר סעיף 2",
        "The person refuses to claim that section 2 is missing.",
        "(א) מסרב לחשב בהתאם לסעיף 2.",
        "(a) The person refuses to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מסרבת",
        "מסרבת לטעון שחסר סעיף 2",
        "The person refuses to claim that section 2 is missing.",
        "(א) מסרבת לחשב בהתאם לסעיף 2.",
        "(a) The person refuses to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מסרבים",
        "מסרבים לטעון שחסר סעיף 2",
        "The person refuses to claim that section 2 is missing.",
        "(א) מסרבים לחשב בהתאם לסעיף 2.",
        "(a) The person refuses to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מסרבות",
        "מסרבות לטעון שחסר סעיף 2",
        "The person refuses to claim that section 2 is missing.",
        "(א) מסרבות לחשב בהתאם לסעיף 2.",
        "(a) The person refuses to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "יסרב",
        "יסרב לטעון שחסר סעיף 2",
        "The person will refuse to claim that section 2 is missing.",
        "(א) יסרב לחשב בהתאם לסעיף 2.",
        "(a) The person will refuse to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "יסרבו",
        "יסרבו לטעון שחסר סעיף 2",
        "The person will refuse to claim that section 2 is missing.",
        "(א) יסרבו לחשב בהתאם לסעיף 2.",
        "(a) The person will refuse to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תסרב",
        "תסרב לטעון שחסר סעיף 2",
        "The person will refuse to claim that section 2 is missing.",
        "(א) תסרב לחשב בהתאם לסעיף 2.",
        "(a) The person will refuse to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תסרבנה",
        "תסרבנה לטעון שחסר סעיף 2",
        "The person will refuse to claim that section 2 is missing.",
        "(א) תסרבנה לחשב בהתאם לסעיף 2.",
        "(a) The person will refuse to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תסרבו",
        "תסרבו לטעון שחסר סעיף 2",
        "The person will refuse to claim that section 2 is missing.",
        "(א) תסרבו לחשב בהתאם לסעיף 2.",
        "(a) The person will refuse to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "סרב",
        "סרב לטעון שחסר סעיף 2",
        "The person refused to claim that section 2 is missing.",
        "(א) סרב לחשב בהתאם לסעיף 2.",
        "(a) The person did not agree to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "סרבה",
        "סרבה לטעון שחסר סעיף 2",
        "The person refused to claim that section 2 is missing.",
        "(א) סרבה לחשב בהתאם לסעיף 2.",
        "(a) The person did not agree to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "סרבו",
        "סרבו לטעון שחסר סעיף 2",
        "The person refused to claim that section 2 is missing.",
        "(א) סרבו לחשב בהתאם לסעיף 2.",
        "(a) The person did not agree to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "סירב",
        "סירב לטעון שחסר סעיף 2",
        "The person refused to claim that section 2 is missing.",
        "(א) סירב לחשב בהתאם לסעיף 2.",
        "(a) The person did not agree to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "סירבה",
        "סירבה לטעון שחסר סעיף 2",
        "The person refused to claim that section 2 is missing.",
        "(א) סירבה לחשב בהתאם לסעיף 2.",
        "(a) The person did not agree to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "סירבו",
        "סירבו לטעון שחסר סעיף 2",
        "The person refused to claim that section 2 is missing.",
        "(א) סירבו לחשב בהתאם לסעיף 2.",
        "(a) The person did not agree to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "לסרב",
        "לסרב לטעון שחסר סעיף 2",
        "The person refuse to claim that section 2 is missing.",
        "(א) לסרב לחשב בהתאם לסעיף 2.",
        "(a) The person refuse to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נבצר",
        "נבצר לטעון שחסר סעיף 2",
        "The person is unable to claim that section 2 is missing.",
        "(א) נבצר לחשב בהתאם לסעיף 2.",
        "(a) The person is unable to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נבצרה",
        "נבצרה לטעון שחסר סעיף 2",
        "The person is unable to claim that section 2 is missing.",
        "(א) נבצרה לחשב בהתאם לסעיף 2.",
        "(a) The person is unable to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מנוע",
        "מנוע מלטעון שחסר סעיף 2",
        "The person is prevented from claiming that section 2 is missing.",
        "(א) מנוע מלחשב בהתאם לסעיף 2.",
        "(a) The person is prevented from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מנועה",
        "מנועה מלטעון שחסר סעיף 2",
        "The person is prevented from claiming that section 2 is missing.",
        "(א) מנועה מלחשב בהתאם לסעיף 2.",
        "(a) The person is prevented from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מנועים",
        "מנועים מלטעון שחסר סעיף 2",
        "The person is prevented from claiming that section 2 is missing.",
        "(א) מנועים מלחשב בהתאם לסעיף 2.",
        "(a) The person is prevented from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מנועות",
        "מנועות מלטעון שחסר סעיף 2",
        "The person is prevented from claiming that section 2 is missing.",
        "(א) מנועות מלחשב בהתאם לסעיף 2.",
        "(a) The person is prevented from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "כושל",
        "כושל להוכיח שחסר סעיף 2",
        "The person fails to prove that section 2 is missing.",
        "(א) כושל לחשב בהתאם לסעיף 2.",
        "(a) The person fails to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "כושלת",
        "כושלת להוכיח שחסר סעיף 2",
        "The person fails to prove that section 2 is missing.",
        "(א) כושלת לחשב בהתאם לסעיף 2.",
        "(a) The person fails to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "כושלים",
        "כושלים להוכיח שחסר סעיף 2",
        "The person fails to prove that section 2 is missing.",
        "(א) כושלים לחשב בהתאם לסעיף 2.",
        "(a) The person fails to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "כושלות",
        "כושלות להוכיח שחסר סעיף 2",
        "The person fails to prove that section 2 is missing.",
        "(א) כושלות לחשב בהתאם לסעיף 2.",
        "(a) The person fails to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נכשל",
        "נכשל להוכיח שחסר סעיף 2",
        "The person failed to prove that section 2 is missing.",
        "(א) נכשל לחשב בהתאם לסעיף 2.",
        "(a) The person was not successful in calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נכשלה",
        "נכשלה להוכיח שחסר סעיף 2",
        "The person failed to prove that section 2 is missing.",
        "(א) נכשלה לחשב בהתאם לסעיף 2.",
        "(a) The person was not successful in calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נכשלו",
        "נכשלו להוכיח שחסר סעיף 2",
        "The person failed to prove that section 2 is missing.",
        "(א) נכשלו לחשב בהתאם לסעיף 2.",
        "(a) The person was not successful in calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נכשלים",
        "נכשלים להוכיח שחסר סעיף 2",
        "The person failed to prove that section 2 is missing.",
        "(א) נכשלים לחשב בהתאם לסעיף 2.",
        "(a) The person was not successful in calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נכשלות",
        "נכשלות להוכיח שחסר סעיף 2",
        "The person failed to prove that section 2 is missing.",
        "(א) נכשלות לחשב בהתאם לסעיף 2.",
        "(a) The person was not successful in calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "כשל",
        "כשל להוכיח שחסר סעיף 2",
        "The person failed to prove that section 2 is missing.",
        "(א) כשל לחשב בהתאם לסעיף 2.",
        "(a) The person was not successful in calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "כשלה",
        "כשלה להוכיח שחסר סעיף 2",
        "The person failed to prove that section 2 is missing.",
        "(א) כשלה לחשב בהתאם לסעיף 2.",
        "(a) The person was not successful in calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "כשלו",
        "כשלו להוכיח שחסר סעיף 2",
        "The person failed to prove that section 2 is missing.",
        "(א) כשלו לחשב בהתאם לסעיף 2.",
        "(a) The person was not successful in calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "יכשל",
        "יכשל להוכיח שחסר סעיף 2",
        "The person will fail to prove that section 2 is missing.",
        "(א) יכשל לחשב בהתאם לסעיף 2.",
        "(a) The person will fail to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "יכשלו",
        "יכשלו להוכיח שחסר סעיף 2",
        "The person will fail to prove that section 2 is missing.",
        "(א) יכשלו לחשב בהתאם לסעיף 2.",
        "(a) The person will fail to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "ייכשל",
        "ייכשל להוכיח שחסר סעיף 2",
        "The person will fail to prove that section 2 is missing.",
        "(א) ייכשל לחשב בהתאם לסעיף 2.",
        "(a) The person will fail to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "ייכשלו",
        "ייכשלו להוכיח שחסר סעיף 2",
        "The person will fail to prove that section 2 is missing.",
        "(א) ייכשלו לחשב בהתאם לסעיף 2.",
        "(a) The person will fail to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תכשל",
        "תכשל להוכיח שחסר סעיף 2",
        "The person will fail to prove that section 2 is missing.",
        "(א) תכשל לחשב בהתאם לסעיף 2.",
        "(a) The person will fail to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תכשלנה",
        "תכשלנה להוכיח שחסר סעיף 2",
        "The person will fail to prove that section 2 is missing.",
        "(א) תכשלנה לחשב בהתאם לסעיף 2.",
        "(a) The person will fail to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תכשלו",
        "תכשלו להוכיח שחסר סעיף 2",
        "The person will fail to prove that section 2 is missing.",
        "(א) תכשלו לחשב בהתאם לסעיף 2.",
        "(a) The person will fail to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תיכשל",
        "תיכשל להוכיח שחסר סעיף 2",
        "The person will fail to prove that section 2 is missing.",
        "(א) תיכשל לחשב בהתאם לסעיף 2.",
        "(a) The person will fail to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תיכשלנה",
        "תיכשלנה להוכיח שחסר סעיף 2",
        "The person will fail to prove that section 2 is missing.",
        "(א) תיכשלנה לחשב בהתאם לסעיף 2.",
        "(a) The person will fail to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תיכשלו",
        "תיכשלו להוכיח שחסר סעיף 2",
        "The person will fail to prove that section 2 is missing.",
        "(א) תיכשלו לחשב בהתאם לסעיף 2.",
        "(a) The person will fail to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "הוחרג",
        "הוחרג מלטעון שחסר סעיף 2",
        "The person was excluded from claiming that section 2 is missing.",
        "(א) הוחרג מלחשב בהתאם לסעיף 2.",
        "(a) The person was excluded from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "הוחרגה",
        "הוחרגה מלטעון שחסר סעיף 2",
        "The person was excluded from claiming that section 2 is missing.",
        "(א) הוחרגה מלחשב בהתאם לסעיף 2.",
        "(a) The person was excluded from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "הוחרגו",
        "הוחרגו מלטעון שחסר סעיף 2",
        "The person was excluded from claiming that section 2 is missing.",
        "(א) הוחרגו מלחשב בהתאם לסעיף 2.",
        "(a) The person was excluded from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מוחרג",
        "מוחרג מלטעון שחסר סעיף 2",
        "The person is excluded from claiming that section 2 is missing.",
        "(א) מוחרג מלחשב בהתאם לסעיף 2.",
        "(a) The person is excluded from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מוחרגת",
        "מוחרגת מלטעון שחסר סעיף 2",
        "The person is excluded from claiming that section 2 is missing.",
        "(א) מוחרגת מלחשב בהתאם לסעיף 2.",
        "(a) The person is excluded from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מוחרגים",
        "מוחרגים מלטעון שחסר סעיף 2",
        "The person is excluded from claiming that section 2 is missing.",
        "(א) מוחרגים מלחשב בהתאם לסעיף 2.",
        "(a) The person is excluded from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מוחרגות",
        "מוחרגות מלטעון שחסר סעיף 2",
        "The person is excluded from claiming that section 2 is missing.",
        "(א) מוחרגות מלחשב בהתאם לסעיף 2.",
        "(a) The person is excluded from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "יוחרג",
        "יוחרג מלטעון שחסר סעיף 2",
        "The person will be excluded from claiming that section 2 is missing.",
        "(א) יוחרג מלחשב בהתאם לסעיף 2.",
        "(a) The person will be excluded from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "יוחרגו",
        "יוחרגו מלטעון שחסר סעיף 2",
        "The person will be excluded from claiming that section 2 is missing.",
        "(א) יוחרגו מלחשב בהתאם לסעיף 2.",
        "(a) The person will be excluded from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תוחרג",
        "תוחרג מלטעון שחסר סעיף 2",
        "The person will be excluded from claiming that section 2 is missing.",
        "(א) תוחרג מלחשב בהתאם לסעיף 2.",
        "(a) The person will be excluded from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תוחרגנה",
        "תוחרגנה מלטעון שחסר סעיף 2",
        "The person will be excluded from claiming that section 2 is missing.",
        "(א) תוחרגנה מלחשב בהתאם לסעיף 2.",
        "(a) The person will be excluded from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נמנע",
        "נמנע מלטעון שחסר סעיף 2",
        "The person is prevented from claiming that section 2 is missing.",
        "(א) נמנע מלחשב בהתאם לסעיף 2.",
        "(a) The person is prevented from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נמנעה",
        "נמנעה מלטעון שחסר סעיף 2",
        "The person is prevented from claiming that section 2 is missing.",
        "(א) נמנעה מלחשב בהתאם לסעיף 2.",
        "(a) The person is prevented from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נמנעו",
        "נמנעו מלטעון שחסר סעיף 2",
        "The person is prevented from claiming that section 2 is missing.",
        "(א) נמנעו מלחשב בהתאם לסעיף 2.",
        "(a) The person is prevented from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נמנעים",
        "נמנעים מלטעון שחסר סעיף 2",
        "The person is prevented from claiming that section 2 is missing.",
        "(א) נמנעים מלחשב בהתאם לסעיף 2.",
        "(a) The person is prevented from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נמנעות",
        "נמנעות מלטעון שחסר סעיף 2",
        "The person is prevented from claiming that section 2 is missing.",
        "(א) נמנעות מלחשב בהתאם לסעיף 2.",
        "(a) The person is prevented from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "יימנע",
        "יימנע מלטעון שחסר סעיף 2",
        "The person will be prevented from claiming that section 2 is missing.",
        "(א) יימנע מלחשב בהתאם לסעיף 2.",
        "(a) The person will be prevented from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "יימנעו",
        "יימנעו מלטעון שחסר סעיף 2",
        "The person will be prevented from claiming that section 2 is missing.",
        "(א) יימנעו מלחשב בהתאם לסעיף 2.",
        "(a) The person will be prevented from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תימנע",
        "תימנע מלטעון שחסר סעיף 2",
        "The person will be prevented from claiming that section 2 is missing.",
        "(א) תימנע מלחשב בהתאם לסעיף 2.",
        "(a) The person will be prevented from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תימנענה",
        "תימנענה מלטעון שחסר סעיף 2",
        "The person will be prevented from claiming that section 2 is missing.",
        "(א) תימנענה מלחשב בהתאם לסעיף 2.",
        "(a) The person will be prevented from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "דוחה",
        "דוחה את הטענה שחסר סעיף 2",
        "The person rejects the claim that section 2 is missing.",
        "(א) דוחה את החישוב בהתאם לסעיף 2.",
        "(a) The person rejects calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "דוחים",
        "דוחים את הטענה שחסר סעיף 2",
        "The person rejects the claim that section 2 is missing.",
        "(א) דוחים את החישוב בהתאם לסעיף 2.",
        "(a) The person rejects calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "דוחות",
        "דוחות את הטענה שחסר סעיף 2",
        "They reject the claim that section 2 is missing.",
        "(א) דוחות את החישוב בהתאם לסעיף 2.",
        "(a) They reject calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "שולל",
        "שולל את הטענה שחסר סעיף 2",
        "The person denies the claim that section 2 is missing.",
        "(א) שולל את החישוב בהתאם לסעיף 2.",
        "(a) The person denies calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "שוללת",
        "שוללת את הטענה שחסר סעיף 2",
        "The person denies the claim that section 2 is missing.",
        "(א) שוללת את החישוב בהתאם לסעיף 2.",
        "(a) The person denies calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "שוללים",
        "שוללים את הטענה שחסר סעיף 2",
        "The person denies the claim that section 2 is missing.",
        "(א) שוללים את החישוב בהתאם לסעיף 2.",
        "(a) The person denies calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "שוללות",
        "שוללות את הטענה שחסר סעיף 2",
        "The person denies the claim that section 2 is missing.",
        "(א) שוללות את החישוב בהתאם לסעיף 2.",
        "(a) The person denies calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נשלל",
        "נשלל מלטעון שחסר סעיף 2",
        "The person was denied from claiming that section 2 is missing.",
        "(א) נשלל מלחשב בהתאם לסעיף 2.",
        "(a) The person was denied from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נשללה",
        "נשללה מלטעון שחסר סעיף 2",
        "The person was denied from claiming that section 2 is missing.",
        "(א) נשללה מלחשב בהתאם לסעיף 2.",
        "(a) The person was denied from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נשללו",
        "נשללו מלטעון שחסר סעיף 2",
        "The person was denied from claiming that section 2 is missing.",
        "(א) נשללו מלחשב בהתאם לסעיף 2.",
        "(a) The person was denied from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נשללים",
        "נשללים מלטעון שחסר סעיף 2",
        "The person was denied from claiming that section 2 is missing.",
        "(א) נשללים מלחשב בהתאם לסעיף 2.",
        "(a) The person was denied from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "נשללות",
        "נשללות מלטעון שחסר סעיף 2",
        "The person was denied from claiming that section 2 is missing.",
        "(א) נשללות מלחשב בהתאם לסעיף 2.",
        "(a) The person was denied from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אוסר",
        "אוסר לטעון שחסר סעיף 2",
        "The person prohibits claiming that section 2 is missing.",
        "(א) אוסר לחשב בהתאם לסעיף 2.",
        "(a) The person prohibits calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אוסרת",
        "אוסרת לטעון שחסר סעיף 2",
        "The person prohibits claiming that section 2 is missing.",
        "(א) אוסרת לחשב בהתאם לסעיף 2.",
        "(a) The person prohibits calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אוסרים",
        "אוסרים לטעון שחסר סעיף 2",
        "The person prohibits claiming that section 2 is missing.",
        "(א) אוסרים לחשב בהתאם לסעיף 2.",
        "(a) The person prohibits calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אוסרות",
        "אוסרות לטעון שחסר סעיף 2",
        "The person prohibits claiming that section 2 is missing.",
        "(א) אוסרות לחשב בהתאם לסעיף 2.",
        "(a) The person prohibits calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אסר",
        "אסר לטעון שחסר סעיף 2",
        "The person prohibited claiming that section 2 is missing.",
        "(א) אסר לחשב בהתאם לסעיף 2.",
        "(a) The person prohibited calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אסרה",
        "אסרה לטעון שחסר סעיף 2",
        "The person prohibited claiming that section 2 is missing.",
        "(א) אסרה לחשב בהתאם לסעיף 2.",
        "(a) The person prohibited calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אסרו",
        "אסרו לטעון שחסר סעיף 2",
        "The person prohibited claiming that section 2 is missing.",
        "(א) אסרו לחשב בהתאם לסעיף 2.",
        "(a) The person prohibited calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "יאסור",
        "יאסור לטעון שחסר סעיף 2",
        "The person will prohibit claiming that section 2 is missing.",
        "(א) יאסור לחשב בהתאם לסעיף 2.",
        "(a) The person will prohibit calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תאסור",
        "תאסור לטעון שחסר סעיף 2",
        "The person will prohibit claiming that section 2 is missing.",
        "(א) תאסור לחשב בהתאם לסעיף 2.",
        "(a) The person will prohibit calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "יאסרו",
        "יאסרו לטעון שחסר סעיף 2",
        "The person will prohibit claiming that section 2 is missing.",
        "(א) יאסרו לחשב בהתאם לסעיף 2.",
        "(a) The person will prohibit calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תאסרנה",
        "תאסרנה לטעון שחסר סעיף 2",
        "The person will prohibit claiming that section 2 is missing.",
        "(א) תאסרנה לחשב בהתאם לסעיף 2.",
        "(a) The person will prohibit calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תאסורנה",
        "תאסורנה לטעון שחסר סעיף 2",
        "The person will prohibit claiming that section 2 is missing.",
        "(א) תאסורנה לחשב בהתאם לסעיף 2.",
        "(a) The person will prohibit calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "ייאסר",
        "ייאסר מלטעון שחסר סעיף 2",
        "The person will be prohibited from claiming that section 2 is missing.",
        "(א) ייאסר מלחשב בהתאם לסעיף 2.",
        "(a) The person will be prohibited from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "ייאסרו",
        "ייאסרו מלטעון שחסר סעיף 2",
        "The person will be prohibited from claiming that section 2 is missing.",
        "(א) ייאסרו מלחשב בהתאם לסעיף 2.",
        "(a) The person will be prohibited from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תיאסר",
        "תיאסר מלטעון שחסר סעיף 2",
        "The person will be prohibited from claiming that section 2 is missing.",
        "(א) תיאסר מלחשב בהתאם לסעיף 2.",
        "(a) The person will be prohibited from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "תיאסרנה",
        "תיאסרנה מלטעון שחסר סעיף 2",
        "The person will be prohibited from claiming that section 2 is missing.",
        "(א) תיאסרנה מלחשב בהתאם לסעיף 2.",
        "(a) The person will be prohibited from calculating in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מכחיש",
        "מכחיש את הטענה שחסר סעיף 2",
        "The person denies the claim that section 2 is missing.",
        "(א) מכחיש את החישוב בהתאם לסעיף 2.",
        "(a) The person denies calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מכחישה",
        "מכחישה את הטענה שחסר סעיף 2",
        "The person denies the claim that section 2 is missing.",
        "(א) מכחישה את החישוב בהתאם לסעיף 2.",
        "(a) The person denies calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מכחישים",
        "מכחישים את הטענה שחסר סעיף 2",
        "The person denies the claim that section 2 is missing.",
        "(א) מכחישים את החישוב בהתאם לסעיף 2.",
        "(a) The person denies calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מכחישות",
        "מכחישות את הטענה שחסר סעיף 2",
        "The person denies the claim that section 2 is missing.",
        "(א) מכחישות את החישוב בהתאם לסעיף 2.",
        "(a) The person denies calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "הכחיש",
        "הכחיש את הטענה שחסר סעיף 2",
        "The person denied the claim that section 2 is missing.",
        "(א) הכחיש את החישוב בהתאם לסעיף 2.",
        "(a) The person denied calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "הכחישה",
        "הכחישה את הטענה שחסר סעיף 2",
        "The person denied the claim that section 2 is missing.",
        "(א) הכחישה את החישוב בהתאם לסעיף 2.",
        "(a) The person denied calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "הכחישו",
        "הכחישו את הטענה שחסר סעיף 2",
        "The person denied the claim that section 2 is missing.",
        "(א) הכחישו את החישוב בהתאם לסעיף 2.",
        "(a) The person denied calculation in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "בלי",
        "בלי לטעון שחסר סעיף 2",
        "Without claiming that section 2 is missing.",
        "(א) בלי חישוב בהתאם לסעיף 2.",
        "(a) Without calculation in accordance with section 2.",
        "both",
        "neither",
    ),
    (
        "מבלי",
        "מבלי לטעון שחסר סעיף 2",
        "Without claiming that section 2 is missing.",
        "(א) מבלי חישוב בהתאם לסעיף 2.",
        "(a) Without calculation in accordance with section 2.",
        "both",
        "neither",
    ),
    (
        "ללא",
        "ללא לטעון שחסר סעיף 2",
        "Without claiming that section 2 is missing.",
        "(א) ללא חישוב בהתאם לסעיף 2.",
        "(a) Without calculation in accordance with section 2.",
        "both",
        "neither",
    ),
    (
        "בלא",
        "בלא לטעון שחסר סעיף 2",
        "Without claiming that section 2 is missing.",
        "(א) בלא חישוב בהתאם לסעיף 2.",
        "(a) Without calculation in accordance with section 2.",
        "both",
        "neither",
    ),
    (
        "למעט",
        "למעט הטענה שחסר סעיף 2",
        "Except for the claim that section 2 is missing.",
        "(א) למעט תשלום בהתאם לסעיף 2.",
        "(a) Except for payment in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "מלבד",
        "מלבד הטענה שחסר סעיף 2",
        "Except for the claim that section 2 is missing.",
        "(א) מלבד תשלום בהתאם לסעיף 2.",
        "(a) Except for payment in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "להוציא",
        "להוציא הטענה שחסר סעיף 2",
        "Except for the claim that section 2 is missing.",
        "(א) להוציא תשלום בהתאם לסעיף 2.",
        "(a) Except for payment in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "זולת",
        "זולת הטענה שחסר סעיף 2",
        "Except for the claim that section 2 is missing.",
        "(א) זולת תשלום בהתאם לסעיף 2.",
        "(a) Except for payment in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אף",
        "אף טענה שחסר סעיף 2 לא הוכחה",
        "No claim that section 2 is missing has been proven.",
        "(א) אף חישוב לא נערך בהתאם לסעיף 2.",
        "(a) Calculation is not made in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אי אפשר",
        "אי אפשר לטעון שחסר סעיף 2",
        "It is not possible to claim that section 2 is missing.",
        "(א) אי אפשר לחשב בהתאם לסעיף 2.",
        "(a) It is not possible to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
    (
        "אי ניתן",
        "אי ניתן לטעון שחסר סעיף 2",
        "It is not possible to claim that section 2 is missing.",
        "(א) אי ניתן לחשב בהתאם לסעיף 2.",
        "(a) It is not possible to calculate in accordance with section 2.",
        "neither",
        "neither",
    ),
)


@pytest.mark.parametrize(
    (
        "form",
        "hebrew_reason",
        "english_reason",
        "hebrew_source",
        "english_source",
        "reason_verdict",
        "source_verdict",
    ),
    ROUND_15_NEGATOR_TRANSLATIONS,
)
def test_every_shared_negator_has_reason_and_source_translation_pairs(
    form,
    hebrew_reason,
    english_reason,
    hebrew_source,
    english_source,
    reason_verdict,
    source_verdict,
):
    assert (
        _accepted("a", hebrew_reason, REVIEW_SOURCE),
        _english_accepted("a", english_reason, REVIEW_ENGLISH_SOURCE),
    ) == _DIFFERENTIAL_VERDICTS[reason_verdict], form
    assert (
        _accepted("a", "חסר סעיף 2", hebrew_source),
        _english_accepted("a", "Missing section 2.", english_source),
    ) == _DIFFERENTIAL_VERDICTS[source_verdict], form


# Lexical exclusions from the original reason grammar apply to every reader.
_ROUND_15_LEXICAL_EXCLUSIONS = (
    "מלא",
    "מלאה",
    "מלאים",
    "מלאכה",
    "כלא",
    "הלא",
    "הלאה",
    "בטרם",
    "מאין",
    "כאין",
    "לאין",
    "האין",
)


def test_reason_and_source_readers_agree_for_every_negator_prefix_and_exclusion():
    """All 585 chains of length 0–3 over all eight prefix letters, per form.

    Compare the actual reason and public source readers, including lookahead
    contexts for אל, מנוע, דוחות and אי אפשר. Prefixes do not get invented
    translations; the unprefixed forms above have faithful translations.
    """

    chains = tuple(
        "".join(chain)
        for length in range(4)
        for chain in itertools.product("והבכלמשד", repeat=length)
    )
    assert len(chains) == 585
    assert len({row[0] for row in ROUND_15_NEGATOR_TRANSLATIONS}) == len(
        ROUND_15_NEGATOR_TRANSLATIONS
    )
    forms = tuple(row[1] for row in ROUND_15_NEGATOR_TRANSLATIONS)
    for chain, phrase in itertools.product(
        chains, (*forms, *_ROUND_15_LEXICAL_EXCLUSIONS)
    ):
        text = chain + phrase
        reason = [match.span() for match in hebrew._REASON_NEGATION.finditer(text)]
        source = [match.span() for match in hebrew.link_negations(text)]
        assert reason == source, text
    for word in _ROUND_15_LEXICAL_EXCLUSIONS:
        assert not hebrew._REASON_NEGATION.search(word), word
        assert not list(hebrew.link_negations(word)), word


@pytest.mark.parametrize("gap", (" ", "   ", "\t", "־", "-", "‐", "‑"))
def test_affirmative_af_connectives_are_not_negative_polarity(gap):
    """Shared אף policy: despite/although stay affirmative in both readers."""

    for text in (f"על{gap}אף האמור", f"אף{gap}כי ההוראה חלה"):
        assert not list(hebrew._REASON_NEGATION.finditer(text)), text
        assert not list(hebrew.link_negations(text)), text
    text = "אף טענה שחסר סעיף 2 לא הוכחה"
    assert next(hebrew._REASON_NEGATION.finditer(text)).group() == "אף"
    assert next(iter(hebrew.link_negations(text))).group() == "אף"


# The source vocabulary includes exclusions; sharing it intentionally gives
# these leading fragments a conservative exclusion reading in reasons too.
# English permits them before the missing claim. חוץ מזה / פרט לזה can also
# mean "besides/apart from that"; the fragment gives no complete clause that
# disambiguates the exclusion. This stricter class is deliberate and pinned
# for both the Hebrew and mixed-English routes, rather than left implicit.
ROUND_15_INTENDED_STRICTER_LEADING_EXCLUSIONS = tuple(
    (route, mixed, english, "hebrew-stricter")
    for route in ("il", "us")
    for mixed, english in (
        ("למעט missing section 2", "Except missing section 2."),
        ("להוציא missing section 2", "Except missing section 2."),
        ("זולת missing section 2", "Except missing section 2."),
        ("חוץ מזה, missing section 2", "Besides that, missing section 2."),
        ("פרט לזה, missing section 2", "Apart from that, missing section 2."),
    )
)


@pytest.mark.parametrize(
    ("route", "mixed", "english", "verdict"),
    ROUND_15_INTENDED_STRICTER_LEADING_EXCLUSIONS,
)
def test_leading_exclusion_fragments_are_intentionally_stricter(
    route, mixed, english, verdict
):
    translated = _english_accepted("a", english, REVIEW_ENGLISH_SOURCE)
    actual = (
        _accepted("a", mixed, REVIEW_SOURCE)
        if route == "il"
        else _english_accepted("a", mixed, REVIEW_ENGLISH_SOURCE)
    )
    assert (actual, translated) == _DIFFERENTIAL_VERDICTS[verdict]


# --- Review round 16 ------------------------------------------------------------
# Every reported form is pinned in reason and source positions, using faithful
# translations. Source פן (lest) is translated with its negative-purpose sense,
# "so that ... not", so the English link guard reads the same scope.
ROUND_16_REPRODUCERS = (
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "ל\u0000א חסר סעיף 2",
        "Section 2 is not missing.",
        "neither",
    ),
    (
        "(א) ל\u0000א מחושב בהתאם לסעיף 2.",
        "(a) It is not computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א\u0000) ישולם בהתאם לסעיף 2.",
        REVIEW_ENGLISH_SOURCE,
        "חסר סעיף 2",
        "Missing section 2.",
        "both",
    ),
    (
        "(א) אל יחושב בהתאם לסעיף 2.",
        "(a) Let it not be calculated in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "אל יטען שחסר סעיף 2",
        "Let it not be claimed that section 2 is missing.",
        "neither",
    ),
    (
        "(א) התשלום ייקבע באין חישוב בהתאם לסעיף 2.",
        "(a) The payment shall be determined without calculation in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "באין בסיס לטענה שחסר סעיף 2",
        "With no basis for claiming that section 2 is missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "פן תטען שחסר סעיף 2",
        "Lest you claim that section 2 is missing.",
        "neither",
    ),
    (
        "(א) פן יחושב בהתאם לסעיף 2.",
        "(a) So that it is not computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        "(א) מס על כבל מחושב בהתאם לסעיף 2.",
        "(a) Tax on a cable is computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "both",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "מס על כבל חסר סעיף 2",
        "Tax on a cable is missing section 2.",
        "both",
    ),
    (
        "(א) הכנסה מעולם העבודה מחושבת בהתאם לסעיף 2.",
        "(a) Income from the world of work is computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "both",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "מידע מעולם העבודה חסר סעיף 2",
        "Information from the world of work is missing section 2.",
        "both",
    ),
    (
        "(א) חוץ מ ־ תשלום בהתאם לסעיף 2.",
        "(a) Except a payment in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "חוץ מ ־ הטענה שחסר סעיף 2",
        "Except the claim that section 2 is missing.",
        "neither",
    ),
    (
        "(א) ישולם בהתאם ל־ סעיף 2.",
        REVIEW_ENGLISH_SOURCE,
        "חסר סעיף 2",
        "Missing section 2.",
        "both",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "חסר בהתאם ל־ סעיף 2",
        "Missing in accordance with section 2.",
        "both",
    ),
)


@pytest.mark.parametrize(
    ("hebrew_source", "english_source", "hebrew_reason", "english_reason", "verdict"),
    ROUND_16_REPRODUCERS,
)
def test_round_16_reproducers_get_the_english_verdict(
    hebrew_source, english_source, hebrew_reason, english_reason, verdict
):
    assert (
        _accepted("a", hebrew_reason, hebrew_source),
        _english_accepted("a", english_reason, english_source),
    ) == _DIFFERENTIAL_VERDICTS[verdict]


def test_closed_negator_class_covers_every_form_and_allowed_prefix():
    """The finite inventory drives both guards, including every person of אל."""

    from axiom_encode.harness import hebrew_negators as grammar

    # Contexts disambiguate the closed inventory's genuine conditional forms.
    tails = {
        "אל": " יחושב",
        "מעולם": " לא חושב",
        "מנוע": " מלפעול",
        "מנועה": " מלפעול",
        "מנועים": " מלפעול",
        "מנועות": " מלפעול",
        "דוחות": " את הטענה",
        "אף": " טענה לא הוכחה",
        "הלא": " מנוצל",
    }
    inventory = {form for family in grammar.NEGATOR_FAMILIES for form in family.forms}
    assert {row[0] for row in ROUND_15_NEGATOR_TRANSLATIONS} <= inventory
    for family in grammar.NEGATOR_FAMILIES:
        for prefix, form in itertools.product(family.prefixes, family.forms):
            phrase = prefix + form + tails.get(form, "")
            reason = list(hebrew._REASON_NEGATION.finditer(phrase))
            source = list(hebrew.link_negations(phrase))
            assert reason, phrase
            assert [m.span() for m in reason] == [m.span() for m in source], phrase
    for person, verb in (("ת", "תחשב"), ("י", "יחשב"), ("א", "אחשב"), ("נ", "נחשב")):
        phrase = f"אל {verb}"
        assert hebrew._REASON_NEGATION.search(phrase), person
        assert list(hebrew.link_negations(phrase)), person
    # Round 18: ב with an absence noun is "without", under the prefix chains of
    # the other "without" forms and with a joint after any letter. Pinned by
    # name, so no edit drops a form silently.
    assert grammar.ABSENCE_NOUNS == ("היעדר", "העדר", "חוסר")
    absence = [
        family for family in grammar.NEGATOR_FAMILIES if "בהיעדר" in family.forms
    ]
    assert [family.forms for family in absence] == [("בהיעדר", "בהעדר", "בחוסר")]
    assert absence[0].prefixes == grammar.ORDINARY_PREFIX_CHAINS
    for prefix, noun in itertools.product(
        grammar.ORDINARY_PREFIX_CHAINS, grammar.ABSENCE_NOUNS
    ):
        for joint in ("", "־"):
            phrase = "".join(letter + joint for letter in prefix + "ב") + noun
            reason = list(hebrew._REASON_NEGATION.finditer(phrase + " חישוב"))
            source = list(hebrew.link_negations(phrase + " חישוב"))
            assert [m.span() for m in reason] == [(0, len(phrase))], phrase
            assert [m.span() for m in source] == [(0, len(phrase))], phrase
    # Nouns, causal uses, בחסר ("understated", "short"), a pronoun suffix and
    # היעדרות (leave) negate nothing.
    for phrase in (
        "היעדר",
        "העדר",
        "חוסר",
        "מחוסר",
        "כמחוסר",
        "עקב היעדר",
        "בשל היעדר",
        "בחסר",
        "ביתר או בחסר",
        "בהעדרה",
        "בהיעדרו",
        "בהיעדרות",
        "לבהיעדר",
        "הבהיעדר",
    ):
        assert not hebrew._REASON_NEGATION.search(phrase + " חישוב"), phrase
        assert not list(hebrew.link_negations(phrase + " חישוב")), phrase


# These conservative readings are intended, rather than unlabelled exceptions
# to translation parity. Unpointed בל also spells the proper name Bel; the
# finite grammar retains its negative reading even in a putative name context.
# Likewise, פן and באין remain negative in fragments and conditional clauses
# that the English guard accepts. English behavior is kept unchanged.
ROUND_16_INTENDED_STRICTER_CONTEXTS = (
    (
        "bel-leading-reason",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "בל: חסר סעיף 2",
        "Bel: Missing section 2.",
        "hebrew-stricter",
    ),
    (
        "bel-contextual-reason",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "לגבי בל: חסר סעיף 2",
        "Concerning Bel: Missing section 2.",
        "hebrew-stricter",
    ),
    (
        "bel-source",
        "(א) מס על בל מחושב בהתאם לסעיף 2.",
        "(a) Tax on Bel is computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "hebrew-stricter",
    ),
    (
        "lest-colon-reason",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "פן תטען: חסר סעיף 2",
        "Lest you claim: Missing section 2.",
        "hebrew-stricter",
    ),
    (
        "lest-source",
        "(א) פן תחשב בהתאם לסעיף 2.",
        "(a) Lest you calculate in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "hebrew-stricter",
    ),
    (
        "without-calculation-fragment",
        "(א) באין חישוב בהתאם לסעיף 2.",
        "(a) Without a calculation in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "hebrew-stricter",
    ),
    (
        "in-the-absence-of-another-provision",
        "(א) באין הוראה אחרת יחושב בהתאם לסעיף 2.",
        "(a) In the absence of another provision it shall be calculated in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "hebrew-stricter",
    ),
    # Both literal translations are incomplete fragments. Their absence
    # particle still receives the conservative negative reading, rather than
    # supplying a complete affirmative missing-dependency claim.
    (
        "incomplete-absence-of-missing-reason",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "באין חסר סעיף 2",
        "In the absence of missing section 2.",
        "hebrew-stricter",
    ),
    (
        "incomplete-absence-reason",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "באין: חסר סעיף 2",
        "in the absence of: Missing section 2.",
        "hebrew-stricter",
    ),
    (
        "incomplete-absence-source",
        "(א) באין בהתאם לסעיף 2.",
        "(a) in the absence of in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "hebrew-stricter",
    ),
)


@pytest.mark.parametrize(
    (
        "name",
        "hebrew_source",
        "english_source",
        "hebrew_reason",
        "english_reason",
        "verdict",
    ),
    ROUND_16_INTENDED_STRICTER_CONTEXTS,
    ids=[row[0] for row in ROUND_16_INTENDED_STRICTER_CONTEXTS],
)
def test_round_16_contexts_are_intentionally_stricter(
    name, hebrew_source, english_source, hebrew_reason, english_reason, verdict
):
    assert (
        _accepted("a", hebrew_reason, hebrew_source),
        _english_accepted("a", english_reason, english_source),
    ) == _DIFFERENTIAL_VERDICTS[verdict], name


# Round 13 already labels negated לפי links stricter than English ``under``.
# Pin the review matrix's complete 31-row non-computation extension too:
# the exclusion remains negative even when its object is delay, rather than
# calculation. This does not widen the special ``without delay`` class.
_ROUND_16_EXCLUSION_WORDS = (
    ("except", "en", "except"),
    ("other than", "en", "other than"),
    ("rather than", "en", "rather than"),
    ("not including", "en", "not including"),
    ("außer", "de", "except"),
    ("ausgenommen", "de", "except"),
    ("abgesehen von", "de", "except for"),
    ("למעט", "he", "except"),
    ("להוציא", "he", "except"),
    ("זולת", "he", "except"),
    ("חוץ מ", "he", "except"),
    ("פרט ל", "he", "except"),
)
_ROUND_16_DELAY_WORDS = (("דיחוי", "he"), ("delay", "en"), ("Verzögerung", "de"))


def _round_16_exclusion_object(exclusion, obj):
    if exclusion.endswith((" מ", " ל")):
        return f"{exclusion}-{obj}"
    return f"{exclusion} {obj}"


ROUND_16_INTENDED_STRICTER_EXCLUSION_OBJECTS = tuple(
    (
        f"{exclusion}-{obj}",
        f"(א) משולם {_round_16_exclusion_object(exclusion, obj)} לפי סעיף 2.",
        f"(a) It is paid {english} delay under section 2.",
        "hebrew-stricter",
    )
    for (exclusion, language, english), (obj, obj_language) in itertools.product(
        _ROUND_16_EXCLUSION_WORDS, _ROUND_16_DELAY_WORDS
    )
    if (language, obj_language) != ("he", "he")
)


@pytest.mark.parametrize(
    ("name", "hebrew_source", "english_source", "verdict"),
    ROUND_16_INTENDED_STRICTER_EXCLUSION_OBJECTS,
    ids=[row[0] for row in ROUND_16_INTENDED_STRICTER_EXCLUSION_OBJECTS],
)
def test_round_16_exclusion_objects_are_intentionally_stricter(
    name, hebrew_source, english_source, verdict
):
    assert len(ROUND_16_INTENDED_STRICTER_EXCLUSION_OBJECTS) == 31
    assert (
        _accepted("a", "חסר סעיף 2", hebrew_source),
        _english_accepted("a", "Missing section 2.", english_source),
    ) == _DIFFERENTIAL_VERDICTS[verdict], name


# Prefix-only imperfect recognition is intentionally conservative in unpointed
# text. Each of the four person prefixes can also begin a recipient noun, as
# the pre-existing תושב case shows. Pin that limitation for the whole four-way
# class at both positions rather than implying only ת has a noun ambiguity.
def test_all_imperfect_person_prefix_noun_ambiguities_are_intentionally_stricter():
    nouns = (
        ("ת", "תושב", "a resident"),
        ("י", "ילד", "a child"),
        ("א", "אזרח", "a citizen"),
        ("נ", "נכה", "a disabled person"),
    )
    assert {prefix for prefix, _word, _english in nouns} == set("תיאנ")
    for prefix, noun, english in nouns:
        assert noun.startswith(prefix)
        # This named verdict deliberately documents a Hebrew-stricter noun
        # reading; it is not claimed to be a prohibitive English translation.
        expected = _DIFFERENTIAL_VERDICTS["hebrew-stricter"]
        assert (
            _accepted("a", f"אל {noun}: חסר סעיף 2", REVIEW_SOURCE),
            _english_accepted(
                "a", f"To {english}: Missing section 2.", REVIEW_ENGLISH_SOURCE
            ),
        ) == expected, ("reason", prefix)
        assert (
            _accepted("a", "חסר סעיף 2", f"(א) ישולם אל {noun} בהתאם לסעיף 2."),
            _english_accepted(
                "a",
                "Missing section 2.",
                f"(a) It shall be paid to {english} in accordance with section 2.",
            ),
        ) == expected, ("source", prefix)


# --- Review round 17 ------------------------------------------------------------
# A governor (``יש``, ``ניתן``, ``כדי`` ...) makes ``להוציא`` the infinitive
# "to issue" across any run of horizontal whitespace, at both positions. The
# English for every gap is the same sentence, which English accepts.
_ROUND_17_GOVERNOR_SOURCE = "(א) יש{gap}להוציא את חישוב המס בהתאם לסעיף 2."
_ROUND_17_GOVERNOR_ENGLISH_SOURCE = (
    "(a) The tax computation must be issued in accordance with section 2."
)
_ROUND_17_GOVERNOR_REASON = "חסר סעיף 2, שלפיו יש{gap}להוציא את החישוב"
_ROUND_17_GOVERNOR_ENGLISH_REASON = (
    "Missing section 2, under which the computation must be issued."
)
_ROUND_17_GAPS = (" ", "  ", "\t", " ", "　", " \t  ")
ROUND_17_REPRODUCERS = (
    *(
        (
            _ROUND_17_GOVERNOR_SOURCE.format(gap=gap),
            _ROUND_17_GOVERNOR_ENGLISH_SOURCE,
            "חסר סעיף 2",
            "Missing section 2.",
            "both",
        )
        for gap in _ROUND_17_GAPS
    ),
    *(
        (
            REVIEW_SOURCE,
            REVIEW_ENGLISH_SOURCE,
            _ROUND_17_GOVERNOR_REASON.format(gap=gap),
            _ROUND_17_GOVERNOR_ENGLISH_REASON,
            "both",
        )
        for gap in _ROUND_17_GAPS
    ),
    # Short prefix-letter runs before a dash. A run grammar cannot stack as
    # proclitics is a word, so the dash after it is punctuation: בבב־לא and
    # של־לא ("of", never ש with ללא) keep their negator, as ובל ("and not")
    # does. A grammatical stack joins: ומ־לא is ומלא ("and full").
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "בבב־לא חסר סעיף 2",
        "Bbb: Section 2 is not missing.",
        "neither",
    ),
    (
        "(א) מס על בבב־לא מחושב בהתאם לסעיף 2.",
        "(a) Tax on Bbb - not computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "של־לא חסר סעיף 2",
        "Of: Section 2 is not missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "הב־לא חסר סעיף 2",
        "Give: Section 2 is not missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "ובל־נכון שחסר סעיף 2",
        "And let it not be true that section 2 is missing.",
        "neither",
    ),
    (
        "(א) מס ובל־מחושב בהתאם לסעיף 2.",
        "(a) Tax, and not computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "כלב־לא חסר סעיף 2",
        "Dog: Section 2 is not missing.",
        "neither",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "כלב־חסר סעיף 2",
        "Dog: Missing section 2.",
        "both",
    ),
    (
        "(א) ישולם בבב־לפי סעיף 2.",
        "(a) Payment shall be made Bbb - under section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "both",
    ),
    (
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "ומ־לא חסר סעיף 2",
        "And full: Missing section 2.",
        "both",
    ),
)


@pytest.mark.parametrize(
    ("hebrew_source", "english_source", "hebrew_reason", "english_reason", "verdict"),
    ROUND_17_REPRODUCERS,
)
def test_round_17_reproducers_get_the_english_verdict(
    hebrew_source, english_source, hebrew_reason, english_reason, verdict
):
    assert (
        _accepted("a", hebrew_reason, hebrew_source),
        _english_accepted("a", english_reason, english_source),
    ) == _DIFFERENTIAL_VERDICTS[verdict]


# Intended stricter readings. A dash after the full governor word is
# punctuation, as after any full word, so ``יש־להוציא`` keeps "except". An
# ambiguous run (ככ ends in a non-final letter; וו is "hook" or ו before a
# ו-initial word) joins for readers that admit a deferral, while readers that
# refuse one also restart after it: each takes the stricter reading.
ROUND_17_INTENDED_STRICTER = (
    (
        "governor-dash-source",
        _ROUND_17_GOVERNOR_SOURCE.format(gap="־"),
        _ROUND_17_GOVERNOR_ENGLISH_SOURCE,
        "חסר סעיף 2",
        "Missing section 2.",
        "hebrew-stricter",
    ),
    (
        "governor-spaced-dash-source",
        _ROUND_17_GOVERNOR_SOURCE.format(gap=" ־ "),
        _ROUND_17_GOVERNOR_ENGLISH_SOURCE,
        "חסר סעיף 2",
        "Missing section 2.",
        "hebrew-stricter",
    ),
    (
        "governor-dash-reason",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        _ROUND_17_GOVERNOR_REASON.format(gap="־"),
        _ROUND_17_GOVERNOR_ENGLISH_REASON,
        "hebrew-stricter",
    ),
    (
        "ambiguous-non-final-run",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "ככ־לא חסר סעיף 2",
        "Like a prison: Missing section 2.",
        "hebrew-stricter",
    ),
    (
        "ambiguous-vav-vav-run",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "וו־חסר סעיף 2",
        "Hook: Missing section 2.",
        "hebrew-stricter",
    ),
)


@pytest.mark.parametrize(
    (
        "name",
        "hebrew_source",
        "english_source",
        "hebrew_reason",
        "english_reason",
        "verdict",
    ),
    ROUND_17_INTENDED_STRICTER,
    ids=[row[0] for row in ROUND_17_INTENDED_STRICTER],
)
def test_round_17_readings_are_intentionally_stricter(
    name, hebrew_source, english_source, hebrew_reason, english_reason, verdict
):
    assert (
        _accepted("a", hebrew_reason, hebrew_source),
        _english_accepted("a", english_reason, english_source),
    ) == _DIFFERENTIAL_VERDICTS[verdict], name


def test_every_governor_takes_every_horizontal_whitespace_run():
    """Exhaustive over governors, horizontal spaces and mixed runs."""

    from axiom_encode.harness.hebrew_negators import (
        INFINITIVE_GOVERNORS,
        governs_infinitive,
    )
    from axiom_encode.harness.hebrew_prefixes import HORIZONTAL_SPACES

    assert HORIZONTAL_SPACES == {
        chr(code)
        for code in range(sys.maxunicode + 1)
        if chr(code).isspace()
        and unicodedata.category(chr(code)) in ("Zs", "Cc")
        and chr(code) not in "\n\v\f\r\x1c\x1d\x1e\x1f\x85"
    }
    runs = (
        *HORIZONTAL_SPACES,
        *(
            "".join(pair)
            for pair in itertools.product(sorted(HORIZONTAL_SPACES), repeat=2)
        ),
        " " * 64,
    )
    readers = (
        hebrew._NEGATIONS,
        hebrew._REASON_NEGATION,
        hebrew._ADVERSATIVE,
        hebrew._ANY_ADVERSATIVE,
        completeness_module._ADVERSATIVE_LANGUAGE,
    )
    for governor in sorted(INFINITIVE_GOVERNORS):
        for gap in runs:
            text = f"{governor}{gap}להוציא את החישוב"
            start = len(governor) + len(gap)
            assert governs_infinitive(text, start), (governor, gap)
            for reader in readers:
                assert reader.search(text) is None, (governor, ascii(gap))
                assert reader.fullmatch(text, start, start + 6) is None
        for gap in ("־", " ־ ", "\n", "\u2028"):
            text = f"{governor}{gap}להוציא את החישוב"
            assert not governs_infinitive(text, len(governor) + len(gap))
            # A dash after the full word is punctuation, so להוציא is "except".
            # Across a line break, the statute-corpus homograph list
            # (hebrew_negators._ISSUING_GOVERNORS) still reads רשאית as before.
            for reader in readers if gap.strip() else ():
                assert reader.search(text) is not None, (governor, ascii(gap))
                start = len(governor) + len(gap)
                assert reader.fullmatch(text, start, start + 6) is not None
    # A longer word ending in a governor is no governor: אפשר but not לאפשר.
    for text in ("לאפשר להוציא", "ממנת  להוציא", "יש\tלהוציא"):
        start = text.index("להוציא")
        assert governs_infinitive(text, start) == text.startswith("יש")


# The English guard is wording-sensitive (``_english_accepted`` at round 16's
# head): ``Missing section 2.`` passes and ``Section 2 is missing.`` fails, so
# one Hebrew sentence has faithful English renderings with different verdicts.
# Parity is with the canonical faithful English, the rendering in the wording
# the English guard recognises. Round 16's fifteen faithful-positive rows are
# parity rows: Hebrew accepts as the canonical English does. Changing English
# to accept more wordings is outside this PR, whose English verdicts hold.
ROUND_17_ENGLISH_WORDING_ROWS = (
    *(
        (
            f"joint-{ascii(joint)}-reason-signal",
            f"ו{joint}חסר סעיף 2",
            "And section 2 is missing.",
            "And missing section 2.",
        )
        for joint in (
            "־",
            "-",
            "‐",
            "‑",
            "–",
            "—",
            "‒",
            "−",
            "﹘",
            "﹣",
            "－",
            " ־",
            "־ ",
            " ־ ",
        )
    ),
    (
        "world-reason",
        "בדוגמה מעולם העבודה חסר סעיף 2",
        "In an example from the world of work, section 2 is missing.",
        "In an example from the world of work: Missing section 2.",
    ),
)


@pytest.mark.parametrize(
    ("name", "hebrew_reason", "alternate_english", "canonical_english"),
    ROUND_17_ENGLISH_WORDING_ROWS,
    ids=[row[0] for row in ROUND_17_ENGLISH_WORDING_ROWS],
)
def test_english_wording_rows_take_the_canonical_english_verdict(
    name, hebrew_reason, alternate_english, canonical_english
):
    assert len(ROUND_17_ENGLISH_WORDING_ROWS) == 15
    hebrew_verdict = _accepted("a", hebrew_reason, REVIEW_SOURCE)
    canonical = _english_accepted("a", canonical_english, REVIEW_ENGLISH_SOURCE)
    assert hebrew_verdict == canonical, name
    # Both renderings stay in the row: the alternate's refusal is the English
    # guard's wording sensitivity, not a Hebrew-only accept.
    assert (
        canonical,
        _english_accepted("a", alternate_english, REVIEW_ENGLISH_SOURCE),
    ) == (
        True,
        False,
    ), name


# --- Review round 18 ------------------------------------------------------------
# "Without" is also ב with an absence noun in construct
# (hebrew_negators.ABSENCE_NOUNS): בהיעדר, its defective spelling בהעדר, and
# בחוסר. Each takes the prefix chains of the other "without" forms
# (ORDINARY_PREFIX_CHAINS) and their joints, in reasons and in sources. Bare
# and causal absence nouns stay nouns, as English "absence" and "lack" do;
# so do בחסר ("understated", "short"), a pronoun-suffixed form and היעדרות
# (absence from work). The canonical English of a "without" source is
# ``Without calculation ...``, which English refuses.
#
# A word reads only as a complete token. A governor or context word that ends
# an identifier (``_יש``, ``1יש``, ``Xיש``) ends no exception
# (hebrew_negators.starts_token), and signals, links and citations, which
# admit a deferral, read a word only where English ``\b`` would.
_ROUND_18_ABSENCE_FORMS = ("בהיעדר", "בהעדר", "בחוסר")
_ROUND_18_CHAIN_GLOSSES = {
    "": "",
    "ו": "and",
    "ש": "that",
    "כש": "when",
    "מש": "since",
    "לכש": "once",
    "בש": "when",
    "ד": "that",
    "וש": "and that",
    "וכש": "and when",
    "ומש": "and since",
    "ולכש": "and once",
    "ובש": "and when",
    "וד": "and that",
}
_ROUND_18_WITHOUT_SOURCE = "(a) Without calculation in accordance with section 2."
_ROUND_18_JOINTED_ABSENCE = (
    "ב־היעדר",
    "ב-העדר",
    "ב־חוסר",
    "ב–היעדר",
    "ב－העדר",
    "ו־בהיעדר",
    "ו־ב־היעדר",
    "ש־ב־העדר",
    "ו ־ בהיעדר",
    "ב ־ היעדר",
    "ו־ב ־ חוסר",
)
# Characters glued to the left of a word: a letter or digit of another
# script, connector punctuation, a combining mark, and a default-ignorable
# (U+200D, U+2060, U+00AD) or Hebrew point that the fold removes.
_ROUND_18_GLUED = (
    "_",
    "1",
    "X",
    "Я",
    "٣",
    "ß",
    "á",
    "_‍",
    "X‍",
    "1⁠",
    "_­",
    "‿",
    "ת״",
    "Xְ",
)
_ROUND_18_OTHER_SCRIPT = ("Я", "ß", "٣", "é")
_ROUND_18_GOVERNOR_CONTROL_ENGLISH = (
    "(a) Tax on {}: a calculation must be issued in accordance with section 2."
)


def _round_18_without(gloss: str) -> str:
    return f"{gloss} without".strip().capitalize()


ROUND_18_REPRODUCERS = (
    # The review's table: ב with an absence noun means "without", as ללא and
    # באין do, so the link it negates applies nothing.
    *(
        (
            f"(א) {form} חישוב בהתאם לסעיף 2.",
            _ROUND_18_WITHOUT_SOURCE,
            "חסר סעיף 2",
            "Missing section 2.",
            "neither",
        )
        for form in (*_ROUND_18_ABSENCE_FORMS, "ללא", "באין")
    ),
    (
        "(א) התשלום ישולם בהיעדר חישוב בהתאם לסעיף 2.",
        "(a) The payment shall be paid without calculation in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    # Parity rows that bound the rule: bare and causal absence nouns.
    *(
        (hebrew_source, english_source, "חסר סעיף 2", "Missing section 2.", "both")
        for hebrew_source, english_source in (
            (
                "(א) עקב היעדר חישוב בהתאם לסעיף 2.",
                "(a) Due to the absence of calculation in accordance with section 2.",
            ),
            (
                "(א) מחוסר חישוב בהתאם לסעיף 2.",
                "(a) For lack of calculation in accordance with section 2.",
            ),
            (
                "(א) אי תשלום בהתאם לסעיף 2.",
                "(a) Non-payment in accordance with section 2.",
            ),
            (
                "(א) באי תשלום בהתאם לסעיף 2.",
                "(a) Upon non-payment in accordance with section 2.",
            ),
            (
                "(א) היעדר חישוב בהתאם לסעיף 2.",
                "(a) Absence of calculation in accordance with section 2.",
            ),
            (
                "(א) חוסר חישוב בהתאם לסעיף 2.",
                "(a) Lack of calculation in accordance with section 2.",
            ),
            (
                "(א) בשל היעדר חישוב בהתאם לסעיף 2.",
                "(a) Because of the absence of calculation in accordance with section 2.",
            ),
            # בחסר in the statutes: "in deficit" and a short sale.
            (
                "(א) כספים שחולקו ביתר או בחסר לפי סעיף 2.",
                "(a) Funds distributed in excess or in deficit under section 2.",
            ),
            (
                "(א) מכירה בחסר בהתאם לסעיף 2.",
                "(a) A short sale in accordance with section 2.",
            ),
            # A pronoun suffix leaves no later object; היעדרות is leave.
            (
                "(א) בהעדרה, ישולם בהתאם לסעיף 2.",
                "(a) In its absence, payment shall be made in accordance with section 2.",
            ),
            (
                "(א) בהיעדרות העובד ישולם בהתאם לסעיף 2.",
                "(a) During the employee's leave, payment shall be made in accordance "
                "with section 2.",
            ),
            # The snapshot's own uses (national health insurance law, schedule 2,
            # items 14 and 19) read as "without" reads in English: after the
            # link, or in a clause of their own, they negate no link.
            (
                "(א) מתן טיפול ראשוני במצבי חולי חריפים, בהתאם לסעיף 2, בהעדר רופא.",
                "(a) Primary care in acute illness, in accordance with section 2, "
                "without a doctor.",
            ),
            (
                "(א) ישולם בהתאם לסעיף 2 לחולים שלגביהם קיימת הוריית נגד לטיפול "
                "בקרינה, בהיעדר עדות לגרורות.",
                "(a) Payment is made in accordance with section 2 to patients for "
                "whom radiotherapy is contraindicated, without evidence of metastases.",
            ),
            (
                "(א) מתן טיפול ראשוני, בהעדר רופא; ישולם בהתאם לסעיף 2.",
                "(a) Primary care, without a doctor; payment is made in accordance "
                "with section 2.",
            ),
        )
    ),
    # Every prefix chain of the "without" forms: in a source before the link,
    # and in a reason after its citation, where it negates nothing cited.
    *(
        (
            f"(א) {chain}{form} חישוב בהתאם לסעיף 2.",
            f"(a) {_round_18_without(gloss)} calculation in accordance with section 2.",
            "חסר סעיף 2",
            "Missing section 2.",
            "neither",
        )
        for chain, gloss in _ROUND_18_CHAIN_GLOSSES.items()
        for form in _ROUND_18_ABSENCE_FORMS
    ),
    *(
        (
            REVIEW_SOURCE,
            REVIEW_ENGLISH_SOURCE,
            f"חסר סעיף 2 {chain}{form} חישוב",
            f"Missing section 2 {_round_18_without(gloss).lower()} calculation.",
            "both",
        )
        for chain, gloss in _ROUND_18_CHAIN_GLOSSES.items()
        for form in _ROUND_18_ABSENCE_FORMS
    ),
    # Joints inside the prefix chain, spaced or not, and after the noun.
    *(
        (
            f"(א) {spelling} חישוב בהתאם לסעיף 2.",
            _ROUND_18_WITHOUT_SOURCE,
            "חסר סעיף 2",
            "Missing section 2.",
            "neither",
        )
        for spelling in _ROUND_18_JOINTED_ABSENCE
    ),
    (
        "(א) בהיעדר־חישוב בהתאם לסעיף 2.",
        _ROUND_18_WITHOUT_SOURCE,
        "חסר סעיף 2",
        "Missing section 2.",
        "neither",
    ),
    # A governor ends no exception as the end of an identifier, a default-
    # ignorable between them notwithstanding.
    *(
        (
            f"(א) מס על המוצר {glued}יש להוציא חישוב בהתאם לסעיף 2.",
            f"(a) Tax on product {glued}יש, except calculation in accordance with "
            "section 2.",
            "חסר סעיף 2",
            "Missing section 2.",
            "neither",
        )
        for glued in _ROUND_18_GLUED
    ),
    *(
        (
            REVIEW_SOURCE,
            REVIEW_ENGLISH_SOURCE,
            f"חסר {glued}יש להוציא סעיף 2",
            f"Missing {glued}יש except section 2.",
            "neither",
        )
        for glued in ("_", "1", "X", "Я")
    ),
    # So do the context words that make להוציא "to issue", a foreign חוץ, a
    # concessive על אף.
    *(
        (hebrew_source, english_source, "חסר סעיף 2", "Missing section 2.", "neither")
        for hebrew_source, english_source in (
            (
                "(א) מס על המוצר _הורשו להוציא חישוב בהתאם לסעיף 2.",
                "(a) Tax on product _הורשו, except calculation in accordance with "
                "section 2.",
            ),
            (
                "(א) מס על המוצר Xהורשו להוציא חישוב בהתאם לסעיף 2.",
                "(a) Tax on product Xהורשו, except calculation in accordance with "
                "section 2.",
            ),
            (
                "(א) מס 1מדינת חוץ מחישוב בהתאם לסעיף 2.",
                "(a) Tax 1מדינת, except calculation in accordance with section 2.",
            ),
            (
                "(א) מס Xמדינת חוץ מחישוב בהתאם לסעיף 2.",
                "(a) Tax Xמדינת, except calculation in accordance with section 2.",
            ),
            (
                "(א) _רשאית קופה להוציא חישוב בהתאם לסעיף 2.",
                "(a) _רשאית fund, except calculation in accordance with section 2.",
            ),
            (
                "(א) _על אף חישוב בהתאם לסעיף 2 לא נעשה.",
                "(a) _על not even a calculation in accordance with section 2 is made.",
            ),
        )
    ),
    # A whole governor or context word still governs: at a space, after a
    # dash, after a mark and its space, after other punctuation, or after two
    # spaces, which no script junction inserts.
    *(
        (
            f"(א) מס על {before}יש להוציא חישוב בהתאם לסעיף 2.",
            _ROUND_18_GOVERNOR_CONTROL_ENGLISH.format(english),
            "חסר סעיף 2",
            "Missing section 2.",
            "both",
        )
        for before, english in (
            ("המוצר ", "the product"),
            ("המוצר ו־", "the product, and"),
            ("המוצר־", "the product -"),
            ("המוצר á ", "product á"),
            ("המוצר X: ", "product X"),
            ("המוצר X  ", "product X"),
        )
    ),
    (
        "(א) מס על המוצר הורשו להוציא חישוב בהתאם לסעיף 2.",
        "(a) Tax on the product: they were permitted to issue a calculation in "
        "accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "both",
    ),
    (
        "(א) מס מדינת חוץ מחושב בהתאם לסעיף 2.",
        "(a) Foreign-state tax is computed in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
        "both",
    ),
    # Signals, links and citations read only complete tokens, on either side
    # of the word, as English reads ``ЯMissing`` and ``MissingX``.
    *(
        row
        for glued in _ROUND_18_OTHER_SCRIPT
        for row in (
            (
                REVIEW_SOURCE,
                REVIEW_ENGLISH_SOURCE,
                f"{glued}חסר סעיף 2",
                f"{glued}Missing section 2.",
                "neither",
            ),
            (
                REVIEW_SOURCE,
                REVIEW_ENGLISH_SOURCE,
                f"{glued}ללא סעיף 2",
                f"{glued}Without section 2.",
                "neither",
            ),
            (
                REVIEW_SOURCE,
                REVIEW_ENGLISH_SOURCE,
                f"חסר {glued}סעיף 2",
                f"Missing {glued}section 2.",
                "neither",
            ),
            (
                f"(א) ישולם {glued}לפי סעיף 2.",
                f"(a) Payment shall be made {glued}under section 2.",
                "חסר סעיף 2",
                "Missing section 2.",
                "neither",
            ),
            (
                f"(א) ישולם {glued}על פי סעיף 2.",
                f"(a) Payment shall be made {glued}under section 2.",
                "חסר סעיף 2",
                "Missing section 2.",
                "neither",
            ),
            (
                f"(א) ישולם {glued}בהתאם לסעיף 2.",
                f"(a) Payment shall be made {glued}in accordance with section 2.",
                "חסר סעיף 2",
                "Missing section 2.",
                "neither",
            ),
        )
    ),
    *(
        row
        for glued in ("X", "Я", "1", "_", "٣")
        for row in (
            (
                REVIEW_SOURCE,
                REVIEW_ENGLISH_SOURCE,
                f"חסר{glued} סעיף 2",
                f"Missing{glued} section 2.",
                "neither",
            ),
            (
                REVIEW_SOURCE,
                REVIEW_ENGLISH_SOURCE,
                f"ללא{glued} סעיף 2",
                f"Without{glued} section 2.",
                "neither",
            ),
        )
    ),
    # A dash after a word of any script is punctuation (``Яו־``, as ``Xו־``),
    # and a combining mark or a space ends no token, in either language.
    *(
        (
            REVIEW_SOURCE,
            REVIEW_ENGLISH_SOURCE,
            hebrew_reason,
            english_reason,
            "both",
        )
        for hebrew_reason, english_reason in (
            *(
                (f"{glued}ו־חסר סעיף 2", f"{glued}ו-Missing section 2.")
                for glued in ("X", *_ROUND_18_OTHER_SCRIPT)
            ),
            ("áחסר סעיף 2", "áMissing section 2."),
            ("חסר́ סעיף 2", "Missinǵ section 2."),
            ("X חסר סעיף 2", "X: Missing section 2."),
            ("X: חסר סעיף 2", "X: Missing section 2."),
            ("חסר סעיף 2 X", "Missing section 2 X."),
        )
    ),
    # Finding 3: the compound בלתי זמין takes a joint, as English reads
    # "unavailable".
    *(
        (
            REVIEW_SOURCE,
            REVIEW_ENGLISH_SOURCE,
            hebrew_reason,
            english_reason,
            "both",
        )
        for hebrew_reason, english_reason in (
            ("בלתי זמינות הוראות סעיף 2", "Unavailable provisions of section 2."),
            ("בלתי־זמינות הוראות סעיף 2", "Unavailable provisions of section 2."),
            ("הוראות בלתי־זמינות שבסעיף 2", "Unavailable provisions of section 2."),
            ("בלתי-זמין סעיף 2", "Unavailable section 2."),
            ("בלתי‐זמינה הוראת סעיף 2", "Unavailable provision of section 2."),
            ("בלתי ־ זמינות הוראות סעיף 2", "Unavailable provisions of section 2."),
            ("בלתי – זמינות הוראות סעיף 2", "Unavailable provisions of section 2."),
        )
    ),
    *(
        (
            REVIEW_SOURCE,
            REVIEW_ENGLISH_SOURCE,
            hebrew_reason,
            english_reason,
            "neither",
        )
        for hebrew_reason, english_reason in (
            (
                "לא נכון שבלתי־זמינות הוראות סעיף 2",
                "It is not true that the provisions of section 2 are unavailable.",
            ),
            (
                "בלתי־זמינות הוראות סעיף 2. אולם הן זמינות",
                "Unavailable provisions of section 2. However they are available.",
            ),
            ("בלתי־מקודד סעיף 2", "Unencoded section 2."),
        )
    ),
)


@pytest.mark.parametrize(
    ("hebrew_source", "english_source", "hebrew_reason", "english_reason", "verdict"),
    ROUND_18_REPRODUCERS,
)
def test_round_18_reproducers_get_the_english_verdict(
    hebrew_source, english_source, hebrew_reason, english_reason, verdict
):
    assert (
        _accepted("a", hebrew_reason, hebrew_source),
        _english_accepted("a", english_reason, english_source),
    ) == _DIFFERENTIAL_VERDICTS[verdict]


# Intended stricter readings. An absence form negates any object, as ללא and
# באין do, and governs the rest of its clause; English ``without`` negates a
# link only before a computation word, and a reason not at all, since English
# reads "without" as a missing-dependency signal. Round 16 labels באין the
# same way (ROUND_16_INTENDED_STRICTER_CONTEXTS).
#
# In round 17's reason frame, English reads "except" after the citation and
# accepts, while Hebrew reads להוציא as the adversative wherever it stands, as
# round 17's dash governor does (governor-dash-reason). And the copy read with
# scripts split puts one space at each junction (``Xיש`` reads ``X יש``), so a
# governor one space after a letter of another script may follow no written
# space at all: no reader takes it as a governor.
ROUND_18_INTENDED_STRICTER = (
    (
        "absence-object-source",
        "(א) בהעדר רופא בהתאם לסעיף 2.",
        "(a) Without a doctor in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
    ),
    (
        "absence-before-comma-source",
        "(א) בהעדר רופא, ישולם בהתאם לסעיף 2.",
        "(a) Without a doctor, payment is made in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
    ),
    (
        "in-the-absence-of-another-provision",
        "(א) בהיעדר הוראה אחרת יחושב בהתאם לסעיף 2.",
        "(a) In the absence of another provision it shall be calculated in "
        "accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
    ),
    (
        "for-want-of-authority-source",
        "(א) בחוסר סמכות ישולם בהתאם לסעיף 2.",
        "(a) Without authority, payment is made in accordance with section 2.",
        "חסר סעיף 2",
        "Missing section 2.",
    ),
    (
        "absence-reason",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "בהיעדר סעיף 2",
        "Without section 2.",
    ),
    (
        "absence-comma-reason",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "בהיעדר הוראות, חסר סעיף 2",
        "Without provisions, missing section 2.",
    ),
    (
        "incomplete-absence-of-missing-reason",
        REVIEW_SOURCE,
        REVIEW_ENGLISH_SOURCE,
        "בהיעדר חסר סעיף 2",
        "In the absence of missing section 2.",
    ),
    *(
        (
            f"absence-before-signal-{chain or 'bare'}-{form}",
            REVIEW_SOURCE,
            REVIEW_ENGLISH_SOURCE,
            f"{chain}{form} הוראות חסר סעיף 2",
            f"{_round_18_without(gloss)} provisions, missing section 2.",
        )
        for chain, gloss in _ROUND_18_CHAIN_GLOSSES.items()
        for form in _ROUND_18_ABSENCE_FORMS
    ),
    *(
        (
            f"identifier-governor-reason-{ascii(glued)}",
            REVIEW_SOURCE,
            REVIEW_ENGLISH_SOURCE,
            f"חסר סעיף 2, שלפיו {glued}יש להוציא את החישוב",
            f"Missing section 2, under which {glued}יש except the computation.",
        )
        for glued in ("_", "1", "X", "Я")
    ),
    *(
        (
            f"junction-space-governor-{ascii(letter)}",
            f"(א) מס על המוצר {letter} יש להוציא חישוב בהתאם לסעיף 2.",
            _ROUND_18_GOVERNOR_CONTROL_ENGLISH.format(f"product {letter}"),
            "חסר סעיף 2",
            "Missing section 2.",
        )
        for letter in ("X", "Я", "á")
    ),
)


@pytest.mark.parametrize(
    ("name", "hebrew_source", "english_source", "hebrew_reason", "english_reason"),
    ROUND_18_INTENDED_STRICTER,
    ids=[row[0] for row in ROUND_18_INTENDED_STRICTER],
)
def test_round_18_readings_are_intentionally_stricter(
    name, hebrew_source, english_source, hebrew_reason, english_reason
):
    assert (
        _accepted("a", hebrew_reason, hebrew_source),
        _english_accepted("a", english_reason, english_source),
    ) == _DIFFERENTIAL_VERDICTS["hebrew-stricter"], name


@pytest.mark.parametrize("obj", ("חישוב", "רופא", "הוראה אחרת יחושב"))
def test_absence_forms_give_exactly_the_issues_of_the_without_forms(obj):
    """No error of a new kind: an absence form reports what ללא reports."""

    content = _deferral_content("a", "חסר סעיף 2")
    expected = _analyze(content, f"(א) ללא {obj} בהתאם לסעיף 2.")
    assert len(expected) == 2
    assert expected[0].startswith(DEFERRAL)
    assert expected[1].startswith("[complete-source-unit:structure]")
    for form in (*_ROUND_18_ABSENCE_FORMS, "באין", "בלי", "בלא"):
        assert _analyze(content, f"(א) {form} {obj} בהתאם לסעיף 2.") == expected, form


_ROUND_18_GOVERNOR_READERS = (
    hebrew._NEGATIONS,
    hebrew._REASON_NEGATION,
    hebrew._ADVERSATIVE,
    hebrew._ANY_ADVERSATIVE,
    completeness_module._ADVERSATIVE_LANGUAGE,
)
_ROUND_18_FOLDED = hebrew.folded_marks()


def _continues_a_word(char: str) -> bool:
    """A word character, digit, ``_`` or combining mark the fold keeps.

    Hebrew letters are left out: one before a governor spells a longer word
    (``איש``), or another governor (``בכדי``), which the round 17 tests cover.
    """

    return (
        ord(char) not in _ROUND_18_FOLDED
        and not "֐" <= char <= "׿"
        and not "יִ" <= char <= "ﭏ"
        and (char.isalnum() or unicodedata.category(char) in ("Mn", "Mc", "Me", "Pc"))
    )


_ROUND_18_JOINING = st.characters(categories=("L", "M", "N", "Pc")).filter(
    _continues_a_word
)
_ROUND_18_IGNORABLES = ("", "‍", "‌", "⁠", "­", "‏", "️")


def _governed_views(text: str) -> tuple[str, ...]:
    """The text as written, folded, and folded with scripts split."""

    folded = hebrew.fold_marks(text)
    return (text, folded, hebrew._split_scripts(folded)[0])


def _reads_exception(reader, view: str, start: int) -> bool:
    return any(match.start() <= start < match.end() for match in reader.finditer(view))


@pytest.mark.parametrize("governor", sorted(hebrew_negators.INFINITIVE_GOVERNORS))
@settings(max_examples=40, deadline=None)
@given(
    glued=st.text(_ROUND_18_JOINING, min_size=1, max_size=3),
    ignorable=st.sampled_from(_ROUND_18_IGNORABLES),
)
def test_a_governor_glued_to_a_word_governs_nothing(governor, glued, ignorable):
    """Every governor, every left neighbour that continues a word: except reads."""

    text = f"מס על {glued}{ignorable}{governor} להוציא את החישוב"
    for view in _governed_views(text):
        start = view.index("להוציא")
        assert not hebrew_negators.governs_infinitive(view, start), ascii(view)
        for reader in _ROUND_18_GOVERNOR_READERS:
            assert _reads_exception(reader, view, start), (ascii(view), reader)


_ROUND_18_BOUNDARIES = (
    " ",
    "\t",
    " ",
    "　",
    "\n",
    " ",
    "(",
    "[",
    ",",
    ";",
    ":",
    ".",
    "־",
    "-",
    "–",
    '"',
    "'",
    "“",
    "”",
)


@pytest.mark.parametrize("governor", sorted(hebrew_negators.INFINITIVE_GOVERNORS))
@settings(max_examples=40, deadline=None)
@given(
    before=st.sampled_from(("", "המוצר", "מס על המוצר")),
    boundary=st.sampled_from(_ROUND_18_BOUNDARIES),
)
def test_a_governor_at_a_token_boundary_still_governs(governor, before, boundary):
    """The exception reads as before wherever the governor starts a token."""

    text = f"{before}{boundary}{governor} להוציא את החישוב"
    for view in _governed_views(text):
        start = view.index("להוציא")
        assert hebrew_negators.governs_infinitive(view, start), ascii(view)
        for reader in _ROUND_18_GOVERNOR_READERS:
            assert not _reads_exception(reader, view, start), (ascii(view), reader)


@settings(max_examples=200, deadline=None)
@given(
    glued=st.text(_ROUND_18_JOINING, min_size=1, max_size=3),
    ignorable=st.sampled_from(_ROUND_18_IGNORABLES),
    context=st.sampled_from(
        (
            *(
                (f"מס על {{}}{word} להוציא את החישוב", "להוציא")
                for word in ("הורשו", "שהורשו", "הרשה", "להרשות", "רשאית", "שלא")
            ),
            ("{}רשאית קופה להוציא את החישוב", "להוציא"),
            *(
                (f"מס {{}}{head} חוץ מחישוב", "חוץ")
                for head in ("מדינת", "תושב", "הכנסות", "מסי", "ומדינת")
            ),
            ("{}על אף טענה לא הוכחה", "אף"),
        )
    ),
)
def test_a_context_word_glued_to_a_word_ends_no_exception(glued, ignorable, context):
    """A context word ends an exception only as a complete token."""

    frame, exception = context
    text = frame.format(glued + ignorable)
    for view in _governed_views(text):
        start = view.index(exception)
        assert _reads_exception(hebrew._NEGATIONS, view, start), ascii(view)
    # At a token boundary the context word still ends the exception.
    for view in _governed_views(frame.format("")):
        start = view.index(exception)
        assert not _reads_exception(hebrew._NEGATIONS, view, start), ascii(view)


_ROUND_18_WORD_CHARACTER = st.characters(categories=("L", "N")).filter(
    _continues_a_word
) | st.just("_")


@settings(max_examples=300, deadline=None)
@given(glued=st.text(_ROUND_18_WORD_CHARACTER, min_size=1, max_size=3))
def test_signals_links_and_citations_read_only_complete_tokens(glued):
    """What admits a deferral reads a word only where English ``\\b`` would."""

    assert all(re.fullmatch(r"\w", char) for char in glued)
    for text in (f"{glued}חסר סעיף 2", f"חסר{glued} סעיף 2"):
        assert not [
            match for match in hebrew._SIGNAL.finditer(text) if match.group("missing")
        ], ascii(text)
    for text in (f"{glued}ללא סעיף 2", f"ללא{glued} סעיף 2"):
        assert not [
            match for match in hebrew._SIGNAL.finditer(text) if match.group("without")
        ], ascii(text)
    assert hebrew.SECTION_DEPENDENCY.search(f"חסר {glued}סעיף 2") is None
    for linker in ("לפי ", "על פי ", "בהתאם ל"):
        text = f"ישולם {glued}{linker}"
        assert hebrew._source_linker(text, 0, len(text)) is None, ascii(text)
    # After a space, each still reads.
    assert hebrew._SIGNAL.search(f"{glued} חסר סעיף 2").group("missing")
    assert hebrew._SIGNAL.search(f"{glued} ללא סעיף 2").group("without")
    assert hebrew.SECTION_DEPENDENCY.search(f"חסר {glued} סעיף 2")
    for linker in ("לפי ", "על פי ", "בהתאם ל"):
        text = f"ישולם {glued} {linker}"
        assert hebrew._source_linker(text, 0, len(text)) is not None, ascii(text)


@pytest.mark.parametrize("joint", hebrew.PREFIX_JOINT_CHARS)
def test_unavailable_takes_every_joint(joint):
    """בלתי זמין is a missing signal across any joint the fold keeps."""

    for gap in (joint, f" {joint} "):
        for adjective in ("זמין", "זמינה", "זמינים", "זמינות"):
            text = f"בלתי{gap}{adjective} הוראות סעיף 2"
            match = hebrew._SIGNAL.search(text)
            assert match and match.group("missing") == text[: match.end()], ascii(text)
