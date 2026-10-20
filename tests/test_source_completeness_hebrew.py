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
import json
import re
import tempfile
import time
from pathlib import Path

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

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
        f"      reason: {json.dumps(reason, ensure_ascii=False)}\n"
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
_INVISIBLE_MARKS = (
    "\u05b0",
    "\u05b4",
    "\u05b5",
    "\u05b7",
    "\u05b8",
    "\u05b9",
    "\u05bc",
    "\u05c1",
    "\u05c7",
    "\u0591",
    "\u05a5",
    "\u00ad",
    "\u061c",
    "\u200b",
    "\u200d",
    "\u200e",
    "\u200f",
    "\u202b",
    "\u202c",
    "\u2067",
    "\u2069",
    "\ufeff",
)


@st.composite
def _marked(draw, text: str) -> str:
    """``text`` with any of its characters followed by an invisible mark."""

    marks = draw(
        st.lists(
            st.sampled_from(("", "", *_INVISIBLE_MARKS)),
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
        f"      reason: {json.dumps(reason)}\n"
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
        assert not hebrew._FOLDABLE.search(folded)
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
        "שמעולם",
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
        "ללא",
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


def test_native_citation_scans_stay_linear():
    """A long whitespace run or many citations cannot make a scan quadratic."""

    pattern = hebrew.current_branch_citation_pattern(CITATION, ("c",))
    # CPU time, not wall time: a shared host's scheduling cannot fail the
    # bound, and a quadratic scan still exceeds it.
    started = time.process_time()
    assert pattern.search("סעיף 14(ג)" + " " * 64_000 + "x")
    many = "חסר " + "סעיף 346 " * 8_000
    assert not hebrew.reason_names_source_bound_dependency(
        many, PRECISION_SOURCE, corpus_citation_path=CITATION
    )
    # Many reason clauses against many source citations of another section.
    assert not hebrew.reason_names_source_bound_dependency(
        "חסר סעיף 346. " * 8_000,
        "(ה1) " + "לפי סעיף 345ב. " * 8_000,
        corpus_citation_path=CITATION,
    )
    # Many reason fragments of one section, each a distinct query.
    assert not hebrew.reason_names_source_bound_dependency(
        " ".join(f"חסר סעיף 346({index})." for index in range(2, 1_000)),
        "(ה1) " + "לפי סעיף 346(1). " * 2_000,
        corpus_citation_path=CITATION,
    )
    # Many contextual references resolved against many named laws.
    assert (
        len(
            hebrew._section_citations(
                "חוק הביטוח הלאומי. " * 16_000 + "סעיף 345ב לפקודה האמורה. " * 16_000
            )
        )
        == 16_000
    )
    # An outline that deepens without end, and a citation chain thousands deep.
    assert hebrew.hebrew_outline_markers("(א) הוראה.\n(1) הוראה.\n" * 4_000) == ()
    assert not hebrew._linked_source_instruments("לפי סעיף 345ב" + "(1)" * 6_400 + ".")
    # Long whitespace runs inside an instrument name, and one long clause of
    # citations.
    assert (
        len(hebrew._section_citations("סעיף 345ב לחוק" + " " * 64_000 + "הביטוח.")) == 1
    )
    assert not hebrew.reason_names_source_bound_dependency(
        "חסר " + "סעיף 346 " * 64_000, PRECISION_SOURCE, corpus_citation_path=CITATION
    )
    # Many punctuation-joined signals before one citation.
    assert not hebrew.reason_names_source_bound_dependency(
        "חסר," * 16_000 + "סעיף 346.", PRECISION_SOURCE, corpus_citation_path=CITATION
    )
    # Many citations after one negation.
    assert not hebrew._linked_source_instruments(
        "אין לפעול " + "בהתאם לסעיף 345ב " * 8_000 + "."
    )
    # Many reason citations against many distinct named laws.
    names = ["צ" + hebrew.hebrew_numeral(value) for value in range(1, 3_201)]
    assert not hebrew.reason_names_source_bound_dependency(
        "חסר סעיף 345ב לחוק תתתתת. " * 3_200,
        "(ה1) " + "".join(f"לפי סעיף 345ב לחוק {name}. " for name in names),
        corpus_citation_path=CITATION,
    )
    # Quadratic forms of these took tens of seconds of CPU; the linear ones take
    # about two in all.
    assert time.process_time() - started < 5


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
