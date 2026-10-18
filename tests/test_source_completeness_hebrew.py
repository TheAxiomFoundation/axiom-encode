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
from hypothesis import given, settings
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
        "אינו מקודד",
        "אינו זמין",
        "אינה זמינה",
        "אינם זמינים",
        "אינן זמינות",
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
    ),
)
def test_hebrew_word_boundaries_reject_lookalikes(text):
    assert not completeness_module._MISSING_DEPENDENCY_LANGUAGE.search(text)


@pytest.mark.parametrize(
    "pattern",
    (
        hebrew.MISSING_DEPENDENCY_LANGUAGE,
        hebrew.ADVERSATIVE_LANGUAGE,
        hebrew.RUNTIME_GAP_LANGUAGE,
        hebrew.ADMINISTRATIVE_ARTIFACT_LANGUAGE,
        hebrew.ADMINISTRATIVE_ACTION_LANGUAGE,
        hebrew.EDITORIAL_OMISSION_WORDS,
        hebrew._SIGNAL.pattern,
        hebrew._EXPLICIT_COMPUTATION.pattern,
    ),
)
def test_vocabulary_writes_final_letters_only_at_word_end(pattern):
    """`זמין(?:ה)?` can never match `זמינה`: a final letter followed by more letters."""

    assert not re.search(r"[ךםןףץ](?:\(\?:)?[א-ת]", pattern)


@pytest.mark.parametrize(
    "text", ("יפורסמו", "פורסמו", "יאושרו", "הוגשו", "ייקבעו", "יודיעו")
)
def test_hebrew_plural_administrative_actions(text):
    assert completeness_module._ADMINISTRATIVE_SOURCE_ACTION_LANGUAGE.search(text)


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


def _accepted(branch: str, reason: str) -> bool:
    issues = _analyze(_deferral_content(branch, reason))
    path = tuple(branch.split("/"))
    return _deferral_issue(issues, 0) is None and path not in _structure_paths(issues)


@pytest.mark.parametrize(
    "reason",
    (
        "סעיף 14(ה1) מחייב תשלום לפי הוראות סעיף 345ב לחוק הביטוח הלאומי בשינויים "
        "המחויבים; חסרה הוראת החישוב המדויקת שבסעיף 345ב.",
        "חסרות הוראות סעיף 345ב לחוק הביטוח הלאומי.",
        "החישוב תלוי בסעיף 345ב טרם קודד.",
        "לא ניתן לחשב את התשלום עד שיקודד סעיף 345ב.",
        "לא ניתן לחשב את התשלום עד שסעיף 345ב יקודד.",
        "לא ניתן לחשב את התשלום כי סעיף 345ב טרם קודד.",
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


def test_hebrew_administrative_runtime_gap_on_exact_branch_is_precise():
    """Counterpart of `test_exact_canonical_state_branch_can_name_source_bound_runtime_gap`."""

    assert _accepted(
        "c",
        "סעיף 14(ג) לא ניתן לקודד: חסרה יכולת לייצג את הגשת העתק הדוח השנתי של המבוטח.",
    )


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


def _english_accepted(branch: str, reason: str) -> bool:
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
        ENGLISH_PRECISION_SOURCE,
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
        "both",
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
        "both",
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
        "לא ניתן לחשב עד שיקודדו התנאים הקבועים בסעיף 345ב.",
        "Cannot be computed until the conditions set in section 345b are encoded.",
        "both",
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
        "both",
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


@pytest.mark.parametrize(
    "reason",
    (
        # The said law is the National Insurance Law, named earlier in the branch.
        "חסרה הוראת החישוב שבסעיף 345ב לחוק הביטוח הלאומי.",
        "חסרה הוראת החישוב שבסעיף 345ב לחוק הביטוח.",
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
    ("phrase", "bounded"),
    (
        ("התנאים", True),
        ("את הוראות החישוב", True),
        ("סכום ההכנסה השנתית", True),
        ("הענף המיוחד הזה", False),
        ("כל הדברים", False),
        ("התנאים 2025", False),
        ("", False),
    ),
)
def test_hebrew_subject_phrases_use_the_english_vocabulary(phrase, bounded):
    assert hebrew._subject_phrase_is_bounded(phrase) is bounded


@pytest.mark.parametrize(
    ("reason", "accepted"),
    (
        ("החישוב תלוי בסעיף 345ב טרם קודד.", True),
        ("לא ניתן לחשב כי החישוב תלוי בסעיף 345ב טרם קודד.", True),
        ("הענף המיוחד הזה תלוי בסעיף 345ב טרם קודד.", False),
        ("לא ניתן לחשב עד שיקודדו התנאים שבסעיף 345ב.", True),
        ("לא ניתן לחשב עד שיקודדו כל הדברים המעניינים שבסעיף 345ב.", False),
        (
            "il/statute/national-health-insurance-law-1994/section-14(e1): "
            "חסרה הוראת סעיף 345ב.",
            False,
        ),
    ),
)
def test_requires_and_until_introductions_are_bounded(reason, accepted):
    assert _accepted("e1", reason) is accepted


# --- Rule `source:` citations ------------------------------------------------


@pytest.mark.parametrize(
    "reference",
    (
        "סעיף 14(ב)(2) לחוק הביטוח הלאומי",
        "סעיף 14(ב) (2) לחוק הביטוח הלאומי",
        "סעיף 14(ב)(2), לחוק הביטוח הלאומי",
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
    started = time.perf_counter()
    assert pattern.search("סעיף 14(ג)" + " " * 64_000 + "x")
    many = "חסר " + "סעיף 346 " * 8_000
    assert not hebrew.reason_names_source_bound_dependency(
        many, PRECISION_SOURCE, corpus_citation_path=CITATION
    )
    # Quadratic forms of these took about 15 s and 40 s; linear ones take
    # milliseconds, so the bound holds on a loaded host.
    assert time.perf_counter() - started < 5


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
