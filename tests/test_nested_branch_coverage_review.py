"""Source obligations remain; authoritative repeal evidence can only refuse."""

from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import validator_pipeline as vp


def _coverage_issues(source, citation):
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": citation}},
            "rules": [],
        }
    )
    return vp.find_source_subparagraph_coverage_issues(
        content, source_texts={citation: source}
    )


def _assert_missing_obligation(source, citation, label):
    issues = _coverage_issues(source, citation)
    assert len(issues) == 1, issues
    assert f"({label})" in issues[0]
    assert "benefit" in issues[0]


@pytest.mark.parametrize(
    "citation",
    [
        "us/regulation/26/1.36B-2/5",
        "us-co/regulation/10-ccr-2506-1/4.207.2/5",
    ],
)
def test_supported_short_regulation_scopes_keep_their_direct_child(citation):
    """Known section layouts must identify the numeric unit's letter child."""
    source = '(5) Definitions:\n(c) "benefit" means the amount payable.\n'
    _assert_missing_obligation(source, citation, "c")


def test_unknown_regulation_scope_conservatively_retains_obligation():
    """An unidentified section root cannot prove that real content is nested."""
    source = '(5) Definitions:\n(c) "benefit" means the amount payable.\n'
    _assert_missing_obligation(source, "us-or/regulation/example/section/5", "c")


@given(
    citation=st.sampled_from(
        [
            "us/regulation/26/1.36B-2",
            "us-co/regulation/10-ccr-2506-1/4.207.2",
            "us-or/regulation/example/section",
        ]
    ),
    number=st.integers(min_value=1, max_value=20),
    letter=st.sampled_from(list("abcf")),
    cue=st.sampled_from(["means", "is eligible for", "is ineligible for"]),
)
@settings(max_examples=50, deadline=None)
def test_known_or_uncertain_regulation_scopes_keep_real_children(
    citation, number, letter, cue
):
    source = (
        f"({number}) Definitions:\n({letter}) A benefit {cue} the amount payable.\n"
    )
    _assert_missing_obligation(source, f"{citation}/{number}", letter)


def test_repeated_combined_heading_keeps_leading_root_obligation():
    """Omitted repeated ancestors must not turn the leading root into a child."""
    source = (
        "(a) Administrative details.\n"
        '(a) (1) "benefit" means the amount payable.\n'
        "(b) Effective date.\n"
    )
    _assert_missing_obligation(source, "us/statute/26/21", "a")


@given(
    letter=st.sampled_from(list("abcf")),
    number=st.integers(min_value=1, max_value=20),
    cue=st.sampled_from(["means", "is eligible for", "is ineligible for"]),
    attached_gap=st.sampled_from(["", " ", "\t"]),
)
@settings(max_examples=50, deadline=None)
def test_repeated_combined_headings_preserve_signal_in_the_leading_root(
    letter, number, cue, attached_gap
):
    successor = chr(ord(letter) + 1)
    source = (
        f"({letter}) Administrative details.\n"
        f"({letter}) ({number}){attached_gap}(A) A benefit {cue} the amount payable.\n"
        f"({successor}) Effective date.\n"
    )
    _assert_missing_obligation(source, "us/statute/26/21", letter)


def _repeal_source(repealed, successor, whitespace, inside_gap, newline, disposition):
    return newline.join(
        [
            "(a) Administrative terms.",
            "(b) Calculation.",
            "(1) Item.",
            "(i) Details.",
            f"{whitespace}[{inside_gap}({repealed}) {disposition}.{inside_gap}]",
            f'({successor}) "benefit" means the amount payable.',
            "",
        ]
    )


@pytest.mark.parametrize(
    "whitespace, inside_gap, newline",
    [
        ("    ", "", "\n"),
        ("\t", "", "\n"),
        (" \t ", " ", "\n"),
        ("    ", "\t", "\r\n"),
    ],
)
def test_repeal_whitespace_keeps_genuine_alphabetic_successor(
    whitespace, inside_gap, newline
):
    """A bracketed repeal's extraction whitespace cannot discard its successor."""
    source = _repeal_source("c", "d", whitespace, inside_gap, newline, "Repealed")
    _assert_missing_obligation(source, "us/statute/26/21", "d")


@pytest.mark.parametrize("inside_gap", ["\n    ", "\r\n\t", "\n\u00a0"])
def test_wrapped_repeal_whitespace_keeps_genuine_alphabetic_successor(inside_gap):
    """Extraction line breaks inside the bracket retain the real successor."""
    source = _repeal_source("c", "d", "    ", inside_gap, "\n", "Repealed")
    _assert_missing_obligation(source, "us/statute/26/21", "d")


@given(
    labels=st.sampled_from([("c", "d"), ("k", "l")]),
    whitespace=st.sampled_from(["", " ", "    ", "\t", " \t ", "\u00a0"]),
    inside_gap=st.sampled_from(["", " ", "\t", "\n    ", "\r\n\t", "\n\u00a0"]),
    newline=st.sampled_from(["\n", "\r\n"]),
    disposition=st.sampled_from(["Repealed", "Omitted"]),
)
@settings(max_examples=50, deadline=None)
def test_bracketed_repeal_whitespace_never_drops_a_real_successor(
    labels, whitespace, inside_gap, newline, disposition
):
    repealed, successor = labels
    source = _repeal_source(
        repealed, successor, whitespace, inside_gap, newline, disposition
    )
    _assert_missing_obligation(source, "us/statute/26/21", successor)


def _parent_fallback_coverage_issues(
    source, summary, *, number=3, status="deferred", scope_method="rules_file"
):
    citation = "us/statute/26/21"
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {
                "status": status,
                "summary": summary,
                "source_verification": {"corpus_citation_path": citation},
            },
            "rules": [],
        }
    )
    scope = (
        {"rules_file": Path("statutes/26/21/a") / f"{number}.yaml"}
        if scope_method == "rules_file"
        else {"requested_source": f"{citation}/a/{number}"}
    )
    return vp.find_source_subparagraph_coverage_issues(
        content, source_texts={citation: source}, **scope
    )


@pytest.mark.parametrize("heading", ["(<i>3</i>)", "(３)", "(<sup>3</sup>)", "(³)"])
@pytest.mark.parametrize("scope_method", ["rules_file", "requested_source"])
def test_parent_fallback_repeal_summary_preserves_formatted_authoritative_subtree(
    heading, scope_method
):
    """A generated repeal cannot waive a formatted, operative source heading."""
    source = (
        f'(a) Administrative details.\n{heading} "benefit" means the amount payable.\n'
    )
    issues = _parent_fallback_coverage_issues(
        source, "(3) Repealed.", scope_method=scope_method
    )
    assert len(issues) == 1, issues
    assert "26 USC 21(a)" in issues[0]
    assert "benefit" in issues[0]
    assert issues == _parent_fallback_coverage_issues(
        source, "Administrative details.", scope_method=scope_method
    )


@pytest.mark.parametrize("authoritative_notice", ["", "(4) Repealed.\n"])
def test_parent_fallback_repeal_summary_requires_matching_source_evidence(
    authoritative_notice,
):
    """An absent branch or another branch's repeal does not prove a repeal."""
    source = '(a) "benefit" means the amount payable.\n' + authoritative_notice
    issues = _parent_fallback_coverage_issues(source, "(3) Repealed.")
    assert len(issues) == 1, issues
    assert "26 USC 21(a)" in issues[0]
    assert "benefit" in issues[0]


@pytest.mark.parametrize(
    "heading", ["(3)", "(<i>3</i>)", "(３)", "(<sup>3</sup>)", "(³)"]
)
@pytest.mark.parametrize("scope_method", ["rules_file", "requested_source"])
def test_parent_fallback_accepts_exact_authoritative_requested_repeal(
    heading, scope_method
):
    """An authenticated requested child may be repealed inside a parent slice."""
    source = (
        '(a) "benefit" means the amount payable.\n'
        f"{heading} Repealed.\n"
        "(b) Effective date.\n"
    )
    assert (
        _parent_fallback_coverage_issues(
            source, "(3) Repealed.", scope_method=scope_method
        )
        == []
    )


@pytest.mark.parametrize("heading", ["(3)", "(<i>3</i>)", "(３)"])
@pytest.mark.parametrize("scope_method", ["rules_file", "requested_source"])
def test_parent_fallback_repeal_in_another_parent_cannot_waive_coverage(
    heading, scope_method
):
    """The same child label under a different parent is not corroboration."""
    source = (
        '(a) "benefit" means the amount payable.\n'
        "(b) Administrative details.\n"
        f"{heading} Repealed.\n"
    )
    issues = _parent_fallback_coverage_issues(
        source, "(3) Repealed.", scope_method=scope_method
    )
    assert len(issues) == 1, issues
    assert "26 USC 21(a)" in issues[0]
    assert "benefit" in issues[0]


@pytest.mark.parametrize(
    "heading", ["(3)", "(<i>3</i>)", "(３)", "(<sup>3</sup>)", "(³)"]
)
@pytest.mark.parametrize("repeal_first", [False, True])
@pytest.mark.parametrize("scope_method", ["rules_file", "requested_source"])
def test_parent_fallback_conflicting_duplicate_heading_cannot_authenticate_repeal(
    heading, repeal_first, scope_method
):
    """A repeal cannot override operative text at the same requested path."""
    notices = [
        f'{heading} "benefit" means the amount payable.\n',
        "(3) Repealed.\n",
    ]
    if repeal_first:
        notices.reverse()
    source = (
        '(a) "benefit" means the amount payable.\n'
        + "".join(notices)
        + "(b) Effective date.\n"
    )
    issues = _parent_fallback_coverage_issues(
        source, "(3) Repealed.", scope_method=scope_method
    )
    assert len(issues) == 1, issues
    assert "26 USC 21(a)" in issues[0]
    assert "benefit" in issues[0]


@pytest.mark.parametrize(
    "boundary",
    [
        "D. Definitions:",
        "6: Definitions:",
        "<i>D.</i> Definitions:",
        "Ｄ． Definitions:",
    ],
)
@pytest.mark.parametrize("scope_method", ["rules_file", "requested_source"])
def test_deep_parent_fallback_repeal_requires_proven_outer_ancestry(
    boundary, scope_method
):
    """A physical immediate parent cannot authenticate an uncertain outer root."""
    citation = "us/statute/26/21"
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {
                "status": "deferred",
                "summary": "(i) Repealed.",
                "source_verification": {"corpus_citation_path": citation},
            },
            "rules": [],
        }
    )
    scope = (
        {"rules_file": Path("statutes/26/21/a/2/i.yaml")}
        if scope_method == "rules_file"
        else {"requested_source": f"{citation}/a/2/i"}
    )
    source = (
        '(a) "benefit" means the amount payable.\n'
        f"{boundary}\n"
        "(2) Administrative details.\n"
        "(i) Repealed.\n"
        "(b) Effective date.\n"
    )
    issues = vp.find_source_subparagraph_coverage_issues(
        content, source_texts={citation: source}, **scope
    )
    assert len(issues) == 1, issues
    assert "26 USC 21(a)" in issues[0]
    assert "benefit" in issues[0]
    assert (
        vp.find_source_subparagraph_coverage_issues(
            content,
            source_texts={citation: source.replace(f"{boundary}\n", "")},
            **scope,
        )
        == []
    )


@given(
    number=st.integers(min_value=1, max_value=20),
    presentation=st.sampled_from(
        ["plain", "i", "em", "b", "strong", "sup", "fullwidth", "superscript"]
    ),
    status=st.sampled_from(["deferred", "entity_not_supported"]),
    scope_method=st.sampled_from(["rules_file", "requested_source"]),
    cue=st.sampled_from(["means", "is eligible for", "is ineligible for"]),
    newline=st.sampled_from(["\n", "\r\n"]),
)
@settings(max_examples=100, deadline=None)
def test_authoritative_heading_presentation_never_authenticates_generated_repeal(
    number, presentation, status, scope_method, cue, newline
):
    """Presentation changes cannot turn an operative branch into a tombstone."""
    if presentation == "plain":
        heading_number = str(number)
    elif presentation == "fullwidth":
        heading_number = "".join(chr(ord(char) + 0xFEE0) for char in str(number))
    elif presentation == "superscript":
        heading_number = str(number).translate(
            str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        )
    else:
        heading_number = f"<{presentation}>{number}</{presentation}>"
    source = (
        f"(a) Administrative details.{newline}"
        f"({heading_number}) A benefit {cue} the amount payable.{newline}"
    )
    issues = _parent_fallback_coverage_issues(
        source,
        f"({number}) Repealed.",
        number=number,
        status=status,
        scope_method=scope_method,
    )
    assert len(issues) == 1, issues
    assert "26 USC 21(a)" in issues[0]
    assert "benefit" in issues[0]
