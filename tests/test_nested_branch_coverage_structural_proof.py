"""Adversarial labels retain baseline demands; repeal waivers only narrow."""

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import validator_pipeline as vp

CITATION = "us/statute/26/21"
CUE = '"benefit" means the amount payable.'


def _coverage_issues(source, *, citation=CITATION, requested_source=None, summary=""):
    module = {
        "source_verification": {"corpus_citation_path": citation},
        "summary": summary,
    }
    if summary:
        module["status"] = "deferred"
    content = yaml.safe_dump({"format": "rulespec/v1", "module": module, "rules": []})
    return vp.find_source_subparagraph_coverage_issues(
        content,
        source_texts={citation: source},
        requested_source=requested_source,
    )


@pytest.mark.parametrize("heading", ["§6", "Sec.6", "IV:", "6—", "6)"])
@pytest.mark.parametrize("kind", ["direct", "borrowed", "waiver"])
def test_uncertain_boundary_preserves_obligation(heading, kind):
    if kind == "direct":
        source = (
            "(5) Prior numbered material.\n"
            f"(a) Details. {heading} Definitions:\n(c) {CUE}\n"
        )
        citation, requested_source, summary = CITATION + "/6", None, ""
        expected_label = "c"
    elif kind == "borrowed":
        source = (
            "(a) Administrative details.\nD. Terms:\n"
            f"(1) Administrative item. {heading} Definitions:\n{CUE}\n"
        )
        citation, requested_source, summary = CITATION, None, ""
        expected_label = "a"
    else:
        source = (
            f"(a) {CUE}\n{heading} Definitions:\n"
            "(2) Administrative details.\n(i) Repealed.\n(b) Effective date.\n"
        )
        citation = CITATION
        requested_source, summary = CITATION + "/a/2/i", "(i) Repealed."
        expected_label = "a"

    issues = _coverage_issues(
        source,
        citation=citation,
        requested_source=requested_source,
        summary=summary,
    )
    assert len(issues) == 1, (heading, kind, issues)
    assert f"26 USC 21({expected_label})" in issues[0]
    assert CUE in issues[0]


@pytest.mark.parametrize(
    "heading",
    ["§ 6", "Sec. 6", "IV.", "6:", "(a)", "— E.", "• E.", '"E."', "（a）"],
)
def test_recognized_uncertainty_control(heading):
    source = (
        "(a) Administrative details.\nD. Terms:\n"
        f"(1) Administrative item. {heading} Definitions:\n{CUE}\n"
    )
    issues = _coverage_issues(source)
    assert len(issues) == 1, issues
    assert "26 USC 21(a)" in issues[0]
    assert CUE in issues[0]


@pytest.mark.parametrize("disposition", ["Reserved", "Expired"])
def test_other_dispositions_are_conservative(disposition):
    assert not vp._authoritative_source_slice_is_editorial_omission(
        f"(3) {disposition}."
    )


@pytest.mark.parametrize("label", ["a", "2", "i"])
@pytest.mark.parametrize("uncertainty", ["<§6>", "<IV:>"])
def test_unfamiliar_markup_inside_authoritative_heading_cannot_authenticate_waiver(
    label, uncertainty
):
    headings = {item: f"({item})" for item in ("a", "2", "i")}
    headings[label] = f"({label}{uncertainty})"
    source = (
        f"{headings['a']} {CUE}\n{headings['2']} Administrative details.\n"
        f"{headings['i']} Repealed.\n(b) Effective date.\n"
    )
    assert not vp._source_scope_is_editorial_omission(source, ("a", "2", "i"))


def _roman_numeral(number):
    result = []
    for value, token in (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ):
        count, number = divmod(number, value)
        result.append(token * count)
    return "".join(result)


NUMERIC_LABELS = st.lists(
    st.integers(min_value=0, max_value=999999), min_size=1, max_size=3
).map(lambda parts: ".".join(map(str, parts)))
ROMAN_LABELS = st.integers(min_value=1, max_value=3999).map(_roman_numeral)
LETTER_LABELS = st.one_of(
    st.characters(categories=("Lu", "Ll", "Lt", "Lo")),
    st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=2, max_size=4),
)
LABEL_TOKENS = st.one_of(
    NUMERIC_LABELS, ROMAN_LABELS, ROMAN_LABELS.map(str.lower), LETTER_LABELS
)
PRESENTATION_GAPS = st.text(alphabet=" \t\u00a0\u200b\u2060", max_size=4)


@st.composite
def structural_label_presentations(draw):
    """Construct labels from independent token, delimiter and typography choices.

    Bare labels are physically line-leading. Other candidates occur after prose
    on a recognized heading's line, where an isolated label cannot establish a
    new ancestor. The generator's truth is the insertion position, not a call
    to the production structure recognizer or boundary scanner.
    """
    family = draw(
        st.sampled_from(
            [
                "delimited",
                "opening",
                "section",
                "bullet",
                "bare",
                "symbol",
                "word",
                "title",
                "closing",
            ]
        )
    )
    token = draw(LABEL_TOKENS)
    gap = draw(PRESENTATION_GAPS)
    if family == "delimited":
        delimiter = draw(st.sampled_from([".", ")", ":", "—", "–", "-"]))
        label = token + gap + delimiter
    elif family == "opening":
        opening = draw(st.sampled_from(["(", "（", "❨", "⦅", "﹙"]))
        closing = draw(st.sampled_from(["", ")", "）", "]", "❩", "⦆", "﹚"]))
        label = opening + gap + token + gap + closing
    elif family == "section":
        section = draw(st.sampled_from(["§", "§§", "Sec", "Section"]))
        if section.isalpha():
            section = "".join(
                draw(st.sampled_from([char.lower(), char.upper()])) for char in section
            )
            section += draw(st.sampled_from(["", "."]))
        label = section + gap + draw(st.one_of(st.just(""), LABEL_TOKENS))
    elif family == "bullet":
        bullet = draw(
            st.sampled_from(["•", "‣", "◦", "⁃", "∙", "●", "▪", "-", "–", "—", "*"])
        )
        label = bullet + gap + draw(st.one_of(st.just(""), LABEL_TOKENS))
    elif family == "symbol":
        label = draw(st.characters(categories=("So", "Sm", "Sc", "Sk")))
    elif family == "word":
        label = draw(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=10)
        ).title()
        label += draw(st.sampled_from([".", ")", ":", "—", "–", "-"]))
    elif family == "title":
        label = " ".join(
            draw(
                st.lists(
                    st.text(
                        alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=10
                    ).map(str.title),
                    min_size=2,
                    max_size=4,
                )
            )
        )
        label += draw(st.sampled_from(["", ".", "?", "!", ":"]))
    elif family == "closing":
        label = (
            draw(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=12)
            )
            + ")"
        )
    else:
        label = token

    if draw(st.booleans()):
        # Fullwidth presentation has the same uncertainty as the base token.
        label = "".join(
            chr(ord(char) + 0xFEE0) if "!" <= char <= "~" else char for char in label
        )
    wrappers = draw(
        st.lists(
            st.sampled_from(
                [
                    ("*", "*"),
                    ("**", "**"),
                    ("_", "_"),
                    ("<b>", "</b>"),
                    ("<em>", "</em>"),
                    ("[", "]"),
                    ("[[", "]]"),
                    ("<", ">"),
                    ('"', '"'),
                    ("\u200b", ""),
                    ("\u2060", ""),
                ]
            ),
            max_size=2,
        )
    )
    for prefix, suffix in wrappers:
        label = prefix + label + suffix
    if family in {"bare", "symbol", "word", "title"}:
        label = draw(PRESENTATION_GAPS) + label
    return label, family in {"bare", "symbol", "word", "title"}


def _inserted_heading(label, line_leading):
    return "\n" + label + "\n" if line_leading else " " + label + " Definitions:\n"


@given(presentation=structural_label_presentations())
@example(presentation=("§6", False))
@example(presentation=("Sec.6", False))
@example(presentation=("IV:", False))
@example(presentation=("6—", False))
@example(presentation=("6)", False))
@example(presentation=('"K"', True))
@example(presentation=("SecIV", False))
@example(presentation=("-0", False))
@example(presentation=("㍑", True))
@example(presentation=("⁶:", True))
@example(presentation=("<§6>", False))
@example(presentation=("<IV:>", False))
@example(presentation=("*Definitions:", False))
@example(presentation=("ﬀ:", False))
@example(presentation=('"-0"', False))
@example(presentation=("[ ]", False))
@example(presentation=("Chapter.", True))
@example(presentation=("\u00a0K", True))
@example(presentation=('"\u00a0Chapter."', True))
@example(presentation=("Annex B", True))
@example(presentation=("Appendix IV", True))
@example(presentation=("aa)", False))
@example(presentation=("General Definitions.", True))
@example(presentation=("\u2028Annex B", False))
@example(presentation=("\u2029Appendix IV", False))
@settings(max_examples=500, deadline=None, derandomize=True)
def test_generated_structural_uncertainty_keeps_demands_and_narrows_waiver(
    presentation,
):
    """Unseen presentations cannot remove demands or widen main's waiver."""
    label, line_leading = presentation
    inserted = _inserted_heading(label, line_leading)
    direct = "(5) Prior numbered material.\n(a) Details." + inserted + f"(c) {CUE}\n"
    direct_issues = _coverage_issues(direct, citation=CITATION + "/6")
    assert len(direct_issues) == 1, (presentation, direct_issues)
    assert "26 USC 21(c)" in direct_issues[0]
    assert CUE in direct_issues[0]

    borrowed = (
        "(a) Administrative details.\nD. Terms:\n(1) Administrative item."
        + inserted
        + CUE
        + "\n"
    )
    borrowed_issues = _coverage_issues(borrowed)
    assert len(borrowed_issues) == 1, (presentation, borrowed_issues)
    assert "26 USC 21(a)" in borrowed_issues[0]
    assert CUE in dict(vp._high_signal_top_level_subparagraphs(borrowed))["a"]

    repeal = (
        f"(a) {CUE}"
        + inserted
        + "(2) Administrative details.\n(i) Repealed.\n(b) Effective date.\n"
    )
    repeal_issues = _coverage_issues(
        repeal, requested_source=CITATION + "/a/2/i", summary="(i) Repealed."
    )
    baseline_demands = _coverage_issues(repeal, requested_source=CITATION + "/a/2/i")
    assert len(baseline_demands) == 1, (presentation, baseline_demands)
    assert "26 USC 21(a)" in baseline_demands[0]
    assert CUE in baseline_demands[0]
    assert repeal_issues in ([], baseline_demands), (presentation, repeal_issues)
    if not repeal_issues:
        assert vp._source_slice_is_editorial_omission("(i) Repealed.")


@given(presentation=structural_label_presentations(), continuation=st.booleans())
@settings(max_examples=500, deadline=None, derandomize=True)
def test_generated_later_boundary_preserves_a_preceding_baseline_cue(
    presentation, continuation
):
    label, line_leading = presentation
    body = "(1) Administrative item.\n" if continuation else "(1) "
    source = (
        "(a) Administrative details.\nD. Terms:\n"
        + body
        + CUE
        + _inserted_heading(label, line_leading)
    )
    issues = _coverage_issues(source)
    assert len(issues) == 1, (presentation, issues)
    assert "26 USC 21(a)" in issues[0]
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("a", source.split("(a) ", 1)[1].strip())
    ]


@pytest.mark.parametrize(
    "prose",
    [
        "Details.",
        "Item.",
        "The supporting discussion continues here.",
        "This text provides further detail without changing the outline.",
        "Additional administrative information follows in ordinary prose.",
    ],
)
def test_ordinary_prose_preserves_baseline_demands_and_matching_repeal(prose):
    # Short fragments are positively prose within a recognized heading body;
    # as standalone dotted lines, they could themselves be structural labels.
    separator = " " if prose in {"Details.", "Item."} else "\n"
    direct = f"(5) Definitions:\n(a) Details.{separator}{prose}\n(c) {CUE}\n"
    direct_issues = _coverage_issues(direct)
    assert len(direct_issues) == 1, direct_issues
    assert "26 USC 21(c)" in direct_issues[0]
    borrowed = (
        "(a) Administrative details.\nD. Terms:\n"
        f"(1) Administrative item.{separator}{prose}\n{CUE}\n"
    )
    borrowed_issues = _coverage_issues(borrowed)
    assert len(borrowed_issues) == 1, borrowed_issues
    assert "26 USC 21(a)" in borrowed_issues[0]
    repeal = (
        f"(a) {CUE}{separator}{prose}\n(2) Administrative details.{separator}{prose}\n"
        "(i) Repealed.\n(b) Effective date.\n"
    )
    assert vp._source_scope_is_editorial_omission(repeal, ("a", "2", "i"))
    assert (
        _coverage_issues(
            repeal, requested_source=CITATION + "/a/2/i", summary="(i) Repealed."
        )
        == []
    )


@pytest.mark.parametrize(
    "heading",
    ["Article 6 Definitions:", "Annex B Definitions:", "General Definitions."],
)
@pytest.mark.parametrize("kind", ["direct", "borrowed", "waiver"])
def test_inline_title_preserves_baseline_demands_and_does_not_widen_waiver(
    heading, kind
):
    if kind == "direct":
        source = f"(5) Prior material.\n(a) Details. {heading}\n(c) {CUE}\n"
        expected = [("c", CUE)]
    elif kind == "borrowed":
        source = f"(a) Administrative details.\nD. Terms:\n(1) Item. {heading}\n{CUE}\n"
        expected = [("a", source.split("(a) ", 1)[1].strip())]
    else:
        source = (
            f"(a) {CUE} {heading}\n(2) Administrative details.\n"
            "(i) Repealed.\n(b) Effective date.\n"
        )
        expected = [("a", source.split("(a) ", 1)[1].split("\n(b) ", 1)[0])]
    assert vp._high_signal_top_level_subparagraphs(source) == expected
    issues = _coverage_issues(source)
    assert len(issues) == 1, issues
    if kind == "waiver":
        waived = _coverage_issues(
            source, requested_source=CITATION + "/a/2/i", summary="(i) Repealed."
        )
        assert waived in ([], issues), waived
        if not waived:
            assert vp._source_slice_is_editorial_omission("(i) Repealed.")


@pytest.mark.parametrize("unknown_label", ["AA", "foo"])
def test_unknown_attached_chain_retains_demand_and_does_not_widen_waiver(
    unknown_label,
):
    source = (
        f"(a) {CUE}\n(2) ({unknown_label}) Administrative details.\n"
        "(i) Repealed.\n(b) Effective date.\n"
    )
    expected = [("a", source.split("(a) ", 1)[1].split("\n(b) ", 1)[0])]
    assert vp._high_signal_top_level_subparagraphs(source) == expected
    demands = _coverage_issues(source, requested_source=CITATION + "/a/2/i")
    assert len(demands) == 1, demands
    issues = _coverage_issues(
        source, requested_source=CITATION + "/a/2/i", summary="(i) Repealed."
    )
    assert issues in ([], demands), issues
    if not issues:
        assert vp._source_slice_is_editorial_omission("(i) Repealed.")
