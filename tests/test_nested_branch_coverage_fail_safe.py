"""Coverage preserves baseline spans, validation and repeal successors."""

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

CITATION = "us/statute/26/21"
RATE_SOURCE = """(a) Administrative details.
D. Calculation:
(1) The applicable percentage determined under section 24 applies.
"""
REPEAL_SOURCE = """(a) Administrative terms.
(b) Calculation.
(1) Item.
(i) Details.
[(c) Repealed. [Effective 2024]]
(d) "benefit" means the amount payable.
"""


def _coverage_issues(source, *, status=None, citation=CITATION):
    module = {
        "summary": source,
        "source_verification": {"corpus_citation_path": citation},
    }
    if status:
        module["status"] = status
    content = yaml.safe_dump({"format": "rulespec/v1", "module": module, "rules": []})
    return vp.find_source_subparagraph_coverage_issues(
        content, source_texts={citation: source}
    )


def test_dotted_heading_cannot_hide_cited_rate_from_numeric_validator(tmp_path):
    repo = tmp_path / "us"
    rules_file = repo / "statutes" / "26" / "21.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {"summary": RATE_SOURCE},
                "rules": [
                    {
                        "name": "applicable_percentage",
                        "kind": "input",
                        "dtype": "Rate",
                    },
                    {
                        "name": "qualifying_expenses",
                        "kind": "input",
                        "dtype": "Money",
                    },
                    {
                        "name": "credit_amount",
                        "kind": "derived",
                        "dtype": "Money",
                        "source": "26 USC 21",
                        "versions": [
                            {
                                "effective_from": "2026-01-01",
                                "formula": "applicable_percentage * qualifying_expenses",
                            }
                        ],
                    },
                ],
            }
        )
    )
    pipeline = vp.ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=None,
        enable_oracles=False,
    )

    assert pipeline._extract_cross_reference_numeric_obligations(
        RATE_SOURCE, title="26"
    ) == [("applicable_percentage", "statutes/26/24", ("a",))]
    issues = pipeline._check_cross_reference_numeric_placeholders(rules_file)
    assert len(issues) == 1, issues
    assert "`credit_amount` uses local `applicable_percentage`" in issues[0]
    assert "statutes/26/24" in issues[0]


def test_annotated_repeal_keeps_genuine_successor_definition():
    issues = _coverage_issues(REPEAL_SOURCE)
    assert len(issues) == 1, issues
    assert "26 USC 21(d)" in issues[0]
    assert '"benefit" means the amount payable.' in issues[0]


@pytest.mark.parametrize(
    "notice",
    [
        "[(c) Repealed. [Effective 2024]]",
        "[(c) Repealed.] [Effective 2024]",
        "[(c) *Repealed*.]",
        "[(c) _Repealed_.]",
        "[(c) <i>Repealed.</i>]",
        "<i>[(c) Repealed.]</i>",
        "*[(c) Repealed.]*",
        "[(c) Repealed.]¹",
        "\u200b[(c) Repealed.]",
        "\t [(c) Omitted. [Effective 2024]] [See editorial note]",
        "Notice: [(c) Repealed.]",
        "EDITORIAL NOTE: [(c) Repealed.]",
        "[Paragraph (c) Repealed.]",
    ],
)
def test_formatted_notice_preserves_genuine_successor(notice):
    source = REPEAL_SOURCE.replace("[(c) Repealed. [Effective 2024]]", notice)
    issues = _coverage_issues(source)
    assert len(issues) == 1, issues
    assert "26 USC 21(d)" in issues[0]


@pytest.mark.parametrize(
    "notice, successor",
    [
        ("[(c) Repealed.]", "e"),
        ("[(c) Repealed—effective 2024.]", "d"),
        ("[(c) Repealed?]", "d"),
        ("[(c) <em class='editorial'>Repealed.</em>]", "d"),
        ("[(c) Omitted.]", "e"),
    ],
)
def test_uncertain_notice_cannot_prove_successor_is_nested(notice, successor):
    source = REPEAL_SOURCE.replace("[(c) Repealed. [Effective 2024]]", notice)
    source = source.replace('(d) "benefit"', f'({successor}) "benefit"')
    issues = _coverage_issues(source)
    assert len(issues) == 1, issues
    assert f"26 USC 21({successor})" in issues[0]


@pytest.mark.parametrize("status", ["deferred", "entity_not_supported"])
def test_editorial_notice_cannot_make_different_label_optional(status):
    source = '(c) Repealed.\n(d) "benefit" means the amount payable.\n'
    issues = _coverage_issues(source, status=status)
    assert len(issues) == 1, issues
    assert "26 USC 21(d)" in issues[0]


@pytest.mark.parametrize(
    "notice",
    [
        "(c) Repealed.",
        "[(c) Repealed. [Effective 2024]]",
        "[(c) <i>Repealed.</i>] [See editorial note]",
        "[(c) Omitted.]",
    ],
)
def test_editorial_notice_keeps_following_numbered_requirement(notice):
    source = (
        f'{notice}\n(d) "benefit" means the amount payable.\n'
        "(100) The amount is 100 dollars.\n"
    )
    issues = _coverage_issues(source, status="deferred")
    assert len(issues) == 1, issues
    assert "26 USC 21(d)" in issues[0]
    assert not vp._authoritative_source_slice_is_editorial_omission(source)
    content = yaml.safe_dump(
        {"format": "rulespec/v1", "module": {"status": "deferred"}, "rules": []}
    )
    analysis = sc.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=CITATION,
        test_cases=[],
        extract_numeric_occurrences=vp.extract_typed_numeric_inventory_occurrences_from_text,
        extract_numeric_grounding_occurrences=vp.extract_typed_numeric_occurrences_from_text,
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
    )
    assert any("numeric value 100" in issue for issue in analysis.issues), analysis


@pytest.mark.parametrize(
    "source",
    [
        '(c) Repealed. (d) "benefit" means the amount payable.',
        '(c) Repealed.\n[(d) "benefit" means the amount payable.]',
        "(c) Repealed.\n(100) The amount is 100 dollars.",
    ],
)
def test_notice_recognition_cannot_consume_other_labels(source):
    assert not vp._authoritative_source_slice_is_editorial_omission(source)


def test_scoped_child_keeps_cue_borrowed_from_uncertain_dotted_content():
    source = (
        "(5) Definitions:\n(c) Administrative details.\n"
        'D. "benefit" means the amount payable.\n'
    )
    citation = CITATION + "/5"
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": citation}},
            "rules": [],
        }
    )
    issues = vp.find_source_subparagraph_coverage_issues(
        content, source_texts={citation: source}
    )
    assert len(issues) == 1, issues
    assert "26 USC 21(c)" in issues[0]
    assert '"benefit" means the amount payable.' in issues[0]


@pytest.mark.parametrize("status", ["deferred", "entity_not_supported"])
def test_repeal_only_summary_cannot_hide_authoritative_successor(status):
    source = '(c) Repealed.\n(d) "benefit" means the amount payable.\n'
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {
                "status": status,
                "summary": "(c) Repealed.",
                "source_verification": {"corpus_citation_path": CITATION},
            },
            "rules": [],
        }
    )
    issues = vp.find_source_subparagraph_coverage_issues(
        content, source_texts={CITATION: source}
    )
    assert len(issues) == 1, issues
    assert "26 USC 21(d)" in issues[0]


def test_missing_alphabetic_sibling_cannot_prove_later_definition_is_nested():
    source = REPEAL_SOURCE.replace("[(c) Repealed. [Effective 2024]]\n", "")
    issues = _coverage_issues(source)
    assert len(issues) == 1, issues
    assert "26 USC 21(d)" in issues[0]


@pytest.mark.parametrize(
    "heading, scope", [("D.", "d"), ("II.", "ii"), ("Section 2.", "2")]
)
def test_isolated_dotted_heading_breaks_prior_numeric_ancestry(heading, scope):
    source = (
        f"(5) Prior numbered material.\n{heading} Definitions:\n"
        '(c) "benefit" means the amount payable.\n'
    )
    citation = CITATION + "/" + scope
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("c", '"benefit" means the amount payable.')
    ]
    assert len(_coverage_issues(source, citation=citation)) == 1


@pytest.mark.parametrize(
    "heading, scope",
    [
        ("6. Definitions:", "6"),
        ("6.1. Definitions:", "6.1"),
        ("6. Repealed.", "6"),
        ("**D.** Definitions:", "d"),
        ("*D.* Definitions:", "d"),
        ("<b>D.</b> Definitions:", "d"),
        ("[D. Definitions:]", "d"),
        ("[[D. Definitions:]]", "d"),
        ("\u200bD. Definitions:", "d"),
        ("\u2060D. Definitions:", "d"),
        ("Ｄ． Definitions:", "d"),
        ("Ⅵ. Definitions:", "vi"),
        ("§ 6. Definitions:", "6"),
        ("Section 6 Definitions:", "6"),
    ],
)
def test_possible_dotted_boundary_preserves_real_scoped_child(heading, scope):
    source = (
        f"(5) Prior numbered material.\n{heading}\n"
        '(c) "benefit" means the amount payable.\n'
    )
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("c", '"benefit" means the amount payable.')
    ]
    assert len(_coverage_issues(source, citation=CITATION + "/" + scope)) == 1


@pytest.mark.parametrize(
    "heading, scope",
    [
        ("## D. Definitions:", "d"),
        ("> D. Definitions:", "d"),
        ('"D." Definitions:', "d"),
        ("（6） Definitions:", "6"),
        ("(6] Definitions:", "6"),
        ("[6) Definitions:", "6"),
        ("(6 Definitions:", "6"),
    ],
)
def test_unaligned_heading_presentation_keeps_real_scoped_child(heading, scope):
    source = (
        f"(5) Prior numbered material.\n{heading}\n"
        '(c) "benefit" means the amount payable.\n'
    )
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("c", '"benefit" means the amount payable.')
    ]
    assert len(_coverage_issues(source, citation=CITATION + "/" + scope)) == 1


@pytest.mark.parametrize(
    "heading",
    ["2.", "*D.*", "<i>D.</i>", "Section 2", "（2）", "(2]", "> D."],
)
def test_possible_heading_cannot_prove_a_borrowed_cue_is_nested(heading):
    source = (
        "(a) Administrative details.\nD. Terms:\n"
        f"(1) Administrative item.\n{heading} New outline:\n"
        '"benefit" means the amount payable.\n'
    )
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("a", source.split("(a) ", 1)[1].strip())
    ]


def test_unrecognized_letter_notice_cannot_own_a_following_borrowed_cue():
    source = (
        "(a) Administrative details.\nD. Terms:\n"
        "(1) Administrative item.\n[(b) Repealed.]\n"
        '"benefit" means the amount payable.\n'
    )
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("a", source.split("(a) ", 1)[1].strip())
    ]


@pytest.mark.parametrize(
    "boundary",
    [
        "[(AA) Definitions:]",
        "[(6.1) Definitions:]",
        "[(<i>E</i>) Definitions:]",
        "（6） Definitions:",
    ],
)
def test_inline_unknown_marker_cannot_own_a_borrowed_cue(boundary):
    source = (
        "(a) Administrative details.\nD. Terms:\n"
        f"(1) Administrative item. {boundary}\n"
        '"benefit" means the amount payable.\n'
    )
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("a", source.split("(a) ", 1)[1].strip())
    ]


@pytest.mark.parametrize("cue_after_marker", [False, True])
def test_inline_uncertainty_retains_cue_before_or_after_the_marker(cue_after_marker):
    cue = '"benefit" means the amount payable.'
    body = f"[(AA) Definitions:] {cue}" if cue_after_marker else f"{cue} [(AA) Note.]"
    source = "(a) Administrative details.\nD. Terms:\n(1) " + body + "\n"
    expected = [("a", source.split("(a) ", 1)[1].strip())]
    assert vp._high_signal_top_level_subparagraphs(source) == expected


@pytest.mark.parametrize(
    "heading",
    [
        "6.Definitions:",
        "6. Definitions:",
        "**6.**Definitions:",
        "<i>6.</i>Definitions:",
        "６．Definitions:",
        "６． Definitions:",
    ],
)
@pytest.mark.parametrize("borrowed", [False, True])
def test_inline_numeric_dot_heading_keeps_uncertain_coverage(heading, borrowed):
    """Removing heading whitespace cannot prove a stale numeric owner."""
    if borrowed:
        source = (
            "(a) Administrative details.\nD. Terms:\n"
            f"(1) Administrative item. {heading}\n"
            '"benefit" means the amount payable.\n'
        )
        citation = CITATION
    else:
        source = (
            "(5) Prior numbered material.\n"
            f"(a) Details. {heading}\n"
            '(c) "benefit" means the amount payable.\n'
        )
        citation = CITATION + "/6"
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": citation}},
            "rules": [],
        }
    )
    issues = vp.find_source_subparagraph_coverage_issues(
        content, source_texts={citation: source}
    )
    assert len(issues) == 1, issues
    assert '"benefit" means the amount payable.' in issues[0]


@pytest.mark.parametrize("chain", ["(1)(A)", "(1) (A)"])
def test_recognized_attached_chain_retains_baseline_borrowed_cue(chain):
    source = (
        "(a) Administrative details.\nD. Terms:\n"
        f'{chain} "benefit" means the amount payable.\n'
    )
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("a", source.split("(a) ", 1)[1].strip())
    ]


@pytest.mark.parametrize(
    "heading",
    [
        "(AA) Definitions:",
        "[(D) Definitions:]",
        "(Β) Definitions:",
        "[(Ａ) Definitions:]",
    ],
)
def test_unproven_alphabetic_boundary_keeps_successor_obligation(heading):
    source = (
        f"(5) Prior numbered material.\n{heading}\n"
        '(c) "benefit" means the amount payable.\n'
    )
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("c", '"benefit" means the amount payable.')
    ]


@pytest.mark.parametrize("number", ["1", "1a"])
@pytest.mark.parametrize("scope_prefix", ["", "5/a/"])
def test_requested_numeric_reset_cannot_inherit_prior_parent(number, scope_prefix):
    source = (
        f"(5) Prior numbered material.\n(a) Details.\n({number}) Definitions:\n"
        '(c) "benefit" means the amount payable.\n'
    )
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("c", '"benefit" means the amount payable.')
    ]
    assert (
        len(_coverage_issues(source, citation=CITATION + "/" + scope_prefix + number))
        == 1
    )


@pytest.mark.parametrize(
    "notice",
    [
        "[(b) Repealed.]",
        "[(b) Repealed. [Effective 2024]]",
        "[(b) *Omitted*.]",
    ],
)
def test_authenticated_alphabetic_notice_retains_baseline_numeric_child_demand(notice):
    source = (
        f"(5) Definitions:\n(a) Details.\n{notice}\n"
        '(c) "benefit" means the amount payable.\n'
    )
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("c", '"benefit" means the amount payable.')
    ]
    assert len(_coverage_issues(source)) == 1


@pytest.mark.parametrize(
    "notice",
    [
        "(6) Repealed.",
        "[(6) Repealed.]",
        "[(6) Repealed. [Effective 2024]]",
        "[(6) *Omitted*.]",
        "[(6.1) Repealed.]",
        "[(6\u20137) Repealed.]",
        "[(<i>6</i>) Repealed.]",
    ],
)
def test_inline_numbered_notice_breaks_uncertain_parent_ancestry(notice):
    source = (
        f"(5) Definitions:\n(a) Details. {notice}\n"
        '(c) "benefit" means the amount payable.\n'
    )
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("c", '"benefit" means the amount payable.')
    ]


@pytest.mark.parametrize("number", ["\uff15", "\u0665"])
@pytest.mark.parametrize("prefix", ["", "D. Definitions:\n"])
def test_unfamiliar_numeric_typography_keeps_canonical_scoped_child(number, prefix):
    source = (
        prefix + f'({number}) Definitions:\n(c) "benefit" means the amount payable.\n'
    )
    citation = CITATION + ("/d" if prefix else "") + "/5"
    assert vp._high_signal_top_level_subparagraphs(source) == [
        ("c", '"benefit" means the amount payable.')
    ]
    assert len(_coverage_issues(source, citation=citation)) == 1


@pytest.mark.parametrize("notice_label", ["3", "4"])
def test_parent_fallback_notice_cannot_hide_a_real_requested_subtree(
    tmp_path, notice_label
):
    source = '(a) "benefit" means the amount payable.\n'
    if notice_label == "3":
        source += "(3) Additional details.\n"
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {
                "status": "deferred",
                "summary": (
                    f"({notice_label}) Repealed. "
                    "Pub. L. 98-21, title III, section 324(a)(3)(B)."
                ),
                "source_verification": {"corpus_citation_path": CITATION},
            },
            "rules": [],
        }
    )
    issues = vp.find_source_subparagraph_coverage_issues(
        content,
        rules_file=tmp_path / "us" / "statutes" / "26" / "21" / "a" / "3.yaml",
        source_texts={CITATION: source},
    )
    assert len(issues) == 1, issues
    assert "26 USC 21(a)" in issues[0]


@pytest.mark.parametrize(
    "case, citation, source",
    [
        (
            "scoped_inline_dotted",
            CITATION + "/d",
            "(5) Prior numbered material.\n(a) Details. D. Definitions:\n"
            '(c) "benefit" means the amount payable.\n',
        ),
        (
            "borrowed_inline_dotted",
            CITATION,
            "(a) Administrative details.\nD. Terms:\n"
            "(1) Administrative item. E. Definitions:\n"
            '"benefit" means the amount payable.\n',
        ),
        (
            "numeric_colon_boundary",
            CITATION + "/6",
            "(5) Prior numbered material.\n6: Definitions:\n"
            '(c) "benefit" means the amount payable.\n',
        ),
    ],
)
def test_inline_letter_and_numeric_colon_boundaries_keep_coverage(
    case, citation, source
):
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": citation}},
            "rules": [],
        }
    )
    issues = vp.find_source_subparagraph_coverage_issues(
        content, source_texts={citation: source}
    )
    assert len(issues) == 1, (case, issues)
    assert '"benefit" means the amount payable.' in issues[0]


@pytest.mark.parametrize(
    "heading", ["E.", "**E.**", "<i>E.</i>", "Ｅ．", "6:", "６：", "Section 6", "§ 6"]
)
@pytest.mark.parametrize("cue_after_heading", [False, True])
def test_inline_heading_uncertainty_retains_cue_in_either_physical_order(
    heading, cue_after_heading
):
    cue = '"benefit" means the amount payable.'
    body = (
        f"{heading} Definitions: {cue}"
        if cue_after_heading
        else f"{cue} {heading} Note."
    )
    source = "(a) Administrative details.\nD. Terms:\n(1) " + body + "\n"
    expected = [("a", source.split("(a) ", 1)[1].strip())]
    assert vp._high_signal_top_level_subparagraphs(source) == expected
