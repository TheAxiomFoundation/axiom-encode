"""Public NZ law excerpts retain legal rows without losing operative numbers.

Excerpts: Social Security Act 2018 (2018/32); amending Act 2026/54, section 15.
Source: New Zealand Legislation (legislation.govt.nz), Crown copyright, CC BY 4.0
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.bare_structure import (
    bare_definition_headers,
    bare_outline,
    mask_bare_labels,
    table_outline,
)
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

S73 = """1
A person entitled to a winter energy payment may make an election not to receive the payment.
2
If the payment would be payable at a rate prescribed in respect of a person who is in a relationship, MSD must be satisfied that the election has been made by both spouses or partners.
3
The election must be made, and may be revoked, in a form and manner approved by MSD.
4
Unless it is revoked, the election stops the person from being entitled to the payment for any days after the election is made."""

S220 = """1
The winter energy payment is not payable for 1 or more days while a beneficiary is absent from New Zealand unless it is so payable under subsection (2).
2
The winter energy payment is payable to a beneficiary for up to a maximum of 28 days of any 1 or more absences (however long each absence lasts in total) of the beneficiary from New Zealand during the winter period (as defined in section 71 ) if—
a
the payment would be payable to the beneficiary were it not for those days of those absences; and
b
MSD is satisfied that those days of those absences do not affect the beneficiary’s eligibility for the payment under section 72 .
3
Sections 219 and 436 do not apply to the winter energy payment."""

RATES_TABLE = """1 | The rates of winter energy payment are as follows:
(a) | to a person who is single and has no dependent children | $ 450 per winter period (b) | to a person who is single and has 1 or more dependent children | $ 700 per winter period (c) | to a person who is in a relationship, and who is the 1 spouse or partner who is entitled under section 72(3) | $ 700 per winter period 2 | For the purposes of clause 1(a) and (b), dependent child , in relation to a person, means any other person who— (a) | is a child— (i) | whose care is primarily the responsibility of the person; and (ii) | who is being maintained as a member of that person’s family; and (iii) | who is financially dependent on that person; and (b) | may be, but need not be, a child in respect of whom payments are being made under section 363 of the Oranga Tamariki Act 1989 ; and (c) | may be, but need not be, a child in respect of whom an orphan’s benefit or an unsupported child’s benefit is being paid; and (d) | is not a child in respect of whom a young parent payment is being paid in relation to a person who is not the child’s parent or step-parent."""

REPLACEMENT_TABLE = """In Schedule 4 , Part 7, replace clauses 4 to 6A with:
4 | To a person (other than a specified person) who has 1 or more dependent children and who is in a relationship, or a sole parent with 2 or more dependent children, whose accommodation costs are the sum of payments required under any mortgage security, and other payments that the chief executive is satisfied are reasonably required to be made in respect of the person's home | 70% of the amount by which an applicant’s weekly qualifying accommodation costs exceeds 40% of the base rate, but not more than— a $305 a week, if the applicant resides in Area 1: b $220 a week, if the applicant resides in Area 2: c $160 a week, if the applicant resides in Area 3: d $120 a week, if the applicant resides in Area 4
4A | To a specified person who has 1 or more dependent children and who is in a relationship, or a sole parent with 2 or more dependent children, whose accommodation costs are the sum of payments required under any mortgage security, and other payments that the chief executive is satisfied are reasonably required to be made in respect of the specified person's home | 70% of the amount by which an applicant’s weekly qualifying accommodation costs exceeds 30% of the base rate, but not more than— a $305 a week, if the applicant resides in Area 1: b $220 a week, if the applicant resides in Area 2: c $160 a week, if the applicant resides in Area 3: d $120 a week, if the applicant resides in Area 4
5 | To a person (other than a specified person) who has no dependent children and who is in a relationship, or a sole parent with 1 dependent child, whose accommodation costs are the sum of payments required under any mortgage security, and other payments that the chief executive is satisfied are reasonably required to be made in respect of the person's home | 70% of the amount by which an applicant’s weekly qualifying accommodation costs exceeds 40% of the base rate, but not more than— a $235 a week, if the applicant resides in Area 1: b $155 a week, if the applicant resides in Area 2: c $105 a week, if the applicant resides in Area 3: d $80 a week, if the applicant resides in Area 4
5A | To a specified person who has no dependent children and who is in a relationship, or a sole parent with 1 dependent child, whose accommodation costs are the sum of payments required under any mortgage security, and other payments that the chief executive is satisfied are reasonably required to be made in respect of the specified person’s home | 70% of the amount by which an applicant’s weekly qualifying accommodation costs exceeds 30% of the base rate, but not more than— a $235 a week, if the applicant resides in Area 1: b $155 a week, if the applicant resides in Area 2: c $105 a week, if the applicant resides in Area 3: d $80 a week, if the applicant resides in Area 4
6 | To any other person (other than a specified person) whose accommodation costs are the sum of payments required under any mortgage security, and other payments that the chief executive is satisfied are reasonably required to be made in respect of the person’s home | 70% of the amount by which an applicant’s weekly qualifying accommodation costs exceeds 40% of the base rate, but not more than— a $165 a week, if the applicant resides in Area 1: b $105 a week, if the applicant resides in Area 2: c $80 a week, if the applicant resides in Area 3: d $70 a week, if the applicant resides in Area 4
6AA | To any other specified person whose accommodation costs are the sum of payments required under any mortgage security, and other payments that the chief executive is satisfied are reasonably required to be made in respect of the specified person’s home | 70% of the amount by which an applicant’s weekly qualifying accommodation costs exceeds 30% of the base rate, but not more than— a $165 a week, if the applicant resides in Area 1: b $105 a week, if the applicant resides in Area 2: c $80 a week, if the applicant resides in Area 3: d $70 a week, if the applicant resides in Area 4
6A | The rate of accommodation supplement payable to a community spouse or partner (as defined in Schedule 2) is the rate in clauses 1 to 6AA that would apply to them if they were single (and their other circumstances were unchanged).
6B | In clauses 4 to 6AA , specified person means a person, or the spouse or partner of a person, who is receiving— a supported living payment on the ground of restricted work capacity or total blindness; or b supported living payment on the ground of caring for another person; or c emergency benefit, but only if the equivalent benefit under section 63(4) is— i supported living payment on the ground of restricted work capacity or total blindness; or ii supported living payment on the ground of caring for another person; or d New Zealand superannuation; or e veteran’s pension."""


# Synthetic legal-identity control; the official unheaded excerpt is kept above.
HEADED_RATES_TABLE = "Clause | Provision\n" + RATES_TABLE
# Synthetic marker-only cells move the value into the same prose cell.
MARKER_ONLY_RATES_TABLE = HEADED_RATES_TABLE.replace(" | $", " $")
MARKER_ONLY_REPLACEMENT_TABLE = REPLACEMENT_TABLE.replace(" | 70%", " 70%")


def test_complete_prose_subsections_with_prescribed_rate_are_authenticated():
    outline = bare_outline(S73)
    assert outline is not None
    assert [label.path for label in outline] == [("1",), ("2",), ("3",), ("4",)]
    assert re.findall(r"\d+", mask_bare_labels(S73, outline)) == []
    for previous, following in zip(outline, outline[1:]):
        assert previous.end == following.start
    election_form = outline[2]
    assert "form and manner" in S73[election_form.marker_end : election_form.end]
    assert "Unless it is revoked" not in S73[election_form.start : election_form.end]


def test_legal_sections_subject_does_not_invalidate_local_subsections():
    outline = bare_outline(S220)
    assert outline is not None
    assert [label.path for label in outline] == [
        ("1",),
        ("2",),
        ("2", "a"),
        ("2", "b"),
        ("3",),
    ]
    exception = next(label for label in outline if label.path == ("2", "b"))
    assert "section 72" in S220[exception.marker_end : exception.end]
    assert "Sections 219" not in S220[exception.start : exception.end]
    assert "Sections 219 and 436 do not apply" in mask_bare_labels(S220, outline)


@settings(max_examples=20, deadline=None)
@given(
    quantity=st.sampled_from(("rate", "amount", "percentage")),
    prescribed=st.sampled_from(("prescribed", "specified", "approved")),
    ending=st.sampled_from(("\n", "\r\n", "\u2028")),
)
def test_complete_prose_outline_is_stable_under_prescribed_quantity_complements(
    quantity, prescribed, ending
):
    source = S73.replace("rate prescribed", f"{quantity} {prescribed}").replace(
        "\n", ending
    )
    outline = bare_outline(source)
    assert outline is not None
    assert [label.path for label in outline] == [("1",), ("2",), ("3",), ("4",)]
    assert len(mask_bare_labels(source, outline)) == len(source)
    assert re.findall(r"\d+", mask_bare_labels(source, outline)) == []


def test_flattened_rate_rows_end_before_later_rows_and_next_clause():
    outline = table_outline(MARKER_ONLY_RATES_TABLE)
    assert outline is not None
    paths = {label.path: label for label in outline}
    single = paths[("1", "a")]
    assert len(MARKER_ONLY_RATES_TABLE[single.marker_end : single.end]) < 100
    assert (
        "$ 450 per winter period"
        in MARKER_ONLY_RATES_TABLE[single.marker_end : single.end]
    )
    assert "$ 700" not in MARKER_ONLY_RATES_TABLE[single.marker_end : single.end]
    assert paths[("1", "c")].end == paths[("2",)].start
    assert paths[("2", "a", "iii")].end == paths[("2", "b")].start
    masked = mask_bare_labels(MARKER_ONLY_RATES_TABLE, outline)
    assert re.findall(r"\d+", masked) == [
        "450",
        "1",
        "700",
        "1",
        "72",
        "3",
        "700",
        "1",
        "363",
        "1989",
    ]


def test_replacement_clause_suffixes_and_cross_reference_keep_their_row():
    outline = bare_outline(MARKER_ONLY_REPLACEMENT_TABLE)
    assert outline is not None
    assert [label.path for label in outline] == [
        ("4",),
        ("4A",),
        ("5",),
        ("5A",),
        ("6",),
        ("6AA",),
        ("6A",),
        ("6B",),
    ]
    row = next(label for label in outline if label.token == "6A")
    owned = MARKER_ONLY_REPLACEMENT_TABLE[row.marker_end : row.end]
    assert "clauses 1 to 6AA" in owned
    assert "specified person means" not in owned
    assert row.end == MARKER_ONLY_REPLACEMENT_TABLE.index("6B |")


@settings(max_examples=30, deadline=None)
@given(
    amount=st.integers(min_value=1, max_value=999999),
    separator=st.sampled_from((" ", "\n", "\r\n", "\u2028")),
)
def test_table_mask_preserves_amounts_offsets_and_all_row_boundaries(amount, separator):
    source = MARKER_ONLY_RATES_TABLE.replace("450", str(amount))
    source = re.sub(r" (?=\([a-z]+\) \||2 \|)", lambda _: separator, source)
    outline = bare_outline(source)
    assert outline is not None
    masked = mask_bare_labels(source, outline)
    assert len(masked) == len(source)
    amount_start = source.index(f"$ {amount}") + 2
    assert masked[amount_start : amount_start + len(str(amount))] == str(amount)
    spans = [(label.start, label.marker_end) for label in outline]
    for index, (before, after) in enumerate(zip(source, masked)):
        if before != after:
            assert any(start <= index < end for start, end in spans)
    by_path = {label.path: label for label in outline}
    for label in outline:
        for depth in range(1, len(label.path)):
            parent = by_path[label.path[:depth]]
            assert parent.start <= label.start < label.end <= parent.end


@settings(max_examples=25, deadline=None)
@given(amount=st.integers(min_value=1, max_value=999999))
def test_numeric_pipe_cells_without_legal_identity_remain_operative(amount):
    for body in (
        f"The amount is ${amount}.",
        "The child is eligible.",
        "The children are eligible.",
    ):
        source = f"1 | {body}\n2 | {body}\n"
        assert table_outline(source) is None
        assert mask_bare_labels(source) == source


@settings(max_examples=30, deadline=None)
@given(
    heading=st.sampled_from(
        (
            "Number of dependent children",
            "Count of eligible dependants",
            "The number of household members",
            "Annual amount",
            "Rate",
            "Percentage",
        )
    ),
    border=st.sampled_from(("", "| ")),
    separator=st.sampled_from(("\n", "\r\n", "\u2028")),
)
def test_value_column_headers_keep_counts_even_with_nested_conditions(
    heading, border, separator
):
    source = separator.join(
        (
            f"{border}{heading} | Eligibility",
            "1 | The child is eligible if— (a) | the child resides here.",
            "2 | The children are eligible if— (a) | the children reside here.",
        )
    )
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source


@pytest.mark.parametrize("prefix", ("Clause | Provision\n", "| Clause | Provision\n"))
def test_clause_column_headers_keep_statutory_rate_rows(prefix):
    outline = bare_outline(prefix + RATES_TABLE.replace(" | $", " $"))
    assert outline is not None
    assert {label.path for label in outline} >= {("1", "a"), ("2", "d")}


@pytest.mark.parametrize(
    "token",
    ("0", "-2", "+2", "2a", "2A1", "2" + "Z" * 9, "9" * 10, "(1)", "(1A)"),
)
def test_unknown_table_marker_invalidates_the_whole_walk(token):
    source = MARKER_ONLY_RATES_TABLE.replace("2 | For", f"{token} | For")
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source


@pytest.mark.parametrize("token", ("1.", "(1.)", "1.2", "(1.2)"))
def test_unknown_dotted_table_row_between_siblings_invalidates_the_whole_walk(token):
    source = MARKER_ONLY_RATES_TABLE.replace(
        "(b) | to a person",
        f"{token} | The unrecognised row is operative. (b) | to a person",
        1,
    )
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source


@pytest.mark.parametrize("token", ("-2", "+2", "−2", "–2", "—2"))
def test_unknown_signed_table_row_between_siblings_invalidates_the_whole_walk(token):
    source = MARKER_ONLY_RATES_TABLE.replace(
        "(b) | to a person",
        f"{token} | The unrecognised row is operative. (b) | to a person",
        1,
    )
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source


@pytest.mark.parametrize("border", ("", "| "))
@pytest.mark.parametrize("token", ("a", "A", "1", "1A"))
def test_unparsed_line_start_children_prevent_a_partial_pipe_walk(border, token):
    source = (
        "In Schedule 4, replace clauses 1 and 2 with:\n"
        "1 | A single person is eligible for payment as follows:\n"
        f"{border}({token}) A single person is eligible for $450.\n"
        f"{border}(b) A sole parent is eligible for $700.\n"
        "2 | A dependent child means a person supported by a parent."
    )
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source


def test_duplicate_or_out_of_order_table_roots_are_never_skipped():
    for token in ("1", "3"):
        assert (
            table_outline(MARKER_ONLY_RATES_TABLE.replace("2 | For", f"{token} | For"))
            is None
        )


@pytest.mark.parametrize(
    "caption",
    [
        "",
        "Family size | Eligibility\n",
        "Age | Eligibility\n",
        "Number of children | Eligibility\n",
        "Years | Eligibility\n",
    ],
)
def test_children_do_not_authenticate_numeric_cell_identity(caption):
    source = (
        caption + "1 | The household is eligible if— (a) | its income is below $450.\n"
        "2 | The household is eligible if— (a) | its income is below $700."
    )
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source


def test_unheaded_nz_rate_table_abstains_without_positive_legal_identity():
    assert table_outline(RATES_TABLE) is None
    assert mask_bare_labels(RATES_TABLE) == RATES_TABLE
    assert table_outline(HEADED_RATES_TABLE) is None
    assert table_outline(MARKER_ONLY_RATES_TABLE) is not None


@pytest.mark.parametrize("separator", [" ", "\n", "\r\n"])
def test_inline_nonpipe_child_cannot_be_absorbed_by_a_pipe_sibling(separator):
    source = (
        "In Schedule 4, replace clauses 1 and 2 with:\n"
        "1 | The conditions apply as follows: (a) | A person is eligible if resident."
        + separator
        + "(b) A person is eligible if aged 65 years.\n"
        "2 | The authority must pay the benefit."
    )
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source


@pytest.mark.parametrize("heading", ["", "Clause | Provision\n"])
def test_pipe_definition_layout_abstains_where_detached_layout_has_independent_owners(
    heading,
):
    pipe = (
        "1 | The following definitions apply:—\nbenefit means—\n"
        "(a) | a main benefit; or\n(b) | a pension.\n"
        "minimum period means a period of at least 22 weeks."
    )
    detached = (
        pipe.replace("1 | ", "1\n").replace("(a) | ", "a\n").replace("(b) | ", "b\n")
    )
    outline = bare_outline(detached)
    assert outline is not None
    assert [term for _, term in bare_definition_headers(detached, outline)] == [
        "benefit",
        "minimum period",
    ]
    assert detached[outline[-1].marker_end : outline[-1].end].strip() == "a pension."
    # The same content cannot be certified with weaker ownership after a layout change.
    assert table_outline(heading + pipe) is None
    assert mask_bare_labels(heading + pipe) == heading + pipe


def test_same_line_label_does_not_hide_a_following_definition_header():
    source = (
        "1 | The following definitions apply:—\nbenefit means—\n(a) | a main benefit."
    )
    from axiom_encode.bare_structure import BareLabel

    labels = (BareLabel("1", ("1",), 0, source.index("The"), len(source)),)
    assert [term for _, term in bare_definition_headers(source, labels)] == ["benefit"]


@pytest.mark.parametrize(
    "introduction", ["Compare clauses 1 and 2 with:", "Clauses 1 and 2 with:"]
)
def test_replacement_identity_requires_an_explicit_replacement(introduction):
    source = (
        introduction
        + "\n1 | The household is eligible if— (a) | its income is below $450.\n"
        "2 | The household is eligible if— (a) | its income is below $700."
    )
    assert table_outline(source) is None
    assert (
        table_outline(source.replace(introduction, "Replace clauses 1 and 2 with:"))
        is not None
    )


@pytest.mark.parametrize("term", ["minimum_period", "minimum.period"])
def test_unfamiliar_detached_definition_terms_prevent_a_partial_table_walk(term):
    source = (
        "Clause | Provision\n1 | The conditions apply as follows:—\n"
        "(a) | A person is eligible if resident.\n(b) | A pension is payable.\n"
        f"{term} means a period of at least 22 weeks."
    )
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source


def test_official_value_cell_tables_abstain_on_ambiguous_ownership():
    for source in (RATES_TABLE, HEADED_RATES_TABLE, REPLACEMENT_TABLE):
        assert table_outline(source) is None
        assert mask_bare_labels(source) == source


JURISDICTIONS = ("nz", "uk", "us", "de", "fr")


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
@pytest.mark.parametrize("border", ("", "| "))
def test_outer_legal_caption_cannot_authenticate_embedded_count_table(
    jurisdiction, border
):
    source = (
        "Clause | Provision\n"
        "1 | The rates are specified in the following table:\n"
        f"{border}Number of children | Eligibility\n"
        "2 | The household is eligible if— (a) | its income is below $450.\n"
        "3 | The household is eligible if— (a) | its income is below $700."
    )
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=f"{jurisdiction}/statute/example/section/15"
    )
    assert vp.extract_numeric_occurrences_from_text(cleaned) == [1, 2, 450, 3, 700]


@pytest.mark.parametrize(
    "heading", ("Number of children", "Family size", "Age", "Class")
)
@pytest.mark.parametrize("columns", ("", " | Eligibility"))
def test_a_new_table_caption_after_first_root_prevents_partial_ownership(
    heading, columns
):
    source = (
        "Clause | Provision\n1 | The following table applies:\n"
        f"{heading}{columns}\n"
        "2 | The household is eligible if— (a) | its income is below $450.\n"
        "3 | The household is eligible if— (a) | its income is below $700."
    )
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source


@pytest.mark.parametrize("line", ("The following applies:", "wrapped prose", "$450"))
def test_unmarked_table_lines_abstain_without_positive_region_ownership(line):
    source = (
        "Clause | Provision\n1 | The following conditions apply:\n"
        f"{line}\n"
        "2 | The household is eligible if— (a) | its income is below $450."
    )
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source


@pytest.mark.parametrize(
    "intervening", ("The following table applies:", "Family size | Eligibility")
)
def test_legal_caption_must_be_immediate_to_its_own_table(intervening):
    source = (
        f"Clause | Provision\n{intervening}\n"
        "1 | The household is eligible if— (a) | its income is below $450.\n"
        "2 | The household is eligible if— (a) | its income is below $700."
    )
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
@pytest.mark.parametrize("separator", (" ", "\n", "\r\n"))
@pytest.mark.parametrize("token", ("b", "B", "bb", "iv", "(B)", "[b]", "b."))
def test_unsupported_pipe_children_abstain_instead_of_belonging_to_previous_child(
    jurisdiction, separator, token
):
    source = (
        "In Schedule 4, replace clauses 1 and 2 with:\n"
        "1 | The conditions apply as follows: (a) | A person is eligible if resident."
        + separator
        + f"{token} | A person is eligible if aged 65 years.\n"
        "2 | The authority must pay the benefit."
    )
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=f"{jurisdiction}/statute/example/section/15"
    )
    assert 65 in vp.extract_numeric_occurrences_from_text(cleaned)


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
@pytest.mark.parametrize("border", ("| ", " | ", "|\t", "|| "))
def test_bordered_detached_definition_heads_prevent_weaker_pipe_ownership(
    jurisdiction, border
):
    source = (
        "Clause | Provision\n1 | The following definitions apply:—\n"
        f"{border}benefit means—\n"
        "(a) | a main benefit; or\n(b) | a pension.\n"
        f"{border}minimum period means a period of at least 22 weeks."
    )
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=f"{jurisdiction}/statute/example/section/15"
    )
    assert 22 in vp.extract_numeric_occurrences_from_text(cleaned)
