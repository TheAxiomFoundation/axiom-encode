"""Public NZ law excerpts retain legal rows without losing operative numbers."""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.bare_structure import bare_outline, mask_bare_labels, table_outline

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
    outline = table_outline(RATES_TABLE)
    assert outline is not None
    paths = {label.path: label for label in outline}
    single = paths[("1", "a")]
    assert len(RATES_TABLE[single.marker_end : single.end]) < 100
    assert "$ 450 per winter period" in RATES_TABLE[single.marker_end : single.end]
    assert "$ 700" not in RATES_TABLE[single.marker_end : single.end]
    assert paths[("1", "c")].end == paths[("2",)].start
    assert paths[("2", "a", "iii")].end == paths[("2", "b")].start
    masked = mask_bare_labels(RATES_TABLE, outline)
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
    outline = bare_outline(REPLACEMENT_TABLE)
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
    owned = REPLACEMENT_TABLE[row.marker_end : row.end]
    assert "clauses 1 to 6AA" in owned
    assert "specified person means" not in owned
    assert row.end == REPLACEMENT_TABLE.index("6B |")


@settings(max_examples=30, deadline=None)
@given(
    amount=st.integers(min_value=1, max_value=999999),
    separator=st.sampled_from((" ", "\n", "\r\n", "\u2028")),
)
def test_table_mask_preserves_amounts_offsets_and_all_row_boundaries(amount, separator):
    source = RATES_TABLE.replace("450", str(amount))
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


@pytest.mark.parametrize(
    "prefix", ("", "Clause | Provision\n", "| Clause | Provision\n")
)
def test_clause_column_headers_keep_statutory_rate_rows(prefix):
    outline = bare_outline(prefix + RATES_TABLE)
    assert outline is not None
    assert {label.path for label in outline} >= {("1", "a"), ("2", "d")}


@pytest.mark.parametrize(
    "token",
    ("0", "-2", "+2", "2a", "2A1", "2" + "Z" * 9, "9" * 10, "(1)", "(1A)"),
)
def test_unknown_table_marker_invalidates_the_whole_walk(token):
    source = RATES_TABLE.replace("2 | For", f"{token} | For")
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source


@pytest.mark.parametrize("token", ("1.", "(1.)", "1.2", "(1.2)"))
def test_unknown_dotted_table_row_between_siblings_invalidates_the_whole_walk(token):
    source = RATES_TABLE.replace(
        "(b) | to a person",
        f"{token} | The unrecognised row is operative. (b) | to a person",
        1,
    )
    assert table_outline(source) is None
    assert mask_bare_labels(source) == source


@pytest.mark.parametrize("token", ("-2", "+2", "−2", "–2", "—2"))
def test_unknown_signed_table_row_between_siblings_invalidates_the_whole_walk(token):
    source = RATES_TABLE.replace(
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
        assert table_outline(RATES_TABLE.replace("2 | For", f"{token} | For")) is None
