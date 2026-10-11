"""Terminal session-law histories are locators; operative values survive."""

from __future__ import annotations

import re
import string
from datetime import date
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
)

OK_HISTORY = (
    "Added by Laws 1988, c. 162, § 106, eff. Jan. 1, 1992. "
    "Amended by Laws 1996, c. 323, § 4, eff. Jan. 1, 1997."
)
VA_HISTORY = (
    "2017, c. 444 ; 2019, cc. 17 , 18 ; "
    "2021, Sp. Sess. I, cc. 117 , 118 , 552 ; 2022, cc. 3 , 648 ; "
    "2022, Sp. Sess. I, cc. 1 , 6 ; 2023, Sp. Sess. I, c. 1 ; "
    "2024, c. 217 ; 2022, cc. 3 , 19 ; 2025, cc. 615 , 658 , 725 ; "
    "2026, c. 7 ; 2026, Sp. Sess. I, c. 1 ."
)


def _values(text: str) -> list[float]:
    return [
        occurrence.value
        for occurrence in extract_typed_numeric_inventory_occurrences_from_text(
            sc.authoritative_numeric_recall_text(text), profile="en-US"
        )
    ]


@pytest.mark.parametrize("history", (OK_HISTORY, VA_HISTORY))
def test_exact_histories_mask_only_validated_terminal_suffix(history):
    operative = "The payable credit is $162 for 65 years and $12,000 of income."
    source = f"{operative}\n\n{history}"

    assert sc._strip_terminal_session_law_history(source) == operative
    assert sc.authoritative_numeric_recall_text(source) == operative
    assert _values(source) == _values(operative)
    # Producing computational recall text must not mutate the authoritative
    # source, including the historical effective dates needed for provenance.
    assert source.endswith(history)


@pytest.mark.parametrize(
    ("run_id", "history", "operative_values"),
    (
        ("37936331759", OK_HISTORY, {12000.0}),
        (
            "37936341740",
            VA_HISTORY,
            {4000.0, 70.0, 500.0, 8750.0, 17500.0, 9200.0, 18400.0},
        ),
    ),
)
def test_real_reencode_source_keeps_every_operative_value(
    run_id, history, operative_values
):
    source = (
        Path(__file__).parent
        / "fixtures/source_completeness"
        / f"signed_reencode_{run_id}"
        / "source.txt"
    ).read_text()
    assert source.endswith(history)
    operative = source[: -len(history)].rstrip()

    assert sc._strip_terminal_session_law_history(source) == operative
    assert _values(source) == _values(operative)
    assert operative_values <= set(_values(source))
    if run_id == "37936331759":
        assert "sixty-five (65) years of age or older" in operative
        assert "only one claim per year" in operative


def test_nj_operative_public_law_reference_does_not_block_authentic_footer():
    # Exact operative reference and terminal history from us-nj/statute/54a:1-2.
    # The period inside P.L. cannot authenticate an inner L. history start.
    operative = (
        'e. "Dependent" means a spouse or child, or a domestic partner as defined '
        "in section 3 of P.L.2003, c.246 (C.26:8A-3), or any individual related "
        "to the taxpayer and who is a dependent pursuant to the provisions of "
        "the Internal Revenue Code during a taxable year."
    )
    history = "L.1976, c.47, s. 54A:1-2; amended 2003, c.246, s.39."
    source = f"{operative}\n\n{history}"

    assert sc._strip_terminal_session_law_history(source) == operative
    assert _values(source) == _values(operative)


@pytest.mark.parametrize(
    "operative",
    (
        "Amended by L.2001, c.10. A deduction of $444 is allowed.",
        "L.2001, c.10 authorizes a deduction of $444.",
        "The deduction is $444. Added by L.2001, c.10 sets the rule.",
        "P.L.2001, c.10 authorizes a deduction of $444.",
        "L.2001, c.10 sets eligibility at $444.",
        "L.2001, c.10 determines the amount as $444.",
        "2017, c. 444 allows a $162 deduction.",
        "L.2001, c.10 A $444 deduction is allowed.",
        "L.2001, c.10 amended criteria permit $444.",
        "L.2001, c.10 Laws require a deduction of $444.",
        "L.2001, c.10 sets\neligibility at $444.",
        "L.2001, c.10 allows $444.",
        "L.2001, c.10 requires $444.",
        "L.2001, c.10 effective criteria require $444.",
        "L.2001, c.10 operative criteria require $444.",
        "Source: L.2001, c.10 authorizes a deduction of $444.",
        "Source: 2017, c.444 allows a $162 deduction.",
        "Amended by L.2001, c.10. $444 is the allowable deduction.",
        "Amended by L.2001, c.10. 65 years is the minimum age.",
        "Amended by L.2001, c.10. (A deduction of $444 is allowed.)",
        "Amended by L.2001, c.10. “The deduction is $444.”",
        "L.2001, c.10. 43 - 7 determines the allowance of $444.",
        "L.2001, c.10 provides $444. Added by L.2002, c.11 sets eligibility at $162.",
    ),
)
def test_operative_citation_paragraph_does_not_block_separate_authentic_footer(
    operative,
):
    # These cite an enactment in body prose. They remain intact while the
    # independent, blank-line-delimited terminal history is fully validated.
    history = "L.1976, c.47, s. 54A:1-2; amended 2003, c.246, s.39."
    source = f"{operative}\n\n{history}"

    assert sc._strip_terminal_session_law_history(source) == operative
    assert _values(source) == _values(operative)


@pytest.mark.parametrize(
    "history",
    (
        "History: 2017, c.615;\n\n2026, c.7.",
        "History: 2017, c.615; 2026, c.7.",
        "Added by L.2002, c.11; Amended by L.2003, c.12.",
    ),
)
def test_later_authenticated_history_after_operative_citation_owns_its_suffix(history):
    operative = "L.2001, c.10 provides $444."
    source = f"{operative} {history}"

    assert sc._strip_terminal_session_law_history(source) == operative
    assert _values(source) == _values(operative)


@pytest.mark.parametrize(
    "metadata",
    (
        "Source: Added monetary allowances.",
        "Source: unknown, c. $444;",
        "Source: 20177;",
        "Source: Laws incomplete.",
        "Source: 2025.",
        "Source: 2025\nA deduction of $444 is allowed.",
        "Added 2025.",
    ),
)
def test_ordinary_source_metadata_does_not_block_authentic_footer(metadata):
    operative = f"{metadata} The credit is $444."
    source = f"{operative}\n\n{OK_HISTORY}"

    assert sc._strip_terminal_session_law_history(source) == operative
    assert _values(source) == _values(operative)


def test_oklahoma_inline_period_chain_retains_immediately_preceding_amount():
    operative = "The credit is $162."

    assert (
        sc.authoritative_numeric_recall_text(f"{operative} {OK_HISTORY}") == operative
    )
    assert _values(f"{operative} {OK_HISTORY}") == [162.0]


def test_virginia_operational_amount_equal_to_chapter_remains_a_value():
    operative = "A deduction of $444 is allowed."

    assert _values(f"{operative}\n\n{VA_HISTORY}") == [444.0]


@pytest.mark.parametrize(
    "history",
    (
        "Added by Laws.\nAmended by Laws 1996, c. 323, § 4, eff. Jan. 1, 1997.",
        "Added by Laws unknown, c. $444;\nAmended by Laws 1996, c. 323, § 4.",
        "Amended by L.\nAdded by Laws 1996, c. 323, § 4.",
        "Amended by P.L. unknown, c. $444;\nAdded by Laws 1996, c. 323, § 4.",
        "History: 2017.\n2026, c. 7.",
        "History: unknown, c. $444;\n2026, c. 7.",
        "History: 20177;\n2026, c. 7.",
        "History: Laws incomplete;\n2026, c. 7.",
        "History:;\n2026, c. 7.",
        "History: unknown\n\n2026, c. 7.",
        "History: 2017, c. $444;\n\n2026, c. 7.",
        "Source: 2017, c. $444;\n\n2026, c. 7.",
        "Source: L.2001, c.10, §;\n\n2026, c. 7.",
        "Source: L.2001, c.10, eff. Feb. 30, 1992 authorizes $444.\n\n2026, c. 7.",
        "Added by Laws.\n\nAmended by Laws 1996, c. 323, § 4.",
        "Added by Laws 1988, cc. 162;\n\n2026, c. 7.",
        "L.2001, c.10; operative prose sets $444.\n\n2026, c. 7.",
        "L.2001, c.10, §;\n\n2026, c. 7.",
        "L.2001, c.10, eff. Feb. 30, 1992 authorizes $444.\n\n2026, c. 7.",
        "L.2001, c.10 amended 2017, c.;\n\n2026, c. 7.",
        "L.2001, c.10 amended by Laws incomplete;\n\n2026, c. 7.",
        "L.2001, c.10 effective unknown;\n\n2026, c. 7.",
        "L.2001, c.10 operative Jan. 1, 2001;\n\n2026, c. 7.",
        "L.2001, c.10 eff Jan. 1, 1992;\n\n2026, c. 7.",
        "L.2001, c.10 s 5;\n\n2026, c. 7.",
        "L.2001, c.10 cc 20;\n\n2026, c. 7.",
        "L.2001, c.10 Sp Sess XII;\n\n2026, c. 7.",
        "L.2001, c.10 amended by unknown;\n\n2026, c. 7.",
        "L.2001, c.10 amended by incomplete;\n\n2026, c. 7.",
        "L.2001, c.10 amended by missing;\n\n2026, c. 7.",
        "L.2001, c.10 effective.\n\n2026, c. 7.",
        "L.2001, c.10 amended.\n\n2026, c. 7.",
        "L.2001, c.10 Laws.\n\n2026, c. 7.",
        "History: L.2001, c.10 authorizes $444.\n\n2026, c. 7.",
        "L.2001, c.10 provides $444. History: 2017, c. $615;\n\n2026, c. 7.",
        "L.2001, c.10 provides $444. History: unknown\n\n2026, c. 7.",
        "L.2001, c.10 provides $444. Added by Laws incomplete;\n\n2026, c. 7.",
        "L.2001, c.10 provides $444.\nHistory: 2017, c. $615;\n\n2026, c. 7.",
        "L.2001, c.10 (History: 2017, c. $615;)\n\n2026, c. 7.",
        "L.2001, c.10 provides $444. History: 2017, c. $615; 2026, c. 7.",
        "L.2001, c.10 provides $444. Source: 2017, c. $615; 2026, c. 7.",
        "L.2001, c.10 provides $444. History: unknown; 2026, c. 7.",
        "History: 2017, Sp. Sess. XII, c. $444;\n2026, c. 7.",
        "History: 2017, c. $444;\n2026, c. 7.",
        "Added by Laws 1988, cc. 162;\nAmended by Laws 1996, c. 323, § 4.",
        "2017, c. $444;\n2026, c. 7.",
        "History: 2017, c.;\n2026, c. 7.",
        "History: 2017;\n2026, c. 7.",
        "History: 20x7, c. 444;\n2026, c. 7.",
        "Added by Laws c. 162;\nAmended by Laws 1996, c. 323, § 4.",
        "2017, Sp. Sess. XI, c. $444;\n2026, c. 7.",
        "Added by Laws 1988, c. 162, §.",
        "Added by Laws 1988, c. 162, § 106, eff. Jan. 1.",
        "Added by Laws 1988, c. 162, § 106, eff. Feb. 30, 1992.",
        "Added by Laws 1988, c. 162, § 106, eff. Jan. 1, 0000.",
        "Added by Laws 1988, c. 162, § 106; the payable credit is $162.",
        "Added by Laws 1988, c. 162, §; Amended by Laws 1996, c. 323, § 4.",
        "Added by Laws 1988, c. 162, §. Amended by Laws 1996, c. 323, § 4.",
        "2017, c. 444; 2019, cc. 17,; 2026, c. 7.",
        "2017, c. 444; 2019, c. 17, 18; 2026, c. 7.",
        "2017, c. 444; 2019, cc. 17; 2026, c. 7.",
        "2017, c. 444; 2021, Sp. Sess. XI, c. 117; 2026, c. 7.",
        "2017, c. 444; prose prescribes $615; 2026, c. 7.",
        "2017, c. 444; 2026, c. 7;",
        f"{OK_HISTORY} The payable credit is $162.",
        f"{VA_HISTORY} A deduction of $444 is allowed.",
    ),
)
def test_malformed_or_nonterminal_history_is_not_partially_masked(history):
    source = f"The payable credit is $162.\n\n{history}"

    assert sc._strip_terminal_session_law_history(source) == source
    # Other pre-existing legal-citation masks may remove a § locator even
    # when the history chain is malformed; amounts must remain in recall.
    assert 162.0 in _values(source)
    for amount in re.findall(r"\$(\d+)", history):
        assert float(amount) in _values(source)


@pytest.mark.parametrize(
    "source",
    (
        "The governing law is Added by Laws 1988, c. 162, § 106.",
        "A deduction of $444 is allowed under 2017, c. 444; 2019, cc. 17, 18.",
        "The amount is $162. Added by Laws 1988, c. 162, § 106.",
    ),
)
def test_inline_single_or_unanchored_citation_is_not_assumed_to_be_history(source):
    assert sc._strip_terminal_session_law_history(source) == source


@st.composite
def _history_entry(draw):
    year = draw(st.integers(min_value=1800, max_value=2099))
    chapters = draw(
        st.lists(st.integers(min_value=1, max_value=9999), min_size=1, max_size=4)
    )
    shape = draw(st.sampled_from(("ok", "va", "nj")))
    if shape == "va":
        session = draw(st.sampled_from(("", "Sp. Sess. I, ", "Sp. Sess. II, ")))
        chapter_text = "c. " if len(chapters) == 1 else "cc. "
        return f"{year}, {session}{chapter_text}{' , '.join(map(str, chapters))}."
    action = draw(st.sampled_from(("Added by", "Amended by", "Supplemented by")))
    section = draw(st.integers(min_value=1, max_value=999))
    publication = "Laws" if shape == "ok" else "L."
    section_marker = "§" if shape == "ok" else "s."
    entry = (
        f"{action} {publication} {year}, c. {chapters[0]}, {section_marker} {section}"
    )
    if draw(st.booleans()):
        effective = draw(
            st.dates(min_value=date(1800, 1, 1), max_value=date(2099, 12, 31))
        )
        entry += f", eff. {effective.strftime('%b').replace('Sep', 'Sept')}. {effective.day}, {effective.year}"
    return f"{entry}."


@st.composite
def _history_chain(draw):
    entries = draw(st.lists(_history_entry(), min_size=1, max_size=8))
    separators = draw(
        st.lists(
            st.sampled_from((" ", "; ", "\n")),
            min_size=len(entries) - 1,
            max_size=len(entries) - 1,
        )
    )
    return (
        "".join(entry + separator for entry, separator in zip(entries, separators))
        + entries[-1]
    )


_OPERATIVE_PROSE = st.text(
    alphabet=string.ascii_letters + string.digits + " ,.!?()[]-/&;", max_size=80
)


@given(_OPERATIVE_PROSE, st.integers(min_value=1, max_value=999999), _history_chain())
@settings(max_examples=150, deadline=None)
@example(prose="Added", amount=162, history=OK_HISTORY)
@example(prose="Source: Added monetary allowances", amount=444, history=VA_HISTORY)
def test_generated_histories_preserve_arbitrary_operative_prefix(
    prose, amount, history
):
    operative = f"{prose} The payable amount is ${amount}.".strip()
    source = f"{operative}\n\n{history}"

    assert sc._strip_terminal_session_law_history(source) == operative
    # Equality, rather than a set difference, handles chapter numbers that
    # also occur as operative amounts: only the latter occurrence survives.
    assert _values(source) == _values(operative)


@given(st.integers(min_value=1, max_value=999999), _history_chain(), st.booleans())
@settings(max_examples=150, deadline=None)
def test_every_generated_amount_outside_history_remains_recalled(
    amount, history, after
):
    operative = f"The payable amount is ${amount}."
    source = f"{history}\n\n{operative}" if after else f"{operative}\n\n{history}"

    assert float(amount) in _values(source)


@given(_OPERATIVE_PROSE, st.integers(min_value=1, max_value=999999), _history_chain())
@settings(max_examples=150, deadline=None)
def test_history_mask_is_idempotent_and_never_lengthens_text(prose, amount, history):
    source = f"{prose} The payable amount is ${amount}.\n\n{history}"
    once = sc._strip_terminal_session_law_history(source)

    assert sc._strip_terminal_session_law_history(once) == once
    assert len(once) <= len(source)


@given(st.integers(min_value=1, max_value=999999), _history_chain())
@settings(max_examples=100, deadline=None)
def test_generated_malformed_first_entry_blocks_later_history_fragments(
    amount, history
):
    source = f"The credit is $162.\n\nHistory: 2017, c. ${amount};\n{history}"

    assert sc._strip_terminal_session_law_history(source) == source
    assert float(amount) in _values(source)


@given(
    _history_entry(),
    st.integers(min_value=1, max_value=999999),
    st.sampled_from(
        ("allows", "determines", "sets eligibility at", "amended criteria permit")
    ),
    _history_chain(),
)
@settings(max_examples=150, deadline=None)
def test_generated_operative_citation_paragraph_preserves_all_prefix_occurrences(
    citation, amount, predicate, history
):
    operative = f"{citation} {predicate} ${amount}."
    source = f"{operative}\n\n{history}"

    assert sc._strip_terminal_session_law_history(source) == operative
    assert _values(source) == _values(operative)
