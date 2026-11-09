from pathlib import Path

import pytest

from axiom_encode.harness.validator_pipeline import (
    _tokenize_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

SOURCE = (
    Path(__file__).parent / "fixtures/numeric_grounding/eitc_percentage_tables.txt"
).read_text()


@pytest.mark.parametrize("profile", ["legacy", "en-US"])
def test_flattened_eitc_table_rates_retain_percentage_column_context(profile):
    tokens = _tokenize_numeric_occurrences_from_text(SOURCE, profile=profile).grounding
    for value in (0.0765, 0.34, 0.4, 0.45, 0.1598, 0.2106):
        assert numeric_value_is_grounded(value, tokens), value
    assert not numeric_value_is_grounded(0.0766, tokens)
    for token in tokens:
        assert SOURCE[token.start : token.end] == token.raw
    for raw in ("34", "40", "45", "15.98", "21.06", "7.65"):
        matches = [t for t in tokens if t.raw == raw]
        assert matches and all(t.has_rate_context for t in matches)
    for token in tokens:
        if token.raw in ("1", "2", "3", "6,330", "6330", "5,000", "5000"):
            assert not token.has_rate_context


@pytest.mark.parametrize(
    "source",
    [
        "| Children | Amount | | --- | --- | | 1 | 7.65 |",
        "The percentage is mentioned in prose. | Children | Amount | | --- | --- | | 1 | 7.65 |",
        "| Children | Percentage | | 1 | 7.65 |",
    ],
)
def test_inline_rates_require_explicit_table_header_and_separator(source):
    tokens = _tokenize_numeric_occurrences_from_text(source, profile="en-US").grounding
    assert not numeric_value_is_grounded(0.0765, tokens)


def test_inline_percentage_context_does_not_leak_into_following_table_or_prose():
    source = (
        "| Children | Percentage | | --- | --- | | 1 | 7.65 | "
        "(2) Dollar amounts | Children | Dollars | | --- | --- | | 1 | 6330 | "
        "The next amount is $5000."
    )
    tokens = _tokenize_numeric_occurrences_from_text(source, profile="en-US").grounding
    assert numeric_value_is_grounded(0.0765, tokens)
    assert not numeric_value_is_grounded(63.3, tokens)
    assert not numeric_value_is_grounded(50, tokens)
    assert all(not t.has_rate_context for t in tokens if t.raw in ("6330", "5000"))


def test_mismatched_inline_header_width_cannot_turn_counts_into_rates():
    source = "| Children | Percentage | Amount | | --- | --- | | 1 | 7.65 |"
    tokens = _tokenize_numeric_occurrences_from_text(source, profile="en-US").grounding
    assert not numeric_value_is_grounded(0.01, tokens)
    assert not numeric_value_is_grounded(0.0765, tokens)


def test_flattened_header_preserves_context_for_following_line():
    source = "| Children | Percentage | | --- | --- |\n| 1 | 7.65 |"
    tokens = _tokenize_numeric_occurrences_from_text(source, profile="en-US").grounding
    assert numeric_value_is_grounded(0.0765, tokens)
    assert not numeric_value_is_grounded(0.01, tokens)


def test_second_inline_percentage_table_after_prose_has_own_context():
    source = (
        "| Children | Percentage | | --- | --- | | 1 | 7.65 | "
        "(2) Rates | Children | Percentage | | --- | --- | | 1 | 8.65 |"
    )
    tokens = _tokenize_numeric_occurrences_from_text(source, profile="en-US").grounding
    assert numeric_value_is_grounded(0.0765, tokens)
    assert numeric_value_is_grounded(0.0865, tokens)
    assert not numeric_value_is_grounded(0.01, tokens)


def test_blank_stub_header_keeps_bordered_table_context():
    source = "| | Percentage | | --- | --- |\n| 1 | 7.65 |"
    tokens = _tokenize_numeric_occurrences_from_text(source, profile="en-US").grounding
    assert numeric_value_is_grounded(0.0765, tokens)
    assert not numeric_value_is_grounded(0.01, tokens)


@pytest.mark.parametrize(
    "extra",
    ["7.65", "(a)", "3", "(iv)", "see note (a)", "1.5 hours", "(B) Joint returns"],
)
def test_inline_row_wider_than_header_is_not_a_rate_row(extra):
    source = f"| Children | Percentage | | --- | --- | | 1 | 3 | {extra} |"
    tokens = _tokenize_numeric_occurrences_from_text(source, profile="en-US").grounding
    assert not numeric_value_is_grounded(0.03, tokens)
    assert not numeric_value_is_grounded(0.0765, tokens)
    source = f"{source} | 2 | 4 |"
    tokens = _tokenize_numeric_occurrences_from_text(source, profile="en-US").grounding
    assert not numeric_value_is_grounded(0.03, tokens)
    assert not numeric_value_is_grounded(0.0765, tokens)


@pytest.mark.parametrize(
    ("source", "grounded", "ungrounded"),
    [
        # Inline table, then a bordered continuation row of a different width.
        (
            "| C | Percentage | | --- | --- | | 1 | 7.65 |\n| 2 | 3 | 34 |",
            (0.0765,),
            (0.34, 0.03, 0.02),
        ),
        # Inline table, then a separate bordered table on following lines.
        (
            "| A | B | | --- | --- | | 1 | 2 |\n| C | Percentage |\n"
            "| --- | --- |\n| 1 | 7.65 |",
            (0.0765,),
            (0.01, 0.02),
        ),
        # Bordered table, then an inline table in the same pipe block.
        (
            "| C | Percentage |\n| --- | --- |\n| 1 | 7.65 |\n"
            "| X | Percentage | | --- | --- | | 2 | 8.65 |",
            (0.0765, 0.0865),
            (0.01, 0.02),
        ),
        # Two inline tables on consecutive lines.
        (
            "| C | Percentage | | --- | --- | | 1 | 7.65 |\n"
            "| C | Percentage | | --- | --- | | 2 | 8.65 |",
            (0.0765, 0.0865),
            (0.01, 0.02),
        ),
        # A separator row after the inline table is not a value row.
        (
            "| C | Percentage | | --- | --- | | 1 | 7.65 |\n| --- | --- |\n"
            "| 2 | 8.65 |",
            (0.0765, 0.0865),
            (0.02,),
        ),
    ],
)
def test_inline_table_next_to_multiline_tables(source, grounded, ungrounded):
    tokens = _tokenize_numeric_occurrences_from_text(source, profile="en-US").grounding
    for value in grounded:
        assert numeric_value_is_grounded(value, tokens), value
    for value in ungrounded:
        assert not numeric_value_is_grounded(value, tokens), value
    for token in tokens:
        assert source[token.start : token.end] == token.raw
