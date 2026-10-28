"""Round 16: a complete statutory vocabulary controls the closed negator class.

The fixture is derived from 1,633 public Israeli-law records. Its expected
readings were audited in actual statutory contexts, independently of the
production regex. Regenerate with the receipted snapshot and the script named
in the fixture; the 11 MB source snapshot is intentionally not vendored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom_encode.harness import hebrew_source_text as hebrew

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/hebrew_negator_corpus.json").read_text()
)


def test_statutory_corpus_receipt_and_complete_distinct_vocabulary():
    assert FIXTURE["snapshot"] == {
        "filename": "2026-09-29-il-taxben-contributions.jsonl",
        "sha256": "fd5f1e4d6fb941c32cfbf2ef4f248146c29baf41906d1cdb3d6c7437b86d0ac1",
        "bytes": 11446347,
        "records": 1633,
        "text_records": 1621,
    }
    words = FIXTURE["vocabulary"]
    assert len(words) == FIXTURE["distinct_token_count"]
    assert words == sorted(set(words))
    assert all(word and all("א" <= char <= "ת" for char in word) for word in words)


def test_every_statutory_word_has_an_audited_negator_or_nonnegator_reading():
    """Every distinct word controls lexical collisions; the allowlist controls misses.

    The neutral following word deliberately does not license contextual אל,
    מנוע, מעולם, אף, אי, חוץ, or פרט. Their real statutory readings are checked
    below. Bare בל is a genuine particle even though its actual statutory
    occurrence happens to be the National Insurance abbreviation בל/50.
    """

    reason_words, source_words = set(), set()
    for word in FIXTURE["vocabulary"]:
        text = word + " חישוב"
        reason = [match.span() for match in hebrew._REASON_NEGATION.finditer(text)]
        source = [match.span() for match in hebrew.link_negations(text)]
        assert reason == source, word
        if any(start == 0 for start, _ in reason):
            reason_words.add(word)
        if any(start == 0 for start, _ in source):
            source_words.add(word)
    expected = set(FIXTURE["probe_negators"])
    assert reason_words == source_words == expected, {
        "unreviewed_negators": sorted(reason_words - expected),
        "missed_negators": sorted(expected - reason_words),
    }


@pytest.mark.parametrize("row", FIXTURE["audited_tokens"], ids=lambda row: row["token"])
def test_statutory_negators_and_commented_exclusions_in_their_real_context(row):
    """Both readers must honor each contextual reading and every lexical exclusion."""

    for context in row["contexts"]:
        text = context["text"]
        assert text[context["start"] : context["end"]] == row["token"]
        assert context["citation"].startswith("il/statute/")
        assert context["explanation"]
        reason = [match.span() for match in hebrew._REASON_NEGATION.finditer(text)]
        source = [match.span() for match in hebrew.link_negations(text)]
        assert reason == source, context
        actual = any(start == context["start"] for start, _ in reason)
        assert actual == context["negator"], context
