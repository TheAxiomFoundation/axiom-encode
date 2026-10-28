"""Round 16: every possible short prefix-only spelling has a primary lexical pin."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path

from axiom_encode.harness import _hebrew_prefix_words as words

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = json.loads((FIXTURES / "hebrew_prefix_lexicon.json").read_text())


def test_primary_lexicon_receipt_and_complete_finite_partition():
    assert FIXTURE["source"]["commit"] == words.HSPELL_COMMIT
    assert FIXTURE["source"]["archive_sha256"] == words.HSPELL_ARCHIVE_SHA256
    assert FIXTURE["source"]["archive_bytes"] == 570733
    assert FIXTURE["source"]["license"] == "AGPL-3.0-only"
    assert FIXTURE["dictionary_words"] == 469509
    assert FIXTURE["prefix_mask_pairs"] == 242
    assert len(FIXTURE["prefix_masks"]) == 242
    alphabet = FIXTURE["alphabet"]
    assert alphabet == "והבכלמשד"
    expected = sorted(
        "".join(chain)
        for length in range(1, 4)
        for chain in itertools.product(alphabet, repeat=length)
    )
    assert FIXTURE["candidate_count"] == len(expected) == 584
    assert [row["token"] for row in FIXTURE["candidates"]] == expected
    assert sorted(FIXTURE["candidate_stem_masks"]) == expected
    assert Counter(row["category"] for row in FIXTURE["candidates"]) == {
        "single_letter_prefix": 8,
        "lexical_word": 286,
        "proper_name": 11,
        "acronym": 8,
        "nonlexical_prefix_chain": 271,
    }
    for name, key in (("LICENSE", "license_sha256"), ("COPYING", "copying_sha256")):
        source = FIXTURES / "licenses" / "hspell" / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == FIXTURE["source"][key]
        # The production Python module is included in the wheel. Its marked
        # comments retain the exact upstream notice and full license bytes.
        module_source = Path(words.__file__).read_text()
        embedded = re.search(
            rf"# HSPELL-UPSTREAM-{name}-BEGIN\n(.*?)# HSPELL-UPSTREAM-{name}-END\n",
            module_source,
            re.DOTALL,
        )
        assert embedded
        restored = (
            "\n".join(
                "" if line == "#" else line.removeprefix("# ")
                for line in embedded.group(1).splitlines()
            )
            + "\n"
        )
        assert hashlib.sha256(restored.encode()).hexdigest() == FIXTURE["source"][key]


def test_generated_whole_words_equal_all_primary_dictionary_readings():
    recognized = set()
    for row in FIXTURE["candidates"]:
        token = row["token"]
        expected_splits = {
            (prefix, token[len(prefix) :])
            for prefix, prefix_mask in FIXTURE["prefix_masks"].items()
            if token.startswith(prefix)
            and prefix_mask
            & FIXTURE["candidate_stem_masks"].get(token[len(prefix) :], 0)
        }
        assert {(r["prefix"], r["stem"]) for r in row["readings"]} == expected_splits
        if row["readings"]:
            recognized.add(token)
        ordinary = False
        for reading in row["readings"]:
            assert reading["prefix"] + reading["stem"] == token
            assert reading["compatible_mask"] > 0
            assert reading["analyses"]
            ordinary |= any(not value["proper_name"] for value in reading["analyses"])
        if row["category"] == "lexical_word":
            assert ordinary
        elif row["category"] == "proper_name":
            assert row["readings"] and not ordinary
        elif row["category"] == "acronym":
            assert not row["readings"] and row["quoted_acronym_analyses"]
        else:
            assert not row["readings"]
    assert recognized == words.HSPELL_PREFIX_WORDS
    assert len(recognized) == 297
    assert all(2 <= len(word) <= 3 for word in recognized)
    # Actual primary noun/verb forms absent from the statute snapshot. These
    # are checks of the dictionary derivation, not the dictionary definition.
    assert set("בד כלב בלש מל הלל הד הוד דוד משה ובו".split()) <= recognized


def test_primary_proper_names_and_quoted_acronyms_remain_distinct():
    by_token = {row["token"]: row for row in FIXTURE["candidates"]}
    assert by_token["לוב"]["category"] == "proper_name"  # Libya
    assert by_token["לוד"]["category"] == "proper_name"  # Lod
    assert by_token["ממ"]["category"] == "acronym"  # e.g. quoted מ"מ
    assert by_token["וכו"]["category"] == "acronym"
    for token in ("דוד", "משה"):
        assert by_token[token]["category"] == "lexical_word"
        readings = by_token[token]["readings"]
        assert any(
            value["proper_name"] for row in readings for value in row["analyses"]
        )
        assert any(
            not value["proper_name"] for row in readings for value in row["analyses"]
        )
