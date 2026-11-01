"""Round 19 (#1882): the audited Hebrew rendering table, closed by grammar.

An audit derived a finite table from the English guard's code and rendered
each entry in Hebrew: 1,123 frames in the reason and source positions, 49
special-route frames and 41 computation frames. The fixture
(``fixtures/hebrew_rendering_audit.json``) holds those generated frames only,
with the Hebrew verdict each must return after round 19 and the verdict of
its canonical English, the faithful rendering in the wording the English
guard recognises. No Hebrew reason or source passes where its English fails;
``stricter`` labels each Hebrew refusal where English accepts.

The rest of this module states the rules for all inputs: every guarded head
under every ordinary prefix chain, joint and quantifier negates a source link
over a computation complement and nothing over a neutral word; the reason
reader reads every head as it did before the round; and a citation never ends
before a digit or a letter English reads.
"""

from __future__ import annotations

import functools
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import hebrew_negators as negators
from axiom_encode.harness import hebrew_source_text as hebrew
from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.test_source_completeness_hebrew import _accepted, _english_accepted

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/hebrew_rendering_audit.json").read_text()
)
_DEFAULT_SOURCE = FIXTURE["defaults"]["reason_source"]
_DEFAULT_REASON = FIXTURE["defaults"]["source_reason"]


def _route_accepted(row: dict, reason: str, source: str) -> bool:
    """The CI verdict for a route row's own citation and branch."""

    citation = row["citation"]
    path = tuple(row["path"])
    output = (
        completeness._rulespec_target_base(citation)
        + "/"
        + "/".join(path)
        + "#audit_deferral"
    )
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {
                "source_verification": {"corpus_citation_path": citation},
                "deferred_outputs": [{"output": output, "reason": reason}],
            },
            "rules": [],
        },
        allow_unicode=True,
        sort_keys=False,
    )
    result = completeness.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    assert path in [branch.path for branch in result.branches], row["id"]
    return not any(
        issue.startswith(
            ("[complete-source-unit:deferral]", "[complete-source-unit:structure]")
        )
        for issue in result.issues
    )


def _verdicts(row: dict, routes: bool) -> tuple[list[bool], list[bool]]:
    """Hebrew and English verdicts, each ``[reason position, source position]``."""

    reason_source = row.get("reason_source", _DEFAULT_SOURCE)
    source_reason = row.get("source_reason", _DEFAULT_REASON)
    if routes:
        accept = functools.partial(_route_accepted, row)
        english = accept
    else:
        accept = functools.partial(_accepted, "a")
        english = functools.partial(_english_accepted, "a")
    return (
        [
            accept(row["reason"][0], reason_source[0]),
            accept(source_reason[0], row["source"][0]),
        ],
        [
            english(row["reason"][1], reason_source[1]),
            english(source_reason[1], row["source"][1]),
        ],
    )


def _classes(key: str) -> list[str]:
    return sorted({row["cls"] for row in FIXTURE[key]})


def test_the_fixture_is_the_audited_table():
    assert len(FIXTURE["main"]) == 1123
    assert len(FIXTURE["routes"]) == 90
    # No row may pass in Hebrew where its canonical English fails.
    for row in FIXTURE["main"] + FIXTURE["routes"]:
        for hebrew_verdict, english_verdict in zip(row["hebrew"], row["english"]):
            assert not (hebrew_verdict and not english_verdict), row["id"]
        stricter = any(
            not hebrew_verdict and english_verdict
            for hebrew_verdict, english_verdict in zip(row["hebrew"], row["english"])
        )
        assert stricter is bool(row.get("stricter")), row["id"]
    labelled = {
        row["id"]
        for row in FIXTURE["routes"]
        if row.get("stricter") == "english-only-route-refuses-hebrew-script"
    }
    assert labelled == {"S013", "S025", "S026", "S038"}


@pytest.mark.parametrize("cls", _classes("main"))
def test_rendering_rows_get_their_round_19_verdicts(cls):
    for row in FIXTURE["main"]:
        if row["cls"] == cls:
            assert _verdicts(row, routes=False) == (row["hebrew"], row["english"]), row[
                "id"
            ]


@pytest.mark.parametrize("cls", _classes("routes"))
def test_route_and_computation_rows_get_their_round_19_verdicts(cls):
    """English-only routes refuse Hebrew script (S038 and three others, labelled)."""

    for row in FIXTURE["routes"]:
        if row["cls"] == cls:
            assert _verdicts(row, routes=True) == (row["hebrew"], row["english"]), row[
                "id"
            ]


# --- The reason reader is unchanged ------------------------------------------

# sha256 of ``_REASON_NEGATION.pattern`` and ``_NEGATED_SIGNAL_PREFIX.pattern``
# at 8cb27cfd7, the commit before round 19. The same pattern reads every input
# the same way, so the split leaves the reason reader's negation unchanged.
_REASON_PATTERN_AT_8CB27 = (
    "4623cd66ac5e78e301c703cf0894f348d0a4d594e6d84bfda058fd6d510be109"
)
_SIGNAL_PREFIX_PATTERN_AT_8CB27 = (
    "96297b982678a1c457d339904d219f23d52fcbc016b30dcf78c2fa822de25ea0"
)


def test_the_reason_negation_is_the_pattern_it_was():
    def digest(pattern: str) -> str:
        return hashlib.sha256(pattern.encode()).hexdigest()

    assert digest(hebrew._REASON_NEGATION.pattern) == _REASON_PATTERN_AT_8CB27
    assert digest(hebrew._NEGATED_SIGNAL_PREFIX.pattern) == (
        _SIGNAL_PREFIX_PATTERN_AT_8CB27
    )
    assert hebrew._NEGATIONS is not hebrew._REASON_NEGATION


@pytest.mark.parametrize(
    "frames",
    [
        FIXTURE["reason_at_8cb27cfd7"][index : index + 30]
        for index in range(0, len(FIXTURE["reason_at_8cb27cfd7"]), 30)
    ],
)
def test_every_head_in_a_reason_gets_its_verdict_from_before_round_19(frames):
    """Each round-19 head before, inside and after a reason's missing claim."""

    for reason, verdict in frames:
        assert _accepted("a", reason, _DEFAULT_SOURCE[0]) is verdict, reason


# --- Source heads, for every prefix chain, joint and quantifier --------------

_DIRECT_HEADS = (
    tuple(head for head in negators.SOURCE_PRIVATIVE_HEADS if head != "מחוסר")
    + negators.SOURCE_CONSTRUCT_HEADS
)
_JOINTS = (" ", "־", "-")
_QUANTIFIERS = ("", *negators.SOURCE_QUANTIFIERS)
_COMPUTATION_SAMPLE = (
    "חישוב",
    "החישוב",
    "קביעה",
    "קביעות",
    "החלת",
    "יישום",
    "הטלת",
    "חיוב",
    "תחולת",
    "היווצרות",
    "calculation",
    "determination",
)
# A neutral word in the same slot: the frame then states nothing negative.
_NEUTRAL = "מדויק"


def _direct_frame(
    prefix: str, head: str, joint: str, quantifier: str, noun: str
) -> str:
    complement = f"{quantifier} {noun}" if quantifier else noun
    return f"(א) התשלום יבוצע {prefix}{head}{joint}{complement} בהתאם לסעיף 2."


def _from_frame(prefix: str, head: str, joint: str, quantifier: str, noun: str) -> str:
    complement = f"{quantifier} {noun}" if quantifier else noun
    return f"(א) הרשות {prefix}{head}{joint}מן {complement} בהתאם לסעיף 2."


def _negation_at(text: str, start: int) -> bool:
    return any(match.start() == start for match in hebrew._NEGATIONS.finditer(text))


def test_every_guarded_head_negates_a_link_over_a_computation_complement():
    """The product, read by the source negations; the neutral frame reads none."""

    for prefix in negators.ORDINARY_PREFIX_CHAINS:
        for joint in _JOINTS:
            for quantifier in _QUANTIFIERS:
                for noun in _COMPUTATION_SAMPLE:
                    if quantifier and noun.startswith("ה"):
                        continue
                    for head in _DIRECT_HEADS:
                        text = _direct_frame(prefix, head, joint, quantifier, noun)
                        start = text.index(prefix + head, 4)
                        assert _negation_at(text, start), text
                        assert not hebrew._REASON_NEGATION.search(text), text
                    for head in negators.SOURCE_FROM_HEADS:
                        text = _from_frame(prefix, head, joint, quantifier, noun)
                        assert _negation_at(text, text.index(prefix + head, 4)), text
                    neutral = _direct_frame(prefix, _NEUTRAL, joint, quantifier, noun)
                    assert not list(hebrew._NEGATIONS.finditer(neutral)), neutral


def test_every_clause_wide_form_negates_under_every_prefix_chain():
    """A form with no statutory homograph negates whatever follows it."""

    for prefix in negators.ORDINARY_PREFIX_CHAINS:
        for form in negators.SOURCE_CLAUSE_NEGATORS:
            text = f"(א) {prefix}{form} התשלום בהתאם לסעיף 2."
            assert _negation_at(text, 4), text
        for phrase in negators.SOURCE_CLAUSE_PHRASES:
            head, tail = phrase.split()
            glued = tail if len(tail) > 1 else tail + "חישוב"
            text = f"(א) {prefix}{head} {glued} בהתאם לסעיף 2."
            assert _negation_at(text, 4), text


@settings(max_examples=40, deadline=None)
@given(
    prefix=st.sampled_from(negators.ORDINARY_PREFIX_CHAINS),
    head=st.sampled_from(_DIRECT_HEADS),
    joint=st.sampled_from(_JOINTS),
    quantifier=st.sampled_from(_QUANTIFIERS),
    noun=st.sampled_from([noun for noun in _COMPUTATION_SAMPLE if noun[0] != "ה"]),
)
def test_a_guarded_head_refuses_the_source_and_a_neutral_word_does_not(
    prefix, head, joint, quantifier, noun
):
    """The whole CI method: the head's frame refuses, the neutral frame accepts."""

    assert not _accepted(
        "a", _DEFAULT_REASON[0], _direct_frame(prefix, head, joint, quantifier, noun)
    )
    assert _accepted(
        "a",
        _DEFAULT_REASON[0],
        _direct_frame(prefix, _NEUTRAL, joint, quantifier, noun),
    )


@pytest.mark.parametrize(
    ("source", "accepted"),
    (
        # Causal מחוסר ("for lack of"), as round 18 pinned it.
        ("(א) התשלום יבוצע מחוסר חישוב בהתאם לסעיף 2.", True),
        ("(א) התשלום קבוע ומחוסר חישוב בהתאם לסעיף 2.", True),
        ("(א) התשלום יבוצע מחוסר כל חישוב בהתאם לסעיף 2.", True),
        ("(א) התשלום יבוצע מחוסרו של חישוב בהתאם לסעיף 2.", True),
        # The adjective: after a copula, in a relative clause, definite, inflected.
        ("(א) התשלום יהיה מחוסר חישוב בהתאם לסעיף 2.", False),
        ("(א) התשלום שמחוסר חישוב ישולם בהתאם לסעיף 2.", False),
        ("(א) התשלום מחוסר החישוב ישולם בהתאם לסעיף 2.", False),
        ("(א) הגמלה מחוסרת קביעה תשולם בהתאם לסעיף 2.", False),
        # A concessive's object concedes it; the payment is still computed.
        ("(א) למרות אי קביעתו מראש, התשלום מחושב בהתאם לסעיף 2.", True),
        ("(א) חרף אי קביעתו מראש, התשלום מחושב בהתאם לסעיף 2.", True),
        ("(א) התשלום נקבע תוך אי חישוב בהתאם לסעיף 2.", False),
        # Bare pronominal absence stays a noun; with של it is "without".
        ("(א) בהיעדרו, יחושב התשלום בהתאם לסעיף 2.", True),
        ("(א) התשלום יבוצע בהיעדרו של חישוב בהתאם לסעיף 2.", False),
        ("(א) התשלום יבוצע בהעדרן המוחלט של קביעות בהתאם לסעיף 2.", False),
        # Statutory senses with no computation complement.
        ("(א) תגמול לחטופים ונעדרים יחושב בהתאם לסעיף 2.", True),
        ("(א) הפטור ממס יחושב בהתאם לסעיף 2.", True),
        ("(א) הוטל עליו כופר כסף בהתאם לסעיף 2.", True),
        ("(א) התשלום במקום אחר יחושב בהתאם לסעיף 2.", True),
        ("(א) אי תשלום בהתאם לסעיף 2.", True),
        # The other order and a claim complement.
        ("(א) החישוב נפקד מן התשלום המחושב בהתאם לסעיף 2.", False),
        ("(א) הפקיד כופר בטענה שהתשלום מחושב בהתאם לסעיף 2.", False),
        # Postposed denial needs a claim with a content clause.
        ("(א) הטענה שהתשלום מחושב בהתאם לסעיף 2 אינה נכונה.", False),
        ("(א) הדוח שהוגש בהתאם לסעיף 2 אינו נכון.", True),
        ("(א) האפשרות לחשב בהתאם לסעיף 2 נבצרה מן הרשות.", False),
        # Disclaimers.
        ("(א) סעיף 2 נותר בעינו והתשלום מחושב בהתאם לסעיף 2.", False),
        ("(א) סעיף 2 נשאר בתוקפו והתשלום מחושב בהתאם לסעיף 2.", True),
    ),
)
def test_each_rule_keeps_its_boundary(source, accepted):
    assert _accepted("a", _DEFAULT_REASON[0], source) is accepted


@pytest.mark.parametrize(
    ("reason", "accepted"),
    (
        ("חסר מידע פרט אם סעיף 2 זמין.", False),
        ("חסר מידע, מצד שני סעיף 2 זמין.", False),
        ("חסר מידע כל עוד סעיף 2 זמין.", False),
        ("חסר סעיף 2", True),
    ),
)
def test_reason_adversatives(reason, accepted):
    assert _accepted("a", reason, _DEFAULT_SOURCE[0]) is accepted


def test_a_conditional_while_in_a_source_still_links():
    source = "(א) כל עוד התנאים מתקיימים, התשלום מחושב בהתאם לסעיף 2."
    assert _accepted("a", _DEFAULT_REASON[0], source)


# --- Citation terminals ------------------------------------------------------

_SOURCE = "(א) ישולם לפי סעיף 2."


@pytest.mark.parametrize("suffix", ("K", "İ", "ı", "ſ", "a", "٣", "٣a", "𝟎", "𝟎א"))
def test_a_citation_never_ends_before_a_digit_or_a_letter_english_reads(suffix):
    assert not _accepted("a", f"חסר סעיף 2{suffix}", _SOURCE)
    assert not _accepted("a", "חסר סעיף 2", f"(א) ישולם לפי סעיף 2{suffix}.")
    assert not _english_accepted(
        "a", f"Missing section 2{suffix}.", "(a) Payment shall be made under section 2."
    )


@pytest.mark.parametrize("suffix", ("Ä", "Ö", "Ü", "ä", "ö", "ü", "ß", "ẞ"))
def test_german_letters_end_a_reason_citation_but_not_a_source_citation(suffix):
    assert _accepted("a", f"חסר סעיף 2{suffix}", _SOURCE)
    assert not _accepted("a", "חסר סעיף 2", f"(א) ישולם לפי סעיף 2{suffix}.")


def test_every_unicode_digit_continues_a_citation():
    """No reader stops a section number before a decimal digit of any script."""

    digits = [
        chr(code)
        for code in range(sys.maxunicode + 1)
        if unicodedata.category(chr(code)) == "Nd"
    ]
    assert len(digits) > 700
    for digit in digits:
        for tail in ("", "a", "אבג", "'", '"', "”"):
            text = f"חסר סעיף 2{digit}{tail}"
            for pattern in (
                hebrew.SECTION_DEPENDENCY,
                hebrew.SOURCE_SECTION_DEPENDENCY,
            ):
                for match in pattern.finditer(text):
                    assert match.group("section") != "2", ascii(text)
