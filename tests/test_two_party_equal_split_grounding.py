"""Numeric grounding of an explicit equal split between exactly two parties.

"The employer and the employee shall contribute equally" states that each of
the two parties bears one half. These tests pin the reading (English and
French), the guards that keep it binary, and the invariants of the change:
it never grounds for three or more parties or for plural, distributive or
unnamed parties; it only ever adds the value 0.5 to the grounding stream
(every other emission, and the whole recall inventory, is unchanged); and it
is deterministic.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from unittest import mock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.validator_pipeline import (
    _iter_two_party_equal_split_matches,
    _tokenize_numeric_occurrences_from_text,
    extract_numbers_from_text,
    extract_numeric_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    find_ungrounded_numeric_issues_scoped,
    numeric_value_is_grounded,
)

RW_LAW_2015_05_ART_8_EN = (
    "The employer and the employee shall contribute equally to the pension scheme."
)
RW_LAW_2015_05_ART_8_FR = (
    "Les cotisations au régime de pension sont réparties à parts égales "
    "entre l’employeur et l’employé."
)
RW_PENSION_CITATION = "rw/statute/law-2015-05/organisation-of-pension-schemes"
RW_ORDER_CITATION = "rw/regulation/po-2024-086-01/pension-contribution-rate"


def _has_half(values) -> bool:
    return any(math.isclose(value, 0.5) for value in values)


def _uncached_tokenization(text: str):
    return _tokenize_numeric_occurrences_from_text.__wrapped__(text)


@contextmanager
def _equal_split_reading_disabled():
    with mock.patch.object(
        validator_pipeline,
        "_iter_two_party_equal_split_matches",
        lambda text: iter(()),
    ):
        yield


# --- examples -------------------------------------------------------------


@pytest.mark.parametrize(
    "source_text",
    [
        RW_LAW_2015_05_ART_8_EN,
        RW_LAW_2015_05_ART_8_FR,
        "Both the employer and the employee shall contribute equally.",
        "The employer and employee shall each pay the contribution in equal shares.",
        "The contribution shall be borne equally by the employer and the employee.",
        "The cost shall be shared in equal shares between the landlord and the tenant.",
        "The premium shall be equally divided between the insurer and the insured.",
        "The amount shall be divided between the husband and the wife equally.",
        "The husband and the wife shall bear the costs equally.",
        "L'employeur et le salarié cotisent à parts égales.",
        "La cotisation est partagée entre l'employeur et le travailleur à parts égales.",
        # The source's own line wrap does not break the reading.
        "The employer and the\nemployee shall contribute equally.",
    ],
)
def test_two_party_equal_split_grounds_one_half(source_text):
    assert _has_half(extract_numbers_from_text(source_text))


@pytest.mark.parametrize(
    "source_text",
    [
        # Rwanda Law No 05/2015 itself: survivors' shares are per capita.
        "his/her portion shall be distributed equally among the eligible "
        "deceased’s children referred to under Paragraph 2 of Article 28",
        "if the deceased’s biological and adoptive parents are still alive, the "
        "lump-sum pension allowance shall be shared equally among them.",
        "ils se partagent à parts égales l’allocation unique",
        "all members shall enjoy equal rights on the basis of their contributions.",
        "L’employeur doit également faire enregistrer un nouvel employé",
        # Bare or non-distributive equality.
        "equally",
        "divided equally among the children",
        "treated equally by the employer and the court",
        "L'employeur et l'employé contribuent également au régime.",
        "The employer and the employee shall contribute equal amounts.",
        "The employer and the employee shall contribute on an equal basis.",
        # Three or more parties.
        "The State, the employer and the employee shall contribute equally.",
        "The employer and the employee and the State shall contribute equally.",
        "The employer and the employee shall contribute equally with the State.",
        "shared equally between the employer and the employee and the State",
        "shared equally between the employer, the employee and the State",
        "réparties à parts égales entre l'employeur, l'employé et l'État",
        "réparties à parts égales entre l'employeur et l'employé et l'État",
        # Plural or distributive parties.
        "divided equally between the widow and the children",
        "divided equally between the widow and each child",
        "Employers and employees shall contribute equally.",
        "The employers and the employees shall contribute equally.",
        "réparties à parts égales entre les employeurs et les employés",
        "The mother and the father shall share custody equally among the children.",
    ],
)
def test_equal_split_reading_rejects_non_binary_or_bare_equality(source_text):
    assert list(_iter_two_party_equal_split_matches(source_text)) == []
    assert not _has_half(extract_numbers_from_text(source_text))


def test_rwanda_law_2015_05_reads_only_the_article_8_split():
    body = "\n\n".join(
        [
            "Revenues must be sufficient to cover pension benefits, working "
            "capital and savings.",
            RW_LAW_2015_05_ART_8_EN,
            "Depending on the nature of employment, a differential rate may be "
            "applied to the contribution.",
            RW_LAW_2015_05_ART_8_FR,
            "In case of remarriage of the surviving spouse, his/her portion "
            "shall be distributed equally among the eligible deceased’s children.",
            "Toutefois, si les parents et les parents adoptifs du défunt sont en "
            "vie, ils se partagent à parts égales l’allocation unique.",
        ]
    )
    spans = list(_iter_two_party_equal_split_matches(body))
    assert [body[start:end] for start, end in spans] == [
        "The employer and the employee shall contribute equally",
        "à parts égales entre l’employeur et l’employé",
    ]


def test_equal_split_is_grounding_only_and_carries_no_rate_context():
    tokenization = _uncached_tokenization(RW_LAW_2015_05_ART_8_EN)
    halves = [
        occurrence
        for occurrence in tokenization.grounding
        if math.isclose(occurrence.value, 0.5)
    ]
    assert len(halves) == 1
    assert halves[0].raw == "The employer and the employee shall contribute equally"
    assert not halves[0].has_rate_context
    assert not halves[0].requires_rate_context
    # No recall obligation: an encoding is never required to restate it.
    assert not _has_half(extract_numeric_occurrences_from_text(RW_LAW_2015_05_ART_8_EN))
    # No percentage rescaling: 0.005 is not "half a percent" of the split.
    occurrences = extract_typed_numeric_occurrences_from_text(RW_LAW_2015_05_ART_8_EN)
    assert numeric_value_is_grounded(0.5, occurrences)
    assert not numeric_value_is_grounded(0.005, occurrences)
    assert not numeric_value_is_grounded(50.0, occurrences)


def _rw_pension_module(excerpt: str) -> str:
    return f"""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {RW_ORDER_CITATION}
rules:
  - name: pension_equal_sharing_fraction
    kind: parameter
    dtype: Rate
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: parameter
            source:
              corpus_citation_path: {RW_PENSION_CITATION}
              excerpt: "{excerpt}"
    versions:
      - effective_from: '2015-05-18'
        formula: |-
          0.5
"""


RW_ORDER_TEXT = (
    "Article One: Contribution rate\n\nThe rate of contribution under the "
    "mandatory pension scheme is fixed at the following percentage of the "
    "remuneration subject to contribution: from 1st January 2025: 12%"
)


def test_scoped_grounding_accepts_rw_pension_split_from_its_proof_atom():
    law_text = "\n\n".join(
        [
            "Article 8: Contribution rate",
            RW_LAW_2015_05_ART_8_EN,
            RW_LAW_2015_05_ART_8_FR,
        ]
    )
    issues = find_ungrounded_numeric_issues_scoped(
        _rw_pension_module(RW_LAW_2015_05_ART_8_EN),
        module_source_text=RW_ORDER_TEXT,
        module_citation_path=RW_ORDER_CITATION,
        proof_source_texts={
            RW_ORDER_CITATION: RW_ORDER_TEXT,
            RW_PENSION_CITATION: law_text,
        },
    )
    assert issues == []


def test_scoped_grounding_still_rejects_half_from_a_per_capita_excerpt():
    excerpt = "his/her portion shall be distributed equally among the children"
    issues = find_ungrounded_numeric_issues_scoped(
        _rw_pension_module(excerpt),
        module_source_text=RW_ORDER_TEXT,
        module_citation_path=RW_ORDER_CITATION,
        proof_source_texts={
            RW_ORDER_CITATION: RW_ORDER_TEXT,
            RW_PENSION_CITATION: f"Article 29\n\n{excerpt}.",
        },
    )
    assert issues == [
        "Ungrounded generated numeric literal: 0.5 does not appear as a "
        "substantive numeric value in the source text."
    ]


# --- properties -----------------------------------------------------------

_EN_SINGULAR_PARTIES = [
    "employer",
    "employee",
    "insured person",
    "worker",
    "State",
    "landlord",
    "tenant",
    "surviving spouse",
    "Fund",
    "insurer",
    "beneficiary",
]
_EN_PLURAL_PARTIES = ["employers", "employees", "children", "workers", "heirs"]
_FR_SINGULAR_PARTIES = [
    ("l’", "employeur"),
    ("l’", "employé"),
    ("le ", "salarié"),
    ("la ", "caisse"),
    ("l'", "État"),
    ("le ", "travailleur"),
]

_EN_SUBJECT = "{parties} shall contribute equally to the scheme."
_EN_PASSIVE = "The contribution shall be borne equally by {parties}."
_EN_BETWEEN = "The cost shall be divided between {parties} in equal shares."
_FR_ADVERBIAL = "Les cotisations sont réparties à parts égales entre {parties}."
_FR_SUBJECT = "{parties} cotisent à parts égales."


def _en_list(parties: list[str], *, serial: str) -> str:
    phrases = [f"the {party}" for party in parties]
    if serial == "and":
        return " and ".join(phrases)
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def _fr_list(parties: list[tuple[str, str]], *, serial: str) -> str:
    phrases = [f"{determiner}{noun}" for determiner, noun in parties]
    if serial == "et":
        return " et ".join(phrases)
    return ", ".join(phrases[:-1]) + " et " + phrases[-1]


def _sentence(template: str, parties: str) -> str:
    sentence = template.format(parties=parties)
    return sentence[0].upper() + sentence[1:]


_PROPERTY_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@_PROPERTY_SETTINGS
@given(
    parties=st.lists(
        st.sampled_from(_EN_SINGULAR_PARTIES), min_size=2, max_size=2, unique=True
    ),
    template=st.sampled_from([_EN_SUBJECT, _EN_PASSIVE, _EN_BETWEEN]),
)
def test_property_two_singular_english_parties_ground_one_half(parties, template):
    text = _sentence(template, _en_list(parties, serial="and"))
    assert list(_iter_two_party_equal_split_matches(text))
    assert _has_half(extract_numbers_from_text(text))


@_PROPERTY_SETTINGS
@given(
    parties=st.lists(
        st.sampled_from(_FR_SINGULAR_PARTIES), min_size=2, max_size=2, unique=True
    ),
    template=st.sampled_from([_FR_ADVERBIAL, _FR_SUBJECT]),
)
def test_property_two_singular_french_parties_ground_one_half(parties, template):
    text = _sentence(template, _fr_list(parties, serial="et"))
    assert list(_iter_two_party_equal_split_matches(text))
    assert _has_half(extract_numbers_from_text(text))


@_PROPERTY_SETTINGS
@given(
    parties=st.lists(
        st.sampled_from(_EN_SINGULAR_PARTIES), min_size=3, max_size=6, unique=True
    ),
    serial=st.sampled_from(["and", "comma"]),
    template=st.sampled_from([_EN_SUBJECT, _EN_PASSIVE, _EN_BETWEEN]),
)
def test_property_never_grounds_for_three_or_more_english_parties(
    parties, serial, template
):
    text = _sentence(template, _en_list(parties, serial=serial))
    assert list(_iter_two_party_equal_split_matches(text)) == []


@_PROPERTY_SETTINGS
@given(
    parties=st.lists(
        st.sampled_from(_FR_SINGULAR_PARTIES), min_size=3, max_size=5, unique=True
    ),
    serial=st.sampled_from(["et", "comma"]),
    template=st.sampled_from([_FR_ADVERBIAL, _FR_SUBJECT]),
)
def test_property_never_grounds_for_three_or_more_french_parties(
    parties, serial, template
):
    text = _sentence(template, _fr_list(parties, serial=serial))
    assert list(_iter_two_party_equal_split_matches(text)) == []


@_PROPERTY_SETTINGS
@given(
    singular=st.sampled_from(_EN_SINGULAR_PARTIES),
    plural=st.sampled_from(_EN_PLURAL_PARTIES),
    plural_first=st.booleans(),
    template=st.sampled_from([_EN_SUBJECT, _EN_PASSIVE, _EN_BETWEEN]),
)
def test_property_never_grounds_for_a_plural_party(
    singular, plural, plural_first, template
):
    parties = [plural, singular] if plural_first else [singular, plural]
    text = _sentence(template, _en_list(parties, serial="and"))
    assert list(_iter_two_party_equal_split_matches(text)) == []


@_PROPERTY_SETTINGS
@given(
    party=st.sampled_from(_EN_SINGULAR_PARTIES),
    unnamed=st.sampled_from(
        ["each child", "every heir", "all members", "them", "the others"]
    ),
    template=st.sampled_from([_EN_PASSIVE, _EN_BETWEEN]),
)
def test_property_never_grounds_for_an_unnamed_or_distributive_party(
    party, unnamed, template
):
    text = _sentence(template, f"the {party} and {unnamed}")
    assert list(_iter_two_party_equal_split_matches(text)) == []


@_PROPERTY_SETTINGS
@given(
    preposition=st.sampled_from(["among", "amongst"]),
    parties=st.lists(
        st.sampled_from(_EN_SINGULAR_PARTIES), min_size=2, max_size=2, unique=True
    ),
)
def test_property_equally_among_never_grounds(preposition, parties):
    text = (
        "The amount shall be distributed equally "
        f"{preposition} {_en_list(parties, serial='and')}."
    )
    assert list(_iter_two_party_equal_split_matches(text)) == []


_PROSE_WORDS = st.sampled_from(
    [
        "the",
        "employer",
        "employee",
        "and",
        "shall",
        "contribute",
        "equally",
        "in",
        "equal",
        "shares",
        "between",
        "among",
        "children",
        "à",
        "parts",
        "égales",
        "entre",
        "l’employeur",
        "et",
        "l’employé",
        "half",
        "one",
        "of",
        "per",
        "cent",
        "12%",
        "5",
        "1/2",
        "2025",
        "Article",
        "8",
        ",",
        ".",
        "\n",
    ]
)
_PROSE = st.lists(_PROSE_WORDS, max_size=40).map(" ".join)
_SPLIT_PHRASES = st.sampled_from(
    [
        RW_LAW_2015_05_ART_8_EN,
        RW_LAW_2015_05_ART_8_FR,
        "The contribution shall be borne equally by the employer and the employee.",
    ]
)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=st.one_of(_PROSE, st.text(max_size=120)))
def test_property_reading_only_adds_half_to_grounding(text):
    """Differential: the change adds 0.5 grounding candidates and nothing else."""
    with_reading = _uncached_tokenization(text)
    with _equal_split_reading_disabled():
        without_reading = _uncached_tokenization(text)
    # The recall inventory is untouched.
    assert with_reading.inventory == without_reading.inventory
    # Every prior grounding emission survives unchanged ...
    remaining = list(with_reading.grounding)
    for occurrence in without_reading.grounding:
        assert occurrence in remaining
        remaining.remove(occurrence)
    # ... and the only additions are non-rate 0.5 candidates.
    assert all(
        occurrence.value == 0.5
        and not occurrence.has_rate_context
        and not occurrence.requires_rate_context
        for occurrence in remaining
    )
    # Hence no literal that grounded before is ungrounded now.
    assert extract_numbers_from_text(text) >= {
        occurrence.value for occurrence in without_reading.grounding
    }


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(prefix=_PROSE, phrase=_SPLIT_PHRASES, suffix=_PROSE)
def test_property_adding_a_split_sentence_grounds_half_and_keeps_its_own_values(
    prefix, phrase, suffix
):
    # The neighbouring paragraph ends its sentence, as statute text does. (An
    # unterminated "parts" or "section" line lets the pre-existing structural
    # cleaner read the next paragraph's "The" as a cross-reference and blank
    # it, which removes the determiner the reading needs; that is cleaner
    # behaviour this change neither causes nor relies on.)
    text = f"{prefix}.\n\n{phrase}\n\n{suffix}"
    assert _has_half(extract_numbers_from_text(text))
    # The split sentence carries no number of its own, so the reading adds
    # exactly the half: the values of the surrounding text are those it had
    # with the reading switched off.
    with _equal_split_reading_disabled():
        before = {
            occurrence.value for occurrence in _uncached_tokenization(text).grounding
        }
    after = {occurrence.value for occurrence in _uncached_tokenization(text).grounding}
    assert after - before <= {0.5}
    assert before <= after


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=st.one_of(_PROSE, _SPLIT_PHRASES), other=_PROSE)
def test_property_reading_is_deterministic(text, other):
    first = list(_iter_two_party_equal_split_matches(text))
    list(_iter_two_party_equal_split_matches(other))
    assert list(_iter_two_party_equal_split_matches(text)) == first
    assert _uncached_tokenization(text) == _uncached_tokenization(text)
