"""Synthetic grammar and work-budget regressions for numeral prefix reuse."""

import pytest

from axiom_encode.harness import validator_pipeline as pipeline


@pytest.mark.parametrize(
    ("words", "start", "expected"),
    [
        ([], 0, None),
        (["חמישה"], 1, None),
        (["חמישה"], 0, (1, 5.0, {"unit"})),
        (["ובכחמישה"], 0, (1, 5.0, {"unit"})),
        (["בשיעור", "של", "ובכחמישה"], 2, (1, 5.0, {"unit"})),
        (["בשיעור", "של", "חמישה"], 0, None),
        (["שני"], 0, None),
        (["שני", "אלפים"], 0, (2, 2000.0, {"thousand"})),
        (["שתי", "מאות"], 0, (2, 200.0, {"hundred"})),
        (["שנים"], 0, None),
        (["שנים", "עשר"], 0, (2, 12.0, {"teen"})),
        (["אפס"], 0, (1, 0.0, {"unit"})),
        (["חצי", "מיליון"], 0, (2, 500000.0, {"million"})),
        (["מיליון", "וחצי"], 0, (2, 1500000.0, {"million", "fraction"})),
        (
            "שלושה מיליון ושני אלפים וחמש מאות".split(),
            0,
            (6, 3002500.0, {"million", "thousand", "hundred"}),
        ),
    ],
)
def test_initial_word_reuse_preserves_grammar(words, start, expected):
    assert pipeline._parse_hebrew_number_run(words, start) == expected


@pytest.mark.parametrize("money_context", [None, False, True])
def test_non_numeral_start_still_rejects_in_every_money_context(money_context):
    assert (
        pipeline._parse_hebrew_number_run(
            ["בשיעור", "של", "חמישה"], money_context=money_context
        )
        is None
    )


def test_percentage_workload_does_not_repeat_prefix_normalization(monkeypatch):
    strip_prefix = pipeline._strip_hebrew_number_prefix
    calls = 0

    def counted_strip(word, vocabulary):
        nonlocal calls
        calls += 1
        return strip_prefix(word, vocabulary)

    monkeypatch.setattr(pipeline, "_strip_hebrew_number_prefix", counted_strip)
    phrase = "בשיעור של חמישה אחוזים מההכנסה; "
    count = 100
    matches = pipeline._iter_hebrew_percent_phrase_matches(phrase * count)
    start = phrase.index("חמישה")
    end = phrase.index("אחוזים") + len("אחוזים")
    assert matches == [
        ((index * len(phrase) + start, index * len(phrase) + end), 0.05)
        for index in range(count)
    ]
    # The old grammar repeated normalization 67 times per phrase while
    # probing rejected prose and several scale tiers. Bound the work
    # independently of processor speed; the existing two-second test stays.
    assert calls <= 8 * count
