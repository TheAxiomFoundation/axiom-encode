"""Differential for English month names: the recognizer on vs off.

The branch reads an English month name used as a date (``June 1939``,
``December 1, 1940``, ``the month of July``) as its month number, the way
#1745 reads Hebrew ones. Turning ``_iter_english_gregorian_month_matches`` off
restores the Hebrew-only reading. These tests compute every validator verdict
the reading can touch, with it on and off, and require the set of changed
verdicts to equal an explicit list:

* over each RuleSpec fixture in the repository (and its ``provision.txt``
  source where one exists). The expected list is empty: no fixture module
  compares a calendar month with a literal that needs grounding, so a month
  number has nothing there to ground, although well over a hundred fixture
  sources date English month names (the corpus guard below);
* over synthetic controls: faithful English Part D must change, and each
  guard must leave its verdicts alone (an annualising 12, a count of months,
  the modal "May", a month beside "percent", the Hebrew rendering).

The numeric tokenizer is cached, so the cache is cleared before each side.

Invariant, as for #1745: the reading may only *ground*; it never adds a recall
obligation. So with it on, every source numeric inventory is unchanged, and
every changed verdict removes issues rather than adding any.
"""

from __future__ import annotations

import pytest

from axiom_encode.harness import validator_pipeline as vp
from tests.test_english_month_grounding import (
    DATED_DECEMBER_TEXT_EN,
    ENGLISH_PART_D_TEXT,
)
from tests.test_hebrew_month_and_index_selector_grounding import (
    PART_D_TEXT,
    _month_module,
    _part_d_birth_month_module,
)
from tests.test_hebrew_month_index_selector_fixture_differential import (
    REPO_ROOT,
    _assert_only_removes_and_keeps_inventory,
    _changed,
    _fixture_verdicts,
    _rulespec_fixtures,
    _verdicts_for,
)

# Every (fixture, verdict) the English reading changes. Empty: see above.
EXPECTED_CHANGED_FIXTURE_VERDICTS: dict[str, list[str]] = {}
# 194 fixture sources date at least one English month name (2026-11-05).
MIN_FIXTURE_SOURCES_WITH_DATED_ENGLISH_MONTHS = 100

# One synthetic module per behaviour, with the verdicts it must change.
CONTROL_CASES: dict[str, tuple[str, str]] = {
    # Part D compared with a numeric birth month, in faithful English: the
    # month names ground the month comparisons, as the Hebrew names do.
    "english-part-d": (_part_d_birth_month_module(), ENGLISH_PART_D_TEXT),
    # A dated month beside an unrelated annualising literal: a month number
    # grounds only a month comparison.
    "english-unrelated-literal": (
        _month_module("monthly_amount * 12 + 500"),
        DATED_DECEMBER_TEXT_EN,
    ),
    # A count of months is a quantity, not a month.
    "english-month-count": (
        _month_module("months_employed >= 6"),
        "Until June 1939 | 65",
    ),
    # The modal "May" dates nothing.
    "english-modal-may": (
        _month_module("birth_month >= 5"),
        "Persons Aged 18 May Apply",
    ),
    # A month beside "percent" grounds no rate scaled from its number.
    "english-rate-neighbour": (
        _month_module("birth_month >= 0.12"),
        "July to December percent",
    ),
    # The Hebrew rendering: its months ground with English on or off.
    "hebrew-part-d": (_part_d_birth_month_module(), PART_D_TEXT),
}
EXPECTED_CHANGED_CONTROL_VERDICTS: dict[str, list[str]] = {
    "english-part-d": ["ungrounded"],
}


def _control_verdicts() -> dict[str, dict[str, object]]:
    return {
        name: _verdicts_for(content, source)
        for name, (content, source) in CONTROL_CASES.items()
    }


def _english_on_and_off(compute):
    """``compute()`` with the English reading off, then on, on a cold cache."""
    vp._tokenize_numeric_occurrences_from_text.cache_clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(vp, "_iter_english_gregorian_month_matches", lambda *_a: [])
        before = compute()
    vp._tokenize_numeric_occurrences_from_text.cache_clear()
    after = compute()
    vp._tokenize_numeric_occurrences_from_text.cache_clear()
    return before, after


@pytest.fixture(scope="module")
def fixture_pair():
    return _english_on_and_off(_fixture_verdicts)


@pytest.fixture(scope="module")
def control_pair():
    return _english_on_and_off(_control_verdicts)


def test_fixture_sources_date_english_months():
    # Guard against a vacuous differential: the reading has months to read.
    dated = [
        path
        for path in _rulespec_fixtures()
        if (path.parent / "provision.txt").is_file()
        and vp._iter_english_gregorian_month_matches(
            (path.parent / "provision.txt").read_text(encoding="utf-8")
        )
    ]
    assert len(dated) >= MIN_FIXTURE_SOURCES_WITH_DATED_ENGLISH_MONTHS


def test_no_fixture_compares_a_month_with_a_literal():
    # Why the expected fixture list is empty: no module has a literal for a
    # month number to ground.
    assert {
        path.relative_to(REPO_ROOT).as_posix(): sorted(values)
        for path in _rulespec_fixtures()
        if (
            values := vp._month_comparison_literal_values(
                path.read_text(encoding="utf-8")
            )
        )
    } == {}


def test_fixture_differential_lists_every_changed_verdict(fixture_pair):
    assert _changed(*fixture_pair) == EXPECTED_CHANGED_FIXTURE_VERDICTS


def test_fixture_verdicts_only_remove_issues_and_keep_inventories(fixture_pair):
    _assert_only_removes_and_keeps_inventory(*fixture_pair)


def test_control_differential_lists_every_changed_verdict(control_pair):
    assert _changed(*control_pair) == EXPECTED_CHANGED_CONTROL_VERDICTS


def test_control_verdicts_only_remove_issues_and_keep_inventories(control_pair):
    _assert_only_removes_and_keeps_inventory(*control_pair)


def test_english_part_d_grounds_like_hebrew_with_english_on(control_pair):
    before, after = control_pair
    # Off, English Part D reports the birth months review finding N8 listed.
    assert before["english-part-d"]["ungrounded"] != []
    assert after["english-part-d"]["ungrounded"] == []
    assert after["hebrew-part-d"]["ungrounded"] == []
    # The guards report their literal either way.
    for side in (before, after):
        assert any(
            "12 does not appear" in issue
            for issue in side["english-unrelated-literal"]["ungrounded"]
        )
        for name, literal in (
            ("english-month-count", "6"),
            ("english-modal-may", "5"),
            ("english-rate-neighbour", "0.12"),
        ):
            assert any(
                f"{literal} does not appear" in issue
                for issue in side[name]["ungrounded"]
            ), (name, side[name]["ungrounded"])
