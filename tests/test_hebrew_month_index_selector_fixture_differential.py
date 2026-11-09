"""Differential for month and selector grounding: recognizers on vs off.

The branch adds two recognizers: Hebrew month names read as dates
(``_iter_hebrew_gregorian_month_matches``) and structural index selectors
(``_rulespec_index_selectors``). Turning both off restores the old verdicts.
These tests compute every validator verdict the recognizers can touch, with
them on and off, and require the set of changed verdicts to equal an explicit
list:

* over each RuleSpec fixture in the repository (and its ``provision.txt``
  source where one exists), where the expected list is empty because no
  fixture carries a Hebrew date or a non-band-named index selector;
* over three synthetic control modules, one per behaviour, so that each
  recognizer has something to detect and the test fails if it stops working
  or widens.

The numeric tokenizer is cached, so the cache is cleared before each side;
otherwise the second side would reuse the first side's tokenization and the
month half could never differ.

Invariant: the recognizers may only *ground* or *exempt*; they never add a
recall obligation. So with them on, every source numeric inventory is
unchanged, and every changed verdict removes issues rather than adding any.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom_encode.harness import validator_pipeline as vp
from tests.test_hebrew_month_and_index_selector_grounding import (
    DATED_DECEMBER_TEXT,
    PART_D_TEXT,
    SIXTEEN_ROWS,
    _cohort_selector_module,
    _month_module,
    _part_d_birth_month_module,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOTS = ("tests/fixtures", "benchmarks")

# Every (fixture, verdict) the recognizers change. Empty: no repository
# fixture carries Hebrew month dates or a non-band-named index selector.
EXPECTED_CHANGED_FIXTURE_VERDICTS: dict[str, list[str]] = {}

# One synthetic module per behaviour, with the verdicts it must change.
CONTROL_CASES: dict[str, tuple[str, str]] = {
    # Part D compared with a numeric birth month: the month names ground the
    # month comparisons.
    "dated-month-comparison": (_part_d_birth_month_module(), PART_D_TEXT),
    # A dated month beside an unrelated annualising literal: nothing changes,
    # because a month number grounds only a month comparison.
    "dated-month-unrelated-literal": (
        _month_module("monthly_amount * 12 + 500"),
        DATED_DECEMBER_TEXT,
    ),
    # A private cohort selector read only as a table subscript: its row
    # labels stop being grounding values and embedded scalars.
    "index-selector": (_cohort_selector_module(SIXTEEN_ROWS), PART_D_TEXT),
}
EXPECTED_CHANGED_CONTROL_VERDICTS: dict[str, list[str]] = {
    "dated-month-comparison": ["ungrounded"],
    "index-selector": ["embedded_scalars", "grounding_values", "ungrounded"],
}


def _rulespec_fixtures() -> list[Path]:
    paths = []
    for root in FIXTURE_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.yaml")):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "rulespec/v1" in text:
                paths.append(path)
    return paths


def _verdicts_for(content: str, source: str | None) -> dict[str, object]:
    pipeline = vp.ValidatorPipeline.__new__(vp.ValidatorPipeline)
    try:
        embedded = pipeline._collect_embedded_scalar_literals(content)
    except Exception as exc:  # noqa: BLE001 - recorded, compared on both sides
        embedded = f"error: {type(exc).__name__}"
    verdicts: dict[str, object] = {
        "grounding_values": sorted(vp.extract_grounding_values(content)),
        "embedded_scalars": embedded,
    }
    if source is not None:
        verdicts["ungrounded"] = vp.find_ungrounded_numeric_issues(
            content, source, authoritative_source_text=source
        )
        verdicts["half_up_helpers"] = vp.source_backed_half_up_rounding_helper_count(
            content, source
        )
        verdicts["source_inventory"] = vp.extract_numeric_occurrences_from_text(source)
    return json.loads(json.dumps(verdicts, default=str))


def _fixture_verdicts() -> dict[str, dict[str, object]]:
    verdicts = {}
    for path in _rulespec_fixtures():
        provision = path.parent / "provision.txt"
        source = provision.read_text(encoding="utf-8") if provision.is_file() else None
        verdicts[path.relative_to(REPO_ROOT).as_posix()] = _verdicts_for(
            path.read_text(encoding="utf-8"), source
        )
    return verdicts


def _control_verdicts() -> dict[str, dict[str, object]]:
    return {
        name: _verdicts_for(content, source)
        for name, (content, source) in CONTROL_CASES.items()
    }


def _on_and_off(compute):
    """``compute()`` with both recognizers off, then on, each on a cold cache."""
    vp._tokenize_numeric_occurrences_from_text.cache_clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(vp, "_iter_hebrew_gregorian_month_matches", lambda *_a, **_k: [])
        patch.setattr(vp, "_rulespec_index_selectors", lambda *_a, **_k: frozenset())
        before = compute()
    vp._tokenize_numeric_occurrences_from_text.cache_clear()
    after = compute()
    vp._tokenize_numeric_occurrences_from_text.cache_clear()
    return before, after


def _changed(before, after) -> dict[str, list[str]]:
    changed = {
        name: sorted(
            key for key in after[name] if after[name][key] != before[name].get(key)
        )
        for name in after
    }
    return {name: keys for name, keys in changed.items() if keys}


def _assert_only_removes_and_keeps_inventory(before, after) -> None:
    for name, verdicts in after.items():
        if "source_inventory" in verdicts:
            assert verdicts["source_inventory"] == before[name]["source_inventory"]
        if "ungrounded" in verdicts:
            assert set(verdicts["ungrounded"]) <= set(before[name]["ungrounded"])
        if isinstance(verdicts["embedded_scalars"], list):
            assert {tuple(item) for item in verdicts["embedded_scalars"]} <= {
                tuple(item) for item in before[name]["embedded_scalars"]
            }


@pytest.fixture(scope="module")
def fixture_pair():
    return _on_and_off(_fixture_verdicts)


@pytest.fixture(scope="module")
def control_pair():
    return _on_and_off(_control_verdicts)


def test_fixture_corpus_is_not_empty(fixture_pair):
    before, _after = fixture_pair
    assert len(before) > 100
    assert sum("ungrounded" in item for item in before.values()) > 100


def test_fixture_differential_lists_every_changed_verdict(fixture_pair):
    assert _changed(*fixture_pair) == EXPECTED_CHANGED_FIXTURE_VERDICTS


def test_fixture_verdicts_only_remove_issues_and_keep_inventories(fixture_pair):
    _assert_only_removes_and_keeps_inventory(*fixture_pair)


def test_control_differential_lists_every_changed_verdict(control_pair):
    assert _changed(*control_pair) == EXPECTED_CHANGED_CONTROL_VERDICTS


def test_control_verdicts_only_remove_issues_and_keep_inventories(control_pair):
    _assert_only_removes_and_keeps_inventory(*control_pair)


def test_control_month_comparison_is_fully_grounded_with_months_on(control_pair):
    before, after = control_pair
    assert after["dated-month-comparison"]["ungrounded"] == []
    assert before["dated-month-comparison"]["ungrounded"] != []
    # The unrelated annualising 12 stays ungrounded either way.
    for side in (before, after):
        assert any(
            "12 does not appear" in issue
            for issue in side["dated-month-unrelated-literal"]["ungrounded"]
        )
