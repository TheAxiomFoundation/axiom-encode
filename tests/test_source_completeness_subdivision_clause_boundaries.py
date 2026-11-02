"""A single subdivision reference ends where a clause cue begins (#1831).

Va. Code § 58.1-322.03 says "No deduction shall be allowed pursuant to this
subdivision 7 if such payments ..." and "the portion of subdivision B 10 of
§ 58.1-301 related to Paycheck Protection Program loans". Numeric recall
demanded 7 and 10 as values, and no faithful encoding can name a structural
unit as a scalar.

A label is masked here only when it carries reference evidence of its own (the
demonstrative ``this`` before the keyword, or a statutory object such as
``of § 58.1-301`` after the label) and a clause cue follows. The negative
controls come first: the same number in an operative position stays demanded.
Production reads U.S. citations with the ``legacy`` numeric profile and the
replay fixtures use ``en-US``, so every check runs under both.
"""

from __future__ import annotations

import copy
import functools
import json
import re
from pathlib import Path

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CITATION = "us-va/statute/58.1/58.1-322.03"
FIXTURE = (
    Path(__file__).parent / "fixtures/source_completeness/signed_reencode_37936341740"
)
PROFILES = ("en-US", "legacy")
RECALL_VALUE = re.compile(
    r"\[complete-source-unit:numeric-recall\] Authoritative corpus numeric "
    r"value (\S+) has no named scalar"
)
EMPTY_MODULE = "format: rulespec/v1\nmodule: {}\nrules: []\n"
# Verbatim sentences from the Virginia fixture's source text.
SUBDIVISION_7 = (
    "No deduction shall be allowed pursuant to this subdivision 7 if such "
    "payments or contributions are deducted on the purchaser's or contributor's "
    "federal income tax return."
)
SUBDIVISION_B_10 = (
    "to the extent excluded from federal adjusted gross income, are not "
    "deductible when computing federal adjusted gross income solely on account "
    "of the portion of subdivision B 10 of § 58.1-301 related to Paycheck "
    "Protection Program loans."
)


def _clean(text: str) -> str:
    return completeness.authoritative_numeric_recall_text(
        text, corpus_citation_path=CITATION
    )


def _values(text: str, profile: str) -> list[float]:
    return [
        item.value
        for item in extract_typed_numeric_inventory_occurrences_from_text(
            text, profile=profile
        )
    ]


def _demands(content: str, source: str, *, profile: str, tests=()) -> set[str]:
    analysis = completeness.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=CITATION,
        test_cases=list(tests),
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return {
        match.group(1)
        for issue in analysis.issues
        if (match := RECALL_VALUE.search(issue))
    }


def _demand_keys(fragment: str, profile: str) -> set[str]:
    return {f"{value:g}" for value in _values(fragment, profile)}


def _assert_kept(source: str, fragments: tuple[str, ...], profile: str) -> None:
    """Every number of every operative fragment survives recall cleanup."""

    remaining = _values(_clean(source), profile)
    for fragment in fragments:
        expected = _values(fragment, profile)
        assert expected, fragment
        for value in expected:
            assert value in remaining, (source, _clean(source), fragment)
            remaining.remove(value)


# Negative controls: the number keeps its demand in an operative position, and
# a reference without evidence of its own keeps the conservative demand.
NEGATIVE_CONTROLS = (
    ("A deduction of $7 is allowed.", ("$7",)),
    ("A deduction of 7 percent is allowed.", ("7 percent",)),
    ("A deduction of 7 months is allowed.", ("7 months",)),
    # The issue's own control: the locator goes, the amount stays.
    ("pursuant to this subdivision 7 if $7 is paid", ("$7",)),
    ("pursuant to this subdivision 7 if 7 percent is paid", ("7 percent",)),
    ("subdivision 7 of this section; the deduction is $7.", ("$7",)),
    # A unit after the number makes it a quantity, not a label.
    ("pursuant to this subdivision 7 percent if such payments", ("7 percent",)),
    ("pursuant to this subdivision 7 months if such payments", ("7 months",)),
    # No reference evidence of its own: no `this`, no statutory object.
    ("pursuant to subdivision 7 if such payments are deducted", ("7",)),
    ("A county may approve a subdivision 5 if the lots are platted.", ("5",)),
    ("the portion of subdivision B 10 related to loans", ("10",)),
    # A suffix letter can be a unit; coordinated labels keep main's demand.
    ("pursuant to this subdivision 7 m if such payments", ("7",)),
    ("pursuant to this subdivision 7 or 8 if such payments", ("7", "8")),
    # The statutory object must be complete before the cue.
    ("subdivision B 10 of § 58.1-301A related to loans", ("10",)),
    ("subdivision 2, 20 of this section's 100 applicants qualify", ("20", "100")),
    # The operative 10 in the same Virginia source stays demanded.
    (
        "b. If the payment is received in a single payment, then 10 percent "
        "of the recognized gain may be subtracted in the taxable year.",
        ("10 percent",),
    ),
    (
        "of the portion of subdivision B 10 of § 58.1-301 related to 10 "
        "percent of the loans.",
        ("10 percent",),
    ),
)


@pytest.mark.parametrize(("source", "fragments"), NEGATIVE_CONTROLS)
@pytest.mark.parametrize("profile", PROFILES)
def test_operative_numbers_stay_demanded(source, fragments, profile):
    _assert_kept(source, fragments, profile)


@pytest.mark.parametrize(
    ("source", "fragment"),
    (
        ("A deduction of $7 is allowed.", "$7"),
        ("pursuant to this subdivision 7 if $7 is paid", "$7"),
        ("pursuant to this subdivision 7 if 7 percent is paid", "7 percent"),
        ("pursuant to subdivision 7 if such payments are deducted", "7"),
    ),
)
@pytest.mark.parametrize("profile", PROFILES)
def test_operative_numbers_stay_demanded_through_the_gate(source, fragment, profile):
    assert _demand_keys(fragment, profile) <= _demands(
        EMPTY_MODULE, source, profile=profile
    )


LOCATOR_SENTENCES = (
    (SUBDIVISION_7, "7"),
    (SUBDIVISION_B_10, "10"),
    ("This subdivision 7 unless the taxpayer elects otherwise.", "7"),
    ("under subdivision 3 of this section if the taxpayer elects", "3"),
    ("under subdivision B 3 of § 58.1-322.02 relating to the credit", "3"),
    ("pursuant to this subdivision 7 when such payments are made", "7"),
    ("pursuant to this subdivision 7 provided that such payments are made", "7"),
    ("pursuant to this subdivision 7 except as provided in this section", "7"),
    ("pursuant to this subdivision 7 if such payments", "7"),
)


@pytest.mark.parametrize(("source", "label"), LOCATOR_SENTENCES)
@pytest.mark.parametrize("profile", PROFILES)
def test_single_subdivision_reference_before_a_clause_cue_is_a_locator(
    source, label, profile
):
    assert float(label) in _values(source, profile)
    cleaned = _clean(source)
    assert float(label) not in _values(cleaned, profile), cleaned
    assert _clean(cleaned) == cleaned
    assert _demands(EMPTY_MODULE, source, profile=profile) == set()
    # The subdivision mask itself blanks digits in place and nothing else.
    masked = completeness._mask_numeric_spans(
        source, completeness._english_subdivision_numeric_recall_spans(source)
    )
    assert len(masked) == len(source)
    changed = [index for index, (a, b) in enumerate(zip(source, masked)) if a != b]
    assert changed
    assert all(source[index].isdigit() and masked[index] == " " for index in changed)


def _fixture_candidate() -> tuple[dict, str, list]:
    recorded = json.loads((FIXTURE / "issues.json").read_text())
    assert recorded["citation"] == CITATION
    candidate = yaml.safe_load((FIXTURE / "candidate.yaml").read_text())
    tests = yaml.safe_load((FIXTURE / "candidate.test.yaml").read_text())
    return candidate, (FIXTURE / "source.txt").read_text(), tests


def _with_named_values(candidate: dict, values, rates) -> str:
    document = copy.deepcopy(candidate)
    for index, value in enumerate(values):
        document["rules"].append(
            {
                "name": f"operative_value_{index}",
                "kind": "parameter",
                "dtype": "Decimal",
                "versions": [{"effective_from": "0001-01-01", "formula": repr(value)}],
            }
        )
    for index, value in enumerate(rates):
        document["rules"].append(
            {
                "name": f"operative_rate_{index}",
                "kind": "parameter",
                "dtype": "Rate",
                "versions": [{"effective_from": "0001-01-01", "formula": repr(value)}],
            }
        )
    return yaml.safe_dump(document, sort_keys=False)


# Every value main demanded from the replayed candidate, minus 7 and 10. Main
# also demanded 7 (only `this subdivision 7 if ...`) and 10 (`subdivision B 10
# of § 58.1-301 related to ...` and the operative `10 percent of the recognized
# gain`). The lists are pinned so that a value this change wrongly masked
# could not drop out of them unnoticed.
OPERATIVE_VALUES = {
    "en-US": (
        "0.65 0.7 1 1.25 1.7 8 12 13 13.5 18 20 30 35 50 65 66 85 100 500 800 900 "
        "930 1000 3000 4500 5000 6000 8000 8500 8750 9000 9200 9300 12000 16000 "
        "17000 17500 18400 18600 20000 30000 50000 75000 100000"
    ),
    "legacy": (
        "0.1 0.2 0.3 0.35 0.5 0.65 0.7 1 1.25 1.7 8 9 12 13 13.5 18 65 66 85 500 "
        "800 900 930 1000 3000 4500 5000 6000 8000 8500 8750 9000 9200 9300 12000 "
        "16000 17000 17500 18400 18600 20000 30000 50000 75000 100000"
    ),
}


def _operative_values(profile: str) -> tuple[str, ...]:
    return tuple(OPERATIVE_VALUES[profile].split())


def _plain(value: str) -> float | int:
    number = float(value)
    return int(number) if number.is_integer() else number


# `10 percent of the recognized gain` is the value 10 under `en-US` and 0.1
# under `legacy`; 0.1 is already in the `legacy` list above.
OPERATIVE_TEN_PERCENT = {"en-US": "10", "legacy": "0.1"}


@pytest.mark.parametrize("profile", PROFILES)
def test_replay_drops_only_the_locator_demands(profile):
    """Main demanded these values plus 7 and 10; only the locators go."""

    candidate, source, tests = _fixture_candidate()
    demanded = _demands(yaml.safe_dump(candidate), source, profile=profile, tests=tests)
    assert demanded == set(_operative_values(profile)) | {
        OPERATIVE_TEN_PERCENT[profile]
    }


@pytest.mark.parametrize("profile", PROFILES)
def test_replayed_candidate_with_operative_values_named_has_no_numeric_demand(
    profile,
):
    candidate, source, tests = _fixture_candidate()
    values = [_plain(value) for value in _operative_values(profile)]
    content = _with_named_values(candidate, values, rates=[0.1])
    assert _demands(content, source, profile=profile, tests=tests) == set()


@pytest.mark.parametrize("unnamed", ("ten percent", "8"))
@pytest.mark.parametrize("profile", PROFILES)
def test_replayed_operative_values_stay_demanded_when_unnamed(unnamed, profile):
    """`10 percent of the recognized gain` and `at least 8.0` are operative."""

    candidate, source, tests = _fixture_candidate()
    dropped = OPERATIVE_TEN_PERCENT[profile] if unnamed == "ten percent" else unnamed
    values = [_plain(value) for value in _operative_values(profile) if value != dropped]
    rates = [] if unnamed == "ten percent" else [0.1]
    content = _with_named_values(candidate, values, rates=rates)
    assert _demands(content, source, profile=profile, tests=tests) == {dropped}


OPERATIVE = (
    "a deduction of ${n}",
    "a deduction of {n} percent",
    "a limit of {n} months",
    "for {n} persons",
    "at least {n} taxable years",
)
LOCATORS = (
    "No deduction shall be allowed pursuant to this subdivision {n} if such "
    "payments are deducted.",
    "the portion of subdivision B {n} of § 58.1-301 related to loans.",
    "under subdivision {n} of this section if the taxpayer elects.",
    "This subdivision {n} unless the taxpayer elects otherwise.",
)


@given(
    number=st.integers(min_value=1, max_value=9999),
    operative=st.sampled_from(OPERATIVE),
    locator=st.sampled_from(LOCATORS),
    locator_first=st.booleans(),
    profile=st.sampled_from(PROFILES),
)
@example(
    number=7,
    operative=OPERATIVE[0],
    locator=LOCATORS[0],
    locator_first=True,
    profile="legacy",
)
@example(
    number=10,
    operative=OPERATIVE[1],
    locator=LOCATORS[1],
    locator_first=False,
    profile="legacy",
)
@settings(max_examples=150, deadline=None)
def test_same_number_in_operative_and_locator_positions(
    number, operative, locator, locator_first, profile
):
    """Only the locator occurrence goes; the operative one keeps its demand."""

    operative_text = operative.format(n=number) + "."
    locator_text = locator.format(n=number)
    parts = (
        (locator_text, operative_text)
        if locator_first
        else (operative_text, locator_text)
    )
    source = " ".join(parts)
    assert sorted(_values(_clean(source), profile)) == sorted(
        _values(_clean(operative_text), profile)
    )
    assert _demand_keys(operative_text, profile) <= _demands(
        EMPTY_MODULE, source, profile=profile
    )
    assert _demands(EMPTY_MODULE, locator_text, profile=profile) == set()


@given(
    number=st.integers(min_value=1, max_value=9999),
    unit=st.sampled_from(
        ("percent", "months", "years", "persons", "dollars", "days", "members")
    ),
    cue=st.sampled_from(("if", "unless", "when", "where", "provided that")),
    profile=st.sampled_from(PROFILES),
)
@settings(max_examples=100, deadline=None)
def test_a_unit_after_the_label_keeps_the_number(number, unit, cue, profile):
    source = (
        f"pursuant to this subdivision {number} {unit} {cue} such payments are made."
    )
    _assert_kept(source, (f"{number} {unit}",), profile)


REPOSITORY = Path(__file__).parents[1]
VIRGINIA_SOURCE = FIXTURE.relative_to(REPOSITORY) / "source.txt"


def _repository_source_texts():
    for root in (REPOSITORY / "tests/fixtures", REPOSITORY / "benchmarks"):
        for path in sorted(root.rglob("*.txt")):
            yield path.relative_to(REPOSITORY), path.read_text()


def _without_clause_cue_branch(monkeypatch):
    """Main's semantics: the new branch never fires."""

    monkeypatch.setattr(
        completeness,
        "_english_subdivision_reference_ends_at_clause_cue",
        lambda *_args: False,
    )


def _inventory_keys(text: str, profile: str) -> list[tuple[float, str]]:
    return [
        (item.value, item.raw)
        for item in extract_typed_numeric_inventory_occurrences_from_text(
            text, profile=profile
        )
    ]


def test_fixture_differential_removes_only_the_named_locators(monkeypatch):
    """Over every repository source text, only the two Virginia locators go.

    The branch can change recall only for a text where it returns true while
    the text is cleaned: anywhere else the elif never fires and the output is
    main's by construction. It reads only `subdivision` citations, so a text
    without that word cannot fire it. Where it fires, compare main's typed
    inventory with this branch's under both numeric profiles.
    """

    fired: list[str] = []
    original = completeness._english_subdivision_reference_ends_at_clause_cue

    def recording(source_text, citation):
        result = original(source_text, citation)
        if result:
            fired.append(source_text[citation.start() : citation.end() + 30])
        return result

    texts = list(_repository_source_texts())
    assert len(texts) > 200
    references: dict[Path, list[str]] = {}
    removed_by_profile: dict[str, dict[Path, list[tuple[float, str]]]] = {}
    for path, text in texts:
        if "subdivision" not in text.lower():
            continue
        fired.clear()
        with monkeypatch.context() as patched:
            patched.setattr(
                completeness,
                "_english_subdivision_reference_ends_at_clause_cue",
                recording,
            )
            current = _clean(text)
        if not fired:
            continue
        references[path] = list(fired)
        with monkeypatch.context() as patched:
            _without_clause_cue_branch(patched)
            baseline = _clean(text)
        for profile in PROFILES:
            before = _inventory_keys(baseline, profile)
            after = _inventory_keys(current, profile)
            removed = list(before)
            for occurrence in after:
                removed.remove(occurrence)
            removed_by_profile.setdefault(profile, {})[path] = sorted(removed)
    assert references == {
        VIRGINIA_SOURCE: [
            "subdivision 7 if such payments or contribut",
            "subdivision B 10 of § 58.1-301 related to Payc",
        ]
    }
    for profile in PROFILES:
        assert removed_by_profile[profile] == {
            VIRGINIA_SOURCE: [(7.0, "7"), (10.0, "10")]
        }


class _CountingPattern:
    """Count the characters each anchored or windowed call reads."""

    def __init__(self, pattern: re.Pattern[str], meter: dict[str, int]):
        self._pattern = pattern
        self._meter = meter

    def match(self, string, pos=0):
        found = self._pattern.match(string, pos)
        self._meter["calls"] += 1
        self._meter["read"] += (found.end() - pos if found else 0) + 1
        return found

    def search(self, string):
        self._meter["calls"] += 1
        self._meter["read"] += len(string)
        return self._pattern.search(string)


@pytest.mark.parametrize("copies", (1, 4, 16, 64))
def test_clause_cue_branch_work_is_linear_in_references(monkeypatch, copies):
    meter = {"calls": 0, "read": 0}
    for name in (
        "_ENGLISH_SUBDIVISION_STATUTORY_OBJECT",
        "_ENGLISH_SUBDIVISION_CLAUSE_CUE",
        "_ENGLISH_SUBDIVISION_DEMONSTRATIVE",
    ):
        monkeypatch.setattr(
            completeness,
            name,
            _CountingPattern(getattr(completeness, name), meter),
        )
    source = " ".join((SUBDIVISION_7, SUBDIVISION_B_10) * copies)
    spans = list(completeness._english_subdivision_numeric_recall_spans(source))
    assert len(spans) == 2 * copies
    # Each reference reaches the branch once: at most three anchored calls,
    # each reading a bounded window, whatever the text length.
    assert meter["calls"] <= 3 * 2 * copies
    assert meter["read"] <= 60 * 2 * copies
