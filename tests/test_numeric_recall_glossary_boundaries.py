"""Numeric recall changes preserve main's glossary and boundary behavior."""

from __future__ import annotations

import functools
import random
import re

import pytest

from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.source_completeness import (
    _collapse_text,
    _source_has_joined_operative_segment,
    _source_has_operative_policy_effect,
)
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.test_numeric_recall_structural_properties import (
    PROFILES,
    US,
    _inventory,
    _recall_issues,
)

SUBDIVISIONS = ("(d)(5)", "(d)(5)(A)", "(d)(5)(A)(i)")
OPERATIVE_PREFIXES = (
    "A household is eligible if it contains ",
    "An applicant qualifies if they belong to ",
    "Applicants receive SNAP benefits if they are ",
    "Qualification requires ",
    "Qualification requires:\n",
    "Eligibility requires:\n",
    "To qualify, applicants must include:\n",
    "A household is eligible:\n",
    # A new line authenticates the apparent heading, but it cannot discard a
    # policy effect that precedes it in the same source branch.
    "A household is eligible if its members include:\n",
    "A household is eligible if "
    + "the relevant immigration status has been documented and verified " * 4
    + "it contains:\n",
)


def _source(subdivisions, *, prefix="", value=1):
    return (
        f"{prefix}Parolees Paroled into the U.S. under section 212{subdivisions} "
        f"of the INA for a period of at least {value} years."
    )


def _root(source):
    return completeness.SourceStructureBranch(
        (), "source-unit", "source unit", source, 0, len(source)
    )


def _bounds(source, profile):
    return [
        (occurrence.value, occurrence.raw)
        for _branch, occurrence in completeness._source_boundary_obligations(
            (_root(source),),
            extract_numeric_occurrences=functools.partial(
                extract_typed_numeric_inventory_occurrences_from_text,
                profile=profile,
            ),
        )
    ]


def _boundary_test_issues(source, profile, *, value=1, witnessed=False):
    rule_name = "household_eligible"
    principal_rules = {
        rule_name: {
            "name": rule_name,
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [{"formula": f"parole_years >= {value}"}],
        }
    }
    cases = (
        {
            "name": "duration endpoint" if witnessed else "above duration endpoint",
            "period": "2026-01-01",
            "input": {"parole_years": value if witnessed else value + 1},
            "output": {rule_name: "holds"},
        },
    )
    issues = completeness._companion_test_issues(
        principal_rules,
        parameter_rules={},
        principal_rule_paths={rule_name: {()}},
        principal_formula_clause_rules={},
        formula_branches=(),
        branches=(_root(source),),
        source_text=source,
        corpus_citation_path=US,
        deferred_paths=set(),
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text,
            profile=profile,
        ),
        numeric_value_is_grounded=numeric_value_is_grounded,
        formula_environment={},
        source_bound_constant_occurrences={},
        declared_input_names={"parole_years"},
    )
    return [issue for issue in issues if "source-stated boundary input" in issue]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("subdivisions", SUBDIVISIONS)
@pytest.mark.parametrize("prefix", OPERATIVE_PREFIXES)
def test_parolee_prefix_duration_boundary_preserves_main_behavior(
    profile, subdivisions, prefix
):
    source = _source(subdivisions, prefix=prefix)
    assert [(item.value, item.raw) for item in _inventory(source, US, profile)] == [
        (1, "1")
    ]
    assert _recall_issues(source, US, profile)
    assert not _recall_issues(source, US, profile, (1,))
    # Main already exempts flat (d)(5) rows even with these operative prefixes.
    # Keep scalar recall strict while testing the pre-existing boundary result.
    assert _bounds(source, profile) == _main_expected_bounds(source)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("subdivisions", SUBDIVISIONS)
@pytest.mark.parametrize("prefix", OPERATIVE_PREFIXES)
@pytest.mark.parametrize("witnessed", (False, True), ids=("missing", "present"))
def test_operative_parolee_duration_requires_an_executed_boundary_case(
    profile, subdivisions, prefix, witnessed
):
    source = _source(subdivisions, prefix=prefix)
    assert bool(_boundary_test_issues(source, profile, witnessed=witnessed)) is (
        not witnessed and bool(_main_expected_bounds(source))
    )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("subdivisions", SUBDIVISIONS)
@pytest.mark.parametrize("heading_prefix", ("", "Alien Group Description:\n"))
def test_actual_parolee_glossary_heading_keeps_scalar_without_boundary_case(
    profile, subdivisions, heading_prefix
):
    source = _source(subdivisions, prefix=heading_prefix)
    assert [(item.value, item.raw) for item in _inventory(source, US, profile)] == [
        (1, "1")
    ]
    assert _recall_issues(source, US, profile)
    assert not _recall_issues(source, US, profile, (1,))
    # Main exempts the flat glossary control and retains nested INA bounds.
    expected = _main_expected_bounds(source)
    assert _bounds(source, profile) == expected
    assert bool(_boundary_test_issues(source, profile)) is bool(expected)


@pytest.mark.parametrize("profile", PROFILES)
def test_generated_glossary_boundaries_preserve_main_semantics(profile):
    """Citation nesting and duration values preserve main's boundary decisions."""

    generator = random.Random(1779)
    values = (1, 1.5, *generator.sample(range(2, 100), 4))
    for subdivisions in SUBDIVISIONS:
        for value in values:
            glossary = _source(subdivisions, value=value)
            expected = _main_expected_bounds(glossary, value=value)
            assert _bounds(glossary, profile) == expected
            assert bool(_boundary_test_issues(glossary, profile, value=value)) is bool(
                expected
            )
            for prefix in OPERATIVE_PREFIXES:
                operative = _source(subdivisions, prefix=prefix, value=value)
                expected = _main_expected_bounds(operative, value=value)
                assert _bounds(operative, profile) == expected
                assert bool(
                    _boundary_test_issues(operative, profile, value=value)
                ) is bool(expected)
                assert not _boundary_test_issues(
                    operative, profile, value=value, witnessed=True
                )


# Verbatim predicate from origin/main commit
# 04ef0c7da14ba219df16c489471e233c5d3cc2a9 (also identical at db7e900e).
def _main_glossary_predicate(
    text: str,
    *,
    context: str,
) -> bool:
    """Exclude a glossary-only admission-duration definition from case bounds."""

    collapsed = _collapse_text(text)
    return bool(
        re.search(
            r"\bParolees\s+Paroled\s+into\s+the\s+U\.?S\.?\s*$",
            _collapse_text(context[: -len(text)] if text else context),
            flags=re.IGNORECASE,
        )
        and re.search(
            r"\bunder\s+\([a-z]\)\(\d+\)\s+of\s+the\s+INA\s+for\s+a\s+"
            r"period\s+of\s+at\s+least\s+\d+(?:\.\d+)?\s+years?\b",
            collapsed,
            flags=re.IGNORECASE,
        )
        and not _source_has_operative_policy_effect(collapsed)
        and not _source_has_joined_operative_segment(collapsed)
    )


# Round 3's p2-gate-repro.py/p2-gate-root.jsonl source families, expanded across
# every round 2 INA subdivision and heading-prefix control. The outline cases
# use recognize_source_structure so child rows have genuine owning paths.
ROUND3_LAYOUTS = (
    "(1) A household is eligible if it contains a parolee:\n(a) {row}.",
    "(1) A household is eligible if it contains a parolee:\n1. {row}.",
    "(1) A household is eligible if it contains a parolee:\n1{row}.",
    "(1) A household is eligible if it contains a parolee:\n(a) {row}.\n"
    "(2) Documentation confirms immigration status.",
    "(a) A household is eligible if it contains a parolee:\n(1) {row}.\n"
    "(b) Documentation confirms immigration status.",
    "A household is eligible if it contains a parolee:\n{row}.",
    "{row} and applicants are eligible if they satisfy this description.",
    "{row}: applicants are eligible if they satisfy this description.",
    "{row}\nand applicants are eligible if they satisfy this description.",
    "{row}; applicants are eligible if they satisfy this description.",
    "{row}. Applicants are eligible if they satisfy this description.",
    "{row}.",
    "Alien Group Description:\n{row}.",
)


def _main_corpus_cleaned_text(text):
    """Frozen main cleaner output for this INA-only differential corpus.

    These corpus transformations were checked against main's entire cleaner at
    04ef0c7da14ba219df16c489471e233c5d3cc2a9, not the PR cleaner. Preserve the
    subdivision residue and character deletion that main fed to its predicate.
    """

    text = text.replace("section 212", "")
    if text.startswith("1Parolees"):
        text = text[1:]
    text = re.sub(r"(?m)^[ \t]*\((?:\d+[a-z]?|[a-z])\)", "", text)
    return re.sub(r"(?m)^[ \t]*\d+[a-z]?\.[ \t]+", "", text)


def _main_boundary_calls(branches):
    calls = []
    for branch in branches:
        if branch.kind not in {
            "paragraph",
            "number",
            "letter",
            "sentence",
            "source-unit",
        }:
            continue
        direct = _main_corpus_cleaned_text(
            completeness._source_branch_direct_text(branch, branches=branches)
        )
        for start, fragment in completeness._source_boundary_fragments(direct):
            text = fragment.split(":", 1)[0]
            context = direct[max(0, start - 160) : start + len(text)]
            calls.append(
                (text, context, _main_glossary_predicate(text, context=context))
            )
    return calls


def _main_expected_bounds(source, *, value=1):
    duration_calls = [
        call for call in _main_boundary_calls((_root(source),)) if "at least" in call[0]
    ]
    assert len(duration_calls) == 1
    return [] if duration_calls[0][2] else [(value, str(value))]


def _recognized_branches(source):
    return completeness.recognize_source_structure(source) or (_root(source),)


def _differential_corpus():
    for subdivisions in SUBDIVISIONS:
        row = _source(subdivisions).removesuffix(".")
        for layout in ROUND3_LAYOUTS:
            yield layout.format(row=row)
        for prefix in OPERATIVE_PREFIXES:
            yield _source(subdivisions, prefix=prefix)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("source", tuple(_differential_corpus()))
def test_glossary_boundary_path_matches_main_text_and_predicate(
    profile, source, monkeypatch
):
    branches = _recognized_branches(source)
    expected_calls = _main_boundary_calls(branches)
    actual_calls = []
    predicate = completeness._source_boundary_is_nonoperative_guidance_definition

    def record_predicate(text, *, context):
        exempt = predicate(text, context=context)
        actual_calls.append((text, context, exempt))
        return exempt

    monkeypatch.setattr(
        completeness,
        "_source_boundary_is_nonoperative_guidance_definition",
        record_predicate,
    )
    completeness._source_boundary_obligations(
        branches,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
    )
    assert actual_calls == expected_calls


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("layout", ROUND3_LAYOUTS[:11])
def test_recognized_nested_and_trailing_operative_cases_require_boundary(
    profile, layout
):
    source = layout.format(row=_source("(d)(5)(A)").removesuffix("."))
    branches = _recognized_branches(source)
    if layout.startswith("(1)") and "(2)" in layout:
        assert any(branch.path == ("1", "a") for branch in branches)
    if layout.startswith("(a)"):
        assert any(branch.path == ("a", "1") for branch in branches)
    extractor = functools.partial(
        extract_typed_numeric_inventory_occurrences_from_text, profile=profile
    )
    assert [
        (occurrence.value, occurrence.raw)
        for _branch, occurrence in completeness._source_boundary_obligations(
            branches, extract_numeric_occurrences=extractor
        )
    ] == [(1, "1")]
    selector = (
        "satisfy_description"
        if "satisfy this description" in source
        else "contains_parolee"
    )
    rule = {
        "name": "household_eligible",
        "kind": "derived",
        "dtype": "Judgment",
        "versions": [{"formula": f"{selector} and parole_years >= 1"}],
    }
    cases = tuple(
        {
            "name": f"case_{index}",
            "period": "2026-01-01",
            "input": {selector: flag, "parole_years": duration},
            "output": {"household_eligible": state},
        }
        for index, (flag, duration, state) in enumerate(
            ((True, 2, "holds"), (False, 2, "not_holds"), (True, 0, "not_holds"))
        )
    )
    for witnessed in (False, True):
        endpoint = (
            {
                "name": "endpoint",
                "period": "2026-01-01",
                "input": {selector: True, "parole_years": 1},
                "output": {"household_eligible": "holds"},
            },
        )
        issues = completeness._companion_test_issues(
            {"household_eligible": rule},
            parameter_rules={},
            principal_rule_paths={
                "household_eligible": {branch.path for branch in branches}
            },
            principal_formula_clause_rules={},
            formula_branches=(),
            branches=branches,
            source_text=source,
            corpus_citation_path=US,
            deferred_paths=set(),
            test_cases=cases + (endpoint if witnessed else ()),
            extract_numeric_occurrences=extractor,
            numeric_value_is_grounded=numeric_value_is_grounded,
            formula_environment={},
            source_bound_constant_occurrences={},
            declared_input_names={selector, "parole_years"},
        )
        assert bool(issues) is not witnessed
        if not witnessed:
            assert any("source-stated boundary input" in issue for issue in issues)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "source,expected_text",
    [
        (source, _main_corpus_cleaned_text(source.splitlines()[0]))
        for source in _differential_corpus()
    ]
    + [
        (
            "Income must be at least section 100 dollars.",
            "Income must be at least  dollars.",
        ),
        (
            "At least section 100 dollars must be paid.",
            "At least  dollars must be paid.",
        ),
    ],
)
def test_narrative_boundary_path_receives_exact_main_cleaned_first_line(
    profile, source, expected_text, monkeypatch
):
    branch = _root(source)
    extractor = functools.partial(
        extract_typed_numeric_inventory_occurrences_from_text, profile=profile
    )
    interval_from_text = completeness._formula_interval_from_text
    expected_interval = interval_from_text(
        expected_text, extract_numeric_occurrences=extractor
    )
    actual_inputs = []

    def record_interval(text, *, extract_numeric_occurrences):
        actual_inputs.append(text)
        return interval_from_text(
            text, extract_numeric_occurrences=extract_numeric_occurrences
        )

    monkeypatch.setattr(completeness, "_formula_interval_from_text", record_interval)
    actual = completeness._source_boundary_obligations(
        (),
        narrative_formula_branches=(branch,),
        extract_numeric_occurrences=extractor,
    )
    assert actual_inputs == [expected_text]
    expected = (
        [
            (boundary.value, boundary.raw, boundary.start, boundary.end)
            for boundary in (expected_interval.lower, expected_interval.upper)
            if boundary is not None
        ]
        if expected_interval is not None
        else []
    )
    assert [
        (boundary.value, boundary.raw, boundary.start, boundary.end)
        for _branch, boundary in actual
    ] == expected
