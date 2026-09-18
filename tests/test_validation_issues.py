"""Tests for structured validation issues (`harness/validation_issues.py`).

The module turns validator issue strings into :class:`ValidationIssue` records
that persist with every encode attempt, so these tests pin the extraction
patterns, the gate labeling rules, and the serialization bounds.
"""

import json
import types
from unittest.mock import MagicMock

import pytest

from axiom_encode.harness.evals import classify_validation_issue
from axiom_encode.harness.validation_issues import (
    ISSUE_SCHEMA_VERSION,
    MAX_ISSUE_MESSAGE_CHARS,
    MAX_ISSUES_JSON_BYTES,
    MAX_ISSUES_PER_ATTEMPT,
    ValidationIssue,
    classify_issue_kind,
    extract_category,
    extract_clause,
    extract_locator,
    extract_value,
    issue_summary_counts,
    issues_from_dicts,
    issues_to_dicts,
    structure_attempt_issues,
    structure_issue,
    structure_labeled_issues,
)

UNGROUNDED_MESSAGE = (
    "Ungrounded generated numeric literal: 600000 does not appear as a "
    "substantive numeric value in the source text."
)
STRUCTURE_MESSAGE = (
    "[complete-source-unit:structure] Source branch (a) at 26 USC 1(a) "
    "[para a] is neither encoded nor precisely deferred."
)
FORMULA_MESSAGE = (
    "[complete-source-unit:formula-output] Explicit source computation "
    "(1)(b)(ii) in us-ms/statute/27-7-5(1) [Absatz 1, Nummer b, Nummer ii] "
    "has no principal derived/relation output."
)
#: The shape `_TEST_OUTPUT_MISMATCH_PATTERN` in cli.py parses back out.
TEST_OUTPUT_MISMATCH_MESSAGE = (
    "Test case `single_filer_credit` output `eitc_amount` expected 1200, got 900."
)


def _metrics(**attributes: object) -> types.SimpleNamespace:
    """A duck-typed `EvalArtifactMetrics` carrying only the named attributes."""
    return types.SimpleNamespace(**attributes)


def _result(metrics: object) -> types.SimpleNamespace:
    """A duck-typed `EvalResult` whose `.metrics` is `metrics`."""
    return types.SimpleNamespace(metrics=metrics)


# ---------------------------------------------------------------------------
# structure_issue: kind, category, locator, line, value, clause
# ---------------------------------------------------------------------------


def test_structure_issue_carries_gate_kind_and_message():
    issue = structure_issue("ci", UNGROUNDED_MESSAGE)

    assert issue.gate == "ci"
    assert issue.kind == "ungrounded_literal"
    assert issue.message == UNGROUNDED_MESSAGE
    assert issue.value == "600000"


def test_structure_issue_kind_delegates_to_the_shared_classifier():
    message = "Companion output coverage is missing for the generated module."

    assert classify_issue_kind(message) == classify_validation_issue(message)
    assert structure_issue("ci", message).kind == classify_validation_issue(message)
    assert structure_issue("ci", message).kind == "companion_coverage"


def test_structure_issue_coerces_non_string_gate_and_message():
    issue = structure_issue(123, ValueError("kaboom"))

    assert issue.gate == "123"
    assert issue.message == "kaboom"


def test_structure_issue_test_case_locator():
    issue = structure_issue(
        "ci",
        "Test case `single_filer_credit` execution failed: parameter `rate` "
        "has no value for key `0`",
    )

    assert issue.locator == "test:single_filer_credit"
    assert issue.kind == "fixture_execution"
    assert issue.line is None


def test_structure_issue_rule_locator():
    issue = structure_issue(
        "compile", "rule `eitc_phase_out` proof atom 0 must declare `path`"
    )

    assert issue.locator == "rule:eitc_phase_out"
    assert issue.kind == "proof_atoms"


def test_structure_issue_path_locator():
    issue = structure_issue(
        "ci",
        "Missing companion test file `rules/us/irs/eitc.test.yaml` for generated module",
    )

    assert issue.locator == "path:rules/us/irs/eitc.test.yaml"


def test_structure_issue_line_locator_and_line_number():
    issue = structure_issue(
        "compile", "Unsupported operator at line 12 in generated formula"
    )

    assert issue.locator == "line:12"
    assert issue.line == 12


def test_structure_issue_keeps_line_alongside_a_named_locator():
    issue = structure_issue("ci", "Test case `alpha` failed at line 7")

    assert issue.locator == "test:alpha"
    assert issue.line == 7


def test_structure_issue_locator_precedence_prefers_the_test_case():
    issue = structure_issue(
        "ci",
        "Test case `alpha` in rule `beta` from `rules/us/irs/eitc.yaml` failed",
    )

    assert issue.locator == "test:alpha"


def test_structure_issue_bracketed_category_and_branch_locator():
    issue = structure_issue("ci", STRUCTURE_MESSAGE)

    assert issue.category == "complete-source-unit:structure"
    assert issue.locator == "26 USC 1(a) [para a]"
    assert issue.clause == "26 USC 1(a)"


def test_structure_issue_formula_output_branch_locator():
    issue = structure_issue("ci", FORMULA_MESSAGE)

    assert issue.category == "complete-source-unit:formula-output"
    assert issue.locator == "us-ms/statute/27-7-5(1) [Absatz 1, Nummer b, Nummer ii]"


def test_structure_issue_branch_locator_collapses_runs_of_whitespace():
    issue = structure_issue(
        "ci",
        "[complete-source-unit:structure] Source branch (a) at 26 USC   1(a)  "
        "[para a] is neither encoded nor precisely deferred.",
    )

    assert issue.locator == "26 USC 1(a) [para a]"


def test_structure_issue_ignores_a_bracketed_aside_as_category():
    issue = structure_issue("ci", "Source branch (a) [para a] was not encoded")

    assert issue.category is None


def test_structure_issue_expected_and_actual_pair():
    issue = structure_issue(
        "ci", "Fixture mismatch: expected 12 but actual 10 for household benefit"
    )

    assert issue.value == "expected=12 actual=10"


def test_structure_issue_expected_value_alone():
    issue = structure_issue("ci", "Test output expected 1200 for the credit")

    assert issue.value == "expected=1200"


def test_structure_issue_actual_value_alone():
    issue = structure_issue("ci", "The formula returned 0 for every household")

    assert issue.value == "actual=0"


def test_structure_issue_policyengine_pair():
    issue = structure_issue(
        "policyengine",
        "Oracle disagreement for case alpha: PE=1200.0 RuleSpec expects=1150.0",
    )

    assert issue.value == "oracle=1200.0 expected=1150.0"
    assert issue.kind == "oracle_coverage"


def test_structure_issue_source_numeric_occurrence_value():
    issue = structure_issue(
        "numeric_occurrence",
        "Source numeric value 7,430 appears 3 times in the source text but "
        "never in the generated module.",
    )

    assert issue.value == "7,430"


@pytest.mark.parametrize(
    ("message", "clause"),
    [
        ("The encoded module omits 26 USC 32(b) entirely.", "26 USC 32(b)"),
        (
            "Deferral target § 1401(b)(1) does not resolve to an encoded module.",
            "§ 1401(b)(1)",
        ),
        (
            "Source branch (a)(2)(B) is present in the source text but absent "
            "from the module.",
            "(a)(2)(B)",
        ),
    ],
)
def test_structure_issue_extracts_the_cited_clause(message, clause):
    assert structure_issue("ci", message).clause == clause
    assert extract_clause(message) == clause


def test_structure_issue_leaves_optional_fields_none_when_nothing_matches():
    issue = structure_issue("ci", "Generated module drifted.")

    assert issue.category is None
    assert issue.locator is None
    assert issue.line is None
    assert issue.value is None
    assert issue.clause is None


def test_structure_issue_caps_the_message_at_the_bound():
    issue = structure_issue("ci", "x" * (MAX_ISSUE_MESSAGE_CHARS + 500))

    assert len(issue.message) == MAX_ISSUE_MESSAGE_CHARS


def test_structure_issue_keeps_a_message_at_exactly_the_bound():
    message = "y" * MAX_ISSUE_MESSAGE_CHARS
    issue = structure_issue("ci", message)

    assert issue.message == message


def test_empty_message_is_unclassified_with_no_extraction():
    issue = structure_issue("attempt", "")

    assert issue.kind == "unclassified"
    assert issue.to_dict() == {"gate": "attempt", "kind": "unclassified", "message": ""}


def test_extract_helpers_agree_with_structure_issue():
    locator, line = extract_locator(STRUCTURE_MESSAGE)

    assert extract_category(STRUCTURE_MESSAGE) == "complete-source-unit:structure"
    assert locator == "26 USC 1(a) [para a]"
    assert line is None
    assert extract_value(UNGROUNDED_MESSAGE) == "600000"


def test_schema_version_is_declared():
    assert ISSUE_SCHEMA_VERSION == "axiom-encode/validation-issue/v1"


def test_extract_value_does_not_keep_the_sentence_comma():
    assert extract_value(TEST_OUTPUT_MISMATCH_MESSAGE) == "expected=1200 actual=900"


def test_extract_clause_prefers_a_real_citation_over_prose():
    message = (
        "A citation-only proof atom does not distinguish one internal formula "
        "clause from another in 26 USC 32(b)."
    )

    assert extract_clause(message) == "26 USC 32(b)"


# ---------------------------------------------------------------------------
# structure_labeled_issues
# ---------------------------------------------------------------------------


def test_structure_labeled_issues_dedupes_across_gates_keeping_the_first_gate():
    structured = structure_labeled_issues(
        [
            ("numeric_occurrence", ["shared message", "occurrence only"]),
            ("ci", ["shared message", "ci only"]),
        ]
    )

    assert [(issue.gate, issue.message) for issue in structured] == [
        ("numeric_occurrence", "shared message"),
        ("numeric_occurrence", "occurrence only"),
        ("ci", "ci only"),
    ]


def test_structure_labeled_issues_dedupes_within_one_gate():
    structured = structure_labeled_issues([("ci", ["same", "same"])])

    assert [issue.message for issue in structured] == ["same"]


def test_structure_labeled_issues_ignores_non_sequence_groups_and_bare_strings():
    structured = structure_labeled_issues(
        [
            ("ci", "a bare string is not an issue list"),
            ("compile", b"bytes are not an issue list"),
            ("taxsim", None),
            ("policyengine", {"issue": "mappings are not sequences"}),
            ("ci", ("kept from a tuple",)),
        ]
    )

    assert [(issue.gate, issue.message) for issue in structured] == [
        ("ci", "kept from a tuple")
    ]


def test_structure_labeled_issues_skips_empty_messages():
    structured = structure_labeled_issues([("ci", ["", "real issue", ""])])

    assert [issue.message for issue in structured] == ["real issue"]


def test_structure_labeled_issues_returns_empty_for_no_groups():
    assert structure_labeled_issues([]) == []


# ---------------------------------------------------------------------------
# structure_attempt_issues
# ---------------------------------------------------------------------------


def test_structure_attempt_issues_skips_passing_gates():
    result = _result(
        _metrics(
            compile_pass=True,
            compile_issues=["advisory compile note"],
            ci_pass=False,
            ci_issues=["Generated RuleSpec failed CI validation"],
        )
    )

    structured = structure_attempt_issues(result)

    assert [(issue.gate, issue.message) for issue in structured] == [
        ("ci", "Generated RuleSpec failed CI validation")
    ]


def test_structure_attempt_issues_keeps_gates_whose_pass_flag_is_unset():
    result = _result(
        _metrics(
            generalist_review_pass=None, generalist_review_issues=["review finding"]
        )
    )

    structured = structure_attempt_issues(result)

    assert [(issue.gate, issue.message) for issue in structured] == [
        ("generalist_review", "review finding")
    ]


def test_structure_attempt_issues_prefers_the_numeric_occurrence_label_over_ci():
    result = _result(
        _metrics(
            numeric_occurrence_pass=False,
            numeric_occurrence_issues=["shared message"],
            ci_pass=False,
            ci_issues=["shared message", "ci only"],
        )
    )

    structured = structure_attempt_issues(result)

    assert [(issue.gate, issue.message) for issue in structured] == [
        ("numeric_occurrence", "shared message"),
        ("ci", "ci only"),
    ]


def test_structure_attempt_issues_labels_only_uncovered_extra_issues():
    result = _result(_metrics(ci_pass=False, ci_issues=["ci only"]))

    structured = structure_attempt_issues(
        result, extra_issues=["ci only", "overlay only"]
    )

    assert [(issue.gate, issue.message) for issue in structured] == [
        ("ci", "ci only"),
        ("overlay", "overlay only"),
    ]


def test_structure_attempt_issues_honors_a_custom_extra_gate():
    structured = structure_attempt_issues(
        _result(None), extra_issues=["preflight complaint"], extra_gate="preflight"
    )

    assert [(issue.gate, issue.message) for issue in structured] == [
        ("preflight", "preflight complaint")
    ]


def test_structure_attempt_issues_dedupes_repeated_extra_issues():
    structured = structure_attempt_issues(
        _result(None), extra_issues=["overlay only", "overlay only", ""]
    )

    assert [issue.message for issue in structured] == ["overlay only"]


def test_structure_attempt_issues_ignores_a_bare_string_extra_issues():
    result = _result(_metrics(ci_pass=False, ci_issues=["ci only"]))

    structured = structure_attempt_issues(result, extra_issues="not a list")

    assert [(issue.gate, issue.message) for issue in structured] == [("ci", "ci only")]


def test_structure_attempt_issues_falls_back_to_the_attempt_error():
    structured = structure_attempt_issues(_result(None), error="attempt exploded")

    assert [(issue.gate, issue.kind, issue.message) for issue in structured] == [
        ("attempt", "attempt_exploded", "attempt exploded")
    ]


def test_structure_attempt_issues_skips_the_error_when_a_gate_spoke():
    result = _result(_metrics(ci_pass=False, ci_issues=["ci only"]))

    structured = structure_attempt_issues(result, error="attempt exploded")

    assert [issue.message for issue in structured] == ["ci only"]


def test_structure_attempt_issues_returns_empty_without_issues_or_error():
    assert structure_attempt_issues(_result(_metrics())) == []
    assert structure_attempt_issues(_result(None)) == []
    assert structure_attempt_issues(_result(None), error="") == []
    assert structure_attempt_issues(_result(None), error=None) == []


def test_structure_attempt_issues_tolerates_a_mock_metrics_object():
    structured = structure_attempt_issues(MagicMock(), error="mock boom")

    assert [(issue.gate, issue.message) for issue in structured] == [
        ("attempt", "mock boom")
    ]


# ---------------------------------------------------------------------------
# Serialization: to_dict / from_dict / bounds
# ---------------------------------------------------------------------------


def test_to_dict_omits_fields_that_are_none():
    issue = ValidationIssue(gate="ci", kind="schema", message="YAML parse failed")

    assert issue.to_dict() == {
        "gate": "ci",
        "kind": "schema",
        "message": "YAML parse failed",
    }


def test_to_dict_keeps_every_populated_field():
    issue = structure_issue("ci", STRUCTURE_MESSAGE)

    assert set(issue.to_dict()) == {
        "gate",
        "kind",
        "message",
        "category",
        "locator",
        "clause",
    }


def test_issues_round_trip_through_dicts():
    issues = [
        structure_issue("ci", UNGROUNDED_MESSAGE),
        structure_issue("compile", "Unsupported operator at line 12 in the formula"),
        structure_issue("attempt", "attempt exploded"),
    ]

    payload, truncated = issues_to_dicts(issues)

    assert truncated == 0
    assert issues_from_dicts(payload) == issues
    assert json.loads(json.dumps(payload)) == payload


def test_issues_to_dicts_reports_the_count_bound():
    issues = [
        structure_issue("ci", f"issue {index}")
        for index in range(MAX_ISSUES_PER_ATTEMPT + 100)
    ]

    payload, truncated = issues_to_dicts(issues)

    assert len(payload) == MAX_ISSUES_PER_ATTEMPT
    assert truncated == 100


def test_issues_to_dicts_reports_the_byte_bound():
    issues = [
        structure_issue("ci", f"{index}: " + "z" * MAX_ISSUE_MESSAGE_CHARS)
        for index in range(MAX_ISSUES_PER_ATTEMPT)
    ]

    payload, truncated = issues_to_dicts(issues)

    assert 0 < len(payload) < MAX_ISSUES_PER_ATTEMPT
    assert truncated == len(issues) - len(payload)
    assert len(json.dumps(payload).encode()) <= MAX_ISSUES_JSON_BYTES


def test_issues_to_dicts_handles_an_empty_list():
    assert issues_to_dicts([]) == ([], 0)


def test_issues_from_dicts_tolerates_non_list_input():
    assert issues_from_dicts("not a list") == []
    assert issues_from_dicts(None) == []
    assert issues_from_dicts({"gate": "ci"}) == []


def test_issues_from_dicts_skips_non_dict_items():
    parsed = issues_from_dicts(
        [
            {"gate": "ci", "kind": "schema", "message": "YAML parse failed"},
            "a bare string",
            None,
            7,
        ]
    )

    assert parsed == [
        ValidationIssue(gate="ci", kind="schema", message="YAML parse failed")
    ]


def test_from_dict_defaults_missing_fields():
    issue = ValidationIssue.from_dict({})

    assert issue == ValidationIssue(gate="attempt", kind="unclassified", message="")


def test_from_dict_rejects_a_bool_line():
    assert ValidationIssue.from_dict({"line": True}).line is None
    assert ValidationIssue.from_dict({"line": False}).line is None
    assert ValidationIssue.from_dict({"line": 12}).line == 12


def test_from_dict_ignores_a_non_integer_line():
    assert ValidationIssue.from_dict({"line": "12"}).line is None
    assert ValidationIssue.from_dict({"line": 12.5}).line is None
    assert ValidationIssue.from_dict({"line": None}).line is None


# ---------------------------------------------------------------------------
# issue_summary_counts
# ---------------------------------------------------------------------------


def test_issue_summary_counts_uses_gate_colon_kind_keys():
    issues = [
        structure_issue("ci", UNGROUNDED_MESSAGE),
        structure_issue("ci", UNGROUNDED_MESSAGE.replace("600000", "700000")),
        structure_issue("attempt", "attempt exploded"),
    ]

    assert issue_summary_counts(issues) == {
        "ci:ungrounded_literal": 2,
        "attempt:attempt_exploded": 1,
    }


def test_issue_summary_counts_is_empty_for_no_issues():
    assert issue_summary_counts([]) == {}
