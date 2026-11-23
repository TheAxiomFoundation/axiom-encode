"""A check's ``enforcement`` decides what its finding does (issue #2300).

``blocking`` (the default) refuses a candidate. ``advisory`` keeps the finding,
records it and shows it to the reviewer, and does not refuse. Everything that
is not positively advisory blocks. In this change exactly one check is
advisory: ``complete-source-unit:source-explicit-conditions/gates-collapsed``.
"""

from __future__ import annotations

import contextlib
import functools
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from axiom_encode import checks, cli, run_log_export
from axiom_encode.checks import (
    ADVISORY,
    BLOCKING,
    Check,
    IssueSubject,
    ValidationIssue,
    advisory_issue_records,
    advisory_review_context,
    issue_is_advisory,
    partition_by_enforcement,
)
from axiom_encode.corpus_resolver import LocalCorpusRelease
from axiom_encode.harness import evals, validator_pipeline
from axiom_encode.harness.evals import EvalArtifactMetrics, EvalResult
from axiom_encode.harness.source_completeness import (
    CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED,
    analyze_complete_source_unit,
)
from axiom_encode.harness.validator_pipeline import ValidationResult, ValidatorPipeline
from axiom_encode.run_log import Severity, StageStatus
from scripts import replay_us_failure_bundles as replay

GATES = "complete-source-unit:source-explicit-conditions/gates-collapsed"
OTHER = "complete-source-unit:tests/output-unasserted"
CITATION = "us-or/manual/odhs/open/page-644"
SOURCE = "Households are eligible unless a member is disqualified."
CONTENT = f"""format: rulespec/v1
module:
  status: deferred
  summary: An explicitly deferred source unit.
  source_verification:
    corpus_citation_path: {CITATION}
rules: []
"""
GATES_MESSAGE = (
    "[complete-source-unit:source-explicit-conditions] Derived formula version(s) "
    "delegate multiple conjunctive factual gates: `rule_a` versions[0]."
)
FIXTURES = Path(__file__).parent / "fixtures" / "az-utility-alternative-lists"
UNIT = "us-az/manual/des/faa5/na-utility-expenses-and-allowances"


@pytest.fixture(autouse=True)
def _registry():
    checks.load_checks()


def _gates_issue(message: str = GATES_MESSAGE) -> ValidationIssue:
    return ValidationIssue(
        message,
        check=GATES,
        subject=IssueSubject(citation=CITATION, rule="rule_a"),
        expected="2 distinct gate inputs",
        found="0 of 2 source gates",
    )


def _other_issue(message: str = "[complete-source-unit:tests] Not asserted."):
    return ValidationIssue(
        message, check=OTHER, subject=IssueSubject(citation=CITATION, rule="rule_b")
    )


def _check(**overrides) -> Check:
    values = dict(
        id="example:enforcement/probe",
        title="t",
        requires="r",
        why="w",
        fix="f",
        cause="validator",
        gate="ci",
        site="a/b.py:c",
    )
    values.update(overrides)
    return Check(**values)


# --------------------------------------------------------------------------
# The declaration: default, allowed values, the one advisory check
# --------------------------------------------------------------------------


def test_a_check_without_the_field_is_blocking():
    check = _check()
    assert check.enforcement == BLOCKING == "blocking"
    assert checks.check_definition_problems(check) == []
    assert "enforcement" not in check.to_dict()


def test_an_advisory_check_names_its_enforcement_in_the_registry():
    assert _check(enforcement="advisory").to_dict()["enforcement"] == "advisory"


@pytest.mark.parametrize(
    "value",
    ["warn", "Advisory", "ADVISORY", " advisory", "advisory ", "", "info", "none"],
)
def test_an_unknown_enforcement_value_is_an_error_not_advisory(value, monkeypatch):
    monkeypatch.setattr(checks, "_REGISTRY", dict(checks._REGISTRY))
    with pytest.raises(ValueError, match="enforcement"):
        checks.declare_check(
            "example:enforcement/unknown",
            title="t",
            requires="r",
            why="w",
            fix="f",
            cause="validator",
            gate="ci",
            site="a/b.py:c",
            enforcement=value,
        )
    assert checks.get_check("example:enforcement/unknown") is None


@pytest.mark.parametrize("value", [None, True, 1, ["advisory"], ("advisory",)])
def test_a_non_string_enforcement_value_is_an_error(value):
    problems = checks.check_definition_problems(_check(enforcement=value))
    assert any("enforcement" in problem for problem in problems)


@pytest.mark.parametrize("gate", ["compile", "proof", "infra", "overlay", "review"])
def test_only_a_ci_check_can_be_advisory(gate):
    # Only the CI gate applies advisory enforcement; another gate would still
    # refuse on the finding, so the declaration would be false.
    problems = checks.check_definition_problems(
        _check(gate=gate, enforcement="advisory")
    )
    assert any("only for a check of the `ci` gate" in problem for problem in problems)
    assert checks.check_definition_problems(_check(gate=gate)) == []


def test_exactly_one_check_is_advisory():
    assert checks.advisory_check_ids() == (GATES,)
    assert CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED.id == GATES
    assert CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED.enforcement == ADVISORY
    assert CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED.gate == "ci"
    blocking = [
        check
        for check in checks.load_checks().values()
        if check.enforcement == BLOCKING
    ]
    assert len(blocking) == len(checks.load_checks()) - 1


def test_the_committed_registry_labels_only_that_check():
    registry = json.loads(
        (Path(checks.__file__).parent / checks.COMMITTED_REGISTRY_PATH).read_text()
    )
    labeled = {
        entry["id"]: entry["enforcement"]
        for entry in registry["checks"]
        if "enforcement" in entry
    }
    assert labeled == {GATES: "advisory"}


# --------------------------------------------------------------------------
# What counts as advisory: only a typed issue of a declared advisory check
# --------------------------------------------------------------------------


def test_only_the_typed_finding_of_the_advisory_check_is_advisory():
    assert issue_is_advisory(_gates_issue())
    assert not issue_is_advisory(_other_issue())


def test_a_plain_string_with_the_same_message_blocks():
    # The message text never decides. An issue that lost its type blocks.
    assert not issue_is_advisory(GATES_MESSAGE)
    assert not issue_is_advisory(str(_gates_issue()))
    assert not issue_is_advisory(f"{_gates_issue()}")
    assert not issue_is_advisory(None)
    assert not issue_is_advisory({"check": GATES, "enforcement": "advisory"})


def test_a_typed_issue_of_an_undeclared_check_blocks():
    issue = ValidationIssue(GATES_MESSAGE, check=GATES + "-not-declared")
    assert checks.get_check(issue.check) is None
    assert not issue_is_advisory(issue)


@pytest.mark.parametrize(
    "value", ["warn", "ADVISORY", "Advisory", "", None, True, 1, ["advisory"]]
)
def test_a_registry_entry_with_an_unknown_value_blocks(value, monkeypatch):
    # A value that got past the declaration (a direct registry write) still
    # blocks: advisory needs exactly the string "advisory".
    bad = replace(CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED, enforcement=value)
    monkeypatch.setitem(checks._REGISTRY, GATES, bad)
    assert not issue_is_advisory(_gates_issue())
    assert partition_by_enforcement([_gates_issue()]) == ([_gates_issue()], [])


def test_every_other_declared_check_blocks():
    registry = checks.load_checks()
    assert len(registry) >= 29
    for check_id in registry:
        issue = ValidationIssue(f"finding of {check_id}", check=check_id)
        blocking, advisory = partition_by_enforcement([issue])
        if check_id == GATES:
            assert (blocking, advisory) == ([], [issue])
        else:
            assert (blocking, advisory) == ([issue], []), check_id


def test_partition_keeps_order_and_every_issue():
    first, second = _other_issue("[x] one"), _other_issue("[x] two")
    gates_a, gates_b = _gates_issue(), _gates_issue(GATES_MESSAGE + " again")
    issues = [first, gates_a, "untyped", second, gates_b]
    blocking, advisory = partition_by_enforcement(issues)
    assert blocking == [first, "untyped", second]
    assert advisory == [gates_a, gates_b]
    assert all(isinstance(issue, ValidationIssue) for issue in advisory)
    assert partition_by_enforcement([]) == ([], [])


def test_a_rewritten_issue_keeps_its_enforcement():
    issue = _gates_issue()
    assert issue_is_advisory(checks.retype(issue, "path prefix: " + issue))
    assert issue_is_advisory(checks.map_issue_text(issue, str.upper))
    assert not issue_is_advisory(checks.map_issue_text(_other_issue(), str.upper))


# --------------------------------------------------------------------------
# Records: the label, the detail, and no way to claim advisory from data
# --------------------------------------------------------------------------


def test_an_issue_record_is_labeled_only_when_advisory():
    advisory = checks.issue_record(_gates_issue(), gate="ci")
    assert advisory["enforcement"] == "advisory"
    assert advisory["check"] == advisory["kind"] == GATES
    assert advisory["expected"] == "2 distinct gate inputs"
    assert advisory["found"] == "0 of 2 source gates"
    assert advisory["fingerprint"] == _gates_issue().fingerprint
    assert "enforcement" not in checks.issue_record(_other_issue(), gate="ci")
    assert "enforcement" not in checks.issue_record(GATES_MESSAGE, gate="ci")


def test_advisory_records_keep_the_whole_message_and_leave_out_the_rest():
    long_message = GATES_MESSAGE + " detail" * 1000
    assert len(long_message) > checks.MAX_RECORDED_MESSAGE_CHARS
    records = advisory_issue_records(
        [_other_issue(), _gates_issue(long_message), GATES_MESSAGE, _gates_issue()]
    )
    assert [record["message"] for record in records] == [long_message, GATES_MESSAGE]
    assert all(record["enforcement"] == "advisory" for record in records)
    assert all(record["gate"] == "ci" for record in records)
    assert all("message_truncated" not in record for record in records)
    assert advisory_issue_records([_other_issue(), "untyped"]) == []
    # The same finding twice is recorded once.
    assert len(advisory_issue_records([_gates_issue(), _gates_issue()])) == 1


def test_a_stored_document_cannot_make_a_blocking_issue_advisory():
    # Re-typing reads the check id; enforcement comes from the live registry.
    message = str(_other_issue())
    document = {
        "issues": [
            {
                **checks.issue_record(_other_issue(), gate="ci"),
                "enforcement": "advisory",
            }
        ]
    }
    (retyped,) = checks.issues_from_document([message], document)
    assert isinstance(retyped, ValidationIssue) and retyped.check == OTHER
    assert not issue_is_advisory(retyped)


def test_the_typed_issue_document_labels_the_advisory_record_and_its_check():
    document = checks.typed_issue_document([("ci", [_gates_issue(), _other_issue()])])
    by_check = {record["check"]: record for record in document["issues"]}
    assert by_check[GATES]["enforcement"] == "advisory"
    assert "enforcement" not in by_check[OTHER]
    assert document["checks"][GATES]["enforcement"] == "advisory"
    assert "enforcement" not in document["checks"][OTHER]


# --------------------------------------------------------------------------
# The reviewer text
# --------------------------------------------------------------------------


def test_review_context_is_empty_without_advisory_records():
    assert advisory_review_context([]) == ""
    blocking_record = checks.issue_record(_other_issue(), gate="ci")
    assert advisory_review_context([blocking_record]) == ""
    assert advisory_review_context([{"message": GATES_MESSAGE, "check": GATES}]) == ""
    assert advisory_review_context(["not a record"]) == ""


def test_review_context_shows_the_check_and_the_whole_finding():
    text = advisory_review_context(advisory_issue_records([_gates_issue()]))
    check = CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED
    assert text.startswith("Validator warnings to examine (advisory")
    assert f"[{GATES}] {check.title}. Requires: {check.requires}" in text
    assert check.why in text
    assert f"Warning: {GATES_MESSAGE}" in text
    assert "fail the review" in text
    assert "not proof of a defect" in text


def test_review_context_is_bounded():
    many = [_gates_issue(f"{GATES_MESSAGE} #{index}") for index in range(14)]
    text = advisory_review_context(advisory_issue_records(many))
    assert text.count("Warning: ") == checks.MAX_REVIEW_ADVISORY_ISSUES
    assert "4 more advisory warning(s) not shown" in text
    huge = advisory_review_context(
        advisory_issue_records([_gates_issue(GATES_MESSAGE + "x" * 9000)])
    )
    assert len(huge) < checks.MAX_REVIEW_ADVISORY_MESSAGE_CHARS + 2500


# --------------------------------------------------------------------------
# The CI result: the real construction site in the validator pipeline
# --------------------------------------------------------------------------


def _pipeline(tmp_path, **kwargs):
    policy_repo = tmp_path / "rulespec-us" / "us-or"
    policy_repo.mkdir(parents=True, exist_ok=True)
    return ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=kwargs.pop("local_corpus_release", None),
        source_citation_path=CITATION,
        enable_oracles=False,
        enforce_repository_layout=False,
        **kwargs,
    )


def _rules_file(pipeline):
    rules_file = pipeline.policy_repo_path / "policies/guidance/candidate.yaml"
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    rules_file.write_text(CONTENT)
    return rules_file


def _ci(tmp_path, monkeypatch, completeness_issues, **pipeline_kwargs):
    """Run the real CI step; only the engine, the corpus and the analyzer are stubs."""

    pipeline = _pipeline(tmp_path, **pipeline_kwargs)
    rules_file = _rules_file(pipeline)
    monkeypatch.setattr(
        validator_pipeline, "_fetch_corpus_source_text", lambda _citation: SOURCE
    )
    monkeypatch.setattr(
        pipeline,
        "_source_texts_for_rulespec_content",
        lambda _content: {CITATION: SOURCE},
    )
    monkeypatch.setattr(
        pipeline,
        "_compile_rulespec_to_artifact",
        lambda _path, _output: (
            subprocess.CompletedProcess(["test-engine"], 0, "", ""),
            {"program": {"parameters": [], "derived": [], "relations": []}},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_complete_source_unit_issues",
        lambda *_args, **_kwargs: list(completeness_issues),
    )
    # `_run_ci` is these two calls inside the corpus-release scope.
    result = pipeline._normalize_validation_staging_result(
        pipeline._run_rulespec_ci(rules_file)
    )
    return pipeline, rules_file, result


def test_ci_passes_with_only_the_advisory_finding_and_keeps_it(tmp_path, monkeypatch):
    _pipeline_, _file, baseline = _ci(tmp_path / "a", monkeypatch, [])
    assert baseline.passed and baseline.issues == [] and baseline.advisory_issues == []

    issue = _gates_issue()
    _pipeline_, _file, result = _ci(tmp_path / "b", monkeypatch, [issue])
    assert result.passed is True
    assert result.issues == []
    assert result.error is None
    assert result.advisory_issues == [issue]
    (kept,) = result.advisory_issues
    assert isinstance(kept, ValidationIssue)
    assert (kept.check, kept.subject, kept.expected, kept.found) == (
        issue.check,
        issue.subject,
        issue.expected,
        issue.found,
    )
    assert result.details == baseline.details
    assert result.raw_output == baseline.raw_output


def test_ci_stays_refused_by_another_finding_beside_the_advisory_one(
    tmp_path, monkeypatch
):
    other, gates = _other_issue(), _gates_issue()
    _p, _f, alone = _ci(tmp_path / "a", monkeypatch, [other])
    _p, _f, both = _ci(tmp_path / "b", monkeypatch, [gates, other])
    assert alone.passed is False and both.passed is False
    assert both.issues == alone.issues == [other]
    assert both.error == alone.error == str(other)
    assert both.advisory_issues == [gates]
    assert alone.advisory_issues == []


def test_ci_refuses_the_same_message_when_it_is_not_typed(tmp_path, monkeypatch):
    _p, _f, result = _ci(tmp_path, monkeypatch, [GATES_MESSAGE])
    assert result.passed is False
    assert result.issues == [GATES_MESSAGE]
    assert result.advisory_issues == []


@pytest.mark.parametrize(
    "check_id",
    [
        OTHER,
        "complete-source-unit:tests/exception-pair-missing",
        "complete-source-unit:tests/boundary-unexercised",
        "complete-source-unit:numeric-recall/value-unnamed",
        "complete-source-unit:formula-output/branch-without-output",
        "complete-source-unit:structure/branch-unencoded",
        "proof:source-evidence/not-found",
    ],
)
def test_ci_still_refuses_on_other_checks(tmp_path, monkeypatch, check_id):
    assert checks.get_check(check_id) is not None
    issue = ValidationIssue(f"finding of {check_id}", check=check_id)
    _p, _f, result = _ci(tmp_path, monkeypatch, [issue])
    assert result.passed is False
    assert result.issues == [issue]
    assert result.advisory_issues == []


def test_ci_refuses_when_the_registry_value_is_unknown(tmp_path, monkeypatch):
    bad = replace(CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED, enforcement="warn")
    monkeypatch.setitem(checks._REGISTRY, GATES, bad)
    _p, _f, result = _ci(tmp_path, monkeypatch, [_gates_issue()])
    assert result.passed is False
    assert result.issues == [_gates_issue()]
    assert result.advisory_issues == []


def test_staging_paths_are_removed_from_the_advisory_finding_too(tmp_path, monkeypatch):
    staging = tmp_path / "staging-root"
    staging.mkdir()
    message = f"{GATES_MESSAGE} See {staging}/us-or/policies/x.yaml."
    _p, _f, result = _ci(
        tmp_path, monkeypatch, [_gates_issue(message)], validation_staging_root=staging
    )
    (kept,) = result.advisory_issues
    assert str(staging) not in kept
    assert issue_is_advisory(kept)
    assert result.passed is True


# --------------------------------------------------------------------------
# The real finding, from the analyzer, on committed real candidates
# --------------------------------------------------------------------------


def _analysis(directory: str, block: int):
    folder = FIXTURES / directory
    citation = f"{UNIT}/block-{block}"
    profile = validator_pipeline._numeric_profile_for_citation_path(citation)
    return analyze_complete_source_unit(
        (folder / "candidate.yaml").read_text(),
        (folder / "source.txt").read_text(),
        corpus_citation_path=citation,
        test_cases=yaml.safe_load((folder / "candidate.test.yaml").read_text()),
        extract_numeric_occurrences=functools.partial(
            validator_pipeline.extract_typed_numeric_inventory_occurrences_from_text,
            profile=profile,
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            validator_pipeline.extract_typed_numeric_occurrences_from_text,
            profile=profile,
        ),
        extract_named_scalars=validator_pipeline.extract_named_scalar_occurrences,
        numeric_value_is_grounded=validator_pipeline.numeric_value_is_grounded,
    )


REAL = [("block-14-e9371ed3", 14), ("block-3-07b0428f", 3), ("block-4-170e8d74", 4)]


@pytest.mark.parametrize(("directory", "block"), REAL)
def test_the_real_finding_is_still_computed_and_is_advisory(
    directory, block, monkeypatch
):
    monkeypatch.setattr(
        "axiom_encode.harness.source_completeness.reviewed_alternative_lists_for",
        lambda citation, text: (),
    )
    analysis = _analysis(directory, block)
    found = [issue for issue in analysis.issues if getattr(issue, "check", "") == GATES]
    assert len(found) == 1, "the analyzer still reports the finding"
    blocking, advisory = partition_by_enforcement(analysis.issues)
    assert advisory == found
    assert all(getattr(issue, "check", None) != GATES for issue in blocking)
    assert sorted(map(str, blocking + advisory)) == sorted(map(str, analysis.issues))
    (record,) = advisory_issue_records(analysis.issues)
    assert record["message"] == str(found[0])
    assert record["enforcement"] == "advisory"
    assert record["fingerprint"] == found[0].fingerprint


@pytest.mark.parametrize(("directory", "block"), REAL)
def test_a_reviewed_alternative_list_entry_still_silences_the_finding(directory, block):
    # The registry path is unchanged: with its approved entry there is no
    # finding at all, advisory or blocking.
    analysis = _analysis(directory, block)
    assert not [i for i in analysis.issues if getattr(i, "check", "") == GATES]
    assert advisory_issue_records(analysis.issues) == []


@pytest.mark.parametrize(("directory", "block"), REAL)
def test_the_real_finding_goes_through_the_ci_step_as_advisory(
    directory, block, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "axiom_encode.harness.source_completeness.reviewed_alternative_lists_for",
        lambda citation, text: (),
    )
    issues = list(_analysis(directory, block).issues)
    expected_blocking, expected_advisory = partition_by_enforcement(issues)
    assert len(expected_advisory) == 1
    _p, _f, result = _ci(tmp_path, monkeypatch, issues)
    assert result.advisory_issues == expected_advisory
    assert result.issues == expected_blocking
    assert result.passed is (not expected_blocking)


# --------------------------------------------------------------------------
# Evaluation: verdict, record, reviewer prompt, no retry
# --------------------------------------------------------------------------


def _evaluate(tmp_path, monkeypatch, ci_result, *, prompts=None, **kwargs):
    pipeline = _pipeline(tmp_path)
    rules_file = _rules_file(pipeline)
    monkeypatch.setattr(
        validator_pipeline, "_fetch_corpus_source_text", lambda _citation: SOURCE
    )
    monkeypatch.setattr(evals, "ValidatorPipeline", lambda **_kwargs: pipeline)
    monkeypatch.setattr(
        evals,
        "_rulespec_validation_target",
        lambda *_args, **_kwargs: contextlib.nullcontext(rules_file),
    )
    monkeypatch.setattr(
        pipeline, "_run_compile_check", lambda _path: ValidationResult("compile", True)
    )
    monkeypatch.setattr(pipeline, "_run_ci", lambda _path: ci_result)

    def fake_reviewer_cli(prompt, **_kwargs):
        if prompts is not None:
            prompts.append(prompt)
        return json.dumps({"score": 9, "passed": True, "issues": []}), 0

    monkeypatch.setattr(validator_pipeline, "run_claude_code", fake_reviewer_cli)
    kwargs.setdefault("skip_reviewers", prompts is None)
    return evals._evaluate_artifact_in_scope(
        rulespec_file=rules_file,
        policy_repo_root=pipeline.policy_repo_path,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        source_text=SOURCE,
        source_citation_path=CITATION,
        local_corpus_release=None,
        **kwargs,
    )


def _advisory_ci_result(*extra_blocking):
    return ValidationResult(
        "ci",
        passed=not extra_blocking,
        issues=list(extra_blocking),
        error=str(extra_blocking[0]) if extra_blocking else None,
        advisory_issues=[_gates_issue()],
    )


def test_evaluation_passes_and_records_the_advisory_finding(tmp_path, monkeypatch):
    metrics = _evaluate(tmp_path, monkeypatch, _advisory_ci_result())
    assert metrics.ci_pass is True
    assert metrics.ci_issues == []
    assert evals._eval_artifact_validation_error(metrics) is None
    (record,) = metrics.advisory_issues
    assert record["check"] == GATES
    assert record["enforcement"] == "advisory"
    assert record["message"] == GATES_MESSAGE
    assert record["gate"] == "ci"
    assert record["expected"] == "2 distinct gate inputs"
    assert record["found"] == "0 of 2 source gates"
    assert record["fingerprint"] == _gates_issue().fingerprint


def test_evaluation_stays_refused_by_another_finding(tmp_path, monkeypatch):
    other = _other_issue()
    metrics = _evaluate(tmp_path, monkeypatch, _advisory_ci_result(other))
    assert metrics.ci_pass is False
    assert metrics.ci_issues == [other]
    assert (
        evals._eval_artifact_validation_error(metrics)
        == "Generated RuleSpec failed CI validation"
    )
    assert [record["check"] for record in metrics.advisory_issues] == [GATES]


def test_the_advisory_finding_reaches_the_reviewer_prompt(tmp_path, monkeypatch):
    prompts: list[str] = []
    metrics = _evaluate(
        tmp_path,
        monkeypatch,
        _advisory_ci_result(),
        prompts=prompts,
        reviewers_require_deterministic_pass=True,
    )
    # The candidate passed the deterministic gates, so the reviewer ran.
    (prompt,) = prompts
    assert "## Review Context" in prompt
    assert "Validator warnings to examine (advisory" in prompt
    assert f"[{GATES}]" in prompt
    assert f"Warning: {GATES_MESSAGE}" in prompt
    assert CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED.requires in prompt
    assert metrics.generalist_review_pass is True
    assert metrics.generalist_review_prompt_sha256


def test_the_reviewer_prompt_is_unchanged_without_an_advisory_finding(
    tmp_path, monkeypatch
):
    plain, warned = [], []
    _evaluate(
        tmp_path / "a", monkeypatch, ValidationResult("ci", passed=True), prompts=plain
    )
    _evaluate(tmp_path / "b", monkeypatch, _advisory_ci_result(), prompts=warned)
    assert "Validator warnings" not in plain[0]
    assert "advisory" not in plain[0].lower()
    section = "\n\n" + advisory_review_context(advisory_issue_records([_gates_issue()]))
    assert section in warned[0]
    # Apart from the temp path of the artifact, the warning is the only
    # difference between the two prompts.
    normalize = lambda text, root: text.replace(str(root), "<root>")  # noqa: E731
    assert normalize(warned[0], tmp_path / "b").replace(section, "") == normalize(
        plain[0], tmp_path / "a"
    )


def test_a_refused_candidate_gets_no_reviewer_call_in_the_encode_lane(
    tmp_path, monkeypatch
):
    prompts: list[str] = []
    _evaluate(
        tmp_path,
        monkeypatch,
        _advisory_ci_result(_other_issue()),
        prompts=prompts,
        reviewers_require_deterministic_pass=True,
    )
    assert prompts == []


def _result(metrics, *, success=True):
    return EvalResult(
        citation=CITATION,
        runner="codex:test",
        backend="codex",
        model="test",
        mode="cold",
        output_file="candidate.yaml",
        trace_file="candidate.trace.json",
        context_manifest_file="candidate.context.json",
        generated_output_sha256="a" * 64,
        trace_sha256="b" * 64,
        context_manifest_sha256="c" * 64,
        duration_ms=0,
        success=success,
        error=None if success else "Generated RuleSpec failed CI validation",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        reasoning_output_tokens=0,
        estimated_cost_usd=None,
        actual_cost_usd=None,
        retrieved_files=[],
        unexpected_accesses=[],
        metrics=metrics,
    )


def _metrics(*, advisory=(), issues=()):
    return EvalArtifactMetrics(
        compile_pass=True,
        compile_issues=[],
        ci_pass=not issues,
        ci_issues=list(issues),
        embedded_source_present=False,
        grounded_numeric_count=0,
        ungrounded_numeric_count=0,
        grounding=[],
        advisory_issues=list(advisory),
    )


def test_the_result_json_carries_the_finding_and_round_trips(tmp_path, monkeypatch):
    metrics = _evaluate(tmp_path, monkeypatch, _advisory_ci_result())
    payload = _result(metrics).to_dict()
    assert payload["metrics"]["advisory_issues"] == metrics.advisory_issues
    assert payload["metrics"]["ci_pass"] is True
    assert payload["metrics"]["ci_issues"] == []
    json.dumps(payload)
    restored = evals._eval_result_from_payload(payload)
    assert asdict(restored.metrics) == asdict(metrics)
    assert restored.success


def test_a_result_without_advisory_findings_keeps_its_serialized_shape():
    payload = _result(_metrics()).to_dict()
    assert "advisory_issues" not in payload["metrics"]
    assert evals._eval_result_from_payload(payload).metrics.advisory_issues == []


def test_the_advisory_finding_alone_starts_no_retry(tmp_path, monkeypatch):
    metrics = _evaluate(tmp_path, monkeypatch, _advisory_ci_result())
    result = _result(metrics)
    assert not cli._encode_attempt_was_validator_rejected(result, {})
    assert not cli._encode_attempt_was_validator_rejected(
        result, {"apply_requested": True, "status": "applied"}
    )


def test_retry_feedback_for_another_finding_leaves_the_advisory_one_out():
    advisory = advisory_issue_records([_gates_issue()])
    failure = cli._FailedEncodeAttempt(
        result=_result(
            _metrics(advisory=advisory, issues=("missing companion case",)),
            success=False,
        ),
        error="Generated RuleSpec failed CI validation",
        validation_issues=("missing companion case",),
        validation_issue_count=1,
        full_validation_issues=("missing companion case",),
    )
    feedback = cli._encode_validation_retry_feedback([failure])
    assert feedback == (failure.error, "missing companion case")
    assert GATES_MESSAGE not in evals._format_validation_retry_feedback(feedback)


def test_resume_revalidation_compares_advisory_findings():
    # They are deterministic validator output, unlike reviewer text.
    advisory = advisory_issue_records([_gates_issue()])
    assert evals._reviewer_independent_metrics(
        _metrics(advisory=advisory)
    ) != evals._reviewer_independent_metrics(_metrics())
    assert "advisory_issues" in evals._reviewer_independent_metrics(_metrics())


# --------------------------------------------------------------------------
# validate(): every reviewer gets the warning; the event log records it
# --------------------------------------------------------------------------


def _validate(tmp_path, monkeypatch, ci_result):
    release = object.__new__(LocalCorpusRelease)
    pipeline = _pipeline(tmp_path, local_corpus_release=release)
    rules_file = _rules_file(pipeline)
    seen: dict[str, str | None] = {}
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        pipeline, "_run_compile_check", lambda _path: ValidationResult("compile", True)
    )
    monkeypatch.setattr(pipeline, "_run_ci", lambda _path: ci_result)

    def reviewer(name, _file, _oracle=None, review_context=None):
        seen[name] = review_context
        return ValidationResult(name, passed=True, score=9.0)

    monkeypatch.setattr(pipeline, "_run_reviewer", reviewer)
    monkeypatch.setattr(
        pipeline,
        "_log_event",
        lambda kind, _content, metadata=None: events.append((kind, metadata or {})),
    )
    result = pipeline._validate_with_authoritative_roots(rules_file)
    return result, seen, dict(events)


def test_validate_gives_every_reviewer_the_warning_and_logs_it(tmp_path, monkeypatch):
    result, seen, events = _validate(tmp_path, monkeypatch, _advisory_ci_result())
    expected = advisory_review_context(advisory_issue_records([_gates_issue()]))
    assert len(seen) == 4
    assert set(seen.values()) == {expected}
    assert result.ci_pass is True
    assert result.results["ci"].advisory_issues == [_gates_issue()]
    (logged,) = events["validation_ci_end"]["advisory_issues"]
    assert logged["check"] == GATES and logged["enforcement"] == "advisory"
    assert events["validation_ci_end"]["passed"] is True
    assert events["validation_ci_end"]["issues"] == []


def test_validate_is_unchanged_without_an_advisory_finding(tmp_path, monkeypatch):
    _result_, seen, events = _validate(
        tmp_path, monkeypatch, ValidationResult("ci", passed=True)
    )
    assert set(seen.values()) == {None}
    assert set(events["validation_ci_end"]) == {"passed", "issues", "duration_ms"}


# --------------------------------------------------------------------------
# Run log export
# --------------------------------------------------------------------------


def _ci_event(metrics):
    (event,) = [
        event
        for event in run_log_export._gate_events_from_metrics(metrics)
        if event["stage"] == "gate.ci"
    ]
    return event


def test_the_run_log_shows_the_advisory_finding_labeled_on_a_passed_gate():
    advisory = advisory_issue_records([_gates_issue()])
    event = _ci_event(_metrics(advisory=advisory))
    assert event["status"] == StageStatus.passed
    assert event["reason_code"] is None
    assert event["attrs"] == {
        "advisory_finding_count": 1,
        "advisory_checks": [GATES],
    }
    (finding,) = event["findings"]
    assert finding.code == GATES
    assert finding.severity == Severity.info
    assert finding.message == f"[advisory] {GATES_MESSAGE}"
    assert finding.locator == "rule:rule_a"


def test_the_run_log_keeps_blocking_findings_critical_beside_the_advisory_one():
    advisory = advisory_issue_records([_gates_issue()])
    event = _ci_event(_metrics(advisory=advisory, issues=[_other_issue()]))
    assert event["status"] == StageStatus.failed
    assert event["reason_code"] == "ci_failure"
    assert [(f.code, f.severity) for f in event["findings"]] == [
        (OTHER, Severity.critical),
        (GATES, Severity.info),
    ]


def test_the_run_log_ci_event_is_unchanged_without_advisory_findings():
    event = _ci_event(_metrics())
    assert set(event) == {"stage", "status", "reason_code", "findings"}
    assert event["findings"] == []
    # A record that is not labeled advisory is never shown as one.
    unlabeled = [{"check": GATES, "message": GATES_MESSAGE}]
    assert _ci_event(_metrics(advisory=unlabeled))["findings"] == []


# --------------------------------------------------------------------------
# The replay benchmark keeps advisory findings apart from the verdict
# --------------------------------------------------------------------------

REPLAY_FIXTURES = replay.DEFAULT_FIXTURE_ROOT
ONLY_GATES_FIXTURE = "36596173514"
GATES_AND_MORE_FIXTURE = "local-t2036-country-case-selection-20261113"
GATES_CATEGORY = "[complete-source-unit:source-explicit-conditions]"
TESTS_FINDING = "[complete-source-unit:tests] missing paired case"


def test_replay_reports_the_finding_as_advisory_and_the_case_as_clean():
    result = replay.replay_case(
        replay.load_fixture(REPLAY_FIXTURES / ONLY_GATES_FIXTURE)
    )
    assert result["error"] is None
    assert result["issues"] == [] and result["rules"] == {}
    (advisory,) = result["advisory"]
    assert advisory.startswith(GATES_CATEGORY)
    assert type(advisory) is str


def test_replay_keeps_a_case_refused_by_its_other_findings():
    result = replay.replay_case(
        replay.load_fixture(REPLAY_FIXTURES / GATES_AND_MORE_FIXTURE)
    )
    assert len(result["advisory"]) == 1
    assert result["issues"], "the other findings still refuse the case"
    assert not any(issue.startswith(GATES_CATEGORY) for issue in result["issues"])
    assert sum(result["rules"].values()) == len(result["issues"])


def test_the_committed_snapshot_pins_advisory_findings_apart():
    snapshot = json.loads((REPLAY_FIXTURES / replay.SNAPSHOT_FILE).read_text())
    advisory_rows = {
        case_id: row["advisory_findings"]
        for case_id, row in snapshot["cases"].items()
        if "advisory_findings" in row
    }
    assert advisory_rows, "the benchmark has fixtures with the finding"
    assert snapshot["totals"]["advisory"] == sum(map(len, advisory_rows.values()))
    for case_id, row in snapshot["cases"].items():
        assert not any(f.startswith(GATES_CATEGORY) for f in row["findings"]), case_id
        for finding in row.get("advisory_findings", []):
            assert finding.startswith(GATES_CATEGORY), case_id
    assert snapshot["cases"][ONLY_GATES_FIXTURE]["issues"] == 0
    assert snapshot["cases"][GATES_AND_MORE_FIXTURE]["issues"] > 0


def _replay_row(case_id, issues, advisory=()):
    return {
        "id": case_id,
        "citation": "us/statute/26/32",
        "issues": list(issues),
        "advisory": list(advisory),
        "rules": replay.rule_counts(issues),
        "error": None,
        "control": False,
        "imports_unresolved": [],
        "deferral_providers_unresolved": [],
    }


def _replay_report(rows):
    return {
        "schema": replay.REPORT_SCHEMA,
        "encoder_version": "0.0.0",
        "context": "workspace",
        "seconds": 0.1,
        "summary": {
            "cases": len(rows),
            "clean": sum(1 for row in rows if not row["issues"]),
            "errors": 0,
            "issues": sum(len(row["issues"]) for row in rows),
            "rules": {},
        },
        "cases": rows,
        "load_errors": [],
    }


def test_the_snapshot_names_advisory_findings_only_where_there_are_some():
    snapshot = replay.snapshot_from_report(
        _replay_report(
            [_replay_row("a", [TESTS_FINDING], [GATES_MESSAGE]), _replay_row("b", [])]
        )
    )
    assert snapshot["cases"]["a"]["advisory_findings"] == [GATES_MESSAGE]
    assert snapshot["cases"]["a"]["findings"] == [TESTS_FINDING]
    assert snapshot["cases"]["a"]["issues"] == 1
    assert "advisory_findings" not in snapshot["cases"]["b"]
    assert snapshot["totals"]["advisory"] == 1
    plain = replay.snapshot_from_report(_replay_report([_replay_row("b", [])]))
    assert "advisory" not in plain["totals"]


def test_a_changed_advisory_finding_is_snapshot_drift():
    pinned = replay.snapshot_from_report(
        _replay_report([_replay_row("a", [], [GATES_MESSAGE])])
    )
    same = _replay_report([_replay_row("a", [], [GATES_MESSAGE])])
    assert replay.snapshot_drift(pinned, same) == []
    gone = _replay_report([_replay_row("a", [])])
    assert replay.snapshot_drift(pinned, gone) == [f"a advisory - {GATES_MESSAGE}"]
    # The finding turned blocking again: both lists changed.
    blocking = _replay_report([_replay_row("a", [GATES_MESSAGE])])
    assert replay.snapshot_drift(pinned, blocking) == [
        f"a + {GATES_MESSAGE}",
        f"a advisory - {GATES_MESSAGE}",
    ]


def test_a_fixture_that_is_clean_only_by_an_advisory_finding_is_not_guarded():
    base = replay.snapshot_from_report(
        _replay_report(
            [_replay_row("warned", [], [GATES_MESSAGE]), _replay_row("clean", [])]
        )
    )
    after = _replay_report(
        [_replay_row("warned", [GATES_MESSAGE]), _replay_row("clean", [TESTS_FINDING])]
    )
    violations = replay.guard_violations(base, after)
    assert len(violations) == 1 and violations[0].startswith("clean ")
