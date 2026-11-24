"""A cited child path that the resolver cannot slice refuses the candidate.

The resolver slices a child citation such as ``.../900/b/1`` out of its
section row. When it cannot (``CorpusSourceSliceError``), CI used to stop
with that exception and a bulk runner counted the attempt as an
infrastructure error. It is now one blocking typed finding for the
candidate. The unit's own path still raises.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from axiom_encode.checks import load_checks, partition_by_enforcement
from axiom_encode.corpus_resolver import (
    CorpusSourceSliceError,
    LocalCorpusRelease,
    resolve_local_corpus_source,
)
from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.evals import evaluate_artifact
from axiom_encode.harness.validator_pipeline import (
    CHECK_PROOF_SOURCE_PATH_UNSLICEABLE,
    ValidationResult,
    ValidatorPipeline,
)
from tests.release_object_fixtures import bind_test_corpus_release

SECTION = "us/regulation/42/435/900"
VERSION = "test-version"
CHECK_ID = "proof:source-evidence/path-unsliceable"

# Synthetic section bodies. The first holds a child marker inline after a
# heading; the resolver cannot isolate `b/1` from it. The second repeats a
# marker; the resolver refuses `b/2` as a malformed sequence.
INLINE_HEADING_BODY = (
    "(a) Scope. The agency must act.\n\n"
    "(b) Eligibility. (1) Criteria. The agency must decide within 45 days.\n\n"
    "(2) Second item.\n\n"
    "(c) Next paragraph."
)
DUPLICATE_MARKER_BODY = (
    "(a) Scope. The agency must decide within 45 days.\n\n"
    "(b) Rules.\n\n"
    "(1) One.\n\n"
    "(2) Two.\n\n"
    "(2) Two again.\n\n"
    "(c) Next."
)
SLICE_CASES = [
    pytest.param(
        INLINE_HEADING_BODY,
        f"{SECTION}/b/1",
        "missing structural marker path",
        id="missing-structural-marker",
    ),
    pytest.param(
        DUPLICATE_MARKER_BODY,
        f"{SECTION}/b/2",
        "Malformed duplicate or backward legal marker sequence",
        id="malformed-marker-sequence",
    ),
]


def _release(tmp_path: Path, body: str) -> LocalCorpusRelease:
    corpus_root = tmp_path / "axiom-corpus"
    provisions = corpus_root / "data" / "corpus" / "provisions" / "us" / "regulation"
    provisions.mkdir(parents=True, exist_ok=True)
    record = {
        "id": f"test:{SECTION}",
        "citation_path": SECTION,
        "body": body,
        "jurisdiction": "us",
        "document_class": "regulation",
        "version": VERSION,
        "source_path": "sources/us/regulation/test",
        "source_as_of": "2026-01-01",
        "expression_date": "2026-01-01",
    }
    (provisions / f"{VERSION}.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    return bind_test_corpus_release(
        corpus_root, "test-release", [("us", "regulation", VERSION)]
    )


def _candidate(proof_path: str, *, interpretation_path: str | None = None) -> str:
    interpretations = ""
    if interpretation_path is not None:
        interpretations = f"""  interpretations:
    - id: decision-period-reading
      question: Which period applies?
      texts:
        - corpus_citation_path: {interpretation_path}
          excerpt: within 45 days
      reading: The stated period applies.
      alternatives:
        - id: other-period
          reading: Another period applies.
      basis: The source states the period.
      status: open
      affects:
        - decision_days
"""
    return f"""format: rulespec/v1
module:
  summary: The agency must decide within 45 days.
  source_verification:
    corpus_citation_path: {SECTION}
{interpretations}rules:
  - name: decision_days
    kind: parameter
    dtype: Number
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: parameter
            source:
              corpus_citation_path: {proof_path}
              excerpt: within 45 days
    versions:
      - effective_from: '2026-01-01'
        formula: '45'
"""


def _pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    body: str,
    content: str,
    unit_path: str = SECTION,
) -> tuple[ValidatorPipeline, Path]:
    release = _release(tmp_path, body)
    policy_repo = tmp_path / "rulespec-us" / "us"
    rules_file = policy_repo / "policies" / "example" / "rules.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(content, encoding="utf-8")
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
        enforce_repository_layout=False,
        local_corpus_release=release,
        source_citation_path=unit_path,
        source_text=body,
    )
    monkeypatch.setattr(
        pipeline,
        "_compile_rulespec_to_artifact",
        lambda *_args: (
            subprocess.CompletedProcess(["axiom-rules-engine"], 0, "", ""),
            {"program": {"parameters": [], "derived": [], "relations": []}},
        ),
    )
    return pipeline, rules_file


def _run_ci_without_refusal(
    pipeline: ValidatorPipeline,
    rules_file: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Run CI as before this change: no slice-failure scope at all."""

    with monkeypatch.context() as patch:
        patch.setattr(
            ValidatorPipeline, "_citation_slice_refusal_own_path", lambda self: None
        )
        return pipeline._run_ci(rules_file)


@pytest.mark.parametrize(("body", "child_path", "message"), SLICE_CASES)
def test_the_synthetic_child_path_raises_the_slicer_error(
    tmp_path, body, child_path, message
):
    release = _release(tmp_path, body)

    assert resolve_local_corpus_source(SECTION, release).body == body
    with pytest.raises(CorpusSourceSliceError, match=message):
        resolve_local_corpus_source(child_path, release)


@pytest.mark.parametrize(("body", "child_path", "message"), SLICE_CASES)
def test_an_unsliceable_cited_child_path_is_a_typed_refusal(
    tmp_path, monkeypatch, body, child_path, message
):
    pipeline, rules_file = _pipeline(
        tmp_path, monkeypatch, body=body, content=_candidate(child_path)
    )
    with pytest.raises(CorpusSourceSliceError, match=message):
        _run_ci_without_refusal(pipeline, rules_file, monkeypatch)

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    typed = [
        issue for issue in result.issues if getattr(issue, "check", None) == CHECK_ID
    ]
    assert len(typed) == 1
    issue = typed[0]
    assert result.issues[0] is issue
    assert result.error == issue
    assert issue.subject.citation == child_path
    assert message in issue.found
    assert f"`{child_path}` cannot be sliced" in issue
    assert f"Cite the unit's own path `{SECTION}`" in issue
    assert "verbatim excerpt" in issue
    assert '"' not in issue
    # Blocking, never advisory: it alone refuses the candidate.
    blocking, advisory = partition_by_enforcement([issue])
    assert blocking == [issue]
    assert advisory == []


@pytest.mark.parametrize(("body", "child_path", "message"), SLICE_CASES)
def test_an_unsliceable_interpretation_path_is_a_typed_refusal(
    tmp_path, monkeypatch, body, child_path, message
):
    pipeline, rules_file = _pipeline(
        tmp_path,
        monkeypatch,
        body=body,
        content=_candidate(SECTION, interpretation_path=child_path),
    )
    with pytest.raises(CorpusSourceSliceError, match=message):
        _run_ci_without_refusal(pipeline, rules_file, monkeypatch)

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert [
        issue.subject.citation
        for issue in result.issues
        if getattr(issue, "check", None) == CHECK_ID
    ] == [child_path]


@pytest.mark.parametrize(("body", "child_path", "message"), SLICE_CASES)
def test_the_same_candidate_citing_the_unit_path_is_unchanged(
    tmp_path, monkeypatch, body, child_path, message
):
    pipeline, rules_file = _pipeline(
        tmp_path, monkeypatch, body=body, content=_candidate(SECTION)
    )

    before = _run_ci_without_refusal(pipeline, rules_file, monkeypatch)
    after = pipeline._run_ci(rules_file)

    assert after.passed == before.passed
    assert after.issues == before.issues
    assert [getattr(issue, "check", None) for issue in after.issues] == [
        getattr(issue, "check", None) for issue in before.issues
    ]
    assert after.error == before.error
    assert after.advisory_issues == before.advisory_issues
    assert not any(getattr(issue, "check", None) == CHECK_ID for issue in after.issues)


@pytest.mark.parametrize(("body", "child_path", "message"), SLICE_CASES)
def test_an_unsliceable_unit_path_still_raises(
    tmp_path, monkeypatch, body, child_path, message
):
    pipeline, rules_file = _pipeline(
        tmp_path,
        monkeypatch,
        body=body,
        content=_candidate(child_path).replace(
            f"corpus_citation_path: {SECTION}\n",
            f"corpus_citation_path: {child_path}\n",
        ),
        unit_path=child_path,
    )

    with pytest.raises(CorpusSourceSliceError, match=message):
        _run_ci_without_refusal(pipeline, rules_file, monkeypatch)
    with pytest.raises(CorpusSourceSliceError, match=message):
        pipeline._run_ci(rules_file)


@pytest.mark.parametrize(("body", "child_path", "message"), SLICE_CASES)
def test_fetchers_record_only_inside_a_scope_with_the_unit_path(
    tmp_path, body, child_path, message
):
    release = _release(tmp_path, body)
    fetchers = (
        validator_pipeline._fetch_corpus_source_text,
        validator_pipeline._fetch_corpus_proof_evidence_text,
    )
    with validator_pipeline._authoritative_corpus_scope(release):
        for fetch in fetchers:
            # Outside a scope, and in a suspended scope, nothing changes.
            with pytest.raises(CorpusSourceSliceError, match=message):
                fetch(child_path)
            with validator_pipeline._citation_slice_failure_scope(SECTION):
                with validator_pipeline._citation_slice_failure_scope(None):
                    with pytest.raises(CorpusSourceSliceError, match=message):
                        fetch(child_path)
            # The unit's own path raises inside a scope too.
            with validator_pipeline._citation_slice_failure_scope(child_path):
                with pytest.raises(CorpusSourceSliceError, match=message):
                    fetch(child_path)
            with validator_pipeline._citation_slice_failure_scope(SECTION) as failures:
                assert fetch(child_path) is None
                assert fetch(SECTION) is not None
            assert failures is not None
            assert list(failures.failures) == [child_path]
            assert message in failures.failures[child_path]
            issues = validator_pipeline._citation_slice_failure_issues(failures)
            assert [issue.check for issue in issues] == [CHECK_ID]


def test_the_check_is_registered_as_a_blocking_ci_check():
    check = load_checks()[CHECK_ID]

    assert check is CHECK_PROOF_SOURCE_PATH_UNSLICEABLE
    assert check.gate == "ci"
    assert check.enforcement == "blocking"
    assert check.cause == "validator"
    assert check.category == "proof:source-evidence"


@pytest.mark.parametrize(("body", "child_path", "message"), SLICE_CASES)
def test_the_eval_numeric_pass_refuses_instead_of_raising(
    tmp_path, monkeypatch, body, child_path, message
):
    """The eval re-reads proof sources after CI; a stub CI that passes shows
    that this second read refuses on its own."""

    release = _release(tmp_path, body)
    policy_repo = tmp_path / "rulespec-us" / "us"
    rules_file = policy_repo / "policies" / "example" / "rules.yaml"
    rules_file.parent.mkdir(parents=True)

    def evaluate(proof_path: str):
        rules_file.write_text(_candidate(proof_path), encoding="utf-8")
        with (
            patch.object(
                ValidatorPipeline,
                "_run_compile_check",
                return_value=ValidationResult("compile", True, issues=[]),
            ),
            patch.object(
                ValidatorPipeline,
                "_run_ci",
                return_value=ValidationResult("ci", True, issues=[]),
            ),
        ):
            return evaluate_artifact(
                rulespec_file=rules_file,
                policy_repo_root=policy_repo,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                source_text=body,
                local_corpus_release=release,
                source_citation_path=SECTION,
            )

    with monkeypatch.context() as disabled:
        disabled.setattr(
            ValidatorPipeline, "_citation_slice_refusal_own_path", lambda self: None
        )
        with pytest.raises(CorpusSourceSliceError, match=message):
            evaluate(child_path)
        before = evaluate(SECTION)

    refused = evaluate(child_path)
    after = evaluate(SECTION)

    assert refused.ci_pass is False
    assert [
        issue.subject.citation
        for issue in refused.ci_issues
        if getattr(issue, "check", None) == CHECK_ID
    ] == [child_path]
    assert after.ci_pass == before.ci_pass
    assert after.ci_issues == before.ci_issues
