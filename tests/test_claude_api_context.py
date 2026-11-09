"""Prompt fidelity and infrastructure boundaries for the API exam."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode import cli
from axiom_encode.harness import evals
from tests.eval_evidence_fixtures import install_test_eval_evidence_keys
from tests.test_evals import (
    _fake_eval_result,
    _fake_source_runner,
    _strict_eval_suite_manifest_payload,
    _write_test_corpus_provision,
)


def _prompt_workspace(root, contents):
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.txt"
    source.write_text("The annual allowance is £100.\n")
    items = []
    for index, content in enumerate(contents):
        path = root / f"context-{index}.txt"
        path.write_bytes(content.encode("utf-8"))
        items.append(
            evals.EvalContextFile(
                source_path=str(path),
                workspace_path=path.name,
                import_path=f"uk:context/{index}",
                kind="precedent",
            )
        )
    return evals.EvalWorkspace(root, source, root / "manifest.json"), items


def _prompt(workspace, items, backend):
    return evals._build_eval_prompt(
        "uk/statute/2020/1/section/1",
        "cold",
        workspace,
        items,
        "statutes/2020/1/1.yaml",
        include_tests=True,
        runner_backend=backend,
    )


def test_openai_prompt_matches_b314efc8_fixture(tmp_path):
    workspace, items = _prompt_workspace(
        tmp_path, ["  context head\n" + "x" * 6100 + "\ncontext tail  \n"]
    )
    expected = Path(__file__).with_name("fixtures") / "openai-prompt-b314efc8.txt"
    assert _prompt(workspace, items, "openai").encode() == expected.read_bytes()


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    st.lists(
        st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=8000),
        min_size=1,
        max_size=4,
    )
)
def test_claude_api_prompt_preserves_every_context_byte(tmp_path, contents):
    workspace, items = _prompt_workspace(tmp_path, contents)
    prompt = _prompt(workspace, items, "claude-api")
    assert "You do not have filesystem tool access" in prompt
    for content in contents:
        assert content.encode() in prompt.encode()


def test_claude_api_prompt_preserves_long_context_and_whitespace(tmp_path):
    contents = [" \r\n" + "a" * 12001 + "\r\n\t ", " \n\t"]
    workspace, items = _prompt_workspace(tmp_path, contents)
    # Amendment copies must also retain their exact bytes in the API prompt.
    items[1].kind = "corpus_amendment_act"
    prompt = _prompt(workspace, items, "claude-api")
    assert all(content.encode() in prompt.encode() for content in contents)


@pytest.mark.parametrize(
    "spec", ["claude-api:claude-opus-5-5", "opus=claude-api:claude-opus-5-5"]
)
def test_parse_claude_api_runner(spec):
    runner = evals.parse_runner_spec(spec)
    assert runner.backend == "claude-api"
    assert runner.model == "claude-opus-5-5"


@given(st.text(), st.text(), st.lists(st.booleans(), min_size=25, max_size=25))
def test_credit_balance_error_is_usage_limit(prefix, suffix, cases):
    phrase = "credit balance is too low"
    varied = "".join(c.upper() if flag else c for c, flag in zip(phrase, cases))
    result = _fake_eval_result("claude-api-opus", "fixture")
    result.error = prefix + varied + suffix
    assert evals._eval_result_indicates_usage_limit(result)


def test_claude_api_missing_key_is_authentication(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = _fake_eval_result("claude-api-opus", "fixture")
    result.backend = "claude-api"
    result.output_file = ""
    result.success = False
    result.failure_kind = "error"
    result.error = "ANTHROPIC_API_KEY is required"
    result.input_tokens = result.output_tokens = 0
    assert (
        evals._eval_result_generation_infrastructure_failure(result) == "authentication"
    )


def test_claude_api_overflow_is_infrastructure_even_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = _fake_eval_result("claude-api-opus", "fixture")
    result.backend = "claude-api"
    result.output_file = ""
    result.success = False
    result.failure_kind = "error"
    result.error = "Claude API context overflow (prompt_chars=1200000)"
    assert (
        evals._eval_result_generation_infrastructure_failure(result)
        == "context-overflow"
    )
    assert not evals._suite_case_results_should_retry([result])
    summary = evals.summarize_readiness([result], evals.EvalReadinessGates())
    assert summary.total_cases == 0
    assert summary.infrastructure_error_count == 1


def test_context_overflow_does_not_mark_readiness_ready():
    passed = _fake_eval_result("claude-api-opus", "passed")
    overflow = _fake_eval_result("claude-api-opus", "overflow")
    overflow.backend = "claude-api"
    overflow.success = False
    overflow.output_file = ""
    overflow.failure_kind = "error"
    overflow.error = "Claude API context overflow (prompt_chars=1200000)"
    summary = evals.summarize_readiness([passed, overflow], evals.EvalReadinessGates())
    assert summary.success_rate == 1.0
    assert not summary.ready


@pytest.mark.parametrize("skip_reviewers", [False, True])
def test_suite_threads_skip_reviewers(tmp_path, monkeypatch, skip_reviewers):
    install_test_eval_evidence_keys(monkeypatch)
    payload = _strict_eval_suite_manifest_payload()
    manifest_path = tmp_path / "suite.yaml"
    manifest_path.write_text(json.dumps(payload))
    manifest = evals.load_eval_suite_manifest(manifest_path)
    result = _fake_eval_result("openai-gpt-5.4", "fixture")
    with patch.object(
        evals, "run_source_eval", side_effect=_fake_source_runner([result])
    ) as run:
        evals.run_eval_suite(
            manifest,
            tmp_path / "out",
            tmp_path / "engine",
            tmp_path / "rulespec-us",
            _write_test_corpus_provision(tmp_path),
            skip_reviewers=skip_reviewers,
        )
    assert run.call_args.kwargs["skip_reviewers"] is skip_reviewers
    state = json.loads((tmp_path / "out/suite-run.json").read_text())
    assert state["execution_identity"].get("skip_reviewers", False) is skip_reviewers


@pytest.mark.parametrize("interrupt_rewind", [False, True])
def test_resume_reruns_credit_exhausted_case(tmp_path, monkeypatch, interrupt_rewind):
    install_test_eval_evidence_keys(monkeypatch)
    payload = _strict_eval_suite_manifest_payload()
    manifest_path = tmp_path / "suite.yaml"
    manifest_path.write_text(json.dumps(payload))
    manifest = evals.load_eval_suite_manifest(manifest_path)
    release = _write_test_corpus_provision(tmp_path)
    limited = _fake_eval_result("openai-gpt-5.4", "fixture")
    limited.error = "CREDIT BALANCE IS TOO LOW"
    limited.success = False
    limited.failure_kind = "error"
    replacement = _fake_eval_result("openai-gpt-5.4", "fixture")
    options = dict(
        manifest=manifest,
        output_root=tmp_path / "out",
        axiom_rules_path=tmp_path / "engine",
        policy_repo_path=tmp_path / "rulespec-us",
        corpus_release=release,
        suite_retry_attempts=0,
    )
    with patch.object(
        evals,
        "run_source_eval",
        side_effect=_fake_source_runner([limited], [replacement]),
    ) as run:
        evals.run_eval_suite(**options)
        assert (
            json.loads((tmp_path / "out/suite-run.json").read_text())["status"]
            == "failed"
        )
        ledger = tmp_path / "out/suite-results.jsonl"
        recorded = ledger.read_bytes()
        verified = cli._load_verified_eval_suite_artifacts(
            output_root=options["output_root"],
            manifest=manifest,
            effective_runners=manifest.runners,
            axiom_rules_path=options["axiom_rules_path"],
            policy_repo_path=options["policy_repo_path"],
            corpus_release=release,
            require_complete=False,
            revalidate_persisted_results=False,
            suite_retry_attempts=0,
            admission_score=False,
        )
        assert len(verified["results"]) == 1
        assert ledger.read_bytes() == recorded
        if interrupt_rewind:
            atomic_write = evals._secure_atomic_eval_write

            def interrupt_after_ledger(root, path, content):
                atomic_write(root, path, content)
                if path == Path("suite-results.jsonl"):
                    raise RuntimeError("synthetic interruption after quota rewind")

            with patch.object(
                evals, "_secure_atomic_eval_write", interrupt_after_ledger
            ):
                with pytest.raises(RuntimeError, match="synthetic interruption"):
                    evals.run_eval_suite(**options, resume_existing=True)
        [resumed] = evals.run_eval_suite(**options, resume_existing=True)
    assert run.call_count == 2
    assert resumed is replacement
    assert len((tmp_path / "out/suite-results.jsonl").read_text().splitlines()) == 1
