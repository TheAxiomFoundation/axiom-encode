"""Tests for the LLM judge stages (maximum-traceability part 2).

Emphasis on the load-bearing invariants:

* a judge failure is a visible ``judge_error`` event and is NEVER treated as a
  pass (fail-open is banned);
* the pre-classifier never drops an entry silently — only skip-with-reason;
* the cross-family guard rejects a judge sharing the generator's family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import types
from pathlib import Path

import pytest
import yaml

from axiom_encode.judges import (
    JUDGE_STAGE,
    JudgeCall,
    JudgeClient,
    JudgeError,
    JudgeEvent,
    JudgeStage,
    TokenCounts,
    Verdict,
    calibration,
    cli_commands,
    disposition,
    drift,
    error_event,
    grid_adequacy,
    model_family,
    preclassifier,
    regeneration,
    statutory_fidelity,
    truncate_provision,
    validate_event_dict,
)
from axiom_encode.judges.client import _extract_json
from axiom_encode.judges.run_log import coerce_confidence
from axiom_encode.run_log import (
    RunLogWriter,
    StageStatus,
    fold_run,
    iter_events,
)

# -- fakes ----------------------------------------------------------------


class FakeClient:
    """A JudgeClient stand-in that returns a preset JudgeCall."""

    def __init__(self, result: JudgeCall, *, generator_model: str = "gpt-5.5"):
        self._result = result
        self.provision_chars = 24_000
        self.generator_model = generator_model
        self.model = result.model
        self.escalation_model = "claude-sonnet-4-5"

    def call(self, **kwargs) -> JudgeCall:
        return self._result


def _ok_call(payload: dict, *, model="claude-haiku-4-5-20251001", escalated=False):
    return JudgeCall(
        payload=payload,
        model=model,
        escalated=escalated,
        tokens=TokenCounts(10, 5),
    )


def _err_call(err_type="server_error", message="boom"):
    return JudgeCall(
        payload=None,
        model="claude-haiku-4-5-20251001",
        escalated=False,
        tokens=TokenCounts(3, 0),
        error=JudgeError(type=err_type, message=message),
    )


# Fake SDK exception classes, defined once so isinstance checks are stable
# across repeated installs within a single test.
class FakeAPIStatusError(Exception):
    def __init__(self, message="", status_code=None):
        super().__init__(message)
        self.status_code = status_code


class FakeRateLimitError(Exception):
    pass


class FakeAPIConnectionError(Exception):
    pass


class FakeInternalServerError(Exception):
    pass


class FakeBadRequestError(Exception):
    pass


def install_fake_anthropic(monkeypatch, responses):
    """Install a fake ``anthropic`` module whose create() replays ``responses``.

    Each item is a str (returned as the response text) or an Exception (raised).
    """

    mod = types.ModuleType("anthropic")
    mod.APIStatusError = FakeAPIStatusError
    mod.RateLimitError = FakeRateLimitError
    mod.APIConnectionError = FakeAPIConnectionError
    mod.InternalServerError = FakeInternalServerError
    mod.BadRequestError = FakeBadRequestError

    queue = list(responses)

    class _Usage:
        input_tokens = 11
        output_tokens = 7

    class _Block:
        type = "text"

        def __init__(self, text):
            self.text = text

    class _Response:
        def __init__(self, text):
            self.content = [_Block(text)]
            self.usage = _Usage()

    class _Messages:
        def create(self, **kwargs):
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return _Response(item)

    class Anthropic:
        def __init__(self, api_key=None):
            self.messages = _Messages()

    mod.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return mod


# -- run_log schema -------------------------------------------------------


def test_event_to_dict_shape_and_validation():
    event = JudgeEvent(
        stage=JudgeStage.STATUTORY_FIDELITY, verdict=Verdict.PASS, confidence=0.9
    )
    d = event.to_dict()
    # Canonical axiom_encode.run_log.v1 wire shape: stage="judge", verdict in attrs.
    assert d["schema"] == "axiom_encode.run_log.v1"
    assert d["stage"] == "judge"
    assert d["status"] == "passed"
    assert d["attrs"]["verdict"] == "pass"
    assert d["attrs"]["judge_stage"] == "statutory_fidelity"
    assert d["attrs"]["tokens"] == {"input": 0, "output": 0}
    assert validate_event_dict(d) == []


def test_error_event_is_never_a_pass():
    ev = error_event(JudgeStage.DISPOSITION, "kaboom", error_type="missing_api_key")
    assert ev.verdict == Verdict.ERROR
    assert ev.passed is False
    assert ev.judge_error is not None
    d = ev.to_dict()
    # Fail-closed: an error maps to status="error" carrying its judge_error.
    assert d["status"] == "error"
    assert d["attrs"]["judge_error"]["type"] == "missing_api_key"
    assert d["reason_code"] == "missing_api_key"
    assert validate_event_dict(d) == []


def test_validate_rejects_fail_open():
    # error status with no judge_error is fail-open masquerading as handled.
    bad = JudgeEvent(stage=JudgeStage.DISPOSITION, verdict=Verdict.ERROR).to_dict()
    bad["attrs"]["judge_error"] = None
    problems = validate_event_dict(bad)
    assert any("judge_error" in p for p in problems)

    # judge_error on a pass verdict is also invalid.
    sneaky = JudgeEvent(stage=JudgeStage.DISPOSITION, verdict=Verdict.PASS).to_dict()
    sneaky["attrs"]["judge_error"] = {"type": "x", "message": "y"}
    assert any("judge_error" in p for p in validate_event_dict(sneaky))


# -- client: cross-family guard + fail-closed -----------------------------


def test_model_family():
    assert model_family("claude-haiku-4-5-20251001") == "anthropic"
    assert model_family("gpt-5.5") == "openai"
    assert model_family("mystery-model") == "unknown"


def test_cross_family_guard_rejects_same_family():
    client = JudgeClient(model="gpt-5.5", api_key="x")
    call = client.call(system="s", user_prompt="p", schema={})
    assert call.error is not None
    assert call.error.type == "cross_family_guard"
    assert not call.ok


def test_missing_api_key_is_fail_closed():
    client = JudgeClient(model="claude-haiku-4-5-20251001", api_key=None)
    # ensure no ambient key leaks in
    client.api_key = None
    call = client.call(system="s", user_prompt="p", schema={})
    assert call.error is not None
    assert call.error.type == "missing_api_key"


def test_client_success_and_escalation(monkeypatch):
    # first (haiku) low confidence -> escalate to sonnet (high confidence).
    install_fake_anthropic(
        monkeypatch,
        [
            '{"verdict":"flag","confidence":0.2,"findings":[]}',
            '{"verdict":"flag","confidence":0.95,"findings":[]}',
        ],
    )
    client = JudgeClient(
        model="claude-haiku-4-5-20251001",
        escalation_model="claude-sonnet-4-5",
        api_key="test",
        escalate_below=0.6,
    )
    call = client.call(system="s", user_prompt="p", schema={})
    assert call.ok
    assert call.escalated is True
    assert call.model == "claude-sonnet-4-5"
    assert call.payload["confidence"] == 0.95
    # tokens accumulate across both calls
    assert call.tokens.input == 22


def test_client_parse_failure_is_error_not_pass(monkeypatch):
    install_fake_anthropic(monkeypatch, ["this is not json"])
    client = JudgeClient(model="claude-haiku-4-5-20251001", api_key="test")
    call = client.call(system="s", user_prompt="p", schema={})
    assert not call.ok
    assert call.error.type == "parse_error"


def test_client_api_failure_retries_then_errors(monkeypatch):
    # two rate-limit errors, then exhausted.
    import axiom_encode.judges.client as clientmod

    monkeypatch.setattr(clientmod.time, "sleep", lambda *_: None)
    install_fake_anthropic(
        monkeypatch,
        [FakeRateLimitError("429"), FakeRateLimitError("429")],
    )
    client = JudgeClient(
        model="claude-haiku-4-5-20251001",
        api_key="test",
        max_attempts=2,
        retry_seconds=0,
    )
    call = client.call(system="s", user_prompt="p", schema={})
    assert not call.ok
    assert call.error.type == "rate_limit"


def test_extract_json_paths():
    assert _extract_json('{"a":1}') == {"a": 1}
    assert _extract_json('```json\n{"a":1}\n```') == {"a": 1}
    assert _extract_json('prefix {"a":1} suffix') == {"a": 1}
    assert _extract_json("nope") is None


def test_truncate_provision_bounded_and_keeps_tail():
    text = "HEAD" + ("x" * 50_000) + "TAILBOUNDARY"
    out = truncate_provision(text, 1000)
    assert len(out) <= 1000
    assert out.startswith("HEAD")
    assert "TAILBOUNDARY" in out


# -- Stage 1: statutory fidelity ------------------------------------------


def test_fidelity_flag_and_needs_review():
    payload = {
        "verdict": "flag",
        "confidence": 0.8,
        "findings": [
            {
                "clause_ref": "26 USC 32(b)",
                "rule_path": "x.yaml#rate",
                "kind": "amount_mismatch",
                "explanation": "600000 not in source",
            }
        ],
    }
    ev = statutory_fidelity.run(
        "prov", "rule", citation="26 USC 32", client=FakeClient(_ok_call(payload))
    )
    assert ev.verdict == Verdict.FLAG
    assert ev.findings[0].kind == "amount_mismatch"
    assert statutory_fidelity.needs_review_label(ev) == "needs-review"
    assert validate_event_dict(ev.to_dict()) == []


def test_fidelity_pass_no_findings():
    payload = {"verdict": "pass", "confidence": 0.9, "findings": []}
    ev = statutory_fidelity.run(
        "prov", "rule", citation="c", client=FakeClient(_ok_call(payload))
    )
    assert ev.verdict == Verdict.PASS
    assert statutory_fidelity.needs_review_label(ev) is None


def test_fidelity_api_failure_is_error_event_not_pass():
    ev = statutory_fidelity.run(
        "prov", "rule", citation="c", client=FakeClient(_err_call())
    )
    assert ev.verdict == Verdict.ERROR
    assert ev.passed is False
    assert ev.judge_error is not None
    # an error is itself a reason for review, never silently dropped
    assert statutory_fidelity.needs_review_label(ev) == "needs-review"


# -- Stage 2: grid adequacy -----------------------------------------------


def test_grid_gaps_become_findings_and_cells():
    payload = {
        "confidence": 0.7,
        "gaps": [
            {
                "region": "35% band",
                "boundary": "phase-out start",
                "clause_ref": "26 USC 32(b)(2)",
                "suggested_case": {"agi": 20000},
                "explanation": "no case in band",
            }
        ],
    }
    ev = grid_adequacy.run(
        "prov", [{"agi": 1000}], suite_name="eitc", client=FakeClient(_ok_call(payload))
    )
    assert ev.verdict == Verdict.FLAG
    assert ev.findings[0].kind == "untested_region"
    cells = grid_adequacy.gaps_to_cells(ev)
    assert cells and cells[0]["suggested_case"] == {"agi": 20000}


def test_grid_no_gaps_pass_and_error_yields_no_cells():
    ev = grid_adequacy.run(
        "p", [], client=FakeClient(_ok_call({"confidence": 0.9, "gaps": []}))
    )
    assert ev.verdict == Verdict.PASS
    assert grid_adequacy.gaps_to_cells(ev) == []
    err = grid_adequacy.run("p", [], client=FakeClient(_err_call()))
    assert err.verdict == Verdict.ERROR
    assert grid_adequacy.gaps_to_cells(err) == []


# -- Stage 3: disposition -------------------------------------------------


def test_disposition_arithmetic():
    d = disposition.Disposition(
        "d",
        "claim",
        residual=42.0,
        records=[{"engine_value": 100, "oracle_value": 142} for _ in range(3)],
    )
    art = disposition.check_arithmetic(d)
    assert art["reproduced"] == 3
    assert art["arithmetic_ok"] is True


def test_disposition_insufficient_records_flags_without_api():
    d = disposition.Disposition(
        "d", "c", residual=1.0, records=[{"engine_value": 1, "oracle_value": 2}]
    )
    # FakeClient.call would raise if used incorrectly; short-circuit must avoid it.
    ev = disposition.run(d, client=FakeClient(_err_call()))
    assert ev.verdict == Verdict.FLAG
    assert ev.findings[0].kind == "insufficient_records"


def test_disposition_pass_requires_arithmetic_and_claim():
    d = disposition.Disposition(
        "d",
        "rounding",
        residual=42.0,
        records=[{"engine_value": 100, "oracle_value": 142} for _ in range(3)],
    )
    ok = FakeClient(
        _ok_call({"consistent": True, "confidence": 0.9, "explanation": "yes"})
    )
    ev = disposition.run(d, client=ok)
    assert ev.verdict == Verdict.PASS

    # arithmetic reproduces but claim rejected -> flag claim_mismatch
    bad = FakeClient(
        _ok_call({"consistent": False, "confidence": 0.9, "explanation": "no"})
    )
    ev2 = disposition.run(d, client=bad)
    assert ev2.verdict == Verdict.FLAG
    assert ev2.findings[0].kind == "claim_mismatch"


def test_disposition_api_error_is_error_event():
    d = disposition.Disposition(
        "d",
        "c",
        residual=42.0,
        records=[{"engine_value": 100, "oracle_value": 142} for _ in range(3)],
    )
    ev = disposition.run(d, client=FakeClient(_err_call()))
    assert ev.verdict == Verdict.ERROR
    assert "arithmetic" in ev.extra


# -- Stage 4: pre-classifier (never drops) --------------------------------


def test_preclassify_heuristics():
    amend = {
        "citation": "Act 1134",
        "source_text": "Section 1 is amended by inserting after paragraph (2) the following new subsection.",
    }
    r = preclassifier.classify(amend, use_llm=False)
    assert r.classification == preclassifier.WorklistClass.AMENDMENT_ACT
    assert r.route == preclassifier.SKIP

    xref = {
        "citation": "NY",
        "source_text": "section 1 section 2 section 3 section 4 section 5 section 6 section 7",
    }
    r2 = preclassifier.classify(xref, use_llm=False)
    assert r2.classification == preclassifier.WorklistClass.XREF_HEAVY


def test_preclassify_llm_self_contained_generates(monkeypatch):
    install_fake_anthropic(
        monkeypatch,
        ['{"classification":"self_contained","confidence":0.9,"reason":"operative"}'],
    )
    client = JudgeClient(model="claude-haiku-4-5-20251001", api_key="test")
    r = preclassifier.classify(
        {
            "citation": "26 USC 21",
            "source_text": "A credit equal to 20 percent of expenses is allowed.",
        },
        client=client,
    )
    assert r.route == preclassifier.GENERATE
    assert r.classification == preclassifier.WorklistClass.SELF_CONTAINED


def test_preclassify_llm_error_skips_never_generates_or_drops():
    # An ambiguous entry whose arbiter fails must be skip-with-reason, not a
    # silent pass-through to generation and not a drop.
    err_client = FakeClient(_err_call(err_type="server_error"))
    r = preclassifier.classify(
        {"citation": "X", "source_text": "A short operative clause with no markers."},
        client=err_client,
    )
    assert r.route == preclassifier.SKIP
    assert r.classification == preclassifier.WorklistClass.NEEDS_CONTAINER
    assert r.llm_error is not None
    assert r.event.verdict == Verdict.SKIP
    # the judge failure remains visible in the run-log even though we skipped
    assert r.event.extra.get("judge_error") is not None


def test_preclassify_batch_never_drops():
    entries = [
        {"citation": "a", "source_text": "Section 1 is amended by striking X."},
        {"citation": "b", "source_text": ""},  # empty
        {
            "citation": "c",
            "source_text": "section 1 section 2 section 3 section 4 section 5 section 6 section 7",
        },
    ]
    out = preclassifier.classify_batch(entries, use_llm=False)
    assert len(out) == len(entries)  # never drops
    # no entry silently routed to generate on uncertainty
    assert all(r.route in {preclassifier.GENERATE, preclassifier.SKIP} for r in out)


# -- drift ----------------------------------------------------------------


def _regeneration_fixture(
    tmp_path: Path,
    *,
    module: str = "us/statutes/26/1.yaml",
    citation: str = "us/statute/26/1",
    backend: str = "openai",
    tool: str = "axiom-encode encode --apply",
    root_mirror: bool = False,
) -> tuple[Path, str, Path]:
    root = tmp_path / "rulespec-us"
    module_path = root / module
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("outputs: {}\n", encoding="utf-8")
    relative = Path(module)
    content_relative = Path(*relative.parts[1:])
    if root_mirror:
        manifest = (
            root / ".axiom" / "encoding-manifests" / relative.with_suffix(".json")
        )
        applied_path = relative.as_posix()
    else:
        manifest = (
            root
            / relative.parts[0]
            / ".axiom"
            / "encoding-manifests"
            / content_relative.with_suffix(".json")
        )
        applied_path = content_relative.as_posix()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "citation": citation,
                "backend": backend,
                "tool": tool,
                "applied_files": [
                    {
                        "path": applied_path,
                        "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root, module, manifest


def _corpus_fixture(
    tmp_path: Path,
    *,
    citation: str = "us/statute/26/1",
) -> Path:
    corpus = tmp_path / "axiom-corpus"
    parts = citation.split("/")
    provisions = corpus / "data" / "corpus" / "provisions" / parts[0] / parts[1]
    provisions.mkdir(parents=True, exist_ok=True)
    with (provisions / "test.jsonl").open("a", encoding="utf-8") as provision_file:
        provision_file.write(
            json.dumps({"citation_path": citation, "body": "authoritative source"})
            + "\n"
        )
    return corpus


def test_drift_regenerator_uses_fixed_argv_and_minimal_environment(
    tmp_path, monkeypatch
):
    root, module, _manifest = _regeneration_fixture(tmp_path)
    corpus = _corpus_fixture(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output_root = Path(command[command.index("--output") + 1])
        generated = output_root / "openai-gpt-5.5" / "statutes/26/1.yaml"
        generated.parent.mkdir(parents=True)
        generated.write_text("outputs:\n  result: 1\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(regeneration.subprocess, "run", fake_run)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    monkeypatch.setenv("GH_TOKEN", "must-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
    monkeypatch.setenv("AXIOM_REPO_TOKEN", "must-not-leak")

    regenerated = regeneration.regenerate_module(
        module,
        "outputs: {}\n",
        root=root,
        corpus_path=corpus,
        backend="openai",
    )

    command = captured["command"]
    kwargs = captured["kwargs"]
    assert command[:4] == [
        sys.executable,
        "-m",
        "axiom_encode.cli",
        "encode",
    ]
    assert command[command.index("--source-id") + 1] == "us:statutes/26/1"
    assert Path(command[command.index("--corpus-path") + 1]) == corpus
    assert Path(command[command.index("--policy-repo-path") + 1]) == root / "us"
    assert command[-2:] == ["--", "us/statute/26/1"]
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == regeneration.REGENERATION_TIMEOUT_SECONDS
    assert kwargs["env"]["OPENAI_API_KEY"] == "openai-test-key"
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]
    assert "GH_TOKEN" not in kwargs["env"]
    assert "GITHUB_TOKEN" not in kwargs["env"]
    assert "AXIOM_REPO_TOKEN" not in kwargs["env"]
    assert regenerated == "outputs:\n  result: 1\n"


def test_drift_regenerator_preserves_requested_child_with_local_only_resolution(
    tmp_path, monkeypatch
):
    root, module, _manifest = _regeneration_fixture(
        tmp_path,
        citation="us/statute/26/1",
    )
    corpus = _corpus_fixture(tmp_path, citation="us/statute/26")
    captured_command: list[str] = []

    def fake_run(command, **_kwargs):
        captured_command.extend(command)
        output_root = Path(command[command.index("--output") + 1])
        generated = output_root / "openai-gpt-5.5" / "statutes/26/1.yaml"
        generated.parent.mkdir(parents=True)
        generated.write_text("outputs: {}\n")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(regeneration.subprocess, "run", fake_run)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    regeneration.regenerate_module(
        module,
        "outputs: {}\n",
        root=root,
        corpus_path=corpus,
    )

    assert "--local-corpus-only" in captured_command
    assert captured_command[-2:] == ["--", "us/statute/26/1"]


def test_drift_error_report_and_published_issue_redact_model_key(tmp_path, monkeypatch):
    root, module, _manifest = _regeneration_fixture(tmp_path)
    corpus = _corpus_fixture(tmp_path)
    secret = "openai-super-secret-test-key"

    def fake_encoder(*_args, **_kwargs):
        return types.SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=f"request failed with credential {secret}",
        )

    monkeypatch.setattr(regeneration.subprocess, "run", fake_encoder)
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    result = drift.check_module(
        module,
        "outputs: {}\n",
        lambda candidate, merged: regeneration.regenerate_module(
            candidate,
            merged,
            root=root,
            corpus_path=corpus,
        ),
    )
    report = drift.DriftReport(checked=[result])
    report_path = tmp_path / "drift-report.json"
    report_path.write_text(json.dumps(report.to_dict()), encoding="utf-8")

    assert result.error is not None
    assert secret not in result.error
    assert "[REDACTED]" in result.error
    assert secret not in report_path.read_text(encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("GH_TOKEN", "github-publisher-test-token")
    published_bodies: list[str] = []

    def fake_github(command, **_kwargs):
        if command[1:3] == ["issue", "create"]:
            body_path = Path(command[command.index("--body-file") + 1])
            published_bodies.append(body_path.read_text(encoding="utf-8"))
            return types.SimpleNamespace(
                returncode=0,
                stdout="https://github.test/issues/1\n",
                stderr="",
            )
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli_commands.subprocess, "run", fake_github)
    status = cli_commands.cmd_publish_drift_report(
        argparse.Namespace(
            report_file=report_path,
            repo="TheAxiomFoundation/axiom-encode",
            json=False,
        )
    )

    assert status == 1
    assert len(published_bodies) == 1
    assert secret not in published_bodies[0]


def test_drift_report_file_redacts_secrets_from_all_result_fields(
    tmp_path, monkeypatch, capsys
):
    secret = "openai-secret-in-generated-diff"
    report = drift.DriftReport(
        checked=[
            drift.DriftResult(
                module="us/statutes/26/1.yaml",
                drifted=True,
                diffs=[
                    {
                        "path": "outputs.result",
                        "change": "value_changed",
                        "merged": 1,
                        "regenerated": secret,
                    }
                ],
            )
        ]
    )
    report_path = tmp_path / "drift-report.json"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setattr(
        cli_commands,
        "_load_drift_modules",
        lambda _args: {"us/statutes/26/1.yaml": "outputs: {}\n"},
    )
    monkeypatch.setattr(drift, "run_drift_check", lambda *_args, **_kwargs: report)

    status = cli_commands.cmd_drift_check(
        argparse.Namespace(
            regenerate=False,
            dry_run=True,
            root=None,
            corpus_path=None,
            modules_file=tmp_path / "unused.json",
            k=3,
            seed=0,
            regenerate_backend="openai",
            report_file=report_path,
            json=True,
        )
    )

    assert status == 1
    assert secret not in report_path.read_text(encoding="utf-8")
    assert secret not in capsys.readouterr().out


def test_drift_publisher_rejects_invalid_diff_before_github_mutation(
    tmp_path, monkeypatch, capsys
):
    report_path = tmp_path / "drift-report.json"
    report_path.write_text(
        json.dumps(
            {
                "n_checked": 1,
                "n_drifted": 1,
                "n_errors": 0,
                "results": [
                    {
                        "module": "us/statutes/26/1.yaml",
                        "drifted": True,
                        "error": None,
                        "diffs": [{}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fail_github(*_args, **_kwargs):
        raise AssertionError("invalid report reached GitHub CLI")

    monkeypatch.setattr(cli_commands.subprocess, "run", fail_github)
    status = cli_commands.cmd_publish_drift_report(
        argparse.Namespace(
            report_file=report_path,
            repo="TheAxiomFoundation/axiom-encode",
            json=False,
        )
    )

    assert status == 2
    assert "invalid required fields" in capsys.readouterr().err


def test_drift_regenerator_supports_root_mirror_state_manifest(tmp_path):
    root, module, manifest = _regeneration_fixture(
        tmp_path,
        module="us-oh/statutes/5747/71.yaml",
        citation="us-oh/statute/5747.71",
        root_mirror=True,
    )
    relative = regeneration.validate_module_path(root, module)

    assert regeneration.read_citation(root, relative) == "us-oh/statute/5747.71"
    assert regeneration.source_id_for_module(root, relative) == (
        "us-oh:statutes/5747/71"
    )
    assert manifest == (
        root / ".axiom" / "encoding-manifests" / "us-oh/statutes/5747/71.json"
    )


def test_drift_regenerator_allows_colon_in_contained_module_path(tmp_path):
    root, module, _manifest = _regeneration_fixture(
        tmp_path,
        module="us-la/statutes/47:294.yaml",
        citation="us-la/statute/47:294",
    )
    relative = regeneration.validate_module_path(root, module)

    assert regeneration.generated_subpath(relative) == Path("statutes/47:294.yaml")
    assert regeneration.source_id_for_module(root, relative) == (
        "us-la:statutes/47:294"
    )


@pytest.mark.parametrize(
    "module",
    [
        "us/statute/26/1.yaml;touch owned",
        "us/statute/26/$(touch owned).yaml",
        "us/statute/26/`touch owned`.yaml",
        "us/statute/26/has space.yaml",
        "../outside.yaml",
    ],
)
def test_drift_regenerator_rejects_unsafe_module_paths(tmp_path, monkeypatch, module):
    root = tmp_path / "rulespec-us"
    corpus = _corpus_fixture(tmp_path)
    root.mkdir()
    candidate = root / module
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("outputs: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsafe path reached subprocess")

    monkeypatch.setattr(regeneration.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="module path|unsafe module"):
        regeneration.regenerate_module(
            module,
            "outputs: {}\n",
            root=root,
            corpus_path=corpus,
        )

    assert called is False
    assert not (tmp_path / "owned").exists()


def test_drift_regenerator_rejects_symlink_escape(tmp_path):
    root = tmp_path / "rulespec-us"
    module_path = root / "us/statute/26/1.yaml"
    module_path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.yaml"
    outside.write_text("outputs: {}\n", encoding="utf-8")
    module_path.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink|escapes"):
        regeneration.validate_module_path(root, "us/statute/26/1.yaml")


def test_drift_regenerator_rejects_manifest_citation_traversal(tmp_path, monkeypatch):
    root, module, manifest = _regeneration_fixture(tmp_path)
    corpus = _corpus_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["citation"] = "us/statute/26/1/../../../../tmp/owned"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsafe citation reached subprocess")

    monkeypatch.setattr(regeneration.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="unsafe manifest citation"):
        regeneration.regenerate_module(
            module,
            "outputs: {}\n",
            root=root,
            corpus_path=corpus,
        )

    assert called is False


def test_drift_loader_samples_only_current_replayable_encodes(tmp_path, capsys):
    root, module, _manifest = _regeneration_fixture(tmp_path)
    unmanifested = root / "us/statutes/26/2.yaml"
    unmanifested.parent.mkdir(parents=True, exist_ok=True)
    unmanifested.write_text("outputs: {}\n", encoding="utf-8")
    _root, deterministic_module, _deterministic_manifest = _regeneration_fixture(
        tmp_path,
        module="us/policies/cms/generated.yaml",
        citation="us/policy/cms/generated",
        backend="deterministic",
        tool="axiom-encode generate-cms-chip-eligibility-composition",
    )

    loaded = cli_commands._load_drift_modules(
        argparse.Namespace(root=root, modules_file=None)
    )

    assert loaded == {module: "outputs: {}\n"}
    assert deterministic_module not in loaded
    assert "skipped 2 non-replayable drift candidates" in capsys.readouterr().err


def test_drift_loader_reports_missing_root_without_traceback(tmp_path, capsys):
    missing = tmp_path / "does-not-exist"

    loaded = cli_commands._load_drift_modules(
        argparse.Namespace(root=missing, modules_file=None)
    )

    assert loaded is None
    assert f"invalid drift root {missing}" in capsys.readouterr().err


def test_drift_selection_skips_citations_missing_from_local_corpus(tmp_path, capsys):
    root, resolvable, _manifest = _regeneration_fixture(tmp_path)
    _root, missing, _missing_manifest = _regeneration_fixture(
        tmp_path,
        module="us/statutes/26/2.yaml",
        citation="policies/irs/non-canonical",
    )
    corpus = _corpus_fixture(tmp_path)
    valid_modules = {resolvable}
    for section in (3, 4):
        citation = f"us/statute/26/{section}"
        _root, valid, _valid_manifest = _regeneration_fixture(
            tmp_path,
            module=f"us/statutes/26/{section}.yaml",
            citation=citation,
        )
        _corpus_fixture(tmp_path, citation=citation)
        valid_modules.add(valid)
    modules = cli_commands._load_drift_modules(
        argparse.Namespace(root=root, modules_file=None)
    )

    selected = cli_commands._select_corpus_backed_modules(
        argparse.Namespace(
            root=root,
            corpus_path=corpus,
            k=3,
            seed=0,
        ),
        modules,
    )

    assert set(selected) == valid_modules
    assert missing not in selected
    assert "does not resolve in the local corpus" in capsys.readouterr().err


def test_drift_selection_fails_if_requested_sample_cannot_be_filled(tmp_path, capsys):
    root, _resolvable, _manifest = _regeneration_fixture(tmp_path)
    _root, _missing, _missing_manifest = _regeneration_fixture(
        tmp_path,
        module="us/statutes/26/2.yaml",
        citation="policies/irs/non-canonical",
    )
    corpus = _corpus_fixture(tmp_path)
    modules = cli_commands._load_drift_modules(
        argparse.Namespace(root=root, modules_file=None)
    )

    selected = cli_commands._select_corpus_backed_modules(
        argparse.Namespace(
            root=root,
            corpus_path=corpus,
            k=3,
            seed=0,
        ),
        modules,
    )

    assert selected is None
    assert "found only 1 of 2 required" in capsys.readouterr().err


def test_drift_selection_rejects_checkout_without_provisions(tmp_path, capsys):
    root, module, _manifest = _regeneration_fixture(tmp_path)
    empty_corpus = tmp_path / "axiom-corpus"
    empty_corpus.mkdir()

    selected = cli_commands._select_corpus_backed_modules(
        argparse.Namespace(
            root=root,
            corpus_path=empty_corpus,
            k=3,
            seed=0,
        ),
        {module: "outputs: {}\n"},
    )

    assert selected is None
    assert "no provisions directory" in capsys.readouterr().err


def test_drift_regenerator_rejects_symlinked_corpus_provisions_tree(tmp_path):
    corpus = tmp_path / "axiom-corpus"
    (corpus / "data" / "corpus").mkdir(parents=True)
    outside = tmp_path / "outside-provisions"
    outside.mkdir()
    (outside / "source.jsonl").write_text(
        json.dumps({"citation_path": "us/statute/26/1", "body": "outside"}) + "\n",
        encoding="utf-8",
    )
    (corpus / "data" / "corpus" / "provisions").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="provisions path contains a symlink"):
        regeneration.validate_corpus_path(corpus)


def test_drift_regenerator_rejects_symlinked_corpus_provision_file(tmp_path):
    corpus = tmp_path / "axiom-corpus"
    provisions = corpus / "data" / "corpus" / "provisions" / "us" / "statute"
    provisions.mkdir(parents=True)
    outside = tmp_path / "outside.jsonl"
    outside.write_text(
        json.dumps({"citation_path": "us/statute/26/1", "body": "outside"}) + "\n",
        encoding="utf-8",
    )
    (provisions / "source.jsonl").symlink_to(outside)

    with pytest.raises(ValueError, match="provisions tree contains a symlink"):
        regeneration.validate_corpus_path(corpus)


def test_drift_regenerator_rejects_symlinked_corpus_claims_tree(tmp_path):
    corpus = _corpus_fixture(tmp_path)
    outside = tmp_path / "outside-claims"
    outside.mkdir()
    (outside / "claims.jsonl").write_text(
        json.dumps({"id": "claim.secret", "status": "accepted"}) + "\n",
        encoding="utf-8",
    )
    claims = corpus / "data" / "corpus" / "claims"
    claims.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="claims path contains a symlink"):
        regeneration.validate_corpus_path(corpus)


def test_drift_regenerator_rejects_symlinked_corpus_claim_file(tmp_path):
    corpus = _corpus_fixture(tmp_path)
    claims = corpus / "data" / "corpus" / "claims" / "us"
    claims.mkdir(parents=True)
    outside = tmp_path / "outside-claim.jsonl"
    outside.write_text(
        json.dumps({"id": "claim.secret", "status": "accepted"}) + "\n",
        encoding="utf-8",
    )
    (claims / "claims.jsonl").symlink_to(outside)

    with pytest.raises(ValueError, match="claims tree contains a symlink"):
        regeneration.validate_corpus_path(corpus)


def test_drift_loader_rejects_module_symlink_before_read(tmp_path, capsys):
    root = tmp_path / "rulespec-us"
    module_path = root / "us/statutes/26/escape.yaml"
    module_path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.yaml"
    outside.write_text("outside\n", encoding="utf-8")
    module_path.symlink_to(outside)

    loaded = cli_commands._load_drift_modules(
        argparse.Namespace(root=root, modules_file=None)
    )

    assert loaded is None
    assert "contains a symlink" in capsys.readouterr().err


def test_drift_loader_rejects_manifest_symlink_before_read(tmp_path, capsys):
    root, _module, manifest = _regeneration_fixture(tmp_path)
    outside = tmp_path / "manifest.json"
    outside.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    manifest.unlink()
    manifest.symlink_to(outside)

    loaded = cli_commands._load_drift_modules(
        argparse.Namespace(root=root, modules_file=None)
    )

    assert loaded is None
    assert "encoding manifest contains a symlink" in capsys.readouterr().err


def test_drift_regeneration_is_openai_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    with pytest.raises(ValueError, match="unsupported regeneration backend"):
        regeneration.child_environment("codex")


def test_drift_github_cli_environment_excludes_model_credentials(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "github-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")

    env = cli_commands._github_cli_environment()

    assert env["GH_TOKEN"] == "github-test-key"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_drift_cli_rejects_removed_shell_template():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    cli_commands.register_judge_subparsers(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "drift-check",
                "--root",
                "rulespec-us",
                "--regenerate-cmd",
                "python script.py {module} {merged} {output}",
            ]
        )


def test_drift_cli_rejects_in_process_issue_publication():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    cli_commands.register_judge_subparsers(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "drift-check",
                "--root",
                "rulespec-us",
                "--regenerate",
                "--create-issues",
            ]
        )


def test_drift_workflow_isolates_model_and_github_credentials_by_job():
    workflow_path = (
        Path(__file__).parents[1] / ".github" / "workflows" / "golden-regeneration.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    live_step = next(
        step
        for step in jobs["drift"]["steps"]
        if step.get("name") == "Run live drift check"
    )
    publish_step = next(
        step
        for step in jobs["publish"]["steps"]
        if step.get("name") == "Publish drift findings"
    )
    upload_step = next(
        step
        for step in jobs["drift"]["steps"]
        if step.get("name") == "Upload sanitized drift report"
    )
    engine_checkout = next(
        step
        for step in jobs["drift"]["steps"]
        if step.get("name") == "Checkout axiom-rules-engine"
    )
    install_steps = [
        step
        for job in jobs.values()
        for step in job["steps"]
        if step.get("name") == "Install encoder"
    ]
    action_steps = [
        step for job in jobs.values() for step in job["steps"] if "uses" in step
    ]
    setup_uv_steps = [
        step for step in action_steps if step["uses"].startswith("astral-sh/setup-uv@")
    ]
    checkout_steps = [
        step for step in action_steps if step["uses"].startswith("actions/checkout@")
    ]

    assert workflow["permissions"] == {"contents": "read"}
    assert jobs["publish"]["needs"] == "drift"
    assert jobs["drift"]["permissions"] == {"contents": "read"}
    assert jobs["publish"]["permissions"] == {
        "contents": "read",
        "issues": "write",
    }
    assert "env" not in workflow
    assert all("env" not in job for job in jobs.values())
    assert all(step["with"]["persist-credentials"] is False for step in checkout_steps)
    assert upload_step["with"]["overwrite"] is True
    assert "OPENAI_API_KEY" in live_step["env"]
    assert "GH_TOKEN" not in live_step["env"]
    assert "GH_TOKEN" in publish_step["env"]
    assert "OPENAI_API_KEY" not in publish_step["env"]
    assert engine_checkout["with"]["ref"] == (
        "38b5f646165f18f64307f1eef226c7a6f2d4936e"
    )
    assert all(
        step["run"] == "uv sync --locked --python 3.13 --extra api"
        for step in install_steps
    )
    assert all(
        re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"]) for step in action_steps
    )
    assert all(step["with"]["version"] == "0.11.7" for step in setup_uv_steps)


def test_drift_ignores_order_and_comments_catches_change():
    a = "outputs:\n  r:\n    value: 0.34\ninputs: [x, y]\n"
    reordered = "inputs: [y, x]  # comment\noutputs:\n  r:\n    value: 0.34\n"
    assert drift.semantic_diff(a, reordered) == []
    changed = "outputs:\n  r:\n    value: 0.40\ninputs: [x, y]\n"
    diffs = drift.semantic_diff(a, changed)
    assert diffs and diffs[0]["change"] == "value_changed"


def test_drift_rejects_yaml_alias_expansion_bomb():
    levels = ["base: &base [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"]
    previous = "base"
    for index in range(8):
        current = f"level_{index}"
        levels.append(
            f"{current}: &{current} [" + ", ".join([f"*{previous}"] * 10) + "]"
        )
        previous = current
    bomb = "\n".join(levels) + "\n"

    with pytest.raises(ValueError, match="aliases are not allowed"):
        drift.semantic_diff(bomb, bomb)


def test_drift_report_round_trip_bounds_paths_values_and_diff_count():
    merged_tree = {f"leaf-{index}": 0 for index in range(1000)}
    regenerated_tree = {f"leaf-{index}": 1 for index in range(1000)}
    for depth in range(6):
        key = f"level-{depth}-" + "x" * 900
        merged_tree = {key: merged_tree}
        regenerated_tree = {key: regenerated_tree}
    merged = json.dumps(
        {
            "evil\npath": 1,
            "deep": merged_tree,
        }
    )
    regenerated = json.dumps(
        {
            "evil\npath": 2,
            "deep": regenerated_tree,
        }
    )
    result = drift.check_module(
        "us/statutes/1.yaml",
        merged,
        lambda _module, _merged: regenerated,
    )
    report = drift.DriftReport(checked=[result])
    payload = json.loads(json.dumps(report.to_dict()))
    loaded = drift.DriftReport.from_dict(payload)

    assert result.diff_count == 1001
    assert len(result.diffs) == drift.MAX_RETAINED_DIFFS
    assert loaded.checked[0].diff_count == 1001
    assert all(
        len(item["path"]) <= drift.MAX_DIFF_PATH_CHARS
        and not any(ord(char) < 32 for char in item["path"])
        for item in payload["results"][0]["diffs"]
    )
    assert len(json.dumps(payload).encode("utf-8")) < 1024 * 1024


def test_drift_report_round_trip_bounds_regeneration_error():
    def fail(_module, _merged):
        raise RuntimeError("x" * 6000)

    result = drift.check_module(
        "us/statutes/1.yaml",
        "format: rulespec/v1\nrules: []\n",
        fail,
    )
    payload = json.loads(json.dumps(drift.DriftReport([result]).to_dict()))
    loaded = drift.DriftReport.from_dict(payload)

    assert len(payload["results"][0]["error"]) == drift.MAX_ERROR_CHARS
    assert "truncated" in payload["results"][0]["error"]
    assert loaded.errors


def test_drift_regeneration_error_is_visible():
    def boom(_m, _t):
        raise RuntimeError("encode failed")

    res = drift.check_module("m.yaml", "outputs: {}\n", boom)
    assert res.error is not None
    assert res.drifted is False  # but recorded as an error, not clean


def test_drift_sample_clamps():
    paths = [f"m{i}" for i in range(20)]
    assert len(drift.sample_modules(paths, k=1, seed=1)) == 3
    assert len(drift.sample_modules(paths, k=99, seed=1)) == 5
    assert len(drift.sample_modules(["a", "b"], k=5)) == 2


def test_run_drift_check_reports_drift():
    modules = {"a.yaml": "v: 1\n", "b.yaml": "v: 2\n", "c.yaml": "v: 3\n"}
    report = drift.run_drift_check(modules, lambda m, t: "v: 9\n", k=3, seed=0)
    assert len(report.drifted) == 3
    body = drift.drift_issue_body(report.drifted[0])
    assert "drift" in body.lower()


def test_drift_issue_body_neutralizes_markdown_and_mentions():
    result = drift.DriftResult(
        module="us/statutes/1.yaml",
        drifted=True,
        diffs=[
            {
                "path": "value` | @axiom [click](https://example.invalid)",
                "change": "value_changed",
                "merged": "before | @axiom",
                "regenerated": "after [link](https://example.invalid)",
            }
        ],
    )

    body = drift.drift_issue_body(result)
    row = next(line for line in body.splitlines() if "value_changed" in line)
    assert row.count("|") == 5
    assert "@axiom" not in body
    assert "[click](" not in body
    assert "&#64;axiom" in body


def test_drift_issue_publisher_bounds_body_and_title_at_report_maxima():
    module = f"us/statutes/{'nested-' * 50}target.yaml"
    result = drift.DriftResult(
        module=module,
        drifted=True,
        total_diffs=1_000,
        diffs=[
            {
                "path": "&" * drift.MAX_DIFF_PATH_CHARS,
                "change": "value_changed",
                "merged": "&" * drift.MAX_DIFF_VALUE_CHARS,
                "regenerated": "<" * drift.MAX_DIFF_VALUE_CHARS,
            }
            for _ in range(drift.MAX_RETAINED_DIFFS)
        ],
    )

    body = drift.drift_issue_body(result)
    title = drift.drift_issue_title(result)

    assert len(body.encode("utf-8")) <= drift.MAX_GITHUB_ISSUE_BODY_BYTES
    assert "more" in body
    assert len(title) <= drift.MAX_GITHUB_ISSUE_TITLE_CHARS
    assert "sha256:" in title
    assert drift.drift_issue_title(result) == title


def test_drift_publisher_reapplies_budgets_after_redaction(monkeypatch):
    secret = "12345678"
    report = drift.DriftReport(
        checked=[
            drift.DriftResult(
                module="us/statutes/1.yaml",
                drifted=True,
                diffs=[
                    {
                        "path": "value",
                        "change": "value_changed",
                        "merged": 1,
                        "regenerated": 2,
                    }
                ],
            )
        ]
    )
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setattr(
        drift,
        "drift_issue_body",
        lambda _result: secret * (drift.MAX_GITHUB_ISSUE_BODY_BYTES // len(secret)),
    )
    monkeypatch.setattr(
        drift,
        "drift_issue_title",
        lambda _result: secret * drift.MAX_GITHUB_ISSUE_TITLE_CHARS,
    )
    published: list[tuple[str, str]] = []

    def fake_github(command, **_kwargs):
        if command[1:3] == ["issue", "create"]:
            title = command[command.index("--title") + 1]
            body_path = Path(command[command.index("--body-file") + 1])
            published.append((title, body_path.read_text(encoding="utf-8")))
            return types.SimpleNamespace(returncode=0, stdout="issue-url\n", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli_commands.subprocess, "run", fake_github)

    assert cli_commands._create_drift_issues(report, "owner/repo") == ["issue-url"]
    title, body = published[0]
    assert secret not in title
    assert secret not in body
    assert len(title) <= drift.MAX_GITHUB_ISSUE_TITLE_CHARS
    assert len(body.encode("utf-8")) <= drift.MAX_GITHUB_ISSUE_BODY_BYTES


def test_drift_error_publisher_neutralizes_markdown_and_mentions(monkeypatch, tmp_path):
    report = drift.DriftReport(
        checked=[
            drift.DriftResult(
                module="us/statutes/1.yaml",
                drifted=False,
                error="failed @axiom [click](https://example.invalid) `payload`",
            )
        ]
    )
    published_bodies: list[str] = []

    def fake_github(command, **_kwargs):
        if command[1:3] == ["issue", "create"]:
            body_path = Path(command[command.index("--body-file") + 1])
            published_bodies.append(body_path.read_text(encoding="utf-8"))
            return types.SimpleNamespace(returncode=0, stdout="issue-url\n", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("GH_TOKEN", "publisher-token")
    monkeypatch.setattr(cli_commands.subprocess, "run", fake_github)

    assert cli_commands._create_drift_issues(report, "owner/repo") == ["issue-url"]
    assert len(published_bodies) == 1
    assert "@axiom" not in published_bodies[0]
    assert "[click](" not in published_bodies[0]
    assert "<pre>" in published_bodies[0]


# -- calibration ----------------------------------------------------------


def test_calibration_rates():
    report = calibration.CalibrationReport(
        n=4,
        n_good=2,
        n_bad=2,
        true_positive=1,
        true_negative=1,
        false_positive=1,
        false_negative=1,
    )
    assert report.false_positive_rate == pytest.approx(0.5)
    assert report.false_negative_rate == pytest.approx(0.5)


def test_calibration_counts_error_separately_not_as_pass():
    cases = [
        calibration.CalibrationCase("g1", "c", "good", "prov", "rule"),
        calibration.CalibrationCase("b1", "c", "bad", "prov", "rule"),
    ]
    # referee always errors -> neither FP nor FN; counted as errors.
    report = calibration.run_calibration(cases, client=FakeClient(_err_call()))
    assert report.errors == 2
    assert report.false_negative == 0
    assert report.false_positive == 0
    assert report.true_negative == 0


def _make_db(path: Path):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE encoding_runs (id TEXT PRIMARY KEY, citation TEXT, "
        "source_text TEXT, rulespec_content TEXT, agent_model TEXT, outcome_json TEXT)"
    )
    rows = []
    for i in range(5):
        rows.append(
            (
                f"good{i}",
                "26 USC 1",
                "src",
                "rule",
                "gpt-5.5",
                '{"status": "apply_applied"}',
            )
        )
        rows.append(
            (
                f"bad{i}",
                "26 USC 2",
                "src",
                "rule",
                "gpt-5.5",
                '{"status": "apply_blocked_validation"}',
            )
        )
    conn.executemany("INSERT INTO encoding_runs VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_calibration_load_cases_balanced(tmp_path):
    db = tmp_path / "e.db"
    _make_db(db)
    cases = calibration.load_cases(db, n=6, seed=0)
    labels = [c.label for c in cases]
    assert labels.count("good") == 3
    assert labels.count("bad") == 3


# -- review-fix regression tests ------------------------------------------


def test_coerce_confidence_clamps_and_tolerates_non_numeric():
    assert coerce_confidence("high") is None
    assert coerce_confidence(None) is None
    assert coerce_confidence(1.5) == 1.0
    assert coerce_confidence(-0.2) == 0.0
    assert coerce_confidence(0.5) == 0.5


def test_fidelity_unrecognized_verdict_is_error_not_pass():
    # A valid-JSON payload with an unrecognized verdict must fail closed, never
    # default to PASS (the fail-open the module bans).
    for bad in ("error", "garbage", ""):
        ev = statutory_fidelity.run(
            "prov",
            "rule",
            citation="c",
            client=FakeClient(
                _ok_call({"verdict": bad, "confidence": 0.9, "findings": []})
            ),
        )
        assert ev.verdict == Verdict.ERROR, bad
        assert ev.passed is False


def test_fidelity_out_of_range_confidence_event_validates():
    ev = statutory_fidelity.run(
        "prov",
        "rule",
        citation="c",
        client=FakeClient(
            _ok_call({"verdict": "pass", "confidence": 1.5, "findings": []})
        ),
    )
    d = ev.to_dict()
    assert d["attrs"]["confidence"] == 1.0
    assert validate_event_dict(d) == []


def test_preclassify_non_numeric_confidence_does_not_crash_batch():
    # A non-numeric arbiter confidence must not raise and take the whole batch
    # down (never-drop invariant).
    bad = FakeClient(
        _ok_call(
            {"classification": "self_contained", "confidence": "high", "reason": "x"}
        )
    )
    r = preclassifier.classify(
        {"citation": "z", "source_text": "An operative clause with no markers here."},
        client=bad,
    )
    assert r.route == preclassifier.GENERATE
    assert r.confidence == 0.0  # coerced from non-numeric


def test_preclassify_empty_source_skips_never_generates():
    r = preclassifier.classify({"citation": "e", "source_text": "   "})
    assert r.route == preclassifier.SKIP
    assert "no source text" in r.reason


def test_generator_unknown_family_is_guarded():
    client = JudgeClient(
        model="claude-haiku-4-5-20251001",
        generator_model="my-inhouse-wrapper",
        api_key="x",
    )
    call = client.call(system="s", user_prompt="p", schema={})
    assert call.error is not None
    assert call.error.type == "cross_family_guard"


def test_client_schema_error_on_missing_required_keys(monkeypatch):
    # Valid JSON that omits a required key is an error, never a usable verdict.
    install_fake_anthropic(monkeypatch, ['{"confidence": 0.9}'])
    client = JudgeClient(model="claude-haiku-4-5-20251001", api_key="test")
    call = client.call(
        system="s",
        user_prompt="p",
        schema={"required": ["verdict", "confidence", "findings"]},
    )
    assert not call.ok
    assert call.error.type == "schema_error"


def test_client_constructor_failure_is_error_not_raise(monkeypatch):
    mod = types.ModuleType("anthropic")
    for name in (
        "APIStatusError",
        "RateLimitError",
        "APIConnectionError",
        "InternalServerError",
        "BadRequestError",
    ):
        setattr(mod, name, type(name, (Exception,), {}))

    class Anthropic:
        def __init__(self, api_key=None):
            raise RuntimeError("no network")

    mod.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    client = JudgeClient(model="claude-haiku-4-5-20251001", api_key="test")
    call = client.call(system="s", user_prompt="p", schema={})
    assert not call.ok
    assert call.error.type == "client_init_error"


# -- canonical run-log emission (axiom_encode.run_log.v1) ------------------


def test_judge_event_emits_into_canonical_run_log(tmp_path):
    # A flagged fidelity verdict written via the canonical RunLogWriter must be
    # a well-formed judge stage event that folds into the per-run run record.
    payload = {
        "verdict": "flag",
        "confidence": 0.8,
        "findings": [
            {
                "clause_ref": "26 USC 32(b)",
                "rule_path": "x.yaml#rate",
                "kind": "amount_mismatch",
                "explanation": "600000 not in source",
            }
        ],
    }
    ev = statutory_fidelity.run(
        "prov",
        "rule",
        citation="26 USC 32",
        run_id="run-abc",
        client=FakeClient(_ok_call(payload)),
    )
    writer = RunLogWriter("run-abc", log_dir=tmp_path)
    written = ev.emit(writer)
    assert written is not None and writer.last_error is None

    events = list(iter_events(tmp_path / "run-abc.jsonl"))
    assert len(events) == 1
    stored = events[0]
    assert stored.stage == JUDGE_STAGE
    # Advisory flag does not gate: the stage completes as "passed".
    assert stored.status == StageStatus.passed.value
    assert stored.attrs["verdict"] == "flag"
    assert stored.attrs["judge_model"] == "claude-haiku-4-5-20251001"
    assert stored.attrs["generator_model"] == "gpt-5.5"
    assert stored.attrs["tokens"] == {"input": 10, "output": 5}
    assert stored.findings[0].code == "amount_mismatch"
    assert stored.findings[0].severity == "critical"

    # The judge event folds into the run record's per-run DAG.
    record = fold_run(events)
    assert record.run_id == "run-abc"


def test_judge_error_emits_error_status_never_silent_pass(tmp_path):
    ev = statutory_fidelity.run(
        "prov", "rule", citation="c", run_id="run-err", client=FakeClient(_err_call())
    )
    writer = RunLogWriter("run-err", log_dir=tmp_path)
    ev.emit(writer)
    stored = list(iter_events(tmp_path / "run-err.jsonl"))[0]
    # Fail-open banned: the judge failure is a visible error event, not a pass.
    assert stored.status == StageStatus.error.value
    assert stored.status != StageStatus.passed.value
    assert stored.attrs["judge_error"]["type"] == "server_error"
    assert validate_event_dict(stored.to_dict()) == []


def test_preclassify_skip_with_error_is_skipped_and_visible():
    # A preclassify skip whose arbiter failed maps to status="skipped" (never a
    # pass, never a drop) while keeping the judge_error visible in attrs.
    err_client = FakeClient(_err_call(err_type="server_error"))
    r = preclassifier.classify(
        {"citation": "X", "source_text": "A short operative clause with no markers."},
        client=err_client,
    )
    d = r.event.to_dict()
    assert d["stage"] == JUDGE_STAGE
    assert d["status"] == "skipped"
    assert d["attrs"]["verdict"] == "skip"
    assert d["attrs"]["judge_error"]["type"] == "server_error"
    assert validate_event_dict(d) == []


def test_advisory_flag_passes_but_promoted_flag_fails():
    advisory = JudgeEvent(stage=JudgeStage.STATUTORY_FIDELITY, verdict=Verdict.FLAG)
    assert advisory.to_dict()["status"] == "passed"

    promoted = JudgeEvent(
        stage=JudgeStage.STATUTORY_FIDELITY, verdict=Verdict.FLAG, advisory=False
    )
    d = promoted.to_dict()
    assert d["status"] == "failed"
    assert d["reason_code"] == "judge_rejected"
