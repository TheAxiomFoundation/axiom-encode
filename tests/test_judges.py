"""Tests for the LLM judge stages (maximum-traceability part 2).

Emphasis on the load-bearing invariants:

* a judge failure is a visible ``judge_error`` event and is NEVER treated as a
  pass (fail-open is banned);
* the pre-classifier never drops an entry silently — only skip-with-reason;
* the cross-family guard rejects a judge sharing the generator's family.
"""

from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path

import pytest

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
    disposition,
    drift,
    error_event,
    grid_adequacy,
    model_family,
    preclassifier,
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


def test_drift_ignores_order_and_comments_catches_change():
    a = "outputs:\n  r:\n    value: 0.34\ninputs: [x, y]\n"
    reordered = "inputs: [y, x]  # comment\noutputs:\n  r:\n    value: 0.34\n"
    assert drift.semantic_diff(a, reordered) == []
    changed = "outputs:\n  r:\n    value: 0.40\ninputs: [x, y]\n"
    diffs = drift.semantic_diff(a, changed)
    assert diffs and diffs[0]["change"] == "value_changed"


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
