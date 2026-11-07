"""Adversarial helper tests, not source-admission or real-engine evidence."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from axiom_encode.harness import mhrtc_execution as m


@pytest.mark.parametrize("raw", ['{"a":1,"a":1}', "NaN", "1e999", "[Infinity]"])
def test_non_json_or_ambiguous_numbers_rejected(raw):
    with pytest.raises(ValueError):
        m.strict_json(raw)


@pytest.mark.parametrize(
    ("left", "right"),
    [(True, 1), ([1, 2], [2, 1]), ({"x": 1}, {"x": "1"})],
)
def test_wire_types_and_order_remain_distinct(left, right):
    assert not m.same_typed_json(left, right)


def test_object_serialization_order_not_semantics():
    assert m.same_typed_json({"x": 1, "y": [False]}, {"y": [False], "x": 1})


@pytest.mark.parametrize(
    "number", ["10000000000000000.0", "10000000000000001.0", "0.145"]
)
def test_floating_tokens_cannot_silently_collapse(number):
    with pytest.raises(ValueError, match="floating"):
        m.strict_json('{"amount":' + number + "}")
    with pytest.raises(ValueError, match="wire strings"):
        m.same_typed_json(float(number), float(number))


@pytest.fixture
def execution(tmp_path, monkeypatch):
    program = tmp_path / "policy.yaml"
    program.write_text("format: rulespec/v1\n")
    artifact = tmp_path / "compiled.json"
    artifact.write_text('{"rules":[{"literal":1}]}')
    binary = tmp_path / "engine"
    binary.write_bytes(b"test stub; not a genuine engine")
    capture = SimpleNamespace(
        artifact=artifact,
        snapshot_root=tmp_path,
        root_relative="policy.yaml",
        binary_sha256=m._digest(binary),
    )
    pipe = SimpleNamespace(
        axiom_rules_path=tmp_path,
        _rulespec_engine_env=lambda: {"SAFE_TEST": "1"},
    )
    # Only the helper transport is tested here. Genuine capture/receipt checks
    # have their own rejection test below and require later runtime integration.
    monkeypatch.setattr(m, "_verified_binary", lambda c, p: binary)
    return capture, pipe, binary


def test_fabricated_capture_rejected_before_engine_resolution(tmp_path):
    with pytest.raises(ValueError, match="materialized"):
        m._verified_binary(SimpleNamespace(), SimpleNamespace())


def test_recompile_is_separate_and_preserves_types(execution, monkeypatch):
    capture, pipe, _ = execution
    original = capture.artifact.read_bytes()

    def compile(**kw):
        assert kw["output"] != capture.artifact
        assert kw["rulespec_roots"] == (capture.snapshot_root,)
        assert kw["timeout"] == 60 and kw["env"] == {"SAFE_TEST": "1"}
        kw["output"].write_text('{"rules": [{"literal": 1.0}]}')
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(m, "run_rulespec_compile", compile)
    with pytest.raises(m.MhrtcExecutionError, match="floating") as caught:
        m.verify_mhrtc_compiled(capture, pipe)
    assert caught.value.attempt.rebuilt_ir == b'{"rules": [{"literal": 1.0}]}'
    assert capture.artifact.read_bytes() == original


def test_compilation_mutation_rejected(execution, monkeypatch):
    capture, pipe, _ = execution

    def compile(**kw):
        kw["output"].write_bytes(capture.artifact.read_bytes())
        capture.artifact.write_text('{"rules":[]}')
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(m, "run_rulespec_compile", compile)
    with pytest.raises(ValueError, match="changed"):
        m.verify_mhrtc_compiled(capture, pipe)


def test_fresh_run_occurs_even_for_matching_report(execution, monkeypatch):
    capture, pipe, binary = execution
    request = {"mode": "explain", "queries": [{"entity_id": "person-1"}]}
    response = {"results": [{"value": {"kind": "decimal", "value": "1"}}]}
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        assert json.loads(kw["input"]) == request
        assert kw["timeout"] == 60 and kw["env"] == {"SAFE_TEST": "1"}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(response), "")

    monkeypatch.setattr(m.subprocess, "run", run)
    receipt = m.reexecute_mhrtc_case(capture, pipe, request, request, response)
    assert len(calls) == 1 and calls[0][0][0] == str(binary)
    assert receipt.artifact_sha256 == m._digest(capture.artifact)


def test_wrong_request_never_runs(execution, monkeypatch):
    capture, pipe, _ = execution
    monkeypatch.setattr(
        m.subprocess, "run", lambda *a, **k: pytest.fail("must not run")
    )
    with pytest.raises(ValueError, match="reconstructed"):
        m.reexecute_mhrtc_case(capture, pipe, {"mode": "explain"}, {}, {})


def test_auxiliary_receipt_requires_an_actual_run_not_a_report(execution, monkeypatch):
    capture, pipe, binary = execution
    request = {"mode": "explain", "queries": [{"entity_id": "other-1"}]}
    response = '{"results":[{"value":{"kind":"decimal","value":"12000"}}]}'
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        assert json.loads(kw["input"]) == request
        assert kw["timeout"] == 60 and kw["env"] == {"SAFE_TEST": "1"}
        return subprocess.CompletedProcess(cmd, 0, response, "diagnostic")

    monkeypatch.setattr(m.subprocess, "run", run)
    receipt = m.execute_mhrtc_auxiliary_request(capture, pipe, request)
    assert len(calls) == 1 and calls[0][0] == str(binary)
    assert receipt.stdout == response and receipt.stderr == "diagnostic"
    assert receipt.artifact_sha256 == m._digest(capture.artifact)


@pytest.mark.parametrize("failure", ["nonzero", "malformed", "mutation"])
def test_auxiliary_failure_retains_real_attempt(execution, monkeypatch, failure):
    capture, pipe, _ = execution

    def run(cmd, **kw):
        if failure == "mutation":
            capture.artifact.write_text("{}")
        return subprocess.CompletedProcess(
            cmd,
            7 if failure == "nonzero" else 0,
            "not-json" if failure == "malformed" else "{}",
            "actual stderr",
        )

    monkeypatch.setattr(m.subprocess, "run", run)
    with pytest.raises(m.MhrtcExecutionError) as caught:
        m.execute_mhrtc_auxiliary_request(capture, pipe, {"mode": "explain"})
    assert caught.value.attempt.stderr == "actual stderr"
    assert caught.value.attempt.command[1] == "run-compiled"


def test_auxiliary_fabricated_capture_cannot_launch(monkeypatch):
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: pytest.fail("no launch"))
    with pytest.raises(ValueError, match="materialized"):
        m.execute_mhrtc_auxiliary_request(
            SimpleNamespace(), SimpleNamespace(), {"mode": "explain"}
        )


@pytest.mark.parametrize(
    "failure", ["wrong", "duplicate", "nonzero", "timeout", "mutation"]
)
def test_fresh_failure_never_yields_record(execution, monkeypatch, failure):
    capture, pipe, _ = execution

    def run(cmd, **kw):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 60)
        if failure == "mutation":
            capture.artifact.write_text("{}")
        stdout = '{"x":1,"x":1}' if failure == "duplicate" else '{"x":1}'
        return subprocess.CompletedProcess(
            cmd, 1 if failure == "nonzero" else 0, stdout, "failure detail"
        )

    monkeypatch.setattr(m.subprocess, "run", run)
    with pytest.raises((m.MhrtcExecutionError, subprocess.TimeoutExpired)) as caught:
        m.reexecute_mhrtc_case(
            capture,
            pipe,
            {"mode": "explain"},
            {"mode": "explain"},
            {"x": 2 if failure == "wrong" else 1},
        )
    if failure != "timeout":
        assert caught.value.attempt.stdout
        assert caught.value.attempt.stderr == "failure detail"
        assert caught.value.attempt.returncode == (1 if failure == "nonzero" else 0)
        assert caught.value.attempt.command[1:] == (
            "run-compiled",
            "--artifact",
            str(capture.artifact),
        )
        assert json.loads(caught.value.attempt.request_json) == {"mode": "explain"}
        assert caught.value.attempt.binary_sha256 == capture.binary_sha256
        assert len(caught.value.attempt.artifact_sha256) == 64


@pytest.mark.parametrize("kind", ["good", "pin", "stale", "path", "hash", "receipt"])
def test_binary_guards(tmp_path, monkeypatch, kind):
    from axiom_encode.engine_binding import EnginePin
    from axiom_encode.harness import same_case_execution

    binary = tmp_path / "engine"
    binary.write_bytes(b"unit-test binary, not a receipt")
    cap = SimpleNamespace(
        fresh=lambda: kind != "stale",
        engine_pin=m.ENGINE_PIN,
        artifact=tmp_path / "compiled.json",
        binary=binary,
        binary_sha256="wrong" if kind == "hash" else m._digest(binary),
    )
    pin = EnginePin("0" * 40 if kind == "pin" else m.ENGINE_PIN, tmp_path / "pin.toml")
    pipe = SimpleNamespace(axiom_rules_path=tmp_path, _declared_engine_pin=lambda: pin)
    monkeypatch.setattr(
        same_case_execution, "_materialized_capture", lambda c: c is cap
    )
    calls = []

    def resolve(root, actual_pin, *, allow_build):
        calls.append(actual_pin)
        assert root == tmp_path and actual_pin == pin and allow_build is False
        if kind == "receipt":
            raise ValueError("invalid receipt")
        return tmp_path / "other" if kind == "path" else binary

    monkeypatch.setattr(m, "resolve_pinned_engine_binary", resolve)
    if kind == "good":
        assert m._verified_binary(cap, pipe) == binary
    else:
        with pytest.raises(ValueError):
            m._verified_binary(cap, pipe)
    assert len(calls) == (0 if kind in {"pin", "stale"} else 1)


def test_post_execution_freshness_failure_retains_attempt(execution, monkeypatch):
    cap, pipe, binary = execution
    calls = []

    def verify(c, p):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("receipt changed")
        return binary

    monkeypatch.setattr(m, "_verified_binary", verify)
    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "{}", "retained"),
    )
    with pytest.raises(m.MhrtcExecutionError, match="receipt changed") as caught:
        m.reexecute_mhrtc_case(cap, pipe, {"mode": "explain"}, {"mode": "explain"}, {})
    assert len(calls) == 2 and caught.value.attempt.stderr == "retained"
