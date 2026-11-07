"""Private lifecycle controls with explicit mocked source/materializer boundaries.

These tests are not corpus authentication, real compilation or source coverage.
"""

import copy
import json
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import mhrtc_capture as m
from axiom_encode.harness import same_case_execution
from axiom_encode.harness.mhrtc_execution import MhrtcAttempt, MhrtcExecutionError
from axiom_encode.harness.mhrtc_request import MhrtcRequestPlan
from tests.test_mhrtc_request import fixture as principal_fixture  # noqa: F401
from tests.test_mhrtc_results import parameters


@dataclass(eq=False)
class Capture:
    original_root: object
    snapshot_root: object
    root_relative: str
    files: tuple
    binary: object
    artifact: object
    binary_sha256: str = "binary"
    engine_pin: str = m.ENGINE_PIN
    git_head: str = "head"
    overlay_origin: object = None
    is_fresh: bool = True
    _mhrtc_context: object = None
    records: list = field(default_factory=list)

    def fresh(self):
        return self.is_fresh


@pytest.fixture
def lifecycle(principal_fixture, monkeypatch, tmp_path):  # noqa: F811
    case, _, _, _ = principal_fixture
    payload, expected = parameters()
    parameter_case = {
        "name": "parameters",
        "entity_id": "p",
        "period": copy.deepcopy(case["period"]),
        "input": {},
        "output": {
            k: float(v) if type(v) is not int else v for k, v in expected.items()
        },
    }
    raw = yaml.safe_dump([case, parameter_case]).encode()
    artifact = tmp_path / "compiled.json"
    artifact.write_text(json.dumps(payload))
    root = SimpleNamespace(
        relative=m.POLICY_PATH, original=tmp_path / m.POLICY_PATH, raw=b"bound policy"
    )
    companion = SimpleNamespace(
        relative=m.POLICY_PATH.removesuffix(".yaml") + ".test.yaml",
        original=tmp_path / "test.yaml",
        raw=raw,
    )
    capture = Capture(
        tmp_path,
        tmp_path,
        m.POLICY_PATH,
        (root, companion),
        tmp_path / "engine",
        artifact,
    )
    release = object()
    pipe = SimpleNamespace(local_corpus_release=release)
    calls = []
    monkeypatch.setattr(
        same_case_execution, "_materialized_capture", lambda value: value is capture
    )
    monkeypatch.setattr(
        m,
        "_source",
        lambda value: (
            ("source", ("source-identity",))
            if value is release
            else (_ for _ in ()).throw(ValueError("release"))
        ),
    )
    monkeypatch.setattr(
        m,
        "bind_mhrtc_program",
        lambda content, source: (
            object() if content == "bound policy" and source == "source" else None
        ),
    )

    def compile_again(c, p):
        calls.append("compile")
        return artifact.read_bytes()

    monkeypatch.setattr(m, "verify_mhrtc_compiled", compile_again)

    def plan(case, payload, **kwargs):
        calls.append(kwargs["case_index"])
        if "relations" in case:
            return MhrtcRequestPlan({"mode": "explain"}, tuple(case["output"]), ())
        return MhrtcRequestPlan(None, (), tuple(case["output"]))

    monkeypatch.setattr(m, "reconstruct_mhrtc_request", plan)
    yield capture, pipe, calls
    m._CAPTURES.pop(id(capture), None)


def mint_attached(capture, pipe):
    assert m.activate(pipe, capture)
    context = m.mint(capture)
    assert context is not None
    capture._mhrtc_context = context
    return context


def test_mint_binds_all_cases_and_separate_parameters(lifecycle):
    capture, pipe, calls = lifecycle
    context = mint_attached(capture, pipe)
    assert calls == ["compile", 1, 2]
    assert context.principal_ordinals == (1,)
    assert context.parameter_ordinals == (2,)
    assert m.valid(context, capture)
    assert not hasattr(context, "cases")  # No retained mutable parsed cases.


def test_unregistered_mint_and_forged_context_rejected(lifecycle):
    capture, pipe, _ = lifecycle
    assert m.mint(capture) is None
    context = mint_attached(capture, pipe)
    forged = replace(context)
    capture._mhrtc_context = forged
    assert not m.valid(forged, capture)


def test_materializer_membership_required(lifecycle, monkeypatch):
    capture, pipe, _ = lifecycle
    monkeypatch.setattr(same_case_execution, "_materialized_capture", lambda _: False)
    assert not m.activate(pipe, capture)


def test_context_does_not_transfer_to_equal_capture(lifecycle):
    capture, pipe, _ = lifecycle
    context = mint_attached(capture, pipe)
    other = copy.copy(capture)
    assert not m.valid(context, other)


@pytest.mark.parametrize(
    "mutation",
    [
        "stale",
        "binary",
        "pin",
        "head",
        "release",
        "source",
        "policy",
        "companion",
        "artifact",
        "extra_file",
    ],
)
def test_mutated_identity_rejected(lifecycle, monkeypatch, mutation):
    capture, pipe, _ = lifecycle
    context = mint_attached(capture, pipe)
    if mutation == "stale":
        capture.is_fresh = False
    elif mutation == "binary":
        capture.binary_sha256 = "changed"
    elif mutation == "pin":
        capture.engine_pin = "changed"
    elif mutation == "head":
        capture.git_head = "changed"
    elif mutation == "release":
        pipe.local_corpus_release = object()
    elif mutation == "source":
        monkeypatch.setattr(m, "_source", lambda _: ("changed", ("source-identity",)))
    elif mutation == "policy":
        capture.files[0].raw += b"changed"
    elif mutation == "companion":
        capture.files[1].raw += b"changed"
    elif mutation == "artifact":
        capture.artifact.write_bytes(b"changed")
    else:
        capture.files += (copy.copy(capture.files[1]),)
    assert not m.valid(context, capture)


def test_reset_invalidates_previous_context_even_same_bytes(lifecycle):
    capture, pipe, _ = lifecycle
    old = mint_attached(capture, pipe)
    new = m.mint(capture)
    assert new is not old
    capture._mhrtc_context = old
    assert not m.valid(old, capture)
    capture._mhrtc_context = new
    assert m.valid(new, capture)


def test_failed_recompile_retains_attempt_and_invalidates_old_context(
    lifecycle, monkeypatch
):
    capture, pipe, _ = lifecycle
    old = mint_attached(capture, pipe)
    attempt = MhrtcAttempt(
        "compile", 1, "", "failure", ("engine",), None, "binary", "artifact"
    )

    def fail(*args):
        raise MhrtcExecutionError("failed", attempt)

    monkeypatch.setattr(m, "verify_mhrtc_compiled", fail)
    assert m.mint(capture) is None
    assert not m.valid(old, capture)
    assert m._CAPTURES[id(capture)][4]["failed_attempt"] is attempt


def test_incomplete_parameter_assertions_prevent_context(lifecycle):
    capture, pipe, _ = lifecycle
    assert m.activate(pipe, capture)
    payload = json.loads(capture.artifact.read_bytes())
    payload["program"]["parameters"][0]["versions"][0]["values"]["0"]["value"] = 1
    capture.artifact.write_text(json.dumps(payload))
    assert m.mint(capture) is None


def test_missing_companion_prevents_context(lifecycle):
    capture, pipe, _ = lifecycle
    capture.files = capture.files[:1]
    assert m.activate(pipe, capture)
    assert m.mint(capture) is None


def test_invalid_later_case_prevents_context(lifecycle, monkeypatch):
    capture, pipe, _ = lifecycle
    assert m.activate(pipe, capture)
    original = m.reconstruct_mhrtc_request

    def fail_later(case, payload, **kwargs):
        if kwargs["case_index"] == 2:
            raise ValueError("bad second case")
        return original(case, payload, **kwargs)

    monkeypatch.setattr(m, "reconstruct_mhrtc_request", fail_later)
    assert m.mint(capture) is None


def test_failed_stale_remint_does_not_revive_old_generation(lifecycle):
    capture, pipe, _ = lifecycle
    old = mint_attached(capture, pipe)
    capture.is_fresh = False
    assert m.mint(capture) is None
    capture.is_fresh = True
    assert not m.valid(old, capture)


@pytest.fixture
def staging(lifecycle, monkeypatch):
    from axiom_encode.harness import mhrtc_execution, mhrtc_results

    capture, pipe, calls = lifecycle
    context = mint_attached(capture, pipe)
    case = yaml.safe_load(context.companion)[0]
    request, response = {"mode": "explain"}, {"reported": True}

    def execute(c, p, reconstructed, reported, result):
        if reconstructed != reported or result != response:
            raise ValueError("fresh comparison differs")
        calls.append("runtime")
        return mhrtc_execution.MhrtcFreshExecution(
            json.dumps(request), json.dumps(response), "", "binary", "artifact"
        )

    monkeypatch.setattr(mhrtc_execution, "reexecute_mhrtc_case", execute)
    monkeypatch.setattr(mhrtc_results, "validate_mhrtc_outputs", lambda *args: True)
    monkeypatch.setattr(m, "_auxiliary", lambda *args: ())
    for item in capture.files:
        for path in (item.original, capture.snapshot_root / item.relative):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item.raw)
    return capture, context, case, request, response, calls


def evidence(capture):
    return same_case_execution.SameCaseEvidence(
        capture.files,
        capture.root_relative,
        capture.snapshot_root,
        capture.git_head,
        capture.binary_sha256,
        capture.engine_pin,
        capture.artifact.read_bytes(),
        tuple(capture.records),
        capture.overlay_origin,
    )


def test_stage_then_complete_private_registration(staging):
    capture, context, case, request, response, calls = staging
    assert m.stage(context, capture, case, request, response, 1)
    assert calls.count("runtime") == 1
    assert len(capture.records) == 1
    result = evidence(capture)
    assert m.register_evidence(context, capture, result)
    assert m.owns_evidence(result)
    assert not m.owns_evidence(replace(result))


def test_incomplete_private_evidence_rejected(staging):
    capture, context, *_ = staging
    assert not m.register_evidence(context, capture, evidence(capture))


@pytest.mark.parametrize("ordinal", [True, 0, 2, 3, 1.0])
def test_wrong_or_static_ordinal_cannot_stage_runtime(staging, ordinal):
    capture, context, case, request, response, calls = staging
    assert not m.stage(context, capture, case, request, response, ordinal)
    assert "runtime" not in calls
    assert not m.register_evidence(context, capture, evidence(capture))


def test_duplicate_stage_poisoned_not_double_counted(staging):
    capture, context, case, request, response, calls = staging
    assert m.stage(context, capture, case, request, response, 1)
    assert not m.stage(context, capture, case, request, response, 1)
    assert calls.count("runtime") == 1
    assert not m.register_evidence(context, capture, evidence(capture))


def test_changed_expected_case_cannot_stage(staging):
    capture, context, case, request, response, calls = staging
    case["name"] += "changed"
    assert not m.stage(context, capture, case, request, response, 1)
    assert "runtime" not in calls


def test_reported_request_different_from_frozen_plan(staging):
    capture, context, case, _, response, calls = staging
    assert not m.stage(context, capture, case, {"mode": "changed"}, response, 1)
    assert "runtime" not in calls


def test_failed_auxiliary_retains_principal_attempt_but_no_complete_record(
    staging, monkeypatch
):
    capture, context, case, request, response, _ = staging

    def fail(*args):
        raise ValueError("required auxiliary missing")

    monkeypatch.setattr(m, "_auxiliary", fail)
    assert not m.stage(context, capture, case, request, response, 1)
    assert len(m._CAPTURES[id(capture)][4]["attempts"]) == 1
    assert capture.records == []
    assert not m.register_evidence(context, capture, evidence(capture))


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "different", "artifact", "head"]
)
def test_copied_or_modified_evidence_cannot_register(staging, mutation):
    capture, context, case, request, response, _ = staging
    assert m.stage(context, capture, case, request, response, 1)
    result = evidence(capture)
    if mutation == "missing":
        result = replace(result, records=())
    elif mutation == "extra":
        result = replace(result, records=result.records * 2)
    elif mutation == "different":
        result = replace(
            result, records=(replace(result.records[0], response_json="{}"),)
        )
    elif mutation == "artifact":
        result = replace(result, artifact_bytes=b"changed")
    else:
        result = replace(result, git_head="different")
    assert not m.register_evidence(context, capture, result)


def test_recompiled_context_invalidates_registered_evidence(staging):
    capture, context, case, request, response, _ = staging
    assert m.stage(context, capture, case, request, response, 1)
    result = evidence(capture)
    assert m.register_evidence(context, capture, result)
    capture._mhrtc_context = m.mint(capture)
    assert not m.owns_evidence(result)


def test_record_mutation_invalidates_registered_evidence(staging):
    capture, context, case, request, response, _ = staging
    assert m.stage(context, capture, case, request, response, 1)
    result = evidence(capture)
    assert m.register_evidence(context, capture, result)
    capture.records.clear()
    assert not m.owns_evidence(result)


def test_stale_stage_is_terminal_after_freshness_restored(staging):
    capture, context, case, request, response, calls = staging
    capture.is_fresh = False
    assert not m.stage(context, capture, case, request, response, 1)
    capture.is_fresh = True
    assert not m.stage(context, capture, case, request, response, 1)
    assert "runtime" not in calls


@pytest.mark.parametrize("location", ["principal", "auxiliary"])
def test_timeout_retained_reraised_and_not_retried(staging, monkeypatch, location):
    import subprocess

    from axiom_encode.harness import mhrtc_execution

    capture, context, case, request, response, _ = staging
    timeout = subprocess.TimeoutExpired(
        ["engine"], 60, output=b"partial", stderr=b"timeout"
    )
    module, attribute = (
        (mhrtc_execution, "reexecute_mhrtc_case")
        if location == "principal"
        else (m, "_auxiliary")
    )
    original = getattr(module, attribute)

    def fail(*args):
        raise timeout

    monkeypatch.setattr(module, attribute, fail)
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        m.stage(context, capture, case, request, response, 1)
    assert raised.value is timeout
    assert m._CAPTURES[id(capture)][4]["failed_process_exception"] is timeout
    monkeypatch.setattr(module, attribute, original)
    assert not m.stage(context, capture, case, request, response, 1)
    assert not m.register_evidence(context, capture, evidence(capture))


def test_unrelated_forged_context_does_not_poison_genuine_generation(staging):
    capture, context, case, request, response, _ = staging
    assert not m.stage(replace(context), capture, case, request, response, 1)
    assert m.stage(context, capture, case, request, response, 1)


@pytest.fixture
def complete(staging):
    capture, context, case, request, response, calls = staging
    assert m.stage(context, capture, case, request, response, 1)
    result = evidence(capture)
    assert m.register_evidence(context, capture, result)
    return capture, context, result


def test_complete_observation_separates_static_assertions(complete):
    capture, context, result = complete
    observed = m.observation(result, context.content.decode())
    assert observed is not None
    assert tuple(r.ordinal for r in observed.principal_receipts) == (1,)
    assert observed.parameter_ordinals == (2,)
    assert not hasattr(observed, "covers_gate")


def test_observation_rejects_detached_or_wrong_content(complete):
    _, context, result = complete
    assert m.observation(replace(result), context.content.decode()) is None
    assert m.observation(result, "other policy") is None
    assert m.observation(result, context.content) is None


@pytest.mark.parametrize("change", ["fresh", "records", "artifact", "source"])
def test_observation_rechecks_liveness(complete, change, monkeypatch):
    capture, context, result = complete
    if change == "fresh":
        capture.is_fresh = False
    elif change == "records":
        capture.records.clear()
    elif change == "artifact":
        capture.artifact.write_bytes(b"changed")
    else:
        monkeypatch.setattr(m, "_source", lambda _: ("changed", ("identity",)))
    assert m.observation(result, context.content.decode()) is None


def test_clear_does_not_revive_when_public_context_restored(complete):
    capture, context, result = complete
    m.clear(capture)
    capture._mhrtc_context = context
    assert not m.valid(context, capture)
    assert m.observation(result, context.content.decode()) is None


def test_refusal_cannot_substitute_for_missing_execution(complete):
    capture, context, result = complete
    assert m.refusal_target(context, capture, {}, 1) is None
    assert not m.stage_refusal(context, capture, {}, {}, 1, "", "failure", 1)
    assert not m.owns_evidence(result)


def test_forged_refusal_context_does_not_clear_genuine_evidence(complete):
    capture, context, result = complete
    assert not m.stage_refusal(replace(context), capture, {}, {}, 1, "", "", 1)
    assert m.owns_evidence(result)
    assert m.observation(result, context.content.decode()) is not None


def test_source_binding_requires_whole_caller_companion(complete):
    _, context, result = complete
    cases = yaml.safe_load(context.companion)
    binding = m.source_binding(result, context.content.decode(), context.source, cases)
    assert binding is not None and binding[0] is context
    assert m.source_binding_live(result, context)
    assert (
        m.source_binding(result, context.content.decode(), context.source, tuple(cases))
        is not None
    )


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "reordered", "expected", "input", "source", "detached"],
)
def test_source_binding_rejects_caller_substitution(complete, change):
    _, context, result = complete
    cases = yaml.safe_load(context.companion)
    source = context.source
    if change == "missing":
        cases.pop()
    elif change == "duplicate":
        cases.append(cases[0])
    elif change == "reordered":
        cases.reverse()
    elif change == "expected":
        cases[0]["output"][next(iter(cases[0]["output"]))] = 999
    elif change == "input":
        cases[0]["input"]["invented"] = 1
    elif change == "source":
        source += "changed"
    else:
        result = replace(result)
    assert m.source_binding(result, context.content.decode(), source, cases) is None


def test_source_binding_live_never_accepts_cloned_context_or_reset(complete):
    capture, context, result = complete
    assert not m.source_binding_live(result, replace(context))
    m.clear(capture)
    assert not m.source_binding_live(result, context)


def test_source_coverage_uses_private_inputs_and_retains_missing(complete, monkeypatch):
    from axiom_encode.harness import mhrtc_coverage as coverage_module

    capture, context, result = complete
    expected = coverage_module.MhrtcCoverage((), ("unfinished",))
    calls = []

    def observe(actual_context, receipts, *, pipe, rules_file):
        calls.append((actual_context, receipts, pipe, rules_file))
        return expected

    monkeypatch.setattr(coverage_module, "observe_mhrtc_coverage", observe)
    binding = m.source_coverage_binding(
        result,
        context.content.decode(),
        context.source,
        yaml.safe_load(context.companion),
    )
    assert binding is not None and binding[0] is context and binding[2] is expected
    assert len(calls) == 1
    assert calls[0] == (
        context,
        m.observation(result, context.content.decode()).principal_receipts,
        m._CAPTURES[id(capture)][1],
        capture.snapshot_root / m.POLICY_PATH,
    )
    assert binding[2].missing == ("unfinished",)


@pytest.mark.parametrize("change", ["detached", "cases", "stale_before"])
def test_source_coverage_rejects_invalid_binding_before_observer(
    complete, monkeypatch, change
):
    from axiom_encode.harness import mhrtc_coverage as coverage_module

    capture, context, result = complete
    cases = yaml.safe_load(context.companion)
    if change == "detached":
        result = replace(result)
    elif change == "cases":
        cases.pop()
    else:
        capture.is_fresh = False

    def forbidden(*args, **kwargs):
        pytest.fail("invalid private binding reached coverage observer")

    monkeypatch.setattr(coverage_module, "observe_mhrtc_coverage", forbidden)
    assert (
        m.source_coverage_binding(
            result, context.content.decode(), context.source, cases
        )
        is None
    )


@pytest.mark.parametrize("change", ["stale_after", "reset_after", "malformed", "error"])
def test_source_coverage_rechecks_generation_after_analysis(
    complete, monkeypatch, change
):
    from axiom_encode.harness import mhrtc_coverage as coverage_module

    capture, context, result = complete

    def observe(*args, **kwargs):
        if change == "stale_after":
            capture.is_fresh = False
        elif change == "reset_after":
            m.clear(capture)
        elif change == "malformed":
            return {"missing": (), "witnesses": ()}
        else:
            raise ValueError("invalid receipt")
        return coverage_module.MhrtcCoverage((), ())

    monkeypatch.setattr(coverage_module, "observe_mhrtc_coverage", observe)
    assert (
        m.source_coverage_binding(
            result,
            context.content.decode(),
            context.source,
            yaml.safe_load(context.companion),
        )
        is None
    )
