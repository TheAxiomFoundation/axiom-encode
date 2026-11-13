"""Actual subprocess provenance mechanics; fixture compiler is not engine895."""

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from axiom_encode.harness import local_compile_custody as c
from axiom_encode.rules_engine_compat import run_rulespec_compile


def make_world(tmp_path, artifact='{"compiled":true}', *, real_engine=None, admit=True):
    from axiom_encode.harness import actual_local_relation_evidence as evidence
    from axiom_encode.harness.same_case_execution import ClosureFile, _payload, digest

    root = tmp_path / "rulespec-ca"
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    relative = "policy.yaml" if real_engine is None else "ca/policies/test.yaml"
    source = root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("format: rulespec/v1\nmodule: {}\nimports: []\n")
    if real_engine is not None:
        source.write_text("""format: rulespec/v1
module: {}
imports: []
rules:
  - name: amount
    kind: parameter
    dtype: Money
    unit: CAD
    versions:
      - effective_from: 2025-01-01
        formula: "10"
""")
    subprocess.run(["git", "-C", str(root), "add", relative], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    binary = tmp_path / "compiler"
    binary.write_text(
        '#!/usr/bin/env python3\nimport sys,pathlib\np=pathlib.Path(sys.argv[sys.argv.index("--output")+1])\np.write_text('
        + repr(artifact)
        + ")\n"
    )
    binary.chmod(0o755)
    if real_engine is not None:
        binary = real_engine.binary
    pin = (
        subprocess.check_output(
            ["git", "-C", str(real_engine.root), "rev-parse", "HEAD"], text=True
        ).strip()
        if real_engine is not None
        else "fixture-pin"
    )
    pipe = SimpleNamespace(
        _declared_engine_pin=lambda: SimpleNamespace(sha=pin),
        _source_texts_for_rulespec_content=lambda content: {},
        _proof_source_texts_for_rulespec_content=lambda content, *, source_texts: (
            source_texts
        ),
    )
    plan = (
        root,
        {source: ClosureFile(relative, source, source.read_bytes())},
        head,
        binary,
        digest(binary.read_bytes()),
        root,
        None,
    )
    # This fixture tests compile custody, not source/recording eligibility.
    # Admit only its exact payload, and restore eligibility before compilation.
    fixture_payload = _payload(source.read_bytes())
    original_eligibility = evidence.local_relation_snapshot_eligible
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            evidence,
            "local_relation_snapshot_eligible",
            lambda payload: admit and payload == fixture_payload,
        )
        capture = c._materialize_registered(
            pipe, source, plan, str(tmp_path / "materialized")
        )
    assert evidence.local_relation_snapshot_eligible is original_eligibility
    if not admit:
        assert capture is None
        return None
    assert capture is not None
    pipeline = SimpleNamespace(
        _rulespec_compile_roots=lambda: (capture.snapshot_root,),
        _rulespec_engine_env=lambda: {"PATH": "/usr/bin:/bin:/usr/local/bin"},
        axiom_rules_path=capture.snapshot_root,
    )
    return (
        capture,
        pipeline,
        capture.snapshot_root / relative,
        tmp_path / "compiled.json",
    )


@pytest.fixture
def world(tmp_path):
    value = make_world(tmp_path)
    yield value
    c._expire(value[0])


def compile_and_bind(world, *, owner=None):
    capture, pipeline, source, output = world
    with c.compile_scope(capture, pipeline, source, output):
        result = run_rulespec_compile(
            binary=capture.binary,
            program=source,
            rulespec_roots=(capture.snapshot_root,),
            output=output,
            cwd=capture.snapshot_root,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin"},
            _execution_owner=pipeline if owner is None else owner,
        )
    capture.artifact = output
    capture.artifact_sha256 = (
        hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None
    )
    return result


def test_refused_recording_admission_creates_no_compile_owner(tmp_path):
    from axiom_encode.harness import actual_local_relation_evidence as evidence

    original_eligibility = evidence.local_relation_snapshot_eligible
    owners_before = dict(c._owners)
    assert make_world(tmp_path, admit=False) is None
    assert c._owners == owners_before
    assert evidence.local_relation_snapshot_eligible is original_eligibility


def test_success_consumed_once(world):
    assert compile_and_bind(world).returncode == 0
    capture, _, _, output = world
    assert json.loads(c.consume_compilation(capture, output)) == {"compiled": True}
    assert c.consume_compilation(capture, output) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "unregistered",
        "different_owner",
        "preexisting",
        "changed_source",
        "changed_artifact",
        "symlink",
        "expired",
    ],
)
def test_invalid_custody_declines_without_changing_compile(world, mutation):
    capture, _, source, output = world
    if mutation == "unregistered":
        c._expire(capture)
    if mutation == "preexisting":
        output.write_text("{}")
    if mutation == "changed_source":
        source.write_text("changed")
    assert (
        compile_and_bind(
            world, owner=object() if mutation == "different_owner" else None
        ).returncode
        == 0
    )
    if mutation == "changed_artifact":
        output.write_text("{}")
    if mutation == "symlink":
        target = output.with_suffix(".other")
        output.rename(target)
        output.symlink_to(target)
    if mutation == "expired":
        c._expire(capture)
    assert c.consume_compilation(capture, output) is None


def test_bare_compiled_path_cannot_mint(world):
    capture, _, _, output = world
    output.write_text("{}")
    capture.artifact = output
    capture.artifact_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    assert c.consume_compilation(capture, output) is None


def test_copied_owner_cannot_borrow(world):
    capture, _, _, output = world
    assert compile_and_bind(world).returncode == 0
    assert c.consume_compilation(SimpleNamespace(**vars(capture)), output) is None


def test_new_attempt_invalidates_prior_success(world):
    capture, pipeline, source, output = world
    assert compile_and_bind(world).returncode == 0
    with c.compile_scope(capture, pipeline, source, output):
        pass
    assert c.consume_compilation(capture, output) is None


def test_exception_and_nested_context_reset(world):
    capture, pipeline, source, output = world
    with (
        pytest.raises(RuntimeError),
        c.compile_scope(capture, pipeline, source, output),
    ):
        with c.compile_scope(None, pipeline, source, output):
            assert c._attempt.get() is None
        assert c._attempt.get() is not None
        raise RuntimeError("body failure")
    assert c._attempt.get() is None
    assert c.consume_compilation(capture, output) is None


@pytest.mark.parametrize("artifact", ['{"a":1,"a":2}', "[]", "not-json"])
def test_invalid_artifact_declines(world, artifact):
    capture, pipeline, source, output = world
    c._expire(capture)
    new_world = make_world(output.parent / "duplicate-case", artifact)
    try:
        assert compile_and_bind(new_world).returncode == 0
        assert c.consume_compilation(new_world[0], new_world[3]) is None
    finally:
        c._expire(new_world[0])


@pytest.mark.parametrize("wrong", ["cwd", "env"])
def test_subprocess_descriptor_mismatch(world, wrong):
    capture, pipeline, source, output = world
    with c.compile_scope(capture, pipeline, source, output):
        result = run_rulespec_compile(
            binary=capture.binary,
            program=source,
            rulespec_roots=(capture.snapshot_root,),
            output=output,
            cwd=output.parent if wrong == "cwd" else capture.snapshot_root,
            env={
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                **({"UNRELATED": "changed"} if wrong == "env" else {}),
            },
            _execution_owner=pipeline,
        )
    assert result.returncode == 0
    capture.artifact = output
    capture.artifact_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    assert c.consume_compilation(capture, output) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("git_head", "replacement"),
        ("engine_pin", "replacement"),
        ("root_relative", "other.yaml"),
        ("files", ()),
    ],
)
def test_capture_binding_cannot_be_rebased(world, field, value):
    capture, _, _, output = world
    assert compile_and_bind(world).returncode == 0
    setattr(capture, field, value)
    assert c.consume_compilation(capture, output) is None


def test_reconstructed_concrete_capture_does_not_register(world):
    from dataclasses import replace

    capture, pipeline, source, output = world
    reconstructed = replace(capture)
    assert compile_and_bind((reconstructed, pipeline, source, output)).returncode == 0
    assert c.consume_compilation(reconstructed, output) is None


def test_optional_setup_failure_yields_normal_body(world):
    capture, pipeline, source, output = world

    def unavailable():
        raise RuntimeError("optional evidence setup")

    pipeline._rulespec_engine_env = unavailable
    with c.compile_scope(capture, pipeline, source, output):
        assert c._attempt.get() is None
        output.write_text("normal-body-executed")
    assert output.read_text() == "normal-body-executed"
    assert c.consume_compilation(capture, output) is None


def test_source_mutation_after_success_rejects(world):
    capture, _, source, output = world
    assert compile_and_bind(world).returncode == 0
    source.write_text("changed after successful compile")
    assert c.consume_compilation(capture, output) is None


def test_pinned_engine_real_materializer_compile_receipt(real_engine, tmp_path):
    value = make_world(tmp_path, real_engine=real_engine)
    assert (
        value[0].engine_pin
        == subprocess.check_output(
            ["git", "-C", str(real_engine.root), "rev-parse", "HEAD"], text=True
        ).strip()
    )
    try:
        result = compile_and_bind(value)
        assert result.returncode == 0, result.stderr
        raw = c.consume_compilation(value[0], value[3])
        assert raw is not None
        assert json.loads(raw) == json.loads(value[3].read_bytes())
    finally:
        c._expire(value[0])


def test_executable_override_cannot_claim_pinned_binary(world):
    capture, pipeline, source, output = world
    alternate = output.parent / "alternate"
    alternate.write_text(
        '#!/usr/bin/env python3\nimport sys,pathlib\npathlib.Path(sys.argv[sys.argv.index("--output")+1]).write_text(\'{"alternate":true}\')\n'
    )
    alternate.chmod(0o755)
    command = [
        str(capture.binary),
        "compile",
        "--program",
        str(source),
        "--rulespec-root",
        str(capture.snapshot_root),
        "--output",
        str(output),
    ]
    with c.compile_scope(capture, pipeline, source, output):
        result = c.run_observed_compile(
            command,
            execution_owner=pipeline,
            composed=False,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(capture.snapshot_root),
            env=pipeline._rulespec_engine_env(),
            executable=str(alternate),
        )
    assert result.returncode == 0
    capture.artifact = output
    capture.artifact_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(output.read_bytes()) == {"alternate": True}
    assert c.consume_compilation(capture, output) is None
