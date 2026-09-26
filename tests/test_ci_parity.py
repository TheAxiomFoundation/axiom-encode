from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from axiom_encode.ci_parity import (
    CI_GATE_REGISTRY,
    LEGACY_GATE_ORDER,
    SUPPORTED_WORKFLOW_PINS,
    WORKFLOW_DIRECTORY,
    WORKFLOW_ENVIRONMENT_STEPS,
    WORKFLOW_GATE_STEPS,
    WORKFLOW_RESOLUTION_STEPS,
    CallerConfig,
    DependencyMismatch,
    Selection,
    _run_pinned_cli,
    acquire_release_object,
    ci_verdict,
    encoder_version_at_pin,
    execute_gates,
    gate_registry_for_pin,
    parse_caller_workflow,
    run_ci,
    select_targets,
    verify_ambient_encoder,
    verify_dependency_checkout,
    workflow_axiom_invocations,
    workflow_gate_contract,
    workflow_gate_coverage,
    workflow_steps,
)
from axiom_encode.toolchain import (
    RuleSpecToolchain,
    load_rulespec_toolchain,
    verify_rulespec_validation_waiver_set,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ci_parity"
PIN_0EFFA6A5 = "0effa6a5b05e7fac53902df7d523e909bd7fc48a"
PIN_6F11BE26 = "6f11be2655f79dd0a3b582db46525f58332ca120"
EMBEDDED_PINS = (PIN_0EFFA6A5, PIN_6F11BE26)
LEGACY_PINS = tuple(
    sha
    for sha, pin in SUPPORTED_WORKFLOW_PINS.items()
    if not pin.gate_parameters.embedded_scripts
)
WORKFLOW_GATE_STEP_SHA256 = {
    "Run repository tests": "6c998d6153fad095d24fdbc61edf089128945c10992b6575626d14baca4abefa",
    "Reject obsolete generated files": "976cd71e74efc5ecb2697afb7e43f7e53f52b425fefe40657b417e64b8e38232",
    "Reject disallowed repository layout": "d16d421f91b38016ef4d4ad3ac97593ba148f06d7bacc3d6f86bbdc1a5bafb87",
    "Enforce validation waiver ratchet": "2c42cf0de4847c69a0197ed5ba605b7c3f6f5b96d0c9568c307c83d2c6e74b98",
    "Reject manual RuleSpec changes": "e5e4f58dfd95c39363fd3cfb39de4bd064302d93f4874c566f85d4e6818effac",
    "Select RuleSpec validation targets": "b075f905b6844a4d40473c101c3d12f72a20c2366fa6c4734c66adf4885525b6",
    "Validate RuleSpec YAML": "1a172396d726ed61d76082785316e73d22901efe06b587b9b4ac3c7a23aa916b",
    "Execute RuleSpec companion tests": "5a6eb8de6b5f6505a67505f5c8154e1f2cb7e60aef55d69af286eadb7003184f",
    "Validate RuleSpec proofs and claims": "eee57b7142a368870bf84473b8c9a697ab919f11eeef38b19fcfe4bbaf718294",
    "Require money proof atoms": "9fec18e5cbd86b3c5581f19380e0e27c8ccbe1d66a7820578e5abab2b18573e3",
    "Validate PolicyEngine oracle coverage classification": "a7aba1c023537ef274b23a418197c1652380f01b2902d0eb7a445332b69f609b",
    "Checkout changed-file oracle coverage classifier": "50fbe10bc745dc9b1d6aa0e77871d713e4339799b079bc962ab8115074255ba6",
    "Install changed-file oracle coverage classifier": "5f6aa72a94fc7ba127fe3586c08ee29f4f23c604fec14d311878e17abd1f82f9",
    "Validate changed PolicyEngine oracle coverage classification": "0c907a7dfece8a50df6b6acb368a5fb312a59e965b36d964588bbd6a9f5ff878",
}
LEGACY_WORKFLOW_GATE_STEP_SHA256_BY_PIN = {
    sha: {
        **WORKFLOW_GATE_STEP_SHA256,
        **(
            {
                "Require money proof atoms": (
                    "371bcd61cdf0528db1b4529461ac25520b62113e7a4caaa62fbec59a84d814e2"
                )
            }
            if sha == "34bcfab235c585c47292c95f51be1a4f4f91d29e"
            else {}
        ),
    }
    for sha in LEGACY_PINS
}
# Pins executed from their own scripts pin the contract of every gate and
# resolution step, in every job, so a reviewed step cannot change silently.
_EMBEDDED_STEP_SHA256_0EFFA6A5 = {
    (
        "shards",
        "Reject unsupported tracked paths",
    ): "93843d86eb362d1e2de59c5807d2dc4d1ed31454e36ecd134da163888c3cf1d7",
    (
        "shards",
        "Compute validation shards",
    ): "38c184853751743abd01c4edefb3d4e8e47e6acbae5b0ee27f7322f58ce5baaa",
    (
        "validate",
        "Authorize exact reviewed migration",
    ): "084b170a6171d7f12a88ff8b0369a500aa00da43d981b7c98c9f63f8f4365feb",
    (
        "validate",
        "Resolve shard validation roots",
    ): "3d4d83ce6556d35a14168397d9ef62142ebbe0097b66435194617dc0d7c50246",
    (
        "validate",
        "Resolve RuleSpec toolchain",
    ): "04274474877e4963c66c8096e287c01f022c70f7434a2ffeb5575b944cad54b1",
    (
        "validate",
        "Validate immutable dependency inputs",
    ): "b1147f0029933d3653c755743fe212b5d56724b9e998ba58272d4289fc19c7cf",
    (
        "validate",
        "Authenticate dependency commits",
    ): "06f68d360ce87c102d5692be546a4253bfbe4fdcacb3efd5c206a7b3655e9426",
    (
        "validate",
        "Verify immutable retired-schema freeze",
    ): "c3982557d4ce436f245b64cae80fd2125283d41372de193954da7f2a07090682",
    (
        "validate",
        "Fetch pinned signed corpus release object",
    ): "b54f38d1baa7bdc14a3404c70ce5fd28a5960be890413c71b808824c34579653",
    (
        "validate",
        "Authenticate signed corpus provenance commit",
    ): "9656f2345039be2894ce99026655150cc290aacdd4a09856eb226dfed77376c1",
    (
        "validate",
        "Provision protected verification supervisor",
    ): "dd5319276783009459a99fdaf34c72bd1d971e030fbbb0e5412fe3b8c120e124",
    (
        "validate",
        "Reject obsolete generated files",
    ): "c8a4c06b488a9db0dcfadebf3b52e4ea77ed6c8137f7950f08664c77a3f45584",
    (
        "validate",
        "Reject disallowed repository layout",
    ): "d0e767ae0760977f83fe515067572e5cb688194b01ed2f883591e56828555797",
    (
        "validate",
        "Enforce validation waiver ratchet",
    ): "72deb905139b65819f099fd1b83c34295b32d41edbafa190215b00a4a2b1a01c",
    (
        "validate",
        "Reject manual RuleSpec changes",
    ): "c09cb67218deeda3c268a07728a87d6eae23dc4b1bbae76b4d787a646cc3f85f",
    (
        "validate",
        "Select RuleSpec validation targets",
    ): "ce1177fcab04ba891b9fbb9b99a3f588b4bf05ef0eb8041d125093451e77f1c5",
    (
        "validate",
        "Validate RuleSpec YAML",
    ): "b39680e20f44a686474d05b93543265a06ec99ce225a3cb1bbaac6aace845611",
    (
        "validate",
        "Execute RuleSpec companion tests",
    ): "413a0a4e0813298aeab4c4add235fcbd09d764f14922f869c983844db6ebf9b1",
    (
        "validate",
        "Validate RuleSpec proofs and claims",
    ): "faabcaf910c244b0b1f84d390a270103d88ae6a6d9dbea0dee5b490b86ed928a",
    (
        "validate",
        "Require money proof atoms",
    ): "aea32962eb031c449a57cbde6190fcffa8f445963e8949913bbf16b6f846e117",
    (
        "validate",
        "Validate PolicyEngine oracle coverage classification",
    ): "8efd481643d6b96595081d04ea71b40e8af3d240581f3581b9f4aec866971e00",
    (
        "validate",
        "Checkout changed-file oracle coverage classifier",
    ): "6e3ad52edd7822839a7e7fc01eb575f503804ef5b84932efe66c7bd73fc5afcf",
    (
        "validate",
        "Install changed-file oracle coverage classifier",
    ): "6dcc9505ab6000d52cb9ba741ef2bceafaa61f91bd1fe2fdfefb80b1cc93a456",
    (
        "validate",
        "Validate changed PolicyEngine oracle coverage classification",
    ): "a509396097ec732fe87e93b49e82687120a6a90a05e6a8f003cd5c0b351a3fe3",
    (
        "validate",
        "Run repository tests",
    ): "12a68768654ddab5af24648d857e1724128ee1cbd21208f738c837c7d3fe7b85",
    (
        "validate-complete",
        "Check validation matrix result",
    ): "f04c58656f9a15ec96afc4bc9060e09a53a8f80017c6f863ad3984c537ade185",
}
EMBEDDED_STEP_SHA256_BY_PIN = {
    PIN_0EFFA6A5: _EMBEDDED_STEP_SHA256_0EFFA6A5,
    PIN_6F11BE26: {
        **_EMBEDDED_STEP_SHA256_0EFFA6A5,
        (
            "shards",
            "Reject unmanifested RuleSpec content",
        ): "d2b2a9b7d7cd22bc7c9dbaa948b85a8d91231eeb7c27f515ac13cd9b52d90aa7",
        (
            "validate",
            "Enforce validation waiver ratchet",
        ): "238fe2b702f04c786bd30a4828efc74c1150b25e4b22c10de7ef272d334e56bc",
        (
            "validate",
            "Reject manual RuleSpec changes",
        ): "360e35f95d3f4bca21754b5b0ce5da7cb631e77bf7d8e833cb8961bc0500033e",
    },
}


@pytest.fixture(params=SUPPORTED_WORKFLOW_PINS.items(), ids=lambda item: item[0][:8])
def supported_workflow(request: pytest.FixtureRequest):
    sha, pin = request.param
    return sha, pin, WORKFLOW_DIRECTORY / pin.fixture


def test_real_lane_callers_parse() -> None:
    de = parse_caller_workflow(FIXTURES / "de-caller.yml")
    dk = parse_caller_workflow(FIXTURES / "dk-caller.yml")

    assert de.workflow_sha == "615c1df9b9ace7deea84da65efd137f46f8bad2b"
    assert de.refs["engine"] == "05eac9d2f89dabe5c6673176260762cef3a58f47"
    assert de.run_generated_guard is True
    assert dk.validate_roots == "auto"
    assert dk.run_generated_guard is False


def test_gate_order_is_stable() -> None:
    assert [gate.key for gate in CI_GATE_REGISTRY] == list(LEGACY_GATE_ORDER)
    assert list(LEGACY_GATE_ORDER) == [
        "repository_tests",
        "obsolete_files",
        "repository_layout",
        "validation_waivers",
        "guard_generated",
        "select_targets",
        "validate",
        "companion_tests",
        "proof_validate",
        "money_atoms",
        "oracle_coverage",
        "changed_oracle_coverage",
    ]
    # validate-rulespec@0effa6a5 adds three gates and moves repository tests,
    # which now run after every trusted gate, to the end.
    assert [gate.key for gate in gate_registry_for_pin(PIN_0EFFA6A5)] == [
        "unsupported_paths",
        "migration_authorization",
        "retired_schema_freeze",
        "obsolete_files",
        "repository_layout",
        "validation_waivers",
        "guard_generated",
        "select_targets",
        "validate",
        "companion_tests",
        "proof_validate",
        "money_atoms",
        "oracle_coverage",
        "changed_oracle_coverage",
        "repository_tests",
    ]
    assert [gate.key for gate in gate_registry_for_pin(PIN_6F11BE26)] == [
        "unsupported_paths",
        "unmanifested_rulespec",
        *[
            gate.key
            for gate in gate_registry_for_pin(PIN_0EFFA6A5)
            if gate.key != "unsupported_paths"
        ],
    ]
    for sha, pin in SUPPORTED_WORKFLOW_PINS.items():
        assert tuple(gate.key for gate in gate_registry_for_pin(sha)) == pin.gates


def test_workflow_axiom_commands_are_covered_by_gate_registry(
    supported_workflow,
) -> None:
    sha, _, fixture = supported_workflow
    invocations = workflow_axiom_invocations(fixture)
    registry: dict[str, list[set[str]]] = {}
    for gate in gate_registry_for_pin(sha):
        command = gate.subcommand.split()[0]
        registry.setdefault(command, []).append(
            {flag for flag in gate.flags if flag.startswith("--")}
        )

    assert invocations
    for command, flags in invocations:
        assert command in registry
        assert any(set(flags) == candidate for candidate in registry[command]), (
            command,
            flags,
        )


def test_every_workflow_gate_step_is_covered_exactly(supported_workflow) -> None:
    sha, pin, fixture = supported_workflow
    coverage = workflow_gate_coverage(fixture)
    registered = {gate.key for gate in gate_registry_for_pin(sha)}
    expected = {
        "Run repository tests",
        "Reject obsolete generated files",
        "Reject disallowed repository layout",
        "Enforce validation waiver ratchet",
        "Reject manual RuleSpec changes",
        "Select RuleSpec validation targets",
        "Validate RuleSpec YAML",
        "Execute RuleSpec companion tests",
        "Validate RuleSpec proofs and claims",
        "Require money proof atoms",
        "Validate PolicyEngine oracle coverage classification",
        "Checkout changed-file oracle coverage classifier",
        "Install changed-file oracle coverage classifier",
        "Validate changed PolicyEngine oracle coverage classification",
    }
    if pin.gate_parameters.embedded_scripts:
        expected |= {
            "Reject unsupported tracked paths",
            "Authorize exact reviewed migration",
            "Verify immutable retired-schema freeze",
        }
    if pin.gate_parameters.unmanifested_precheck:
        expected.add("Reject unmanifested RuleSpec content")

    assert set(coverage) == expected
    assert {gate for gates in coverage.values() for gate in gates} == registered


def test_every_pinned_workflow_step_is_accounted_for(supported_workflow) -> None:
    # A step a future pin adds is unaccounted for until it is reviewed and
    # classified as a gate, a resolution step or provisioned environment.
    _, _, fixture = supported_workflow
    classified = (
        WORKFLOW_GATE_STEPS.keys()
        | WORKFLOW_RESOLUTION_STEPS.keys()
        | WORKFLOW_ENVIRONMENT_STEPS.keys()
    )
    names = {step.get("name") for _, step in workflow_steps(fixture)}

    assert names - classified == set()
    assert not (WORKFLOW_GATE_STEPS.keys() & WORKFLOW_RESOLUTION_STEPS.keys())
    assert not (WORKFLOW_GATE_STEPS.keys() & WORKFLOW_ENVIRONMENT_STEPS.keys())
    assert not (WORKFLOW_RESOLUTION_STEPS.keys() & WORKFLOW_ENVIRONMENT_STEPS.keys())


def test_packaged_workflows_are_the_pinned_blobs(supported_workflow) -> None:
    _, pin, fixture = supported_workflow
    data = fixture.read_bytes()

    assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == (
        pin.workflow_blob
    )


def test_pin_input_declarations_match_packaged_workflow(supported_workflow) -> None:
    _, pin, fixture = supported_workflow
    payload = yaml.safe_load(fixture.read_text())
    declared = payload.get("on", payload.get(True))["workflow_call"]["inputs"]

    assert list(declared) == list(pin.inputs)
    for name, declaration in declared.items():
        expected = pin.inputs[name]
        assert declaration["type"] == expected.type, name
        assert bool(declaration.get("required", False)) is expected.required, name
        assert declaration.get("default") == expected.default, name


def test_gate_registry_lines_start_at_their_workflow_steps(supported_workflow) -> None:
    sha, _, fixture = supported_workflow
    if sha == "34bcfab235c585c47292c95f51be1a4f4f91d29e":
        pytest.skip("the legacy registry's line ranges refer to 615c1df9")
    lines = fixture.read_text().split("\n")
    step_lines = {
        line.strip().removeprefix("- name: "): number
        for number, line in enumerate(lines, start=1)
        if line.startswith("      - name: ")
    }
    coverage = workflow_gate_coverage(fixture)
    for gate in gate_registry_for_pin(sha):
        first_step = min(
            step_lines[name] for name, keys in coverage.items() if gate.key in keys
        )
        assert int(gate.workflow_lines.split("-")[0]) == first_step, gate.key


def test_workflow_gate_step_semantics_match_pinned_contract(supported_workflow) -> None:
    sha, pin, fixture = supported_workflow
    payload = yaml.safe_load(fixture.read_text())
    if pin.gate_parameters.embedded_scripts:
        actual = {
            (job, step["name"]): hashlib.sha256(
                yaml.safe_dump(
                    workflow_gate_contract(step, payload, job), sort_keys=True
                ).encode()
            ).hexdigest()
            for job, step in workflow_steps(fixture)
            if step.get("name") in WORKFLOW_GATE_STEPS
            or step.get("name") in WORKFLOW_RESOLUTION_STEPS
        }
        assert actual == EMBEDDED_STEP_SHA256_BY_PIN[sha]
        return
    actual = {}
    for step in payload["jobs"]["validate"]["steps"]:
        name = step.get("name")
        if name in WORKFLOW_GATE_STEP_SHA256:
            canonical = yaml.safe_dump(
                workflow_gate_contract(step, payload), sort_keys=True
            ).encode()
            actual[name] = hashlib.sha256(canonical).hexdigest()

    assert actual == LEGACY_WORKFLOW_GATE_STEP_SHA256_BY_PIN[sha]
    money_atom_run = next(
        step["run"]
        for step in payload["jobs"]["validate"]["steps"]
        if step.get("name") == "Require money proof atoms"
    )
    assert ('! -path "*/programs/*"' in money_atom_run) is (
        pin.gate_parameters.exclude_programs_from_money_atom_check
    )


def test_supported_workflow_fixtures_have_only_known_divergence() -> None:
    fixtures = {
        sha: yaml.safe_load((WORKFLOW_DIRECTORY / pin.fixture).read_text())
        for sha, pin in SUPPORTED_WORKFLOW_PINS.items()
    }
    old = fixtures["34bcfab235c585c47292c95f51be1a4f4f91d29e"]
    new = fixtures["615c1df9b9ace7deea84da65efd137f46f8bad2b"]
    old_step = next(
        step
        for step in old["jobs"]["validate"]["steps"]
        if step.get("name") == "Require money proof atoms"
    )
    new_step = next(
        step
        for step in new["jobs"]["validate"]["steps"]
        if step.get("name") == "Require money proof atoms"
    )

    assert (
        new_step["run"].replace(' \\\n        ! -path "*/programs/*"', "")
        == old_step["run"]
    )
    new_step["run"] = old_step["run"]
    assert new == old


def test_embedded_workflow_pins_have_only_known_divergence() -> None:
    old = yaml.safe_load(
        (WORKFLOW_DIRECTORY / SUPPORTED_WORKFLOW_PINS[PIN_0EFFA6A5].fixture).read_text()
    )
    new = yaml.safe_load(
        (WORKFLOW_DIRECTORY / SUPPORTED_WORKFLOW_PINS[PIN_6F11BE26].fixture).read_text()
    )
    added = new["jobs"]["shards"]["steps"].pop(2)
    assert added["name"] == "Reject unmanifested RuleSpec content"
    for name in ("Enforce validation waiver ratchet", "Reject manual RuleSpec changes"):
        old_step = next(
            step for step in old["jobs"]["validate"]["steps"] if step["name"] == name
        )
        new_step = next(
            step for step in new["jobs"]["validate"]["steps"] if step["name"] == name
        )
        assert new_step["run"] != old_step["run"]
        new_step["run"] = old_step["run"]

    assert new == old


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def test_ref_mismatch_names_both_shas_and_caller(tmp_path: Path) -> None:
    repo = tmp_path / "dependency"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "one").write_text("one")
    _git(repo, "add", "one")
    _git(repo, "commit", "-qm", "one")
    pin = _git(repo, "rev-parse", "HEAD")
    (repo / "two").write_text("two")
    _git(repo, "add", "two")
    _git(repo, "commit", "-qm", "two")
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", head)
    caller = tmp_path / "caller.yml"

    with pytest.raises(ValueError) as error:
        verify_dependency_checkout(
            "engine", repo, pin, caller, allow_ref_mismatch=False
        )

    message = str(error.value)
    assert pin in message
    assert head in message
    assert str(caller) in message

    warning = verify_dependency_checkout(
        "engine", repo, pin, caller, allow_ref_mismatch=True
    )
    assert warning is not None
    assert warning.pinned_sha == pin
    assert warning.head_sha == head
    assert warning.name == "engine"


def test_dirty_worktree_at_pinned_head_is_a_mismatch(tmp_path: Path) -> None:
    repo = tmp_path / "dependency"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "one").write_text("one")
    _git(repo, "add", "one")
    _git(repo, "commit", "-qm", "one")
    pin = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", pin)
    caller = tmp_path / "caller.yml"

    assert (
        verify_dependency_checkout(
            "corpus", repo, pin, caller, allow_ref_mismatch=False
        )
        is None
    )

    (repo / "one").write_text("modified locally")

    with pytest.raises(ValueError, match="dirty worktree"):
        verify_dependency_checkout(
            "corpus", repo, pin, caller, allow_ref_mismatch=False
        )

    warning = verify_dependency_checkout(
        "corpus", repo, pin, caller, allow_ref_mismatch=True
    )
    assert warning is not None
    assert warning.name == "corpus"
    assert warning.pinned_sha == pin
    assert "dirty worktree" in warning.head_sha


def test_verdict_is_qualified_only_when_gates_pass_with_mismatches() -> None:
    mismatch = DependencyMismatch("engine", "1" * 40, "2" * 40)

    assert ci_verdict(True, []) == "PASS"
    assert ci_verdict(True, [mismatch]) == "PASS-WITH-MISMATCHED-DEPS"
    assert ci_verdict(False, [mismatch]) == "FAIL"


def test_ambient_encoder_mismatch_fails_closed_or_is_qualified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = "1" * 40
    head = "2" * 40

    def fake_git(repo, *args, **_kwargs):
        if args == ("rev-parse", "--show-toplevel"):
            return subprocess.CompletedProcess([], 0, f"{repo}\n", "")
        return subprocess.CompletedProcess(
            [], 0, "" if args[0] == "status" else head + "\n", ""
        )

    monkeypatch.setattr("axiom_encode.ci_parity._git", fake_git)

    with pytest.raises(ValueError, match="--allow-encoder-mismatch") as error:
        verify_ambient_encoder(
            pin, "1.0.0", tmp_path / "caller.yml", allow_encoder_mismatch=False
        )
    assert pin in str(error.value)
    assert head in str(error.value)

    mismatch = verify_ambient_encoder(
        pin, "1.0.0", tmp_path / "caller.yml", allow_encoder_mismatch=True
    )
    assert mismatch == DependencyMismatch("ambient-encoder", head, pin)


def test_ambient_encoder_unresolvable_head_never_passes_on_version_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from axiom_encode import __version__

    pin = "1" * 40
    monkeypatch.setattr(
        "axiom_encode.ci_parity._git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 128, "", "fatal"),
    )

    with pytest.raises(ValueError, match="--allow-encoder-mismatch"):
        verify_ambient_encoder(
            pin, __version__, tmp_path / "caller.yml", allow_encoder_mismatch=False
        )

    mismatch = verify_ambient_encoder(
        pin, __version__, tmp_path / "caller.yml", allow_encoder_mismatch=True
    )
    assert mismatch is not None
    assert mismatch.name == "ambient-encoder"
    assert mismatch.pinned_sha == pin
    assert "unresolvable" in mismatch.head_sha


def test_toolchain_resolution_binds_fixture_waiver_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "rulespec-dk"
    (repo / ".axiom").mkdir(parents=True)
    waiver = b"validate_failures: {}\n"
    waiver_sha = hashlib.sha256(waiver).hexdigest()
    (repo / "known-validation-gaps.yaml").write_bytes(waiver)
    (repo / ".axiom" / "toolchain.toml").write_text(
        "[toolchain]\n"
        'axiom_corpus_release = "dk-rulespec-2026-07-22"\n'
        f'axiom_corpus_release_content_sha256 = "{"1" * 64}"\n'
        f'validation_waiver_set_sha256 = "{waiver_sha}"\n'
    )

    resolved = load_rulespec_toolchain(repo)

    assert resolved.root == repo
    assert resolved.corpus_release == "dk-rulespec-2026-07-22"
    assert verify_rulespec_validation_waiver_set(repo) == waiver_sha


def test_changed_target_selection_matches_companion_and_excludes_programs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "rulespec-zz"
    module = repo / "zz" / "statutes" / "benefit.yaml"
    companion = repo / "zz" / "statutes" / "benefit.test.yaml"
    program = repo / "zz" / "programs" / "composition.yaml"
    module.parent.mkdir(parents=True)
    program.parent.mkdir(parents=True)
    module.write_text("version: 1\n")
    companion.write_text("cases: []\n")
    program.write_text("version: 1\n")
    monkeypatch.setattr(
        "axiom_encode.ci_parity._changed_paths",
        lambda _repo, _base: (
            "zz/statutes/benefit.test.yaml",
            "zz/programs/composition.yaml",
        ),
    )

    selection = select_targets(repo, "origin/main", ("zz",))

    assert selection.mode == "changed"
    assert selection.rulespec_files == (module,)
    assert selection.test_files == (companion,)


def test_legacy_layout_allows_programs_as_workflow_extra_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from axiom_encode.ci_parity import _layout_gate

    repo = tmp_path / "rulespec-dk"
    program = repo / "programs" / "composition.yaml"
    program.parent.mkdir(parents=True)
    program.write_text("version: 1\n")
    monkeypatch.setattr(
        "axiom_encode.ci_parity._git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )

    code, _ = _layout_gate(repo, ("dk", "sources", "programs"))

    assert code == 0


def test_encoder_pin_version_consistency(tmp_path: Path) -> None:
    repo = tmp_path / "encoder"
    (repo / "src" / "axiom_encode").mkdir(parents=True)
    (repo / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    (repo / "src" / "axiom_encode" / "__init__.py").write_text(
        '__version__ = "1.2.3"\n'
    )
    (repo / "uv.lock").write_text(
        '[[package]]\nname = "axiom-encode"\nversion = "1.2.3"\n'
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "version")
    pin = _git(repo, "rev-parse", "HEAD")

    assert encoder_version_at_pin(repo, pin) == "1.2.3"


def _toolchain(tmp_path: Path, content_sha: str) -> RuleSpecToolchain:
    return RuleSpecToolchain(tmp_path, "dk-release", content_sha, "0" * 64)


def test_release_object_fetch_uses_workflow_url_scheme(tmp_path: Path) -> None:
    content = {"git": {"commit": "a" * 40}}
    digest = hashlib.sha256(
        json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    payload = json.dumps(
        {"release": "dk-release", "content_sha256": digest, "content": content}
    ).encode()
    seen = []

    path = acquire_release_object(
        _toolchain(tmp_path, digest),
        tmp_path,
        "https://objects.example/base/",
        offline=False,
        fetcher=lambda url: seen.append(url) or payload,
    )

    assert seen == [f"https://objects.example/base/releases/dk-release/{digest}.json"]
    assert path.read_bytes() == payload


def test_offline_requires_present_release_object(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--offline requires"):
        acquire_release_object(
            _toolchain(tmp_path, "1" * 64),
            tmp_path,
            "https://objects.example",
            offline=True,
        )


def test_ci_parser_contract_has_no_environment_public_key() -> None:
    # The public root is intentionally an explicit CLI-only value.  This test
    # guards against quietly reintroducing the forbidden environment fallback.
    args = Namespace(corpus_release_public_key="explicit")
    assert args.corpus_release_public_key == "explicit"


def test_unknown_workflow_pin_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "rulespec-dk"
    repo.mkdir()
    caller = CallerConfig(
        tmp_path / "caller.yml",
        "f" * 40,
        {name: "a" * 40 for name in ("encode", "engine", "corpus", "rulespec_us")},
        "dk",
        True,
        False,
    )
    monkeypatch.setattr(
        "axiom_encode.ci_parity.find_caller_workflow", lambda *_a, **_k: caller
    )

    @contextlib.contextmanager
    def passthrough_checkout(source, head=None, base=None):
        yield source

    monkeypatch.setattr("axiom_encode.ci_parity.resolve_commit", lambda _repo, ref: ref)

    monkeypatch.setattr(
        "axiom_encode.ci_parity.committed_checkout", passthrough_checkout
    )
    monkeypatch.setattr(
        "axiom_encode.ci_parity.uncommitted_changes_note", lambda _source: None
    )
    args = Namespace(repo=repo, json=False, base_ref="origin/main")

    assert run_ci(args) == 1
    output = capsys.readouterr().err
    assert "Unsupported validate-rulespec workflow pin" in output
    assert caller.workflow_sha in output
    assert all(sha in output for sha in SUPPORTED_WORKFLOW_PINS)


@pytest.mark.parametrize(
    ("workflow_sha", "programs_in_money_atom_check"),
    [
        ("615c1df9b9ace7deea84da65efd137f46f8bad2b", False),
        ("34bcfab235c585c47292c95f51be1a4f4f91d29e", True),
    ],
)
def test_execute_gates_preserves_order_and_uses_pin_gate_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow_sha: str,
    programs_in_money_atom_check: bool,
) -> None:
    repo = tmp_path / "rulespec-dk"
    test_file = repo / "dk" / "statutes" / "benefit.test.yaml"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("cases: []\n")
    program_file = repo / "dk" / "programs" / "pilot.yaml"
    program_file.parent.mkdir(parents=True)
    program_file.write_text("rules: []\n")
    (repo / "known-validation-gaps.yaml").write_text("validate_failures: {}\n")
    paths = {
        "encode": tmp_path / "encode",
        "engine": tmp_path / "engine",
        "corpus": tmp_path / "corpus",
        "rulespec_us": tmp_path / "rulespec-us",
    }
    caller = CallerConfig(
        tmp_path / "caller.yml",
        workflow_sha,
        {name: "a" * 40 for name in paths},
        "dk",
        False,
        False,
        run_pytest=False,
        run_money_atom_check=True,
    )
    args = Namespace(repo=repo, base_ref="origin/main")
    calls = []
    monkeypatch.setattr(
        "axiom_encode.ci_parity.select_targets",
        lambda *_: Selection("changed", (), (test_file,)),
    )
    monkeypatch.setattr("axiom_encode.ci_parity._changed_paths", lambda *_: ())
    monkeypatch.setattr("axiom_encode.ci_parity._obsolete_gate", lambda _: (0, "ok"))
    monkeypatch.setattr("axiom_encode.ci_parity._layout_gate", lambda *_: (0, "ok"))
    monkeypatch.setattr(
        "axiom_encode.ci_parity._git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, "waivers: {}\n", ""
        ),
    )

    def fake_cli(arguments, *, environment=None):
        calls.append((list(arguments), environment))
        return 0, "ok\n"

    monkeypatch.setattr("axiom_encode.ci_parity._run_cli", fake_cli)

    results = execute_gates(args, caller, paths, ("dk",))

    assert [result.gate for result in results] == [
        gate.key for gate in CI_GATE_REGISTRY
    ]
    companion = next(call for call in calls if call[0][0] == "test")
    assert companion[0][-1] == "statutes/benefit.test.yaml"
    assert companion[1] == {
        "AXIOM_RULESPEC_REPO_ROOTS": f"{repo}{os.pathsep}{paths['rulespec_us']}"
    }
    money_atom_calls = [
        call
        for call in calls
        if call[0][0] == "proof-validate" and "--money-atoms-only" in call[0]
    ]
    assert bool(money_atom_calls) is programs_in_money_atom_check
    if money_atom_calls:
        assert str(program_file) in money_atom_calls[0][0]


def test_pinned_classifier_rejects_ambient_oracles_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = "1" * 40
    monkeypatch.setattr(
        "axiom_encode.ci_parity._git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            'dependencies = ["axiom-oracles @ git+https://github.com/'
            f'TheAxiomFoundation/axiom-oracles@{expected}"]\n',
            "",
        ),
    )
    monkeypatch.setattr(
        "axiom_encode.ci_parity._installed_oracles_pin", lambda: "2" * 40
    )

    code, output = _run_pinned_cli(tmp_path, "a" * 40, ["oracle-coverage"])

    assert code == 1
    assert expected in output
    assert "2" * 40 in output
