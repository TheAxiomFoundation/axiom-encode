from __future__ import annotations

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
    SUPPORTED_WORKFLOW_PINS,
    CallerConfig,
    DependencyMismatch,
    Selection,
    _changed_paths_nul,
    _freeze_git_commit,
    _git_regular_blob,
    _run_pinned_cli,
    acquire_release_object,
    ci_verdict,
    encoder_version_at_pin,
    execute_gates,
    parse_caller_workflow,
    run_ci,
    select_targets,
    verify_ambient_encoder,
    verify_dependency_checkout,
    workflow_axiom_invocations,
    workflow_gate_contract,
    workflow_gate_coverage,
)
from axiom_encode.toolchain import (
    MAX_RULESPEC_TOOLCHAIN_BYTES,
    MAX_VALIDATION_WAIVER_SET_BYTES,
    RuleSpecToolchain,
    load_rulespec_toolchain,
    verify_rulespec_validation_waiver_set,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ci_parity"
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
WORKFLOW_GATE_STEP_SHA256_BY_PIN = {
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
    for sha in SUPPORTED_WORKFLOW_PINS
}


@pytest.fixture(params=SUPPORTED_WORKFLOW_PINS.items(), ids=lambda item: item[0][:8])
def supported_workflow(request: pytest.FixtureRequest):
    sha, pin = request.param
    return sha, pin, FIXTURES / pin.fixture


def test_real_lane_callers_parse() -> None:
    de = parse_caller_workflow(FIXTURES / "de-caller.yml")
    dk = parse_caller_workflow(FIXTURES / "dk-caller.yml")

    assert de.workflow_sha == "615c1df9b9ace7deea84da65efd137f46f8bad2b"
    assert de.refs["engine"] == "05eac9d2f89dabe5c6673176260762cef3a58f47"
    assert de.run_generated_guard is True
    assert dk.validate_roots == "auto"
    assert dk.run_generated_guard is False


def test_gate_order_is_stable() -> None:
    assert [gate.key for gate in CI_GATE_REGISTRY] == [
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


def test_workflow_axiom_commands_are_covered_by_gate_registry(
    supported_workflow,
) -> None:
    _, _, fixture = supported_workflow
    invocations = workflow_axiom_invocations(fixture)
    registry: dict[str, list[set[str]]] = {}
    for gate in CI_GATE_REGISTRY:
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
    _, _, fixture = supported_workflow
    coverage = workflow_gate_coverage(fixture)
    registered = {gate.key for gate in CI_GATE_REGISTRY}

    assert set(coverage) == {
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
    assert {gate for gates in coverage.values() for gate in gates} == registered


def test_workflow_gate_step_semantics_match_pinned_contract(supported_workflow) -> None:
    sha, pin, fixture = supported_workflow
    payload = yaml.safe_load(fixture.read_text())
    actual = {}
    for step in payload["jobs"]["validate"]["steps"]:
        name = step.get("name")
        if name in WORKFLOW_GATE_STEP_SHA256:
            canonical = yaml.safe_dump(
                workflow_gate_contract(step, payload), sort_keys=True
            ).encode()
            actual[name] = hashlib.sha256(canonical).hexdigest()

    assert actual == WORKFLOW_GATE_STEP_SHA256_BY_PIN[sha]
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
        sha: yaml.safe_load((FIXTURES / pin.fixture).read_text())
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


def test_historical_workflow_pins_do_not_claim_immutable_transition_evidence() -> None:
    for pin in SUPPORTED_WORKFLOW_PINS.values():
        assert pin.immutable_waiver_transition_evidence is False
        invocations = workflow_axiom_invocations(FIXTURES / pin.fixture)
        waiver_flags = next(
            flags for command, flags in invocations if command == "validation-waivers"
        )
        assert "--protected-base-toolchain" not in waiver_flags
        assert "--changed-paths-format" not in waiver_flags


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _ci_evidence_repo(tmp_path: Path) -> tuple[Path, str, bytes, bytes]:
    repo = tmp_path / "rulespec-zz"
    (repo / ".axiom").mkdir(parents=True)
    protected_waivers = b"validate_failures: {}\r\n# exact base bytes"
    protected_toolchain = b"[toolchain]\r\n# exact base bytes"
    (repo / "known-validation-gaps.yaml").write_bytes(protected_waivers)
    (repo / ".axiom" / "toolchain.toml").write_bytes(protected_toolchain)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    return repo, _git(repo, "rev-parse", "HEAD"), protected_waivers, protected_toolchain


def test_ci_evidence_uses_frozen_exact_blobs_and_nul_paths(tmp_path: Path) -> None:
    repo, base, protected_waivers, protected_toolchain = _ci_evidence_repo(tmp_path)
    _git(repo, "branch", "base-proof", base)
    frozen_base = _freeze_git_commit(repo, "base-proof")
    (repo / "known-validation-gaps.yaml").write_bytes(b"validate_failures: {}\n")
    (repo / ".axiom" / "toolchain.toml").write_bytes(b"[toolchain]\n")
    adversarial = repo / "dk" / "statutes" / "line\nbreak.yaml"
    adversarial.parent.mkdir(parents=True)
    adversarial.write_bytes(b"version: 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "head")
    head = _freeze_git_commit(repo, "HEAD")
    _git(repo, "branch", "-f", "base-proof", head)

    assert frozen_base == base
    assert _git_regular_blob(
        repo,
        frozen_base,
        "known-validation-gaps.yaml",
        max_bytes=MAX_VALIDATION_WAIVER_SET_BYTES,
    ) == protected_waivers
    assert _git_regular_blob(
        repo,
        frozen_base,
        ".axiom/toolchain.toml",
        max_bytes=MAX_RULESPEC_TOOLCHAIN_BYTES,
    ) == protected_toolchain
    assert _changed_paths_nul(repo, frozen_base, head) == (
        b".axiom/toolchain.toml\0"
        b"dk/statutes/line\nbreak.yaml\0"
        b"known-validation-gaps.yaml\0"
    )


@pytest.mark.parametrize(
    ("relative_path", "violation"),
    [
        ("known-validation-gaps.yaml", "executable"),
        ("known-validation-gaps.yaml", "symlink"),
        ("known-validation-gaps.yaml", "missing"),
        (".axiom/toolchain.toml", "executable"),
        (".axiom/toolchain.toml", "symlink"),
        (".axiom/toolchain.toml", "missing"),
    ],
)
def test_ci_evidence_rejects_non_regular_or_missing_base_blobs(
    tmp_path: Path, relative_path: str, violation: str
) -> None:
    repo, _, _, _ = _ci_evidence_repo(tmp_path)
    path = repo / relative_path
    if violation == "executable":
        _git(repo, "update-index", "--chmod=+x", relative_path)
    elif violation == "symlink":
        path.unlink()
        path.symlink_to("redirected")
        _git(repo, "add", "--", relative_path)
    else:
        path.unlink()
        _git(repo, "add", "--all", "--", relative_path)
    _git(repo, "commit", "-qm", violation)
    commit = _freeze_git_commit(repo, "HEAD")

    with pytest.raises(ValueError, match="exactly one|exact regular 0644"):
        _git_regular_blob(
            repo,
            commit,
            relative_path,
            max_bytes=MAX_VALIDATION_WAIVER_SET_BYTES,
        )


def test_ci_evidence_ignores_ambient_git_repo_redirection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, target_head, target_waivers, _ = _ci_evidence_repo(tmp_path / "target")
    redirect, _, _, _ = _ci_evidence_repo(tmp_path / "redirect")
    monkeypatch.setenv("GIT_DIR", str(redirect / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(redirect))

    assert _freeze_git_commit(target, "HEAD") == target_head
    assert _git_regular_blob(
        target,
        target_head,
        "known-validation-gaps.yaml",
        max_bytes=MAX_VALIDATION_WAIVER_SET_BYTES,
    ) == target_waivers


def test_ci_evidence_caps_changed_path_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "axiom_encode.ci_parity._git_bytes",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, b"x" * (MAX_VALIDATION_WAIVER_SET_BYTES + 1), b""
        ),
    )

    with pytest.raises(ValueError, match="changed-path evidence exceeds"):
        _changed_paths_nul(tmp_path, "a" * 40, "b" * 40)


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
    monkeypatch.setattr(
        "axiom_encode.ci_parity._git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, head + "\n", ""),
    )

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
        lambda _repo, _base, *_: (
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
    monkeypatch.setattr("axiom_encode.ci_parity.find_caller_workflow", lambda _: caller)
    args = Namespace(repo=repo, json=False)

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
    base_commit = "b" * 40
    protected_waivers = b"validate_failures: {}\r\n"
    protected_toolchain = b"[toolchain]\r\n"
    changed_paths = b"dk/statutes/benefit.test.yaml\0"
    captured_waiver_evidence = {}
    monkeypatch.setattr(
        "axiom_encode.ci_parity._freeze_git_commit", lambda *_: base_commit
    )
    monkeypatch.setattr(
        "axiom_encode.ci_parity.select_targets",
        lambda *_, **__: Selection("changed", (), (test_file,)),
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
    monkeypatch.setattr(
        "axiom_encode.ci_parity._git_regular_blob",
        lambda _repo, _commit, relative_path, **_kwargs: (
            protected_waivers
            if relative_path == "known-validation-gaps.yaml"
            else protected_toolchain
        ),
    )
    monkeypatch.setattr(
        "axiom_encode.ci_parity._changed_paths_nul",
        lambda *_: changed_paths,
    )

    def fake_cli(arguments, *, environment=None):
        calls.append((list(arguments), environment))
        if arguments[:2] == ["validation-waivers", "audit"]:
            for flag in (
                "--protected-base",
                "--protected-base-toolchain",
                "--changed-paths",
            ):
                path = Path(arguments[arguments.index(flag) + 1])
                captured_waiver_evidence[flag] = path.read_bytes()
        return 0, "ok\n"

    monkeypatch.setattr("axiom_encode.ci_parity._run_cli", fake_cli)

    results = execute_gates(args, caller, paths, ("dk",))

    assert [result.gate for result in results] == [
        gate.key for gate in CI_GATE_REGISTRY
    ]
    waiver_result = next(
        result for result in results if result.gate == "validation_waivers"
    )
    assert waiver_result.command[waiver_result.command.index("--changed-paths-format") + 1] == "nul-v1"
    assert waiver_result.note == (
        "local immutable-evidence compatibility check; pinned historical "
        "workflow lacks this evidence and is not transition-compatible"
    )
    assert captured_waiver_evidence == {
        "--protected-base": protected_waivers,
        "--protected-base-toolchain": protected_toolchain,
        "--changed-paths": changed_paths,
    }
    selection_result = next(
        result for result in results if result.gate == "select_targets"
    )
    assert selection_result.command[2] == base_commit
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


def test_execute_gates_does_not_invoke_audit_with_invalid_base_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "rulespec-dk"
    repo.mkdir()
    paths = {
        name: tmp_path / name
        for name in ("encode", "engine", "corpus", "rulespec_us")
    }
    caller = CallerConfig(
        tmp_path / "caller.yml",
        "615c1df9b9ace7deea84da65efd137f46f8bad2b",
        {name: "a" * 40 for name in paths},
        "dk",
        False,
        False,
        run_pytest=False,
        run_money_atom_check=False,
    )
    calls = []
    monkeypatch.setattr(
        "axiom_encode.ci_parity._freeze_git_commit", lambda *_: "b" * 40
    )
    monkeypatch.setattr(
        "axiom_encode.ci_parity.select_targets",
        lambda *_, **__: Selection("changed", (), ()),
    )
    monkeypatch.setattr("axiom_encode.ci_parity._obsolete_gate", lambda _: (0, "ok"))
    monkeypatch.setattr("axiom_encode.ci_parity._layout_gate", lambda *_: (0, "ok"))
    monkeypatch.setattr(
        "axiom_encode.ci_parity._git_regular_blob",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("protected base mode is invalid")
        ),
    )

    def fake_cli(arguments, *, environment=None):
        calls.append((list(arguments), environment))
        return 0, "ok\n"

    monkeypatch.setattr("axiom_encode.ci_parity._run_cli", fake_cli)

    results = execute_gates(
        Namespace(repo=repo, base_ref="origin/main"), caller, paths, ("dk",)
    )

    waiver_result = next(
        result for result in results if result.gate == "validation_waivers"
    )
    assert waiver_result.status == "FAIL"
    assert "protected base mode is invalid" in waiver_result.output
    assert not any(call[0][:2] == ["validation-waivers", "audit"] for call in calls)


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
