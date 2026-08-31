"""Executed projected-state validation for legacy cleanup receipts."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from base64 import b64decode
from dataclasses import replace
from pathlib import Path

import pytest

import axiom_encode.legacy_cleanup_validation as cleanup_validation
from axiom_encode.legacy_cleanup import LEGACY_CLEANUP_VALIDATION_CHECKS
from axiom_encode.legacy_cleanup_git import plan_legacy_cleanup_base
from axiom_encode.legacy_cleanup_validation import (
    LEGACY_CLEANUP_MAX_COMMAND_OUTPUT_BYTES,
    LEGACY_CLEANUP_VALIDATION_SCHEMA,
    LegacyCleanupValidationError,
    ValidationCommandResult,
    _internal_main,
    execute_projected_validation,
    validation_execution_issues,
    verification_only_axiom_encode_command,
)

DELETED_PRIMARY = Path("be/statutes/deleted_aggregation.yaml")
DELETED_COMPANION = Path("be/statutes/deleted_aggregation.test.yaml")
SURVIVING_PRIMARY = Path("be/statutes/surviving_rule.yaml")
SURVIVING_COMPANION = Path("be/statutes/surviving_rule.test.yaml")
WAIVER_BYTES = b"version: 1\nwaivers: []\n"


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_git_commands_disable_hostile_local_checkout_accelerators(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(cleanup_validation.subprocess, "run", fake_run)

    cleanup_validation._git_bytes(Path("/fixture"), "status", "--porcelain=v2")

    assert captured["command"] == [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.autocrlf=false",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "core.sparseCheckout=false",
        "-C",
        "/fixture",
        "status",
        "--porcelain=v2",
    ]


def _init_git_repo(path: Path, *, origin: str | None = None) -> Path:
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.email", "cleanup-validation@example.com")
    _git(path, "config", "user.name", "Cleanup Validation Test")
    if origin is not None:
        _git(path, "remote", "add", "origin", origin)
    return path


def _commit_all(repo: Path, message: str = "fixture") -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _write_group(repo: Path, primary: Path, *, companion: bool = True) -> None:
    target = repo / primary
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "module:\n"
        f"  name: {primary.stem}\n"
        "rules:\n"
        "  - name: amount\n"
        "    type: parameter\n"
        "    value: 1\n",
        encoding="utf-8",
    )
    if companion:
        test_path = target.with_name(f"{target.stem}.test.yaml")
        test_path.write_text(
            "tests:\n"
            "  - name: amount\n"
            "    inputs: {}\n"
            "    expected:\n"
            "      amount: 1\n",
            encoding="utf-8",
        )


def _write_contract(repo: Path) -> None:
    toolchain = repo / ".axiom/toolchain.toml"
    toolchain.parent.mkdir(parents=True, exist_ok=True)
    toolchain.write_text(
        "[toolchain]\n"
        'axiom_corpus_release = "cleanup-validation-test"\n'
        f'axiom_corpus_release_content_sha256 = "{"c" * 64}"\n'
        "validation_waiver_set_sha256 = "
        f'"{hashlib.sha256(WAIVER_BYTES).hexdigest()}"\n',
        encoding="utf-8",
    )
    (repo / "known-validation-gaps.yaml").write_bytes(WAIVER_BYTES)


def _dependency_repo(tmp_path: Path, name: str) -> Path:
    repo = _init_git_repo(tmp_path / name)
    (repo / "README.md").write_text(f"# {name}\n", encoding="utf-8")
    _commit_all(repo)
    return repo


def _fixture(
    tmp_path: Path,
    *,
    survivor: bool = True,
    surviving_companion: bool = True,
    repository_tests: bool = True,
) -> tuple[Path, object, Path, Path]:
    repo = _init_git_repo(
        tmp_path / "rulespec-be",
        origin="https://github.com/TheAxiomFoundation/rulespec-be.git",
    )
    _write_contract(repo)
    _write_group(repo, DELETED_PRIMARY)
    if survivor:
        _write_group(
            repo,
            SURVIVING_PRIMARY,
            companion=surviving_companion,
        )
    if repository_tests:
        tests = repo / "tests"
        tests.mkdir()
        (tests / "test_repository.py").write_text(
            "def test_repository_fixture():\n    assert True\n",
            encoding="utf-8",
        )
    base = _commit_all(repo, "protected base")
    plan = plan_legacy_cleanup_base(
        repo,
        base_ref=base,
        primary_paths=[DELETED_PRIMARY],
        require_clean_checkout=True,
    )
    corpus = _dependency_repo(tmp_path, "axiom-corpus")
    engine = _dependency_repo(tmp_path, "axiom-rules-engine")
    return repo, plan, corpus, engine


class _PassingRunner:
    def __init__(self, *, run_internal: bool = False) -> None:
        self.calls: list[tuple[tuple[str, ...], Path]] = []
        self.run_internal = run_internal

    def __call__(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        environment,
        output_limit: int,
    ) -> ValidationCommandResult:
        self.calls.append((command, cwd))
        assert not (cwd / DELETED_PRIMARY).exists()
        assert not (cwd / DELETED_COMPANION).exists()
        direct_internal = False
        if "axiom_encode.legacy_cleanup_validation" in command:
            module_index = command.index("axiom_encode.legacy_cleanup_validation")
            direct_internal = command[module_index + 1] in {
                "repository-tests-presence",
                "repository-layout",
                "metadata-reference-closure-index-scan",
            }
        real_pytest = "pytest" in command and "axiom-encode-verification" not in command
        if self.run_internal and (direct_internal or real_pytest):
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=dict(environment),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            return ValidationCommandResult(completed.returncode, completed.stdout)
        return ValidationCommandResult(0, f"validated {cwd}\n".encode())


def _execute(repo, plan, corpus, engine, runner):
    return execute_projected_validation(
        repo,
        plan,
        corpus_checkout=corpus,
        rules_engine_checkout=engine,
        axiom_encode_command=verification_only_axiom_encode_command((b"k" * 32,)),
        command_runner=runner,
    )


def test_real_projected_tree_runs_exact_matrix_and_returns_bounded_evidence(
    tmp_path,
):
    repo, plan, corpus, engine = _fixture(tmp_path)
    runner = _PassingRunner(run_internal=True)

    evidence = _execute(repo, plan, corpus, engine, runner)

    assert evidence == {
        **evidence,
        "schema": LEGACY_CLEANUP_VALIDATION_SCHEMA,
        "status": "passed",
        "engine_execution": True,
        "projected_post_deletion_tree": plan.projected_post_deletion_tree,
    }
    assert [check["name"] for check in evidence["checks"]] == list(
        LEGACY_CLEANUP_VALIDATION_CHECKS
    )
    assert all(set(check) == {
        "name",
        "command",
        "target_count",
        "target_list_sha256",
        "exit_code",
        "output_sha256",
    } for check in evidence["checks"])
    assert not validation_execution_issues(
        evidence,
        expected_projected_tree=plan.projected_post_deletion_tree,
    )
    assert len(runner.calls) == len(LEGACY_CLEANUP_VALIDATION_CHECKS)
    assert "tests" in runner.calls[0][0]
    assert str(SURVIVING_PRIMARY) not in runner.calls[0][0]
    assert all(
        len(command) <= cleanup_validation.LEGACY_CLEANUP_MAX_COMMAND_TOKENS
        for command, _cwd in runner.calls
    )
    oracle_command = next(
        command for command, _cwd in runner.calls if "oracle-coverage" in command
    )
    assert {
        "--fail-on-unmapped",
        "--fail-on-untested-comparable",
        "--fail-on-incomplete-comparable",
        "--fail-on-stale-pending",
        "--fail-on-empty",
    }.issubset(oracle_command)
    assert sum(
        "--money-atoms-only" in command for command, _cwd in runner.calls
    ) == 1
    engine_head = _git(engine, "rev-parse", "HEAD")
    engine_commands = [
        command
        for command, _cwd in runner.calls
        if "--axiom-rules-engine-path" in command
    ]
    assert len(engine_commands) == 3
    assert all(
        command[command.index("--axiom-rules-engine-ref") + 1] == engine_head
        for command in engine_commands
    )
    evidence_by_name = {check["name"]: check for check in evidence["checks"]}
    for name in (
        "validation-waivers",
        "remaining-rulespec-validation",
        "remaining-companion-tests",
    ):
        portable = evidence_by_name[name]["command"]
        assert portable[portable.index("--axiom-rules-engine-ref") + 1] == (
            "{axiom-rules-engine-commit}"
        )
    rendered = repr(evidence)
    assert str(tmp_path) not in rendered
    assert all(
        not token.startswith("/")
        for check in evidence["checks"]
        for token in check["command"]
    )


def test_evidence_is_deterministic_across_private_snapshot_paths(tmp_path):
    repo, plan, corpus, engine = _fixture(tmp_path)

    first = _execute(repo, plan, corpus, engine, _PassingRunner())
    second = _execute(repo, plan, corpus, engine, _PassingRunner())

    assert first == second


def test_real_pytest_output_normalization_removes_only_presentation_noise(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_pass.py").write_text(
        "def test_pass():\n    assert True\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
        }
    )
    command = (
        sys.executable,
        "-P",
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests",
    )

    first = cleanup_validation._bounded_subprocess_runner(
        command,
        cwd=tmp_path,
        environment=environment,
        output_limit=4096,
    )
    second = cleanup_validation._bounded_subprocess_runner(
        command,
        cwd=tmp_path,
        environment=environment,
        output_limit=4096,
    )

    assert first.exit_code == second.exit_code == 0
    assert cleanup_validation._normalized_output(
        first.output,
        replacements=(),
    ) == cleanup_validation._normalized_output(second.output, replacements=())
    assert cleanup_validation._normalized_output(
        b"\x1b[32m1 passed in 0.01s\x1b[0m\n",
        replacements=(),
    ) == b"1 passed in {elapsed}\n"


def test_verification_only_command_carries_canonical_public_keyring():
    first = b"a" * 32
    second = b"b" * 32

    command = verification_only_axiom_encode_command((second, first))

    assert "axiom-encode-verification" in command
    encoded = [
        command[index + 1]
        for index, token in enumerate(command)
        if token == "--corpus-release-public-key"
    ]
    assert encoded == sorted(encoded)
    assert {b64decode(value) for value in encoded} == {first, second}
    assert not any("BROKER" in token or "PRIVATE" in token for token in command)
    assert command[-1] == "--"

    with pytest.raises(LegacyCleanupValidationError, match="1..16"):
        verification_only_axiom_encode_command(())
    with pytest.raises(LegacyCleanupValidationError, match="duplicate"):
        verification_only_axiom_encode_command((first, first))


def test_verification_only_wrapper_dispatches_inside_keyring_context(monkeypatch):
    import axiom_encode.entrypoint as entrypoint
    import axiom_encode.toolchain as toolchain

    public_key = b"v" * 32
    observed = {}

    def fake_main():
        observed["argv"] = sys.argv[:]
        observed["keys"] = toolchain._LOCAL_CORPUS_RELEASE_PUBLIC_KEYS.get()
        return 0

    monkeypatch.setattr(entrypoint, "main", fake_main)
    prefix = verification_only_axiom_encode_command((public_key,))
    wrapper_arguments = prefix[prefix.index("axiom-encode-verification") :]

    assert _internal_main(
        (*wrapper_arguments, "inventory", "--root", "rulespec-be")
    ) == 0
    assert observed["argv"] == [
        "axiom-encode",
        "inventory",
        "--root",
        "rulespec-be",
    ]
    assert tuple(b64decode(key) for key in observed["keys"]) == (public_key,)
    assert toolchain._LOCAL_CORPUS_RELEASE_PUBLIC_KEYS.get() is None


def test_verification_wrapper_expands_large_target_list_below_argv_limit(
    tmp_path,
    monkeypatch,
):
    import axiom_encode.entrypoint as entrypoint

    targets = tuple(
        Path(f"be/statutes/rule_{index:04d}.yaml") for index in range(300)
    )
    target_list = tmp_path / "targets.txt"
    target_list.write_bytes(cleanup_validation._target_list_bytes(targets))
    observed = {}

    def fake_main():
        observed["argv"] = sys.argv[:]
        return 0

    monkeypatch.setattr(entrypoint, "main", fake_main)
    prefix = verification_only_axiom_encode_command((b"v" * 32,))
    command = cleanup_validation._verification_command_with_targets(
        prefix,
        target_list=target_list,
        target_prefix=Path("/projected"),
    )
    wrapper_arguments = command[command.index("axiom-encode-verification") :]

    assert len(command) < cleanup_validation.LEGACY_CLEANUP_MAX_COMMAND_TOKENS
    assert _internal_main((*wrapper_arguments, "validate", "--skip-reviewers")) == 0
    assert observed["argv"][:3] == [
        "axiom-encode",
        "validate",
        "--skip-reviewers",
    ]
    assert observed["argv"][3:] == [
        str(Path("/projected") / target) for target in targets
    ]


@pytest.mark.parametrize(
    ("subcommand", "handler_name", "target_attribute", "arguments", "strip"),
    [
        (
            "validate",
            "cmd_validate",
            "files",
            (
                "--skip-reviewers",
                "--corpus-path",
                "/corpus",
                "--axiom-rules-engine-path",
                "/engine",
                "--axiom-rules-engine-ref",
                "a" * 40,
            ),
            0,
        ),
        (
            "proof-validate",
            "cmd_proof_validate",
            "files",
            ("--corpus-path", "/corpus"),
            0,
        ),
        (
            "test",
            "cmd_test",
            "paths",
            (
                "--root",
                "/projected/be",
                "--axiom-rules-engine-path",
                "/engine",
                "--axiom-rules-engine-ref",
                "a" * 40,
            ),
            1,
        ),
    ],
)
def test_target_list_expansion_is_accepted_by_real_cli_parser(
    tmp_path,
    monkeypatch,
    subcommand,
    handler_name,
    target_attribute,
    arguments,
    strip,
):
    import axiom_encode.cli as cli
    import axiom_encode.entrypoint as entrypoint

    target = Path("be/statutes/example.yaml")
    target_list = tmp_path / "targets.txt"
    target_list.write_bytes(cleanup_validation._target_list_bytes((target,)))
    observed = {}

    def fake_handler(args):
        observed["targets"] = getattr(args, target_attribute)

    monkeypatch.setattr(cli, handler_name, fake_handler)
    monkeypatch.setattr(entrypoint, "main", cli.main)
    prefix = verification_only_axiom_encode_command((b"v" * 32,))
    command = cleanup_validation._verification_command_with_targets(
        prefix,
        target_list=target_list,
        target_prefix=Path("/projected") if not strip else None,
        target_strip_components=strip,
    )
    wrapper_arguments = command[command.index("axiom-encode-verification") :]

    assert _internal_main((*wrapper_arguments, subcommand, *arguments)) == 0
    expected = Path("statutes/example.yaml") if strip else Path(
        "/projected/be/statutes/example.yaml"
    )
    assert observed["targets"] == [expected]


def test_validation_waiver_engine_ref_is_accepted_by_real_cli_parser(monkeypatch):
    import axiom_encode.cli as cli
    import axiom_encode.entrypoint as entrypoint

    expected_ref = "a" * 40
    observed = {}

    def fake_handler(args):
        observed["ref"] = args.axiom_rules_engine_ref

    monkeypatch.setattr(cli, "cmd_validation_waivers", fake_handler)
    monkeypatch.setattr(entrypoint, "main", cli.main)
    prefix = verification_only_axiom_encode_command((b"v" * 32,))
    wrapper_arguments = prefix[prefix.index("axiom-encode-verification") :]

    assert _internal_main(
        (
            *wrapper_arguments,
            "validation-waivers",
            "audit",
            "--root",
            "/projected",
            "--corpus-path",
            "/corpus",
            "--protected-base",
            "/protected-waivers",
            "--changed-paths",
            "/changed-paths",
            "--axiom-rules-engine-path",
            "/engine",
            "--axiom-rules-engine-ref",
            expected_ref,
        )
    ) == 0
    assert observed["ref"] == expected_ref


def test_absent_repository_tests_execute_the_presence_check(tmp_path):
    repo, plan, corpus, engine = _fixture(tmp_path, repository_tests=False)
    runner = _PassingRunner(run_internal=True)

    evidence = _execute(repo, plan, corpus, engine, runner)

    assert evidence["checks"][0]["target_count"] == 0
    assert "repository-tests-presence" in runner.calls[0][0]


def test_nonzero_required_command_cannot_fabricate_pass(tmp_path):
    repo, plan, corpus, engine = _fixture(tmp_path)

    class FailingRunner(_PassingRunner):
        def __call__(self, command, **kwargs):
            if "oracle-coverage" in command:
                return ValidationCommandResult(9, b"oracle coverage failed\n")
            return super().__call__(command, **kwargs)

    with pytest.raises(
        LegacyCleanupValidationError,
        match=r"required validation check failed \(oracle-coverage, exit 9\)",
    ):
        _execute(repo, plan, corpus, engine, FailingRunner())


def test_runner_must_return_executed_result(tmp_path):
    repo, plan, corpus, engine = _fixture(tmp_path)

    def missing_result(command, **kwargs):
        del command, kwargs
        return None

    with pytest.raises(LegacyCleanupValidationError, match="no executed result"):
        _execute(repo, plan, corpus, engine, missing_result)


def test_projected_tree_mismatch_fails_before_validation_commands(tmp_path):
    repo, plan, corpus, engine = _fixture(tmp_path)
    runner = _PassingRunner()
    wrong = replace(plan, projected_post_deletion_tree="0" * len(plan.base_tree))

    with pytest.raises(LegacyCleanupValidationError, match="projected post-deletion"):
        _execute(repo, wrong, corpus, engine, runner)

    assert not runner.calls


@pytest.mark.parametrize("attack", ["reappear", "mutate"])
def test_command_cannot_reappear_deletion_or_mutate_projection(tmp_path, attack):
    repo, plan, corpus, engine = _fixture(tmp_path)

    class MutatingRunner(_PassingRunner):
        def __call__(self, command, *, cwd, **kwargs):
            if attack == "reappear":
                target = cwd / DELETED_PRIMARY
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("reappeared: true\n", encoding="utf-8")
            else:
                (cwd / SURVIVING_PRIMARY).write_text(
                    "module:\n  name: mutated\n",
                    encoding="utf-8",
                )
            return ValidationCommandResult(0, b"claimed pass\n")

    match = "reappeared" if attack == "reappear" else "mutated"
    with pytest.raises(LegacyCleanupValidationError, match=match):
        _execute(repo, plan, corpus, engine, MutatingRunner())


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_command_cannot_hide_survivor_mutation_with_index_flags(
    tmp_path,
    index_flag,
):
    repo, plan, corpus, engine = _fixture(tmp_path)

    class HiddenMutationRunner(_PassingRunner):
        def __call__(self, command, *, cwd, **kwargs):
            _git(cwd, "update-index", index_flag, "--", SURVIVING_PRIMARY.as_posix())
            (cwd / SURVIVING_PRIMARY).write_text(
                "module:\n  name: hidden_mutation\n",
                encoding="utf-8",
            )
            return ValidationCommandResult(0, b"claimed pass\n")

    with pytest.raises(LegacyCleanupValidationError, match="prohibited Git index flag"):
        _execute(repo, plan, corpus, engine, HiddenMutationRunner())


def test_external_checkout_hidden_mutation_is_rejected_at_admission(tmp_path):
    repo, plan, corpus, engine = _fixture(tmp_path)
    _git(corpus, "update-index", "--assume-unchanged", "--", "README.md")
    (corpus / "README.md").write_text("hidden mutation\n", encoding="utf-8")

    with pytest.raises(LegacyCleanupValidationError, match="prohibited Git index flag"):
        _execute(repo, plan, corpus, engine, _PassingRunner())


def test_command_cannot_mutate_exact_target_transport(tmp_path):
    repo, plan, corpus, engine = _fixture(tmp_path)

    class TargetListMutatingRunner(_PassingRunner):
        def __call__(self, command, **kwargs):
            if "--target-list" in command:
                index = command.index("--target-list")
                Path(command[index + 1]).write_text(
                    "be/statutes/not-authorized.yaml\n",
                    encoding="utf-8",
                )
            return ValidationCommandResult(0, b"claimed pass\n")

    with pytest.raises(LegacyCleanupValidationError, match="support input mutated"):
        _execute(repo, plan, corpus, engine, TargetListMutatingRunner())


def test_command_output_limit_is_fail_closed(tmp_path):
    repo, plan, corpus, engine = _fixture(tmp_path)

    def excessive_output(command, **kwargs):
        del command, kwargs
        return ValidationCommandResult(
            0,
            b"x" * (LEGACY_CLEANUP_MAX_COMMAND_OUTPUT_BYTES + 1),
        )

    with pytest.raises(LegacyCleanupValidationError, match="output exceeds"):
        _execute(repo, plan, corpus, engine, excessive_output)


def test_bounded_runner_kills_successful_leader_background_descendants(tmp_path):
    delayed_output = tmp_path / "late-mutation.txt"
    child = (
        "import pathlib,time; "
        "time.sleep(0.25); "
        f"pathlib.Path({str(delayed_output)!r}).write_text('mutated')"
    )
    leader = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}])"
    )

    result = cleanup_validation._bounded_subprocess_runner(
        (sys.executable, "-c", leader),
        cwd=tmp_path,
        environment=os.environ,
        output_limit=4096,
    )
    time.sleep(0.4)

    assert result.exit_code == 0
    assert not delayed_output.exists()


def test_deleted_reference_scan_uses_token_boundaries():
    patterns = cleanup_validation._reference_variants((DELETED_PRIMARY,))

    assert cleanup_validation._contains_deleted_reference(
        b'"be/statutes/deleted_aggregation.yaml"',
        patterns,
    )
    assert cleanup_validation._contains_deleted_reference(
        b'"be:statutes/deleted_aggregation#amount"',
        patterns,
    )
    assert cleanup_validation._contains_deleted_reference(
        b'"/abs/project/be/statutes/deleted_aggregation.yaml"',
        patterns,
    )
    assert not cleanup_validation._contains_deleted_reference(
        b'"be/statutes/deleted_aggregation.yaml.backup"',
        patterns,
    )
    assert not cleanup_validation._contains_deleted_reference(
        b'"deleted_aggregation_extra"',
        patterns,
    )
    assert not cleanup_validation._contains_deleted_reference(
        b'"statutes/deleted_aggregation:derived"',
        patterns,
    )


def test_command_count_limit_is_fail_closed(tmp_path, monkeypatch):
    repo, plan, corpus, engine = _fixture(tmp_path)
    monkeypatch.setattr(cleanup_validation, "LEGACY_CLEANUP_MAX_ACTUAL_COMMANDS", 0)

    with pytest.raises(LegacyCleanupValidationError, match="command-count limit"):
        _execute(repo, plan, corpus, engine, _PassingRunner())


@pytest.mark.parametrize(
    ("survivor", "companion", "match"),
    [
        (False, False, "zero surviving primaries"),
        (True, False, "zero surviving companions"),
    ],
)
def test_projected_validation_refuses_vacuous_surviving_corpus(
    tmp_path,
    survivor,
    companion,
    match,
):
    repo, plan, corpus, engine = _fixture(
        tmp_path,
        survivor=survivor,
        surviving_companion=companion,
    )

    with pytest.raises(LegacyCleanupValidationError, match=match):
        _execute(repo, plan, corpus, engine, _PassingRunner())


def test_validation_execution_schema_rejects_incomplete_or_vacuous_evidence(
    tmp_path,
):
    repo, plan, corpus, engine = _fixture(tmp_path)
    evidence = _execute(repo, plan, corpus, engine, _PassingRunner())

    incomplete = {**evidence, "checks": evidence["checks"][:-1]}
    assert "matrix is incomplete" in " ".join(
        validation_execution_issues(incomplete)
    )

    vacuous = {
        **evidence,
        "checks": [dict(check) for check in evidence["checks"]],
    }
    vacuous["checks"][3]["target_count"] = 0
    assert "passed vacuously" in " ".join(validation_execution_issues(vacuous))
