"""Regression tests for the ci parity hardening found in review.

Each test pins one way a local run could disagree with validate-rulespec CI:
state CI never sees (developer pytest options, an edited checkout, another
interpreter, ambient Git steering, a shadowing package), a glob the fallback
layout check must honour, or a caller setting that would change how a
recognized resolver runs.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from axiom_encode import ci_parity, toolchain
from axiom_encode.ci_parity import (
    CliInvocation,
    DependencyMismatch,
    ShardPlan,
    _repository_test_environment,
    _run_cli_isolated,
    committed_checkout,
    parse_caller_workflow,
    uncommitted_changes_note,
    verify_ambient_encoder,
    verify_dependency_checkout,
    verify_python_version,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ci_parity"
CURRENT_KEY = base64.b64encode(bytes(range(32))).decode("ascii")
APPLY_KEY = base64.b64encode(bytes(range(64, 96))).decode("ascii")
EVAL_KEY = base64.b64encode(bytes(range(96, 128))).decode("ascii")


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "rulespec-zz"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "tracked.txt").write_text("one\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "one")
    return repo, _git(repo, "rev-parse", "HEAD")


# -- repository tests see no developer pytest options ------------------------


def test_repository_tests_run_without_pytest_or_git_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k test_pass")
    monkeypatch.setenv("PYTEST_PLUGINS", "shadow")
    monkeypatch.setenv("GIT_DIR", "/elsewhere/.git")
    monkeypatch.setenv("KEEP_ME", "1")

    environment = _repository_test_environment()

    assert "PYTEST_ADDOPTS" not in environment
    assert "PYTEST_PLUGINS" not in environment
    assert "GIT_DIR" not in environment
    assert environment["KEEP_ME"] == "1"


def test_repository_tests_gate_cannot_be_deselected_by_pytest_addopts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = _repo(tmp_path)
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_gate.py").write_text(
        "def test_pass():\n    pass\n\n\ndef test_regression():\n    assert False\n"
    )
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k test_pass -p no:cacheprovider")
    run = ci_parity.WorkflowRun(
        caller=ci_parity.CallerConfig(
            tmp_path / "caller.yml",
            "0effa6a5b05e7fac53902df7d523e909bd7fc48a",
            {},
            "auto",
            True,
            False,
            inputs={"run-pytest": True},
        ),
        workflow=ci_parity.PinnedWorkflow("0effa6a5b05e7fac53902df7d523e909bd7fc48a"),
        paths={},
        repo=repo,
        simulation=ci_parity.PullRequestSimulation(
            "a" * 40, "b" * 40, "main", "", None, True
        ),
        plan=ShardPlan(("zz",), "zz", "zz", "sources programs", '["zz"]', "test"),
        validate_roots_input="auto",
        keyring=(CURRENT_KEY,),
    )
    spec = next(
        gate
        for gate in ci_parity.gate_registry_for_pin(run.workflow.sha)
        if gate.key == "repository_tests"
    )

    result = ci_parity._gate_repository_tests(run, spec)

    assert result.status == "FAIL", result.output
    assert "test_regression" in result.output


# -- gates see the committed HEAD, as actions/checkout does ------------------


@pytest.mark.parametrize(
    "hide",
    ["modified", "untracked", "ignored-conftest", "assume-unchanged", "skip-worktree"],
)
def test_committed_checkout_contains_only_the_committed_tree(
    tmp_path: Path, hide: str
) -> None:
    repo, _ = _repo(tmp_path)
    (repo / ".gitignore").write_text("conftest.py\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "ignore")
    if hide == "modified":
        (repo / "tracked.txt").write_text("edited\n")
    elif hide == "untracked":
        (repo / "untracked.py").write_text("x = 1\n")
    elif hide == "ignored-conftest":
        (repo / "conftest.py").write_text("collect_ignore_glob = ['*']\n")
    else:
        flag = "--assume-unchanged" if hide == "assume-unchanged" else "--skip-worktree"
        _git(repo, "update-index", flag, "tracked.txt")
        (repo / "tracked.txt").write_text("hidden edit\n")

    with committed_checkout(repo) as checkout:
        assert checkout.name == repo.name
        assert _git(checkout, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")
        assert (checkout / "tracked.txt").read_text() == "one\n"
        assert not (checkout / "untracked.py").exists()
        assert not (checkout / "conftest.py").exists()
        assert _git(checkout, "status", "--porcelain", "--untracked-files=all") == ""
    assert not checkout.exists()
    assert str(checkout) not in _git(repo, "worktree", "list")
    note = uncommitted_changes_note(repo)
    if hide == "ignored-conftest":
        # Ignored files are not changes; the committed checkout omits them.
        assert note is None
    else:
        assert note is not None and "committed HEAD" in note


def test_clean_source_has_no_uncommitted_note(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)

    assert uncommitted_changes_note(repo) is None


def test_committed_checkout_is_removed_when_the_run_fails(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)

    with pytest.raises(RuntimeError, match="gate crashed"):
        with committed_checkout(repo) as checkout:
            raise RuntimeError("gate crashed")
    assert not checkout.exists()
    assert str(checkout) not in _git(repo, "worktree", "list")


# -- index flags cannot hide edits in dependency or encoder checkouts ----------


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_index_flags_make_a_dependency_checkout_dirty(
    tmp_path: Path, flag: str
) -> None:
    repo, head = _repo(tmp_path)
    _git(repo, "update-ref", "refs/remotes/origin/main", head)
    _git(repo, "update-index", flag, "tracked.txt")
    (repo / "tracked.txt").write_text("hidden edit\n")
    assert _git(repo, "status", "--porcelain") == ""

    with pytest.raises(ValueError, match="dirty worktree"):
        verify_dependency_checkout(
            "corpus", repo, head, tmp_path / "c.yml", allow_ref_mismatch=False
        )


def test_index_flags_under_encoder_sources_are_a_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = tmp_path / "axiom-encode"
    (encoder / "src" / "axiom_encode").mkdir(parents=True)
    module = encoder / "src" / "axiom_encode" / "ci_parity.py"
    module.write_text("# stand-in\n")
    (encoder / "pyproject.toml").write_text('[project]\nversion = "1"\n')
    _git(encoder, "init", "-q")
    _git(encoder, "config", "user.email", "test@example.com")
    _git(encoder, "config", "user.name", "Test")
    _git(encoder, "add", "-A")
    _git(encoder, "commit", "-qm", "encoder")
    pin = _git(encoder, "rev-parse", "HEAD")
    _git(encoder, "update-index", "--assume-unchanged", "src/axiom_encode/ci_parity.py")
    module.write_text("# edited but hidden from status\n")
    monkeypatch.setattr(ci_parity, "__file__", str(module))

    assert verify_ambient_encoder(
        pin, "1", tmp_path / "c.yml", allow_encoder_mismatch=True
    ) == DependencyMismatch("ambient-encoder", f"{pin} (dirty worktree)", pin)


# -- the interpreter is the caller's python-version ---------------------------


def test_python_version_matches_the_running_interpreter(tmp_path: Path) -> None:
    running = f"{sys.version_info.major}.{sys.version_info.minor}"

    assert (
        verify_python_version(running, tmp_path / "c.yml", allow_encoder_mismatch=False)
        is None
    )


def test_python_version_mismatch_fails_closed_or_is_qualified(tmp_path: Path) -> None:
    running = f"{sys.version_info.major}.{sys.version_info.minor}"

    with pytest.raises(ValueError, match="PYTHON MISMATCH"):
        verify_python_version("3.12", tmp_path / "c.yml", allow_encoder_mismatch=False)
    assert verify_python_version(
        "3.12", tmp_path / "c.yml", allow_encoder_mismatch=True
    ) == DependencyMismatch("python", running, "3.12")


# -- ci's own git calls ignore ambient Git steering ---------------------------


def test_unreadable_status_is_not_a_clean_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, head = _repo(tmp_path)
    _git(repo, "update-ref", "refs/remotes/origin/main", head)
    real = ci_parity._git

    def failing_status(path: Path, *arguments: str, check: bool = True):
        if arguments[:1] == ("status",):
            return subprocess.CompletedProcess(["git"], 128, "", "fatal: bad config")
        return real(path, *arguments, check=check)

    monkeypatch.setattr(ci_parity, "_git", failing_status)

    with pytest.raises(ValueError, match="dirty worktree"):
        verify_dependency_checkout(
            "engine", repo, head, tmp_path / "c.yml", allow_ref_mismatch=False
        )


def test_injected_git_config_cannot_blind_the_dependency_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, head = _repo(tmp_path)
    _git(repo, "update-ref", "refs/remotes/origin/main", head)
    (repo / "tracked.txt").write_text("edited\n")
    # Without the clean environment, this makes `git status` exit 128 with
    # empty stdout, which used to read as clean.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "status.relativePaths")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "invalid")

    with pytest.raises(ValueError, match="dirty worktree"):
        verify_dependency_checkout(
            "engine", repo, head, tmp_path / "c.yml", allow_ref_mismatch=False
        )


def test_ambient_git_dir_does_not_redirect_identity_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, head = _repo(tmp_path)
    other, _ = _repo(tmp_path / "other")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))

    assert ci_parity._git(repo, "rev-parse", "HEAD").stdout.strip() == head


# -- the fallback layout check matches the workflow's -name globs --------------


@pytest.mark.parametrize("name", [".yaml", ".yml", "x.yaml"])
def test_fallback_layout_rejects_every_yaml_name_the_find_globs_match(
    tmp_path: Path, name: str
) -> None:
    repo, _ = _repo(tmp_path)
    (repo / name).write_text("metadata: true\n")
    run = ci_parity.WorkflowRun(
        caller=ci_parity.CallerConfig(
            tmp_path / "c.yml", "0" * 40, {}, "auto", True, False
        ),
        workflow=ci_parity.PinnedWorkflow("0effa6a5b05e7fac53902df7d523e909bd7fc48a"),
        paths={},
        repo=repo,
        simulation=ci_parity.PullRequestSimulation(
            "a" * 40, "b" * 40, "main", "", None, True
        ),
        plan=ShardPlan(("zz",), "zz", "zz", "sources programs", '["zz"]', "test"),
        validate_roots_input="auto",
        keyring=(CURRENT_KEY,),
    )
    spec = next(
        gate
        for gate in ci_parity.gate_registry_for_pin(run.workflow.sha)
        if gate.key == "repository_layout"
    )

    result = ci_parity._gate_repository_layout(run, spec)

    assert result.status == "FAIL"
    assert f"./{name}" in result.output


# -- verification-only apply and eval roots ------------------------------------


def test_local_apply_and_eval_roots_are_scoped_to_the_context() -> None:
    assert toolchain.local_signing_public_key("apply") is None
    with toolchain.local_corpus_release_verification(
        CURRENT_KEY, apply_public_key=APPLY_KEY, eval_public_key=EVAL_KEY
    ):
        assert toolchain.local_signing_public_key("apply") == base64.b64decode(
            APPLY_KEY
        )
        assert toolchain.local_signing_public_key("eval") == base64.b64decode(EVAL_KEY)
    assert toolchain.local_signing_public_key("apply") is None
    assert toolchain.local_signing_public_key("eval") is None


def test_malformed_apply_root_names_its_flag() -> None:
    with pytest.raises(toolchain.RuleSpecToolchainError, match="--apply-public-key"):
        with toolchain.local_corpus_release_verification(
            CURRENT_KEY, apply_public_key="nope"
        ):
            pass


def test_apply_manifest_and_eval_verifiers_use_the_local_roots() -> None:
    from axiom_encode import cli
    from axiom_encode.harness import eval_evidence

    with toolchain.local_corpus_release_verification(
        CURRENT_KEY, apply_public_key=APPLY_KEY, eval_public_key=EVAL_KEY
    ):
        apply_key = cli._applied_encoding_manifest_verifier()
        eval_key = eval_evidence.load_eval_evidence_public_key_from_broker()
    assert cli._raw_ed25519_public_key(apply_key) == base64.b64decode(APPLY_KEY)
    assert cli._raw_ed25519_public_key(eval_key) == base64.b64decode(EVAL_KEY)


def test_isolated_worker_receives_signing_roots_on_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any):
        seen["command"] = command
        seen["request"] = json.loads(kwargs["input"])
        seen["env"] = kwargs["env"]
        seen["cwd"] = kwargs["cwd"]
        return subprocess.CompletedProcess(command, 0, b"ok\n", b"")

    monkeypatch.setattr(ci_parity.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "shadow"))

    code, _ = _run_cli_isolated(
        CliInvocation(("--help",), cwd=tmp_path),
        (CURRENT_KEY,),
        2,
        {"apply": APPLY_KEY},
    )

    assert code == 0
    assert seen["command"][:3] == [sys.executable, "-I", "-c"]
    assert seen["request"]["signing_roots"] == {"apply": APPLY_KEY}
    assert seen["request"]["module_file"] == str(Path(ci_parity.__file__).resolve())
    assert not any(name.startswith("GIT_") for name in seen["env"])
    assert seen["cwd"] == tmp_path


# -- isolated workers import only the verified encoder -------------------------


def test_isolated_worker_ignores_a_shadow_package_in_its_cwd(tmp_path: Path) -> None:
    shadow = tmp_path / "axiom_encode"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("")
    (shadow / "ci_parity.py").write_text(
        "def _isolated_cli_main(request):\n    return 0\n"
    )

    code, output = _run_cli_isolated(
        CliInvocation(("ci", "--repo"), cwd=tmp_path), (CURRENT_KEY,), 1
    )

    # The real encoder ran (and rejected the incomplete argv); the shadow did not.
    assert code != 0
    assert "--repo" in output


def test_isolated_worker_refuses_a_different_encoder_module(tmp_path: Path) -> None:
    request = {
        "package_root": str(Path(ci_parity.__file__).resolve().parents[1]),
        "module_file": str(tmp_path / "elsewhere" / "ci_parity.py"),
        "keyring": [CURRENT_KEY],
        "arguments": ["--help"],
        "cwd": str(tmp_path),
        "supervised": False,
        "environment": {},
    }

    result = subprocess.run(
        [sys.executable, "-I", "-c", ci_parity._ISOLATED_CLI_BOOTSTRAP],
        input=json.dumps(request).encode(),
        capture_output=True,
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode != 0
    assert b"isolated ci worker imported" in result.stderr


# -- workflow-level context cannot change a recognized resolver ---------------


def _us_caller_repo(tmp_path: Path, extra: dict[str, Any]) -> Path:
    repo = tmp_path / "rulespec-us"
    workflows = repo / ".github" / "workflows"
    workflows.mkdir(parents=True)
    payload = yaml.safe_load((FIXTURES / "us-caller.yml").read_text())
    text = (FIXTURES / "us-caller.yml").read_text()
    prefix = yaml.safe_dump(extra, sort_keys=False)
    (workflows / "repository-checks.yml").write_text(prefix + text)
    assert payload["jobs"]["validate"]["needs"] == "workflow-toolchain"
    (repo / ".axiom").mkdir()
    (repo / ".axiom" / "workflow-toolchain.toml").write_text(
        (FIXTURES / "us-workflow-toolchain.toml").read_text()
    )
    return repo


@pytest.mark.parametrize(
    "extra",
    [
        {"defaults": {"run": {"working-directory": "scripts"}}},
        {"env": {"PYTHONPATH": "shadow"}},
    ],
    ids=["defaults", "env"],
)
def test_workflow_level_context_makes_the_resolver_unrecognized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: dict[str, Any]
) -> None:
    monkeypatch.setenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "anon")
    repo = _us_caller_repo(tmp_path, extra)

    with pytest.raises(ValueError, match="the caller workflow sets"):
        parse_caller_workflow(repo / ".github" / "workflows" / "repository-checks.yml")


def test_rulespec_us_caller_without_workflow_context_still_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "anon")
    repo = _us_caller_repo(tmp_path, {"concurrency": {"group": "checks"}})

    caller = parse_caller_workflow(
        repo / ".github" / "workflows" / "repository-checks.yml"
    )

    assert caller.refs["encode"] == "f856cfcb886d9bd050b228aa60aeb4b96939f739"
    assert os.environ["NEXT_PUBLIC_SUPABASE_ANON_KEY"] == "anon"


def test_optimized_interpreter_is_a_python_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    flags = type("Flags", (), {"optimize": 1})()
    monkeypatch.setattr(ci_parity.sys, "flags", flags)

    assert verify_python_version(
        running, tmp_path / "c.yml", allow_encoder_mismatch=True
    ) == DependencyMismatch("python", f"{running} -O", running)


# -- interpreter overrides never reach repository tests or embedded scripts ----


def test_repository_tests_run_without_python_interpreter_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")

    environment = _repository_test_environment()

    assert not any(name.startswith("PYTHON") for name in environment)


def test_embedded_scripts_keep_their_assertions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")

    code, _, stderr = ci_parity._run_embedded_python(
        "assert False, 'embedded assertion ran'\n", cwd=tmp_path
    )

    assert code != 0
    assert "embedded assertion ran" in stderr


# -- trust roots must differ across roles, as the supervisor requires -----------


@pytest.mark.parametrize(
    ("apply_key", "eval_key"),
    [(CURRENT_KEY, None), (None, CURRENT_KEY), (APPLY_KEY, APPLY_KEY)],
    ids=["apply-equals-corpus", "eval-equals-corpus", "apply-equals-eval"],
)
def test_trust_roots_must_be_distinct_across_roles(
    apply_key: str | None, eval_key: str | None
) -> None:
    with pytest.raises(toolchain.RuleSpecToolchainError, match="must be distinct"):
        with toolchain.local_corpus_release_verification(
            CURRENT_KEY, apply_public_key=apply_key, eval_public_key=eval_key
        ):
            pass


# -- the committed checkout is a fresh clone, not a linked worktree ------------


def _checked_in(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", f"add {relative}")


def test_source_sparse_patterns_do_not_reach_the_checkout(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    _checked_in(repo, "tests/test_regression.py", "def test_x():\n    assert False\n")
    _git(repo, "sparse-checkout", "set", "--no-cone", "/*", "!/tests/")
    assert not (repo / "tests" / "test_regression.py").exists()

    with committed_checkout(repo) as checkout:
        assert (
            (checkout / "tests" / "test_regression.py")
            .read_text()
            .endswith("assert False\n")
        )


def test_source_smudge_filters_do_not_rewrite_the_checkout(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    _checked_in(repo, ".gitattributes", "tests/*.py filter=local\n")
    _checked_in(repo, "tests/test_regression.py", "def test_x():\n    assert False\n")
    _git(repo, "config", "filter.local.smudge", "sed s/False/True/g")
    _git(repo, "config", "filter.local.clean", "cat")

    with committed_checkout(repo) as checkout:
        assert "assert False" in (checkout / "tests" / "test_regression.py").read_text()


def test_source_hooks_do_not_run_for_the_checkout(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    marker = tmp_path / "hook-ran"
    hook = repo / ".git" / "hooks" / "post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    hook.chmod(0o755)

    with committed_checkout(repo) as checkout:
        assert not (checkout / ".git" / "hooks" / "post-checkout").exists()
    assert not marker.exists()


def test_checkout_origin_is_the_source_origin(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    url = "https://github.com/TheAxiomFoundation/rulespec-zz.git"
    _git(repo, "remote", "add", "origin", url)

    with committed_checkout(repo) as checkout:
        assert _git(checkout, "remote", "get-url", "origin") == url


def test_checkout_without_source_origin_has_none(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)

    with committed_checkout(repo) as checkout:
        assert _git(checkout, "remote") == ""


def test_base_refs_resolve_in_the_source_checkout(tmp_path: Path) -> None:
    upstream, _ = _repo(tmp_path / "upstream")
    clone = tmp_path / "rulespec-zz"
    _git(tmp_path, "clone", "-q", str(upstream), str(clone))
    base = _git(clone, "rev-parse", "HEAD")
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    _checked_in(clone, "later.txt", "later\n")

    assert ci_parity.resolve_commit(clone, "@{upstream}") == base
    with pytest.raises(ValueError, match="does not name a commit"):
        ci_parity.resolve_commit(clone, "no-such-ref")


# -- cycle 4: attributes, object integrity, path collisions, base branch -------


def test_default_user_attributes_do_not_rewrite_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = _repo(tmp_path)
    _checked_in(repo, "tests/test_lf.py", "def test_x():\n    pass\n")
    xdg = tmp_path / "xdg"
    (xdg / "git").mkdir(parents=True)
    (xdg / "git" / "attributes").write_text("*.py text eol=crlf\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    with committed_checkout(repo) as checkout:
        assert b"\r\n" not in (checkout / "tests" / "test_lf.py").read_bytes()


def test_a_substituted_object_fails_the_checkout_closed(tmp_path: Path) -> None:
    import zlib

    repo, _ = _repo(tmp_path)
    _checked_in(repo, "tests/test_regression.py", "assert False\n")
    blob = _git(repo, "rev-parse", "HEAD:tests/test_regression.py")
    loose = repo / ".git" / "objects" / blob[:2] / blob[2:]
    assert loose.exists()
    forged = b"assert True\n"
    loose.chmod(0o644)
    loose.write_bytes(zlib.compress(b"blob %d\0" % len(forged) + forged))

    with pytest.raises(subprocess.CalledProcessError):
        with committed_checkout(repo):
            pass


def test_paths_that_collide_on_this_filesystem_fail_closed(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    upper = (
        subprocess.run(
            ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
            input=b"assert False\n",
            stdout=subprocess.PIPE,
            check=True,
        )
        .stdout.decode()
        .strip()
    )
    lower = (
        subprocess.run(
            ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
            input=b"assert True\n",
            stdout=subprocess.PIPE,
            check=True,
        )
        .stdout.decode()
        .strip()
    )
    for blob, path in ((upper, "tests/test_CASE.py"), (lower, "tests/test_case.py")):
        _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
    _git(repo, "commit", "-qm", "case pair")
    probe = tmp_path / "Probe"
    probe.write_text("")
    case_insensitive = (tmp_path / "probe").exists()

    if case_insensitive:
        with pytest.raises(ValueError, match="collide on this filesystem"):
            with committed_checkout(repo):
                pass
    else:
        with committed_checkout(repo) as checkout:
            assert (checkout / "tests" / "test_CASE.py").read_text() == "assert False\n"
            assert (checkout / "tests" / "test_case.py").read_text() == "assert True\n"


def test_upstream_base_names_its_branch_for_the_pull_request(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    _git(upstream, "config", "user.email", "test@example.com")
    _git(upstream, "config", "user.name", "Test")
    (upstream / "a.txt").write_text("a\n")
    _git(upstream, "add", "-A")
    _git(upstream, "commit", "-qm", "a")
    clone = tmp_path / "rulespec-zz"
    _git(tmp_path, "clone", "-q", str(upstream), str(clone))

    for ref in ("@{upstream}", "@{u}", "origin/main", "refs/remotes/origin/main"):
        simulation = ci_parity.simulate_pull_request(clone, ref)
        assert simulation.base_branch == "main", ref
    sha = _git(clone, "rev-parse", "HEAD")
    assert ci_parity.simulate_pull_request(clone, sha).base_branch == sha
