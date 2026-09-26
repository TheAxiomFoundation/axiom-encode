"""Parity tests for the embedded-script gates of validate-rulespec 0effa6a5/6f11be26.

Every gate that runs one of the pinned workflow's inline scripts runs the REAL
script here (it needs only git, python and PyYAML), over a tiny git repository
built in ``tmp_path``. Supervised and unsupervised encoder subcommands are
replaced by recorders, so each test asserts the exact argv, working directory,
supervision and environment the corresponding workflow step implies. Expected
values come from the packaged workflow files, not from ``ci_parity``.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from axiom_encode import ci_parity
from axiom_encode.ci_parity import (
    DEPENDENCY_INPUTS,
    SUPPORTED_WORKFLOW_PINS,
    WORKFLOW_GATE_ORDER_0EFFA6A5,
    WORKFLOW_GATE_ORDER_6F11BE26,
    WORKFLOW_INPUTS_0EFFA6A5,
    CallerConfig,
    CliInvocation,
    GateResult,
    GateSpec,
    PinnedWorkflow,
    ShardPlan,
    ShardSelection,
    WorkflowRun,
    compute_shard_plan,
    execute_workflow_gates,
    gate_registry_for_pin,
    resolve_roots,
    simulate_pull_request,
)

PIN_0EFFA6A5 = "0effa6a5b05e7fac53902df7d523e909bd7fc48a"
PIN_6F11BE26 = "6f11be2655f79dd0a3b582db46525f58332ca120"
EMBEDDED_PINS = pytest.mark.parametrize(
    "pin", [PIN_0EFFA6A5, PIN_6F11BE26], ids=lambda sha: sha[:8]
)
RULESPEC_US = "TheAxiomFoundation/rulespec-us"
RULESPEC_US_REMOTE = f"https://github.com/{RULESPEC_US}.git"
OTHER_REMOTE = "https://github.com/TheAxiomFoundation/rulespec-zz.git"
REFS = {
    "encode": "1" * 40,
    "engine": "2" * 40,
    "corpus": "3" * 40,
    "rulespec_us": "4" * 40,
}
KEYRING = ("test-corpus-release-key",)
ISSUE = f"https://github.com/{RULESPEC_US}/issues/1"
PLAIN = "module:\n  id: plain\n"
CHANGED = "module:\n  id: changed\n"
RETIRED = (
    "module:\n"
    "  source_verification:\n"
    "    corpus_citation_paths:\n"
    "      - us/statute/26/1\n"
)
FREEZE_PATH = ".axiom/retired-schema-freeze.json"
AUTHORIZATION_PATH = ".axiom/reviewed-migrations.json"
LEDGER = "known-validation-gaps.yaml"
TOOLCHAIN = ".axiom/toolchain.toml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _sha256(data: bytes | str) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def _write(repo: Path, files: Mapping[str, str | bytes | None]) -> None:
    for relative, content in files.items():
        path = repo / relative
        if content is None:
            path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")


def _commit(
    repo: Path,
    files: Mapping[str, str | bytes | None] | None = None,
    message: str = "change",
) -> str:
    _write(repo, files or {})
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _repo(
    tmp_path: Path,
    files: Mapping[str, str | bytes | None],
    *,
    name: str = "rulespec-zz",
    remote: str | None = None,
) -> Path:
    """A checkout whose first commit is also refs/remotes/origin/main."""

    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    if remote is not None:
        _git(repo, "remote", "add", "origin", remote)
    _commit(repo, files, "base")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


@functools.cache
def _workflow(sha: str) -> PinnedWorkflow:
    return PinnedWorkflow(sha)


def _spec(key: str, sha: str = PIN_0EFFA6A5) -> GateSpec:
    return next(spec for spec in gate_registry_for_pin(sha) if spec.key == key)


def _caller(
    tmp_path: Path, sha: str, inputs: Mapping[str, Any] | None = None
) -> CallerConfig:
    values: dict[str, Any] = {
        name: declaration.default
        for name, declaration in WORKFLOW_INPUTS_0EFFA6A5.items()
    }
    for dependency, name in DEPENDENCY_INPUTS.items():
        values[name] = REFS[dependency]
    values.update(inputs or {})
    return CallerConfig(
        path=tmp_path / "caller.yml",
        workflow_sha=sha,
        refs=dict(REFS),
        validate_roots=str(values["validate-roots"]),
        run_generated_guard=bool(values["run-generated-guard"]),
        guard_programs_root=bool(values["guard-programs-root"]),
        release_base_url=str(values["corpus-release-base-url"]),
        run_pytest=bool(values["run-pytest"]),
        run_money_atom_check=bool(values["run-money-atom-check"]),
        inputs=values,
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path / "deps"
    return {
        "encode": base / "axiom-encode",
        "engine": base / "axiom-rules-engine",
        "corpus": base / "axiom-corpus",
        "rulespec_us": base / "rulespec-us",
    }


def _run(
    repo: Path,
    tmp_path: Path,
    *,
    sha: str = PIN_0EFFA6A5,
    inputs: Mapping[str, Any] | None = None,
    number: int | None = None,
    base_ref: str = "origin/main",
    plan: ShardPlan | None = None,
) -> WorkflowRun:
    caller = _caller(tmp_path, sha, {"validate-roots": "auto", **(inputs or {})})
    workflow = _workflow(sha)
    simulation = simulate_pull_request(repo, base_ref, number)
    roots_input = str(caller.inputs["validate-roots"])
    if plan is None:
        plan = compute_shard_plan(
            workflow, repo, simulation, roots_input, caller.guard_programs_root
        )
    temp = tmp_path / "runner-temp"
    temp.mkdir(exist_ok=True)
    return WorkflowRun(
        caller=caller,
        workflow=workflow,
        paths=_paths(tmp_path),
        repo=repo,
        simulation=simulation,
        plan=plan,
        validate_roots_input=roots_input,
        keyring=KEYRING,
        temp=temp,
    )


def _plan(*shards: str, roots: str | None = None) -> ShardPlan:
    return ShardPlan(
        matrix=shards,
        first=shards[0],
        roots=roots if roots is not None else " ".join(shards),
        allowed_extra="sources programs",
        matrix_json=json.dumps(list(shards)),
        scope="test plan",
    )


class _BatchRecorder:
    """Stands in for _run_cli_batch; records every invocation it is handed."""

    def __init__(
        self,
        outcome: Callable[[CliInvocation], tuple[int, str]] | None = None,
        events: list[tuple[str, tuple[str, ...]]] | None = None,
    ) -> None:
        self.batches: list[list[CliInvocation]] = []
        self.outcome = outcome or (lambda _invocation: (0, "ok\n"))
        self.events = events

    def __call__(
        self,
        invocations: list[CliInvocation],
        *,
        jobs: int,
        keyring: tuple[str, ...],
        signing_roots: dict[str, str] | None = None,
    ) -> list[tuple[int, str]]:
        assert jobs == 1
        assert tuple(keyring) == KEYRING
        assert not signing_roots
        batch = list(invocations)
        self.batches.append(batch)
        if self.events is not None:
            self.events.extend(("batch", item.arguments) for item in batch)
        return [self.outcome(invocation) for invocation in batch]

    @property
    def invocations(self) -> list[CliInvocation]:
        return [invocation for batch in self.batches for invocation in batch]


class _CliRecorder:
    """Stands in for _run_cli; records argv, environment, cwd and supervision."""

    def __init__(
        self,
        result: tuple[int, str] = (0, "ok\n"),
        events: list[tuple[str, tuple[str, ...]]] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result
        self.events = events

    def __call__(
        self,
        arguments: list[str],
        *,
        environment: dict[str, str] | None = None,
        cwd: Path | None = None,
        supervised: bool = False,
    ) -> tuple[int, str]:
        self.calls.append(
            {
                "arguments": list(arguments),
                "environment": environment,
                "cwd": cwd,
                "supervised": supervised,
            }
        )
        if self.events is not None:
            self.events.append(("cli", tuple(arguments)))
        return self.result


def _patch_batch(
    monkeypatch: pytest.MonkeyPatch,
    outcome: Callable[[CliInvocation], tuple[int, str]] | None = None,
) -> _BatchRecorder:
    recorder = _BatchRecorder(outcome)
    monkeypatch.setattr(ci_parity, "_run_cli_batch", recorder)
    return recorder


def _patch_cli(
    monkeypatch: pytest.MonkeyPatch, result: tuple[int, str] = (0, "ok\n")
) -> _CliRecorder:
    recorder = _CliRecorder(result)
    monkeypatch.setattr(ci_parity, "_run_cli", recorder)
    return recorder


@pytest.fixture(autouse=True)
def _no_real_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a gate reaches a real encoder, classifier or pytest run."""

    def refuse(name: str) -> Callable[..., Any]:
        def refused(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(f"unexpected real {name} call: {args!r}")

        return refused

    for name in ("_run_cli", "_run_cli_batch", "_run_pinned_process", "_run_process"):
        monkeypatch.setattr(ci_parity, name, refuse(name))


def _freeze(modules: Mapping[str, str]) -> str:
    return (
        json.dumps(
            {
                "format": "axiom/retired-schema-freeze/v1",
                "artifacts": {path: _sha256(text) for path, text in modules.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _record(tag: str) -> dict[str, str]:
    return {"fingerprint": f"sha256:{tag}", "issue": ISSUE, "expires": "2999-01-01"}


def _ledger(entries: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump({"validate_failures": dict(entries)}, sort_keys=True).encode()


def _toolchain(
    ledger: bytes,
    *,
    release: str = "us-rulespec-2026-09-01",
    content: str = "c" * 64,
) -> str:
    return (
        "[toolchain]\n"
        f'axiom_corpus_release = "{release}"\n'
        f'axiom_corpus_release_content_sha256 = "{content}"\n'
        f'validation_waiver_set_sha256 = "{_sha256(ledger)}"\n'
    )


# ---------------------------------------------------------------------------
# Packaged workflow and gate order
# ---------------------------------------------------------------------------


@EMBEDDED_PINS
def test_pinned_workflow_rejects_a_packaged_file_that_is_not_the_pinned_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str
) -> None:
    fixture = SUPPORTED_WORKFLOW_PINS[pin].fixture
    assert PinnedWorkflow(pin).path.name == fixture
    tampered = tmp_path / fixture
    shutil.copyfile(ci_parity.WORKFLOW_DIRECTORY / fixture, tampered)
    tampered.write_bytes(tampered.read_bytes() + b"# local edit\n")
    monkeypatch.setattr(ci_parity, "WORKFLOW_DIRECTORY", tmp_path)

    with pytest.raises(ValueError, match="is blob"):
        PinnedWorkflow(pin)


@EMBEDDED_PINS
def test_execute_workflow_gates_runs_every_handler_in_pin_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN})
    run = _run(repo, tmp_path, sha=pin, plan=_plan("us"))
    seen: list[tuple[str, GateSpec]] = []
    temps: list[Path] = []
    for key in WORKFLOW_GATE_ORDER_6F11BE26:

        def handler(run: WorkflowRun, spec: GateSpec, key: str = key) -> GateResult:
            assert run.temp.is_dir()
            temps.append(run.temp)
            seen.append((key, spec))
            return GateResult(spec.key, spec.name, "PASS")

        monkeypatch.setattr(ci_parity, f"_gate_{key}", handler)

    results = execute_workflow_gates(run)

    expected = (
        WORKFLOW_GATE_ORDER_0EFFA6A5
        if pin == PIN_0EFFA6A5
        else WORKFLOW_GATE_ORDER_6F11BE26
    )
    assert tuple(key for key, _ in seen) == expected
    assert all(key == spec.key for key, spec in seen)
    assert tuple(result.gate for result in results) == expected
    assert tuple(spec.key for spec in gate_registry_for_pin(pin)) == expected
    assert run.workflow.pin.gates == expected
    # Repository-controlled tests run after every trusted gate, as in CI.
    assert expected[-1] == "repository_tests"
    assert ("unmanifested_rulespec" in expected) is (pin == PIN_6F11BE26)
    if pin == PIN_6F11BE26:
        assert expected[:3] == (
            "unsupported_paths",
            "unmanifested_rulespec",
            "migration_authorization",
        )
    # One shared RUNNER_TEMP for the whole run, removed afterwards.
    assert len(set(temps)) == 1
    assert not temps[0].exists()


@EMBEDDED_PINS
def test_execute_workflow_gates_end_to_end_runs_repository_tests_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str
) -> None:
    ledger = b"validate_failures: {}\n"
    module = "us/statutes/a.yaml"
    manifest = ".axiom/encoding-manifests/us/statutes/a.json"

    def manifested(content: str) -> dict[str, str]:
        return {
            module: content,
            manifest: json.dumps(
                {"applied_files": [{"path": module, "sha256": _sha256(content)}]}
            ),
        }

    repo = _repo(
        tmp_path,
        {
            LEDGER: ledger,
            TOOLCHAIN: _toolchain(ledger),
            "us/statutes/a.test.yaml": "cases: []\n",
            "tests/test_repository.py": "def test_ok():\n    pass\n",
            **manifested(PLAIN),
        },
        remote=OTHER_REMOTE,
    )
    head = _commit(repo, manifested(CHANGED))
    run = _run(repo, tmp_path, sha=pin)
    events: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(ci_parity, "_run_cli_batch", _BatchRecorder(events=events))
    monkeypatch.setattr(ci_parity, "_run_cli", _CliRecorder(events=events))

    def pinned(
        encode_path: Path, encode_pin: str, arguments: list[str], *, stderr: int
    ) -> tuple[int, str, str]:
        events.append(("pinned", tuple(arguments)))
        return 0, json.dumps({"items": []}), ""

    def process(
        arguments: list[str], cwd: Path, *, environment: dict[str, str] | None = None
    ) -> tuple[int, str]:
        assert cwd == repo
        assert environment is not None and not any(
            name.startswith("PYTEST_") for name in environment
        )
        events.append(("process", tuple(arguments)))
        return 0, "1 passed\n"

    monkeypatch.setattr(ci_parity, "_run_pinned_process", pinned)
    monkeypatch.setattr(ci_parity, "_run_process", process)

    results = execute_workflow_gates(run)

    assert [result.gate for result in results] == list(run.workflow.pin.gates)
    assert {result.gate: result.status for result in results} == {
        key: "PASS" for key in run.workflow.pin.gates
    }, "\n".join(result.output for result in results if result.status != "PASS")
    assert run.plan.matrix == ("us",)
    assert run.mode == "changed"
    assert run.migration_authorized == "false"
    assert run.retired_skip == frozenset()
    assert events[-1] == (
        "process",
        (sys.executable, "-m", "pytest", "-q", "tests"),
    )
    assert [kind for kind, _ in events] == [
        "batch",  # validation-waivers audit (one partition)
        "cli",  # guard-generated
        "batch",  # validate
        "batch",  # companion tests
        "batch",  # proof-validate
        "cli",  # money atoms
        "pinned",  # changed-file oracle coverage classifier
        "process",  # repository tests
    ]
    guard = next(arguments for kind, arguments in events if kind == "cli")
    assert guard[guard.index("--head-ref") + 1] == head


# ---------------------------------------------------------------------------
# simulate_pull_request
# ---------------------------------------------------------------------------


def test_simulated_pull_request_carries_base_head_and_event_context(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN}, remote=RULESPEC_US_REMOTE)
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, {"us/statutes/a.yaml": CHANGED})

    simulation = simulate_pull_request(repo, "origin/main", 911)

    assert (simulation.base_sha, simulation.head_sha) == (base, head)
    assert simulation.base_branch == "main"
    assert simulation.repository == RULESPEC_US
    assert simulation.number == 911
    assert simulation.base_is_ancestor is True
    assert simulation.ref == "refs/pull/911/merge"
    assert simulation.github_environment(repo) == {
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_REF": "refs/pull/911/merge",
        "GITHUB_REPOSITORY": RULESPEC_US,
        "GITHUB_SHA": head,
        "GITHUB_WORKSPACE": str(repo),
    }
    assert simulate_pull_request(repo, "origin/main").ref == ""


@pytest.mark.parametrize(
    ("base_ref", "branch"),
    [
        ("origin/main", "main"),
        ("refs/remotes/origin/main", "main"),
        ("refs/heads/main", "main"),
        ("main", "main"),
        ("origin/release", "release"),
        ("refs/remotes/origin/release", "release"),
    ],
)
def test_simulated_base_branch_strips_ref_prefixes(
    tmp_path: Path, base_ref: str, branch: str
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN})
    _git(repo, "update-ref", "refs/remotes/origin/release", "HEAD")

    simulation = simulate_pull_request(repo, base_ref)

    assert simulation.base_branch == branch
    assert simulation.base_sha == _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    ("remote", "slug"),
    [
        (f"https://github.com/{RULESPEC_US}.git", RULESPEC_US),
        (f"https://github.com/{RULESPEC_US}", RULESPEC_US),
        (f"https://github.com/{RULESPEC_US}/", RULESPEC_US),
        (f"git@github.com:{RULESPEC_US}.git", RULESPEC_US),
        (f"git@github.com:{RULESPEC_US}", RULESPEC_US),
        (f"ssh://git@github.com/{RULESPEC_US}.git", RULESPEC_US),
        ("https://github.com/owner/rulespec.us.git", "owner/rulespec.us"),
        (f"https://gitlab.com/{RULESPEC_US}.git", ""),
        (None, ""),
    ],
)
def test_simulated_repository_slug_comes_from_the_origin_remote(
    tmp_path: Path, remote: str | None, slug: str
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN}, remote=remote)

    assert simulate_pull_request(repo, "origin/main").repository == slug


def test_simulated_pull_request_flags_a_head_that_lacks_the_base(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN})
    head = _git(repo, "rev-parse", "HEAD")
    base = _commit(repo, {"us/statutes/b.yaml": PLAIN})
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    _git(repo, "reset", "-q", "--hard", head)

    simulation = simulate_pull_request(repo, "origin/main")

    assert simulation.base_sha == base
    assert simulation.head_sha == head
    assert simulation.base_is_ancestor is False


@pytest.mark.parametrize("base_ref", ["origin/nope", "HEAD^{tree}"])
def test_simulated_pull_request_rejects_a_base_ref_that_is_not_a_commit(
    tmp_path: Path, base_ref: str
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN})

    with pytest.raises(ValueError, match="does not name a commit") as error:
        simulate_pull_request(repo, base_ref)
    assert f"--base-ref {base_ref}" in str(error.value)


# ---------------------------------------------------------------------------
# compute_shard_plan (the shards job, with the REAL embedded scope script)
# ---------------------------------------------------------------------------

AUTO_LAYOUT = {
    "us/statutes/a.yaml": PLAIN,
    "us-ca/regulations/b.yaml": PLAIN,
    "tz-znz/legislation/c.yaml": PLAIN,
    "yy/policies/d.yaml": PLAIN,
    # Not jurisdiction shards: leading "_" or ".", a name failing
    # ^[a-z]{2}(-[a-z0-9-]+)*$, or no statutes/regulations/policies/legislation.
    "_x/statutes/e.yaml": PLAIN,
    ".x/statutes/f.yaml": PLAIN,
    "Qq/statutes/g.yaml": PLAIN,
    "usa/statutes/h.yaml": PLAIN,
    "u/statutes/i.yaml": PLAIN,
    "us_ny/statutes/j.yaml": PLAIN,
    "zz/manual/k.yaml": PLAIN,
    "sources/us/l.yaml": PLAIN,
    "README.md": "readme\n",
}
# In the order bash visits `for dir in */` on the runner: "us-ca/" sorts
# before "us/" because '-' < '/'.
AUTO_SHARDS = ("tz-znz", "us-ca", "us", "yy")


@EMBEDDED_PINS
@pytest.mark.parametrize(
    ("change", "matrix", "scope"),
    [
        (
            {"us/statutes/a.yaml": CHANGED},
            ("us",),
            "scope=one-jurisdiction (us)",
        ),
        (
            {"us/statutes/a.yaml": None},
            ("us",),
            "scope=one-jurisdiction (us)",
        ),
        (
            {"us-ca/regulations/new.yaml": PLAIN},
            ("us-ca",),
            "scope=one-jurisdiction (us-ca)",
        ),
        (
            {"us/statutes/a.yaml": CHANGED, ".axiom/index/us.json": "{}\n"},
            ("us",),
            "scope=one-jurisdiction (us)",
        ),
        (
            {"us/statutes/a.yaml": CHANGED, "us-ca/regulations/b.yaml": CHANGED},
            AUTO_SHARDS,
            "scope=full (2 jurisdiction subtrees touched)",
        ),
        (
            {"README.md": "changed\n"},
            AUTO_SHARDS,
            "scope=full (cross-cutting path README.md)",
        ),
        (
            {".github/workflows/x.yml": "on: push\n"},
            AUTO_SHARDS,
            "scope=full (cross-cutting path .github/workflows/x.yml)",
        ),
        (
            {"us/statutes/a.yaml": CHANGED, "_x/statutes/e.yaml": CHANGED},
            AUTO_SHARDS,
            "scope=full (cross-cutting path _x/statutes/e.yaml)",
        ),
        (
            {"us/statutes/a.yaml": CHANGED, "sources/us/l.yaml": CHANGED},
            AUTO_SHARDS,
            "scope=full (cross-cutting path sources/us/l.yaml)",
        ),
        (
            {".axiom/index/us.json": "{}\n"},
            AUTO_SHARDS,
            "scope=full (0 jurisdiction subtrees touched)",
        ),
        ({}, AUTO_SHARDS, "scope=full (empty diff)"),
    ],
    ids=[
        "one-jurisdiction",
        "one-jurisdiction-deletion",
        "one-jurisdiction-addition",
        "index-alongside-one-jurisdiction",
        "two-jurisdictions",
        "root-file",
        "workflow",
        "non-shard-directory",
        "sources",
        "index-only",
        "empty-diff",
    ],
)
def test_auto_shard_plan_scopes_like_the_workflow(
    tmp_path: Path,
    pin: str,
    change: dict[str, str | None],
    matrix: tuple[str, ...],
    scope: str,
) -> None:
    repo = _repo(tmp_path, AUTO_LAYOUT)
    _commit(repo, change)

    plan = compute_shard_plan(
        _workflow(pin), repo, simulate_pull_request(repo, "origin/main"), "auto", False
    )

    assert plan.matrix == matrix
    assert plan.first == matrix[0]
    assert plan.scope == scope
    assert json.loads(plan.matrix_json) == list(matrix)
    assert plan.matrix_json == json.dumps(list(matrix))
    # roots always covers every discovered jurisdiction, even when scoped.
    assert plan.roots == " ".join(AUTO_SHARDS)
    assert plan.allowed_extra == "sources programs"
    assert plan.shard_roots(matrix[0], "auto") == matrix[0]


@EMBEDDED_PINS
def test_auto_shard_plan_orders_shards_like_the_rulespec_us_runner(
    tmp_path: Path, pin: str
) -> None:
    # Repository Checks run 36184228440 (rulespec-us 54d90a72) printed
    # Shards: ["us-ak", "us-al", ..., "us-wy", "us"]: the country root sorts
    # last, so us-ak is the first shard that runs the repository-wide gates.
    repo = _repo(
        tmp_path,
        {
            "us/statutes/a.yaml": PLAIN,
            "us-ak/statutes/b.yaml": PLAIN,
            "us-al/regulations/c.yaml": PLAIN,
            "us-wy/policies/d.yaml": PLAIN,
        },
    )
    _commit(repo, {})

    plan = compute_shard_plan(
        _workflow(pin), repo, simulate_pull_request(repo, "origin/main"), "auto", False
    )

    assert plan.matrix == ("us-ak", "us-al", "us-wy", "us")
    assert plan.first == "us-ak"
    assert plan.roots == "us-ak us-al us-wy us"
    assert resolve_roots(repo, "auto") == ("us-ak", "us-al", "us-wy", "us")


def test_auto_shard_plan_guards_programs_as_a_root(tmp_path: Path) -> None:
    repo = _repo(tmp_path, AUTO_LAYOUT)
    _commit(repo, {"us/statutes/a.yaml": CHANGED})

    plan = compute_shard_plan(
        _workflow(PIN_0EFFA6A5),
        repo,
        simulate_pull_request(repo, "origin/main"),
        "auto",
        True,
    )

    assert plan.matrix == ("us",)
    assert plan.roots == " ".join(AUTO_SHARDS) + " programs"
    assert plan.roots.endswith(" programs")
    assert plan.allowed_extra == "sources"


@pytest.mark.parametrize(
    ("validate_roots", "guard", "roots"),
    [
        ("statutes regulations policies", False, "statutes regulations policies"),
        ("statutes regulations", True, "statutes regulations programs"),
        ("programs statutes", True, "programs statutes"),
        ("statutes programs", True, "statutes programs"),
        ("myprograms", True, "myprograms programs"),
        ("us", False, "us"),
    ],
)
def test_explicit_validate_roots_run_one_all_shard(
    tmp_path: Path, validate_roots: str, guard: bool, roots: str
) -> None:
    repo = _repo(tmp_path, AUTO_LAYOUT)
    _commit(repo, {"us/statutes/a.yaml": CHANGED})

    plan = compute_shard_plan(
        _workflow(PIN_0EFFA6A5),
        repo,
        simulate_pull_request(repo, "origin/main"),
        validate_roots,
        guard,
    )

    assert plan.matrix == ("__all__",)
    assert plan.first == "__all__"
    assert plan.matrix_json == '["__all__"]'
    assert plan.roots == roots
    assert plan.allowed_extra == "sources"
    assert plan.scope == "explicit validate-roots"
    # "Resolve shard validation roots": the __all__ leg uses the raw input.
    assert plan.shard_roots("__all__", validate_roots) == validate_roots


def test_auto_shard_plan_without_jurisdictions_fails(tmp_path: Path) -> None:
    repo = _repo(
        tmp_path,
        {
            "statutes/a.yaml": PLAIN,
            "_us/statutes/b.yaml": PLAIN,
            "zz/manual/c.yaml": PLAIN,
        },
    )

    with pytest.raises(ValueError, match="found no jurisdiction directories"):
        compute_shard_plan(
            _workflow(PIN_0EFFA6A5),
            repo,
            simulate_pull_request(repo, "origin/main"),
            "auto",
            False,
        )


@pytest.mark.parametrize("validate_roots", ["us*", "statutes us-?", "[s]tatutes"])
def test_glob_characters_in_validate_roots_are_refused(
    tmp_path: Path, validate_roots: str
) -> None:
    repo = _repo(tmp_path, AUTO_LAYOUT)

    with pytest.raises(ValueError, match="glob characters"):
        compute_shard_plan(
            _workflow(PIN_0EFFA6A5),
            repo,
            simulate_pull_request(repo, "origin/main"),
            validate_roots,
            False,
        )


# ---------------------------------------------------------------------------
# Reject unsupported tracked paths (REAL python3 -c script)
# ---------------------------------------------------------------------------

UNSUPPORTED_NAMES = ["bad\tname.yaml", "bad\nname.yaml", "bad\x7fname.yaml"]


@EMBEDDED_PINS
@pytest.mark.parametrize(
    "name", UNSUPPORTED_NAMES + ["bad name.yaml"], ids=["tab", "lf", "del", "ls"]
)
def test_unsupported_paths_rejects_a_tracked_control_character(
    tmp_path: Path, pin: str, name: str
) -> None:
    path = f"us/statutes/{name}"
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN, path: PLAIN})
    assert path in _git(repo, "ls-files", "-z").split("\0")
    _commit(repo, {"us/statutes/a.yaml": CHANGED})
    run = _run(repo, tmp_path, sha=pin)

    result = ci_parity._gate_unsupported_paths(run, _spec("unsupported_paths", pin))

    assert result.status == "FAIL"
    assert f"unsupported tracked path: {path!r}" in result.failures
    assert result.command == [
        "tracked-paths",
        run.simulation.base_sha,
        run.simulation.head_sha,
    ]


@pytest.mark.parametrize("name", UNSUPPORTED_NAMES, ids=["tab", "lf", "del"])
def test_unsupported_paths_also_checks_paths_only_in_the_diff(
    tmp_path: Path, name: str
) -> None:
    path = f"us/statutes/{name}"
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN, path: PLAIN})
    _commit(repo, {path: None})
    assert path not in _git(repo, "ls-files", "-z").split("\0")
    run = _run(repo, tmp_path)

    result = ci_parity._gate_unsupported_paths(run, _spec("unsupported_paths"))

    assert result.status == "FAIL"
    assert f"unsupported tracked path: {path!r}" in result.failures


def test_unsupported_paths_passes_an_ordinary_repository(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN, "README.md": "é\n"})
    _commit(repo, {"us/statutes/ü.yaml": PLAIN})
    run = _run(repo, tmp_path)

    result = ci_parity._gate_unsupported_paths(run, _spec("unsupported_paths"))

    assert result.status == "PASS", result.output
    assert result.output == "No unsupported tracked or changed paths.\n"


# ---------------------------------------------------------------------------
# Authorize exact reviewed migration (REAL heredoc)
# ---------------------------------------------------------------------------


def test_migration_authorization_not_requested_outputs_false(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN}, remote=OTHER_REMOTE)
    run = _run(repo, tmp_path, plan=_plan("us"))

    result = ci_parity._gate_migration_authorization(
        run, _spec("migration_authorization")
    )

    assert result.status == "PASS", result.output
    assert run.migration_authorized == "false"
    assert run.migration_candidate == ""
    assert result.output == "authorized=false\ncandidate=\n"
    assert result.note is None


@pytest.mark.parametrize(
    "inputs",
    [
        {"run-generated-guard": False},
        {"retired-schema-bootstrap-sha256": "a" * 64},
        {"validation-waiver-bootstrap-sha256": "b" * 64},
    ],
    ids=["guard-disabled", "retired-bootstrap", "waiver-bootstrap"],
)
def test_migration_authorization_is_restricted_to_rulespec_us(
    tmp_path: Path, inputs: dict[str, Any]
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN}, remote=OTHER_REMOTE)
    run = _run(repo, tmp_path, inputs=inputs, number=911, plan=_plan("us"))

    result = ci_parity._gate_migration_authorization(
        run, _spec("migration_authorization")
    )

    assert result.status == "FAIL"
    assert (
        "ReviewedMigrationAuthorizationError: migration authorization is "
        "restricted to rulespec-us"
    ) in result.failures
    assert run.migration_authorized == ""


def test_migration_authorization_without_pull_request_number_points_at_flag(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN}, remote=RULESPEC_US_REMOTE)
    run = _run(
        repo,
        tmp_path,
        inputs={
            "run-generated-guard": False,
            "migration-authorization-path": AUTHORIZATION_PATH,
        },
        plan=_plan("us"),
    )

    result = ci_parity._gate_migration_authorization(
        run, _spec("migration_authorization")
    )

    assert result.status == "FAIL"
    assert (
        "ReviewedMigrationAuthorizationError: pull-request number is missing "
        "or malformed"
    ) in result.failures
    assert result.note is not None
    assert "--pull-request" in result.note


@pytest.mark.parametrize(
    "path", ["", "/abs/auth.json", "../auth.json", "a/./auth.json", "auth.yaml"]
)
def test_migration_authorization_requires_a_canonical_json_path(
    tmp_path: Path, path: str
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN}, remote=RULESPEC_US_REMOTE)
    run = _run(
        repo,
        tmp_path,
        number=911,
        inputs={"run-generated-guard": False, "migration-authorization-path": path},
        plan=_plan("us"),
    )

    result = ci_parity._gate_migration_authorization(
        run, _spec("migration_authorization")
    )

    assert result.status == "FAIL"
    assert (
        "ReviewedMigrationAuthorizationError: authorization path is not a "
        "canonical repository JSON path"
    ) in result.failures
    assert result.note is None


def _authorized_migration_repo(
    tmp_path: Path, *, pull_request: int = 911, waiver_digest: str = "b" * 64
) -> tuple[Path, str]:
    """rulespec-us with a candidate topic head authorized by the base commit."""

    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN}, remote=RULESPEC_US_REMOTE)
    _git(repo, "checkout", "-q", "-b", "topic")
    candidate = _commit(repo, {"us/statutes/a.yaml": CHANGED})
    _git(repo, "checkout", "-q", "main")
    authorization = {
        "format": "axiom/reviewed-migrations/v1",
        "migrations": [
            {
                "pull_request": pull_request,
                "head": candidate,
                "retired_schema_bootstrap_sha256": "a" * 64,
                "validation_waiver_bootstrap_sha256": waiver_digest,
            }
        ],
    }
    base = _commit(repo, {AUTHORIZATION_PATH: json.dumps(authorization, indent=2)})
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    _git(repo, "checkout", "-q", "topic")
    return repo, candidate


MIGRATION_INPUTS = {
    "run-generated-guard": False,
    "migration-authorization-path": AUTHORIZATION_PATH,
    "retired-schema-bootstrap-sha256": "a" * 64,
    "validation-waiver-bootstrap-sha256": "b" * 64,
}


def test_migration_authorization_accepts_the_exact_reviewed_candidate(
    tmp_path: Path,
) -> None:
    repo, candidate = _authorized_migration_repo(tmp_path)
    run = _run(repo, tmp_path, number=911, inputs=MIGRATION_INPUTS)
    assert run.simulation.head_sha == candidate

    result = ci_parity._gate_migration_authorization(
        run, _spec("migration_authorization")
    )

    assert result.status == "PASS", result.output
    assert run.migration_authorized == "true"
    assert run.migration_candidate == candidate
    assert result.output == f"authorized=true\ncandidate={candidate}\n"


@pytest.mark.parametrize(
    ("number", "inputs", "base_ref", "message"),
    [
        (
            912,
            MIGRATION_INPUTS,
            "origin/main",
            "candidate is not exactly authorized by the protected base",
        ),
        (
            911,
            {**MIGRATION_INPUTS, "validation-waiver-bootstrap-sha256": "c" * 64},
            "origin/main",
            "validation-waiver bootstrap digest differs from the protected "
            "authorization",
        ),
        (
            911,
            MIGRATION_INPUTS,
            "origin/release",
            "pull request must target the protected main branch",
        ),
        (
            911,
            {**MIGRATION_INPUTS, "migration-authorization-path": ".axiom/other.json"},
            "origin/main",
            "protected base does not contain exactly one authorization file",
        ),
    ],
    ids=["wrong-pr", "digest-differs", "not-main", "missing-file"],
)
def test_migration_authorization_rejects_an_inexact_candidate(
    tmp_path: Path,
    number: int,
    inputs: dict[str, Any],
    base_ref: str,
    message: str,
) -> None:
    repo, _ = _authorized_migration_repo(tmp_path)
    _git(repo, "update-ref", "refs/remotes/origin/release", "refs/remotes/origin/main")
    run = _run(repo, tmp_path, number=number, inputs=inputs, base_ref=base_ref)

    result = ci_parity._gate_migration_authorization(
        run, _spec("migration_authorization")
    )

    assert result.status == "FAIL"
    assert f"ReviewedMigrationAuthorizationError: {message}" in result.failures
    assert run.migration_authorized == ""


# ---------------------------------------------------------------------------
# Verify immutable retired-schema freeze (REAL heredoc)
# ---------------------------------------------------------------------------


def _freeze_gate(run: WorkflowRun) -> GateResult:
    return ci_parity._gate_retired_schema_freeze(run, _spec("retired_schema_freeze"))


def test_retired_schema_prefreeze_is_restricted_to_rulespec_us(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN}, remote=OTHER_REMOTE)
    run = _run(
        repo,
        tmp_path,
        inputs={"allow-retired-schema-prefreeze": True},
        plan=_plan("us"),
    )

    result = _freeze_gate(run)

    assert result.status == "FAIL"
    assert (
        "RetiredSchemaFreezeError: pre-freeze compatibility is restricted to "
        "rulespec-us"
    ) in result.failures


def test_retired_schema_prefreeze_on_rulespec_us_without_freeze_passes(
    tmp_path: Path,
) -> None:
    # Pre-freeze tolerates retired-schema modules that no freeze lists yet.
    repo = _repo(
        tmp_path,
        {"us/statutes/a.yaml": PLAIN, "us/statutes/x.yaml": RETIRED},
        remote=RULESPEC_US_REMOTE,
    )
    _commit(repo, {"us/statutes/a.yaml": CHANGED})
    run = _run(repo, tmp_path, inputs={"allow-retired-schema-prefreeze": True})

    result = _freeze_gate(run)

    assert result.status == "PASS", result.output
    assert run.retired_skip == frozenset()
    assert (run.temp / "retired-schema-skip.txt").read_text() == ""
    assert "Verified 0 immutable retired-schema module(s)." in result.output


@pytest.mark.parametrize(
    ("inputs", "files", "message"),
    [
        (
            {"allow-retired-schema-prefreeze": True},
            {FREEZE_PATH: _freeze({})},
            "pre-freeze compatibility cannot be used with a freeze",
        ),
        (
            {"allow-retired-schema-prefreeze": True, "run-generated-guard": False},
            {},
            "pre-freeze compatibility requires the generated guard",
        ),
        (
            {"run-generated-guard": False},
            {},
            "generated guard bypass requires an authorized bootstrap",
        ),
        (
            {"retired-schema-bootstrap-sha256": "d" * 64},
            {},
            "bootstrap is not bound to an exact protected authorization",
        ),
    ],
    ids=["prefreeze-with-freeze", "prefreeze-without-guard", "bypass", "bootstrap"],
)
def test_retired_schema_rulespec_us_policy_failures(
    tmp_path: Path, inputs: dict[str, Any], files: dict[str, str], message: str
) -> None:
    repo = _repo(
        tmp_path,
        {"us/statutes/a.yaml": PLAIN, **files},
        remote=RULESPEC_US_REMOTE,
    )
    run = _run(repo, tmp_path, inputs=inputs, plan=_plan("us"))
    run.migration_authorized = "false"

    result = _freeze_gate(run)

    assert result.status == "FAIL"
    assert f"RetiredSchemaFreezeError: {message}" in result.failures


@pytest.mark.parametrize("remote", [OTHER_REMOTE, RULESPEC_US_REMOTE])
def test_retired_schema_freeze_verifies_listed_modules_and_skips_them(
    tmp_path: Path, remote: str
) -> None:
    modules = {"us/statutes/x.yaml": RETIRED}
    repo = _repo(
        tmp_path,
        {**modules, "us/statutes/a.yaml": PLAIN, FREEZE_PATH: _freeze(modules)},
        remote=remote,
    )
    _commit(repo, {"us/statutes/a.yaml": CHANGED})
    run = _run(repo, tmp_path)

    result = _freeze_gate(run)

    assert result.status == "PASS", result.output
    assert "Verified 1 immutable retired-schema module(s)." in result.output
    assert run.retired_skip == frozenset({"us/statutes/x.yaml"})
    assert (run.temp / "retired-schema-skip.txt").read_text() == "us/statutes/x.yaml\n"


def test_retired_schema_freeze_allows_a_decrement(tmp_path: Path) -> None:
    modules = {"us/statutes/x.yaml": RETIRED, "us-ca/statutes/z.yaml": RETIRED}
    repo = _repo(tmp_path, {**modules, FREEZE_PATH: _freeze(modules)})
    _commit(
        repo,
        {
            "us-ca/statutes/z.yaml": PLAIN,
            FREEZE_PATH: _freeze({"us/statutes/x.yaml": RETIRED}),
        },
    )
    run = _run(repo, tmp_path)

    result = _freeze_gate(run)

    assert result.status == "PASS", result.output
    assert run.retired_skip == frozenset({"us/statutes/x.yaml"})


@pytest.mark.parametrize(
    ("base", "head", "message"),
    [
        (
            {
                "us/statutes/x.yaml": RETIRED,
                FREEZE_PATH: _freeze({"us/statutes/x.yaml": RETIRED}),
            },
            {"us/statutes/x.yaml": RETIRED + "# edited\n"},
            "frozen module digest mismatch for us/statutes/x.yaml",
        ),
        (
            {"us/statutes/x.yaml": RETIRED},
            {FREEZE_PATH: _freeze({"us/statutes/x.yaml": RETIRED})},
            "freeze is decrement-only; added=['us/statutes/x.yaml'], changed=[]",
        ),
        (
            {
                "us/statutes/x.yaml": RETIRED,
                FREEZE_PATH: _freeze({"us/statutes/x.yaml": RETIRED}),
            },
            {"us/statutes/x.yaml": PLAIN, FREEZE_PATH: None},
            "the sealed freeze file cannot be removed",
        ),
        (
            {FREEZE_PATH: _freeze({})},
            {"us/statutes/y.yaml": RETIRED},
            "inventory mismatch; unlisted=['us/statutes/y.yaml'], missing=[]",
        ),
        (
            {"us/statutes/x.yaml": PLAIN, FREEZE_PATH: _freeze({})},
            {FREEZE_PATH: _freeze({"us/statutes/x.yaml": PLAIN})},
            "module has no retired source-verification field: us/statutes/x.yaml",
        ),
    ],
    ids=["digest-mismatch", "added", "removed", "unlisted", "no-retired-field"],
)
def test_retired_schema_freeze_failures(
    tmp_path: Path,
    base: dict[str, str],
    head: dict[str, str | None],
    message: str,
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN, **base})
    _commit(repo, head)
    run = _run(repo, tmp_path, plan=_plan("us"))

    result = _freeze_gate(run)

    assert result.status == "FAIL"
    assert any(
        failure == f"RetiredSchemaFreezeError: {message}"
        or failure.startswith(f"RetiredSchemaFreezeError: {message}: ")
        for failure in result.failures
    ), result.failures
    assert run.retired_skip == frozenset()


def test_retired_schema_freeze_reads_the_head_commit_not_the_worktree(
    tmp_path: Path,
) -> None:
    modules = {"us/statutes/x.yaml": RETIRED}
    repo = _repo(tmp_path, {**modules, FREEZE_PATH: _freeze(modules)})
    _commit(repo, {"us/statutes/a.yaml": PLAIN})
    _write(repo, {"us/statutes/x.yaml": "uncommitted: edit\n"})
    run = _run(repo, tmp_path)

    result = _freeze_gate(run)

    assert result.status == "PASS", result.output
    assert run.retired_skip == frozenset({"us/statutes/x.yaml"})


@pytest.mark.parametrize(
    ("authorized", "status"), [("true", "PASS"), ("false", "FAIL"), ("", "FAIL")]
)
def test_retired_schema_bootstrap_is_bound_to_the_migration_authorization(
    tmp_path: Path, authorized: str, status: str
) -> None:
    modules = {"us/statutes/x.yaml": RETIRED}
    freeze = _freeze(modules)
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN}, remote=RULESPEC_US_REMOTE)
    _commit(repo, {**modules, FREEZE_PATH: freeze})
    run = _run(
        repo,
        tmp_path,
        inputs={
            "run-generated-guard": False,
            "retired-schema-bootstrap-sha256": _sha256(freeze),
        },
    )
    run.migration_authorized = authorized

    result = _freeze_gate(run)

    assert result.status == status, result.output
    if status == "PASS":
        assert run.retired_skip == frozenset({"us/statutes/x.yaml"})
    else:
        assert (
            "RetiredSchemaFreezeError: bootstrap is not bound to an exact "
            "protected authorization"
        ) in result.failures
        assert run.retired_skip == frozenset()


# ---------------------------------------------------------------------------
# Reject obsolete generated files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("files", "listed"),
    [
        ({"us/statutes/a.rac": ""}, "./us/statutes/a.rac"),
        ({"a.rac.test": ""}, "./a.rac.test"),
        ({"us/.venv/lib/b.rac": ""}, "./us/.venv/lib/b.rac"),
        ({"us/_axiom/c.rac.test": ""}, "./us/_axiom/c.rac.test"),
        ({"x.rac/keep.txt": ""}, "./x.rac"),
    ],
    ids=["nested-file", "root-test-file", "nested-venv", "nested-axiom", "directory"],
)
def test_obsolete_files_are_found_anywhere_but_pruned_roots(
    tmp_path: Path, files: dict[str, str], listed: str
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN})
    run = _run(repo, tmp_path, plan=_plan("us"))
    _write(repo, files)

    result = ci_parity._gate_obsolete_files(run, _spec("obsolete_files"))

    assert result.status == "FAIL"
    assert listed in result.failures


def test_obsolete_files_prune_top_level_tool_directories(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN})
    run = _run(repo, tmp_path, plan=_plan("us"))
    _write(
        repo,
        {
            ".git/x.rac": "",
            "_axiom/axiom-encode/y.rac": "",
            ".venv/lib/z.rac.test": "",
            ".pytest_cache/w.rac": "",
            "us/statutes/a.rac.yaml": "",
        },
    )

    result = ci_parity._gate_obsolete_files(run, _spec("obsolete_files"))

    assert result.status == "PASS", result.output
    assert result.output == "No obsolete generated files.\n"


# ---------------------------------------------------------------------------
# Reject disallowed repository layout
# ---------------------------------------------------------------------------

STRUCTURE = """\
version: 1
allowed_root_directories: [".axiom", ".github", "us"]
allowed_root_files: ["known-validation-gaps.yaml"]
path_rules:
  - patterns: [".axiom/*"]
    allow_extensions: [".yaml", ".toml", ".json"]
  - patterns: [".github/*"]
    allow_extensions: [".yml"]
  - patterns: ["us/*"]
    allow_extensions: [".yaml"]
"""


def _layout(run: WorkflowRun) -> GateResult:
    return ci_parity._gate_repository_layout(run, _spec("repository_layout"))


@pytest.mark.parametrize(
    ("extra", "status", "problem"),
    [
        ({}, "PASS", None),
        # The configured structure replaces the legacy layout rules.
        ({"us/statutes/parameters.yaml": PLAIN}, "PASS", None),
        (
            {"docs/notes.md": "notes\n"},
            "FAIL",
            "- docs/notes.md: top-level directory docs/ is not allowed",
        ),
        (
            {"README.md": "readme\n"},
            "FAIL",
            "- README.md: top-level file is not allowed",
        ),
        (
            {"us/statutes/a.txt": "text\n"},
            "FAIL",
            "- us/statutes/a.txt: file name/extension is not allowed by matched "
            "path rule",
        ),
    ],
    ids=["valid", "legacy-name-allowed", "directory", "root-file", "extension"],
)
def test_layout_with_repository_structure_runs_the_workflow_script(
    tmp_path: Path, extra: dict[str, str], status: str, problem: str | None
) -> None:
    repo = _repo(
        tmp_path,
        {
            ".axiom/repository-structure.yaml": STRUCTURE,
            LEDGER: "validate_failures: {}\n",
            "us/statutes/a.yaml": PLAIN,
            ".github/workflows/ci.yml": "on: push\n",
            **extra,
        },
    )
    run = _run(repo, tmp_path)

    result = _layout(run)

    assert result.status == status, result.output
    if problem is None:
        assert (
            "Repository layout matches .axiom/repository-structure.yaml."
            in result.output
        )
    else:
        assert problem in result.failures


def test_layout_with_repository_structure_judges_tracked_files_only(
    tmp_path: Path,
) -> None:
    repo = _repo(
        tmp_path,
        {".axiom/repository-structure.yaml": STRUCTURE, "us/statutes/a.yaml": PLAIN},
    )
    run = _run(repo, tmp_path)
    _write(repo, {"docs/untracked.yaml": PLAIN})

    assert _layout(run).status == "PASS"


@pytest.mark.parametrize(
    ("extra", "status", "listed"),
    [
        ({}, "PASS", None),
        ({"us/statutes/sub/b.yml": PLAIN}, "PASS", None),
        ({".github/workflows/ci.yml": "on: push\n"}, "PASS", None),
        ({"sources/us/s.yaml": PLAIN}, "PASS", None),
        ({"programs/p.yaml": PLAIN}, "PASS", None),
        ({LEDGER: "validate_failures: {}\n"}, "PASS", None),
        ({"docs/known-dangling.yaml": PLAIN}, "PASS", None),
        ({".venv/lib/v.yaml": PLAIN}, "PASS", None),
        ({"_axiom/dep/v.yaml": PLAIN}, "PASS", None),
        ({"us/notes.md": "notes\n"}, "PASS", None),
        (
            {"us/statutes/parameters.yaml": PLAIN},
            "FAIL",
            "./us/statutes/parameters.yaml",
        ),
        ({"us/statutes/tests.yaml": PLAIN}, "FAIL", "./us/statutes/tests.yaml"),
        ({"tests/cases.yaml": PLAIN}, "FAIL", "tests/cases.yaml"),
        ({"docs/config.yaml": PLAIN}, "FAIL", "./docs/config.yaml"),
        ({"config.yml": PLAIN}, "FAIL", "./config.yml"),
        ({".axiom/index/us.yaml": PLAIN}, "FAIL", "./.axiom/index/us.yaml"),
        ({"statute/readme.txt": "x\n"}, "FAIL", "statute/"),
        ({"policy": "x\n"}, "FAIL", "policy/"),
        ({"x/us/a.yaml": PLAIN}, "FAIL", "./x/us/a.yaml"),
    ],
)
def test_legacy_layout_rules_without_repository_structure(
    tmp_path: Path, extra: dict[str, str], status: str, listed: str | None
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN, **extra})
    run = _run(repo, tmp_path)
    assert run.plan.roots == "us"
    assert run.plan.allowed_extra == "sources programs"

    result = _layout(run)

    assert result.status == status, result.output
    assert result.command == ["layout", "us", "sources programs"]
    if listed is not None:
        assert listed in result.failures
        assert result.failures[0] == (
            "Repository layout is not allowed. Use RuleSpec YAML under "
            "statutes/, regulations/, or policies/."
        )


def test_legacy_layout_flags_tests_yaml_even_when_tests_is_a_root(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, {"statutes/a.yaml": PLAIN, "tests/cases.yaml": PLAIN})
    run = _run(repo, tmp_path, inputs={"validate-roots": "statutes tests"})

    result = _layout(run)

    assert result.status == "FAIL"
    assert "tests/cases.yaml" in result.failures
    assert "./tests/cases.yaml" not in result.failures


@pytest.mark.parametrize(("guard", "status"), [(False, "FAIL"), (True, "PASS")])
def test_explicit_roots_tolerate_programs_only_when_guarded(
    tmp_path: Path, guard: bool, status: str
) -> None:
    repo = _repo(tmp_path, {"statutes/a.yaml": PLAIN, "programs/p.yaml": PLAIN})
    run = _run(
        repo,
        tmp_path,
        inputs={"validate-roots": "statutes", "guard-programs-root": guard},
    )

    result = _layout(run)

    assert result.status == status, result.output
    assert ("./programs/p.yaml" in result.failures) is (not guard)


# ---------------------------------------------------------------------------
# Enforce validation waiver ratchet (REAL ratchet heredoc; audit recorded)
# ---------------------------------------------------------------------------

JURISDICTION_FILES = {
    "us/statutes/a.yaml": PLAIN,
    "us/statutes/b.yaml": PLAIN,
    "us-ca/statutes/c.yaml": PLAIN,
}


def _waiver_repo(
    tmp_path: Path,
    entries: Mapping[str, Any],
    *,
    ledger: bool = True,
    toolchain: bool = True,
) -> tuple[Path, bytes]:
    base_ledger = _ledger(entries)
    files: dict[str, str | bytes] = dict(JURISDICTION_FILES)
    if ledger:
        files[LEDGER] = base_ledger
    if toolchain:
        files[TOOLCHAIN] = _toolchain(base_ledger)
    return _repo(tmp_path, files), base_ledger


def _repin(entries: Mapping[str, Any], **toolchain: str) -> dict[str, str | bytes]:
    head_ledger = _ledger(entries)
    return {LEDGER: head_ledger, TOOLCHAIN: _toolchain(head_ledger, **toolchain)}


class _AuditRecorder(_BatchRecorder):
    """Records audits and snapshots the files they are handed."""

    def __init__(self) -> None:
        super().__init__()
        self.protected: list[bytes] = []
        self.changed: list[bytes] = []

    def __call__(
        self,
        invocations: list[CliInvocation],
        *,
        jobs: int,
        keyring: tuple[str, ...],
        signing_roots: dict[str, str] | None = None,
    ) -> list[tuple[int, str]]:
        for invocation in invocations:
            arguments = invocation.arguments
            protected = arguments[arguments.index("--protected-base") + 1]
            changed = arguments[arguments.index("--changed-paths") + 1]
            self.protected.append(Path(protected).read_bytes())
            self.changed.append(Path(changed).read_bytes())
        return super().__call__(invocations, jobs=jobs, keyring=keyring)


def _waiver_gate(
    run: WorkflowRun, monkeypatch: pytest.MonkeyPatch, sha: str = PIN_0EFFA6A5
) -> tuple[GateResult, _AuditRecorder]:
    recorder = _AuditRecorder()
    monkeypatch.setattr(ci_parity, "_run_cli_batch", recorder)
    return (
        ci_parity._gate_validation_waivers(run, _spec("validation_waivers", sha)),
        recorder,
    )


def _expected_audit(run: WorkflowRun, shard: str) -> CliInvocation:
    return CliInvocation(
        (
            "validation-waivers",
            "audit",
            "--root",
            str(run.repo),
            "--corpus-path",
            str(run.paths["corpus"]),
            "--protected-base",
            str(run.temp / "protected-known-validation-gaps.yaml"),
            "--changed-paths",
            str(run.temp / "waiver-audit-changed-paths.txt"),
            "--partition-key",
            shard,
            "--partition-keys-json",
            run.plan.matrix_json,
            "--axiom-rules-engine-path",
            str(run.paths["engine"]),
        ),
        cwd=run.repo,
        supervised=True,
    )


@EMBEDDED_PINS
def test_waiver_ratchet_growth_fails_before_any_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str
) -> None:
    base = {"us/statutes/a.yaml": {"active": _record("a")}}
    repo, _ = _waiver_repo(tmp_path, base)
    _commit(repo, _repin({**base, "us/statutes/b.yaml": {"active": _record("b")}}))
    run = _run(repo, tmp_path, sha=pin)

    result, recorder = _waiver_gate(run, monkeypatch, pin)

    assert result.status == "FAIL"
    assert any(
        failure.startswith("ValidationWaiverRatchetGrowth")
        and "us/statutes/b.yaml:active" in failure
        for failure in result.failures
    ), result.failures
    assert recorder.invocations == []


@EMBEDDED_PINS
def test_waiver_ratchet_decrement_audits_every_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str
) -> None:
    base = {
        "us/statutes/a.yaml": {"active": _record("a")},
        "us/statutes/b.yaml": {"active": _record("b")},
    }
    repo, base_ledger = _waiver_repo(tmp_path, base)
    _commit(
        repo,
        {
            **_repin({"us/statutes/b.yaml": {"active": _record("b")}}),
            "us/statutes/a.yaml": CHANGED,
        },
    )
    run = _run(repo, tmp_path, sha=pin)
    assert run.plan.matrix == ("us-ca", "us")

    result, recorder = _waiver_gate(run, monkeypatch, pin)

    assert result.status == "PASS", result.output
    assert (
        "Validation waiver ratchet is decrement-only: 2 base module(s) -> 1 head "
        "module(s)."
    ) in result.output
    assert recorder.invocations == [
        _expected_audit(run, "us-ca"),
        _expected_audit(run, "us"),
    ]
    assert json.loads(run.plan.matrix_json) == ["us-ca", "us"]
    assert recorder.protected == [base_ledger, base_ledger]
    # Not a companion-only PR, so the audit sees every changed path.
    changed = f"{TOOLCHAIN}\n{LEDGER}\nus/statutes/a.yaml\n".encode()
    assert recorder.changed == [changed, changed]
    assert (run.temp / "waiver-changed-paths.txt").read_bytes() == changed


@EMBEDDED_PINS
def test_waiver_pending_approval_may_grow_only_in_a_digest_only_companion_pr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str
) -> None:
    base = {"us/statutes/a.yaml": {"active": _record("a")}}
    repo, base_ledger = _waiver_repo(tmp_path, base)
    _commit(repo, _repin({**base, "us/statutes/b.yaml": {"pending": _record("b")}}))
    run = _run(repo, tmp_path, sha=pin)

    result, recorder = _waiver_gate(run, monkeypatch, pin)

    assert result.status == "PASS", result.output
    assert len(recorder.invocations) == len(run.plan.matrix) == 2
    assert recorder.protected == [base_ledger] * 2
    # The digest-only toolchain edit is dropped from the audit's changed paths.
    assert recorder.changed == [b"known-validation-gaps.yaml\n"] * 2
    assert (
        run.temp / "waiver-changed-paths.txt"
    ).read_bytes() == f"{TOOLCHAIN}\n{LEDGER}\n".encode()


def test_waiver_pending_growth_alongside_other_changes_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = {"us/statutes/a.yaml": {"active": _record("a")}}
    repo, _ = _waiver_repo(tmp_path, base)
    _commit(
        repo,
        {
            **_repin({**base, "us/statutes/b.yaml": {"pending": _record("b")}}),
            "us/statutes/b.yaml": CHANGED,
        },
    )
    run = _run(repo, tmp_path)

    result, recorder = _waiver_gate(run, monkeypatch)

    assert result.status == "FAIL"
    assert any(
        "ValidationWaiverRatchetGrowth" in failure
        and "us/statutes/b.yaml:pending" in failure
        for failure in result.failures
    ), result.failures
    assert recorder.invocations == []


def test_waiver_consuming_a_protected_pending_approval_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = _waiver_repo(tmp_path, {"us/statutes/b.yaml": {"pending": _record("b")}})
    _commit(
        repo,
        {
            **_repin({"us/statutes/b.yaml": {"active": _record("b")}}),
            "us/statutes/b.yaml": CHANGED,
        },
    )
    run = _run(repo, tmp_path)

    result, recorder = _waiver_gate(run, monkeypatch)

    assert result.status == "PASS", result.output
    assert len(recorder.invocations) == 2


def test_waiver_companion_pin_must_match_the_ledger_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = {"us/statutes/a.yaml": {"active": _record("a")}}
    repo, base_ledger = _waiver_repo(tmp_path, base)
    _commit(repo, {LEDGER: _ledger({}), TOOLCHAIN: _toolchain(base_ledger + b"#\n")})
    run = _run(repo, tmp_path)

    result, recorder = _waiver_gate(run, monkeypatch)

    assert result.status == "FAIL"
    assert (
        "ValidationWaiverPinCompanionMismatch: waiver ledger digest pin is not exact"
        in result.failures
    )
    assert recorder.invocations == []


@pytest.mark.parametrize(
    ("pin", "status"), [(PIN_0EFFA6A5, "FAIL"), (PIN_6F11BE26, "PASS")]
)
def test_waiver_corpus_release_repin_with_a_retirement_is_pin_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str, status: str
) -> None:
    base = {
        "us/statutes/a.yaml": {"active": _record("a")},
        "us/statutes/b.yaml": {"active": _record("b")},
    }
    repo, base_ledger = _waiver_repo(tmp_path, base)
    _commit(
        repo,
        _repin(
            {"us/statutes/b.yaml": {"active": _record("b")}},
            release="us-rulespec-2026-09-15",
            content="d" * 64,
        ),
    )
    run = _run(repo, tmp_path, sha=pin)

    result, recorder = _waiver_gate(run, monkeypatch, pin)

    assert result.status == status, result.output
    if pin == PIN_0EFFA6A5:
        assert (
            "ValidationWaiverPinCompanionScope: toolchain companion may change only "
            "the waiver-set digest"
        ) in result.failures
        assert recorder.invocations == []
    else:
        assert (
            "Corpus release re-pin ledger transition: 1 retired, 0 staged "
            "approval(s) consumed."
        ) in result.output
        assert recorder.protected == [base_ledger] * 2
        # A release re-pin keeps the full changed paths for the audit.
        assert recorder.changed == [f"{TOOLCHAIN}\n{LEDGER}\n".encode()] * 2


def test_waiver_corpus_release_repin_cannot_add_records_on_6f11be26(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = {"us/statutes/a.yaml": {"active": _record("a")}}
    repo, _ = _waiver_repo(tmp_path, base)
    _commit(
        repo,
        _repin(
            {**base, "us/statutes/b.yaml": {"pending": _record("b")}},
            release="us-rulespec-2026-09-15",
            content="d" * 64,
        ),
    )
    run = _run(repo, tmp_path, sha=PIN_6F11BE26)

    result, recorder = _waiver_gate(run, monkeypatch, PIN_6F11BE26)

    assert result.status == "FAIL"
    assert any(
        failure.startswith("ValidationWaiverPinCompanionScope")
        and "us/statutes/b.yaml (new entry)" in failure
        for failure in result.failures
    ), result.failures
    assert recorder.invocations == []


@pytest.mark.parametrize(
    ("ledger", "toolchain", "message"),
    [
        (
            True,
            False,
            "ValidationWaiverBaseToolchainMissing: protected base must contain "
            ".axiom/toolchain.toml",
        ),
        (
            False,
            True,
            "ValidationWaiverBaseMissing: protected base must contain "
            "known-validation-gaps.yaml",
        ),
    ],
    ids=["no-base-toolchain", "no-base-ledger"],
)
def test_waiver_protected_base_must_carry_ledger_and_toolchain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ledger: bool,
    toolchain: bool,
    message: str,
) -> None:
    repo, _ = _waiver_repo(tmp_path, {}, ledger=ledger, toolchain=toolchain)
    _commit(repo, _repin({}))
    run = _run(repo, tmp_path)

    result, recorder = _waiver_gate(run, monkeypatch)

    assert result.status == "FAIL"
    assert result.failures == [message]
    assert recorder.invocations == []


@pytest.mark.parametrize("authorized", ["", "false"])
def test_waiver_bootstrap_requires_an_exact_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, authorized: str
) -> None:
    repo, base_ledger = _waiver_repo(tmp_path, {})
    run = _run(
        repo,
        tmp_path,
        inputs={"validation-waiver-bootstrap-sha256": _sha256(base_ledger)},
        plan=_plan("us"),
    )
    run.migration_authorized = authorized

    result, recorder = _waiver_gate(run, monkeypatch)

    assert result.status == "FAIL"
    assert result.failures == [
        "ValidationWaiverBootstrapUnauthorized: bootstrap is not bound to an "
        "exact protected authorization"
    ]
    assert recorder.invocations == []


def _bootstrap_run(tmp_path: Path, head_ledger: bytes) -> tuple[WorkflowRun, bytes]:
    # A bootstrap needs neither a protected-base ledger nor toolchain: the
    # authorized candidate's own ledger and toolchain become the comparison.
    repo = _repo(tmp_path, dict(JURISDICTION_FILES))
    _commit(repo, {LEDGER: head_ledger, TOOLCHAIN: _toolchain(head_ledger)})
    run = _run(
        repo,
        tmp_path,
        inputs={"validation-waiver-bootstrap-sha256": _sha256(head_ledger)},
        plan=_plan("us"),
    )
    run.migration_authorized = "true"
    run.migration_candidate = run.simulation.head_sha
    return run, head_ledger


def test_waiver_bootstrap_uses_the_authorized_candidate_ledger_as_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, head_ledger = _bootstrap_run(
        tmp_path, _ledger({"us/statutes/a.yaml": {"active": _record("a")}})
    )

    result, recorder = _waiver_gate(run, monkeypatch)

    assert result.status == "PASS", result.output
    assert recorder.invocations == [_expected_audit(run, "us")]
    assert recorder.protected == [head_ledger]
    assert (run.temp / "protected-toolchain.toml").read_bytes() == (
        run.repo / TOOLCHAIN
    ).read_bytes()


def test_waiver_bootstrap_rejects_a_ledger_that_differs_from_the_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, head_ledger = _bootstrap_run(tmp_path, _ledger({}))
    _write(run.repo, {LEDGER: head_ledger + b"# local edit\n"})

    result, recorder = _waiver_gate(run, monkeypatch)

    assert result.status == "FAIL"
    assert result.failures[0].startswith("ValidationWaiverBootstrapHashMismatch")
    assert recorder.invocations == []


def test_waiver_bootstrap_hashes_the_authorized_candidate_not_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = _bootstrap_run(tmp_path, _ledger({}))
    # The protected base commit has no ledger at all.
    run.migration_candidate = run.simulation.base_sha

    result, recorder = _waiver_gate(run, monkeypatch)

    assert result.status == "FAIL"
    assert result.failures == [
        "ValidationWaiverBootstrapCandidateMismatch: candidate waiver hash does "
        "not match bootstrap"
    ]
    assert recorder.invocations == []


def test_waiver_bootstrap_hashes_candidate_ledger_bytes_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The workflow pipes raw `git show` bytes into sha256sum; a text-mode read
    # (universal newlines) would turn CRLF into LF and reject this bootstrap.
    crlf = b'validate_failures:\r\n  us/statutes/a.yaml:\r\n    active: {fingerprint: "x"}\r\n'
    assert yaml.safe_load(crlf) == {
        "validate_failures": {"us/statutes/a.yaml": {"active": {"fingerprint": "x"}}}
    }
    run, _ = _bootstrap_run(tmp_path, crlf)
    # The workflow's `git show <candidate>:known-validation-gaps.yaml | sha256sum`.
    shown = subprocess.run(
        ["git", "-C", str(run.repo), "show", f"{run.migration_candidate}:{LEDGER}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert _sha256(shown) == run.caller.inputs["validation-waiver-bootstrap-sha256"]

    result, _ = _waiver_gate(run, monkeypatch)

    assert result.status == "PASS", result.output


# ---------------------------------------------------------------------------
# Reject manual RuleSpec changes
# ---------------------------------------------------------------------------


@EMBEDDED_PINS
def test_guard_generated_disabled_by_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN})
    run = _run(
        repo, tmp_path, sha=pin, inputs={"run-generated-guard": False}, plan=_plan("us")
    )
    recorder = _patch_cli(monkeypatch)

    result = ci_parity._gate_guard_generated(run, _spec("guard_generated", pin))

    assert result.status == "PASS"
    assert result.command == []
    assert result.output == "Disabled by caller run-generated-guard.\n"
    assert recorder.calls == []


@EMBEDDED_PINS
@pytest.mark.parametrize(("code", "status"), [(0, "PASS"), (1, "FAIL")])
def test_guard_generated_runs_one_supervised_invocation_on_exact_shas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pin: str,
    code: int,
    status: str,
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN})
    head = _commit(repo, {"us/statutes/a.yaml": CHANGED})
    run = _run(repo, tmp_path, sha=pin)
    recorder = _patch_cli(monkeypatch, (code, "guard said so\n"))

    result = ci_parity._gate_guard_generated(run, _spec("guard_generated", pin))

    assert result.status == status
    expected = [
        "guard-generated",
        "--repo",
        str(repo),
        "--base-ref",
        run.simulation.base_sha,
        "--head-ref",
        head,
        "--corpus-path",
        str(run.paths["corpus"]),
        "--expected-encoder-checkout",
        str(run.paths["encode"]),
    ]
    assert recorder.calls == [
        {"arguments": expected, "environment": None, "cwd": repo, "supervised": True}
    ]
    assert result.command == expected
    assert result.failures == ([] if code == 0 else ["guard said so"])


@pytest.mark.parametrize(
    ("pin", "status", "calls"), [(PIN_0EFFA6A5, "PASS", 1), (PIN_6F11BE26, "FAIL", 0)]
)
def test_guard_generated_exact_base_ref_is_enforced_from_6f11be26(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str, status: str, calls: int
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN})
    run = _run(repo, tmp_path, sha=pin, plan=_plan("us"))
    run.simulation = dataclasses.replace(run.simulation, base_sha="HEAD~1")
    recorder = _patch_cli(monkeypatch)

    result = ci_parity._gate_guard_generated(run, _spec("guard_generated", pin))

    assert result.status == status
    assert len(recorder.calls) == calls
    if status == "FAIL":
        assert result.failures == [
            "GeneratedGuardBaseRef: base ref must be an exact 40-hex commit, "
            "got 'HEAD~1'"
        ]


# ---------------------------------------------------------------------------
# Select RuleSpec validation targets (REAL classify heredoc)
# ---------------------------------------------------------------------------

SELECTION_LAYOUT = {
    "us/statutes/a.yaml": PLAIN,
    "us/statutes/a.test.yaml": "cases: []\n",
    "us/statutes/b.yaml": PLAIN,
    "us/statutes/b.test.yaml": "cases: []\n",
    "us/regulations/r.yaml": PLAIN,
    "us/manual/m.yaml": PLAIN,
    "us/programs/p.yaml": PLAIN,
    "us/statutes/programs/q.yaml": PLAIN,
    "us-ca/regulations/c.yml": PLAIN,
    "us-ca/regulations/c.test.yml": "cases: []\n",
}


def _select(run: WorkflowRun, sha: str = PIN_0EFFA6A5) -> GateResult:
    result = ci_parity._gate_select_targets(run, _spec("select_targets", sha))
    assert result.status == "PASS", result.output
    return result


def _selected(run: WorkflowRun) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    return {
        shard: (selection.rulespec_files, selection.test_files)
        for shard, selection in run.selections.items()
    }


def test_select_targets_pairs_companions_and_drops_non_canonical_paths(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, SELECTION_LAYOUT)
    _commit(
        repo,
        {
            "us/statutes/a.test.yaml": "cases: [1]\n",
            "us/statutes/b.yaml": CHANGED,
            "us/manual/m.yaml": CHANGED,
            "us/programs/p.yaml": CHANGED,
            "us/statutes/programs/q.yaml": CHANGED,
            "us/statutes/notes.md": "notes\n",
        },
    )
    run = _run(repo, tmp_path)
    assert run.plan.matrix == ("us",)

    result = _select(run)

    assert run.mode == "changed"
    assert _selected(run) == {
        "us": (
            ("us/statutes/a.yaml", "us/statutes/b.yaml"),
            ("us/statutes/a.test.yaml", "us/statutes/b.test.yaml"),
        )
    }
    assert run.selections["us"].roots == ("us",)
    assert result.command[-2:] == ["--mode", "changed"]
    assert run.changed == (
        "us/manual/m.yaml",
        "us/programs/p.yaml",
        "us/statutes/a.test.yaml",
        "us/statutes/b.yaml",
        "us/statutes/notes.md",
        "us/statutes/programs/q.yaml",
    )


def test_select_targets_pairs_yml_companions_and_skips_missing_files(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, SELECTION_LAYOUT)
    _commit(
        repo,
        {"us-ca/regulations/c.yml": CHANGED, "us-ca/regulations/c.test.yml": None},
    )
    run = _run(repo, tmp_path)
    assert run.plan.matrix == ("us-ca",)

    _select(run)

    # The deleted companion is a changed path but no longer a file.
    assert _selected(run) == {"us-ca": (("us-ca/regulations/c.yml",), ())}


@pytest.mark.parametrize(
    "change",
    [
        {".github/workflows/ci.yml": "on: push\n"},
        {".github/workflows/nested/ci.yaml": "on: push\n"},
    ],
    ids=["yml", "nested-yaml"],
)
def test_workflow_change_selects_every_canonical_file_of_every_shard(
    tmp_path: Path, change: dict[str, str]
) -> None:
    repo = _repo(tmp_path, SELECTION_LAYOUT)
    _commit(repo, change)
    run = _run(repo, tmp_path)
    assert run.plan.matrix == ("us-ca", "us")

    _select(run)

    assert run.mode == "full-toolchain-bump"
    assert _selected(run) == {
        "us": (
            ("us/regulations/r.yaml", "us/statutes/a.yaml", "us/statutes/b.yaml"),
            ("us/statutes/a.test.yaml", "us/statutes/b.test.yaml"),
        ),
        "us-ca": (("us-ca/regulations/c.yml",), ("us-ca/regulations/c.test.yml",)),
    }


def _toolchain_change_repo(
    tmp_path: Path, base_toolchain: str | None, head_toolchain: str
) -> WorkflowRun:
    files: dict[str, str] = dict(SELECTION_LAYOUT)
    if base_toolchain is not None:
        files[TOOLCHAIN] = base_toolchain
    repo = _repo(tmp_path, files)
    _commit(repo, {TOOLCHAIN: head_toolchain, "us/statutes/a.yaml": CHANGED})
    run = _run(repo, tmp_path)
    assert run.plan.matrix == ("us-ca", "us")
    return run


LEDGER_BYTES = b"validate_failures: {}\n"


@EMBEDDED_PINS
def test_corpus_only_toolchain_change_validates_changed_files_only(
    tmp_path: Path, pin: str
) -> None:
    run = _toolchain_change_repo(
        tmp_path,
        _toolchain(LEDGER_BYTES),
        _toolchain(LEDGER_BYTES, release="us-rulespec-2026-09-15", content="d" * 64),
    )

    _select(run, pin)

    assert run.mode == "changed-corpus-toolchain-bump"
    assert _selected(run) == {
        "us": (("us/statutes/a.yaml",), ("us/statutes/a.test.yaml",)),
        "us-ca": ((), ()),
    }


@pytest.mark.parametrize(
    ("base_toolchain", "head_toolchain"),
    [
        (_toolchain(LEDGER_BYTES), _toolchain(LEDGER_BYTES + b"#\n")),
        (
            _toolchain(LEDGER_BYTES),
            _toolchain(LEDGER_BYTES + b"#\n", release="us-rulespec-2026-09-15"),
        ),
        (None, _toolchain(LEDGER_BYTES)),
        (_toolchain(LEDGER_BYTES), "[toolchain\nnot toml\n"),
        (_toolchain(LEDGER_BYTES), _toolchain(LEDGER_BYTES) + 'extra = "x"\n'),
    ],
    ids=["waiver-digest", "digest-and-release", "no-base", "invalid-head", "new-key"],
)
def test_other_toolchain_changes_select_everything(
    tmp_path: Path, base_toolchain: str | None, head_toolchain: str
) -> None:
    run = _toolchain_change_repo(tmp_path, base_toolchain, head_toolchain)

    _select(run)

    assert run.mode == "full-toolchain-bump"
    assert run.selections["us"].rulespec_files == (
        "us/regulations/r.yaml",
        "us/statutes/a.yaml",
        "us/statutes/b.yaml",
    )
    assert run.selections["us-ca"].rulespec_files == ("us-ca/regulations/c.yml",)


def test_workflow_change_wins_over_a_corpus_only_toolchain_change(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, {**SELECTION_LAYOUT, TOOLCHAIN: _toolchain(LEDGER_BYTES)})
    _commit(
        repo,
        {
            TOOLCHAIN: _toolchain(LEDGER_BYTES, release="us-rulespec-2026-09-15"),
            ".github/workflows/ci.yml": "on: push\n",
        },
    )
    run = _run(repo, tmp_path)

    _select(run)

    assert run.mode == "full-toolchain-bump"


def test_explicit_roots_have_no_canonical_filter_but_drop_programs(
    tmp_path: Path,
) -> None:
    repo = _repo(
        tmp_path,
        {
            "statutes/x.yaml": PLAIN,
            "statutes/x.test.yaml": "cases: []\n",
            "manual/m.yaml": PLAIN,
            "statutes/programs/p.yaml": PLAIN,
            "programs/q.yaml": PLAIN,
            "other/o.yaml": PLAIN,
        },
    )
    _commit(
        repo,
        {
            "statutes/x.yaml": CHANGED,
            "manual/m.yaml": CHANGED,
            "statutes/programs/p.yaml": CHANGED,
            "programs/q.yaml": CHANGED,
            "other/o.yaml": CHANGED,
        },
    )
    run = _run(
        repo,
        tmp_path,
        inputs={
            "validate-roots": "statutes manual programs",
            "guard-programs-root": True,
        },
    )

    _select(run)

    assert run.mode == "changed"
    assert _selected(run) == {
        "__all__": (("manual/m.yaml", "statutes/x.yaml"), ("statutes/x.test.yaml",))
    }
    assert run.selections["__all__"].roots == ("statutes", "manual", "programs")


# ---------------------------------------------------------------------------
# Validate RuleSpec YAML (REAL skip heredoc; validator recorded)
# ---------------------------------------------------------------------------


def _validate_invocation(run: WorkflowRun, files: list[str]) -> CliInvocation:
    return CliInvocation(
        (
            "validate",
            *(str(run.repo / file) for file in files),
            "--skip-reviewers",
            "--corpus-path",
            str(run.paths["corpus"]),
            "--axiom-rules-engine-path",
            str(run.paths["engine"]),
        ),
        cwd=run.repo,
        supervised=True,
    )


def _module_run(
    tmp_path: Path,
    selections: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]],
    *,
    ledger: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
    retired: frozenset[str] = frozenset(),
) -> WorkflowRun:
    files: dict[str, str | bytes] = {"us/statutes/placeholder.yaml": PLAIN}
    for rules, tests in selections.values():
        files.update(dict.fromkeys(rules, PLAIN))
        files.update(dict.fromkeys(tests, "cases: []\n"))
    if ledger is not None:
        files[LEDGER] = _ledger(ledger)
    repo = _repo(tmp_path, files)
    run = _run(repo, tmp_path, inputs=inputs, plan=_plan(*selections))
    run.selections = {
        shard: ShardSelection(shard, (shard,), rules, tests)
        for shard, (rules, tests) in selections.items()
    }
    run.retired_skip = retired
    return run


def _validate(run: WorkflowRun) -> GateResult:
    return ci_parity._gate_validate(run, _spec("validate"))


def test_validate_skips_only_active_waivers_and_retired_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = tuple(f"us/statutes/{name}.yaml" for name in "abcdef")
    run = _module_run(
        tmp_path,
        {"us": (files, ())},
        ledger={
            "us/statutes/a.yaml": {"active": _record("a")},
            "us/statutes/b.yaml": {"pending": _record("b")},
            "us/statutes/c.yaml": {"active": _record("c"), "pending": _record("c2")},
        },
        retired=frozenset({"us/statutes/d.yaml"}),
    )
    recorder = _patch_batch(monkeypatch)

    result = _validate(run)

    assert result.status == "PASS", result.output
    assert recorder.invocations == [
        _validate_invocation(
            run, ["us/statutes/b.yaml", "us/statutes/e.yaml", "us/statutes/f.yaml"]
        )
    ]
    for skipped in ("a", "c", "d"):
        assert (
            f"SKIPPED (known-validation-gaps validate_failures): us/statutes/{skipped}.yaml"
            in result.output
        )
    assert "us/statutes/b.yaml\n" not in result.output
    assert set(run.validate_skip) == {"us"}
    assert result.note is not None and result.note.startswith("validation-workers=1")


def test_validate_splits_files_round_robin_across_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = tuple(f"us/statutes/{name}.yaml" for name in "abcde")
    run = _module_run(tmp_path, {"us": (files, ())}, inputs={"validation-workers": 4})
    recorder = _patch_batch(monkeypatch)

    result = _validate(run)

    assert result.status == "PASS", result.output
    assert recorder.invocations == [
        _validate_invocation(run, list(files[worker::4])) for worker in range(4)
    ]
    assert [len(item.arguments) - 6 for item in recorder.invocations] == [2, 1, 1, 1]


@pytest.mark.parametrize(("workers", "count"), [(1, 1), (2, 2), (4, 3)])
def test_validate_worker_count_is_capped_by_selected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int, count: int
) -> None:
    files = tuple(f"us/statutes/{name}.yaml" for name in "abc")
    run = _module_run(
        tmp_path, {"us": (files, ())}, inputs={"validation-workers": workers}
    )
    recorder = _patch_batch(monkeypatch)

    assert _validate(run).status == "PASS"
    assert recorder.invocations == [
        _validate_invocation(run, list(files[worker::count])) for worker in range(count)
    ]


@pytest.mark.parametrize("workers", [0, 5, -1, 1.5])
def test_validate_rejects_out_of_range_workers_when_files_remain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int | float
) -> None:
    run = _module_run(
        tmp_path,
        {"us": (("us/statutes/a.yaml",), ())},
        inputs={"validation-workers": workers},
    )
    recorder = _patch_batch(monkeypatch)

    result = _validate(run)

    assert result.status == "FAIL"
    assert result.failures == ["validation-workers must be an integer between 1 and 4"]
    assert recorder.invocations == []


@pytest.mark.parametrize(
    ("files", "ledger", "message"),
    [
        (
            ("us/statutes/a.yaml",),
            {"us/statutes/a.yaml": {"active": _record("a")}},
            "All selected RuleSpec YAML files are skipped.",
        ),
        ((), None, "No RuleSpec YAML files selected for validation."),
    ],
    ids=["all-skipped", "no-files"],
)
def test_validate_worker_range_is_not_checked_without_remaining_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: tuple[str, ...],
    ledger: dict[str, Any] | None,
    message: str,
) -> None:
    run = _module_run(
        tmp_path,
        {"us": (files, ())},
        ledger=ledger,
        inputs={"validation-workers": 5},
    )
    recorder = _patch_batch(monkeypatch)

    result = _validate(run)

    assert result.status == "PASS", result.output
    assert message in result.output
    assert recorder.invocations == []
    assert set(run.validate_skip) == ({"us"} if files else set())


def test_validate_failures_are_labelled_by_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _module_run(
        tmp_path,
        {
            "us": (("us/statutes/a.yaml",), ()),
            "us-ca": (("us-ca/statutes/c.yaml",), ()),
        },
    )
    recorder = _patch_batch(
        monkeypatch,
        lambda invocation: (
            (1, "invalid module\n")
            if "us-ca" in invocation.arguments[1]
            else (0, "valid\n")
        ),
    )

    result = _validate(run)

    assert result.status == "FAIL"
    assert result.failures == ["[us-ca] invalid module"]
    assert [item.arguments[1] for item in recorder.invocations] == [
        str(run.repo / "us/statutes/a.yaml"),
        str(run.repo / "us-ca/statutes/c.yaml"),
    ]
    assert set(run.validate_skip) == {"us", "us-ca"}


def test_validate_silent_chunk_failure_still_fails_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = tuple(f"us/statutes/{name}.yaml" for name in "ab")
    run = _module_run(tmp_path, {"us": (files, ())}, inputs={"validation-workers": 2})
    _patch_batch(
        monkeypatch,
        lambda invocation: (
            (3, "") if invocation.arguments[1].endswith("b.yaml") else (0, "ok\n")
        ),
    )

    result = _validate(run)

    # The step exits 1 when any chunk fails; its log is every chunk's log.
    assert result.status == "FAIL"
    assert result.failures


@pytest.mark.parametrize("gate", ["validate", "companion_tests", "proof_validate"])
def test_waiver_ledger_shape_is_not_judged_when_a_shard_selects_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    run = _module_run(tmp_path, {"us": ((), ())})
    _write(run.repo, {LEDGER: "- us/statutes/a.yaml\n"})
    recorder = _patch_batch(monkeypatch)

    result = getattr(ci_parity, f"_gate_{gate}")(run, _spec(gate))

    assert recorder.invocations == []
    assert result.status == "PASS", result.output


# ---------------------------------------------------------------------------
# Execute RuleSpec companion tests
# ---------------------------------------------------------------------------


def _companion_invocation(
    run: WorkflowRun, jurisdiction: str, files: list[str]
) -> CliInvocation:
    return CliInvocation(
        (
            "test",
            "--root",
            str(run.repo / jurisdiction),
            "--axiom-rules-engine-path",
            str(run.paths["engine"]),
            *files,
        ),
        cwd=run.repo,
        supervised=False,
        environment={
            "AXIOM_RULESPEC_REPO_ROOTS": f"{run.repo}:{run.paths['rulespec_us']}"
        },
    )


def _companions(run: WorkflowRun) -> GateResult:
    return ci_parity._gate_companion_tests(run, _spec("companion_tests"))


def test_companion_tests_group_by_jurisdiction_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _module_run(
        tmp_path,
        {
            "__all__": (
                (),
                (
                    "regulations/b.test.yaml",
                    "statutes/a.test.yaml",
                    "statutes/sub/c.test.yml",
                ),
            )
        },
    )
    recorder = _patch_batch(monkeypatch)

    result = _companions(run)

    assert result.status == "PASS", result.output
    assert recorder.invocations == [
        _companion_invocation(run, "regulations", ["b.test.yaml"]),
        _companion_invocation(run, "statutes", ["a.test.yaml", "sub/c.test.yml"]),
    ]


def test_companion_tests_run_per_shard_with_jurisdiction_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _module_run(
        tmp_path,
        {
            "us": ((), ("us/statutes/a.test.yaml",)),
            "us-ca": ((), ("us-ca/regulations/c.test.yml",)),
            "yy": ((), ()),
        },
    )
    recorder = _patch_batch(
        monkeypatch,
        lambda invocation: (
            (1, "case failed\n")
            if invocation.arguments[2].endswith("us-ca")
            else (0, "ok\n")
        ),
    )

    result = _companions(run)

    assert result.status == "FAIL"
    assert recorder.invocations == [
        _companion_invocation(run, "us", ["statutes/a.test.yaml"]),
        _companion_invocation(run, "us-ca", ["regulations/c.test.yml"]),
    ]
    assert result.failures == ["[us-ca] case failed"]
    assert "[yy] No RuleSpec companion tests selected; skipping." in result.output


@pytest.mark.parametrize("skip", ["active-waiver", "retired"])
def test_waived_module_companion_skips_only_where_validate_wrote_its_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skip: str
) -> None:
    ledger = (
        {"us/statutes/a.yaml": {"active": _record("a")}}
        if skip == "active-waiver"
        else {}
    )
    retired = frozenset({"us/statutes/a.yaml"}) if skip == "retired" else frozenset()
    # us: the waived module is itself selected, so validate writes the list.
    # us-ca: only the companion changed and its module is gone, so the
    # validate step exits before writing any skip list.
    run = _module_run(
        tmp_path,
        {
            "us": (("us/statutes/a.yaml",), ("us/statutes/a.test.yaml",)),
            "us-ca": ((), ("us-ca/statutes/z.test.yaml",)),
        },
        ledger={
            **ledger,
            "us-ca/statutes/z.yaml": {"active": _record("z")},
        },
        retired=retired | {"us-ca/statutes/z.yaml"},
    )
    recorder = _patch_batch(monkeypatch)

    assert _validate(run).status == "PASS"
    assert set(run.validate_skip) == {"us"}
    validated = recorder.invocations
    assert validated == []

    result = _companions(run)

    assert result.status == "PASS", result.output
    assert (
        "[us] SKIPPED companion (known-validation-gaps validate_failures): "
        "us/statutes/a.test.yaml"
    ) in result.output
    assert recorder.invocations == [
        _companion_invocation(run, "us-ca", ["statutes/z.test.yaml"])
    ]


def test_companion_tests_without_selection_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _module_run(tmp_path, {"us": ((), ())})
    recorder = _patch_batch(monkeypatch)

    result = _companions(run)

    assert result.status == "PASS"
    assert result.output == "No RuleSpec companion tests selected; skipping.\n"
    assert recorder.invocations == []


# ---------------------------------------------------------------------------
# Validate RuleSpec proofs and claims
# ---------------------------------------------------------------------------


def test_proof_validate_skips_active_and_retired_modules_per_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _module_run(
        tmp_path,
        {
            "us": (
                ("us/statutes/a.yaml", "us/statutes/b.yaml", "us/statutes/w.yaml"),
                (),
            ),
            "us-ca": (("us-ca/statutes/c.yaml", "us-ca/statutes/d.yaml"), ()),
            "us-ny": (("us-ny/statutes/n.yaml",), ()),
            "yy": ((), ()),
        },
        ledger={
            "us/statutes/w.yaml": {"active": _record("w")},
            "us/statutes/b.yaml": {"pending": _record("b")},
            "us-ny/statutes/n.yaml": {"active": _record("n")},
        },
        retired=frozenset({"us-ca/statutes/d.yaml"}),
    )
    recorder = _patch_batch(monkeypatch)

    result = ci_parity._gate_proof_validate(run, _spec("proof_validate"))

    assert result.status == "PASS", result.output

    def proof(*files: str) -> CliInvocation:
        return CliInvocation(
            (
                "proof-validate",
                *(str(run.repo / file) for file in files),
                "--corpus-path",
                str(run.paths["corpus"]),
            ),
            cwd=run.repo,
            supervised=True,
        )

    assert recorder.invocations == [
        proof("us/statutes/a.yaml", "us/statutes/b.yaml"),
        proof("us-ca/statutes/c.yaml"),
    ]
    assert "[us-ny] All selected RuleSpec YAML files are skipped." in result.output
    assert "[yy] No RuleSpec YAML files selected for proof validation." in result.output
    assert (
        "[us-ca] SKIPPED (known-validation-gaps validate_failures): "
        "us-ca/statutes/d.yaml"
    ) in result.output


def test_proof_validate_failure_fails_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _module_run(tmp_path, {"us": (("us/statutes/a.yaml",), ())})
    _patch_batch(monkeypatch, lambda _invocation: (1, "missing claim\n"))

    result = ci_parity._gate_proof_validate(run, _spec("proof_validate"))

    assert result.status == "FAIL"
    assert result.failures == ["missing claim"]


# ---------------------------------------------------------------------------
# Require money proof atoms
# ---------------------------------------------------------------------------

MONEY_LAYOUT = {
    "us/statutes/a.yaml": PLAIN,
    "us/statutes/a.test.yaml": "cases: []\n",
    "us/statutes/programs/p.yaml": PLAIN,
    "us/manual/m.yaml": PLAIN,
    "us/programs/q.yaml": PLAIN,
    "us/root.yaml": PLAIN,
    "us-ca/legislation/l.yml": PLAIN,
    "us-ca/regulations/r.yaml": PLAIN,
    "us-ca/regulations/r.test.yml": "cases: []\n",
    "sources/s.yaml": PLAIN,
}


def _money(run: WorkflowRun) -> GateResult:
    return ci_parity._gate_money_atoms(run, _spec("money_atoms"))


@pytest.mark.parametrize("ratchet", [False, True])
def test_money_atoms_scan_canonical_content_roots_in_auto_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ratchet: bool
) -> None:
    files = dict(MONEY_LAYOUT)
    if ratchet:
        files["known-missing-money-atoms.yaml"] = "missing: {}\n"
    repo = _repo(tmp_path, files)
    _commit(repo, {"us/statutes/a.yaml": CHANGED})
    run = _run(repo, tmp_path)
    assert run.plan.matrix == ("us",)
    assert run.plan.roots == "us-ca us"
    recorder = _patch_cli(monkeypatch)

    result = _money(run)

    assert result.status == "PASS", result.output
    [call] = recorder.calls
    arguments = call["arguments"]
    tail = ["--money-atoms-only", "--corpus-path", str(run.paths["corpus"])]
    if ratchet:
        tail += ["--ratchet-file", "known-missing-money-atoms.yaml"]
    assert arguments[0] == "proof-validate"
    assert arguments[-len(tail) :] == tail
    assert sorted(arguments[1 : -len(tail)]) == [
        "us-ca/legislation/l.yml",
        "us-ca/regulations/r.yaml",
        "us/statutes/a.yaml",
    ]
    assert call["cwd"] == repo
    assert call["supervised"] is True
    assert call["environment"] is None
    assert result.command == ["proof-validate", "{3 files}", *tail]


def test_money_atoms_explicit_guarded_roots_include_top_level_programs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(
        tmp_path,
        {
            "statutes/a.yaml": PLAIN,
            "statutes/programs/p.yaml": PLAIN,
            "programs/q.yaml": PLAIN,
            "manual/m.yaml": PLAIN,
        },
    )
    run = _run(
        repo,
        tmp_path,
        inputs={"validate-roots": "statutes regulations", "guard-programs-root": True},
    )
    assert run.plan.roots == "statutes regulations programs"
    recorder = _patch_cli(monkeypatch)

    assert _money(run).status == "PASS"
    [call] = recorder.calls
    assert sorted(call["arguments"][1:-3]) == ["programs/q.yaml", "statutes/a.yaml"]


def test_money_atoms_disabled_or_empty_pass_without_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, MONEY_LAYOUT)
    disabled = _run(repo, tmp_path, inputs={"run-money-atom-check": False})
    recorder = _patch_cli(monkeypatch)

    result = _money(disabled)

    assert result.status == "PASS"
    assert result.output == "Disabled by caller run-money-atom-check.\n"

    empty = _run(repo, tmp_path, inputs={"validate-roots": "policies"})
    result = _money(empty)

    assert result.status == "PASS"
    assert result.output == "No RuleSpec YAML files found for money-atom check.\n"
    assert recorder.calls == []


def test_money_atoms_failure_fails_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, MONEY_LAYOUT)
    run = _run(repo, tmp_path)
    _patch_cli(monkeypatch, (1, "us/statutes/a.yaml: missing money atom\n"))

    result = _money(run)

    assert result.status == "FAIL"
    assert result.failures == ["us/statutes/a.yaml: missing money atom"]


# ---------------------------------------------------------------------------
# PolicyEngine oracle coverage (full vs changed; REAL changed-file filter)
# ---------------------------------------------------------------------------


def _oracle_run(
    tmp_path: Path,
    mode: str,
    selections: Mapping[str, tuple[str, ...]],
    *,
    name: str = "rulespec-zz",
) -> WorkflowRun:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN}, name=name)
    run = _run(repo, tmp_path, plan=_plan(*selections))
    run.mode = mode
    run.selections = {
        shard: ShardSelection(shard, (shard,), files, ())
        for shard, files in selections.items()
    }
    return run


class _Classifier:
    def __init__(self, payload: Any, code: int = 0, stderr: str = "") -> None:
        self.payload = payload
        self.code = code
        self.stderr = stderr
        self.calls: list[tuple[Path, str, list[str], int]] = []

    def __call__(
        self, encode_path: Path, pin: str, arguments: list[str], *, stderr: int
    ) -> tuple[int, str, str]:
        self.calls.append((encode_path, pin, list(arguments), stderr))
        text = (
            self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        )
        return self.code, text, self.stderr


def _oracle_gates(run: WorkflowRun) -> tuple[GateResult, GateResult]:
    return (
        ci_parity._gate_oracle_coverage(run, _spec("oracle_coverage")),
        ci_parity._gate_changed_oracle_coverage(run, _spec("changed_oracle_coverage")),
    )


def test_full_toolchain_bump_runs_full_oracle_coverage_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _oracle_run(tmp_path, "full-toolchain-bump", {"us": ("us/statutes/a.yaml",)})
    recorder = _patch_cli(monkeypatch)
    classifier = _Classifier({"items": []})
    monkeypatch.setattr(ci_parity, "_run_pinned_process", classifier)

    full, changed = _oracle_gates(run)

    assert full.status == "PASS"
    assert recorder.calls == [
        {
            "arguments": [
                "oracle-coverage",
                "--root",
                str(run.repo),
                "--fail-on-unmapped",
                "--fail-on-untested-comparable",
                "--limit",
                "50",
            ],
            "environment": None,
            "cwd": run.repo,
            "supervised": False,
        }
    ]
    assert changed.status == "PASS"
    assert changed.output == "Not used for full-toolchain-bump mode.\n"
    assert classifier.calls == []


@pytest.mark.parametrize("mode", ["changed", "changed-corpus-toolchain-bump"])
def test_changed_modes_run_the_pinned_classifier_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    run = _oracle_run(
        tmp_path, mode, {"us": ("us/statutes/a.yaml",), "us-ca": (), "yy": ()}
    )
    recorder = _patch_cli(monkeypatch)
    classifier = _Classifier(
        {
            "items": [
                {
                    "file": f"{run.repo.name}/us/statutes/a.yaml",
                    "legal_id": "us/a#tested",
                    "status": "comparable",
                    "tested": True,
                },
                {
                    "file": f"{run.repo.name}/us/statutes/a.yaml",
                    "legal_id": "us/a#other",
                    "status": "not_comparable",
                },
                {
                    "file": f"{run.repo.name}/us/statutes/unchanged.yaml",
                    "legal_id": "us/u#x",
                    "status": "unmapped",
                },
            ]
        }
    )
    monkeypatch.setattr(ci_parity, "_run_pinned_process", classifier)

    full, changed = _oracle_gates(run)

    assert full.status == "PASS"
    assert (
        full.output == "Full oracle coverage is not selected for changed-file mode.\n"
    )
    assert recorder.calls == []
    assert changed.status == "PASS", changed.output
    assert classifier.calls == [
        (
            run.paths["encode"],
            REFS["encode"],
            ["oracle-coverage", "--root", str(run.repo), "--json"],
            subprocess.PIPE,
        )
    ]
    assert "[us] Changed PolicyEngine oracle coverage passed for 2 output(s)." in (
        changed.output
    )
    assert (
        "[us-ca] No changed RuleSpec YAML files selected for oracle coverage."
        in changed.output
    )


def test_changed_oracle_coverage_fails_on_unmapped_and_untested_changed_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _oracle_run(tmp_path, "changed", {"us": ("us/statutes/a.yaml",)})
    changed_file = f"{run.repo.name}/us/statutes/a.yaml"
    monkeypatch.setattr(
        ci_parity,
        "_run_pinned_process",
        _Classifier(
            {
                "items": [
                    {"file": changed_file, "legal_id": "us/a#x", "status": "unmapped"},
                    {
                        "file": changed_file,
                        "legal_id": "us/a#y",
                        "status": "comparable",
                        "tested": False,
                    },
                    {
                        "file": changed_file,
                        "legal_id": "us/a#z",
                        "status": "comparable",
                        "tested": True,
                    },
                    {
                        "file": f"{run.repo.name}/us/statutes/b.yaml",
                        "legal_id": "us/b#x",
                        "status": "unmapped",
                    },
                    {
                        "file": "us/statutes/a.yaml",
                        "legal_id": "us/a#unprefixed",
                        "status": "unmapped",
                    },
                ]
            }
        ),
    )

    _, changed = _oracle_gates(run)

    assert changed.status == "FAIL"
    assert changed.failures == [
        "Changed PolicyEngine oracle coverage is incomplete.",
        "- us/a#x: unmapped",
        "- us/a#y: comparable but not covered by companion tests",
    ]


def test_changed_oracle_coverage_skips_belgium_by_workspace_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _oracle_run(
        tmp_path, "changed", {"__all__": ("statutes/a.yaml",)}, name="rulespec-be"
    )
    monkeypatch.setattr(
        ci_parity,
        "_run_pinned_process",
        _Classifier(
            {
                "items": [
                    {
                        "file": "rulespec-be/statutes/a.yaml",
                        "legal_id": "be/a#x",
                        "status": "unmapped",
                    }
                ]
            }
        ),
    )

    _, changed = _oracle_gates(run)

    assert changed.status == "PASS", changed.output
    assert "skipped for Belgium" in changed.output


def test_changed_oracle_coverage_classifier_failure_fails_every_selected_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _oracle_run(
        tmp_path,
        "changed",
        {"us": ("us/statutes/a.yaml",), "us-ca": ("us-ca/statutes/c.yaml",)},
    )
    classifier = _Classifier("", code=2, stderr="classifier crashed\n")
    monkeypatch.setattr(ci_parity, "_run_pinned_process", classifier)

    _, changed = _oracle_gates(run)

    assert changed.status == "FAIL"
    assert changed.failures == ["[us] classifier crashed", "[us-ca] classifier crashed"]
    assert len(classifier.calls) == 1


def test_changed_oracle_coverage_without_selection_skips_the_classifier(
    tmp_path: Path,
) -> None:
    run = _oracle_run(tmp_path, "changed", {"us": ()})

    _, changed = _oracle_gates(run)

    assert changed.status == "PASS"
    assert changed.output == (
        "No changed RuleSpec YAML files selected for oracle coverage.\n"
    )


# ---------------------------------------------------------------------------
# Run repository tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("files", "run_pytest", "output", "calls"),
    [
        (
            {"tests/test_repo.py": "def test(): pass\n"},
            False,
            "Disabled by caller run-pytest.\n",
            0,
        ),
        (
            {"tests/helper.py": "x = 1\n"},
            True,
            "No Python tests found; skipping pytest.\n",
            0,
        ),
        (
            {"test_root.py": "def test(): pass\n"},
            True,
            "No Python tests found; skipping pytest.\n",
            0,
        ),
        ({"tests/unit/test_repo.py": "def test(): pass\n"}, True, "1 passed\n", 1),
        ({"tests/repo_test.py": "def test(): pass\n"}, True, "1 passed\n", 1),
    ],
    ids=["disabled", "no-test-files", "outside-tests", "nested", "suffix"],
)
def test_repository_tests_follow_the_workflow_find(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, str],
    run_pytest: bool,
    output: str,
    calls: int,
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN, **files})
    run = _run(repo, tmp_path, inputs={"run-pytest": run_pytest}, plan=_plan("us"))
    seen: list[tuple[list[str], Path]] = []

    def process(
        arguments: list[str], cwd: Path, *, environment: dict[str, str] | None = None
    ) -> tuple[int, str]:
        seen.append((list(arguments), cwd))
        return 0, "1 passed\n"

    monkeypatch.setattr(ci_parity, "_run_process", process)

    result = ci_parity._gate_repository_tests(run, _spec("repository_tests"))

    assert result.status == "PASS"
    assert result.output == output
    assert seen == [([sys.executable, "-m", "pytest", "-q", "tests"], repo)] * calls


# ---------------------------------------------------------------------------
# 6f11be26: Reject unmanifested RuleSpec content (REAL heredoc)
# ---------------------------------------------------------------------------

EXCEPTION = ".axiom/generated-guard-exception.md"


def _unmanifested(run: WorkflowRun) -> GateResult:
    return ci_parity._gate_unmanifested_rulespec(
        run, _spec("unmanifested_rulespec", PIN_6F11BE26)
    )


def _manifest(path: str, content: str | None) -> str:
    return json.dumps(
        {
            "applied_files": [
                {
                    "path": path,
                    "sha256": "deleted" if content is None else _sha256(content),
                }
            ]
        }
    )


def test_unmanifested_rulespec_rejects_hand_authored_modules(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"README.md": "readme\n"})
    _commit(repo, {"us/statutes/x.yaml": PLAIN})
    run = _run(repo, tmp_path, sha=PIN_6F11BE26)

    result = _unmanifested(run)

    assert result.status == "FAIL"
    assert (
        "Unmanifested RuleSpec content (1 of 1 guarded files, scan=diff):"
        in result.failures
    )
    assert (
        "  us/statutes/x.yaml has no tracked encoder apply manifest "
        "(.axiom/encoding-manifests/us/statutes/x.json)"
    ) in result.output
    assert result.command == [
        "manifest-precheck",
        f"{run.simulation.base_sha}...{run.simulation.head_sha}",
    ]


@pytest.mark.parametrize(
    "manifest",
    [
        {
            ".axiom/encoding-manifests/us/statutes/x.json": _manifest(
                "us/statutes/x.yaml", PLAIN
            )
        },
        {
            "us/.axiom/encoding-manifests/statutes/x.json": _manifest(
                "statutes/x.yaml", PLAIN
            )
        },
        {
            ".axiom/encoding-manifests/other-name.json": _manifest(
                "us/statutes/x.yaml", PLAIN
            )
        },
        {".axiom/encoding-manifests/us/statutes/x.json": "{}"},
    ],
    ids=["checkout-root", "jurisdiction-root", "content-resolved", "filename-only"],
)
def test_unmanifested_rulespec_accepts_a_tracked_manifest(
    tmp_path: Path, manifest: dict[str, str]
) -> None:
    repo = _repo(tmp_path, {"README.md": "readme\n"})
    _commit(repo, {"us/statutes/x.yaml": PLAIN, **manifest})
    run = _run(repo, tmp_path, sha=PIN_6F11BE26)

    result = _unmanifested(run)

    assert result.status == "PASS", result.output
    assert (
        "1 guarded RuleSpec file(s) in scope (scan=diff); every one carries an "
        "encoder apply manifest."
    ) in result.output


def test_unmanifested_rulespec_rejects_content_that_no_manifest_records(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, {"README.md": "readme\n"})
    _commit(
        repo,
        {
            "us/statutes/x.yaml": CHANGED,
            ".axiom/encoding-manifests/us/statutes/x.json": _manifest(
                "us/statutes/x.yaml", PLAIN
            ),
        },
    )
    run = _run(repo, tmp_path, sha=PIN_6F11BE26)

    result = _unmanifested(run)

    assert result.status == "FAIL"
    assert (
        "  us/statutes/x.yaml content matches no sha256 recorded by any encoder "
        "apply manifest"
    ) in result.output


@pytest.mark.parametrize(
    "change",
    [
        {"us/legislation/x.yaml": PLAIN},
        {"us/statutes/x.test.yaml": "cases: []\n"},
        {"us/statutes/composed/x.yaml": PLAIN},
        {"us/manual/x.yaml": PLAIN},
        {"docs/x.yaml": PLAIN},
    ],
    ids=["legislation", "companion-test", "composed", "manual", "outside-roots"],
)
def test_unmanifested_rulespec_ignores_unguarded_paths(
    tmp_path: Path, change: dict[str, str]
) -> None:
    repo = _repo(tmp_path, {"README.md": "readme\n"})
    _commit(repo, change)
    run = _run(repo, tmp_path, sha=PIN_6F11BE26, plan=_plan("us"))

    result = _unmanifested(run)

    assert result.status == "PASS", result.output
    assert "No guarded RuleSpec content in scope (scan=diff)" in result.output


@pytest.mark.parametrize(
    "path", ["programs/p.yaml", "statutes/s.yml", "us-ca/programs/p.yaml"]
)
def test_unmanifested_rulespec_guards_programs_and_legacy_roots(
    tmp_path: Path, path: str
) -> None:
    repo = _repo(tmp_path, {"README.md": "readme\n"})
    _commit(repo, {path: PLAIN})
    run = _run(repo, tmp_path, sha=PIN_6F11BE26, plan=_plan("us"))

    result = _unmanifested(run)

    assert result.status == "FAIL"
    assert any(path in line for line in result.failures), result.failures


@pytest.mark.parametrize(
    ("exception", "status", "message"),
    [
        (
            None,
            "FAIL",
            "run-generated-guard=false without .axiom/generated-guard-exception.md",
        ),
        (
            f"https://github.com/{RULESPEC_US}/pull/911\nexpires: 2999-12-31\n",
            "PASS",
            f"run-generated-guard=false honored under https://github.com/"
            f"{RULESPEC_US}/pull/911 (expires 2999-12-31).",
        ),
        (
            f"https://github.com/{RULESPEC_US}/issues/7\nexpires: 2000-01-01\n",
            "FAIL",
            f"{EXCEPTION} expired on 2000-01-01",
        ),
        (
            "not a url\nexpires: 2999-12-31\n",
            "FAIL",
            f"{EXCEPTION} line 1 is not a GitHub issue or pull-request URL",
        ),
    ],
    ids=["missing", "valid", "expired", "bad-url"],
)
def test_unmanifested_rulespec_guard_bypass_needs_a_live_exception(
    tmp_path: Path, exception: str | None, status: str, message: str
) -> None:
    files = {"README.md": "readme\n"}
    if exception is not None:
        files[EXCEPTION] = exception
    repo = _repo(tmp_path, files)
    # Hand-authored content is not judged once a live exception is honored.
    _commit(repo, {"us/statutes/x.yaml": PLAIN})
    run = _run(
        repo,
        tmp_path,
        sha=PIN_6F11BE26,
        inputs={"run-generated-guard": False},
        plan=_plan("us"),
    )

    result = _unmanifested(run)

    assert result.status == status, result.output
    assert message in result.output
