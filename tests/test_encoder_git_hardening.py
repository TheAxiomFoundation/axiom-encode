"""Producer history uses the provisioned repository's unmodified commit graph."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from axiom_encode import cli
from tests.test_encoder_manifest_ancestry import (
    _expected,
    _git,
    _Graph,
    _identity,
    _write_git_ref,
)
from tests.test_encoder_manifest_ancestry import (
    graph as _graph_fixture,
)
from tests.test_encoder_manifest_ancestry import (
    identity_checkout as _identity_checkout_fixture,
)

# The suite's autouse fixture replaces the current guard resolver for ordinary
# manifest tests. These Git-boundary tests must exercise the actual resolver.
_current_guard_identity = cli._current_guard_encoder_execution_identity
graph = _graph_fixture
identity_checkout = _identity_checkout_fixture


_NEUTRAL_GIT_ENVIRONMENT = {
    "GIT_GRAFT_FILE": "/dev/null",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
}
_HOSTILE_GIT_ENVIRONMENT = {
    "GIT_DIR": "/unavailable/alternate.git",
    "GIT_WORK_TREE": "/unavailable/work-tree",
    "GIT_OBJECT_DIRECTORY": "/unavailable/objects",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/unavailable/alternate-objects",
    "GIT_INDEX_FILE": "/unavailable/index",
    "GIT_COMMON_DIR": "/unavailable/common",
    "GIT_REPLACE_REF_BASE": "refs/attacker/",
    "GIT_NAMESPACE": "attacker",
    "GIT_CONFIG": "/unavailable/config",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "core.bare",
    "GIT_CONFIG_VALUE_0": "true",
    "GIT_CONFIG_KEY_99": "core.bare",
    "GIT_CONFIG_VALUE_99": "true",
    "GIT_CONFIG_PARAMETERS": "'core.bare=true'",
    **dict.fromkeys(_NEUTRAL_GIT_ENVIRONMENT, "hostile"),
}


def _issue(graph: _Graph, producer: int, pin: int = 1) -> str | None:
    return cli._encoder_producer_admission_issue(
        _identity(graph.commits[producer]), _expected(graph, pin)
    )


def _grafts(checkout: Path, content: str) -> Path:
    path = checkout / ".git/info/grafts"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.mark.parametrize("producer", [0, 5], ids=["older-main", "never-main"])
def test_unmodified_graph_rejects_older_and_never_main_producers(graph, producer):
    assert _issue(graph, producer)


def test_repository_grafts_cannot_add_feature_to_first_parent_main(graph):
    assert _issue(graph, 4) is None
    assert _issue(graph, 5)
    _grafts(
        graph.root,
        f"{graph.commits[4]} {graph.commits[5]}\n{graph.commits[5]} {graph.commits[1]}\n",
    )
    assert _issue(graph, 5)


def test_ambient_graft_file_cannot_add_feature_to_first_parent_main(
    graph, tmp_path, monkeypatch
):
    assert _issue(graph, 4) is None
    grafts = tmp_path / "external-grafts"
    grafts.write_text(
        f"{graph.commits[4]} {graph.commits[5]}\n{graph.commits[5]} {graph.commits[1]}\n"
    )
    monkeypatch.setenv("GIT_GRAFT_FILE", str(grafts))
    assert _issue(graph, 5)


def test_symbolic_origin_main_cannot_authorize_a_feature(graph):
    _write_git_ref(graph.root, "refs/remotes/origin/main", "ref: refs/heads/feature")
    assert _issue(graph, 5)


def test_symbolic_local_main_cannot_authorize_a_feature(graph):
    _write_git_ref(graph.root, "refs/heads/main", "ref: refs/heads/feature")
    assert _issue(graph, 5)


def test_ambient_git_dir_cannot_substitute_the_history_checkout(
    graph, tmp_path, monkeypatch
):
    alternate = _Graph(tmp_path / "alternate-history")
    shutil.copytree(
        graph.root / ".git/objects", alternate.root / ".git/objects", dirs_exist_ok=True
    )
    _write_git_ref(alternate.root, "refs/heads/main", graph.commits[5])
    monkeypatch.setenv("GIT_DIR", str(alternate.root / ".git"))
    assert _issue(graph, 5)


@pytest.mark.parametrize("producer", [4, 5], ids=["legitimate-main", "feature"])
def test_any_replace_ref_rejects_history_even_when_replacements_are_disabled(
    graph, producer
):
    _write_git_ref(graph.root, f"refs/replace/{graph.commits[4]}", graph.commits[5])
    assert _issue(graph, producer)


def test_even_empty_repository_grafts_rejects_legitimate_main(graph):
    _grafts(graph.root, "")
    assert _issue(graph, 4)


@pytest.mark.parametrize("layout", ["git-file", "linked-worktree"])
def test_graft_refusal_checks_the_common_git_directory(graph, layout):
    common = graph.root.parent / "encoder-common.git"
    shutil.move(graph.root / ".git", common)
    git_dir = common
    if layout == "linked-worktree":
        git_dir = common / "worktrees/encoder"
        git_dir.mkdir(parents=True)
        (git_dir / "commondir").write_text("../..\n")
        (git_dir / "gitdir").write_text(str(graph.root / ".git") + "\n")
        (git_dir / "HEAD").write_text(graph.commits[1] + "\n")
    (graph.root / ".git").write_text(f"gitdir: {git_dir}\n")
    assert _issue(graph, 4) is None
    (common / "info").mkdir()
    (common / "info/grafts").write_text("")
    assert _issue(graph, 4)


@pytest.mark.parametrize("resolver", ["explicit", "current"])
@pytest.mark.parametrize("metadata", ["empty-grafts", "grafts", "replacement"])
def test_git_backed_resolvers_refuse_graph_metadata_even_at_the_exact_pin(
    identity_checkout, monkeypatch, resolver, metadata
):
    checkout, pin, producer = identity_checkout
    if metadata == "replacement":
        _write_git_ref(checkout, f"refs/replace/{pin}", producer)
    else:
        _grafts(checkout, "" if metadata == "empty-grafts" else f"{pin} {producer}\n")
    monkeypatch.setattr(
        cli,
        "_apply_encoder_execution_identity",
        lambda: {"kind": "git", "dirty": False, "commit": pin, "path": str(checkout)},
    )
    with pytest.raises(RuntimeError, match="forbidden (graft|replacement)"):
        if resolver == "explicit":
            cli._read_only_guard_encoder_execution_identity(checkout)
        else:
            _current_guard_identity()


@pytest.mark.parametrize("producer", ["feature", "older-main"])
def test_clean_checkout_resolver_cannot_use_grafts_to_admit_a_producer(
    identity_checkout, producer
):
    checkout, pin, feature = identity_checkout
    (checkout / ".git/HEAD").write_text(pin + "\n")
    # The version-bearing tree remains identical and clean throughout the attack.
    tree = _git(checkout, "rev-parse", "HEAD^{tree}")
    graph = object.__new__(_Graph)
    graph.root, graph.tree = checkout, tree
    graph.commits, graph.parents, graph.main = [pin, feature], [(), (0,)], []
    _write_git_ref(checkout, "refs/remotes/origin/main", pin)
    if producer == "feature":
        main = graph.add(0, main=True)
        claimed = feature
        grafts = f"{graph.commits[main]} {feature}\n{feature} {pin}\n"
    else:
        old = graph.add(0, main=True)
        new_pin = graph.add(old, main=True)
        main = graph.add(new_pin, main=True)
        pin = graph.commits[new_pin]
        claimed = graph.commits[old]
        (checkout / ".git/HEAD").write_text(pin + "\n")
        grafts = (
            f"{graph.commits[main]} {claimed}\n"
            f"{claimed} {pin}\n{pin} {graph.commits[0]}\n"
        )
    # Remove the tracking ref so canonical local main is the graph being proved.
    (checkout / ".git/refs/remotes/origin/main").unlink()
    baseline = cli._read_only_guard_encoder_execution_identity(checkout)
    assert baseline["commit"] == pin
    assert cli._encoder_producer_admission_issue(
        _identity(claimed, cli.__version__), baseline
    )
    _grafts(checkout, grafts)
    try:
        expected = cli._read_only_guard_encoder_execution_identity(checkout)
    except RuntimeError:
        return
    assert cli._encoder_producer_admission_issue(
        _identity(claimed, cli.__version__), expected
    )


@pytest.mark.parametrize(
    "operation", ["history", "resolver", "current", "live-current", "versions"]
)
def test_every_guard_git_command_scrubs_overrides_and_disables_graph_rewriting(
    graph, identity_checkout, monkeypatch, operation
):
    calls = []
    real_run = subprocess.run

    def capture(command, **kwargs):
        calls.append((command, kwargs.get("env")))
        return real_run(command, **kwargs)

    monkeypatch.setattr(cli.subprocess, "run", capture)
    for name, value in _HOSTILE_GIT_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    checkout, pin, _producer = identity_checkout
    if operation == "live-current":
        monkeypatch.setattr(cli, "_axiom_encode_repo_root", lambda: checkout)
    else:
        monkeypatch.setattr(
            cli,
            "_apply_encoder_execution_identity",
            lambda: {
                "kind": "git",
                "dirty": False,
                "commit": pin,
                "path": str(checkout),
            },
        )
    if operation == "history":
        result = _issue(graph, 4)
    elif operation in {"resolver", "current", "live-current"}:
        try:
            result = (
                cli._read_only_guard_encoder_execution_identity(checkout)
                if operation == "resolver"
                else _current_guard_identity()
            )
        except RuntimeError:
            result = None
    else:
        result = cli._axiom_encode_versions_at_ref(checkout, "HEAD")
    assert calls
    for _command, environment in calls:
        assert environment is not None
        assert {
            key: value for key, value in environment.items() if key.startswith("GIT_")
        } == (_NEUTRAL_GIT_ENVIRONMENT)
    if operation == "history":
        assert result is None
    elif operation in {"resolver", "current", "live-current"}:
        assert result["commit"] == pin
    else:
        assert set(result.values()) == {cli.__version__}


def _fetch_record(commit: str, branch: str = "main", repository: str | None = None):
    repository = repository or "https://github.com/TheAxiomFoundation/axiom-encode"
    return f"{commit}\t\tbranch '{branch}' of {repository}\n"


@pytest.mark.parametrize("merge_status", ["", "not-for-merge"])
def test_explicit_official_main_fetch_head_authorizes_newer_main(graph, merge_status):
    _write_git_ref(graph.root, "refs/heads/main", graph.commits[1])
    (graph.root / ".git/FETCH_HEAD").write_text(
        _fetch_record(graph.commits[4]).replace("\t\t", f"\t{merge_status}\t")
    )
    assert _issue(graph, 4) is None


@pytest.mark.parametrize(
    "record",
    ["malformed", "feature", "unofficial", "ambiguous", "tag"],
)
@pytest.mark.parametrize("local_main", [False, True])
def test_fetch_head_requires_one_explicit_official_main_branch(
    graph, record, local_main
):
    if not local_main:
        (graph.root / ".git/refs/heads/main").unlink()
    commit = graph.commits[5]
    records = {
        "malformed": f"{commit}\n",
        "feature": _fetch_record(commit, branch="feature"),
        "unofficial": _fetch_record(commit, repository="https://example.com/encoder"),
        "ambiguous": _fetch_record(commit) + _fetch_record(graph.commits[4]),
        "tag": _fetch_record(commit).replace("branch 'main'", "tag 'main'"),
    }
    (graph.root / ".git/FETCH_HEAD").write_text(records[record])
    assert _issue(graph, 5)
    assert _issue(graph, 4)


@pytest.mark.parametrize(
    "kind", ["fifo", "directory", "oversize", "symlink", "invalid-encoding"]
)
def test_unusable_fetch_head_fails_closed_without_reading_unbounded_metadata(
    graph, kind
):
    fetch_head = graph.root / ".git/FETCH_HEAD"
    if kind == "fifo":
        os.mkfifo(fetch_head)
    elif kind == "directory":
        fetch_head.mkdir()
    elif kind == "symlink":
        fetch_head.symlink_to(graph.root / ".git/HEAD")
    else:
        fetch_head.write_bytes(b"x" * 4097 if kind == "oversize" else b"\xff")
    assert _issue(graph, 4)


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    steps=st.lists(st.booleans(), min_size=1, max_size=4),
    producer_selector=st.integers(min_value=0, max_value=100),
    override=st.sampled_from(list(_HOSTILE_GIT_ENVIRONMENT)),
    value=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789/._-", min_size=1, max_size=30
    ),
)
@example(
    steps=[False],
    producer_selector=2,
    override="GIT_DIR",
    value="/unavailable/alternate.git",
)
def test_generated_graph_admission_is_invariant_under_ambient_git_overrides(
    tmp_path, monkeypatch, steps, producer_selector, override, value
):
    counter = len(list(tmp_path.iterdir()))
    history = _Graph(tmp_path / f"ambient-graph-{counter}")
    pin = history.add(main=True)
    for merge in steps:
        main = history.main[-1]
        side = history.add(main)
        history.add(main, *([side] if merge else []), main=True)
    producer = producer_selector % len(history.commits)
    allowed = producer in history.main and history.is_ancestor(pin, producer)
    with monkeypatch.context() as context:
        context.setenv(override, value)
        assert (_issue(history, producer, pin) is None) is allowed


@pytest.mark.parametrize("failure", ["exit-128", "oserror", "timeout"])
@pytest.mark.parametrize("fetched", [False, True])
def test_every_git_proof_stage_fails_closed(graph, monkeypatch, failure, fetched):
    if fetched:
        (graph.root / ".git/FETCH_HEAD").write_text(_fetch_record(graph.commits[4]))
    real_run = subprocess.run
    commands = []

    def record(command, **kwargs):
        commands.append(tuple(command))
        return real_run(command, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(cli.subprocess, "run", record)
        assert _issue(graph, 4) is None
    assert commands
    for index, target in enumerate(commands):
        count = 0

        def fail(command, **kwargs):
            nonlocal count
            current = count
            count += 1
            if current == index:
                assert tuple(command) == target
                if failure == "oserror":
                    raise OSError("injected Git launch failure")
                if failure == "timeout":
                    raise subprocess.TimeoutExpired(command, 30)
                return subprocess.CompletedProcess(command, 128, "", "injected error")
            return real_run(command, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(cli.subprocess, "run", fail)
            assert _issue(graph, 4), (failure, target)
