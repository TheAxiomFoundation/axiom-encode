"""Producer admission works through the supervisor's trusted Git wrapper.

The generated-change guard runs under the signing supervisor, whose PATH holds
only the trusted Git wrapper. These tests route every Git call through that
wrapper, so the ancestry proof from encode#1859 is exercised as CI runs it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from axiom_encode import cli
from scripts.provision_verification_supervisor import _install_trusted_git_wrapper
from tests.test_encoder_manifest_ancestry import _expected, _identity
from tests.test_encoder_manifest_ancestry import graph as _graph_fixture
from tests.test_encoder_manifest_ancestry import (
    identity_checkout as _identity_checkout_fixture,
)

graph = _graph_fixture
identity_checkout = _identity_checkout_fixture

_HISTORY_PREFIX = ["--no-replace-objects", "-c", "core.commitGraph=false"]


def _route_git_through_wrapper(tmp_path: Path, monkeypatch) -> Path:
    """Install the provisioned wrapper and make it the only Git on PATH."""

    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required")
    tools = tmp_path / "supervised-tools"
    tools.mkdir()
    wrapper = _install_trusted_git_wrapper(
        tools, Path(sys.executable).resolve(), Path(git).resolve()
    )
    monkeypatch.setenv("PATH", str(tools))
    assert shutil.which("git") == str(wrapper)
    return wrapper


@pytest.mark.parametrize(
    ("producer", "accepted"),
    [(1, True), (2, False), (3, True), (4, True), (5, False)],
    ids=["pin", "second-parent", "merge", "main-head", "feature"],
)
def test_supervised_wrapper_admits_only_first_parent_main_descendants(
    graph, tmp_path, monkeypatch, producer, accepted
):
    _route_git_through_wrapper(tmp_path, monkeypatch)
    issue = cli._encoder_producer_admission_issue(
        _identity(graph.commits[producer]), _expected(graph)
    )
    assert (issue is None) is accepted, issue
    if not accepted:
        assert "trusted git wrapper" not in issue


def test_supervised_wrapper_reads_the_fetched_main_record(graph, tmp_path, monkeypatch):
    (graph.root / ".git/FETCH_HEAD").write_text(
        f"{graph.commits[4]}\t\tbranch 'main' of "
        "https://github.com/TheAxiomFoundation/axiom-encode\n"
    )
    subprocess.run(
        ["git", "-C", str(graph.root), "update-ref", "refs/heads/main", graph.commits[1]],
        check=True,
    )
    _route_git_through_wrapper(tmp_path, monkeypatch)
    assert (
        cli._encoder_producer_admission_issue(
            _identity(graph.commits[4]), _expected(graph)
        )
        is None
    )


def test_supervised_wrapper_keeps_shallow_history_refused(graph, tmp_path, monkeypatch):
    (graph.root / ".git/shallow").write_text(graph.commits[0] + "\n")
    _route_git_through_wrapper(tmp_path, monkeypatch)
    issue = cli._encoder_producer_admission_issue(
        _identity(graph.commits[4]), _expected(graph)
    )
    assert isinstance(issue, str) and "shallow" in issue.lower()


def test_supervised_wrapper_resolves_the_read_only_guard_identity(
    identity_checkout, tmp_path, monkeypatch
):
    checkout, pin, producer = identity_checkout
    _route_git_through_wrapper(tmp_path, monkeypatch)
    identity = cli._read_only_guard_encoder_execution_identity(checkout)
    assert identity.history_checkout == checkout
    assert dict(identity) == _identity(pin, cli.__version__)
    assert (
        cli._encoder_producer_admission_issue(
            _identity(producer, cli.__version__), identity
        )
        is None
    )


_OTHER_COMMIT_GRAPH = ["--no-replace-objects", "-c", "core.commitGraph=true"]


@pytest.mark.parametrize(
    ("global_options", "arguments"),
    [
        # History queries need the exact commit-graph-disabling prefix.
        ([], ["for-each-ref", "--format=%(refname)", "refs/replace/"]),
        ([], ["rev-list", "--first-parent", "a" * 40]),
        ([], ["rev-parse", "--is-shallow-repository"]),
        ([], ["show-ref", "--verify", "--hash", "refs/heads/main"]),
        ([], ["symbolic-ref", "--quiet", "refs/heads/main"]),
        # The prefix does not widen the accepted argument shapes.
        (_HISTORY_PREFIX, ["for-each-ref", "refs/heads/"]),
        (_HISTORY_PREFIX, ["rev-list", "--first-parent", "main"]),
        (_HISTORY_PREFIX, ["rev-list", "--first-parent", "--all"]),
        (_HISTORY_PREFIX, ["rev-parse", "--git-path", "config"]),
        (_HISTORY_PREFIX, ["show-ref", "--verify", "--hash", "refs/heads/feature"]),
        (_HISTORY_PREFIX, ["symbolic-ref", "refs/heads/main", "refs/heads/feature"]),
        (_HISTORY_PREFIX, ["update-ref", "refs/heads/main", "a" * 40]),
        (_HISTORY_PREFIX, ["fetch", "origin", "main"]),
        # Any other global prefix is refused before the command is read.
        (_OTHER_COMMIT_GRAPH, ["rev-parse", "HEAD"]),
        (["-c", "core.commitGraph=false"], ["rev-parse", "HEAD"]),
    ],
)
def test_supervised_wrapper_refuses_other_history_shapes(
    graph, tmp_path, monkeypatch, global_options, arguments
):
    wrapper = _route_git_through_wrapper(tmp_path, monkeypatch)
    result = subprocess.run(
        [str(wrapper), *global_options, "-C", str(graph.root), *arguments],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "trusted git wrapper refused" in result.stderr
