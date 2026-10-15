"""Cycle-24 regressions: supervised and isolated gates keep ci's fetch switches.

Cycle 24's executed review ran both scenarios below against the code before
the fix. A supervised gate placed corpus bytes despite the caller's
``AXIOM_CORPUS_NO_FETCH``, because the supervised strip withheld it. An
isolated ``--jobs`` worker under ``--offline`` asked R2 for a missing
artifact, because the remote-fetch switch is a context variable that worker
threads and processes do not inherit.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from axiom_encode import ci_parity, cli
from axiom_encode import corpus_materialize as cm
from axiom_encode.toolchain import local_corpus_release_verification
from tests.release_object_fixtures import TEST_RELEASE_PUBLIC_KEY
from tests.test_corpus_materialize import (
    PROVISIONS,
    REG_PROVISIONS,
    REGULATION,
    RELEASE,
    STATUTE,
    _pre_switch_corpus,
    _release,
    _switch,
)
from tests.test_toolchain import _write_toolchain


@pytest.fixture(autouse=True)
def _isolated_corpus_environment(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    for name in (
        cm.CACHE_ENV,
        cm.NO_FETCH_ENV,
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_ENDPOINT",
        "R2_ACCOUNT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    cm._PRUNED_STAGING_DIRECTORIES.clear()


def _partial_checkout(tmp_path: Path) -> tuple[Path, str]:
    """A lock-file corpus checkout holding the statute scope, not the regulation."""

    root = tmp_path / "axiom-corpus"
    _, sha = _pre_switch_corpus(root, scopes=(STATUTE, REGULATION))
    statute = (root / PROVISIONS).read_bytes()
    _switch(root)
    (root / PROVISIONS).parent.mkdir(parents=True)
    (root / PROVISIONS).write_bytes(statute)
    return root, sha


@pytest.mark.parametrize("supervised", [False, True])
def test_no_fetch_opt_out_reaches_every_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supervised: bool
) -> None:
    root, sha = _partial_checkout(tmp_path)
    monkeypatch.setenv(cm.NO_FETCH_ENV, "1")
    _release(root, sha)  # ci's own bind places nothing
    assert not (root / REG_PROVISIONS).exists()
    seen: dict[str, str | None] = {}

    def gate_main() -> None:
        # validate, proof-validate and the waiver audit bind the release.
        seen["no_fetch"] = os.environ.get(cm.NO_FETCH_ENV)
        _release(root, sha)

    monkeypatch.setattr(cli, "main", gate_main)
    code, _ = ci_parity._run_cli(["validate"], cwd=root, supervised=supervised)

    assert code == 0
    assert cm.fetch_disabled({cm.NO_FETCH_ENV: seen["no_fetch"] or ""})
    assert not (root / REG_PROVISIONS).exists()
    assert os.environ.get(cm.NO_FETCH_ENV) == "1"


def _rulespec_checkout(tmp_path: Path, sha: str) -> Path:
    repo = tmp_path / "rulespec-us"
    (repo / "us").mkdir(parents=True)
    _write_toolchain(repo, RELEASE, content_sha256=sha)
    return repo


@pytest.mark.parametrize("jobs", [1, 2], ids=["in-process", "isolated"])
def test_offline_keeps_every_gate_off_r2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: int
) -> None:
    root = tmp_path / "axiom-corpus"
    _, sha = _pre_switch_corpus(root, scopes=(STATUTE,))
    _switch(root)
    # No cache and no git objects: only R2 could supply the missing bytes.
    shutil.rmtree(root / ".git")
    repo = _rulespec_checkout(tmp_path, sha)
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "fake")
    monkeypatch.setenv("R2_ENDPOINT", "https://127.0.0.1:1")  # closed port
    invocation = ci_parity.CliInvocation(
        ("manifest-census", "--repo", str(repo), "--corpus-path", str(root)),
        cwd=repo,
        supervised=True,
    )

    # ci --offline: run_ci runs every gate inside remote_fetch_disabled().
    with (
        cm.remote_fetch_disabled(),
        local_corpus_release_verification(TEST_RELEASE_PUBLIC_KEY),
    ):
        results = ci_parity._run_cli_batch(
            [invocation] * jobs, jobs=jobs, keyring=(TEST_RELEASE_PUBLIC_KEY,)
        )

    assert len(results) == jobs
    for _, output in results:
        assert "r2:" not in output
