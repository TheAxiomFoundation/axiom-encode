"""Cycle-24 regressions: supervised and isolated gates keep ci's fetch switches.

Cycle 24's executed review ran both scenarios below against the code before
the fix. A supervised gate placed corpus bytes despite the caller's
``AXIOM_CORPUS_NO_FETCH``, because the supervised strip withheld it. An
isolated ``--jobs`` worker under ``--offline`` asked R2 for a missing
artifact, because the remote-fetch switch is a context variable that worker
threads and processes do not inherit.
"""

from __future__ import annotations

import locale
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _repository_run(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paths={"engine": tmp_path / "engine"},
        temp=tmp_path / "runner-temp",
        simulation=ci_parity.PullRequestSimulation(
            "a" * 40, "b" * 40, "main", "Org/rulespec-us", 7, True
        ),
        repo=tmp_path / "rulespec-us",
        mode="changed",
    )


@pytest.mark.parametrize(
    ("value", "kept"), [("1", True), ("yes", True), ("0", False), (None, False)]
)
def test_repository_pytest_environment_keeps_only_the_no_fetch_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None, kept: bool
) -> None:
    # Cycle 25: repository pytest runs in CI's step environment, built from
    # the run's inputs; the caller's opt-out is its one ambient input.
    run = _repository_run(tmp_path)
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ambient")
    monkeypatch.setenv(cm.CACHE_ENV, str(tmp_path / "cache"))
    if value is not None:
        monkeypatch.setenv(cm.NO_FETCH_ENV, value)

    environment = ci_parity._repository_test_environment(run)

    assert environment.get(cm.NO_FETCH_ENV) == ("1" if kept else None)
    assert not any(name.startswith("R2_") for name in environment)
    assert cm.CACHE_ENV not in environment
    assert environment["HOME"] == str(run.temp / "repository-test-home")
    monkeypatch.delenv(cm.NO_FETCH_ENV, raising=False)
    without = ci_parity._repository_test_environment(run)
    assert {
        name: value for name, value in environment.items() if name != cm.NO_FETCH_ENV
    } == without


@pytest.mark.parametrize("opt_out", [True, False], ids=["opt-out", "default"])
def test_repository_test_child_honors_the_no_fetch_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opt_out: bool
) -> None:
    # Cycle 25 reproduced a repository test binding the release from the
    # step environment and placing the missing scope despite the opt-out.
    root, sha = _partial_checkout(tmp_path)
    if opt_out:
        monkeypatch.setenv(cm.NO_FETCH_ENV, "1")
    run = _repository_run(tmp_path)
    (run.temp / "repository-test-home").mkdir(parents=True)
    environment = ci_parity._repository_test_environment(
        run, interpreter=sys.executable
    )
    bind = (
        "import sys\n"
        "from axiom_encode.corpus_resolver import LocalCorpusRelease\n"
        "LocalCorpusRelease(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", bind, str(root), RELEASE, sha, TEST_RELEASE_PUBLIC_KEY],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (root / REG_PROVISIONS).exists() is not opt_out


def test_utf8_ctype_decodes_rulespec_text_as_ci_does(tmp_path: Path) -> None:
    # Cycle 25: LC_ALL=C.UTF-8 in os.environ does not change the running
    # interpreter's LC_CTYPE, so in-process gates decoded files with the
    # caller's codeset. Outside UTF-8 mode, under ISO-8859-1, "§" became "Â§".
    source = tmp_path / "statute.yaml"
    source.write_bytes("7 USC § 2014(e)\n".encode())
    probe = (
        "import locale, sys\n"
        "from pathlib import Path\n"
        "from axiom_encode import ci_parity\n"
        "before = Path(sys.argv[1]).read_text()\n"
        "ci_parity._adopt_utf8_ctype()\n"
        "print(ascii(before))\n"
        "print(ascii(Path(sys.argv[1]).read_text()))\n"
    )
    environment = {**os.environ, "LC_ALL": "en_US.ISO8859-1", "LANG": ""}
    environment.pop("PYTHONUTF8", None)

    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", probe, str(source)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    before, after = result.stdout.splitlines()
    if before == after:
        pytest.skip("no ISO-8859-1 locale on this host")
    assert after == ascii("7 USC § 2014(e)\n")


def test_utf8_ctype_fails_closed_without_a_utf8_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(category: int, name: str | None = None) -> str:
        if name is None:
            return "C"
        raise locale.Error("unsupported locale setting")

    monkeypatch.setattr(ci_parity.locale, "setlocale", refuse)
    monkeypatch.setattr(
        ci_parity, "sys", SimpleNamespace(flags=SimpleNamespace(utf8_mode=0))
    )

    with pytest.raises(ValueError, match="UTF-8 LC_CTYPE"):
        ci_parity._adopt_utf8_ctype()
