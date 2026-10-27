"""Cycle-24 regressions: supervised and isolated gates keep ci's fetch switches.

Cycle 24's executed review ran both scenarios below against the code before
the fix. A supervised gate placed corpus bytes despite the caller's
``AXIOM_CORPUS_NO_FETCH``, because the supervised strip withheld it. An
isolated ``--jobs`` worker under ``--offline`` asked R2 for a missing
artifact, because the remote-fetch switch is a context variable that worker
threads and processes do not inherit.
"""

from __future__ import annotations

import base64
import io
import locale
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from cryptography.hazmat.primitives import serialization

from axiom_encode import ci_parity, cli
from axiom_encode import corpus_materialize as cm
from axiom_encode.harness import eval_evidence
from axiom_encode.toolchain import local_corpus_release_verification
from tests.environment_assertions import changed_names
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
    no_fetch = os.environ.get(cm.NO_FETCH_ENV)
    assert no_fetch == "1"


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

    no_fetch = environment.get(cm.NO_FETCH_ENV)
    assert no_fetch == ("1" if kept else None)
    assert not any(name.startswith("R2_") for name in environment)
    names = set(environment)
    assert cm.CACHE_ENV not in names
    assert environment["HOME"] == str(run.temp / "repository-test-home")
    monkeypatch.delenv(cm.NO_FETCH_ENV, raising=False)
    without = ci_parity._repository_test_environment(run)
    expected = {
        name: value for name, value in environment.items() if name != cm.NO_FETCH_ENV
    }
    changed = changed_names(expected, without)
    assert changed == []


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


@pytest.mark.parametrize(
    "caller_locale", ["C", "en_US.ISO8859-1", "xx_XX.not-a-locale"]
)
def test_utf8_ctype_decodes_rulespec_text_as_ci_does(
    tmp_path: Path, caller_locale: str
) -> None:
    # Cycle 25: LC_ALL=C.UTF-8 in os.environ does not change the running
    # interpreter's LC_CTYPE, so in-process gates decoded files with the
    # caller's codeset: outside UTF-8 mode "§" became "Â§" (ISO-8859-1) or
    # raised (ASCII). C exists on every host, so this always runs somewhere.
    source = tmp_path / "statute.yaml"
    source.write_bytes("7 USC \u00a7 2014(e)\n".encode())
    expected = ascii("7 USC \u00a7 2014(e)\n")
    probe = (
        "import sys\n"
        "from pathlib import Path\n"
        "from axiom_encode import ci_parity\n"
        "def read():\n"
        "    try:\n"
        "        return ascii(Path(sys.argv[1]).read_text())\n"
        "    except UnicodeDecodeError:\n"
        "        return 'UnicodeDecodeError'\n"
        "print(read())\n"
        "ci_parity._adopt_utf8_ctype()\n"
        "print(read())\n"
    )
    # LC_ALL set: no C-locale coercion; -X utf8=0: no UTF-8 mode.
    environment = {**os.environ, "LC_ALL": caller_locale, "LANG": ""}
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
    if before == expected:
        pytest.skip(f"{caller_locale} already decodes UTF-8 on this host")
    assert after == expected


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


def test_children_get_the_adopted_utf8_locale_without_c_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Cycle 26: on a host without C.UTF-8, exporting LC_ALL=C.UTF-8 left
    # spawned workers in an unknown locale (ASCII outside UTF-8 mode); the
    # run exports the UTF-8 locale it actually adopted.
    real = locale.setlocale
    saved = real(locale.LC_CTYPE)
    try:
        real(locale.LC_CTYPE, "en_US.UTF-8")
    except locale.Error:
        pytest.skip("no en_US.UTF-8 locale on this host")
    finally:
        real(locale.LC_CTYPE, saved)

    def without_c_utf8(category: int, name: str | None = None) -> str:
        if name == "C.UTF-8":
            raise locale.Error("unsupported locale setting")
        return real(category, name)

    seen: dict[str, str] = {}

    def resolve(*_: object) -> str:
        seen.update(LANG=os.environ["LANG"], LC_ALL=os.environ["LC_ALL"])
        raise ValueError("stop after inspecting the runner environment")

    monkeypatch.setattr(ci_parity, "resolve_commit", resolve)
    ambient = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "PATH": os.defpath}
    # Such a host starts in a locale it has: C.
    real(locale.LC_CTYPE, "C")
    try:
        monkeypatch.setattr(ci_parity.locale, "setlocale", without_c_utf8)
        with (
            mock.patch.dict(os.environ, ambient, clear=True),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            args = SimpleNamespace(
                repo=Path("/unavailable"), base_ref="base", json=True
            )
            assert ci_parity.run_ci(args) == 1
            changed = changed_names(ambient, os.environ)
            assert changed == []
        monkeypatch.undo()
        assert real(locale.LC_CTYPE) == "C"
    finally:
        real(locale.LC_CTYPE, saved)

    assert seen == {"LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"}


def test_utf8_ctype_keeps_a_callers_own_utf8_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Cycle 26: a host with only de_DE.UTF-8 failed closed although its
    # LC_CTYPE already decoded UTF-8.
    current = {"name": "de_DE.UTF-8"}

    def only_the_callers(category: int, name: str | None = None) -> str:
        if name is None or name == "de_DE.UTF-8":
            return current["name"]
        raise locale.Error("unsupported locale setting")

    monkeypatch.setattr(ci_parity.locale, "setlocale", only_the_callers)
    monkeypatch.setattr(ci_parity.locale, "getencoding", lambda: "UTF-8")
    monkeypatch.setattr(
        ci_parity, "sys", SimpleNamespace(flags=SimpleNamespace(utf8_mode=0))
    )

    assert ci_parity._adopt_utf8_ctype() == "de_DE.UTF-8"


@pytest.mark.parametrize("caller_ctype", ["C", "en_US.UTF-8", "UTF-8"])
def test_c_utf8_is_adopted_first_wherever_it_exists(
    monkeypatch: pytest.MonkeyPatch, caller_ctype: str
) -> None:
    # Cycle 27: C.UTF-8 is CI's locale, so it wins over a caller's own UTF-8
    # locale wherever it exists; the caller's LC_CTYPE comes back afterwards.
    real = locale.setlocale
    saved = real(locale.LC_CTYPE)
    try:
        real(locale.LC_CTYPE, "C.UTF-8")
        real(locale.LC_CTYPE, caller_ctype)
    except locale.Error:
        real(locale.LC_CTYPE, saved)
        pytest.skip(f"host lacks C.UTF-8 or {caller_ctype}")
    caller = real(locale.LC_CTYPE)
    seen: dict[str, str] = {}

    def resolve(*_: object) -> str:
        seen.update(
            LANG=os.environ["LANG"],
            LC_ALL=os.environ["LC_ALL"],
            LC_CTYPE=real(locale.LC_CTYPE),
        )
        raise ValueError("stop after inspecting the runner environment")

    monkeypatch.setattr(ci_parity, "resolve_commit", resolve)
    ambient = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "PATH": os.defpath}
    try:
        with (
            mock.patch.dict(os.environ, ambient, clear=True),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            args = SimpleNamespace(
                repo=Path("/unavailable"), base_ref="base", json=True
            )
            assert ci_parity.run_ci(args) == 1
        restored = real(locale.LC_CTYPE)
    finally:
        real(locale.LC_CTYPE, saved)

    assert seen == {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "LC_CTYPE": "C.UTF-8"}
    assert restored == caller


_CURRENT_ROOT = base64.b64encode(bytes(range(32))).decode("ascii")
_APPLY_ROOT = bytes(range(64, 96))
_EVAL_ROOT = bytes(range(96, 128))


class _RetiredRootBroker:
    """An attached broker whose apply trust includes a retired root (#1903)."""

    apply_public_key_raw = bytes(range(160, 192))
    retired_apply_public_keys_raw = (bytes(range(192, 224)),)
    eval_public_key_raw = bytes(range(224, 256))


def _attach_broker(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    calls: list[object] = []

    def attached(*args: object, **kwargs: object) -> _RetiredRootBroker:
        calls.append((args, kwargs))
        return _RetiredRootBroker()

    monkeypatch.setattr(cli, "get_signing_broker", attached)
    monkeypatch.setattr(eval_evidence, "get_signing_broker", attached)
    return calls


def _raw(key: object) -> bytes:
    return key.public_bytes(  # type: ignore[attr-defined]
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def test_ci_verifies_apply_and_eval_only_with_its_own_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Cycle 28: without --apply-public-key, ci fell back to an attached
    # broker, whose retired roots (#1903) judge historical manifests unlike
    # the pinned CI; ci's own roots are now its only apply and eval trust.
    calls = _attach_broker(monkeypatch)

    with local_corpus_release_verification(_CURRENT_ROOT):
        assert cli._applied_encoding_manifest_verifier() is None
        with pytest.raises(ValueError, match="--eval-public-key"):
            eval_evidence.load_eval_evidence_public_key_from_broker()
    with local_corpus_release_verification(
        _CURRENT_ROOT,
        apply_public_key=base64.b64encode(_APPLY_ROOT).decode("ascii"),
        eval_public_key=base64.b64encode(_EVAL_ROOT).decode("ascii"),
    ):
        assert _raw(cli._applied_encoding_manifest_verifier()) == _APPLY_ROOT
        assert _raw(eval_evidence.load_eval_evidence_public_key_from_broker()) == (
            _EVAL_ROOT
        )
    assert calls == []

    # Outside ci the attached broker is the trust, as before.
    assert isinstance(cli._applied_encoding_manifest_verifier(), _RetiredRootBroker)
    assert _raw(eval_evidence.load_eval_evidence_public_key_from_broker()) == (
        _RetiredRootBroker.eval_public_key_raw
    )
    assert len(calls) == 2


@pytest.mark.parametrize("callers", ["de_DE.UTF-8", "en_US.UTF-8", "UTF-8"])
def test_c_utf8_is_preferred_to_a_callers_own_utf8_locale(
    monkeypatch: pytest.MonkeyPatch, callers: str
) -> None:
    # Cycle 28: the real-locale ordering test can skip on a host whose only
    # UTF-8 locale is C.UTF-8; this one pins the order everywhere.
    current = {"name": callers}

    def host(category: int, name: str | None = None) -> str:
        if name is None:
            return current["name"]
        if name in {"C.UTF-8", callers}:
            current["name"] = name
            return name
        raise locale.Error("unsupported locale setting")

    monkeypatch.setattr(ci_parity.locale, "setlocale", host)
    monkeypatch.setattr(ci_parity.locale, "getencoding", lambda: "UTF-8")

    assert ci_parity._adopt_utf8_ctype() == "C.UTF-8"
