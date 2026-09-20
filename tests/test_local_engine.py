"""Explicit local-engine selection must never silently skip or change engines."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import local_engine


def test_explicit_missing_engine_fails_even_when_a_default_is_available(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AXIOM_TEST_ENGINE_CHECKOUT", str(tmp_path / "missing"))
    monkeypatch.delenv("AXIOM_TEST_ENGINE_REF", raising=False)
    default_binary = tmp_path / "target/release/axiom-rules-engine"
    default_binary.parent.mkdir(parents=True)
    default_binary.write_bytes(b"synthetic probe fixture")
    monkeypatch.setattr(local_engine, "DEFAULT_ENGINE_CHECKOUTS", (tmp_path,))
    monkeypatch.setattr(
        local_engine.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="compile --rulespec-root", stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="no compatible executable"):
        local_engine.local_engine_paths()


def test_incompatible_release_is_not_hidden_by_compatible_debug(tmp_path, monkeypatch):
    monkeypatch.setenv("AXIOM_TEST_ENGINE_CHECKOUT", str(tmp_path))
    monkeypatch.delenv("AXIOM_TEST_ENGINE_REF", raising=False)
    for profile in ("release", "debug"):
        binary = tmp_path / "target" / profile / "axiom-rules-engine"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"synthetic probe fixture")
    calls = []

    def probe(argv, **kwargs):
        calls.append(argv[0])
        return SimpleNamespace(returncode=0, stdout="legacy help", stderr="")

    monkeypatch.setattr(local_engine.subprocess, "run", probe)
    with pytest.raises(RuntimeError, match="no compatible executable"):
        local_engine.local_engine_paths()
    assert calls == [str(tmp_path / "target/release/axiom-rules-engine")]


def test_explicit_receipt_binding_must_select_the_executed_binary(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AXIOM_TEST_ENGINE_CHECKOUT", str(tmp_path))
    monkeypatch.setenv("AXIOM_TEST_ENGINE_REF", "a" * 40)
    binary = tmp_path / "target/release/axiom-rules-engine"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"synthetic probe fixture")
    calls = []

    def bind(checkout, ref, *, allow_build):
        calls.append((checkout, ref, allow_build))
        return {"binary": str(tmp_path / "different-engine")}

    monkeypatch.setattr(local_engine, "bind_clean_engine_checkout", bind)
    with pytest.raises(RuntimeError, match="different binary"):
        local_engine.local_engine_paths()
    assert calls == [(Path(tmp_path), "a" * 40, False)]


def test_pin_without_checkout_cannot_bind_an_ambient_engine(monkeypatch):
    monkeypatch.delenv("AXIOM_TEST_ENGINE_CHECKOUT", raising=False)
    monkeypatch.setenv("AXIOM_TEST_ENGINE_REF", "a" * 40)
    with pytest.raises(RuntimeError, match="requires AXIOM_TEST_ENGINE_CHECKOUT"):
        local_engine.local_engine_paths()


@pytest.mark.parametrize(
    "name", ["AXIOM_TEST_ENGINE_CHECKOUT", "AXIOM_TEST_ENGINE_REF"]
)
@pytest.mark.parametrize("value", ["", " ", "\t"])
def test_explicit_blank_selection_refuses_before_probe(
    tmp_path, monkeypatch, name, value
):
    monkeypatch.setenv("AXIOM_TEST_ENGINE_CHECKOUT", str(tmp_path))
    monkeypatch.delenv("AXIOM_TEST_ENGINE_REF", raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=f"{name} must not be blank"):
        local_engine.local_engine_paths()


def test_release_directory_cannot_be_hidden_by_debug(tmp_path, monkeypatch):
    monkeypatch.setenv("AXIOM_TEST_ENGINE_CHECKOUT", str(tmp_path))
    monkeypatch.delenv("AXIOM_TEST_ENGINE_REF", raising=False)
    release = tmp_path / "target/release/axiom-rules-engine"
    release.mkdir(parents=True)
    debug = tmp_path / "target/debug/axiom-rules-engine"
    debug.parent.mkdir(parents=True)
    debug.write_bytes(b"synthetic probe fixture")
    with pytest.raises(RuntimeError, match="not a file"):
        local_engine.local_engine_paths()


@pytest.mark.parametrize(
    "effect",
    [
        SimpleNamespace(returncode=1, stdout="--rulespec-root", stderr=""),
        subprocess.TimeoutExpired("probe", 30),
        PermissionError("synthetic probe failure"),
    ],
)
def test_explicit_probe_failure_cannot_skip(tmp_path, monkeypatch, effect):
    monkeypatch.setenv("AXIOM_TEST_ENGINE_CHECKOUT", str(tmp_path))
    monkeypatch.delenv("AXIOM_TEST_ENGINE_REF", raising=False)
    binary = tmp_path / "target/release/axiom-rules-engine"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"synthetic probe fixture")

    def probe(*args, **kwargs):
        if isinstance(effect, BaseException):
            raise effect
        return effect

    monkeypatch.setattr(local_engine.subprocess, "run", probe)
    with pytest.raises((RuntimeError, OSError, subprocess.TimeoutExpired)):
        local_engine.local_engine_paths()


def test_invalid_binding_refuses_before_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("AXIOM_TEST_ENGINE_CHECKOUT", str(tmp_path))
    monkeypatch.setenv("AXIOM_TEST_ENGINE_REF", "a" * 40)
    binary = tmp_path / "target/release/axiom-rules-engine"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"synthetic probe fixture")

    def bind(*args, **kwargs):
        raise RuntimeError("invalid receipt")

    def probe(*args, **kwargs):
        pytest.fail("invalid binding must refuse before probing")

    monkeypatch.setattr(local_engine, "bind_clean_engine_checkout", bind)
    monkeypatch.setattr(local_engine.subprocess, "run", probe)
    with pytest.raises(RuntimeError, match="invalid receipt"):
        local_engine.local_engine_paths()


def test_matching_explicit_pin_probes_same_binary_without_build(tmp_path, monkeypatch):
    monkeypatch.setenv("AXIOM_TEST_ENGINE_CHECKOUT", str(tmp_path))
    monkeypatch.setenv("AXIOM_TEST_ENGINE_REF", "a" * 40)
    binary = tmp_path / "target/release/axiom-rules-engine"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"synthetic probe fixture")
    calls = []

    def bind(checkout, ref, *, allow_build):
        calls.append((checkout, ref, allow_build))
        return {"binary": str(binary.resolve())}

    def probe(argv, **kwargs):
        assert argv == [str(binary), "compile", "--help"]
        return SimpleNamespace(returncode=0, stdout="--rulespec-root", stderr="")

    monkeypatch.setattr(local_engine, "bind_clean_engine_checkout", bind)
    monkeypatch.setattr(local_engine.subprocess, "run", probe)
    assert local_engine.local_engine_paths() == (tmp_path, binary)
    assert calls == [(tmp_path, "a" * 40, False)]
