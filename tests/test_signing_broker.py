"""Security and protocol tests for the out-of-process signing broker."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
from base64 import b64encode
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from hypothesis import given
from hypothesis import strategies as st

from axiom_encode.corpus_release import RELEASE_OBJECT_PUBLIC_KEY_ENV
from axiom_encode.retired_apply import validate_retired_apply_inventory
from axiom_encode.signing_broker import (
    APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV,
    BROKER_FD_ENV,
    BROKER_MARKER_ENV,
    BROKER_PID_ENV,
    EVAL_EVIDENCE_PRIVATE_KEY_ENV,
    LEGACY_APPLY_SIGNING_KEY_ENV,
    SigningBrokerClient,
    SigningBrokerError,
    attach_signing_broker_from_environment,
    canonical_signing_message,
    reject_direct_private_signing_environment,
    scrub_private_signing_environment,
)
from tests.environment_assertions import changed_names
from tests.signing_broker_fixtures import SigningBrokerFixture

APPLY_MANIFEST_SIGNING_PUBLIC_KEY_ENV = "AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY"
EVAL_EVIDENCE_PUBLIC_KEY_ENV = "AXIOM_ENCODE_EVAL_SIGNING_PUBLIC_KEY"


def _keypair(seed: bytes) -> tuple[str, str, Ed25519PrivateKey]:
    private_key = Ed25519PrivateKey.from_private_bytes(seed)
    private_text = b64encode(seed).decode("ascii")
    public_text = b64encode(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode("ascii")
    return private_text, public_text, private_key


def test_typed_apply_and_eval_operations() -> None:
    apply_private, apply_public, apply_private_key = _keypair(b"\xab" * 32)
    eval_private, eval_public, eval_private_key = _keypair(b"\xcd" * 32)
    corpus_release_public = b64encode(b"\x17" * 32).decode("ascii")
    broker = SigningBrokerFixture(
        apply_private_key=apply_private,
        apply_public_key=apply_public,
        eval_private_key=eval_private,
        eval_public_key=eval_public,
        corpus_release_public_key=corpus_release_public,
    )
    payload = b"typed broker payload"

    apply_private_key.public_key().verify(
        broker.apply_ed25519_sign(payload),
        canonical_signing_message("apply_ed25519", payload),
    )
    eval_private_key.public_key().verify(
        broker.eval_ed25519_sign(payload),
        canonical_signing_message("eval_ed25519", payload),
    )
    assert broker.capabilities == frozenset({"apply_ed25519", "eval_ed25519"})
    assert broker.corpus_release_public_key_raw == b"\x17" * 32
    assert broker.corpus_release_public_keys_raw == (b"\x17" * 32,)


def test_apply_and_eval_signatures_are_not_cross_usable_even_with_one_test_key() -> (
    None
):
    private, public, private_key = _keypair(b"\xab" * 32)
    broker = SigningBrokerFixture(
        apply_private_key=private,
        apply_public_key=public,
        eval_private_key=private,
        eval_public_key=public,
    )
    payload = b"same canonical artifact bytes"
    apply_signature = broker.apply_ed25519_sign(payload)
    eval_signature = broker.eval_ed25519_sign(payload)

    with pytest.raises(InvalidSignature):
        private_key.public_key().verify(
            apply_signature,
            canonical_signing_message("eval_ed25519", payload),
        )
    with pytest.raises(InvalidSignature):
        private_key.public_key().verify(
            eval_signature,
            canonical_signing_message("apply_ed25519", payload),
        )


def test_missing_capability_fails_closed() -> None:
    apply_private, apply_public, _private_key = _keypair(b"\xab" * 32)
    broker = SigningBrokerFixture(
        apply_private_key=apply_private,
        apply_public_key=apply_public,
    )

    with pytest.raises(SigningBrokerError, match="not provisioned"):
        broker.eval_ed25519_sign(b"payload")


@pytest.mark.parametrize("key_kind", ["private", "public"])
def test_apply_key_requires_valid_ed25519_material(key_kind: str) -> None:
    apply_private, apply_public, _private_key = _keypair(b"\xab" * 32)
    kwargs = {
        "apply_private_key": apply_private,
        "apply_public_key": apply_public,
    }
    kwargs[f"apply_{key_kind}_key"] = b64encode(b"short").decode("ascii")
    with pytest.raises(SigningBrokerError, match="32 raw bytes"):
        SigningBrokerFixture(**kwargs)


def test_apply_private_and_public_keys_must_match() -> None:
    apply_private, _apply_public, _private_key = _keypair(b"\xab" * 32)
    _other_private, other_public, _other_key = _keypair(b"\xac" * 32)

    with pytest.raises(SigningBrokerError, match="do not match"):
        SigningBrokerFixture(
            apply_private_key=apply_private,
            apply_public_key=other_public,
        )


@pytest.mark.parametrize(
    ("private_name", "private_value"),
    [
        (LEGACY_APPLY_SIGNING_KEY_ENV, "private"),
        (LEGACY_APPLY_SIGNING_KEY_ENV, ""),
        (APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV, "private"),
        (APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV, ""),
        (EVAL_EVIDENCE_PRIVATE_KEY_ENV, "private"),
        (EVAL_EVIDENCE_PRIVATE_KEY_ENV, ""),
    ],
)
def test_production_rejects_direct_private_key_environment(
    private_name: str,
    private_value: str,
) -> None:
    with pytest.raises(SigningBrokerError, match="externally provisioned"):
        reject_direct_private_signing_environment({private_name: private_value})


def test_console_entrypoint_rejects_private_environment_before_cli_dispatch(
    tmp_path,
) -> None:
    environment = scrub_private_signing_environment()
    environment[APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV] = "must-not-enter-cli"
    source_root = Path(__file__).parents[1] / "src"
    dispatch_current_checkout = (
        "import sys; "
        f"sys.path.insert(0, {str(source_root)!r}); "
        "from axiom_encode.entrypoint import main; "
        "raise SystemExit(main())"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", dispatch_current_checkout, "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "externally provisioned signing broker" in completed.stderr
    assert "Axiom Encode CLI" not in completed.stdout


def test_readme_does_not_name_private_signing_environment_variables() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text()

    assert APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV not in readme
    assert EVAL_EVIDENCE_PRIVATE_KEY_ENV not in readme
    assert LEGACY_APPLY_SIGNING_KEY_ENV not in readme


def test_scrub_removes_private_keys_and_broker_handles() -> None:
    environment = {
        "SAFE": "kept",
        APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV: "apply-private",
        APPLY_MANIFEST_SIGNING_PUBLIC_KEY_ENV: "apply-public",
        EVAL_EVIDENCE_PRIVATE_KEY_ENV: "eval-private",
        EVAL_EVIDENCE_PUBLIC_KEY_ENV: "eval-public",
        RELEASE_OBJECT_PUBLIC_KEY_ENV: "corpus-release-public",
        BROKER_MARKER_ENV: "1",
        BROKER_FD_ENV: "7",
        BROKER_PID_ENV: "123",
        "PATH": "/hostile/bin",
        "HOME": "/hostile/home",
        "GIT_CONFIG_GLOBAL": "/hostile/gitconfig",
    }

    scrubbed = scrub_private_signing_environment(environment)
    changed = changed_names(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "PATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "XDG_CONFIG_HOME": "/nonexistent",
            "XDG_DATA_HOME": "/nonexistent",
        },
        scrubbed,
    )
    assert changed == []


@pytest.mark.parametrize(
    ("descriptor", "broker_pid"),
    [("2", "1"), ("3", "0"), ("-1", "42")],
)
def test_attach_rejects_non_capability_descriptor_or_pid(
    monkeypatch,
    descriptor: str,
    broker_pid: str,
) -> None:
    monkeypatch.setenv(BROKER_MARKER_ENV, "1")
    monkeypatch.setenv(BROKER_FD_ENV, descriptor)
    monkeypatch.setenv(BROKER_PID_ENV, broker_pid)

    with pytest.raises(SigningBrokerError, match="Malformed"):
        attach_signing_broker_from_environment()


def test_environment_pid_cannot_counterfeit_kernel_broker_peer(monkeypatch) -> None:
    client, counterfeit_peer = socket.socketpair()
    responder: threading.Thread | None = None
    if sys.platform.startswith("linux"):

        def respond() -> None:
            assert counterfeit_peer.recv(1) == b"\xa5"
            counterfeit_peer.sendall(b"\x5a")

        responder = threading.Thread(target=respond, daemon=True)
        responder.start()
    try:
        monkeypatch.setenv(BROKER_MARKER_ENV, "1")
        monkeypatch.setenv(BROKER_FD_ENV, str(client.fileno()))
        monkeypatch.setenv(BROKER_PID_ENV, str(os.getpid() + 100_000))

        with pytest.raises(SigningBrokerError, match="peer identity mismatch"):
            attach_signing_broker_from_environment()
    finally:
        try:
            client.close()
        except OSError:
            pass
        counterfeit_peer.close()
        if responder is not None:
            responder.join(timeout=2)


def _frozen_status_inventory() -> dict[str, object]:
    return {
        "schema": "axiom-encode/retired-apply-inventory/v1",
        "repositories": [
            {
                "repository": "TheAxiomFoundation/rulespec-us",
                "cutoff_commit": "a" * 40,
                "artifacts": [
                    {
                        "path": ".axiom/encoding-manifests/frozen.json",
                        "sha256": "b" * 64,
                        "covered_files": [
                            {"path": "statute/frozen.yaml", "sha256": "c" * 64}
                        ],
                    }
                ],
            }
        ],
    }


def _base_protocol_status() -> dict[str, object]:
    return {
        "capabilities": [],
        "apply_public_key": b64encode(b"a" * 32).decode(),
        "eval_public_key": b64encode(b"b" * 32).decode(),
        "corpus_release_public_key": b64encode(b"c" * 32).decode(),
        "corpus_release_public_keys": [b64encode(b"c" * 32).decode()],
    }


def _client_for_status(status: dict[str, object]) -> SigningBrokerClient:
    connection, peer = socket.socketpair()
    peer.close()
    try:
        with patch.object(SigningBrokerClient, "_request", return_value=status):
            return SigningBrokerClient(connection)
    except BaseException:
        connection.close()
        raise


def test_broker_preserves_retired_roots_and_indexes_inventory_once() -> None:
    status = _base_protocol_status()
    inventory = _frozen_status_inventory()
    status["retired_apply_public_keys"] = [b64encode(b"d" * 32).decode()]
    status["retired_apply_inventory"] = inventory
    with patch(
        "axiom_encode.retired_apply.validate_retired_apply_inventory",
        wraps=validate_retired_apply_inventory,
    ) as validate:
        client = _client_for_status(status)
        try:
            assert client.capabilities == frozenset()
            assert client.apply_public_key_raw == b"a" * 32
            assert client.retired_apply_public_keys_raw == (b"d" * 32,)
            exposed = client.retired_apply_inventory
            assert exposed is not None
            assert not hasattr(exposed, "__dict__")
            exposed.__init__(inventory)  # Tuple initialization cannot replace data.
            with pytest.raises(AttributeError):
                exposed._repositories = {}
            repository = exposed["TheAxiomFoundation/rulespec-us"]
            artifacts = repository["artifacts"]
            artifact = artifacts[".axiom/encoding-manifests/frozen.json"]
            covered = artifact["covered_files"]
            assert repository["cutoff_commit"] == "a" * 40
            assert artifact["sha256"] == "b" * 64
            assert covered == ({"path": "statute/frozen.yaml", "sha256": "c" * 64},)
            with pytest.raises(TypeError):
                exposed["TheAxiomFoundation/rulespec-us"] = repository
            with pytest.raises(TypeError):
                repository["cutoff_commit"] = "d" * 40
            with pytest.raises(TypeError):
                artifacts[artifact["path"]] = artifact
            with pytest.raises(TypeError):
                artifact["sha256"] = "d" * 64
            with pytest.raises(TypeError):
                covered[0] = {"path": "changed.yaml", "sha256": "d" * 64}
            with pytest.raises(TypeError):
                covered[0]["sha256"] = "d" * 64
            raw_repository = inventory["repositories"][0]
            raw_artifact = raw_repository["artifacts"][0]
            raw_covered = raw_artifact["covered_files"][0]
            inventory["schema"] = "caller-mutated"
            raw_repository["cutoff_commit"] = "d" * 40
            raw_artifact["sha256"] = "d" * 64
            raw_covered["sha256"] = "d" * 64
            raw_artifact["covered_files"].clear()
            raw_repository["artifacts"].clear()
            inventory["repositories"].clear()
            assert repository["cutoff_commit"] == "a" * 40
            assert artifact["sha256"] == "b" * 64
            assert covered == ({"path": "statute/frozen.yaml", "sha256": "c" * 64},)
            for _ in range(10):
                assert client.retired_apply_inventory is exposed
            assert validate.call_count == 1
        finally:
            client.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_roots",
        "missing_inventory",
        "empty",
        "malformed",
        "wrong_length",
        "duplicate",
        "active_overlap",
        "eval_overlap",
        "corpus_overlap",
        "unknown_inventory_field",
    ],
)
def test_broker_rejects_malformed_retired_status(mutation: str) -> None:
    status = _base_protocol_status()
    status["retired_apply_public_keys"] = [b64encode(b"d" * 32).decode()]
    status["retired_apply_inventory"] = deepcopy(_frozen_status_inventory())
    if mutation == "missing_roots":
        del status["retired_apply_public_keys"]
    elif mutation == "missing_inventory":
        del status["retired_apply_inventory"]
    elif mutation == "empty":
        status["retired_apply_public_keys"] = []
    elif mutation == "malformed":
        status["retired_apply_public_keys"] = ["bad-base64!!"]
    elif mutation == "wrong_length":
        status["retired_apply_public_keys"] = [b64encode(b"short").decode()]
    elif mutation == "duplicate":
        status["retired_apply_public_keys"] *= 2
    elif mutation.endswith("_overlap"):
        field = {
            "active_overlap": "apply_public_key",
            "eval_overlap": "eval_public_key",
            "corpus_overlap": "corpus_release_public_key",
        }[mutation]
        status["retired_apply_public_keys"] = [status[field]]
    else:
        status["retired_apply_inventory"]["caller_override"] = True
    with pytest.raises(SigningBrokerError, match="retired apply"):
        _client_for_status(status)


@given(
    st.lists(
        st.integers(min_value=0, max_value=255), min_size=3, max_size=3, unique=True
    )
)
def test_legacy_broker_status_keeps_original_roots_without_retirement(
    values: list[int],
) -> None:
    roots = [bytes([value]) * 32 for value in values]
    status = {
        "capabilities": [],
        "apply_public_key": b64encode(roots[0]).decode(),
        "eval_public_key": b64encode(roots[1]).decode(),
        "corpus_release_public_key": b64encode(roots[2]).decode(),
    }
    for plural in (False, True):
        message = dict(status)
        if plural:
            message["corpus_release_public_keys"] = [
                message["corpus_release_public_key"]
            ]
        client = _client_for_status(message)
        try:
            assert client.apply_public_key_raw == roots[0]
            assert client.eval_public_key_raw == roots[1]
            assert client.corpus_release_public_keys_raw == (roots[2],)
            assert client.retired_apply_public_keys_raw == ()
            assert client.retired_apply_inventory is None
        finally:
            client.close()
