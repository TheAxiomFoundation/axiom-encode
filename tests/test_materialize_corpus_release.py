"""Tests for safe registry-backed corpus release materialization."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "materialize_corpus_release",
    ROOT / "scripts" / "materialize_corpus_release.py",
)
assert _SPEC is not None and _SPEC.loader is not None
release_acquisition = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release_acquisition)


def _response(
    release_name: str = "test-release",
    commit: str = "a" * 40,
) -> tuple[bytes, str, str]:
    content = {"git": {"commit": commit}}
    digest = hashlib.sha256(
        json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    payload = {
        "release": release_name,
        "content_sha256": digest,
        "content": content,
        "signature": {"value": "verified by the protected signer"},
    }
    return json.dumps([{"release_object": payload}]).encode(), digest, commit


def _write_toolchain(path: Path, release_name: str, digest: str) -> None:
    path.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{release_name}"\n'
        f'axiom_corpus_release_content_sha256 = "{digest}"\n'
    )


def test_materializes_valid_registry_response(tmp_path: Path) -> None:
    raw, digest, commit = _response()
    response = tmp_path / "response.json"
    response.write_bytes(raw)
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    destination, actual_commit = release_acquisition.materialize_registry_response(
        response,
        corpus,
        release_name="test-release",
        release_sha=digest,
    )

    assert actual_commit == commit
    assert destination == corpus / "releases" / "test-release" / f"{digest}.json"
    assert json.loads(destination.read_text())["content_sha256"] == digest


@pytest.mark.parametrize(
    ("response_factory", "message"),
    [
        (lambda raw, _digest: b"not-json", "invalid JSON"),
        (lambda raw, _digest: json.dumps([]).encode(), "exactly one"),
        (
            lambda raw, _digest: json.dumps(json.loads(raw) * 2).encode(),
            "exactly one",
        ),
        (
            lambda raw, _digest: raw.replace(
                b'"content_sha256": "', b'"content_sha256": "0'
            ),
            "content-address mismatch",
        ),
    ],
)
def test_rejects_invalid_registry_responses(
    tmp_path: Path,
    response_factory,
    message: str,
) -> None:
    raw, digest, _ = _response()
    response = tmp_path / "response.json"
    response.write_bytes(response_factory(raw, digest))
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    with pytest.raises(release_acquisition.ReleaseAcquisitionError, match=message):
        release_acquisition.materialize_registry_response(
            response,
            corpus,
            release_name="test-release",
            release_sha=digest,
        )


def test_rejects_invalid_git_provenance(tmp_path: Path) -> None:
    raw, digest, _ = _response(commit="invalid-commit")
    response = tmp_path / "response.json"
    response.write_bytes(raw)
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    with pytest.raises(
        release_acquisition.ReleaseAcquisitionError, match="Git provenance"
    ):
        release_acquisition.materialize_registry_response(
            response,
            corpus,
            release_name="test-release",
            release_sha=digest,
        )


def test_rejects_oversized_registry_response(tmp_path: Path) -> None:
    response = tmp_path / "response.json"
    response.write_bytes(b"x" * (release_acquisition.MAX_REGISTRY_RESPONSE_BYTES + 1))
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    with pytest.raises(release_acquisition.ReleaseAcquisitionError, match="16 MiB"):
        release_acquisition.materialize_registry_response(
            response,
            corpus,
            release_name="test-release",
            release_sha="0" * 64,
        )


def test_rejects_symlinked_release_directory(tmp_path: Path) -> None:
    raw, digest, _ = _response()
    response = tmp_path / "response.json"
    response.write_bytes(raw)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "releases").symlink_to(tmp_path / "outside", target_is_directory=True)

    with pytest.raises(
        release_acquisition.ReleaseAcquisitionError, match="safe directory"
    ):
        release_acquisition.materialize_registry_response(
            response,
            corpus,
            release_name="test-release",
            release_sha=digest,
        )
    assert not (tmp_path / "outside").exists()


def test_load_release_pin_rejects_unsafe_name(tmp_path: Path) -> None:
    toolchain = tmp_path / "toolchain.toml"
    _write_toolchain(toolchain, "../escape", "0" * 64)

    with pytest.raises(release_acquisition.ReleaseAcquisitionError, match="name"):
        release_acquisition.load_release_pin(toolchain)
