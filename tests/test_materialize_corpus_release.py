"""Tests for safe registry-backed corpus release materialization."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path

import pytest
import yaml

from axiom_encode.corpus_resolver import MAX_RELEASE_OBJECT_BYTES

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
    *,
    artifact_count: int = 0,
) -> tuple[bytes, str, str]:
    content: dict[str, object] = {"git": {"commit": commit}}
    if artifact_count:
        content["artifacts"] = [
            {
                "artifact_class": "sources",
                "path": f"data/corpus/sources/us/statute/v/source-{index:06d}.html",
                "sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            }
            for index in range(artifact_count)
        ]
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


def test_materializes_registry_response_larger_than_former_cap(
    tmp_path: Path,
) -> None:
    raw, digest, commit = _response(artifact_count=120_000)
    assert len(raw) > 16 * 1024 * 1024
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
    assert destination.stat().st_size > len(raw)
    assert json.loads(destination.read_text())["content_sha256"] == digest


def test_every_release_object_fetch_uses_the_resolver_cap() -> None:
    cap = release_acquisition.MAX_REGISTRY_RESPONSE_BYTES
    assert cap == MAX_RELEASE_OBJECT_BYTES
    fetch_steps = []
    for workflow in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        for job in (document.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                command = step.get("run") or ""
                if "release_objects" not in command and "/releases/" not in command:
                    continue
                fetch_steps.append((workflow.name, step.get("name")))
                assert not re.search(
                    r"\bwget\b|urlopen|gh release download", command
                ), (workflow.name, step.get("name"))
                curls = re.findall(r"curl (?:[^\n]*\\\n)*[^\n]*", command)
                assert curls, (workflow.name, step.get("name"))
                for curl in curls:
                    assert re.findall(r"--max-filesize[= ]+(\S+)", curl) == [
                        str(cap)
                    ], (workflow.name, curl)
    assert len(fetch_steps) >= 4, fetch_steps


def _grow_after_fstat(monkeypatch, target: Path) -> list[int]:
    real_fstat = os.fstat
    real_read = os.read
    returned: list[int] = []

    def fstat_then_grow(descriptor):
        result = real_fstat(descriptor)
        if os.path.samestat(result, os.stat(target)):
            with target.open("ab") as stream:
                stream.write(b"y" * (4 * 1024 * 1024))
        return result

    def counting_read(descriptor, size):
        chunk = real_read(descriptor, size)
        returned.append(len(chunk))
        return chunk

    monkeypatch.setattr(os, "fstat", fstat_then_grow)
    monkeypatch.setattr(os, "read", counting_read)
    return returned


def test_response_read_stops_at_limit_when_file_grows_after_fstat(
    tmp_path: Path, monkeypatch
) -> None:
    response = tmp_path / "response.json"
    response.write_bytes(b"x" * 16)
    monkeypatch.setattr(release_acquisition, "MAX_REGISTRY_RESPONSE_BYTES", 16)
    returned = _grow_after_fstat(monkeypatch, response)

    with pytest.raises(release_acquisition.ReleaseAcquisitionError):
        release_acquisition._read_bounded_regular_file(response)
    assert sum(returned) == 17


def test_oversized_response_is_rejected_before_read(
    tmp_path: Path, monkeypatch
) -> None:
    response = tmp_path / "response.json"
    response.write_bytes(b"x" * 17)
    monkeypatch.setattr(release_acquisition, "MAX_REGISTRY_RESPONSE_BYTES", 16)

    def unexpected_read(descriptor, size):
        raise AssertionError("an oversized registry response must not be read")

    monkeypatch.setattr(os, "read", unexpected_read)

    with pytest.raises(release_acquisition.ReleaseAcquisitionError):
        release_acquisition._read_bounded_regular_file(response)


def test_rejects_oversized_registry_response(tmp_path: Path) -> None:
    response = tmp_path / "response.json"
    with response.open("wb") as stream:
        stream.truncate(release_acquisition.MAX_REGISTRY_RESPONSE_BYTES + 1)
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    with pytest.raises(release_acquisition.ReleaseAcquisitionError, match="64 MiB"):
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
