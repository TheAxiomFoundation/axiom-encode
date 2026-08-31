from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/seal_targeted_reencode_artifact.py"
SPEC = importlib.util.spec_from_file_location("targeted_artifact_seal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
artifact_seal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifact_seal)


def test_copy_seal_and_verify_exact_flat_artifact(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "metadata.json").write_text('{"tree":"abc"}\n', encoding="utf-8")
    (candidate / "status.txt").write_text("M example.yaml\n", encoding="utf-8")
    sealed = tmp_path / "sealed"

    artifact_seal.copy_candidate(candidate, sealed)
    (candidate / "metadata.json").write_text('{"tree":"raced"}\n', encoding="utf-8")
    digest = artifact_seal.seal(sealed)

    assert artifact_seal.verify(sealed, require_root_owned=False) == digest
    assert (sealed / "metadata.json").read_text(encoding="utf-8") == (
        '{"tree":"abc"}\n'
    )
    manifest = json.loads(
        (sealed / artifact_seal.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["schema"] == artifact_seal.MANIFEST_SCHEMA
    assert [item["path"] for item in manifest["files"]] == [
        "metadata.json",
        "status.txt",
    ]
    assert not (sealed.stat().st_mode & 0o222)
    assert all(not (path.stat().st_mode & 0o222) for path in sealed.iterdir())


def test_copy_rejects_links_and_reserved_seal_entries(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside\n", encoding="utf-8")
    (candidate / "metadata.json").symlink_to(outside)

    with pytest.raises(artifact_seal.ArtifactSealError, match="unsafe"):
        artifact_seal.copy_candidate(candidate, tmp_path / "linked-seal")

    (candidate / "metadata.json").unlink()
    (candidate / artifact_seal.MANIFEST_NAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(artifact_seal.ArtifactSealError, match="reserved"):
        artifact_seal.copy_candidate(candidate, tmp_path / "reserved-seal")


def test_copy_can_drop_mutable_intermediate_guard_logs(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "metadata.json").write_text("{}\n", encoding="utf-8")
    (candidate / "guard-generated.json").write_text("{}\n", encoding="utf-8")
    (candidate / "source-01-guard-generated.json").write_text("{}\n", encoding="utf-8")
    sealed = tmp_path / "sealed"

    artifact_seal.copy_candidate(
        candidate,
        sealed,
        exclude_guard_logs=True,
    )

    assert sorted(path.name for path in sealed.iterdir()) == ["metadata.json"]


def test_copy_rejects_unreviewed_candidate_filename(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "metadata.json").write_text("{}\n", encoding="utf-8")
    (candidate / "forged.txt").write_text("forged\n", encoding="utf-8")

    with pytest.raises(artifact_seal.ArtifactSealError, match="unexpected entries"):
        artifact_seal.copy_candidate(candidate, tmp_path / "sealed")


def test_verify_rejects_post_seal_mutation_and_unmanifested_file(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "metadata.json").write_text("{}\n", encoding="utf-8")
    sealed = tmp_path / "sealed"
    artifact_seal.copy_candidate(candidate, sealed)
    artifact_seal.seal(sealed)

    os.chmod(sealed, 0o755)
    os.chmod(sealed / "metadata.json", 0o644)
    (sealed / "metadata.json").write_text('{"forged":true}\n', encoding="utf-8")
    with pytest.raises(artifact_seal.ArtifactSealError, match="differs"):
        artifact_seal.verify(sealed, require_root_owned=False)

    (sealed / "metadata.json").write_text("{}\n", encoding="utf-8")
    os.chmod(sealed / "metadata.json", 0o444)
    (sealed / "extra.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(artifact_seal.ArtifactSealError, match="unmanifested"):
        artifact_seal.verify(sealed, require_root_owned=False)


def test_publication_receipt_requires_exact_protected_head_and_base(
    tmp_path: Path,
) -> None:
    payload = {
        "base": {
            "ref": "main",
            "repo": {"full_name": "TheAxiomFoundation/rulespec-us"},
            "sha": "a" * 40,
        },
        "draft": True,
        "head": {
            "ref": "axiom-encode/run-1",
            "repo": {"full_name": "TheAxiomFoundation/rulespec-us"},
            "sha": "b" * 40,
        },
        "number": 123,
        "state": "open",
    }
    destination = tmp_path / "publication-receipt.json"
    artifact_seal.write_publication_receipt(
        destination,
        json.dumps(payload).encode("utf-8"),
        base_branch="main",
        base_commit="a" * 40,
        head_branch="axiom-encode/run-1",
        head_commit="b" * 40,
        repository="TheAxiomFoundation/rulespec-us",
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert not (destination.stat().st_mode & 0o222)

    with pytest.raises(artifact_seal.ArtifactSealError, match="does not bind"):
        artifact_seal.write_publication_receipt(
            tmp_path / "wrong.json",
            json.dumps(payload).encode("utf-8"),
            base_branch="main",
            base_commit="a" * 40,
            head_branch="axiom-encode/run-1",
            head_commit="c" * 40,
            repository="TheAxiomFoundation/rulespec-us",
        )


def test_publication_receipt_rejects_nonfinite_json_and_malformed_expectations(
    tmp_path: Path,
) -> None:
    with pytest.raises(artifact_seal.ArtifactSealError, match="non-finite"):
        artifact_seal.write_publication_receipt(
            tmp_path / "nan.json",
            b'{"base":NaN}',
            base_branch="main",
            base_commit="a" * 40,
            head_branch="axiom-encode/run-1",
            head_commit="b" * 40,
            repository="TheAxiomFoundation/rulespec-us",
        )

    with pytest.raises(artifact_seal.ArtifactSealError, match="base commit"):
        artifact_seal.write_publication_receipt(
            tmp_path / "malformed.json",
            b"{}",
            base_branch="main",
            base_commit="not-a-commit",
            head_branch="axiom-encode/run-1",
            head_commit="b" * 40,
            repository="TheAxiomFoundation/rulespec-us",
        )
