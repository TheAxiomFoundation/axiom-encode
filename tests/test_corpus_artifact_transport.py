"""Artifact transport cannot add source authority or change old dispatch modes."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import stat
import subprocess
import zipfile
from base64 import b64encode

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from axiom_encode import corpus_artifact_transport as transport
from axiom_encode.corpus_materialize import LockEntry, _canonical_lock_bytes
from axiom_encode.corpus_release import canonical_release_object_bytes
from axiom_encode.prepare_signed_backfill import (
    immutable_atomic_source_contract,
    split_atomic_source_input,
)
from tests.test_corpus_release import _signed_release_object

INNER = {
    "schema": "axiom-encode/atomic-source-transaction/v3",
    "source_bundle": [],
    "canonical_refresh_bundle": [],
    "primary_required_test_cases": [],
    "require_complete_source_unit": True,
}
WRAPPER = {
    "schema": "axiom-encode/corpus-artifact-transport/v1",
    "transaction": INNER,
    "corpus_artifact_run_id": "123",
}
DISPATCH = {
    "CITATION": "us/regulation/7/273/4",
    "REPLACE_RULESPEC_PATH": "us/regulations/7-cfr/273/4.yaml",
}


def test_wrapper_projects_only_source_contract():
    raw = json.dumps(WRAPPER)
    assert transport.request_run_id(raw, DISPATCH) == "123"
    assert immutable_atomic_source_contract(raw) == split_atomic_source_input(
        json.dumps(INNER)
    )
    assert json.loads(raw)["corpus_artifact_run_id"] == "123"


@pytest.mark.parametrize(
    "field",
    [
        "QUEUE_ID",
        "QUEUE_ITEM_ID",
        "QUEUE_MANIFEST_SHA256",
        "QUEUE_ITEM_GENERATION_SHA256",
        "QUEUE_DISPATCHER_RUN_ID",
        "REPAIR_RUN_ID",
        "DEPENDENT_CITATION",
        "SECOND_DEPENDENT_CITATION",
        "DEPENDENT_REVIEW_FINDING",
        "SECOND_DEPENDENT_REVIEW_FINDING",
        "REPLACE_LEGACY_RULESPEC_PATH",
        "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
        "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
        "EXISTING_SIGNED_IMPORTS_JSON",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON",
    ],
)
def test_excluded_dispatch_before_transport(field):
    with pytest.raises(ValueError):
        transport.request_run_id(json.dumps(WRAPPER), {**DISPATCH, field: '["x"]'})


@pytest.mark.parametrize(
    "change",
    ["extra", "nested", "v6", "false", "cases", "source", "refresh", "id", "path"],
)
def test_wrapper_strictness(change):
    wrapper = copy.deepcopy(WRAPPER)
    dispatch = dict(DISPATCH)
    if change == "extra":
        wrapper["ignored"] = True
    elif change == "nested":
        wrapper["transaction"] = copy.deepcopy(WRAPPER)
    elif change == "v6":
        wrapper["transaction"]["schema"] = "axiom-encode/atomic-source-transaction/v6"
    elif change == "false":
        wrapper["transaction"]["require_complete_source_unit"] = False
    elif change == "cases":
        wrapper["transaction"]["primary_required_test_cases"] = [{"name": "x"}]
    elif change == "source":
        wrapper["transaction"]["source_bundle"] = ["us/statute/x"]
    elif change == "refresh":
        wrapper["transaction"]["canonical_refresh_bundle"] = [{}]
    elif change == "id":
        wrapper["corpus_artifact_run_id"] = 123
    else:
        dispatch["REPLACE_RULESPEC_PATH"] = "other.yaml"
    with pytest.raises(ValueError):
        transport.request_run_id(json.dumps(wrapper), dispatch)


def test_duplicate_keys_and_oversized_wrapper_refuse():
    with pytest.raises(ValueError):
        transport.request_run_id(
            json.dumps(WRAPPER).replace(
                '"123"', '"123", "corpus_artifact_run_id": "124"'
            ),
            DISPATCH,
        )
    with pytest.raises(ValueError):
        split_atomic_source_input(" " * (512 * 1024) + json.dumps(WRAPPER))


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        json.dumps(INNER),
        '{"schema_version":"axiom-encode/retired-source-metadata-migration-plan/v2"}',
    ],
)
def test_old_request_never_selects_transport(raw):
    assert transport.request_run_id(raw, {}) == ""


def make_archive(payload=b"body", receipt=None, extra=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("payload.bin", payload)
        archive.writestr("receipt.json", json.dumps(receipt or {"test": True}))
        if extra is not None:
            archive.writestr(extra, b"bad")
    data = stream.getvalue()
    return data, {
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size_in_bytes": len(data),
    }


@pytest.mark.parametrize("extra", ["../escape", "payload.bin", "other", "/absolute"])
def test_archive_extra_duplicate_traversal_rejected(extra):
    data, artifact = make_archive(extra=extra)
    with pytest.raises(ValueError):
        transport.archive_members(data, artifact)


def test_archive_digest_and_size_and_symlink_refuse():
    data, artifact = make_archive()
    with pytest.raises(ValueError):
        transport.archive_members(data + b"tamper", artifact)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        item = zipfile.ZipInfo("payload.bin")
        item.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(item, b"target")
        archive.writestr("receipt.json", b"{}")
    data = stream.getvalue()
    with pytest.raises(ValueError):
        transport.archive_members(
            data,
            {
                "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
                "size_in_bytes": len(data),
            },
        )
    data, artifact = make_archive(payload=b"x" * (transport.MAX_PAYLOAD + 1))
    with pytest.raises(ValueError):
        transport.archive_members(data, artifact)


@pytest.fixture
def signed(tmp_path):
    key = Ed25519PrivateKey.generate()
    release, public = _signed_release_object(private_key=key)
    body = b"x" * 30
    member = next(
        a
        for a in release["content"]["artifacts"]
        if a["artifact_class"] == "provisions"
    )
    member["sha256"] = hashlib.sha256(body).hexdigest()
    member["r2_key"] = f"objects/sha256/{member['sha256'][:2]}/{member['sha256']}"
    release["content"]["validation"]["r2_readback"]["verified_keys"] = [
        a["r2_key"] for a in release["content"]["artifacts"]
    ]
    repo = tmp_path / "corpus"
    repo.mkdir()
    transport.git(repo, "init", "-q")
    transport.git(repo, "config", "user.name", "Test")
    transport.git(repo, "config", "user.email", "test@example.com")
    lock = transport.lock_path_for(member["path"])
    (repo / lock).parent.mkdir(parents=True)
    (repo / lock).write_bytes(
        _canonical_lock_bytes(
            ("nz", "statute", "2026-07-10-nz-rulespec"),
            {member["path"]: LockEntry(member["sha256"], len(body))},
        )
    )
    transport.git(repo, "add", ".")
    transport.git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    head = transport.git(repo, "rev-parse", "HEAD")
    release["content"]["git"]["commit"] = head
    release["content_sha256"] = hashlib.sha256(
        json.dumps(
            release["content"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    release["signature"]["value"] = b64encode(
        key.sign(canonical_release_object_bytes(release))
    ).decode()
    run = {"id": 123, "head_sha": head, "run_attempt": 1}
    receipt = {
        "schema_version": "axiom-corpus/release-artifact-export/v1",
        "release": release["release"],
        "content_sha256": release["content_sha256"],
        "source_commit": head,
        "path": member["path"],
        "sha256": member["sha256"],
        "bytes": len(body),
        "payload_filename": "payload.bin",
        "exporter_commit": head,
        "exporter_run_id": "123",
        "exporter_run_attempt": "1",
        "exporter_workflow_ref": f"{transport.REPOSITORY}/{transport.WORKFLOW}@refs/heads/main",
    }
    return dict(
        payload=body,
        receipt=receipt,
        release=release,
        public_key=public,
        pin=(release["release"], release["content_sha256"]),
        run=run,
        corpus=repo,
    )


def test_signed_member_lock_and_placement(signed):
    path = transport.verify_member(**signed)
    transport.place_member(signed["corpus"], path, signed["payload"])
    transport.place_member(signed["corpus"], path, signed["payload"])
    assert (signed["corpus"] / path).read_bytes() == signed["payload"]
    with pytest.raises(ValueError):
        transport.place_member(signed["corpus"], path, b"different")


@pytest.mark.parametrize(
    "change",
    [
        "signature",
        "release",
        "source",
        "path",
        "hash",
        "size",
        "run",
        "attempt",
        "workflow",
        "body",
        "lock",
    ],
)
def test_signed_authority_refusals(signed, change):
    if change == "signature":
        signed["release"]["signature"]["value"] = "a" * 88
    elif change == "release":
        signed["pin"] = ("wrong", signed["pin"][1])
    elif change == "source":
        signed["receipt"]["source_commit"] = "a" * 40
    elif change == "path":
        signed["receipt"]["path"] = "../escape"
    elif change == "hash":
        signed["receipt"]["sha256"] = "f" * 64
    elif change == "size":
        signed["receipt"]["bytes"] = True
    elif change == "run":
        signed["receipt"]["exporter_run_id"] = "456"
    elif change == "attempt":
        signed["receipt"]["exporter_run_attempt"] = "2"
    elif change == "workflow":
        signed["receipt"]["exporter_workflow_ref"] = "unreviewed"
    elif change == "body":
        signed["payload"] = b"y" * 30
    else:
        repo = signed["corpus"]
        lock = transport.lock_path_for(signed["receipt"]["path"])
        original = (repo / lock).read_text()
        (repo / lock).write_text(
            original.replace(signed["receipt"]["sha256"], "f" * 64)
        )
        transport.git(repo, "add", ".")
        transport.git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "bad-lock")
        commit = transport.git(repo, "rev-parse", "HEAD")
        signed["release"]["content"]["git"]["commit"] = commit
        signed["receipt"]["source_commit"] = commit
        content_sha = hashlib.sha256(
            json.dumps(
                signed["release"]["content"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
        signed["release"]["content_sha256"] = content_sha
        signed["receipt"]["content_sha256"] = content_sha
        signed["pin"] = (signed["pin"][0], content_sha)
        key = Ed25519PrivateKey.generate()
        signed["public_key"] = b64encode(
            key.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
        ).decode()
        signed["release"]["signature"]["value"] = b64encode(
            key.sign(canonical_release_object_bytes(signed["release"]))
        ).decode()
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
        transport.verify_member(**signed)


def test_symlink_parent_refused(tmp_path):
    (tmp_path / "data").symlink_to(tmp_path)
    with pytest.raises((ValueError, OSError)):
        transport.place_member(
            tmp_path, "data/corpus/provisions/us/statute/test.jsonl", b"x"
        )


def test_run_refusals_before_git(tmp_path):
    run = {
        "id": 123,
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "path": transport.WORKFLOW,
        "head_branch": "main",
        "repository": {"full_name": transport.REPOSITORY},
        "head_repository": {"full_name": transport.REPOSITORY},
        "head_sha": "a" * 40,
        "run_attempt": 1,
    }
    for field in (
        "id",
        "event",
        "status",
        "conclusion",
        "path",
        "head_branch",
        "repository",
        "head_repository",
        "head_sha",
        "run_attempt",
    ):
        bad = copy.deepcopy(run)
        bad[field] = {} if field.endswith("repository") else "bad"
        with pytest.raises(ValueError):
            transport.validate_run(bad, "123", tmp_path)


def test_artifact_metadata_exact_binding_and_negatives():
    run = {
        "id": 123,
        "head_sha": "a" * 40,
        "repository": {"id": 10},
        "head_repository": {"id": 10},
    }
    good = {
        "id": 456,
        "name": "authenticated-release-artifact",
        "expired": False,
        "digest": "sha256:" + "b" * 64,
        "size_in_bytes": 100,
        "workflow_run": {
            "id": 123,
            "head_sha": "a" * 40,
            "repository_id": 10,
            "head_repository_id": 10,
        },
    }
    assert (
        transport.select_artifact({"total_count": 1, "artifacts": [good]}, run) == good
    )
    for field, value in [
        ("name", "other"),
        ("expired", True),
        ("id", True),
        ("digest", None),
        ("size_in_bytes", True),
        ("size_in_bytes", transport.MAX_ARCHIVE + 1),
        ("workflow_run", {}),
    ]:
        bad = {**good, field: value}
        with pytest.raises(ValueError):
            transport.select_artifact({"total_count": 1, "artifacts": [bad]}, run)
    with pytest.raises(ValueError):
        transport.select_artifact({"total_count": 2, "artifacts": [good]}, run)


@pytest.mark.parametrize("changed_path", tuple(transport.REVIEWED_BLOBS))
def test_reviewed_exporter_ancestry_and_exact_blobs(signed, monkeypatch, changed_path):
    repo = signed["corpus"]
    for path in transport.REVIEWED_BLOBS:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("reviewed fixture " + path)
    transport.git(repo, "add", ".")
    transport.git(
        repo, "-c", "commit.gpgsign=false", "commit", "-qm", "reviewed-exporter"
    )
    head = transport.git(repo, "rev-parse", "HEAD")
    transport.git(
        repo,
        "remote",
        "add",
        "origin",
        f"https://github.com/{transport.REPOSITORY}.git",
    )
    transport.git(repo, "update-ref", "refs/remotes/origin/main", head)
    monkeypatch.setattr(transport, "REVIEWED_EXPORTER", head)
    monkeypatch.setattr(
        transport,
        "REVIEWED_BLOBS",
        {
            p: transport.git(repo, "rev-parse", f"HEAD:{p}")
            for p in transport.REVIEWED_BLOBS
        },
    )
    run = {
        "id": 123,
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "path": transport.WORKFLOW,
        "head_branch": "main",
        "repository": {"full_name": transport.REPOSITORY},
        "head_repository": {"full_name": transport.REPOSITORY},
        "head_sha": head,
        "run_attempt": 1,
    }
    transport.validate_run(run, "123", repo)
    (repo / changed_path).write_text("unreviewed")
    transport.git(repo, "add", ".")
    transport.git(
        repo, "-c", "commit.gpgsign=false", "commit", "-qm", "changed-exporter"
    )
    changed = transport.git(repo, "rev-parse", "HEAD")
    transport.git(repo, "update-ref", "refs/remotes/origin/main", changed)
    with pytest.raises(ValueError, match="blobs"):
        transport.validate_run({**run, "head_sha": changed}, "123", repo)
    transport.git(repo, "update-ref", "refs/remotes/origin/main", "HEAD~2")
    with pytest.raises(subprocess.CalledProcessError):
        transport.validate_run(run, "123", repo)


def test_destination_link_rejected(signed, tmp_path):
    path = signed["receipt"]["path"]
    target = signed["corpus"] / path
    target.parent.mkdir(parents=True)
    untouched = tmp_path / "untouched"
    untouched.write_bytes(b"original")
    target.symlink_to(untouched)
    with pytest.raises(OSError):
        transport.place_member(signed["corpus"], path, signed["payload"])
    assert untouched.read_bytes() == b"original"


def test_cli_excluded_modes_never_touch_artifact_service(monkeypatch):
    monkeypatch.setenv("ATOMIC_SOURCE_JSON", json.dumps(WRAPPER))
    for field, value in DISPATCH.items():
        monkeypatch.setenv(field, value)
    monkeypatch.setenv("QUEUE_ID", "queue")
    monkeypatch.setattr("sys.argv", ["transport", "materialize"])
    monkeypatch.setattr(
        transport,
        "read_regular",
        lambda *a, **k: pytest.fail("unexpected metadata read"),
    )
    with pytest.raises(ValueError, match="fresh ordinary"):
        transport.main()


def test_archive_length_must_match_github_metadata():
    data, artifact = make_archive()
    with pytest.raises(ValueError):
        transport.archive_members(data, {**artifact, "size_in_bytes": len(data) + 1})


def test_transport_still_enforces_fresh_attempt_budget(monkeypatch):
    from tests import test_attempt_budget as fixtures

    contract = fixtures.TestMainExitContract()
    contract._set_env(monkeypatch)
    contract._stub_api(monkeypatch, contract.FAILING_HISTORY)
    monkeypatch.setenv("SOURCE_BUNDLE_JSON", json.dumps(WRAPPER))
    assert (
        fixtures.budget_mod.is_retired_source_metadata_plan(json.dumps(WRAPPER))
        is False
    )
    assert fixtures.budget_mod.main() == 1


def test_staged_cli_revalidates_metadata_without_credentials(
    signed, monkeypatch, capsys, tmp_path
):
    """Placement never trusts the earlier validated artifact-ID output alone."""
    repo = signed["corpus"]
    for path in transport.REVIEWED_BLOBS:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("reviewed fixture " + path)
    transport.git(repo, "add", ".")
    transport.git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "exporter")
    head = transport.git(repo, "rev-parse", "HEAD")
    transport.git(
        repo,
        "remote",
        "add",
        "origin",
        f"https://github.com/{transport.REPOSITORY}.git",
    )
    transport.git(repo, "update-ref", "refs/remotes/origin/main", head)
    monkeypatch.setattr(transport, "REVIEWED_EXPORTER", head)
    monkeypatch.setattr(
        transport,
        "REVIEWED_BLOBS",
        {
            p: transport.git(repo, "rev-parse", f"HEAD:{p}")
            for p in transport.REVIEWED_BLOBS
        },
    )
    run = {
        "id": 123,
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "path": transport.WORKFLOW,
        "head_branch": "main",
        "head_sha": head,
        "run_attempt": 1,
        "repository": {"id": 10, "full_name": transport.REPOSITORY},
        "head_repository": {"id": 10, "full_name": transport.REPOSITORY},
    }
    receipt = {**signed["receipt"], "exporter_commit": head}
    data, archive_identity = make_archive(signed["payload"], receipt)
    artifact = {
        **archive_identity,
        "id": 456,
        "name": "authenticated-release-artifact",
        "expired": False,
        "workflow_run": {
            "id": 123,
            "head_sha": head,
            "repository_id": 10,
            "head_repository_id": 10,
        },
    }
    run_path, metadata_path, archive_path, release_path, toolchain = [
        tmp_path / n
        for n in (
            "run.json",
            "artifacts.json",
            "archive.zip",
            "release.json",
            "toolchain.toml",
        )
    ]
    run_path.write_text(json.dumps(run))
    metadata_path.write_text(json.dumps({"total_count": 1, "artifacts": [artifact]}))
    archive_path.write_bytes(data)
    release_path.write_text(json.dumps(signed["release"]))
    toolchain.write_text(
        f'[toolchain]\naxiom_corpus_release = "{signed["pin"][0]}"\naxiom_corpus_release_content_sha256 = "{signed["pin"][1]}"\n'
    )
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "AXIOM_REPO_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {
        **DISPATCH,
        "ATOMIC_SOURCE_JSON": json.dumps(WRAPPER),
        "TRUST_CORPUS_RELEASE_ROOT": signed["public_key"],
    }.items():
        monkeypatch.setenv(name, value)
    common = [
        "--corpus-root",
        str(repo),
        "--run-metadata",
        str(run_path),
        "--artifact-metadata",
        str(metadata_path),
    ]
    monkeypatch.setattr("sys.argv", ["transport", "validate-export", *common])
    transport.main()
    assert capsys.readouterr().out.strip() == "456"
    materialize = [
        "transport",
        "materialize",
        *common,
        "--toolchain",
        str(toolchain),
        "--release-object",
        str(release_path),
        "--archive",
        str(archive_path),
    ]
    monkeypatch.setattr("sys.argv", materialize)
    run_path.write_text(json.dumps({**run, "conclusion": "failure"}))
    with pytest.raises(ValueError, match="successful official"):
        transport.main()
    assert not (repo / receipt["path"]).exists()
    run_path.write_text(json.dumps(run))
    metadata_path.write_text(
        json.dumps(
            {
                "total_count": 1,
                "artifacts": [{**artifact, "digest": "sha256:" + "0" * 64}],
            }
        )
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        transport.main()
    assert not (repo / receipt["path"]).exists()
    metadata_path.write_text(json.dumps({"total_count": 1, "artifacts": [artifact]}))
    transport.main()
    assert (repo / receipt["path"]).read_bytes() == signed["payload"]
    assert json.loads(capsys.readouterr().out)["artifact_id"] == 456
