"""Artifact transport cannot add source authority or change old dispatch modes."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import stat
import subprocess
import sys
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
    resolve_atomic_repair_mode,
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
PREFETCH = {
    "schema": "axiom-encode/corpus-provision-prefetch/v1",
    "transaction": INNER,
    "provision_export_run_ids": ["123", "456", "789"],
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


def test_provision_prefetch_projects_only_source_contract():
    raw = json.dumps(PREFETCH)
    assert transport.request_run_id(raw, DISPATCH) == ""
    assert transport.provision_run_ids(raw) == ["123", "456", "789"]
    assert immutable_atomic_source_contract(raw) == split_atomic_source_input(
        json.dumps(INNER)
    )


def test_provision_run_ids_cli_emits_no_blank_run(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["corpus_artifact_transport", "provision-run-ids"])
    monkeypatch.setenv("ATOMIC_SOURCE_JSON", "[]")
    transport.main()
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("ATOMIC_SOURCE_JSON", json.dumps(PREFETCH))
    transport.main()
    assert capsys.readouterr().out == "123\n456\n789\n"


def test_provision_materialization_rejects_unlisted_run_before_any_io(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["corpus_artifact_transport", "materialize", "--provision-run-id", "999"],
    )
    monkeypatch.setenv("ATOMIC_SOURCE_JSON", json.dumps(PREFETCH))
    with pytest.raises(ValueError, match="not authorized"):
        transport.main()


@pytest.mark.parametrize(
    "ids",
    [[], ["0"], ["01"], ["123", "123"], [123], ["1" * 21], ["1"] * 9],
)
def test_provision_prefetch_rejects_invalid_run_ids(ids):
    with pytest.raises(ValueError, match="run IDs"):
        transport.provision_run_ids(
            json.dumps({**PREFETCH, "provision_export_run_ids": ids})
        )


@pytest.mark.parametrize(
    "change", ["extra", "nested", "repair", "not_object", "v4", "v5"]
)
def test_provision_prefetch_rejects_mixed_or_unknown_envelopes(change):
    wrapper = copy.deepcopy(PREFETCH)
    if change == "extra":
        wrapper["unknown"] = True
    elif change == "nested":
        wrapper["transaction"] = copy.deepcopy(WRAPPER)
    elif change == "repair":
        wrapper["transaction"]["schema"] = "axiom-encode/atomic-source-transaction/v6"
    elif change in {"v4", "v5"}:
        wrapper["transaction"]["schema"] = (
            f"axiom-encode/atomic-source-transaction/{change}"
        )
    else:
        wrapper["transaction"] = []
    with pytest.raises(ValueError):
        transport.provision_run_ids(json.dumps(wrapper))


@pytest.mark.parametrize(
    "run_id",
    ["123;id", "123\n", "$(id)", "0", "1" * 21, "１２３"],
)
def test_transport_run_id_must_be_ascii_canonical_decimal(run_id):
    wrapper = {**WRAPPER, "corpus_artifact_run_id": run_id}
    with pytest.raises(ValueError, match="run|transport"):
        transport.request_run_id(json.dumps(wrapper), DISPATCH)


def test_recursive_request_is_rejected(monkeypatch):
    def recurse(_raw):
        raise RecursionError("nested transport")

    monkeypatch.setattr(transport.json, "loads", recurse)
    assert transport.request_run_id("{}", DISPATCH) == ""


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
        "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
        "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
        "EXISTING_SIGNED_IMPORTS_JSON",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON",
    ],
)
def test_excluded_dispatch_before_transport(field):
    with pytest.raises(ValueError):
        transport.request_run_id(json.dumps(WRAPPER), {**DISPATCH, field: '["x"]'})


def test_in_place_legacy_replacement_transport_preserves_ece_dependents():
    legacy_path = (
        "us-az/policies/des/faa5/na-categorical-eligibility/"
        "expanded-categorical-eligibility.yaml"
    )
    legacy_identity = (
        "us-az:policies/des/faa5/na-categorical-eligibility/"
        "expanded-categorical-eligibility"
    )
    dispatch = {
        "CITATION": "us-az/manual/des/faa5/na-categorical-eligibility/block-4",
        "REPLACE_RULESPEC_PATH": legacy_path,
        "REPLACE_LEGACY_RULESPEC_PATH": legacy_path,
    }
    assert transport.request_run_id(json.dumps(WRAPPER), dispatch) == "123"

    with_cases = copy.deepcopy(WRAPPER)
    with_cases["transaction"]["primary_required_test_cases"] = [
        {
            "name": "2026 February ECE rate",
            "period": {
                "period_kind": "month",
                "start": "2026-02-01",
                "end": "2026-02-28",
            },
            "input": {},
            "required_output": {
                f"{legacy_identity}#expanded_categorical_eligibility_fpl_limit_rate": 1.85
            },
        },
        {
            "name": "2026 March ECE rate and deemed tests",
            "period": {
                "period_kind": "month",
                "start": "2026-03-01",
                "end": "2026-03-31",
            },
            "input": {
                f"{legacy_identity}#input.gross_income": 1900,
                f"{legacy_identity}#input.current_federal_poverty_level_amount": 1000,
                f"{legacy_identity}#input.budgetary_unit_meets_basic_categorical_eligibility_requirements": False,
                f"{legacy_identity}#input.resources_of_participants_with_certain_disqualifications_are_not_over_na_resource_limit": True,
                f"{legacy_identity}#input.budgetary_unit_has_participant_with_disqualification": False,
            },
            "required_output": {
                f"{legacy_identity}#expanded_categorical_eligibility_fpl_limit_rate": 2.0,
                f"{legacy_identity}#expanded_categorical_eligibility_deems_resources_met": "holds",
            },
        },
    ]
    assert transport.request_run_id(json.dumps(with_cases), dispatch) == "123"
    assert resolve_atomic_repair_mode(json.dumps(with_cases), dispatch) == {
        "mode": "full_artifact",
        "tests_only": False,
    }


def test_corpus_transport_rejects_arbitrary_legacy_path_move():
    dispatch = {
        "CITATION": "us-az/manual/des/faa5/na-categorical-eligibility/block-4",
        "REPLACE_RULESPEC_PATH": (
            "us-az/policies/des/faa5/na-categorical-eligibility/block-4.yaml"
        ),
        "REPLACE_LEGACY_RULESPEC_PATH": (
            "us-az/policies/des/faa5/na-categorical-eligibility/"
            "expanded-categorical-eligibility.yaml"
        ),
    }
    with pytest.raises(ValueError, match="in-place protected RuleSpec target"):
        transport.request_run_id(json.dumps(WRAPPER), dispatch)


def test_corpus_transport_rejects_even_normalizable_legacy_path_move():
    dispatch = {
        "CITATION": "us-az/manual/des/faa5/na-categorical-eligibility/block-4",
        "REPLACE_RULESPEC_PATH": (
            "us-az/policies/des/faa5/na-categorical-eligibility/block-4.yaml"
        ),
        "REPLACE_LEGACY_RULESPEC_PATH": (
            "us-az/policies/des/faa5/na-categorical-eligibility/Block 4.yaml"
        ),
    }
    with pytest.raises(ValueError, match="in-place protected RuleSpec target"):
        transport.request_run_id(json.dumps(WRAPPER), dispatch)


@pytest.mark.parametrize(
    "legacy_source",
    [
        "us-az/../../other.yaml",
        "us-az/policies/des/faa5//expanded.yaml",
        "us-az/policies/des/faa5\\expanded.yaml",
        "us-az/.axiom/expanded.yaml",
        "us-az/policies/des/faa5/expanded.test.yaml",
    ],
)
def test_legacy_replacement_transport_rejects_unsafe_in_place_path(legacy_source):
    dispatch = {
        "CITATION": "us-az/manual/des/faa5/na-categorical-eligibility/block-4",
        "REPLACE_RULESPEC_PATH": legacy_source,
        "REPLACE_LEGACY_RULESPEC_PATH": legacy_source,
    }
    with pytest.raises(ValueError):
        transport.request_run_id(json.dumps(WRAPPER), dispatch)


def test_legacy_replacement_transport_rejects_other_jurisdiction():
    legacy_source = "us/policies/des/faa5/na-categorical-eligibility/expanded.yaml"
    dispatch = {
        "CITATION": "us-az/manual/des/faa5/na-categorical-eligibility/block-4",
        "REPLACE_RULESPEC_PATH": legacy_source,
        "REPLACE_LEGACY_RULESPEC_PATH": legacy_source,
    }
    with pytest.raises(ValueError, match="legacy source must be"):
        transport.request_run_id(json.dumps(WRAPPER), dispatch)


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


def _resign_fixture(signed):
    release = signed["release"]
    release["content_sha256"] = hashlib.sha256(
        json.dumps(
            release["content"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    signed["receipt"]["content_sha256"] = release["content_sha256"]
    signed["receipt"]["source_commit"] = release["content"]["git"]["commit"]
    signed["pin"] = (release["release"], release["content_sha256"])
    key = Ed25519PrivateKey.generate()
    signed["public_key"] = b64encode(
        key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    ).decode()
    release["signature"]["value"] = b64encode(
        key.sign(canonical_release_object_bytes(release))
    ).decode()


def _historical_lock_commit(signed, lock_sha):
    repo = signed["corpus"]
    path = signed["receipt"]["path"]
    lock = repo / transport.lock_path_for(path)
    original = lock.read_bytes()
    lock.write_bytes(
        _canonical_lock_bytes(
            ("nz", "statute", "2026-07-10-nz-rulespec"),
            {path: LockEntry(lock_sha, len(signed["payload"]))},
        )
    )
    transport.git(repo, "add", ".")
    transport.git(
        repo, "-c", "commit.gpgsign=false", "commit", "-qm", "historical-lock"
    )
    historical = transport.git(repo, "rev-parse", "HEAD")
    lock.write_bytes(original)
    transport.git(repo, "add", ".")
    transport.git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "current-lock")
    signed["release"]["content"]["git"]["commit"] = historical
    _resign_fixture(signed)


def test_payload_must_match_signed_member_even_when_current_lock_agrees(signed):
    different = hashlib.sha256(b"y" * len(signed["payload"])).hexdigest()
    _historical_lock_commit(signed, different)
    member = next(
        item
        for item in signed["release"]["content"]["artifacts"]
        if item["artifact_class"] == "provisions"
    )
    member["sha256"] = different
    member["r2_key"] = f"objects/sha256/{different[:2]}/{different}"
    signed["release"]["content"]["validation"]["r2_readback"]["verified_keys"] = [
        item["r2_key"] for item in signed["release"]["content"]["artifacts"]
    ]
    signed["receipt"]["sha256"] = different
    _resign_fixture(signed)
    with pytest.raises(ValueError, match="payload differs from signed member"):
        transport.verify_member(**signed)


def test_release_commit_lock_must_match_signed_member(signed):
    _historical_lock_commit(signed, "f" * 64)
    with pytest.raises(ValueError, match="immutable lock disagrees"):
        transport.verify_member(**signed)


def test_reingested_checkout_lock_refuses_historical_member(signed):
    path = transport.verify_member(**signed)
    repo = signed["corpus"]
    lock = transport.lock_path_for(path)
    (repo / lock).write_bytes(
        _canonical_lock_bytes(
            ("nz", "statute", "2026-07-10-nz-rulespec"),
            {path: LockEntry("f" * 64, len(signed["payload"]))},
        )
    )
    transport.git(repo, "add", str(lock))
    transport.git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "reingest")

    with pytest.raises(ValueError, match="current corpus checkout lock disagrees"):
        transport.verify_member(**signed)
    with pytest.raises(ValueError, match="current corpus checkout lock disagrees"):
        transport.place_member(repo, path, signed["payload"])
    assert not (repo / path).exists()


@pytest.mark.parametrize("change", ["missing", "symlink"])
def test_current_checkout_lock_requires_regular_file(signed, change, tmp_path):
    path = transport.verify_member(**signed)
    repo = signed["corpus"]
    lock = repo / transport.lock_path_for(path)
    original = lock.read_bytes()
    lock.unlink()
    if change == "symlink":
        target = tmp_path / "outside-lock.json"
        target.write_bytes(original)
        lock.symlink_to(target)

    with pytest.raises(ValueError, match="current corpus checkout lock does not pin"):
        transport.verify_member(**signed)
    with pytest.raises(ValueError, match="current corpus checkout lock does not pin"):
        transport.place_member(repo, path, signed["payload"])
    assert not (repo / path).exists()


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
    bad_head = copy.deepcopy(good)
    bad_head["workflow_run"]["head_sha"] = "b" * 40
    with pytest.raises(ValueError, match="bound"):
        transport.select_artifact({"total_count": 1, "artifacts": [bad_head]}, run)


def test_run_head_must_descend_from_reviewed_exporter(signed, monkeypatch):
    repo = signed["corpus"]
    for path in transport.REVIEWED_BLOBS:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("reviewed fixture " + path)
    transport.git(repo, "add", ".")
    transport.git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "same-blobs")
    prior_head = transport.git(repo, "rev-parse", "HEAD")
    transport.git(
        repo,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--allow-empty",
        "-qm",
        "reviewed-exporter-marker",
    )
    reviewed_head = transport.git(repo, "rev-parse", "HEAD")
    transport.git(
        repo,
        "remote",
        "add",
        "origin",
        f"https://github.com/{transport.REPOSITORY}.git",
    )
    transport.git(repo, "update-ref", "refs/remotes/origin/main", reviewed_head)
    monkeypatch.setattr(transport, "REVIEWED_EXPORTER", reviewed_head)
    monkeypatch.setattr(
        transport,
        "REVIEWED_BLOBS",
        {
            path: transport.git(repo, "rev-parse", f"{prior_head}:{path}")
            for path in transport.REVIEWED_BLOBS
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
        "head_sha": prior_head,
        "run_attempt": 1,
    }
    with pytest.raises(subprocess.CalledProcessError):
        transport.validate_run(run, "123", repo)


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


def test_reviewed_exporter_runtime_pairs_are_atomic(signed, monkeypatch):
    repo = signed["corpus"]
    for path in transport.REVIEWED_BLOBS:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("reviewed fixture " + path)
    transport.git(repo, "add", ".")
    transport.git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "reviewed")
    reviewed_head = transport.git(repo, "rev-parse", "HEAD")
    original_lock = (repo / "uv.lock").read_text()
    baseline_blobs = {
        path: transport.git(repo, "rev-parse", f"HEAD:{path}")
        for path in transport.REVIEWED_BLOBS
    }
    transport.git(
        repo,
        "remote",
        "add",
        "origin",
        f"https://github.com/{transport.REPOSITORY}.git",
    )
    monkeypatch.setattr(transport, "REVIEWED_EXPORTER", reviewed_head)
    monkeypatch.setattr(transport, "REVIEWED_BLOBS", baseline_blobs)
    run = {
        "id": 123,
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "path": transport.WORKFLOW,
        "head_branch": "main",
        "repository": {"full_name": transport.REPOSITORY},
        "head_repository": {"full_name": transport.REPOSITORY},
        "head_sha": reviewed_head,
        "run_attempt": 1,
    }
    transport.git(repo, "update-ref", "refs/remotes/origin/main", reviewed_head)
    transport.validate_run(run, "123", repo)

    (repo / "pyproject.toml").write_text("second reviewed pyproject")
    (repo / "uv.lock").write_text("second reviewed lock")
    transport.git(repo, "add", "pyproject.toml", "uv.lock")
    transport.git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "second pair")
    second_head = transport.git(repo, "rev-parse", "HEAD")
    second_pair = (
        transport.git(repo, "rev-parse", "HEAD:pyproject.toml"),
        transport.git(repo, "rev-parse", "HEAD:uv.lock"),
    )
    monkeypatch.setattr(transport, "REVIEWED_PYMUPDF_RUNTIME_BLOB_PAIR", second_pair)
    transport.git(repo, "update-ref", "refs/remotes/origin/main", second_head)
    transport.validate_run({**run, "head_sha": second_head}, "123", repo)

    (repo / "uv.lock").write_text(original_lock)
    transport.git(repo, "add", "uv.lock")
    transport.git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "mixed pair")
    mixed_head = transport.git(repo, "rev-parse", "HEAD")
    transport.git(repo, "update-ref", "refs/remotes/origin/main", mixed_head)
    with pytest.raises(ValueError, match="reviewed blobs"):
        transport.validate_run({**run, "head_sha": mixed_head}, "123", repo)


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
    with pytest.raises(ValueError, match="one fresh primary"):
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
        "--corpus-checkout",
        str(repo),
        "--run-metadata",
        str(run_path),
        "--artifact-metadata",
        str(metadata_path),
    ]
    # The authenticated checkout argument does not retain the unsigned legacy alias.
    monkeypatch.setattr(
        "sys.argv", ["transport", "validate-export", "--corpus-root", str(repo)]
    )
    with pytest.raises(SystemExit) as rejected:
        transport.main()
    assert rejected.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
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
