"""Adversarial tests for the pure cleanup receipt and immutable-base proof."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from base64 import b64encode
from copy import deepcopy
from pathlib import Path

import pytest

from axiom_encode.legacy_cleanup import (
    LEGACY_CLEANUP_BASE_PROOF_SCHEMA,
    LEGACY_CLEANUP_PROVENANCE_ASSERTIONS,
    LEGACY_CLEANUP_RECEIPT_DIR,
    LEGACY_CLEANUP_RECEIPT_SCHEMA,
    LEGACY_CLEANUP_SIGNATURE_ALGORITHM,
    LEGACY_CLEANUP_SIGNATURE_DOMAIN,
    LEGACY_CLEANUP_TOOL,
    LEGACY_CLEANUP_VALIDATION_CHECKS,
    LEGACY_CLEANUP_VALIDATION_SCHEMA,
    LegacyCleanupReceiptError,
    canonical_primary_paths,
    canonical_receipt_bytes,
    cleanup_signature_payload,
    companion_path,
    decode_strict_json_object,
    deleted_paths,
    is_legacy_cleanup_receipt_path,
    parse_receipt_bytes,
    receipt_identity_sha256,
    receipt_path,
)
from axiom_encode.legacy_cleanup_git import (
    LegacyCleanupGitError,
    plan_legacy_cleanup_base,
    plan_payload_issues,
)

PRIMARY = Path("be/statutes/legacy.yaml")
COMPANION = Path("be/statutes/legacy.test.yaml")
WAIVER_BYTES = b"version: 1\nwaivers: []\n"
WAIVER_SHA256 = hashlib.sha256(WAIVER_BYTES).hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "rulespec-be"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "legacy-cleanup@example.com")
    _git(repo, "config", "user.name", "Legacy Cleanup Test")
    _git(
        repo,
        "remote",
        "add",
        "origin",
        "https://github.com/TheAxiomFoundation/rulespec-be.git",
    )
    return repo


def _write_contract(repo: Path) -> None:
    toolchain = repo / ".axiom/toolchain.toml"
    toolchain.parent.mkdir(parents=True, exist_ok=True)
    toolchain.write_text(
        "[toolchain]\n"
        'axiom_corpus_release = "legacy-cleanup-test"\n'
        f'axiom_corpus_release_content_sha256 = "{"c" * 64}"\n'
        f'validation_waiver_set_sha256 = "{WAIVER_SHA256}"\n',
        encoding="utf-8",
    )
    (repo / "known-validation-gaps.yaml").write_bytes(WAIVER_BYTES)


def _write_group(repo: Path, primary: Path = PRIMARY) -> None:
    for path, body in (
        (primary, "module:\n  name: legacy\n"),
        (companion_path(primary), "tests:\n  - name: legacy\n"),
    ):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")


def _base_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = _init_repo(tmp_path)
    _write_contract(repo)
    _write_group(repo)
    return repo, _commit(repo, "legacy base")


def _plan(repo: Path, base: str, *, clean: bool = True):
    return plan_legacy_cleanup_base(
        repo,
        base_ref=base,
        primary_paths=[PRIMARY],
        require_clean_checkout=clean,
    )


def _payload(plan) -> dict[str, object]:
    validation_checks = [
        {
            "name": name,
            "command": ["internal", name],
            "target_count": 1,
            "target_list_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "exit_code": 0,
            "output_sha256": hashlib.sha256(f"passed:{name}".encode()).hexdigest(),
        }
        for name in LEGACY_CLEANUP_VALIDATION_CHECKS
    ]
    payload: dict[str, object] = {
        "schema_version": LEGACY_CLEANUP_RECEIPT_SCHEMA,
        "tool": LEGACY_CLEANUP_TOOL,
        "provenance_class": "unmanifested-legacy-contraction",
        "generated_at": "2026-08-30T20:00:00Z",
        "reason": "Remove an exact unmanifested legacy aggregation",
        "repository": {
            "repository": plan.repository,
            "object_format": plan.object_format,
            "base_commit": plan.base_commit,
            "base_tree": plan.base_tree,
            "projected_post_deletion_tree": plan.projected_post_deletion_tree,
        },
        "toolchain": {
            "axiom_encode": {
                "repository": "github.com/TheAxiomFoundation/axiom-encode",
                "object_format": "sha1",
                "commit": "a" * 40,
                "version": "0.2.test",
            },
            "axiom_rules_engine": {
                "repository": "github.com/TheAxiomFoundation/axiom-rules-engine",
                "object_format": "sha1",
                "commit": "b" * 40,
            },
            "corpus_release": {
                "name": plan.toolchain_values["axiom_corpus_release"],
                "content_sha256": plan.toolchain_values[
                    "axiom_corpus_release_content_sha256"
                ],
                "selector_sha256": "d" * 64,
            },
            "validation_waiver_set_sha256": plan.toolchain_values[
                "validation_waiver_set_sha256"
            ],
            "base_files": deepcopy(plan.base_files),
        },
        "base_proof": {
            "schema": LEGACY_CLEANUP_BASE_PROOF_SCHEMA,
            "ownership_inventory_sha256": plan.ownership_inventory_sha256,
            "provenance_record_count": len(plan.provenance_records),
            "surviving_reference_inventory_sha256": (
                plan.surviving_reference_inventory_sha256
            ),
            "surviving_blob_count": plan.surviving_blob_count,
        },
        "validation_execution": {
            "schema": LEGACY_CLEANUP_VALIDATION_SCHEMA,
            "status": "passed",
            "engine_execution": True,
            "projected_post_deletion_tree": plan.projected_post_deletion_tree,
            "checks": validation_checks,
        },
        "provenance_assertions": dict(LEGACY_CLEANUP_PROVENANCE_ASSERTIONS),
        "groups": deepcopy(list(plan.groups)),
    }
    payload["receipt_identity_sha256"] = receipt_identity_sha256(payload)
    payload["signature"] = {
        "algorithm": LEGACY_CLEANUP_SIGNATURE_ALGORITHM,
        "key_id": f"sha256:{'e' * 64}",
        "domain": LEGACY_CLEANUP_SIGNATURE_DOMAIN,
        "value": b64encode(b"s" * 64).decode("ascii"),
    }
    return payload


def test_base_plan_binds_exact_preimages_and_projected_tree(tmp_path):
    repo, base = _base_repo(tmp_path)

    plan = _plan(repo, base)

    assert plan.base_commit == base
    assert plan.groups[0]["primary"] == {
        **plan.groups[0]["primary"],
        "path": PRIMARY.as_posix(),
        "base_mode": "100644",
        "result": "absent",
    }
    assert plan.groups[0]["companion"]["path"] == COMPANION.as_posix()
    assert plan.projected_post_deletion_tree != plan.base_tree
    assert (repo / PRIMARY).is_file()
    assert (repo / COMPANION).is_file()

    (repo / PRIMARY).unlink()
    (repo / COMPANION).unlink()
    _git(repo, "add", "--", PRIMARY.as_posix(), COMPANION.as_posix())
    assert _git(repo, "write-tree") == plan.projected_post_deletion_tree


def test_receipt_identity_is_semantic_not_containing_head_or_timestamp(tmp_path):
    repo, base = _base_repo(tmp_path)
    payload = _payload(_plan(repo, base))
    identity = payload["receipt_identity_sha256"]

    changed_envelope = deepcopy(payload)
    changed_envelope["generated_at"] = "2026-08-30T20:00:01Z"
    changed_envelope["signature"]["value"] = b64encode(b"t" * 64).decode("ascii")
    assert receipt_identity_sha256(changed_envelope) == identity
    assert "head_commit" not in payload["repository"]
    assert "final_tree" not in payload["repository"]

    changed_base = deepcopy(payload)
    changed_base["repository"]["base_commit"] = "f" * len(base)
    assert receipt_identity_sha256(changed_base) != identity


def test_parser_accepts_exact_canonical_receipt_and_identity_path(tmp_path):
    repo, base = _base_repo(tmp_path)
    payload = _payload(_plan(repo, base))
    path = receipt_path(payload["receipt_identity_sha256"])

    parsed = parse_receipt_bytes(
        canonical_receipt_bytes(payload),
        expected_path=path,
    )

    assert parsed == payload
    assert is_legacy_cleanup_receipt_path(path)
    assert deleted_paths(parsed) == (PRIMARY.as_posix(), COMPANION.as_posix())
    assert cleanup_signature_payload(parsed).startswith(
        LEGACY_CLEANUP_SIGNATURE_DOMAIN.encode() + b"\0"
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "axiom-encode/applied-rulespec/v5", "schema_version"),
        ("tool", "axiom-encode retire", "tool is invalid"),
        ("provenance_class", "generated", "provenance class"),
        ("generated_at", "2026-08-30T20:00:00+00:00", "generated_at"),
    ],
)
def test_parser_rejects_cross_contract_replay(tmp_path, field, value, match):
    repo, base = _base_repo(tmp_path)
    payload = _payload(_plan(repo, base))
    payload[field] = value
    payload["receipt_identity_sha256"] = receipt_identity_sha256(payload)

    with pytest.raises(LegacyCleanupReceiptError, match=match):
        parse_receipt_bytes(canonical_receipt_bytes(payload))


@pytest.mark.parametrize(
    ("signature_field", "value", "match"),
    [
        ("domain", "axiom-encode/applied-rulespec/v5", "domain"),
        ("algorithm", "ed25519", "algorithm"),
        ("key_id", "apply-root", "key ID"),
        ("value", "not-base64", "value"),
    ],
)
def test_parser_rejects_signature_scope_and_shape_replay(
    tmp_path,
    signature_field,
    value,
    match,
):
    repo, base = _base_repo(tmp_path)
    payload = _payload(_plan(repo, base))
    payload["signature"][signature_field] = value

    with pytest.raises(LegacyCleanupReceiptError, match=match):
        parse_receipt_bytes(canonical_receipt_bytes(payload))


def test_parser_rejects_duplicate_keys_nonfinite_and_noncanonical_bytes(tmp_path):
    repo, base = _base_repo(tmp_path)
    payload = _payload(_plan(repo, base))
    canonical = canonical_receipt_bytes(payload)

    duplicate = canonical.replace(
        b'  "generated_at": "2026-08-30T20:00:00Z",\n',
        b'  "generated_at": "2026-08-30T20:00:00Z",\n'
        b'  "generated_at": "2026-08-30T20:00:00Z",\n',
        1,
    )
    with pytest.raises(LegacyCleanupReceiptError, match="duplicate JSON key"):
        parse_receipt_bytes(duplicate)
    with pytest.raises(LegacyCleanupReceiptError, match="invalid JSON constant"):
        decode_strict_json_object(b'{"value": NaN}', label="test record")
    with pytest.raises(LegacyCleanupReceiptError, match="canonical persisted JSON"):
        parse_receipt_bytes(json.dumps(payload).encode())


@pytest.mark.parametrize(
    "bad_path",
    [
        "be/statutes/legacy.test.yaml",
        "../be/statutes/legacy.yaml",
        "be/other/legacy.yaml",
        "/be/statutes/legacy.yaml",
        "be\\statutes\\legacy.yaml",
        "be/statutes/legacy.yml",
        "BE/statutes/legacy.yaml",
        "be/statutes/control\x00.yaml",
    ],
)
def test_primary_input_rejects_path_and_companion_attacks(bad_path):
    with pytest.raises(LegacyCleanupReceiptError, match="canonical primary"):
        canonical_primary_paths([bad_path])


def test_primary_input_is_bounded_unique_and_mechanically_grouped():
    paths = [f"be/statutes/legacy_{index:02d}.yaml" for index in range(64)]
    assert len(canonical_primary_paths(paths)) == 64
    assert companion_path(Path(paths[0])).name == "legacy_00.test.yaml"
    with pytest.raises(LegacyCleanupReceiptError, match="unique"):
        canonical_primary_paths([paths[0], paths[0]])
    with pytest.raises(LegacyCleanupReceiptError, match="1..64"):
        canonical_primary_paths([])
    with pytest.raises(LegacyCleanupReceiptError, match="1..64"):
        canonical_primary_paths([*paths, "be/statutes/overflow.yaml"])


@pytest.mark.parametrize("kind", ["executable", "symlink", "gitlink"])
def test_base_plan_rejects_non_regular_group_modes(tmp_path, kind):
    repo = _init_repo(tmp_path)
    _write_contract(repo)
    _write_group(repo)
    if kind == "executable":
        os.chmod(repo / PRIMARY, 0o755)
    elif kind == "symlink":
        (repo / PRIMARY).unlink()
        (repo / PRIMARY).symlink_to(COMPANION.name)
    else:
        first = _commit(repo, "temporary object")
        (repo / PRIMARY).unlink()
        _git(repo, "add", "-A")
        _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{first},{PRIMARY}")
        _git(repo, "commit", "-m", f"{kind} base")
        base = _git(repo, "rev-parse", "HEAD")
    if kind != "gitlink":
        base = _commit(repo, f"{kind} base")

    with pytest.raises(LegacyCleanupGitError, match="regular 100644 blob"):
        _plan(repo, base)


@pytest.mark.parametrize(
    "owner_kind",
    [
        "canonical",
        "supplemental",
        "retirement",
        "migration",
        "replacement",
        "manual",
        "deterministic",
        "prior-cleanup",
    ],
)
def test_base_plan_rejects_every_prior_ownership_class(tmp_path, owner_kind):
    repo = _init_repo(tmp_path)
    _write_contract(repo)
    _write_group(repo)
    if owner_kind == "canonical":
        path = repo / ".axiom/encoding-manifests/be/statutes/legacy.json"
        payload = {}
    elif owner_kind == "migration":
        path = repo / f".axiom/path-migrations/{'1' * 64}.json"
        payload = {
            "moves": [{"from": PRIMARY.as_posix(), "to": "be/statutes/new.yaml"}]
        }
    elif owner_kind == "replacement":
        path = repo / f".axiom/legacy-replacements/{'2' * 64}.json"
        payload = {"legacy": {"files": [{"path": PRIMARY.as_posix()}]}}
    elif owner_kind == "prior-cleanup":
        path = repo / LEGACY_CLEANUP_RECEIPT_DIR / f"{'3' * 64}.json"
        payload = {"groups": [{"companion": {"path": COMPANION.as_posix()}}]}
    else:
        path = repo / f".axiom/encoding-manifests/supplemental-{owner_kind}.json"
        if owner_kind == "retirement":
            payload = {
                "retired_manifest": {"applied_files": [{"path": PRIMARY.as_posix()}]}
            }
        elif owner_kind == "manual":
            payload = {
                "backend": "manual",
                "applied_files": [{"path": PRIMARY.as_posix()}],
            }
        elif owner_kind == "deterministic":
            payload = {
                "backend": "legacy",
                "deterministic_execution": True,
                "applied_files": [{"path": PRIMARY.as_posix()}],
            }
        else:
            payload = {"applied_files": [{"path": PRIMARY.as_posix()}]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    base = _commit(repo, f"{owner_kind} owner")

    with pytest.raises(LegacyCleanupGitError, match="already (owns|covers)"):
        _plan(repo, base)


@pytest.mark.parametrize(
    "malformation", ["duplicate-key", "invalid-utf8", "symlink", "non-json"]
)
def test_base_plan_fails_closed_on_malformed_provenance(tmp_path, malformation):
    repo = _init_repo(tmp_path)
    _write_contract(repo)
    _write_group(repo)
    path = repo / ".axiom/encoding-manifests/unrelated.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if malformation == "duplicate-key":
        path.write_bytes(b'{"applied_files": [], "applied_files": []}')
    elif malformation == "invalid-utf8":
        path.write_bytes(b'{"unrelated": "\xff"}')
    elif malformation == "symlink":
        target = repo / "unrelated.json"
        target.write_text("{}")
        path.symlink_to(os.path.relpath(target, path.parent))
    else:
        path = path.with_suffix(".txt")
        path.write_text("unrelated")
    base = _commit(repo, "malformed provenance")

    with pytest.raises(LegacyCleanupGitError, match="provenance"):
        _plan(repo, base)


def test_removed_or_corrupted_live_owner_does_not_create_eligibility(tmp_path):
    repo = _init_repo(tmp_path)
    _write_contract(repo)
    _write_group(repo)
    owner = repo / ".axiom/encoding-manifests/supplemental.json"
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text(json.dumps({"applied_files": [{"path": PRIMARY.as_posix()}]}))
    base = _commit(repo, "owned base")
    owner.unlink()

    with pytest.raises(LegacyCleanupGitError, match="already covers"):
        _plan(repo, base, clean=False)


@pytest.mark.parametrize(
    ("path", "content"),
    [
        ("be/statutes/survivor.yaml", "imports:\n  - be:statutes/legacy\n"),
        ("known-validation-gaps.yaml", f"path: {PRIMARY.as_posix()}\n"),
        ("oracle-coverage.yaml", f"module: {PRIMARY.as_posix()}\n"),
        ("docs/legacy.md", f"See {PRIMARY.as_posix()} before cleanup.\n"),
    ],
)
def test_base_plan_rejects_surviving_import_and_metadata_references(
    tmp_path,
    path,
    content,
):
    repo = _init_repo(tmp_path)
    _write_contract(repo)
    _write_group(repo)
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    if path == "known-validation-gaps.yaml":
        toolchain = repo / ".axiom/toolchain.toml"
        waiver_digest = hashlib.sha256(target.read_bytes()).hexdigest()
        toolchain.write_text(
            toolchain.read_text().replace(WAIVER_SHA256, waiver_digest),
            encoding="utf-8",
        )
    base = _commit(repo, "stale reference")

    with pytest.raises(LegacyCleanupGitError, match="references a cleanup target"):
        _plan(repo, base)


@pytest.mark.parametrize("dirty_kind", ["staged", "unstaged", "untracked", "ignored"])
def test_base_plan_requires_exact_clean_base_checkout(tmp_path, dirty_kind):
    repo, base = _base_repo(tmp_path)
    if dirty_kind == "staged":
        (repo / PRIMARY).write_text("staged\n")
        _git(repo, "add", PRIMARY.as_posix())
    elif dirty_kind == "unstaged":
        (repo / PRIMARY).write_text("unstaged\n")
    elif dirty_kind == "untracked":
        (repo / "untracked.txt").write_text("untracked\n")
    else:
        (repo / ".gitignore").write_text("ignored.txt\n")
        _commit(repo, "ignore contract")
        base = _git(repo, "rev-parse", "HEAD")
        (repo / "ignored.txt").write_text("ignored\n")

    with pytest.raises(LegacyCleanupGitError, match="requires no staged"):
        _plan(repo, base)


def test_base_plan_rejects_stale_head_even_with_clean_worktree(tmp_path):
    repo, base = _base_repo(tmp_path)
    _git(repo, "commit", "--allow-empty", "-m", "advanced head")

    with pytest.raises(LegacyCleanupGitError, match="HEAD to equal"):
        _plan(repo, base)


def test_plan_payload_comparison_is_base_to_projected_tree_not_head(tmp_path):
    repo, base = _base_repo(tmp_path)
    plan = _plan(repo, base)
    payload = _payload(plan)

    assert plan_payload_issues(payload, plan) == []
    forged = deepcopy(payload)
    forged["repository"]["projected_post_deletion_tree"] = "f" * len(base)
    assert any("projected-tree" in issue for issue in plan_payload_issues(forged, plan))
    assert "head_commit" not in payload["repository"]
    assert receipt_path(payload["receipt_identity_sha256"]).parent == (
        LEGACY_CLEANUP_RECEIPT_DIR
    )
