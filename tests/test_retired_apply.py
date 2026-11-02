"""Frozen retired-root authority binds exact Git blobs and live history."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.retired_apply import (
    RETIRED_APPLY_INVENTORY_SCHEMA,
    RetiredApplyContext,
    retired_apply_admission_issue,
    retired_apply_payload_admission_issue,
    retrospective_head_issue,
    validate_retired_apply_inventory,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        [shutil.which("git") or "git", "-C", str(repo), *arguments],
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": "/nonexistent",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Synthetic fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.com",
            "GIT_COMMITTER_NAME": "Synthetic fixture",
            "GIT_COMMITTER_EMAIL": "fixture@example.com",
            "GIT_TERMINAL_PROMPT": "0",
        },
    ).stdout.strip()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _signed(payload: dict[str, object]) -> bytes:
    # Every key used here is newly generated synthetic test material.
    key = Ed25519PrivateKey.generate()
    message = json.dumps(payload, sort_keys=True).encode()
    return (
        json.dumps(
            {**payload, "signature": base64.b64encode(key.sign(message)).decode()},
            sort_keys=True,
        ).encode()
        + b"\n"
    )


@dataclass
class _Frozen:
    repo: Path
    manifest: Path
    generated: Path
    manifest_bytes: bytes
    generated_bytes: bytes
    cutoff: str
    inventory: dict[str, object]

    @property
    def context(self) -> RetiredApplyContext:
        return RetiredApplyContext(self.repo, self.manifest, self.manifest_bytes)

    def commit(self, message: str = "Synthetic fixture change") -> str:
        _git(self.repo, "add", "--all")
        _git(self.repo, "commit", "--allow-empty", "-m", message)
        return _git(self.repo, "rev-parse", "HEAD")


@pytest.fixture
def frozen(tmp_path: Path) -> _Frozen:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    _git(
        repo,
        "remote",
        "add",
        "origin",
        "https://github.com/TheAxiomFoundation/rulespec-us.git",
    )
    generated = repo / "us/example.yaml"
    generated.parent.mkdir()
    generated_bytes = b"synthetic generated fixture bytes\n"
    generated.write_bytes(generated_bytes)
    manifest = repo / ".axiom/encoding-manifests/example.json"
    manifest.parent.mkdir(parents=True)
    manifest_bytes = _signed(
        {
            "applied_files": [
                {"path": "us/example.yaml", "sha256": _sha(generated_bytes)}
            ]
        }
    )
    manifest.write_bytes(manifest_bytes)
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", "Synthetic frozen main")
    cutoff = _git(repo, "rev-parse", "HEAD")
    inventory = {
        "schema": RETIRED_APPLY_INVENTORY_SCHEMA,
        "repositories": [
            {
                "repository": "TheAxiomFoundation/rulespec-us",
                "cutoff_commit": cutoff,
                "artifacts": [
                    {
                        "path": manifest.relative_to(repo).as_posix(),
                        "sha256": _sha(manifest_bytes),
                        "covered_files": [
                            {"path": "us/example.yaml", "sha256": _sha(generated_bytes)}
                        ],
                    }
                ],
            }
        ],
    }
    return _Frozen(
        repo, manifest, generated, manifest_bytes, generated_bytes, cutoff, inventory
    )


def test_inventory_validation_copies_exact_schema(frozen):
    result = validate_retired_apply_inventory(frozen.inventory)
    assert result == frozen.inventory
    result["repositories"][0]["artifacts"][0]["covered_files"].clear()
    assert frozen.inventory["repositories"][0]["artifacts"][0]["covered_files"]


@pytest.mark.parametrize("location", ["root", "repository", "artifact", "covered"])
def test_inventory_rejects_unknown_fields(frozen, location):
    value = copy.deepcopy(frozen.inventory)
    entries = {
        "root": value,
        "repository": value["repositories"][0],
        "artifact": value["repositories"][0]["artifacts"][0],
        "covered": value["repositories"][0]["artifacts"][0]["covered_files"][0],
    }
    entries[location]["caller_authority"] = True
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_retired_apply_inventory(value)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/absolute.json",
        "../outside",
        "a/../b",
        "./a",
        "a//b",
        "a/",
        "a\\b",
        "-option",
        ":magic",
        "a*",
        "a?",
        "a[",
        "a\0b",
        "a\nb",
        "a\tb",
        "a\x7fb",
        "a\x80b",
        "a\ud800b",
    ],
)
def test_inventory_rejects_noncanonical_paths(frozen, path):
    value = copy.deepcopy(frozen.inventory)
    value["repositories"][0]["artifacts"][0]["path"] = path
    with pytest.raises(ValueError, match="path must be canonical"):
        validate_retired_apply_inventory(value)


@pytest.mark.parametrize(
    "path", ["us/statutes/26:1.yaml", "us/statutes/42/1437c–1.yaml", "us/He-W 704.yaml"]
)
def test_inventory_preserves_exact_historical_utf8_paths(frozen, path):
    value = copy.deepcopy(frozen.inventory)
    value["repositories"][0]["artifacts"][0]["covered_files"][0]["path"] = path
    assert validate_retired_apply_inventory(value) == value


@pytest.mark.parametrize(
    "identity",
    [
        "github.com/TheAxiomFoundation/rulespec-us",
        "OtherOwner/rulespec-us",
        "TheAxiomFoundation/Rulespec-us",
        "TheAxiomFoundation/rulespec-us.git",
        "TheAxiomFoundation/rulespec-us/",
        "TheAxiomFoundation/rulespec-us--ca",
    ],
)
def test_inventory_rejects_noncanonical_repositories(frozen, identity):
    value = copy.deepcopy(frozen.inventory)
    value["repositories"][0]["repository"] = identity
    with pytest.raises(ValueError, match="repository must be canonical"):
        validate_retired_apply_inventory(value)


@pytest.mark.parametrize("location", ["repository", "artifact", "covered"])
def test_inventory_rejects_duplicate_bindings(frozen, location):
    value = copy.deepcopy(frozen.inventory)
    if location == "repository":
        target = value["repositories"]
    elif location == "artifact":
        target = value["repositories"][0]["artifacts"]
    else:
        target = value["repositories"][0]["artifacts"][0]["covered_files"]
    target.append(copy.deepcopy(target[0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_retired_apply_inventory(value)


def test_inventory_prior_source_is_explicit_and_preserved(frozen):
    value = copy.deepcopy(frozen.inventory)
    covered = value["repositories"][0]["artifacts"][0]["covered_files"][0]
    covered["source_commit"] = "a" * 40
    assert validate_retired_apply_inventory(value) == value
    duplicate = {**covered, "source_commit": "b" * 40}
    value["repositories"][0]["artifacts"][0]["covered_files"].append(duplicate)
    assert validate_retired_apply_inventory(value) == value
    covered["source_commit"] = "main"
    with pytest.raises(ValueError, match="source must be a full commit"):
        validate_retired_apply_inventory(value)


def test_frozen_live_artifact_is_admitted(frozen):
    assert retired_apply_admission_issue(frozen.inventory, frozen.context) is None
    assert (
        retired_apply_payload_admission_issue(
            frozen.inventory, frozen.context, json.loads(frozen.manifest_bytes)
        )
        is None
    )
    relative = replace(
        frozen.context, artifact_path=frozen.manifest.relative_to(frozen.repo)
    )
    assert retired_apply_admission_issue(frozen.inventory, relative) is None


def test_unchanged_frozen_artifact_survives_unrelated_postcutoff_commit(frozen):
    (frozen.repo / "README.txt").write_text("Unrelated fixture documentation\n")
    frozen.commit()
    assert retired_apply_admission_issue(frozen.inventory, frozen.context) is None


@pytest.mark.parametrize("mutation", ["manifest", "generated", "new", "relocated"])
def test_changed_or_relocated_old_artifact_is_rejected(frozen, mutation):
    context = frozen.context
    if mutation == "manifest":
        frozen.manifest.write_bytes(frozen.manifest_bytes + b" ")
        context = replace(context, artifact_bytes=frozen.manifest.read_bytes())
    elif mutation == "generated":
        frozen.generated.write_bytes(frozen.generated_bytes + b"changed")
    else:
        target = frozen.manifest.with_name("introduced.json")
        target.write_bytes(frozen.manifest_bytes)
        context = replace(context, artifact_path=target)
        if mutation == "relocated":
            frozen.manifest.unlink()
    assert retired_apply_admission_issue(frozen.inventory, context) is not None


@pytest.mark.parametrize("changed_path", ["manifest", "generated"])
@pytest.mark.parametrize("change", ["replace", "delete"])
def test_restoration_after_supersession_is_rejected(frozen, changed_path, change):
    target = frozen.manifest if changed_path == "manifest" else frozen.generated
    original = target.read_bytes()
    if change == "replace":
        target.write_bytes(_signed({"new_active_fixture": True}))
    else:
        target.unlink()
    frozen.commit("Synthetic new-root replacement or deletion")
    target.write_bytes(original)
    frozen.commit("Synthetic attempted historical replay")
    assert "superseded" in retired_apply_admission_issue(
        frozen.inventory, frozen.context
    )


def test_superseded_merged_side_branch_is_rejected(frozen):
    _git(frozen.repo, "checkout", "-b", "synthetic-branch")
    frozen.manifest.write_bytes(_signed({"new_active_fixture": True}))
    frozen.commit()
    frozen.manifest.write_bytes(frozen.manifest_bytes)
    frozen.commit()
    _git(frozen.repo, "checkout", "main")
    _git(frozen.repo, "merge", "--no-ff", "synthetic-branch", "-m", "Synthetic merge")
    assert "superseded" in retired_apply_admission_issue(
        frozen.inventory, frozen.context
    )


@pytest.mark.parametrize(
    "origin",
    [
        "https://github.com/OtherOwner/rulespec-us.git",
        "https://github.com/TheAxiomFoundation/rulespec-uk.git",
        "https://example.com/TheAxiomFoundation/rulespec-us.git",
    ],
)
def test_repository_identity_is_owner_and_path_bound(frozen, origin):
    _git(frozen.repo, "remote", "set-url", "origin", origin)
    assert retired_apply_admission_issue(frozen.inventory, frozen.context) is not None


@pytest.mark.parametrize("head", ["main", "origin/main", "HEAD~1", "a" * 40])
def test_historical_head_must_be_exact_checked_out_commit(frozen, head):
    assert retired_apply_admission_issue(
        frozen.inventory, replace(frozen.context, head_ref=head)
    )


def test_retrospective_cutoff_and_ancestor_pass_but_later_head_fails(frozen):
    _git(frozen.repo, "commit", "--allow-empty", "-m", "Later frozen main")
    cutoff = _git(frozen.repo, "rev-parse", "HEAD")
    frozen.inventory["repositories"][0]["cutoff_commit"] = cutoff
    assert retrospective_head_issue(frozen.inventory, frozen.repo, cutoff) is None
    assert (
        retired_apply_admission_issue(
            frozen.inventory, replace(frozen.context, retrospective=True)
        )
        is None
    )
    _git(frozen.repo, "checkout", "--detach", frozen.cutoff)
    assert (
        retrospective_head_issue(frozen.inventory, frozen.repo, frozen.cutoff) is None
    )
    assert (
        retired_apply_admission_issue(
            frozen.inventory, replace(frozen.context, retrospective=True)
        )
        is None
    )
    _git(frozen.repo, "checkout", "main")
    frozen.commit()
    assert "newer" in retrospective_head_issue(frozen.inventory, frozen.repo)


@pytest.mark.parametrize("metadata", ["grafts", "replacement", "shallow"])
def test_history_rewriting_or_incomplete_metadata_fails_closed(frozen, metadata):
    if metadata == "grafts":
        path = frozen.repo / ".git/info/grafts"
        path.write_text(frozen.cutoff + "\n")
    elif metadata == "replacement":
        _git(frozen.repo, "update-ref", "refs/replace/" + frozen.cutoff, frozen.cutoff)
    else:
        (frozen.repo / ".git/shallow").write_text(frozen.cutoff + "\n")
    assert retired_apply_admission_issue(frozen.inventory, frozen.context)


def test_symlinked_live_generated_file_fails(frozen):
    target = frozen.repo / "identical-fixture-bytes"
    target.write_bytes(frozen.generated_bytes)
    frozen.generated.unlink()
    frozen.generated.symlink_to(target)
    assert "symlink" in retired_apply_admission_issue(frozen.inventory, frozen.context)


def test_payload_cannot_borrow_a_different_frozen_container(frozen):
    payload = json.loads(frozen.manifest_bytes)
    payload["new_old_key_authority"] = True
    assert "does not match" in retired_apply_payload_admission_issue(
        frozen.inventory, frozen.context, payload
    )
    assert "not exact frozen evidence" in retired_apply_payload_admission_issue(
        frozen.inventory, frozen.context, payload, nested=True
    )


def test_nested_frozen_evidence_survives_active_replacement_but_cannot_authorize_it(
    frozen,
):
    payload = json.loads(frozen.manifest_bytes)
    frozen.manifest.write_bytes(_signed({"new_active_fixture": True, "prior": payload}))
    frozen.generated.unlink()
    frozen.commit()
    context = replace(frozen.context, artifact_bytes=frozen.manifest.read_bytes())
    assert (
        retired_apply_payload_admission_issue(
            frozen.inventory, context, payload, nested=True
        )
        is None
    )
    assert (
        retired_apply_payload_admission_issue(frozen.inventory, context, payload)
        is not None
    )


def test_deleted_prefreeze_nested_bytes_bind_explicit_prior_source(frozen):
    nested = json.loads(frozen.manifest_bytes)
    frozen.manifest_bytes = _signed(
        {
            "retired_manifest": nested,
            "applied_files": [
                {
                    "path": "us/example.yaml",
                    "sha256": _sha(frozen.generated_bytes),
                    "deleted": True,
                }
            ],
        }
    )
    frozen.manifest.write_bytes(frozen.manifest_bytes)
    frozen.generated.unlink()
    cutoff = frozen.commit("Synthetic frozen retirement receipt")
    repository = frozen.inventory["repositories"][0]
    repository["cutoff_commit"] = cutoff
    artifact = repository["artifacts"][0]
    artifact["sha256"] = _sha(frozen.manifest_bytes)
    artifact["covered_files"][0]["source_commit"] = frozen.cutoff
    assert retired_apply_admission_issue(frozen.inventory, frozen.context) is None
    assert (
        retired_apply_payload_admission_issue(
            frozen.inventory, frozen.context, nested, nested=True
        )
        is None
    )
    del artifact["covered_files"][0]["source_commit"]
    assert retired_apply_payload_admission_issue(
        frozen.inventory, frozen.context, nested, nested=True
    )


def test_inventory_cannot_omit_covered_generated_bytes(frozen):
    frozen.inventory["repositories"][0]["artifacts"][0]["covered_files"].clear()
    assert "every generated-file" in retired_apply_admission_issue(
        frozen.inventory, frozen.context
    )


@pytest.mark.parametrize("field", ["artifact", "covered", "cutoff"])
def test_inventory_digest_and_cutoff_must_match_real_git_blobs(frozen, field):
    repository = frozen.inventory["repositories"][0]
    artifact = repository["artifacts"][0]
    if field == "artifact":
        artifact["sha256"] = "f" * 64
    elif field == "covered":
        artifact["covered_files"][0]["sha256"] = "f" * 64
    else:
        repository["cutoff_commit"] = "f" * 40
    assert retired_apply_admission_issue(frozen.inventory, frozen.context)


def test_historical_covered_source_cannot_be_newer_than_cutoff(frozen):
    source = frozen.commit("Synthetic commit newer than owner cutoff")
    frozen.inventory["repositories"][0]["artifacts"][0]["covered_files"][0][
        "source_commit"
    ] = source
    assert "source must be an ancestor" in retired_apply_admission_issue(
        frozen.inventory, frozen.context
    )


@pytest.mark.parametrize("filename", ["26:1.yaml", "1437c–1.yaml", "He-W 704.yaml"])
def test_exact_historical_utf8_paths_bind_real_frozen_blobs(frozen, filename):
    new_generated = frozen.generated.with_name(filename)
    frozen.generated.rename(new_generated)
    relative = new_generated.relative_to(frozen.repo).as_posix()
    frozen.manifest_bytes = _signed(
        {"applied_files": [{"path": relative, "sha256": _sha(frozen.generated_bytes)}]}
    )
    frozen.manifest.write_bytes(frozen.manifest_bytes)
    cutoff = frozen.commit("Synthetic historical path")
    repository = frozen.inventory["repositories"][0]
    repository["cutoff_commit"] = cutoff
    artifact = repository["artifacts"][0]
    artifact["sha256"] = _sha(frozen.manifest_bytes)
    artifact["covered_files"][0]["path"] = relative
    frozen.commit("Synthetic unrelated commit")
    assert retired_apply_admission_issue(frozen.inventory, frozen.context) is None


def test_git_uses_authenticated_execution_environment(frozen, monkeypatch):
    from axiom_encode import signing_broker

    monkeypatch.setattr(
        signing_broker,
        "_trusted_subprocess_base_environment",
        {"PATH": "/usr/bin:/bin"},
    )
    monkeypatch.setenv("PATH", "/unavailable/attacker-tools")
    monkeypatch.setenv("GIT_DIR", "/unavailable/attacker.git")
    monkeypatch.setenv("GIT_REPLACE_REF_BASE", "refs/attacker/")
    assert retired_apply_admission_issue(frozen.inventory, frozen.context) is None


def test_retrospective_mode_requires_authenticated_inventory(frozen):
    assert retrospective_head_issue(None, frozen.repo)


def test_prior_source_cannot_bind_live_generated_bytes(frozen):
    prior = frozen.cutoff
    frozen.cutoff = frozen.commit("Synthetic later cutoff")
    frozen.inventory["repositories"][0]["cutoff_commit"] = frozen.cutoff
    frozen.inventory["repositories"][0]["artifacts"][0]["covered_files"][0][
        "source_commit"
    ] = prior
    assert "live coverage" in retired_apply_admission_issue(
        frozen.inventory, frozen.context
    )


@settings(
    max_examples=12,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(st.binary(min_size=1, max_size=24))
def test_any_live_generated_byte_mutation_fails(frozen, suffix):
    frozen.generated.write_bytes(frozen.generated_bytes + suffix)
    assert retired_apply_admission_issue(frozen.inventory, frozen.context)
    frozen.generated.write_bytes(frozen.generated_bytes)
    assert retired_apply_admission_issue(frozen.inventory, frozen.context) is None


@pytest.mark.parametrize("target_name", ("manifest", "generated"))
def test_live_read_rejects_symlink_swap_at_open(
    frozen, tmp_path, monkeypatch, target_name
):
    target = getattr(frozen, target_name)
    original_bytes = target.read_bytes()
    outside = tmp_path / "synthetic-outside-bytes"
    outside.write_bytes(original_bytes)
    real_open = os.open
    raced = False

    def swap_at_open(path, flags, *args, **kwargs):
        nonlocal raced
        if not raced and path == target.name and "dir_fd" in kwargs:
            raced = True
            target.unlink()
            target.symlink_to(outside)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_at_open)
    issue = retired_apply_admission_issue(frozen.inventory, frozen.context)
    assert raced
    assert issue is not None
    assert target.is_symlink()


def test_side_branch_head_is_not_directly_admitted(frozen: _Frozen) -> None:
    """A head that neither descends from nor precedes the cutoff is refused."""
    early = frozen.cutoff
    (frozen.repo / "README.txt").write_text("later frozen main\n")
    frozen.inventory["repositories"][0]["cutoff_commit"] = frozen.commit("later cutoff")
    _git(frozen.repo, "checkout", "-b", "side", early)
    (frozen.repo / "side.txt").write_text("side\n")
    frozen.commit("side commit")
    issue = retired_apply_admission_issue(frozen.inventory, frozen.context)
    assert issue is not None and "descend" in issue
