"""Unreceipted retirement trust boundaries, using only fixture signing keys.

The signed mutations deliberately re-sign structural changes so each refusal
exercises the named check instead of merely failing signature verification.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import itertools
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from axiom_encode import cli
from axiom_encode.retired_apply import retrospective_head_issue
from tests import test_cli as fixtures
from tests.eval_evidence_fixtures import install_test_eval_evidence_keys
from tests.test_retired_apply_paths import _RotatedBroker


@pytest.fixture(autouse=True)
def retirement_test_signer(monkeypatch):
    previous_umask = os.umask(0o022)
    install_test_eval_evidence_keys(
        monkeypatch,
        apply_private_key=fixtures.TEST_APPLY_PRIVATE_KEY_B64,
        apply_public_key=fixtures.TEST_APPLY_PUBLIC_KEY_B64,
    )
    monkeypatch.setattr(
        cli,
        "_read_only_guard_encoder_execution_identity",
        lambda *_args: dict(fixtures.TEST_PINNED_ENCODER_IDENTITY),
    )
    monkeypatch.setattr(
        cli,
        "_require_clean_axiom_encode_git_provenance",
        _encoder_provenance,
    )
    try:
        yield
    finally:
        os.umask(previous_umask)


def _encoder_provenance():
    return {
        "root": "/repo/axiom-encode",
        "commit": fixtures.TEST_PINNED_ENCODER_IDENTITY["commit"],
        "dirty_tracked": False,
        "version": fixtures.AXIOM_ENCODE_TEST_VERSION,
        "version_commit": "b" * 40,
        "identity_source": "git",
    }


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _snapshot(repo: Path) -> dict[str, bytes | str]:
    """Include every checkout file, including untracked transaction residue."""

    return {
        path.relative_to(repo).as_posix(): (
            os.readlink(path) if path.is_symlink() else path.read_bytes()
        )
        for path in repo.rglob("*")
        if ".git" not in path.relative_to(repo).parts
        and (path.is_symlink() or path.is_file())
    }


@dataclass
class RetirementRepo:
    repo: Path
    corpus_path: Path
    release: object
    primary: str
    companion: str
    base: str
    originals: dict[str, bytes]

    @property
    def manifest_relative(self):
        return cli._applied_encoding_manifest_path(Path(self.primary)).as_posix()

    @property
    def manifest(self):
        return self.repo / self.manifest_relative

    def commit(self, message="test retirement"):
        fixtures._git(self.repo, "add", "-A")
        fixtures._git(self.repo, "commit", "-m", message)
        return fixtures._git(self.repo, "rev-parse", "HEAD").stdout.strip()

    def retire(self, **overrides):
        args = {
            "paths": [self.primary],
            "policy_repo_path": self.repo,
            "corpus_path": self.corpus_path,
            "reason": "replace merged council module with section encodings",
            "unreceipted": True,
        }
        args.update(overrides)
        cli.cmd_retire(SimpleNamespace(**args))

    def guard(self, *, all_files=False, base=None):
        return cli.guard_generated_change_issues(
            self.repo,
            corpus_path=self.corpus_path,
            base_ref=base or self.base,
            head_ref="HEAD",
            all_files=all_files,
        )

    def load(self, proof_cache=None):
        return cli._load_verified_applied_encoding_manifest_payload(
            self.repo,
            self.manifest_relative,
            historical_verification=True,
            local_corpus_release=self.release,
            expected_encoder_identity=fixtures.TEST_PINNED_ENCODER_IDENTITY,
            path_migration_receipt_proof_cache=proof_cache,
        )

    def receipt(self):
        payload = {
            "schema_version": cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
            "tool": cli.APPLIED_ENCODING_UNRECEIPTED_RETIRE_TOOL,
            "generated_at": "2026-11-04T00:00:00+00:00",
            "reason": "superseded merged module",
            "axiom_encode_version": fixtures.AXIOM_ENCODE_TEST_VERSION,
            "axiom_encode_git": _encoder_provenance(),
            "validation_waiver_set_sha256": _sha256(
                (self.repo / "known-validation-gaps.yaml").read_bytes()
            ),
            "base_commit": self.base,
            "applied_files": [
                {"path": path, "deleted": True, "sha256": _sha256(raw)}
                for path, raw in self.originals.items()
            ],
        }
        cli._sign_applied_encoding_manifest(payload, fixtures.TEST_APPLY_SIGNING_BROKER)
        return payload

    def write_deleted(self, payload):
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text(json.dumps(payload) + "\n")
        for path in self.originals:
            (self.repo / path).unlink(missing_ok=True)

    def base_issues(self, payload):
        return cli._unreceipted_retirement_base_issues(
            payload,
            repo_path=self.repo,
            manifest_label=self.manifest_relative,
            base_ref=self.base,
            roots=tuple(sorted(cli.RULESPEC_ATOMIC_MODULE_ROOTS)),
        )


def _repo(tmp_path, *, name="example", suffix="", waiver_state=None):
    repo = tmp_path / "rulespec-us"
    fixtures._init_test_git_repo(repo)
    primary = f"us/policies/council/{name}.yaml"
    companion = f"us/policies/council/{name}.test.yaml"
    release = fixtures._bind_test_corpus_release(
        repo,
        tmp_path / "axiom-corpus",
        body="authoritative source\n",
    )
    attestation = fixtures.resolve_corpus_source_unit(
        "us/statute/26/1", release
    ).source_attestation
    attestation["generation_input_sha256"] = attestation["resolved_text_sha256"]
    attestation["rulespec_root"] = "rulespec-us/us"
    module = (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_path: us/statute/26/1\n"
        f"    source_sha256: {attestation['source_sha256']}\n"
        "rules: []\n"
        f"# {suffix}\n"
    ).encode()
    originals = {primary: module, companion: f"[]\n# {suffix}\n".encode()}
    for path, raw in originals.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    waiver = repo / "known-validation-gaps.yaml"
    if waiver_state is not None:
        expires = (date.today() + timedelta(days=30)).isoformat()
        waiver.write_text(
            "validate_failures:\n"
            f"  {primary}:\n"
            f"    {waiver_state}:\n"
            f"      fingerprint: sha256:{'1' * 64}\n"
            "      owner: '@axiom-test'\n"
            "      issue: https://github.com/TheAxiomFoundation/axiom-encode/issues/1\n"
            f"      expires: '{expires}'\n"
        )
        fixtures._write_test_rulespec_toolchain(
            repo, release_content_sha256=release.content_sha256
        )
    # Keep a live authenticated module so an unchanged --all audit also checks
    # the retirement receipt instead of the empty-corpus refusal.
    sentinel = repo / "us/statutes/26/1.yaml"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(module)
    sentinel_manifest = repo / cli._applied_encoding_manifest_path(
        sentinel.relative_to(repo)
    )
    sentinel_manifest.parent.mkdir(parents=True)
    sentinel_payload = fixtures._signed_manifest_payload(
        {
            "schema_version": cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
            "backend": "codex",
            "source_attestation": attestation,
            "validation_waiver_set_sha256": _sha256(waiver.read_bytes()),
            "applied_files": [
                {
                    "path": sentinel.relative_to(repo).as_posix(),
                    "sha256": _sha256(module),
                }
            ],
        }
    )
    sentinel_manifest.write_text(json.dumps(sentinel_payload) + "\n")
    fixtures._git(repo, "add", ".")
    fixtures._git(repo, "commit", "-m", "unreceipted protected base module")
    return RetirementRepo(
        repo,
        release.root,
        release,
        primary,
        companion,
        fixtures._git(repo, "rev-parse", "HEAD").stdout.strip(),
        originals,
    )


@pytest.mark.parametrize("batch", [0, 1])
@settings(
    max_examples=4,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    name=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=20),
    suffix=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 \t", max_size=100),
)
def test_command_round_trip_binds_exact_deleted_bytes(tmp_path, name, suffix, batch):
    """Soundness, pairing, round trip: random base trees reach both guard modes."""

    with tempfile.TemporaryDirectory(prefix=f"batch-{batch}-", dir=tmp_path) as case:
        checkout = _repo(Path(case), name=name, suffix=suffix)
        checkout.retire()
        payload = json.loads(checkout.manifest.read_text())
        assert payload == checkout.load()[0]
        assert payload["schema_version"] == cli.APPLIED_ENCODING_MANIFEST_SCHEMA
        assert payload["tool"] == cli.APPLIED_ENCODING_UNRECEIPTED_RETIRE_TOOL
        assert payload["base_commit"] == checkout.base
        assert set(payload) == {
            "schema_version",
            "tool",
            "generated_at",
            "reason",
            "axiom_encode_version",
            "axiom_encode_git",
            "validation_waiver_set_sha256",
            "applied_files",
            "base_commit",
            "signature",
        }
        assert payload["applied_files"] == [
            {"path": path, "deleted": True, "sha256": _sha256(raw)}
            for path, raw in checkout.originals.items()
        ]
        assert all(not (checkout.repo / path).exists() for path in checkout.originals)
        assert (
            cli._applied_encoding_manifest_signature_issue(
                payload, fixtures.TEST_APPLY_SIGNING_BROKER
            )
            is None
        )
        checkout.commit()
        assert checkout.guard() == []
        # No changed receipt is available at this boundary: --all must discover
        # and authenticate the unchanged receipt independently.
        assert checkout.guard(all_files=True, base="HEAD") == []


def _mutate_receipt(payload, mutation):
    changed = copy.deepcopy(payload)
    if mutation == "path":
        changed["applied_files"][0]["path"] = "us/policies/council/other.yaml"
    elif mutation == "hash":
        changed["applied_files"][0]["sha256"] = "f" * 64
    elif mutation == "tool":
        changed["tool"] = cli.APPLIED_ENCODING_RETIRE_TOOL
    elif mutation == "schema":
        changed["schema_version"] = "axiom-encode/applied-rulespec/v1"
    elif mutation == "added-field":
        changed["deterministic_execution"] = {}
    elif mutation == "removed-field":
        changed.pop("reason")
    elif mutation == "signature":
        changed["signature"]["value"] = "A" * len(changed["signature"]["value"])
    elif mutation == "key-id":
        changed["signature"]["key_id"] = "f" * 64
    elif mutation == "producer":
        changed["axiom_encode_git"]["commit"] = "c" * 40
    elif mutation == "pair":
        changed["applied_files"].pop()
    elif mutation == "deleted":
        changed["applied_files"][0]["deleted"] = False
    elif mutation == "base":
        changed["base_commit"] = "d" * 40
    elif mutation == "reason":
        changed["reason"] = ""
    else:
        raise AssertionError(mutation)
    return changed


@pytest.mark.parametrize(
    "mutation",
    [
        "path",
        "hash",
        "tool",
        "schema",
        "added-field",
        "removed-field",
        "signature",
        "key-id",
    ],
)
@settings(
    max_examples=4,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(suffix=st.text(alphabet="abcXYZ012 \t", max_size=80))
def test_every_single_field_tamper_is_refused(tmp_path, mutation, suffix):
    """Kills omission of signature verification for independently signed data."""

    with tempfile.TemporaryDirectory(dir=tmp_path) as case:
        checkout = _repo(Path(case), suffix=suffix)
        payload = checkout.receipt()
        checkout.write_deleted(payload)
        checkout.commit()
        assert checkout.guard() == []
        tampered = _mutate_receipt(payload, mutation)
        checkout.write_deleted(tampered)
        checkout.commit("single unsigned field mutation")
        issues = checkout.guard()
        expected = (
            "not an encoder apply manifest"
            if mutation == "schema"
            else "signing key"
            if mutation == "key-id"
            else "signature"
        )
        assert any(expected in issue for issue in issues), issues


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("path", "exactly one protected primary and its companion"),
        ("tool", "retired_manifest"),
        ("schema", "encoder apply manifest"),
        ("added-field", "exact unreceipted retirement v5 schema"),
        ("removed-field", "exact unreceipted retirement v5 schema"),
        ("signature", "signature"),
        ("key-id", "signing key"),
        ("producer", "encoder"),
        ("pair", "exactly one protected primary and its companion"),
        ("deleted", "malformed deleted file entries"),
        ("reason", "reason"),
        ("hash", "base sha256 does not match"),
        ("base", "recorded base_commit is unavailable"),
    ],
)
def test_signed_refusals_and_all_loader_agree(tmp_path, mutation, expected):
    """Differential; kills the matching schema, tool, producer or pairing check."""

    checkout = _repo(tmp_path)
    payload = checkout.receipt()
    checkout.write_deleted(payload)
    checkout.commit()
    assert checkout.guard(all_files=True, base="HEAD") == []
    assert checkout.load()[3] == []
    changed = _mutate_receipt(payload, mutation)
    if mutation not in {"signature", "key-id"}:
        cli._sign_applied_encoding_manifest(changed, fixtures.TEST_APPLY_SIGNING_BROKER)
    checkout.write_deleted(changed)
    checkout.commit("signed invalid contract")
    guard_issues = checkout.guard(all_files=True, base="HEAD")
    loader_issues = checkout.load()[3]
    assert bool(guard_issues) == bool(loader_issues) is True
    assert any(expected in issue for issue in guard_issues), guard_issues
    assert any(expected in issue for issue in loader_issues), loader_issues


@pytest.mark.parametrize("mutation", ["hash", "base"])
def test_signed_receipt_must_match_immutable_base(tmp_path, mutation):
    """Kills base-byte hashing and base-commit identity checks respectively."""

    checkout = _repo(tmp_path)
    payload = checkout.receipt()
    assert checkout.base_issues(payload) == []
    changed = _mutate_receipt(payload, mutation)
    cli._sign_applied_encoding_manifest(changed, fixtures.TEST_APPLY_SIGNING_BROKER)
    expected = (
        "base sha256 does not match"
        if mutation == "hash"
        else "recorded base_commit is unavailable"
    )
    assert any(expected in issue for issue in checkout.base_issues(changed))
    checkout.write_deleted(changed)
    checkout.commit()
    assert any(expected in issue for issue in checkout.guard())


def test_signed_retirement_wrong_historical_waiver_hash_is_refused(tmp_path):
    """A settled receipt must bind the waiver bytes at its recorded base."""

    checkout = _repo(tmp_path)
    payload = checkout.receipt()
    payload["validation_waiver_set_sha256"] = "f" * 64
    cli._sign_applied_encoding_manifest(payload, fixtures.TEST_APPLY_SIGNING_BROKER)
    checkout.write_deleted(payload)
    checkout.commit("signed retirement with an incorrect historical waiver hash")
    expected = (
        "validation_waiver_set_sha256 does not match the RuleSpec toolchain-bound "
        "waiver set"
    )
    assert any(expected in issue for issue in checkout.guard())
    # With no changed receipt or deleted path, --all reaches historical waiver
    # verification without the gate's fresh-receipt waiver check masking it.
    all_issues = checkout.guard(all_files=True, base="HEAD")
    assert any(expected in issue for issue in all_issues), all_issues
    loader_issues = checkout.load()[3]
    assert any(expected in issue for issue in loader_issues), loader_issues


@pytest.mark.parametrize("recorded_fault", ["owner", "bytes"])
def test_retirement_replays_recorded_base_distinct_from_pr_base(
    tmp_path, recorded_fault
):
    """A clean PR base cannot hide ownership or different recorded-base bytes."""

    checkout = _repo(tmp_path)
    owner = checkout.repo / ".axiom/legacy-replacements/recorded-owner.json"
    if recorded_fault == "owner":
        owner.parent.mkdir(parents=True)
        owner.write_text(
            json.dumps({"replacement": {"source": checkout.primary}}) + "\n"
        )
        expected = "already covered"
    else:
        (checkout.repo / checkout.primary).write_bytes(
            checkout.originals[checkout.primary] + b"# different recorded bytes\n"
        )
        expected = "base sha256 does not match"
    recorded_base = checkout.commit("recorded base cannot authorize retirement")
    if recorded_fault == "owner":
        owner.unlink()
    else:
        (checkout.repo / checkout.primary).write_bytes(
            checkout.originals[checkout.primary]
        )
    checkout.base = checkout.commit("unowned exact bytes at the PR base")
    payload = checkout.receipt()
    assert checkout.base_issues(payload) == []
    payload["base_commit"] = recorded_base
    cli._sign_applied_encoding_manifest(payload, fixtures.TEST_APPLY_SIGNING_BROKER)
    proof_issues = checkout.base_issues(payload)
    assert any(
        expected in issue and "at the recorded retirement base" in issue
        for issue in proof_issues
    ), proof_issues
    assert all("at the immutable base" not in issue for issue in proof_issues)
    checkout.write_deleted(payload)
    checkout.commit("retirement incorrectly records the earlier disallowed base")
    for all_files in (False, True):
        issues = checkout.guard(all_files=all_files)
        assert any(expected in issue for issue in issues), issues


@pytest.mark.parametrize("field", ["tool", "backend", "identity_source"])
@settings(
    max_examples=4,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@example(value=[])
@example(value={})
@example(value=None)
@given(
    value=st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-10, max_value=10),
        st.lists(st.integers(min_value=-10, max_value=10), max_size=3),
        st.dictionaries(
            st.sampled_from(["claim", "tool", "backend"]), st.booleans(), max_size=3
        ),
    )
)
def test_signed_malformed_contract_values_refuse_without_crashing(
    tmp_path, field, value
):
    """Untrusted JSON container/scalar shapes must become issues, never crashes."""

    with tempfile.TemporaryDirectory(dir=tmp_path) as case:
        checkout = _repo(Path(case))
        payload = checkout.receipt()
        if field == "identity_source":
            payload["axiom_encode_git"]["identity_source"] = value
            expected = "identity_source"
        else:
            payload[field] = value
            expected = (
                "allowlisted tool/backend contract"
                if field == "tool"
                else "exact unreceipted retirement v5 schema"
                if value is None
                else "backend is not canonical"
            )
        cli._sign_applied_encoding_manifest(payload, fixtures.TEST_APPLY_SIGNING_BROKER)
        checkout.write_deleted(payload)
        checkout.commit("signed malformed contract value")
        guard_issues = checkout.guard(all_files=True, base="HEAD")
        loader_issues = checkout.load()[3]
        assert isinstance(guard_issues, list) and isinstance(loader_issues, list)
        assert any(expected in issue for issue in guard_issues), guard_issues
        assert any(expected in issue for issue in loader_issues), loader_issues


def test_protected_base_audit_checks_absence_in_its_snapshot(tmp_path):
    checkout = _repo(tmp_path)
    checkout.retire()
    retirement_commit = checkout.commit()
    raw = checkout.manifest.read_bytes()
    (checkout.repo / checkout.primary).write_bytes(checkout.originals[checkout.primary])
    checkout.commit("later module at a retired path")
    claims, issues = cli._guard_base_manifest_claims(
        raw,
        repo_path=checkout.repo,
        manifest_path=checkout.manifest_relative,
        historical_head_ref="HEAD",
        retrospective=False,
        roots=tuple(sorted(cli.RULESPEC_ATOMIC_MODULE_ROOTS)),
        signing_broker=fixtures.TEST_APPLY_SIGNING_BROKER,
        snapshot_commit=retirement_commit,
    )
    assert issues == []
    assert claims == {checkout.primary, checkout.companion}


def test_retirement_after_unrelated_commit_preserves_pr_base_proof(tmp_path):
    checkout = _repo(tmp_path)
    original_base = checkout.base
    (checkout.repo / "README.md").write_text("unrelated earlier PR work\n")
    retirement_base = checkout.commit("unrelated commit before supervised retirement")
    checkout.retire()
    assert json.loads(checkout.manifest.read_text())["base_commit"] == retirement_base
    checkout.commit()
    assert checkout.guard(base=original_base) == []
    assert checkout.guard(all_files=True, base="HEAD") == []


@pytest.mark.parametrize("member", ["primary", "companion"])
@pytest.mark.parametrize(
    "owner",
    [
        "mirrored",
        "v5",
        "v1",
        "v1-local",
        "v1-local-mapping",
        "legacy-owned",
        "legacy-replacement",
        "exact-dependent",
        "retained-successor",
        "path-migration",
        "retired-source-metadata",
        "retired-source-aux",
    ],
)
def test_any_base_receipt_owner_excludes_unreceipted_route(tmp_path, owner, member):
    """Exclusivity; kills the selected namespace/class/path ownership check."""

    checkout = _repo(tmp_path)
    claimed = getattr(checkout, member)
    payload = {
        "schema_version": cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
        "tool": "axiom-encode encode",
        "applied_files": [
            {"path": claimed, "sha256": _sha256(checkout.originals[claimed])}
        ],
    }
    receipt = checkout.repo / ".axiom/encoding-manifests/us/policies/owner.json"
    if owner == "mirrored":
        receipt = checkout.repo / cli._applied_encoding_manifest_path(Path(claimed))
    elif owner in {"v1", "v1-local", "v1-local-mapping", "legacy-owned"}:
        payload["schema_version"] = "axiom-encode/applied-rulespec/v1"
        if owner in {"v1-local", "v1-local-mapping"}:
            receipt = checkout.repo / "us/.axiom/encoding-manifests/policies/owner.json"
            local = claimed.removeprefix("us/")
            payload["applied_files"] = (
                {local: _sha256(checkout.originals[claimed])}
                if owner == "v1-local-mapping"
                else [{"path": local, "sha256": _sha256(checkout.originals[claimed])}]
            )
        if owner == "legacy-owned":
            payload["backend"] = "manual"
            payload["tool"] = "axiom-encode sign-applied-files"
    elif owner in {"legacy-replacement", "exact-dependent", "retained-successor"}:
        receipt = checkout.repo / ".axiom/legacy-replacements/owner.json"
        payload = {
            "tool": {
                "legacy-replacement": cli.APPLIED_ENCODING_LEGACY_REPLACEMENT_TOOL,
                "exact-dependent": cli.APPLIED_ENCODING_LEGACY_EXACT_DEPENDENT_TOOL,
                "retained-successor": cli.APPLIED_ENCODING_LEGACY_RETAINED_SUCCESSOR_TOOL,
            }[owner],
            "replacement": {"source": claimed},
        }
    elif owner == "path-migration":
        receipt = checkout.repo / ".axiom/path-migrations/owner.json"
        payload = {"moves": [{"from": claimed, "to": "us/policies/moved.yaml"}]}
    elif owner == "retired-source-metadata":
        payload["tool"] = cli.RETIRED_SOURCE_METADATA_TOOL
    elif owner == "retired-source-aux":
        receipt = checkout.repo / ".axiom/retired-source-metadata-migrations/owner.json"
        payload = {"files": [{"path": claimed}]}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload) + "\n")
    checkout.base = checkout.commit("prior ownership claim")
    expected = "already covered at the immutable base"
    assert any(expected in issue for issue in checkout.base_issues(checkout.receipt()))
    before = _snapshot(checkout.repo)
    with pytest.raises(SystemExit, match="already covered|unreceipted ownership"):
        checkout.retire()
    assert _snapshot(checkout.repo) == before


@pytest.mark.parametrize(
    "raw",
    [
        b"{invalid JSON",
        b"[]",
        b'{"applied_files": [], "applied_files": []}',
        b'{"unrecognized_claim": NaN}',
        b'{"unrecognized_claim": Infinity}',
        b'{"unrecognized_claim": 1e999}',
    ],
    ids=["invalid-json", "nonobject", "duplicate-key", "nan", "infinity", "overflow"],
)
def test_ambiguous_receipt_inventory_refuses_without_mutation(tmp_path, raw):
    checkout = _repo(tmp_path)
    receipt = checkout.repo / ".axiom/legacy-replacements/ambiguous.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(raw)
    checkout.base = checkout.commit("ambiguous prior receipt inventory")
    issues = checkout.base_issues(checkout.receipt())
    assert any("cannot establish unreceipted ownership" in issue for issue in issues), (
        issues
    )
    before = _snapshot(checkout.repo)
    with pytest.raises(SystemExit, match="cannot establish unreceipted ownership"):
        checkout.retire()
    assert _snapshot(checkout.repo) == before


def test_uncommitted_receipt_owner_refuses_without_mutation(tmp_path):
    checkout = _repo(tmp_path)
    receipt = checkout.repo / ".axiom/legacy-replacements/new-owner.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"replacement": {"source": checkout.primary}}) + "\n")
    payload = checkout.receipt()
    issues = cli._unreceipted_retirement_current_ownership_issues(
        payload,
        repo_path=checkout.repo,
        manifest_label=checkout.manifest_relative,
        roots=tuple(sorted(cli.RULESPEC_ATOMIC_MODULE_ROOTS)),
    )
    assert any("already covered in the working tree" in issue for issue in issues), (
        issues
    )
    before = _snapshot(checkout.repo)
    with pytest.raises(SystemExit, match="already covered in the working tree"):
        checkout.retire()
    assert _snapshot(checkout.repo) == before


@pytest.mark.parametrize("owner_state", ["committed", "working-tree"])
def test_pre_install_recheck_refuses_owner_added_during_signing(
    tmp_path, monkeypatch, owner_state
):
    """Signing may not leave a stale planning proof in control of installation."""

    checkout = _repo(tmp_path)
    before = _snapshot(checkout.repo)
    owner_label = ".axiom/legacy-replacements/concurrent-owner.json"
    owner = checkout.repo / owner_label
    owner_bytes = (
        json.dumps({"replacement": {"source": checkout.primary}}) + "\n"
    ).encode()
    broker = cli._require_applied_encoding_manifest_signer()
    original_sign = broker.apply_ed25519_sign
    sign_calls = []

    def sign_and_add_owner(message):
        signature = original_sign(message)
        sign_calls.append(message)
        owner.parent.mkdir(parents=True)
        owner.write_bytes(owner_bytes)
        if owner_state == "committed":
            checkout.commit("owner added after retirement planning")
            # Only the refreshed HEAD proof can see this owner; the checkout
            # inventory has already lost it again before installation starts.
            owner.unlink()
        return signature

    monkeypatch.setattr(broker, "apply_ed25519_sign", sign_and_add_owner)
    with pytest.raises(
        RuntimeError, match="Retirement ownership changed:.*already covered"
    ):
        checkout.retire()
    assert len(sign_calls) == 1
    expected = (
        before if owner_state == "committed" else before | {owner_label: owner_bytes}
    )
    assert _snapshot(checkout.repo) == expected
    assert not checkout.manifest.exists()


@pytest.mark.parametrize("member", ["primary", "companion"])
def test_uncommitted_deleted_bytes_refuse_without_mutation(tmp_path, member):
    checkout = _repo(tmp_path)
    path = checkout.repo / getattr(checkout, member)
    path.write_bytes(path.read_bytes() + b"# uncommitted change\n")
    before = _snapshot(checkout.repo)
    with pytest.raises(SystemExit, match="base sha256 does not match"):
        checkout.retire()
    assert _snapshot(checkout.repo) == before


@pytest.mark.parametrize("member", ["primary", "companion"])
def test_symlink_target_refuses_without_mutation(tmp_path, member):
    checkout = _repo(tmp_path)
    target = checkout.repo / getattr(checkout, member)
    outside = tmp_path / "outside.yaml"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    before = _snapshot(checkout.repo)
    outside_before = outside.read_bytes()
    with pytest.raises(RuntimeError, match="Apply target path contains a symlink"):
        checkout.retire()
    assert _snapshot(checkout.repo) == before
    assert outside.read_bytes() == outside_before


@pytest.mark.parametrize(
    "preflight",
    [
        "_recover_apply_transaction",
        "load_rulespec_local_corpus_release",
        "verify_rulespec_validation_waiver_set",
        "_current_guard_encoder_execution_identity",
        "_require_clean_axiom_encode_git_provenance",
    ],
)
def test_unreceipted_retirement_keeps_existing_preflights(
    tmp_path, monkeypatch, preflight
):
    checkout = _repo(tmp_path)
    before = _snapshot(checkout.repo)

    def refuse(*_args, **_kwargs):
        raise RuntimeError(f"required preflight: {preflight}")

    monkeypatch.setattr(cli, preflight, refuse)
    with pytest.raises(RuntimeError, match=f"required preflight: {preflight}"):
        checkout.retire()
    assert _snapshot(checkout.repo) == before


@pytest.mark.parametrize("unreceipted", [False, True])
def test_refused_command_preserves_every_checkout_byte(tmp_path, unreceipted):
    """Kills implicit fallback and the prohibition on companion-only commands."""

    checkout = _repo(tmp_path)
    before = _snapshot(checkout.repo)
    with pytest.raises(SystemExit, match="companion test|current apply manifest"):
        checkout.retire(
            paths=[checkout.companion] if unreceipted else [checkout.primary],
            unreceipted=unreceipted,
        )
    assert _snapshot(checkout.repo) == before


def test_missing_companion_refuses_without_mutation(tmp_path):
    """Kills the command's mandatory live companion check."""

    checkout = _repo(tmp_path)
    (checkout.repo / checkout.companion).unlink()
    checkout.base = checkout.commit("missing companion at base")
    before = _snapshot(checkout.repo)
    with pytest.raises(SystemExit, match="companion|exist|regular base"):
        checkout.retire()
    assert _snapshot(checkout.repo) == before


@pytest.mark.parametrize("member", ["primary", "companion"])
@pytest.mark.parametrize("all_files", [False, True])
def test_recreated_deleted_member_is_refused_by_guard_and_loader(
    tmp_path, member, all_files
):
    """Kills absence checks for each member, including unchanged --all receipts."""

    checkout = _repo(tmp_path)
    checkout.retire()
    checkout.commit()
    proof_cache = {}
    assert checkout.load(proof_cache)[3] == []
    path = getattr(checkout, member)
    receipt_bytes = checkout.manifest.read_bytes()
    (checkout.repo / path).write_bytes(checkout.originals[path] + b"# recreated\n")
    recreated_base = checkout.commit("restore one retired member")
    guard_issues = checkout.guard(all_files=True, base="HEAD")
    loader_issues = checkout.load(proof_cache)[3]
    expected = f"lists {path} as deleted but it exists"
    assert any(expected in issue for issue in guard_issues), guard_issues
    assert any(expected in issue for issue in loader_issues), loader_issues
    (checkout.repo / path).unlink()
    checkout.commit("delete recreated member using unchanged receipt")
    assert checkout.manifest.read_bytes() == receipt_bytes
    assert checkout.load(proof_cache)[3] == []
    issues = checkout.guard(all_files=all_files, base=recreated_base)
    assert issues
    if all_files:
        assert any(f"{path} base sha256 does not match" in issue for issue in issues), (
            issues
        )


def _freeze_retirement_replay(checkout, monkeypatch, cutoff, broker):
    artifacts = []
    for label in cli._all_applied_encoding_manifest_paths(checkout.repo):
        raw = (checkout.repo / label).read_bytes()
        payload = json.loads(raw)
        covered = []
        for entry in payload["applied_files"]:
            evidence = {"path": entry["path"], "sha256": entry["sha256"]}
            if entry.get("deleted") is True:
                evidence["source_commit"] = checkout.base
            covered.append(evidence)
        artifacts.append(
            {"path": label, "sha256": _sha256(raw), "covered_files": covered}
        )
    monkeypatch.setattr(
        broker,
        "_retired_apply_inventory",
        {
            "schema": "axiom-encode/retired-apply-inventory/v1",
            "repositories": [
                {
                    "repository": "TheAxiomFoundation/rulespec-us",
                    "cutoff_commit": cutoff,
                    "artifacts": artifacts,
                }
            ],
        },
    )
    monkeypatch.setattr(cli, "get_signing_broker", lambda **_kwargs: broker)
    monkeypatch.setattr(cli, "_applied_encoding_manifest_verifier", lambda: broker)
    assert (
        retrospective_head_issue(broker.retired_apply_inventory, checkout.repo, cutoff)
        is None
    )


@pytest.mark.parametrize("all_files", [False, True])
@pytest.mark.parametrize("retrospective", [False, True])
@pytest.mark.parametrize("restoration", ["unowned", "receipt-owner", "model-group"])
def test_unchanged_receipt_cannot_retire_recreated_pair(
    tmp_path, monkeypatch, all_files, retrospective, restoration
):
    checkout = _repo(tmp_path)
    broker = _RotatedBroker({}) if retrospective else None
    fixtures._git(
        checkout.repo,
        "remote",
        "add",
        "origin",
        "https://github.com/TheAxiomFoundation/rulespec-us.git",
    )
    model_path = checkout.repo / cli._applied_encoding_manifest_path(
        Path("us/statutes/26/1.yaml")
    )
    original_model = model_path.read_bytes()
    checkout.retire()
    first_retirement = checkout.commit()
    receipt_bytes = checkout.manifest.read_bytes()
    for path, raw in checkout.originals.items():
        (checkout.repo / path).write_bytes(raw + b"# fresh replacement\n")
    replacement_entries = [
        {"path": path, "sha256": _sha256((checkout.repo / path).read_bytes())}
        for path in checkout.originals
    ]
    if restoration == "model-group":
        group = json.loads(original_model)
        group["applied_files"].extend(replacement_entries)
        fixtures._signed_manifest_payload(group)
        model_path.write_text(json.dumps(group) + "\n")
    elif restoration == "receipt-owner":
        owner = (
            checkout.repo
            / cli.APPLIED_ENCODING_LEGACY_REPLACEMENT_RECEIPT_DIR
            / ("e" * 64 + ".json")
        )
        owner.parent.mkdir(parents=True, exist_ok=True)
        owner.write_text(json.dumps({"applied_files": replacement_entries}) + "\n")
    recreated_base = checkout.commit("restore pair under a fresh owner")
    if restoration == "model-group":
        assert checkout.guard(base=first_retirement) == []
        if broker is not None:
            cli._sign_applied_encoding_manifest(group, broker)
            model_path.write_text(json.dumps(group) + "\n")
            recreated_base = checkout.commit("rotate fixture model owner to active key")
    for path in checkout.originals:
        (checkout.repo / path).unlink()
    if restoration == "model-group":
        model_path.write_bytes(original_model)
    cutoff = checkout.commit("shrink model group and delete recreated pair")
    assert checkout.manifest.read_bytes() == receipt_bytes
    assert checkout.load()[3] == []
    if retrospective:
        _freeze_retirement_replay(checkout, monkeypatch, cutoff, broker)
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.corpus_path,
        base_ref=recreated_base,
        head_ref=cutoff,
        all_files=all_files,
        retrospective=retrospective,
    )
    assert issues
    for path in checkout.originals:
        assert any(path in issue for issue in issues), issues
        if all_files or restoration == "model-group":
            assert any(
                f"{path} base sha256 does not match" in issue for issue in issues
            ), issues
    if (all_files or restoration == "model-group") and restoration != "unowned":
        expected_owner = model_path if restoration == "model-group" else owner
        owner_label = expected_owner.relative_to(checkout.repo).as_posix()
        assert any(
            "already covered" in issue and f"by {owner_label}" in issue
            for issue in issues
        ), issues


def test_settled_untouched_retirement_stays_historical(tmp_path):
    checkout = _repo(tmp_path)
    checkout.retire()
    settled_base = checkout.commit()
    receipt_bytes = checkout.manifest.read_bytes()
    (checkout.repo / "README.md").write_text("Unrelated documentation change.\n")
    checkout.commit("leave settled retirement untouched")
    assert checkout.guard(all_files=True, base=settled_base) == []
    assert (
        cli.guard_generated_change_issues(
            checkout.repo, corpus_path=checkout.corpus_path, all_files=True
        )
        == []
    )
    assert checkout.load()[3] == []
    census = cli._manifest_census(
        checkout.repo,
        roots=tuple(sorted(cli.RULESPEC_ATOMIC_MODULE_ROOTS)),
        local_corpus_release=checkout.release,
    )
    assert census["total"] == census["encoder"] == 1
    assert census["unmanifested"] == 0
    assert checkout.manifest.read_bytes() == receipt_bytes


@pytest.mark.parametrize("member", ["primary", "companion"])
def test_no_base_all_guard_checks_dirty_second_deletion(tmp_path, member):
    checkout = _repo(tmp_path)
    checkout.retire()
    checkout.commit()
    receipt_bytes = checkout.manifest.read_bytes()
    for path, raw in checkout.originals.items():
        (checkout.repo / path).write_bytes(raw + b"# recreated\n")
    checkout.commit("restore pair before dirty deletion")
    for path in checkout.originals:
        (checkout.repo / path).unlink()
    assert checkout.manifest.read_bytes() == receipt_bytes
    issues = cli.guard_generated_change_issues(
        checkout.repo, corpus_path=checkout.corpus_path, all_files=True
    )
    path = getattr(checkout, member)
    assert any(f"{path} base sha256 does not match" in issue for issue in issues), (
        issues
    )


def test_coverage_claimant_cannot_reuse_unchanged_retirement(tmp_path):
    checkout = _repo(tmp_path)
    model_relative = cli._applied_encoding_manifest_path(Path("us/statutes/26/1.yaml"))
    model_path = checkout.repo / model_relative
    original_model = model_path.read_bytes()
    checkout.retire()
    checkout.commit()
    for path, raw in checkout.originals.items():
        (checkout.repo / path).write_bytes(raw + b"# recreated\n")
    group = json.loads(original_model)
    group["applied_files"].extend(
        {"path": path, "sha256": _sha256((checkout.repo / path).read_bytes())}
        for path in checkout.originals
    )
    fixtures._signed_manifest_payload(group)
    model_path.write_text(json.dumps(group) + "\n")
    recreated_base = checkout.commit("restore pair in model group")
    for path in checkout.originals:
        (checkout.repo / path).unlink()
    model_path.write_bytes(original_model)
    checkout.commit("shrink model group and delete pair")
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.corpus_path,
        base_ref=recreated_base,
        changed_files=[model_relative.as_posix()],
    )
    assert any("base sha256 does not match" in issue for issue in issues), issues
    assert any("already covered" in issue for issue in issues), issues


def test_unreceipted_retirement_rolls_back_second_delete_failure(tmp_path, monkeypatch):
    checkout = _repo(tmp_path)
    before = _snapshot(checkout.repo)
    original_unlink = Path.unlink

    def fail_second(path, *args, **kwargs):
        if path == checkout.repo / checkout.companion and not kwargs.get(
            "missing_ok", False
        ):
            raise OSError("injected second deletion failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_second)
    with pytest.raises(RuntimeError, match="transaction failed"):
        checkout.retire()
    assert _snapshot(checkout.repo) == before


@pytest.mark.parametrize("state", ["active", "pending"])
def test_retire_waived_unreceipted_module_then_retire_waiver(tmp_path, state):
    """The pre-retirement waiver hash survives the existing exact-decrement flow."""

    checkout = _repo(tmp_path, waiver_state=state)
    old_waiver_hash = _sha256(
        (checkout.repo / "known-validation-gaps.yaml").read_bytes()
    )
    checkout.retire()
    payload = json.loads(checkout.manifest.read_text())
    assert payload["validation_waiver_set_sha256"] == old_waiver_hash
    checkout.commit("retire waived unreceipted module")
    assert checkout.guard() == []
    (checkout.repo / "known-validation-gaps.yaml").write_text(
        fixtures.TEST_VALIDATION_WAIVER_TEXT
    )
    fixtures._write_test_rulespec_toolchain(
        checkout.repo,
        release_content_sha256=checkout.release.content_sha256,
    )
    checkout.commit("remove obsolete validation waiver and refresh toolchain binding")
    assert json.loads(checkout.manifest.read_text()) == payload
    assert checkout.guard() == []
    assert checkout.guard(all_files=True) == []
    # The unrelated live model fixture also predates waiver removal. Give it a
    # fresh valid model receipt so this final unchanged --all audit isolates the
    # historical retirement receipt's pre-retirement waiver identity.
    sentinel_manifest = (
        checkout.repo / ".axiom/encoding-manifests/us/statutes/26/1.json"
    )
    sentinel = json.loads(sentinel_manifest.read_text())
    current_waiver_hash = _sha256(
        (checkout.repo / "known-validation-gaps.yaml").read_bytes()
    )
    sentinel["validation_waiver_set_sha256"] = current_waiver_hash
    sentinel["validation_execution"]["policy_pre_apply"][
        "validation_waiver_set_sha256"
    ] = current_waiver_hash
    cli._sign_applied_encoding_manifest(sentinel, fixtures.TEST_APPLY_SIGNING_BROKER)
    sentinel_manifest.write_text(json.dumps(sentinel) + "\n")
    checkout.commit("fresh unrelated fixture model receipt after waiver retirement")
    assert checkout.guard(all_files=True, base="HEAD") == []
    assert checkout.load()[3] == []


@pytest.mark.parametrize("mutation", ["path", "hash", "base"])
@settings(
    max_examples=3,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(nonce=st.binary(min_size=1, max_size=32))
def test_immutable_proof_cache_preserves_cold_refusals(
    tmp_path, monkeypatch, mutation, nonce
):
    """A warm proof cannot authorize changed deletion bytes, paths or history."""

    base = "a" * 40
    primary = "us/policies/council/example.yaml"
    companion = "us/policies/council/example.test.yaml"
    originals = {primary: b"module\n" + nonce, companion: b"tests\n" + nonce}
    payload = {
        "base_commit": base,
        "applied_files": [
            {"path": path, "deleted": True, "sha256": _sha256(raw)}
            for path, raw in originals.items()
        ],
    }
    blob_reads = []

    def git_snapshot(_repo, *_args):
        requested = _args[-1].removesuffix("^{commit}")
        if requested in {base, "HEAD"}:
            return base
        raise RuntimeError("recorded commit unavailable")

    def read_snapshot(_repo, commit, path, **_kwargs):
        blob_reads.append((commit, path.as_posix()))
        return originals.get(path.as_posix()), []

    monkeypatch.setattr(cli, "_rulespec_migration_git", git_snapshot)
    monkeypatch.setattr(
        cli,
        "_guard_base_receipt_tree",
        lambda *_args: SimpleNamespace(issue=None, records={}),
    )
    monkeypatch.setattr(cli, "_guard_base_receipt_bytes", read_snapshot)
    proof_cache = {}

    def prove(candidate, cache, *, reference=base):
        return cli._unreceipted_retirement_base_issues(
            candidate,
            repo_path=tmp_path,
            manifest_label=".axiom/encoding-manifests/us/policies/council/example.json",
            base_ref=reference,
            roots=tuple(sorted(cli.RULESPEC_ATOMIC_MODULE_ROOTS)),
            proof_cache=cache,
        )

    assert prove(payload, proof_cache) == []
    read_count = len(blob_reads)
    assert prove(payload, proof_cache) == []
    assert len(blob_reads) == read_count
    assert prove(payload, {}) == []
    assert len(blob_reads) > read_count
    changed = copy.deepcopy(payload)
    if mutation == "path":
        changed["applied_files"][0]["path"] = "us/policies/council/other.yaml"
    elif mutation == "hash":
        changed["applied_files"][0]["sha256"] = "f" * 64
    else:
        changed["base_commit"] = "b" * 40
    warm = prove(changed, proof_cache)
    cold = prove(changed, {})
    assert warm == cold and cold
    before_mutable = len(blob_reads)
    assert prove(payload, proof_cache, reference="HEAD") == []
    assert prove(payload, proof_cache, reference="HEAD") == []
    assert len(blob_reads) >= before_mutable + 2 * len(originals)


@pytest.mark.parametrize("all_files", [False, True])
def test_changed_retirement_requires_guard_base_waiver_identity(tmp_path, all_files):
    """A historical waiver match cannot admit a new stale-waiver retirement."""

    checkout = _repo(tmp_path)
    payload = checkout.receipt()
    recorded_waiver_hash = payload["validation_waiver_set_sha256"]
    waiver = checkout.repo / "known-validation-gaps.yaml"
    waiver.write_text(waiver.read_text() + "# later protected waiver identity\n")
    fixtures._write_test_rulespec_toolchain(
        checkout.repo,
        release_content_sha256=checkout.release.content_sha256,
    )
    current_waiver_hash = _sha256(waiver.read_bytes())
    assert current_waiver_hash != recorded_waiver_hash
    sentinel_manifest = (
        checkout.repo / ".axiom/encoding-manifests/us/statutes/26/1.json"
    )
    sentinel = json.loads(sentinel_manifest.read_text())
    sentinel["validation_waiver_set_sha256"] = current_waiver_hash
    sentinel["validation_execution"]["policy_pre_apply"][
        "validation_waiver_set_sha256"
    ] = current_waiver_hash
    cli._sign_applied_encoding_manifest(sentinel, fixtures.TEST_APPLY_SIGNING_BROKER)
    sentinel_manifest.write_text(json.dumps(sentinel) + "\n")
    guard_base = checkout.commit("change protected waiver identity before retirement")

    # The recorded and guard bases both contain the exact, unowned pair.
    assert checkout.base_issues(payload) == []
    assert (
        cli._unreceipted_retirement_base_issues(
            payload,
            repo_path=checkout.repo,
            manifest_label=checkout.manifest_relative,
            base_ref=guard_base,
            roots=tuple(sorted(cli.RULESPEC_ATOMIC_MODULE_ROOTS)),
        )
        == []
    )
    checkout.write_deleted(payload)
    checkout.commit("introduce fixture-signed stale-waiver retirement")
    assert checkout.load()[3] == []
    issues = checkout.guard(all_files=all_files, base=guard_base)
    assert any("validation_waiver_set_sha256" in issue for issue in issues), issues


@pytest.mark.parametrize("member", ["primary", "companion"])
def test_protected_base_audit_refuses_present_deleted_member(tmp_path, member):
    """Base receipt obligations independently reject a false deletion claim."""

    checkout = _repo(tmp_path)
    checkout.retire()
    checkout.commit("retire original pair")
    raw = checkout.manifest.read_bytes()
    path = getattr(checkout, member)
    (checkout.repo / path).write_bytes(checkout.originals[path])
    recreated_base = checkout.commit("recreate one deleted member")
    claims, issues = cli._guard_base_manifest_claims(
        raw,
        repo_path=checkout.repo,
        manifest_path=checkout.manifest_relative,
        historical_head_ref="HEAD",
        retrospective=False,
        roots=tuple(sorted(cli.RULESPEC_ATOMIC_MODULE_ROOTS)),
        signing_broker=fixtures.TEST_APPLY_SIGNING_BROKER,
        snapshot_commit=recreated_base,
    )
    assert claims == {checkout.primary, checkout.companion}
    assert any(f"lists {path} as deleted but it exists" in issue for issue in issues), (
        issues
    )


@pytest.mark.parametrize("all_files", [False, True])
def test_guard_refuses_replacing_false_base_deletion_with_live_model(
    tmp_path, all_files
):
    """A valid head model cannot hide a base receipt's false deletion claim."""

    checkout = _repo(tmp_path)
    checkout.retire()
    checkout.commit("retire original pair")
    for path, raw in checkout.originals.items():
        (checkout.repo / path).write_bytes(raw)
    recreated_base = checkout.commit("restore pair under unchanged deletion receipt")
    model_path = checkout.repo / ".axiom/encoding-manifests/us/statutes/26/1.json"
    model = json.loads(model_path.read_text())
    model["applied_files"] = [
        {"path": path, "sha256": _sha256(raw)}
        for path, raw in checkout.originals.items()
    ]
    fixtures._signed_manifest_payload(model)
    checkout.manifest.write_text(json.dumps(model) + "\n")
    checkout.commit("replace false deletion receipt with fixture model coverage")
    assert checkout.load()[3] == []
    issues = checkout.guard(base=recreated_base, all_files=all_files)
    assert any("as deleted but it exists" in issue for issue in issues), issues


@pytest.mark.parametrize("committed", [False, True], ids=["working-tree", "HEAD"])
@pytest.mark.parametrize("member", ["primary", "companion"])
@pytest.mark.parametrize("shape", ["list", "nested-value", "local-list"])
def test_any_string_receipt_owner_refuses_without_mutation(
    tmp_path, committed, member, shape
):
    checkout = _repo(tmp_path)
    claimed = getattr(checkout, member)
    receipt = checkout.repo / ".axiom/encoding-manifests/owner.json"
    if shape == "local-list":
        receipt = checkout.repo / "us/.axiom/encoding-manifests/owner.json"
        payload = {"covered": [claimed.removeprefix("us/")]}
    elif shape == "nested-value":
        payload = {"unknown": [{"metadata": {"reference": claimed}}]}
    else:
        payload = {"covered": [claimed]}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload) + "\n")
    if committed:
        checkout.base = checkout.commit("plain-string prior receipt owner")
        issues = checkout.base_issues(checkout.receipt())
    else:
        issues = cli._unreceipted_retirement_current_ownership_issues(
            checkout.receipt(),
            repo_path=checkout.repo,
            manifest_label=checkout.manifest_relative,
            roots=tuple(sorted(cli.RULESPEC_ATOMIC_MODULE_ROOTS)),
        )
    before = _snapshot(checkout.repo)
    with pytest.raises(SystemExit, match="already covered"):
        checkout.retire()
    assert _snapshot(checkout.repo) == before
    assert any("already covered" in issue for issue in issues), issues


def _other_receipt_issues(claimed, receipt):
    return cli._unreceipted_retirement_ownership_issues(
        {"applied_files": [{"path": claimed}]},
        manifest_label="retiring.json",
        inventory={
            Path(".axiom/encoding-manifests/other.json"): json.dumps(receipt).encode()
        },
        location="at HEAD",
    )


_RETIREMENT_SHAPES = {
    "receipted": cli._RETIRE_APPLY_MANIFEST_FIELDS,
    "unreceipted": cli._UNRECEIPTED_RETIRE_APPLY_MANIFEST_FIELDS,
}


def _retirement_shaped(shape, **values):
    return {field: values.get(field, "") for field in _RETIREMENT_SHAPES[shape]}


def test_reason_naming_another_pair_leaves_that_pair_retirable(tmp_path):
    checkout = _repo(tmp_path)
    second = "us/policies/council/second.yaml"
    second_companion = "us/policies/council/second.test.yaml"
    for path, raw in zip((second, second_companion), checkout.originals.values()):
        (checkout.repo / path).write_bytes(raw)
    checkout.base = checkout.commit("second unreceipted pair")

    checkout.retire(reason=second)
    checkout.commit("retire the first pair under a reason that names the second")
    checkout.retire(paths=[second])

    assert not (checkout.repo / second).exists()
    assert not (checkout.repo / second_companion).exists()


@pytest.mark.parametrize("retire_owner", [False, True], ids=["live", "retired"])
@pytest.mark.parametrize("backend", sorted(cli.APPLIED_ENCODING_ENCODER_BACKENDS))
def test_model_run_id_naming_independent_pair_does_not_create_ownership(
    tmp_path, retire_owner, backend
):
    """Production refresh and retirement writers preserve descriptive run IDs."""

    checkout = _repo(tmp_path)
    owner = checkout.repo / "us/statutes/26/1.yaml"
    owner_companion = owner.with_name("1.test.yaml")
    owner_companion.write_text("[]\n")
    owner_manifest = checkout.repo / cli._applied_encoding_manifest_path(
        owner.relative_to(checkout.repo)
    )
    historical = json.loads(owner_manifest.read_text())
    historical["applied_files"].append(
        {
            "path": owner_companion.relative_to(checkout.repo).as_posix(),
            "sha256": _sha256(owner_companion.read_bytes()),
        }
    )
    cli._sign_applied_encoding_manifest(historical, fixtures.TEST_APPLY_SIGNING_BROKER)
    owner_manifest.write_text(json.dumps(historical) + "\n")

    # Keep an authenticated live model for --all after both requested deletions.
    sentinel = owner.with_name("sentinel.yaml")
    sentinel.write_bytes(owner.read_bytes())
    sentinel_manifest = checkout.repo / cli._applied_encoding_manifest_path(
        sentinel.relative_to(checkout.repo)
    )
    sentinel_payload = copy.deepcopy(historical)
    sentinel_payload["applied_files"] = [
        {
            "path": sentinel.relative_to(checkout.repo).as_posix(),
            "sha256": _sha256(sentinel.read_bytes()),
        }
    ]
    cli._sign_applied_encoding_manifest(
        sentinel_payload, fixtures.TEST_APPLY_SIGNING_BROKER
    )
    sentinel_manifest.write_text(json.dumps(sentinel_payload) + "\n")
    checkout.base = checkout.commit("prepare independent owner and live sentinel")

    output_root = tmp_path / "refresh-output"
    output_root.mkdir()
    generated = output_root / "1.yaml"
    generated.write_bytes(owner.read_bytes())
    source_text = output_root / "source.txt"
    source_text.write_text(
        fixtures.resolve_corpus_source_unit("us/statute/26/1", checkout.release).body,
        encoding="utf-8",
    )
    context_manifest = output_root / "context-manifest.json"
    context_manifest.write_text(
        json.dumps(
            {
                "source_text_file": source_text.name,
                "source_metadata": {
                    "source_attestation": historical["source_attestation"]
                },
            }
        )
        + "\n"
    )
    result = SimpleNamespace(
        output_file=str(generated),
        runner=backend,
        backend=backend,
        model="fixture-refresh-model",
        tool=cli.APPLIED_ENCODING_MODEL_TOOL,
        citation="us/statute/26/1",
        generation_prompt_sha256=None,
        trace_file=None,
        context_manifest_file=str(context_manifest),
        source_attestation=historical["source_attestation"],
    )
    if backend == "codex":
        result.codex_cli_version = "codex-cli fixture"
        result.codex_cli_sha256 = "c" * 64
    setattr(result, cli._MANIFEST_ONLY_REFRESH_ATTR, True)
    setattr(
        result,
        cli._APPLY_VALIDATION_SNAPSHOT_ATTR,
        {"manifest_validation_execution": historical["validation_execution"]},
    )
    refreshed_manifest = cli._write_applied_encoding_manifest(
        result,
        output_root=output_root,
        policy_repo_path=checkout.repo / "us",
        corpus_path=checkout.corpus_path,
        relative_output=Path("statutes/26/1.yaml"),
        applied_files=[owner, owner_companion],
        run_id=checkout.primary,
        signing_broker=fixtures.TEST_APPLY_SIGNING_BROKER,
        axiom_encode_git=_encoder_provenance(),
    )
    assert refreshed_manifest == owner_manifest
    # The target repository has no protected encoder_verifier = "main", so
    # the writer keeps the repository waiver-set digest (#2092).
    expected_fields = cli._MODEL_APPLY_MANIFEST_FIELDS
    if backend == "codex":
        expected_fields |= {"codex_cli_version", "codex_cli_sha256"}
    assert set(json.loads(owner_manifest.read_text())) == expected_fields
    assert checkout.guard() == []
    assert (
        cli._load_verified_applied_encoding_manifest_payload(
            checkout.repo,
            owner_manifest.relative_to(checkout.repo).as_posix(),
            historical_verification=True,
            local_corpus_release=checkout.release,
            expected_encoder_identity=fixtures.TEST_PINNED_ENCODER_IDENTITY,
        )[3]
        == []
    )
    checkout.commit("refresh live owner with independent pair as run ID")

    if retire_owner:
        checkout.retire(
            paths=[owner.relative_to(checkout.repo).as_posix()],
            unreceipted=False,
        )
        checkout.commit("retire owner while preserving its signed model")
        owner_payload = json.loads(owner_manifest.read_text())
        assert owner_payload["retired_manifest"]["run_id"] == checkout.primary
        assert checkout.guard() == []
        assert (
            cli._load_verified_applied_encoding_manifest_payload(
                checkout.repo,
                owner_manifest.relative_to(checkout.repo).as_posix(),
                historical_verification=True,
                local_corpus_release=checkout.release,
                expected_encoder_identity=fixtures.TEST_PINNED_ENCODER_IDENTITY,
            )[3]
            == []
        )
    else:
        assert json.loads(owner_manifest.read_text())["run_id"] == checkout.primary

    parent = fixtures._git(checkout.repo, "rev-parse", "HEAD").stdout.strip()
    checkout.retire()
    final_receipt = json.loads(checkout.manifest.read_text())
    assert set(final_receipt) == cli._UNRECEIPTED_RETIRE_APPLY_MANIFEST_FIELDS
    assert final_receipt["schema_version"] == cli.APPLIED_ENCODING_MANIFEST_SCHEMA
    assert final_receipt["tool"] == cli.APPLIED_ENCODING_UNRECEIPTED_RETIRE_TOOL
    assert final_receipt["base_commit"] == parent
    assert final_receipt["applied_files"] == [
        {"path": path, "deleted": True, "sha256": _sha256(raw)}
        for path, raw in checkout.originals.items()
    ]
    assert (
        cli._applied_encoding_manifest_signature_issue(
            final_receipt, fixtures.TEST_APPLY_SIGNING_BROKER
        )
        is None
    )
    checkout.commit("retire independent unreceipted pair")
    assert not (checkout.repo / checkout.primary).exists()
    assert not (checkout.repo / checkout.companion).exists()
    assert checkout.guard(base=parent) == []
    assert checkout.guard(all_files=True, base=parent) == []
    assert (
        cli.guard_generated_change_issues(
            checkout.repo, corpus_path=checkout.corpus_path, all_files=True
        )
        == []
    )
    assert checkout.load()[3] == []


@pytest.mark.parametrize("shape", sorted(_RETIREMENT_SHAPES))
def test_retirement_receipt_reason_owns_nothing(shape):
    claimed = "us/policies/council/second.yaml"
    assert (
        _other_receipt_issues(claimed, _retirement_shaped(shape, reason=claimed)) == []
    )


@pytest.mark.parametrize("shape", sorted(_RETIREMENT_SHAPES))
@pytest.mark.parametrize(
    "values",
    [
        {"applied_files": [{"path": "us/policies/council/second.yaml"}]},
        {"tool": "us/policies/council/second.yaml"},
        {"reason": ["us/policies/council/second.yaml"]},
        {"reason": {"covered": "us/policies/council/second.yaml"}},
    ],
    ids=["applied-files", "other-string-field", "reason-list", "reason-object"],
)
def test_retirement_shape_still_owns_outside_a_text_reason(shape, values):
    issues = _other_receipt_issues(
        "us/policies/council/second.yaml", _retirement_shaped(shape, **values)
    )
    assert any("already covered" in issue for issue in issues), issues


@pytest.mark.parametrize(
    "receipt",
    [
        {"reason": "us/policies/council/second.yaml"},
        {
            **_retirement_shaped(
                "unreceipted", reason="us/policies/council/second.yaml"
            ),
            "unknown": "",
        },
    ],
    ids=["reason-only", "retirement-shape-plus-a-field"],
)
def test_reason_in_an_unknown_shape_still_owns(receipt):
    issues = _other_receipt_issues("us/policies/council/second.yaml", receipt)
    assert any("already covered" in issue for issue in issues), issues


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    shape=st.sampled_from(sorted(_RETIREMENT_SHAPES)),
    reason=st.text(max_size=60),
    owned=st.booleans(),
)
@example(shape="unreceipted", reason="us/policies/council/second.yaml", owned=False)
@example(shape="receipted", reason="us/policies/council/second.yaml", owned=True)
def test_invariant_retirement_reason_never_changes_ownership(shape, reason, owned):
    claimed = "us/policies/council/second.yaml"
    entries = [{"path": claimed}] if owned else []
    assert _other_receipt_issues(
        claimed, _retirement_shaped(shape, applied_files=entries, reason=reason)
    ) == _other_receipt_issues(
        claimed, _retirement_shaped(shape, applied_files=entries, reason="unrelated")
    )


def test_no_base_all_snapshot_does_not_replay_settled_deletion(tmp_path):
    checkout = _repo(tmp_path)
    checkout.retire()
    checkout.commit()
    for path, raw in checkout.originals.items():
        (checkout.repo / path).write_bytes(raw + b"# recreated after retirement\n")
    recreated_base = checkout.commit("restore retired pair")
    for path in checkout.originals:
        (checkout.repo / path).unlink()
    checkout.commit("settled second deletion under unchanged receipt")
    assert (
        cli.guard_generated_change_issues(
            checkout.repo, corpus_path=checkout.corpus_path, all_files=True
        )
        == []
    )
    issues = checkout.guard(all_files=True, base=recreated_base)
    assert any("base sha256 does not match" in issue for issue in issues), issues


# These expectations are deliberately independent of the production classes: a
# classification mutation must change behavior, rather than its own test cases.
_SCAN_SCHEMA_NAMES = (
    "_MODEL_APPLY_MANIFEST_FIELDS",
    "_MODEL_CODEX_APPLY_MANIFEST_FIELDS",
    "_REVIEWED_CANDIDATE_MANIFEST_FIELDS",
    "_REVIEWED_REPAIR_MANIFEST_FIELDS",
    "_RETIRE_APPLY_MANIFEST_FIELDS",
    "_UNRECEIPTED_RETIRE_APPLY_MANIFEST_FIELDS",
    "_PATH_MIGRATION_APPLY_MANIFEST_FIELDS",
    "_LEGACY_REPLACEMENT_APPLY_MANIFEST_FIELDS",
    "_LEGACY_EXACT_DEPENDENT_APPLY_MANIFEST_FIELDS",
    "_LEGACY_RETAINED_SUCCESSOR_APPLY_MANIFEST_FIELDS",
    "_RETIRED_SOURCE_METADATA_MANIFEST_FIELDS",
    "_LEGACY_REPLACEMENT_RECEIPT_FIELDS",
    "_MODEL_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS",
    "_MODEL_CODEX_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS",
    "_REVIEWED_CANDIDATE_PER_FILE_WAIVER_MANIFEST_FIELDS",
    "_MODEL_APPLY_REVIEW_RECORD_MANIFEST_FIELDS",
    "_MODEL_CODEX_APPLY_REVIEW_RECORD_MANIFEST_FIELDS",
)
# Independent audited sets: adding a field requires an explicit classification.
_EXPECTED_COMMON_SCANNED = frozenset(
    {
        "schema_version",
        "generated_at",
        "tool",
        "axiom_encode_version",
        "axiom_encode_git",
        "validation_waiver_set_sha256",
        "signature",
    }
)
_EXPECTED_MANIFEST_CLASSES = {
    "_MODEL_APPLY_MANIFEST_FIELDS": (
        frozenset({"run_id", "runner", "model"}),
        _EXPECTED_COMMON_SCANNED
        | {
            "generation_prompt_sha256",
            "citation",
            "backend",
            "generated_output_root",
            "generated_output_file",
            "generated_output_sha256",
            "trace_file",
            "trace_sha256",
            "context_manifest_file",
            "context_manifest_sha256",
            "applied_files",
            "source_attestation",
            "validation_execution",
        },
    ),
    "_REVIEWED_CANDIDATE_MANIFEST_FIELDS": (
        frozenset({"run_id"}),
        _EXPECTED_COMMON_SCANNED
        | {
            "citation",
            "reviewed_rulespec_ref",
            "applied_files",
            "source_attestation",
            "validation_execution",
        },
    ),
    "_REVIEWED_REPAIR_MANIFEST_FIELDS": (
        frozenset({"run_id"}),
        _EXPECTED_COMMON_SCANNED
        | {
            "citation",
            "reviewed_repair",
            "applied_files",
            "source_attestation",
            "validation_execution",
        },
    ),
    "_RETIRE_APPLY_MANIFEST_FIELDS": (
        frozenset({"reason"}),
        _EXPECTED_COMMON_SCANNED
        | {"applied_files", "retired_manifest", "source_attestation"},
    ),
    "_UNRECEIPTED_RETIRE_APPLY_MANIFEST_FIELDS": (
        frozenset({"reason"}),
        _EXPECTED_COMMON_SCANNED | {"base_commit", "applied_files"},
    ),
    "_PATH_MIGRATION_APPLY_MANIFEST_FIELDS": (
        frozenset(),
        _EXPECTED_COMMON_SCANNED
        | {"applied_files", "migrated_manifest", "migration", "source_attestation"},
    ),
    "_LEGACY_REPLACEMENT_APPLY_MANIFEST_FIELDS": (
        frozenset(),
        _EXPECTED_COMMON_SCANNED
        | {
            "applied_files",
            "replacement_manifest",
            "replacement",
            "source_attestation",
        },
    ),
    "_LEGACY_EXACT_DEPENDENT_APPLY_MANIFEST_FIELDS": (
        frozenset(),
        _EXPECTED_COMMON_SCANNED | {"applied_files", "legacy_migration"},
    ),
    "_LEGACY_RETAINED_SUCCESSOR_APPLY_MANIFEST_FIELDS": (
        frozenset(),
        _EXPECTED_COMMON_SCANNED
        | {"applied_files", "legacy_migration", "retained_successor_manifest"},
    ),
    "_RETIRED_SOURCE_METADATA_MANIFEST_FIELDS": (
        frozenset(),
        _EXPECTED_COMMON_SCANNED
        | {"applied_files", "retired_source_metadata", "prior_manifest"},
    ),
    "_LEGACY_REPLACEMENT_RECEIPT_FIELDS": (
        frozenset(),
        _EXPECTED_COMMON_SCANNED
        | {
            "repository",
            "corpus_release",
            "validation_execution",
            "legacy",
            "replacement",
            "replacement_manifest",
        },
    ),
}
_EXPECTED_MANIFEST_CLASSES["_MODEL_CODEX_APPLY_MANIFEST_FIELDS"] = (
    _EXPECTED_MANIFEST_CLASSES["_MODEL_APPLY_MANIFEST_FIELDS"][0],
    _EXPECTED_MANIFEST_CLASSES["_MODEL_APPLY_MANIFEST_FIELDS"][1]
    | {"codex_cli_version", "codex_cli_sha256"},
)
# Per-file waiver variants (#2092): the waiver bindings replace the set
# digest and are scanned, because they name covered paths.
for _set_name, _per_file_name in (
    ("_MODEL_APPLY_MANIFEST_FIELDS", "_MODEL_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS"),
    (
        "_MODEL_CODEX_APPLY_MANIFEST_FIELDS",
        "_MODEL_CODEX_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS",
    ),
    (
        "_REVIEWED_CANDIDATE_MANIFEST_FIELDS",
        "_REVIEWED_CANDIDATE_PER_FILE_WAIVER_MANIFEST_FIELDS",
    ),
):
    _EXPECTED_MANIFEST_CLASSES[_per_file_name] = (
        _EXPECTED_MANIFEST_CLASSES[_set_name][0],
        (_EXPECTED_MANIFEST_CLASSES[_set_name][1] - {"validation_waiver_set_sha256"})
        | {"validation_waiver_bindings"},
    )
# Generalist review record variants (#2146): the per-file model shapes plus the
# record, which is scanned. It holds digests and enumerated words only.
for _per_file_name, _review_name in (
    (
        "_MODEL_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS",
        "_MODEL_APPLY_REVIEW_RECORD_MANIFEST_FIELDS",
    ),
    (
        "_MODEL_CODEX_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS",
        "_MODEL_CODEX_APPLY_REVIEW_RECORD_MANIFEST_FIELDS",
    ),
):
    _EXPECTED_MANIFEST_CLASSES[_review_name] = (
        _EXPECTED_MANIFEST_CLASSES[_per_file_name][0],
        _EXPECTED_MANIFEST_CLASSES[_per_file_name][1] | {"generalist_review"},
    )
_EXPECTED_DESCRIPTIVE_FIELDS = frozenset({"reason", "run_id", "runner", "model"})
_SCAN_SCHEMAS = {name: getattr(cli, name) for name in _SCAN_SCHEMA_NAMES}
_SCAN_PATH = "us/policies/council/second.yaml"
_SCAN_NESTED_POSITIONS = (
    ("_RETIRE_APPLY_MANIFEST_FIELDS", "retired_manifest"),
    ("_PATH_MIGRATION_APPLY_MANIFEST_FIELDS", "migrated_manifest"),
    ("_LEGACY_REPLACEMENT_APPLY_MANIFEST_FIELDS", "replacement_manifest"),
    ("_LEGACY_REPLACEMENT_RECEIPT_FIELDS", "replacement_manifest"),
    ("_LEGACY_RETAINED_SUCCESSOR_APPLY_MANIFEST_FIELDS", "retained_successor_manifest"),
    ("_RETIRED_SOURCE_METADATA_MANIFEST_FIELDS", "prior_manifest"),
)


def _scan_shaped(schema_name, **values):
    return {field: values.get(field, "") for field in _SCAN_SCHEMAS[schema_name]}


def _scan_nested(inner, position, depth):
    schema_name, field = position
    for _ in range(depth):
        value = (
            {"path": "old-owner.json", "sha256": "0" * 64, "manifest": inner}
            if field == "prior_manifest"
            else inner
        )
        inner = _scan_shaped(schema_name, **{field: value})
    return inner


def _scan_strings(receipt):
    strings, nonfinite_count = cli._unreceipted_retirement_manifest_strings(receipt)
    assert nonfinite_count == 0
    return strings


def _assert_scan_ownership(receipt, *, covered, context):
    assert (_SCAN_PATH in _scan_strings(receipt)) is covered, context
    issues = _other_receipt_issues(_SCAN_PATH, receipt)
    assert bool(issues) is covered, (context, issues)
    if covered:
        assert any("already covered" in issue for issue in issues), (context, issues)


def test_manifest_scan_classes_cover_each_schema_and_all_schema_constants():
    discovered = {
        name
        for name, value in vars(cli).items()
        if name.endswith("_MANIFEST_FIELDS") and isinstance(value, frozenset)
    }
    recognized = set(_SCAN_SCHEMA_NAMES)
    assert discovered <= recognized
    classes = cli._UNRECEIPTED_RETIREMENT_MANIFEST_FIELD_CLASSES
    assert set(classes) == set(_SCAN_SCHEMAS.values())
    admitted = set().union(*map(set, cli._APPLIED_MANIFEST_SCHEMA_VARIANTS.values()))
    assert set(classes) == admitted | {cli._LEGACY_REPLACEMENT_RECEIPT_FIELDS}
    for name, schema in _SCAN_SCHEMAS.items():
        descriptive, scanned = classes[schema]
        assert not descriptive & scanned, name
        assert descriptive | scanned == schema, name
        expected_descriptive, expected_scanned = _EXPECTED_MANIFEST_CLASSES[name]
        assert expected_descriptive | expected_scanned == schema, name
        assert descriptive == expected_descriptive, name
        assert scanned == expected_scanned, name


_SCAN_ADMISSION_TOOLS = {
    "_REVIEWED_CANDIDATE_MANIFEST_FIELDS": cli.APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL,
    "_REVIEWED_REPAIR_MANIFEST_FIELDS": cli.APPLIED_ENCODING_REVIEWED_REPAIR_TOOL,
    "_RETIRE_APPLY_MANIFEST_FIELDS": cli.APPLIED_ENCODING_RETIRE_TOOL,
    "_UNRECEIPTED_RETIRE_APPLY_MANIFEST_FIELDS": cli.APPLIED_ENCODING_UNRECEIPTED_RETIRE_TOOL,
    "_PATH_MIGRATION_APPLY_MANIFEST_FIELDS": cli.APPLIED_ENCODING_PATH_MIGRATION_TOOL,
    "_LEGACY_REPLACEMENT_APPLY_MANIFEST_FIELDS": cli.APPLIED_ENCODING_LEGACY_REPLACEMENT_TOOL,
    "_LEGACY_EXACT_DEPENDENT_APPLY_MANIFEST_FIELDS": cli.APPLIED_ENCODING_LEGACY_EXACT_DEPENDENT_TOOL,
    "_LEGACY_RETAINED_SUCCESSOR_APPLY_MANIFEST_FIELDS": cli.APPLIED_ENCODING_LEGACY_RETAINED_SUCCESSOR_TOOL,
    "_RETIRED_SOURCE_METADATA_MANIFEST_FIELDS": cli.RETIRED_SOURCE_METADATA_TOOL,
    "_REVIEWED_CANDIDATE_PER_FILE_WAIVER_MANIFEST_FIELDS": (
        cli.APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL
    ),
}


def _admission_shape_issues(keys, schema_name, backend=None):
    """Invoke real admission, separating field-set errors from value errors."""
    payload = dict.fromkeys(keys, "")
    if "backend" in keys:
        payload["backend"] = backend
    if schema_name == "_LEGACY_REPLACEMENT_RECEIPT_FIELDS":
        payload["tool"] = cli.APPLIED_ENCODING_LEGACY_REPLACEMENT_TOOL
        payload["schema_version"] = (
            cli.APPLIED_ENCODING_LEGACY_REPLACEMENT_RECEIPT_SCHEMA
        )
        issues = cli._guard_auxiliary_receipt_authentication_issues(
            payload,
            receipt_path=(
                cli.APPLIED_ENCODING_LEGACY_REPLACEMENT_RECEIPT_DIR / "test.json"
            ).as_posix(),
            signing_broker=fixtures.TEST_APPLY_SIGNING_BROKER,
        )
    else:
        if "backend" in keys:
            payload["tool"] = cli.APPLIED_ENCODING_MODEL_TOOL
        else:
            payload["tool"] = _SCAN_ADMISSION_TOOLS[schema_name]
        issues = cli._applied_manifest_exact_schema_issues(
            payload, manifest_label="shape-probe.json"
        )
    return [issue for issue in issues if "does not match the exact" in issue]


def _admission_field_probes():
    """Include admission-local extensions, not just module schema constants.

    Literal extension sets are probed together and separately, so a newly
    admitted optional pair is detected even when neither singleton is admitted.
    """
    fields = set().union(*_SCAN_SCHEMAS.values()) | {"unclassified_optional_field"}
    groups = []
    for function in (
        cli._applied_manifest_exact_schema_issues,
        cli._guard_auxiliary_receipt_authentication_issues,
    ):
        tree = ast.parse(inspect.getsource(function))
        fields.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.isidentifier()
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Set):
                literal = {
                    item.value
                    for item in node.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                }
                if literal:
                    groups.append(literal)
            elif isinstance(node, ast.Name):
                value = vars(cli).get(node.id)
                if isinstance(value, (set, frozenset)) and all(
                    isinstance(item, str) for item in value
                ):
                    fields.update(value)
                    groups.append(value)
    probes = {frozenset({field}) for field in fields}
    for group in groups:
        probes.add(frozenset(group))
        if len(group) <= 8:
            for size in range(2, len(group) + 1):
                probes.update(
                    map(frozenset, itertools.combinations(sorted(group), size))
                )
    return probes


@pytest.mark.parametrize("schema_name", _SCAN_SCHEMA_NAMES)
def test_manifest_scan_recognition_matches_direct_schema_admission(schema_name):
    schema = _SCAN_SCHEMAS[schema_name]
    backends = (
        sorted(cli.APPLIED_ENCODING_ENCODER_BACKENDS) if "backend" in schema else [None]
    )
    probes = {schema}
    probes.update(schema | extension for extension in _admission_field_probes())
    probes.update(
        schema - {field} for field in schema - {"backend", "tool", "schema_version"}
    )
    recognized = cli._UNRECEIPTED_RETIREMENT_MANIFEST_FIELD_CLASSES
    for keys in probes:
        # Recognition concerns keys alone. A candidate can be admitted under a
        # different contract: e.g. removing retained_successor_manifest produces
        # the exact-dependent schema. Exercise admission for every selector.
        admitted = any(
            not _admission_shape_issues(keys, name, backend)
            for name, fields in _SCAN_SCHEMAS.items()
            if ("backend" in fields) == ("backend" in keys)
            for backend in (
                sorted(cli.APPLIED_ENCODING_ENCODER_BACKENDS)
                if "backend" in fields
                else [None]
            )
        )
        assert admitted == (keys in recognized), (schema_name, sorted(keys))
    for backend in backends:
        assert (
            not _admission_shape_issues(schema, schema_name, backend)
            or (
                schema_name
                in {
                    "_MODEL_CODEX_APPLY_MANIFEST_FIELDS",
                    "_MODEL_CODEX_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS",
                    "_MODEL_CODEX_APPLY_REVIEW_RECORD_MANIFEST_FIELDS",
                }
                and backend != "codex"
            )
            or (
                # A per-file Codex manifest must carry CLI provenance; only
                # historical set-digest Codex receipts may omit it.
                schema_name
                in {
                    "_MODEL_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS",
                    "_MODEL_APPLY_REVIEW_RECORD_MANIFEST_FIELDS",
                }
                and backend == "codex"
            )
        )


def test_manifest_scan_covers_guard_receipt_directories():
    # Discover named directories independently, including new receipt families.
    directories = {
        Path(value)
        for name, value in vars(cli).items()
        if name.endswith(("_RECEIPT_DIR", "_MANIFEST_DIR"))
        and isinstance(value, (str, Path))
        and Path(value).parts[:1] == (".axiom",)
    }
    assert directories
    for directory in directories:
        assert cli._provenance_inventory_prefix(
            directory / "probe.json"
        ) == directory.parts or directory == Path(
            cli.RETIRED_SOURCE_METADATA_RECEIPT_DIR
        ), directory
        for prefix in (Path(), Path("us")):
            receipt = prefix / directory / "new-family" / "probe.json"
            assert (
                cli._unreceipted_retirement_inventory_prefix(receipt)
                == (prefix / directory).parts
            ), receipt


@pytest.mark.parametrize(
    ("schema_name", "field"),
    [
        (name, field)
        for name, (descriptive, _scanned) in _EXPECTED_MANIFEST_CLASSES.items()
        for field in sorted(descriptive)
    ],
)
def test_manifest_scan_descriptive_strings_do_not_own(schema_name, field):
    receipt = _scan_shaped(schema_name, **{field: _SCAN_PATH})
    _assert_scan_ownership(receipt, covered=False, context=(schema_name, field))
    for position in _SCAN_NESTED_POSITIONS:
        for depth in (1, 2):
            nested = _scan_nested(receipt, position, depth)
            _assert_scan_ownership(
                nested, covered=False, context=(schema_name, field, position, depth)
            )


@pytest.mark.parametrize(
    ("schema_name", "field"),
    [
        (name, field)
        for name, (_descriptive, scanned) in _EXPECTED_MANIFEST_CLASSES.items()
        for field in sorted(scanned)
    ],
)
@pytest.mark.parametrize("form", ["string", "list", "dict"])
def test_manifest_scan_every_other_field_still_owns(schema_name, field, form):
    value = {
        "string": _SCAN_PATH,
        "list": [[_SCAN_PATH]],
        "dict": {"nested": {"covered": _SCAN_PATH}},
    }[form]
    receipt = _scan_shaped(schema_name, **{field: value})
    _assert_scan_ownership(receipt, covered=True, context=(schema_name, field, form))
    for position in _SCAN_NESTED_POSITIONS:
        for depth in (1, 2):
            nested = _scan_nested(receipt, position, depth)
            _assert_scan_ownership(
                nested,
                covered=True,
                context=(schema_name, field, form, position, depth),
            )


@pytest.mark.parametrize(
    ("schema_name", "field"),
    [
        (name, field)
        for name, (descriptive, _scanned) in _EXPECTED_MANIFEST_CLASSES.items()
        for field in sorted(descriptive)
    ],
)
@pytest.mark.parametrize("form", ["list", "dict"])
def test_manifest_scan_descriptive_containers_still_own(schema_name, field, form):
    value = [_SCAN_PATH] if form == "list" else {"covered": _SCAN_PATH}
    receipt = _scan_shaped(schema_name, **{field: value})
    _assert_scan_ownership(receipt, covered=True, context=(schema_name, field, form))
    for position in _SCAN_NESTED_POSITIONS:
        for depth in (1, 2):
            _assert_scan_ownership(
                _scan_nested(receipt, position, depth),
                covered=True,
                context=(schema_name, field, form, position, depth),
            )


@pytest.mark.parametrize("key_change", ["added", "removed"])
def test_manifest_scan_nested_model_requires_exact_keys(key_change):
    receipt = _scan_shaped("_MODEL_APPLY_MANIFEST_FIELDS", run_id=_SCAN_PATH)
    if key_change == "added":
        receipt["unknown"] = ""
    else:
        del receipt["generated_at"]
    for position in _SCAN_NESTED_POSITIONS:
        for depth in (1, 2):
            _assert_scan_ownership(
                _scan_nested(receipt, position, depth),
                covered=True,
                context=(key_change, position, depth),
            )


@pytest.mark.parametrize("envelope", ["source-attestation", "unknown-top", "list"])
@pytest.mark.parametrize(
    "schema_name",
    ["_MODEL_APPLY_MANIFEST_FIELDS", "_MODEL_CODEX_APPLY_MANIFEST_FIELDS"],
)
def test_manifest_scan_schema_keys_at_other_positions_still_own(envelope, schema_name):
    nested = _scan_shaped(schema_name, run_id=_SCAN_PATH)
    receipt = {
        "source-attestation": _scan_shaped(
            "_MODEL_APPLY_MANIFEST_FIELDS", source_attestation=nested
        ),
        "unknown-top": {"unknown": nested},
        "list": _scan_shaped(
            "_RETIRE_APPLY_MANIFEST_FIELDS", retired_manifest=[nested]
        ),
    }[envelope]
    _assert_scan_ownership(receipt, covered=True, context=envelope)


@pytest.mark.parametrize("key_change", ["added", "removed"])
def test_manifest_scan_prior_manifest_wrapper_requires_exact_keys(key_change):
    nested = _scan_shaped("_MODEL_APPLY_MANIFEST_FIELDS", run_id=_SCAN_PATH)
    wrapper = {"path": "old-owner.json", "sha256": "0" * 64, "manifest": nested}
    if key_change == "added":
        wrapper["unknown"] = ""
    else:
        del wrapper["path"]
    receipt = _scan_shaped(
        "_RETIRED_SOURCE_METADATA_MANIFEST_FIELDS", prior_manifest=wrapper
    )
    _assert_scan_ownership(receipt, covered=True, context=key_change)


def test_manifest_scan_mapping_applied_files_keys_still_own():
    receipt = _scan_shaped(
        "_MODEL_APPLY_MANIFEST_FIELDS", applied_files={_SCAN_PATH: "0" * 64}
    )
    _assert_scan_ownership(receipt, covered=True, context="mapping-applied-files")
    for position in _SCAN_NESTED_POSITIONS:
        _assert_scan_ownership(
            _scan_nested(receipt, position, 2),
            covered=True,
            context=("mapping-applied-files", position),
        )


def test_manifest_scan_is_iterative_for_deep_values():
    value = _SCAN_PATH
    for _ in range(2000):
        value = {"nested": [value]}
    strings, nonfinite_count = cli._unreceipted_retirement_manifest_strings(
        {"unknown": value}
    )
    assert strings == {_SCAN_PATH}
    assert nonfinite_count == 0


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_manifest_scan_keeps_nonfinite_numbers(number):
    receipt = _scan_shaped(
        "_MODEL_APPLY_MANIFEST_FIELDS", run_id=[number, {"nested": number}]
    )
    strings, nonfinite_count = cli._unreceipted_retirement_manifest_strings(receipt)
    assert strings == {""}
    assert nonfinite_count == 2
    list_only = _scan_shaped(
        "_MODEL_APPLY_MANIFEST_FIELDS", run_id=[[number], [number]]
    )
    assert any(
        "nonfinite" in issue for issue in _other_receipt_issues(_SCAN_PATH, list_only)
    )


def _reference_manifest_strings(receipt):
    """The previous conservative scanner, including historical mapping keys."""
    pending = [receipt]
    strings = set()
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key == "applied_files" and isinstance(value, dict):
                    strings.update(value)
                pending.append(value)
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, str):
            strings.add(item)
    return strings


def _recognized_descriptive_strings(receipt):
    """Independent positional specification of the permitted exclusions."""
    pending = [(receipt, True)]
    excluded = set()
    while pending:
        item, recognized_position = pending.pop()
        if not isinstance(item, dict):
            continue
        schema = frozenset(item)
        if not recognized_position or schema not in _SCAN_SCHEMAS.values():
            continue
        for field in schema & _EXPECTED_DESCRIPTIVE_FIELDS:
            if isinstance(item[field], str):
                excluded.add(item[field])
        for _name, field in _SCAN_NESTED_POSITIONS:
            nested = item.get(field)
            if field == "prior_manifest":
                if isinstance(nested, dict) and set(nested) == {
                    "path",
                    "sha256",
                    "manifest",
                }:
                    pending.append((nested["manifest"], True))
            else:
                pending.append((nested, True))
    return excluded


_SCAN_JSON_VALUES = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.sampled_from([_SCAN_PATH, "us/statutes/26/1.yaml", "", "ordinary text"]),
        st.text(max_size=30),
    ),
    lambda child: st.one_of(
        st.lists(child, max_size=3),
        st.dictionaries(
            st.one_of(st.text(max_size=12), st.just("applied_files")),
            child,
            max_size=3,
        ),
    ),
    max_leaves=15,
)


@st.composite
def _manifest_scan_examples(draw):
    schema_name = draw(st.sampled_from(_SCAN_SCHEMA_NAMES))
    fields = sorted(_SCAN_SCHEMAS[schema_name])
    values = draw(
        st.lists(_SCAN_JSON_VALUES, min_size=len(fields), max_size=len(fields))
    )
    receipt = dict(zip(fields, values))
    for _ in range(draw(st.integers(min_value=0, max_value=2))):
        receipt = _scan_nested(
            receipt, draw(st.sampled_from(_SCAN_NESTED_POSITIONS)), 1
        )
    return receipt


@settings(max_examples=70, deadline=None)
@given(receipt=_manifest_scan_examples())
@example(receipt=_scan_shaped("_MODEL_APPLY_MANIFEST_FIELDS", run_id=_SCAN_PATH))
@example(receipt=_scan_shaped("_MODEL_CODEX_APPLY_MANIFEST_FIELDS", run_id=_SCAN_PATH))
@example(
    receipt=_scan_nested(
        _scan_shaped("_RETIRE_APPLY_MANIFEST_FIELDS", reason=_SCAN_PATH),
        _SCAN_NESTED_POSITIONS[-1],
        2,
    )
)
def test_manifest_scan_only_removes_recognized_descriptive_strings(receipt):
    conservative = _reference_manifest_strings(receipt)
    actual = _scan_strings(receipt)
    assert actual <= conservative
    assert conservative - actual <= _recognized_descriptive_strings(receipt)
    unknown = {"unknown": receipt}
    assert _scan_strings(unknown) == _reference_manifest_strings(unknown)


@pytest.mark.parametrize("field", ["path", "sha256"])
def test_manifest_scan_prior_manifest_wrapper_provenance_stays_scanned(field):
    wrapper = {
        "path": "old-owner.json",
        "sha256": "0" * 64,
        "manifest": _scan_shaped("_MODEL_APPLY_MANIFEST_FIELDS", run_id=_SCAN_PATH),
    }
    wrapper[field] = _SCAN_PATH
    receipt = _scan_shaped(
        "_RETIRED_SOURCE_METADATA_MANIFEST_FIELDS", prior_manifest=wrapper
    )
    _assert_scan_ownership(receipt, covered=True, context=("prior-provenance", field))


@pytest.mark.parametrize("field", ["root", "rulespec_root"])
@pytest.mark.parametrize("depth", [1, 2])
def test_manifest_scan_caller_chosen_nested_provenance_still_owns(field, depth):
    values = (
        {"axiom_encode_git": {"root": _SCAN_PATH}}
        if field == "root"
        else {
            "validation_execution": {
                "rulespec_dependencies": [
                    {"jurisdiction_roots": [{"rulespec_root": _SCAN_PATH}]}
                ]
            }
        }
    )
    receipt = _scan_shaped("_MODEL_APPLY_MANIFEST_FIELDS", **values)
    _assert_scan_ownership(receipt, covered=True, context=(field, "root"))
    for position in _SCAN_NESTED_POSITIONS:
        _assert_scan_ownership(
            _scan_nested(receipt, position, depth),
            covered=True,
            context=(field, position, depth),
        )
