"""I6: the signed class admits only one whole deterministic migration."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from base64 import b64encode
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import axiom_encode.cli as cli
from axiom_encode import __version__
from axiom_encode.retired_source_metadata import PLAN_SCHEMA, RECEIPT_DIR
from scripts.provision_verification_supervisor import _install_trusted_git_wrapper
from tests.signing_broker_fixtures import SigningBrokerFixture

PRIMARY = "us/statutes/26/25A.yaml"
COMPANION = "us/statutes/26/25A.test.yaml"
IMPORTER = "us/policies/importer.yaml"
WAIVER = "a" * 64
ENCODER = {
    "repository": cli.APPLIED_ENCODING_OFFICIAL_REPOSITORY,
    "commit": "a" * 40,
    "version": __version__,
}
PROVENANCE = {
    "root": "/repo/axiom-encode",
    "commit": ENCODER["commit"],
    "dirty_tracked": False,
    "version": __version__,
    "version_commit": "b" * 40,
    "identity_source": "git",
}
PRIVATE = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
BROKER = SigningBrokerFixture(
    apply_private_key=b64encode(bytes(range(32))).decode(),
    apply_public_key=b64encode(
        PRIVATE.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    ).decode(),
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=cli._rulespec_migration_git_environment(),
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.name", "Migration test")
    git(repo, "config", "user.email", "migration@example.com")
    source = repo / PRIMARY
    source.parent.mkdir(parents=True)
    source.write_text("""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/25A
    values:
      rate: 0.2
rules: []
""")
    (repo / COMPANION).write_text("tests: []\n")
    importer = repo / IMPORTER
    importer.parent.mkdir(parents=True)
    importer.write_text(f"""format: rulespec/v1
rules:
  - name: imported
    metadata:
      proof:
        atoms:
          - kind: import
            import:
              target: us:statutes/26/25A#amount
              hash: sha256:{hashlib.sha256(source.read_bytes()).hexdigest()}
""")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    base = git(repo, "rev-parse", "HEAD")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {"schema_version": PLAN_SCHEMA, "base_commit": base, "modules": [PRIMARY]}
        )
    )
    corpus = tmp_path / "axiom-corpus"
    corpus.mkdir()
    release = SimpleNamespace(
        root=corpus,
        name="test-release",
        content_sha256="b" * 64,
        selector_sha256="c" * 64,
    )
    monkeypatch.setattr(cli, "load_rulespec_local_corpus_release", lambda *_: release)
    monkeypatch.setattr(cli, "load_rulespec_toolchain", lambda *_: SimpleNamespace())
    monkeypatch.setattr(cli, "verify_rulespec_validation_waiver_set", lambda *_: WAIVER)
    monkeypatch.setattr(
        cli, "_require_applied_encoding_manifest_signer", lambda: BROKER
    )
    monkeypatch.setattr(
        cli, "_applied_encoding_manifest_verifier", lambda: PRIVATE.public_key()
    )
    monkeypatch.setattr(
        cli, "_current_guard_encoder_execution_identity", lambda: ENCODER
    )
    monkeypatch.setattr(
        cli, "_read_only_guard_encoder_execution_identity", lambda *_: ENCODER
    )
    monkeypatch.setattr(
        cli, "_require_clean_axiom_encode_git_provenance", lambda: PROVENANCE
    )
    cli._RETIRED_SOURCE_METADATA_REPLAY_CACHE.clear()
    return SimpleNamespace(
        repo=repo,
        base=base,
        args=SimpleNamespace(
            policy_repo_path=repo, corpus_path=corpus, plan=plan, dry_run=False
        ),
    )


def migrate(checkout, capsys):
    cli.cmd_migrate_retired_source_metadata(checkout.args)
    return json.loads(capsys.readouterr().out)


def changed(report):
    return sorted(
        [
            report["receipt_path"],
            *report["manifest_paths"],
            *(item["path"] for item in report["files"]),
        ]
    )


def guard(checkout, paths):
    return cli.guard_generated_change_issues(
        checkout.repo, corpus_path=checkout.args.corpus_path, changed_files=paths
    )


def test_signed_replay_class_and_complete_guard_round_trip(checkout, capsys):
    report = migrate(checkout, capsys)
    assert guard(checkout, changed(report)) == []
    assert len(report["files"]) == 2
    assert len(report["cascade_rewrites"]) == 1
    primary_manifest = json.loads(
        (checkout.repo / cli._applied_encoding_manifest_path(Path(PRIMARY))).read_text()
    )
    assert primary_manifest["axiom_encode_git"] == PROVENANCE
    assert "source_attestation" not in primary_manifest
    assert primary_manifest["applied_files"] == sorted(
        [
            {
                "path": PRIMARY,
                "sha256": hashlib.sha256(
                    (checkout.repo / PRIMARY).read_bytes()
                ).hexdigest(),
            },
            {
                "path": COMPANION,
                "sha256": hashlib.sha256(
                    (checkout.repo / COMPANION).read_bytes()
                ).hexdigest(),
            },
        ],
        key=lambda item: item["path"],
    )
    inventory = cli.retired_source_metadata_change_set(
        checkout.repo, Path(report["receipt_path"])
    )
    assert inventory["changed_paths"] == changed(report)
    assert COMPANION not in inventory["changed_paths"]


def test_signed_cli_runs_through_production_trusted_git_allowlist(
    checkout, capsys, monkeypatch, tmp_path
):
    trusted_git = shutil.which("git")
    assert trusted_git is not None
    tools = tmp_path / "trusted-tools"
    tools.mkdir()
    _install_trusted_git_wrapper(
        tools, Path(sys.executable).resolve(), Path(trusted_git).resolve()
    )
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    report = migrate(checkout, capsys)
    assert guard(checkout, changed(report)) == []


def test_dry_run_never_requests_signing_and_replays(checkout, capsys, monkeypatch):
    checkout.args.dry_run = True

    def forbidden():
        raise AssertionError("dry run requested signer")

    monkeypatch.setattr(cli, "_require_applied_encoding_manifest_signer", forbidden)
    report = migrate(checkout, capsys)
    assert report["dry_run"] is True
    assert report["manifest_paths"] == []
    replay, _ = cli._retired_source_metadata_replay(
        checkout.repo, Path(report["receipt_path"])
    )
    assert all(
        (checkout.repo / item.path).read_bytes() == item.after for item in replay.files
    )
    assert not (checkout.repo / cli.APPLIED_ENCODING_MANIFEST_DIR).exists()


@pytest.mark.parametrize("tamper", ["primary", "cascade", "receipt", "companion"])
def test_I6_postimage_and_receipt_tampering_refused(checkout, capsys, tamper):
    report = migrate(checkout, capsys)
    relative = {
        "primary": PRIMARY,
        "cascade": IMPORTER,
        "receipt": report["receipt_path"],
        "companion": COMPANION,
    }[tamper]
    with (checkout.repo / relative).open("ab") as handle:
        handle.write(b"\n# changed\n")
    assert guard(checkout, changed(report))


@pytest.mark.parametrize(
    "split",
    [
        "receipt_only",
        "without_receipt",
        "without_primary",
        "without_cascade",
        "without_manifest",
        "unrelated",
    ],
)
def test_I6_receipt_and_exact_change_set_cannot_be_split(checkout, capsys, split):
    report = migrate(checkout, capsys)
    paths = changed(report)
    if split == "receipt_only":
        paths = [report["receipt_path"]]
    elif split == "without_receipt":
        paths.remove(report["receipt_path"])
    elif split == "without_primary":
        paths.remove(PRIMARY)
    elif split == "without_cascade":
        paths.remove(IMPORTER)
    elif split == "without_manifest":
        paths.remove(report["manifest_paths"][0])
    else:
        paths.append("README.md")
    assert guard(checkout, paths)


def test_I6_class_cannot_be_claimed_without_receipt(checkout, capsys):
    report = migrate(checkout, capsys)
    manifest_path = report["manifest_paths"][0]
    payload = json.loads((checkout.repo / manifest_path).read_text())
    del payload["retired_source_metadata"]
    cli._sign_applied_encoding_manifest(payload, BROKER)
    (checkout.repo / manifest_path).write_text(json.dumps(payload))
    assert guard(checkout, changed(report))
    assert cli._applied_manifest_exact_schema_issues(
        payload, manifest_label=manifest_path
    )


def test_I6_receipt_digest_and_encoder_pin_required(checkout, capsys):
    report = migrate(checkout, capsys)
    path = report["manifest_paths"][0]
    payload = json.loads((checkout.repo / path).read_text())
    payload["retired_source_metadata"]["receipt_sha256"] = "0" * 64
    cli._sign_applied_encoding_manifest(payload, BROKER)
    (checkout.repo / path).write_text(json.dumps(payload))
    assert guard(checkout, changed(report))
    payload = copy.deepcopy(payload)
    payload["axiom_encode_git"]["commit"] = "f" * 40
    assert cli._applied_manifest_tool_execution_issues(
        payload, manifest_label=path, expected_encoder_identity=ENCODER
    )


def test_I6_stale_receipt_cannot_be_transplanted_to_another_protected_base(
    checkout, capsys
):
    report = migrate(checkout, capsys)
    assert (
        cli.guard_generated_change_issues(
            checkout.repo,
            corpus_path=checkout.args.corpus_path,
            changed_files=changed(report),
            base_ref=checkout.base,
        )
        == []
    )
    (checkout.repo / "README.md").write_text("intervening base commit\n")
    git(checkout.repo, "add", "README.md")
    git(checkout.repo, "commit", "-m", "intervening base")
    new_base = git(checkout.repo, "rev-parse", "HEAD")
    issues = cli.guard_generated_change_issues(
        checkout.repo,
        corpus_path=checkout.args.corpus_path,
        changed_files=changed(report),
        base_ref=new_base,
    )
    assert any(
        "base_commit does not match the protected base" in issue for issue in issues
    )


def test_dry_run_refusal_and_wrong_base_leave_bytes_unchanged(checkout):
    original = (checkout.repo / PRIMARY).read_bytes()
    payload = json.loads(checkout.args.plan.read_text())
    payload["base_commit"] = "f" * 40
    checkout.args.plan.write_text(json.dumps(payload))
    checkout.args.dry_run = True
    with pytest.raises(SystemExit, match="does not match RuleSpec HEAD"):
        cli.cmd_migrate_retired_source_metadata(checkout.args)
    assert (checkout.repo / PRIMARY).read_bytes() == original
    assert not (checkout.repo / RECEIPT_DIR).exists()


def test_install_receipt_allowlist_is_one_digest_json(checkout):
    assert cli._is_canonical_apply_transaction_target(
        checkout.repo, RECEIPT_DIR / ("a" * 64 + ".json")
    )
    for tail in ("history.json", "a" * 64 + ".yaml", "nested/" + "a" * 64 + ".json"):
        assert not cli._is_canonical_apply_transaction_target(
            checkout.repo, RECEIPT_DIR / tail
        )


@pytest.mark.parametrize("owner", ["overlapping", "shrinking"])
def test_migration_refuses_stale_or_shrinking_existing_v5_ownership(checkout, owner):
    path = cli._applied_encoding_manifest_path(
        Path(PRIMARY if owner == "shrinking" else "us/policies/other.yaml")
    )
    target = checkout.repo / path
    target.parent.mkdir(parents=True)
    entries = [
        {
            "path": PRIMARY,
            "sha256": hashlib.sha256(
                (checkout.repo / PRIMARY).read_bytes()
            ).hexdigest(),
        }
    ]
    if owner == "shrinking":
        entries.append({"path": "us/policies/other.yaml", "sha256": "0" * 64})
    target.write_text(
        json.dumps(
            {
                "schema_version": cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
                "applied_files": entries,
            }
        )
    )
    git(checkout.repo, "add", path.as_posix())
    git(checkout.repo, "commit", "-m", "prior v5 ownership")
    plan = json.loads(checkout.args.plan.read_text())
    plan["base_commit"] = git(checkout.repo, "rev-parse", "HEAD")
    checkout.args.plan.write_text(json.dumps(plan))
    original = (checkout.repo / PRIMARY).read_bytes()
    with pytest.raises(SystemExit, match="manifest ownership"):
        cli.cmd_migrate_retired_source_metadata(checkout.args)
    assert (checkout.repo / PRIMARY).read_bytes() == original
    assert not (checkout.repo / RECEIPT_DIR).exists()


def test_receipt_postcheck_failure_rolls_back_every_file(checkout, monkeypatch):
    before = {
        path: (checkout.repo / path).read_bytes()
        for path in (PRIMARY, IMPORTER, COMPANION)
    }

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected replay refusal")

    monkeypatch.setattr(cli, "_retired_source_metadata_replay", fail)
    with pytest.raises(RuntimeError, match="injected replay refusal"):
        cli.cmd_migrate_retired_source_metadata(checkout.args)
    assert all(
        (checkout.repo / path).read_bytes() == raw for path, raw in before.items()
    )
    assert not (checkout.repo / RECEIPT_DIR).exists()
    assert not (checkout.repo / cli.APPLIED_ENCODING_MANIFEST_DIR).exists()
    assert not (checkout.repo / ".axiom/.apply-transaction").exists()
