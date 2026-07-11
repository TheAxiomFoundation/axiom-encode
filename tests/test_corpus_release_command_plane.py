"""Adversarial signed-corpus checks through the public command plane."""

from __future__ import annotations

import hashlib
import json
import sys
from base64 import b64encode
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from axiom_encode import entrypoint, signing_broker
from axiom_encode.corpus_resolver import InvalidCorpusReleaseError, LocalCorpusRelease
from tests.release_object_fixtures import bind_test_corpus_release
from tests.signing_broker_fixtures import SigningBrokerFixture

RELEASE_NAME = "public-command-test-release"
RELEASE_VERSION = "2026-public-command-test"
MODULE_RELATIVE = Path("us/statutes/module.yaml")


@dataclass(frozen=True)
class CommandPlaneFixture:
    rulespec_root: Path
    corpus_root: Path
    engine_root: Path
    module: Path
    protected_base: Path
    changed_paths: Path
    release: LocalCorpusRelease


def _waiver_text() -> str:
    expiry = (date.today() + timedelta(days=30)).isoformat()
    return (
        "validate_failures:\n"
        f"  {MODULE_RELATIVE.as_posix()}:\n"
        "    active:\n"
        f"      fingerprint: sha256:{'a' * 64}\n"
        "      owner: '@axiom-security'\n"
        "      issue: "
        "https://github.com/TheAxiomFoundation/axiom-encode/issues/1108\n"
        f"      expires: '{expiry}'\n"
    )


def _write_command_plane_fixture(tmp_path: Path) -> CommandPlaneFixture:
    rulespec_root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    engine_root = tmp_path / "axiom-rules-engine"
    engine_root.mkdir()

    module = rulespec_root / MODULE_RELATIVE
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n")

    waiver = rulespec_root / "known-validation-gaps.yaml"
    waiver.write_text(_waiver_text())
    protected_base = tmp_path / "protected-base-waivers.yaml"
    protected_base.write_bytes(waiver.read_bytes())
    changed_paths = tmp_path / "changed-paths.txt"
    changed_paths.write_text("")

    provision = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / "us"
        / "statute"
        / f"{RELEASE_VERSION}.jsonl"
    )
    provision.parent.mkdir(parents=True)
    provision.write_text(
        json.dumps(
            {
                "id": "public-command-source",
                "citation_path": "us/statute/1",
                "body": "The signed public-command source states 451.",
                "jurisdiction": "us",
                "document_class": "statute",
                "version": RELEASE_VERSION,
                "source_path": "sources/public-command-source.txt",
                "source_as_of": "2026-07-11",
                "expression_date": "2026-07-11",
            }
        )
        + "\n"
    )
    release = bind_test_corpus_release(
        corpus_root,
        RELEASE_NAME,
        [("us", "statute", RELEASE_VERSION)],
    )

    toolchain = rulespec_root / ".axiom" / "toolchain.toml"
    toolchain.parent.mkdir()
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{RELEASE_NAME}"\n'
        "axiom_corpus_release_content_sha256 = "
        f'"{release.content_sha256}"\n'
        "validation_waiver_set_sha256 = "
        f'"{hashlib.sha256(waiver.read_bytes()).hexdigest()}"\n'
    )
    return CommandPlaneFixture(
        rulespec_root=rulespec_root,
        corpus_root=corpus_root,
        engine_root=engine_root,
        module=module,
        protected_base=protected_base,
        changed_paths=changed_paths,
        release=release,
    )


def _public_argv(fixture: CommandPlaneFixture, command: str) -> list[str]:
    if command == "validate":
        return [
            "validate",
            str(fixture.module),
            "--skip-reviewers",
            "--corpus-path",
            str(fixture.corpus_root),
            "--axiom-rules-engine-path",
            str(fixture.engine_root),
        ]
    if command == "proof-validate":
        return [
            "proof-validate",
            str(fixture.module),
            "--corpus-path",
            str(fixture.corpus_root),
        ]
    if command == "waiver-audit":
        return [
            "validation-waivers",
            "audit",
            "--root",
            str(fixture.rulespec_root),
            "--corpus-path",
            str(fixture.corpus_root),
            "--protected-base",
            str(fixture.protected_base),
            "--changed-paths",
            str(fixture.changed_paths),
            "--axiom-rules-engine-path",
            str(fixture.engine_root),
            "--json",
        ]
    raise AssertionError(f"unsupported public command fixture: {command}")


def _invoke_public(monkeypatch, fixture: CommandPlaneFixture, command: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["axiom-encode", *_public_argv(fixture, command)],
    )
    entrypoint.main()


def _assert_public_release_rejection(
    monkeypatch,
    capsys,
    fixture: CommandPlaneFixture,
    command: str,
    expected: str,
) -> None:
    if command == "waiver-audit":
        with pytest.raises(SystemExit) as error:
            _invoke_public(monkeypatch, fixture, command)
        assert error.value.code == 2
        output = capsys.readouterr()
        assert expected in output.out + output.err
        return
    with pytest.raises(InvalidCorpusReleaseError, match=expected):
        _invoke_public(monkeypatch, fixture, command)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(b"", "not valid UTF-8 JSON", id="zero-byte"),
        pytest.param(b'""\n', "must be a JSON object", id="empty-json-string"),
        pytest.param(b"null\n", "must be a JSON object", id="empty-json-null"),
        pytest.param(b"[]\n", "must be a JSON object", id="empty-json-array"),
        pytest.param(b"{}\n", "unsupported schema version", id="empty-object"),
    ],
)
@pytest.mark.parametrize("command", ["validate", "proof-validate", "waiver-audit"])
def test_public_commands_reject_malformed_on_disk_release_object(
    tmp_path,
    monkeypatch,
    capsys,
    command,
    raw,
    expected,
):
    fixture = _write_command_plane_fixture(tmp_path)
    fixture.release.release_object_path.write_bytes(raw)

    _assert_public_release_rejection(
        monkeypatch,
        capsys,
        fixture,
        command,
        expected,
    )


@pytest.mark.parametrize("command", ["validate", "proof-validate", "waiver-audit"])
def test_public_commands_reject_wrong_key_release_object(
    tmp_path,
    monkeypatch,
    capsys,
    command,
):
    fixture = _write_command_plane_fixture(tmp_path)
    wrong_key = Ed25519PrivateKey.from_private_bytes(b"\x99" * 32)
    wrong_public = b64encode(
        wrong_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    ).decode("ascii")
    monkeypatch.setattr(
        signing_broker,
        "_active_broker",
        SigningBrokerFixture(corpus_release_public_key=wrong_public),
    )

    _assert_public_release_rejection(
        monkeypatch,
        capsys,
        fixture,
        command,
        "signature is invalid",
    )


@pytest.mark.parametrize("command", ["validate", "proof-validate", "waiver-audit"])
def test_public_commands_reject_tampered_release_object(
    tmp_path,
    monkeypatch,
    capsys,
    command,
):
    fixture = _write_command_plane_fixture(tmp_path)
    payload = json.loads(fixture.release.release_object_path.read_text())
    payload["content"]["created_at"] = "2026-07-11T00:00:01Z"
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(
            payload["content"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    fixture.release.release_object_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    _assert_public_release_rejection(
        monkeypatch,
        capsys,
        fixture,
        command,
        "signature is invalid",
    )


@pytest.mark.parametrize("command", ["validate", "proof-validate", "waiver-audit"])
def test_public_commands_reject_unsigned_cut_plan_without_signed_object(
    tmp_path,
    monkeypatch,
    capsys,
    command,
):
    fixture = _write_command_plane_fixture(tmp_path)
    fixture.release.release_object_path.unlink()
    cut_plan = (
        fixture.corpus_root / "manifests" / "releases" / f"{fixture.release.name}.json"
    )
    cut_plan.parent.mkdir(parents=True)
    cut_plan.write_text(
        json.dumps(
            {
                "name": fixture.release.name,
                "scopes": [
                    {
                        "jurisdiction": "us",
                        "document_class": "statute",
                        "version": RELEASE_VERSION,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    _assert_public_release_rejection(
        monkeypatch,
        capsys,
        fixture,
        command,
        "Corpus release object not found",
    )


def test_public_proof_validate_rejects_source_outside_release_inventory(
    tmp_path,
    monkeypatch,
    capsys,
):
    fixture = _write_command_plane_fixture(tmp_path)
    rogue_citation = "us/statute/rogue"
    rogue = (
        fixture.corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / "us"
        / "statute"
        / "rogue.jsonl"
    )
    rogue.write_text(
        json.dumps(
            {
                "id": "unattested-local-source",
                "citation_path": rogue_citation,
                "body": "The unattested local file says the amount is 999.",
                "jurisdiction": "us",
                "document_class": "statute",
                "version": "rogue",
                "source_path": "sources/unattested-local-source.txt",
                "source_as_of": "2026-07-11",
                "expression_date": "2026-07-11",
            }
        )
        + "\n"
    )
    fixture.module.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  proof_validation:\n"
        "    required: true\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {rogue_citation}\n"
        "rules:\n"
        "  - name: unattested_amount\n"
        "    kind: parameter\n"
        "    dtype: Money\n"
        "    unit: USD\n"
        "    metadata:\n"
        "      proof:\n"
        "        atoms:\n"
        "          - path: versions[0].formula\n"
        "            kind: parameter\n"
        "            source:\n"
        f"              corpus_citation_path: {rogue_citation}\n"
        "              excerpt: The unattested local file says the amount is 999.\n"
        "    versions:\n"
        "      - effective_from: '2026-01-01'\n"
        "        formula: '999'\n"
    )

    with pytest.raises(SystemExit) as error:
        _invoke_public(monkeypatch, fixture, "proof-validate")

    assert error.value.code == 1
    output = capsys.readouterr().out
    assert "Proof corpus resolution failed" in output
    assert "CorpusSourceNotFoundError" in output
