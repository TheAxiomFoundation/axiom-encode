import base64
import subprocess

import pytest

from axiom_encode.notary.canonical import jcs_dumps, strict_parse
from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.producer_client import apply_export, packet_files, save_result

from .test_producer import emission
from .test_producers import submission as _submission_fixture

submission = _submission_fixture


def test_export_never_contains_private_auth(submission, tmp_path):
    export = emission(submission)
    result = {
        "state": "complete",
        "run_id": "1" * 32,
        "export_base64": base64.b64encode(export).decode(),
        "refreshed_auth_base64": base64.b64encode(b'{"fixture_secret":true}').decode(),
    }
    public, private = tmp_path / "export.json", tmp_path / "auth.json"
    save_result(jcs_dumps(result), public, private)
    assert public.read_bytes() == export
    assert b"fixture_secret" not in public.read_bytes()
    assert private.stat().st_mode & 0o777 == 0o600
    with pytest.raises(IdentityRefusal):
        save_result(jcs_dumps(result), private, private)


def test_arbitrary_export_paths_and_tampering_are_rejected(submission):
    packet = strict_parse(emission(submission))
    row = next(row for row in packet["files"] if row["path"] == "rules/example.yaml")
    row["base64"] = base64.b64encode(b"tampered").decode()
    with pytest.raises(IdentityRefusal, match="output_digest"):
        packet_files(jcs_dumps(packet))
    row["path"] = "../escape.yaml"
    with pytest.raises(IdentityRefusal):
        packet_files(jcs_dumps(packet))


def test_no_credential_response_needs_no_private_destination(submission, tmp_path):
    export = emission(submission)
    result = {
        "state": "complete",
        "run_id": "1" * 32,
        "export_base64": base64.b64encode(export).decode(),
        "refreshed_auth_base64": None,
    }
    public = tmp_path / "export.json"
    save_result(jcs_dumps(result), public)
    assert public.read_bytes() == export
    result["refreshed_auth_base64"] = base64.b64encode(b'{"secret":true}').decode()
    second = tmp_path / "second.json"
    with pytest.raises(IdentityRefusal, match="private_auth_destination_required"):
        save_result(jcs_dumps(result), second)
    assert not second.exists()


@pytest.fixture
def checkout(submission, tmp_path, monkeypatch):
    epoch, _, _, args = submission
    root = tmp_path / "checkout"
    root.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    git("init", "-q")
    git("config", "user.email", "fixture@example.test")
    git("config", "user.name", "Fixture")
    for path, raw in args["base"].blobs.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    git("add", ".")
    git("commit", "-qm", "fixture base")
    packet = strict_parse(emission(submission))
    packet["base_commit_git_oid"] = git("rev-parse", "HEAD").decode().strip()
    monkeypatch.setattr(
        "axiom_encode.notary.producer_client.registry_for_base",
        lambda *args: epoch.state().registry,
    )
    return root, packet, git


def test_apply_verifies_signature_before_any_write(checkout):
    root, packet, _ = checkout
    row = next(row for row in packet["files"] if row["path"].endswith(".producer.sig"))
    row["base64"] = base64.b64encode(b"not a signature").decode()
    with pytest.raises(IdentityRefusal, match="signature"):
        apply_export(jcs_dumps(packet), root)
    assert not (root / "rules/example.yaml").exists()


def test_ignored_local_file_is_preserved(checkout):
    root, packet, git = checkout
    # Git's local exclude does not alter HEAD and git status remains clean.
    (root / ".git/info/exclude").write_text("rules/example.yaml\n")
    target = root / "rules/example.yaml"
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(b"private local work")
    assert git("status", "--porcelain") == b""
    with pytest.raises(IdentityRefusal, match="existing_local_file"):
        apply_export(jcs_dumps(packet), root)
    assert target.read_bytes() == b"private local work"


def test_clean_apply_preserves_exact_emitted_bytes(checkout):
    root, packet, _ = checkout
    apply_export(jcs_dumps(packet), root)
    assert (root / "rules/example.yaml").read_bytes() == b"generated\n"
