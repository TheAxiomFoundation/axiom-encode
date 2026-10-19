"""Acquire one signed corpus member from a reviewed official export run.

GitHub artifacts supply transport, never source authority. No archive code is run.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

from axiom_encode.corpus_materialize import _parse_scope_lock, lock_path_for
from axiom_encode.corpus_release import verify_release_object
from axiom_encode.prepare_signed_backfill import (
    resolve_atomic_repair_mode,
    split_atomic_source_input,
)

REPOSITORY = "TheAxiomFoundation/axiom-corpus"
WORKFLOW = ".github/workflows/export-release-artifact.yml"
REVIEWED_EXPORTER = "d77b54c474e9bbb350f79ccbd6805bcfaf940e0b"
REVIEWED_BLOBS = {
    WORKFLOW: "aa33c31545926e12c62eaf3b3e11376a533b0832",
    "scripts/export_release_artifact.py": "f3e4764363c609c419e763badbbeb6f428042578",
    ".github/scripts/install-locked.sh": "d49d89cfccf6f53b38169583976749d7a225c922",
    ".github/scripts/uv-requirements.txt": "75951a305fc786c7dc98dd3785a8cd15614cce3f",
    "pyproject.toml": "d2e466184798193640ed498fe00c0546679c0099",
    "uv.lock": "438ba675bc1acd5d4228b466c7de116fc0a2ddc5",
}
MAX_PAYLOAD = 16 * 1024 * 1024
MAX_ARCHIVE = 17 * 1024 * 1024
MAX_RECEIPT = 16 * 1024
MAX_JSON = 4 * 1024 * 1024
MAX_RELEASE = 64 * 1024 * 1024


def request_run_id(raw: str, dispatch: dict[str, str]) -> str:
    # All old inputs, including the separate metadata lane, retain their route.
    try:
        schema = json.loads(raw).get("schema")
    except (AttributeError, ValueError):
        return ""
    if schema != "axiom-encode/corpus-artifact-transport/v1":
        return ""
    normalized = split_atomic_source_input(raw)
    resolve_atomic_repair_mode(raw, dispatch)
    return str(normalized["corpus_artifact_run_id"])


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()


def validate_run(run: dict, run_id: str, corpus: Path) -> None:
    if (
        str(run.get("id")) != run_id
        or run.get("event") != "workflow_dispatch"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or run.get("path") != WORKFLOW
        or run.get("head_branch") != "main"
        or run.get("repository", {}).get("full_name") != REPOSITORY
        or run.get("head_repository", {}).get("full_name") != REPOSITORY
    ):
        raise ValueError("export is not a successful official main workflow run")
    head = run.get("head_sha")
    if not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("invalid exporter head")
    attempt = run.get("run_attempt")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError("invalid exporter attempt")
    origin = git(corpus, "remote", "get-url", "origin")
    if origin not in {
        f"https://github.com/{REPOSITORY}",
        f"https://github.com/{REPOSITORY}.git",
    }:
        raise ValueError("corpus origin is not official")
    git(corpus, "merge-base", "--is-ancestor", REVIEWED_EXPORTER, head)
    git(corpus, "merge-base", "--is-ancestor", head, "refs/remotes/origin/main")
    for path, blob in REVIEWED_BLOBS.items():
        if git(corpus, "rev-parse", f"{head}:{path}") != blob:
            raise ValueError("exporter implementation differs from reviewed blobs")


def select_artifact(document: dict, run: dict) -> dict:
    artifacts = document.get("artifacts")
    if (
        document.get("total_count") != 1
        or not isinstance(artifacts, list)
        or len(artifacts) != 1
    ):
        raise ValueError("export run must contain exactly one artifact")
    artifact = artifacts[0]
    binding = artifact.get("workflow_run", {})
    if (
        artifact.get("name") != "authenticated-release-artifact"
        or artifact.get("expired") is not False
        or binding.get("id") != run["id"]
        or binding.get("head_sha") != run["head_sha"]
        or binding.get("repository_id") != run["repository"]["id"]
        or binding.get("head_repository_id") != run["head_repository"]["id"]
    ):
        raise ValueError("artifact is not bound to the authenticated run")
    if (
        not isinstance(artifact.get("id"), int)
        or isinstance(artifact["id"], bool)
        or artifact["id"] < 1
    ):
        raise ValueError("invalid artifact ID")
    if (
        type(artifact.get("size_in_bytes")) is not int
        or not 0 < artifact["size_in_bytes"] <= MAX_ARCHIVE
    ):
        raise ValueError("artifact archive exceeds size bound")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(artifact.get("digest"))):
        raise ValueError("artifact lacks an immutable SHA256 digest")
    return artifact


def archive_members(data: bytes, artifact: dict) -> tuple[bytes, dict]:
    if (
        len(data) > MAX_ARCHIVE
        or len(data) != artifact["size_in_bytes"]
        or hashlib.sha256(data).hexdigest() != artifact["digest"][7:]
    ):
        raise ValueError("archive size or GitHub digest mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) != 2 or {m.filename for m in members} != {
            "payload.bin",
            "receipt.json",
        }:
            raise ValueError("archive must contain exactly the two fixed regular files")
        results = {}
        for member in members:
            mode = member.external_attr >> 16
            if (
                member.flag_bits & 1
                or member.is_dir()
                or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
            ):
                raise ValueError("nonregular or encrypted archive member")
            limit = MAX_PAYLOAD if member.filename == "payload.bin" else MAX_RECEIPT
            if member.file_size <= 0 or member.file_size > limit:
                raise ValueError("archive member exceeds size bound")
            with archive.open(member) as stream:
                value = stream.read(limit + 1)
            if len(value) != member.file_size:
                raise ValueError("archive member length mismatch")
            results[member.filename] = value
    return results["payload.bin"], json.loads(results["receipt.json"])


def verify_member(
    payload: bytes,
    receipt: dict,
    release: dict,
    public_key: str,
    pin: tuple[str, str],
    run: dict,
    corpus: Path,
) -> str:
    verified = verify_release_object(release, public_key=public_key)
    if (verified.name, verified.content_sha256) != pin:
        raise ValueError("release differs from actual RuleSpec toolchain")
    expected_keys = {
        "schema_version",
        "release",
        "content_sha256",
        "source_commit",
        "path",
        "sha256",
        "bytes",
        "payload_filename",
        "exporter_commit",
        "exporter_run_id",
        "exporter_run_attempt",
        "exporter_workflow_ref",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ValueError("unexpected export receipt schema")
    if (
        receipt["schema_version"] != "axiom-corpus/release-artifact-export/v1"
        or receipt["release"] != verified.name
        or receipt["content_sha256"] != verified.content_sha256
        or receipt["source_commit"] != verified.git_commit
        or receipt["exporter_commit"] != run["head_sha"]
        or receipt["exporter_run_id"] != str(run["id"])
        or receipt["exporter_run_attempt"] != str(run["run_attempt"])
        or receipt["exporter_workflow_ref"]
        != f"{REPOSITORY}/{WORKFLOW}@refs/heads/main"
        or receipt["payload_filename"] != "payload.bin"
    ):
        raise ValueError("receipt identity mismatch")
    path = receipt["path"]
    if (
        not isinstance(path, str)
        or PurePosixPath(path).as_posix() != path
        or ".." in PurePosixPath(path).parts
    ):
        raise ValueError("unsafe provision path")
    matches = [
        a
        for a in release["content"]["artifacts"]
        if a["path"] == path and a["artifact_class"] == "provisions"
    ]
    if len(matches) != 1:
        raise ValueError("receipt is not an exact signed provision member")
    member = matches[0]
    if (
        receipt["sha256"] != member["sha256"]
        or type(receipt["bytes"]) is not int
        or receipt["bytes"] != member["bytes"]
        or len(payload) != member["bytes"]
        or not 0 < len(payload) <= MAX_PAYLOAD
        or hashlib.sha256(payload).hexdigest() != member["sha256"]
    ):
        raise ValueError("payload differs from signed member")
    git(corpus, "merge-base", "--is-ancestor", verified.git_commit, "HEAD")
    lock = lock_path_for(path)
    if lock is None or not git(
        corpus, "ls-tree", verified.git_commit, "--", str(lock)
    ).startswith("100644 blob "):
        raise ValueError("missing ordinary immutable lock")
    # Existing strict lock parser; never execute code from the exported archive.
    raw = subprocess.check_output(
        ["git", "-C", str(corpus), "show", f"{verified.git_commit}:{lock}"]
    )
    entry = _parse_scope_lock(raw, lock).get(path)
    if entry is None or entry.content != (member["sha256"], member["bytes"]):
        raise ValueError("immutable lock disagrees with signed member")
    return path


def place_member(corpus: Path, path: str, data: bytes) -> None:
    parts = PurePosixPath(path).parts
    if (
        len(parts) != 6
        or parts[:3] != ("data", "corpus", "provisions")
        or any(p in {".", ".."} for p in parts)
    ):
        raise ValueError("invalid provision destination")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(corpus, directory_flags)
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o755, dir_fd=parent_fd)
            except FileExistsError:
                pass
            child_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        try:
            descriptor = os.open(parts[-1], flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            descriptor = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
            )
            with os.fdopen(descriptor, "rb") as existing:
                if (
                    not stat.S_ISREG(os.fstat(existing.fileno()).st_mode)
                    or existing.read(MAX_PAYLOAD + 1) != data
                ):
                    raise ValueError(
                        "existing destination differs from authenticated bytes"
                    )
            return
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
        except BaseException:
            os.unlink(parts[-1], dir_fd=parent_fd)
            raise
    finally:
        os.close(parent_fd)


def read_regular(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise ValueError("transport input is not a bounded regular file")
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("transport input exceeds size bound")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["request", "validate-export", "materialize"])
    parser.add_argument("--corpus-checkout", type=Path)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--release-object", type=Path)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--artifact-metadata", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    run_id = request_run_id(
        os.environ.get("ATOMIC_SOURCE_JSON") or "[]", dict(os.environ)
    )
    if args.mode == "request":
        print(run_id)
        return
    if not run_id:
        return
    if not all((args.corpus_checkout, args.run_metadata, args.artifact_metadata)):
        raise ValueError("transport requires exact corpus and GitHub metadata paths")
    run = json.loads(read_regular(args.run_metadata, MAX_JSON))
    validate_run(run, run_id, args.corpus_checkout)
    artifact = select_artifact(
        json.loads(read_regular(args.artifact_metadata, MAX_JSON)), run
    )
    if args.mode == "validate-export":
        print(artifact["id"])
        return
    if not all((args.toolchain, args.release_object, args.archive)):
        raise ValueError(
            "materialization requires exact toolchain/release/archive paths"
        )
    pin_doc = tomllib.loads(read_regular(args.toolchain, MAX_JSON).decode())[
        "toolchain"
    ]
    pin = (
        pin_doc["axiom_corpus_release"],
        pin_doc["axiom_corpus_release_content_sha256"],
    )
    release = json.loads(read_regular(args.release_object, MAX_RELEASE))
    archive = read_regular(args.archive, MAX_ARCHIVE)
    payload, receipt = archive_members(archive, artifact)
    path = verify_member(
        payload,
        receipt,
        release,
        os.environ["TRUST_CORPUS_RELEASE_ROOT"],
        pin,
        run,
        args.corpus_checkout,
    )
    place_member(args.corpus_checkout, path, payload)
    print(
        json.dumps(
            {
                "transport_run": run_id,
                "artifact_id": artifact["id"],
                "archive_digest": artifact["digest"],
                "receipt": receipt,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
