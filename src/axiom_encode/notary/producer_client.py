"""Personal-Codex client and SSH stdio relay; never writes auth into an export.

SSH uses an explicit host-key file and identity, with no ambient SSH config or
agent forwarding. The enrolled host's Unix peer UID is the operator identity.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import socket
import stat
import subprocess
import sys
from pathlib import Path

from ._schema import decode_base64, digest, fields, relative_path
from .canonical import jcs_dumps, sha256_hex, strict_parse
from .chain import Anchor, reconstruct
from .consumer import parse_consumer
from .identity import IdentityRefusal
from .lineage import (
    GENERATION_SCHEMAS,
    STORE_PREFIX,
    parse_path_policy,
    parse_record,
    signature_role,
)
from .manifest import manifest_sha256
from .producer_host import MAX_REQUEST, write_private
from .protocol import oid
from .refusal import Refusal
from .remote import RemoteRepository
from .signatures import verify_detached
from .verification import Snapshot

MAX_RESULT = 100_000_000


def packet_files(raw: bytes):
    packet = strict_parse(raw)
    if (
        len(raw) > MAX_RESULT
        or not fields(
            packet,
            {
                "schema",
                "lane",
                "epoch_sha256",
                "base_commit_git_oid",
                "run_id",
                "record_sha256",
                "files",
            },
        )
        or packet["schema"] != "axiom/producer-export/v1"
        or not oid(packet["base_commit_git_oid"])
        or not digest(packet["epoch_sha256"])
        or not digest(packet["record_sha256"])
        or not isinstance(packet["files"], list)
        or not 1 <= len(packet["files"]) <= 4000
    ):
        raise IdentityRefusal("producer_export_schema")
    files, previous = {}, ""
    for row in packet["files"]:
        if not fields(row, {"path", "mode", "base64"}):
            raise IdentityRefusal("producer_export_file")
        path = row["path"]
        if (
            not relative_path(path)
            or path.startswith("/")
            or "\\" in path
            or path <= previous
        ):
            raise IdentityRefusal("producer_export_path")
        previous = path
        raw = decode_base64(row["base64"]) if row["base64"] is not None else None
        if (
            (row["mode"] is None and row["base64"] is not None)
            or (row["mode"] == "100644" and raw is None)
            or row["mode"] not in {None, "100644"}
        ):
            raise IdentityRefusal("producer_export_mode")
        files[path] = raw
    records, allowed, endpoints = {}, set(), {}
    for path, raw in files.items():
        if path.startswith(STORE_PREFIX) and path.endswith(".json"):
            body = parse_record(raw) if raw is not None else None
            address = path.removeprefix(STORE_PREFIX).removesuffix(".json")
            if (
                body is None
                or sha256_hex(raw) != address
                or body["lane"] != packet["lane"]
                or body["epoch_sha256"] != packet["epoch_sha256"]
            ):
                raise IdentityRefusal("producer_export_record")
            records[address] = body
            role = "producer" if body["schema"] in GENERATION_SCHEMAS else "actor"
            if files.get(path + "." + role + ".sig") is None:
                raise IdentityRefusal("producer_export_evidence")
            allowed.update({path, path + "." + role + ".sig"})
            for transition in body["transitions"]:
                output = transition["path"]
                if not output.endswith((".yaml", ".yml")):
                    raise IdentityRefusal("producer_export_output")
                allowed.add(output)
                endpoints.setdefault(output, set()).add(transition["after_blob_sha256"])
    if packet["record_sha256"] not in records or set(files) != allowed:
        raise IdentityRefusal("producer_export_contents")
    for path, choices in endpoints.items():
        if (
            sha256_hex(files[path]) if files[path] is not None else None
        ) not in choices:
            raise IdentityRefusal("producer_export_output_digest")
    return packet, files


def unix_request(path: str, raw: bytes) -> bytes:
    if len(raw) > MAX_REQUEST or b"\n" in raw:
        raise IdentityRefusal("producer_request_size")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(7500)
        client.connect(path)
        client.sendall(raw + b"\n")
        with client.makefile("rb") as stream:
            response = stream.readline(MAX_RESULT + 1)
    if len(response) > MAX_RESULT or not response.endswith(b"\n"):
        raise IdentityRefusal("producer_response_size")
    return response[:-1]


def transport(args, raw):
    if args.socket:
        return unix_request(args.socket, raw)
    if (
        not args.ssh_host
        or not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+", args.ssh_host)
        or not args.known_hosts
        or not args.identity_file
    ):
        raise IdentityRefusal("producer_ssh_configuration")
    result = subprocess.run(
        [
            "/usr/bin/ssh",
            "-F",
            "/dev/null",
            "-T",
            "-oBatchMode=yes",
            "-oIdentitiesOnly=yes",
            "-oStrictHostKeyChecking=yes",
            "-oUserKnownHostsFile=" + str(Path(args.known_hosts).resolve()),
            "-oForwardAgent=no",
            "-oClearAllForwardings=yes",
            "-i",
            str(Path(args.identity_file).resolve()),
            args.ssh_host,
            "/opt/axiom/producer/bin/python -I -m axiom_encode.notary.producer_client relay --socket /run/axiom-producer/producer.sock",
        ],
        input=raw + b"\n",
        capture_output=True,
        timeout=7500,
    )
    if result.returncode != 0 or len(result.stdout) > MAX_RESULT:
        raise IdentityRefusal("producer_ssh_unavailable")
    return result.stdout.strip()


def save_result(raw: bytes, export_file: Path, refreshed_auth_file: Path | None = None):
    result = strict_parse(raw)
    if (
        not fields(
            result, {"state", "run_id", "export_base64", "refreshed_auth_base64"}
        )
        or result["state"] != "complete"
    ):
        raise IdentityRefusal("producer_run_not_complete_use_status")
    export = decode_base64(result["export_base64"])
    if export is None:
        raise IdentityRefusal("producer_response_export")
    packet_files(export)
    if (
        refreshed_auth_file is not None
        and export_file.resolve() == refreshed_auth_file.resolve()
    ):
        raise IdentityRefusal("producer_auth_output_separation")
    auth = (
        decode_base64(result["refreshed_auth_base64"])
        if result["refreshed_auth_base64"] is not None
        else None
    )
    if result["refreshed_auth_base64"] is not None and (
        auth is None or not isinstance(strict_parse(auth), dict)
    ):
        raise IdentityRefusal("producer_refreshed_auth")
    if export_file.exists():
        raise IdentityRefusal("producer_export_already_exists")
    if auth is not None:
        # Explicit, private credential output; never stdout, archive, or Git.
        if refreshed_auth_file is None:
            raise IdentityRefusal("producer_private_auth_destination_required")
        if refreshed_auth_file.is_symlink():
            raise IdentityRefusal("producer_auth_symlink")
        write_private(refreshed_auth_file, auth)
    fd = os.open(
        export_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(fd, "wb") as stream:
        stream.write(export)


def apply_export(raw: bytes, checkout: Path):
    packet, files = packet_files(raw)

    def git(*args):
        return subprocess.check_output(
            ["git", "--no-replace-objects", "-C", str(checkout), *args]
        )

    if git("rev-parse", "HEAD").decode().strip() != packet[
        "base_commit_git_oid"
    ] or git("status", "--porcelain", "--untracked-files=all"):
        raise IdentityRefusal("producer_apply_requires_clean_exact_base")
    base = Snapshot.read(checkout, packet["base_commit_git_oid"])
    if isinstance(base, Refusal):
        raise IdentityRefusal("producer_apply_base")
    registry = registry_for_base(base, packet)
    for path, content in files.items():
        if path.startswith(STORE_PREFIX) and path.endswith(".json"):
            record = parse_record(content)
            role = "producer" if record["schema"] in GENERATION_SCHEMAS else "actor"
            if not verify_detached(
                files[path + "." + role + ".sig"],
                body_sha256=sha256_hex(content),
                role=signature_role(record),
                registry=registry,
            ):
                raise IdentityRefusal("producer_apply_signature")
    policy = parse_path_policy(
        base.blobs.get(".axiom/notary/path-policy.json", b""), lane=packet["lane"]
    )
    if isinstance(policy, Refusal):
        raise IdentityRefusal("producer_apply_policy")
    for path in files:
        if not path.startswith(STORE_PREFIX) and not policy.protects(path):
            raise IdentityRefusal("producer_apply_path")
        if path.startswith(STORE_PREFIX) and path in base.blobs:
            raise IdentityRefusal("producer_apply_existing_evidence")
        target = checkout / path
        if any(parent.is_symlink() for parent in [target, *target.parents]):
            raise IdentityRefusal("producer_apply_symlink")
        if target.exists() and (not target.is_file() or target.stat().st_nlink != 1):
            raise IdentityRefusal("producer_apply_file")
        if path not in base.blobs:
            if target.exists():
                raise IdentityRefusal("producer_apply_existing_local_file")
        elif (
            not target.exists()
            or target.read_bytes() != base.blobs[path]
            or bool(target.stat().st_mode & 0o111)
        ):
            raise IdentityRefusal("producer_apply_local_file_changed")
    for path, content in files.items():
        target = checkout / path
        if content is None:
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            write_private(target, content)
            target.chmod(0o644)


def registry_for_base(base, packet):
    consumer = parse_consumer(base.blobs.get(".axiom/notary/consumer.json", b""))
    if (
        consumer is None
        or consumer["lane"] != packet["lane"]
        or consumer["epoch_sha256"] != packet["epoch_sha256"]
    ):
        raise IdentityRefusal("producer_apply_consumer")
    anchor = Anchor(
        consumer["lane"],
        consumer["epoch_sha256"],
        consumer["notary_repository"],
        consumer["notary_spki_sha256"],
    )
    with RemoteRepository(anchor.notary_repository) as remote:
        tip = remote.fetch("refs/heads/chain")
        state = reconstruct(remote.chain_history(tip), anchor)
    if (
        isinstance(state, Refusal)
        or not state.activated
        or state.tip.manifest != manifest_sha256(base.manifest)
    ):
        raise IdentityRefusal("producer_apply_finalized_base")
    return state.registry


def main():
    parser = argparse.ArgumentParser(
        description="Use an enrolled producer with personal Codex auth"
    )
    parser.add_argument(
        "operation",
        choices=("encode", "generate", "status", "correction", "relay", "apply"),
    )
    parser.add_argument("--socket")
    parser.add_argument("--ssh-host")
    parser.add_argument("--known-hosts")
    parser.add_argument("--identity-file")
    parser.add_argument("--run-id")
    parser.add_argument("--citation")
    parser.add_argument("--draw-set-id")
    parser.add_argument("--auth-file", type=Path)
    parser.add_argument("--correction-request", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--refreshed-auth", type=Path)
    parser.add_argument("--checkout", type=Path)
    args = parser.parse_args()
    try:
        if args.operation == "relay":
            raw = sys.stdin.buffer.readline(MAX_REQUEST + 1)
            if len(raw) > MAX_REQUEST or not raw.endswith(b"\n"):
                raise IdentityRefusal("producer_request_size")
            sys.stdout.buffer.write(unix_request(args.socket, raw[:-1]) + b"\n")
            return
        if args.operation == "apply":
            apply_export(args.export.read_bytes(), args.checkout.resolve())
            return
        if (
            not args.run_id
            or not re.fullmatch(r"[0-9a-f]{32}", args.run_id)
            or not args.export
            or (args.operation == "encode" and not args.refreshed_auth)
        ):
            raise IdentityRefusal("producer_client_arguments")
        request = {"operation": args.operation, "run_id": args.run_id}
        if args.operation == "encode":
            info = args.auth_file.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_mode & 0o077
                or info.st_size > 131072
            ):
                raise IdentityRefusal("producer_auth_file_permissions")
            request |= {
                "citation": args.citation,
                "draw_set_id": args.draw_set_id,
                "auth_base64": base64.b64encode(args.auth_file.read_bytes()).decode(),
            }
        elif args.operation == "correction":
            edits = strict_parse(args.correction_request.read_bytes())
            if not fields(edits, {"edits", "reason", "predecessor_run_id"}):
                raise IdentityRefusal("producer_correction_request")
            request |= edits
        save_result(
            transport(args, jcs_dumps(request)), args.export, args.refreshed_auth
        )
        print("Saved public export. Any refreshed credentials were saved separately.")
    except Exception:
        raise SystemExit(
            "Producer request refused or incomplete; preserve the run ID and use status. No credentials printed."
        ) from None


if __name__ == "__main__":
    main()
