"""Custodian-owned Unix-socket producer service, with peer-UID operator binding.

No operation takes a signing key, scope, runtime identity, encoder command or
arbitrary generation body. Only `encode` invokes the model. Reusing a run ID
never reruns generation; completed results remain available to their owner.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import os
import re
import socket
import socketserver
import stat
import struct
import tempfile
import threading
import tomllib
from contextlib import contextmanager
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ._schema import decode_base64, digest, fields, lane_name
from .canonical import jcs_dumps, sha256_hex, strict_parse
from .chain import Anchor, reconstruct
from .consumer import parse_consumer
from .deployment import custodian_file, custodian_parent
from .identity import IdentityRefusal
from .lineage import STORE_PREFIX
from .manifest import manifest_sha256
from .producer import (
    correction_export,
    generation_export,
    key_fingerprint,
    observed_transitions,
)
from .producer_runtime import LinuxRuntime
from .producers import ENROLLMENT_PATH, parse_enrollments
from .protocol import decimal_id, parse_artifact
from .refusal import Refusal
from .remote import RemoteRepository
from .verification import Snapshot

MAX_REQUEST = 32_000_000


def parse_config(raw):
    body = strict_parse(raw)
    names = {
        "schema",
        "lane",
        "content_branch",
        "epoch_sha256",
        "notary_spki_sha256",
        "producer_key_file",
        "actor_key_file",
        "state_directory",
        "socket_path",
        "socket_gid",
        "operators",
        "encoder_identity",
        "dependency_inventory",
        "codex_binary",
        "python",
        "worker_uid",
        "worker_gid",
        "corpus_path",
        "engine_path",
        "dependency_roots",
        "model",
        "sampling",
        "references",
        "timeout_seconds",
    }
    if not fields(body, names) or body["schema"] != "axiom/supervised-producer-host/v1":
        raise IdentityRefusal("producer_host_configuration")
    if (
        not lane_name(body["lane"])
        or not body["lane"].startswith("TheAxiomFoundation/")
        or not isinstance(body["content_branch"], str)
        or not re.fullmatch(r"[A-Za-z0-9_-]+", body["content_branch"])
        or not all(digest(body[k]) for k in ("epoch_sha256", "notary_spki_sha256"))
    ):
        raise IdentityRefusal("producer_host_lane")
    if (
        type(body["timeout_seconds"]) is not int
        or not 60 <= body["timeout_seconds"] <= 7200
        or type(body["socket_gid"]) is not int
        or body["socket_gid"] <= 0
    ):
        raise IdentityRefusal("producer_host_limits")
    if (
        not isinstance(body["model"], str)
        or re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", body["model"]) is None
    ):
        raise IdentityRefusal("producer_host_model")
    if not fields(body["sampling"], {"temperature", "seed"}) or not fields(
        body["references"], {"oracles", "reference_data"}
    ):
        raise IdentityRefusal("producer_host_metadata")
    operators = body["operators"]
    if (
        not isinstance(operators, list)
        or not operators
        or any(
            not fields(item, {"uid", "github_user_id"})
            or type(item["uid"]) is not int
            or item["uid"] < 1000
            or item["uid"] == body["worker_uid"]
            or not decimal_id(item["github_user_id"])
            for item in operators
        )
        or len({item["uid"] for item in operators}) != len(operators)
    ):
        raise IdentityRefusal("producer_host_operators")
    if not isinstance(body["dependency_roots"], dict) or any(
        not re.fullmatch(r"[a-z]{2}", name) for name in body["dependency_roots"]
    ):
        raise IdentityRefusal("producer_host_dependencies")
    for path in [
        body[k]
        for k in (
            "producer_key_file",
            "actor_key_file",
            "state_directory",
            "socket_path",
            "codex_binary",
            "python",
            "corpus_path",
            "engine_path",
        )
    ] + list(body["dependency_roots"].values()):
        if (
            not isinstance(path, str)
            or not re.fullmatch(r"/[A-Za-z0-9_./-]+", path)
            or Path(path) != Path(path).resolve()
        ):
            raise IdentityRefusal("producer_host_path")
    inventory = jcs_dumps(body["dependency_inventory"])
    if parse_artifact(inventory, "dependency-inventory") is None:
        raise IdentityRefusal("producer_host_inventory")
    return body | {"dependency_inventory": inventory}


def write_private(path: Path, raw: bytes):
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(fd)
        temporary.unlink(missing_ok=True)


class ProducerHost:
    def __init__(self, config, *, runtime=None):
        self.config = config
        self.runtime = runtime or LinuxRuntime(config)
        self.operators = {
            row["uid"]: row["github_user_id"] for row in config["operators"]
        }
        self.keys = {}
        for role in ("producer", "actor"):
            key = serialization.load_pem_private_key(
                custodian_file(config[role + "_key_file"], private=True, limit=10000),
                password=None,
            )
            if not isinstance(key, Ed25519PrivateKey):
                raise IdentityRefusal("producer_host_key")
            self.keys[role] = key
        if key_fingerprint(self.keys["producer"]) == key_fingerprint(
            self.keys["actor"]
        ):
            raise IdentityRefusal("producer_actor_keys_not_separate")
        self.root = Path(config["state_directory"])
        if not self.root.exists():
            self.root.mkdir(mode=0o711, parents=False)
            self.root.chmod(0o711)
        custodian_parent(self.root / "lock")
        if (
            self.root.stat().st_uid != os.geteuid()
            or stat.S_IMODE(self.root.stat().st_mode) != 0o711
        ):
            raise IdentityRefusal("producer_state_directory")
        for name, mode in (("jobs", 0o711), ("records", 0o700), ("results", 0o700)):
            directory = self.root / name
            if not directory.exists():
                directory.mkdir(mode=mode)
                directory.chmod(mode)
            if (
                directory.is_symlink()
                or directory.stat().st_uid != os.geteuid()
                or stat.S_IMODE(directory.stat().st_mode) != mode
            ):
                raise IdentityRefusal("producer_state_directory")

    @contextmanager
    def _lock(self):
        fd = os.open(self.root / "lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    @contextmanager
    def _base(self, github_user_id):
        c = self.config
        with RemoteRepository(c["lane"]) as lane:
            tip = lane.fetch("refs/heads/" + c["content_branch"])
            base = lane.snapshot(tip)
            consumer = parse_consumer(
                base.blobs.get(".axiom/notary/consumer.json", b"")
            )
            if (
                consumer is None
                or consumer["epoch_sha256"] != c["epoch_sha256"]
                or consumer["notary_spki_sha256"] != c["notary_spki_sha256"]
                or consumer["lane"] != c["lane"]
                or consumer["notary_repository"] != c["lane"] + "-notary"
            ):
                raise IdentityRefusal("producer_consumer_binding")
            anchor = Anchor(
                c["lane"],
                c["epoch_sha256"],
                c["lane"] + "-notary",
                c["notary_spki_sha256"],
            )
            with RemoteRepository(anchor.notary_repository) as chain:
                chain_tip = chain.fetch("refs/heads/chain")
                state = reconstruct(chain.chain_history(chain_tip), anchor)
            if (
                isinstance(state, Refusal)
                or not state.activated
                or state.tip.manifest != manifest_sha256(base.manifest)
            ):
                raise IdentityRefusal("producer_finalized_base")
            enrollments = parse_enrollments(
                base.blobs.get(ENROLLMENT_PATH, b""), state.registry
            )
            enrollment = enrollments.get(key_fingerprint(self.keys["producer"]))
            if (
                enrollment is None
                or enrollment.body["actor_spki_sha256"]
                != key_fingerprint(self.keys["actor"])
                or github_user_id not in enrollment.body["github_user_ids"]
                or enrollment.body["encoder"] != c["encoder_identity"]
            ):
                raise IdentityRefusal("producer_enrollment")
            try:
                pins = tomllib.loads(
                    base.blobs[".axiom/workflow-toolchain.toml"].decode()
                )["workflow_toolchain"]
                if (
                    pins["axiom_encode_ref"] != c["encoder_identity"]["git_oid"]
                    or pins["axiom_encode_version"] != c["encoder_identity"]["version"]
                ):
                    raise ValueError
            except (KeyError, UnicodeError, ValueError, TypeError):
                raise IdentityRefusal("producer_encoder_pin") from None
            yield (
                lane,
                base,
                enrollment,
                [
                    key.public_bytes(
                        serialization.Encoding.Raw, serialization.PublicFormat.Raw
                    )
                    for key in state.registry.keys["corpus-release"].values()
                ],
            )

    def _status(self, run_id, uid):
        path = self.root / "records" / (run_id + ".json")
        if not path.exists():
            raise IdentityRefusal("producer_run_unknown")
        record = strict_parse(custodian_file(str(path), private=True))
        if record["uid"] != uid or record.get("github_user_id") != self.operators[uid]:
            raise IdentityRefusal("producer_run_owner")
        if record["state"] == "complete":
            return strict_parse(
                custodian_file(
                    str(self.root / "results" / (run_id + ".json")),
                    private=True,
                    limit=100_000_000,
                )
            )
        return {"state": record["state"], "run_id": run_id}

    def perform(self, uid: int, request: dict):
        if uid not in self.operators:
            raise IdentityRefusal("producer_operator_not_enrolled")
        if (
            not isinstance(request, dict)
            or not isinstance(request.get("run_id"), str)
            or re.fullmatch(r"[0-9a-f]{32}", request["run_id"]) is None
        ):
            raise IdentityRefusal("producer_request")
        operation, run_id = request.get("operation"), request["run_id"]
        if operation == "status":
            if not fields(request, {"operation", "run_id"}):
                raise IdentityRefusal("producer_request")
            return self._status(run_id, uid)
        if operation == "encode":
            if (
                not fields(
                    request,
                    {"operation", "run_id", "citation", "draw_set_id", "auth_base64"},
                )
                or any(
                    not isinstance(request[k], str)
                    or not request[k].strip()
                    or len(request[k]) > 500
                    or "\0" in request[k]
                    for k in ("citation", "draw_set_id")
                )
                or request["citation"].startswith("-")
            ):
                raise IdentityRefusal("producer_encode_request")
            auth = decode_base64(request["auth_base64"])
            if (
                auth is None
                or len(auth) > 131072
                or not isinstance(strict_parse(auth), dict)
            ):
                raise IdentityRefusal("producer_personal_auth")
            public_request = {
                key: value for key, value in request.items() if key != "auth_base64"
            }
        elif operation == "correction":
            if (
                not fields(
                    request,
                    {"operation", "run_id", "edits", "reason", "predecessor_run_id"},
                )
                or not isinstance(request["edits"], dict)
                or not request["edits"]
                or len(request["edits"]) > 1000
            ):
                raise IdentityRefusal("producer_correction_request")
            public_request = request
        else:
            raise IdentityRefusal("producer_operation")
        with self._lock():
            record_path = self.root / "records" / (run_id + ".json")
            fingerprint = sha256_hex(jcs_dumps(public_request))
            if record_path.exists():
                record = strict_parse(custodian_file(str(record_path), private=True))
                if (
                    record["uid"] != uid
                    or record.get("github_user_id") != self.operators[uid]
                    or record["request_sha256"] != fingerprint
                ):
                    raise IdentityRefusal("producer_run_reuse")
                return self._status(run_id, uid)
            record = {
                "uid": uid,
                "github_user_id": self.operators[uid],
                "request_sha256": fingerprint,
                "state": "running-or-interrupted",
            }
            write_private(record_path, jcs_dumps(record))
            try:
                with self._base(self.operators[uid]) as (
                    lane,
                    base,
                    enrollment,
                    corpus_public_keys,
                ):
                    common = dict(
                        enrollment=enrollment,
                        base=base,
                        lane=self.config["lane"],
                        epoch=self.config["epoch_sha256"],
                        run_id=run_id,
                    )
                    refreshed = None
                    if operation == "encode":
                        job = self.root / "jobs" / run_id
                        job.mkdir(mode=0o700)
                        observed, refreshed = self.runtime.run(
                            job=job,
                            request=request,
                            lane_remote=lane,
                            base=base,
                            enrollment=enrollment,
                            auth=auth,
                            corpus_public_keys=corpus_public_keys,
                        )
                        export = generation_export(
                            self.keys["producer"],
                            **common,
                            **observed,
                            draw_set_id=request["draw_set_id"],
                            sampling=self.config["sampling"],
                            independence={
                                "sibling_draws_visible": "no",
                                "incumbent_encoding_visible": "yes",
                            },
                            references=self.config["references"],
                        )
                    else:
                        inherited = None
                        predecessor = None
                        if request["predecessor_run_id"] is not None:
                            prior_run = request["predecessor_run_id"]
                            if (
                                not isinstance(prior_run, str)
                                or not re.fullmatch(r"[0-9a-f]{32}", prior_run)
                                or prior_run == run_id
                            ):
                                raise IdentityRefusal("producer_correction_predecessor")
                            prior = self._status(prior_run, uid)
                            if prior["state"] != "complete":
                                raise IdentityRefusal("producer_correction_predecessor")
                            inherited = strict_parse(
                                decode_base64(prior["export_base64"])
                            )
                            if (
                                inherited["base_commit_git_oid"] != base.commit
                                or inherited["epoch_sha256"]
                                != self.config["epoch_sha256"]
                            ):
                                raise IdentityRefusal("producer_correction_stale_base")
                            blobs = dict(base.blobs)
                            modes = {path: mode for path, mode, _ in base.manifest}
                            for row in inherited["files"]:
                                raw = (
                                    decode_base64(row["base64"])
                                    if row["base64"] is not None
                                    else None
                                )
                                if raw is None:
                                    blobs.pop(row["path"], None)
                                    modes.pop(row["path"], None)
                                else:
                                    blobs[row["path"]] = raw
                                    modes[row["path"]] = "100644"
                            common["base"] = Snapshot(
                                base.commit,
                                sorted(
                                    (path, modes[path], sha256_hex(raw))
                                    for path, raw in blobs.items()
                                ),
                                blobs,
                            )
                            predecessor = inherited["record_sha256"]
                        edits = {
                            path: decode_base64(raw) if raw is not None else None
                            for path, raw in request["edits"].items()
                        }
                        if any(
                            request["edits"][path] is not None and raw is None
                            for path, raw in edits.items()
                        ):
                            raise IdentityRefusal("producer_correction_encoding")
                        if inherited is not None:
                            transitions, _ = observed_transitions(
                                common["base"], edits, lane=self.config["lane"]
                            )
                            latest = strict_parse(
                                common["base"].blobs[
                                    STORE_PREFIX + predecessor + ".json"
                                ]
                            )
                            endpoints = {
                                row["path"]: row["after_blob_sha256"]
                                for row in latest["transitions"]
                            }
                            if any(
                                row["path"] not in endpoints
                                or endpoints[row["path"]] != row["before_blob_sha256"]
                                for row in transitions
                            ):
                                # V33 permits null. The coverage graph still
                                # authenticates every per-path predecessor; a
                                # latest run is not necessarily the latest
                                # record for all paths in this correction.
                                predecessor = None
                        export = correction_export(
                            self.keys["actor"],
                            **common,
                            outputs=edits,
                            github_user_id=self.operators[uid],
                            reason=request["reason"],
                            predecessor=predecessor,
                        )
                        if inherited is not None:
                            packet = strict_parse(export)
                            rows = {row["path"]: row for row in inherited["files"]}
                            rows.update({row["path"]: row for row in packet["files"]})
                            packet["files"] = [rows[path] for path in sorted(rows)]
                            export = jcs_dumps(packet)
                    result = {
                        "state": "complete",
                        "run_id": run_id,
                        "export_base64": base64.b64encode(export).decode(),
                        "refreshed_auth_base64": base64.b64encode(refreshed).decode()
                        if refreshed is not None
                        else None,
                    }
                    write_private(
                        self.root / "results" / (run_id + ".json"), jcs_dumps(result)
                    )
                    write_private(
                        record_path, jcs_dumps(record | {"state": "complete"})
                    )
                    return result
            except BaseException:
                write_private(
                    record_path, jcs_dumps(record | {"state": "failed-or-interrupted"})
                )
                raise


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        _, uid, _ = struct.unpack(
            "3i", self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        try:
            if uid not in self.server.host.operators:
                raise IdentityRefusal("producer_operator_not_enrolled")
            self.connection.settimeout(30)
            line = self.rfile.readline(MAX_REQUEST + 1)
            if len(line) > MAX_REQUEST or not line.endswith(b"\n"):
                raise IdentityRefusal("producer_request_size")
            result = self.server.host.perform(uid, strict_parse(line[:-1]))
        except IdentityRefusal as exc:
            reason = str(exc)
            result = {
                "error": reason
                if re.fullmatch(r"[a-z_]{1,100}", reason)
                else "producer_refused"
            }
        except Exception:
            result = {"error": "producer_unavailable"}
        self.wfile.write(jcs_dumps(result) + b"\n")


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def main():
    parser = argparse.ArgumentParser(
        description="Run the Linux custodian-owned supervised producer"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    config = parse_config(custodian_file(args.config))
    host = ProducerHost(config)
    target = Path(config["socket_path"])
    custodian_parent(target, socket=True)
    if target.exists():
        raise SystemExit("producer socket already exists; refuse replacement")

    with Server(str(target), Handler) as server:
        server.host = host
        os.chown(target, 0, config["socket_gid"])
        target.chmod(0o660)
        server.serve_forever()


if __name__ == "__main__":
    main()
