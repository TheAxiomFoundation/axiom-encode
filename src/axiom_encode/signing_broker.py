"""Out-of-process signing boundary for untrusted model execution.

A trusted external supervisor provisions this broker over an anonymous socket
and launches the CLI with only the attached descriptor. The model-capable CLI
rejects private signing keys in its environment and immediately makes the
descriptor non-inheritable, so model/reviewer subprocesses receive neither
private signing material nor an IPC capability.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import resource
import socket
import struct
import sys
import threading
from base64 import b64decode, b64encode
from binascii import Error as BinasciiError
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

LEGACY_APPLY_SIGNING_KEY_ENV: Final = "AXIOM_ENCODE_APPLY_SIGNING_KEY"
APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV: Final = "AXIOM_ENCODE_APPLY_SIGNING_PRIVATE_KEY"
EVAL_EVIDENCE_PRIVATE_KEY_ENV: Final = "AXIOM_ENCODE_EVAL_SIGNING_PRIVATE_KEY"

BROKER_FD_ENV: Final = "AXIOM_ENCODE_SIGNING_BROKER_FD"
BROKER_PID_ENV: Final = "AXIOM_ENCODE_SIGNING_BROKER_PID"
BROKER_MARKER_ENV: Final = "AXIOM_ENCODE_SIGNING_BROKER_ACTIVE"

_PRIVATE_ENV_NAMES: Final = (
    LEGACY_APPLY_SIGNING_KEY_ENV,
    APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV,
    EVAL_EVIDENCE_PRIVATE_KEY_ENV,
)
_BROKER_ENV_NAMES: Final = (BROKER_FD_ENV, BROKER_PID_ENV, BROKER_MARKER_ENV)
_PROTOCOL_VERSION: Final = 4
_MAX_FRAME_BYTES: Final = 64 * 1024 * 1024
_SIGNATURE_DOMAIN_PREFIX: Final = b"axiom-encode/external-signer-sign/v2\x00"
_SIGNING_SCOPES: Final = frozenset({"apply_ed25519", "eval_ed25519"})
_TRUSTED_SUBPROCESS_BASE_NAMES: Final = (
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_NOSYSTEM",
    "GIT_TERMINAL_PROMPT",
    "HOME",
    "LANG",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
)
_EXPLICIT_SUBPROCESS_ENV_NAMES: Final = frozenset(
    {
        "LC_ALL",
        "PYTHONHASHSEED",
        "TZ",
    }
)
_trusted_subprocess_base_environment: dict[str, str] | None = None


def canonical_signing_message(scope: str, payload: bytes) -> bytes:
    """Bind one persisted signature to its exact operation scope."""

    if scope not in _SIGNING_SCOPES:
        raise ValueError(f"Unsupported signing scope: {scope!r}")
    if not isinstance(payload, bytes):
        raise TypeError("Signing payload must be bytes")
    return _SIGNATURE_DOMAIN_PREFIX + scope.encode("ascii") + b"\x00" + payload


class SigningBrokerError(RuntimeError):
    """The trusted signing broker is unavailable or rejected a request."""


class SigningBroker(Protocol):
    """Typed operations exposed to signing callers."""

    @property
    def capabilities(self) -> frozenset[str]: ...

    @property
    def apply_public_key_raw(self) -> bytes | None: ...

    @property
    def eval_public_key_raw(self) -> bytes | None: ...

    @property
    def corpus_release_public_key_raw(self) -> bytes | None: ...

    @property
    def corpus_release_public_keys_raw(self) -> tuple[bytes, ...]: ...

    def apply_ed25519_sign(self, payload: bytes) -> bytes: ...

    def eval_ed25519_sign(self, payload: bytes) -> bytes: ...


@dataclass(frozen=True, slots=True)
class BrokerStatus:
    """Non-secret broker capabilities returned after provisioning."""

    capabilities: frozenset[str]
    apply_public_key_raw: bytes | None
    eval_public_key_raw: bytes | None
    corpus_release_public_key_raw: bytes | None
    corpus_release_public_keys_raw: tuple[bytes, ...]


def scrub_private_signing_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the purpose-minimal environment allowed for untrusted children.

    Signing processes do not forward ambient credentials, proxy settings,
    executable search paths, Git configuration, telemetry exporters, or
    Supabase configuration. Callers that need an additional public, non-secret
    value must add it explicitly for that one subprocess after this scrub.
    The optional input can add only the narrow RuleSpec/locale/hash settings
    above; it cannot replace the trusted HOME, PATH, or Git neutralizers.
    """

    clean = dict(
        _trusted_subprocess_base_environment
        or {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "PATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "XDG_CONFIG_HOME": "/nonexistent",
            "XDG_DATA_HOME": "/nonexistent",
        }
    )
    if environment is not None:
        for name, value in environment.items():
            if name in _EXPLICIT_SUBPROCESS_ENV_NAMES:
                clean[name] = value
    return clean


def reject_direct_private_signing_environment(
    environment: Mapping[str, str] | None = None,
) -> None:
    """Reject production startup with private signing bytes in the environment.

    Production signing must be provisioned by an external supervisor through an
    already-attached broker descriptor. This module never turns a mutable Python
    process that received private environment bytes into the model-capable CLI.
    """

    source = os.environ if environment is None else environment
    present = [name for name in _PRIVATE_ENV_NAMES if name in source]
    if present:
        raise SigningBrokerError(
            "Private signing keys must not be supplied to axiom-encode through "
            "its environment; attach an externally provisioned signing broker "
            "instead (forbidden: " + ", ".join(present) + ")"
        )


class SigningBrokerClient:
    """Synchronous client for one anonymous signing-broker socket."""

    def __init__(self, connection: socket.socket, *, broker_pid: int | None = None):
        self._connection = connection
        self._broker_pid = broker_pid
        self._lock = threading.Lock()
        self._next_request_id = 1
        self._closed = False
        os.set_inheritable(self._connection.fileno(), False)
        status = self._request("status")
        legacy_status_fields = {
            "capabilities",
            "apply_public_key",
            "eval_public_key",
            "corpus_release_public_key",
        }
        status_fields = set(status)
        if status_fields not in {
            frozenset(legacy_status_fields),
            frozenset({*legacy_status_fields, "corpus_release_public_keys"}),
        }:
            raise SigningBrokerError("Signing broker returned malformed status")
        raw_capabilities = status.get("capabilities")
        if not isinstance(raw_capabilities, list) or any(
            not isinstance(item, str) for item in raw_capabilities
        ):
            raise SigningBrokerError("Signing broker returned malformed capabilities")
        if len(raw_capabilities) != len(set(raw_capabilities)) or not set(
            raw_capabilities
        ) <= {"apply_ed25519", "eval_ed25519"}:
            raise SigningBrokerError("Signing broker returned malformed capabilities")
        apply_public_key_raw = self._decode_status_public_key(
            status.get("apply_public_key"),
            label="apply",
        )
        eval_public_key_raw = self._decode_status_public_key(
            status.get("eval_public_key"),
            label="eval",
        )
        corpus_release_public_key_raw = self._decode_status_public_key(
            status.get("corpus_release_public_key"),
            label="corpus release",
        )
        encoded_corpus_release_public_keys = status.get("corpus_release_public_keys")
        if "corpus_release_public_keys" not in status:
            corpus_release_public_keys_raw = (
                (corpus_release_public_key_raw,)
                if corpus_release_public_key_raw is not None
                else ()
            )
        elif not isinstance(encoded_corpus_release_public_keys, list) or not (
            encoded_corpus_release_public_keys
        ):
            raise SigningBrokerError(
                "Signing broker returned a malformed corpus release public keyring"
            )
        else:
            corpus_release_public_keys_raw = tuple(
                self._decode_required_status_public_key(
                    encoded_public,
                    label="corpus release",
                )
                for encoded_public in encoded_corpus_release_public_keys
            )
        if (
            not corpus_release_public_keys_raw
            or corpus_release_public_keys_raw[0] != corpus_release_public_key_raw
        ):
            raise SigningBrokerError(
                "Signing broker returned a conflicting corpus release public keyring"
            )
        protected_roots = {
            apply_public_key_raw,
            eval_public_key_raw,
            *corpus_release_public_keys_raw,
        }
        if None in protected_roots or len(protected_roots) != (
            2 + len(corpus_release_public_keys_raw)
        ):
            raise SigningBrokerError(
                "Signing broker must expose distinct protected apply, eval, and "
                "corpus release trust roots"
            )
        self._status = BrokerStatus(
            frozenset(raw_capabilities),
            apply_public_key_raw,
            eval_public_key_raw,
            corpus_release_public_key_raw,
            corpus_release_public_keys_raw,
        )

    @staticmethod
    def _decode_status_public_key(
        encoded_public: object, *, label: str
    ) -> bytes | None:
        """Decode one optional raw Ed25519 public key from broker status."""

        if encoded_public is None:
            return None
        elif isinstance(encoded_public, str):
            try:
                public_key_raw = b64decode(
                    encoded_public.encode("ascii"), validate=True
                )
            except (BinasciiError, UnicodeEncodeError) as exc:
                raise SigningBrokerError(
                    f"Signing broker returned a malformed {label} public key"
                ) from exc
            if len(public_key_raw) != 32:
                raise SigningBrokerError(
                    f"Signing broker returned a malformed {label} public key"
                )
            return public_key_raw
        else:
            raise SigningBrokerError(
                f"Signing broker returned a malformed {label} public key"
            )

    @classmethod
    def _decode_required_status_public_key(
        cls, encoded_public: object, *, label: str
    ) -> bytes:
        public_key = cls._decode_status_public_key(encoded_public, label=label)
        if public_key is None:
            raise SigningBrokerError(
                f"Signing broker returned a malformed {label} public key"
            )
        return public_key

    @property
    def capabilities(self) -> frozenset[str]:
        return self._status.capabilities

    @property
    def apply_public_key_raw(self) -> bytes | None:
        return self._status.apply_public_key_raw

    @property
    def eval_public_key_raw(self) -> bytes | None:
        return self._status.eval_public_key_raw

    @property
    def corpus_release_public_key_raw(self) -> bytes | None:
        return self._status.corpus_release_public_key_raw

    @property
    def corpus_release_public_keys_raw(self) -> tuple[bytes, ...]:
        return self._status.corpus_release_public_keys_raw

    @property
    def broker_pid(self) -> int | None:
        return self._broker_pid

    def apply_ed25519_sign(self, payload: bytes) -> bytes:
        return self._request_ed25519_signature("apply_ed25519_sign", payload)

    def eval_ed25519_sign(self, payload: bytes) -> bytes:
        return self._request_ed25519_signature("eval_ed25519_sign", payload)

    def _request_ed25519_signature(self, operation: str, payload: bytes) -> bytes:
        response = self._request(operation, payload=_encode_bytes(payload))
        if set(response) != {"signature"}:
            raise SigningBrokerError(
                "Signing broker returned a malformed Ed25519 signature"
            )
        encoded = response.get("signature")
        if not isinstance(encoded, str):
            raise SigningBrokerError(
                "Signing broker returned a malformed Ed25519 signature"
            )
        try:
            signature = b64decode(encoded.encode("ascii"), validate=True)
        except (BinasciiError, UnicodeEncodeError) as exc:
            raise SigningBrokerError(
                "Signing broker returned a malformed Ed25519 signature"
            ) from exc
        if len(signature) != 64:
            raise SigningBrokerError(
                "Signing broker returned a malformed Ed25519 signature"
            )
        return signature

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            if self._closed:
                return
            try:
                request_id = self._next_request_id
                self._next_request_id += 1
                _send_frame(
                    self._connection,
                    {"version": _PROTOCOL_VERSION, "id": request_id, "op": "shutdown"},
                )
            except (OSError, SigningBrokerError):
                pass
            finally:
                self._closed = True
                self._connection.close()

    def close_after_fork_in_child(self) -> None:
        """Drop the inherited socket without shutting down the parent's broker."""

        # An at-fork callback must not acquire ``self._lock``: another thread may
        # have held it at the instant of fork, leaving it permanently locked in
        # the child.  Closing the duplicated descriptor is process-local and the
        # parent retains its independent descriptor and live broker session.
        if self._closed:
            return
        self._closed = True
        self._connection.close()

    def _request(self, operation: str, **fields: object) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise SigningBrokerError("Signing broker connection is closed")
            request_id = self._next_request_id
            self._next_request_id += 1
            request = {
                "version": _PROTOCOL_VERSION,
                "id": request_id,
                "op": operation,
                **fields,
            }
            try:
                _send_frame(self._connection, request)
                response = _receive_frame(self._connection)
            except (EOFError, OSError, ValueError) as exc:
                raise SigningBrokerError("Signing broker connection failed") from exc
            if response.get("version") != _PROTOCOL_VERSION:
                raise SigningBrokerError("Signing broker protocol version mismatch")
            if response.get("id") != request_id:
                raise SigningBrokerError("Signing broker response identity mismatch")
            if response.get("ok") is not True:
                if set(response) != {"version", "id", "ok", "error", "result"}:
                    raise SigningBrokerError(
                        "Signing broker returned a malformed error response"
                    )
                detail = response.get("error")
                raise SigningBrokerError(
                    str(detail)
                    if isinstance(detail, str) and detail
                    else "Signing failed"
                )
            if set(response) != {"version", "id", "ok", "result"}:
                raise SigningBrokerError("Signing broker returned a malformed response")
            result = response.get("result")
            if not isinstance(result, dict):
                raise SigningBrokerError("Signing broker returned a malformed response")
            return result


_active_broker: SigningBroker | None = None
_active_broker_pid: int | None = None


def _drop_active_broker_after_fork_in_child() -> None:
    """Ensure a fork-only child cannot reuse the parent's signing capability."""

    global _active_broker, _active_broker_pid
    broker = _active_broker
    _active_broker = None
    _active_broker_pid = None
    if isinstance(broker, SigningBrokerClient):
        broker.close_after_fork_in_child()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_drop_active_broker_after_fork_in_child)


def install_signing_broker(
    client: SigningBroker, *, broker_pid: int | None = None
) -> None:
    """Install the sole process-local broker capability."""

    global _active_broker, _active_broker_pid
    if _active_broker is not None and _active_broker is not client:
        raise SigningBrokerError("A signing broker is already installed")
    _active_broker = client
    _active_broker_pid = broker_pid


def get_signing_broker(*, capability: str | None = None) -> SigningBroker:
    """Return the installed broker and optionally require one capability."""

    broker = _active_broker
    if broker is None:
        raise SigningBrokerError(
            "No trusted signing broker is attached; launch axiom-encode under "
            "a trusted supervisor that attaches one before the console entrypoint"
        )
    if capability is not None and capability not in broker.capabilities:
        raise SigningBrokerError(
            f"Trusted signing broker does not provide {capability!r}"
        )
    return broker


def attach_signing_broker_from_environment() -> SigningBrokerClient | None:
    """Attach the supervised CLI to its inherited broker descriptor."""

    marker = os.environ.get(BROKER_MARKER_ENV)
    descriptor_text = os.environ.get(BROKER_FD_ENV)
    pid_text = os.environ.get(BROKER_PID_ENV)
    if marker != "1":
        if BROKER_FD_ENV in os.environ or BROKER_PID_ENV in os.environ:
            raise SigningBrokerError("Incomplete signing-broker attachment marker")
        return None
    try:
        descriptor = int(descriptor_text or "")
        broker_pid = int(pid_text or "")
    except ValueError as exc:
        raise SigningBrokerError("Malformed signing-broker attachment marker") from exc
    if descriptor < 3 or broker_pid <= 0:
        raise SigningBrokerError("Malformed signing-broker attachment marker")
    for name in _BROKER_ENV_NAMES:
        os.environ.pop(name, None)
    os.set_inheritable(descriptor, False)
    _harden_signing_capability_process(role="client")
    connection = socket.socket(fileno=descriptor)
    try:
        _authenticate_broker_peer(connection, expected_pid=broker_pid)
    except Exception:
        connection.close()
        raise
    client = SigningBrokerClient(
        connection,
        broker_pid=broker_pid,
    )
    install_signing_broker(client, broker_pid=broker_pid)
    global _trusted_subprocess_base_environment
    _trusted_subprocess_base_environment = {
        name: os.environ[name]
        for name in _TRUSTED_SUBPROCESS_BASE_NAMES
        if name in os.environ
    }
    atexit.register(_close_active_production_broker)
    return client


def _authenticate_broker_peer(
    connection: socket.socket,
    *,
    expected_pid: int,
) -> None:
    """Authenticate the anonymous broker peer with kernel credentials."""

    previous_timeout = connection.gettimeout()
    connection.settimeout(5.0)
    try:
        if sys.platform.startswith("linux"):
            if not hasattr(socket, "SO_PASSCRED") or not hasattr(
                socket, "SCM_CREDENTIALS"
            ):
                raise SigningBrokerError(
                    "Linux signing broker credential messages unavailable"
                )
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            connection.sendall(b"\xa5")
            credential_size = struct.calcsize("3i")
            data, ancillary, flags, _address = connection.recvmsg(
                1,
                socket.CMSG_SPACE(credential_size),
            )
            credentials = [
                value
                for level, kind, value in ancillary
                if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS
            ]
            if (
                data != b"\x5a"
                or flags & socket.MSG_CTRUNC
                or len(credentials) != 1
                or len(credentials[0]) != credential_size
            ):
                raise SigningBrokerError("Signing broker peer credential is malformed")
            peer_pid, peer_uid, peer_gid = struct.unpack("3i", credentials[0])
            if (
                peer_pid != expected_pid
                or peer_uid != os.geteuid()
                or peer_gid != os.getegid()
            ):
                raise SigningBrokerError("Signing broker peer identity mismatch")
            return
        if sys.platform == "darwin":
            getpeereid = getattr(connection, "getpeereid", None)
            if getpeereid is not None:
                peer_uid, peer_gid = getpeereid()
                if peer_uid != os.geteuid() or peer_gid != os.getegid():
                    raise SigningBrokerError("Signing broker peer identity mismatch")
            # LOCAL_PEERPID is kernel-authenticated; unlike the environment
            # marker, it identifies the process holding the peer endpoint.
            peer_pid = ctypes.c_int()
            peer_pid_size = ctypes.c_uint32(ctypes.sizeof(peer_pid))
            libc = ctypes.CDLL(None, use_errno=True)
            if (
                libc.getsockopt(
                    connection.fileno(),
                    0,  # SOL_LOCAL
                    2,  # LOCAL_PEERPID
                    ctypes.byref(peer_pid),
                    ctypes.byref(peer_pid_size),
                )
                != 0
                or peer_pid_size.value != ctypes.sizeof(peer_pid)
                or peer_pid.value != expected_pid
            ):
                raise SigningBrokerError("Signing broker peer identity mismatch")
            return
        raise SigningBrokerError("Signing broker peer authentication is unsupported")
    finally:
        connection.settimeout(previous_timeout)


def _close_active_production_broker() -> None:
    global _active_broker
    broker = _active_broker
    _active_broker = None
    if isinstance(broker, SigningBrokerClient):
        broker.close()


def _send_frame(connection: socket.socket, payload: Mapping[str, object]) -> None:
    try:
        raw = json.dumps(
            dict(payload),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SigningBrokerError(
            "Signing broker request is not canonical JSON"
        ) from exc
    if len(raw) > _MAX_FRAME_BYTES:
        raise SigningBrokerError("Signing broker frame exceeds its size limit")
    connection.sendall(struct.pack(">I", len(raw)) + raw)


def _receive_frame(connection: socket.socket) -> dict[str, Any]:
    header = _receive_exact(connection, 4)
    length = struct.unpack(">I", header)[0]
    if length > _MAX_FRAME_BYTES:
        raise SigningBrokerError("Signing broker frame exceeds its size limit")
    raw = _receive_exact(connection, length)
    try:

        def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError("duplicate object key")
                result[name] = value
            return result

        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=exact_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SigningBrokerError("Signing broker frame is malformed") from exc
    if not isinstance(payload, dict):
        raise SigningBrokerError("Signing broker frame must be an object")
    return payload


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("Signing broker connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _encode_bytes(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise TypeError("Signing payload must be bytes")
    return b64encode(payload).decode("ascii")


def _harden_signing_capability_process(*, role: str) -> None:
    """Deny core dumps and same-user debugger attachment for one capability."""

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        # Linux prctl(PR_SET_DUMPABLE, 0) blocks /proc/<pid>/mem and ptrace from
        # sibling model processes even on hosts without Yama ptrace_scope.
        if libc.prctl(4, 0, 0, 0, 0) != 0:
            raise SigningBrokerError(
                f"Could not disable signing-broker {role} process dumping"
            )
    elif sys.platform == "darwin":
        # Darwin ptrace(PT_DENY_ATTACH, ...) provides the corresponding
        # same-user debugger boundary.
        if libc.ptrace(31, 0, None, 0) != 0:
            raise SigningBrokerError(
                f"Could not deny signing-broker {role} debugger attachment"
            )
