"""Typed, digest-bound human approval using a custodian-enrolled YubiHSM 2.

Hardware selection is a reference deployment proposal, not an assertion that
any production reviewer already owns an enrolled device. No key creation,
import, export, software-key fallback, or general signing command is exposed.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ._schema import digest, fields, lane_name
from .canonical import jcs_dumps, sha256_hex, strict_parse
from .deployment import custodian_file
from .identity import IdentityRefusal
from .lineage import CORRECTION, parse_record
from .protocol import parse_artifact
from .signatures import ROLE_SCOPES


def parse_device(raw: bytes):
    body = strict_parse(raw)
    if (
        not fields(
            body,
            {
                "schema",
                "role",
                "lane",
                "epoch_sha256",
                "serial",
                "auth_key_id",
                "signing_key_id",
                "spki_sha256",
                "custody_evidence_sha256",
            },
        )
        or body["schema"] != "axiom/notary-hardware-approval/v1"
    ):
        raise IdentityRefusal("hardware_configuration")
    if (
        body["role"] not in {"review", "approver", "admin-approver"}
        or not lane_name(body["lane"])
        or not body["lane"].startswith("TheAxiomFoundation/")
        or not all(
            digest(body[k])
            for k in ("epoch_sha256", "spki_sha256", "custody_evidence_sha256")
        )
    ):
        raise IdentityRefusal("hardware_binding")
    if (
        type(body["serial"]) is not int
        or body["serial"] <= 0
        or any(
            type(body[k]) is not int or not 0 < body[k] <= 65535
            for k in ("auth_key_id", "signing_key_id")
        )
    ):
        raise IdentityRefusal("hardware_identity")
    return body


def approval_subject(raw: bytes, expected_digest: str, device: dict):
    if not digest(expected_digest) or sha256_hex(raw) != expected_digest:
        raise IdentityRefusal("approval_digest")
    role = device["role"]
    if role == "review":
        body = parse_record(raw)
        if body is None or body["schema"] != CORRECTION:
            raise IdentityRefusal("approval_role")
    elif role == "approver":
        body = parse_artifact(raw, "receipt-candidate")
    else:
        body = parse_artifact(raw, "genesis") or parse_artifact(raw, "transition")
    if body is None or body["lane"] != device["lane"]:
        raise IdentityRefusal("approval_subject")
    epoch = (
        expected_digest
        if body["schema"] == "axiom/notary-genesis/v1"
        else body["epoch_sha256"]
    )
    if epoch != device["epoch_sha256"]:
        raise IdentityRefusal("approval_epoch")
    return body


def sign_with_device(
    raw: bytes, expected_digest: str, device: dict, password: str
) -> bytes:
    approval_subject(raw, expected_digest, device)
    try:
        from yubihsm import YubiHsm
        from yubihsm.defs import ALGORITHM, CAPABILITY, ORIGIN
        from yubihsm.objects import AsymmetricKey
    except ImportError:
        raise IdentityRefusal("hardware_dependency_missing") from None
    # Direct USB only. Caller cannot send authentication to a remote connector.
    hsm = YubiHsm.connect("yhusb://serial=" + str(device["serial"]))
    try:
        if hsm.get_device_info().serial != device["serial"]:
            raise IdentityRefusal("hardware_serial")
        session = hsm.create_session_derived(device["auth_key_id"], password)
        try:
            key = AsymmetricKey(session, device["signing_key_id"])
            info = key.get_info()
            if (
                info.algorithm != ALGORITHM.EC_ED25519
                or info.origin != ORIGIN.GENERATED
                or info.capabilities != CAPABILITY.SIGN_EDDSA
            ):
                # Requiring this exact capability also excludes wrapped export.
                raise IdentityRefusal("hardware_key_custody")
            public = key.get_public_key()
            if not isinstance(public, Ed25519PublicKey):
                raise IdentityRefusal("hardware_key_type")
            fingerprint = sha256_hex(
                public.public_bytes(
                    serialization.Encoding.DER,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            if fingerprint != device["spki_sha256"]:
                raise IdentityRefusal("hardware_public_key")
            scope = ROLE_SCOPES[device["role"]]
            frame = (
                b"axiom-encode/external-signer-sign/v2\0"
                + scope.encode()
                + b"\0"
                + expected_digest.encode()
            )
            signature = key.sign_eddsa(frame)
            try:
                public.verify(signature, frame)
            except (InvalidSignature, ValueError, TypeError):
                raise IdentityRefusal("hardware_signature") from None
            return jcs_dumps(
                {
                    "schema": "axiom/detached-signature/v1",
                    "body_sha256": expected_digest,
                    "scope": scope,
                    "signer_spki_sha256": fingerprint,
                    "signature_base64": base64.b64encode(signature).decode(),
                }
            )
        finally:
            session.close()
    finally:
        hsm.close()


def main():
    parser = argparse.ArgumentParser(
        description="Review and approve one exact notary candidate or correction with an enrolled hardware key"
    )
    parser.add_argument(
        "--device", required=True, help="Custodian-owned device configuration"
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--output", required=True, help="New public signature-sidecar file"
    )
    args = parser.parse_args()
    device = parse_device(custodian_file(args.device, limit=20000))
    with Path(args.candidate).open("rb") as stream:
        raw = stream.read(8_000_001)
    if len(raw) > 8_000_000:
        raise SystemExit("candidate is too large")
    body = approval_subject(raw, args.expected_sha256, device)
    print(
        f"Role: {device['role']}\nLane: {device['lane']}\nEpoch: {device['epoch_sha256']}\nCandidate SHA-256: {args.expected_sha256}\nHardware serial: {device['serial']}\nPublic key SPKI: {device['spki_sha256']}"
    )
    print(jcs_dumps(body).decode())
    if (
        input(
            "After reviewing the candidate, type its first 12 digest characters: "
        ).strip()
        != args.expected_sha256[:12]
    ):
        raise SystemExit("approval cancelled")
    password = getpass.getpass("Hardware authentication password: ")
    signature = sign_with_device(raw, args.expected_sha256, device, password)
    del password
    fd = os.open(
        args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644
    )
    with os.fdopen(fd, "wb") as stream:
        stream.write(signature)
    print(f"Wrote {device['role']} approval for {args.expected_sha256}")


if __name__ == "__main__":
    main()
