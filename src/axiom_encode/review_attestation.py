"""Verify a signature over a candidate review record (phase 3 of #2319).

A candidate review record (``candidate_review_record``) is not signed: who can
edit the candidate pull request body can write one. A *review attestation* is
an Ed25519 signature over the record by a key that is enrolled in the review
attestation root. This module holds the verifying half: the message that is
signed, the attestation, the root, and the checks of the signing run.

**The signing half is not here, and no command signs.** Evidence that a
reviewer ran needs a signer that the account which runs the encoder cannot
use directly. The signing supervisor is not that: it refuses to run as root
and runs with the uid of the user and of the encoder child
(``cmd/axiom-encode-signing-supervisor/execution_trust.go``), so a key that
the supervisor can read is a key that the user, the child and every agent
session of that user can read. ``sign_record`` exists for the component that
will hold the key (docs/reviewed-pr-promotion.md, "Review attestation") and
for tests, which make their own throwaway keys.

The root ships in the package (``data/review_attestation_roots.json``), so it
is versioned with the encoder commit, needs a code-owner review to change,
and reaches the signing run through the attested runtime. It is not a
repository variable. An empty root admits nothing.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from importlib import resources
from typing import Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from axiom_encode.candidate_review_record import (
    ReviewRecordError,
    load_record_file_text,
    record_issues,
    record_sha256,
)

ATTESTATION_SCHEMA = "axiom-encode/review-attestation/v1"
ROOTS_SCHEMA = "axiom-encode/review-attestation-roots/v1"
ROOTS_RESOURCE = "review_attestation_roots.json"
BEGIN_MARKER = "<!-- axiom-encode:candidate-review-attestation:v1:begin -->"
END_MARKER = "<!-- axiom-encode:candidate-review-attestation:v1:end -->"
_MARKER_PREFIX = "<!-- axiom-encode:candidate-review-attestation:"
# The signature covers this domain line and the record digest, so a signature
# made for another purpose with the same key is never a review attestation.
_DOMAIN = b"axiom-encode/review-attestation/v1\n"
MAX_ROOT_KEYS = 64
MAX_ATTESTATIONS_PER_BODY = 64

_ATTESTATION_FIELDS = frozenset({"schema", "key_id", "record_sha256", "signature"})
_ROOT_KEY_FIELDS = frozenset({"key_id", "ed25519_public_key", "owner", "enrolled"})
_KEY_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._@/-]{0,99}")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _raw_public_key(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)


def key_id(public_key: Ed25519PublicKey) -> str:
    """The identifier of a key: ``sha256:`` and the digest of its 32 raw bytes."""

    return "sha256:" + hashlib.sha256(_raw_public_key(public_key)).hexdigest()


def attestation_message(record_digest: str) -> bytes:
    """The exact bytes that a review attestation signs."""

    if _SHA256.fullmatch(record_digest) is None:
        raise ReviewRecordError("the record digest is not a SHA-256")
    return _DOMAIN + record_digest.encode("ascii") + b"\n"


def sign_record(record: object, private_key: Ed25519PrivateKey) -> dict[str, str]:
    """Sign one well-formed record. For the key holder and for tests only.

    The caller must have run the reviewer itself. A signature over a record
    that the caller was merely given proves nothing about a review.
    """

    issues = record_issues(record)
    if issues:
        raise ReviewRecordError(
            "the review record cannot be attested: " + "; ".join(issues)
        )
    digest = record_sha256(record)
    return {
        "schema": ATTESTATION_SCHEMA,
        "key_id": key_id(private_key.public_key()),
        "record_sha256": digest,
        "signature": base64.b64encode(
            private_key.sign(attestation_message(digest))
        ).decode("ascii"),
    }


def load_roots(text: str) -> dict[str, Ed25519PublicKey]:
    """Read the review attestation root: key id -> public key, strictly.

    Refused: another schema or field set, more than ``MAX_ROOT_KEYS`` keys, a
    key that is not 32 raw bytes in canonical base64, a key id that is not the
    digest of its key, a key or a key id that is listed twice, an owner or an
    enrolment date that is not plain.
    """

    document = load_record_file_text(text)
    keys = document.get("keys")
    if (
        set(document) != {"schema", "keys"}
        or document["schema"] != ROOTS_SCHEMA
        or not isinstance(keys, list)
        or len(keys) > MAX_ROOT_KEYS
    ):
        raise ReviewRecordError(f"the file is not a {ROOTS_SCHEMA} document")
    roots: dict[str, Ed25519PublicKey] = {}
    for entry in keys:
        if not isinstance(entry, dict) or set(entry) != _ROOT_KEY_FIELDS:
            raise ReviewRecordError("a review attestation root entry is malformed")
        encoded = entry["ed25519_public_key"]
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (TypeError, ValueError, binascii.Error) as exc:
            raise ReviewRecordError(
                "a review attestation root key is not base64"
            ) from exc
        if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != encoded:
            raise ReviewRecordError(
                "a review attestation root key is not 32 raw bytes in canonical base64"
            )
        public_key = Ed25519PublicKey.from_public_bytes(raw)
        identifier = entry["key_id"]
        if identifier != key_id(public_key):
            raise ReviewRecordError(
                "a review attestation root key id is not the digest of its key"
            )
        if (
            not isinstance(entry["owner"], str)
            or _OWNER.fullmatch(entry["owner"]) is None
            or not isinstance(entry["enrolled"], str)
            or _DATE.fullmatch(entry["enrolled"]) is None
        ):
            raise ReviewRecordError(
                "a review attestation root entry needs a plain owner and an "
                "enrolment date"
            )
        if identifier in roots:
            raise ReviewRecordError("a review attestation root key is listed twice")
        roots[identifier] = public_key
    return roots


def shipped_roots() -> dict[str, Ed25519PublicKey]:
    """The root of this encoder commit, from the package's own data."""

    text = (
        resources.files("axiom_encode")
        .joinpath("data", ROOTS_RESOURCE)
        .read_text(encoding="utf-8")
    )
    return load_roots(text)


def attestation_issues(
    record: object,
    attestation: object,
    roots: Mapping[str, Ed25519PublicKey],
) -> list[str]:
    """Return every reason ``attestation`` does not attest ``record``."""

    if (
        not isinstance(attestation, dict)
        or set(attestation) != _ATTESTATION_FIELDS
        or attestation["schema"] != ATTESTATION_SCHEMA
        or not isinstance(attestation["key_id"], str)
        or _KEY_ID.fullmatch(attestation["key_id"]) is None
        or not isinstance(attestation["record_sha256"], str)
        or _SHA256.fullmatch(attestation["record_sha256"]) is None
        or not isinstance(attestation["signature"], str)
    ):
        return ["the review attestation is malformed"]
    if record_issues(record):
        return ["the attested review record is not well formed"]
    digest = record_sha256(record)
    if attestation["record_sha256"] != digest:
        return ["the review attestation is for another record"]
    public_key = roots.get(attestation["key_id"])
    if public_key is None:
        return [
            f"the review attestation key {attestation['key_id']} is not enrolled "
            "in the review attestation root"
        ]
    try:
        signature = base64.b64decode(attestation["signature"], validate=True)
        public_key.verify(signature, attestation_message(digest))
    except (ValueError, binascii.Error, InvalidSignature):
        return ["the review attestation signature does not verify"]
    return []


def require_attested_records(
    records: Sequence[dict],
    attestations: Sequence[object],
    roots: Mapping[str, Ed25519PublicKey],
) -> list[dict]:
    """Return the attestation of every record, or refuse the whole set.

    Each record needs exactly one attestation that names its digest, and that
    attestation must verify against an enrolled key. An empty root refuses
    every record: nobody is enrolled.
    """

    admitted: list[dict] = []
    refusals: list[str] = []
    for record in records:
        digest = record_sha256(record)
        named = [
            attestation
            for attestation in attestations
            if isinstance(attestation, dict)
            and attestation.get("record_sha256") == digest
        ]
        label = f"{record.get('citation')} ({record.get('rulespec_path')})"
        if len(named) != 1:
            refusals.append(
                f"{label}: {len(named)} review attestations name the record; "
                "exactly one is required"
            )
            continue
        issues = attestation_issues(record, named[0], roots)
        if issues:
            refusals.append(f"{label}: " + "; ".join(issues))
        else:
            admitted.append(named[0])
    if refusals:
        raise ReviewRecordError(
            "a review attestation by an enrolled key is required for every "
            "unit and is missing: " + " | ".join(refusals)
        )
    return admitted


def render_attestation_block(attestation: Mapping[str, object]) -> str:
    """The fixed block of one attestation for a pull request body."""

    if (
        set(attestation) != _ATTESTATION_FIELDS
        or attestation.get("schema") != ATTESTATION_SCHEMA
    ):
        raise ReviewRecordError("the review attestation cannot be rendered")
    from axiom_encode.candidate_review_record import canonical_json

    return (
        "\n".join(
            [
                BEGIN_MARKER,
                "```json",
                canonical_json(dict(attestation)),
                "```",
                END_MARKER,
            ]
        )
        + "\n"
    )


def extract_attestations(body: object) -> list[dict]:
    """Read every attestation block from a pull request body.

    A block is exactly five lines: the begin marker, a ``json`` fence, one
    line of JSON, the closing fence, the end marker. Any other use of a
    marker refuses the whole body. A body without a marker has none.
    """

    if not isinstance(body, str):
        raise ReviewRecordError("the pull request body is not text")
    lines = [
        line.strip()
        for line in body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    found: list[dict] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line == BEGIN_MARKER:
            block = lines[index : index + 5]
            if (
                len(block) != 5
                or block[1] != "```json"
                or block[3] != "```"
                or block[4] != END_MARKER
            ):
                raise ReviewRecordError(
                    "a review attestation block is not in its fixed five-line form"
                )
            found.append(load_record_file_text(block[2]))
            if len(found) > MAX_ATTESTATIONS_PER_BODY:
                raise ReviewRecordError("the body holds too many review attestations")
            index += 5
            continue
        if _MARKER_PREFIX in line:
            raise ReviewRecordError(
                "the pull request body holds a review attestation marker "
                "outside a block"
            )
        index += 1
    return found
