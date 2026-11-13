"""Bounded artifact transport; these bytes never supply signature authority."""

from __future__ import annotations

import base64

from ._schema import canonical_object, decode_base64, digest, fields
from .canonical import jcs_dumps
from .github_publication import _NAME
from .identity import IdentityRefusal

BUNDLE_ARTIFACT = "axiom-notary-signed"


def pack_bundle(kind: str, address: str, files: dict[str, bytes]) -> bytes:
    raw = jcs_dumps(
        {
            "schema": "axiom/notary-bundle-transport/v1",
            "kind": kind,
            "artifact_sha256": address,
            "files": [
                {"path": path, "base64": base64.b64encode(value).decode()}
                for path, value in sorted(files.items())
            ],
        }
    )
    unpack_bundle(raw)
    return raw


def unpack_bundle(raw: bytes) -> tuple[str, str, dict[str, bytes]]:
    if len(raw) > 7_000_000:
        raise IdentityRefusal("bundle_size")
    body = canonical_object(raw)
    if (
        not fields(body, {"schema", "kind", "artifact_sha256", "files"})
        or body["schema"] != "axiom/notary-bundle-transport/v1"
        or body["kind"] not in {"receipt", "transition", "genesis"}
        or not digest(body["artifact_sha256"])
        or not isinstance(body["files"], list)
        or not body["files"]
    ):
        raise IdentityRefusal("bundle_schema")
    files = {}
    previous = ""
    for row in body["files"]:
        if not fields(row, {"path", "base64"}) or not isinstance(row["path"], str):
            raise IdentityRefusal("bundle_file")
        path = row["path"]
        value = decode_base64(row["base64"])
        if (
            path <= previous
            or path == "HEAD.json"
            or not _NAME.fullmatch(path)
            or value is None
        ):
            raise IdentityRefusal("bundle_file")
        previous = path
        files[path] = value
    return body["kind"], body["artifact_sha256"], files
