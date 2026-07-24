"""Shared signed release-object/v2 fixtures for corpus consumer tests."""

from __future__ import annotations

import hashlib
import json
from base64 import b64encode
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from axiom_encode.corpus_release import canonical_release_object_bytes
from axiom_encode.corpus_resolver import LocalCorpusRelease

_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x17" * 32)
TEST_RELEASE_PUBLIC_KEY = b64encode(
    _PRIVATE_KEY.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
).decode()


def bind_test_corpus_release(
    corpus_root: Path,
    name: str,
    scopes: list[tuple[str, str, str]],
) -> LocalCorpusRelease:
    """Create and load one signed v2 release over canonical local artifacts."""

    content_sha256 = write_test_release_object(corpus_root, name, scopes)
    return LocalCorpusRelease(
        corpus_root,
        name,
        content_sha256,
        TEST_RELEASE_PUBLIC_KEY,
    )


def write_test_release_object(
    corpus_root: Path,
    name: str,
    scopes: list[tuple[str, str, str]],
    *,
    git_commit: str = "a" * 40,
) -> str:
    """Write a production-shaped signed release object and return its identity."""

    artifact_rows: list[dict[str, object]] = []
    scope_rows: list[dict[str, object]] = []
    for jurisdiction, document_class, version in scopes:
        prefix = Path(jurisdiction) / document_class / version
        provisions = (
            corpus_root
            / "data"
            / "corpus"
            / "provisions"
            / prefix.with_suffix(".jsonl")
        )
        if not provisions.is_file():
            raise AssertionError(
                f"test release requires canonical provisions artifact: {provisions}"
            )
        row_count = sum(
            1 for line in provisions.read_bytes().splitlines() if line.strip()
        )
        if row_count <= 0:
            raise AssertionError(
                f"test release provisions artifact must contain rows: {provisions}"
            )
        scope_identity = f"{jurisdiction}/{document_class}/{version}"
        provision_projection_sha256 = hashlib.sha256(
            f"{scope_identity}:provision-projection".encode()
        ).hexdigest()
        navigation_projection_sha256 = hashlib.sha256(
            f"{scope_identity}:navigation-projection".encode()
        ).hexdigest()
        inventory = (
            corpus_root / "data" / "corpus" / "inventory" / prefix.with_suffix(".json")
        )
        coverage = (
            corpus_root / "data" / "corpus" / "coverage" / prefix.with_suffix(".json")
        )
        source = (
            corpus_root / "data" / "corpus" / "sources" / prefix / "fixture-source.txt"
        )
        for path, body in (
            (inventory, "{}\n"),
            (coverage, "{}\n"),
            (source, "authoritative fixture source\n"),
        ):
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body, encoding="utf-8")
        scope_rows.append(
            {
                "jurisdiction": jurisdiction,
                "document_class": document_class,
                "version": version,
                "provision_rows": row_count,
                "navigation_rows": row_count,
                "provision_projection_sha256": provision_projection_sha256,
                "navigation_projection_sha256": navigation_projection_sha256,
            }
        )
        for artifact_class, paths in (
            ("inventory", [inventory]),
            ("provisions", [provisions]),
            ("coverage", [coverage]),
            (
                "sources",
                sorted(path for path in source.parent.rglob("*") if path.is_file()),
            ),
        ):
            for path in paths:
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                entry: dict[str, object] = {
                    "artifact_class": artifact_class,
                    "path": path.relative_to(corpus_root).as_posix(),
                    "sha256": digest,
                    "bytes": len(raw),
                    "r2_bucket": "axiom-corpus",
                    "r2_key": f"objects/sha256/{digest[:2]}/{digest}",
                }
                if artifact_class == "provisions":
                    entry["rows"] = row_count
                artifact_rows.append(entry)
    artifact_rows.sort(key=lambda item: str(item["path"]))
    selector = {
        "name": name,
        "scopes": [
            {
                "jurisdiction": scope["jurisdiction"],
                "document_class": scope["document_class"],
                "version": scope["version"],
            }
            for scope in scope_rows
        ],
    }
    content = {
        "release": name,
        "created_at": "2026-07-10T00:00:00Z",
        "selector_sha256": _canonical_sha256(selector),
        "corpus_base": "data/corpus",
        "git": {
            "commit": git_commit,
            "committed_at": "2026-07-10T00:00:00Z",
        },
        "r2": {"bucket": "axiom-corpus", "addressing": "sha256"},
        "scopes": scope_rows,
        "artifacts": artifact_rows,
        "validation": {
            "passed": True,
            "deep_validation": {
                "error_count": 0,
                "warning_count": 0,
                "scope_count": len(scope_rows),
            },
            "r2_readback": {
                "bucket": "axiom-corpus",
                "artifact_count": len(artifact_rows),
                "artifact_bytes": sum(int(item["bytes"]) for item in artifact_rows),
                "verified_keys": [str(item["r2_key"]) for item in artifact_rows],
            },
            "supabase_projection_evidence": [
                {
                    "jurisdiction": scope["jurisdiction"],
                    "document_class": scope["document_class"],
                    "version": scope["version"],
                    "expected": scope["provision_rows"],
                    "actual": scope["provision_rows"],
                    "expected_navigation": scope["navigation_rows"],
                    "actual_navigation": scope["navigation_rows"],
                    "expected_provision_projection_sha256": scope[
                        "provision_projection_sha256"
                    ],
                    "actual_provision_projection_sha256": scope[
                        "provision_projection_sha256"
                    ],
                    "expected_navigation_projection_sha256": scope[
                        "navigation_projection_sha256"
                    ],
                    "actual_navigation_projection_sha256": scope[
                        "navigation_projection_sha256"
                    ],
                }
                for scope in scope_rows
            ],
        },
    }
    payload = {
        "schema_version": "axiom-corpus/release-object/v2",
        "release": name,
        "content_sha256": _canonical_sha256(content),
        "content": content,
    }
    payload["signature"] = {
        "algorithm": "ed25519",
        "key_id": "axiom-corpus-release-v2",
        "value": b64encode(
            _PRIVATE_KEY.sign(canonical_release_object_bytes(payload))
        ).decode(),
    }
    release_object = (
        corpus_root / "releases" / name / f"{payload['content_sha256']}.json"
    )
    release_object.parent.mkdir(parents=True, exist_ok=True)
    release_object.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(payload["content_sha256"])


def _canonical_sha256(value: dict[str, object]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(raw).hexdigest()
