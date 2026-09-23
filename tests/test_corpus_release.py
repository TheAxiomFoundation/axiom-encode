"""Consumer conformance tests for signed axiom-corpus release objects."""

from __future__ import annotations

import copy
import hashlib
import json
from base64 import b64encode

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from axiom_encode.corpus_release import (
    COMPLETE_EXPRESSION_DATES_PROFILE,
    RELEASE_OBJECT_SCHEMA_V2,
    RELEASE_OBJECT_SCHEMA_VERSION,
    CorpusReleaseObjectError,
    canonical_release_object_bytes,
    verify_release_object,
)


def _canonical_sha256(payload: dict) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _signed_release_object(
    *,
    schema_version: str = RELEASE_OBJECT_SCHEMA_V2,
    private_key: Ed25519PrivateKey | None = None,
) -> tuple[dict, str]:
    private_key = private_key or Ed25519PrivateKey.generate()
    public_key = b64encode(
        private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    ).decode()
    release = "nz-rulespec-2026-07-10"
    version = "2026-07-10-nz-rulespec"
    prefix = f"nz/statute/{version}"
    paths = [
        ("coverage", f"data/corpus/coverage/{prefix}.json", 10, None),
        ("inventory", f"data/corpus/inventory/{prefix}.json", 20, None),
        ("provisions", f"data/corpus/provisions/{prefix}.jsonl", 30, 1),
        ("sources", f"data/corpus/sources/{prefix}/act.html", 40, None),
    ]
    artifacts = []
    for index, (artifact_class, path, byte_count, rows) in enumerate(paths):
        digest = hashlib.sha256(f"artifact-{index}".encode()).hexdigest()
        artifact = {
            "artifact_class": artifact_class,
            "path": path,
            "sha256": digest,
            "bytes": byte_count,
            "r2_bucket": "axiom-corpus",
            "r2_key": f"objects/sha256/{digest[:2]}/{digest}",
        }
        if rows is not None:
            artifact["rows"] = rows
        artifacts.append(artifact)
    provision_projection_sha256 = hashlib.sha256(
        b"fixture provision projection"
    ).hexdigest()
    navigation_projection_sha256 = hashlib.sha256(
        b"fixture navigation projection"
    ).hexdigest()
    scopes = [
        {
            "jurisdiction": "nz",
            "document_class": "statute",
            "version": version,
            "provision_rows": 1,
            "navigation_rows": 1,
            "provision_projection_sha256": provision_projection_sha256,
            "navigation_projection_sha256": navigation_projection_sha256,
        }
    ]
    selector = {
        "name": release,
        "scopes": [
            {
                "jurisdiction": "nz",
                "document_class": "statute",
                "version": version,
            }
        ],
    }
    if schema_version == RELEASE_OBJECT_SCHEMA_VERSION:
        selector["quality_profile"] = COMPLETE_EXPRESSION_DATES_PROFILE
    selector_sha256 = _canonical_sha256(selector)
    content = {
        "release": release,
        "created_at": "2026-07-10T00:00:00Z",
        "selector_sha256": selector_sha256,
        "corpus_base": "data/corpus",
        "git": {
            "commit": "a" * 40,
            "committed_at": "2026-07-10T00:00:00Z",
        },
        "r2": {"bucket": "axiom-corpus", "addressing": "sha256"},
        "scopes": scopes,
        "artifacts": artifacts,
        "validation": {
            "passed": True,
            "deep_validation": {
                "error_count": 0,
                "warning_count": 0,
                "scope_count": 1,
            },
            "r2_readback": {
                "bucket": "axiom-corpus",
                "artifact_count": len(artifacts),
                "artifact_bytes": sum(item[2] for item in paths),
                "verified_keys": [item["r2_key"] for item in artifacts],
            },
            "supabase_projection_evidence": [
                {
                    "jurisdiction": "nz",
                    "document_class": "statute",
                    "version": version,
                    "expected": 1,
                    "actual": 1,
                    "expected_navigation": 1,
                    "actual_navigation": 1,
                    "expected_provision_projection_sha256": (
                        provision_projection_sha256
                    ),
                    "actual_provision_projection_sha256": (provision_projection_sha256),
                    "expected_navigation_projection_sha256": (
                        navigation_projection_sha256
                    ),
                    "actual_navigation_projection_sha256": (
                        navigation_projection_sha256
                    ),
                }
            ],
        },
    }
    if schema_version == RELEASE_OBJECT_SCHEMA_VERSION:
        content["quality_profile"] = COMPLETE_EXPRESSION_DATES_PROFILE
        content["validation"]["quality_profile"] = COMPLETE_EXPRESSION_DATES_PROFILE
    payload = {
        "schema_version": schema_version,
        "release": release,
        "content_sha256": _canonical_sha256(content),
        "content": content,
    }
    payload["signature"] = {
        "algorithm": "ed25519",
        "key_id": "axiom-corpus-release-v2",
        "value": b64encode(
            private_key.sign(canonical_release_object_bytes(payload))
        ).decode(),
    }
    return payload, public_key


def test_verified_release_object_exposes_only_signed_inventory() -> None:
    payload, public_key = _signed_release_object()

    verified = verify_release_object(payload, public_key=public_key)

    assert verified.name == "nz-rulespec-2026-07-10"
    assert verified.content_sha256 == payload["content_sha256"]
    assert verified.selector_sha256 == payload["content"]["selector_sha256"]
    assert (
        verified.scopes[0].provision_projection_sha256
        == payload["content"]["scopes"][0]["provision_projection_sha256"]
    )
    assert (
        verified.scopes[0].navigation_projection_sha256
        == payload["content"]["scopes"][0]["navigation_projection_sha256"]
    )
    assert [item.artifact_class for item in verified.artifacts] == [
        "coverage",
        "inventory",
        "provisions",
        "sources",
    ]


def test_verified_release_object_accepts_profiled_v3() -> None:
    payload, public_key = _signed_release_object(
        schema_version=RELEASE_OBJECT_SCHEMA_VERSION
    )

    verified = verify_release_object(payload, public_key=public_key)

    assert verified.content_sha256 == payload["content_sha256"]
    assert payload["content"]["quality_profile"] == (COMPLETE_EXPRESSION_DATES_PROFILE)


def test_release_object_verifies_with_current_or_retired_key(caplog) -> None:
    current_private_key = Ed25519PrivateKey.generate()
    retired_private_key = Ed25519PrivateKey.generate()
    old_payload, retired_public_key = _signed_release_object(
        private_key=retired_private_key
    )
    new_payload, current_public_key = _signed_release_object(
        private_key=current_private_key
    )
    keyring = (current_public_key, retired_public_key)

    with caplog.at_level("DEBUG", logger="axiom_encode.corpus_release"):
        old_verified = verify_release_object(old_payload, public_key=keyring)
        new_verified = verify_release_object(new_payload, public_key=keyring)

    assert old_verified.content_sha256 == old_payload["content_sha256"]
    assert new_verified.content_sha256 == new_payload["content_sha256"]
    assert "release object signature verified with key index 1" in caplog.messages
    assert "release object signature verified with key index 0" in caplog.messages


def test_release_object_rejects_empty_keyring() -> None:
    payload, _public_key = _signed_release_object()

    with pytest.raises(CorpusReleaseObjectError, match="keyring must not be empty"):
        verify_release_object(payload, public_key=())


@pytest.mark.parametrize("schema_version", [[], {}])
def test_release_object_rejects_non_scalar_schema_version(
    schema_version: object,
) -> None:
    payload, public_key = _signed_release_object()
    payload["schema_version"] = schema_version

    with pytest.raises(CorpusReleaseObjectError, match="unsupported schema version"):
        verify_release_object(payload, public_key=public_key)


@pytest.mark.parametrize("profile", [None, "different-profile"])
def test_v3_release_object_requires_supported_content_profile(
    profile: str | None,
) -> None:
    payload, public_key = _signed_release_object(
        schema_version=RELEASE_OBJECT_SCHEMA_VERSION
    )
    if profile is None:
        payload["content"].pop("quality_profile")
    else:
        payload["content"]["quality_profile"] = profile
    payload["content_sha256"] = _canonical_sha256(payload["content"])

    with pytest.raises(
        CorpusReleaseObjectError,
        match="content does not match|unsupported quality profile",
    ):
        verify_release_object(payload, public_key=public_key)


@pytest.mark.parametrize("profile", [None, "different-profile"])
def test_v3_release_object_requires_matching_validation_profile(
    profile: str | None,
) -> None:
    payload, public_key = _signed_release_object(
        schema_version=RELEASE_OBJECT_SCHEMA_VERSION
    )
    validation = payload["content"]["validation"]
    if profile is None:
        validation.pop("quality_profile")
    else:
        validation["quality_profile"] = profile
    payload["content_sha256"] = _canonical_sha256(payload["content"])

    with pytest.raises(
        CorpusReleaseObjectError,
        match="validation does not match|quality profile is inconsistent",
    ):
        verify_release_object(payload, public_key=public_key)


def test_v3_selector_sha256_attests_quality_profile() -> None:
    payload, public_key = _signed_release_object(
        schema_version=RELEASE_OBJECT_SCHEMA_VERSION
    )
    payload["content"]["selector_sha256"] = _canonical_sha256(
        {
            "name": payload["release"],
            "scopes": [
                {
                    key: scope[key]
                    for key in ("jurisdiction", "document_class", "version")
                }
                for scope in payload["content"]["scopes"]
            ],
        }
    )
    payload["content_sha256"] = _canonical_sha256(payload["content"])

    with pytest.raises(CorpusReleaseObjectError, match="selector sha256"):
        verify_release_object(payload, public_key=public_key)


def test_release_object_rejects_content_tamper() -> None:
    payload, public_key = _signed_release_object()
    payload["content"]["scopes"][0]["provision_rows"] = 2

    with pytest.raises(CorpusReleaseObjectError, match="content sha256"):
        verify_release_object(payload, public_key=public_key)


def test_release_object_rejects_wrong_public_key() -> None:
    payload, _public_key = _signed_release_object()
    wrong_key = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    )

    with pytest.raises(CorpusReleaseObjectError, match="signature is invalid"):
        verify_release_object(payload, public_key=b64encode(wrong_key).decode())


def test_release_object_rejects_artifact_outside_signed_scopes() -> None:
    payload, public_key = _signed_release_object()
    forged = copy.deepcopy(payload)
    extra = copy.deepcopy(forged["content"]["artifacts"][-1])
    extra["path"] = "data/corpus/sources/nz/statute/zzzz/other.html"
    forged["content"]["artifacts"].append(extra)
    forged["content_sha256"] = _canonical_sha256(forged["content"])

    with pytest.raises(CorpusReleaseObjectError, match="outside its declared scopes"):
        verify_release_object(forged, public_key=public_key)


def test_release_object_rejects_mutable_current_name() -> None:
    payload, public_key = _signed_release_object()
    payload["release"] = "current"

    with pytest.raises(CorpusReleaseObjectError, match="legacy mutable"):
        verify_release_object(payload, public_key=public_key)


@pytest.mark.parametrize("release", ["nz.rulespec", "nz_rulespec", "nz--rulespec"])
def test_release_object_rejects_noncanonical_release_name(release: str) -> None:
    payload, public_key = _signed_release_object()
    payload["release"] = release

    with pytest.raises(CorpusReleaseObjectError, match="invalid release name"):
        verify_release_object(payload, public_key=public_key)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jurisdiction", "../nz"),
        ("document_class", "unknown"),
        ("version", "../v1"),
    ],
)
def test_release_object_rejects_noncanonical_scope_identity(
    field: str, value: str
) -> None:
    payload, public_key = _signed_release_object()
    payload["content"]["scopes"][0][field] = value
    payload["content_sha256"] = _canonical_sha256(payload["content"])

    with pytest.raises(CorpusReleaseObjectError, match="invalid or duplicate scope"):
        verify_release_object(payload, public_key=public_key)


def test_release_object_rejects_rows_field_on_non_provisions_artifact() -> None:
    payload, public_key = _signed_release_object()
    payload["content"]["artifacts"][0]["rows"] = None
    payload["content_sha256"] = _canonical_sha256(payload["content"])

    with pytest.raises(CorpusReleaseObjectError, match="artifact does not match"):
        verify_release_object(payload, public_key=public_key)


def test_release_object_rejects_invalid_public_key_pem() -> None:
    payload, _public_key = _signed_release_object()

    with pytest.raises(CorpusReleaseObjectError, match="public key PEM is invalid"):
        verify_release_object(
            payload,
            public_key="-----BEGIN PUBLIC KEY-----\ninvalid\n-----END PUBLIC KEY-----",
        )


@pytest.mark.parametrize(
    "field",
    ["provision_projection_sha256", "navigation_projection_sha256"],
)
def test_release_object_rejects_scope_without_projection_digest(field: str) -> None:
    payload, public_key = _signed_release_object()
    payload["content"]["scopes"][0].pop(field)
    payload["content_sha256"] = _canonical_sha256(payload["content"])

    with pytest.raises(CorpusReleaseObjectError, match="scope does not match"):
        verify_release_object(payload, public_key=public_key)


def test_release_object_rejects_uppercase_projection_digest() -> None:
    payload, public_key = _signed_release_object()
    payload["content"]["scopes"][0]["provision_projection_sha256"] = "A" * 64
    payload["content_sha256"] = _canonical_sha256(payload["content"])

    with pytest.raises(CorpusReleaseObjectError, match="invalid projection sha256"):
        verify_release_object(payload, public_key=public_key)


def test_release_object_rejects_legacy_count_only_validation_shape() -> None:
    payload, public_key = _signed_release_object()
    evidence = payload["content"]["validation"].pop("supabase_projection_evidence")
    for item in evidence:
        for field in tuple(item):
            if "projection_sha256" in field:
                item.pop(field)
    payload["content"]["validation"]["supabase_counts"] = evidence
    payload["content_sha256"] = _canonical_sha256(payload["content"])

    with pytest.raises(CorpusReleaseObjectError, match="validation does not match"):
        verify_release_object(payload, public_key=public_key)


@pytest.mark.parametrize(
    "field",
    [
        "expected_provision_projection_sha256",
        "actual_provision_projection_sha256",
        "expected_navigation_projection_sha256",
        "actual_navigation_projection_sha256",
    ],
)
def test_release_object_rejects_projection_evidence_mismatch(field: str) -> None:
    payload, public_key = _signed_release_object()
    payload["content"]["validation"]["supabase_projection_evidence"][0][field] = (
        "f" * 64
    )
    payload["content_sha256"] = _canonical_sha256(payload["content"])

    with pytest.raises(CorpusReleaseObjectError, match="projection evidence"):
        verify_release_object(payload, public_key=public_key)


def test_canonical_bytes_exclude_signature_without_mutating_payload() -> None:
    payload, _public_key = _signed_release_object()
    snapshot = copy.deepcopy(payload)

    canonical = canonical_release_object_bytes(payload)

    assert payload == snapshot
    unsigned = {key: value for key, value in snapshot.items() if key != "signature"}
    assert canonical == json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    assert canonical == canonical_release_object_bytes(unsigned)


def _reference_scope_artifact_membership(scopes, artifacts) -> None:
    """The per-scope full scan that the grouped membership check replaced."""

    by_path = {artifact.path: artifact for artifact in artifacts}
    claimed: set[str] = set()
    for scope in scopes:
        prefix = f"{scope.jurisdiction}/{scope.document_class}/{scope.version}"
        required = {
            "inventory": f"data/corpus/inventory/{prefix}.json",
            "provisions": f"data/corpus/provisions/{prefix}.jsonl",
            "coverage": f"data/corpus/coverage/{prefix}.json",
        }
        for artifact_class, path in required.items():
            artifact = by_path.get(path)
            if artifact is None or artifact.artifact_class != artifact_class:
                raise CorpusReleaseObjectError(
                    f"release scope lacks its {artifact_class} artifact: {prefix}"
                )
            claimed.add(path)
        if by_path[required["provisions"]].row_count != scope.provision_rows:
            raise CorpusReleaseObjectError(
                "release scope row count does not match its provisions artifact: "
                f"{prefix}"
            )
        source_prefix = f"data/corpus/sources/{prefix}/"
        source_paths = [
            path
            for path, artifact in by_path.items()
            if path.startswith(source_prefix) and artifact.artifact_class == "sources"
        ]
        if not source_paths:
            raise CorpusReleaseObjectError(
                f"release scope lacks source artifacts: {prefix}"
            )
        claimed.update(source_paths)
    if set(by_path) != claimed:
        raise CorpusReleaseObjectError(
            "release object contains artifacts outside its declared scopes"
        )


def _membership_outcome(check, scopes, artifacts) -> str | None:
    try:
        check(scopes, artifacts)
    except CorpusReleaseObjectError as exc:
        return str(exc)
    return None


def _random_membership_case(rng):
    from axiom_encode.corpus_release import (
        VerifiedReleaseArtifact,
        VerifiedReleaseScope,
    )

    versions = ("v1", "v10", "v1.0", "v1-a", "v2")
    identities = rng.sample(
        [
            (jurisdiction, document_class, version)
            for jurisdiction in ("us", "us-ca")
            for document_class in ("statute", "manual")
            for version in versions
        ],
        rng.randint(1, 6),
    )
    scopes = [
        VerifiedReleaseScope(*identity, rng.randint(1, 3), "a" * 64, "b" * 64)
        for identity in identities
    ]
    artifacts: dict[str, VerifiedReleaseArtifact] = {}

    def add(artifact_class: str, path: str, rows: int | None = None) -> None:
        artifacts[path] = VerifiedReleaseArtifact(
            artifact_class, path, "c" * 64, 1, rows
        )

    for scope in scopes:
        prefix = f"{scope.jurisdiction}/{scope.document_class}/{scope.version}"
        if rng.random() > 0.05:
            add("inventory", f"data/corpus/inventory/{prefix}.json")
        if rng.random() > 0.05:
            rows = scope.provision_rows if rng.random() > 0.05 else 99
            add("provisions", f"data/corpus/provisions/{prefix}.jsonl", rows)
        if rng.random() > 0.05:
            add("coverage", f"data/corpus/coverage/{prefix}.json")
        for _ in range(rng.choice((0, 1, 1, 2, 3))):
            leaf = rng.choice(("act.html", "a/b.xml", "a/b/c.pdf", "x"))
            add("sources", f"data/corpus/sources/{prefix}/{leaf}")
    for _ in range(rng.choice((0, 0, 0, 1))):
        jurisdiction = rng.choice(("us", "us-ca"))
        document_class = rng.choice(("statute", "manual"))
        version = rng.choice(versions)
        prefix = f"{jurisdiction}/{document_class}/{version}"
        add(
            rng.choice(("sources", "inventory")),
            rng.choice(
                (
                    f"data/corpus/sources/{prefix}/stray.html",
                    f"data/corpus/sources/{prefix}",
                    f"data/corpus/sources/{jurisdiction}/{document_class}/x.html",
                    f"data/corpus/sources/{prefix}x/stray.html",
                )
            ),
        )
    ordered = sorted(artifacts.values(), key=lambda artifact: artifact.path)
    return tuple(scopes), tuple(ordered)


def test_grouped_membership_check_matches_reference_scan() -> None:
    import random

    from axiom_encode.corpus_release import _validate_scope_artifact_membership

    rng = random.Random(20260923)
    outcomes: set[str | None] = set()
    for _ in range(3000):
        scopes, artifacts = _random_membership_case(rng)
        expected = _membership_outcome(
            _reference_scope_artifact_membership, scopes, artifacts
        )
        assert (
            _membership_outcome(_validate_scope_artifact_membership, scopes, artifacts)
            == expected
        )
        outcomes.add(expected if expected is None else expected.split(":")[0])
    assert outcomes == {
        None,
        "release scope lacks its inventory artifact",
        "release scope lacks its provisions artifact",
        "release scope lacks its coverage artifact",
        "release scope row count does not match its provisions artifact",
        "release scope lacks source artifacts",
        "release object contains artifacts outside its declared scopes",
    }


def test_membership_does_not_credit_sibling_version_sources() -> None:
    from axiom_encode.corpus_release import (
        VerifiedReleaseArtifact,
        VerifiedReleaseScope,
        _validate_scope_artifact_membership,
    )

    def artifact(artifact_class: str, path: str, rows: int | None = None):
        return VerifiedReleaseArtifact(artifact_class, path, "c" * 64, 1, rows)

    scopes = tuple(
        VerifiedReleaseScope("us", "statute", version, 1, "a" * 64, "b" * 64)
        for version in ("v1", "v10")
    )
    artifacts = []
    for version in ("v1", "v10"):
        artifacts += [
            artifact("inventory", f"data/corpus/inventory/us/statute/{version}.json"),
            artifact(
                "provisions", f"data/corpus/provisions/us/statute/{version}.jsonl", 1
            ),
            artifact("coverage", f"data/corpus/coverage/us/statute/{version}.json"),
        ]
    artifacts.append(artifact("sources", "data/corpus/sources/us/statute/v10/a.html"))

    with pytest.raises(
        CorpusReleaseObjectError, match="lacks source artifacts: us/statute/v1$"
    ):
        _validate_scope_artifact_membership(
            scopes, tuple(sorted(artifacts, key=lambda item: item.path))
        )
