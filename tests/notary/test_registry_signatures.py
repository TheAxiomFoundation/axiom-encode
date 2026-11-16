"""Adversarial key separation and domain tests for v33 authentication."""

import base64
import itertools
import json
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key

from axiom_encode.notary.canonical import jcs_dumps, sha256_hex
from axiom_encode.notary.refusal import Refusal
from axiom_encode.notary.registry import ROLES, KeyRegistry, parse_registry
from axiom_encode.notary.signatures import ROLE_SCOPES, verify_detached
from tests.notary.lineage_fixtures import LANE, Identities, public_entry


@pytest.fixture
def identities():
    return Identities.create()


def test_valid_registry_and_read_only_keys(identities):
    registry = identities.registry()
    assert isinstance(registry, KeyRegistry)
    assert set(registry.keys) == set(ROLES)
    with pytest.raises(TypeError):
        registry.keys["producer"]["forged"] = object()


@pytest.mark.parametrize(
    "left,right",
    itertools.combinations((*ROLES, "legacy_apply_root", "legacy_eval_root"), 2),
)
def test_every_pair_of_roles_and_legacy_roots_must_be_distinct(identities, left, right):
    entry = public_entry(identities.keys[left])
    if right in ROLES:
        identities.body[right] = [entry]
        if right == "notary":
            identities.pins["notary_spki_sha256"] = entry["spki_sha256"]
    else:
        identities.pins[right] = entry["spki_sha256"]
    assert isinstance(identities.registry(), Refusal)


@pytest.mark.parametrize(
    "fault",
    [
        "unknown-field",
        "missing-role",
        "wrong-lane",
        "unknown-schema",
        "duplicate-key",
        "bad-fingerprint",
        "bad-base64",
        "unpadded-base64",
        "not-a-key",
        "rsa-key",
        "trailing-der",
        "wrong-notary-pin",
        "unsorted",
        "wrong-type",
    ],
)
def test_invalid_registry_refuses(identities, fault):
    body = identities.body
    entry = body["producer"][0]
    if fault == "unknown-field":
        body["allow"] = True
    elif fault == "missing-role":
        del body["review"]
    elif fault == "wrong-lane":
        body["lane"] = "elsewhere/rules"
    elif fault == "unknown-schema":
        body["schema"] += "2"
    elif fault == "duplicate-key":
        body["producer"].append(deepcopy(entry))
    elif fault == "bad-fingerprint":
        entry["spki_sha256"] = "0" * 64
    elif fault == "bad-base64":
        entry["public_key_spki_der_base64"] = "!!!!"
    elif fault == "unpadded-base64":
        entry["public_key_spki_der_base64"] = entry[
            "public_key_spki_der_base64"
        ].rstrip("=")
    elif fault in {"not-a-key", "trailing-der"}:
        der = (
            b"invalid"
            if fault == "not-a-key"
            else base64.b64decode(entry["public_key_spki_der_base64"]) + b"tail"
        )
        entry.update(
            spki_sha256=sha256_hex(der),
            public_key_spki_der_base64=base64.b64encode(der).decode(),
        )
    elif fault == "rsa-key":
        body["producer"] = [public_entry(generate_private_key(65537, 2048))]
    elif fault == "wrong-notary-pin":
        identities.pins["notary_spki_sha256"] = "0" * 64
    elif fault == "unsorted":
        body["producer"] = sorted(
            [entry, public_entry(Ed25519PrivateKey.generate())],
            key=lambda e: e["spki_sha256"],
            reverse=True,
        )
    elif fault == "wrong-type":
        body["producer"] = {"key": entry}
    result = identities.registry()
    assert isinstance(result, Refusal)
    assert result.code == "policy-invalid"


@pytest.mark.parametrize(
    "raw", [b'{"schema":1,"schema":2}', b'{"x":NaN}', b'{"x":"\\ud800"}', b"{}\n"]
)
def test_registry_rejects_noncanonical_and_duplicate_json(identities, raw):
    assert isinstance(parse_registry(raw, lane=LANE, **identities.pins), Refusal)


def test_empty_producer_role_permits_full_revocation(identities):
    identities.body["producer"] = []
    assert isinstance(identities.registry(), KeyRegistry)


@pytest.mark.parametrize("role", ROLE_SCOPES)
def test_required_role_verifies_exact_frame(identities, role):
    raw = b'"fixture body"'
    assert verify_detached(
        identities.sidecar(raw, role),
        body_sha256=sha256_hex(raw),
        role=role,
        registry=identities.registry(),
    )


@pytest.mark.parametrize(
    "signed_role,required_role", itertools.permutations(ROLE_SCOPES, 2)
)
def test_every_cross_scope_signature_fails(identities, signed_role, required_role):
    raw = b'"fixture body"'
    sidecar = identities.sidecar(raw, signed_role)
    assert not verify_detached(
        sidecar,
        body_sha256=sha256_hex(raw),
        role=required_role,
        registry=identities.registry(),
    )
    # Rewriting the declared scope does not rewrite the cryptographic domain.
    rewritten = json.loads(sidecar)
    rewritten["scope"] = ROLE_SCOPES[required_role]
    assert not verify_detached(
        jcs_dumps(rewritten),
        body_sha256=sha256_hex(raw),
        role=required_role,
        registry=identities.registry(),
    )


@pytest.mark.parametrize("scope", ["apply_ed25519", "eval_ed25519"])
def test_legacy_domains_cannot_authenticate_producer(identities, scope):
    raw = b'"fixture body"'
    sidecar = json.loads(identities.sidecar(raw, "producer", scope=scope))
    sidecar["scope"] = ROLE_SCOPES["producer"]
    assert not verify_detached(
        jcs_dumps(sidecar),
        body_sha256=sha256_hex(raw),
        role="producer",
        registry=identities.registry(),
    )


@pytest.mark.parametrize(
    "fault",
    [
        "unknown-key",
        "wrong-role-key",
        "wrong-body",
        "noncanonical",
        "missing-field",
        "extra-field",
        "bad-signature",
        "bad-fingerprint",
        "wrong-schema",
        "non-string-scope",
    ],
)
def test_sidecar_cannot_choose_its_authority(identities, fault):
    raw = b'"fixture body"'
    sidecar = json.loads(identities.sidecar(raw, "producer"))
    if fault == "unknown-key":
        sidecar = json.loads(
            identities.sidecar(raw, "producer", key=Ed25519PrivateKey.generate())
        )
    elif fault == "wrong-role-key":
        sidecar = json.loads(
            identities.sidecar(raw, "producer", key=identities.keys["actor"])
        )
    elif fault == "wrong-body":
        sidecar["body_sha256"] = "0" * 64
    elif fault == "missing-field":
        del sidecar["scope"]
    elif fault == "extra-field":
        sidecar["allow"] = True
    elif fault == "bad-signature":
        sidecar["signature_base64"] = base64.b64encode(b"0" * 64).decode()
    elif fault == "bad-fingerprint":
        sidecar["signer_spki_sha256"] = []
    elif fault == "wrong-schema":
        sidecar["schema"] = "other"
    elif fault == "non-string-scope":
        sidecar["scope"] = []
    payload = jcs_dumps(sidecar) + (b"\n" if fault == "noncanonical" else b"")
    assert not verify_detached(
        payload,
        body_sha256=sha256_hex(raw),
        role="producer",
        registry=identities.registry(),
    )


def test_rotation_refuses_removed_signer_without_rewriting_old_signature(identities):
    raw = b'"fixture body"'
    signature = identities.sidecar(raw, "producer")
    old_registry = identities.registry()
    identities.body["producer"] = [public_entry(Ed25519PrivateKey.generate())]
    assert not verify_detached(
        signature,
        body_sha256=sha256_hex(raw),
        role="producer",
        registry=identities.registry(),
    )
    assert verify_detached(
        signature, body_sha256=sha256_hex(raw), role="producer", registry=old_registry
    )
