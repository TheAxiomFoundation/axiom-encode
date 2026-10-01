from types import SimpleNamespace

import pytest
from yubihsm.defs import ALGORITHM, CAPABILITY, ORIGIN

from axiom_encode.notary.approval import (
    approval_subject,
    parse_device,
    sign_with_device,
)
from axiom_encode.notary.canonical import jcs_dumps, sha256_hex
from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.signatures import verify_detached

from .lineage_fixtures import EPOCH, LANE, Identities, correction
from .test_protocol import candidate


@pytest.fixture
def hardware(monkeypatch):
    identities = Identities.create()
    role = "approver"
    device = {
        "schema": "axiom/notary-hardware-approval/v1",
        "role": role,
        "lane": LANE,
        "epoch_sha256": EPOCH,
        "serial": 12345,
        "auth_key_id": 7,
        "signing_key_id": 8,
        "spki_sha256": identities.body[role][0]["spki_sha256"],
        "custody_evidence_sha256": "f" * 64,
    }
    info = SimpleNamespace(
        algorithm=ALGORITHM.EC_ED25519,
        origin=ORIGIN.GENERATED,
        capabilities=CAPABILITY.SIGN_EDDSA,
    )
    observations = []
    key = SimpleNamespace(
        get_info=lambda: info,
        get_public_key=lambda: identities.keys[role].public_key(),
        sign_eddsa=lambda data: (
            observations.append(data),
            identities.keys[role].sign(data),
        )[1],
    )
    session = SimpleNamespace(close=lambda: observations.append("session closed"))

    def authenticate(key_id, password):
        assert key_id == 7 and password == "fixture-password"
        return session

    hsm = SimpleNamespace(
        get_device_info=lambda: SimpleNamespace(serial=12345),
        create_session_derived=authenticate,
        close=lambda: observations.append("device closed"),
    )

    def connect(url):
        assert url == "yhusb://serial=12345"
        return hsm

    monkeypatch.setattr("yubihsm.YubiHsm.connect", connect)
    monkeypatch.setattr(
        "yubihsm.objects.AsymmetricKey",
        lambda actual, key_id: key if actual is session and key_id == 8 else None,
    )
    return SimpleNamespace(
        identities=identities,
        device=device,
        key=key,
        info=info,
        observations=observations,
        raw=jcs_dumps(candidate()),
    )


def test_typed_hardware_signature_verifies_under_v33_role(hardware):
    h = hardware
    device = parse_device(jcs_dumps(h.device))
    signature = sign_with_device(h.raw, sha256_hex(h.raw), device, "fixture-password")
    assert verify_detached(
        signature,
        body_sha256=sha256_hex(h.raw),
        role="approver",
        registry=h.identities.registry(),
    )
    assert h.observations[-2:] == ["session closed", "device closed"]
    assert not verify_detached(
        signature,
        body_sha256=sha256_hex(h.raw),
        role="review",
        registry=h.identities.registry(),
    )


@pytest.mark.parametrize(
    "mutation",
    ["imported", "exportable", "wrong-key", "wrong-algorithm", "bad-signature"],
)
def test_device_refuses_wrong_custody_or_response(hardware, mutation):
    h = hardware
    if mutation == "imported":
        h.info.origin = ORIGIN.IMPORTED
    elif mutation == "exportable":
        h.info.capabilities |= CAPABILITY.EXPORTABLE_UNDER_WRAP
    elif mutation == "wrong-key":
        h.device["spki_sha256"] = "0" * 64
    elif mutation == "wrong-algorithm":
        h.info.algorithm = ALGORITHM.EC_P256
    else:
        h.key.sign_eddsa = lambda _: b"bad"
    with pytest.raises(IdentityRefusal):
        sign_with_device(h.raw, sha256_hex(h.raw), h.device, "fixture-password")
    assert h.observations == ["session closed", "device closed"]


@pytest.mark.parametrize(
    "mutation",
    ["wrong-digest", "wrong-role", "wrong-lane", "wrong-epoch", "notary-role"],
)
def test_candidate_must_match_before_device_is_contacted(hardware, mutation):
    h = hardware
    address = sha256_hex(h.raw)
    if mutation == "wrong-digest":
        address = "0" * 64
    elif mutation == "wrong-role":
        h.device["role"] = "review"
    elif mutation == "wrong-lane":
        h.device["lane"] = "TheAxiomFoundation/rulespec-us"
    elif mutation == "wrong-epoch":
        h.device["epoch_sha256"] = "1" * 64
    else:
        h.device["role"] = "notary"
    with pytest.raises(IdentityRefusal):
        device = parse_device(jcs_dumps(h.device))
        sign_with_device(h.raw, address, device, "fixture-password")
    assert h.observations == []


def test_correction_review_accepts_only_correction(hardware):
    h = hardware
    h.device["role"] = "review"
    raw = jcs_dumps(correction())
    assert approval_subject(raw, sha256_hex(raw), h.device)["reason"]
    with pytest.raises(IdentityRefusal, match="approval_role"):
        approval_subject(h.raw, sha256_hex(h.raw), h.device)
