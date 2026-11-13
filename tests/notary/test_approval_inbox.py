from concurrent.futures import ThreadPoolExecutor

import pytest
from starlette.testclient import TestClient

from axiom_encode.notary.approval_inbox import ApprovalInbox
from axiom_encode.notary.canonical import jcs_dumps, sha256_hex, strict_parse
from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.service import create_app

from .lineage_fixtures import Identities


def test_public_signature_deposit_is_idempotent_atomic_and_not_a_signing_api(tmp_path):
    tmp_path.chmod(0o700)
    inbox = ApprovalInbox(tmp_path.resolve())
    identities = Identities.create()
    candidate = jcs_dumps({"fixture": "candidate"})
    approval = identities.sidecar(candidate, "approver")
    address = sha256_hex(candidate)
    assert inbox.read(address) is None
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert set(pool.map(inbox.deposit, [approval] * 16)) == {address}
    assert inbox.read(address) == approval
    mutated = jcs_dumps(strict_parse(approval) | {"signature_base64": "YQ=="})
    with pytest.raises(IdentityRefusal):
        inbox.deposit(mutated)
    with pytest.raises(IdentityRefusal):
        inbox.read("../private-key")
    with pytest.raises(IdentityRefusal):
        inbox.deposit(identities.sidecar(candidate, "producer"))


def test_waiting_for_approval_authenticates_before_inbox_read():
    calls = []

    class Signer:
        def _authenticate(self, token, digest):
            calls.append("authenticate")
            if token != "fixture":
                raise IdentityRefusal("fixture_bad_identity")

        def receipt(self, *args):
            pytest.fail("no signature until hardware approval exists")

    def load(address):
        calls.append("read")
        return None

    client = TestClient(create_app(signer_factory=Signer, approval_loader=load))
    body = {"candidate_sha256": "a" * 64, "approval_base64": None}
    assert (
        client.post(
            "/v1/receipt", json=body, headers={"Authorization": "Bearer invalid"}
        ).status_code
        == 403
    )
    assert calls == ["authenticate"]
    result = client.post(
        "/v1/receipt", json=body, headers={"Authorization": "Bearer fixture"}
    )
    assert result.json() == {"state": "awaiting_hardware_approval"}
    assert calls == ["authenticate", "authenticate", "read"]
