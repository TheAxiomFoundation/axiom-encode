import pytest

from axiom_encode.notary.canonical import jcs_dumps, strict_parse
from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.merge_guard import inspect

from .chain_fixtures import Epoch
from .test_producer import decode, emission
from .test_producers import API, enrollment_base
from .test_verification import snapshot


@pytest.fixture
def packet(monkeypatch):
    inventory = strict_parse(Epoch.create().inventory)
    epoch = Epoch.create(
        prepare_base=lambda identities: (
            enrollment_base(identities)
            | {
                ".axiom/notary/runner.json": jcs_dumps(
                    {
                        "deployment": {"dependency_inventory": inventory},
                        "encoder_identity": {},
                    }
                )
            }
        )
    )
    policy = strict_parse(epoch.active.blobs[".axiom/notary/producers.json"])
    _, files = decode(emission((epoch, policy, None, {"base": epoch.active})))
    subject = snapshot("c" * 40, epoch.active.blobs | files)
    monkeypatch.setattr(
        "axiom_encode.notary.merge_guard.require_running_identity", lambda *_: None
    )
    return epoch, subject, API(subject.commit)


def test_preflight_passes_before_pending_receipt_without_claiming_admission(packet):
    epoch, subject, api = packet
    assert not epoch.state().pending
    assert (
        inspect(
            epoch.active,
            subject,
            lane=epoch.anchor.lane,
            history=epoch.history,
            api=api,
            pr_number="42",
        )
        == 0
    )


def test_outsider_fork_readonly_and_tampered_bytes_refuse(packet):
    epoch, subject, api = packet
    args = dict(lane=epoch.anchor.lane, history=epoch.history, api=api, pr_number="42")
    api.pr["user"]["id"] = 456
    with pytest.raises(IdentityRefusal):
        inspect(epoch.active, subject, **args)
    api.pr["user"]["id"] = 123
    api.pr["head"]["repo"]["full_name"] = "outsider/fork"
    with pytest.raises(IdentityRefusal):
        inspect(epoch.active, subject, **args)
    api.pr["head"]["repo"]["full_name"] = epoch.anchor.lane
    api.permission["permission"] = "read"
    with pytest.raises(IdentityRefusal):
        inspect(epoch.active, subject, **args)
    api.permission["permission"] = "write"
    bad = snapshot(subject.commit, subject.blobs | {"rules/example.yaml": b"changed"})
    with pytest.raises(IdentityRefusal):
        inspect(epoch.active, bad, **args)


def test_legacy_fallback_only_when_consumer_absent_from_trusted_base(packet):
    epoch, subject, api = packet
    args = dict(lane=epoch.anchor.lane, history=epoch.history, api=api, pr_number="42")
    assert inspect(epoch.base, subject, **args) == 78
    bad = snapshot(
        epoch.active.commit,
        epoch.active.blobs | {".axiom/notary/consumer.json": b"malformed"},
    )
    with pytest.raises(IdentityRefusal):
        inspect(bad, subject, **args)


@pytest.mark.parametrize("rotates", [False, True])
def test_administrative_merge_rewrite_and_finalized_rotation(packet, rotates):
    from dataclasses import replace

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from axiom_encode.notary.administration import build_transition
    from axiom_encode.notary.canonical import sha256_hex
    from axiom_encode.notary.publication import plan_finalization

    from .lineage_fixtures import public_entry

    epoch, _, api = packet
    blobs = dict(epoch.active.blobs)
    if rotates:
        successor = public_entry(Ed25519PrivateKey.generate())
        registry = strict_parse(blobs[".axiom/notary/keys.json"])
        registry["notary"] = [successor]
        consumer = strict_parse(blobs[".axiom/notary/consumer.json"])
        consumer["notary_spki_sha256"] = successor["spki_sha256"]
        blobs[".axiom/notary/keys.json"] = jcs_dumps(registry)
        blobs[".axiom/notary/consumer.json"] = jcs_dumps(consumer)
    else:
        blobs[".axiom/notary/runner.json"] = jcs_dumps({"replacement": True})
    subject = snapshot("d" * 40, blobs)
    body = strict_parse(
        build_transition(epoch.active, subject, epoch.state(), reason="fixture")
    )
    address, bundle = epoch.signed(body, ("transition", "admin-approver"))
    bundle.update(
        {
            sha256_hex(blobs[row["path"]]) + ".raw": blobs[row["path"]]
            for row in body["delta"]
        }
    )
    epoch.append(bundle)
    merged = replace(subject, commit="e" * 40)
    args = dict(lane=epoch.anchor.lane, history=epoch.history)
    assert inspect(epoch.active, merged, **args) == 0
    # Rewritten PR heads still require a fresh authorization; only post-merge
    # processing may use the pending artifact's authenticated commit locator.
    with pytest.raises(IdentityRefusal):
        inspect(epoch.active, merged, api=api, pr_number="42", **args)
    wrong = snapshot(merged.commit, merged.blobs | {"README": b"extra"})
    with pytest.raises(IdentityRefusal):
        inspect(epoch.active, wrong, **args)
    final = plan_finalization(epoch.history, epoch.anchor, address, merged)
    assert final.admitted
    epoch.append(final.files)
    assert inspect(epoch.active, merged, **(args | {"history": epoch.history})) == 0
    with pytest.raises(IdentityRefusal):
        inspect(epoch.active, wrong, **(args | {"history": epoch.history}))
