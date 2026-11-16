from dataclasses import replace

import pytest

from axiom_encode.notary.chain import InvalidChain
from axiom_encode.notary.publication import plan_finalization, plan_publication

from .chain_fixtures import Epoch
from .test_verification import snapshot


@pytest.fixture
def epoch():
    return Epoch.create()


def test_publish_complete_bundle_then_finalize_rewritten_commit(epoch):
    address, bundle, subject = epoch.receipt()
    plan = plan_publication(
        epoch.history, epoch.anchor, bundle, address, "receipt", epoch.active, subject
    )
    assert plan.expected_chain_commit == epoch.history[-1].commit
    assert plan.state.tip.address == epoch.state().tip.address
    assert address in plan.state.pending and plan.admitted
    epoch.append(plan.files)
    merged = replace(subject, commit="d" * 40)
    final = plan_finalization(epoch.history, epoch.anchor, address, merged)
    assert final.admitted and final.state.tip.address == address
    assert final.state.sequence == 3
    assert final.state.terminal[address] == "finalized"


def test_publication_retry_is_idempotent(epoch):
    address, bundle, subject = epoch.receipt()
    epoch.append(bundle)
    plan = plan_publication(
        epoch.history, epoch.anchor, bundle, address, "receipt", epoch.active, subject
    )
    assert not plan.files and plan.admitted


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "pointer", "stale-base", "wrong-subject", "wrong-kind"],
)
def test_publication_refuses_unbound_or_open_bundle(epoch, mutation):
    address, bundle, subject = epoch.receipt()
    base, kind = epoch.active, "receipt"
    if mutation == "missing":
        bundle.pop(next(iter(bundle)))
    elif mutation == "extra":
        bundle["orphan"] = b"unsigned"
    elif mutation == "pointer":
        bundle["HEAD.json"] = b"{}"
    elif mutation == "stale-base":
        base = epoch.base
    elif mutation == "wrong-subject":
        subject = snapshot(subject.commit, subject.blobs | {"README": b"different"})
    else:
        kind = "transition"
    with pytest.raises(InvalidChain):
        plan_publication(
            epoch.history, epoch.anchor, bundle, address, kind, base, subject
        )


def test_wrong_merged_bytes_permanently_void_attempt(epoch):
    address, bundle, subject = epoch.receipt()
    epoch.append(bundle)
    changed = snapshot(
        "d" * 40, subject.blobs | {"rules/example.yaml": b"changed at merge"}
    )
    plan = plan_finalization(epoch.history, epoch.anchor, address, changed)
    assert not plan.admitted and plan.state.terminal[address] == "void"
    assert plan.state.tip.address == epoch.state().tip.address
    epoch.append(plan.files)
    with pytest.raises(InvalidChain, match="not_pending"):
        plan_finalization(epoch.history, epoch.anchor, address, subject)


def test_success_voids_same_state_sibling_in_same_commit(epoch):
    one, bundle, subject = epoch.receipt(run_id="1")
    epoch.append(bundle)
    two, second, same = epoch.receipt(run_id="2")
    assert one != two and subject.manifest == same.manifest
    epoch.append(second)
    plan = plan_finalization(epoch.history, epoch.anchor, one, subject)
    assert plan.admitted
    assert plan.state.terminal[one] == "finalized"
    assert plan.state.terminal[two] == "void"
    assert len(plan.files) == 3


def test_never_pending_cannot_finalize(epoch):
    with pytest.raises(InvalidChain, match="not_pending"):
        plan_finalization(epoch.history, epoch.anchor, "f" * 64, epoch.active)


def test_notary_rotation_finalizes_under_authorized_successor_pin(epoch):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from axiom_encode.notary.administration import build_transition
    from axiom_encode.notary.canonical import jcs_dumps, sha256_hex, strict_parse
    from axiom_encode.notary.chain import reconstruct

    from .lineage_fixtures import public_entry

    blobs = dict(epoch.active.blobs)
    successor = public_entry(Ed25519PrivateKey.generate())
    registry = strict_parse(blobs[".axiom/notary/keys.json"])
    registry["notary"] = [successor]
    consumer = strict_parse(blobs[".axiom/notary/consumer.json"])
    consumer["notary_spki_sha256"] = successor["spki_sha256"]
    blobs[".axiom/notary/keys.json"] = jcs_dumps(registry)
    blobs[".axiom/notary/consumer.json"] = jcs_dumps(consumer)
    subject = snapshot("d" * 40, blobs)
    raw = build_transition(epoch.active, subject, epoch.state(), reason="rotate notary")
    body = strict_parse(raw)
    address, bundle = epoch.signed(body, ("transition", "admin-approver"))
    bundle.update(
        {
            sha256_hex(blobs[row["path"]]) + ".raw": blobs[row["path"]]
            for row in body["delta"]
        }
    )
    epoch.append(bundle)
    plan = plan_finalization(epoch.history, epoch.anchor, address, subject)
    assert (
        plan.admitted
        and plan.state.anchor.notary_spki_sha256 == successor["spki_sha256"]
    )
    epoch.append(plan.files)
    assert reconstruct(epoch.history, plan.state.anchor).tip.address == address
