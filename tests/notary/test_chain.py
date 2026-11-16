from dataclasses import replace

import pytest

from axiom_encode.notary.canonical import jcs_dumps, strict_parse
from axiom_encode.notary.chain import ChainState, reconstruct
from axiom_encode.notary.manifest import manifest_sha256
from axiom_encode.notary.refusal import Refusal

from .chain_fixtures import Epoch, addressed
from .test_verification import snapshot


@pytest.fixture
def epoch():
    e = Epoch.create()
    assert isinstance(e.state(), ChainState), e.state()
    return e


def test_genesis_activation_and_exact_output_receipt(epoch):
    assert epoch.state().activated
    assert epoch.state().sequence == 2
    receipt, bundle, subject = epoch.receipt()
    old = epoch.state().tip.address
    epoch.append(bundle)
    pending = epoch.state()
    assert pending.tip.address == old
    assert receipt in pending.pending
    epoch.finalize(receipt, "receipt", manifest_sha256(subject.manifest), 3)
    finalized = epoch.state()
    assert finalized.sequence == 3 and finalized.tip.address == receipt
    assert finalized.terminal[receipt] == "finalized"


@pytest.mark.parametrize("suffix", [".json.notary.sig", ".json.approver.sig"])
def test_missing_receipt_bundle_member(epoch, suffix):
    _, bundle, _ = epoch.receipt()
    del bundle[next(name for name in bundle if name.endswith(suffix))]
    epoch.append(bundle)
    assert isinstance(epoch.state(), Refusal)


def test_unpublished_report_or_candidate(epoch):
    _, bundle, _ = epoch.receipt()
    for schema in ("axiom/notary-report-pass/v1", "axiom/notary-receipt-candidate/v1"):
        name = next(
            name
            for name, raw in bundle.items()
            if name.endswith(".json") and strict_parse(raw)["schema"] == schema
        )
        files = dict(bundle)
        del files[name]
        prior = list(epoch.history)
        epoch.append(files)
        assert isinstance(epoch.state(), Refusal)
        epoch.history = prior


@pytest.mark.parametrize(
    "operation",
    [
        "delete",
        "rewrite",
        "late-sidecar",
        "orphan-json",
        "orphan-raw",
        "sidecar-to-raw",
        "unknown-role",
    ],
)
def test_append_only_and_reachability(epoch, operation):
    blobs = dict(epoch.history[-1].blobs)
    existing = next(
        name for name in blobs if name.endswith(".json") and name != "HEAD.json"
    )
    if operation == "delete":
        del blobs[existing]
    elif operation == "rewrite":
        blobs[existing] = b"{}"
    elif operation in ("late-sidecar", "unknown-role"):
        blobs[
            existing
            + (".approver.sig" if operation == "late-sidecar" else ".review.sig")
        ] = b"{}"
    elif operation == "sidecar-to-raw":
        blobs[next(name for name in blobs if name.endswith(".raw")) + ".notary.sig"] = (
            b"{}"
        )
    elif operation == "orphan-raw":
        address, raw = addressed({"schema": "looks-like-an-artifact"})
        blobs[address + ".raw"] = raw
    else:
        from .test_protocol import report

        address, raw = addressed(report())
        blobs[address + ".json"] = raw
    epoch.history.append(snapshot("e" * 40, blobs))
    assert isinstance(epoch.state(), Refusal)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sequence", "4"),
        ("sequence", "03"),
        ("lane", "Other/repo"),
        ("epoch_sha256", "f" * 64),
        ("target_kind", "transition"),
        ("merged_tip_manifest_sha256", "f" * 64),
    ],
)
def test_marker_binding_and_sequence(epoch, field, value):
    r, bundle, subject = epoch.receipt()
    epoch.append(bundle)
    epoch.finalize(r, "receipt", manifest_sha256(subject.manifest), 3)
    previous, latest = epoch.history[-2:]
    new = latest.blobs.keys() - previous.blobs.keys()
    name = next(iter(new))
    marker = strict_parse(latest.blobs[name])
    marker[field] = value
    address, raw = addressed(marker)
    blobs = dict(latest.blobs)
    del blobs[name]
    blobs[address + ".json"] = raw
    epoch.history[-1] = snapshot(latest.commit, blobs)
    assert isinstance(epoch.state(), Refusal)


def test_same_state_sibling_is_voided(epoch):
    first, bundle1, subject = epoch.receipt(run_id="1")
    second, bundle2, _ = epoch.receipt(run_id="2")
    epoch.append(bundle1)
    epoch.append(bundle2)
    assert len(epoch.state().pending) == 2
    epoch.finalize(
        first,
        "receipt",
        manifest_sha256(subject.manifest),
        3,
        voids=[(second, "receipt")],
    )
    state = epoch.state()
    assert state.terminal[first] == "finalized" and state.terminal[second] == "void"
    epoch.finalize(second, "receipt", manifest_sha256(subject.manifest), 4)
    assert isinstance(epoch.state(), Refusal)


def test_missing_sibling_void_refuses(epoch):
    first, bundle1, subject = epoch.receipt(run_id="1")
    _, bundle2, _ = epoch.receipt(run_id="2")
    epoch.append(bundle1)
    epoch.append(bundle2)
    epoch.finalize(first, "receipt", manifest_sha256(subject.manifest), 3)
    assert isinstance(epoch.state(), Refusal)


def test_head_pointer_does_not_supply_authority(epoch):
    blobs = dict(epoch.history[-1].blobs)
    head = strict_parse(blobs["HEAD.json"])
    head["tip_sha256"] = "f" * 64
    blobs["HEAD.json"] = jcs_dumps(head)
    epoch.history.append(snapshot("f" * 40, blobs))
    assert isinstance(epoch.state(), Refusal)


def test_consumer_notary_pin_is_enforced(epoch):
    assert isinstance(
        reconstruct(epoch.history, replace(epoch.anchor, notary_spki_sha256="f" * 64)),
        Refusal,
    )


@pytest.mark.parametrize("commit_index", [0, 2])
def test_bootstrap_missing_preimage(epoch, commit_index):
    # Remove from this and later snapshots, so failure is missing bundle, not mutation.
    name = next(
        name
        for name in epoch.history[commit_index].blobs
        if name.endswith(".raw")
        and (commit_index == 0 or name not in epoch.history[commit_index - 1].blobs)
    )
    for i in range(commit_index, len(epoch.history)):
        snap = epoch.history[i]
        epoch.history[i] = snapshot(
            snap.commit, {k: v for k, v in snap.blobs.items() if k != name}
        )
    assert isinstance(epoch.state(), Refusal)
