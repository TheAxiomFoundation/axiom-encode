import io
import zipfile
from dataclasses import replace
from types import SimpleNamespace

import pytest

from axiom_encode.notary.canonical import sha256_hex, strict_parse
from axiom_encode.notary.identity import IdentityRefusal, authenticate_job
from axiom_encode.notary.manifest import manifest_sha256
from axiom_encode.notary.provenance import REPORT_ARTIFACT
from axiom_encode.notary.signatures import verify_detached
from axiom_encode.notary.signer import NotarySigner, ReceiptInputs, make_candidate

from .test_identity import API as IdentityAPI
from .test_identity import claims, token
from .test_identity import policy as _policy_fixture
from .test_identity import rsa_key as _rsa_fixture
from .test_producers import submission as _submission_fixture

policy = _policy_fixture
rsa_key = _rsa_fixture
submission = _submission_fixture


def archive(raw, *, member="report.json"):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as stream:
        stream.writestr(member, raw)
    return output.getvalue()


class API(IdentityAPI):
    def get(self, path):
        if "/artifacts?" in path:
            return {"total_count": len(self.artifacts), "artifacts": self.artifacts}
        return super().get(path)


class Inputs:
    def __init__(self, packet):
        self.packet = packet

    def receipt(self, identity, address):
        return self.packet


@pytest.fixture
def service(submission, policy, rsa_key, monkeypatch):
    epoch, _, _, args = submission
    api = API(policy)
    api.pr["head"]["sha"] = args["subject"].commit
    api.pr["base"]["sha"] = args["base"].commit
    for name, check in (("verify", 3), ("recompute", 4)):
        api.jobs.append(
            {
                "name": name,
                "run_id": 1,
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
                "check_run_url": f"https://api.github.com/repos/{policy.repository}/check-runs/{check}",
            }
        )
    report_archive = archive(args["report_raw"])
    api.artifacts = [
        {
            "id": 5,
            "name": REPORT_ARTIFACT,
            "expired": False,
            "workflow_run": {"id": 1},
            "digest": "sha256:" + sha256_hex(report_archive),
        }
    ]
    state = epoch.state()
    state = replace(
        state,
        tip=replace(
            state.tip,
            body=state.tip.body
            | {"subject_tree_manifest_sha256": manifest_sha256(args["base"].manifest)},
        ),
    )
    reader = Inputs(
        ReceiptInputs(
            args["base"],
            args["subject"],
            state,
            epoch.inventory,
            report_archive,
            b"",
            "42",
        )
    )
    oidc = token(claims(policy), rsa_key)
    monkeypatch.setattr(
        "jwt.PyJWKClient.get_signing_key_from_jwt",
        lambda *_: SimpleNamespace(key=rsa_key.public_key()),
    )
    identity = authenticate_job(oidc, policy, api)
    candidate = make_candidate(api, identity, reader.packet, require_recompute=True)
    reader.packet = replace(reader.packet, candidate_raw=candidate)
    signer = NotarySigner(epoch.identities.keys["notary"], policy, api, reader)
    approval = epoch.identities.sidecar(candidate, "approver")
    return signer, oidc, candidate, approval, reader, api, epoch


def test_signer_recomputes_and_derives_wrapper(service):
    signer, oidc, candidate, approval, _, _, epoch = service
    address = sha256_hex(candidate)
    bundle = signer.receipt(oidc, address, approval)
    wrapper_name = next(
        n
        for n in bundle
        if n.endswith(".json")
        and strict_parse(bundle[n])["schema"] == "axiom/notary-receipt/v1"
    )
    wrapper = strict_parse(bundle[wrapper_name])
    assert wrapper["candidate_sha256"] == address
    assert wrapper["authorization"] == {
        "environment": "notary-signing",
        "approve_check_run_id": "2",
        "approval_signature_sha256": sha256_hex(approval),
    }
    assert verify_detached(
        bundle[wrapper_name + ".notary.sig"],
        body_sha256=wrapper_name[:-5],
        role="notary",
        registry=epoch.state().registry,
    )
    assert len(bundle) == 5


@pytest.mark.parametrize(
    "mutation", ["missing", "wrong-body", "wrong-role", "wrong-key"]
)
def test_signer_requires_digest_bound_approval(service, mutation):
    signer, oidc, candidate, approval, _, _, epoch = service
    if mutation == "missing":
        approval = b""
    elif mutation == "wrong-body":
        approval = epoch.identities.sidecar(b"different", "approver")
    elif mutation == "wrong-role":
        approval = epoch.identities.sidecar(candidate, "review")
    else:
        approval = epoch.identities.sidecar(
            candidate, "approver", key=epoch.identities.keys["producer"]
        )
    with pytest.raises(IdentityRefusal, match="digest_bound_approval"):
        signer.receipt(oidc, sha256_hex(candidate), approval)


@pytest.mark.parametrize(
    "mutation",
    [
        "failed-verify",
        "failed-recompute",
        "wrong-attempt",
        "duplicate-artifact",
        "missing-artifact",
        "expired-artifact",
        "wrong-digest",
        "moved-base",
        "permission-revoked",
        "fork",
    ],
)
def test_signer_rereads_live_control_plane(service, mutation):
    signer, oidc, candidate, approval, _, api, _ = service
    if mutation == "failed-verify":
        api.jobs[1]["conclusion"] = "failure"
    elif mutation == "failed-recompute":
        api.jobs[2]["conclusion"] = "failure"
    elif mutation == "wrong-attempt":
        api.run["run_attempt"] = 2
    elif mutation == "duplicate-artifact":
        api.artifacts *= 2
    elif mutation == "missing-artifact":
        api.artifacts = []
    elif mutation == "expired-artifact":
        api.artifacts[0]["expired"] = True
    elif mutation == "wrong-digest":
        api.artifacts[0]["digest"] = "sha256:" + "0" * 64
    elif mutation == "moved-base":
        api.pr["base"]["sha"] = "f" * 40
    elif mutation == "permission-revoked":
        api.permission["permission"] = "read"
    else:
        api.pr["head"]["repo"]["full_name"] = "stranger/fork"
    with pytest.raises(IdentityRefusal):
        signer.receipt(oidc, sha256_hex(candidate), approval)


def test_candidate_bytes_cannot_be_swapped_after_recomputation(service):
    signer, oidc, candidate, approval, reader, _, _ = service
    reader.packet = replace(reader.packet, candidate_raw=b"{}")
    with pytest.raises(IdentityRefusal, match="candidate_reconciliation"):
        signer.receipt(oidc, sha256_hex(candidate), approval)


def test_changed_output_cannot_be_laundered_by_report(service):
    from .test_verification import snapshot

    signer, oidc, candidate, approval, reader, _, _ = service
    subject = reader.packet.subject
    reader.packet = replace(
        reader.packet,
        subject=snapshot(
            subject.commit, subject.blobs | {"rules/example.yaml": b"hand edit"}
        ),
    )
    with pytest.raises(IdentityRefusal, match="report_reconciliation"):
        signer.receipt(oidc, sha256_hex(candidate), approval)


def test_wrong_notary_key_cannot_issue_current_epoch_receipt(service, policy):
    _, oidc, candidate, approval, reader, api, epoch = service
    signer = NotarySigner(epoch.identities.keys["producer"], policy, api, reader)
    with pytest.raises(IdentityRefusal, match="signer_not_current_root"):
        signer.receipt(oidc, sha256_hex(candidate), approval)


@pytest.mark.parametrize("member", ["../report.json", "/report.json", "other.json"])
def test_report_archive_does_not_extract_paths(service, member):
    signer, oidc, candidate, approval, reader, api, _ = service
    altered = archive(b"{}", member=member)
    reader.packet = replace(reader.packet, report_archive=altered)
    api.artifacts[0]["digest"] = "sha256:" + sha256_hex(altered)
    with pytest.raises(IdentityRefusal, match="artifact_members"):
        signer.receipt(oidc, sha256_hex(candidate), approval)


def test_genesis_requires_exact_ceremony_administrative_key(service):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from axiom_encode.notary.administration import build_genesis
    from axiom_encode.notary.canonical import jcs_dumps
    from axiom_encode.notary.signer import GenesisInputs

    from .lineage_fixtures import public_entry
    from .test_administration import genesis_args

    signer, oidc, _, _, reader, _, epoch = service
    args = genesis_args(epoch)
    extra = Ed25519PrivateKey.generate()
    registry = strict_parse(args["prospective"][".axiom/notary/keys.json"])
    registry["admin-approver"].append(public_entry(extra))
    registry["admin-approver"].sort(key=lambda row: row["spki_sha256"])
    args["prospective"][".axiom/notary/keys.json"] = jcs_dumps(registry)
    raw = build_genesis(epoch.base, **args)
    reader.genesis = lambda *_: GenesisInputs(epoch.base, args, raw)
    wrong = epoch.identities.sidecar(raw, "admin-approver", key=extra)
    with pytest.raises(IdentityRefusal, match="bootstrap_ceremony_approver"):
        signer.genesis(oidc, sha256_hex(raw), wrong)
    correct = epoch.identities.sidecar(raw, "admin-approver")
    bundle = signer.genesis(oidc, sha256_hex(raw), correct)
    assert len(bundle) == 7
    assert verify_detached(
        bundle[sha256_hex(raw) + ".json.genesis.sig"],
        body_sha256=sha256_hex(raw),
        role="genesis",
        registry=epoch.state().registry,
    )


def test_transition_requires_admin_approval_over_recomputed_candidate(service):
    from axiom_encode.notary.administration import build_transition
    from axiom_encode.notary.canonical import jcs_dumps
    from axiom_encode.notary.signer import TransitionInputs

    from .test_verification import snapshot

    signer, oidc, _, _, reader, _, epoch = service
    registry = strict_parse(epoch.active.blobs[".axiom/notary/keys.json"])
    registry["producer"] = []
    subject = snapshot(
        "d" * 40, epoch.active.blobs | {".axiom/notary/keys.json": jcs_dumps(registry)}
    )
    candidate = build_transition(
        epoch.active, subject, epoch.state(), reason="revoke producer"
    )
    reader.transition = lambda *_: TransitionInputs(
        epoch.active, subject, epoch.state(), candidate
    )
    wrong = epoch.identities.sidecar(candidate, "approver")
    with pytest.raises(IdentityRefusal, match="digest_bound_approval"):
        signer.transition(oidc, sha256_hex(candidate), wrong)
    correct = epoch.identities.sidecar(candidate, "admin-approver")
    bundle = signer.transition(oidc, sha256_hex(candidate), correct)
    assert len(bundle) == 4
    assert verify_detached(
        bundle[sha256_hex(candidate) + ".json.transition.sig"],
        body_sha256=sha256_hex(candidate),
        role="transition",
        registry=epoch.state().registry,
    )
