import base64
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization

from axiom_encode.notary.canonical import jcs_dumps, strict_parse
from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.lineage import STORE_PREFIX
from axiom_encode.notary.producer_host import ProducerHost
from axiom_encode.notary.producers import Enrollment
from axiom_encode.notary.signer import _signed_sidecar
from axiom_encode.notary.verification import verify_snapshots

from .test_producer import decode
from .test_producers import submission as _submission_fixture
from .test_verification import snapshot

submission = _submission_fixture


@pytest.fixture
def host(tmp_path, submission, monkeypatch):
    epoch, policy, _, args = submission
    # Fixture files only. Deployment custody is tested separately; this test
    # exercises the host's durable ownership and completed-run semantics.
    monkeypatch.setattr(
        "axiom_encode.notary.producer_host.custodian_file",
        lambda path, **kwargs: __import__("pathlib").Path(path).read_bytes(),
    )
    config = {
        "operators": [{"uid": 1234, "github_user_id": "123"}],
        "state_directory": str(tmp_path / "state"),
        "lane": epoch.anchor.lane,
        "epoch_sha256": epoch.anchor.epoch_sha256,
        "sampling": {"temperature": "0.5", "seed": None},
        "references": {"oracles": [], "reference_data": []},
    }
    for role in ("producer", "actor"):
        path = tmp_path / (role + ".pem")
        path.write_bytes(
            epoch.identities.keys[role].private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        config[role + "_key_file"] = str(path)
    calls = []

    def run(**kwargs):
        calls.append(kwargs["request"]["run_id"])
        return {
            "outputs": {"rules/example.yaml": b"generated\n"},
            "model": "fixture",
            "prompts": ["a" * 64],
            "source_id": "fixture",
            "source_bytes": b"source",
        }, b'{"fixture_refreshed": true}'

    service = ProducerHost(config, runtime=SimpleNamespace(run=run))

    @contextmanager
    def base(user_id):
        assert user_id == "123"
        yield None, args["base"], Enrollment(jcs_dumps(policy["runtimes"][0])), []

    service._base = base
    service.calls = calls
    return service


def request(**changes):
    return (
        dict(
            operation="encode",
            run_id="1" * 32,
            citation="fixture",
            draw_set_id="draw",
            auth_base64=base64.b64encode(b'{"fixture": true}').decode(),
        )
        | changes
    )


def test_retry_and_status_do_not_generate_again_or_publish_auth(host):
    result = host.perform(1234, request())
    assert result == host.perform(1234, request())
    assert result == host.perform(1234, {"operation": "status", "run_id": "1" * 32})
    assert host.calls == ["1" * 32]
    export = base64.b64decode(result["export_base64"])
    assert b"fixture_refreshed" not in export
    assert b"auth" not in export
    assert (
        b"auth_base64"
        not in (host.root / "records" / ("1" * 32 + ".json")).read_bytes()
    )


def test_uid_reassignment_cannot_retrieve_prior_contributor_auth(host):
    host.perform(1234, request())
    host.operators[1234] = "999"
    with pytest.raises(IdentityRefusal, match="run_owner"):
        host.perform(1234, {"operation": "status", "run_id": "1" * 32})
    with pytest.raises(IdentityRefusal, match="run_reuse"):
        host.perform(1234, request())
    assert len(host.calls) == 1


def test_request_cannot_supply_outputs_or_choose_signing_identity(host):
    for extra in (
        {"outputs": {}},
        {"runtime_identity": "forged"},
        {"command": "anything"},
    ):
        with pytest.raises(IdentityRefusal, match="encode_request"):
            host.perform(1234, request(**extra))
    with pytest.raises(IdentityRefusal, match="not_enrolled"):
        host.perform(5678, request())
    assert not host.calls


def test_interrupted_run_is_not_automatically_regenerated(host):
    def fail(**kwargs):
        raise RuntimeError("fixture failure")

    host.runtime.run = fail
    with pytest.raises(RuntimeError):
        host.perform(1234, request())
    assert host.perform(1234, request())["state"] == "failed-or-interrupted"


def test_correction_of_unmerged_generation_preserves_both_records(host, submission):
    epoch, _, _, args = submission
    host.perform(1234, request())
    result = host.perform(
        1234,
        {
            "operation": "correction",
            "run_id": "2" * 32,
            "edits": {"rules/example.yaml": base64.b64encode(b"corrected\n").decode()},
            "reason": "Policy correction",
            "predecessor_run_id": "1" * 32,
        },
    )
    packet, files = decode(base64.b64decode(result["export_base64"]))
    body_path = STORE_PREFIX + packet["record_sha256"] + ".json"
    files[body_path + ".review.sig"] = _signed_sidecar(
        epoch.identities.keys["review"], files[body_path], "review"
    )
    report = strict_parse(
        verify_snapshots(
            args["base"],
            snapshot("c" * 40, args["base"].blobs | files),
            epoch.state().predecessor(),
            epoch.inventory,
            [{"gate_id": "compile", "outcome": "pass"}],
        )
    )
    assert report["schema"] == "axiom/notary-report-pass/v1"
    assert len(report["eligible_records"]) == 2
    assert host.calls == ["1" * 32]
