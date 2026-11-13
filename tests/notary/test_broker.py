from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from axiom_encode.notary.broker import PublisherTokenBroker
from axiom_encode.notary.github_inputs import Deployment
from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.leases import owner

from .lineage_fixtures import LANE
from .test_identity import API, claims, token
from .test_identity import rsa_key as _rsa_fixture

rsa_key = _rsa_fixture


class Leases:
    def __init__(self):
        self.issued, self.released = [], []

    @contextmanager
    def finalization(self, identity, policy):
        try:
            yield self.acquire(identity, policy, kind="finalize")
        finally:
            self.release(identity)

    def acquire(self, identity, policy, *, kind):
        self.issued.append((identity, policy, kind))
        return "fixture-chain-token", "fixture-lane-token"

    def policy_for_release(self, key):
        return next(p for i, p, _ in self.issued if owner(i) == key)

    def release(self, identity):
        self.released.append(identity)


class Inputs:
    def __init__(self):
        self.config = Deployment(
            LANE,
            "101",
            "102",
            "e" * 64,
            "main",
            ".axiom/notary/consumer.json",
            ".github/workflows/notary.yml",
            "sign",
            frozenset({123}),
            1,
            2,
            3,
            123,
            456,
            "admission",
            b"{}",
        )
        self.tip = "a" * 40
        self.audits = []

    def _tip(self, repo, branch):
        return self.tip

    def _audit(self, environment):
        self.audits.append(environment)


@pytest.fixture
def broker(rsa_key, monkeypatch):
    inputs, leases = Inputs(), Leases()
    apps = SimpleNamespace(
        lane=SimpleNamespace(
            scope=SimpleNamespace(
                repository=LANE, repository_id=101, owner_id=102, app_id=456
            )
        ),
        chain=SimpleNamespace(scope=SimpleNamespace(app_id=123)),
    )
    finalizer = SimpleNamespace(finalize=lambda *args: {"state": "finalized"})
    service = PublisherTokenBroker(inputs, apps, leases, finalizer=finalizer)
    monkeypatch.setattr(
        "jwt.PyJWKClient.get_signing_key_from_jwt",
        lambda *_: SimpleNamespace(key=rsa_key.public_key()),
    )
    return service


def caller(broker, rsa_key, operation):
    policy = broker._policy(operation)
    body = claims(policy) | {
        "workflow_ref": LANE + "/" + policy.workflow_path + "@" + policy.ref,
        "environment": policy.environment,
        "sub": f"repo:{LANE}:environment:{policy.environment}",
    }
    api = API(policy)
    api.run["event"] = policy.event
    api.run["head_sha"] = policy.workflow_sha_git_oid
    api.jobs[0]["name"] = operation
    for check, name in enumerate(("verify", "recompute", "approve"), 3):
        api.jobs.append(
            api.jobs[0]
            | {
                "name": name,
                "status": "completed",
                "conclusion": "success",
                "check_run_url": f"https://api.github.com/repos/{LANE}/check-runs/{check}",
            }
        )
    broker.inputs.api = api
    return token(body, rsa_key), body


def test_only_publish_job_receives_limited_tokens_and_release_survives_merge(
    broker, rsa_key
):
    signed, _ = caller(broker, rsa_key, "publish")
    grant = broker.publish_tokens(signed)
    assert grant["chain_token"] == "fixture-chain-token"
    assert grant["lane_token"] == "fixture-lane-token"
    broker.inputs.tip = "b" * 40
    assert broker.release(signed, grant["lease_owner"]) == {"state": "released"}
    assert len(broker.leases.released) == 1


def test_finalizer_returns_no_credential(broker, rsa_key):
    signed, _ = caller(broker, rsa_key, "finalize")
    assert broker.finalize(signed) == {"state": "finalized"}
    assert broker.leases.issued[0][2] == "finalize"
    assert len(broker.leases.released) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment", "notary-signing"),
        ("workflow_ref", "wrong"),
        ("workflow_sha", "b" * 40),
        ("run_attempt", "2"),
    ],
)
def test_wrong_identity_receives_no_tokens(broker, rsa_key, field, value):
    _, body = caller(broker, rsa_key, "publish")
    with pytest.raises(IdentityRefusal):
        broker.publish_tokens(token(body | {field: value}, rsa_key))
    assert not broker.leases.issued


def test_signer_job_or_failed_approval_cannot_get_publisher_tokens(broker, rsa_key):
    signed, _ = caller(broker, rsa_key, "publish")
    broker.inputs.api.jobs[0]["name"] = "approve"
    with pytest.raises(IdentityRefusal):
        broker.publish_tokens(signed)
    assert not broker.leases.issued
    signed, _ = caller(broker, rsa_key, "publish")
    broker.inputs.api.jobs[-1]["conclusion"] = "failure"
    with pytest.raises(IdentityRefusal, match="required_job"):
        broker.publish_tokens(signed)
    assert not broker.leases.issued
