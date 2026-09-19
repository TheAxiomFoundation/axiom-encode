import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from axiom_encode.notary.identity import (
    ISSUER,
    IdentityRefusal,
    JobPolicy,
    authenticate_job,
    require_writer_submission,
    verify_oidc,
)

from .lineage_fixtures import LANE


@pytest.fixture(scope="module")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def policy():
    return JobPolicy(
        LANE,
        "101",
        "102",
        ".github/workflows/notary.yml",
        "a" * 40,
        "refs/heads/main",
        "notary-signing",
        "https://notary.example.test/sign",
        "approve",
    )


def claims(policy):
    now = int(time.time())
    return {
        "iss": ISSUER,
        "aud": policy.audience,
        "iat": now - 1,
        "nbf": now - 1,
        "exp": now + 300,
        "jti": "fixture-nonce",
        "repository": LANE,
        "repository_id": "101",
        "repository_owner_id": "102",
        "workflow_ref": LANE + "/.github/workflows/notary.yml@refs/heads/main",
        "workflow_sha": "a" * 40,
        "ref": "refs/heads/main",
        "environment": "notary-signing",
        "run_id": "1",
        "run_attempt": "1",
        "check_run_id": "2",
        "sub": "repo:" + LANE + ":environment:notary-signing",
    }


def token(body, key):
    return jwt.encode(body, key, algorithm="RS256", headers={"kid": "fixture"})


class API:
    def __init__(self, policy):
        self.run = {
            "id": 1,
            "run_attempt": 1,
            "path": policy.workflow_path,
            "event": "workflow_dispatch",
            "head_branch": "main",
            "status": "in_progress",
            "repository": {"id": 101, "full_name": LANE},
        }
        self.jobs = [
            {
                "name": "approve",
                "run_id": 1,
                "run_attempt": 1,
                "status": "in_progress",
                "check_run_url": "https://api.github.com/repos/"
                + LANE
                + "/check-runs/2",
            }
        ]
        self.pr = {
            "state": "open",
            "draft": False,
            "user": {"id": 123, "login": "writer"},
            "head": {"sha": "b" * 40, "repo": {"full_name": LANE}},
            "base": {"repo": {"full_name": LANE}},
        }
        self.permission = {
            "permission": "write",
            "user": {"id": 123, "login": "writer"},
        }

    def get(self, path):
        if path.endswith("/actions/runs/1"):
            return self.run
        if "/attempts/1/jobs?" in path:
            return {"total_count": len(self.jobs), "jobs": self.jobs}
        if path.endswith("/pulls/42"):
            return self.pr
        if path.endswith("/collaborators/writer/permission"):
            return self.permission
        raise IdentityRefusal("unexpected lookup")


def test_authenticates_signature_and_live_job(policy, rsa_key):
    result = authenticate_job(
        token(claims(policy), rsa_key),
        policy,
        API(policy),
        signing_key=rsa_key.public_key(),
    )
    assert result.check_run_id == "2" and result.environment == "notary-signing"


@pytest.mark.parametrize(
    "field,value",
    [
        ("iss", "https://attacker.invalid"),
        ("aud", "wrong"),
        ("repository", "Other/repo"),
        ("repository_id", "999"),
        ("repository_owner_id", "999"),
        ("workflow_ref", "wrong"),
        ("workflow_sha", "f" * 40),
        ("ref", "refs/heads/attacker"),
        ("environment", "production-signing"),
        ("run_attempt", "2"),
        ("check_run_id", "02"),
        ("run_id", 1),
        ("sub", "wrong"),
        ("exp", 1),
        ("nbf", 9999999999),
        ("jti", ""),
        ("job_workflow_ref", "reusable"),
        ("job_workflow_sha", "f" * 40),
    ],
)
def test_oidc_identity_negative_matrix(policy, rsa_key, field, value):
    with pytest.raises(IdentityRefusal):
        verify_oidc(
            token(claims(policy) | {field: value}, rsa_key),
            policy,
            signing_key=rsa_key.public_key(),
        )


def test_wrong_signature_and_algorithm(policy, rsa_key):
    another = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(IdentityRefusal):
        verify_oidc(
            token(claims(policy), another), policy, signing_key=rsa_key.public_key()
        )
    forged = jwt.encode(
        claims(policy), "attacker-secret", algorithm="HS256", headers={"kid": "fixture"}
    )
    with pytest.raises(IdentityRefusal):
        verify_oidc(forged, policy, signing_key=rsa_key.public_key())


@pytest.mark.parametrize(
    "kind",
    [
        "job-name",
        "job-id",
        "job-attempt",
        "job-run",
        "duplicate-job",
        "completed-job",
        "new-attempt",
        "wrong-workflow",
        "run-repository",
        "completed-run",
    ],
)
def test_control_plane_binding(policy, rsa_key, kind):
    api = API(policy)
    if kind == "job-name":
        api.jobs[0]["name"] = "publish"
    elif kind == "job-id":
        api.jobs[0]["check_run_url"] += "3"
    elif kind == "job-attempt":
        api.jobs[0]["run_attempt"] = 2
    elif kind == "job-run":
        api.jobs[0]["run_id"] = 2
    elif kind == "duplicate-job":
        api.jobs *= 2
    elif kind == "completed-job":
        api.jobs[0]["status"] = "completed"
    elif kind == "new-attempt":
        api.run["run_attempt"] = 2
    elif kind == "wrong-workflow":
        api.run["path"] = ".github/workflows/attacker.yml"
    elif kind == "run-repository":
        api.run["repository"]["id"] = 999
    elif kind == "completed-run":
        api.run["status"] = "completed"
    with pytest.raises(IdentityRefusal):
        authenticate_job(
            token(claims(policy), rsa_key),
            policy,
            api,
            signing_key=rsa_key.public_key(),
        )


def test_publisher_cannot_call_signer(policy, rsa_key):
    claims_body = claims(policy) | {
        "environment": "notary-publishing",
        "sub": "repo:" + LANE + ":environment:notary-publishing",
    }
    with pytest.raises(IdentityRefusal):
        authenticate_job(
            token(claims_body, rsa_key),
            policy,
            API(policy),
            signing_key=rsa_key.public_key(),
        )


@pytest.mark.parametrize(
    "permission,accepted",
    [
        ("write", True),
        ("admin", True),
        ("read", False),
        ("none", False),
        ("triage", False),
    ],
)
def test_current_repository_permission_is_required(policy, permission, accepted):
    api = API(policy)
    api.permission["permission"] = permission
    if accepted:
        assert (
            require_writer_submission(
                api, LANE, "42", "b" * 40, contributor_ids=frozenset(["123"])
            )
            == api.pr
        )
    else:
        with pytest.raises(IdentityRefusal):
            require_writer_submission(
                api, LANE, "42", "b" * 40, contributor_ids=frozenset(["123"])
            )


@pytest.mark.parametrize(
    "kind", ["unknown-operator", "fork", "stale-head", "closed", "wrong-user-response"]
)
def test_submission_identity_negatives(policy, kind):
    api = API(policy)
    ids = frozenset(["123"])
    if kind == "unknown-operator":
        ids = frozenset(["456"])
    elif kind == "fork":
        api.pr["head"]["repo"]["full_name"] = "stranger/rulespec-nz"
    elif kind == "stale-head":
        api.pr["head"]["sha"] = "c" * 40
    elif kind == "closed":
        api.pr["state"] = "closed"
    elif kind == "wrong-user-response":
        api.permission["user"]["id"] = 456
    with pytest.raises(IdentityRefusal):
        require_writer_submission(api, LANE, "42", "b" * 40, contributor_ids=ids)
