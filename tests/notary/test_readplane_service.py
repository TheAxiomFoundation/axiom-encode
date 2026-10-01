import time
from types import SimpleNamespace

import jwt
import pytest
from starlette.testclient import TestClient

from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.readplane import (
    ReadGateway,
    ReadOperations,
    RPCClient,
    gateway_bearer,
)
from axiom_encode.notary.service import create_app

from .test_apps import pair as _pair_fixture
from .test_broker import Inputs

pair = _pair_fixture


class API:
    def __init__(self):
        self.calls = []

    def get(self, path):
        self.calls.append(path)
        return {
            "full_name": Inputs().config.repository,
            "allow_rebase_merge": False,
            "allow_squash_merge": True,
            "temp_clone_token": "must-not-escape",
        }

    def collection(self, path):
        self.calls.append(path)
        return []

    def archive(self, repository, artifact):
        self.calls.append((repository, artifact))
        return b"zip"

    def optional_ref(self, repository, branch):
        return None


@pytest.fixture
def gateway(pair):
    operations = ReadOperations(Inputs().config, API())
    return ReadGateway(
        operations,
        pair.chain._key.public_key(),
        app_id=pair.chain.scope.app_id,
        audience="https://reads.example.test",
    )


def test_app_key_read_proof_has_separate_audience_and_never_adds_permissions(
    gateway, pair
):
    bearer = gateway_bearer(pair.chain, gateway.audience)
    result = gateway.read(
        bearer, {"method": "get", "value": "/repos/" + Inputs().config.repository}
    )
    assert result["value"]["allow_rebase_merge"] is False
    assert "temp_clone_token" not in result["value"]
    assert pair.chain._api.calls == []  # No GitHub App token minting.


@pytest.mark.parametrize(
    "mutation", ["wrong-key", "wrong-audience", "too-long", "wrong-subject", "expired"]
)
def test_bad_read_proof_is_rejected_before_fetch(gateway, pair, mutation):
    now = int(time.time())
    body = {
        "iss": gateway.issuer,
        "aud": gateway.audience,
        "sub": "control-plane-read",
        "iat": now - 1,
        "nbf": now - 1,
        "exp": now + 60,
    }
    key = pair.chain._key
    if mutation == "wrong-key":
        key = pair.lane._key
    elif mutation == "wrong-audience":
        body["aud"] = "github"
    elif mutation == "too-long":
        body["exp"] = now + 1000
    elif mutation == "wrong-subject":
        body["sub"] = "sign"
    else:
        body.update(iat=now - 60, nbf=now - 60, exp=now - 1)
    with pytest.raises(IdentityRefusal):
        gateway.read(
            jwt.encode(body, key, algorithm="RS256"),
            {"method": "get", "value": "/repos/" + Inputs().config.repository},
        )
    assert gateway.operations.api.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        {"method": "post", "value": "/repos/TheAxiomFoundation/rulespec-nz"},
        {"method": "get", "value": "/repos/stranger/repo"},
        {
            "method": "get",
            "value": "/repos/TheAxiomFoundation/rulespec-nz/actions/secrets",
        },
        {"method": "get", "value": "/repos/TheAxiomFoundation/rulespec-nz/../secrets"},
        {"method": "get", "value": "/orgs/TheAxiomFoundation/teams/other"},
        {
            "method": "archive",
            "value": {"repository": "stranger/repo", "artifact_id": "1"},
        },
        {
            "method": "optional_ref",
            "value": {"repository": "TheAxiomFoundation/rulespec-nz", "branch": "main"},
        },
    ],
)
def test_closed_read_surface(gateway, pair, operation):
    with pytest.raises(IdentityRefusal):
        gateway.read(gateway_bearer(pair.chain, gateway.audience), operation)
    assert gateway.operations.api.calls == []


def test_service_capabilities_are_physically_separate(gateway):
    with pytest.raises(ValueError, match="one capability"):
        create_app(read_gateway=gateway, signer_factory=lambda: None)
    client = TestClient(create_app(read_gateway=gateway))
    for operation in ("receipt", "publish-tokens", "finalize"):
        response = client.post(
            "/v1/" + operation, json={}, headers={"Authorization": "Bearer fixture"}
        )
        assert response.status_code == 403
    assert gateway.operations.api.calls == []


def test_read_http_round_trip_is_bounded_and_no_store(gateway, pair):
    client = TestClient(create_app(read_gateway=gateway))
    response = client.post(
        "/v1/read",
        json={"method": "get", "value": "/repos/" + Inputs().config.repository},
        headers={
            "Authorization": "Bearer " + gateway_bearer(pair.chain, gateway.audience)
        },
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "must-not-escape" not in response.text
    response = client.post(
        "/v1/read",
        content=b"x" * 131073,
        headers={"Authorization": "Bearer fixture", "Content-Type": "application/json"},
    )
    assert response.status_code == 403


def test_unexpected_exception_never_discloses_credentials():
    def fail(*args):
        raise RuntimeError("PRIVATE-KEY-OR-TOKEN")

    broker = SimpleNamespace(publish_tokens=fail)
    client = TestClient(create_app(broker=broker))
    response = client.post(
        "/v1/publish-tokens", json={}, headers={"Authorization": "Bearer fixture"}
    )
    assert response.status_code == 503
    assert "PRIVATE" not in response.text


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.test",
        "https://user:pass@example.test",
        "https://example.test/?secret",
        "https://example.test:444",
        "https://example.test/foreign",
    ],
)
def test_service_endpoint_cannot_redirect_credentials(endpoint):
    with pytest.raises(IdentityRefusal):
        RPCClient(endpoint, lambda: "secret")


def test_nested_repository_credentials_are_never_forwarded(gateway, pair):
    gateway.operations.api.get = lambda path: {
        "head": {"repo": {"temp_clone_token": "secret", "full_name": "example"}},
        "repository": {"temp_clone_token": "secret"},
        "head_repository": {"temp_clone_token": "secret"},
    }
    result = gateway.read(
        gateway_bearer(pair.chain, gateway.audience),
        {
            "method": "get",
            "value": "/repos/" + Inputs().config.repository + "/pulls/42",
        },
    )
    assert "secret" not in str(result)
    assert result["value"]["head"]["repo"]["full_name"] == "example"
    gateway.operations.api.collection = lambda path: [
        {"base": {"repo": {"temp_clone_token": "secret"}}}
    ]
    result = gateway.read(
        gateway_bearer(pair.chain, gateway.audience),
        {
            "method": "collection",
            "value": "/repos/"
            + Inputs().config.repository
            + "/commits/"
            + "a" * 40
            + "/pulls",
        },
    )
    assert result == {"value": [{"base": {"repo": {}}}]}
