from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from axiom_encode.notary.apps import (
    CHAIN_PERMISSIONS,
    LANE_PERMISSIONS,
    AppCredential,
    AppScope,
    PublisherApps,
)
from axiom_encode.notary.identity import IdentityRefusal

from .lineage_fixtures import LANE


class API:
    def __init__(self, scope, permissions):
        self.scope = scope
        self.permissions = permissions
        self.calls = []
        self.app = {
            "id": scope.app_id,
            "permissions": dict(permissions),
            "owner": {"id": scope.owner_id},
        }
        self.installations = [
            {
                "id": scope.installation_id,
                "app_id": scope.app_id,
                "permissions": dict(permissions),
                "repository_selection": "selected",
                "suspended_at": None,
                "target_type": "Organization",
                "account": {"id": scope.owner_id},
            }
        ]
        self.repositories = {
            "total_count": 1,
            "repositories": [
                {
                    "id": scope.repository_id,
                    "full_name": scope.repository,
                    "owner": {"id": scope.owner_id},
                }
            ],
        }

    def request(self, method, path, token, *, body=None, status=200):
        self.calls.append((method, path, body, token))
        if path == "/app":
            return deepcopy(self.app)
        if path.startswith("/app/installations?"):
            return deepcopy(self.installations)
        if path.endswith("/access_tokens"):
            assert status == 201
            return {
                "token": "fixture-only",
                "permissions": self.permissions,
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(hours=1)
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        if path.startswith("/installation/repositories?"):
            return deepcopy(self.repositories)
        if path == "/installation/token":
            assert method == "DELETE" and status == (204, 401)
            return None
        raise AssertionError(path)


@pytest.fixture
def pair():
    lane = AppScope(10, 20, 30, LANE, 40)
    chain = AppScope(11, 21, 31, LANE + "-notary", 40)
    roots = []
    for scope, permissions in [(chain, CHAIN_PERMISSIONS), (lane, LANE_PERMISSIONS)]:
        api = API(scope, permissions)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        roots.append(AppCredential(scope, key, api))
    return PublisherApps(*roots)


def test_both_roots_are_audited_undownscoped_then_operation_tokens_are_scoped(pair):
    with pair.credentials() as tokens:
        assert tokens == ("fixture-only", "fixture-only")
        for root in (pair.chain, pair.lane):
            mint = [c for c in root._api.calls if c[1].endswith("/access_tokens")]
            assert len(mint) == 2
            assert mint[0][2] == {}
            assert mint[1][2]["repository_ids"] == [root.scope.repository_id]
            claims = jwt.decode(
                mint[1][3], root._key.public_key(), algorithms=["RS256"]
            )
            assert claims["iss"] == str(root.scope.app_id)
            assert claims["exp"] - claims["iat"] == 360
    for root in (pair.chain, pair.lane):
        assert sum(c[0] == "DELETE" for c in root._api.calls) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-root-permission",
        "extra-installation-permission",
        "extra-installation",
        "all-repositories",
        "extra-repository",
        "wrong-repository",
        "wrong-app",
        "wrong-owner",
        "suspended",
        "missing-suspension",
    ],
)
def test_root_scope_failures_never_mint_operation_credentials(pair, mutation):
    api = pair.lane._api
    if mutation == "extra-root-permission":
        api.app["permissions"]["statuses"] = "write"
    elif mutation == "extra-installation-permission":
        api.installations[0]["permissions"]["contents"] = "write"
    elif mutation == "extra-installation":
        api.installations.append(deepcopy(api.installations[0]))
    elif mutation == "all-repositories":
        api.installations[0]["repository_selection"] = "all"
    elif mutation == "extra-repository":
        api.repositories["total_count"] = 2
    elif mutation == "wrong-repository":
        api.repositories["repositories"][0]["full_name"] = LANE + "-other"
    elif mutation == "wrong-app":
        api.app["id"] = 99
    elif mutation == "wrong-owner":
        api.installations[0]["account"]["id"] = 99
    elif mutation == "suspended":
        api.installations[0]["suspended_at"] = "2026-09-20T00:00:00Z"
    else:
        del api.installations[0]["suspended_at"]
    with pytest.raises(IdentityRefusal):
        with pair.credentials():
            pytest.fail("rejected root vended a token")
    assert not any(c[2] for root in (pair.chain, pair.lane) for c in root._api.calls)


@pytest.mark.parametrize("mutation", ["app", "installation", "key", "repository"])
def test_publisher_roots_cannot_share_authority(pair, mutation):
    chain, lane = pair.chain, pair.lane
    if mutation == "key":
        chain = AppCredential(chain.scope, lane._key, chain._api)
    else:
        field = {
            "app": "app_id",
            "installation": "installation_id",
            "repository": "repository_id",
        }[mutation]
        chain = AppCredential(
            replace(chain.scope, **{field: getattr(lane.scope, field)}),
            chain._key,
            chain._api,
        )
    with pytest.raises(IdentityRefusal, match="not_separate"):
        PublisherApps(chain, lane)


def test_operation_failure_still_revokes_both_tokens(pair):
    with pytest.raises(ValueError, match="operation failed"):
        with pair.credentials():
            raise ValueError("operation failed")
    for root in (pair.chain, pair.lane):
        assert sum(c[0] == "DELETE" for c in root._api.calls) == 2
