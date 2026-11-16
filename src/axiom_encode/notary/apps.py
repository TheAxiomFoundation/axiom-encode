"""Two-App publisher credentials, audited at the minting authority boundary.

Only the separately deployed publisher broker instantiates these objects.
Runner requests never select an App, installation, repository or permission.
Auditing a downscoped token is insufficient: inspect both App registrations,
every installation, and the repositories visible to an undownscoped token.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

import jwt
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from .canonical import sha256_hex, strict_parse
from .identity import IdentityRefusal
from .protocol import lane_name
from .remote import _bounded

CHAIN_PERMISSIONS = {"contents": "write", "metadata": "read"}
LANE_PERMISSIONS = {"checks": "write", "contents": "read", "metadata": "read"}


class GitHubAppAPI:
    """Fixed-origin HTTPS transport; credentials and error bodies never logged."""

    def request(self, method, path, token, *, body=None, status=200):
        expected_statuses = status if isinstance(status, tuple) else (status,)
        if isinstance(status, tuple) and (
            method != "DELETE" or path != "/installation/token" or status != (204, 401)
        ):
            raise IdentityRefusal("app_api_status_policy")
        if (
            method not in {"GET", "POST", "PATCH", "DELETE"}
            or not path.startswith(("/app", "/installation/", "/repos/"))
            or any(value in path for value in ("..", "#", "\\", "\r", "\n"))
        ):
            raise IdentityRefusal("app_api_path")
        with requests.Session() as session:
            session.trust_env = False
            try:
                response = session.request(
                    method,
                    "https://api.github.com" + path,
                    headers={
                        "Authorization": "Bearer " + token,
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                    json=body,
                    timeout=(5, 30),
                    allow_redirects=False,
                    stream=True,
                )
                if response.status_code not in expected_statuses:
                    response.close()
                    raise IdentityRefusal("app_api_refused_or_uncertain")
                raw = _bounded(response)
            except requests.RequestException:
                raise IdentityRefusal("app_api_refused_or_uncertain") from None
        if status == 204 or status == (204, 401):
            return None
        parsed = strict_parse(raw)
        if not isinstance(parsed, (dict, list)):
            raise IdentityRefusal("app_api_response")
        return parsed


@dataclass(frozen=True)
class AppScope:
    app_id: int
    installation_id: int
    repository_id: int
    repository: str
    owner_id: int

    def __post_init__(self):
        if (
            any(
                type(value) is not int or value <= 0
                for value in (
                    self.app_id,
                    self.installation_id,
                    self.repository_id,
                    self.owner_id,
                )
            )
            or not lane_name(self.repository)
            or not self.repository.startswith("TheAxiomFoundation/")
        ):
            raise IdentityRefusal("app_configuration")


@dataclass(frozen=True)
class InstallationToken:
    value: str
    expires_at: int


class AppCredential:
    """Custodian-installed App root. No key-loading or caller-selected scope."""

    def __init__(self, scope: AppScope, key: RSAPrivateKey, api: GitHubAppAPI):
        if not isinstance(key, RSAPrivateKey) or key.key_size < 2048:
            raise IdentityRefusal("app_key_type")
        self.scope, self._key, self._api = scope, key, api
        self.key_fingerprint = sha256_hex(
            key.public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

    def _jwt(self):
        now = int(time.time())
        return jwt.encode(
            {"iat": now - 60, "exp": now + 300, "iss": str(self.scope.app_id)},
            self._key,
            algorithm="RS256",
        )

    def _roots(self, bearer, permissions):
        app = self._api.request("GET", "/app", bearer)
        if (
            not isinstance(app, dict)
            or app.get("id") != self.scope.app_id
            or app.get("permissions") != permissions
            or app.get("owner", {}).get("id") != self.scope.owner_id
        ):
            raise IdentityRefusal("app_root_permissions")
        installations = []
        for page in range(1, 101):
            rows = self._api.request(
                "GET", f"/app/installations?per_page=100&page={page}", bearer
            )
            if not isinstance(rows, list):
                raise IdentityRefusal("app_installation_listing")
            installations.extend(rows)
            if len(rows) < 100:
                break
        else:
            raise IdentityRefusal("app_installation_listing")
        if len(installations) != 1 or not isinstance(installations[0], dict):
            raise IdentityRefusal("app_installation_scope")
        installation = installations[0]
        if (
            installation.get("id") != self.scope.installation_id
            or installation.get("app_id") != self.scope.app_id
            or installation.get("permissions") != permissions
            or installation.get("repository_selection") != "selected"
            or "suspended_at" not in installation
            or installation["suspended_at"] is not None
            or installation.get("target_type") != "Organization"
            or installation.get("account", {}).get("id") != self.scope.owner_id
        ):
            raise IdentityRefusal("app_installation_scope")

    def _issue(self, bearer, body, permissions):
        result = self._api.request(
            "POST",
            f"/app/installations/{self.scope.installation_id}/access_tokens",
            bearer,
            body=body,
            status=201,
        )
        token = result.get("token") if isinstance(result, dict) else None
        if not isinstance(token, str) or not token:
            raise IdentityRefusal("app_token_response")
        if result.get("permissions") != permissions:
            self.revoke(token)
            raise IdentityRefusal("app_token_permissions")
        try:
            expiry = result["expires_at"]
            if not isinstance(expiry, str) or not expiry.endswith("Z"):
                raise ValueError
            expires_at = int(datetime.fromisoformat(expiry).timestamp())
            if not int(time.time()) + 60 < expires_at <= int(time.time()) + 3700:
                raise ValueError
        except (KeyError, ValueError, TypeError, OverflowError):
            self.revoke(token)
            raise IdentityRefusal("app_token_expiry") from None
        return InstallationToken(token, expires_at)

    def revoke(self, token):
        # A token already revoked/expired is unusable (401), so revocation is
        # idempotent across a broker crash. Other errors never imply success.
        self._api.request("DELETE", "/installation/token", token, status=(204, 401))

    @contextmanager
    def _token(self, bearer, body, permissions):
        token = self._issue(bearer, body, permissions)
        try:
            yield token.value
        finally:
            self.revoke(token.value)

    def audit(self, permissions):
        bearer = self._jwt()
        self._roots(bearer, permissions)
        # Deliberately no repository or permission downscope for this audit.
        # Otherwise a root installation with extra repositories stays hidden.
        with self._token(bearer, {}, permissions) as token:
            listing = self._api.request(
                "GET", "/installation/repositories?per_page=100&page=1", token
            )
            if (
                not isinstance(listing, dict)
                or type(listing.get("total_count")) is not int
                or listing["total_count"] != 1
                or not isinstance(listing.get("repositories"), list)
                or len(listing["repositories"]) != 1
            ):
                raise IdentityRefusal("app_repository_scope")
            repo = listing["repositories"][0]
            if (
                not isinstance(repo, dict)
                or repo.get("id") != self.scope.repository_id
                or repo.get("full_name") != self.scope.repository
                or repo.get("owner", {}).get("id") != self.scope.owner_id
            ):
                raise IdentityRefusal("app_repository_scope")
        self._roots(bearer, permissions)

    @contextmanager
    def operation(self, permissions):
        with self._token(
            self._jwt(),
            {
                "repository_ids": [self.scope.repository_id],
                "permissions": permissions,
            },
            permissions,
        ) as token:
            yield token

    def issue_operation(self, permissions):
        """Only used after the broker has authenticated the publish job."""
        return self._issue(
            self._jwt(),
            {"repository_ids": [self.scope.repository_id], "permissions": permissions},
            permissions,
        )


class PublisherApps:
    """Both roots must pass the exact audit before either capability is used."""

    def __init__(self, chain: AppCredential, lane: AppCredential):
        if (
            chain.scope.app_id == lane.scope.app_id
            or chain.scope.installation_id == lane.scope.installation_id
            or chain.scope.repository_id == lane.scope.repository_id
            or chain.key_fingerprint == lane.key_fingerprint
            or chain.scope.repository != lane.scope.repository + "-notary"
            or chain.scope.owner_id != lane.scope.owner_id
        ):
            raise IdentityRefusal("publisher_apps_not_separate")
        self.chain, self.lane = chain, lane

    @contextmanager
    def credentials(self):
        self.chain.audit(CHAIN_PERMISSIONS)
        self.lane.audit(LANE_PERMISSIONS)
        with self.chain.operation(CHAIN_PERMISSIONS) as chain:
            with self.lane.operation(LANE_PERMISSIONS) as lane:
                yield chain, lane

    def issue(self):
        """Issue the two limited tokens to the authenticated publisher job.

        App-minting keys never leave the broker. The pinned publisher revokes
        both installation tokens in its finally block; GitHub's token expiry
        remains the bound if a runner is abruptly lost.
        """
        self.chain.audit(CHAIN_PERMISSIONS)
        self.lane.audit(LANE_PERMISSIONS)
        chain = self.chain.issue_operation(CHAIN_PERMISSIONS)
        try:
            lane = self.lane.issue_operation(LANE_PERMISSIONS)
        except BaseException:
            self.chain.revoke(chain.value)
            raise
        return chain, lane
