"""Closed read transport; neither publisher App gains Actions/org permissions.

The custodian's read-only GitHub credential stays in the signer's read plane.
The broker authenticates to that plane using its existing chain-App RSA key,
with a distinct, audience-bound JWT. Runners contact only the broker using
GitHub OIDC. This deployment proposal does not widen either App installation.
"""

from __future__ import annotations

import base64
import re
import time
from urllib.parse import urlparse

import jwt
import requests

from ._schema import decode_base64, fields
from .canonical import strict_parse
from .github_inputs import Deployment
from .identity import IdentityRefusal
from .protocol import decimal_id
from .remote import _bounded


def public_metadata(value):
    # GitHub embeds repository objects inside PR/run/association responses.
    # Temporary clone credentials can occur at any depth, including arrays.
    if isinstance(value, dict):
        return {
            key: public_metadata(item)
            for key, item in value.items()
            if not re.search(
                r"(^|_)(token|secret|password|authorization|private_key)($|_)",
                key,
                re.IGNORECASE,
            )
        }
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    return value


class ReadOperations:
    """Closed allowlist of GETs needed by the verifier and publisher."""

    def __init__(self, config: Deployment, api, *, ceremony=None):
        self.config, self.api, self.ceremony = config, api, ceremony

    def perform(self, request: dict):
        if not fields(request, {"method", "value"}):
            raise IdentityRefusal("read_operation")
        method, value = request["method"], request["value"]
        c = self.config
        lane, chain = c.repository, c.repository + "-notary"
        if method == "archive":
            if (
                not fields(value, {"repository", "artifact_id"})
                or value["repository"] != lane
                or not decimal_id(value["artifact_id"])
            ):
                raise IdentityRefusal("read_artifact")
            raw = self.api.archive(lane, value["artifact_id"])
            return {"archive_base64": base64.b64encode(raw).decode()}
        if method == "optional_ref":
            if value != {"repository": chain, "branch": "chain"}:
                raise IdentityRefusal("read_ref")
            return {"value": public_metadata(self.api.optional_ref(chain, "chain"))}
        if method not in {"get", "collection"} or not isinstance(value, str):
            raise IdentityRefusal("read_operation")
        exact = {
            f"/repos/{lane}",
            f"/repos/{lane}/git/ref/heads/{c.content_branch}",
            f"/repos/{chain}/git/ref/heads/chain",
            f"/repos/{lane}/rulesets/{c.lane_ruleset_id}",
            f"/repos/{chain}/rulesets/{c.chain_writer_ruleset_id}",
            f"/repos/{chain}/rulesets/{c.chain_integrity_ruleset_id}",
        }
        for environment in ("notary-signing", "notary-publishing"):
            exact.add(f"/repos/{lane}/environments/{environment}")
            exact.add(
                f"/repos/{lane}/environments/{environment}/deployment-branch-policies?per_page=100"
            )
        if self.ceremony:
            exact.add(f"/repos/{lane}/rulesets/{self.ceremony.lock_ruleset_id}")
            team = (
                f"/orgs/{lane.split('/')[0]}/teams/{self.ceremony.bootstrap_team_slug}"
            )
            exact.add(team)
        prefix = re.escape(f"/repos/{lane}")
        allowed_get = value in exact or any(
            re.fullmatch(prefix + suffix, value)
            for suffix in (
                r"/pulls/[1-9][0-9]*",
                r"/collaborators/[A-Za-z0-9-]+/permission",
                r"/actions/runs/[1-9][0-9]*",
                r"/actions/runs/[1-9][0-9]*/attempts/1/jobs\?per_page=100&page=[1-9][0-9]*",
                r"/actions/runs/[1-9][0-9]*/artifacts\?per_page=100&page=[1-9][0-9]*",
            )
        )
        allowed_collection = re.fullmatch(
            prefix + r"/commits/[0-9a-f]{40}/pulls", value
        ) is not None or bool(self.ceremony and value == team + "/members")
        if (method == "get" and not allowed_get) or (
            method == "collection" and not allowed_collection
        ):
            raise IdentityRefusal("read_path")
        response = getattr(self.api, method)(value)
        if value == f"/repos/{lane}":
            # GitHub repository responses may contain a temp_clone_token.
            # Forward only the metadata used by the deployment audit.
            response = {
                key: response.get(key)
                for key in (
                    "full_name",
                    "allow_rebase_merge",
                    "allow_squash_merge",
                    "allow_merge_commit",
                )
            }
        return {"value": public_metadata(response)}


class ReadGateway:
    """Broker-only read endpoint. Public RSA pin is custodian-installed."""

    def __init__(
        self, operations: ReadOperations, public_key, *, app_id: int, audience: str
    ):
        self.operations, self.public_key = operations, public_key
        self.issuer = "axiom-notary-broker:" + str(app_id)
        self.audience = audience

    def read(self, bearer: str, request: dict):
        if not isinstance(bearer, str) or len(bearer) > 32768:
            raise IdentityRefusal("read_gateway_identity")
        try:
            claims = jwt.decode(
                bearer,
                self.public_key,
                algorithms=["RS256"],
                issuer=self.issuer,
                audience=self.audience,
                options={
                    "require": ["iss", "aud", "sub", "iat", "nbf", "exp"],
                    "strict_aud": True,
                },
            )
            if (
                claims["sub"] != "control-plane-read"
                or any(type(claims[k]) is not int for k in ("iat", "nbf", "exp"))
                or not 0 < claims["exp"] - claims["iat"] <= 120
                or claims["nbf"] != claims["iat"]
            ):
                raise ValueError
        except (jwt.PyJWTError, ValueError, TypeError):
            raise IdentityRefusal("read_gateway_identity") from None
        return self.operations.perform(request)


def gateway_bearer(app, audience):
    now = int(time.time())
    return jwt.encode(
        {
            "iss": "axiom-notary-broker:" + str(app.scope.app_id),
            "aud": audience,
            "sub": "control-plane-read",
            "iat": now - 5,
            "nbf": now - 5,
            "exp": now + 60,
        },
        app._key,
        algorithm="RS256",
    )


class RPCClient:
    def __init__(self, endpoint: str, bearer):
        parsed = urlparse(endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.port not in (None, 443)
            or parsed.path not in ("", "/")
        ):
            raise IdentityRefusal("service_endpoint")
        self.endpoint, self.bearer = endpoint.rstrip("/"), bearer

    def call(self, operation: str, body: dict):
        if operation not in {
            "read",
            "receipt",
            "transition",
            "genesis",
            "publish-tokens",
            "release",
            "finalize",
        }:
            raise IdentityRefusal("service_operation")
        with requests.Session() as session:
            session.trust_env = False
            try:
                response = session.post(
                    self.endpoint + "/v1/" + operation,
                    headers={
                        "Authorization": "Bearer " + self.bearer(),
                        "Accept": "application/json",
                    },
                    json=body,
                    timeout=(5, 180),
                    stream=True,
                    allow_redirects=False,
                )
                if response.status_code != 200:
                    response.close()
                    raise IdentityRefusal("service_refused_or_uncertain")
                result = strict_parse(_bounded(response, 32_000_000))
            except requests.RequestException:
                raise IdentityRefusal("service_refused_or_uncertain") from None
        if not isinstance(result, dict):
            raise IdentityRefusal("service_response")
        return result


class ProxyReader:
    def __init__(self, client: RPCClient):
        self.client = client

    def _read(self, method, value):
        result = self.client.call("read", {"method": method, "value": value})
        if not fields(result, {"value"}):
            raise IdentityRefusal("read_response")
        return result["value"]

    def get(self, path):
        result = self._read("get", path)
        if not isinstance(result, dict):
            raise IdentityRefusal("read_response")
        return result

    def collection(self, path):
        result = self._read("collection", path)
        if not isinstance(result, list):
            raise IdentityRefusal("read_response")
        return result

    def optional_ref(self, repository, branch):
        result = self._read(
            "optional_ref", {"repository": repository, "branch": branch}
        )
        if result is not None and not isinstance(result, dict):
            raise IdentityRefusal("read_response")
        return result

    def archive(self, repository, artifact_id):
        result = self.client.call(
            "read",
            {
                "method": "archive",
                "value": {"repository": repository, "artifact_id": artifact_id},
            },
        )
        raw = decode_base64(result.get("archive_base64"))
        if (
            not fields(result, {"archive_base64"})
            or raw is None
            or len(raw) > 8_000_000
        ):
            raise IdentityRefusal("read_response")
        return raw
