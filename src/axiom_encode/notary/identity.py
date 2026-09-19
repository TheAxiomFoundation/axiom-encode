"""External-service authentication of exact GitHub Actions job identities.

OIDC proves the caller. GitHub's current control plane supplies job names,
attempt state and artifact metadata; runner-provided labels are not authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

import jwt
import requests

from .canonical import strict_parse
from .protocol import decimal_id, oid

ISSUER = "https://token.actions.githubusercontent.com"
JWKS = ISSUER + "/.well-known/jwks"


class IdentityRefusal(ValueError):
    pass


class ReadAPI(Protocol):
    def get(self, path: str) -> dict: ...


class GitHubReadAPI:
    """Read-only methods over a deployment-owned credential, never a job token."""

    def __init__(self, token: str):
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.headers.update(
            {
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def get(self, path: str) -> dict:
        if not path.startswith("/repos/") or ".." in path or "#" in path:
            raise IdentityRefusal("invalid_control_plane_path")
        response = self._session.get(
            "https://api.github.com" + path, timeout=(5, 20), allow_redirects=False
        )
        if response.status_code != 200 or len(response.content) > 8_000_000:
            raise IdentityRefusal("control_plane_unavailable")
        body = strict_parse(response.content)
        if not isinstance(body, dict):
            raise IdentityRefusal("control_plane_malformed")
        return body


@dataclass(frozen=True)
class JobPolicy:
    repository: str
    repository_id: str
    repository_owner_id: str
    workflow_path: str
    workflow_sha_git_oid: str
    ref: str
    environment: str
    audience: str
    job_name: str


@dataclass(frozen=True)
class JobIdentity:
    repository: str
    workflow_ref: str
    workflow_sha_git_oid: str
    ref: str
    run_id: str
    run_attempt: str
    check_run_id: str
    environment: str
    jti: str


def _jwt_shape(token: str):
    if not isinstance(token, str) or len(token) > 32768 or len(token.split(".")) != 3:
        raise IdentityRefusal("oidc_malformed")
    try:
        parts = [
            strict_parse(jwt.utils.base64url_decode(p)) for p in token.split(".")[:2]
        ]
    except (ValueError, TypeError) as exc:
        raise IdentityRefusal("oidc_malformed") from exc
    if any(not isinstance(p, dict) for p in parts):
        raise IdentityRefusal("oidc_malformed")
    if (
        parts[0].get("alg") != "RS256"
        or not isinstance(parts[0].get("kid"), str)
        or not parts[0]["kid"]
    ):
        raise IdentityRefusal("oidc_algorithm")


def verify_oidc(token: str, policy: JobPolicy, *, signing_key=None) -> dict:
    """Verify GitHub's signature and registered claims before reading identity.

    Production omits signing_key and always uses GitHub's fixed JWKS endpoint.
    The injectable public key supports offline adversarial signature tests.
    No URL or key from the token is trusted.
    """
    _jwt_shape(token)
    try:
        key = (
            signing_key
            if signing_key is not None
            else jwt.PyJWKClient(JWKS, timeout=10).get_signing_key_from_jwt(token).key
        )
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=policy.audience,
            issuer=ISSUER,
            options={
                "require": ["iss", "aud", "exp", "nbf", "iat", "jti"],
                "strict_aud": True,
            },
        )
    except (jwt.PyJWTError, OSError, ValueError) as exc:
        raise IdentityRefusal("oidc_signature_or_registered_claim") from exc
    exact = {
        "repository": policy.repository,
        "repository_id": policy.repository_id,
        "repository_owner_id": policy.repository_owner_id,
        "workflow_ref": policy.repository
        + "/"
        + policy.workflow_path
        + "@"
        + policy.ref,
        "workflow_sha": policy.workflow_sha_git_oid,
        "ref": policy.ref,
        "environment": policy.environment,
        "run_attempt": "1",
        "sub": f"repo:{policy.repository}:environment:{policy.environment.replace(':', '%3A')}",
    }
    if any(claims.get(k) != v for k, v in exact.items()):
        raise IdentityRefusal("oidc_identity_mismatch")
    if "job_workflow_ref" in claims or "job_workflow_sha" in claims:
        raise IdentityRefusal("reusable_workflow")
    if (
        any(
            not decimal_id(claims.get(k))
            for k in ("run_id", "run_attempt", "check_run_id")
        )
        or not oid(claims.get("workflow_sha"))
        or not isinstance(claims["jti"], str)
        or not claims["jti"]
    ):
        raise IdentityRefusal("oidc_identity_encoding")
    return claims


def jobs_for_attempt(
    api: ReadAPI, repository: str, run_id: str, attempt: str
) -> list[dict]:
    jobs = []
    for page in range(1, 101):
        data = api.get(
            f"/repos/{repository}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100&page={page}"
        )
        if (
            not isinstance(data.get("jobs"), list)
            or type(data.get("total_count")) is not int
            or data["total_count"] < 0
        ):
            raise IdentityRefusal("jobs_malformed")
        jobs.extend(data["jobs"])
        if len(jobs) == data["total_count"]:
            return jobs
        if not data["jobs"] or len(jobs) > data["total_count"]:
            break
    raise IdentityRefusal("jobs_incomplete")


def _check_id(job: dict, repository: str) -> str | None:
    raw = job.get("check_run_url")
    if not isinstance(raw, str):
        return None
    parsed = urlparse(raw)
    prefix = "/repos/" + repository + "/check-runs/"
    value = parsed.path.removeprefix(prefix)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.github.com"
        or not parsed.path.startswith(prefix)
        or parsed.query
        or parsed.fragment
        or not decimal_id(value)
    ):
        return None
    return value


def authenticate_job(
    token: str, policy: JobPolicy, api: ReadAPI, *, signing_key=None
) -> JobIdentity:
    claims = verify_oidc(token, policy, signing_key=signing_key)
    run = api.get(f"/repos/{policy.repository}/actions/runs/{claims['run_id']}")
    if (
        str(run.get("id")) != claims["run_id"]
        or str(run.get("run_attempt")) != claims["run_attempt"]
        or run.get("path") != policy.workflow_path
        or run.get("event") != "workflow_dispatch"
        or run.get("head_branch") != policy.ref.removeprefix("refs/heads/")
        or run.get("status") != "in_progress"
    ):
        raise IdentityRefusal("run_identity_or_state")
    if (
        not isinstance(run.get("repository"), dict)
        or str(run["repository"].get("id")) != policy.repository_id
        or run["repository"].get("full_name") != policy.repository
    ):
        raise IdentityRefusal("run_repository")
    jobs = jobs_for_attempt(
        api, policy.repository, claims["run_id"], claims["run_attempt"]
    )
    matching = [
        job
        for job in jobs
        if isinstance(job, dict)
        and _check_id(job, policy.repository) == claims["check_run_id"]
    ]
    if (
        len(matching) != 1
        or matching[0].get("name") != policy.job_name
        or matching[0].get("status") != "in_progress"
        or str(matching[0].get("run_attempt")) != claims["run_attempt"]
        or str(matching[0].get("run_id")) != claims["run_id"]
    ):
        raise IdentityRefusal("job_identity_or_state")
    return JobIdentity(
        policy.repository,
        claims["workflow_ref"],
        claims["workflow_sha"],
        claims["ref"],
        claims["run_id"],
        claims["run_attempt"],
        claims["check_run_id"],
        claims["environment"],
        claims["jti"],
    )


def require_writer_submission(
    api: ReadAPI,
    repository: str,
    pr_number: str,
    subject: str,
    *,
    contributor_ids: frozenset[str],
) -> dict:
    """Authenticate PR metadata and current write access; reject forks and drafts.

    contributor_ids comes from the custodian-approved host enrollment, never
    a request's author field. It binds an enrolled producer to its operators.
    """
    if not decimal_id(pr_number) or not oid(subject):
        raise IdentityRefusal("submission_encoding")
    pr = api.get(f"/repos/{repository}/pulls/{pr_number}")
    try:
        login, author_id = pr["user"]["login"], str(pr["user"]["id"])
        if (
            not isinstance(login, str)
            or re_login(login) is False
            or author_id not in contributor_ids
            or pr["state"] != "open"
            or pr["head"]["sha"] != subject
            or pr["head"]["repo"]["full_name"] != repository
            or pr["base"]["repo"]["full_name"] != repository
            or pr.get("draft") is not False
        ):
            raise IdentityRefusal("submission_identity")
        permission = api.get(f"/repos/{repository}/collaborators/{login}/permission")
        if (
            permission.get("permission") not in ("write", "admin")
            or str(permission["user"]["id"]) != author_id
            or permission["user"]["login"] != login
        ):
            raise IdentityRefusal("writer_permission")
    except (KeyError, TypeError) as exc:
        raise IdentityRefusal("submission_metadata") from exc
    return pr


def re_login(login: str) -> bool:
    import re

    return re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", login) is not None
