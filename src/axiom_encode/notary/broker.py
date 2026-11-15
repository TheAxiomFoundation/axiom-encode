"""Separate two-App broker: publish tokens or broker-executed finalization.

No notary key, model credential or generic signing operation belongs here.
Deployment configuration and App roots are installed by the custodian.
"""

from __future__ import annotations

from dataclasses import asdict

from .apps import GitHubAppAPI, PublisherApps
from .finalizer import Finalizer
from .github_inputs import GitHubSignerInputs
from .identity import IdentityRefusal, JobPolicy, authenticate_job
from .leases import PublisherLeases, owner
from .publisher import completed_job


class PublisherTokenBroker:
    def __init__(
        self,
        inputs: GitHubSignerInputs,
        apps: PublisherApps,
        leases: PublisherLeases,
        *,
        finalizer=None,
    ):
        c = inputs.config
        if (
            apps.lane.scope.repository != c.repository
            or str(apps.lane.scope.repository_id) != c.repository_id
            or str(apps.lane.scope.owner_id) != c.repository_owner_id
            or apps.lane.scope.app_id != c.lane_app_id
            or apps.chain.scope.app_id != c.chain_app_id
        ):
            raise IdentityRefusal("broker_app_configuration")
        self.inputs, self.apps, self.leases = inputs, apps, leases
        self.finalizer = finalizer or Finalizer(inputs, app_api=GitHubAppAPI())

    def _policy(self, operation: str) -> JobPolicy:
        if operation not in {"publish", "finalize"}:
            raise IdentityRefusal("broker_operation")
        c = self.inputs.config
        return JobPolicy(
            c.repository,
            c.repository_id,
            c.repository_owner_id,
            c.workflow_path if operation == "publish" else c.finalizer_workflow_path,
            self.inputs._tip(c.repository, c.content_branch),
            "refs/heads/" + c.content_branch,
            "notary-publishing",
            c.publishing_audience,
            operation,
            "workflow_dispatch" if operation == "publish" else "push",
        )

    def publish_tokens(self, oidc: str) -> dict:
        policy = self._policy("publish")
        identity = authenticate_job(oidc, policy, self.inputs.api)
        self.inputs._audit("notary-publishing")
        for name in ("verify", "recompute", "approve"):
            completed_job(self.inputs.api, identity, name)
        chain, lane = self.leases.acquire(identity, policy, kind="publish")
        # This is the only operation returning tokens. It runs exclusively for
        # the authenticated publish job, never the request-only finalizer.
        return {
            "identity": asdict(identity),
            "lease_owner": owner(identity),
            "chain_token": chain,
            "lane_token": lane,
        }

    def read(self, oidc: str, request: dict) -> dict:
        # Runner uses only broker OIDC; it never receives the service's read
        # credential or contacts the signing operation. No lease/token minted.
        from .readplane import ReadOperations

        policy = self._policy("publish")
        authenticate_job(oidc, policy, self.inputs.api)
        return ReadOperations(
            self.inputs.config, self.inputs.api, ceremony=self.inputs._ceremony
        ).perform(request)

    def release(self, oidc: str, lease_owner: str) -> dict:
        if not isinstance(lease_owner, str) or len(lease_owner) > 300:
            raise IdentityRefusal("broker_release_identity")
        # A merged branch may have moved since issuance. Release authenticates
        # the original lease policy; it cannot mint or extend any capability.
        policy = self.leases.policy_for_release(lease_owner)
        identity = authenticate_job(oidc, policy, self.inputs.api)
        if owner(identity) != lease_owner or policy.job_name != "publish":
            raise IdentityRefusal("broker_release_identity")
        self.leases.release(identity)
        return {"state": "released"}

    def finalize(self, oidc: str) -> dict:
        policy = self._policy("finalize")
        identity = authenticate_job(oidc, policy, self.inputs.api)
        self.inputs._audit("notary-publishing")
        with self.leases.finalization(identity, policy) as (chain, lane):
            return self.finalizer.finalize(identity, chain, lane)
