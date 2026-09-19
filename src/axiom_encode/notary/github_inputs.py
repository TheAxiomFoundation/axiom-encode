"""Concrete read-only external-signer inputs from authenticated GitHub state."""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import sha256_hex
from .chain import Anchor, reconstruct
from .consumer import parse_consumer
from .identity import IdentityRefusal, JobIdentity, JobPolicy
from .manifest import manifest_sha256
from .protection import (
    require_bootstrap_lock,
    require_chain_protection,
    require_environment,
    require_lane_protection,
)
from .protocol import oid, parse_artifact
from .provenance import REPORT_ARTIFACT, artifact_for_run, read_report_archive
from .refusal import Refusal
from .remote import GitHubReader, RemoteRepository
from .signer import GenesisInputs, ReceiptInputs, TransitionInputs

CANDIDATE_ARTIFACT = "axiom-notary-candidate"


@dataclass(frozen=True)
class Deployment:
    repository: str
    repository_id: str
    repository_owner_id: str
    epoch_sha256: str
    content_branch: str
    consumer_spec_path: str
    workflow_path: str
    signing_audience: str
    reviewer_ids: frozenset[int]
    lane_ruleset_id: int
    chain_writer_ruleset_id: int
    chain_integrity_ruleset_id: int
    chain_app_id: int
    lane_app_id: int
    check_name: str
    dependency_inventory: bytes


@dataclass(frozen=True)
class BootstrapCeremony:
    """Custodian-installed values, never workflow arguments or request fields."""

    arguments: dict
    lock_ruleset_id: int
    bootstrap_team_id: int
    bootstrap_team_slug: str
    bootstrap_user_id: int


class GitHubSignerInputs:
    """Reads the live lane, chain, artifacts and protections for each request.

    Configuration is installed by the service custodian. No constructor value
    comes from an HTTP request. Bootstrap uses a separate ceremony adapter;
    until it is configured this production reader refuses genesis outright.
    """

    def __init__(
        self,
        config: Deployment,
        api: GitHubReader,
        *,
        read_token: str | None = None,
        ceremony: BootstrapCeremony | None = None,
    ):
        self.config, self.api, self._read_token, self._ceremony = (
            config,
            api,
            read_token,
            ceremony,
        )

    def _tip(self, repository: str, branch: str) -> str:
        body = self.api.get(f"/repos/{repository}/git/ref/heads/{branch}")
        value = body.get("object", {})
        if (
            body.get("ref") != "refs/heads/" + branch
            or not isinstance(value, dict)
            or value.get("type") != "commit"
            or not oid(value.get("sha"))
        ):
            raise IdentityRefusal("protected_ref_unavailable")
        return value["sha"]

    def job_policy(self) -> JobPolicy:
        c = self.config
        # A workflow SHA is a commit locator, not the workflow blob hash. It
        # changes with ordinary merges, so derive it from the live protected
        # base for this request instead of accepting a caller claim or pinning
        # yesterday's commit forever.
        return JobPolicy(
            c.repository,
            c.repository_id,
            c.repository_owner_id,
            c.workflow_path,
            self._tip(c.repository, c.content_branch),
            "refs/heads/" + c.content_branch,
            "notary-signing",
            c.signing_audience,
            "approve",
        )

    def _audit(self, environment="notary-signing"):
        c = self.config
        env = self.api.get(f"/repos/{c.repository}/environments/{environment}")
        branches = self.api.get(
            f"/repos/{c.repository}/environments/{environment}/deployment-branch-policies?per_page=100"
        )
        require_environment(
            env,
            branches,
            name=environment,
            protected_branch=c.content_branch,
            reviewer_ids=c.reviewer_ids,
        )
        lane = self.api.get(f"/repos/{c.repository}/rulesets/{c.lane_ruleset_id}")
        require_lane_protection(
            lane,
            repository=c.repository,
            ref="refs/heads/" + c.content_branch,
            lane_app_id=c.lane_app_id,
            check_name=c.check_name,
        )
        chain = c.repository + "-notary"
        require_chain_protection(
            repository=chain,
            chain_app_id=c.chain_app_id,
            writer_ruleset=self.api.get(
                f"/repos/{chain}/rulesets/{c.chain_writer_ruleset_id}"
            ),
            integrity_ruleset=self.api.get(
                f"/repos/{chain}/rulesets/{c.chain_integrity_ruleset_id}"
            ),
        )

    def _archive(self, identity: JobIdentity, name: str) -> bytes:
        artifact = artifact_for_run(
            self.api, self.config.repository, identity.run_id, name
        )
        archive = self.api.archive(self.config.repository, str(artifact["id"]))
        if artifact.get("digest") != "sha256:" + sha256_hex(archive):
            raise IdentityRefusal("artifact_digest")
        return archive

    def _candidate(self, identity: JobIdentity, address: str, kind: str) -> bytes:
        archive = self._archive(identity, CANDIDATE_ARTIFACT)
        raw = read_report_archive(archive, sha256_hex(archive), member="candidate.json")
        if sha256_hex(raw) != address or parse_artifact(raw, kind) is None:
            raise IdentityRefusal("candidate_address_or_schema")
        return raw

    def _snapshots(self, identity: JobIdentity, candidate: dict):
        c = self.config
        self._audit()
        if (
            identity.repository != c.repository
            or identity.workflow_sha_git_oid
            != self._tip(c.repository, c.content_branch)
        ):
            raise IdentityRefusal("workflow_base_moved")
        subject_commit = candidate["subject_commit_git_oid"]
        prs = self.api.collection(
            f"/repos/{c.repository}/commits/{subject_commit}/pulls"
        )
        matching = [
            p
            for p in prs
            if isinstance(p, dict)
            and p.get("state") == "open"
            and isinstance(p.get("head"), dict)
            and p["head"].get("sha") == subject_commit
            and isinstance(p.get("base"), dict)
            and p["base"].get("ref") == c.content_branch
        ]
        if len(matching) != 1:
            raise IdentityRefusal("candidate_pull_request_not_unique")
        pr_number = str(matching[0].get("number"))
        from .protocol import decimal_id

        if not decimal_id(pr_number):
            raise IdentityRefusal("candidate_pull_request_number")
        with RemoteRepository(c.repository, read_token=self._read_token) as lane:
            base_commit = lane.fetch("refs/heads/" + c.content_branch)
            if base_commit != identity.workflow_sha_git_oid:
                raise IdentityRefusal("workflow_base_moved")
            base = lane.snapshot(base_commit)
            actual_subject = lane.fetch(f"refs/pull/{pr_number}/head")
            if actual_subject != subject_commit:
                raise IdentityRefusal("candidate_head_moved")
            subject = lane.snapshot(actual_subject)
        consumer = parse_consumer(base.blobs.get(c.consumer_spec_path, b""))
        bootstrap = (
            consumer is None
            or consumer["epoch_sha256"] != c.epoch_sha256
            or consumer["lane"] != c.repository
        )
        if bootstrap:
            if self._ceremony is None:
                raise IdentityRefusal("deployed_consumer_binding")
            self._bootstrap_lock()
            anchor = Anchor(
                c.repository,
                c.epoch_sha256,
                c.repository + "-notary",
                self._ceremony.arguments["ceremony_notary_spki_sha256"],
            )
        else:
            if (
                consumer["lane"] != c.repository
                or consumer["epoch_sha256"] != c.epoch_sha256
            ):
                raise IdentityRefusal("deployed_consumer_binding")
            anchor = Anchor(
                c.repository,
                c.epoch_sha256,
                consumer["notary_repository"],
                consumer["notary_spki_sha256"],
            )
        chain_tip = self._tip(anchor.notary_repository, "chain")
        with RemoteRepository(
            anchor.notary_repository, read_token=self._read_token
        ) as chain:
            if chain.fetch("refs/heads/chain") != chain_tip:
                raise IdentityRefusal("chain_moved")
            state = reconstruct(chain.chain_history(chain_tip), anchor)
        if isinstance(state, Refusal) or state.tip.manifest != manifest_sha256(
            base.manifest
        ):
            raise IdentityRefusal("chain_or_finalized_base")
        if bootstrap and state.activated:
            raise IdentityRefusal("active_consumer_missing")
        if (
            self._tip(c.repository, c.content_branch) != base.commit
            or self._tip(anchor.notary_repository, "chain") != chain_tip
        ):
            raise IdentityRefusal("remote_state_moved")
        return base, subject, state, pr_number

    def receipt(self, identity: JobIdentity, candidate_sha256: str) -> ReceiptInputs:
        raw = self._candidate(identity, candidate_sha256, "receipt-candidate")
        candidate = parse_artifact(raw, "receipt-candidate")
        base, subject, state, pr_number = self._snapshots(identity, candidate)
        return ReceiptInputs(
            base,
            subject,
            state,
            self.config.dependency_inventory,
            self._archive(identity, REPORT_ARTIFACT),
            raw,
            pr_number,
        )

    def transition(
        self, identity: JobIdentity, candidate_sha256: str
    ) -> TransitionInputs:
        raw = self._candidate(identity, candidate_sha256, "transition")
        base, subject, state, _ = self._snapshots(
            identity, parse_artifact(raw, "transition")
        )
        return TransitionInputs(base, subject, state, raw)

    def genesis(self, identity: JobIdentity, candidate_sha256: str) -> GenesisInputs:
        c = self.config
        self._audit()
        self._bootstrap_lock()
        if self.api.optional_ref(c.repository + "-notary", "chain") is not None:
            raise IdentityRefusal("second_genesis")
        if identity.repository != c.repository:
            raise IdentityRefusal("bootstrap_identity")
        raw = self._candidate(identity, candidate_sha256, "genesis")
        tip = self._tip(c.repository, c.content_branch)
        if tip != identity.workflow_sha_git_oid:
            raise IdentityRefusal("bootstrap_tip_moved")
        with RemoteRepository(c.repository, read_token=self._read_token) as lane:
            if lane.fetch("refs/heads/" + c.content_branch) != tip:
                raise IdentityRefusal("bootstrap_tip_moved")
            snapshot = lane.snapshot(tip)
        self._bootstrap_lock()
        if self._tip(c.repository, c.content_branch) != tip:
            raise IdentityRefusal("bootstrap_tip_moved")
        if self.api.optional_ref(c.repository + "-notary", "chain") is not None:
            raise IdentityRefusal("second_genesis")
        return GenesisInputs(snapshot, self._ceremony.arguments, raw)

    def _bootstrap_lock(self):
        c, ceremony = self.config, self._ceremony
        if ceremony is None:
            raise IdentityRefusal("bootstrap_ceremony_required")
        ruleset = self.api.get(
            f"/repos/{c.repository}/rulesets/{ceremony.lock_ruleset_id}"
        )
        require_bootstrap_lock(
            ruleset,
            repository=c.repository,
            ref="refs/heads/" + c.content_branch,
            bootstrap_actor_id=ceremony.bootstrap_team_id,
        )
        organization = c.repository.split("/")[0]
        team = self.api.get(
            f"/orgs/{organization}/teams/{ceremony.bootstrap_team_slug}"
        )
        members = self.api.collection(
            f"/orgs/{organization}/teams/{ceremony.bootstrap_team_slug}/members"
        )
        if (
            team.get("id") != ceremony.bootstrap_team_id
            or len(members) != 1
            or not isinstance(members[0], dict)
            or members[0].get("id") != ceremony.bootstrap_user_id
        ):
            raise IdentityRefusal("bootstrap_sole_actor")
