"""Pinned publisher runner: reconcile, publish, then set the App-owned check.

The separate broker authenticates this job and provisions limited tokens.
This code executes no candidate programs and has no signing operation.
"""

from __future__ import annotations

from .administration import build_genesis, build_transition
from .apps import GitHubAppAPI
from .bundle import BUNDLE_ARTIFACT, unpack_bundle
from .canonical import sha256_hex
from .chain import Anchor, reconstruct
from .github_inputs import GitHubSignerInputs
from .github_publication import AdmissionChecks, ChainWriter
from .identity import (
    IdentityRefusal,
    JobIdentity,
    _check_id,
    jobs_for_attempt,
    require_writer_submission,
)
from .producers import require_enrolled_submission
from .protocol import parse_artifact
from .provenance import REPORT_ARTIFACT, read_report_archive
from .publication import plan_genesis, plan_publication
from .refusal import Refusal
from .remote import GitHubReader, RemoteRepository
from .signer import ReceiptInputs, make_candidate


def completed_job(api, identity: JobIdentity, name: str) -> str:
    jobs = jobs_for_attempt(
        api, identity.repository, identity.run_id, identity.run_attempt
    )
    rows = [j for j in jobs if isinstance(j, dict) and j.get("name") == name]
    if (
        len(rows) != 1
        or rows[0].get("status") != "completed"
        or rows[0].get("conclusion") != "success"
        or str(rows[0].get("run_id")) != identity.run_id
        or str(rows[0].get("run_attempt")) != identity.run_attempt
        or _check_id(rows[0], identity.repository) is None
    ):
        raise IdentityRefusal("publisher_required_job")
    return _check_id(rows[0], identity.repository)


def chain_history(anchor, token):
    with RemoteRepository(anchor.notary_repository, read_token=token) as repository:
        tip = repository.fetch("refs/heads/chain")
        history = repository.chain_history(tip)
    state = reconstruct(history, anchor)
    if isinstance(state, Refusal):
        raise IdentityRefusal("publisher_chain")
    return history, state


class Publisher:
    def __init__(self, inputs: GitHubSignerInputs, *, app_api=None):
        self.inputs, self.config = inputs, inputs.config
        self.app_api = app_api or GitHubAppAPI()

    def publish(self, identity: JobIdentity, chain_token: str, lane_token: str):
        c, inputs = self.config, self.inputs
        if (
            identity.repository != c.repository
            or identity.environment != "notary-publishing"
        ):
            raise IdentityRefusal("publisher_identity")
        inputs._audit("notary-publishing")
        approve_id = completed_job(inputs.api, identity, "approve")
        archive = inputs._archive(identity, BUNDLE_ARTIFACT)
        transport = read_report_archive(
            archive, sha256_hex(archive), member="bundle.json"
        )
        kind, address, bundle = unpack_bundle(transport)
        body = parse_artifact(bundle.get(address + ".json", b""), kind)
        if body is None:
            raise IdentityRefusal("publisher_artifact")
        if kind == "genesis":
            return self._genesis(identity, address, bundle, chain_token)
        candidate_raw = (
            bundle.get(body["candidate_sha256"] + ".json", b"")
            if kind == "receipt"
            else bundle[address + ".json"]
        )
        claims = parse_artifact(
            candidate_raw, "receipt-candidate" if kind == "receipt" else "transition"
        )
        if claims is None:
            raise IdentityRefusal("publisher_candidate")
        base, subject, state, number = inputs._snapshots(identity, claims)
        author_api = GitHubReader(lane_token)
        if kind == "receipt":
            material = ReceiptInputs(
                base,
                subject,
                state,
                c.dependency_inventory,
                inputs._archive(identity, REPORT_ARTIFACT),
                candidate_raw,
                number,
            )
            if (
                make_candidate(inputs.api, identity, material, require_recompute=True)
                != candidate_raw
            ):
                raise IdentityRefusal("publisher_candidate_reconciliation")
            if body["authorization"]["approve_check_run_id"] != approve_id:
                raise IdentityRefusal("publisher_approval_job")
            require_enrolled_submission(
                author_api,
                pr_number=number,
                base=base,
                subject=subject,
                registry=state.registry,
                report_raw=bundle[claims["report_sha256"] + ".json"],
            )
        else:
            if (
                build_transition(base, subject, state, reason=claims["reason"])
                != candidate_raw
            ):
                raise IdentityRefusal("publisher_transition_reconciliation")
            # Administrative approval supplies key authority; the PR must
            # still have a current same-repository writer as its author.
            pr = author_api.get(f"/repos/{c.repository}/pulls/{number}")
            author = str(pr.get("user", {}).get("id"))
            require_writer_submission(
                author_api,
                c.repository,
                number,
                subject.commit,
                contributor_ids=frozenset({author}),
            )
        history, current = chain_history(state.anchor, chain_token)
        if current.tip.address != state.tip.address:
            raise IdentityRefusal("publisher_predecessor_moved")
        plan = plan_publication(
            history, state.anchor, bundle, address, kind, base, subject
        )
        commit = ChainWriter(c.repository + "-notary", chain_token).publish(
            plan, history[-1]
        )
        # Re-read the actual branch and live PR immediately before green.
        # The broker lease serializes all publisher/finalizer check mutations.
        _, confirmed = chain_history(state.anchor, chain_token)
        if (
            address not in confirmed.pending
            or confirmed.tip.address != state.tip.address
        ):
            raise IdentityRefusal("publisher_not_pending")
        fresh_base, fresh_subject, _, _ = inputs._snapshots(identity, claims)
        if fresh_base.commit != base.commit or fresh_subject.commit != subject.commit:
            raise IdentityRefusal("publisher_subject_moved")
        if kind == "receipt":
            require_enrolled_submission(
                author_api,
                pr_number=number,
                base=base,
                subject=subject,
                registry=state.registry,
                report_raw=bundle[claims["report_sha256"] + ".json"],
            )
        else:
            require_writer_submission(
                author_api,
                c.repository,
                number,
                subject.commit,
                contributor_ids=frozenset({author}),
            )
        check = AdmissionChecks(
            self.app_api, c.repository, c.lane_app_id, c.check_name, lane_token
        )
        check_id = check.write(
            subject.commit, state.anchor.epoch_sha256, address, admitted=True
        )
        return {
            "artifact_sha256": address,
            "chain_commit": commit,
            "check_run_id": check_id,
            "state": "pending",
        }

    def _genesis(self, identity, address, bundle, chain_token):
        material = self.inputs.genesis(identity, address)
        expected = build_genesis(material.snapshot, **material.arguments)
        if bundle.get(address + ".json") != expected or sha256_hex(expected) != address:
            raise IdentityRefusal("publisher_genesis_reconciliation")
        anchor = Anchor(
            self.config.repository,
            address,
            self.config.repository + "-notary",
            material.arguments["ceremony_notary_spki_sha256"],
        )
        plan = plan_genesis(anchor, bundle, material.snapshot)
        # Re-audit the frozen lane after all local work; the lease closes the
        # first-push race, while the external lane lock prevents source motion.
        self.inputs._bootstrap_lock()
        if (
            self.inputs._tip(self.config.repository, self.config.content_branch)
            != material.snapshot.commit
        ):
            raise IdentityRefusal("publisher_bootstrap_moved")
        commit = ChainWriter(anchor.notary_repository, chain_token).bootstrap(
            plan.first, plan.final
        )
        _, actual = chain_history(anchor, chain_token)
        if actual.tip.address != address or actual.sequence != 1:
            raise IdentityRefusal("publisher_bootstrap_reconstruction")
        return {
            "artifact_sha256": address,
            "chain_commit": commit,
            "state": "genesis-finalized",
        }
