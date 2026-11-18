"""Broker-side finalization. A workflow supplies identity, never markers or tokens."""

from __future__ import annotations

from .chain import Anchor, reconstruct
from .consumer import parse_consumer
from .github_inputs import GitHubSignerInputs
from .github_publication import AdmissionChecks, ChainWriter
from .identity import IdentityRefusal, JobIdentity, require_writer_submission
from .manifest import manifest_sha256
from .producers import require_enrolled_submission
from .protocol import decimal_id, oid, parse_artifact
from .publication import plan_finalization, plan_void
from .refusal import Refusal
from .remote import GitHubReader, RemoteRepository


def repair_recent_void_checks(history, state, checks):
    """Retry check invalidation after a successful chain write and API failure.

    Inspect voids since (and including) the latest finalized advancement;
    older subjects are also stale under strict up-to-date branch protection.
    """
    head = history[-1].blobs["HEAD.json"]
    for index in range(len(history) - 1, 0, -1):
        snapshot, parent = history[index], history[index - 1]
        for name in snapshot.blobs.keys() - parent.blobs.keys():
            if not name.endswith(".json"):
                continue
            marker = parse_artifact(snapshot.blobs[name], "void")
            if marker is None:
                continue
            target = marker["target_sha256"]
            body = parse_artifact(
                snapshot.blobs[target + ".json"], marker["target_kind"]
            )
            if marker["target_kind"] == "receipt":
                body = parse_artifact(
                    snapshot.blobs[body["candidate_sha256"] + ".json"],
                    "receipt-candidate",
                )
            checks.invalidate(
                body["subject_commit_git_oid"], state.anchor.epoch_sha256, target
            )
        if parent.blobs.get("HEAD.json") != head:
            break


class Finalizer:
    def __init__(self, inputs: GitHubSignerInputs, *, app_api):
        self.inputs, self.config, self.app_api = inputs, inputs.config, app_api

    def finalize(self, identity: JobIdentity, chain_token: str, lane_token: str):
        c = self.config
        self.inputs._audit("notary-publishing")
        if (
            identity.repository != c.repository
            or identity.environment != "notary-publishing"
            or self.inputs._tip(c.repository, c.content_branch)
            != identity.workflow_sha_git_oid
        ):
            raise IdentityRefusal("finalizer_identity_or_tip")
        with RemoteRepository(c.repository, read_token=lane_token) as lane:
            commit = lane.fetch("refs/heads/" + c.content_branch)
            if commit != identity.workflow_sha_git_oid:
                raise IdentityRefusal("finalizer_tip_moved")
            merged = lane.snapshot(commit)
            parents = (
                lane._run("show", "-s", "--format=%P", commit).decode().strip().split()
            )
            if len(parents) not in (1, 2) or not all(oid(p) for p in parents):
                raise IdentityRefusal("finalizer_parent")
            base = lane.snapshot(parents[0])
        with RemoteRepository(
            c.repository + "-notary", read_token=chain_token
        ) as chain:
            tip = chain.fetch("refs/heads/chain")
            history = chain.chain_history(tip)
        state = None
        # A rotation's new consumer pin cannot authenticate the *pending*
        # predecessor. Try both authenticated Git states; full reconstruction
        # proves every rotation, and only the old base can finalize a pending
        # transition. The current pin handles idempotent retries after success.
        for snapshot in (merged, base):
            consumer = parse_consumer(snapshot.blobs.get(c.consumer_spec_path, b""))
            if consumer is None or consumer["epoch_sha256"] != c.epoch_sha256:
                continue
            anchor = Anchor(
                c.repository,
                c.epoch_sha256,
                c.repository + "-notary",
                consumer["notary_spki_sha256"],
            )
            candidate = reconstruct(history, anchor)
            if not isinstance(candidate, Refusal):
                state = candidate
                break
        if state is None and self.inputs._ceremony is not None:
            self.inputs._bootstrap_lock()
            anchor = Anchor(
                c.repository,
                c.epoch_sha256,
                c.repository + "-notary",
                self.inputs._ceremony.arguments["ceremony_notary_spki_sha256"],
            )
            candidate = reconstruct(history, anchor)
            if not isinstance(candidate, Refusal) and not candidate.activated:
                state = candidate
        if state is None:
            raise IdentityRefusal("finalizer_chain")
        checks = AdmissionChecks(
            self.app_api, c.repository, c.lane_app_id, c.check_name, lane_token
        )
        if state.tip.manifest == manifest_sha256(merged.manifest):
            repair_recent_void_checks(history, state, checks)
            return {
                "state": "already-finalized",
                "artifact_sha256": state.tip.address,
                "chain_commit": tip,
            }
        associated = self.inputs.api.collection(
            f"/repos/{c.repository}/commits/{commit}/pulls"
        )
        matches = [
            p
            for p in associated
            if isinstance(p, dict)
            and p.get("merge_commit_sha") == commit
            and isinstance(p.get("base"), dict)
            and p["base"].get("ref") == c.content_branch
        ]
        if len(matches) != 1 or not decimal_id(str(matches[0].get("number"))):
            raise IdentityRefusal("finalizer_merge_pr")
        number = str(matches[0]["number"])
        author_api = GitHubReader(lane_token)
        pr = author_api.get(f"/repos/{c.repository}/pulls/{number}")
        head = pr.get("head", {}).get("sha")
        if (
            not oid(head)
            or pr.get("merged") is not True
            or pr.get("merge_commit_sha") != commit
        ):
            raise IdentityRefusal("finalizer_merge_pr")
        target = checks.target(head, c.epoch_sha256, include_invalidated=True)
        if state.terminal.get(target) == "void":
            raw = history[-1].blobs.get(target + ".json", b"")
            artifact = parse_artifact(raw, "receipt") or parse_artifact(
                raw, "transition"
            )
            if artifact and "candidate_sha256" in artifact:
                artifact = parse_artifact(
                    history[-1].blobs.get(artifact["candidate_sha256"] + ".json", b""),
                    "receipt-candidate",
                )
            if artifact is None or artifact.get("subject_commit_git_oid") != head:
                raise IdentityRefusal("finalizer_void_subject")
            checks.invalidate(head, c.epoch_sha256, target)
            return {
                "state": "already-void",
                "artifact_sha256": target,
                "chain_commit": tip,
            }
        # An invalidated check can identify a durable void for recovery, but
        # never authorizes a new finalization.
        checks.target(head, c.epoch_sha256)
        if target not in state.pending or state.pending[target].commit != head:
            checks.invalidate(head, c.epoch_sha256, target)
            raise IdentityRefusal("finalizer_not_pending")
        pending = state.pending[target]
        reason = None
        if (
            manifest_sha256(base.manifest)
            != pending.claims["base_tree_manifest_sha256"]
        ):
            reason = "actual-merge-base-mismatch"
        else:
            try:
                if pending.kind == "receipt":
                    with RemoteRepository(c.repository, read_token=lane_token) as lane:
                        lane.fetch(head)
                        subject = lane.snapshot(head)
                    require_enrolled_submission(
                        author_api,
                        pr_number=number,
                        base=base,
                        subject=subject,
                        registry=state.registry,
                        report_raw=history[-1].blobs[
                            pending.claims["report_sha256"] + ".json"
                        ],
                        merged_commit=commit,
                    )
                else:
                    author = str(pr.get("user", {}).get("id"))
                    require_writer_submission(
                        author_api,
                        c.repository,
                        number,
                        head,
                        contributor_ids=frozenset({author}),
                        merged_commit=commit,
                    )
            except IdentityRefusal as exc:
                # Transport/metadata failures refuse for retry; only a live,
                # successful permission response establishes lost write access.
                if str(exc) != "writer_permission":
                    raise
                reason = "contributor-write-access-revoked"
        plan = (
            plan_void(history, state.anchor, [target], reason=reason)
            if reason
            else plan_finalization(history, state.anchor, target, merged)
        )
        if self.inputs._tip(c.repository, c.content_branch) != merged.commit:
            raise IdentityRefusal("finalizer_tip_moved")
        new_tip = ChainWriter(c.repository + "-notary", chain_token).publish(
            plan, history[-1]
        )
        # Void markers and HEAD committed atomically above. Invalidate every
        # corresponding subject check while the broker still holds its lease.
        for address, artifact in state.pending.items():
            if plan.state.terminal.get(address) == "void":
                checks.invalidate(artifact.commit, c.epoch_sha256, address)
        return {
            "state": "finalized" if plan.admitted else "void",
            "artifact_sha256": target,
            "chain_commit": new_tip,
        }
