"""Generated-file defense for activated lanes; never substitutes for admission.

A missing consumer at the trusted base returns 78 so the shared workflow runs
its mandatory legacy guard. A malformed consumer never selects that fallback.
The source-App-bound notary check remains a separate required merge condition.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import replace
from pathlib import Path

from .administration import build_transition
from .canonical import jcs_dumps, strict_parse
from .chain import Anchor, reconstruct
from .consumer import parse_consumer
from .deployment import require_running_identity
from .identity import IdentityRefusal, require_writer_submission
from .manifest import manifest_sha256
from .producers import enrolled_contributor_ids
from .protocol import decimal_id, parse_artifact
from .refusal import Refusal
from .remote import GitHubReader, RemoteRepository
from .verification import Snapshot, verify_snapshots


def inspect(base, subject, *, lane, history, api=None, pr_number=None):
    raw = base.blobs.get(".axiom/notary/consumer.json")
    if raw is None:
        return 78
    consumer = parse_consumer(raw)
    if consumer is None or consumer["lane"] != lane:
        raise IdentityRefusal("guard_consumer")
    anchor = Anchor(
        lane,
        consumer["epoch_sha256"],
        consumer["notary_repository"],
        consumer["notary_spki_sha256"],
    )
    state = reconstruct(history, anchor)
    if isinstance(state, Refusal) and pr_number is None:
        successor = parse_consumer(
            subject.blobs.get(".axiom/notary/consumer.json", b"")
        )
        if successor is not None and all(
            successor[key] == consumer[key]
            for key in ("lane", "epoch_sha256", "notary_repository")
        ):
            # The entire predecessor-authorized rotation must reconstruct,
            # and only an exact finalized subject can use the successor pin.
            candidate = reconstruct(
                history,
                replace(anchor, notary_spki_sha256=successor["notary_spki_sha256"]),
            )
            if (
                not isinstance(candidate, Refusal)
                and candidate.activated
                and candidate.tip.manifest == manifest_sha256(subject.manifest)
            ):
                return 0
    if isinstance(state, Refusal) or not state.activated:
        raise IdentityRefusal("guard_chain")
    base_manifest, subject_manifest = (
        manifest_sha256(base.manifest),
        manifest_sha256(subject.manifest),
    )
    if state.tip.manifest != base_manifest:
        # Post-merge CI may race after successful broker finalization. It can
        # accept only the exact newly finalized manifest, never another head.
        if pr_number is None and state.tip.manifest == subject_manifest:
            return 0
        raise IdentityRefusal("guard_finalized_base")
    config = strict_parse(base.blobs.get(".axiom/notary/runner.json", b""))
    if not isinstance(config, dict):
        raise IdentityRefusal("guard_runner_configuration")
    inventory = jcs_dumps(config["deployment"]["dependency_inventory"])
    if parse_artifact(inventory, "dependency-inventory") is None:
        raise IdentityRefusal("guard_inventory")
    require_running_identity(config["encoder_identity"], inventory)
    raw_report = verify_snapshots(
        base, subject, state.predecessor(lane_commit=base.commit), inventory, []
    )
    report = strict_parse(raw_report)
    if (
        report.get("stage") == "gates"
        or parse_artifact(raw_report, "report-pass") is not None
    ):
        allowed = enrolled_contributor_ids(
            base, subject, state.registry, report["coverage_assignment"]
        )
        if pr_number is not None:
            require_writer_submission(
                api, lane, pr_number, subject.commit, contributor_ids=allowed
            )
        return 0
    if report.get("refusal", {}).get("code") == "trust-surface-change":
        for artifact in state.pending.values():
            if (
                artifact.kind != "transition"
                or (pr_number is not None and artifact.commit != subject.commit)
                or artifact.manifest != subject_manifest
            ):
                continue
            if (
                build_transition(
                    base,
                    subject
                    if pr_number is not None
                    else replace(subject, commit=artifact.commit),
                    state,
                    reason=artifact.body["reason"],
                )
                == history[-1].blobs[artifact.address + ".json"]
            ):
                if pr_number is not None:
                    pr = api.get(f"/repos/{lane}/pulls/{pr_number}")
                    require_writer_submission(
                        api,
                        lane,
                        pr_number,
                        subject.commit,
                        contributor_ids=frozenset({str(pr.get("user", {}).get("id"))}),
                    )
                return 0
    raise IdentityRefusal("guard_lineage_or_administrative_authorization")


def command(args):
    def snapshot(ref):
        commit = (
            subprocess.check_output(
                [
                    "git",
                    "--no-replace-objects",
                    "-C",
                    str(args.repo),
                    "rev-parse",
                    "--verify",
                    "--end-of-options",
                    ref + "^{commit}",
                ]
            )
            .decode()
            .strip()
        )
        value = Snapshot.read(args.repo, commit)
        if isinstance(value, Refusal):
            raise IdentityRefusal("guard_snapshot")
        return value

    try:
        base, subject = snapshot(args.base_ref), snapshot(args.head_ref)
        raw = base.blobs.get(".axiom/notary/consumer.json")
        if raw is None:
            return 78
        consumer = parse_consumer(raw)
        if consumer is None or consumer["lane"] != args.lane:
            raise IdentityRefusal("guard_consumer")
        if args.pr_number is not None and not decimal_id(args.pr_number):
            raise IdentityRefusal("guard_pull_request")
        with RemoteRepository(consumer["notary_repository"]) as remote:
            history = remote.chain_history(remote.fetch("refs/heads/chain"))
        token = os.environ.get("GITHUB_TOKEN")
        if getattr(args, "github_token_file", None):
            # The protected CI wrapper writes its contents-read job token to
            # a private temporary file; the verification supervisor strips
            # ambient tokens. No credential is copied into evidence.
            from .deployment import custodian_file

            token = (
                custodian_file(str(args.github_token_file), private=True, limit=10000)
                .decode()
                .strip()
            )
        return inspect(
            base,
            subject,
            lane=args.lane,
            history=history,
            api=GitHubReader(token),
            pr_number=args.pr_number,
        )
    except Exception:
        return 1


def add_arguments(parser):
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--head-ref", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--pr-number")
    parser.add_argument("--github-token-file", type=Path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    raise SystemExit(command(parser.parse_args()))


if __name__ == "__main__":
    main()
