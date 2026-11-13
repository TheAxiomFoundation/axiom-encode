"""Candidate report construction and independent trusted reconciliation.

The caller supplies a reconstructed finalized predecessor and pinned verifier
inventory. This module never treats command-line base/pin choices as authority.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from axiom_encode.toolchain import parse_rulespec_toolchain_bytes

from .canonical import jcs_dumps, sha256_hex
from .coverage import Coverage, compute_coverage
from .lineage import (
    POLICY_PATH,
    STORE_PREFIX,
    StoreFile,
    classify_lineage,
    parse_path_policy,
)
from .manifest import (
    Manifest,
    build_tree_manifest,
    fsck_clean,
    manifest_diff,
    manifest_sha256,
)
from .protocol import (
    ESTABLISHED,
    PROFILE_PATH,
    REFUSAL_DETAILS,
    TRANSITION_POLICY_PATH,
    check_gates,
    gate_declarations_well_formed,
    oid,
    parse_artifact,
    parse_transition_policy,
)
from .refusal import Refusal
from .registry import REGISTRY_PATH, parse_registry


@dataclass(frozen=True)
class Predecessor:
    lane: str
    epoch_sha256: str
    address: str
    kind: str
    commit: str
    manifest_sha256: str
    notary_spki_sha256: str
    legacy_apply_spki_sha256: str
    legacy_eval_spki_sha256: str
    consumer_spec_path: str
    activated: bool


@dataclass(frozen=True)
class Snapshot:
    commit: str
    manifest: Manifest
    blobs: Mapping[str, bytes]

    @classmethod
    def read(cls, repository: Path, commit: str) -> Snapshot | Refusal:
        if not oid(commit):
            return Refusal(
                "subject-unresolvable", None, REFUSAL_DETAILS["subject-unresolvable"]
            )
        try:
            kind = subprocess.run(
                [
                    "git",
                    "--no-replace-objects",
                    "-C",
                    str(repository),
                    "cat-file",
                    "-t",
                    commit,
                ],
                capture_output=True,
                check=False,
            )
        except OSError:
            return Refusal(
                "subject-unresolvable", None, REFUSAL_DETAILS["subject-unresolvable"]
            )
        if kind.returncode or kind.stdout != b"commit\n":
            return Refusal(
                "subject-unresolvable", None, REFUSAL_DETAILS["subject-unresolvable"]
            )
        manifest = build_tree_manifest(repository, commit)
        if isinstance(manifest, Refusal):
            return manifest
        blobs = {}
        for path, _, digest in manifest:
            result = subprocess.run(
                [
                    "git",
                    "--no-replace-objects",
                    "-C",
                    str(repository),
                    "cat-file",
                    "blob",
                    f"{commit}:{path}",
                ],
                capture_output=True,
                check=False,
            )
            if result.returncode or sha256_hex(result.stdout) != digest:
                return Refusal("structural", path, REFUSAL_DETAILS["structural"])
            blobs[path] = result.stdout
        return cls(commit, manifest, blobs)

    def store(self) -> dict[str, StoreFile]:
        return {
            path[len(STORE_PREFIX) :]: StoreFile(self.blobs[path], mode)
            for path, mode, _ in self.manifest
            if path.startswith(STORE_PREFIX)
        }


def _commit_resolves(repository: Path, commit: str) -> bool:
    if not oid(commit):
        return False
    try:
        result = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(repository),
                "cat-file",
                "-t",
                commit,
            ],
            capture_output=True,
            check=False,
        )
        tree = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(repository),
                "cat-file",
                "-t",
                commit + "^{tree}",
            ],
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return (
        result.returncode == 0
        and result.stdout == b"commit\n"
        and tree.returncode == 0
        and tree.stdout == b"tree\n"
    )


def refusal_report(
    lane: str,
    epoch: str,
    subject: str,
    stage: str,
    refusal: Refusal,
    established: dict,
    gate_evidence: dict | None = None,
) -> bytes:
    body = {
        "schema": "axiom/notary-report-refusal/v1",
        "lane": lane,
        "epoch_sha256": epoch,
        "subject_commit_git_oid": subject,
        "stage": stage,
        "refusal": {
            "code": refusal.code,
            "path": refusal.path,
            "detail": REFUSAL_DETAILS[refusal.code],
        },
        "established": established,
    }
    if stage == "gates":
        body.update(gate_evidence or {})
    return jcs_dumps(body)


def _hard_trust_surface(path: str, consumer_spec_path: str) -> bool:
    # The transition policy may add surfaces; it cannot make these mutable in
    # an ordinary candidate. The store is governed by its own append-only wall.
    return (
        path == consumer_spec_path
        or path
        in {
            ".axiom/toolchain.toml",
            ".axiom/workflow-toolchain.toml",
            "known-validation-gaps.yaml",
            "CODEOWNERS",
            "docs/CODEOWNERS",
            "repository-structure.yaml",
        }
        or path == ".github"
        or path.startswith(".github/")
        or path == ".axiom/notary"
        or path.startswith(".axiom/notary/")
        or path.split("/")[-1] == ".gitattributes"
    )


def _trust_category(path: str, consumer_spec_path: str) -> int:
    # §4 sentence order, then bytewise-least path within that category.
    if path == ".github/workflows" or path.startswith(".github/workflows/"):
        return 0
    if path == ".github/actions" or path.startswith(".github/actions/"):
        return 1
    if path in {".axiom/toolchain.toml", ".axiom/workflow-toolchain.toml"}:
        return 2
    if path in {REGISTRY_PATH, consumer_spec_path}:
        return 3
    if path in {POLICY_PATH, TRANSITION_POLICY_PATH}:
        return 4
    if path == PROFILE_PATH:
        return 5
    if path == "known-validation-gaps.yaml":
        return 6
    return 7


def verify_snapshots(
    base: Snapshot,
    subject: Snapshot,
    predecessor: Predecessor,
    dependency_inventory: bytes,
    gates: list[dict],
) -> bytes:
    """Construct a report over already-total immutable Git snapshots.

    This is also the trusted recomputation function: gate outcomes are copied
    as declarations and checked against the profile, never upgraded to facts.
    """
    established = dict.fromkeys(ESTABLISHED)
    established.update(
        base_commit_git_oid=predecessor.commit,
        chain_predecessor_sha256=predecessor.address,
        chain_predecessor_kind=predecessor.kind,
    )

    def refuse(stage, code, path=None, gate_evidence=None):
        return refusal_report(
            predecessor.lane,
            predecessor.epoch_sha256,
            subject.commit,
            stage,
            Refusal(code, path, REFUSAL_DETAILS[code]),
            established,
            gate_evidence,
        )

    established["base_tree_manifest_sha256"] = manifest_sha256(base.manifest)
    established["subject_tree_manifest_sha256"] = manifest_sha256(subject.manifest)
    if not gate_declarations_well_formed(gates):
        return refuse("structural", "structural")
    if (
        base.commit != predecessor.commit
        or established["base_tree_manifest_sha256"] != predecessor.manifest_sha256
        or not predecessor.activated
    ):
        return refuse("preflight", "predecessor-stale")
    profile_raw = base.blobs.get(PROFILE_PATH, b"")
    profile = parse_artifact(profile_raw, "profile")
    if profile is None or profile["lane"] != predecessor.lane:
        return refuse("preflight", "policy-invalid", PROFILE_PATH)
    established["profile_sha256"] = sha256_hex(profile_raw)
    policy_raw = base.blobs.get(POLICY_PATH, b"")
    policy = parse_path_policy(policy_raw, lane=predecessor.lane)
    if isinstance(policy, Refusal):
        return refuse("preflight", "policy-invalid", POLICY_PATH)
    established["path_policy_sha256"] = sha256_hex(policy_raw)
    transition_policy = parse_transition_policy(
        base.blobs.get(TRANSITION_POLICY_PATH, b""), predecessor.lane
    )
    if isinstance(transition_policy, Refusal):
        return refuse("preflight", "policy-invalid", TRANSITION_POLICY_PATH)
    registry = parse_registry(
        base.blobs.get(REGISTRY_PATH, b""),
        lane=predecessor.lane,
        notary_spki_sha256=predecessor.notary_spki_sha256,
        legacy_apply_root=predecessor.legacy_apply_spki_sha256,
        legacy_eval_root=predecessor.legacy_eval_spki_sha256,
    )
    if isinstance(registry, Refusal):
        return refuse("preflight", "policy-invalid", REGISTRY_PATH)
    inventory = parse_artifact(dependency_inventory, "dependency-inventory")
    if inventory is None or inventory["lane"] != predecessor.lane:
        return refuse("preflight", "policy-invalid")
    # Base-bound toolchain and raw waiver bytes preserve the existing contract.
    try:
        toolchain = parse_rulespec_toolchain_bytes(
            base.blobs[".axiom/toolchain.toml"],
            root=Path("/" + predecessor.lane.split("/")[-1]),
        )
        waiver = base.blobs["known-validation-gaps.yaml"]
        if (
            len(waiver) > 2_000_000
            or sha256_hex(waiver) != toolchain.validation_waiver_set_sha256
        ):
            return refuse("preflight", "policy-invalid", "known-validation-gaps.yaml")
        corpus = {
            "name": toolchain.corpus_release,
            "content_sha256": toolchain.corpus_release_content_sha256,
        }
    except (KeyError, ValueError, UnicodeError, TypeError):
        return refuse("preflight", "policy-invalid", ".axiom/toolchain.toml")
    for change in sorted(
        manifest_diff(base.manifest, subject.manifest),
        key=lambda c: (
            _trust_category(c.path, predecessor.consumer_spec_path),
            c.path.encode(),
        ),
    ):
        if _hard_trust_surface(
            change.path, predecessor.consumer_spec_path
        ) or transition_policy.protects(change.path):
            return refuse("preflight", "trust-surface-change", change.path)
    # History mutation is a trust-surface preflight refusal; the underlying
    # classifier still reports structural when used by lineage diagnostics.
    base_store, subject_store = base.store(), subject.store()
    for name in sorted(base_store, key=str.encode):
        if subject_store.get(name) != base_store[name]:
            return refuse("preflight", "trust-surface-change", STORE_PREFIX + name)
    if (
        established["base_tree_manifest_sha256"]
        == established["subject_tree_manifest_sha256"]
    ):
        return refuse("preflight", "state-identical")
    lineage = classify_lineage(
        base_store,
        subject_store,
        lane=predecessor.lane,
        epoch_sha256=predecessor.epoch_sha256,
        registry=registry,
        path_policy=policy,
    )
    if isinstance(lineage, Refusal):
        return refuse("eligibility", lineage.code, lineage.path)
    established.update(
        eligible_records=[r.body_sha256 for r in lineage.eligible],
        ineligible_records=[
            asdict(r) | {"reasons": list(r.reasons)} for r in lineage.ineligible
        ],
    )
    coverage = compute_coverage(
        base.manifest, subject.manifest, lineage.eligible, policy
    )
    if isinstance(coverage, Refusal):
        return refuse("assignment", coverage.code, coverage.path)
    gate_refusal = check_gates(profile, gates)
    assert isinstance(coverage, Coverage)
    gate_evidence = dict(
        coverage_assignment=coverage.assignment_json(),
        unused_eligible_records=list(coverage.unused_eligible_records),
        unprotected_changes=list(coverage.unprotected_changes),
        gates=gates,
    )
    if gate_refusal:
        return refuse("gates", gate_refusal.code, gate_refusal.path, gate_evidence)
    body = {
        "schema": "axiom/notary-report-pass/v1",
        "lane": predecessor.lane,
        "epoch_sha256": predecessor.epoch_sha256,
        "subject_commit_git_oid": subject.commit,
        **established,
        **gate_evidence,
        "corpus_release": corpus,
        "waiver_set_sha256": sha256_hex(waiver),
        "dependency_pins_sha256": sha256_hex(dependency_inventory),
        "verifier": inventory["verifier"],
        "diff_coverage": "pass",
    }
    raw = jcs_dumps(body)
    if parse_artifact(raw, "report-pass") is None:
        # An invalid toolchain value must not leak a schema-invalid pass report.
        return refuse("preflight", "policy-invalid", ".axiom/toolchain.toml")
    return raw


def verify_repository(
    repository: Path,
    subject: str,
    predecessor: Predecessor,
    dependency_inventory: bytes,
    gates: list[dict],
) -> bytes:
    """Resolve structural inputs in the v33 base-before-subject order."""
    established = dict.fromkeys(ESTABLISHED)
    established.update(
        base_commit_git_oid=predecessor.commit,
        chain_predecessor_sha256=predecessor.address,
        chain_predecessor_kind=predecessor.kind,
    )

    def fail(stage, refusal):
        return refusal_report(
            predecessor.lane,
            predecessor.epoch_sha256,
            subject,
            stage,
            refusal,
            established,
        )

    if not _commit_resolves(repository, predecessor.commit):
        established = dict.fromkeys(ESTABLISHED)
        return fail("resolution", Refusal("chain-unresolvable", None, ""))
    if not _commit_resolves(repository, subject):
        return fail("resolution", Refusal("subject-unresolvable", None, ""))
    clean = fsck_clean(repository)
    if clean is not True:
        return fail("structural", Refusal("structural", None, ""))
    base = Snapshot.read(repository, predecessor.commit)
    if isinstance(base, Refusal):
        if base.code == "subject-unresolvable":
            established = dict.fromkeys(ESTABLISHED)
        return fail(
            "resolution" if base.code == "subject-unresolvable" else "structural",
            Refusal(
                "chain-unresolvable"
                if base.code == "subject-unresolvable"
                else base.code,
                base.path,
                "",
            ),
        )
    established["base_tree_manifest_sha256"] = manifest_sha256(base.manifest)
    candidate = Snapshot.read(repository, subject)
    if isinstance(candidate, Refusal):
        if candidate.code == "subject-unresolvable":
            established["base_tree_manifest_sha256"] = None
        return fail(
            "resolution" if candidate.code == "subject-unresolvable" else "structural",
            candidate,
        )
    return verify_snapshots(base, candidate, predecessor, dependency_inventory, gates)


def reconcile(proposed: bytes, recomputed: bytes) -> dict | Refusal:
    """Only a byte-identical, well-formed pass can become a receipt candidate."""
    body = parse_artifact(proposed, "report-pass")
    if (
        body is None
        or proposed != recomputed
        or parse_artifact(recomputed, "report-pass") is None
    ):
        return Refusal("structural", None, "report_reconciliation_failed")
    return body
