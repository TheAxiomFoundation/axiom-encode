"""Closed v33 chain, report, policy, and provenance wire schemas.

Parsing establishes syntax only. Signature, base-state, GitHub identity and
recomputation checks are deliberately separate from this module.
"""

from __future__ import annotations

import re

from ._schema import (
    canonical_object,
    decode_base64,
    digest,
    fields,
    lane_name,
    nonempty,
    ordered_strings,
    relative_path,
)
from .lineage import PathPolicy
from .refusal import Refusal

PREFIX = "axiom/notary-"
PROFILE_PATH = ".axiom/notary/profile.json"
TRANSITION_POLICY_PATH = ".axiom/notary/transition-path-policy.json"
KINDS = {"genesis", "receipt", "transition"}
REASONS = {
    "address-mismatch",
    "unrecognized-store-name",
    "malformed-record",
    "invalid-signature",
    "wrong-lane",
    "wrong-epoch",
    "duplicate-transition-paths",
    "unprotected-path-transition",
}
REPORT_FIELDS = {
    "schema",
    "lane",
    "epoch_sha256",
    "subject_commit_git_oid",
    "subject_tree_manifest_sha256",
    "base_commit_git_oid",
    "base_tree_manifest_sha256",
    "chain_predecessor_sha256",
    "chain_predecessor_kind",
    "profile_sha256",
    "path_policy_sha256",
    "corpus_release",
    "waiver_set_sha256",
    "eligible_records",
    "unused_eligible_records",
    "coverage_assignment",
    "gates",
    "diff_coverage",
    "unprotected_changes",
    "ineligible_records",
    "dependency_pins_sha256",
    "verifier",
}
ESTABLISHED = (
    "base_commit_git_oid",
    "chain_predecessor_sha256",
    "chain_predecessor_kind",
    "base_tree_manifest_sha256",
    "subject_tree_manifest_sha256",
    "profile_sha256",
    "path_policy_sha256",
    "eligible_records",
    "ineligible_records",
)
REFUSAL_DETAILS = {
    "subject-unresolvable": "subject commit or tree cannot be resolved",
    "chain-unresolvable": "finalized chain state cannot be reconstructed",
    "uncovered-path": "no eligible transition covers this path",
    "ambiguous-assignment": "more than one valid assignment",
    "inconsistent-chain": "eligible transitions exist but no valid chain",
    "record-cycle": "consumed records admit no execution order",
    "no-valid-execution": "no topological order yields realizable trees",
    "inadmissible-entry": "entry mode or type inadmissible",
    "structural": "tree or store structurally malformed",
    "state-identical": "base and subject states are identical",
    "trust-surface-change": "ordinary candidate changes a trust surface",
    "policy-invalid": "required policy, profile, or key registry absent or invalid at the base, or a path-policy expansion unsupported before v34",
    "predecessor-stale": "base is not the finalized chain tip",
    "gate-unacceptable": "gate outcome outside the acceptable set",
    "gate-missing": "required gate absent",
    "gate-extra": "gate outside the profile",
}


def oid(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def decimal_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value) is not None


def _objects(value: object, key: str, members: set[str]) -> bool:
    return (
        isinstance(value, list)
        and all(fields(v, members) for v in value)
        and ordered_strings([v[key] for v in value])
    )


def _digests(value: object) -> bool:
    return ordered_strings(value) and all(digest(v) for v in value)


def _paths(value: object) -> bool:
    return ordered_strings(value) and all(relative_path(v) for v in value)


def _named_commit(value: object) -> bool:
    return (
        fields(value, {"repo", "git_oid"})
        and lane_name(value["repo"])
        and oid(value["git_oid"])
    )


def _ineligible(value: object) -> bool:
    return _objects(value, "store_name", {"store_name", "reasons"}) and all(
        nonempty(v["store_name"])
        and ordered_strings(v["reasons"])
        and bool(v["reasons"])
        and set(v["reasons"]) <= REASONS
        and (
            "unrecognized-store-name" not in v["reasons"]
            or v["reasons"] == ["unrecognized-store-name"]
        )
        and (
            "malformed-record" not in v["reasons"]
            or set(v["reasons"]) <= {"address-mismatch", "malformed-record"}
        )
        for v in value
    )


def _assignment(value: object) -> bool:
    return _objects(value, "path", {"path", "record_sha256s"}) and all(
        relative_path(v["path"])
        and isinstance(v["record_sha256s"], list)
        and bool(v["record_sha256s"])
        and all(digest(d) for d in v["record_sha256s"])
        and len(v["record_sha256s"]) == len(set(v["record_sha256s"]))
        for v in value
    )


def _gates(value: object) -> bool:
    return _objects(value, "gate_id", {"gate_id", "outcome"}) and all(
        nonempty(v["gate_id"]) and nonempty(v["outcome"]) for v in value
    )


def gate_declarations_well_formed(value: object) -> bool:
    return _gates(value)


def _profile(body: dict) -> bool:
    if not fields(body, {"schema", "lane", "required_gates", "oracle_policy"}):
        return False
    if body["oracle_policy"] not in ("fail-closed", "reduced-tier") or not _objects(
        body["required_gates"], "gate_id", {"gate_id", "acceptable_outcomes", "tier"}
    ):
        return False
    return all(
        nonempty(g["gate_id"])
        and g["tier"] in ("public", "restricted", "ci-attested")
        and ordered_strings(g["acceptable_outcomes"])
        and all(nonempty(v) for v in g["acceptable_outcomes"])
        and (
            body["oracle_policy"] != "fail-closed"
            or "oracle-unavailable" not in g["acceptable_outcomes"]
        )
        for g in body["required_gates"]
    )


def _pass(body: dict) -> bool:
    if not (oid(body["subject_commit_git_oid"]) and oid(body["base_commit_git_oid"])):
        return False
    if not all(digest(body[name]) for name in REPORT_FIELDS if name.endswith("sha256")):
        return False
    if (
        not isinstance(body["chain_predecessor_kind"], str)
        or body["chain_predecessor_kind"] not in KINDS
        or body["diff_coverage"] != "pass"
    ):
        return False
    corpus = body["corpus_release"]
    if not (
        fields(corpus, {"name", "content_sha256"})
        and nonempty(corpus["name"])
        and digest(corpus["content_sha256"])
    ):
        return False
    if not (
        _digests(body["eligible_records"])
        and _digests(body["unused_eligible_records"])
        and _assignment(body["coverage_assignment"])
        and _ineligible(body["ineligible_records"])
        and _paths(body["unprotected_changes"])
        and _gates(body["gates"])
        and _named_commit(body["verifier"])
    ):
        return False
    return _partition(body)


def _partition(body: dict) -> bool:
    consumed = {r for a in body["coverage_assignment"] for r in a["record_sha256s"]}
    unused = set(body["unused_eligible_records"])
    return not consumed & unused and consumed | unused == set(body["eligible_records"])


def _job1(value: object) -> bool:
    names = {
        "workflow_ref",
        "workflow_sha_git_oid",
        "ref",
        "run_id",
        "run_attempt",
        "check_run_id",
        "conclusion",
        "artifact_name",
        "artifact_id",
        "artifact_sha256",
    }
    return (
        fields(value, names)
        and all(nonempty(value[k]) for k in names)
        and all(
            decimal_id(value[k])
            for k in ("run_id", "run_attempt", "check_run_id", "artifact_id")
        )
        and oid(value["workflow_sha_git_oid"])
        and digest(value["artifact_sha256"])
        and value["conclusion"] == "success"
        and value["ref"].startswith("refs/")
        and value["workflow_ref"].endswith("@" + value["ref"])
    )


def _delta(value: object) -> bool:
    if not _objects(
        value,
        "path",
        {
            "path",
            "before_entry_sha256",
            "before_mode",
            "after_entry_sha256",
            "after_mode",
        },
    ):
        return False
    for change in value:
        if not relative_path(change["path"]):
            return False
        for side in ("before", "after"):
            d, mode = change[f"{side}_entry_sha256"], change[f"{side}_mode"]
            if not (
                (d is None and mode is None)
                or (digest(d) and mode in ("100644", "100755"))
            ):
                return False
    return True


def _inventory(value: object, width: int) -> bool:
    return (
        isinstance(value, list)
        and all(
            isinstance(row, list)
            and len(row) == width
            and relative_path(row[0])
            and all(digest(v) for v in row[1:])
            for row in value
        )
        and ordered_strings([row[0] for row in value])
    )


def _genesis(body: dict) -> bool:
    if not fields(
        body,
        {
            "schema",
            "lane",
            "genesis_commit_git_oid",
            "genesis_tree_manifest_sha256",
            "bootstrap_policies",
            "activation_spec_template_sha256",
            "consumer_spec_path",
            "notary_repository",
            "legacy_apply_root",
            "legacy_eval_root",
            "v5_attested",
            "baseline_unattested",
        },
    ):
        return False
    if not (
        oid(body["genesis_commit_git_oid"])
        and digest(body["genesis_tree_manifest_sha256"])
        and digest(body["activation_spec_template_sha256"])
        and relative_path(body["consumer_spec_path"])
        and lane_name(body["notary_repository"])
    ):
        return False
    policies = body["bootstrap_policies"]
    if not (
        fields(
            policies,
            {
                "path_policy_sha256",
                "transition_path_policy_sha256",
                "profile_sha256",
                "key_registry_sha256",
            },
        )
        and all(digest(v) for v in policies.values())
    ):
        return False
    # Key bytes, raw id/SPKI agreement, and role disjointness are cryptographic
    # validation, performed by the genesis verifier after this closed parse.
    for name in ("legacy_apply_root", "legacy_eval_root"):
        root = body[name]
        if not (
            fields(root, {"raw_key_id", "public_key_spki_der_base64"})
            and isinstance(root["raw_key_id"], str)
            and root["raw_key_id"].startswith("sha256:")
            and digest(root["raw_key_id"][7:])
            and nonempty(root["public_key_spki_der_base64"])
            and decode_base64(root["public_key_spki_der_base64"]) is not None
        ):
            return False
    return _inventory(body["v5_attested"], 3) and _inventory(
        body["baseline_unattested"], 2
    )


def _refusal(body: dict) -> bool:
    extra = (
        {
            "coverage_assignment",
            "unused_eligible_records",
            "unprotected_changes",
            "gates",
        }
        if body.get("stage") == "gates"
        else set()
    )
    if not fields(
        body,
        {
            "schema",
            "lane",
            "epoch_sha256",
            "subject_commit_git_oid",
            "stage",
            "refusal",
            "established",
        }
        | extra,
    ):
        return False
    if not (digest(body["epoch_sha256"]) and oid(body["subject_commit_git_oid"])):
        return False
    stage = body["stage"]
    if stage not in (
        "resolution",
        "structural",
        "preflight",
        "eligibility",
        "assignment",
        "gates",
    ):
        return False
    refusal = body["refusal"]
    if not (
        fields(refusal, {"code", "path", "detail"})
        and isinstance(refusal["code"], str)
        and refusal["code"] in REFUSAL_DETAILS
        and refusal["detail"] == REFUSAL_DETAILS[refusal["code"]]
        and (refusal["path"] is None or isinstance(refusal["path"], str))
    ):
        return False
    established = body["established"]
    if not fields(established, set(ESTABLISHED)):
        return False
    seen_null = False
    validators = (
        oid,
        digest,
        lambda v: isinstance(v, str) and v in KINDS,
        digest,
        digest,
        digest,
        digest,
        _digests,
        _ineligible,
    )
    for name, validate in zip(ESTABLISHED, validators, strict=True):
        value = established[name]
        if value is None:
            seen_null = True
        elif seen_null or not validate(value):
            return False
    if stage == "gates":
        return (
            not seen_null
            and _assignment(body["coverage_assignment"])
            and _digests(body["unused_eligible_records"])
            and _paths(body["unprotected_changes"])
            and _gates(body["gates"])
            and _partition(established | {name: body[name] for name in extra})
        )
    return True


def parse_artifact(raw: bytes, expected: str | None = None) -> dict | None:
    """Return a fresh closed-schema body, never an authenticity assertion."""
    body = canonical_object(raw)
    if body is None or not lane_name(body.get("lane")):
        return None
    schema = body.get("schema")
    if not isinstance(schema, str) or (
        expected is not None and schema != PREFIX + expected + "/v1"
    ):
        return None
    if schema == PREFIX + "profile/v1":
        return body if _profile(body) else None
    if schema == PREFIX + "genesis/v1":
        return body if _genesis(body) else None
    if schema == PREFIX + "report-refusal/v1":
        return body if _refusal(body) else None
    if schema in (PREFIX + "report-pass/v1", PREFIX + "receipt-candidate/v1"):
        candidate = schema == PREFIX + "receipt-candidate/v1"
        names = REPORT_FIELDS | ({"report_sha256", "job1"} if candidate else set())
        if not fields(body, names) or not _pass(body):
            return None
        if candidate and not (digest(body["report_sha256"]) and _job1(body["job1"])):
            return None
        return body
    if schema == PREFIX + "dependency-inventory/v1":
        if (
            not fields(
                body,
                {
                    "schema",
                    "lane",
                    "actions",
                    "containers",
                    "python_lock_sha256",
                    "verifier",
                },
            )
            or not digest(body["python_lock_sha256"])
            or not _named_commit(body["verifier"])
        ):
            return None
        for name, validate in (("actions", oid), ("containers", digest)):
            values = body[name]
            if not (
                isinstance(values, list)
                and all(
                    isinstance(v, list)
                    and len(v) == 2
                    and nonempty(v[0])
                    and validate(v[1])
                    for v in values
                )
                and ordered_strings([v[0] for v in values])
            ):
                return None
        return body
    if not digest(body.get("epoch_sha256")):
        return None
    if schema == PREFIX + "receipt/v1":
        if not fields(
            body,
            {"schema", "lane", "epoch_sha256", "candidate_sha256", "authorization"},
        ) or not digest(body["candidate_sha256"]):
            return None
        auth = body["authorization"]
        return (
            body
            if fields(
                auth,
                {"environment", "approve_check_run_id", "approval_signature_sha256"},
            )
            and nonempty(auth["environment"])
            and decimal_id(auth["approve_check_run_id"])
            and digest(auth["approval_signature_sha256"])
            else None
        )
    if schema == PREFIX + "transition/v1":
        if not fields(
            body,
            {
                "schema",
                "lane",
                "epoch_sha256",
                "chain_predecessor_sha256",
                "chain_predecessor_kind",
                "base_tree_manifest_sha256",
                "subject_tree_manifest_sha256",
                "subject_commit_git_oid",
                "delta",
                "reason",
            },
        ):
            return None
        return (
            body
            if all(
                digest(body[k])
                for k in (
                    "chain_predecessor_sha256",
                    "base_tree_manifest_sha256",
                    "subject_tree_manifest_sha256",
                )
            )
            and isinstance(body["chain_predecessor_kind"], str)
            and body["chain_predecessor_kind"] in KINDS
            and oid(body["subject_commit_git_oid"])
            and _delta(body["delta"])
            and nonempty(body["reason"])
            else None
        )
    if schema in (PREFIX + "finalization/v1", PREFIX + "void/v1"):
        final = schema == PREFIX + "finalization/v1"
        names = {"schema", "lane", "epoch_sha256", "target_sha256", "target_kind"} | (
            {"merged_tip_manifest_sha256", "sequence"} if final else {"reason"}
        )
        if (
            not fields(body, names)
            or not digest(body["target_sha256"])
            or not isinstance(body["target_kind"], str)
            or body["target_kind"] not in KINDS
        ):
            return None
        return (
            body
            if (
                digest(body["merged_tip_manifest_sha256"])
                and decimal_id(body["sequence"])
                if final
                else nonempty(body["reason"])
            )
            else None
        )
    return None


def parse_transition_policy(raw: bytes, lane: str) -> PathPolicy | Refusal:
    body = canonical_object(raw)
    refusal = Refusal(
        "policy-invalid", TRANSITION_POLICY_PATH, REFUSAL_DETAILS["policy-invalid"]
    )
    if (
        not fields(body, {"schema", "lane", "rules"})
        or body["schema"] != PREFIX + "transition-path-policy/v1"
        or body["lane"] != lane
        or not isinstance(body["rules"], list)
    ):
        return refusal
    if any(
        not fields(r, {"action", "prefix"})
        or r["action"] not in ("include", "exclude")
        or not relative_path(r["prefix"])
        for r in body["rules"]
    ):
        return refusal
    policy = PathPolicy(lane, tuple((r["action"], r["prefix"]) for r in body["rules"]))
    for root, required in ((".axiom/notary", True), (".axiom/lineage", False)):
        boundaries = {root} | {p for _, p in policy.rules if p.startswith(root + "/")}
        if any(policy.protects(p) != required for p in boundaries):
            return refusal
    return policy


def check_gates(profile: dict, gates: list[dict]) -> Refusal | None:
    if not _gates(gates):
        return Refusal("structural", None, REFUSAL_DETAILS["structural"])
    required = {g["gate_id"]: g for g in profile["required_gates"]}
    supplied = {g["gate_id"]: g["outcome"] for g in gates}
    for code, paths in (
        ("gate-missing", required.keys() - supplied.keys()),
        ("gate-extra", supplied.keys() - required.keys()),
        (
            "gate-unacceptable",
            {
                k
                for k in required.keys() & supplied.keys()
                if supplied[k] not in required[k]["acceptable_outcomes"]
            },
        ),
    ):
        if paths:
            return Refusal(code, min(paths, key=str.encode), REFUSAL_DETAILS[code])
    return None
