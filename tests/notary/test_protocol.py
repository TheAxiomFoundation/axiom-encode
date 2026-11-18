"""Closed wire formats are tested separately from trust and recomputation."""

from copy import deepcopy

import pytest

from axiom_encode.notary.canonical import jcs_dumps
from axiom_encode.notary.protocol import (
    ESTABLISHED,
    REFUSAL_DETAILS,
    check_gates,
    parse_artifact,
    parse_transition_policy,
)
from axiom_encode.notary.refusal import Refusal

from .lineage_fixtures import EPOCH, LANE


def report():
    return {
        "schema": "axiom/notary-report-pass/v1",
        "lane": LANE,
        "epoch_sha256": EPOCH,
        "subject_commit_git_oid": "1" * 40,
        "base_commit_git_oid": "2" * 40,
        "subject_tree_manifest_sha256": "3" * 64,
        "base_tree_manifest_sha256": "4" * 64,
        "chain_predecessor_sha256": "5" * 64,
        "chain_predecessor_kind": "transition",
        "profile_sha256": "6" * 64,
        "path_policy_sha256": "7" * 64,
        "corpus_release": {"name": "fixture", "content_sha256": "8" * 64},
        "waiver_set_sha256": "9" * 64,
        "eligible_records": ["a" * 64],
        "unused_eligible_records": [],
        "coverage_assignment": [{"path": "rules/x", "record_sha256s": ["a" * 64]}],
        "gates": [{"gate_id": "compile", "outcome": "pass"}],
        "diff_coverage": "pass",
        "unprotected_changes": ["README"],
        "ineligible_records": [],
        "dependency_pins_sha256": "b" * 64,
        "verifier": {"repo": "TheAxiomFoundation/axiom-encode", "git_oid": "c" * 40},
    }


def profile():
    return {
        "schema": "axiom/notary-profile/v1",
        "lane": LANE,
        "oracle_policy": "reduced-tier",
        "required_gates": [
            {"gate_id": "compile", "acceptable_outcomes": ["pass"], "tier": "public"}
        ],
    }


def candidate():
    body = report()
    body.update(
        schema="axiom/notary-receipt-candidate/v1",
        report_sha256="d" * 64,
        job1={
            "workflow_ref": LANE + "/.github/workflows/notary.yml@refs/heads/main",
            "workflow_sha_git_oid": "e" * 40,
            "ref": "refs/heads/main",
            "run_id": "1",
            "run_attempt": "1",
            "check_run_id": "2",
            "conclusion": "success",
            "artifact_name": "notary-report",
            "artifact_id": "3",
            "artifact_sha256": "f" * 64,
        },
    )
    return body


def wrapper():
    return {
        "schema": "axiom/notary-receipt/v1",
        "lane": LANE,
        "epoch_sha256": EPOCH,
        "candidate_sha256": "1" * 64,
        "authorization": {
            "environment": "notary-signing",
            "approve_check_run_id": "4",
            "approval_signature_sha256": "2" * 64,
        },
    }


def transition():
    return {
        "schema": "axiom/notary-transition/v1",
        "lane": LANE,
        "epoch_sha256": EPOCH,
        "chain_predecessor_sha256": "a" * 64,
        "chain_predecessor_kind": "receipt",
        "base_tree_manifest_sha256": "b" * 64,
        "subject_tree_manifest_sha256": "c" * 64,
        "subject_commit_git_oid": "d" * 40,
        "reason": "rotate key",
        "delta": [
            {
                "path": ".axiom/notary/keys.json",
                "before_entry_sha256": "e" * 64,
                "before_mode": "100644",
                "after_entry_sha256": "f" * 64,
                "after_mode": "100644",
            }
        ],
    }


def marker():
    return {
        "schema": "axiom/notary-finalization/v1",
        "lane": LANE,
        "epoch_sha256": EPOCH,
        "target_sha256": "f" * 64,
        "target_kind": "receipt",
        "merged_tip_manifest_sha256": "a" * 64,
        "sequence": "3",
    }


def refusal():
    return {
        "schema": "axiom/notary-report-refusal/v1",
        "lane": LANE,
        "epoch_sha256": EPOCH,
        "subject_commit_git_oid": "a" * 40,
        "stage": "resolution",
        "refusal": {
            "code": "chain-unresolvable",
            "path": None,
            "detail": REFUSAL_DETAILS["chain-unresolvable"],
        },
        "established": dict.fromkeys(ESTABLISHED),
    }


FACTORIES = [report, profile, candidate, wrapper, transition, marker, refusal]


def genesis():
    from cryptography.hazmat.primitives import serialization

    from axiom_encode.notary.canonical import sha256_hex

    from .lineage_fixtures import Identities, public_entry

    identities = Identities.create()
    roots = {}
    for name in ("legacy_apply_root", "legacy_eval_root"):
        key = identities.keys[name].public_key()
        roots[name] = {
            "raw_key_id": "sha256:"
            + sha256_hex(
                key.public_bytes(
                    serialization.Encoding.Raw, serialization.PublicFormat.Raw
                )
            ),
            "public_key_spki_der_base64": public_entry(identities.keys[name])[
                "public_key_spki_der_base64"
            ],
        }
    return {
        "schema": "axiom/notary-genesis/v1",
        "lane": LANE,
        "genesis_commit_git_oid": "a" * 40,
        "genesis_tree_manifest_sha256": "b" * 64,
        "bootstrap_policies": {
            name: "c" * 64
            for name in (
                "path_policy_sha256",
                "transition_path_policy_sha256",
                "profile_sha256",
                "key_registry_sha256",
            )
        },
        "activation_spec_template_sha256": "d" * 64,
        "consumer_spec_path": ".axiom/notary/consumer.json",
        "notary_repository": LANE + "-notary",
        **roots,
        "v5_attested": [],
        "baseline_unattested": [["rules/x", "e" * 64]],
    }


FACTORIES.append(genesis)


@pytest.mark.parametrize("factory", FACTORIES)
def test_known_formats_roundtrip(factory):
    body = factory()
    assert parse_artifact(jcs_dumps(body)) == body


@pytest.mark.parametrize("factory", FACTORIES)
def test_every_top_level_member_required_and_unknown_refused(factory):
    body = factory()
    for key in body:
        invalid = deepcopy(body)
        del invalid[key]
        assert parse_artifact(jcs_dumps(invalid)) is None, key
    assert parse_artifact(jcs_dumps(body | {"unexpected": True})) is None


@pytest.mark.parametrize("factory", FACTORIES)
def test_wrong_member_types_are_refusals_not_exceptions(factory):
    body = factory()
    for name, value in body.items():
        # Exercise every container/scalar boundary, including unhashable enum values.
        replacements = [[], {}, None, True, 123]
        for replacement in replacements:
            if type(replacement) is type(value):
                continue
            assert parse_artifact(jcs_dumps(body | {name: replacement})) is None, (
                name,
                replacement,
            )


@pytest.mark.parametrize(
    "field",
    [
        "eligible_records",
        "unused_eligible_records",
        "unprotected_changes",
        "gates",
        "coverage_assignment",
        "ineligible_records",
    ],
)
def test_semantic_arrays_refuse_duplicates(field):
    body = report()
    values = {
        "unused_eligible_records": ["b" * 64],
        "ineligible_records": [{"store_name": "bad", "reasons": ["malformed-record"]}],
    }
    body[field] = values.get(field, body[field]) * 2
    assert parse_artifact(jcs_dumps(body)) is None


def test_record_partition_is_exact_and_disjoint():
    body = report()
    body["unused_eligible_records"] = body["eligible_records"]
    assert parse_artifact(jcs_dumps(body)) is None
    body["unused_eligible_records"] = []
    body["eligible_records"].append("b" * 64)
    assert parse_artifact(jcs_dumps(body)) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "01"),
        ("run_attempt", 1),
        ("conclusion", "failure"),
        ("workflow_sha_git_oid", "f" * 64),
        ("workflow_ref", "wrong@refs/heads/elsewhere"),
        ("artifact_name", ""),
        ("artifact_sha256", "sha256:" + "a" * 64),
    ],
)
def test_candidate_job_contract(field, value):
    body = candidate()
    body["job1"][field] = value
    assert parse_artifact(jcs_dumps(body)) is None


def test_refusal_prefix_and_stage_restrictions():
    body = refusal()
    body["established"]["subject_tree_manifest_sha256"] = "b" * 64
    assert parse_artifact(jcs_dumps(body)) is None
    body = refusal()
    body["established"]["coverage_assignment"] = []
    assert parse_artifact(jcs_dumps(body)) is None


def test_refusal_gates_carries_complete_assignment():
    body, passed = refusal(), report()
    body["stage"] = "gates"
    body["refusal"] = {
        "code": "gate-missing",
        "path": "compile",
        "detail": REFUSAL_DETAILS["gate-missing"],
    }
    body["established"] = {name: passed[name] for name in ESTABLISHED}
    body.update(
        {
            name: passed[name]
            for name in (
                "coverage_assignment",
                "unused_eligible_records",
                "unprotected_changes",
                "gates",
            )
        }
    )
    assert parse_artifact(jcs_dumps(body)) == body


@pytest.mark.parametrize(
    "outcomes,policy,valid",
    [
        (["oracle-unavailable", "pass"], "reduced-tier", True),
        (["oracle-unavailable", "pass"], "fail-closed", False),
        (["pass", "pass"], "reduced-tier", False),
        (["pass"], "unknown", False),
    ],
)
def test_oracle_policy(outcomes, policy, valid):
    body = profile()
    body["oracle_policy"] = policy
    body["required_gates"][0]["acceptable_outcomes"] = outcomes
    assert (parse_artifact(jcs_dumps(body)) is not None) == valid


def test_gate_failure_priority_missing_extra_unacceptable():
    p = profile()
    assert (
        check_gates(p, [{"gate_id": "extra", "outcome": "pass"}]).code == "gate-missing"
    )
    assert (
        check_gates(
            p,
            [
                {"gate_id": "compile", "outcome": "fail"},
                {"gate_id": "extra", "outcome": "pass"},
            ],
        ).code
        == "gate-extra"
    )
    assert (
        check_gates(p, [{"gate_id": "compile", "outcome": "fail"}]).code
        == "gate-unacceptable"
    )
    assert check_gates(p, [{"gate_id": "compile", "outcome": "pass"}]) is None


def test_transition_policy_reserved_prefix_obligations():
    body = {
        "schema": "axiom/notary-transition-path-policy/v1",
        "lane": LANE,
        "rules": [{"action": "include", "prefix": ".axiom/notary"}],
    }
    assert not isinstance(parse_transition_policy(jcs_dumps(body), LANE), Refusal)
    for action, prefix in [
        ("exclude", ".axiom/notary/keys.json"),
        ("include", ".axiom/lineage"),
        ("include", ".axiom/lineage/nested"),
    ]:
        assert isinstance(
            parse_transition_policy(
                jcs_dumps(
                    body
                    | {"rules": body["rules"] + [{"action": action, "prefix": prefix}]}
                ),
                LANE,
            ),
            Refusal,
        )


@pytest.mark.parametrize(
    "reasons",
    [
        ["malformed-record", "wrong-lane"],
        ["invalid-signature", "unrecognized-store-name"],
    ],
)
def test_ineligible_reason_prerequisites(reasons):
    body = report()
    body["ineligible_records"] = [{"store_name": "bad", "reasons": reasons}]
    assert parse_artifact(jcs_dumps(body)) is None


@pytest.mark.parametrize("mutation", ["overlap", "missing", "unlisted"])
def test_gates_refusal_partition(mutation):
    body, passed = refusal(), report()
    body["stage"] = "gates"
    body["refusal"] = {
        "code": "gate-missing",
        "path": "compile",
        "detail": REFUSAL_DETAILS["gate-missing"],
    }
    body["established"] = {name: passed[name] for name in ESTABLISHED}
    body.update(
        {
            name: passed[name]
            for name in (
                "coverage_assignment",
                "unused_eligible_records",
                "unprotected_changes",
                "gates",
            )
        }
    )
    established = body["established"]
    if mutation == "overlap":
        body["unused_eligible_records"] = ["a" * 64]
    elif mutation == "missing":
        established["eligible_records"].append("b" * 64)
    else:
        established["eligible_records"] = []
    assert parse_artifact(jcs_dumps(body)) is None


@pytest.mark.parametrize("name", ["legacy_apply_root", "legacy_eval_root"])
@pytest.mark.parametrize("value", ["not base64!", "YQ", "YR==", "", "YQ==\n"])
def test_legacy_key_base64_is_canonical(name, value):
    body = genesis()
    body[name]["public_key_spki_der_base64"] = value
    assert parse_artifact(jcs_dumps(body)) is None
