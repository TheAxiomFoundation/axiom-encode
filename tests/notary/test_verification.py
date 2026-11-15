from copy import deepcopy
from dataclasses import replace

import pytest

from axiom_encode.notary.canonical import jcs_dumps, sha256_hex, strict_parse
from axiom_encode.notary.lineage import POLICY_PATH, STORE_PREFIX
from axiom_encode.notary.manifest import manifest_sha256
from axiom_encode.notary.protocol import (
    ESTABLISHED,
    PROFILE_PATH,
    TRANSITION_POLICY_PATH,
    parse_artifact,
)
from axiom_encode.notary.refusal import Refusal
from axiom_encode.notary.registry import REGISTRY_PATH
from axiom_encode.notary.verification import (
    Predecessor,
    Snapshot,
    reconcile,
    verify_snapshots,
)

from .lineage_fixtures import EPOCH, LANE, Identities, policy_body
from .test_protocol import profile


def snapshot(commit, blobs):
    return Snapshot(
        commit,
        sorted((path, "100644", sha256_hex(raw)) for path, raw in blobs.items()),
        dict(blobs),
    )


@pytest.fixture
def packet():
    identities = Identities.create()
    waiver = b"validate_failures: {}\n"
    base = snapshot(
        "a" * 40,
        {
            POLICY_PATH: jcs_dumps(policy_body()),
            PROFILE_PATH: jcs_dumps(profile()),
            REGISTRY_PATH: jcs_dumps(identities.body),
            TRANSITION_POLICY_PATH: jcs_dumps(
                {
                    "schema": "axiom/notary-transition-path-policy/v1",
                    "lane": LANE,
                    "rules": [{"action": "include", "prefix": ".axiom/notary"}],
                }
            ),
            ".axiom/toolchain.toml": (
                f'[toolchain]\naxiom_corpus_release = "fixture"\naxiom_corpus_release_content_sha256 = "{"c" * 64}"\nvalidation_waiver_set_sha256 = "{sha256_hex(waiver)}"\n'
            ).encode(),
            "known-validation-gaps.yaml": waiver,
        },
    )
    predecessor = Predecessor(
        LANE,
        EPOCH,
        "f" * 64,
        "transition",
        base.commit,
        manifest_sha256(base.manifest),
        identities.pins["notary_spki_sha256"],
        identities.pins["legacy_apply_root"],
        identities.pins["legacy_eval_root"],
        ".axiom/notary/consumer.json",
        True,
    )
    subject = snapshot(
        "b" * 40,
        base.blobs
        | {"rules/example.yaml": b"generated\n"}
        | {STORE_PREFIX + name: file.raw for name, file in identities.store().items()},
    )
    inventory = jcs_dumps(
        {
            "schema": "axiom/notary-dependency-inventory/v1",
            "lane": LANE,
            "actions": [],
            "containers": [],
            "python_lock_sha256": "d" * 64,
            "verifier": {
                "repo": "TheAxiomFoundation/axiom-encode",
                "git_oid": "e" * 40,
            },
        }
    )
    return (
        base,
        subject,
        predecessor,
        inventory,
        [{"gate_id": "compile", "outcome": "pass"}],
    )


def test_exact_generated_bytes_produce_well_formed_report(packet):
    raw = verify_snapshots(*packet)
    body = parse_artifact(raw, "report-pass")
    assert body is not None
    assert body["coverage_assignment"][0]["path"] == "rules/example.yaml"
    assert body["diff_coverage"] == "pass"
    assert reconcile(raw, raw) == body


def test_tampered_generated_bytes_refuse(packet):
    base, subject, *rest = packet
    tampered = snapshot(
        subject.commit, subject.blobs | {"rules/example.yaml": b"hand edit\n"}
    )
    body = parse_artifact(verify_snapshots(base, tampered, *rest), "report-refusal")
    assert body["refusal"]["code"] == "inconsistent-chain"
    assert body["stage"] == "assignment"
    assert set(body["established"]) == set(ESTABLISHED)


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/ci.yml",
        ".github/actions/check/action.yml",
        ".axiom/workflow-toolchain.toml",
        "rules/.gitattributes",
        "CODEOWNERS",
        "docs/CODEOWNERS",
        "repository-structure.yaml",
        ".axiom/notary/consumer.json",
    ],
)
def test_trust_changes_cannot_ride_ordinary_candidate(packet, path):
    base, subject, *rest = packet
    altered = snapshot(subject.commit, subject.blobs | {path: b"altered"})
    body = parse_artifact(verify_snapshots(base, altered, *rest), "report-refusal")
    assert body["refusal"]["code"] == "trust-surface-change"
    assert body["refusal"]["path"] == path


@pytest.mark.parametrize(
    "missing",
    [
        PROFILE_PATH,
        POLICY_PATH,
        TRANSITION_POLICY_PATH,
        REGISTRY_PATH,
        ".axiom/toolchain.toml",
        "known-validation-gaps.yaml",
    ],
)
def test_missing_base_authority_has_truthful_prefix(packet, missing):
    base, subject, predecessor, *rest = packet
    base = snapshot(
        base.commit, {path: raw for path, raw in base.blobs.items() if path != missing}
    )
    predecessor = replace(predecessor, manifest_sha256=manifest_sha256(base.manifest))
    body = parse_artifact(
        verify_snapshots(base, subject, predecessor, *rest), "report-refusal"
    )
    assert body is not None
    assert body["refusal"]["code"] == "policy-invalid"
    if missing == PROFILE_PATH:
        assert body["established"]["profile_sha256"] is None
        assert body["established"]["path_policy_sha256"] is None


def test_state_identical_refuses(packet):
    base, _, predecessor, *rest = packet
    body = parse_artifact(
        verify_snapshots(base, base, predecessor, *rest), "report-refusal"
    )
    assert body["refusal"]["code"] == "state-identical"


def test_stale_predecessor_and_inactive_epoch(packet):
    base, subject, predecessor, *rest = packet
    for pred in [
        replace(predecessor, manifest_sha256="0" * 64),
        replace(predecessor, activated=False),
    ]:
        body = parse_artifact(
            verify_snapshots(base, subject, pred, *rest), "report-refusal"
        )
        assert body["refusal"]["code"] == "predecessor-stale"


def test_bad_new_lineage_is_enumerated_without_poisoning(packet):
    base, subject, *rest = packet
    subject = snapshot(subject.commit, subject.blobs | {STORE_PREFIX + "alias": b"bad"})
    body = parse_artifact(verify_snapshots(base, subject, *rest), "report-pass")
    assert body["ineligible_records"] == [
        {"store_name": "alias", "reasons": ["unrecognized-store-name"]}
    ]


def test_gate_refusal_retains_successful_coverage(packet):
    body = parse_artifact(verify_snapshots(*packet[:-1], []), "report-refusal")
    assert body["stage"] == "gates"
    assert body["refusal"]["code"] == "gate-missing"
    assert body["coverage_assignment"]
    assert set(body["established"]) == set(ESTABLISHED)


@pytest.mark.parametrize(
    "field",
    [
        "eligible_records",
        "unused_eligible_records",
        "coverage_assignment",
        "unprotected_changes",
        "ineligible_records",
        "corpus_release",
        "waiver_set_sha256",
        "dependency_pins_sha256",
        "verifier",
        "subject_tree_manifest_sha256",
    ],
)
def test_trusted_reconciliation_rejects_forged_fields(packet, field):
    recomputed = verify_snapshots(*packet)
    forged = deepcopy(strict_parse(recomputed))
    forged[field] = None
    assert isinstance(reconcile(jcs_dumps(forged), recomputed), Refusal)


def test_refusal_never_signable(packet):
    refused = verify_snapshots(*packet[:-1], [])
    assert isinstance(reconcile(refused, refused), Refusal)


@pytest.mark.parametrize(
    "gates", [None, {}, [None], [{"gate_id": "compile", "outcome": 0}]]
)
def test_malformed_gate_declarations_emit_valid_structural_refusal(packet, gates):
    body = parse_artifact(verify_snapshots(*packet[:-1], gates), "report-refusal")
    assert body is not None and body["stage"] == "structural"
    assert body["established"]["subject_tree_manifest_sha256"]
    assert body["established"]["profile_sha256"] is None


@pytest.mark.parametrize(
    "extra", [b'extra="unrecognized"\n', b'axiom_corpus_release="../bad"\n']
)
def test_toolchain_uses_existing_closed_contract(packet, extra):
    base, subject, predecessor, *rest = packet
    raw = base.blobs[".axiom/toolchain.toml"] + extra
    base = snapshot(base.commit, base.blobs | {".axiom/toolchain.toml": raw})
    predecessor = replace(predecessor, manifest_sha256=manifest_sha256(base.manifest))
    body = parse_artifact(
        verify_snapshots(base, subject, predecessor, *rest), "report-refusal"
    )
    assert body["refusal"]["code"] == "policy-invalid"
    assert body["refusal"]["path"] == ".axiom/toolchain.toml"


def test_trust_category_priority_precedes_lexical_path_order(packet):
    base, subject, *rest = packet
    subject = snapshot(
        subject.commit,
        subject.blobs
        | {".axiom/workflow-toolchain.toml": b"bad", ".github/workflows/z.yml": b"bad"},
    )
    body = parse_artifact(verify_snapshots(base, subject, *rest), "report-refusal")
    assert body["refusal"]["path"] == ".github/workflows/z.yml"


@pytest.mark.parametrize(
    "resolves,code",
    [([False], "chain-unresolvable"), ([True, False], "subject-unresolvable")],
)
def test_resolution_failure_precedes_fsck(
    packet, tmp_path, monkeypatch, resolves, code
):
    from axiom_encode.notary import verification

    decisions = iter(resolves)
    monkeypatch.setattr(verification, "_commit_resolves", lambda *_: next(decisions))

    def forbidden(*_):
        pytest.fail("fsck must not run before commit resolution")

    monkeypatch.setattr(verification, "fsck_clean", forbidden)
    _, subject, predecessor, inventory, gates = packet
    raw = verification.verify_repository(
        tmp_path, subject.commit, predecessor, inventory, gates
    )
    body = parse_artifact(raw, "report-refusal")
    assert body["refusal"]["code"] == code
    values = body["established"]
    assert sum(v is not None for v in values.values()) == (
        0 if len(resolves) == 1 else 3
    )
