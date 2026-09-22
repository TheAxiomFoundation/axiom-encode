from copy import deepcopy

import pytest

from axiom_encode.notary.canonical import jcs_dumps
from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.lineage import STORE_PREFIX
from axiom_encode.notary.producers import (
    ENROLLMENT_PATH,
    Enrollment,
    parse_enrollments,
    require_enrolled_submission,
)
from axiom_encode.notary.verification import verify_snapshots

from .chain_fixtures import Epoch
from .lineage_fixtures import LANE, generation
from .test_verification import snapshot


class API:
    def __init__(self, subject):
        self.pr = {
            "user": {"login": "contributor", "id": 123},
            "state": "open",
            "draft": False,
            "head": {"sha": subject, "repo": {"full_name": LANE}},
            "base": {"repo": {"full_name": LANE}},
        }
        self.permission = {"permission": "write", "user": self.pr["user"]}

    def get(self, path):
        return self.pr if "/pulls/" in path else self.permission


def enrollment_base(identities):
    entry = {
        "producer_spki_sha256": identities.body["producer"][0]["spki_sha256"],
        "actor_spki_sha256": identities.body["actor"][0]["spki_sha256"],
        "github_user_ids": ["123"],
        "encoder": {
            "repository": "TheAxiomFoundation/axiom-encode",
            "git_oid": "e" * 40,
            "version": "0.2.2015",
            "package_tree_sha256": "f" * 64,
        },
        "codex_cli": {"version": "fixture-cli", "sha256": "c" * 64},
        "custody_evidence_sha256": "d" * 64,
    }
    policy = {
        "schema": "axiom/notary-producer-enrollments/v1",
        "lane": LANE,
        "runtimes": [entry],
    }
    return {
        ENROLLMENT_PATH: jcs_dumps(policy),
        ".axiom/workflow-toolchain.toml": (
            '[workflow_toolchain]\naxiom_encode_version="0.2.2015"\naxiom_encode_ref="'
            + "e" * 40
            + '"\n'
        ).encode(),
    }


@pytest.fixture
def submission():
    # Enrollment is committed BEFORE genesis, so all manifests and signed
    # historical chain anchors are real; no forged in-memory predecessor.
    from axiom_encode.notary.canonical import strict_parse

    epoch = Epoch.create(prepare_base=enrollment_base)
    base = epoch.active
    policy = strict_parse(base.blobs[ENROLLMENT_PATH])
    enrollment = Enrollment(jcs_dumps(policy["runtimes"][0]))
    registry = epoch.state().registry
    record = generation() | {
        "epoch_sha256": epoch.anchor.epoch_sha256,
        "runtime_identity": enrollment.runtime_identity,
        "cli_version": "fixture-cli",
    }
    subject = snapshot(
        "c" * 40,
        base.blobs
        | {"rules/example.yaml": b"generated\n"}
        | {
            STORE_PREFIX + name: file.raw
            for name, file in epoch.identities.store(record).items()
        },
    )
    predecessor = epoch.state().predecessor()
    report = verify_snapshots(
        base,
        subject,
        predecessor,
        epoch.inventory,
        [{"gate_id": "compile", "outcome": "pass"}],
    )
    return (
        epoch,
        policy,
        API(subject.commit),
        dict(
            pr_number="1",
            base=base,
            subject=subject,
            registry=registry,
            report_raw=report,
        ),
    )


def test_approved_writer_exact_runtime_and_encoder_pass(submission):
    _, _, api, args = submission
    assert require_enrolled_submission(api, **args)["user"]["id"] == 123


@pytest.mark.parametrize(
    "mutation", ["outsider", "read-only", "fork", "stale-head", "draft"]
)
def test_live_contributor_authorization_is_required(submission, mutation):
    _, _, api, args = submission
    if mutation == "outsider":
        api.pr["user"]["id"] = 999
    elif mutation == "read-only":
        api.permission["permission"] = "read"
    elif mutation == "fork":
        api.pr["head"]["repo"]["full_name"] = "contributor/rulespec-nz"
    elif mutation == "stale-head":
        api.pr["head"]["sha"] = "a" * 40
    else:
        api.pr["draft"] = True
    with pytest.raises(IdentityRefusal):
        require_enrolled_submission(api, **args)


@pytest.mark.parametrize("field", ["version", "git_oid"])
def test_current_base_pin_must_match_enrollment(submission, field):
    _, policy, api, args = submission
    policy["runtimes"][0]["encoder"][field] = (
        "0.2.1" if field == "version" else "1" * 40
    )
    args["base"] = snapshot(
        args["base"].commit, args["base"].blobs | {ENROLLMENT_PATH: jcs_dumps(policy)}
    )
    with pytest.raises(IdentityRefusal, match="encoder_pin_mismatch"):
        require_enrolled_submission(api, **args)


def test_enrollment_cannot_be_supplied_by_candidate(submission):
    _, _, api, args = submission
    args["base"] = snapshot(
        args["base"].commit,
        {p: b for p, b in args["base"].blobs.items() if p != ENROLLMENT_PATH},
    )
    with pytest.raises(IdentityRefusal, match="enrollment_schema"):
        require_enrolled_submission(api, **args)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown-field",
        "duplicate",
        "wrong-actor",
        "empty-operators",
        "wrong-lane",
        "missing-producer",
        "bool-user",
    ],
)
def test_closed_enrollment_contract(submission, mutation):
    _, original, _, args = submission
    policy = deepcopy(original)
    entry = policy["runtimes"][0]
    if mutation == "unknown-field":
        entry["approved"] = True
    elif mutation == "duplicate":
        policy["runtimes"].append(deepcopy(entry))
    elif mutation == "wrong-actor":
        entry["actor_spki_sha256"] = entry["producer_spki_sha256"]
    elif mutation == "empty-operators":
        entry["github_user_ids"] = []
    elif mutation == "wrong-lane":
        policy["lane"] = "TheAxiomFoundation/rulespec-us"
    elif mutation == "missing-producer":
        policy["runtimes"] = []
    else:
        entry["github_user_ids"] = [True]
    with pytest.raises(IdentityRefusal):
        parse_enrollments(jcs_dumps(policy), args["registry"])


def test_same_key_cannot_claim_a_different_runtime(submission):
    _, policy, api, args = submission
    policy["runtimes"][0]["codex_cli"]["version"] = "other-version"
    args["base"] = snapshot(
        args["base"].commit, args["base"].blobs | {ENROLLMENT_PATH: jcs_dumps(policy)}
    )
    with pytest.raises(IdentityRefusal, match="runtime_mismatch"):
        require_enrolled_submission(api, **args)
