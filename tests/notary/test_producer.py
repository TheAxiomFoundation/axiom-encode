import base64

import pytest

from axiom_encode.notary.canonical import jcs_dumps, sha256_hex, strict_parse
from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.lineage import (
    POLICY_PATH,
    STORE_PREFIX,
    StoreFile,
    classify_lineage,
    parse_path_policy,
)
from axiom_encode.notary.producer import correction_export, generation_export
from axiom_encode.notary.producers import Enrollment
from axiom_encode.notary.verification import verify_snapshots

from .lineage_fixtures import LANE
from .test_producers import submission as _submission_fixture
from .test_verification import snapshot

submission = _submission_fixture


def emission(submission, **overrides):
    epoch, policy, _, args = submission
    data = dict(
        enrollment=Enrollment(jcs_dumps(policy["runtimes"][0])),
        base=args["base"],
        lane=LANE,
        epoch=epoch.anchor.epoch_sha256,
        run_id="fixture-run",
        outputs={"rules/example.yaml": b"generated\n"},
        model="fixture",
        prompts=["a" * 64],
        source_id="fixture-citation",
        source_bytes=b"captured source",
        draw_set_id="fixture-draw",
        sampling={"temperature": "0.5", "seed": None},
        independence={
            "sibling_draws_visible": "no",
            "incumbent_encoding_visible": "yes",
        },
        references={"oracles": [], "reference_data": []},
    )
    return generation_export(epoch.identities.keys["producer"], **(data | overrides))


def decode(export):
    data = strict_parse(export)
    return data, {
        row["path"]: base64.b64decode(row["base64"])
        for row in data["files"]
        if row["base64"] is not None
    }


def test_host_observed_output_produces_valid_covered_report(submission):
    epoch, _, _, args = submission
    packet, files = decode(emission(submission))
    record = strict_parse(files[STORE_PREFIX + packet["record_sha256"] + ".json"])
    assert record["source_capture"]["content_sha256"] == sha256_hex(b"captured source")
    subject = snapshot("c" * 40, args["base"].blobs | files)
    report = verify_snapshots(
        args["base"],
        subject,
        epoch.state().predecessor(),
        epoch.inventory,
        [{"gate_id": "compile", "outcome": "pass"}],
    )
    assert strict_parse(report)["schema"] == "axiom/notary-report-pass/v1"


@pytest.mark.parametrize(
    "path",
    [
        "../escape.yaml",
        "/absolute.yaml",
        ".axiom/notary/keys.json",
        ".github/workflows/a.yml",
        "rules/file.py",
        "unprotected/a.yaml",
    ],
)
def test_emission_cannot_sign_trust_or_unprotected_paths(submission, path):
    with pytest.raises(IdentityRefusal, match="output_path"):
        emission(submission, outputs={path: b"anything"})


def test_output_tampering_after_emission_fails_coverage(submission):
    epoch, _, _, args = submission
    _, files = decode(emission(submission))
    files["rules/example.yaml"] = b"hand-edit"
    report = verify_snapshots(
        args["base"],
        snapshot("c" * 40, args["base"].blobs | files),
        epoch.state().predecessor(),
        epoch.inventory,
        [{"gate_id": "compile", "outcome": "pass"}],
    )
    assert strict_parse(report)["schema"] == "axiom/notary-report-refusal/v1"


def test_correction_is_actor_signed_and_ineligible_without_hardware_review(submission):
    epoch, policy, _, args = submission
    packet, files = decode(
        correction_export(
            epoch.identities.keys["actor"],
            enrollment=Enrollment(jcs_dumps(policy["runtimes"][0])),
            base=args["base"],
            lane=LANE,
            epoch=epoch.anchor.epoch_sha256,
            run_id="correction",
            outputs={"rules/example.yaml": b"corrected\n"},
            github_user_id="123",
            reason="Explicit policy correction",
            predecessor=None,
        )
    )
    name = packet["record_sha256"] + ".json"
    record = files[STORE_PREFIX + name]
    assert strict_parse(record)["actor"] == "github:123"
    store = {
        path.removeprefix(STORE_PREFIX): StoreFile(raw)
        for path, raw in files.items()
        if path.startswith(STORE_PREFIX)
    }
    context = dict(
        lane=LANE,
        epoch_sha256=epoch.anchor.epoch_sha256,
        registry=epoch.state().registry,
        path_policy=parse_path_policy(args["base"].blobs[POLICY_PATH], lane=LANE),
    )
    classified = classify_lineage({}, store, **context)
    assert not classified.eligible
    store[name + ".review.sig"] = StoreFile(epoch.identities.sidecar(record, "review"))
    assert len(classify_lineage({}, store, **context).eligible) == 1
