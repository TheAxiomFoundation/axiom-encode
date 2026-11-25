"""The review attestation: a signature over a candidate review record.

Every key here is a throwaway key made inside the test. No test reads, makes
or prints a key of any machine, and the shipped root enrols nobody.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from axiom_encode import candidate_review_record as crr
from axiom_encode import prepare_signed_backfill as helper
from axiom_encode import review_attestation as ra
from axiom_encode.candidate_review_record import ReviewRecordError
from tests.test_candidate_review_record import FAIL, make_record
from tests.test_generalist_review_promotion import _Promotion
from tests.test_review_record_promotion import (
    Pull,
    dispatch,
    load_script,
    unit_record,
    write_document,
)
from tests.test_review_record_workflow import (
    BASE_COMMAND,
    needs_shell,
    promotion_command,
    wrapper,
)


def enrol(*private_keys, owner="test machine") -> str:
    """The text of a root that enrols the public halves of throwaway keys."""

    keys = []
    for private_key in private_keys:
        public_key = private_key.public_key()
        keys.append(
            {
                "key_id": ra.key_id(public_key),
                "ed25519_public_key": base64.b64encode(
                    public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
                ).decode(),
                "owner": owner,
                "enrolled": "2026-11-25",
            }
        )
    return json.dumps({"schema": ra.ROOTS_SCHEMA, "keys": keys})


@pytest.fixture
def key():
    return Ed25519PrivateKey.generate()


@pytest.fixture
def roots(key):
    return ra.load_roots(enrol(key))


# --- The shipped root ---------------------------------------------------------------


def test_the_shipped_root_enrols_nobody_and_so_admits_nothing(tmp_path, key):
    assert ra.shipped_roots() == {}
    record = make_record(tmp_path)
    with pytest.raises(ReviewRecordError, match="is not enrolled"):
        ra.require_attested_records(
            [record], [ra.sign_record(record, key)], ra.shipped_roots()
        )


# --- Sign and verify -----------------------------------------------------------------


def test_a_signature_by_an_enrolled_key_attests_the_record(tmp_path, key, roots):
    record = make_record(tmp_path)
    attestation = ra.sign_record(record, key)
    assert set(attestation) == {"schema", "key_id", "record_sha256", "signature"}
    assert attestation["record_sha256"] == crr.record_sha256(record)
    assert ra.attestation_issues(record, attestation, roots) == []
    assert ra.require_attested_records([record], [attestation], roots) == [attestation]


def test_a_key_that_is_not_enrolled_is_refused(tmp_path, roots):
    record = make_record(tmp_path)
    other = Ed25519PrivateKey.generate()
    (issue,) = ra.attestation_issues(record, ra.sign_record(record, other), roots)
    assert "is not enrolled" in issue


def test_a_removed_key_no_longer_attests(tmp_path, key):
    """Revocation: remove the key id from the root."""

    second = Ed25519PrivateKey.generate()
    record = make_record(tmp_path)
    attestation = ra.sign_record(record, key)
    assert (
        ra.attestation_issues(record, attestation, ra.load_roots(enrol(key, second)))
        == []
    )
    assert (
        ra.attestation_issues(record, attestation, ra.load_roots(enrol(second))) != []
    )


def test_a_record_changed_after_the_signature_is_refused(tmp_path, key, roots):
    record = make_record(tmp_path, output=FAIL)
    attestation = ra.sign_record(record, key)
    changed = copy.deepcopy(record)
    changed["run"]["attempt"] = 2
    assert ra.attestation_issues(changed, attestation, roots) == [
        "the review attestation is for another record"
    ]
    forged = {**attestation, "record_sha256": crr.record_sha256(changed)}
    assert ra.attestation_issues(changed, forged, roots) == [
        "the review attestation signature does not verify"
    ]


def test_a_signature_for_another_purpose_is_not_an_attestation(tmp_path, key, roots):
    record = make_record(tmp_path)
    digest = crr.record_sha256(record)
    bare = {
        "schema": ra.ATTESTATION_SCHEMA,
        "key_id": ra.key_id(key.public_key()),
        "record_sha256": digest,
        # The same key signs the digest without the domain line.
        "signature": base64.b64encode(key.sign(digest.encode())).decode(),
    }
    assert ra.attestation_issues(record, bare, roots) == [
        "the review attestation signature does not verify"
    ]


@pytest.mark.parametrize(
    "change",
    [
        lambda a: a.update(extra=1),
        lambda a: a.pop("signature"),
        lambda a: a.update(schema="axiom-encode/review-attestation/v2"),
        lambda a: a.update(key_id="sha256:" + "A" * 64),
        lambda a: a.update(signature="not base64 !"),
        lambda a: a.update(signature=None),
    ],
)
def test_a_malformed_attestation_is_refused(tmp_path, key, roots, change):
    record = make_record(tmp_path)
    attestation = ra.sign_record(record, key)
    change(attestation)
    assert ra.attestation_issues(record, attestation, roots) != []


def test_every_record_needs_exactly_one_attestation(tmp_path, key, roots):
    first = make_record(tmp_path)
    second = make_record(tmp_path, attempt=2)
    signed = ra.sign_record(first, key)
    with pytest.raises(ReviewRecordError, match="0 review attestations name"):
        ra.require_attested_records([first, second], [signed], roots)
    with pytest.raises(ReviewRecordError, match="2 review attestations name"):
        ra.require_attested_records([first], [signed, dict(signed)], roots)


def test_a_record_that_is_not_well_formed_is_not_signed(tmp_path, key):
    record = make_record(tmp_path)
    record["status"] = "failed"
    with pytest.raises(ReviewRecordError, match="cannot be attested"):
        ra.sign_record(record, key)


# --- The root file --------------------------------------------------------------------


def test_the_root_lists_several_machines(key):
    second = Ed25519PrivateKey.generate()
    roots = ra.load_roots(enrol(key, second))
    assert set(roots) == {ra.key_id(key.public_key()), ra.key_id(second.public_key())}


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema="other"),
        lambda d: d.update(extra=1),
        lambda d: d["keys"].append(dict(d["keys"][0])),
        lambda d: d["keys"][0].update(key_id="sha256:" + "0" * 64),
        lambda d: d["keys"][0].update(ed25519_public_key="AAAA"),
        lambda d: d["keys"][0].update(ed25519_public_key="!"),
        lambda d: d["keys"][0].update(owner=""),
        lambda d: d["keys"][0].update(enrolled="yesterday"),
        lambda d: d["keys"][0].pop("owner"),
        lambda d: d["keys"][0].update(private_key="x"),
        lambda d: d.update(keys={}),
    ],
)
def test_a_malformed_root_is_refused_whole(key, change):
    document = json.loads(enrol(key))
    change(document)
    with pytest.raises(ReviewRecordError):
        ra.load_roots(json.dumps(document))


# --- The pull request body --------------------------------------------------------------


def test_an_attestation_block_round_trips_beside_its_record(tmp_path, key):
    record = make_record(tmp_path)
    attestation = ra.sign_record(record, key)
    body = (
        "Candidate.\n\n"
        + crr.render_block(record)
        + "\n"
        + ra.render_attestation_block(attestation)
    )
    assert crr.extract_records(body) == [record]
    assert ra.extract_attestations(body) == [attestation]
    assert ra.extract_attestations(crr.render_block(record)) == []


@pytest.mark.parametrize(
    "damage",
    [
        lambda b: b.replace(ra.END_MARKER, ""),
        lambda b: b.replace("```json\n", "```json\n\n"),
        lambda b: b.replace(ra.BEGIN_MARKER + "\n", ""),
        lambda b: b.replace(":v1:begin -->", ":v2:begin -->"),
        lambda b: b.replace('{"key_id"', '{"key_id'),
    ],
)
def test_a_damaged_attestation_block_refuses_the_body(tmp_path, key, damage):
    block = ra.render_attestation_block(ra.sign_record(make_record(tmp_path), key))
    damaged = damage(block)
    assert damaged != block
    with pytest.raises(ReviewRecordError):
        ra.extract_attestations("intro\n" + damaged + "outro\n")


# --- The wrapper key and the supervised command ------------------------------------------


def test_the_attestation_key_needs_the_record_key_and_exactly_true():
    payload = helper.split_atomic_source_input(
        wrapper(require_review_record=True, require_review_attestation=True)
    )
    assert payload["promotion_review_record_required"] is True
    assert payload["promotion_review_attestation_required"] is True
    plain = helper.split_atomic_source_input(wrapper(require_review_record=True))
    assert "promotion_review_attestation_required" not in plain
    for extra in (
        {"require_review_attestation": True},
        {"require_review_record": True, "require_review_attestation": False},
        {"require_review_record": True, "require_review_attestation": "true"},
    ):
        with pytest.raises(ValueError, match="require_review_attestation"):
            helper.split_atomic_source_input(wrapper(**extra))


@needs_shell
def test_only_the_attestation_key_adds_the_attestation_flag():
    record_only = helper.split_atomic_source_input(wrapper(require_review_record=True))
    assert "--require-review-attestation" not in promotion_command(record_only)
    assert (
        promotion_command(helper.split_atomic_source_input(wrapper())) == BASE_COMMAND
    )
    both = helper.split_atomic_source_input(
        wrapper(require_review_record=True, require_review_attestation=True)
    )
    assert promotion_command(both) == [
        *BASE_COMMAND,
        "--require-review-record",
        "/runner/temp/promotion-review-records/candidate-review-records.json",
        "--require-review-attestation",
    ]


# --- The promotion -----------------------------------------------------------------------


def attested_promotion(tmp_path, monkeypatch, key, *, sign_with=None, enrolled=True):
    promotion = _Promotion(tmp_path / "p")
    record = unit_record(tmp_path, promotion)
    document = json.loads(write_document(tmp_path, [record], promotion.ref).read_text())
    if sign_with is not None:
        document["attestations"] = [ra.sign_record(record, sign_with)]
    target = tmp_path / "attested.json"
    target.write_text(json.dumps(document))
    promotion.args.require_review_record = target
    promotion.args.require_review_attestation = True
    monkeypatch.setattr(
        ra, "shipped_roots", lambda: ra.load_roots(enrol(key)) if enrolled else {}
    )
    return promotion


def test_the_promotion_signs_a_record_attested_by_an_enrolled_key(
    tmp_path, monkeypatch, key, capsys
):
    promotion = attested_promotion(tmp_path, monkeypatch, key, sign_with=key)
    promotion.run()
    assert promotion.order == ["validate", "sign"]
    assert ra.key_id(key.public_key()) in capsys.readouterr().out


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"sign_with": None}, "0 review attestations name"),
        ({"sign_with": "other"}, "is not enrolled"),
        ({"sign_with": "own", "enrolled": False}, "is not enrolled"),
    ],
)
def test_the_promotion_refuses_without_an_enrolled_attestation(
    tmp_path, monkeypatch, key, arguments, message
):
    chosen = {"other": Ed25519PrivateKey.generate(), "own": key, None: None}
    arguments = {**arguments, "sign_with": chosen[arguments["sign_with"]]}
    promotion = attested_promotion(tmp_path, monkeypatch, key, **arguments)
    with pytest.raises(RuntimeError, match=message):
        promotion.run()
    assert promotion.order == [] and promotion.signed == []
    assert not promotion.manifest.exists()


def test_the_record_switch_alone_asks_for_no_attestation(tmp_path, monkeypatch, key):
    promotion = attested_promotion(tmp_path, monkeypatch, key, sign_with=None)
    promotion.args.require_review_attestation = False
    promotion.run()
    assert promotion.order == ["validate", "sign"]


def test_the_attestation_flag_needs_the_record_flag(tmp_path):
    promotion = _Promotion(tmp_path / "p")
    promotion.args.require_review_attestation = True
    with pytest.raises(ValueError, match="requires --require-review-record"):
        promotion.run()
    assert promotion.order == []


# --- The workflow's reader ----------------------------------------------------------------


def test_the_reader_keeps_and_checks_the_attestations(tmp_path, monkeypatch, key):
    promotion = _Promotion(tmp_path / "p")
    record = unit_record(tmp_path, promotion)
    attestation = ra.sign_record(record, key)
    body = crr.render_block(record) + "\n" + ra.render_attestation_block(attestation)
    request = dispatch(promotion.ref)
    outer = json.loads(request["source_bundle_json"])
    outer["require_review_attestation"] = True
    request["source_bundle_json"] = json.dumps(outer)
    script = load_script()

    def call(text, out):
        return script.read_records(
            dispatch=request,
            repo=promotion.repo,
            out_dir=tmp_path / out,
            run_id="123",
            run_attempt=1,
            fetch=Pull(text, promotion.ref),
        )

    monkeypatch.setattr(ra, "shipped_roots", lambda: ra.load_roots(enrol(key)))
    required, summary = call(body, "good")
    assert required and ra.key_id(key.public_key()) in summary
    document = crr.load_read_document(
        Path(tmp_path / "good" / "candidate-review-records.json").read_text()
    )
    assert document["attestations"] == [attestation]
    with pytest.raises(ValueError, match="0 review attestations name"):
        call(crr.render_block(record), "missing")
    monkeypatch.setattr(ra, "shipped_roots", lambda: {})
    with pytest.raises(ValueError, match="is not enrolled"):
        call(body, "empty-root")
