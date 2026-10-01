"""V33 closed bodies, historical immutability, and complete reason classification."""

from copy import deepcopy

import pytest

from axiom_encode.notary.canonical import jcs_dumps, sha256_hex
from axiom_encode.notary.lineage import (
    LineageClassification,
    StoreFile,
    classify_lineage,
    parse_path_policy,
    parse_record,
)
from axiom_encode.notary.refusal import Refusal
from tests.notary.lineage_fixtures import (
    EPOCH,
    LANE,
    Identities,
    correction,
    generation,
    policy,
    policy_body,
)


@pytest.fixture
def identities():
    return Identities.create()


def classify(identities, subject, base=None, **kwargs):
    context = dict(
        lane=LANE,
        epoch_sha256=EPOCH,
        registry=identities.registry(),
        path_policy=policy(),
    )
    context.update(kwargs)
    return classify_lineage(base or {}, subject, **context)


@pytest.mark.parametrize("factory", [generation, correction])
def test_signed_generation_and_correction_eligible(identities, factory):
    result = classify(identities, identities.store(factory()))
    assert isinstance(result, LineageClassification)
    assert len(result.eligible) == 1 and result.ineligible == ()
    # A mutable caller view cannot alter authenticated raw bytes.
    result.eligible[0].body["lane"] = "forged/repo"
    assert result.eligible[0].body["lane"] == LANE


@pytest.mark.parametrize("missing_role", ["actor", "review"])
def test_correction_needs_both_signatures(identities, missing_role):
    store = identities.store(correction())
    del store[next(name for name in store if name.endswith(f".{missing_role}.sig"))]
    result = classify(identities, store)
    assert result.eligible == ()
    assert result.ineligible[0].reasons == ("invalid-signature",)


@pytest.mark.parametrize("field", list(generation()))
def test_every_generation_field_is_required(field):
    body = generation()
    del body[field]
    assert parse_record(jcs_dumps(body)) is None


@pytest.mark.parametrize("field", list(correction()))
def test_every_correction_field_is_required(field):
    body = correction()
    del body[field]
    assert parse_record(jcs_dumps(body)) is None


@pytest.mark.parametrize(
    "temperature", [None, "0", "1", "-1", "0.125", "-0.125", "123456789123456789.1"]
)
def test_canonical_decimal_temperature_is_not_a_json_number(temperature):
    body = generation()
    body["sampling"]["temperature"] = temperature
    assert parse_record(jcs_dumps(body)) is not None


@pytest.mark.parametrize(
    "temperature",
    [
        0,
        0.5,
        True,
        [],
        {},
        "-0",
        "-0.0",
        "1.0",
        "01",
        "+1",
        ".5",
        "1.",
        "1e-2",
        "0.50",
        " 1",
    ],
)
def test_noncanonical_decimal_refuses(temperature):
    body = generation()
    body["sampling"]["temperature"] = temperature
    assert parse_record(jcs_dumps(body)) is None


@pytest.mark.parametrize(
    "seed,valid",
    [
        (None, True),
        ("0", True),
        ("-1", True),
        ("9007199254740993", True),
        ("01", False),
        ("-0", False),
        ("1.5", False),
        (0, False),
    ],
)
def test_seed_is_canonical_integer_or_null(seed, valid):
    body = generation()
    body["sampling"]["seed"] = seed
    assert (parse_record(jcs_dumps(body)) is not None) == valid


@pytest.mark.parametrize(
    "timestamp,valid",
    [
        ("2026-09-18T12:00:00Z", True),
        ("2026-09-18t12:00:00z", True),
        ("2026-09-18T12:00:00.123+00:00", True),
        ("2016-12-31T23:59:60Z", True),
        ("2026-02-30T12:00:00Z", False),
        ("2026-09-18T12:00:00-04:00", False),
        ("2026-09-18T12:00:00-00:00", False),
        ("2026-09-18", False),
    ],
)
def test_rfc3339_utc_timestamp_is_informational(timestamp, valid):
    body = generation()
    body["emitted_at"] = timestamp
    assert (parse_record(jcs_dumps(body)) is not None) == valid


@pytest.mark.parametrize(
    "fault",
    [
        "extra-body",
        "extra-sampling",
        "extra-source",
        "extra-independence",
        "independence-value",
        "oracle-duplicate",
        "oracle-unsorted",
        "reference-duplicate",
        "prompt-duplicate",
        "prompt-unsorted",
        "invalid-source-digest",
        "numeric-cli",
        "empty-actor",
        "invalid-predecessor",
        "mode-without-blob",
        "blob-without-mode",
        "symlink-mode",
        "uppercase-digest",
        "path-traversal",
        "unsorted-transitions",
        "extra-transition",
    ],
)
def test_closed_nested_schemas_and_order(fault):
    body = generation()
    if fault == "extra-body":
        body["trusted"] = True
    elif fault == "extra-sampling":
        body["sampling"]["top_p"] = "1"
    elif fault == "extra-source":
        body["source_capture"]["verified"] = True
    elif fault == "extra-independence":
        body["independence"]["proof"] = "yes"
    elif fault == "independence-value":
        body["independence"]["sibling_draws_visible"] = True
    elif fault in {"oracle-duplicate", "oracle-unsorted", "reference-duplicate"}:
        field = "reference_data" if fault == "reference-duplicate" else "oracles"
        names = ["z", "a"] if fault == "oracle-unsorted" else ["same", "same"]
        body["source_capture"][field] = [
            {"name": n, "version": "fixture", "content_sha256": None} for n in names
        ]
    elif fault == "prompt-duplicate":
        body["prompt_sha256s"] *= 2
    elif fault == "prompt-unsorted":
        body["prompt_sha256s"] = ["f" * 64, "a" * 64]
    elif fault == "invalid-source-digest":
        body["source_capture"]["content_sha256"] = "bad"
    elif fault == "numeric-cli":
        body["cli_version"] = 1
    elif fault == "empty-actor":
        body = correction()
        body["actor"] = ""
    elif fault == "invalid-predecessor":
        body = correction()
        body["predecessor_record_sha256"] = False
    elif fault == "mode-without-blob":
        body["transitions"][0]["before_mode"] = "100644"
    elif fault == "blob-without-mode":
        body["transitions"][0]["after_mode"] = None
    elif fault == "symlink-mode":
        body["transitions"][0]["after_mode"] = "120000"
    elif fault == "uppercase-digest":
        body["transitions"][0]["after_blob_sha256"] = "A" * 64
    elif fault == "path-traversal":
        body["transitions"][0]["path"] = "rules/../other"
    elif fault == "unsorted-transitions":
        body["transitions"] *= 2
        body["transitions"] = deepcopy(body["transitions"])
        body["transitions"][0]["path"] = "rules/z"
        body["transitions"][1] = {**body["transitions"][1], "path": "rules/a"}
    elif fault == "extra-transition":
        body["transitions"][0]["verified"] = True
    assert parse_record(jcs_dumps(body)) is None


def test_multifault_record_keeps_every_applicable_reason(identities):
    body = generation()
    body.update(lane="elsewhere/repo", epoch_sha256="0" * 64)
    body["transitions"][0]["path"] = "outside/file"
    body["transitions"] *= 2
    result = classify(identities, {"a" * 64 + ".json": StoreFile(jcs_dumps(body))})
    assert result.ineligible[0].reasons == (
        "address-mismatch",
        "duplicate-transition-paths",
        "invalid-signature",
        "unprotected-path-transition",
        "wrong-epoch",
        "wrong-lane",
    )


def test_malformed_body_only_gets_reasons_with_met_prerequisites(identities):
    result = classify(identities, {"a" * 64 + ".json": StoreFile(b"not json")})
    assert result.ineligible[0].reasons == ("address-mismatch", "malformed-record")


@pytest.mark.parametrize(
    "name",
    [
        "nested/" + "a" * 64 + ".json",
        "A" * 64 + ".json",
        "no-digest.json",
        "a" * 64 + ".json.notary.sig",
        "a" * 64 + ".json.producer.sig",
    ],
)
def test_unrecognized_names_are_not_parsed(identities, name):
    result = classify(identities, {name: StoreFile(b"malformed!")})
    assert result.ineligible[0].reasons == ("unrecognized-store-name",)


def test_bad_new_record_does_not_veto_valid_record_or_poison_history(identities):
    store = identities.store()
    bad = "0" * 64 + ".json"
    store[bad] = StoreFile(b"bad")
    result = classify(identities, store)
    assert len(result.eligible) == 1
    assert result.ineligible[0].store_name == bad
    # Once base-present, every file is inert even after every producer is revoked.
    identities.body["producer"] = []
    later = classify(identities, store, base=store)
    assert later.eligible == later.ineligible == ()


@pytest.mark.parametrize("change", ["delete", "bytes", "mode"])
def test_inherited_malformed_history_is_immutable(identities, change):
    base = {"historical-garbage": StoreFile(b"bad")}
    subject = dict(base)
    if change == "delete":
        del subject["historical-garbage"]
    elif change == "bytes":
        subject["historical-garbage"] = StoreFile(b"different")
    else:
        subject["historical-garbage"] = StoreFile(b"bad", "100755")
    result = classify(identities, subject, base)
    assert isinstance(result, Refusal) and result.code == "structural"
    assert result.path == ".axiom/lineage/historical-garbage"


def test_alias_and_eligible_body_are_distinct_entries(identities):
    store = identities.store()
    name = next(name for name in store if name.endswith(".json"))
    alias = "0" * 64 + ".json"
    store[alias] = store[name]
    store[alias + ".producer.sig"] = store[name + ".producer.sig"]
    result = classify(identities, store)
    assert len(result.eligible) == 1
    assert [(r.store_name, r.reasons) for r in result.ineligible] == [
        (alias, ("address-mismatch",))
    ]


def test_old_orphan_signature_cannot_become_new_evidence(identities):
    store = identities.store()
    sidecar = next(name for name in store if name.endswith(".sig"))
    base = {sidecar: store[sidecar]}
    result = classify(identities, store, base)
    assert result.eligible == ()
    assert result.ineligible[0].reasons == ("invalid-signature",)


def test_new_sidecar_for_inherited_body_is_inert(identities):
    store = identities.store()
    body = next(name for name in store if name.endswith(".json"))
    result = classify(identities, store, {body: store[body]})
    assert result.eligible == ()
    assert result.ineligible[0].store_name.endswith(".producer.sig")
    assert result.ineligible[0].reasons == ("unrecognized-store-name",)


def test_policy_component_boundaries_and_last_match():
    body = policy_body()
    body["rules"].extend(
        [
            {"action": "exclude", "prefix": "rules/private"},
            {"action": "include", "prefix": "rules/private/public"},
        ]
    )
    rules = parse_path_policy(jcs_dumps(body), lane=LANE)
    assert rules.protects("rules") and rules.protects("rules/example")
    assert not rules.protects("rules-evil/example")
    assert not rules.protects("rules/private/example")
    assert rules.protects("rules/private/public/example")


@pytest.mark.parametrize(
    "rules",
    [
        [("include", ".axiom")],
        [("include", ".axiom/lineage")],
        [("include", ".axiom/notary/keys.json")],
        [("exclude", ".axiom"), ("include", ".axiom/lineage/nested")],
        [
            ("include", ".axiom"),
            ("exclude", ".axiom/notary"),
            ("exclude", ".axiom/lineage"),
            ("include", ".axiom/notary/nested"),
        ],
    ],
)
def test_policy_cannot_protect_reserved_store_subtrees(rules):
    body = policy_body()
    body["rules"] = [{"action": action, "prefix": prefix} for action, prefix in rules]
    result = parse_path_policy(jcs_dumps(body), lane=LANE)
    assert isinstance(result, Refusal) and result.code == "policy-invalid"


def test_explicit_reserved_exclusions_and_similar_prefixes_are_valid():
    body = policy_body()
    body["rules"] = [
        {"action": "include", "prefix": ".axiom"},
        {"action": "exclude", "prefix": ".axiom/notary"},
        {"action": "exclude", "prefix": ".axiom/lineage"},
        {"action": "include", "prefix": ".axiom/notary-elsewhere"},
    ]
    result = parse_path_policy(jcs_dumps(body), lane=LANE)
    assert not isinstance(result, Refusal)
    assert result.protects(".axiom/notary-elsewhere/file")


def test_authenticated_declarations_are_not_runtime_measurements(identities):
    body = generation()
    body["runtime_identity"] = "producer-declared-value"
    body["source_capture"]["oracles"] = [
        {"name": "fixture", "version": "declared", "content_sha256": None}
    ]
    result = classify(identities, identities.store(body))
    assert len(result.eligible) == 1


def test_changed_raw_bytes_invalidate_address_and_signature(identities):
    store = identities.store()
    name = next(name for name in store if name.endswith(".json"))
    body = generation()
    body["transitions"][0]["after_blob_sha256"] = "0" * 64
    store[name] = StoreFile(jcs_dumps(body))
    result = classify(identities, store)
    assert result.ineligible[0].reasons == ("address-mismatch", "invalid-signature")


@pytest.mark.parametrize(
    "raw", [b"{}\n", b'{"schema":1,"schema":2}', b'{"x":NaN}', b'{"x":"\\ud800"}']
)
def test_noncanonical_or_invalid_json_is_malformed(identities, raw):
    result = classify(identities, {sha256_hex(raw) + ".json": StoreFile(raw)})
    assert result.ineligible[0].reasons == ("malformed-record",)
