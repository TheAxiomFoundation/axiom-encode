"""The applied-rulespec/v6 reader: off by default, exact, encoding lane only.

Change 2 of docs/content-bound-merge-guard.md section 8 (#2172). The approval
rules of the merge guard itself (content hash binding, independence,
enrolment, failed reads) are tested in tests/test_content_merge_guard.py.
"""

from __future__ import annotations

import copy
import json

import pytest

from axiom_encode import applied_manifest_v6 as v6
from axiom_encode import cli
from axiom_encode import content_merge_guard as guard
from tests.test_cli import TEST_APPLY_SIGNING_BROKER
from tests.test_encoder_manifest_ancestry import (  # noqa: F401 (fixtures)
    _expected,
    _load,
    _reviewed_promotion,
    _set_producer,
    graph,
    manifest_fixture,
)

MANIFEST = ".axiom/encoding-manifests/us/statutes/26/1.json"
NOT_A_MANIFEST = f"{MANIFEST} is not an encoder apply manifest"
V6_SCHEMA_ISSUE = (
    f"{MANIFEST} does not match the exact unreviewed candidate promotion v6 schema"
)


def _scope(ref: str) -> dict:
    return {
        "schema": v6.ATTESTATION_SCOPE_SCHEMA,
        "validation": v6.ATTESTATION_VALIDATION,
        "owner_review": v6.ATTESTATION_OWNER_REVIEW,
        "origin": v6.ATTESTATION_ORIGIN,
        "candidate": {
            "repository": "TheAxiomFoundation/rulespec-us",
            "pull_request": 2044,
            "head_sha": ref,
            "base_sha": "a" * 40,
            "author_id": 110687043,
        },
        "dispatcher": {"id": 110687043, "login": "PavelMakarchuk"},
        "merge_requirement": v6.MERGE_REQUIREMENT,
    }


def _to_v6(v5_promotion: dict) -> dict:
    payload = copy.deepcopy(v5_promotion)
    payload.pop("signature", None)
    ref = payload.pop("reviewed_rulespec_ref")
    payload["schema_version"] = v6.MANIFEST_SCHEMA_V6
    payload[v6.CANDIDATE_REF_FIELD] = ref
    payload[v6.ATTESTATION_SCOPE_FIELD] = _scope(ref)
    return payload


def _sign(payload: dict) -> dict:
    cli._sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_BROKER)
    return payload


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(v6, "V6_READER_ENABLED", True)


@pytest.fixture
def promotion(manifest_fixture, graph):  # noqa: F811
    """A v5 promotion manifest the live verifier admits, and its checkout."""

    _fixtures, repo, _rule, payload = manifest_fixture
    _set_producer(payload, graph.commits[1])
    return repo, _reviewed_promotion(payload), _expected(graph)


# --- the switch --------------------------------------------------------------


def test_reader_is_off_by_default_and_has_no_runtime_override(monkeypatch):
    assert v6.V6_READER_ENABLED is False
    source = (cli.Path(v6.__file__)).read_text()
    assert "os.environ" not in source and "getenv" not in source
    monkeypatch.setenv("AXIOM_ENCODE_V6_READER", "1")
    assert v6.v6_admissible({"schema_version": v6.MANIFEST_SCHEMA_V6}) is False


def test_only_an_exact_true_switch_enables(monkeypatch, promotion):
    payload = _to_v6(promotion[1])
    monkeypatch.setattr(v6, "V6_READER_ENABLED", 1)
    assert v6.v6_admissible(payload) is False
    monkeypatch.setattr(v6, "V6_READER_ENABLED", True)
    assert v6.v6_admissible(payload) is True


def test_off_a_validly_signed_exact_v6_manifest_is_refused(promotion):
    repo, v5, expected = promotion
    verified, _root, _digest, issues = _load(repo, _sign(_to_v6(v5)), expected)
    assert verified is None
    assert issues == [NOT_A_MANIFEST]


def test_off_the_immutable_base_reader_refuses_v6(promotion, tmp_path):
    repo, v5, _expected_identity = promotion
    raw = json.dumps(_sign(_to_v6(v5))).encode()
    claims, issues = cli._guard_base_manifest_claims(
        raw,
        repo_path=repo,
        manifest_path=MANIFEST,
        historical_head_ref="HEAD",
        retrospective=False,
        roots=("statutes",),
        signing_broker=TEST_APPLY_SIGNING_BROKER,
    )
    assert claims == set()
    assert issues == [
        f"{MANIFEST} at the immutable base is not an encoder apply manifest"
    ]


# --- enabled: signature and exact contract -----------------------------------


def test_enabled_exact_signed_v6_manifest_is_admitted(enabled, promotion):
    repo, v5, expected = promotion
    payload = _sign(_to_v6(v5))
    verified, _root, _digest, issues = _load(repo, payload, expected)
    assert issues == []
    assert verified == payload
    assert v6.owner_review_state(verified) == "none"
    assert v6.owner_review_state(v5) == "before-signing"


def test_enabled_unsigned_v6_manifest_is_refused(enabled, promotion):
    repo, v5, expected = promotion
    verified, _root, _digest, issues = _load(repo, _to_v6(v5), expected)
    assert verified is None
    assert issues == [f"{MANIFEST} is missing an encoder apply manifest signature"]


def test_enabled_v5_signature_does_not_carry_over_to_a_relabelled_manifest(
    enabled, promotion
):
    repo, v5, expected = promotion
    signature = _sign(copy.deepcopy(v5))["signature"]
    forged = _to_v6(v5) | {"signature": signature}
    verified, _root, _digest, issues = _load(repo, forged, expected)
    assert verified is None
    assert len(issues) == 1 and "signature" in issues[0]


def test_enabled_scope_edited_after_signing_is_refused(enabled, promotion):
    repo, v5, expected = promotion
    payload = _sign(_to_v6(v5))
    payload["attestation_scope"]["candidate"]["pull_request"] = 1
    verified, _root, _digest, issues = _load(repo, payload, expected)
    assert verified is None
    assert len(issues) == 1 and "signature" in issues[0]


def _mutations():
    def scope(path, value):
        def apply(payload):
            target = payload["attestation_scope"]
            for key in path[:-1]:
                target = target[key]
            if value is _DROP:
                del target[path[-1]]
            else:
                target[path[-1]] = value

        return apply

    cases = {
        "review-claimed": scope(("owner_review",), "approved"),
        "review-null": scope(("owner_review",), None),
        "scope-schema": scope(("schema",), "axiom-encode/attestation-scope/v2"),
        "validation": scope(("validation",), "none"),
        "origin": scope(("origin",), "model"),
        "merge-requirement": scope(("merge_requirement",), "none"),
        "merge-requirement-missing": scope(("merge_requirement",), _DROP),
        "extra-scope-field": scope(("approved_by",), 6076111),
        "candidate-missing-author": scope(("candidate", "author_id"), _DROP),
        "candidate-author-string": scope(("candidate", "author_id"), "110687043"),
        "candidate-author-bool": scope(("candidate", "author_id"), True),
        "candidate-author-zero": scope(("candidate", "author_id"), 0),
        "candidate-pr-float": scope(("candidate", "pull_request"), 2044.0),
        "candidate-other-org": scope(("candidate", "repository"), "evil/rulespec-us"),
        "candidate-short-name": scope(("candidate", "repository"), "rulespec-us"),
        "candidate-head-short": scope(("candidate", "head_sha"), "d" * 39),
        "candidate-head-other": scope(("candidate", "head_sha"), "e" * 40),
        "candidate-base-upper": scope(("candidate", "base_sha"), "A" * 40),
        "candidate-extra": scope(("candidate", "reviewer"), 1),
        "dispatcher-missing": scope(("dispatcher",), _DROP),
        "dispatcher-id-string": scope(("dispatcher", "id"), "110687043"),
        "dispatcher-login": scope(("dispatcher", "login"), "not a login"),
        "dispatcher-extra": scope(("dispatcher", "type"), "User"),
        "scope-not-object": lambda payload: payload.update(attestation_scope="none"),
    }
    return [pytest.param(apply, id=name) for name, apply in cases.items()]


_DROP = object()


@pytest.mark.parametrize("mutate", _mutations())
def test_enabled_signed_v6_with_a_wrong_scope_is_refused(enabled, promotion, mutate):
    repo, v5, expected = promotion
    payload = _to_v6(v5)
    mutate(payload)
    verified, _root, _digest, issues = _load(repo, _sign(payload), expected)
    assert verified is None
    assert any("attestation scope is invalid" in issue for issue in issues)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda p: p.update(reviewed_rulespec_ref=p.pop("candidate_rulespec_ref")),
            id="v5-ref-name",
        ),
        pytest.param(
            lambda p: p.update(reviewed_rulespec_ref="d" * 40), id="both-ref-names"
        ),
        pytest.param(lambda p: p.pop("attestation_scope"), id="no-scope"),
        pytest.param(lambda p: p.update(review={"approved": True}), id="extra-field"),
        pytest.param(lambda p: p.pop("validation_execution"), id="no-validation"),
        pytest.param(
            lambda p: p.update(
                validation_waiver_bindings={}, validation_waiver_set_sha256="c" * 64
            ),
            id="both-waiver-fields",
        ),
    ],
)
def test_enabled_signed_v6_outside_the_exact_field_set_is_refused(
    enabled, promotion, mutate
):
    repo, v5, expected = promotion
    payload = _to_v6(v5)
    mutate(payload)
    verified, _root, _digest, issues = _load(repo, _sign(payload), expected)
    assert verified is None
    assert V6_SCHEMA_ISSUE in issues


def test_enabled_invalid_candidate_ref_is_refused(enabled, promotion):
    repo, v5, expected = promotion
    payload = _to_v6(v5)
    payload["candidate_rulespec_ref"] = "HEAD"
    payload["attestation_scope"]["candidate"]["head_sha"] = "HEAD"
    verified, _root, _digest, issues = _load(repo, _sign(payload), expected)
    assert verified is None
    assert f"{MANIFEST} reviewed RuleSpec ref is invalid" in issues


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda p: p.update(tool=cli.APPLIED_ENCODING_MODEL_TOOL), id="model-tool"
        ),
        pytest.param(lambda p: p.update(backend="codex"), id="model-backend"),
        pytest.param(lambda p: p.update(backend=None), id="null-backend-key"),
        pytest.param(
            lambda p: p.update(tool=cli.APPLIED_ENCODING_RETIRE_TOOL), id="retire-tool"
        ),
        pytest.param(
            lambda p: p.update(tool=cli.APPLIED_ENCODING_REVIEWED_REPAIR_TOOL),
            id="repair-tool",
        ),
        pytest.param(
            lambda p: p.update(schema_version="axiom-encode/applied-rulespec/v7"),
            id="v7",
        ),
        pytest.param(lambda p: p.update(schema_version=6), id="number-label"),
    ],
)
def test_enabled_v6_label_belongs_to_the_promotion_contract_only(
    enabled, promotion, mutate
):
    repo, v5, expected = promotion
    payload = _to_v6(v5)
    mutate(payload)
    verified, _root, _digest, issues = _load(repo, _sign(payload), expected)
    assert verified is None
    assert issues == [NOT_A_MANIFEST]


def test_enabled_nested_and_historical_readers_still_refuse_v6(enabled, promotion):
    payload = _sign(_to_v6(promotion[1]))
    assert cli._applied_manifest_exact_schema_issues(
        payload, manifest_label="outer.retired_manifest"
    ) == ["outer.retired_manifest is not an encoder apply manifest"]
    assert (
        cli._applied_manifest_exact_schema_issues(
            payload, manifest_label=MANIFEST, admit_v6=True
        )
        == []
    )


def test_enabled_v5_promotion_cannot_carry_the_v6_fields(enabled, promotion):
    repo, v5, expected = promotion
    payload = copy.deepcopy(v5)
    payload["attestation_scope"] = _scope(payload["reviewed_rulespec_ref"])
    verified, _root, _digest, issues = _load(repo, _sign(payload), expected)
    assert verified is None
    assert any(
        "exact reviewed candidate promotion v5 schema" in issue for issue in issues
    )


# --- unreadable input is a refusal, never "nothing there" --------------------


@pytest.mark.parametrize("switch", [False, True])
@pytest.mark.parametrize(
    "raw",
    [b"", b"{", b"\xff\xfe", b"[]", b"null", b'"v6"', b"[" * 100_000],
    ids=["empty", "truncated", "not-utf8", "array", "null", "string", "deep"],
)
def test_unreadable_manifest_is_refused(monkeypatch, promotion, switch, raw):
    monkeypatch.setattr(v6, "V6_READER_ENABLED", switch)
    repo, _v5, expected = promotion
    path = repo / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    verified, _root, _digest, issues = (
        cli._load_verified_applied_encoding_manifest_payload(
            repo,
            MANIFEST,
            signing_broker=TEST_APPLY_SIGNING_BROKER,
            expected_encoder_identity=expected,
        )
    )
    assert verified is None
    assert len(issues) == 1 and issues[0].startswith(MANIFEST)


@pytest.mark.parametrize("switch", [False, True])
@pytest.mark.parametrize("kind", ["missing", "symlink", "directory"])
def test_manifest_that_cannot_be_read_is_refused(monkeypatch, promotion, switch, kind):
    monkeypatch.setattr(v6, "V6_READER_ENABLED", switch)
    repo, v5, expected = promotion
    path = repo / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "symlink":
        target = repo / "elsewhere.json"
        target.write_text(json.dumps(_sign(_to_v6(v5))))
        path.symlink_to(target)
    elif kind == "directory":
        path.mkdir()
    verified, _root, _digest, issues = (
        cli._load_verified_applied_encoding_manifest_payload(
            repo,
            MANIFEST,
            signing_broker=TEST_APPLY_SIGNING_BROKER,
            expected_encoder_identity=expected,
        )
    )
    assert verified is None
    assert issues and all(issue.startswith(MANIFEST) for issue in issues)


@pytest.mark.parametrize("payload", [None, [], "v6", 6, {}, {"schema_version": None}])
def test_predicates_refuse_non_manifests(enabled, payload):
    assert v6.v6_admissible(payload) is False
    assert v6.is_v6_labelled(payload) is False


# --- boundary: only v6-labelled manifests take the new paths ------------------


def _non_v6_payloads(v5_promotion: dict) -> dict[str, dict]:
    model = {field: "x" for field in cli._MODEL_APPLY_MANIFEST_FIELDS} | {
        "schema_version": cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
        "tool": cli.APPLIED_ENCODING_MODEL_TOOL,
        "backend": "codex",
    }
    retire = {field: "x" for field in cli._RETIRE_APPLY_MANIFEST_FIELDS} | {
        "schema_version": cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
        "tool": cli.APPLIED_ENCODING_RETIRE_TOOL,
    }
    return {
        "v5-promotion": v5_promotion,
        "v5-model": model,
        "v5-retirement": retire,
        "v1-historic": {"schema_version": "axiom-encode/applied-rulespec/v1"},
        "no-label": {"tool": v6.PROMOTION_TOOL},
        "toolchain-pin-shaped": {"axiom_encode_ref": "b" * 40},
        "empty": {},
    }


def test_a_manifest_not_labelled_v6_is_judged_the_same_on_and_off(
    monkeypatch, promotion
):
    """Infrastructure and earlier-lane changes carry no v6 manifest.

    Their verification must not depend on the switch. A workflow, pin, ledger,
    tooling or ProgramSpec change has no manifest at all, so no reader in this
    change is reached for it.
    """

    repo, v5, expected = promotion
    results = {}
    for switch in (False, True):
        monkeypatch.setattr(v6, "V6_READER_ENABLED", switch)
        results[switch] = {
            name: (
                cli._apply_manifest_schema_admitted(payload),
                cli._applied_manifest_exact_schema_issues(
                    payload, manifest_label=name, admit_v6=True
                ),
                cli._applied_manifest_exact_schema_issues(payload, manifest_label=name),
            )
            for name, payload in _non_v6_payloads(v5).items()
        }
        loaded = _load(repo, _sign(copy.deepcopy(v5)), expected)
        assert loaded[3] == [] and loaded[0] is not None
    assert results[False] == results[True]
    assert results[False]["v5-promotion"] == (True, [], [])
    assert results[False]["v1-historic"][0] is False
    assert results[False]["toolchain-pin-shaped"][0] is False


def test_v6_hooks_are_reached_only_through_the_label():
    """Every reference to the v6 module in cli.py goes through the label."""

    source = cli.Path(cli.__file__).read_text()
    uses = {
        line.strip().split("_manifest_v6.")[1].split("(")[0].split(")")[0].split()[0]
        for line in source.splitlines()
        if "_manifest_v6." in line
    }
    assert uses == {
        "v6_field_variants",
        "v6_admissible",
        "is_v6_labelled",
        "CANDIDATE_REF_FIELD",
        "contract_issues",
    }
    assert "V6_READER_ENABLED" not in source
    assert "applied-rulespec/v6" not in source.replace(
        "# applied-rulespec/v6 (#2172)", ""
    )


def test_no_writer_emits_v6():
    source = cli.Path(cli.__file__).read_text()
    assert '"schema_version": _manifest_v6' not in source
    assert "MANIFEST_SCHEMA_V6" not in source
    assert cli.APPLIED_ENCODING_MANIFEST_SCHEMA == "axiom-encode/applied-rulespec/v5"


# --- one contract with the evaluator ------------------------------------------


def test_reader_and_merge_guard_parse_the_same_contract():
    for name in (
        "MANIFEST_SCHEMA_V6",
        "PROMOTION_TOOL",
        "ATTESTATION_SCOPE_SCHEMA",
        "ATTESTATION_VALIDATION",
        "ATTESTATION_OWNER_REVIEW",
        "ATTESTATION_ORIGIN",
        "MERGE_REQUIREMENT",
        "ATTESTATION_SCOPE_FIELDS",
        "ATTESTATION_CANDIDATE_FIELDS",
        "ATTESTATION_DISPATCHER_FIELDS",
    ):
        assert getattr(v6, name) == getattr(guard, name), name
    assert set(cli._REVIEWED_CANDIDATE_V6_MANIFEST_VARIANTS) == set(
        guard.V6_MANIFEST_FIELD_VARIANTS
    )
    assert v6.PROMOTION_TOOL == cli.APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL


def test_a_scope_the_reader_admits_is_one_the_merge_guard_parses(promotion):
    payload = _sign(_to_v6(promotion[1]))
    assert v6.attestation_scope_issues(payload, manifest_label="m") == []
    payload["applied_files"] = [
        {"path": "us/statutes/26/1.yaml", "sha256": "0" * 64},
        {"path": "us/statutes/26/1.test.yaml", "sha256": "1" * 64},
    ]
    payload["run_id"] = "12345"
    parsed = guard._parse_manifest(
        payload,
        repository="TheAxiomFoundation/rulespec-us",
        number=2044,
        author_id=110687043,
    )
    assert parsed.dispatcher_id == 110687043
