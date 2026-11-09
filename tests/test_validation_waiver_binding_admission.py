# ruff: noqa: F811 - tests take the imported frozen_repo fixture
"""Signed-manifest admission with per-file waiver bindings (#2092).

These tests run the real manifest verifier. They show that a waiver change
for a file that a manifest does not cover leaves the manifest admissible,
that drift of a covered file's waiver still fails, that old-format manifests
with the repository waiver-set digest still verify, and that malformed
bindings fail closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re

import pytest

from axiom_encode import cli
from axiom_encode import validation_waivers as waivers_module
from tests.test_cli import TEST_PINNED_ENCODER_IDENTITY
from tests.test_retired_apply_paths import (
    _commit,
    _set_producer_identity,
    frozen_repo,  # noqa: F401 - fixture
)

OTHER = "us-ma/regulations/106-cmr/1.yaml"


@pytest.fixture(autouse=True)
def _clear_waiver_ledger_caches():
    """The ledger caches are process-global; keep tests order-independent."""

    cli._VALIDATION_WAIVER_SETS_BY_SHA256.clear()
    cli._VALIDATION_WAIVER_LEDGER_BYTES_BY_SHA256.clear()
    yield
    cli._VALIDATION_WAIVER_SETS_BY_SHA256.clear()
    cli._VALIDATION_WAIVER_LEDGER_BYTES_BY_SHA256.clear()


EXPIRES = "2026-12-21"


def _record(marker: str, *, state: str = "active", expires: str = EXPIRES) -> str:
    return (
        f"    {state}:\n"
        f'      fingerprint: "sha256:{marker * 64}"\n'
        '      owner: "@MaxGhenis"\n'
        '      issue: "https://github.com/TheAxiomFoundation/rulespec-us/issues/1980"\n'
        f'      expires: "{expires}"\n'
    )


def _write_ledger(
    fixture,
    *,
    covered: str | None,
    other: str | None,
    covered_expires: str = EXPIRES,
    covered_pending: str | None = None,
) -> str:
    """Write the waiver ledger, rebind the toolchain digest, return it."""

    text = "validate_failures:\n"
    if covered is not None:
        text += f"  {fixture.rule_relative}:\n" + _record(
            covered, expires=covered_expires
        )
        if covered_pending is not None:
            text += _record(covered_pending, state="pending")
    if other is not None:
        text += f"  {OTHER}:\n" + _record(other)
    if covered is None and other is None:
        text = "validate_failures: {}\n"
    ledger = fixture.repo / waivers_module.DEFAULT_WAIVER_PATH
    ledger.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    toolchain = fixture.repo / ".axiom" / "toolchain.toml"
    updated, count = re.subn(
        r'(?m)^validation_waiver_set_sha256 = "[0-9a-f]{64}"$',
        f'validation_waiver_set_sha256 = "{digest}"',
        toolchain.read_text(encoding="utf-8"),
    )
    assert count == 1
    toolchain.write_text(updated, encoding="utf-8")
    return digest


def _sign(fixture, payload: dict) -> None:
    _set_producer_identity(payload, dict(TEST_PINNED_ENCODER_IDENTITY))
    cli._sign_applied_encoding_manifest(payload, fixture.broker)
    fixture.manifest.write_text(json.dumps(payload) + "\n")


def _per_file_payload(fixture, ledger_digest: str) -> dict:
    payload = copy.deepcopy(fixture.payload)
    payload.pop("signature", None)
    del payload[cli.VALIDATION_WAIVER_SET_SHA256_FIELD]
    policy = payload["validation_execution"]["policy_pre_apply"]
    policy["validation_waiver_set_sha256"] = ledger_digest
    ledger = waivers_module.load_validation_waivers_bytes(
        (fixture.repo / waivers_module.DEFAULT_WAIVER_PATH).read_bytes(),
        allow_expired=True,
        require_paths=False,
    )
    payload[cli.VALIDATION_WAIVER_BINDINGS_FIELD] = (
        waivers_module.build_file_waiver_bindings(ledger, {fixture.rule_relative})
    )
    if payload.get("backend") == "codex":
        # New Codex manifests always carry CLI provenance; only historical
        # set-digest receipts may omit it.
        payload["codex_cli_version"] = "codex-cli 0.0.0"
        payload["codex_cli_sha256"] = "c" * 64
    return payload


def _set_digest_payload(fixture, ledger_digest: str) -> dict:
    payload = copy.deepcopy(fixture.payload)
    payload.pop("signature", None)
    payload[cli.VALIDATION_WAIVER_SET_SHA256_FIELD] = ledger_digest
    payload["validation_execution"]["policy_pre_apply"][
        "validation_waiver_set_sha256"
    ] = ledger_digest
    return payload


def _admission_issues(fixture) -> list[str]:
    expected = cli.verify_rulespec_validation_waiver_set(fixture.repo)
    _payload, _prefix, _digest, issues = (
        cli._load_verified_applied_encoding_manifest_payload(
            fixture.repo,
            fixture.manifest_relative,
            signing_broker=fixture.broker,
            expected_waiver_set_sha256=expected,
            local_corpus_release=fixture.release,
        )
    )
    return issues


def test_per_file_manifest_is_admissible(frozen_repo):
    digest = _write_ledger(frozen_repo, covered="a", other="b")
    _sign(frozen_repo, _per_file_payload(frozen_repo, digest))
    _commit(frozen_repo.repo, "per-file manifest")
    assert _admission_issues(frozen_repo) == []


@pytest.mark.parametrize("other_after", ["c", None])
def test_uncovered_waiver_change_keeps_per_file_manifest_admissible(
    frozen_repo, other_after
):
    digest = _write_ledger(frozen_repo, covered="a", other="b")
    _sign(frozen_repo, _per_file_payload(frozen_repo, digest))
    _commit(frozen_repo.repo, "per-file manifest")
    _write_ledger(frozen_repo, covered="a", other=other_after)
    _commit(frozen_repo.repo, "renew or drop an unrelated waiver")
    assert _admission_issues(frozen_repo) == []


def test_uncovered_waiver_change_still_rejects_set_digest_manifest(frozen_repo):
    """The old binding is the behavior that #2092 removes for new manifests."""

    digest = _write_ledger(frozen_repo, covered="a", other="b")
    _sign(frozen_repo, _set_digest_payload(frozen_repo, digest))
    _commit(frozen_repo.repo, "set-digest manifest")
    assert _admission_issues(frozen_repo) == []
    _write_ledger(frozen_repo, covered="a", other="c")
    _commit(frozen_repo.repo, "renew an unrelated waiver")
    assert any(
        "toolchain-bound waiver set" in issue
        for issue in _admission_issues(frozen_repo)
    )


@pytest.mark.parametrize("covered_after", ["c", None])
def test_covered_waiver_drift_fails_per_file_manifest(frozen_repo, covered_after):
    digest = _write_ledger(frozen_repo, covered="a", other="b")
    _sign(frozen_repo, _per_file_payload(frozen_repo, digest))
    _commit(frozen_repo.repo, "per-file manifest")
    _write_ledger(frozen_repo, covered=covered_after, other="b")
    _commit(frozen_repo.repo, "change the covered file's waiver")
    issues = _admission_issues(frozen_repo)
    assert any(
        f"waiver identity for {frozen_repo.rule_relative} is not current" in issue
        for issue in issues
    ), issues


def test_new_waiver_on_unwaived_covered_file_fails(frozen_repo):
    digest = _write_ledger(frozen_repo, covered=None, other="b")
    _sign(frozen_repo, _per_file_payload(frozen_repo, digest))
    _commit(frozen_repo.repo, "per-file manifest without a covered waiver")
    assert _admission_issues(frozen_repo) == []
    _write_ledger(frozen_repo, covered="a", other="b")
    _commit(frozen_repo.repo, "waive the covered file after signing")
    issues = _admission_issues(frozen_repo)
    assert any(
        f"waiver identity for {frozen_repo.rule_relative} is not current" in issue
        for issue in issues
    ), issues


def test_covered_waiver_renewal_makes_per_file_manifest_stale(frozen_repo):
    """A renewal binds a new record (owner, issue, expires), so it re-signs."""

    digest = _write_ledger(frozen_repo, covered="a", other="b")
    _sign(frozen_repo, _per_file_payload(frozen_repo, digest))
    _commit(frozen_repo.repo, "per-file manifest")
    _write_ledger(frozen_repo, covered="a", other="b", covered_expires="2027-01-15")
    _commit(frozen_repo.repo, "renew the covered file's waiver")
    issues = _admission_issues(frozen_repo)
    assert any(
        f"waiver identity for {frozen_repo.rule_relative} is not current" in issue
        for issue in issues
    ), issues


def test_pending_record_on_covered_file_keeps_per_file_manifest_admissible(
    frozen_repo,
):
    """A pending record never waives a failure, so it is not bound."""

    digest = _write_ledger(frozen_repo, covered="a", other="b")
    _sign(frozen_repo, _per_file_payload(frozen_repo, digest))
    _commit(frozen_repo.repo, "per-file manifest")
    _write_ledger(frozen_repo, covered="a", other="b", covered_pending="d")
    _commit(frozen_repo.repo, "stage a pending record for the covered file")
    assert _admission_issues(frozen_repo) == []


def test_per_file_codex_manifest_requires_cli_provenance(frozen_repo):
    digest = _write_ledger(frozen_repo, covered="a", other="b")
    payload = _per_file_payload(frozen_repo, digest)
    assert payload["backend"] == "codex"
    del payload["codex_cli_version"]
    del payload["codex_cli_sha256"]
    _sign(frozen_repo, payload)
    _commit(frozen_repo.repo, "per-file Codex manifest without CLI provenance")
    issues = _admission_issues(frozen_repo)
    assert any(
        issue.endswith("does not match the exact model v5 schema") for issue in issues
    ), issues


def test_old_format_set_digest_manifest_still_verifies(frozen_repo):
    digest = _write_ledger(frozen_repo, covered="a", other="b")
    _sign(frozen_repo, _set_digest_payload(frozen_repo, digest))
    _commit(frozen_repo.repo, "set-digest manifest")
    assert _admission_issues(frozen_repo) == []


def _malformed_both_fields(payload: dict, digest: str) -> None:
    payload[cli.VALIDATION_WAIVER_SET_SHA256_FIELD] = digest


def _malformed_schema(payload: dict, _digest: str) -> None:
    payload[cli.VALIDATION_WAIVER_BINDINGS_FIELD]["schema"] = "v0"


def _malformed_extra_file(payload: dict, _digest: str) -> None:
    payload[cli.VALIDATION_WAIVER_BINDINGS_FIELD]["files"].append(
        {"path": "zz/unrelated.yaml", "fingerprint": None, "record_sha256": None}
    )


def _malformed_no_files(payload: dict, _digest: str) -> None:
    payload[cli.VALIDATION_WAIVER_BINDINGS_FIELD]["files"] = []


def _malformed_bad_record(payload: dict, _digest: str) -> None:
    payload[cli.VALIDATION_WAIVER_BINDINGS_FIELD]["files"][0]["record_sha256"] = (
        "not-a-digest"
    )


def _malformed_null_bindings(payload: dict, _digest: str) -> None:
    payload[cli.VALIDATION_WAIVER_BINDINGS_FIELD] = None


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            _malformed_both_fields,
            "unsupported both validation_waiver_set_sha256 and "
            "validation_waiver_bindings",
        ),
        (_malformed_schema, "validation_waiver_bindings.schema is not supported"),
        (
            _malformed_extra_file,
            "validation_waiver_bindings binds files it does not cover: "
            "zz/unrelated.yaml",
        ),
        (_malformed_no_files, "validation_waiver_bindings.files is malformed"),
        (
            _malformed_bad_record,
            "validation_waiver_bindings.files[0] waiver record identity is invalid",
        ),
        (_malformed_null_bindings, "validation_waiver_bindings is malformed"),
    ],
)
def test_malformed_bindings_fail_closed_at_admission(frozen_repo, mutate, expected):
    digest = _write_ledger(frozen_repo, covered="a", other="b")
    payload = _per_file_payload(frozen_repo, digest)
    mutate(payload, digest)
    _sign(frozen_repo, payload)
    _commit(frozen_repo.repo, "malformed per-file manifest")
    issues = _admission_issues(frozen_repo)
    assert any(expected in issue for issue in issues), issues


def test_per_file_manifest_fails_when_validated_ledger_is_unavailable(frozen_repo):
    """Admission reads the expected ledger by digest and fails if it is absent."""

    digest = _write_ledger(frozen_repo, covered="a", other="b")
    _sign(frozen_repo, _per_file_payload(frozen_repo, digest))
    _commit(frozen_repo.repo, "per-file manifest")
    cli._VALIDATION_WAIVER_SETS_BY_SHA256.clear()
    cli._VALIDATION_WAIVER_LEDGER_BYTES_BY_SHA256.clear()
    _payload, _prefix, _digest, issues = (
        cli._load_verified_applied_encoding_manifest_payload(
            frozen_repo.repo,
            frozen_repo.manifest_relative,
            signing_broker=frozen_repo.broker,
            expected_waiver_set_sha256="f" * 64,
            local_corpus_release=frozen_repo.release,
        )
    )
    assert any("cannot be checked" in issue for issue in issues), issues


def test_writer_binds_from_the_checkout_root_ledger_of_a_content_root(frozen_repo):
    digest = _write_ledger(frozen_repo, covered="a", other="b")
    bindings = cli._build_applied_manifest_file_waiver_bindings(
        frozen_repo.repo / "us",
        covered_paths={frozen_repo.rule_relative},
        validated_waiver_set_sha256=digest,
    )
    assert bindings["files"] == [
        cli._validation_waivers.file_waiver_binding(
            cli._validation_waiver_set_for_sha256(frozen_repo.repo, digest),
            frozen_repo.rule_relative,
        )
    ]
    assert bindings["files"][0]["fingerprint"] == f"sha256:{'a' * 64}"


@pytest.mark.parametrize("validated", ["0" * 64, None])
def test_writer_refuses_to_sign_when_the_ledger_changed_after_validation(
    frozen_repo, validated
):
    _write_ledger(frozen_repo, covered="a", other="b")
    with pytest.raises(RuntimeError, match="ledger changed after validation"):
        cli._build_applied_manifest_file_waiver_bindings(
            frozen_repo.repo,
            covered_paths={frozen_repo.rule_relative},
            validated_waiver_set_sha256=validated,
        )


def test_validated_waiver_digest_of_each_manifest_shape():
    digest = "a" * 64
    per_file = {
        cli.VALIDATION_WAIVER_BINDINGS_FIELD: {},
        "validation_execution": {
            "policy_pre_apply": {"validation_waiver_set_sha256": digest}
        },
    }
    assert cli._manifest_validated_waiver_set_sha256(per_file) == digest
    assert (
        cli._manifest_validated_waiver_set_sha256(
            {cli.VALIDATION_WAIVER_SET_SHA256_FIELD: digest}
        )
        == digest
    )
    both = {**per_file, cli.VALIDATION_WAIVER_SET_SHA256_FIELD: digest}
    assert cli._manifest_validated_waiver_set_sha256(both) is None
    malformed = copy.deepcopy(per_file)
    malformed["validation_execution"]["policy_pre_apply"][
        "validation_waiver_set_sha256"
    ] = "not-a-digest"
    assert cli._manifest_validated_waiver_set_sha256(malformed) is None
    assert cli._manifest_validated_waiver_set_sha256({}) is None


def test_nested_bindings_read_the_ledger_at_the_snapshot_commit(frozen_repo):
    """A nested manifest's ledger may exist only at the protected base."""

    digest = _write_ledger(frozen_repo, covered="a", other="b")
    payload = _per_file_payload(frozen_repo, digest)
    snapshot = _commit(frozen_repo.repo, "ledger at the protected base")
    _write_ledger(frozen_repo, covered="c", other="d")
    _commit(frozen_repo.repo, "the working ledger moved on")
    cli._VALIDATION_WAIVER_SETS_BY_SHA256.clear()
    cli._VALIDATION_WAIVER_LEDGER_BYTES_BY_SHA256.clear()
    assert (
        cli._nested_file_waiver_binding_issues(
            payload,
            repo_path=frozen_repo.repo,
            manifest_label="nested",
            expected_waiver_set_sha256=digest,
            base_commit=snapshot,
        )
        == []
    )
    cli._VALIDATION_WAIVER_SETS_BY_SHA256.clear()
    cli._VALIDATION_WAIVER_LEDGER_BYTES_BY_SHA256.clear()
    # Without the snapshot commit, only the moved-on working ledger exists.
    assert cli._nested_file_waiver_binding_issues(
        payload,
        repo_path=frozen_repo.repo,
        manifest_label="nested",
        expected_waiver_set_sha256=digest,
    )


def test_nested_bindings_reject_covered_drift(frozen_repo):
    digest = _write_ledger(frozen_repo, covered="a", other="b")
    payload = _per_file_payload(frozen_repo, digest)
    drifted = _write_ledger(frozen_repo, covered="c", other="b")
    issues = cli._nested_file_waiver_binding_issues(
        payload,
        repo_path=frozen_repo.repo,
        manifest_label="nested",
        expected_waiver_set_sha256=drifted,
    )
    assert any("is not current" in issue for issue in issues), issues


def test_nested_set_digest_manifests_need_no_bindings_check(frozen_repo):
    digest = _write_ledger(frozen_repo, covered="a", other="b")
    assert (
        cli._nested_file_waiver_binding_issues(
            _set_digest_payload(frozen_repo, digest),
            repo_path=frozen_repo.repo,
            manifest_label="nested",
            expected_waiver_set_sha256=digest,
        )
        == []
    )
