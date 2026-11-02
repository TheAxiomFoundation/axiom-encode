# ruff: noqa: F811  (the imported `checkout` fixture is a test parameter)
"""A landed migration receipt is settled history for ``guard-generated --all`` (#1942).

After a retired-source-metadata migration lands, ``--all`` with the landing
commit as the protected base verifies the unchanged receipt and admits later
trees under the ordinary guard rules. Editing or removing a historical receipt
stays refused, and a second migration lands as its own transaction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import axiom_encode.cli as cli
from axiom_encode.harness.evals import resolve_corpus_source_unit
from axiom_encode.retired_source_metadata import PLAN_SCHEMA
from tests.test_cli import TestGuardGenerated as _GuardFixtures
from tests.test_retired_source_metadata_cli import (  # noqa: F401
    BROKER,
    PRIMARY,
    changed,
    checkout,
    git,
    migrate,
)

HISTORICAL = "historical migration receipts cannot be edited or removed"
OTHER = "us/statutes/26/36B.yaml"


def _guard(checkout, *, base_ref, all_files=False, changed_files=None):
    options = {"base_ref": base_ref}
    if all_files:
        options["all_files"] = True
    if changed_files is not None:
        options["changed_files"] = changed_files
    return cli.guard_generated_change_issues(
        checkout.repo, corpus_path=checkout.args.corpus_path, **options
    )


def _land(checkout, capsys, message="land migration"):
    report = migrate(checkout, capsys)
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", message)
    cli._RETIRED_SOURCE_METADATA_REPLAY_CACHE.clear()
    return report, git(checkout.repo, "rev-parse", "HEAD")


def _commit_unrelated_change(checkout, text="later change outside the migration\n"):
    (checkout.repo / "README.md").write_text(text)
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "later unrelated change")


def test_all_admits_the_landing_tree_itself(checkout, capsys):
    report, landing = _land(checkout, capsys)

    assert _guard(checkout, base_ref=landing, all_files=True) == []
    assert report["receipt_path"]


def test_all_admits_a_later_tree_with_the_landing_commit_as_base(checkout, capsys):
    _report, landing = _land(checkout, capsys)
    _commit_unrelated_change(checkout)

    assert _guard(checkout, base_ref=landing, all_files=True) == []
    # Ordinary diff mode with the same base agreed before the fix and still does.
    assert _guard(checkout, base_ref=landing) == []


@pytest.mark.parametrize("committed", [True, False], ids=["committed", "worktree"])
@pytest.mark.parametrize("operation", ["modify", "delete"])
def test_all_still_refuses_an_edited_historical_receipt(
    checkout, capsys, operation, committed
):
    report, landing = _land(checkout, capsys)
    receipt = checkout.repo / report["receipt_path"]
    if operation == "modify":
        receipt.write_bytes(receipt.read_bytes() + b"\n")
    else:
        receipt.unlink()
    if committed:
        git(checkout.repo, "add", "--all")
        git(checkout.repo, "commit", "-m", f"{operation} historical receipt")

    issues = _guard(checkout, base_ref=landing, all_files=True)

    assert any(
        report["receipt_path"] in issue and HISTORICAL in issue for issue in issues
    ), issues


def test_all_refuses_a_later_edit_to_a_migrated_file_without_a_new_manifest(
    checkout, capsys
):
    _report, landing = _land(checkout, capsys)
    primary = checkout.repo / PRIMARY
    primary.write_text(primary.read_text() + "# hand edit\n")
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "edit migrated file by hand")

    assert _guard(checkout, base_ref=landing, all_files=True)


def test_all_refuses_a_changed_migration_manifest_bound_to_a_historical_receipt(
    checkout, capsys
):
    report, landing = _land(checkout, capsys)
    manifest_path = checkout.repo / report["manifest_paths"][0]
    payload = json.loads(manifest_path.read_text())
    payload["generated_at"] = "2099-01-01T00:00:00+00:00"
    cli._sign_applied_encoding_manifest(payload, BROKER)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "re-sign migration manifest")

    issues = _guard(checkout, base_ref=landing, all_files=True)

    assert any(
        report["manifest_paths"][0] in issue and report["receipt_path"] in issue
        for issue in issues
    ), issues


def test_all_with_the_original_base_still_requires_the_exact_transaction(
    checkout, capsys
):
    _report, _landing = _land(checkout, capsys)
    _commit_unrelated_change(checkout)

    issues = _guard(checkout, base_ref=checkout.base, all_files=True)

    assert any("README.md" in issue for issue in issues), issues


def _add_independent_module(checkout):
    """Give the generator base a second, unrelated module with its own owner."""

    release = cli.load_rulespec_local_corpus_release(
        checkout.repo, checkout.args.corpus_path
    )
    fixtures = _GuardFixtures()
    fixtures.source_attestation = resolve_corpus_source_unit(
        "us/statute/26/1", release
    ).source_attestation
    fixtures.source_attestation["generation_input_sha256"] = (
        fixtures.source_attestation["resolved_text_sha256"]
    )
    fixtures.source_attestation["rulespec_root"] = "rulespec-us/us"
    source_sha256 = fixtures.source_attestation["source_sha256"]
    other = checkout.repo / OTHER
    other.write_text(f"""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/1
    source_sha256: {source_sha256}
    values:
      rate: 0.3
rules: []
""")
    payload = fixtures._model_manifest(
        [{"path": OTHER, "sha256": hashlib.sha256(other.read_bytes()).hexdigest()}]
    )
    cli._sign_applied_encoding_manifest(payload, BROKER)
    owner = checkout.repo / cli._applied_encoding_manifest_path(Path(OTHER))
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text(json.dumps(payload) + "\n")
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "add an independent module")
    checkout.base = git(checkout.repo, "rev-parse", "HEAD")
    checkout.generator_base = checkout.base
    _set_plan(checkout, checkout.base, [PRIMARY])


def _set_plan(checkout, base_commit, modules):
    checkout.args.plan.write_text(
        json.dumps(
            {
                "schema_version": PLAN_SCHEMA,
                "base_commit": base_commit,
                "modules": modules,
            }
        )
    )


def test_second_migration_on_a_landed_one_is_its_own_transaction(checkout, capsys):
    _add_independent_module(checkout)
    first, landing = _land(checkout, capsys, "land first migration")
    _set_plan(checkout, landing, [OTHER])

    second = migrate(checkout, capsys)
    cli._RETIRED_SOURCE_METADATA_REPLAY_CACHE.clear()
    assert second["receipt_path"] != first["receipt_path"]

    # Uncommitted and committed, the second transaction is admitted in every
    # mode that names the first migration's landing commit as the base.
    assert _guard(checkout, base_ref=landing, changed_files=changed(second)) == []
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "land second migration")
    cli._RETIRED_SOURCE_METADATA_REPLAY_CACHE.clear()
    assert _guard(checkout, base_ref=landing) == []
    assert _guard(checkout, base_ref=landing, all_files=True) == []

    # The second transaction still has to be exact.
    _commit_unrelated_change(checkout)
    issues = _guard(checkout, base_ref=landing, all_files=True)
    assert any(
        second["receipt_path"] in issue and "README.md" in issue for issue in issues
    ), issues


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    mode=st.sampled_from(["all", "diff", "changed_files"]),
    receipt_edit=st.sampled_from(["none", "modify", "delete"]),
    later_change=st.booleans(),
)
def test_property_landed_receipt_verdicts(
    checkout, capsys, mode, receipt_edit, later_change
):
    """With the landing commit as base, every mode agrees with the diff mode.

    A tree is admitted exactly when the historical receipt is unchanged; the
    later unrelated change never affects the verdict. Before #1942 the ``all``
    mode refused every landed tree.
    """

    if not hasattr(checkout, "landed"):
        checkout.landed = _land(checkout, capsys)
    report, landing = checkout.landed
    git(checkout.repo, "reset", "--hard", landing)
    git(checkout.repo, "clean", "-fd")
    cli._RETIRED_SOURCE_METADATA_REPLAY_CACHE.clear()
    if later_change:
        _commit_unrelated_change(checkout)
    receipt = checkout.repo / report["receipt_path"]
    if receipt_edit == "modify":
        receipt.write_bytes(receipt.read_bytes() + b" ")
    elif receipt_edit == "delete":
        receipt.unlink()
    if receipt_edit != "none":
        git(checkout.repo, "add", "--all")
        git(checkout.repo, "commit", "-m", "touch historical receipt")
    discovered = git(checkout.repo, "diff", "--name-only", landing).splitlines()

    issues = _guard(
        checkout,
        base_ref=landing,
        all_files=mode == "all",
        changed_files=discovered if mode == "changed_files" else None,
    )

    admitted = receipt_edit == "none"
    assert (issues == []) is admitted, (mode, receipt_edit, later_change, issues)
    if not admitted:
        assert any(HISTORICAL in issue for issue in issues), issues
