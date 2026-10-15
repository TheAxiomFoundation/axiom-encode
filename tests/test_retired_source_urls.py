"""Versioned neutral URL history, exact replay, and protected admission."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

import axiom_encode.cli as cli
from axiom_encode.retired_source_metadata import (
    PLAN_SCHEMA,
    PLAN_SCHEMA_V2,
    RECEIPT_SCHEMA,
    RECEIPT_SCHEMA_V2,
    RetiredSourceMetadataError,
    build_migration,
    canonical_json_bytes,
    load_plan_bytes,
    rewrite_source_metadata,
    verify_migration_replay,
)
from axiom_encode.retired_source_metadata_workflow import resolve_request
from scripts.enforce_attempt_budget import is_retired_source_metadata_plan
from tests.test_retired_source_metadata_cli import (
    BROKER,
    COMPANION,
    IMPORTER,
    PRIMARY,
    changed,
    checkout,
    git,
    guard,
    migrate,
)

# Import the existing isolated signing fixture; these are synthetic test keys.
__all__ = ["checkout"]
PATH = "ca/statutes/example.yaml"
PREFIX = "format: rulespec/v1\nmodule:\n  source_verification:\n    corpus_citation_path: ca/statute/example\n"
URLS = "    source_urls:\n      - 'https://example.gov/old?a=1&b=2'\n      - https://example.gov/a%20b\n      - 'https://example.gov/old?a=1&b=2'\n"


def plan(schema=PLAN_SCHEMA_V2):
    return load_plan_bytes(
        json.dumps(
            {"schema_version": schema, "base_commit": "a" * 40, "modules": [PATH]}
        ).encode()
    )


def test_v2_preserves_order_raw_history_and_every_other_byte():
    tail = "    source_sha256: unchanged\n  source_url: https://example.gov/module\nrules: []\n"
    raw = (PREFIX + URLS + tail).encode()
    migration = build_migration(plan(), base_tree="b" * 40, base_files={PATH: raw})
    assert migration.files[0].after == (PREFIX + tail).encode()
    assert migration.receipt["schema_version"] == RECEIPT_SCHEMA_V2
    history = migration.receipt["primaries"][0]
    assert history["removed_source_urls"] == [
        "https://example.gov/old?a=1&b=2",
        "https://example.gov/a%20b",
        "https://example.gov/old?a=1&b=2",
    ]
    assert history["removed_source_urls_yaml"] == URLS
    assert (
        verify_migration_replay(migration.receipt_bytes, base_files={PATH: raw})
        == migration
    )
    with pytest.raises(RetiredSourceMetadataError, match="nothing-to-migrate"):
        rewrite_source_metadata(migration.files[0].after, plan_schema=PLAN_SCHEMA_V2)


def test_v1_known_receipt_remains_byte_identical_and_does_not_remove_urls():
    raw = (
        PREFIX
        + "    values:\n      amount: 42\n    source_urls:\n      - https://example.gov/old?a=1&b=2\nrules: []\n"
    ).encode()
    migration = build_migration(
        plan(PLAN_SCHEMA), base_tree="b" * 40, base_files={PATH: raw}
    )
    # Frozen from unmodified dde29/2195, including the v1 history shape.
    assert (
        migration.receipt_sha256
        == "79834da8ae6b4e0a398d536ecee344ecaf8ba2aade3addc58ae0733c98747ba1"
    )
    assert b"source_urls:" in migration.files[0].after
    assert "removed_source_urls" not in migration.receipt["primaries"][0]
    assert (
        verify_migration_replay(migration.receipt_bytes, base_files={PATH: raw})
        == migration
    )
    with pytest.raises(RetiredSourceMetadataError, match="nothing-to-migrate"):
        build_migration(
            plan(PLAN_SCHEMA),
            base_tree="b" * 40,
            base_files={PATH: (PREFIX + URLS + "rules: []\n").encode()},
        )


@pytest.mark.parametrize(
    "block",
    [
        "    source_urls: []\n",
        "    source_urls: null\n",
        "    source_urls:\n      value: https://example.gov\n",
        "    source_urls:\n      - 42\n",
        "    source_urls:\n      - ''\n",
        "    source_urls:\n      - &url https://example.gov\n      - *url\n",
        "    source_urls:\n     - https://example.gov\n",
        "    source_urls:\n      - |\n        https://example.gov\n",
        "    source_urls:\n      - https://example.gov\n    source_urls:\n      - https://example.gov/other\n",
    ],
)
def test_v2_refuses_malformed_documentary_lists(block):
    with pytest.raises(RetiredSourceMetadataError):
        rewrite_source_metadata(
            (PREFIX + block + "rules: []\n").encode(), plan_schema=PLAN_SCHEMA_V2
        )


def test_v2_requires_singular_citation_and_preserves_unknown_siblings():
    raw = (
        PREFIX + URLS + "    unknown_sibling: must-not-be-silently-removed\nrules: []\n"
    ).encode()
    result = rewrite_source_metadata(raw, plan_schema=PLAN_SCHEMA_V2)
    assert b"unknown_sibling: must-not-be-silently-removed" in result.after
    with pytest.raises(RetiredSourceMetadataError, match="singular citation"):
        rewrite_source_metadata(
            raw.replace(b"    corpus_citation_path: ca/statute/example\n", b""),
            plan_schema=PLAN_SCHEMA_V2,
        )


@pytest.mark.parametrize("mutation", ["history", "schema", "plan", "postimage"])
def test_v2_replay_rejects_tampering(mutation):
    raw = (PREFIX + URLS + "rules: []\n").encode()
    migration = build_migration(plan(), base_tree="b" * 40, base_files={PATH: raw})
    receipt = copy.deepcopy(migration.receipt)
    if mutation == "history":
        receipt["primaries"][0]["removed_source_urls"].reverse()
        receipt["primaries"][0]["removed_source_urls"].pop()
    elif mutation == "schema":
        receipt["schema_version"] = RECEIPT_SCHEMA
    elif mutation == "plan":
        receipt["plan"]["schema_version"] = PLAN_SCHEMA
    else:
        receipt["files"][0]["after_sha256"] = "0" * 64
    with pytest.raises(RetiredSourceMetadataError):
        verify_migration_replay(
            canonical_json_bytes(receipt) + b"\n", base_files={PATH: raw}
        )


def test_v2_workflow_selection_is_exact_and_excludes_generation_inputs():
    raw = plan().canonical_bytes.decode()
    assert resolve_request(raw, base_ref="a" * 40, environment={}) == plan().payload
    assert is_retired_source_metadata_plan(raw)
    with pytest.raises(ValueError, match="cannot mix"):
        resolve_request(
            raw, base_ref="a" * 40, environment={"REVIEW_FINDING": "generate instead"}
        )
    with pytest.raises(ValueError, match="base_commit"):
        resolve_request(raw, base_ref="b" * 40, environment={})


def test_v2_signed_transaction_cascade_and_tampered_receipt_fail_closed(
    checkout, capsys
):
    source = checkout.repo / PRIMARY
    old_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text(
        source.read_text().replace("    values:\n      rate: 0.2\n", URLS)
    )
    importer = checkout.repo / IMPORTER
    importer.write_text(
        importer.read_text().replace(
            old_hash, hashlib.sha256(source.read_bytes()).hexdigest()
        )
    )
    # This synthetic base needs generator owners for its new module bytes.
    for relative, raw_owner in checkout.owners.items():
        owner = json.loads(raw_owner)
        for entry in owner["applied_files"]:
            entry["sha256"] = hashlib.sha256(
                (checkout.repo / entry["path"]).read_bytes()
            ).hexdigest()
        cli._sign_applied_encoding_manifest(owner, BROKER)
        owner_path = checkout.repo / cli._applied_encoding_manifest_path(Path(relative))
        owner_path.write_text(json.dumps(owner) + "\n")
        checkout.owners[relative] = owner_path.read_bytes()
    git(checkout.repo, "add", ".")
    git(checkout.repo, "commit", "-m", "URL-only synthetic base")
    checkout.base = git(checkout.repo, "rev-parse", "HEAD")
    payload = json.loads(checkout.args.plan.read_text())
    payload.update(
        schema_version=PLAN_SCHEMA_V2,
        base_commit=checkout.base,
    )
    checkout.args.plan.write_text(json.dumps(payload))
    companion = (checkout.repo / COMPANION).read_bytes()
    report = migrate(checkout, capsys)
    assert len(report["cascade_rewrites"]) == 1
    assert guard(checkout, changed(report)) == []
    assert (checkout.repo / COMPANION).read_bytes() == companion
    receipt_path = checkout.repo / report["receipt_path"]
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema_version"] == RECEIPT_SCHEMA_V2
    receipt["primaries"][0]["removed_source_urls"] = ["https://changed.invalid"]
    receipt_path.write_text(json.dumps(receipt))
    cli._RETIRED_SOURCE_METADATA_REPLAY_CACHE.clear()
    assert guard(checkout, changed(report))
