"""Release-selector blast-radius inventory tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode.cli import (
    _migration_inventory_report,
    cmd_migration_inventory,
    migration_inventory,
)


def _record(citation: str, *, record_id: str, body="Text", version="active"):
    return {
        "id": record_id,
        "citation_path": citation,
        "body": body,
        "jurisdiction": "us",
        "document_class": "statute",
        "version": version,
        "source_path": f"sources/{record_id}.xml",
        "source_as_of": "2026-01-01",
        "expression_date": "2026-01-01",
    }


def _write_fixture(tmp_path):
    corpus = tmp_path / "axiom-corpus"
    selector = corpus / "manifests/releases/current.json"
    selector.parent.mkdir(parents=True)
    selector.write_text(
        json.dumps(
            {
                "name": "current",
                "scopes": [
                    {
                        "jurisdiction": "us",
                        "document_class": "statute",
                        "version": "active",
                    }
                ],
            }
        )
    )
    provisions = corpus / "data/corpus/provisions/us/statute"
    provisions.mkdir(parents=True)
    rows = [
        _record(
            "us/statute/7/2014",
            record_id="parent",
            body="(a) First.\n(e) Target.\n(f) Sibling.",
        ),
        _record("us/statute/26/1", record_id="duplicate-a"),
        _record("us/statute/26/1", record_id="duplicate-b"),
    ]
    provisions.joinpath("active.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    rules = tmp_path / "rulespec-us"
    rules.joinpath("statutes/7/2014").mkdir(parents=True)
    rules.joinpath("statutes/7/2014/e.yaml").write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        "    corpus_citation_path: us/statute/7/2014/e\nrules: []\n"
    )
    rules.joinpath("statutes/26").mkdir(parents=True)
    rules.joinpath("statutes/26/1.yaml").write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        "    corpus_citation_path: us/statute/26/1\nrules: []\n"
    )
    return rules, corpus


def test_migration_inventory_reports_duplicate_and_slice_fallback(tmp_path):
    rules, corpus = _write_fixture(tmp_path)

    rows = migration_inventory([rules], corpus)

    assert {(row["citation"], row["reason"]) for row in rows} == {
        ("us/statute/7/2014/e", "parent-slice-dependency"),
        ("us/statute/26/1", "duplicated"),
    }


def test_migration_inventory_json_is_machine_readable(tmp_path, capsys):
    rules, corpus = _write_fixture(tmp_path)

    cmd_migration_inventory(
        SimpleNamespace(checkouts=[rules], corpus_root=corpus, json=True)
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 2
    assert {row["reason"] for row in payload["failures"]} == {
        "duplicated",
        "parent-slice-dependency",
    }
    assert payload["scanned_modules"] == 2
    assert payload["release_selector_sha256"]
    assert payload["complete"] is True


def test_migration_inventory_rejects_nonexistent_checkout(tmp_path):
    _rules, corpus = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="existing directory"):
        migration_inventory([tmp_path / "missing"], corpus)


def test_migration_inventory_counts_unreadable_module_as_incomplete(tmp_path):
    rules, corpus = _write_fixture(tmp_path)
    denied = rules / "statutes/7/2014/e.yaml"
    original = type(denied).read_text

    def read_text(path, *args, **kwargs):
        if path == denied:
            raise PermissionError("denied")
        return original(path, *args, **kwargs)

    with patch.object(type(denied), "read_text", read_text):
        report = _migration_inventory_report([rules], corpus)
    assert report["complete"] is False
    assert any("invalid-rulespec" in item for item in report["incomplete_scan_reasons"])
    assert any(row["reason"] == "invalid-rulespec" for row in report["failures"])


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("inactive", "inactive-only"),
        ("bodyless", "bodyless-branch"),
        ("missing-version", "missing-version"),
        ("missing-jurisdiction", "missing-jurisdiction"),
        ("missing-document-class", "missing-document-class"),
        ("missing-release-metadata", "missing-release-metadata"),
    ],
)
def test_migration_inventory_classifies_release_failure_classes(
    tmp_path, mutation, expected
):
    rules, corpus = _write_fixture(tmp_path)
    provision = corpus / "data/corpus/provisions/us/statute/active.jsonl"
    row = _record("us/statute/7/2014/e", record_id="target")
    if mutation == "inactive":
        row["version"] = "inactive"
    elif mutation == "bodyless":
        row["body"] = None
    elif mutation == "missing-version":
        row.pop("version")
    elif mutation == "missing-jurisdiction":
        row.pop("jurisdiction")
    elif mutation == "missing-document-class":
        row.pop("document_class")
    else:
        row.pop("source_as_of")
    provision.write_text(json.dumps(row) + "\n")
    (rules / "statutes/26/1.yaml").unlink()
    rows = migration_inventory([rules], corpus)
    assert rows[0]["reason"] == expected


def test_migration_inventory_classifies_unversioned_corpus_checkout(tmp_path):
    rules, corpus = _write_fixture(tmp_path)
    (corpus / "manifests/releases/current.json").unlink()
    reasons = {row["reason"] for row in migration_inventory([rules], corpus)}
    assert reasons == {"unversioned-checkout"}


def test_migration_inventory_distinguishes_exact_alias_from_parent_slice(tmp_path):
    rules = tmp_path / "rulespec-uk"
    module = rules / "statutes/ukpga/2020/1/3.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        "    corpus_citation_path: uk/statute/ukpga/2020/1/3\nrules: []\n"
    )
    corpus = tmp_path / "axiom-corpus"
    selector = corpus / "manifests/releases/current.json"
    selector.parent.mkdir(parents=True)
    selector.write_text(
        json.dumps(
            {
                "name": "current",
                "scopes": [
                    {
                        "jurisdiction": "uk",
                        "document_class": "statute",
                        "version": "active",
                    }
                ],
            }
        )
    )
    provision = corpus / "data/corpus/provisions/uk/statute/active.jsonl"
    provision.parent.mkdir(parents=True)
    row = _record(
        "uk/statute/legislation.gov.uk/ukpga/2020/1/section/3",
        record_id="alias",
    )
    row["jurisdiction"] = "uk"
    provision.write_text(json.dumps(row) + "\n")
    rows = migration_inventory([rules], corpus)
    assert rows[0]["reason"] == "exact-alias-dependency"
