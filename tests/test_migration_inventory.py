"""Release-selector blast-radius inventory tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

from axiom_encode.cli import cmd_migration_inventory, migration_inventory


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
        ("us/statute/7/2014/e", "slice-fallback-dependency"),
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
        "slice-fallback-dependency",
    }
