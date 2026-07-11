"""Release-selector blast-radius inventory tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode.cli import (
    _migration_inventory_report,
    cmd_migration_inventory,
    main,
    migration_inventory,
)
from axiom_encode.corpus_resolver import (
    InvalidCorpusReleaseError,
)
from tests.release_object_fixtures import bind_test_corpus_release

RELEASE_SELECTOR = "nz-rulespec-current"


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


def _write_selector(corpus, name=RELEASE_SELECTOR):
    selector = corpus / "manifests" / "releases" / f"{name}.json"
    selector.parent.mkdir(parents=True, exist_ok=True)
    selector.write_text(
        json.dumps(
            {
                "name": name,
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
    return selector


def _write_toolchain(
    rules,
    name=RELEASE_SELECTOR,
    *,
    content_sha256="0" * 64,
):
    waiver = rules / "known-validation-gaps.yaml"
    waiver.parent.mkdir(parents=True, exist_ok=True)
    waiver.write_text("validate_failures: {}\n", encoding="utf-8")
    waiver_sha256 = hashlib.sha256(waiver.read_bytes()).hexdigest()
    toolchain = rules / ".axiom" / "toolchain.toml"
    toolchain.parent.mkdir(parents=True, exist_ok=True)
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{name}"\n'
        f'axiom_corpus_release_content_sha256 = "{content_sha256}"\n'
        f'validation_waiver_set_sha256 = "{waiver_sha256}"\n'
    )


def _bind_release(corpus, rules, *, scopes=None):
    release = bind_test_corpus_release(
        corpus,
        RELEASE_SELECTOR,
        scopes or [("us", "statute", "active")],
    )
    _write_toolchain(
        rules,
        content_sha256=release.content_sha256,
    )
    return release


def _commit_fixture_checkout(rules):
    subprocess.run(["git", "init"], cwd=rules, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=rules,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=rules,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=rules, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=rules,
        check=True,
        capture_output=True,
    )


def _write_fixture(tmp_path):
    corpus = tmp_path / "axiom-corpus"
    _write_selector(corpus)
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
    _bind_release(corpus, rules)
    rules.joinpath("us/statutes/7/2014").mkdir(parents=True)
    rules.joinpath("us/statutes/7/2014/e.yaml").write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        "    corpus_citation_path: us/statute/7/2014/e\nrules: []\n"
    )
    rules.joinpath("us/statutes/26").mkdir(parents=True)
    rules.joinpath("us/statutes/26/1.yaml").write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        "    corpus_citation_path: us/statute/26/1\nrules: []\n"
    )
    _commit_fixture_checkout(rules)
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
    with pytest.raises(SystemExit) as exc_info:
        cmd_migration_inventory(
            SimpleNamespace(
                checkouts=[rules],
                corpus_path=corpus,
                json=True,
            )
        )
    assert exc_info.value.code == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 2
    assert {row["reason"] for row in payload["failures"]} == {
        "duplicated",
        "parent-slice-dependency",
    }
    assert payload["scanned_modules"] == 2
    release = payload["releases"][str(rules.resolve())]
    assert release["name"] == RELEASE_SELECTOR
    assert len(release["content_sha256"]) == 64
    assert len(release["selector_sha256"]) == 64
    assert payload["complete"] is True


def test_migration_inventory_rejects_nonexistent_checkout(tmp_path):
    _rules, corpus = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        migration_inventory(
            [tmp_path / "missing"],
            corpus,
        )


@pytest.mark.parametrize("name", ["workspace", "rulespec-us-co", "rulespec"])
def test_migration_inventory_rejects_noncanonical_checkout_roots(tmp_path, name):
    _rules, corpus = _write_fixture(tmp_path)
    invalid = tmp_path / name
    invalid.mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="exact canonical rulespec-<country>"):
        migration_inventory([invalid], corpus)


def test_migration_inventory_rejects_noncanonical_yml_module(tmp_path):
    rules, corpus = _write_fixture(tmp_path)
    legacy = rules / "us/statutes/26/legacy.yml"
    legacy.write_text("format: rulespec/v1\nrules: []\n")

    with pytest.raises(ValueError, match="exact .yaml extension"):
        migration_inventory([rules], corpus)


def test_migration_inventory_zero_modules_is_incomplete(tmp_path):
    rules, corpus = _write_fixture(tmp_path)
    for module in (rules / "us/statutes").rglob("*.yaml"):
        module.unlink()

    report = _migration_inventory_report([rules], corpus)

    assert report["scanned_modules"] == 0
    assert report["complete"] is False
    assert any(
        "zero canonical RuleSpec modules" in reason
        for reason in report["incomplete_scan_reasons"]
    )
    with pytest.raises(ValueError, match="scan is incomplete"):
        migration_inventory([rules], corpus)


def test_migration_inventory_counts_unreadable_module_as_incomplete(tmp_path):
    rules, corpus = _write_fixture(tmp_path)
    denied = rules / "us/statutes/7/2014/e.yaml"
    original = type(denied).read_text

    def read_text(path, *args, **kwargs):
        if path == denied:
            raise PermissionError("denied")
        return original(path, *args, **kwargs)

    with patch.object(type(denied), "read_text", read_text):
        report = _migration_inventory_report([rules], corpus)
        with pytest.raises(ValueError, match="scan is incomplete"):
            migration_inventory([rules], corpus)
    assert report["complete"] is False
    assert any("invalid-rulespec" in item for item in report["incomplete_scan_reasons"])
    assert report["incomplete_rulespec_count"] == 1
    assert report["resolution_finding_count"] == len(report["failures"])
    assert all(row["reason"] != "invalid-rulespec" for row in report["failures"])


def test_migration_inventory_rejects_missing_source_verification(tmp_path):
    rules, corpus = _write_fixture(tmp_path)
    (rules / "us/policies").mkdir()
    (rules / "us/policies/no-verification.yaml").write_text(
        "format: rulespec/v1\nmodule: {}\nrules: []\n"
    )
    (rules / "us/statutes/7/2014/e.yaml").write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        "    corpus_citation_path: us/statute/7/2014/e\nrules: []\n"
    )
    report = _migration_inventory_report([rules], corpus)
    assert report["scanned_modules"] == 3
    assert report["complete"] is False
    assert report["incomplete_rulespec_count"] == 1
    assert any("invalid-rulespec" in item for item in report["incomplete_scan_reasons"])


@pytest.mark.parametrize(
    "source_verification",
    [
        "source_verification: {}",
        "source_verification:\n    corpus_citation_path: ''",
        "source_verification:\n    corpus_citation_path: [us/statute/7/2014/e]",
    ],
)
def test_migration_inventory_rejects_missing_or_nonstring_singular_source(
    tmp_path,
    source_verification,
):
    rules, corpus = _write_fixture(tmp_path)
    (rules / "us/statutes/7/2014/e.yaml").write_text(
        "format: rulespec/v1\nmodule:\n  " + source_verification + "\nrules: []\n"
    )

    report = _migration_inventory_report([rules], corpus)

    assert report["complete"] is False
    assert report["incomplete_rulespec_count"] == 1
    assert any("invalid-rulespec" in item for item in report["incomplete_scan_reasons"])


def test_migration_inventory_rejects_plural_source_locator(tmp_path):
    rules, corpus = _write_fixture(tmp_path)
    (rules / "us/statutes/7/2014/e.yaml").write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        "    corpus_citation_paths:\n"
        "      - us/statute/7/2014/e\nrules: []\n"
    )

    report = _migration_inventory_report([rules], corpus)

    assert report["complete"] is False
    assert report["incomplete_rulespec_count"] == 1
    assert any("invalid-rulespec" in item for item in report["incomplete_scan_reasons"])


@pytest.mark.parametrize(
    "citation",
    [
        " us/statute/7/2014/e",
        "us/statute/7/2014/e/",
        "us:statutes/7/2014/e",
    ],
)
def test_migration_inventory_rejects_noncanonical_citation_identity(
    tmp_path,
    citation,
):
    rules, corpus = _write_fixture(tmp_path)
    (rules / "us/statutes/7/2014/e.yaml").write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {json.dumps(citation)}\nrules: []\n"
    )

    report = _migration_inventory_report([rules], corpus)

    assert report["complete"] is False
    assert any(
        "exact canonical" in reason for reason in report["incomplete_scan_reasons"]
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("inactive", "inactive-only"),
        ("bodyless", "malformed-descendants"),
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
    (rules / "us/statutes/26/1.yaml").unlink()
    _bind_release(corpus, rules)
    rows = migration_inventory([rules], corpus)
    assert rows[0]["reason"] == expected


def test_migration_inventory_missing_named_selector_cannot_fall_back_to_current(
    tmp_path,
):
    rules, corpus = _write_fixture(tmp_path)
    for release_object in (corpus / "releases" / RELEASE_SELECTOR).glob("*.json"):
        release_object.unlink()

    with pytest.raises(
        InvalidCorpusReleaseError,
        match=rf"{RELEASE_SELECTOR}",
    ):
        migration_inventory([rules], corpus)


@pytest.mark.parametrize(
    "release_selector",
    ["", "../current", "nested/current", " current"],
)
def test_migration_inventory_rejects_unsafe_release_selector_names(
    tmp_path, release_selector
):
    rules, corpus = _write_fixture(tmp_path)
    _write_toolchain(rules, release_selector)

    with pytest.raises(ValueError):
        migration_inventory([rules], corpus)


def test_migration_inventory_cli_uses_toolchain_selector(tmp_path, capsys):
    rules, corpus = _write_fixture(tmp_path)

    with (
        patch(
            "sys.argv",
            [
                "axiom-encode",
                "migration-inventory",
                str(rules),
                "--corpus-path",
                str(corpus),
            ],
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert "RELEASE" in capsys.readouterr().out


def test_migration_inventory_does_not_invent_exact_uk_alias(tmp_path):
    rules = tmp_path / "rulespec-uk"
    _write_toolchain(rules)
    module = rules / "uk/statutes/ukpga/2020/1/3.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        "    corpus_citation_path: uk/statute/ukpga/2020/1/3\nrules: []\n"
    )
    corpus = tmp_path / "axiom-corpus"
    selector = corpus / "manifests/releases" / f"{RELEASE_SELECTOR}.json"
    selector.parent.mkdir(parents=True)
    selector.write_text(
        json.dumps(
            {
                "name": RELEASE_SELECTOR,
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
    _bind_release(
        corpus,
        rules,
        scopes=[("uk", "statute", "active")],
    )
    _commit_fixture_checkout(rules)
    rows = migration_inventory([rules], corpus)
    assert rows[0]["reason"] == "absent"


def test_migration_inventory_does_not_invent_uk_alias_parent(tmp_path):
    rules = tmp_path / "rulespec-uk"
    _write_toolchain(rules)
    module = rules / "uk/statutes/ukpga/2020/1/3/a.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        "    corpus_citation_path: uk/statute/ukpga/2020/1/3/a\nrules: []\n"
    )
    corpus = tmp_path / "axiom-corpus"
    selector = corpus / "manifests/releases" / f"{RELEASE_SELECTOR}.json"
    selector.parent.mkdir(parents=True)
    selector.write_text(
        json.dumps(
            {
                "name": RELEASE_SELECTOR,
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
        record_id="alias-parent",
        body="(a) Target.\n(b) Sibling.",
    )
    row["jurisdiction"] = "uk"
    provision.write_text(json.dumps(row) + "\n")
    _bind_release(
        corpus,
        rules,
        scopes=[("uk", "statute", "active")],
    )
    _commit_fixture_checkout(rules)

    rows = migration_inventory([rules], corpus)

    assert rows[0]["reason"] == "absent"
