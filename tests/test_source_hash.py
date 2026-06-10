"""Tests for source-hash pinning, staleness checking, and stamping helpers."""

import hashlib
import json
from pathlib import Path

from axiom_encode.source_hash import (
    PinnedModule,
    StaleModule,
    check_staleness,
    iter_pinned_modules,
    provenance_block,
    read_corpus_provision_text,
    run_check_source_staleness,
    source_text_sha256,
    source_verification_block,
)

CITATION_PATH = "us/statute/7/2014/e"
SOURCE_TEXT = "The standard deduction shall be $198 for fiscal year 2026."


def _write_corpus(tmp_path: Path, body: str = SOURCE_TEXT) -> Path:
    corpus_root = tmp_path / "axiom-corpus"
    provision_file = corpus_root / "provisions" / "us" / "statute" / "title-7.jsonl"
    provision_file.parent.mkdir(parents=True)
    provision_file.write_text(
        json.dumps({"citation_path": CITATION_PATH, "body": body}) + "\n",
        encoding="utf-8",
    )
    return corpus_root


def _write_module(tmp_path: Path, source_sha256: str) -> Path:
    rulespec_root = tmp_path / "rulespec-us"
    module_path = rulespec_root / "statutes" / "7" / "2014" / "e.yaml"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {CITATION_PATH}\n"
        f"    source_sha256: {source_sha256}\n"
        "rules:\n"
        "  - name: snap_standard_deduction\n"
        "    kind: parameter\n"
        "    dtype: Money\n"
        "    versions:\n"
        "      - effective_from: 2025-10-01\n"
        "        formula: '198'\n",
        encoding="utf-8",
    )
    return module_path


def test_source_text_sha256_matches_hashlib():
    expected = hashlib.sha256(SOURCE_TEXT.encode("utf-8")).hexdigest()
    assert source_text_sha256(SOURCE_TEXT) == expected


def test_source_verification_block_pins_citation_and_hash():
    block = source_verification_block(CITATION_PATH, SOURCE_TEXT)
    assert block == {
        "corpus_citation_path": CITATION_PATH,
        "source_sha256": source_text_sha256(SOURCE_TEXT),
    }


def test_provenance_block_records_encoder_model_and_run():
    block = provenance_block(model="claude-fable-5", run_id="run-001")
    assert block["encoder"].startswith("axiom-encode/")
    assert block["model"] == "claude-fable-5"
    assert block["run_id"] == "run-001"
    assert set(block) == {"encoder", "model", "run_id"}


def test_read_corpus_provision_text_returns_exact_match(tmp_path):
    corpus_root = _write_corpus(tmp_path)
    assert read_corpus_provision_text(corpus_root, CITATION_PATH) == SOURCE_TEXT


def test_read_corpus_provision_text_prefers_latest_source_as_of(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    provision_file = corpus_root / "provisions" / "us" / "statute" / "title-7.jsonl"
    provision_file.parent.mkdir(parents=True)
    records = [
        {
            "citation_path": CITATION_PATH,
            "body": "old text",
            "source_as_of": "2025-01-01",
        },
        {
            "citation_path": CITATION_PATH,
            "body": "new text",
            "source_as_of": "2026-01-01",
        },
    ]
    provision_file.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert read_corpus_provision_text(corpus_root, CITATION_PATH) == "new text"


def test_read_corpus_provision_text_falls_back_to_descendants(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    provision_file = corpus_root / "provisions" / "us" / "statute" / "title-7.jsonl"
    provision_file.parent.mkdir(parents=True)
    records = [
        {"citation_path": CITATION_PATH, "body": None, "heading": "Deductions"},
        {
            "citation_path": f"{CITATION_PATH}/2",
            "body": "second child",
            "level": 1,
            "ordinal": 2,
        },
        {
            "citation_path": f"{CITATION_PATH}/1",
            "body": "first child",
            "level": 1,
            "ordinal": 1,
            "heading": "In general",
        },
    ]
    provision_file.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    text = read_corpus_provision_text(corpus_root, CITATION_PATH)
    assert text == "In general\n\nfirst child\n\nsecond child"


def test_read_corpus_provision_text_missing_returns_none(tmp_path):
    corpus_root = _write_corpus(tmp_path)
    assert read_corpus_provision_text(corpus_root, "us/statute/26/1") is None


def test_iter_pinned_modules_skips_unpinned_modules(tmp_path):
    rulespec_root = tmp_path / "rulespec-us"
    unpinned = rulespec_root / "statutes" / "7" / "2017.yaml"
    unpinned.parent.mkdir(parents=True)
    unpinned.write_text(
        "format: rulespec/v1\nmodule:\n  title: No pin here\nrules: []\n",
        encoding="utf-8",
    )

    assert list(iter_pinned_modules(rulespec_root)) == []


def test_iter_pinned_modules_yields_pin_and_citation(tmp_path):
    pinned_sha = source_text_sha256(SOURCE_TEXT)
    module_path = _write_module(tmp_path, pinned_sha)

    assert list(iter_pinned_modules(module_path.parents[3])) == [
        PinnedModule(module_path, CITATION_PATH, pinned_sha)
    ]


def test_check_staleness_passes_matching_pin(tmp_path):
    corpus_root = _write_corpus(tmp_path)
    module_path = _write_module(tmp_path, source_text_sha256(SOURCE_TEXT))

    assert check_staleness(module_path.parents[3], corpus_root) == []


def test_check_staleness_reports_mismatched_pin(tmp_path):
    corpus_root = _write_corpus(tmp_path, body="republished text")
    pinned_sha = source_text_sha256(SOURCE_TEXT)
    module_path = _write_module(tmp_path, pinned_sha)

    assert check_staleness(module_path.parents[3], corpus_root) == [
        StaleModule(module_path, pinned_sha, source_text_sha256("republished text"))
    ]


def test_check_staleness_reports_missing_provision_as_unverifiable(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    (corpus_root / "provisions").mkdir(parents=True)
    pinned_sha = source_text_sha256(SOURCE_TEXT)
    module_path = _write_module(tmp_path, pinned_sha)

    assert check_staleness(module_path.parents[3], corpus_root) == [
        StaleModule(module_path, pinned_sha, None)
    ]


def test_check_staleness_flags_yaml_integer_pin_instead_of_skipping(tmp_path):
    # An unquoted all-digit digest parses as a YAML integer. A present pin
    # must surface as stale, never silently pass the check.
    corpus_root = _write_corpus(tmp_path)
    module_path = _write_module(tmp_path, "0" * 64)

    stale = check_staleness(module_path.parents[3], corpus_root)

    assert len(stale) == 1
    assert stale[0].module_path == module_path
    assert stale[0].current_sha == source_text_sha256(SOURCE_TEXT)


def test_run_check_source_staleness_exit_zero_when_fresh(tmp_path, capsys):
    corpus_root = _write_corpus(tmp_path)
    module_path = _write_module(tmp_path, source_text_sha256(SOURCE_TEXT))

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(module_path.parents[3]),
            "--corpus-root",
            str(corpus_root),
        ]
    )

    assert exit_code == 0
    assert "1 pinned module(s) match" in capsys.readouterr().out


def test_run_check_source_staleness_exit_one_when_stale(tmp_path, capsys):
    corpus_root = _write_corpus(tmp_path, body="republished text")
    module_path = _write_module(tmp_path, source_text_sha256(SOURCE_TEXT))

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(module_path.parents[3]),
            "--corpus-root",
            str(corpus_root),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"STALE {module_path}" in output
    assert "1 of 1 pinned module(s) are stale." in output


def test_entrypoint_dispatches_check_source_staleness(tmp_path, monkeypatch):
    from axiom_encode import entrypoint

    corpus_root = _write_corpus(tmp_path)
    module_path = _write_module(tmp_path, source_text_sha256(SOURCE_TEXT))
    monkeypatch.setattr(
        "sys.argv",
        [
            "axiom-encode",
            "check-source-staleness",
            "--rulespec-root",
            str(module_path.parents[3]),
            "--corpus-root",
            str(corpus_root),
        ],
    )

    assert entrypoint.main() == 0


def test_entrypoint_delegates_other_commands_to_cli(monkeypatch):
    import axiom_encode.cli
    from axiom_encode import entrypoint

    monkeypatch.setattr(axiom_encode.cli, "main", lambda: 42)
    monkeypatch.setattr("sys.argv", ["axiom-encode", "stats"])

    assert entrypoint.main() == 42
