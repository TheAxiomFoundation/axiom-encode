"""Tests for source-hash pinning, staleness checking, and stamping helpers."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from axiom_encode.corpus_resolver import (
    InvalidCorpusCitationError,
    InvalidCorpusReleaseError,
    LocalCorpusRelease,
)
from axiom_encode.source_hash import (
    PinnedModule,
    RuleSpecScanError,
    StaleModule,
    check_staleness,
    iter_pinned_modules,
    provenance_block,
    resolved_source_verification_block,
    run_check_source_staleness,
    source_text_sha256,
)
from tests.release_object_fixtures import (
    TEST_RELEASE_PUBLIC_KEY,
    bind_test_corpus_release,
)

CITATION_PATH = "us/statute/7/2014/e"
SOURCE_TEXT = "The standard deduction shall be $198 for fiscal year 2026."
TEST_RELEASE = "test-release"
TEST_SELECTOR = "source-hash-test-release"


def _release_row(citation_path: str, body: str | None, **extra) -> dict:
    return {
        "id": f"test:{citation_path}",
        "citation_path": citation_path,
        "body": body,
        "jurisdiction": "us",
        "document_class": "statute",
        "version": TEST_RELEASE,
        "source_path": "sources/us/statute/test.xml",
        "source_as_of": "2026-01-01",
        "expression_date": "2026-01-01",
        **extra,
    }


def _write_corpus(tmp_path: Path, body: str = SOURCE_TEXT) -> Path:
    corpus_root = tmp_path / "axiom-corpus"
    provision_file = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / "us"
        / "statute"
        / f"{TEST_RELEASE}.jsonl"
    )
    provision_file.parent.mkdir(parents=True)
    provision_file.write_text(
        json.dumps(_release_row(CITATION_PATH, body)) + "\n",
        encoding="utf-8",
    )
    bind_test_corpus_release(
        corpus_root,
        TEST_SELECTOR,
        [("us", "statute", TEST_RELEASE)],
    )
    return corpus_root


def _local_release(corpus_root: Path) -> LocalCorpusRelease:
    return bind_test_corpus_release(
        corpus_root,
        TEST_SELECTOR,
        [("us", "statute", TEST_RELEASE)],
    )


def _write_module(
    tmp_path: Path,
    source_sha256: str,
    *,
    citation_path: str = CITATION_PATH,
) -> Path:
    rulespec_checkout = tmp_path / "rulespec-us"
    content_root = rulespec_checkout / "us"
    waiver = rulespec_checkout / "known-validation-gaps.yaml"
    waiver.parent.mkdir(parents=True, exist_ok=True)
    waiver.write_text("validate_failures: {}\n", encoding="utf-8")
    waiver_sha256 = hashlib.sha256(waiver.read_bytes()).hexdigest()
    toolchain = rulespec_checkout / ".axiom" / "toolchain.toml"
    toolchain.parent.mkdir(parents=True, exist_ok=True)
    release_objects = sorted(
        (tmp_path / "axiom-corpus" / "releases" / TEST_SELECTOR).glob("*.json")
    )
    release_content_sha256 = (
        release_objects[0].stem if len(release_objects) == 1 else "0" * 64
    )
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{TEST_SELECTOR}"\n'
        f'axiom_corpus_release_content_sha256 = "{release_content_sha256}"\n'
        f'validation_waiver_set_sha256 = "{waiver_sha256}"\n',
        encoding="utf-8",
    )
    module_path = content_root / "statutes" / "7" / "2014" / "e.yaml"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {citation_path}\n"
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


def test_provenance_block_records_encoder_model_and_run():
    block = provenance_block(model="claude-fable-5", run_id="run-001")
    assert block["encoder"].startswith("axiom-encode/")
    assert block["model"] == "claude-fable-5"
    assert block["run_id"] == "run-001"
    assert set(block) == {"encoder", "model", "run_id"}


def test_iter_pinned_modules_reports_unpinned_modules(tmp_path):
    rulespec_root = tmp_path / "rulespec-us" / "us"
    unpinned = rulespec_root / "statutes" / "7" / "2017.yaml"
    unpinned.parent.mkdir(parents=True)
    unpinned.write_text(
        "format: rulespec/v1\nmodule:\n  title: No pin here\nrules: []\n",
        encoding="utf-8",
    )

    assert list(iter_pinned_modules(rulespec_root)) == [
        PinnedModule(unpinned, None, "<missing>")
    ]


def test_iter_pinned_modules_reports_verification_without_source_hash(tmp_path):
    rulespec_root = tmp_path / "rulespec-us" / "us"
    unpinned = rulespec_root / "statutes" / "7" / "2017.yaml"
    unpinned.parent.mkdir(parents=True)
    unpinned.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {CITATION_PATH}\n"
        "rules: []\n",
        encoding="utf-8",
    )

    assert list(iter_pinned_modules(rulespec_root)) == [
        PinnedModule(unpinned, CITATION_PATH, "<missing>")
    ]


def test_iter_pinned_modules_yields_pin_and_citation(tmp_path):
    pinned_sha = source_text_sha256(SOURCE_TEXT)
    module_path = _write_module(tmp_path, pinned_sha)

    assert list(iter_pinned_modules(module_path.parents[3])) == [
        PinnedModule(module_path, CITATION_PATH, pinned_sha)
    ]


@pytest.mark.parametrize(
    "citation_path",
    [
        f"/{CITATION_PATH}/",
        f"'{CITATION_PATH} '",
        "us:statutes/7/2014/e",
    ],
)
def test_iter_pinned_modules_rejects_noncanonical_persisted_citation_identity(
    tmp_path,
    citation_path,
):
    module_path = _write_module(
        tmp_path,
        source_text_sha256(SOURCE_TEXT),
        citation_path=citation_path,
    )

    with pytest.raises(RuleSpecScanError, match="exact canonical"):
        list(iter_pinned_modules(module_path.parents[3]))


@pytest.mark.parametrize(
    "source_root",
    ["legislation", "policies", "regulations", "statutes"],
)
def test_iter_pinned_modules_accepts_each_atomic_module_root(tmp_path, source_root):
    pinned_sha = source_text_sha256(SOURCE_TEXT)
    original = _write_module(tmp_path, pinned_sha)
    content_root = tmp_path / "rulespec-us" / "us"
    module_path = content_root / source_root / "example.yaml"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    original.replace(module_path)

    expected = [PinnedModule(module_path, CITATION_PATH, pinned_sha)]
    assert list(iter_pinned_modules(content_root)) == expected
    assert list(iter_pinned_modules(content_root.parent)) == expected


def test_iter_pinned_modules_ignores_composition_specs(tmp_path):
    pinned_sha = source_text_sha256(SOURCE_TEXT)
    module_path = _write_module(tmp_path, pinned_sha)
    content_root = tmp_path / "rulespec-us" / "us"
    program_spec = content_root / "programs" / "snap" / "fy-2026.yaml"
    program_spec.parent.mkdir(parents=True)
    outside = tmp_path / "outside.yaml"
    outside.write_text("secret: do-not-read\n", encoding="utf-8")
    # A RuleSpec scanner normally rejects YAML symlinks before reading. The
    # ProgramSpec subtree must be absent from that scanner altogether.
    program_spec.symlink_to(outside)

    expected = [PinnedModule(module_path, CITATION_PATH, pinned_sha)]
    assert list(iter_pinned_modules(content_root)) == expected
    assert list(iter_pinned_modules(content_root.parent)) == expected


def test_iter_pinned_modules_ignores_checkout_program_specs(tmp_path):
    pinned_sha = source_text_sha256(SOURCE_TEXT)
    module_path = _write_module(tmp_path, pinned_sha)
    checkout = tmp_path / "rulespec-us"
    program_spec = checkout / "programs" / "us" / "snap" / "fy-2026.yaml"
    program_spec.parent.mkdir(parents=True)
    outside = tmp_path / "outside.yaml"
    outside.write_text("secret: do-not-read\n", encoding="utf-8")
    program_spec.symlink_to(outside)

    assert list(iter_pinned_modules(checkout)) == [
        PinnedModule(module_path, CITATION_PATH, pinned_sha)
    ]


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("regulation/example.yaml", "outside the canonical"),
        ("sources/example.yaml", "outside the canonical"),
        ("statutes/example.yaml", "exact canonical country checkout"),
    ],
)
def test_iter_pinned_modules_rejects_noncanonical_layout(
    tmp_path,
    relative_path,
    message,
):
    original = _write_module(tmp_path, source_text_sha256(SOURCE_TEXT))
    checkout = tmp_path / "rulespec-us"
    module_path = checkout / relative_path
    module_path.parent.mkdir(parents=True, exist_ok=True)
    original.replace(module_path)

    with pytest.raises(RuleSpecScanError, match=message):
        list(iter_pinned_modules(checkout))


@pytest.mark.parametrize("suffix", [".yml", ".test.yml"])
def test_iter_pinned_modules_rejects_removed_yml_extension(tmp_path, suffix):
    content_root = tmp_path / "rulespec-us" / "us"
    path = content_root / "statutes" / f"example{suffix}"
    path.parent.mkdir(parents=True)
    path.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n", encoding="utf-8")

    with pytest.raises(RuleSpecScanError, match=r"\.yml is removed"):
        list(iter_pinned_modules(content_root))


def test_iter_pinned_modules_rejects_plural_corpus_citation_paths_anywhere(tmp_path):
    module_path = _write_module(tmp_path, source_text_sha256(SOURCE_TEXT))
    module_path.write_text(
        module_path.read_text(encoding="utf-8").replace(
            f"    corpus_citation_path: {CITATION_PATH}\n",
            f"    corpus_citation_path: {CITATION_PATH}\n"
            "    corpus_citation_paths:\n"
            f"      - {CITATION_PATH}\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuleSpecScanError, match="corpus_citation_paths"):
        list(iter_pinned_modules(module_path.parents[3]))


@pytest.mark.parametrize("suffix", [".test.yaml"])
def test_iter_pinned_modules_skips_explicit_companion_tests(tmp_path, suffix):
    rulespec_root = tmp_path / "rulespec-us" / "us"
    companion = rulespec_root / "statutes" / "7" / f"2014{suffix}"
    companion.parent.mkdir(parents=True)
    companion.write_text("malformed: [", encoding="utf-8")

    assert list(iter_pinned_modules(rulespec_root)) == []


def test_malformed_candidate_rulespec_fails_staleness_scan_closed(
    tmp_path,
    capsys,
):
    rulespec_root = tmp_path / "rulespec-us" / "us"
    module_path = rulespec_root / "statutes" / "7" / "malformed.yaml"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("format: rulespec/v1\nmodule: [", encoding="utf-8")

    with pytest.raises(RuleSpecScanError, match="could not parse"):
        list(iter_pinned_modules(rulespec_root))

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(rulespec_root),
            "--corpus-path",
            str(tmp_path / "unused-corpus"),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"ERROR {module_path}" in output
    assert "could not parse candidate RuleSpec YAML" in output
    assert "No modules under" not in output


def test_unreadable_candidate_rulespec_fails_staleness_scan_closed(
    tmp_path,
    monkeypatch,
    capsys,
):
    module_path = _write_module(tmp_path, source_text_sha256(SOURCE_TEXT))
    rulespec_root = module_path.parents[3]
    original_open = os.open

    def deny_candidate_read(path, flags, mode=0o777, *, dir_fd=None):
        if path == module_path.name and dir_fd is not None:
            raise PermissionError("test read denied")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("axiom_encode.source_hash.os.open", deny_candidate_read)

    with pytest.raises(RuleSpecScanError, match="could not securely open"):
        list(iter_pinned_modules(rulespec_root))

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(rulespec_root),
            "--corpus-path",
            str(tmp_path / "unused-corpus"),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"ERROR {module_path}" in output
    assert "could not securely open candidate RuleSpec YAML" in output
    assert "No modules under" not in output


def test_duplicate_source_hash_key_fails_staleness_scan_closed(tmp_path, capsys):
    module_path = _write_module(tmp_path, "a" * 64)
    rulespec_root = module_path.parents[3]
    content = module_path.read_text(encoding="utf-8")
    module_path.write_text(
        content.replace(
            "    source_sha256: " + "a" * 64 + "\n",
            "    source_sha256: "
            + "a" * 64
            + "\n"
            + "    source_sha256: "
            + "b" * 64
            + "\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuleSpecScanError, match="duplicate key 'source_sha256'"):
        list(iter_pinned_modules(rulespec_root))

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(rulespec_root),
            "--corpus-path",
            str(tmp_path / "unused-corpus"),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"ERROR {module_path}" in output
    assert "duplicate key 'source_sha256'" in output
    assert "No modules under" not in output


@pytest.mark.parametrize("root_kind", ["missing", "file", "symlink"])
def test_invalid_rulespec_root_fails_staleness_scan_closed(
    tmp_path,
    capsys,
    root_kind,
):
    rulespec_root = tmp_path / "rulespec-root"
    if root_kind == "file":
        rulespec_root.write_text("not a directory", encoding="utf-8")
    elif root_kind == "symlink":
        target = tmp_path / "real-rulespec-root"
        target.mkdir()
        rulespec_root.symlink_to(target, target_is_directory=True)

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(rulespec_root),
            "--corpus-path",
            str(tmp_path / "unused-corpus"),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"ERROR {rulespec_root}" in output
    assert "No modules under" not in output


@pytest.mark.parametrize("link_kind", ["broken", "outside"])
def test_yaml_symlink_fails_staleness_scan_closed(tmp_path, capsys, link_kind):
    rulespec_root = tmp_path / "rulespec-us"
    rulespec_root.mkdir()
    target = tmp_path / "outside.yaml"
    if link_kind == "outside":
        target.write_text("module: {}\n", encoding="utf-8")
    module_path = rulespec_root / "linked.yaml"
    module_path.symlink_to(target)

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(rulespec_root),
            "--corpus-path",
            str(tmp_path / "unused-corpus"),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"ERROR {module_path}" in output
    assert "YAML symlinks are not permitted" in output
    assert "No modules under" not in output


@pytest.mark.parametrize("entry_kind", ["directory", "fifo"])
def test_non_regular_yaml_entry_fails_staleness_scan_closed(
    tmp_path,
    capsys,
    entry_kind,
):
    rulespec_root = tmp_path / "rulespec-us"
    rulespec_root.mkdir()
    module_path = rulespec_root / "not-regular.yaml"
    if entry_kind == "directory":
        module_path.mkdir()
    else:
        os.mkfifo(module_path)

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(rulespec_root),
            "--corpus-path",
            str(tmp_path / "unused-corpus"),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"ERROR {module_path}" in output
    assert "YAML paths must be regular files" in output
    assert "No modules under" not in output


def test_traversal_error_fails_staleness_scan_closed(tmp_path, monkeypatch, capsys):
    rulespec_root = tmp_path / "rulespec-us"
    rulespec_root.mkdir()

    def deny_directory_scan(path):
        raise PermissionError("test directory read denied")

    monkeypatch.setattr("axiom_encode.source_hash.os.scandir", deny_directory_scan)

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(rulespec_root),
            "--corpus-path",
            str(tmp_path / "unused-corpus"),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"ERROR {rulespec_root}" in output
    assert "could not enumerate RuleSpec directory" in output
    assert "No modules under" not in output


def test_entry_budget_stops_scandir_before_sort_or_exhaustion(tmp_path, monkeypatch):
    rulespec_root = tmp_path / "rulespec-us"
    rulespec_root.mkdir()

    class FakeEntry:
        def __init__(self, name):
            self.name = name

    class GuardedScanner:
        consumed = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            self.consumed += 1
            if self.consumed > 2:
                pytest.fail("scanner was exhausted before enforcing the entry budget")
            return FakeEntry(f"entry-{self.consumed}")

    scanner = GuardedScanner()
    monkeypatch.setattr("axiom_encode.source_hash._MAX_RULESPEC_SCAN_ENTRIES", 1)
    monkeypatch.setattr("axiom_encode.source_hash.os.scandir", lambda path: scanner)

    with pytest.raises(RuleSpecScanError, match="entry limit"):
        list(iter_pinned_modules(rulespec_root))
    assert scanner.consumed == 2


def test_scan_depth_limit_fails_closed(tmp_path, monkeypatch):
    rulespec_root = tmp_path / "rulespec-us"
    (rulespec_root / "one" / "two").mkdir(parents=True)
    monkeypatch.setattr("axiom_encode.source_hash._MAX_RULESPEC_SCAN_DEPTH", 1)

    with pytest.raises(RuleSpecScanError, match="directory tree exceeds"):
        list(iter_pinned_modules(rulespec_root))


def test_ignored_infrastructure_symlink_and_noise_are_not_scanned(
    tmp_path,
    monkeypatch,
):
    rulespec_root = tmp_path / "rulespec-us"
    rulespec_root.mkdir()
    outside = tmp_path / "outside-infrastructure"
    outside.mkdir()
    (outside / "hidden.yaml").write_text("module: []\n", encoding="utf-8")
    (rulespec_root / ".venv").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr("axiom_encode.source_hash._MAX_RULESPEC_SCAN_ENTRIES", 0)

    assert list(iter_pinned_modules(rulespec_root)) == []


def test_non_yaml_symlink_is_skipped_without_following(tmp_path):
    rulespec_root = tmp_path / "rulespec-us"
    rulespec_root.mkdir()
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    (outside / "hidden.yaml").write_text("module: []\n", encoding="utf-8")
    (rulespec_root / "vendor").symlink_to(outside, target_is_directory=True)

    assert list(iter_pinned_modules(rulespec_root)) == []


def test_candidate_swap_to_symlink_during_open_fails_closed(tmp_path, monkeypatch):
    module_path = _write_module(tmp_path, source_text_sha256(SOURCE_TEXT))
    rulespec_root = module_path.parents[3]
    outside = tmp_path / "outside.yaml"
    outside.write_text("module: {}\n", encoding="utf-8")
    original_open = os.open
    swapped = False

    def swap_before_candidate_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == module_path.name and dir_fd is not None and not swapped:
            swapped = True
            module_path.unlink()
            module_path.symlink_to(outside)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        "axiom_encode.source_hash.os.open",
        swap_before_candidate_open,
    )

    with pytest.raises(RuleSpecScanError, match="securely open"):
        list(iter_pinned_modules(rulespec_root))
    assert swapped


def test_per_file_yaml_byte_limit_fails_closed(tmp_path, monkeypatch):
    module_path = _write_module(tmp_path, source_text_sha256(SOURCE_TEXT))
    monkeypatch.setattr("axiom_encode.source_hash._MAX_RULESPEC_YAML_BYTES", 32)

    with pytest.raises(RuleSpecScanError, match="byte scan limit"):
        list(iter_pinned_modules(module_path.parents[3]))


def test_aggregate_yaml_byte_limit_fails_closed(tmp_path, monkeypatch):
    rulespec_root = tmp_path / "rulespec-us"
    rulespec_root.mkdir()
    for index in range(2):
        (rulespec_root / f"metadata-{index}.yaml").write_text(
            "metadata: " + "x" * 30 + "\n",
            encoding="utf-8",
        )
    monkeypatch.setattr("axiom_encode.source_hash._MAX_RULESPEC_SCAN_YAML_BYTES", 50)

    with pytest.raises(RuleSpecScanError, match="aggregate"):
        list(iter_pinned_modules(rulespec_root))


@pytest.mark.parametrize(
    ("budget_name", "budget_value", "error"),
    [
        ("_MAX_YAML_TOKENS", 5, "token limit"),
        ("_MAX_YAML_NODES", 3, "node limit"),
        ("_MAX_YAML_DEPTH", 2, "level limit"),
    ],
)
def test_yaml_structure_budgets_fail_closed(
    tmp_path,
    monkeypatch,
    budget_name,
    budget_value,
    error,
):
    module_path = _write_module(tmp_path, source_text_sha256(SOURCE_TEXT))
    monkeypatch.setattr(f"axiom_encode.source_hash.{budget_name}", budget_value)

    with pytest.raises(RuleSpecScanError, match=error):
        list(iter_pinned_modules(module_path.parents[3]))


def test_bounded_yaml_anchor_and_alias_remain_supported(tmp_path):
    rulespec_root = tmp_path / "rulespec-us" / "us"
    module_path = rulespec_root / "statutes" / "anchored.yaml"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  summary: &summary Shared summary\n"
        "  description: *summary\n",
        encoding="utf-8",
    )

    assert list(iter_pinned_modules(rulespec_root)) == [
        PinnedModule(module_path, None, "<missing>")
    ]


def test_yaml_alias_budget_fails_closed(tmp_path, monkeypatch):
    rulespec_root = tmp_path / "rulespec-us" / "us"
    module_path = rulespec_root / "statutes" / "anchored.yaml"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "module:\n  summary: &summary Shared summary\n  description: *summary\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("axiom_encode.source_hash._MAX_YAML_ALIASES", 0)

    with pytest.raises(RuleSpecScanError, match="anchor limit"):
        list(iter_pinned_modules(rulespec_root))


def test_top_level_yaml_merge_cannot_hide_pinned_module(tmp_path, capsys):
    rulespec_root = tmp_path / "rulespec-us" / "us"
    module_path = rulespec_root / "statutes" / "merged.yaml"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "defaults: &defaults\n"
        "  module:\n"
        "    source_verification:\n"
        f"      corpus_citation_path: {CITATION_PATH}\n"
        f"      source_sha256: {'a' * 64}\n"
        "<<: *defaults\n",
        encoding="utf-8",
    )

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(rulespec_root),
            "--corpus-path",
            str(tmp_path / "unused-corpus"),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"ERROR {module_path}" in output
    assert "top-level YAML merge key" in output
    assert "No modules under" not in output


def test_nested_yaml_merge_in_rulespec_module_fails_closed(tmp_path):
    rulespec_root = tmp_path / "rulespec-us" / "us"
    module_path = rulespec_root / "statutes" / "merged.yaml"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "defaults: &defaults\n"
        "  corpus_citation_path: us/statute/7/2014/e\n"
        "module:\n"
        "  source_verification:\n"
        "    <<: *defaults\n",
        encoding="utf-8",
    )

    with pytest.raises(RuleSpecScanError, match="merge keys are not permitted"):
        list(iter_pinned_modules(rulespec_root))


def test_nested_yaml_merge_in_non_rulespec_metadata_is_ignored(tmp_path):
    rulespec_root = tmp_path / "rulespec-us"
    metadata_path = rulespec_root / "metadata.yaml"
    rulespec_root.mkdir()
    metadata_path.write_text(
        "defaults: &defaults\n  note: shared\nmetadata:\n  <<: *defaults\n",
        encoding="utf-8",
    )

    assert list(iter_pinned_modules(rulespec_root)) == []


@pytest.mark.parametrize(
    ("module_yaml", "error"),
    [
        ("module: null\n", "module must be a mapping"),
        ("module: []\n", "module must be a mapping"),
        (
            "module:\n  source_verification: []\n",
            "source_verification must be a mapping",
        ),
    ],
)
def test_structurally_invalid_rulespec_candidate_fails_closed(
    tmp_path,
    capsys,
    module_yaml,
    error,
):
    rulespec_root = tmp_path / "rulespec-us" / "us"
    module_path = rulespec_root / "statutes" / "invalid.yaml"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(module_yaml, encoding="utf-8")

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(rulespec_root),
            "--corpus-path",
            str(tmp_path / "unused-corpus"),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"ERROR {module_path}" in output
    assert error in output
    assert "No modules under" not in output


def test_duplicate_keys_in_non_rulespec_metadata_remain_ignored(tmp_path):
    rulespec_root = tmp_path / "rulespec-us"
    metadata_path = rulespec_root / "metadata.yaml"
    rulespec_root.mkdir()
    metadata_path.write_text(
        "items:\n  - note: first\n    note: second\n",
        encoding="utf-8",
    )

    assert list(iter_pinned_modules(rulespec_root)) == []


def test_multiple_yaml_documents_fail_staleness_scan_closed(tmp_path, capsys):
    rulespec_root = tmp_path / "rulespec-us" / "us"
    module_path = rulespec_root / "statutes" / "multiple.yaml"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "module: {}\n---\nmodule: {}\n",
        encoding="utf-8",
    )

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(rulespec_root),
            "--corpus-path",
            str(tmp_path / "unused-corpus"),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"ERROR {module_path}" in output
    assert "could not parse candidate RuleSpec YAML" in output
    assert "No modules under" not in output


def test_check_staleness_passes_matching_pin(tmp_path):
    corpus_root = _write_corpus(tmp_path)
    module_path = _write_module(tmp_path, source_text_sha256(SOURCE_TEXT))

    assert check_staleness(module_path.parents[3], _local_release(corpus_root)) == []


def test_check_staleness_reports_mismatched_pin(tmp_path):
    corpus_root = _write_corpus(tmp_path, body="republished text")
    pinned_sha = source_text_sha256(SOURCE_TEXT)
    module_path = _write_module(tmp_path, pinned_sha)

    assert check_staleness(module_path.parents[3], _local_release(corpus_root)) == [
        StaleModule(module_path, pinned_sha, source_text_sha256("republished text"))
    ]


def test_check_staleness_hashes_full_stored_parent_body(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    provision_file = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / "us"
        / "statute"
        / f"{TEST_RELEASE}.jsonl"
    )
    provision_file.parent.mkdir(parents=True)
    parent_citation = "us/statute/7/2014"
    parent_body = "(e) The standard deduction is $198.\n(f) Other rules apply."
    provision_file.write_text(
        json.dumps(_release_row(parent_citation, parent_body)) + "\n",
        encoding="utf-8",
    )
    module_path = _write_module(
        tmp_path,
        source_text_sha256(parent_body),
        citation_path=CITATION_PATH,
    )

    assert check_staleness(module_path.parents[3], _local_release(corpus_root)) == []


def test_resolved_source_verification_block_pins_full_stored_parent_body(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    provision_file = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / "us"
        / "statute"
        / f"{TEST_RELEASE}.jsonl"
    )
    provision_file.parent.mkdir(parents=True)
    parent_citation = "us/statute/7/2014"
    parent_body = "(e) The standard deduction is $198.\n(f) Other rules apply."
    provision_file.write_text(
        json.dumps(_release_row(parent_citation, parent_body)) + "\n",
        encoding="utf-8",
    )
    block = resolved_source_verification_block(
        _local_release(corpus_root),
        CITATION_PATH,
    )
    module_path = _write_module(
        tmp_path,
        block["source_sha256"],
        citation_path=block["corpus_citation_path"],
    )

    assert block == {
        "corpus_citation_path": CITATION_PATH,
        "source_sha256": source_text_sha256(parent_body),
    }
    assert check_staleness(module_path.parents[3], _local_release(corpus_root)) == []


def test_resolved_source_verification_block_rejects_noncanonical_identity(tmp_path):
    corpus_root = _write_corpus(tmp_path)

    with pytest.raises(InvalidCorpusCitationError, match="exact canonical"):
        resolved_source_verification_block(
            _local_release(corpus_root),
            f"/{CITATION_PATH}/",
        )


def test_check_staleness_reports_missing_provision_as_unverifiable(tmp_path):
    corpus_root = _write_corpus(tmp_path)
    release = _local_release(corpus_root)
    (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / "us"
        / "statute"
        / f"{TEST_RELEASE}.jsonl"
    ).unlink()
    pinned_sha = source_text_sha256(SOURCE_TEXT)
    module_path = _write_module(tmp_path, pinned_sha)

    stale = check_staleness(module_path.parents[3], release)

    assert len(stale) == 1
    assert stale[0].module_path == module_path
    assert stale[0].pinned_sha == pinned_sha
    assert stale[0].current_sha is None
    assert stale[0].resolution_error is not None


def test_check_staleness_surfaces_ambiguous_corpus_source(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    provision_file = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / "us"
        / "statute"
        / f"{TEST_RELEASE}.jsonl"
    )
    provision_file.parent.mkdir(parents=True)
    provision_file.write_text(
        json.dumps(_release_row(CITATION_PATH, "first active text", id="first"))
        + "\n"
        + json.dumps(_release_row(CITATION_PATH, "second active text", id="second"))
        + "\n",
        encoding="utf-8",
    )
    pinned_sha = source_text_sha256("first active text")
    module_path = _write_module(tmp_path, pinned_sha)

    stale = check_staleness(module_path.parents[3], _local_release(corpus_root))

    assert len(stale) == 1
    assert stale[0].current_sha is None
    assert stale[0].resolution_error is not None
    assert "AmbiguousCorpusSourceError" in stale[0].resolution_error


def test_local_release_rejects_missing_release_object_without_fallback(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    provision_file = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / "us"
        / "statute"
        / "legacy.jsonl"
    )
    provision_file.parent.mkdir(parents=True)
    provision_file.write_text(
        json.dumps({"citation_path": CITATION_PATH, "body": SOURCE_TEXT}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(InvalidCorpusReleaseError, match="release object not found"):
        LocalCorpusRelease(
            corpus_root,
            TEST_SELECTOR,
            "0" * 64,
            TEST_RELEASE_PUBLIC_KEY,
        )


@pytest.mark.parametrize(
    "yaml_pin",
    [
        pytest.param("null", id="null"),
        pytest.param("''", id="empty"),
        pytest.param("123456", id="non-string"),
        pytest.param("A" * 64, id="uppercase"),
        pytest.param("a" * 63, id="short"),
        pytest.param("g" * 64, id="non-hex"),
    ],
)
def test_malformed_source_hash_pin_is_unverifiable_without_resolution(
    tmp_path,
    monkeypatch,
    capsys,
    yaml_pin,
):
    corpus_root = _write_corpus(tmp_path)
    module_path = _write_module(tmp_path, yaml_pin)
    rulespec_root = module_path.parents[3]

    def fail_if_resolved(*args, **kwargs):
        pytest.fail("malformed source_sha256 must not trigger source resolution")

    monkeypatch.setattr(
        "axiom_encode.source_hash.resolve_local_corpus_source",
        fail_if_resolved,
    )

    pinned = list(iter_pinned_modules(rulespec_root))
    assert len(pinned) == 1

    stale = check_staleness(rulespec_root, _local_release(corpus_root))
    assert len(stale) == 1
    assert stale[0].module_path == module_path
    assert stale[0].current_sha is None
    assert stale[0].resolution_error is not None
    assert "64 lowercase hexadecimal characters" in stale[0].resolution_error

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(rulespec_root),
            "--corpus-path",
            str(corpus_root),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"STALE {module_path}" in output
    assert "1 of 1 pinned module(s) are stale." in output


def test_run_check_source_staleness_exit_zero_when_fresh(tmp_path, capsys):
    corpus_root = _write_corpus(tmp_path)
    module_path = _write_module(tmp_path, source_text_sha256(SOURCE_TEXT))

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(module_path.parents[3]),
            "--corpus-path",
            str(corpus_root),
        ]
    )

    assert exit_code == 0
    assert "1 pinned module(s) match" in capsys.readouterr().out


def test_run_check_source_staleness_binds_release_before_empty_scan(tmp_path, capsys):
    corpus_root = _write_corpus(tmp_path)
    rulespec_root = tmp_path / "rulespec-us"
    waiver = rulespec_root / "known-validation-gaps.yaml"
    waiver.parent.mkdir(parents=True, exist_ok=True)
    waiver.write_text("validate_failures: {}\n", encoding="utf-8")
    waiver_sha256 = hashlib.sha256(waiver.read_bytes()).hexdigest()
    toolchain = rulespec_root / ".axiom" / "toolchain.toml"
    toolchain.parent.mkdir(parents=True)
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{TEST_SELECTOR}"\n'
        f'axiom_corpus_release_content_sha256 = "{_local_release(corpus_root).content_sha256}"\n'
        f'validation_waiver_set_sha256 = "{waiver_sha256}"\n',
        encoding="utf-8",
    )

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(rulespec_root),
            "--corpus-path",
            str(corpus_root),
        ]
    )

    assert exit_code == 0
    assert "No RuleSpec modules found" in capsys.readouterr().out


def test_run_check_source_staleness_rejects_empty_unbound_checkout(tmp_path, capsys):
    corpus_root = _write_corpus(tmp_path)
    rulespec_root = tmp_path / "rulespec-us"
    rulespec_root.mkdir()

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(rulespec_root),
            "--corpus-path",
            str(corpus_root),
        ]
    )

    assert exit_code == 1
    assert "ERROR corpus release" in capsys.readouterr().out


def test_run_check_source_staleness_exit_one_when_stale(tmp_path, capsys):
    corpus_root = _write_corpus(tmp_path, body="republished text")
    module_path = _write_module(tmp_path, source_text_sha256(SOURCE_TEXT))

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(module_path.parents[3]),
            "--corpus-path",
            str(corpus_root),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert f"STALE {module_path}" in output
    assert "1 of 1 pinned module(s) are stale." in output


def test_run_check_source_staleness_reports_ambiguity_and_exits_one(
    tmp_path,
    capsys,
):
    corpus_root = tmp_path / "axiom-corpus"
    provision_file = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / "us"
        / "statute"
        / f"{TEST_RELEASE}.jsonl"
    )
    provision_file.parent.mkdir(parents=True)
    provision_file.write_text(
        json.dumps(_release_row(CITATION_PATH, "active A", id="active-a"))
        + "\n"
        + json.dumps(_release_row(CITATION_PATH, "active B", id="active-b"))
        + "\n",
        encoding="utf-8",
    )
    bind_test_corpus_release(
        corpus_root,
        TEST_SELECTOR,
        [("us", "statute", TEST_RELEASE)],
    )
    module_path = _write_module(tmp_path, source_text_sha256("active A"))

    exit_code = run_check_source_staleness(
        [
            "--rulespec-root",
            str(module_path.parents[3]),
            "--corpus-path",
            str(corpus_root),
        ]
    )

    assert exit_code == 1
    assert "AmbiguousCorpusSourceError" in capsys.readouterr().out


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
            "--corpus-path",
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
