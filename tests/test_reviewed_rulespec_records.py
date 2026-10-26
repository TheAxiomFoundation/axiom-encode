"""Reviewed RuleSpec admission records (data/reviewed-rulespec-refs)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from axiom_encode import prepare_signed_backfill as records_module
from axiom_encode.prepare_signed_backfill import (
    REVIEWED_RULESPEC_PR_BASES,
    REVIEWED_RULESPEC_RECORD_SCHEMA,
    REVIEWED_RULESPEC_REFS,
    UNBOUND_PROMOTION_REVIEWED_RULESPEC_REFS,
    load_reviewed_rulespec_records,
    reviewed_rulespec_pr_bases_from_records,
    reviewed_rulespec_refs_from_records,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD_ROOT = ROOT / "data" / "reviewed-rulespec-refs"

# The exact allowlist as it stood in src/axiom_encode/prepare_signed_backfill.py
# before records replaced it (origin/main 2a13e5f08). These pairs and bases are
# frozen: the migration must preserve every one of them unchanged.
PRE_RECORD_REVIEWED_RULESPEC_REFS = frozenset(
    {
        ("ca", "09327ea52b2c09d20ce5f826fe61dceb33253abc"),
        ("dk", "06489d04e7d4b8d424d1711d99df883c6411248a"),
        ("us", "b61918da93fe8a1a29b35b9330aef2085291a5d0"),
        ("us", "251d8d66dabdebcb763d9e7c9b8322a281440c36"),
        ("us", "68cca4a6fa806b63f95277c129575d88d2ac07f1"),
        ("us", "1e04e456ab404860050586c34eef51321eea95e9"),
        ("us", "b1a6e07af093d62f613f83afe26fcb4dd87de491"),
        ("us", "38ddc92d4160a0d39af13bfe232a446b554a15c5"),
        ("us", "ef9dd5f72d529ebc70f539c42144361e536d7563"),
        ("us", "f4fd3203db560c0d4661542388b6ae2f353e0bd3"),
        ("us", "e942ce50546b1c3a1c0c8f3f0404a217eddbe071"),
        ("us", "dc87ef6212accbc4ff67b81f97b6ddf0cf3b5a5c"),
        ("us", "2a503a5c9a2227c363aceaece6c547429c3c0878"),
        ("us", "6535019ce780d9e78f10509f2fe7a2607fb2bdc4"),
        ("us", "c482ef6506c50b54236354926bbce1bcd6434132"),
        ("us", "297aec1691edf7b3a21781c8a825690db1e7c988"),
        ("us", "cab4b7bc6d4b82124d0331964d1cd6c78b1d0683"),
        ("us", "79ffd74fe3d3c83665335ec64feb7458d9cc877a"),
        ("us", "b5273061fc5765dea04bf36f63de39bf40afc2d8"),
        ("us", "d4c168e5a7d8ff28d848deba2b7faaba65a3e09a"),
        ("us", "87c766a7ecc76d7d36272c648f6daf87b4669317"),
        ("us", "746bf3b4cc13ea24ef2aefb3509d24bbe9139b6e"),
        ("ca", "f60f7a84c30e38c7d4961d70647eb0457e7d76c2"),
    }
)
PRE_RECORD_REVIEWED_RULESPEC_PR_BASES = frozenset(
    {
        (
            "dk",
            "06489d04e7d4b8d424d1711d99df883c6411248a",
            "pin/dk-rulespec-2026-08-07",
        ),
        (
            "us",
            "2a503a5c9a2227c363aceaece6c547429c3c0878",
            "hard-cut/canonical-layout-us",
        ),
        (
            "us",
            "297aec1691edf7b3a21781c8a825690db1e7c988",
            "axiom/signed-backfill-us-35001504609-1",
        ),
        (
            "us",
            "cab4b7bc6d4b82124d0331964d1cd6c78b1d0683",
            "axiom/signed-backfill-us-35145159769-1",
        ),
        (
            "us",
            "79ffd74fe3d3c83665335ec64feb7458d9cc877a",
            "axiom/signed-backfill-us-35160240952-1",
        ),
        (
            "us",
            "b5273061fc5765dea04bf36f63de39bf40afc2d8",
            "fix/1248-snap-immigration-status",
        ),
        (
            "us",
            "d4c168e5a7d8ff28d848deba2b7faaba65a3e09a",
            "codex/az-nested-engine-pin",
        ),
        (
            "us",
            "87c766a7ecc76d7d36272c648f6daf87b4669317",
            "axiom/signed-backfill-us-38756101655-1",
        ),
        (
            "us",
            "746bf3b4cc13ea24ef2aefb3509d24bbe9139b6e",
            "fl-snap-bbce-page5",
        ),
    }
)
SHA = "a" * 40


def _record(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": REVIEWED_RULESPEC_RECORD_SCHEMA,
        "country": "us",
        "rulespec_ref": SHA,
        "pr_base_branch": "contrib/candidate",
        "base_sha": None,
        "rulespec_pr": 12,
        "purpose": "Reviewed candidate.",
        "binding": {
            "citation": "us/statute/7/2015/f",
            "rulespec_path": "us/statutes/7/2015/f.yaml",
            "rulespec_sha256": "b" * 64,
            "companion_sha256": "c" * 64,
        },
        "declared": {
            "encoder_commit": "d" * 40,
            "encoder_version": "0.0.0-test",
            "failure_class": None,
            "generator_cli": "codex-cli 1.2.3",
            "generator_model": "gpt-6-luna",
            "outcome_sha256": "e" * 64,
            "pre_encode_gate": "pass",
        },
    }
    payload.update(overrides)
    return payload


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write(
    root: Path,
    payload: object,
    *,
    country: str = "us",
    name: str = f"{SHA}.json",
    raw: bytes | None = None,
    mode: int = 0o644,
) -> Path:
    path = root / country / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(payload) if raw is None else raw)
    path.chmod(mode)
    return path


def test_records_preserve_every_pre_record_admission() -> None:
    assert PRE_RECORD_REVIEWED_RULESPEC_REFS <= REVIEWED_RULESPEC_REFS
    assert {
        base
        for base in REVIEWED_RULESPEC_PR_BASES
        if base[:2] in PRE_RECORD_REVIEWED_RULESPEC_REFS
    } == PRE_RECORD_REVIEWED_RULESPEC_PR_BASES


def test_checkout_records_load_strictly_and_match_derived_sets() -> None:
    records = load_reviewed_rulespec_records(RECORD_ROOT)

    assert reviewed_rulespec_refs_from_records(records) == REVIEWED_RULESPEC_REFS
    assert (
        reviewed_rulespec_pr_bases_from_records(records) == REVIEWED_RULESPEC_PR_BASES
    )
    migrated = [
        record
        for record in records
        if (record.country, record.rulespec_ref) in PRE_RECORD_REVIEWED_RULESPEC_REFS
    ]
    assert len(migrated) == len(PRE_RECORD_REVIEWED_RULESPEC_REFS)
    # Migrated entries never recorded which module they admitted, so they carry
    # no binding and no declared generation metadata.
    assert all(record.payload["binding"] is None for record in migrated)
    assert all(record.payload["declared"] is None for record in migrated)


def test_loader_accepts_complete_record_and_derives_base(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    _write(root, _record())
    _write(
        root,
        _record(
            country="ca",
            rulespec_ref="f" * 40,
            pr_base_branch=None,
            binding=None,
            declared=None,
            rulespec_pr=None,
        ),
        country="ca",
        name=f"{'f' * 40}.json",
    )

    records = load_reviewed_rulespec_records(root)

    assert reviewed_rulespec_refs_from_records(records) == {
        ("ca", "f" * 40),
        ("us", SHA),
    }
    assert reviewed_rulespec_pr_bases_from_records(records) == {
        ("us", SHA, "contrib/candidate")
    }


def test_loader_accepts_empty_root(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    root.mkdir()

    assert load_reviewed_rulespec_records(root) == ()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_record(extra=True), "exactly the v1 schema"),
        (
            {key: value for key, value in _record().items() if key != "purpose"},
            "exactly the v1 schema",
        ),
        (_record(schema="axiom-encode/reviewed-rulespec-admission/v2"), "schema"),
        (_record(country="ca"), "does not match its path"),
        (_record(rulespec_ref="b" * 40), "does not match its path"),
        (_record(rulespec_ref=SHA.upper()), "does not match its path"),
        (_record(pr_base_branch="bad..branch"), "pr_base_branch"),
        (_record(pr_base_branch="-leading"), "pr_base_branch"),
        (_record(pr_base_branch="ends.lock"), "pr_base_branch"),
        (_record(pr_base_branch=""), "pr_base_branch"),
        (_record(base_sha="abc"), "base_sha"),
        (_record(rulespec_pr=True), "rulespec_pr"),
        (_record(rulespec_pr=0), "rulespec_pr"),
        (_record(rulespec_pr="12"), "rulespec_pr"),
        (_record(purpose=""), "purpose"),
        (_record(purpose="two\nlines"), "purpose"),
        (_record(purpose="x" * 401), "purpose"),
        (_record(binding={"citation": "us/statute/7"}), "binding fields"),
        (
            _record(binding={**_record()["binding"], "rulespec_sha256": "B" * 64}),
            "rulespec_sha256",
        ),
        (
            _record(binding={**_record()["binding"], "companion_sha256": None}),
            "companion_sha256",
        ),
        (
            _record(binding={**_record()["binding"], "rulespec_path": "../x.yaml"}),
            "rulespec_path",
        ),
        (
            _record(
                binding={
                    **_record()["binding"],
                    "rulespec_path": "us/statutes/7/2015/f.test.yaml",
                }
            ),
            "rulespec_path",
        ),
        (
            _record(binding={**_record()["binding"], "citation": "/us/statute"}),
            "citation",
        ),
        (_record(declared={"model": "x"}), "declared fields"),
        (
            _record(declared={**_record()["declared"], "encoder_commit": "abc"}),
            "encoder_commit",
        ),
        (
            _record(declared={**_record()["declared"], "generator_model": 5}),
            "generator_model",
        ),
        ([], "JSON object"),
    ],
)
def test_loader_rejects_invalid_record_content(
    tmp_path: Path, payload: object, message: str
) -> None:
    root = tmp_path / "refs"
    _write(root, payload)

    with pytest.raises(ValueError, match=message):
        load_reviewed_rulespec_records(root)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"{not json", "not valid UTF-8 JSON"),
        (b"\xff\xfe", "not valid UTF-8 JSON"),
        (
            _canonical(_record())[:-2] + b',\n  "country": "us"\n}\n',
            "duplicate JSON key",
        ),
        (json.dumps(_record(), sort_keys=True).encode() + b"\n", "not canonical"),
        (_canonical(_record())[:-1], "not canonical"),
        (b" " * (16 * 1024 + 1), "too large"),
    ],
)
def test_loader_rejects_malformed_or_noncanonical_bytes(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    root = tmp_path / "refs"
    _write(root, None, raw=raw)

    with pytest.raises(ValueError, match=message):
        load_reviewed_rulespec_records(root)


@pytest.mark.parametrize("mode", [0o600, 0o664, 0o755])
def test_loader_rejects_wrong_mode(tmp_path: Path, mode: int) -> None:
    root = tmp_path / "refs"
    _write(root, _record(), mode=mode)

    with pytest.raises(ValueError, match="mode 0644"):
        load_reviewed_rulespec_records(root)


@pytest.mark.parametrize(
    ("country", "name"),
    [
        ("us", f"{SHA.upper()}.json"),
        ("us", f"{'a' * 39}.json"),
        ("us", f"{SHA}.json.bak"),
        ("us", ".hidden"),
        ("us", "README.md"),
        ("usa", f"{SHA}.json"),
        ("US", f"{SHA}.json"),
    ],
)
def test_loader_rejects_unexpected_names(
    tmp_path: Path, country: str, name: str
) -> None:
    root = tmp_path / "refs"
    _write(root, _record(country=country), country=country, name=name)

    with pytest.raises(ValueError, match="unexpected entry|named <full-sha>.json"):
        load_reviewed_rulespec_records(root)


def test_loader_rejects_extra_file_at_root(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    _write(root, _record())
    (root / "README.md").write_text("notes\n")

    with pytest.raises(ValueError, match="unexpected entry"):
        load_reviewed_rulespec_records(root)


def test_loader_rejects_nested_directory(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    (root / "us" / f"{SHA}.json").mkdir(parents=True)

    with pytest.raises(ValueError, match="not a regular file"):
        load_reviewed_rulespec_records(root)


def test_loader_rejects_fifo_record(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes are unavailable")
    root = tmp_path / "refs"
    (root / "us").mkdir(parents=True)
    os.mkfifo(root / "us" / f"{SHA}.json", 0o644)

    with pytest.raises(ValueError, match="not a regular file"):
        load_reviewed_rulespec_records(root)


def test_loader_rejects_symlinked_record(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    target = _write(tmp_path / "elsewhere", _record())
    (root / "us").mkdir(parents=True)
    (root / "us" / f"{SHA}.json").symlink_to(target)

    with pytest.raises(ValueError, match="cannot be read safely"):
        load_reviewed_rulespec_records(root)


def test_loader_rejects_symlinked_country_directory(tmp_path: Path) -> None:
    root = tmp_path / "refs"
    _write(tmp_path / "elsewhere", _record())
    root.mkdir()
    (root / "us").symlink_to(tmp_path / "elsewhere" / "us", target_is_directory=True)

    with pytest.raises(ValueError, match="unexpected entry"):
        load_reviewed_rulespec_records(root)


def test_loader_rejects_symlinked_or_missing_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    _write(real, _record())
    link = tmp_path / "refs"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="real directory"):
        load_reviewed_rulespec_records(link)
    with pytest.raises(ValueError, match="unavailable"):
        load_reviewed_rulespec_records(tmp_path / "missing")


def test_installed_copy_admits_no_reviewed_head(tmp_path: Path) -> None:
    """Outside a source checkout the module derives empty sets, fail-closed."""

    package = tmp_path / "site-packages" / "axiom_encode"
    package.mkdir(parents=True)
    copy = package / "prepare_signed_backfill.py"
    shutil.copyfile(records_module.__file__, copy)
    # A stray record tree next to the installed copy must not be consulted.
    _write(tmp_path / "data" / "reviewed-rulespec-refs", _record())
    spec = importlib.util.spec_from_file_location("_installed_psb", copy)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    assert module.REVIEWED_RULESPEC_REFS == frozenset()
    assert module.REVIEWED_RULESPEC_PR_BASES == frozenset()


def test_checkout_without_record_root_admits_no_reviewed_head(
    tmp_path: Path,
) -> None:
    package = tmp_path / "checkout" / "src" / "axiom_encode"
    package.mkdir(parents=True)
    copy = package / "prepare_signed_backfill.py"
    shutil.copyfile(records_module.__file__, copy)
    spec = importlib.util.spec_from_file_location("_bare_checkout_psb", copy)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.REVIEWED_RULESPEC_REFS == frozenset()
        _write(tmp_path / "checkout" / "data" / "reviewed-rulespec-refs", _record())
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    assert module.REVIEWED_RULESPEC_REFS == frozenset({("us", SHA)})


def test_unbound_promotion_set_is_frozen_to_the_migrated_records() -> None:
    """Only pre-record admissions may promote without a record binding."""

    assert UNBOUND_PROMOTION_REVIEWED_RULESPEC_REFS == (
        PRE_RECORD_REVIEWED_RULESPEC_REFS
    )
    records = load_reviewed_rulespec_records(RECORD_ROOT)
    assert UNBOUND_PROMOTION_REVIEWED_RULESPEC_REFS <= {
        (record.country, record.rulespec_ref)
        for record in records
        if record.payload["binding"] is None
    }
