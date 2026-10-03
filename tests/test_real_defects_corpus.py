"""The real-defects verifier corpus must reproduce from Git and the release objects."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CORPUS_DIR = ROOT / "benchmarks" / "verifier" / "real_defects_v0"
_SPEC = importlib.util.spec_from_file_location(
    "verify_real_defects", ROOT / "scripts" / "verify_real_defects.py"
)
assert _SPEC is not None and _SPEC.loader is not None
verify_real_defects = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify_real_defects)


def _sibling_checkout(env_name: str, repo_name: str) -> Path | None:
    """Locate a sibling checkout: env override, then the org folder next to this repo."""

    override = os.environ.get(env_name)
    candidates = [Path(override)] if override else []
    main_checkout = ROOT
    # Worktrees live under <checkout>/.claude/worktrees/<name>; climb to the checkout.
    if ".claude" in main_checkout.parts:
        main_checkout = Path(
            *main_checkout.parts[: main_checkout.parts.index(".claude")]
        )
    candidates.append(main_checkout.parent / repo_name)
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return None


def _index() -> dict:
    return json.loads((CORPUS_DIR / "index.json").read_text(encoding="utf-8"))


def test_index_lists_every_case_directory_once():
    index = _index()
    ids = [entry["id"] for entry in index["cases"]]
    assert ids, "corpus index is empty"
    assert len(ids) == len(set(ids))
    on_disk = sorted(p.name for p in (CORPUS_DIR / "cases").iterdir() if p.is_dir())
    assert sorted(ids) == on_disk
    assert index["counts"]["cases"] == len(ids)
    by_kind = index["counts"]["by_kind"]
    assert sum(by_kind.values()) == len(ids)
    assert set(by_kind) == set(verify_real_defects.DEFECT_KINDS)
    for entry in index["cases"]:
        assert entry["defect_kind"] in verify_real_defects.DEFECT_KINDS
        assert 0 <= entry["confidence"] <= 1


def test_shipped_files_hash_to_case_digests():
    report = verify_real_defects.verify(
        CORPUS_DIR,
        repos={},
        corpus_repo=None,
        release_cache=None,
        roots_dir=None,
        with_release=False,
        main_ref="",
    )
    assert report.checked == _index()["counts"]["cases"]
    assert report.failures == []


def test_case_json_carries_the_documented_schema():
    for entry in _index()["cases"]:
        case = json.loads(
            (CORPUS_DIR / "cases" / entry["id"] / "case.json").read_text(
                encoding="utf-8"
            )
        )
        for key in verify_real_defects.REQUIRED_CASE_KEYS:
            assert key in case, f"{entry['id']} lacks {key}"
        assert len(case["commit"]) == 40 and len(case["parent_commit"]) == 40
        assert case["fix_stage"] in {"post_merge", "pre_merge_review", "unknown"}
        assert case["triage_status"] in {"fidelity", "unclear"}
        resolution = case["provision_resolution"]
        assert resolution["mode"] in {"axiom_encode_resolver", "direct_row_exact"}
        assert resolution["provision_file"].startswith("data/corpus/provisions/")
        assert case["locator"]["pre_fix_lines"] or case["locator"]["post_fix_lines"]
        assert case["provision_chars"] > 0


@pytest.mark.skipif(
    _sibling_checkout("AXIOM_REAL_DEFECTS_RULESPEC_US", "rulespec-us") is None
    or _sibling_checkout("AXIOM_REAL_DEFECTS_RULESPEC_UK", "rulespec-uk") is None,
    reason="rulespec checkouts are not available",
)
def test_artifacts_reproduce_from_rulespec_git():
    repos = {
        "us": _sibling_checkout("AXIOM_REAL_DEFECTS_RULESPEC_US", "rulespec-us"),
        "uk": _sibling_checkout("AXIOM_REAL_DEFECTS_RULESPEC_UK", "rulespec-uk"),
    }
    report = verify_real_defects.verify(
        CORPUS_DIR,
        repos=repos,
        corpus_repo=None,
        release_cache=None,
        roots_dir=None,
        with_release=False,
        main_ref="",
    )
    assert report.failures == []


@pytest.mark.skipif(
    _sibling_checkout("AXIOM_REAL_DEFECTS_AXIOM_CORPUS", "axiom-corpus") is None,
    reason="axiom-corpus checkout is not available",
)
def test_provision_rows_reproduce_from_corpus_git():
    corpus_repo = _sibling_checkout("AXIOM_REAL_DEFECTS_AXIOM_CORPUS", "axiom-corpus")
    report = verify_real_defects.verify(
        CORPUS_DIR,
        repos={},
        corpus_repo=corpus_repo,
        release_cache=None,
        roots_dir=None,
        with_release=False,
        main_ref="",
    )
    assert report.failures == []
