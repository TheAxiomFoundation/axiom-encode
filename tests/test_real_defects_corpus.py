"""The real-defects verifier corpus must reproduce from Git and the release objects.

Tiers, as in ``scripts/verify_real_defects.py``:

* shipped files: always run.
* rulespec Git: runs when the rulespec-us and rulespec-uk checkouts are found
  (siblings of this checkout, or ``AXIOM_REAL_DEFECTS_RULESPEC_US`` and
  ``AXIOM_REAL_DEFECTS_RULESPEC_UK``).
* axiom-corpus Git: opt-in. It streams about 140 MB of provision blobs out of
  a multi-gigabyte pack, so it runs only when ``AXIOM_REAL_DEFECTS_CORPUS_TIER=1``
  is set and the axiom-corpus checkout is found (sibling, or
  ``AXIOM_REAL_DEFECTS_AXIOM_CORPUS``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import subprocess
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


def _cases() -> list[dict]:
    return [
        json.loads(
            (CORPUS_DIR / "cases" / entry["id"] / "case.json").read_text(
                encoding="utf-8"
            )
        )
        for entry in _index()["cases"]
    ]


def test_commit_metadata_is_the_commits_own_not_the_screen_paraphrase():
    for case in _cases():
        assert case["commit_date"], case["id"]
        assert not case["commit_subject"].startswith("(screen-flagged)"), case["id"]
        reason = case["triage"]["screen_reason"]
        assert (reason is None) == (case["triage"]["candidate_source"] == "keyword")
        if reason:
            assert reason not in case["commit_subject"], case["id"]


def test_only_label_bearing_fields_carry_the_label():
    """No field outside LABEL_BEARING_KEYS names a defect kind or repeats the
    case's screen reason, reader reasoning, or verifier justification."""

    label_keys = set(verify_real_defects.LABEL_BEARING_KEYS)
    for case in _cases():
        triage = case["triage"]
        probes = [
            text[:80]
            for text in (
                triage.get("screen_reason") or "",
                triage["pre_fix_wrong_because"],
                triage["verifier_justification"],
                case["triage_notes"],
            )
            if len(text) >= 20
        ]
        for key, value in case.items():
            if key in label_keys:
                continue
            rendered = json.dumps(value, ensure_ascii=False)
            assert "(screen-flagged)" not in rendered, (case["id"], key)
            for kind in verify_real_defects.DEFECT_KINDS:
                if kind != "other":
                    assert kind not in rendered, (case["id"], key, kind)
            for probe in probes:
                assert probe not in rendered, (case["id"], key)


CORPUS_TIER_ENV = "AXIOM_REAL_DEFECTS_CORPUS_TIER"


@pytest.mark.skipif(
    os.environ.get(CORPUS_TIER_ENV) != "1",
    reason=f"corpus tier is opt-in; set {CORPUS_TIER_ENV}=1 to run it",
)
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


# --------------------------------------------------------------------------
# The streaming reader the git and corpus tiers use
# --------------------------------------------------------------------------

# Characters str.splitlines() treats as line boundaries, plus ordinary text.
_ALPHABET = ["a", "b", "{", "}", '"', " ", "\n", "\r", "\r\n", "\x0b", "\x0c"]
_ALPHABET += ["\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029", "é", "€", "😀"]


def _chunked(raw: bytes, size: int):
    for start in range(0, len(raw), size):
        yield raw[start : start + size]


def test_scan_lines_matches_splitlines_for_any_text_and_chunking():
    """Property: for every text and chunk size, scan_lines reports the blob's
    sha256, the str.splitlines() line count, and exactly the wanted lines."""

    rng = random.Random(1659)
    for trial in range(3000):
        text = "".join(rng.choice(_ALPHABET) for _ in range(rng.randrange(0, 60)))
        raw = text.encode("utf-8")
        lines = text.splitlines()
        wanted = {rng.randrange(1, len(lines) + 3) for _ in range(3)}
        size = rng.choice([1, 2, 3, 5, 7, 64, 1 << 20])
        digest, count, found = verify_real_defects.scan_lines(
            _chunked(raw, size), wanted
        )
        assert digest == hashlib.sha256(raw).hexdigest(), trial
        assert count == len(lines), (trial, text)
        assert found == {n: lines[n - 1] for n in wanted if n <= len(lines)}, (
            trial,
            text,
        )


def test_git_object_reader_streams_blobs_and_reports_missing(tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    payload = ("row one\r\nrow\u2028two\n" * 50_000).encode("utf-8")
    (repo / "big.jsonl").write_bytes(payload)
    (repo / "small.txt").write_bytes(b"x")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "commit",
            "-qm",
            "init",
        ],
        check=True,
    )
    with verify_real_defects.GitObjectReader(repo) as reader:
        reader.CHUNK_BYTES = 4096
        assert reader.blob_chunks("HEAD", "absent.txt") is None
        assert reader.blob_chunks("HEAD", "") is None  # a tree, not a blob
        chunks = reader.blob_chunks("HEAD", "big.jsonl")
        assert chunks is not None
        sizes = []
        digest, count, found = verify_real_defects.scan_lines(
            (sizes.append(len(c)) or c for c in chunks), {1, 3}
        )
        assert max(sizes) <= 4096
        text = payload.decode("utf-8")
        assert digest == hashlib.sha256(payload).hexdigest()
        assert count == len(text.splitlines())
        assert found == {1: "row one", 3: "two"}
        # An abandoned iterator drains, so the next request stays in sync.
        partial = reader.blob_chunks("HEAD", "big.jsonl")
        assert partial is not None
        next(partial)
        partial.close()
        assert (
            reader.blob_sha256("HEAD", "small.txt") == hashlib.sha256(b"x").hexdigest()
        )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_git_tier_checks_parent_bytes_and_commit_metadata(tmp_path):
    repo = tmp_path / "rulespec"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    module = repo / "us" / "m.yaml"
    module.parent.mkdir()
    module.write_bytes(b"format: rulespec/v1\nrules: []\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Encode m")
    parent = _git(repo, "rev-parse", "HEAD")
    module.write_bytes(b"format: rulespec/v1\nrules: [x]\n")
    _git(repo, "commit", "-qam", "Fix m")
    commit = _git(repo, "rev-parse", "HEAD")
    date = _git(repo, "log", "-1", "--format=%cI", commit)
    good = {
        "id": "us-001-test",
        "jurisdiction": "us",
        "commit": commit,
        "parent_commit": parent,
        "commit_date": date,
        "commit_subject": "Fix m",
        "module_path": "us/m.yaml",
        "pre_fix_artifact_sha256": hashlib.sha256(
            b"format: rulespec/v1\nrules: []\n"
        ).hexdigest(),
        "post_fix_artifact_sha256": hashlib.sha256(
            b"format: rulespec/v1\nrules: [x]\n"
        ).hexdigest(),
    }
    bad = dict(
        good,
        id="us-002-test",
        commit_date="",
        commit_subject="(screen-flagged) Wrong amount",
        parent_commit=commit,
        post_fix_artifact_sha256="0" * 64,
    )
    report = verify_real_defects.Report()
    verify_real_defects.check_git([good, bad], {"us": repo}, report, "HEAD")
    assert sorted(report.failures) == [
        "us-002-test: commit_date does not match the commit",
        "us-002-test: commit_subject does not match the commit",
        "us-002-test: parent_commit is not the first parent of commit",
        "us-002-test: post_fix artifact digest does not reproduce",
        "us-002-test: pre_fix artifact digest does not reproduce",
    ]
