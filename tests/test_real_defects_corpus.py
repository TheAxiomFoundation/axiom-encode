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


# --------------------------------------------------------------------------
# evidence_in_provision (tools/check_evidence.py)
# --------------------------------------------------------------------------

_EVIDENCE_SPEC = importlib.util.spec_from_file_location(
    "check_evidence", CORPUS_DIR / "tools" / "check_evidence.py"
)
assert _EVIDENCE_SPEC is not None and _EVIDENCE_SPEC.loader is not None
check_evidence = importlib.util.module_from_spec(_EVIDENCE_SPEC)
_EVIDENCE_SPEC.loader.exec_module(check_evidence)


def _module(rules: str) -> str:
    return "format: rulespec/v1\nmodule:\n  summary: s\nrules:\n" + rules


_PRE = _module(
    """  - name: cap
    kind: parameter
    source: 7 CFR 273.10(e)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            source:
              corpus_citation_path: us/regulation/7/273/10
              excerpt: up to the maximum of $143.
    versions:
      - effective_from: '2025-10-01'
        formula: '143'
"""
)


def _case(rule_names, **triage):
    return {
        "artifacts_shipped": True,
        "locator": {"rule_names": rule_names},
        "triage": {
            "pre_fix_wrong_because": "",
            "triage_notes": "",
            "verifier_justification": "",
            "verifier_notes": "",
            **triage,
        },
    }


def test_evidence_removed_value_marks_the_provision_as_supporting_pre_fix():
    post = _PRE.replace("formula: '143'", "formula: '198.99'")
    status, check = check_evidence.check_case(
        _case(["cap"]), _PRE, post, "Subtract the deduction up to the maximum of $143."
    )
    assert status == "absent"
    assert check["reason"] == "provision_supports_pre_fix"
    by_text = {s["text"]: s for s in check["strings"]}
    assert by_text["143"]["side"] == "pre" and by_text["143"]["matched"]
    assert by_text["198.99"]["side"] == "post" and not by_text["198.99"]["matched"]
    # The unchanged excerpt carries the removed value, so it is pre-side too.
    assert by_text["up to the maximum of $143."]["side"] == "pre"
    status, _ = check_evidence.check_case(
        _case(["cap"]), _PRE, post, "the maximum is $198.99 from October 1, 2025"
    )
    assert status == "present"


def test_evidence_context_excerpt_counts_only_when_the_fix_added_nothing():
    logic_pre = _PRE.replace("formula: '143'", "formula: x <= 143")
    logic_post = _PRE.replace("formula: '143'", "formula: x < 143")
    provision = "Applies when income is up to the maximum of $143. Other text."
    status, check = check_evidence.check_case(
        _case(["cap"]), logic_pre, logic_post, provision
    )
    assert status == "present"
    [excerpt] = [s for s in check["strings"] if s["origin"] == "context_excerpt"]
    assert excerpt["counts"] and excerpt["matched"]
    added = logic_post.replace(
        "              excerpt: up to the maximum of $143.\n",
        "              excerpt: up to the maximum of $143.\n"
        "          - path: versions[0].formula\n"
        "            source:\n"
        "              corpus_citation_path: us/statute/7/2014\n"
        "              excerpt: adjusted annually for inflation\n",
    )
    status, check = check_evidence.check_case(
        _case(["cap"]), logic_pre, added, provision
    )
    assert status == "absent"
    context = [s for s in check["strings"] if s["origin"] == "context_excerpt"]
    assert context and not any(s["counts"] for s in context)


def test_evidence_unknown_for_metadata_only_and_untestable_cases():
    status, check = check_evidence.check_case(_case(["cap"]), None, None, None)
    assert (status, check["reason"]) == ("unknown", "metadata_only")
    status, check = check_evidence.check_case(
        _case(["no_such_rule"]), _PRE, _PRE.replace("summary: s", "summary: t"), "x"
    )
    assert (status, check["reason"]) == ("unknown", "nothing_to_test")
    assert check["rules_missing"] == ["no_such_rule"]


def test_evidence_quotes_match_whole_or_by_ordered_nearby_fragments():
    provision = check_evidence.normalize(
        "The combined income of the others with whom the individual resides "
        "(excluding the income of the individual and spouse) must not exceed 165%."
    )
    assert check_evidence.text_in(
        "the combined income of the others ... must not exceed 165%", provision
    )
    assert not check_evidence.text_in(
        "must not exceed 165% ... the combined income of the others", provision
    )
    far = provision + " " + "x" * 400 + " tail words here"
    assert not check_evidence.text_in("must not exceed 165% ... tail words here", far)
    case = _case(
        ["cap"],
        verifier_notes='The source says "the combined income of the others with '
        'whom the individual ... resides".',
    )
    status, check = check_evidence.check_case(
        case, _PRE, _PRE.replace("summary: s", "summary: t"), provision
    )
    assert status == "present"
    assert check["strings"][-1]["origin"] == "triage_quote"


def test_evidence_number_forms_do_not_match_inside_other_numbers():
    assert check_evidence.value_forms("0.15")[1:] == [
        "15 percent",
        "15 per cent",
        "15%",
    ]
    assert "250,000" in check_evidence.value_forms("250000")
    assert "November 14, 2025" in check_evidence.value_forms("2025-11-14")
    text = check_evidence.normalize("rates of 1.15 and 150 and 2,150")
    for form in check_evidence.value_forms("15"):
        assert not check_evidence._form_pattern(form).search(text)
    assert check_evidence.value_tokens("heading_9903_01_30 or 7") == set()
    assert check_evidence.value_tokens("9903.01.77 and 165(d) on 2026-01-01") == {
        "9903.01.77",
        "165(d)",
        "2026-01-01",
    }


def test_committed_evidence_fields_reproduce_and_obey_the_status_rules():
    """Invariants for every case: the committed fields equal a fresh run, and
    the status follows from the counts (present iff a post-side string
    matched; absent iff something was tested and none did; unknown iff
    nothing was tested or the case is metadata-only)."""

    for entry in _index()["cases"]:
        case_dir = CORPUS_DIR / "cases" / entry["id"]
        case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        status, check = check_evidence.check_case_dir(case_dir, case)
        assert (case["evidence_in_provision"], case["evidence_check"]) == (
            status,
            check,
        ), entry["id"]
        assert entry["evidence_in_provision"] == status
        tested = check["post_side_tested"] + check["pre_side_tested"]
        if status == "present":
            assert check["post_side_matched"] > 0
        elif status == "absent":
            assert tested > 0 and check["post_side_matched"] == 0
            assert (check["reason"] == "provision_supports_pre_fix") == (
                check["pre_side_matched"] > 0
            )
        else:
            assert check["reason"] in {"metadata_only", "nothing_to_test"}
            assert check["reason"] == "metadata_only" or tested == 0
        assert (check["reason"] == "metadata_only") == (not case["artifacts_shipped"])


def test_evidence_agreement_with_the_recorded_hand_calls():
    """The agreement figures the README reports, recomputed from
    triage/evidence_validation.json and the committed fields."""

    record = json.loads(
        (CORPUS_DIR / "triage" / "evidence_validation.json").read_text(encoding="utf-8")
    )
    assert record["method"] == check_evidence.METHOD
    agreement = {}
    for labelled in record["sets"]:
        hits = 0
        for case_id, call in labelled["labels"].items():
            case = json.loads(
                (CORPUS_DIR / "cases" / case_id / "case.json").read_text(
                    encoding="utf-8"
                )
            )
            hits += case["evidence_in_provision"] == call["label"]
        agreement[labelled["name"]] = (hits, len(labelled["labels"]))
    assert agreement == {
        "review_round2": (14, 15),
        "blind_round1": (19, 24),
        "blind_round2": (20, 24),
    }
