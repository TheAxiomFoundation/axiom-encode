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

import copy
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CORPUS_DIR = ROOT / "benchmarks" / "verifier" / "real_defects_v0"
TOOLS_DIR = CORPUS_DIR / "tools"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify_real_defects = _load_module(
    "verify_real_defects", ROOT / "scripts" / "verify_real_defects.py"
)
build_real_defects = _load_module(
    "build_real_defects", TOOLS_DIR / "build_real_defects.py"
)
merge_triage = _load_module("merge_triage", TOOLS_DIR / "merge_triage.py")
provision_review = _load_module("provision_review", TOOLS_DIR / "provision_review.py")


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
    """Every case has exactly the documented keys, in the documented order,
    and the README's schema table names each of them."""

    readme = (CORPUS_DIR / "README.md").read_text(encoding="utf-8")
    assert set(verify_real_defects.REQUIRED_CASE_KEYS) == set(
        build_real_defects.CASE_KEY_ORDER
    )
    for key in build_real_defects.CASE_KEY_ORDER:
        assert f"`{key}`" in readme, f"README does not document {key}"
    for entry in _index()["cases"]:
        case = json.loads(
            (CORPUS_DIR / "cases" / entry["id"] / "case.json").read_text(
                encoding="utf-8"
            )
        )
        assert list(case) == list(build_real_defects.CASE_KEY_ORDER), entry["id"]
        assert len(case["commit"]) == 40 and len(case["parent_commit"]) == 40
        assert case["fix_stage"] in verify_real_defects.FIX_STAGES
        assert case["triage_status"] in verify_real_defects.TRIAGE_STATUSES
        for component in verify_real_defects.provision_components(case):
            resolution = component["resolution"]
            assert resolution["mode"] in {"axiom_encode_resolver", "direct_row_exact"}
            assert resolution["provision_file"].startswith("data/corpus/provisions/")
            assert resolution["corpus_commit"] == case["corpus_commit"]
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


# --------------------------------------------------------------------------
# Extended provisions and review quotes (scripts/verify_real_defects.py)
# --------------------------------------------------------------------------

_TEXT_ALPHABET = ["a", "B", "ß", " ", "\n", "\t", "—", "’", "“", "x", "é", ";"]
_TEXT_ALPHABET += ["--- Source: q ---", "\n\n", "$143"]


def _random_text(rng: random.Random, low: int = 0, high: int = 30) -> str:
    return "".join(rng.choice(_TEXT_ALPHABET) for _ in range(rng.randrange(low, high)))


def test_compose_and_split_provision_round_trip_for_any_components():
    """Property: split_provision inverts compose_provision for every list of
    component texts, including texts that contain header-like lines; one
    component composes to itself; a wrong length or path does not split."""

    rng = random.Random(885)
    for trial in range(4000):
        components = [
            (f"us/statute/{index}", _random_text(rng))
            for index in range(rng.randrange(1, 5))
        ]
        text = verify_real_defects.compose_provision(components)
        sized = [(path, len(part)) for path, part in components]
        assert verify_real_defects.split_provision(text, sized) == [
            part for _path, part in components
        ], trial
        if len(components) == 1:
            assert text == components[0][1]
            continue
        grown = [(sized[0][0], sized[0][1] + 1), *sized[1:]]
        assert verify_real_defects.split_provision(text, grown) is None, trial
        renamed = [("uk/other", sized[0][1]), *sized[1:]]
        assert verify_real_defects.split_provision(text, renamed) is None, trial
        regions = verify_real_defects.component_regions(sized)
        for path, part in components:
            start, end = regions[path]
            assert text[start:end] == part, trial
    for bad in (-1, 1.0, True, "1", None):
        assert (
            verify_real_defects.split_provision(
                "--- Source: a ---\n\n--- Source: b ---\nZ", [("a", bad), ("b", 1)]
            )
            is None
        )


def test_locate_quote_finds_any_substring_and_reports_a_covering_span():
    """Property: any substring of a text is located, whatever its whitespace
    or case, and the span returned covers a passage that folds to it."""

    rng = random.Random(1659)
    fold = verify_real_defects._folded
    for trial in range(4000):
        text = _random_text(rng, 3, 60)
        start = rng.randrange(0, len(text) - 1)
        quote = text[start : rng.randrange(start + 1, len(text) + 1)]
        if not quote.strip():
            continue
        varied = " ".join(quote.split()).swapcase() if trial % 2 else quote
        span = verify_real_defects.locate_quote(varied, text)
        assert span is not None, (trial, quote, text)
        assert 0 <= span[0] < span[1] <= len(text)
        assert fold(varied)[0] in fold(text[span[0] : span[1]])[0], (trial, quote)
        # Whatever a span covers, the quote matches it exactly from end to end.
        covered = text[span[0] : span[1]]
        assert verify_real_defects.locate_quote(varied, covered) == (0, len(covered))
    assert verify_real_defects.locate_quote(
        "the  Maximum of $143", "up to the\nmaximum of $143."
    ) == (6, 25)
    assert verify_real_defects.locate_quote("a b ... e f", "a b c d e f") == (0, 11)
    assert verify_real_defects.locate_quote("e f ... a b", "a b c d e f") is None
    far = "a b" + " x" * 200 + " e f"
    assert verify_real_defects.locate_quote("a b ... e f", far) is None
    assert verify_real_defects.locate_quote("...", "a b") is None
    # A match never starts or ends inside one character's case-folded
    # expansion ("ß" folds to "ss").
    assert verify_real_defects.locate_quote("the tax as", "the tax aß") is None
    assert verify_real_defects.locate_quote("s", "aß") is None
    assert verify_real_defects.locate_quote("the tax ass", "the tax aß") == (0, 10)
    assert verify_real_defects.locate_quote("SS", "aß") == (1, 2)
    assert verify_real_defects.locate_quote("ss ... b", "ß x b") == (0, 5)
    assert verify_real_defects.locate_quote("‘quoted’ – text", "'quoted' - text") == (
        0,
        15,
    )


def test_scrub_local_paths_is_idempotent_and_leaves_no_local_path():
    scratch = (
        "/private/tmp/claude-501/-Users-someone-TheAxiomFoundation-axiom-encode"
        "--claude-worktrees-zealous-heyrovsky-31a96b/"
        "86103d8f-b97f-4559-a96c-7e79b0000000/scratchpad"
    )
    samples = {
        f"cat {scratch}/diffs/us/198bee703e/012.diff.": "cat <scratch>/diffs/us/198bee703e/012.diff.",
        f"{scratch}/inputs/us/a9562cfd44.json": "<scratch>/inputs/us/a9562cfd44.json",
        "git -C /Users/someone/TheAxiomFoundation/rulespec-us show": "git -C rulespec-us show",
        "/Users/someone/TheAxiomFoundation/axiom-encode/.claude/worktrees/zealous-x-1/src/a.py": "axiom-encode/src/a.py",
        "under ~/TheAxiomFoundation/rulespec-uk and ~/TheAxiomFoundation": "under rulespec-uk and <checkouts>",
        "/tmp/real-defects-enum and us/statute/26/24": "/tmp/real-defects-enum and us/statute/26/24",
    }
    for raw, expected in samples.items():
        once = verify_real_defects.scrub_local_paths(raw)
        assert once == expected
        assert verify_real_defects.scrub_local_paths(once) == once
        assert verify_real_defects.LOCAL_PATH.search(once) is None
    nested = {"a": [next(iter(samples))], "b": 3, "c": None}
    assert verify_real_defects.scrub_json(nested) == {
        "a": [samples[next(iter(samples))]],
        "b": 3,
        "c": None,
    }


def test_the_corpus_carries_no_local_absolute_path():
    """No file under the corpus directory names a path on the machine that
    built it (a home directory, a session scratchpad)."""

    offenders = []
    for path in sorted(CORPUS_DIR.rglob("*")):
        if not path.is_file() or path.suffix == ".pyc":
            continue
        match = verify_real_defects.LOCAL_PATH.search(
            path.read_text(encoding="utf-8", errors="replace")
        )
        if match:
            offenders.append((str(path.relative_to(CORPUS_DIR)), match.group(0)))
    assert offenders == []


def _reviewed_case(provision: str, **review) -> dict:
    return {
        "corpus_citation_path": "us/statute/1",
        "provision_sha256": verify_real_defects.sha256_text(provision),
        "provision_chars": len(provision),
        "provision_resolution": {},
        "provision_extension": None,
        "judgeable_from_provision": review.get("verdict") != "not_in_sources",
        "provision_review": {
            "verdict": "in_provision",
            "outcome": "kept",
            "decisive_quotes": [],
            "missing_basis": "",
            **review,
        },
    }


def test_provision_record_problems_names_each_inconsistency():
    provision = "Subtract the deduction up to the maximum of $143."
    span = list(verify_real_defects.locate_quote("maximum of $143", provision))
    good = _reviewed_case(
        provision,
        decisive_quotes=[
            {"citation_path": "us/statute/1", "quote": "maximum of $143", "span": span}
        ],
    )
    problems = verify_real_defects.provision_record_problems
    assert problems(good, provision) == []
    assert problems(good, None) == []

    moved = copy.deepcopy(good)
    moved["provision_review"]["decisive_quotes"][0]["span"] = [0, 5]
    assert any("not at its span" in p for p in problems(moved, provision))

    unquoted = _reviewed_case(provision)
    assert any("needs a decisive quote" in p for p in problems(unquoted, provision))

    flipped = copy.deepcopy(good)
    flipped["judgeable_from_provision"] = False
    assert any("does not follow" in p for p in problems(flipped, provision))

    excluded = _reviewed_case(
        provision, verdict="not_in_sources", outcome="not_judgeable"
    )
    assert any("what the fix rests on" in p for p in problems(excluded, provision))
    excluded["provision_review"]["missing_basis"] = "an uncited notice"
    assert problems(excluded, provision) == []
    quoted = copy.deepcopy(excluded)
    quoted["provision_review"]["decisive_quotes"] = good["provision_review"][
        "decisive_quotes"
    ]
    assert any("has decisive quotes" in p for p in problems(quoted, provision))
    other = copy.deepcopy(good)
    other["provision_review"]["decisive_quotes"][0]["citation_path"] = "us/x"
    assert any("not in the provision" in p for p in problems(other, provision))

    unreviewed = _reviewed_case(provision)
    unreviewed["provision_review"] = None
    assert any("without a review" in p for p in problems(unreviewed, provision))
    unreviewed["judgeable_from_provision"] = None
    assert problems(unreviewed, provision) == []

    # An extended provision: the decisive quote must sit in the added text.
    added = "The maximum is $198.99 from October 1, 2025."
    text = verify_real_defects.compose_provision(
        [("us/statute/1", provision), ("us/guidance/2", added)]
    )
    extended = _reviewed_case(
        text,
        verdict="in_other_citation",
        outcome="provision_extended",
        decisive_quotes=[
            {
                "citation_path": "us/guidance/2",
                "quote": "maximum is $198.99",
                "span": list(
                    verify_real_defects.locate_quote("maximum is $198.99", text)
                ),
            }
        ],
    )
    extended["provision_extension"] = {
        "composition": verify_real_defects.PROVISION_COMPOSITION,
        "first_citation_text_sha256": verify_real_defects.sha256_text(provision),
        "first_citation_chars": len(provision),
        "added": [
            {
                "citation_path": "us/guidance/2",
                "text_sha256": verify_real_defects.sha256_text(added),
                "chars": len(added),
                "resolution": {},
            }
        ],
    }
    assert problems(extended, text) == []
    in_first = copy.deepcopy(extended)
    in_first["provision_review"]["decisive_quotes"] = [
        {
            "citation_path": "us/statute/1",
            "quote": "maximum of $143",
            "span": list(verify_real_defects.locate_quote("maximum of $143", text)),
        }
    ]
    assert any("added citation" in p for p in problems(in_first, text))
    # A quote must sit inside the citation it names.
    misnamed = copy.deepcopy(extended)
    misnamed["provision_review"]["decisive_quotes"][0]["citation_path"] = "us/statute/1"
    assert any("lies outside" in p for p in problems(misnamed, text))
    stranger = copy.deepcopy(extended)
    stranger["provision_review"]["decisive_quotes"][0]["citation_path"] = (
        "us/statute/26/999999"
    )
    assert any("not in the provision" in p for p in problems(stranger, text))
    tampered = text.replace("$198.99", "$199.99")
    assert any("recorded digest" in p for p in problems(extended, tampered))
    assert any(
        "not its recorded composition" in p for p in problems(extended, provision)
    )
    bare = copy.deepcopy(good)
    bare["provision_extension"] = extended["provision_extension"]
    assert any("disagree" in p for p in problems(bare, text))


# --------------------------------------------------------------------------
# Ids, families and filters (tools/build_real_defects.py)
# --------------------------------------------------------------------------


def _family_case(
    module_path, kind, *, status="fidelity", inferred=None, commit="a" * 40
):
    return {
        "jurisdiction": "us",
        "commit": commit,
        "module_path": module_path,
        "defect_kind": kind,
        "other_kind": None,
        "triage_status": status,
        "locator": {"rule_path": "rules[r].versions[0].formula"},
        "triage": {"verifier_inferred_from_module_index": inferred},
    }


def test_families_key_on_commit_and_rule_path_and_settle_one_kind():
    cases = [
        _family_case("us/ch03.yaml", "wrong_entity_or_scope"),
        _family_case("us/ch01.yaml", "untraceable_branch", status="unclear"),
        _family_case("us/ch02.yaml", "untraceable_branch"),
        _family_case("us/ch04.yaml", "wrong_entity_or_scope", inferred=0),
        _family_case("us/ch05.yaml", "wrong_entity_or_scope", inferred=0),
        _family_case("us/other.yaml", "amount_mismatch", commit="b" * 40),
    ]
    build_real_defects.assign_families(cases)
    family = [c for c in cases if c["commit"].startswith("a")]
    assert len({c["family_id"] for c in family}) == 1
    assert {c["family_size"] for c in family} == {5}
    # Directly verified members vote: two untraceable_branch against one
    # wrong_entity_or_scope; the two inherited verdicts do not vote.
    assert {c["defect_kind"] for c in family} == {"untraceable_branch"}
    changed = [c for c in family if c["triage"]["kind_before_family_settlement"]]
    assert sorted(c["module_path"] for c in changed) == [
        "us/ch03.yaml",
        "us/ch04.yaml",
        "us/ch05.yaml",
    ]
    assert all(
        c["triage"]["family_kind_votes"]
        == {"untraceable_branch": 2, "wrong_entity_or_scope": 1}
        for c in family
    )
    # The representative is the first fidelity member, not the unclear ch01.
    assert [c["module_path"] for c in family if c["family_representative"]] == [
        "us/ch02.yaml"
    ]
    single = cases[-1]
    assert single["family_size"] == 1 and single["family_representative"]
    assert single["triage"]["kind_before_family_settlement"] is None
    assert single["triage"]["family_kind_votes"] is None
    # Settling again changes nothing (the original label is not lost).
    snapshot = copy.deepcopy(cases)
    build_real_defects.assign_families(cases)
    assert cases == snapshot


def test_family_invariants_hold_for_any_grouping():
    """Property: every family has exactly one representative and one kind,
    its members share commit and rule path, and family_size is its count."""

    rng = random.Random(4)
    kinds = ["untraceable_branch", "wrong_entity_or_scope", "amount_mismatch"]
    for _trial in range(300):
        cases = []
        for index in range(rng.randrange(1, 25)):
            case = _family_case(
                f"us/m{index:02d}.yaml",
                rng.choice(kinds),
                status=rng.choice(["fidelity", "fidelity", "unclear"]),
                inferred=rng.choice([None, None, 0]),
                commit=rng.choice(["a", "b", "c"]) * 40,
            )
            case["locator"]["rule_path"] = rng.choice(["rules[x]", "rules[y]"])
            cases.append(case)
        build_real_defects.assign_families(cases)
        families: dict[str, list[dict]] = {}
        for case in cases:
            families.setdefault(case["family_id"], []).append(case)
        keys = set()
        for members in families.values():
            assert sum(c["family_representative"] for c in members) == 1
            assert len({c["defect_kind"] for c in members}) == 1
            assert {c["family_size"] for c in members} == {len(members)}
            key = {(c["commit"], c["locator"]["rule_path"]) for c in members}
            assert len(key) == 1
            keys |= key
            representative = next(c for c in members if c["family_representative"])
            if any(c["triage_status"] == "fidelity" for c in members):
                assert representative["triage_status"] == "fidelity"
        assert len(keys) == len(families)


def test_a_kept_representative_ships_its_files_whenever_a_member_can():
    """Property: under the default shipping policy, pruning a family to its
    representative keeps a case that ships files whenever any fidelity
    member was read directly by a verifier."""

    rng = random.Random(7)
    for _trial in range(300):
        cases = []
        for index in range(rng.randrange(1, 12)):
            case = _family_case(
                f"us/m{index:02d}.yaml",
                "untraceable_branch",
                status=rng.choice(["fidelity", "unclear"]),
                inferred=rng.choice([None, 0, 0]),
            )
            case["id"] = f"us-{index + 1:03d}-x"
            cases.append(case)
        build_real_defects.assign_families(cases)
        built = [{"case": case, "files": {"x": b""}} for case in cases]
        kept, _removed = build_real_defects.filter_cases(
            built, drop_unclear=True, family_members="representatives"
        )
        build_real_defects.apply_shipping_policy(kept, "verified")
        can_ship = any(
            c["triage_status"] == "fidelity"
            and c["triage"]["verifier_inferred_from_module_index"] is None
            for c in cases
        )
        if can_ship:
            assert [item["case"]["artifacts_shipped"] for item in kept] == [True]


def test_check_evidence_write_keeps_the_documented_key_order():
    case = _cases()[0]
    status, check = check_evidence.check_case_dir(
        CORPUS_DIR / "cases" / case["id"], case
    )
    assert list(check_evidence.with_evidence(case, status, check)) == list(
        build_real_defects.CASE_KEY_ORDER
    )
    shuffled = {
        k: v
        for k, v in case.items()
        if k not in {"evidence_in_provision", "evidence_check"}
    }
    assert list(check_evidence.with_evidence(shuffled, status, check)) == list(
        build_real_defects.CASE_KEY_ORDER
    )


def test_case_ids_do_not_depend_on_order_or_on_what_a_build_filters():
    """Property: a row's id is fixed once assigned. Building the rows in
    another order, or building a subset, gives every row the same id, and a
    new row takes the next free number of its jurisdiction."""

    rng = random.Random(9)
    for _trial in range(200):
        rows = [
            {
                "jurisdiction": rng.choice(["us", "uk"]),
                "commit": f"{rng.randrange(16**10):010x}" + "0" * 30,
                "module_path": f"us/statutes/{index}.yaml",
            }
            for index in range(rng.randrange(1, 30))
        ]
        registry = build_real_defects.CaseIds([])
        first = {r["module_path"]: registry.id_for(r) for r in rows}
        assert len(set(first.values())) == len(rows)
        for jurisdiction in ("us", "uk"):
            seqs = sorted(
                int(i.split("-")[1])
                for i in first.values()
                if i.startswith(jurisdiction)
            )
            assert seqs == list(range(1, len(seqs) + 1))
        again = build_real_defects.CaseIds(copy.deepcopy(registry.entries))
        subset = rng.sample(rows, rng.randrange(0, len(rows) + 1))
        assert {r["module_path"]: again.id_for(r) for r in subset} == {
            r["module_path"]: first[r["module_path"]] for r in subset
        }
        newcomer = {
            "jurisdiction": "us",
            "commit": "f" * 40,
            "module_path": "us/new.yaml",
        }
        expected = 1 + sum(1 for r in rows if r["jurisdiction"] == "us")
        assert again.id_for(newcomer).startswith(f"us-{expected:03d}-ffffffff-us-new")
    with pytest.raises(ValueError, match="twice"):
        build_real_defects.CaseIds([*registry.entries, registry.entries[0]])


def test_filter_cases_logs_every_removed_case_with_its_id():
    cases = [
        _family_case("us/ch01.yaml", "untraceable_branch"),
        _family_case("us/ch02.yaml", "untraceable_branch"),
        _family_case("us/ch03.yaml", "untraceable_branch", status="unclear"),
        _family_case(
            "us/solo.yaml", "amount_mismatch", status="unclear", commit="b" * 40
        ),
    ]
    for number, case in enumerate(cases, start=1):
        case["id"] = f"us-{number:03d}-x"
    build_real_defects.assign_families(cases)
    built = [{"case": case, "files": {}} for case in cases]
    kept, removed = build_real_defects.filter_cases(
        built, drop_unclear=False, family_members="all"
    )
    assert (len(kept), removed) == (4, [])
    kept, removed = build_real_defects.filter_cases(
        built, drop_unclear=True, family_members="representatives"
    )
    assert [item["case"]["id"] for item in kept] == ["us-001-x"]
    assert {row["id"]: row["reason"] for row in removed} == {
        "us-002-x": "family_non_representative",
        "us-003-x": "triage_unclear",
        "us-004-x": "triage_unclear",
    }
    # The representative keeps the size of the family it stands for.
    assert kept[0]["case"]["family_size"] == 3


def test_committed_cases_removed_rows_and_id_registry_account_for_each_other():
    """Conservation: every id the registry has issued is a case in the corpus
    or a row the build log says a flag removed, never both, never neither."""

    index = _index()
    log = json.loads((CORPUS_DIR / "triage" / "build_log.json").read_text("utf-8"))
    registry = json.loads((CORPUS_DIR / "triage" / "case_ids.json").read_text("utf-8"))
    issued = [entry["id"] for entry in registry["ids"]]
    kept = [entry["id"] for entry in index["cases"]]
    removed = [row["id"] for row in log["removed"]]
    assert len(set(issued)) == len(issued)
    assert set(kept) | set(removed) == set(issued)
    assert set(kept) & set(removed) == set()
    assert len(removed) == len(set(removed))
    counts = index["counts"]
    assert counts["cases"] == len(kept) == log["kept"]
    assert counts["built_before_filters"] == len(issued) == log["built"]
    assert sum(counts["removed_by_flag"].values()) == len(removed)
    assert counts["dropped_at_build"] == len(log["dropped"])
    assert log["flags"] == index["build_flags"]
    if index["build_flags"]["drop_unclear"]:
        assert counts["by_triage_status"]["unclear"] == 0
    if index["build_flags"]["family_members"] == "representatives":
        assert all(entry["family_representative"] for entry in index["cases"])
        assert counts["families"] == counts["cases"]
    for entry in registry["ids"]:
        expected = (
            f"{entry['jurisdiction']}-{entry['seq']:03d}-{entry['commit'][:8]}-"
            f"{build_real_defects.slugify(entry['module_path'])}"
        )
        assert entry["id"] == expected


def test_index_and_readme_counts_are_regenerated_from_the_cases():
    """The committed index equals build_index over the committed cases, and
    the README's generated block equals render_counts over the committed
    index and logs."""

    index = _index()
    log = json.loads((CORPUS_DIR / "triage" / "build_log.json").read_text("utf-8"))
    summary = json.loads((CORPUS_DIR / "triage" / "summary.json").read_text("utf-8"))
    rebuilt = build_real_defects.build_index(
        sorted(_cases(), key=lambda case: case["id"]),
        shipping_policy=index["shipping_policy"],
        flags=index["build_flags"],
        dropped_at_build=len(log["dropped"]),
        removed=log["removed"],
    )
    assert rebuilt == index
    readme = (CORPUS_DIR / "README.md").read_text(encoding="utf-8")
    block = build_real_defects.render_counts(index, summary, log)
    assert block in readme
    assert build_real_defects.splice_counts(readme, block) == readme


# --------------------------------------------------------------------------
# The provision review (decision d885)
# --------------------------------------------------------------------------

# (agreeing, compared) per set of earlier hand calls; see the README.
# Fields apply_provision_review copies from the settled record unchanged.
COPIED_REVIEW_KEYS = (
    "verdict",
    "defect_real",
    "nearest_quotes",
    "why",
    "missing_basis",
    "pre_fix_restates_it",
    "basis",
    "adjudication_note",
    "readers",
)
REVIEW_DEFAULTS = {"adjudication_note": ""}
HAND_CALL_AGREEMENT = {
    "review_round2": (14, 15),
    "blind_round1": (21, 24),
    "blind_round2": (22, 24),
}


def test_every_board_eligible_case_carries_its_provision_review():
    """Each shipped fidelity representative was checked against its provision,
    the case record equals the review record it was built from, and the
    outcome follows from the verdict."""

    record = json.loads(
        (CORPUS_DIR / "triage" / "provision_review.json").read_text(encoding="utf-8")
    )
    reviewed = record["cases"]
    eligible = [c for c in _cases() if build_real_defects.is_board_eligible(c)]
    assert eligible
    assert {c["id"] for c in eligible} <= set(reviewed)
    assert set(reviewed) <= {c["id"] for c in _cases()}
    for case in _cases():
        settled = reviewed.get(case["id"])
        review = case["provision_review"]
        if settled is None:
            assert review is None and case["judgeable_from_provision"] is None
            continue
        assert review["method"] == record["method"]
        assert review["verdict"] == settled["verdict"]
        assert (
            review["outcome"]
            == (verify_real_defects.PROVISION_REVIEW_OUTCOMES[settled["verdict"]])
        )
        # Every field copied from the settled record is equal; span (and the
        # derived fields below) are the build's own.
        assert [
            {k: v for k, v in q.items() if k != "span"}
            for q in review["decisive_quotes"]
        ] == settled["decisive_quotes"], case["id"]
        for key in COPIED_REVIEW_KEYS:
            assert review[key] == settled.get(key, REVIEW_DEFAULTS.get(key)), (
                case["id"],
                key,
            )
        assert review["nearest_quotes"] == settled["nearest_quotes"]
        if settled["verdict"] == "not_in_sources":
            assert review["decisive_quotes"] == []
        else:
            assert review["decisive_quotes"] and review["nearest_quotes"] == []
        assert review["readers"] == settled["readers"]
        assert len(settled["readers"]) >= 1
        assert case["judgeable_from_provision"] is (
            settled["verdict"] != "not_in_sources"
        )
        added = [
            q["citation_path"]
            for q in settled["decisive_quotes"]
            if q["citation_path"] != case["corpus_citation_path"]
        ]
        extension = case["provision_extension"]
        if settled["verdict"] == "in_other_citation":
            assert [a["citation_path"] for a in extension["added"]] == list(
                dict.fromkeys(added)
            )
        else:
            assert extension is None
    assert record["counts"] == _index()["counts"]["by_provision_review_verdict"]
    assert sum(record["counts"].values()) == len(reviewed)


def test_a_case_said_to_restate_the_passage_has_a_quoting_proof_atom():
    """pre_fix_restates_it means the pre-fix module quotes the passage in a
    proof atom (``excerpt``, or ``text`` or ``span`` in older modules), so a
    case flagged true must have at least one such atom."""

    import yaml

    flagged = 0
    for case in _cases():
        review = case["provision_review"]
        if review is None or not review["pre_fix_restates_it"]:
            continue
        flagged += 1
        module = yaml.safe_load(
            (CORPUS_DIR / "cases" / case["id"] / "pre_fix.yaml").read_text("utf-8")
        )
        quoting = [
            atom
            for rule in module.get("rules") or []
            for atom in ((rule.get("metadata") or {}).get("proof") or {}).get("atoms")
            or []
            if isinstance(atom, dict)
            and any(
                isinstance((atom.get("source") or {}).get(key), str)
                and (atom.get("source") or {})[key].strip()
                for key in ("excerpt", "text", "span")
            )
        ]
        assert quoting, case["id"]
    assert flagged


def _hand_call_agreement() -> dict[str, tuple[int, int]]:
    """Per recorded set of earlier hand calls: (agreeing, compared).

    A hand call of ``present`` (the provision carries the evidence) agrees
    with a review verdict of ``in_provision``; ``absent`` agrees with the
    other two verdicts.
    """

    record = json.loads(
        (CORPUS_DIR / "triage" / "evidence_validation.json").read_text(encoding="utf-8")
    )
    verdicts = {
        case["id"]: case["provision_review"]["verdict"]
        for case in _cases()
        if case["provision_review"] is not None
    }
    agreement = {}
    for labelled in record["sets"]:
        calls = [
            (call["label"] == "present") == (verdicts[case_id] == "in_provision")
            for case_id, call in labelled["labels"].items()
            if case_id in verdicts
        ]
        agreement[labelled["name"]] = (sum(calls), len(calls))
    return agreement


def test_review_agreement_with_the_earlier_hand_calls():
    """The agreement the README reports between the provision review and the
    63 hand calls recorded before it (triage/evidence_validation.json)."""

    agreement = _hand_call_agreement()
    assert agreement == HAND_CALL_AGREEMENT
    readme = (CORPUS_DIR / "README.md").read_text(encoding="utf-8")
    agree = sum(hits for hits, _total in agreement.values())
    total = sum(total for _hits, total in agreement.values())
    assert f"agrees with {agree} of the {total}" in " ".join(readme.split())


def test_the_loader_default_set_is_the_judgeable_cases_the_window_shows():
    """Differential: the benchmark loader's default selection equals what the
    corpus index says is board-eligible, judgeable and visible in the
    judges' default window."""

    verifier_root = ROOT / "benchmarks" / "verifier"
    sys.path.insert(0, str(verifier_root))
    try:
        from encodebench_verifier.sources import real as real_source
    finally:
        sys.path.remove(str(verifier_root))
    from axiom_encode.judges.client import DEFAULT_PROVISION_CHARS, truncate_provision

    index = _index()
    expected = sorted(
        entry["id"]
        for entry in index["cases"]
        if entry["family_representative"]
        and entry["triage_status"] == "fidelity"
        and entry["artifacts_shipped"]
        and entry["judgeable_from_provision"] is True
        and entry["decisive_text_in_default_window"]
    )
    suite, report = real_source.build_real_suite(
        CORPUS_DIR, provision_chars=DEFAULT_PROVISION_CHARS, truncate=truncate_provision
    )
    assert sorted(suite.source_identity["case_ids"]) == expected
    counts = index["counts"]
    assert len(expected) == counts["judgeable_with_decisive_text_in_default_window"]
    assert report["skipped"].get("not_judgeable_from_provision", 0) == sum(
        1
        for entry in index["cases"]
        if entry["family_representative"]
        and entry["triage_status"] == "fidelity"
        and entry["judgeable_from_provision"] is False
    )
    everything, _ = real_source.build_real_suite(
        CORPUS_DIR,
        provision_chars=DEFAULT_PROVISION_CHARS,
        truncate=truncate_provision,
        judgeable_only=False,
    )
    assert (
        len(everything.source_identity["case_ids"])
        == counts["fidelity_representatives"]
    )


class _StubResolver:
    """Stands in for the release resolver: citation path to text."""

    def __init__(self, texts):
        self.texts = texts
        self.asked = []

    def resolve(self, case, citation):
        self.asked.append(citation)
        text = self.texts[citation]
        return {
            "mode": "axiom_encode_resolver",
            "resolved_citation_path": citation,
            "corpus_commit": case["corpus_commit"],
            "text": text,
        }


def _reviewable_item(provision: str) -> dict:
    module = _PRE
    case = {
        "id": "us-001-test",
        "corpus_citation_path": "us/regulation/7/273/10",
        "corpus_commit": "c" * 40,
        "provision_sha256": verify_real_defects.sha256_text(provision),
        "provision_chars": len(provision),
        "provision_resolution": {"corpus_commit": "c" * 40},
        "artifacts_shipped": True,
        "locator": {"rule_names": ["cap"]},
        "triage": {
            "pre_fix_wrong_because": "",
            "triage_notes": "",
            "verifier_justification": "",
            "verifier_notes": "",
        },
    }
    files = {
        "pre_fix.yaml": module.encode(),
        "post_fix.yaml": module.replace("formula: '143'", "formula: '198.99'").encode(),
        "provision.txt": provision.encode(),
    }
    return {"case": case, "files": files}


def test_apply_provision_review_extends_resets_and_agrees_with_the_verifier():
    """Differential: what the build writes for each verdict is what the
    verify script accepts, and re-applying a different record starts again
    from the first citation's text."""

    meta = {"method": "provision_review_v1", "reviewed_on": "2026-10-10"}
    first = "Subtract the deduction up to the maximum of $143. The maximum is adjusted each year."
    memo = "Header. The maximum is adjusted each year. For fiscal year 2026 the maximum is $198.99."
    resolver = _StubResolver({"us/guidance/cola-2026": memo})
    item = _reviewable_item(first)
    problems = verify_real_defects.provision_record_problems
    apply = build_real_defects.apply_provision_review

    # in_other_citation: the quote also occurs in the first citation, and is
    # still located in the citation the reader quoted it from.
    record = {
        "verdict": "in_other_citation",
        "defect_real": "yes",
        "decisive_quotes": [
            {
                "citation_path": "us/guidance/cola-2026",
                "quote": "the maximum is adjusted each year",
            },
            {
                "citation_path": "us/guidance/cola-2026",
                "quote": "the maximum is $198.99",
            },
            {
                "citation_path": "us/regulation/7/273/10",
                "quote": "up to the maximum of $143",
            },
        ],
        "nearest_quotes": [],
        "basis": "readers_agree",
        "readers": {"r1": {}, "r2": {}},
    }
    apply(item, record, meta, resolver)
    case = item["case"]
    text = item["files"]["provision.txt"].decode()
    assert resolver.asked == ["us/guidance/cola-2026"]
    assert text == verify_real_defects.compose_provision(
        [("us/regulation/7/273/10", first), ("us/guidance/cola-2026", memo)]
    )
    assert case["provision_sha256"] == verify_real_defects.sha256_text(text)
    assert case["provision_chars"] == len(text)
    extension = case["provision_extension"]
    assert extension["first_citation_chars"] == len(first)
    assert [a["citation_path"] for a in extension["added"]] == ["us/guidance/cola-2026"]
    assert "text" not in extension["added"][0]["resolution"]
    review = case["provision_review"]
    assert case["judgeable_from_provision"] is True
    assert review["outcome"] == "provision_extended"
    assert review["mechanical_before_review"] == "absent"
    assert review["default_window"] == {"chars": 24_000, "decisive_text_visible": True}
    added_from = text.index(memo)
    spans = [q["span"] for q in review["decisive_quotes"]]
    assert spans[0][0] >= added_from and spans[1][0] >= added_from
    assert spans[2][1] <= added_from
    for quote in review["decisive_quotes"]:
        start, end = quote["span"]
        assert text[start:end].casefold() == quote["quote"].casefold()
    assert problems(case, text) == []

    # A decisive passage the default window cuts away is reported as such.
    middle = "filler " * 3000 + first + " filler" * 3000
    long_item = _reviewable_item(middle)
    in_middle = {
        "verdict": "in_provision",
        "defect_real": "yes",
        "decisive_quotes": [
            {"citation_path": "us/regulation/7/273/10", "quote": "adjusted each year"}
        ],
        "nearest_quotes": [],
        "basis": "readers_agree",
        "readers": {},
    }
    apply(long_item, in_middle, meta, resolver)
    assert long_item["case"]["provision_review"]["default_window"] == {
        "chars": 24_000,
        "decisive_text_visible": False,
    }
    assert problems(long_item["case"], middle) == []

    # Re-applying another verdict resets the provision to the first citation.
    kept = {
        "verdict": "in_provision",
        "defect_real": "yes",
        "decisive_quotes": [
            {"citation_path": "us/regulation/7/273/10", "quote": "adjusted each year"}
        ],
        "nearest_quotes": [],
        "basis": "adjudicated",
        "readers": {},
    }
    apply(item, kept, meta, resolver)
    assert item["files"]["provision.txt"].decode() == first
    assert case["provision_extension"] is None
    assert case["provision_sha256"] == verify_real_defects.sha256_text(first)
    assert problems(case, first) == []

    excluded = {
        "verdict": "not_in_sources",
        "defect_real": "unsure",
        "decisive_quotes": [],
        "nearest_quotes": [
            {
                "citation_path": "us/regulation/7/273/10",
                "quote": "up to the maximum of $143",
            }
        ],
        "missing_basis": "the COLA memorandum, which neither module cites",
        "basis": "readers_agree",
        "readers": {},
    }
    apply(item, excluded, meta, resolver)
    assert case["judgeable_from_provision"] is False
    assert case["provision_review"]["decisive_quotes"] == []
    assert case["provision_review"]["default_window"] is None
    assert len(case["provision_review"]["nearest_quotes"]) == 1
    assert problems(case, first) == []

    apply(item, None, meta, resolver)
    assert case["provision_review"] is None and case["judgeable_from_provision"] is None
    assert problems(case, first) == []

    # A quote that is not in the citation it names stops the build.
    wrong = dict(
        kept,
        decisive_quotes=[
            {
                "citation_path": "us/regulation/7/273/10",
                "quote": "the maximum is $198.99",
            }
        ],
    )
    with pytest.raises(RuntimeError, match="decisive quote is not in"):
        apply(_reviewable_item(first), wrong, meta, resolver)
    silent = dict(record, decisive_quotes=[record["decisive_quotes"][2]])
    with pytest.raises(RuntimeError, match="names no other one"):
        apply(_reviewable_item(first), silent, meta, resolver)


def test_reader_records_are_checked_against_the_bundle_texts(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "other").mkdir(parents=True)
    (bundle / "provision.txt").write_text("Subtract up to the maximum of $143.")
    (bundle / "other" / "001.txt").write_text("The maximum is $198.99 for 2026.")
    manifest = {"others": [{"file": "other/001.txt", "citation": "us/guidance/x"}]}
    check = provision_review.check_reader_record

    def record(**fields):
        return {"verdict": "in_provision", "defect_real": "yes", **fields}

    quote = {"file": "provision.txt", "quote": "up to the maximum of $143"}
    assert check(record(decisive_quotes=[quote]), bundle, manifest) == []
    invented = {"file": "provision.txt", "quote": "up to the maximum of $198.99"}
    assert any(
        "not found" in p
        for p in check(record(decisive_quotes=[invented]), bundle, manifest)
    )
    other = {"file": "other/001.txt", "quote": "The maximum is $198.99"}
    assert any(
        "provision.txt only" in p
        for p in check(record(decisive_quotes=[other]), bundle, manifest)
    )
    assert (
        check(
            record(verdict="in_other_citation", decisive_quotes=[other]),
            bundle,
            manifest,
        )
        == []
    )
    assert any(
        "needs a quote from an other/ file" in p
        for p in check(
            record(verdict="in_other_citation", decisive_quotes=[quote]),
            bundle,
            manifest,
        )
    )
    assert any(
        "missing_basis" in p
        for p in check(
            record(verdict="not_in_sources", defect_real="unsure"), bundle, manifest
        )
    )
    assert (
        check(
            record(
                verdict="not_in_sources",
                defect_real="unsure",
                missing_basis="an uncited notice",
            ),
            bundle,
            manifest,
        )
        == []
    )
    assert any(
        "defect_real is not yes" in p
        for p in check(
            record(defect_real="unsure", decisive_quotes=[quote]), bundle, manifest
        )
    )
    unknown = {"file": "other/999.txt", "quote": "The maximum is $198.99"}
    assert any(
        "unknown file" in p
        for p in check(record(decisive_quotes=[unknown]), bundle, manifest)
    )


def test_settle_needs_agreeing_readers_or_an_adjudication():
    first = {
        "verdict": "in_provision",
        "defect_real": "yes",
        "decisive_quotes": [{"file": "provision.txt", "quote": "a b c"}],
        "why_decisive": "w",
    }
    second = dict(
        first, verdict="not_in_sources", decisive_quotes=[], missing_basis="m"
    )
    clean = {"r1": [], "r2": []}
    settle = provision_review.settle
    record, why = settle("c", {"r1": first, "r2": first}, clean, None)
    assert why is None and record["basis"] == "readers_agree"
    # One reader is not enough unless the caller says so.
    record, why = settle("c", {"r1": first}, {"r1": []}, None)
    assert record is None and "2 needed" in why
    record, why = settle("c", {"r1": first}, {"r1": []}, None, min_readers=1)
    assert record["basis"] == "single_reader"
    record, why = settle("c", {"r1": first, "r2": second}, clean, None)
    assert record is None and "disagree" in why
    # A record whose quote failed the check never settles a case by default.
    record, why = settle(
        "c", {"r1": first, "r2": first}, {"r1": ["quote not found"], "r2": []}, None
    )
    assert record is None and "does not check: r1" in why
    record, why = settle(
        "c", {"r1": first, "r2": second}, clean, dict(second, adjudication_note="x")
    )
    assert record["verdict"] == "not_in_sources" and record["basis"] == "adjudicated"
    one = dict(
        first,
        verdict="in_other_citation",
        decisive_quotes=[{"file": "other/001.txt", "quote": "a b c"}],
    )
    two = dict(
        first,
        verdict="in_other_citation",
        decisive_quotes=[{"file": "other/002.txt", "quote": "a b c"}],
    )
    record, why = settle("c", {"r1": one, "r2": two}, clean, None)
    assert record is None and "different other citations" in why
    record, why = settle("c", {"r1": one, "r2": one}, clean, None)
    assert record["verdict"] == "in_other_citation"


# --------------------------------------------------------------------------
# The triage record (tools/merge_triage.py)
# --------------------------------------------------------------------------


def _override_row(**fields):
    row = {
        "keep": True,
        "jurisdiction": "us",
        "commit": "095d680d47",
        "module_path": "statutes/26/24.yaml",
        "rule_path": "rules[a].versions[0].formula",
        "rule_names": [
            "a",
            "ch01_s232_steel_heading_rate",
            "section_232_steel_component_rate",
        ],
        "triage_status": "unclear",
        "confidence": 0.49,
        "triage_confidence": 0.55,
        "verifier": {"confidence": 0.62},
        "review_override": None,
    }
    row.update(fields)
    return row


def test_review_overrides_promote_and_trim_and_must_match():
    rows = [_override_row(), _override_row(module_path="statutes/26/1.yaml")]
    overrides = [
        {
            "id": "promote",
            "source": "s",
            "reason": "r",
            "match": {
                "jurisdiction": "us",
                "commit": "095d680d47",
                "module_path": "statutes/26/24.yaml",
            },
            "set_triage_status": "fidelity",
        },
        {
            "id": "trim",
            "source": "s",
            "reason": "r",
            "match": {
                "rule_path": "rules[a].versions[0].formula",
                "module_path": "statutes/26/1.yaml",
            },
            "drop_rule_names_matching": r"(ch\w+_)?s232_steel_heading_rate|section_232_steel_component_rate",
        },
    ]
    assert merge_triage.apply_review_overrides(rows, overrides) == ["promote", "trim"]
    promoted, trimmed = rows
    assert promoted["triage_status"] == "fidelity"
    # The lower of the two readers' confidences, without the unclear cap.
    assert promoted["confidence"] == 0.55
    assert promoted["review_override"]["changed"] == {
        "triage_status": ["unclear", "fidelity"],
        "confidence": [0.49, 0.55],
    }
    assert trimmed["rule_names"] == ["a"]
    assert trimmed["triage_status"] == "unclear"
    assert trimmed["review_override"]["id"] == "trim"
    stale = [dict(overrides[0], id="stale", match={"commit": "nope"})]
    with pytest.raises(ValueError, match="matches no kept row"):
        merge_triage.apply_review_overrides(rows, stale)
    with pytest.raises(ValueError, match="two review overrides"):
        merge_triage.apply_review_overrides(
            [_override_row()],
            [overrides[0], dict(overrides[1], match={"commit": "095d680d47"})],
        )


@pytest.mark.skipif(
    _sibling_checkout("AXIOM_REAL_DEFECTS_RULESPEC_US", "rulespec-us") is None
    or _sibling_checkout("AXIOM_REAL_DEFECTS_RULESPEC_UK", "rulespec-uk") is None,
    reason="rulespec checkouts are not available",
)
def test_the_triage_record_reproduces_from_the_workflow_output(tmp_path):
    """Differential: re-running the merge over the committed workflow output,
    PR URL record and review overrides writes the committed triage files."""

    triage = tmp_path / "corpus" / "triage"
    triage.mkdir(parents=True)
    for name in ("workflow_output.json", "pr_urls.json", "review_overrides.json"):
        shutil.copy(CORPUS_DIR / "triage" / name, triage / name)
    assert (
        merge_triage.main(
            [
                "--workflow-output",
                str(triage / "workflow_output.json"),
                "--corpus-dir",
                str(tmp_path / "corpus"),
                "--rulespec-us",
                str(_sibling_checkout("AXIOM_REAL_DEFECTS_RULESPEC_US", "rulespec-us")),
                "--rulespec-uk",
                str(_sibling_checkout("AXIOM_REAL_DEFECTS_RULESPEC_UK", "rulespec-uk")),
                "--scrub-record",
            ]
        )
        == 0
    )
    for name in (
        "workflow_output.json",
        "triage_merged.json",
        "screen.json",
        "summary.json",
        "pr_urls.json",
    ):
        assert (triage / name).read_bytes() == (
            CORPUS_DIR / "triage" / name
        ).read_bytes(), name
    rows = json.loads((triage / "triage_merged.json").read_text(encoding="utf-8"))
    kept = {
        (row["jurisdiction"], row["commit"], row["module_path"]): row
        for row in rows
        if row["keep"]
    }
    for case in _cases():
        row = kept[(case["jurisdiction"], case["commit"][:10], case["module_path"])]
        assert case["triage_status"] == row["triage_status"], case["id"]
        assert case["confidence"] == row["confidence"], case["id"]
        assert case["locator"]["rule_names"] == row["rule_names"], case["id"]
        assert case["pr_url"] == row["pr_url"], case["id"]
        assert case["triage"]["review_override"] == row["review_override"], case["id"]
