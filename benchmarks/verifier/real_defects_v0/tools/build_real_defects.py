#!/usr/bin/env python3
"""Build the real-defects corpus from the triage record.

Input: ``triage/triage_merged.json`` (one row per kept module case, produced
by ``tools/merge_triage.py`` from the workflow output). Output: ``cases/<id>/``
directories with ``case.json``, ``pre_fix.yaml``, ``post_fix.yaml`` and
``provision.txt``; ``index.json``; and ``triage/build_log.json`` listing every
row that was dropped and why.

Everything the build derives comes from Git (the rulespec checkouts and the
axiom-corpus checkout) and from signed corpus release objects fetched from the
public registry. Nothing is hand-written.

Usage (from the axiom-encode checkout)::

    uv run python benchmarks/verifier/real_defects_v0/tools/build_real_defects.py \
        --corpus-dir benchmarks/verifier/real_defects_v0 \
        --rulespec-us ../rulespec-us --rulespec-uk ../rulespec-uk \
        --axiom-corpus ../axiom-corpus --release-cache ~/.cache/axiom-real-defects
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[4]
_SPEC = importlib.util.spec_from_file_location(
    "verify_real_defects", ROOT / "scripts" / "verify_real_defects.py"
)
assert _SPEC is not None and _SPEC.loader is not None
lib = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lib)

MAIN_REF = "origin/main"
FIRST_PARENT_PROBE = 400


def slugify(path: str) -> str:
    stem = re.sub(r"\.ya?ml$", "", path)
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:80]


def list_registry_releases(prefix: str, cache_dir: Path) -> list[dict[str, Any]]:
    """Return every registry release for one jurisdiction prefix, oldest first."""

    listing = cache_dir / "registry_releases.json"
    if listing.exists():
        rows = json.loads(listing.read_text(encoding="utf-8"))
    else:
        url_base, key = lib.registry_credentials()
        url = (
            f"{url_base.rstrip('/')}/rest/v1/release_objects"
            "?select=release_name,content_sha256,created_at&order=release_name.asc&limit=1000"
        )
        request = urllib.request.Request(
            url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Accept-Profile": "corpus",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            rows = json.loads(response.read())
        cache_dir.mkdir(parents=True, exist_ok=True)
        listing.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    releases = []
    for row in rows:
        name = row["release_name"]
        if not name.startswith(prefix + "-"):
            continue
        payload = lib.fetch_release_object(name, row["content_sha256"], cache_dir)
        git_meta = payload["content"]["git"]
        releases.append(
            {
                "name": name,
                "content_sha256": row["content_sha256"],
                "commit": git_meta["commit"],
                "committed_at": git_meta.get("committed_at", ""),
                "created_at": row.get("created_at", ""),
                "payload": payload,
            }
        )
    releases.sort(key=lambda item: (item["committed_at"], item["created_at"]))
    return releases


def toolchain_at(repo: Path, commit: str) -> dict[str, Any]:
    raw = lib.git_blob(repo, commit, ".axiom/toolchain.toml")
    if raw is None:
        return {}
    try:
        return tomllib.loads(raw.decode("utf-8")).get("toolchain", {})
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return {}


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    probe = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant]
    )
    return probe.returncode == 0


def select_releases(
    jurisdiction: str,
    repo: Path,
    corpus_repo: Path,
    commit: str,
    releases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Order the candidate releases for one fix commit.

    The fix commit's toolchain decides: a signed release pin wins; otherwise
    the latest release whose corpus commit is at or before the toolchain's
    ``axiom_corpus_ref``; otherwise the earliest release after it.
    """

    toolchain = toolchain_at(repo, commit)
    pin = toolchain.get("axiom_corpus_release")
    ref = toolchain.get("axiom_corpus_ref")
    if pin:
        first = [r for r in releases if r["name"] == pin]
        rest = [
            r
            for r in releases
            if r["name"] != pin
            and (not first or r["committed_at"] >= first[0]["committed_at"])
        ]
        return first + rest, "toolchain_release_pin", ref
    if ref:
        before = [
            r
            for r in releases
            if r["commit"] == ref or is_ancestor(corpus_repo, r["commit"], ref)
        ]
        after = [r for r in releases if r not in before]
        if before:
            return (
                list(reversed(before)) + after,
                "latest_release_at_or_before_toolchain_corpus_ref",
                ref,
            )
        return after, "earliest_release_after_toolchain_corpus_ref", ref
    return list(releases), "no_toolchain_corpus_pin", None


def fix_time_corpus_match(
    corpus_repo: Path, corpus_ref: str | None, resolution: dict[str, Any]
) -> str:
    """Does the same provision row at the toolchain corpus ref carry the same body?"""

    if not corpus_ref:
        return "no_corpus_ref"
    if corpus_ref == resolution["corpus_commit"]:
        return "same_commit"
    raw = lib.git_blob(corpus_repo, corpus_ref, resolution["provision_file"])
    if raw is None:
        return "provision_file_absent_at_corpus_ref"
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("citation_path") != resolution["resolved_citation_path"]:
            continue
        body = record.get("body")
        if not isinstance(body, str) or not body.strip():
            return "row_body_empty_at_corpus_ref"
        if lib.sha256_text(body) == resolution["stored_body_sha256"]:
            return "same_body"
        return "different_body"
    return "row_absent_at_corpus_ref"


def citation_paths(module_text: str) -> list[str]:
    try:
        doc = yaml.safe_load(module_text)
    except yaml.YAMLError:
        return []
    if not isinstance(doc, dict):
        return []
    module = doc.get("module") or {}
    verification = module.get("source_verification") or {}
    paths: list[str] = []
    single = verification.get("corpus_citation_path")
    if isinstance(single, str):
        paths.append(single)
    plural = verification.get("corpus_citation_paths")
    if isinstance(plural, list):
        paths.extend(p for p in plural if isinstance(p, str))
    return paths


def hunk_ranges(repo: Path, parent: str, commit: str, path: str) -> dict[str, list]:
    diff = lib.git(repo, "diff", "--no-color", "-U0", parent, commit, "--", path)
    pre: list[list[int]] = []
    post: list[list[int]] = []
    for match in re.finditer(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", diff, flags=re.M
    ):
        a, b, c, d = match.groups()
        b = int(b) if b is not None else 1
        d = int(d) if d is not None else 1
        if b:
            pre.append([int(a), int(a) + b - 1])
        if d:
            post.append([int(c), int(c) + d - 1])
    return {"pre_fix_lines": pre, "post_fix_lines": post}


def fix_stage(repo: Path, parent: str, path: str) -> str:
    """post_merge when the pre-fix blob was ever on the first-parent history of main."""

    try:
        pre_blob = lib.git(repo, "rev-parse", f"{parent}:{path}").strip()
    except subprocess.CalledProcessError:
        return "unknown"
    history = lib.git(
        repo,
        "log",
        "--first-parent",
        f"-n{FIRST_PARENT_PROBE}",
        "--format=%H",
        MAIN_REF,
        "--",
        path,
    ).split()
    for candidate in history:
        try:
            blob = lib.git(repo, "rev-parse", f"{candidate}:{path}").strip()
        except subprocess.CalledProcessError:
            continue
        if blob == pre_blob:
            return "post_merge"
    return "pre_merge_review"


ARTIFACT_FILES = ("pre_fix.yaml", "post_fix.yaml", "provision.txt")


def apply_shipping_policy(cases_dir: Path, policy: str) -> dict[str, int]:
    """Decide which cases keep pre_fix.yaml, post_fix.yaml and provision.txt on disk.

    ``all`` keeps every case's files. ``verified`` (the default) keeps them for
    every case a verifier read directly and removes them for cases whose
    verdict was inherited from a sibling (the generator-family members), which
    stay as metadata-only records reproducible from Git and the release
    object. ``case.json`` records the outcome under ``artifacts_shipped``.
    """

    shipped = removed = 0
    for case_path in sorted(cases_dir.glob("*/case.json")):
        case = json.loads(case_path.read_text(encoding="utf-8"))
        inherited = (
            case["triage"].get("verifier_inferred_from_module_index") is not None
        )
        keep = policy == "all" or not inherited
        case["artifacts_shipped"] = keep
        if not keep:
            for name in ARTIFACT_FILES:
                target = case_path.parent / name
                if target.exists():
                    target.unlink()
            removed += 1
        else:
            shipped += 1
        case_path.write_text(
            json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return {"artifacts_shipped": shipped, "metadata_only": removed}


def assign_families(cases: list[dict[str, Any]]) -> None:
    """Group cases that carry one correction applied to many modules.

    A family is one commit, one defect kind, and one rule path (the generator
    regenerations touch 100 chapter compositions with the same hunk). The
    representative is the family's first module path in sorted order.
    """

    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        key = f"{case['jurisdiction']}-{case['commit'][:8]}-{case['defect_kind']}-{case['locator'].get('rule_path') or ''}"
        groups.setdefault(key, []).append(case)
    for key, members in groups.items():
        members.sort(key=lambda c: c["module_path"])
        digest = lib.sha256_text(key)[:12]
        family_id = f"{members[0]['jurisdiction']}-{members[0]['commit'][:8]}-{digest}"
        for position, case in enumerate(members):
            case["family_id"] = family_id
            case["family_size"] = len(members)
            case["family_representative"] = position == 0


def build_case(
    row: dict[str, Any],
    *,
    repos: dict[str, Path],
    corpus_repo: Path,
    releases_by_jur: dict[str, list[dict[str, Any]]],
    roots_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    jur = row["jurisdiction"]
    repo = repos[jur]
    if row.get("defect_kind") not in lib.DEFECT_KINDS:
        return None, None, "defect_kind_missing"
    commit = lib.git(repo, "rev-parse", row["commit"]).strip()
    parents = lib.git(repo, "rev-list", "--parents", "-n", "1", commit).split()[1:]
    if not parents:
        return None, None, "root_commit"
    parent = parents[0]
    if not is_ancestor(repo, commit, MAIN_REF):
        return None, None, "commit_not_on_main"
    path = row["module_path"]
    pre = lib.git_blob(repo, parent, path)
    post = lib.git_blob(repo, commit, path)
    if pre is None:
        return None, None, "module_added_in_fix_commit"
    if post is None:
        return None, None, "module_deleted_in_fix_commit"
    if pre == post:
        return None, None, "module_unchanged"
    pre_text = pre.decode("utf-8", errors="replace")
    post_text = post.decode("utf-8", errors="replace")
    if not post_text.startswith("format: rulespec/v1") or not pre_text.startswith(
        "format: rulespec/v1"
    ):
        return None, None, "not_rulespec_v1_on_both_sides"
    citations = citation_paths(post_text) or citation_paths(pre_text)
    if not citations:
        return None, None, "no_corpus_citation_path"
    ordered, basis, corpus_ref = select_releases(
        jur, repo, corpus_repo, commit, releases_by_jur[jur]
    )
    resolution = None
    chosen = None
    attempts = []
    for index, release in enumerate(ordered):
        try:
            resolution = lib.resolve_provision(
                release["payload"], corpus_repo, roots_dir, citations[0]
            )
        except Exception as exc:  # noqa: BLE001
            attempts.append({"release": release["name"], "error": str(exc)[:200]})
            continue
        chosen = dict(release)
        chosen["fallback_index"] = index
        break
    if resolution is None or chosen is None:
        return None, {"attempts": attempts[:5]}, "provision_unresolved"
    provision_text = resolution.pop("text")
    resolution["fix_time_corpus_match"] = fix_time_corpus_match(
        corpus_repo, corpus_ref, resolution
    )
    resolution["selection_basis"] = basis
    resolution["fallback_index"] = chosen["fallback_index"]
    resolution["toolchain_corpus_ref"] = corpus_ref
    resolution["requested_citation_path"] = citations[0]
    resolution["release_errors_before_success"] = attempts
    locator = hunk_ranges(repo, parent, commit, path)
    locator["rule_names"] = row.get("rule_names") or []
    locator["rule_path"] = row.get("rule_path") or ""
    verifier = row.get("verifier") or {}
    case = {
        "id": None,
        "jurisdiction": jur,
        "repo": lib.REPO_SLUGS[jur],
        "commit": commit,
        "parent_commit": parent,
        "commit_date": row.get("date"),
        "commit_subject": row.get("subject"),
        "pr_url": row.get("pr_url"),
        "module_path": path,
        "corpus_citation_path": citations[0],
        "corpus_citation_paths_all": citations,
        "corpus_release": chosen["name"],
        "corpus_release_content_sha256": chosen["content_sha256"],
        "corpus_commit": resolution["corpus_commit"],
        "defect_kind": row["defect_kind"],
        "other_kind": row.get("other_kind") or None,
        "confidence": round(float(row["confidence"]), 3),
        "description": row.get("description") or "",
        "description_source": row.get("description_source") or "none",
        "locator": locator,
        "pre_fix_artifact_sha256": lib.sha256_bytes(pre),
        "post_fix_artifact_sha256": lib.sha256_bytes(post),
        "provision_sha256": lib.sha256_text(provision_text),
        "provision_chars": len(provision_text),
        "provision_resolution": resolution,
        "fix_stage": fix_stage(repo, parent, path),
        "triage_status": row.get("triage_status"),
        "triage": {
            "classification": row.get("classification"),
            "pre_fix_wrong_because": row.get("pre_fix_wrong_because") or "",
            "triage_notes": row.get("triage_notes") or "",
            "verifier_verdict": verifier.get("verdict"),
            "verifier_defect_kind": verifier.get("defect_kind"),
            "verifier_confidence": verifier.get("confidence"),
            "verifier_justification": verifier.get("justification") or "",
            "verifier_quote": verifier.get("quote") or "",
            "verifier_notes": verifier.get("notes") or "",
            "verifier_inferred_from_module_index": verifier.get(
                "inferred_from_module_index"
            ),
            "candidate_source": row.get("candidate_source"),
        },
        "triage_notes": row.get("triage_notes_combined")
        or row.get("triage_notes")
        or "",
    }
    files = {
        "pre_fix.yaml": pre,
        "post_fix.yaml": post,
        "provision.txt": provision_text.encode("utf-8"),
    }
    return case, files, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--rulespec-us", type=Path, required=True)
    parser.add_argument("--rulespec-uk", type=Path, required=True)
    parser.add_argument("--axiom-corpus", type=Path, required=True)
    parser.add_argument("--release-cache", type=Path, required=True)
    parser.add_argument("--roots-dir", type=Path)
    parser.add_argument(
        "--ship-artifacts",
        choices=["verified", "all"],
        default="verified",
        help="which cases keep pre_fix/post_fix/provision files on disk",
    )
    parser.add_argument(
        "--apply-shipping-policy-only",
        action="store_true",
        help="skip the build; re-apply --ship-artifacts to the existing cases and index",
    )
    args = parser.parse_args(argv)
    corpus_dir = args.corpus_dir
    if args.apply_shipping_policy_only:
        shipping = apply_shipping_policy(corpus_dir / "cases", args.ship_artifacts)
        index_path = corpus_dir / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["counts"].update(shipping)
        index["shipping_policy"] = args.ship_artifacts
        by_id = {
            p.parent.name: json.loads(p.read_text(encoding="utf-8"))
            for p in (corpus_dir / "cases").glob("*/case.json")
        }
        for entry in index["cases"]:
            entry["artifacts_shipped"] = by_id[entry["id"]]["artifacts_shipped"]
        index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(shipping))
        return 0
    repos = {"us": args.rulespec_us, "uk": args.rulespec_uk}
    roots_dir = args.roots_dir or Path(tempfile.mkdtemp(prefix="real-defects-roots-"))
    rows = json.loads(
        (corpus_dir / "triage" / "triage_merged.json").read_text(encoding="utf-8")
    )
    releases_by_jur = {
        jur: list_registry_releases(jur, args.release_cache) for jur in ("us", "uk")
    }
    kept_rows = [r for r in rows if r.get("keep")]
    kept_rows.sort(
        key=lambda r: (
            r["jurisdiction"],
            r.get("date") or "",
            r["commit"],
            r["module_path"],
        )
    )
    cases_dir = corpus_dir / "cases"
    if cases_dir.exists():
        shutil.rmtree(cases_dir)
    cases_dir.mkdir(parents=True)
    built: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    counters = {"us": 0, "uk": 0}
    for row in kept_rows:
        case, files, reason = build_case(
            row,
            repos=repos,
            corpus_repo=args.axiom_corpus,
            releases_by_jur=releases_by_jur,
            roots_dir=roots_dir,
        )
        if case is None or files is None:
            dropped.append(
                {
                    "jurisdiction": row["jurisdiction"],
                    "commit": row["commit"],
                    "module_path": row["module_path"],
                    "reason": reason,
                    "detail": files,
                }
            )
            print(
                f"drop {row['jurisdiction']} {row['commit'][:10]} {row['module_path']}: {reason}"
            )
            continue
        counters[case["jurisdiction"]] += 1
        case["id"] = (
            f"{case['jurisdiction']}-{counters[case['jurisdiction']]:03d}-"
            f"{case['commit'][:8]}-{slugify(case['module_path'])}"
        )
        case_dir = cases_dir / case["id"]
        case_dir.mkdir()
        for name, raw in files.items():
            (case_dir / name).write_bytes(raw)
        (case_dir / "case.json").write_text(
            json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        built.append(case)
        print(f"case {case['id']} kind={case['defect_kind']} conf={case['confidence']}")
    assign_families(built)
    for case in built:
        case_path = cases_dir / case["id"] / "case.json"
        case_path.write_text(
            json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    shipping = apply_shipping_policy(cases_dir, args.ship_artifacts)
    for case in built:
        case["artifacts_shipped"] = json.loads(
            (cases_dir / case["id"] / "case.json").read_text(encoding="utf-8")
        )["artifacts_shipped"]
    families = {c["family_id"] for c in built}
    index = {
        "schema_version": "real_defects/v0",
        "shipping_policy": args.ship_artifacts,
        "generated_from": {
            "triage_record": "triage/triage_merged.json",
            "rulespec_main_ref": MAIN_REF,
        },
        "counts": {
            "cases": len(built),
            "by_jurisdiction": {
                jur: sum(1 for c in built if c["jurisdiction"] == jur)
                for jur in ("us", "uk")
            },
            "by_kind": {
                kind: sum(1 for c in built if c["defect_kind"] == kind)
                for kind in lib.DEFECT_KINDS
            },
            "by_triage_status": {
                status: sum(1 for c in built if c["triage_status"] == status)
                for status in sorted({c["triage_status"] for c in built})
            },
            "by_fix_stage": {
                stage: sum(1 for c in built if c["fix_stage"] == stage)
                for stage in sorted({c["fix_stage"] for c in built})
            },
            "dropped_at_build": len(dropped),
            **shipping,
            "families": len(families),
            "family_representatives": sum(
                1 for c in built if c["family_representative"]
            ),
            "by_kind_representatives": {
                kind: sum(
                    1
                    for c in built
                    if c["defect_kind"] == kind and c["family_representative"]
                )
                for kind in lib.DEFECT_KINDS
            },
        },
        "cases": [
            {
                "id": c["id"],
                "jurisdiction": c["jurisdiction"],
                "commit": c["commit"],
                "module_path": c["module_path"],
                "corpus_citation_path": c["corpus_citation_path"],
                "corpus_release": c["corpus_release"],
                "defect_kind": c["defect_kind"],
                "confidence": c["confidence"],
                "triage_status": c["triage_status"],
                "fix_stage": c["fix_stage"],
                "provision_chars": c["provision_chars"],
                "pr_url": c["pr_url"],
                "family_id": c["family_id"],
                "family_size": c["family_size"],
                "family_representative": c["family_representative"],
                "artifacts_shipped": c["artifacts_shipped"],
            }
            for c in built
        ],
    }
    (corpus_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (corpus_dir / "triage" / "build_log.json").write_text(
        json.dumps({"dropped": dropped, "built": len(built)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"built {len(built)} cases, dropped {len(dropped)}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
