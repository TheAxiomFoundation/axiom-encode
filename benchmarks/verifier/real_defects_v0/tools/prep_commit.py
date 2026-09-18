#!/usr/bin/env python3
"""Write a read-only triage input bundle for one rulespec commit.

usage: prep_commit.py <us|uk> <sha> [--no-pr]   (scratch dir from AXIOM_REAL_DEFECTS_SCRATCH, repos from AXIOM_REAL_DEFECTS_RULESPEC_US/UK)
Writes <scratch>/inputs/<jur>/<sha>.json plus diffs/<jur>/<sha>/<n>.diff.
Never modifies the rulespec repositories; only `git show`/`git log` and `gh api`.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

S = Path(os.environ.get("AXIOM_REAL_DEFECTS_SCRATCH", Path(__file__).resolve().parent))
REPOS = {
    "us": os.environ.get(
        "AXIOM_REAL_DEFECTS_RULESPEC_US",
        "/Users/maxghenis/TheAxiomFoundation/rulespec-us",
    ),
    "uk": os.environ.get(
        "AXIOM_REAL_DEFECTS_RULESPEC_UK",
        "/Users/maxghenis/TheAxiomFoundation/rulespec-uk",
    ),
}
GH = {"us": "TheAxiomFoundation/rulespec-us", "uk": "TheAxiomFoundation/rulespec-uk"}


def git(repo, *args, text=True):
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=text, check=True
    ).stdout


def is_module(f):
    if not re.search(r"\.ya?ml$", f):
        return False
    if "/" not in f:
        return False
    top = f.split("/")[0]
    if not re.fullmatch(r"(us|uk)(-[a-z0-9-]+)?|statutes|regulations|policies", top):
        return False
    if "/.axiom/" in f or "/tests/" in f or "/data/" in f:
        return False
    if f.endswith(".test.yaml") or f.endswith(".test.yml"):
        return False
    return True


def main():
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    jur, sha = sys.argv[1], sys.argv[2]
    want_pr = "--no-pr" not in sys.argv
    repo = REPOS[jur]
    out_json = S / "inputs" / jur / f"{sha[:10]}.json"
    if out_json.exists():
        print(out_json)
        return
    full = git(repo, "rev-parse", sha).strip()
    meta = git(
        repo,
        "show",
        "-s",
        "--format=%H%x00%P%x00%ad%x00%an%x00%s%x00%b",
        "--date=short",
        full,
    ).split("\x00")
    parents = meta[1].split()
    parent = parents[0] if parents else None
    files = [
        f
        for f in git(repo, "show", "--name-only", "--format=", full).splitlines()
        if f.strip()
    ]
    modules = [f for f in files if is_module(f)]
    tests = [f for f in files if re.search(r"\.test\.ya?ml$", f)]
    others = [f for f in files if f not in modules and f not in tests]
    diff_dir = S / "diffs" / jur / sha[:10]
    diff_dir.mkdir(parents=True, exist_ok=True)
    mods = []
    for i, path in enumerate(modules):
        diff = git(repo, "show", "--format=", "--no-color", full, "--", path)
        dpath = diff_dir / f"{i:03d}.diff"
        dpath.write_text(diff)
        status = "modified"
        try:
            post = git(repo, "show", f"{full}:{path}")
        except subprocess.CalledProcessError:
            post = None
            status = "deleted"
        pre = None
        if parent:
            try:
                pre = git(repo, "show", f"{parent}:{path}")
            except subprocess.CalledProcessError:
                status = "added"
        head = (post or pre or "")[:1200]
        cit = re.findall(
            r"corpus_citation_paths?:\s*(?:\n\s*-\s*(\S+)|\s*(\S+))", post or pre or ""
        )
        cits = [a or b for a, b in cit][:6]
        mods.append(
            {
                "index": i,
                "path": path,
                "status": status,
                "diff_file": str(dpath),
                "diff_lines": diff.count("\n"),
                "pre_lines": pre.count("\n") if pre else 0,
                "post_lines": post.count("\n") if post else 0,
                "post_header": head,
                "corpus_citation_paths_seen": cits,
                "is_rulespec_v1": bool(post and post.startswith("format: rulespec/v1")),
            }
        )
    pr = None
    if want_pr:
        try:
            raw = subprocess.run(
                ["gh", "api", f"repos/{GH[jur]}/commits/{full}/pulls"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            prs = json.loads(raw)
            if prs:
                p = prs[0]
                n = p["number"]
                reviews = json.loads(
                    subprocess.run(
                        ["gh", "api", f"repos/{GH[jur]}/pulls/{n}/reviews"],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout
                )
                rcomments = json.loads(
                    subprocess.run(
                        [
                            "gh",
                            "api",
                            f"repos/{GH[jur]}/pulls/{n}/comments",
                            "--paginate",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout
                )
                icomments = json.loads(
                    subprocess.run(
                        [
                            "gh",
                            "api",
                            f"repos/{GH[jur]}/issues/{n}/comments",
                            "--paginate",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout
                )
                pr = {
                    "number": n,
                    "url": p["html_url"],
                    "title": p["title"],
                    "merged_at": p.get("merged_at"),
                    "body": (p.get("body") or "")[:6000],
                    "reviews": [
                        {
                            "user": r["user"]["login"],
                            "state": r["state"],
                            "body": (r.get("body") or "")[:3000],
                        }
                        for r in reviews
                        if (r.get("body") or "").strip()
                    ],
                    "review_comments": [
                        {
                            "user": c["user"]["login"],
                            "path": c.get("path"),
                            "body": (c.get("body") or "")[:3000],
                        }
                        for c in rcomments
                    ],
                    "issue_comments": [
                        {
                            "user": c["user"]["login"],
                            "body": (c.get("body") or "")[:3000],
                        }
                        for c in icomments
                    ],
                }
        except Exception as exc:  # noqa: BLE001
            pr = {"error": str(exc)[:300]}
    bundle = {
        "jurisdiction": jur,
        "repo": GH[jur],
        "repo_path": repo,
        "commit": full,
        "parent_commit": parent,
        "all_parents": parents,
        "date": meta[2],
        "author": meta[3],
        "subject": meta[4],
        "body": meta[5][:8000],
        "module_files": mods,
        "test_files": tests,
        "other_files": others[:60],
        "pr": pr,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(bundle, indent=1))
    print(out_json)


if __name__ == "__main__":
    main()
