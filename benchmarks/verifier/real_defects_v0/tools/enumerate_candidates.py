#!/usr/bin/env python3
"""Enumerate module-touching commits on a rulespec repository's main branch.

Writes ``<out>/<jur>_commits.json`` (every commit that changed a rule module,
with its module and companion-test files), ``<jur>_keyword.json`` (commits
whose subject matches the correction keyword list) and
``<jur>_nonkeyword.json`` (the rest, subject only, for the screening pass).

usage: enumerate_candidates.py <us|uk> <repo-path> <out-dir> [--ref origin/main]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

MODULE_ROOT = re.compile(r"(us|uk)(-[a-z0-9-]+)?|statutes|regulations|policies")
KEYWORDS = re.compile(
    r"fix|correct|wrong|repair|bug|finding|time-bound|polarity|address|re-?anchor|"
    r"re-?verdict|round|closeout|taper|boundar|mis(s|t)|revert|amend|patch|resolve|"
    r"cross-review|review",
    re.I,
)


def is_module(path: str) -> bool:
    if not re.search(r"\.ya?ml$", path) or "/" not in path:
        return False
    top = path.split("/")[0]
    if MODULE_ROOT.fullmatch(top) is None:
        return False
    if "/.axiom/" in path or "/tests/" in path or "/data/" in path:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("jurisdiction", choices=["us", "uk"])
    parser.add_argument("repo", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--ref", default="origin/main")
    args = parser.parse_args()
    raw = subprocess.run(
        [
            "git",
            "-C",
            str(args.repo),
            "log",
            args.ref,
            "--name-only",
            "--format=::%H|%P|%ad|%an|%s",
            "--date=short",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    commits: list[dict] = []
    current: dict | None = None
    for line in raw.splitlines():
        if line.startswith("::"):
            sha, parents, date, author, subject = line[2:].split("|", 4)
            current = {
                "commit": sha,
                "parents": parents.split(),
                "date": date,
                "author": author,
                "subject": subject,
                "files": [],
            }
            commits.append(current)
        elif line.strip() and current is not None:
            current["files"].append(line.strip())
    rows = []
    for commit in commits:
        modules = [f for f in commit["files"] if is_module(f)]
        if not modules:
            continue
        tests = [f for f in modules if re.search(r"\.test\.ya?ml$", f)]
        rows.append(
            {
                **commit,
                "jurisdiction": args.jurisdiction,
                "module_files": [f for f in modules if f not in tests],
                "test_files": tests,
            }
        )
    rows = [r for r in rows if r["module_files"]]
    keyword = [r for r in rows if KEYWORDS.search(r["subject"])]
    nonkeyword = [
        {
            "commit": r["commit"][:10],
            "date": r["date"],
            "subject": r["subject"],
            "n_modules": len(r["module_files"]),
        }
        for r in rows
        if not KEYWORDS.search(r["subject"])
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    jur = args.jurisdiction
    (args.out / f"{jur}_commits.json").write_text(json.dumps(rows, indent=1))
    (args.out / f"{jur}_keyword.json").write_text(json.dumps(keyword, indent=1))
    (args.out / f"{jur}_nonkeyword.json").write_text(json.dumps(nonkeyword, indent=1))
    print(
        f"{jur}: module commits {len(rows)}, keyword {len(keyword)}, "
        f"non-keyword {len(nonkeyword)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
