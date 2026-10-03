#!/usr/bin/env python3
"""Apply the round-two review revisions to the committed corpus.

A full rebuild through ``tools/build_real_defects.py`` re-resolves every
provision against signed release objects from the public registry, and the
release cache the 2026-09-18 build used was not kept. The revisions the
2026-10-03 review of PR #1659 asked for change no module or provision bytes,
so this tool applies them to the committed cases instead, reading only the
committed files and the rulespec checkouts (``git log`` and
``git cat-file``; never a write).

Every stage is deterministic and idempotent: running the tool on its own
output changes nothing. Stages, in order:

``commit_metadata``
    ``commit_date`` and ``commit_subject`` become the commit's committer date
    (``%cI``) and subject (``%s``); ``triage.screen_reason`` carries the screen
    pass's one-line reason (null for keyword candidates). The triage record's
    ``date`` and ``subject`` (``triage/triage_merged.json`` and the flagged
    entries in ``triage/screen.json``) get the same values, and the merged
    rows are re-sorted the way ``tools/merge_triage.py`` sorts them.

Usage (from the axiom-encode checkout)::

    uv run python benchmarks/verifier/real_defects_v0/tools/revise_corpus.py \\
        --corpus-dir benchmarks/verifier/real_defects_v0 \\
        --rulespec-us ../rulespec-us --rulespec-uk ../rulespec-uk
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[3]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lib = _load("verify_real_defects", ROOT / "scripts" / "verify_real_defects.py")
merge = _load("merge_triage", TOOLS / "merge_triage.py")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any, indent: int) -> bool:
    """Write ``payload`` the way the build tools do; report whether bytes changed."""

    text = json.dumps(payload, indent=indent, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


class Corpus:
    """The committed corpus, loaded once and written back by ``save``."""

    def __init__(self, corpus_dir: Path, repos: dict[str, Path]) -> None:
        self.dir = corpus_dir
        self.repos = repos
        self.triage_dir = corpus_dir / "triage"
        self.rows: list[dict[str, Any]] = read_json(
            self.triage_dir / "triage_merged.json"
        )
        self.screen: dict[str, Any] = read_json(self.triage_dir / "screen.json")
        self.cases: dict[str, dict[str, Any]] = {
            path.parent.name: read_json(path)
            for path in sorted((corpus_dir / "cases").glob("*/case.json"))
        }
        self._metadata: dict[str, dict[str, dict[str, Any]]] = {}

    def commit_meta(self, jurisdiction: str, commit: str) -> dict[str, Any]:
        """``%cI``/``%s``/parents for a commit, read once per jurisdiction."""

        if jurisdiction not in self._metadata:
            commits = {
                r["commit"] for r in self.rows if r["jurisdiction"] == jurisdiction
            }
            commits |= {
                e["commit"]
                for e in self.screen.get("flagged") or []
                if e["jur"] == jurisdiction
            }
            commits |= {
                c["commit"]
                for c in self.cases.values()
                if c["jurisdiction"] == jurisdiction
            }
            self._metadata[jurisdiction] = lib.commit_metadata(
                self.repos[jurisdiction], commits
            )
        return self._metadata[jurisdiction][commit]

    def row_for(self, case: dict[str, Any]) -> dict[str, Any]:
        """The kept triage row a case was built from."""

        matches = [
            row
            for row in self.rows
            if row["keep"]
            and row["jurisdiction"] == case["jurisdiction"]
            and case["commit"].startswith(row["commit"])
            and row["module_path"] == case["module_path"]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"{case['id']}: {len(matches)} triage rows match")
        return matches[0]

    def save(self) -> list[str]:
        changed = []
        if write_json(self.triage_dir / "triage_merged.json", self.rows, indent=1):
            changed.append("triage/triage_merged.json")
        if write_json(self.triage_dir / "screen.json", self.screen, indent=1):
            changed.append("triage/screen.json")
        for case_id, case in self.cases.items():
            path = self.dir / "cases" / case_id / "case.json"
            if write_json(path, case, indent=2):
                changed.append(f"cases/{case_id}/case.json")
        return changed


def stage_commit_metadata(corpus: Corpus) -> None:
    for row in corpus.rows:
        meta = corpus.commit_meta(row["jurisdiction"], row["commit"])
        row["date"] = meta["date"]
        row["subject"] = meta["subject"]
    corpus.rows.sort(key=merge.row_sort_key)
    corpus.screen["flagged"] = [
        merge.screen_entry_from_git(
            entry, corpus.commit_meta(entry["jur"], entry["commit"])
        )
        for entry in corpus.screen.get("flagged") or []
    ]
    for case in corpus.cases.values():
        meta = corpus.commit_meta(case["jurisdiction"], case["commit"])
        case["commit_date"] = meta["date"]
        case["commit_subject"] = meta["subject"]
        case["triage"]["screen_reason"] = corpus.row_for(case).get("screen_reason")


STAGES = (("commit_metadata", stage_commit_metadata),)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--rulespec-us", type=Path, required=True)
    parser.add_argument("--rulespec-uk", type=Path, required=True)
    args = parser.parse_args(argv)
    corpus = Corpus(args.corpus_dir, {"us": args.rulespec_us, "uk": args.rulespec_uk})
    for name, stage in STAGES:
        stage(corpus)
        print(f"stage {name}: done")
    changed = corpus.save()
    print(f"{len(changed)} files changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
