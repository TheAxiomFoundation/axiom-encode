#!/usr/bin/env python3
"""Generate the one-time frozen v1 source-attestation rollout inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE
    ).stdout


def build_inventory(workspace: Path) -> dict[str, object]:
    repositories: dict[str, list[dict[str, str]]] = {}
    for repo in sorted(workspace.glob("rulespec-*")):
        if not (repo / ".git").exists():
            continue
        names = (
            _git(
                repo,
                "ls-tree",
                "-r",
                "--name-only",
                "HEAD",
                ".axiom/encoding-manifests",
            )
            .decode()
            .splitlines()
        )
        entries: list[dict[str, str]] = []
        for path in names:
            if not path.endswith(".json"):
                continue
            content = _git(repo, "show", f"HEAD:{path}")
            try:
                payload = json.loads(content)
            except (UnicodeError, json.JSONDecodeError):
                continue
            if (
                payload.get("schema_version") != "axiom-encode/applied-rulespec/v1"
                or payload.get("backend") not in {"codex", "openai", "anthropic"}
                or "source_attestation" in payload
            ):
                continue
            entries.append(
                {"path": path, "sha256": hashlib.sha256(content).hexdigest()}
            )
        if entries:
            repositories[repo.name] = entries
    return {
        "captured_at": "2026-07-10T00:00:00Z",
        "description": (
            "Frozen path and sha256 allowlist for signed v1 manifests present "
            "when resolver source attestations rolled out. Entries are removed, "
            "never added, as manifests migrate to v2."
        ),
        "repositories": repositories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_inventory(args.workspace.resolve())
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
