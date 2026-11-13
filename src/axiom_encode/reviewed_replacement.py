"""Read-only inventory for same-path reviewed replacements; never authorizes signing.

This is preparation for replacement admission, not a replacement for the
protected predecessor verifier. All bytes come from explicit Git commits, not
the working tree. In particular, an ownership manifest hash is NOT proof that
the manifest is valid or owns the candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

_SHA = re.compile(r"[0-9a-f]{40}")
_ROOTS = {"legislation", "policies", "regulations", "statutes"}
_MAX_BLOB = 10 * 1024 * 1024


def _git(repo: Path, *args: str) -> bytes:
    from .signing_broker import scrub_private_signing_environment

    environment = {
        key: value
        for key, value in scrub_private_signing_environment().items()
        if not key.startswith("GIT_")
    }
    environment.update(
        GIT_GRAFT_FILE=os.devnull,
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_NO_LAZY_FETCH="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="core.commitGraph",
        GIT_CONFIG_VALUE_0="false",
        GIT_TERMINAL_PROMPT="0",
        GIT_LITERAL_PATHSPECS="1",
        GIT_OPTIONAL_LOCKS="0",
    )
    executable = shutil.which("git", path=environment.get("PATH") or None)
    if executable is None:
        raise ValueError("replacement inventory requires Git")
    return subprocess.run(
        [executable, "-C", str(repo), *args],
        check=True,
        capture_output=True,
        timeout=60,
        env=environment,
    ).stdout


def _blob(repo: Path, ref: str, path: str) -> bytes:
    entry = _git(repo, "ls-tree", "-z", ref, "--", path).split(b"\0")
    if len(entry) != 2 or not entry[0]:
        raise ValueError(f"expected one tracked regular file: {path}")
    metadata, actual = entry[0].split(b"\t", 1)
    mode, kind, oid = metadata.split()
    if mode != b"100644" or kind != b"blob" or actual.decode() != path:
        raise ValueError(f"expected a non-executable regular file: {path}")
    if int(_git(repo, "cat-file", "-s", oid.decode())) > _MAX_BLOB:
        raise ValueError(f"file exceeds inventory limit: {path}")
    return _git(repo, "cat-file", "blob", oid.decode())


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def inventory(repo: Path, base: str, candidate: str) -> dict[str, object]:
    """Inventory only modified, same-path primary/companion pairs and their owners.

    Refuse additions/deletions, manifest edits, and unrelated file changes. A
    candidate must preserve historical ownership evidence verbatim. Path-moving
    migrations and new modules need a different explicit contract.
    """
    repo = Path(_git(repo, "rev-parse", "--show-toplevel").decode().strip())
    for ref in (base, candidate):
        if _SHA.fullmatch(ref) is None:
            raise ValueError("base and candidate must be full lowercase commit SHAs")
        if _git(repo, "cat-file", "-t", ref).strip() != b"commit":
            raise ValueError("base and candidate must identify commits")
    _git(repo, "merge-base", "--is-ancestor", base, candidate)
    raw = _git(
        repo,
        "diff",
        "--no-renames",
        "--ignore-submodules=none",
        "--no-relative",
        "--name-status",
        "-z",
        base,
        candidate,
    )
    fields = raw.split(b"\0")[:-1]
    changed: set[str] = set()
    for index in range(0, len(fields), 2):
        status, encoded = fields[index : index + 2]
        path = encoded.decode()
        parts = PurePosixPath(path).parts
        if (
            status != b"M"
            or len(parts) < 3
            or re.fullmatch(r"[a-z]{2}(?:-[a-z0-9]+)?", parts[0]) is None
            or parts[1] not in _ROOTS
            or not path.endswith(".yaml")
            or any(ch.isspace() or ch == "\\" for ch in path)
        ):
            raise ValueError(
                f"unsupported replacement change: {status.decode()} {path}"
            )
        changed.add(path)
    if not changed:
        raise ValueError("candidate changes no RuleSpec files")
    primaries = {
        p.removesuffix(".test.yaml") + ".yaml" if p.endswith(".test.yaml") else p
        for p in changed
    }
    entries = []
    for primary in sorted(primaries):
        companion = primary.removesuffix(".yaml") + ".test.yaml"
        parts = PurePosixPath(primary).parts
        tail = str(PurePosixPath(*parts[1:])).removesuffix(".yaml") + ".json"
        owner_paths = (
            f".axiom/encoding-manifests/{parts[0]}/{tail}",
            f"{parts[0]}/.axiom/encoding-manifests/{tail}",
            f".axiom/encoding-manifests/{tail}",
        )
        owners = []
        for owner in owner_paths:
            if not _git(repo, "ls-tree", "-z", base, "--", owner):
                continue
            old = _blob(repo, base, owner)
            if _blob(repo, candidate, owner) != old:
                raise ValueError(f"candidate changed predecessor evidence: {owner}")
            owners.append({"path": owner, "sha256": _digest(old)})
        if len(owners) != 1:
            raise ValueError(f"expected exactly one predecessor manifest for {primary}")
        entries.append(
            {
                "rulespec_path": primary,
                "companion_path": companion,
                "base_rulespec_sha256": _digest(_blob(repo, base, primary)),
                "base_companion_sha256": _digest(_blob(repo, base, companion)),
                "candidate_rulespec_sha256": _digest(_blob(repo, candidate, primary)),
                "candidate_companion_sha256": _digest(
                    _blob(repo, candidate, companion)
                ),
                "predecessor_manifest": owners[0],
            }
        )
    return {
        "schema": "axiom-encode/reviewed-replacement-inventory/v1",
        "authorizes_signing": False,
        "base_sha": base,
        "candidate_sha": candidate,
        "replacements": entries,
        "unchecked": [
            "generation provenance and attempt-budget admission",
            "predecessor signature, ownership, and replacement receipt authority",
            "source completeness and atomic dependent closure",
            "independent review, toolchain compatibility, and exact-head CI",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    print(json.dumps(inventory(args.repo, args.base, args.candidate), indent=2))


if __name__ == "__main__":
    main()
