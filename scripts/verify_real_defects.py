#!/usr/bin/env python3
"""Re-derive and verify the real-defects verifier corpus.

The corpus under ``benchmarks/verifier/real_defects_v0`` stores, per case, the
pre-fix and post-fix RuleSpec module bytes, the provision text the module
cites, and their sha256 digests. Every digest must reproduce from the named
Git commits and the named signed corpus release. This script checks that.

Tiers (each one is skipped, and reported as skipped, when its inputs are absent):

* ``shipped``  - the shipped files hash to what ``case.json`` says.
* ``git``      - the rulespec checkouts reproduce the pre-fix and post-fix
                 module bytes from ``parent_commit`` and ``commit``.
* ``corpus``   - the axiom-corpus checkout reproduces the provision row: the
                 provision file at the release's corpus commit hashes to the
                 release artifact digest and the row body hashes to the stored
                 body digest.

The ``git`` and ``corpus`` tiers read objects through one long-lived
``git cat-file --batch`` process per repository. Each distinct
``(corpus_commit, provision_file)`` pair is read once for all the cases that
cite it: the blob is hashed in bounded chunks and only the row lines those
cases name are kept, so memory stays at one chunk plus the longest line.
* ``release``  - the signed release object is fetched from the public registry
                 (or a local cache) and the provision text is re-resolved
                 through ``axiom_encode.corpus_resolver``; the resolved text
                 must hash to ``provision_sha256``.

Usage::

    uv run python scripts/verify_real_defects.py \
        --corpus-dir benchmarks/verifier/real_defects_v0 \
        --rulespec-us ../rulespec-us --rulespec-uk ../rulespec-uk \
        --axiom-corpus ../axiom-corpus --release-cache ~/.cache/axiom-real-defects

Registry access reads ``AXIOM_CORPUS_RELEASE_REGISTRY_URL`` and
``AXIOM_CORPUS_RELEASE_REGISTRY_ANON_KEY`` (the same public values the org
validate-rulespec workflow reads from repository variables). When they are
unset the script tries ``gh variable get`` on TheAxiomFoundation/rulespec-us.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# Published verification keys for signed corpus release objects. These are the
# org repository variables AXIOM_CORPUS_RELEASE_PUBLIC_KEY and
# AXIOM_CORPUS_RELEASE_RETIRED_PUBLIC_KEY as read on 2026-09-17; both are
# public verification roots, not signing material. Override with
# AXIOM_CORPUS_RELEASE_PUBLIC_KEYS (comma separated) if they rotate.
DEFAULT_RELEASE_PUBLIC_KEYS = (
    "KwIYEbbs/905yxn/9Yi6jYTF8oyZcq1FlxiG4e0y0tg=",
    "9EJzeLnZsdBoDEDkU8xbi7HjYVVvd4HDX/BxPHPvXTk=",
)
REGISTRY_URL_ENV = "AXIOM_CORPUS_RELEASE_REGISTRY_URL"
REGISTRY_KEY_ENV = "AXIOM_CORPUS_RELEASE_REGISTRY_ANON_KEY"
REGISTRY_VARIABLES_REPO = "TheAxiomFoundation/rulespec-us"
REPO_SLUGS = {
    "us": "TheAxiomFoundation/rulespec-us",
    "uk": "TheAxiomFoundation/rulespec-uk",
}
REQUIRED_CASE_KEYS = (
    "id",
    "jurisdiction",
    "repo",
    "commit",
    "parent_commit",
    "pr_url",
    "module_path",
    "corpus_citation_path",
    "corpus_release",
    "corpus_release_content_sha256",
    "corpus_commit",
    "defect_kind",
    "confidence",
    "description",
    "locator",
    "pre_fix_artifact_sha256",
    "post_fix_artifact_sha256",
    "provision_sha256",
    "provision_resolution",
    "fix_stage",
    "family_id",
    "family_size",
    "family_representative",
    "artifacts_shipped",
    "triage_notes",
)
# Fields that carry the answer a judge is scored against, or the triage
# readers' words about it. A benchmark must not show them to a judge; every
# other case.json field is checked to be free of defect-kind names and of its
# case's triage text (tests/test_real_defects_corpus.py).
LABEL_BEARING_KEYS = (
    "defect_kind",
    "other_kind",
    "confidence",
    "description",
    "description_source",
    "locator",
    "triage_status",
    "triage",
    "triage_notes",
    "evidence_in_provision",
    "evidence_check",
)
DEFECT_KINDS = (
    "amount_mismatch",
    "boundary_direction",
    "unrepresented_clause",
    "untraceable_branch",
    "wrong_period_or_effective_date",
    "wrong_entity_or_scope",
    "polarity_or_logic",
    "other",
)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def git(repo: Path | str, *args: str, binary: bool = False) -> Any:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=not binary,
        check=True,
    )
    return completed.stdout


def git_blob(repo: Path | str, commit: str, path: str) -> bytes | None:
    try:
        return git(repo, "show", f"{commit}:{path}", binary=True)
    except subprocess.CalledProcessError:
        return None


class GitObjectReader:
    """Stream blobs from one repository through ``git cat-file --batch``.

    One process serves every request, so a run costs one fork per repository
    instead of one per object, and a blob arrives in chunks of at most
    ``CHUNK_BYTES`` rather than as one buffer.
    """

    CHUNK_BYTES = 1 << 20

    def __init__(self, repo: Path | str) -> None:
        self.repo = Path(repo)
        self._process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> GitObjectReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        assert process.stdin is not None and process.stdout is not None
        process.stdin.close()
        process.stdout.close()
        process.wait()

    def _pipes(self) -> tuple[Any, Any]:
        if self._process is None:
            self._process = subprocess.Popen(
                ["git", "-C", str(self.repo), "cat-file", "--batch"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
        assert self._process.stdin is not None and self._process.stdout is not None
        return self._process.stdin, self._process.stdout

    def blob_chunks(self, commit: str, path: str) -> Iterator[bytes] | None:
        """Return an iterator over the blob's bytes, or None when it is absent.

        The iterator must be consumed (or closed) before the next request; an
        early close drains the rest of the object from the pipe.
        """

        if "\n" in path or "\n" in commit:
            raise ValueError("object names must not contain newlines")
        stdin, stdout = self._pipes()
        stdin.write(f"{commit}:{path}\n".encode())
        stdin.flush()
        header = stdout.readline().decode("utf-8", errors="replace").split()
        if len(header) != 3:
            # "<name> missing" or "<name> ambiguous": no content follows.
            return None
        size = int(header[2])
        if header[1] != "blob":
            self._drain(stdout, size)
            return None
        return self._chunks(stdout, size)

    def _chunks(self, stdout: Any, size: int) -> Iterator[bytes]:
        remaining = size
        try:
            while remaining:
                chunk = stdout.read(min(self.CHUNK_BYTES, remaining))
                if not chunk:
                    raise RuntimeError(f"git cat-file ended early in {self.repo}")
                remaining -= len(chunk)
                yield chunk
        finally:
            self._drain(stdout, remaining)

    @staticmethod
    def _drain(stdout: Any, remaining: int) -> None:
        while remaining:
            chunk = stdout.read(min(GitObjectReader.CHUNK_BYTES, remaining))
            if not chunk:
                raise RuntimeError("git cat-file ended early")
            remaining -= len(chunk)
        if stdout.read(1) != b"\n":
            raise RuntimeError("git cat-file output is out of sync")

    def blob_sha256(self, commit: str, path: str) -> str | None:
        chunks = self.blob_chunks(commit, path)
        if chunks is None:
            return None
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(chunk)
        return digest.hexdigest()


# Line boundaries str.splitlines() honours besides "\n" (UTF-8 encoded). The
# corpus resolver numbers JSONL rows with str.splitlines(), so a streaming
# reader that split on "\n" alone could disagree with it on these.
_EXTRA_LINE_BREAKS = re.compile(
    rb"[\r\x0b\x0c\x1c\x1d\x1e]|\xc2\x85|\xe2\x80[\xa8\xa9]"
)


def scan_lines(
    chunks: Iterator[bytes], wanted: set[int]
) -> tuple[str, int, dict[int, str]]:
    """Hash a blob and keep only the wanted lines, without holding the blob.

    Returns ``(sha256, line_count, {line_number: text})``. Line numbers are
    1-based and match ``raw.decode("utf-8").splitlines()`` exactly: segments
    are cut at ``"\n"`` (which never occurs inside a UTF-8 sequence) and only
    a segment carrying another line boundary is decoded and re-split.
    """

    digest = hashlib.sha256()
    found: dict[int, str] = {}
    count = 0
    carry = b""

    def emit(segment: bytes, terminated: bool) -> None:
        nonlocal count
        if _EXTRA_LINE_BREAKS.search(segment) is None:
            count += 1
            if count in wanted:
                found[count] = segment.decode("utf-8")
            return
        text = segment.decode("utf-8")
        for piece in (text + "\n" if terminated else text).splitlines():
            count += 1
            if count in wanted:
                found[count] = piece

    for chunk in chunks:
        digest.update(chunk)
        data = carry + chunk if carry else chunk
        start = 0
        while (end := data.find(b"\n", start)) >= 0:
            emit(data[start:end], True)
            start = end + 1
        carry = data[start:]
    if carry:
        emit(carry, False)
    return digest.hexdigest(), count, found


# --------------------------------------------------------------------------
# Release objects and provision resolution
# --------------------------------------------------------------------------


def registry_credentials() -> tuple[str, str]:
    url = os.environ.get(REGISTRY_URL_ENV, "").strip()
    key = os.environ.get(REGISTRY_KEY_ENV, "").strip()
    if url and key:
        return url, key
    for name, current in (
        ("NEXT_PUBLIC_SUPABASE_URL", url),
        ("NEXT_PUBLIC_SUPABASE_ANON_KEY", key),
    ):
        if current:
            continue
        value = subprocess.run(
            ["gh", "variable", "get", name, "-R", REGISTRY_VARIABLES_REPO],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if name.endswith("URL"):
            url = value
        else:
            key = value
    if not url.startswith("https://"):
        raise RuntimeError("corpus release registry URL must use HTTPS")
    return url, key


def canonical_content_sha256(content: dict[str, Any]) -> str:
    canonical = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(canonical)


def fetch_release_object(
    name: str, content_sha256: str, cache_dir: Path | None
) -> dict[str, Any]:
    """Return the signed release object, from cache or the public registry."""

    cached = cache_dir / f"{name}.json" if cache_dir else None
    if cached and cached.exists():
        payload = json.loads(cached.read_text(encoding="utf-8"))
        if canonical_content_sha256(payload["content"]) == content_sha256:
            return payload
    url_base, key = registry_credentials()
    url = (
        f"{url_base.rstrip('/')}/rest/v1/release_objects?select=release_object"
        f"&release_name=eq.{name}&content_sha256=eq.{content_sha256}&limit=2"
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
    if len(rows) != 1:
        raise RuntimeError(f"registry returned {len(rows)} rows for {name}")
    payload = rows[0]["release_object"]
    if payload.get("release") != name:
        raise RuntimeError(f"registry returned the wrong release for {name}")
    actual = canonical_content_sha256(payload["content"])
    if actual != content_sha256 or payload.get("content_sha256") != content_sha256:
        raise RuntimeError(f"release content digest mismatch for {name}")
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            "w", dir=cache_dir, delete=False, encoding="utf-8"
        )
        tmp.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp.close()
        os.replace(tmp.name, cached)
    return payload


def provision_artifacts(
    content: dict[str, Any], jurisdiction: str, document_class: str
) -> list[dict[str, Any]]:
    versions = {
        scope["version"]
        for scope in content["scopes"]
        if scope["jurisdiction"] == jurisdiction
        and scope["document_class"] == document_class
    }
    selected = []
    for artifact in content["artifacts"]:
        parts = artifact["path"].split("/")
        if (
            artifact["artifact_class"] != "provisions"
            or len(parts) != 6
            or parts[:3] != ["data", "corpus", "provisions"]
            or parts[3] != jurisdiction
            or parts[4] != document_class
            or parts[5].removesuffix(".jsonl") not in versions
        ):
            continue
        selected.append(artifact)
    return selected


def materialize_sparse_root(
    payload: dict[str, Any],
    corpus_repo: Path,
    root: Path,
    jurisdiction: str,
    document_class: str,
) -> str:
    """Write the release object plus the provision files one lookup needs."""

    content = payload["content"]
    name = payload["release"]
    sha = payload["content_sha256"]
    commit = content["git"]["commit"]
    release_dir = root / "releases" / name
    release_dir.mkdir(parents=True, exist_ok=True)
    target = release_dir / f"{sha}.json"
    if not target.exists():
        tmp = tempfile.NamedTemporaryFile(
            "w", dir=release_dir, delete=False, encoding="utf-8"
        )
        tmp.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp.close()
        os.replace(tmp.name, target)
    for artifact in provision_artifacts(content, jurisdiction, document_class):
        dest = root / artifact["path"]
        if dest.exists() and sha256_bytes(dest.read_bytes()) == artifact["sha256"]:
            continue
        raw = git_blob(corpus_repo, commit, artifact["path"])
        if raw is None:
            raise RuntimeError(
                f"corpus commit {commit} lacks release artifact {artifact['path']}"
            )
        if sha256_bytes(raw) != artifact["sha256"]:
            raise RuntimeError(
                f"artifact digest mismatch at corpus commit {commit}: "
                f"{artifact['path']}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile("wb", dir=dest.parent, delete=False)
        tmp.write(raw)
        tmp.close()
        os.replace(tmp.name, dest)
    return commit


def public_keys() -> tuple[str, ...]:
    raw = os.environ.get("AXIOM_CORPUS_RELEASE_PUBLIC_KEYS", "").strip()
    if raw:
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    return DEFAULT_RELEASE_PUBLIC_KEYS


def direct_row_exact(
    root: Path, content: dict[str, Any], citation: str
) -> dict[str, Any]:
    """Fallback for release rows the current resolver rejects (rows without an
    ``id``): the one active-scope row whose citation_path equals the request."""

    jurisdiction, document_class = citation.split("/")[:2]
    versions = {
        scope["version"]
        for scope in content["scopes"]
        if scope["jurisdiction"] == jurisdiction
        and scope["document_class"] == document_class
    }
    hits: list[tuple[str, str, int, dict[str, Any], str]] = []
    for artifact in provision_artifacts(content, jurisdiction, document_class):
        raw = (root / artifact["path"]).read_bytes()
        for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                record.get("citation_path") != citation
                or record.get("version") not in versions
            ):
                continue
            body = record.get("body")
            if isinstance(body, str) and body.strip():
                hits.append(
                    (artifact["path"], artifact["sha256"], number, record, body)
                )
    if len(hits) != 1:
        raise RuntimeError(
            f"direct_row_exact found {len(hits)} body rows for {citation}"
        )
    path, sha, number, record, body = hits[0]
    digest = sha256_text(body)
    return {
        "mode": "direct_row_exact",
        "resolved_citation_path": citation,
        "provision_file": path,
        "provision_file_sha256": sha,
        "line_number": number,
        "record_id": str(record.get("id") or ""),
        "stored_body_sha256": digest,
        "resolved_text_sha256": digest,
        "slice_required": False,
        "component_rows": 0,
        "text": body,
        "expression_date": record.get("expression_date"),
        "source_as_of": record.get("source_as_of"),
    }


def resolve_provision(
    payload: dict[str, Any],
    corpus_repo: Path,
    roots_dir: Path,
    citation: str,
) -> dict[str, Any]:
    """Resolve one citation through the release with the axiom-encode resolver.

    Returns the provision text plus the row identity the digests bind to.
    """

    from axiom_encode.corpus_resolver import (
        CorpusRowStructureError,
        LocalCorpusRelease,
        resolve_local_corpus_source,
    )

    jurisdiction, document_class = citation.split("/")[:2]
    name = payload["release"]
    root = roots_dir / name
    commit = materialize_sparse_root(
        payload, corpus_repo, root, jurisdiction, document_class
    )
    release = LocalCorpusRelease(root, name, payload["content_sha256"], public_keys())
    try:
        resolved = resolve_local_corpus_source(citation, release)
    except CorpusRowStructureError as exc:
        if getattr(exc, "reason", "") != "missing-release-metadata":
            raise
        result = direct_row_exact(root, payload["content"], citation)
        result["resolver_error"] = str(exc)[:200]
    else:
        result = {
            "mode": "axiom_encode_resolver",
            "resolved_citation_path": resolved.citation_path,
            "provision_file": resolved.provision_file,
            "provision_file_sha256": resolved.provision_file_sha256,
            "line_number": resolved.row.line_number,
            "record_id": resolved.row.record_id,
            "stored_body_sha256": resolved.stored_body_sha256,
            "resolved_text_sha256": resolved.resolved_text_sha256,
            "slice_required": resolved.slice_required,
            "component_rows": len(resolved.component_rows),
            "text": resolved.body,
            "expression_date": resolved.row.expression_date,
            "source_as_of": resolved.row.source_as_of,
        }
    result["corpus_commit"] = commit
    return result


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


class Report:
    """Counts and failure messages from one verification run."""

    def __init__(self) -> None:
        self.checked = 0
        self.failures: list[str] = []
        self.skipped: dict[str, int] = {}

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def skip(self, tier: str) -> None:
        self.skipped[tier] = self.skipped.get(tier, 0) + 1


def load_cases(corpus_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    index = json.loads((corpus_dir / "index.json").read_text(encoding="utf-8"))
    cases = []
    for entry in index["cases"]:
        case_dir = corpus_dir / "cases" / entry["id"]
        case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        cases.append((case_dir, case))
    return cases


def check_shipped(case_dir: Path, case: dict[str, Any], report: Report) -> None:
    cid = case.get("id", case_dir.name)
    for key in REQUIRED_CASE_KEYS:
        if key not in case:
            report.fail(f"{cid}: missing key {key}")
    if case.get("defect_kind") not in DEFECT_KINDS:
        report.fail(f"{cid}: unknown defect_kind {case.get('defect_kind')!r}")
    if case.get("id") != case_dir.name:
        report.fail(f"{cid}: id does not match directory name")
    if case.get("repo") != REPO_SLUGS.get(case.get("jurisdiction", "")):
        report.fail(f"{cid}: repo does not match jurisdiction")
    shipped = case.get("artifacts_shipped", True)
    for filename, key in (
        ("pre_fix.yaml", "pre_fix_artifact_sha256"),
        ("post_fix.yaml", "post_fix_artifact_sha256"),
        ("provision.txt", "provision_sha256"),
    ):
        path = case_dir / filename
        if not path.exists():
            if shipped:
                report.fail(f"{cid}: {filename} is missing")
            else:
                report.skip("shipped")
            continue
        actual = sha256_bytes(path.read_bytes())
        if actual != case.get(key):
            report.fail(f"{cid}: {filename} sha256 {actual} != {case.get(key)}")
    if case.get("pre_fix_artifact_sha256") == case.get("post_fix_artifact_sha256"):
        report.fail(f"{cid}: pre-fix and post-fix artifacts are identical")
    locator = case.get("locator") or {}
    for key in ("pre_fix_lines", "post_fix_lines", "rule_path"):
        if key not in locator:
            report.fail(f"{cid}: locator lacks {key}")
    confidence = case.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        report.fail(f"{cid}: confidence must be within [0, 1]")


COMMIT_METADATA_FORMAT = "%H%x00%P%x00%cI%x00%s"


def commit_metadata(
    repo: Path | str, commits: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Parents, committer date and subject per commit, from one ``git log`` call.

    Keys are full shas (abbreviated input is resolved first). ``date`` is
    ``%cI`` (strict ISO 8601 committer date) and ``subject`` is ``%s``, the
    same values ``git log -1 --format=%cI%n%s <commit>`` prints.
    """

    refs = sorted(set(commits))
    if not refs:
        return {}
    full = git(repo, "rev-parse", *(f"{ref}^{{commit}}" for ref in refs)).split()
    out = git(
        repo, "log", "--no-walk=unsorted", f"--format={COMMIT_METADATA_FORMAT}", *full
    )
    metadata: dict[str, dict[str, Any]] = {}
    for line in out.split("\n"):
        if not line:
            continue
        sha, parents, date, subject = line.split("\x00")
        metadata[sha] = {"parents": parents.split(), "date": date, "subject": subject}
    for ref, sha in zip(refs, full, strict=True):
        if ref != sha:
            metadata[ref] = metadata[sha]
    return metadata


def check_git(
    cases: list[dict[str, Any]], repos: dict[str, Path], report: Report, main_ref: str
) -> None:
    """Git tier: parentage, ancestry, and the module bytes on both sides.

    Also checks that ``commit_date`` and ``commit_subject`` equal the commit's
    ``%cI`` and ``%s``. Commit metadata and main-branch ancestry are read once
    per repository; module blobs stream through one ``git cat-file --batch``
    process.
    """

    by_jurisdiction: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        if repos.get(case.get("jurisdiction", "")) is None:
            report.skip("git")
            continue
        by_jurisdiction.setdefault(case["jurisdiction"], []).append(case)
    for jurisdiction, group in by_jurisdiction.items():
        repo = repos[jurisdiction]
        commits = sorted({str(case.get("commit")) for case in group})
        try:
            metadata = commit_metadata(repo, commits)
        except subprocess.CalledProcessError as exc:
            for case in group:
                report.fail(f"{case['id']}: git tier error: {exc}")
            continue
        on_main: set[str] | None = None
        if main_ref:
            try:
                on_main = set(git(repo, "rev-list", main_ref).split())
            except subprocess.CalledProcessError:
                on_main = set()
        with GitObjectReader(repo) as reader:
            for case in group:
                try:
                    _check_git_case(case, metadata, on_main, main_ref, reader, report)
                except (KeyError, RuntimeError, ValueError) as exc:
                    report.fail(f"{case.get('id')}: git tier error: {exc}")


def _check_git_case(
    case: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    on_main: set[str] | None,
    main_ref: str,
    reader: GitObjectReader,
    report: Report,
) -> None:
    cid = case["id"]
    commit = case["commit"]
    parent = case["parent_commit"]
    meta = metadata.get(commit) or {"parents": [], "date": None, "subject": None}
    if meta["parents"][:1] != [parent]:
        report.fail(f"{cid}: parent_commit is not the first parent of commit")
    if case.get("commit_date") != meta["date"]:
        report.fail(f"{cid}: commit_date does not match the commit")
    if case.get("commit_subject") != meta["subject"]:
        report.fail(f"{cid}: commit_subject does not match the commit")
    if on_main is not None and commit not in on_main:
        report.fail(f"{cid}: commit {commit[:10]} is not on {main_ref}")
    for label, ref, key in (
        ("pre_fix", parent, "pre_fix_artifact_sha256"),
        ("post_fix", commit, "post_fix_artifact_sha256"),
    ):
        digest = reader.blob_sha256(ref, case["module_path"])
        if digest is None:
            report.fail(f"{cid}: {label} module missing at {ref[:10]}")
        elif digest != case[key]:
            report.fail(f"{cid}: {label} artifact digest does not reproduce")


def check_corpus(
    cases: list[dict[str, Any]], corpus_repo: Path | None, report: Report
) -> None:
    """Corpus tier: the provision file and row each case names reproduce.

    Cases are grouped by ``(corpus_commit, provision_file)``; each pair is
    streamed once through :func:`scan_lines`, which hashes the whole blob and
    keeps only the row lines the group's cases point at.
    """

    if corpus_repo is None:
        for _case in cases:
            report.skip("corpus")
        return
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in cases:
        try:
            pair = (
                case["corpus_commit"],
                case["provision_resolution"]["provision_file"],
            )
        except KeyError as exc:
            report.fail(f"{case.get('id')}: corpus tier error: {exc}")
            continue
        groups.setdefault(pair, []).append(case)
    with GitObjectReader(corpus_repo) as reader:
        for (commit, provision_file), group in sorted(groups.items()):
            try:
                wanted = {
                    int(case["provision_resolution"]["line_number"]) for case in group
                }
                chunks = reader.blob_chunks(commit, provision_file)
                scanned = None if chunks is None else scan_lines(chunks, wanted)
            except (KeyError, RuntimeError, ValueError) as exc:
                for case in group:
                    report.fail(f"{case['id']}: corpus tier error: {exc}")
                continue
            for case in group:
                try:
                    _check_corpus_case(case, scanned, report)
                except (KeyError, ValueError) as exc:
                    report.fail(f"{case['id']}: corpus tier error: {exc}")


def _check_corpus_case(
    case: dict[str, Any],
    scanned: tuple[str, int, dict[int, str]] | None,
    report: Report,
) -> None:
    cid = case["id"]
    resolution = case["provision_resolution"]
    if scanned is None:
        report.fail(f"{cid}: provision file missing at corpus commit")
        return
    file_sha256, line_count, lines = scanned
    if file_sha256 != resolution["provision_file_sha256"]:
        report.fail(f"{cid}: provision file digest does not reproduce")
        return
    number = resolution["line_number"]
    if number < 1 or number > line_count:
        report.fail(f"{cid}: provision row line {number} is out of range")
        return
    record = json.loads(lines[number])
    if record.get("citation_path") != resolution["resolved_citation_path"]:
        report.fail(f"{cid}: provision row citation_path does not match")
    body = record.get("body")
    composed = bool(resolution.get("component_rows")) or bool(
        resolution.get("slice_required")
    )
    if isinstance(body, str) and body.strip():
        if sha256_text(body) != resolution["stored_body_sha256"]:
            report.fail(f"{cid}: provision row body digest does not reproduce")
    elif composed:
        # A document-level row with no body: the resolver composed the text
        # from descendant rows, which only the release tier re-derives.
        report.skip("corpus_body_composed")
    else:
        report.fail(f"{cid}: provision row has no body to reproduce")
    if resolution["mode"] == "direct_row_exact" or (
        not resolution.get("slice_required") and not resolution.get("component_rows")
    ):
        if isinstance(body, str) and sha256_text(body) != case["provision_sha256"]:
            report.fail(f"{cid}: provision text digest does not equal the row body")


def check_release(
    case: dict[str, Any],
    corpus_repo: Path | None,
    cache_dir: Path | None,
    roots_dir: Path | None,
    report: Report,
) -> None:
    cid = case["id"]
    if corpus_repo is None or roots_dir is None:
        report.skip("release")
        return
    payload = fetch_release_object(
        case["corpus_release"], case["corpus_release_content_sha256"], cache_dir
    )
    if payload["content"]["git"]["commit"] != case["corpus_commit"]:
        report.fail(f"{cid}: release names a different corpus commit")
    result = resolve_provision(
        payload, corpus_repo, roots_dir, case["corpus_citation_path"]
    )
    if result["mode"] != case["provision_resolution"]["mode"]:
        report.fail(f"{cid}: resolution mode changed to {result['mode']}")
    if sha256_text(result["text"]) != case["provision_sha256"]:
        report.fail(f"{cid}: re-resolved provision text digest does not reproduce")


def verify(
    corpus_dir: Path,
    *,
    repos: dict[str, Path],
    corpus_repo: Path | None,
    release_cache: Path | None,
    roots_dir: Path | None,
    with_release: bool,
    main_ref: str,
) -> Report:
    report = Report()
    seen: set[str] = set()
    loaded = load_cases(corpus_dir)
    for case_dir, case in loaded:
        report.checked += 1
        cid = case.get("id", case_dir.name)
        if cid in seen:
            report.fail(f"{cid}: duplicate id")
        seen.add(cid)
        check_shipped(case_dir, case, report)
    cases = [case for _case_dir, case in loaded]
    check_git(cases, repos, report, main_ref)
    check_corpus(cases, corpus_repo, report)
    for case in cases:
        if with_release:
            try:
                check_release(case, corpus_repo, release_cache, roots_dir, report)
            except Exception as exc:  # noqa: BLE001
                report.fail(f"{case.get('id')}: release tier error: {exc}")
        else:
            report.skip("release")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--rulespec-us", type=Path)
    parser.add_argument("--rulespec-uk", type=Path)
    parser.add_argument("--axiom-corpus", type=Path)
    parser.add_argument(
        "--main-ref",
        default="origin/main",
        help="ref every case commit must be an ancestor of ('' to skip)",
    )
    parser.add_argument(
        "--with-release",
        action="store_true",
        help="also fetch release objects and re-resolve provisions",
    )
    parser.add_argument("--release-cache", type=Path)
    parser.add_argument("--roots-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repos = {}
    if args.rulespec_us and args.rulespec_us.exists():
        repos["us"] = args.rulespec_us
    if args.rulespec_uk and args.rulespec_uk.exists():
        repos["uk"] = args.rulespec_uk
    corpus_repo = (
        args.axiom_corpus if args.axiom_corpus and args.axiom_corpus.exists() else None
    )
    roots_dir = args.roots_dir
    if args.with_release and roots_dir is None:
        roots_dir = Path(tempfile.mkdtemp(prefix="real-defects-roots-"))
    report = verify(
        args.corpus_dir,
        repos=repos,
        corpus_repo=corpus_repo,
        release_cache=args.release_cache,
        roots_dir=roots_dir,
        with_release=args.with_release,
        main_ref=args.main_ref,
    )
    print(
        f"checked {report.checked} cases; failures {len(report.failures)}; "
        f"skipped {json.dumps(report.skipped, sort_keys=True)}"
    )
    for failure in report.failures:
        print(f"FAIL {failure}")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
