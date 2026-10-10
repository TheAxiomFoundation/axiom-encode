#!/usr/bin/env python3
"""Re-derive and verify the real-defects verifier corpus.

The corpus under ``benchmarks/verifier/real_defects_v0`` stores, per case, the
pre-fix and post-fix RuleSpec module bytes, the provision text the module
cites, and their sha256 digests. Every digest must reproduce from the named
Git commits and the named signed corpus release. This script checks that.

Tiers (each one is skipped, and reported as skipped, when its inputs are absent):

* ``shipped``  - the shipped files hash to what ``case.json`` says; an
                 extended provision splits into its recorded components; the
                 provision review's quotes sit at their recorded spans.
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
                 (or a local cache) and every citation of the provision is
                 re-resolved through ``axiom_encode.corpus_resolver``; the
                 text (composed, when the provision was extended) must hash
                 to ``provision_sha256``.

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
# Every key the corpus README's case-schema table documents.
REQUIRED_CASE_KEYS = (
    "id",
    "jurisdiction",
    "repo",
    "commit",
    "parent_commit",
    "commit_date",
    "commit_subject",
    "pr_url",
    "module_path",
    "corpus_citation_path",
    "corpus_citation_paths_all",
    "corpus_release",
    "corpus_release_content_sha256",
    "corpus_commit",
    "defect_kind",
    "other_kind",
    "confidence",
    "description",
    "description_source",
    "locator",
    "pre_fix_artifact_sha256",
    "post_fix_artifact_sha256",
    "provision_sha256",
    "provision_chars",
    "provision_resolution",
    "provision_extension",
    "evidence_in_provision",
    "evidence_check",
    "judgeable_from_provision",
    "provision_review",
    "fix_stage",
    "artifacts_shipped",
    "family_id",
    "family_size",
    "family_representative",
    "triage_status",
    "triage",
    "triage_notes",
)
FIX_STAGES = ("post_merge", "pre_merge_review")
TRIAGE_STATUSES = ("fidelity", "unclear")
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
    "judgeable_from_provision",
    "provision_review",
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


# --------------------------------------------------------------------------
# Extended provisions, review quotes, local paths
# --------------------------------------------------------------------------

PROVISION_COMPOSITION = "sources_joined_v1"
PROVISION_REVIEW_VERDICTS = ("in_provision", "in_other_citation", "not_in_sources")
PROVISION_REVIEW_OUTCOMES = {
    "in_provision": "kept",
    "in_other_citation": "provision_extended",
    "not_in_sources": "not_judgeable",
}


def provision_source_header(citation_path: str) -> str:
    return f"--- Source: {citation_path} ---"


def compose_provision(components: list[tuple[str, str]]) -> str:
    """The provision text for ``(citation_path, text)`` components.

    One component is its text, unchanged. Several are joined as
    ``sources_joined_v1``: each text under a ``--- Source: <path> ---`` line,
    with one blank line between components, in the order given.
    """

    if len(components) == 1:
        return components[0][1]
    return "\n\n".join(
        f"{provision_source_header(path)}\n{text}" for path, text in components
    )


def split_provision(text: str, components: list[tuple[str, int]]) -> list[str] | None:
    """Invert :func:`compose_provision` given each ``(citation_path, chars)``.

    Returns the component texts, or None when ``text`` is not that
    composition. Lengths make the split exact even when a component's own
    text contains a line that looks like a header.
    """

    for _path, chars in components:
        if isinstance(chars, bool) or not isinstance(chars, int) or chars < 0:
            return None
    if len(components) == 1:
        return [text] if len(text) == components[0][1] else None
    parts: list[str] = []
    position = 0
    for index, (path, chars) in enumerate(components):
        lead = ("\n\n" if index else "") + provision_source_header(path) + "\n"
        if not text.startswith(lead, position):
            return None
        position += len(lead)
        if position + chars > len(text):
            return None
        parts.append(text[position : position + chars])
        position += chars
    return parts if position == len(text) else None


def component_regions(components: list[tuple[str, int]]) -> dict[str, tuple[int, int]]:
    """Where each component's own text sits in :func:`compose_provision`'s output.

    ``{citation_path: (start, end)}`` from each ``(citation_path, chars)``,
    as ``[start, end)`` character offsets; headers and separators lie outside
    every region.
    """

    if len(components) == 1:
        path, chars = components[0]
        return {path: (0, chars)}
    regions: dict[str, tuple[int, int]] = {}
    position = 0
    for index, (path, chars) in enumerate(components):
        position += len("\n\n" if index else "") + len(
            provision_source_header(path) + "\n"
        )
        regions[path] = (position, position + chars)
        position += chars
    return regions


_TYPOGRAPHY = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-"}
    | {"\u2014": "-", "\u00a0": " "}
)
_ELLIPSIS = re.compile(r"\.\.\.|\u2026")
MAX_QUOTE_ELISION_CHARS = 300


def _folded(text: str) -> tuple[str, list[int]]:
    """``text`` case-folded with whitespace runs as one space and typographic
    quotes and dashes flattened, plus each folded character's source offset."""

    out: list[str] = []
    offsets: list[int] = []
    pending_space = False
    for offset, char in enumerate(text):
        if char.isspace() or char == "\u00a0":
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            offsets.append(offset - 1)
            pending_space = False
        for piece in char.translate(_TYPOGRAPHY).casefold():
            out.append(piece)
            offsets.append(offset)
    return "".join(out), offsets


def locate_quote(quote: str, text: str) -> tuple[int, int] | None:
    """Where ``quote`` occurs in ``text``, as a ``[start, end)`` character span.

    Matching ignores case, whitespace runs and typographic quote and dash
    forms. Every match starts and ends on whole characters of ``text``: a
    quote never matches part of a character's case-folded expansion (``as``
    does not match inside ``aß``, which folds to ``ass``). An ellipsis in the
    quote splits it into fragments that must occur in order, each starting
    within ``MAX_QUOTE_ELISION_CHARS`` folded characters of the end of the
    one before; the span then runs from the first fragment's start to the
    last one's end. None when it does not occur.
    """

    haystack, offsets = _folded(text)
    fragments = [_folded(part)[0] for part in _ELLIPSIS.split(quote)]
    fragments = [part for part in fragments if part]
    if not fragments:
        return None

    def aligned(start: int, end: int) -> bool:
        return (start == 0 or offsets[start] != offsets[start - 1]) and (
            end == len(haystack) or offsets[end] != offsets[end - 1]
        )

    def find(part: str, begin: int) -> int:
        found = haystack.find(part, begin)
        while found >= 0 and not aligned(found, found + len(part)):
            found = haystack.find(part, found + 1)
        return found

    start = find(fragments[0], 0)
    while start >= 0:
        end = start + len(fragments[0])
        for part in fragments[1:]:
            found = find(part, end)
            if found < 0 or found - end > MAX_QUOTE_ELISION_CHARS:
                break
            end = found + len(part)
        else:
            return offsets[start], offsets[end - 1] + 1
        start = find(fragments[0], start + 1)
    return None


_LOCAL_PATH_RULES = (
    # A Claude session scratchpad: /private/tmp/claude-<uid>/<session>/<id>/scratchpad
    (
        re.compile(
            r"(?:/private)?/tmp/claude-\d+/[A-Za-z0-9._-]+/[0-9a-f-]+(?:/scratchpad)?"
        ),
        "<scratch>",
    ),
    # A session worktree of a checkout under the org folder.
    (
        re.compile(
            r"(?:/Users/[^/\s\"']+|~)/TheAxiomFoundation/"
            r"([A-Za-z0-9._-]+)/\.claude/worktrees/[A-Za-z0-9._-]+"
        ),
        r"\1",
    ),
    # A checkout under the org folder: keep the repository name.
    (re.compile(r"(?:/Users/[^/\s\"']+|~)/TheAxiomFoundation/(?=[A-Za-z0-9._-])"), ""),
    (re.compile(r"(?:/Users/[^/\s\"']+|~)/TheAxiomFoundation\b"), "<checkouts>"),
)
LOCAL_PATH = re.compile(r"/Users/[A-Za-z0-9._-]+/|/private/(?:tmp|var)/|/tmp/claude-")


def scrub_local_paths(text: str) -> str:
    """Replace the triage machine's absolute paths with portable forms.

    Session scratchpads become ``<scratch>`` and checkouts under the org
    folder become the repository name (``rulespec-us/...``). Idempotent.
    """

    for pattern, replacement in _LOCAL_PATH_RULES:
        text = pattern.sub(replacement, text)
    return text


def scrub_json(value: Any) -> Any:
    """:func:`scrub_local_paths` over every string in a JSON value."""

    if isinstance(value, str):
        return scrub_local_paths(value)
    if isinstance(value, list):
        return [scrub_json(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub_json(item) for key, item in value.items()}
    return value


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
    # Always present, so a release that carries nothing for this scope fails
    # in the resolver the same way whatever an earlier lookup materialized.
    (root / "data" / "corpus" / "provisions").mkdir(parents=True, exist_ok=True)
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


def portable_error(exc: BaseException, root: Path) -> str:
    """An error's message without the local path of the sparse corpus root."""

    message = str(exc).replace(f"{root}/", "").replace(str(root), "<corpus root>")
    return message[:200]


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
        result["resolver_error"] = portable_error(exc, root)
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
    if case.get("fix_stage") not in FIX_STAGES:
        report.fail(f"{cid}: unknown fix_stage {case.get('fix_stage')!r}")
    if case.get("triage_status") not in TRIAGE_STATUSES:
        report.fail(f"{cid}: unknown triage_status {case.get('triage_status')!r}")
    provision_path = case_dir / "provision.txt"
    provision_text = (
        provision_path.read_text(encoding="utf-8") if provision_path.exists() else None
    )
    try:
        problems = provision_record_problems(case, provision_text)
    except (KeyError, TypeError) as exc:
        problems = [f"provision records are malformed: {exc!r}"]
    for message in problems:
        report.fail(f"{cid}: {message}")
    locator = case.get("locator") or {}
    for key in ("pre_fix_lines", "post_fix_lines", "rule_path"):
        if key not in locator:
            report.fail(f"{cid}: locator lacks {key}")
    confidence = case.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        report.fail(f"{cid}: confidence must be within [0, 1]")


def provision_components(case: dict[str, Any]) -> list[dict[str, Any]]:
    """The citations ``provision.txt`` is made of, first citation first.

    Each entry has ``citation_path``, ``text_sha256``, ``chars`` and the
    ``resolution`` record of its row. An unextended case has one: the first
    citation, whose text is the whole provision.
    """

    extension = case.get("provision_extension")
    first = {
        "citation_path": case["corpus_citation_path"],
        "text_sha256": case["provision_sha256"],
        "chars": case["provision_chars"],
        "resolution": case["provision_resolution"],
    }
    if not extension:
        return [first]
    first["text_sha256"] = extension["first_citation_text_sha256"]
    first["chars"] = extension["first_citation_chars"]
    return [first, *extension["added"]]


def provision_record_problems(
    case: dict[str, Any], provision_text: str | None
) -> list[str]:
    """What is inconsistent in a case's extension and review records.

    ``provision_text`` is the shipped ``provision.txt``, or None for a
    metadata-only case (the text checks are then skipped).
    """

    problems: list[str] = []
    extension = case.get("provision_extension")
    components = provision_components(case)
    sized = [(c["citation_path"], c["chars"]) for c in components]
    first = components[0]["citation_path"]
    if extension is not None:
        if extension.get("composition") != PROVISION_COMPOSITION:
            problems.append("provision_extension has an unknown composition")
        if not extension.get("added"):
            problems.append("provision_extension adds no citation")
        paths = [path for path, _chars in sized]
        if len(set(paths)) != len(paths):
            problems.append("provision_extension repeats a citation")
        if provision_text is not None:
            parts = split_provision(provision_text, sized)
            if parts is None:
                problems.append("provision.txt is not its recorded composition")
            else:
                for component, part in zip(components, parts, strict=True):
                    if sha256_text(part) != component["text_sha256"]:
                        problems.append(
                            f"provision component {component['citation_path']} "
                            "does not hash to its recorded digest"
                        )
    review = case.get("provision_review")
    judgeable = case.get("judgeable_from_provision")
    if review is None:
        if judgeable is not None:
            problems.append("judgeable_from_provision is set without a review")
        if extension is not None:
            problems.append("provision_extension is set without a review")
        return problems
    verdict = review.get("verdict")
    if verdict not in PROVISION_REVIEW_VERDICTS:
        problems.append(f"unknown provision_review verdict {verdict!r}")
        return problems
    if review.get("outcome") != PROVISION_REVIEW_OUTCOMES[verdict]:
        problems.append("provision_review outcome does not follow from its verdict")
    if judgeable is not (verdict != "not_in_sources"):
        problems.append("judgeable_from_provision does not follow from the verdict")
    if (verdict == "in_other_citation") != (extension is not None):
        problems.append("provision_extension and the review verdict disagree")
    quotes = review.get("decisive_quotes") or []
    if verdict == "not_in_sources":
        if quotes:
            problems.append("a not_in_sources review has decisive quotes")
        if not (review.get("missing_basis") or "").strip():
            problems.append("a not_in_sources review must say what the fix rests on")
        return problems
    if not quotes:
        problems.append("a judgeable case needs a decisive quote")
    regions = component_regions(sized)
    cited = set()
    for quote in quotes:
        citation = quote.get("citation_path")
        text = quote.get("quote") or ""
        if citation not in regions:
            problems.append(f"decisive quote cites {citation!r}, not in the provision")
            continue
        cited.add(citation)
        span = quote.get("span")
        if not (
            isinstance(span, list)
            and len(span) == 2
            and all(isinstance(x, int) and not isinstance(x, bool) for x in span)
        ):
            problems.append("a decisive quote lacks its span")
            continue
        start, end = span
        low, high = regions[citation]
        if not low <= start < end <= high:
            problems.append(f"decisive quote lies outside {citation}: {text[:60]!r}")
            continue
        if provision_text is not None and locate_quote(
            text, provision_text[start:end]
        ) != (0, end - start):
            problems.append(f"decisive quote is not at its span: {text[:60]!r}")
    if verdict == "in_provision" and cited - {first}:
        problems.append("an in_provision review quotes another citation")
    if verdict == "in_other_citation" and not cited - {first}:
        problems.append("no decisive quote lies in an added citation")
    return problems


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
    """Corpus tier: the provision file and row of every component reproduce.

    A case's components are its first citation and any citation its provision
    was extended with. Components are grouped by ``(corpus_commit,
    provision_file)``; each pair is streamed once through :func:`scan_lines`,
    which hashes the whole blob and keeps only the row lines the group points
    at.
    """

    if corpus_repo is None:
        for _case in cases:
            report.skip("corpus")
        return
    groups: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for case in cases:
        try:
            for component in provision_components(case):
                resolution = component["resolution"]
                pair = (
                    resolution.get("corpus_commit") or case["corpus_commit"],
                    resolution["provision_file"],
                )
                groups.setdefault(pair, []).append((case["id"], component))
        except (KeyError, TypeError) as exc:
            report.fail(f"{case.get('id')}: corpus tier error: {exc!r}")
    with GitObjectReader(corpus_repo) as reader:
        for (commit, provision_file), group in sorted(groups.items()):
            try:
                wanted = {
                    int(component["resolution"]["line_number"])
                    for _cid, component in group
                }
                chunks = reader.blob_chunks(commit, provision_file)
                scanned = None if chunks is None else scan_lines(chunks, wanted)
            except (KeyError, RuntimeError, ValueError) as exc:
                for cid, _component in group:
                    report.fail(f"{cid}: corpus tier error: {exc}")
                continue
            for cid, component in group:
                try:
                    _check_corpus_component(cid, component, scanned, report)
                except (KeyError, ValueError) as exc:
                    report.fail(f"{cid}: corpus tier error: {exc}")


def _check_corpus_component(
    cid: str,
    component: dict[str, Any],
    scanned: tuple[str, int, dict[int, str]] | None,
    report: Report,
) -> None:
    resolution = component["resolution"]
    cid = f"{cid} [{component['citation_path']}]"
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
        if isinstance(body, str) and sha256_text(body) != component["text_sha256"]:
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
    texts: list[tuple[str, str]] = []
    for component in provision_components(case):
        citation = component["citation_path"]
        result = resolve_provision(payload, corpus_repo, roots_dir, citation)
        if result["mode"] != component["resolution"]["mode"]:
            report.fail(
                f"{cid}: resolution mode of {citation} changed to {result['mode']}"
            )
        if sha256_text(result["text"]) != component["text_sha256"]:
            report.fail(f"{cid}: re-resolved text of {citation} does not reproduce")
        texts.append((citation, result["text"]))
    if sha256_text(compose_provision(texts)) != case["provision_sha256"]:
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
