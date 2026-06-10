"""Source-hash pinning, staleness checking, and provenance stamping.

RuleSpec modules ground to legal text through
``module.source_verification.corpus_citation_path``. The helpers here add
the mechanical half of that grounding:

- :func:`source_text_sha256` / :func:`source_verification_block` pin the
  SHA-256 of the exact provision text a module was encoded from.
- :func:`provenance_block` builds the ``module.encoding_provenance`` block
  recording the encoder version, model, and run id in module content.
- :func:`check_staleness` recomputes pinned hashes against a local corpus
  checkout and reports every module whose source text no longer matches.

The local-corpus read mirrors how the validator pipeline reads provision
JSONL files (exact citation match with best-body selection, then a
descendant fallback for metadata-only nodes). It is re-implemented here so
this module depends only on the standard library and PyYAML, not on
harness internals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Sequence

import yaml


class PinnedModule(NamedTuple):
    """A module that declares ``source_verification.source_sha256``."""

    module_path: Path
    citation_path: str | None
    pinned_sha: str


class StaleModule(NamedTuple):
    """A pinned module whose corpus text no longer matches its pin.

    ``current_sha`` is ``None`` when the pinned provision text cannot be
    found in the corpus checkout (including pins that declare no
    ``corpus_citation_path``), which is itself a staleness signal: the pin
    can no longer be verified.
    """

    module_path: Path
    pinned_sha: str
    current_sha: str | None


def source_text_sha256(source_text: str) -> str:
    """SHA-256 hex digest of provision text, exactly as stored (UTF-8)."""
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


def source_verification_block(citation_path: str, source_text: str) -> dict[str, str]:
    """Build a ``module.source_verification`` block pinning ``source_text``.

    ``source_text`` must be the provision body exactly as read from the
    corpus (for example via :func:`read_corpus_provision_text`), so that
    :func:`check_staleness` recomputes an identical digest later.
    """
    return {
        "corpus_citation_path": citation_path,
        "source_sha256": source_text_sha256(source_text),
    }


def provenance_block(model: str, run_id: str) -> dict[str, str]:
    """Build a ``module.encoding_provenance`` block for the current encoder."""
    return {
        "encoder": f"axiom-encode/{_encoder_version()}",
        "model": model,
        "run_id": run_id,
    }


def _encoder_version() -> str:
    try:
        from axiom_encode import __version__
    except ImportError:  # pragma: no cover - the package always ships one
        return "unknown"
    return __version__


def corpus_provisions_root(corpus_root: Path) -> Path | None:
    """Locate the ``provisions/`` directory inside a corpus checkout."""
    root = Path(corpus_root).expanduser()
    candidates = (
        root,
        root / "provisions",
        root / "data" / "corpus",
        root / "data" / "corpus" / "provisions",
    )
    for candidate in candidates:
        provisions_root = (
            candidate if candidate.name == "provisions" else candidate / "provisions"
        )
        try:
            resolved = provisions_root.resolve()
            if resolved.is_dir():
                return resolved
        except OSError:
            continue
    return None


def read_corpus_provision_text(corpus_root: Path, citation_path: str) -> str | None:
    """Read the body text for ``citation_path`` from a local corpus checkout.

    Exact-citation records are preferred, picking the best body when files
    contain duplicate citations (latest ``source_as_of``, then official
    source format, then ``version``). Metadata-only nodes fall back to the
    concatenated bodies of their child provisions.
    """
    normalized_path = citation_path.strip().strip("/")
    if not normalized_path:
        return None
    provisions_root = corpus_provisions_root(corpus_root)
    if provisions_root is None:
        return None

    provision_files = _candidate_provision_files(provisions_root, normalized_path)
    exact_records: list[dict[str, Any]] = []
    for provision_file in provision_files:
        exact_records.extend(_read_provision_records(provision_file, normalized_path))
    source_text = _select_record_body(exact_records)
    if source_text is not None:
        return source_text

    for provision_file in provision_files:
        source_text = _read_descendant_text(provision_file, normalized_path)
        if source_text is not None:
            return source_text
    return None


def _candidate_provision_files(
    provisions_root: Path,
    citation_path: str,
) -> tuple[Path, ...]:
    parts = citation_path.split("/")
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_files(base: Path) -> None:
        for path in sorted(base.glob("*.jsonl")):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)

    if len(parts) >= 2:
        add_files(provisions_root / parts[0] / parts[1])
    if not candidates:
        for path in sorted(provisions_root.rglob("*.jsonl")):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)
    return tuple(candidates)


def _iter_provision_records(provision_file: Path) -> Iterator[dict[str, Any]]:
    try:
        lines = provision_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def _read_provision_records(
    provision_file: Path,
    citation_path: str,
) -> list[dict[str, Any]]:
    return [
        record
        for record in _iter_provision_records(provision_file)
        if record.get("citation_path") == citation_path
    ]


def _select_record_body(records: list[dict[str, Any]]) -> str | None:
    """Select the best body when corpus files contain duplicate citations."""
    body_records = [record for record in records if record.get("body") is not None]
    if not body_records:
        return None

    def record_key(record: dict[str, Any]) -> tuple[str, int, str]:
        source_as_of = str(record.get("source_as_of") or "")
        source_format = str(record.get("source_format") or "")
        official_source = 1 if source_format == "legislation.gov.uk-clml" else 0
        version = str(record.get("version") or "")
        return (source_as_of, official_source, version)

    selected = max(body_records, key=record_key)
    return str(selected["body"])


def _read_descendant_text(provision_file: Path, citation_path: str) -> str | None:
    """Read body-bearing child provisions for a metadata-only node."""
    child_prefix = f"{citation_path}/"
    descendants: list[tuple[int, int, str | None, str]] = []
    for record in _iter_provision_records(provision_file):
        record_path = str(record.get("citation_path") or "")
        if not record_path.startswith(child_prefix):
            continue
        body = record.get("body")
        if body is None:
            continue
        descendants.append(
            (
                int(record.get("level") or 0),
                int(record.get("ordinal") or 0),
                str(record.get("heading") or "") or None,
                str(body),
            )
        )

    if not descendants:
        return None
    chunks: list[str] = []
    for _, _, heading, body in sorted(descendants, key=lambda item: item[:2]):
        if heading:
            chunks.append(f"{heading}\n\n{body}")
        else:
            chunks.append(body)
    return "\n\n".join(chunks)


def iter_pinned_modules(rulespec_root: Path) -> Iterator[PinnedModule]:
    """Yield every module under ``rulespec_root`` that pins a source hash."""
    root = Path(rulespec_root)
    paths = sorted(
        path
        for pattern in ("*.yaml", "*.yml")
        for path in root.rglob(pattern)
        if path.is_file()
    )
    for path in paths:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(payload, dict):
            continue
        module = payload.get("module")
        if not isinstance(module, dict):
            continue
        verification = module.get("source_verification")
        if not isinstance(verification, dict):
            continue
        raw_sha = verification.get("source_sha256")
        if raw_sha is None:
            continue
        # Coerce non-string scalars (for example an unquoted all-digit
        # digest that YAML parses as an integer) instead of skipping them:
        # a present pin must always be checked, and a coerced value can
        # never match a recomputed digest, so it surfaces as stale rather
        # than silently passing.
        pinned_sha = str(raw_sha).strip()
        if not pinned_sha:
            continue
        citation_path = verification.get("corpus_citation_path")
        if not isinstance(citation_path, str) or not citation_path.strip():
            citation_path = None
        yield PinnedModule(path, citation_path, pinned_sha)


def check_staleness(rulespec_root: Path, corpus_root: Path) -> list[StaleModule]:
    """Report every pinned module whose corpus text no longer matches.

    Returns ``(module_path, pinned_sha, current_sha)`` tuples for each
    module declaring ``source_verification.source_sha256`` whose recomputed
    hash differs from the pin. ``current_sha`` is ``None`` when the pinned
    provision text cannot be found (missing provision, or a pin without a
    ``corpus_citation_path``). Modules whose hashes still match are omitted.
    """
    stale: list[StaleModule] = []
    for module_path, citation_path, pinned_sha in iter_pinned_modules(rulespec_root):
        current_sha: str | None = None
        if citation_path is not None:
            source_text = read_corpus_provision_text(corpus_root, citation_path)
            if source_text is not None:
                current_sha = source_text_sha256(source_text)
        if current_sha != pinned_sha.lower():
            stale.append(StaleModule(module_path, pinned_sha, current_sha))
    return stale


def run_check_source_staleness(argv: Sequence[str] | None = None) -> int:
    """CLI for ``axiom-encode check-source-staleness``."""
    parser = argparse.ArgumentParser(
        prog="axiom-encode check-source-staleness",
        description=(
            "Recompute module.source_verification.source_sha256 pins against "
            "a local corpus checkout and report modules whose source text "
            "changed."
        ),
    )
    parser.add_argument(
        "--rulespec-root",
        required=True,
        type=Path,
        help="Root of a jurisdiction RuleSpec checkout to scan for pinned modules.",
    )
    parser.add_argument(
        "--corpus-root",
        required=True,
        type=Path,
        help="Root of a local axiom-corpus checkout (containing provisions/).",
    )
    args = parser.parse_args(argv)

    pinned = list(iter_pinned_modules(args.rulespec_root))
    if not pinned:
        print(f"No modules under {args.rulespec_root} pin source_sha256.")
        return 0

    stale = check_staleness(args.rulespec_root, args.corpus_root)
    if not stale:
        print(f"All {len(pinned)} pinned module(s) match the current corpus text.")
        return 0

    for entry in stale:
        current = entry.current_sha or "<provision text not found>"
        print(f"STALE {entry.module_path}")
        print(f"  pinned  {entry.pinned_sha}")
        print(f"  current {current}")
    print(f"{len(stale)} of {len(pinned)} pinned module(s) are stale.")
    return 1
