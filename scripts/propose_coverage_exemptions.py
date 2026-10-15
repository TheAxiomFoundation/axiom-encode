#!/usr/bin/env python3
"""Print unapproved coverage-exemption candidates from verified release bytes.

The withdrawn ownership scanner is used only here, as a proposal aid. A printed
candidate is not an exemption: its ownership and exact requested scope still
need the review described in docs/coverage-exemptions.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axiom_encode.corpus_release import (
    VerifiedCorpusReleaseObject,
    VerifiedReleaseArtifact,
    verify_release_object,
)
from axiom_encode.corpus_resolver import (
    MAX_CORPUS_PROVISION_BYTES,
    _record_status,
    read_bounded_regular_file,
    require_canonical_corpus_citation_path,
    validate_corpus_release_name,
)
from axiom_encode.harness import validator_pipeline as vp

try:
    from . import retired_coverage_heuristic as retired
except ImportError:
    import retired_coverage_heuristic as retired

# Published verification roots, also used by scripts/verify_real_defects.py.
# These are public keys, not signing material. Rotation can be supplied through
# AXIOM_CORPUS_RELEASE_PUBLIC_KEYS (comma separated).
DEFAULT_RELEASE_PUBLIC_KEYS = (
    "KwIYEbbs/905yxn/9Yi6jYTF8oyZcq1FlxiG4e0y0tg=",
    "9EJzeLnZsdBoDEDkU8xbi7HjYVVvd4HDX/BxPHPvXTk=",
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CorpusRow:
    citation_path: str
    text: str
    corpus_artifact: str
    artifact_sha256: str
    row: int
    status: str | None


@dataclass(frozen=True)
class ReleaseCorpus:
    root: Path
    release: VerifiedCorpusReleaseObject
    selected_artifacts: tuple[VerifiedReleaseArtifact, ...] | None = None

    @property
    def artifacts(self) -> tuple[VerifiedReleaseArtifact, ...]:
        if self.selected_artifacts is not None:
            return self.selected_artifacts
        return tuple(
            artifact
            for artifact in self.release.artifacts
            if artifact.artifact_class == "provisions"
        )


def load_release(
    corpus_path: Path,
    name: str,
    content_sha256: str | None = None,
    release_object: Path | None = None,
    *,
    public_keys: Sequence[str] | None = None,
) -> ReleaseCorpus:
    """Verify an offline signed release; never fetch or materialize corpus data."""
    validate_corpus_release_name(name)
    root = corpus_path.resolve(strict=True)
    if release_object is None:
        directory = root / "releases" / name
        if content_sha256 is None:
            objects = sorted(directory.glob("*.json"))
            if len(objects) != 1:
                raise ValueError(
                    "Expected exactly one release object; specify --release-sha256"
                )
            release_object = objects[0]
        else:
            # Validation below also checks that the requested digest is valid.
            if len(content_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in content_sha256
            ):
                raise ValueError("--release-sha256 must be a lowercase SHA-256 digest")
            release_object = directory / f"{content_sha256}.json"
    if public_keys is None:
        configured = os.environ.get("AXIOM_CORPUS_RELEASE_PUBLIC_KEYS", "").strip()
        public_keys = (
            tuple(part.strip() for part in configured.split(",") if part.strip())
            if configured
            else DEFAULT_RELEASE_PUBLIC_KEYS
        )
    payload = json.loads(release_object.read_text(encoding="utf-8"))
    verified = verify_release_object(payload, public_key=public_keys)
    if verified.name != name or (
        content_sha256 is not None and verified.content_sha256 != content_sha256
    ):
        raise ValueError("Release object does not match the requested release pin")
    return ReleaseCorpus(root, verified)


def select_artifacts(corpus: ReleaseCorpus, manifest: Path | None) -> ReleaseCorpus:
    """Select an explicitly recorded historical slice of a verified release."""
    if manifest is None:
        return corpus
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        payload.get("release") != corpus.release.name
        or payload.get("release_sha256") != corpus.release.content_sha256
    ):
        raise ValueError("Artifact manifest does not match the verified release")
    inventory = {artifact.path: artifact for artifact in corpus.artifacts}
    selected = []
    for entry in payload["artifacts"]:
        artifact = inventory.get(entry["path"])
        if artifact is None or any(
            entry.get(key) != expected
            for key, expected in (
                ("sha256", artifact.sha256),
                ("bytes", artifact.byte_count),
                ("rows", artifact.row_count),
                ("artifact_class", artifact.artifact_class),
            )
        ):
            raise ValueError("Artifact manifest differs from the signed inventory")
        selected.append(artifact)
    if len({artifact.path for artifact in selected}) != len(selected):
        raise ValueError("Duplicate artifact in manifest")
    return ReleaseCorpus(corpus.root, corpus.release, tuple(selected))


def iter_release_rows(
    corpus: ReleaseCorpus,
    *,
    artifact_start: int = 0,
    artifact_stop: int | None = None,
) -> Iterator[CorpusRow]:
    """Read each selected artifact once, verifying digest, length, and rows."""
    artifacts = corpus.artifacts
    stop = len(artifacts) if artifact_stop is None else artifact_stop
    if not 0 <= artifact_start <= stop <= len(artifacts):
        raise ValueError("Invalid artifact range")
    for artifact in artifacts[artifact_start:stop]:
        raw = read_bounded_regular_file(
            corpus.root,
            corpus.root / artifact.path,
            label="coverage proposal corpus artifact",
            max_bytes=MAX_CORPUS_PROVISION_BYTES,
        )
        if (
            len(raw) != artifact.byte_count
            or hashlib.sha256(raw).hexdigest() != artifact.sha256
        ):
            raise ValueError(f"Release artifact bytes do not match: {artifact.path}")
        # JSONL physical rows are delimited by LF bytes. Unicode line separators
        # inside a JSON string belong to the source body, not another record.
        lines = raw.split(b"\n")
        records = []
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            record = json.loads(line.decode("utf-8"))
            if not isinstance(record, dict):
                raise ValueError(f"Corpus row is not an object: {artifact.path}")
            citation = record.get("citation_path")
            body = record.get("body")
            if not isinstance(citation, str) or not citation:
                raise ValueError(f"Corpus citation is missing: {artifact.path}")
            if body is None:
                body = ""
            if not isinstance(body, str):
                raise ValueError(f"Corpus body is not text: {artifact.path}")
            records.append(
                CorpusRow(
                    citation,
                    body,
                    artifact.path,
                    artifact.sha256,
                    line_number,
                    _record_status(record),
                )
            )
        if artifact.row_count is not None and len(records) != artifact.row_count:
            raise ValueError(f"Release artifact row count differs: {artifact.path}")
        yield from records


def scan_main_demands(source_text: str):
    """Use the unchanged demand scanner, before reviewed-list filtering."""
    return vp._high_signal_top_level_coverage_demands(source_text)


def propose_row(
    row: CorpusRow,
    requested_scope: list[str] | None = None,
    *,
    requested_source: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Return proposals only; retain both scanners' physical context for review."""
    main_demands = scan_main_demands(row.text)
    if requested_scope:
        scope_top = vp._top_level_subparagraph_path(tuple(requested_scope))
        main_demands = [d for d in main_demands if (d.label,) == scope_top]
    heuristic_citation = row.citation_path
    retired_demands = retired._high_signal_top_level_subparagraphs(
        row.text, corpus_citation_path=heuristic_citation
    )
    kept = {(str(label), label.physical_start) for label, _ in retired_demands}
    branches = retired.recognize_source_structure(
        row.text, corpus_citation_path=heuristic_citation
    )
    for demand in main_demands:
        if (demand.label, demand.physical_start) in kept:
            continue
        window = row.text[
            max(0, demand.physical_start - 80) : demand.physical_start + 80
        ]
        preceding = [
            branch for branch in branches if branch.start <= demand.physical_start
        ]
        owner = preceding[-1] if preceding else None
        yield {
            "candidate": True,
            "citation_path": row.citation_path,
            "requested_scope": requested_scope,
            "source_text_sha256": sha256_text(row.text),
            "label": demand.label,
            "physical_start": demand.physical_start,
            "window": window,
            "window_sha256": sha256_text(window),
            "reason": None,
            "owner_evidence": None,
            "corpus_artifact": row.corpus_artifact,
            "artifact_sha256": row.artifact_sha256,
            "row": row.row,
            "reviewer": None,
            "reviewed_at": None,
            "proposal_context": {
                "retired_heuristic_commit": retired.RETIRED_HEURISTIC_COMMIT,
                "main_demand_body": demand.body,
                "source_text": row.text,
                "corpus_status": row.status,
                "requested_source": requested_source,
                "nearest_recognized_heading": (
                    {
                        "path": list(owner.path),
                        "physical_start": owner.start,
                        "label": owner.label,
                    }
                    if owner is not None
                    else None
                ),
                "review_required": (
                    "Verify bytes, scope, and owner or complete tombstone. "
                    "Reject inline titles, unknown marker chains, and ambiguous "
                    "typography. The heuristic is not ownership evidence."
                ),
            },
        }


def add_release_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--release", required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--release-sha256")
    parser.add_argument("--release-object", type=Path)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--artifact-start", type=int, default=0)
    parser.add_argument("--artifact-stop", type=int)


def requested_scope_for_source(
    citation_path: str, requested_source: str | None
) -> list[str] | None:
    if requested_source is None:
        return None
    target = require_canonical_corpus_citation_path(
        vp._normalize_requested_source_to_corpus_path(requested_source)
    )
    if target != citation_path and not target.startswith(citation_path + "/"):
        raise ValueError("Requested source must be this exact row or its descendant")
    scope = vp._scope_prefix_from_requested_source(
        citation_path, requested_source, preserve_case=True
    )
    return list(scope) if scope else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_release_arguments(parser)
    parser.add_argument("--citation", help="Propose only for this exact row citation")
    parser.add_argument(
        "--requested-source",
        help="Actual requested citation when inspecting a parent-fallback row",
    )
    args = parser.parse_args(argv)
    if args.requested_source is not None and args.citation is None:
        parser.error("--requested-source requires --citation")
    corpus = select_artifacts(
        load_release(
            args.corpus_path, args.release, args.release_sha256, args.release_object
        ),
        args.artifact_manifest,
    )
    count = 0
    for row in iter_release_rows(
        corpus, artifact_start=args.artifact_start, artifact_stop=args.artifact_stop
    ):
        if args.citation is not None and row.citation_path != args.citation:
            continue
        scope = requested_scope_for_source(row.citation_path, args.requested_source)
        for candidate in propose_row(
            row, scope, requested_source=args.requested_source
        ):
            candidate["proposal_context"]["release"] = corpus.release.name
            candidate["proposal_context"]["release_sha256"] = (
                corpus.release.content_sha256
            )
            print(json.dumps(candidate, ensure_ascii=True))
            count += 1
    print(f"Emitted {count} unapproved candidates.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
