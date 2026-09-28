"""Compare a resolver snapshot with the working tree against local corpus anchors.

The corpus is read only. Example (save the baseline before editing the resolver)::

    git show BASE:src/axiom_encode/corpus_resolver.py > /tmp/resolver-before.py
    uv run python scripts/check_cfr_slice_differential.py \
        --baseline-resolver /tmp/resolver-before.py \
        --corpus-root ../axiom-corpus --output /tmp/cfr-differential.json

The bounded sweep includes actual anchor requests, then applies anchor-derived
child paths to a deterministic sample of section bodies from both document
classes. This permits a useful sweep even when a local corpus has fewer than
2,000 anchors; generated requests are explicitly counted separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from axiom_encode import corpus_resolver

SNAP_ANCHORS = "anchors/us/regulation/2026-05-10-snap-7-cfr-273.jsonl"
SNAP_PROVISIONS = (
    "provisions/us/regulation/"
    "2026-05-10-snap-7-cfr-273-r2026-07-15-self-contained.jsonl"
)
MAX_SECTION_CHARS = 16_000


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalized(text: str | None) -> str:
    return " ".join((text or "").split())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_baseline(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_cfr_resolver_baseline", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load baseline resolver: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def memoize_pure_helpers(module: ModuleType) -> None:
    """Reuse immutable helper results; never cache mutable parsing state.

    Every anchor in a section replays the same delimiter scan and reference
    context checks. Caching by every argument changes neither result nor parser
    state, and keeps the independent oracle affordable on large CFR sections.
    """
    for name in (
        "_parenthetical_marker_is_structural",
        "_parenthetical_marker_has_strong_boundary",
        "_marker_follows_explicit_reference_cue",
    ):
        setattr(module, name, lru_cache(maxsize=100_000)(getattr(module, name)))
    delimiter_scan = lru_cache(maxsize=1_024)(module._delimiter_enclosed_marker_starts)

    def cached_delimiter_scan(text: str, marker_starts: Iterable[int]) -> frozenset[int]:
        return delimiter_scan(text, tuple(marker_starts))

    module._delimiter_enclosed_marker_starts = cached_delimiter_scan


@dataclass(frozen=True)
class Section:
    citation_path: str
    document_class: str
    body: str
    body_sha256: str
    provision_file: str


@dataclass(frozen=True)
class Request:
    section: Section
    fragments: tuple[str, ...]
    anchored: bool

    @property
    def citation_path(self) -> str:
        return "/".join((self.section.citation_path, *self.fragments))

    @property
    def key(self) -> tuple[str, str, tuple[str, ...]]:
        return self.section.citation_path, self.section.body_sha256, self.fragments


@cache
def slice_result(module: ModuleType, request: Request) -> dict[str, Any]:
    try:
        text = module._slice_us_legal_hierarchy(
            request.section.body,
            request.fragments,
            document_class=request.section.document_class,
            ancestor_fragments=(),
        )
    except module.CorpusSourceSliceError as exc:
        return {"status": "error", "error": str(exc)}
    if text is None:
        return {"status": "missing"}
    return {"status": "success", "text": text}


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    if result["status"] != "success":
        return result
    text = result["text"]
    return {
        "status": "success",
        "chars": len(text),
        "sha256": sha256(text),
        "start": text[:160],
        "end": text[-160:],
    }


def sections_and_anchors(
    root: Path,
    *,
    include_recovery_blocks: bool = False,
) -> tuple[list[Section], list[tuple[str, dict[str, Any]]], dict[str, str]]:
    sections = {}
    anchors = []
    hashes = {}
    for document_class in ("statute", "regulation"):
        for path in sorted((root / "provisions/us" / document_class).rglob("*.jsonl")):
            relative_path = str(path.relative_to(root))
            hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in read_rows(path):
                body = row.get("body")
                citation = row.get("citation_path", "")
                parts = citation.split("/")
                section_depth = 4 if document_class == "statute" else 5
                if (
                    not body
                    or len(parts) != section_depth
                    or parts[:2] != ["us", document_class]
                    or (not include_recovery_blocks and not parts[2].isdigit())
                ):
                    continue
                section = Section(
                    citation, document_class, body, sha256(body), relative_path
                )
                sections.setdefault((citation, section.body_sha256), section)
        for path in sorted((root / "anchors/us" / document_class).rglob("*.jsonl")):
            relative_path = str(path.relative_to(root))
            hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
            anchors.extend((relative_path, row) for row in read_rows(path))
    return list(sections.values()), anchors, hashes


def anchor_request(anchor: dict[str, Any], section: Section) -> Request:
    parent = anchor["parent_citation_path"]
    citation = anchor["citation_path"]
    if not citation.startswith(parent + "/"):
        raise ValueError(f"Anchor is not a child of its parent: {citation}")
    return Request(section, tuple(citation[len(parent) + 1 :].split("/")), True)


def compare_snap(
    root: Path, baseline: ModuleType, anchors: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    snap_anchors = [anchor for path, anchor in anchors if path == SNAP_ANCHORS]
    if not (root / SNAP_ANCHORS).is_file() or len(snap_anchors) != 192:
        raise ValueError(
            f"SNAP oracle requires {SNAP_ANCHORS} with exactly 192 anchors; "
            f"found {len(snap_anchors)}"
        )
    rows = {row["citation_path"]: row for row in read_rows(root / SNAP_PROVISIONS)}
    before_matches = []
    after_matches = []
    mismatches = []
    for anchor in snap_anchors:
        row = rows[anchor["parent_citation_path"]]
        body = row["body"]
        section = Section(
            row["citation_path"], "regulation", body, sha256(body), SNAP_PROVISIONS
        )
        if anchor["parent_body_sha256"] != section.body_sha256:
            raise ValueError(f"SNAP anchor body hash differs: {anchor['citation_path']}")
        request = anchor_request(anchor, section)
        before = slice_result(baseline, request)
        after = slice_result(corpus_resolver, request)
        before_match = normalized(before.get("text")) == normalized(anchor["text"])
        after_match = normalized(after.get("text")) == normalized(anchor["text"])
        if before_match:
            before_matches.append(request.citation_path)
        if after_match:
            after_matches.append(request.citation_path)
        else:
            start, end = anchor["char_start"], anchor["char_end"]
            mismatches.append(
                {
                    "citation_path": request.citation_path,
                    "char_start": start,
                    "char_end": end,
                    "anchor_matches_source_span": normalized(body[start:end])
                    == normalized(anchor["text"]),
                    "anchor_text": anchor["text"],
                    "before": summarize(before),
                    "after": summarize(after),
                }
            )
    return {
        "total": len(snap_anchors),
        "matched_before": len(before_matches),
        "matched_after": len(after_matches),
        "lost_matches": sorted(set(before_matches) - set(after_matches)),
        "gained_matches": sorted(set(after_matches) - set(before_matches)),
        "remaining_mismatches": mismatches,
    }


def build_sweep(
    sections: list[Section],
    anchors: list[tuple[str, dict[str, Any]]],
    *,
    sample_size: int,
    seed: int,
) -> tuple[list[Request], list[str]]:
    by_hash = {(s.citation_path, s.body_sha256): s for s in sections}
    requests = {}
    unavailable = []
    patterns: dict[str, set[tuple[str, ...]]] = {"statute": set(), "regulation": set()}
    for _file, anchor in anchors:
        section = by_hash.get(
            (anchor["parent_citation_path"], anchor["parent_body_sha256"])
        )
        if section is None:
            unavailable.append(anchor["citation_path"])
            continue
        request = anchor_request(anchor, section)
        requests.setdefault(request.key, request)
        patterns[section.document_class].add(request.fragments)
    rng = random.Random(seed)
    candidates: dict[str, list[Request]] = {"statute": [], "regulation": []}
    for section in sorted(sections, key=lambda s: (s.citation_path, s.body_sha256)):
        if len(section.body) > MAX_SECTION_CHARS:
            continue
        for fragments in sorted(patterns[section.document_class]):
            # Require each component to appear in the section to make the
            # generated requests useful without trusting either slicer.
            if all(f"({fragment})" in section.body for fragment in fragments):
                request = Request(section, fragments, False)
                if request.key not in requests:
                    candidates[section.document_class].append(request)
    for values in candidates.values():
        rng.shuffle(values)
    while len(requests) < sample_size and any(candidates.values()):
        for document_class in ("statute", "regulation"):
            if candidates[document_class] and len(requests) < sample_size:
                request = candidates[document_class].pop()
                requests.setdefault(request.key, request)
    if len(requests) < sample_size:
        raise ValueError(f"Only {len(requests)} distinct requests; need {sample_size}")
    return list(requests.values()), unavailable


def compare_sweep(baseline: ModuleType, requests: list[Request]) -> dict[str, Any]:
    statuses_before: Counter[str] = Counter()
    statuses_after: Counter[str] = Counter()
    regressions = []
    improvements = []
    for request in requests:
        before = slice_result(baseline, request)
        after = slice_result(corpus_resolver, request)
        statuses_before[before["status"]] += 1
        statuses_after[after["status"]] += 1
        evidence = {
            "citation_path": request.citation_path,
            "section_body_sha256": request.section.body_sha256,
            "provision_file": request.section.provision_file,
            "anchored": request.anchored,
            "before": summarize(before),
            "after": summarize(after),
        }
        if before["status"] == "success":
            if after["status"] != "success" or before["text"] != after["text"]:
                regressions.append(evidence)
        elif after["status"] == "success":
            improvements.append(evidence)
    return {
        "requests": len(requests),
        "actual_anchor_requests": sum(request.anchored for request in requests),
        "anchor_derived_requests": sum(not request.anchored for request in requests),
        "document_classes": dict(Counter(r.section.document_class for r in requests)),
        "distinct_section_bodies": len(
            {(r.section.citation_path, r.section.body_sha256) for r in requests}
        ),
        "statuses_before": dict(statuses_before),
        "statuses_after": dict(statuses_after),
        "unchanged_successes": statuses_before["success"] - len(regressions),
        "regressions": regressions,
        "new_successes": improvements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-resolver", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument(
        "--no-memoize", action="store_true", help="Disable pure-helper memoization"
    )
    parser.add_argument(
        "--include-recovery-blocks",
        action="store_true",
        help="Also sample recovery containers with section-shaped path depth",
    )
    args = parser.parse_args()
    if args.sample_size < 2_000:
        parser.error("--sample-size must be at least 2000")
    root = args.corpus_root / "data/corpus"
    baseline = load_baseline(args.baseline_resolver)
    if not args.no_memoize:
        memoize_pure_helpers(baseline)
        memoize_pure_helpers(corpus_resolver)
    sections, anchors, hashes = sections_and_anchors(
        root, include_recovery_blocks=args.include_recovery_blocks
    )
    requests, unavailable = build_sweep(
        sections, anchors, sample_size=args.sample_size, seed=args.seed
    )
    snap = compare_snap(root, baseline, anchors)
    print(
        f"SNAP: {snap['matched_before']}/{snap['total']} -> "
        f"{snap['matched_after']}/{snap['total']}; "
        f"lost matches: {len(snap['lost_matches'])}",
        flush=True,
    )
    report = {
        "baseline_resolver_sha256": sha256(args.baseline_resolver.read_text()),
        "current_resolver_sha256": sha256(Path(corpus_resolver.__file__).read_text()),
        "seed": args.seed,
        "memoize_pure_helpers": not args.no_memoize,
        "include_recovery_blocks": args.include_recovery_blocks,
        "generated_request_max_section_chars": MAX_SECTION_CHARS,
        "input_files_sha256": hashes,
        "unavailable_anchor_parents": unavailable,
        "snap": snap,
        "sweep": compare_sweep(baseline, requests),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    snap, sweep = report["snap"], report["sweep"]
    print(
        f"Sweep: {sweep['requests']} requests; "
        f"{sweep['unchanged_successes']}/{sweep['statuses_before'].get('success', 0)} "
        f"baseline successes byte-identical; "
        f"{len(sweep['new_successes'])} new successes; "
        f"{len(sweep['regressions'])} changed baseline successes"
    )
    print(f"Detailed evidence: {args.output}")
    return int(
        bool(snap["lost_matches"])
        or snap["matched_after"] < snap["matched_before"]
        or bool(sweep["regressions"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
