#!/usr/bin/env python3
"""Check reviewed exemptions against pinned corpus bytes and record a ledger.

Artifact ranges permit bounded, repeatable shards. A shard records untested
entries; only a complete selected range certifies that no selected entry is
stale. --entries-only is the small CI stale-entry check, not a full release
differential. Full differentials omit it and may use an explicit snapshot
artifact manifest, whose bytes are checked against the signed release object.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from axiom_encode.harness import coverage_exemptions as registry

try:
    from . import propose_coverage_exemptions as proposal
except ImportError:
    import propose_coverage_exemptions as proposal


def check_rows(
    corpus: proposal.ReleaseCorpus,
    exemptions: Sequence[registry.CoverageExemption],
    *,
    artifact_start: int = 0,
    artifact_stop: int | None = None,
    entries_only: bool = False,
) -> dict[str, Any]:
    """Compare the actual filter with independently computed literal matches."""
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    validated_entries: set[tuple[Any, ...]] = set()
    matched_entries: set[tuple[Any, ...]] = set()
    selected = corpus.artifacts[artifact_start:artifact_stop]
    selected_paths = {artifact.path for artifact in selected}
    expected_entries = {
        entry.key for entry in exemptions if entry.corpus_artifact in selected_paths
    }
    by_row: dict[tuple[str, int], list[registry.CoverageExemption]] = {}
    by_citation: dict[str, list[registry.CoverageExemption]] = {}
    for entry in exemptions:
        by_row.setdefault((entry.corpus_artifact, entry.row), []).append(entry)
        by_citation.setdefault(entry.citation_path, []).append(entry)
    for row in proposal.iter_release_rows(
        corpus, artifact_start=artifact_start, artifact_stop=artifact_stop
    ):
        counts["rows"] += 1
        counts["nonempty_rows"] += bool(row.text)
        counts["source_bytes"] += len(row.text.encode("utf-8"))
        demands = proposal.scan_main_demands(row.text)
        counts["main_physical_demands"] += len(demands)
        counts["main_row_labels"] += len({demand.label for demand in demands})
        text_digest = proposal.sha256_text(row.text)
        demand_keys = {(demand.label, demand.physical_start) for demand in demands}
        for entry in by_row.get((row.corpus_artifact, row.row), ()):
            if (
                entry.citation_path == row.citation_path
                and entry.artifact_sha256 == row.artifact_sha256
                and entry.source_text_sha256 == text_digest
                and (entry.label, entry.physical_start) in demand_keys
                and (
                    entry.requested_scope is None
                    or (entry.label,)
                    == proposal.vp._top_level_subparagraph_path(entry.requested_scope)
                )
                and entry.window
                == registry.source_window(row.text, entry.physical_start)
                and entry.window_sha256 == proposal.sha256_text(entry.window)
            ):
                validated_entries.add(entry.key)
            else:
                errors.append({"kind": "stale_entry", "key": entry.key})
        # Check the ordinary section target and every explicitly reviewed target.
        scopes = {None} | {
            entry.requested_scope for entry in by_citation.get(row.citation_path, ())
        }
        for scope in scopes:
            scoped_demands = (
                demands
                if scope is None
                else [
                    demand
                    for demand in demands
                    if (demand.label,)
                    == proposal.vp._top_level_subparagraph_path(scope)
                ]
            )
            scoped_keys = {
                (demand.label, demand.physical_start) for demand in scoped_demands
            }
            expected = {
                (entry.label, entry.physical_start)
                for entry in exemptions
                if entry.citation_path == row.citation_path
                and entry.requested_scope == scope
                and entry.source_text_sha256 == text_digest
                and (entry.label, entry.physical_start) in scoped_keys
                and entry.window
                == registry.source_window(row.text, entry.physical_start)
                and entry.window_sha256 == proposal.sha256_text(entry.window)
            }
            after = registry.filter_coverage_demands(
                scoped_demands,
                citation_path=row.citation_path,
                requested_scope=scope,
                source_text=row.text,
                exemptions=exemptions,
            )
            after_keys = {(demand.label, demand.physical_start) for demand in after}
            removed = scoped_keys - after_keys
            added = after_keys - scoped_keys
            counts["evaluated_scopes"] += 1
            counts["expected_physical_removals"] += len(expected)
            counts["physical_removals"] += len(removed)
            counts["physical_additions"] += len(added)
            counts["row_labels_removed"] += len(
                {demand.label for demand in scoped_demands}
                - {demand.label for demand in after}
            )
            if removed:
                counts["rows_with_physical_removals"] += 1
            if removed != expected or added:
                errors.append(
                    {
                        "kind": "differential_mismatch",
                        "corpus_artifact": row.corpus_artifact,
                        "row": row.row,
                        "requested_scope": scope,
                        "expected": sorted(expected),
                        "removed": sorted(removed),
                        "added": sorted(added),
                    }
                )
            for entry in by_citation.get(row.citation_path, ()):
                if (
                    entry.requested_scope == scope
                    and (
                        entry.label,
                        entry.physical_start,
                    )
                    in expected
                ):
                    matched_entries.add(entry.key)
    missing = expected_entries - validated_entries
    for key in sorted(missing, key=repr):
        if not any(error.get("key") == key for error in errors):
            errors.append({"kind": "unseen_entry", "key": key})
    complete = artifact_start == 0 and (
        artifact_stop is None or artifact_stop == len(corpus.artifacts)
    )
    if complete:
        for entry in exemptions:
            if entry.corpus_artifact not in selected_paths:
                errors.append(
                    {"kind": "entry_artifact_outside_checked_slice", "key": entry.key}
                )
    # Explicit zeros keep independent ledger merges and review comparisons clear.
    for field in (
        "rows",
        "nonempty_rows",
        "source_bytes",
        "main_physical_demands",
        "main_row_labels",
        "evaluated_scopes",
        "expected_physical_removals",
        "physical_removals",
        "physical_additions",
        "row_labels_removed",
        "rows_with_physical_removals",
    ):
        counts.setdefault(field, 0)
    return {
        "schema": "axiom-encode/coverage-exemption-ledger/v1",
        "release": corpus.release.name,
        "release_sha256": corpus.release.content_sha256,
        "release_commit": corpus.release.git_commit,
        "scope": "entry artifacts only"
        if entries_only
        else "selected release artifacts",
        "complete_selected_range": complete,
        "artifact_start": artifact_start,
        "artifact_stop": artifact_start + len(selected),
        "checked_artifacts": len(selected),
        "selected_artifacts": len(corpus.artifacts),
        "release_provision_artifacts": sum(
            artifact.artifact_class == "provisions"
            for artifact in corpus.release.artifacts
        ),
        "approved_entries": len(exemptions),
        "entries_validated": len(validated_entries),
        "entries_matched": len(matched_entries),
        "entries_untested": len(exemptions) - len(expected_entries),
        "counts": dict(counts),
        "errors": errors,
        "passed": not errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    proposal.add_release_arguments(parser)
    parser.add_argument(
        "--entries", type=Path, help="Alternate approved JSONL registry"
    )
    parser.add_argument(
        "--listed-artifacts-only",
        "--entries-only",
        dest="entries_only",
        action="store_true",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    corpus = proposal.select_artifacts(
        proposal.load_release(
            args.corpus_path, args.release, args.release_sha256, args.release_object
        ),
        args.artifact_manifest,
    )
    exemptions = (
        tuple(
            registry.CoverageExemption.from_mapping(json.loads(line))
            for line in args.entries.read_text(encoding="utf-8").splitlines()
            if line
        )
        if args.entries is not None
        else registry.load_coverage_exemptions()
    )
    if len({entry.key for entry in exemptions}) != len(exemptions):
        raise ValueError("Duplicate approved exemption key")
    if args.entries_only:
        paths = {entry.corpus_artifact for entry in exemptions}
        corpus = proposal.ReleaseCorpus(
            corpus.root,
            corpus.release,
            tuple(artifact for artifact in corpus.artifacts if artifact.path in paths),
        )
    ledger = check_rows(
        corpus,
        exemptions,
        artifact_start=args.artifact_start,
        artifact_stop=args.artifact_stop,
        entries_only=args.entries_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(
        f"{ledger['entries_validated']} entries validated; "
        f"{ledger['counts']['physical_removals']} removals; "
        f"{ledger['counts']['physical_additions']} additions; "
        f"{len(ledger['errors'])} errors.",
        file=sys.stderr,
    )
    return 0 if ledger["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
