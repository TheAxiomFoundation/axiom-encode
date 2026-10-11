"""Compare locator recall against a supplied baseline on a reproducible sample.

Select the first two nonempty body rows per corpus file, plus all body rows
in the three statute issue files and the retained Maryland guidance control.
Retain rendered sources no longer than 12,000 characters and always include body
rows at or below the four selected citation roots. Large document rows make
the existing numeric extractor expensive. The manifest, every excluded row,
and every changed numeric occurrence are written to the requested output.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from axiom_encode.harness import source_completeness as current
from axiom_encode.harness.validator_pipeline import (
    _numeric_profile_for_citation_path,
    extract_typed_numeric_inventory_occurrences_from_text,
)

ISSUE_FILES = {
    "us-ok/statute/2026-07-16-pit-central-us-ok-title-68-r2026-07-24-immutable.jsonl",
    "us-az/statute/2026-07-13-recovery.jsonl",
    "us-va/statute/2026-07-13-recovery.jsonl",
    "us-md/guidance/2026-07-12-md-fia-snap-fy2026.jsonl",
}


TARGET_CITATIONS = {
    "us-ok/statute/68-2906",
    "us-az/statute/43-1072",
    "us-va/statute/58.1/58.1-322.03",
    "us-md/guidance/dhs/fia/snap-manual-214/page-5",
}


def rendered_source(row):
    return "\n\n".join(part for part in (row.get("heading"), row["body"]) if part)


def bounded_rows(rows, max_source_chars):
    selected, excluded = [], []
    for file, line, row in rows:
        length = len(rendered_source(row))
        if length <= max_source_chars or any(
            row["citation_path"] == target
            or row["citation_path"].startswith(target + "/")
            for target in TARGET_CITATIONS
        ):
            selected.append((file, line, row))
        else:
            excluded.append(
                {
                    "file": file,
                    "line": line,
                    "citation": row["citation_path"],
                    "source_chars": length,
                }
            )
    return selected, excluded


def load_baseline(path: Path):
    name = "axiom_encode.harness._locator_audit_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_computation_unchanged(baseline_path: Path):
    """Verify that the numeric masks leave the computation predicate unchanged."""

    def predicate(path):
        tree = ast.parse(path.read_text())
        return next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "source_states_explicit_computation"
        )

    old = predicate(baseline_path)
    new = predicate(Path(current.__file__))
    if ast.dump(old, include_attributes=False) != ast.dump(
        new, include_attributes=False
    ):
        raise ValueError("Computation predicate differs from the supplied baseline")


def tracked_provision_files(root: Path):
    return sorted(
        subprocess.check_output(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(root),
                "ls-files",
                "--",
                "*.jsonl",
            ],
            text=True,
        ).splitlines()
    )


def select_rows(root: Path, relative_files):
    selected = []
    files = []
    total = 0
    for file_index, relative in enumerate(relative_files, 1):
        path = root / relative
        digest = hashlib.sha256()
        choices = []
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                total += 1
                if relative not in ISSUE_FILES and len(choices) == 2:
                    continue
                row = json.loads(raw)
                if not isinstance(row.get("body"), str) or not row["body"].strip():
                    continue
                entry = (relative, line_number, row)
                choices.append(entry)
        files.append({"file": relative, "sha256": digest.hexdigest()})
        selected.extend(choices)
        if file_index % 100 == 0:
            print(f"Selected from {file_index} files", flush=True)
    return sorted(selected, key=lambda item: (item[0], item[1])), files, total


def numeric_tokens(recall, citation):
    profile = _numeric_profile_for_citation_path(citation)
    occurrences = extract_typed_numeric_inventory_occurrences_from_text(
        recall, profile=profile
    )
    return Counter((item.raw, item.value) for item in occurrences)


def serialize_tokens(tokens):
    return [
        {"raw": raw, "value": value, "count": count}
        for (raw, value), count in sorted(tokens.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-sample", type=Path)
    parser.add_argument("--max-source-chars", type=int, default=12000)
    args = parser.parse_args()
    started = time.monotonic()
    baseline = load_baseline(args.baseline)
    verify_computation_unchanged(args.baseline)
    relative_files = tracked_provision_files(args.corpus)
    args.output.with_suffix(".tracked-files.txt").write_text(
        "\n".join(relative_files) + "\n"
    )
    if args.reuse_sample:
        sample = json.loads(args.reuse_sample.read_text())
        rows, files, total = sample["rows"], sample["files"], sample["total"]
        if relative_files != sorted(item["file"] for item in files):
            raise ValueError("Saved sample differs from tracked corpus file manifest")
    else:
        rows, files, total = select_rows(args.corpus, relative_files)
        args.output.with_suffix(".sample.json").write_text(
            json.dumps({"rows": rows, "files": files, "total": total}) + "\n"
        )
    unbounded_selected_body_rows = len(rows)
    rows, excluded = bounded_rows(rows, args.max_source_chars)
    args.output.with_suffix(".bounded.sample.json").write_text(
        json.dumps({"rows": rows, "files": files, "total": total}) + "\n"
    )
    result = {
        "selection": "first two nonempty body rows per file and all body rows "
        "in ISSUE_FILES, retaining rendered sources at or below max_source_chars; "
        "always include body rows at or below the four TARGET_CITATIONS roots",
        "max_source_chars": args.max_source_chars,
        "target_citations": sorted(TARGET_CITATIONS),
        "unbounded_selected_body_rows": unbounded_selected_body_rows,
        "excluded_oversize_rows": excluded,
        "issue_files": sorted(ISSUE_FILES),
        "files": files,
        "total_jsonl_rows": total,
        "selected_body_rows": len(rows),
        "baseline_module_sha256": hashlib.sha256(
            args.baseline.read_bytes()
        ).hexdigest(),
        "current_module_sha256": hashlib.sha256(
            Path(current.__file__).read_bytes()
        ).hexdigest(),
        "manifest": [
            {
                "file": file,
                "line": line,
                "citation": row["citation_path"],
            }
            for file, line, row in rows
        ],
        "computation_method": "whole-source explicit-computation boolean on "
        "both versions for every selected row. Exact predicate AST equality "
        "confirms the computation-predicate function is unchanged.",
        "whole_source_computation_counts": {"before": 0, "after": 0},
        "whole_source_computation_changes": [],
        "changes": [],
        "errors": [],
    }
    print(
        json.dumps(
            {
                "selected_body_rows": len(rows),
                "excluded_oversize_rows": len(excluded),
                "max_source_chars": args.max_source_chars,
            }
        ),
        flush=True,
    )
    # Checkpoint the selection before running potentially expensive analyses.
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for index, (file, line, row) in enumerate(rows, 1):
        citation = row["citation_path"]
        source = rendered_source(row)
        try:
            old_recall = baseline.authoritative_numeric_recall_text(
                source, corpus_citation_path=citation
            )
            new_recall = current.authoritative_numeric_recall_text(
                source, corpus_citation_path=citation
            )
            # Both branches use the same deterministic numeric extractor.
            # Equal recall strings prove equal demands without tokenizing huge
            # unchanged documents twice (the legacy word-number regex is slow).
            old_tokens = new_tokens = Counter()
            if old_recall != new_recall:
                old_tokens = numeric_tokens(old_recall, citation)
                new_tokens = numeric_tokens(new_recall, citation)
            old_whole = baseline.source_states_explicit_computation(source)
            new_whole = current.source_states_explicit_computation(source)
            result["whole_source_computation_counts"]["before"] += old_whole
            result["whole_source_computation_counts"]["after"] += new_whole
            if old_whole != new_whole:
                result["whole_source_computation_changes"].append(
                    {
                        "file": file,
                        "line": line,
                        "citation": row["citation_path"],
                        "before": old_whole,
                        "after": new_whole,
                    }
                )
            if old_tokens != new_tokens:
                result["changes"].append(
                    {
                        "file": file,
                        "line": line,
                        "citation": row["citation_path"],
                        "removed_numeric": serialize_tokens(old_tokens - new_tokens),
                        "added_numeric": serialize_tokens(new_tokens - old_tokens),
                        "old_numeric_count": sum(old_tokens.values()),
                        "new_numeric_count": sum(new_tokens.values()),
                        "old_recall": old_recall,
                        "new_recall": new_recall,
                    }
                )
        except Exception as error:
            result["errors"].append(
                {"file": file, "line": line, "citation": citation, "error": repr(error)}
            )
        if index % 100 == 0:
            result["completed_rows"] = index
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"{index}/{len(rows)} rows; {len(result['changes'])} changed",
                flush=True,
            )
    result["completed_rows"] = len(rows)
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("total_jsonl_rows", "selected_body_rows", "elapsed_seconds")
            }
            | {"changed_rows": len(result["changes"]), "errors": len(result["errors"])}
        )
    )
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
