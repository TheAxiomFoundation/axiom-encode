"""Audit every statute body in a pinned NZ release against the pre-change code.

Run repeatedly with the same output directory to resume bounded batches. The
checkpoint lists exact provision files; this command never searches the corpus.
Every change is retained in JSONL, including numeric label evidence and each
before/after gate obligation. All public artifacts use corpus-relative names.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from axiom_encode import corpus_resolver as current_resolver
from axiom_encode.bare_structure import bare_outline, mask_bare_labels
from axiom_encode.harness import source_completeness as current
from axiom_encode.harness.validator_pipeline import (
    _numeric_profile_for_citation_path,
    extract_typed_numeric_inventory_occurrences_from_text,
)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_baseline(ref: str, relative: str, module_name: str) -> Any:
    source = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "show", f"{ref}:{relative}"],
        text=True,
    )
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = relative
    sys.modules[module_name] = module
    exec(compile(source, relative, "exec"), module.__dict__)
    return module


def collapse(text: str) -> str:
    return " ".join(text.split())


def branch_record(branch: Any) -> dict[str, Any]:
    return {
        "path": list(branch.path),
        "kind": branch.kind,
        "label": branch.label,
        "start": branch.start,
        "end": branch.end,
        "text": collapse(branch.text),
    }


def occurrence_record(item: Any) -> dict[str, Any]:
    return {
        "raw": item.raw,
        "value": item.value,
        "start": item.start,
        "end": item.end,
    }


def anchored_occurrences(
    text: str, recall: str, occurrences: Any
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Restore corpus offsets after the legacy recall cleanup deletes citations."""
    blocks = difflib.SequenceMatcher(
        a=text, b=recall, autojunk=False
    ).get_matching_blocks()
    result = {}
    for item in occurrences:
        block = next(
            (
                block
                for block in blocks
                if block.b <= item.start and item.end <= block.b + block.size
            ),
            None,
        )
        start = None if block is None else block.a + item.start - block.b
        end = None if start is None else start + item.end - item.start
        record = {**occurrence_record(item), "source_start": start, "source_end": end}
        result[(start, end, item.raw, item.value)] = record
    return result


def gates(module: Any, text: str, citation: str, extractor: Any) -> dict[str, Any]:
    branches = module.recognize_source_structure(text, corpus_citation_path=citation)
    shared = {
        "branches": branches,
        "active_branches": branches,
        "deferred_paths": set(),
    }
    formula = module._source_formula_branches(text, **shared)
    exceptions = module._source_exception_branches(text, **shared)
    control = module._source_control_branches(
        text,
        **shared,
        formula_branches=formula,
        extract_numeric_occurrences=extractor,
    )
    return {
        "branches": branches,
        "clauses": tuple(
            module.SourceStructureBranch(
                (), "excerpt", "source clause", value, start, end
            )
            for start, end, value in module._source_clause_spans(
                text, branches=branches
            )
        ),
        "tests_conditions": exceptions,
        "formula": formula,
        "control": control,
    }


def structural_removal_evidence(
    removed: Any, replacements: Any, clauses: Any, text: str, outline: Any, gate: str
) -> dict[str, Any]:
    """Account for old clauses split/cleaned or assigned to an authentic owner."""
    affected = [
        label for label in outline if removed.start <= label.start < removed.end
    ]
    overlapping = [
        item
        for item in replacements
        if item.start < removed.end and removed.start < item.end
    ]
    masked = mask_bare_labels(text, outline)
    clean_text = collapse(masked[removed.start : removed.end])
    retained_clauses = [
        item
        for item in clauses
        if item.start < removed.end and removed.start < item.end
    ]
    # Check source offsets and actual excerpt characters, not merely whether a
    # changed interval happens to contain a structural marker. Every operative
    # character from the former obligation must remain in an extracted clause.
    uncovered = [
        offset
        for offset in range(removed.start, removed.end)
        if not masked[offset].isspace()
        and not any(
            item.start <= offset < item.end
            and item.text[offset - item.start] == masked[offset]
            for item in retained_clauses
        )
    ]
    aligned_replacements = [
        item
        for item in overlapping
        if collapse(item.text) == collapse(masked[item.start : item.end])
    ]
    uncovered_gate_offsets = [
        offset
        for offset in range(removed.start, removed.end)
        if not masked[offset].isspace()
        and not any(
            item.start <= offset < item.end
            for item in aligned_replacements
        )
    ]
    ownership_only = any(
        collapse(item.text) == collapse(removed.text) and item.path != removed.path
        for item in overlapping
    )
    return {
        "authenticated_labels": [
            {"token": label.token, "path": list(label.path), "start": label.start}
            for label in affected
        ],
        "cleaned_original": clean_text,
        "overlapping_replacements": [branch_record(item) for item in overlapping],
        "retained_source_clauses": [branch_record(item) for item in retained_clauses],
        "uncovered_operative_offsets": uncovered,
        "uncovered_same_gate_operative_offsets": uncovered_gate_offsets,
        "ownership_only": ownership_only,
        "structural_artifact": bool(affected or ownership_only)
        and bool(overlapping)
        and not uncovered
        and (gate == "tests_conditions" or not uncovered_gate_offsets),
    }


def compare_gates(old: Any, new: Any, text: str, outline: Any) -> dict[str, Any]:
    changes = {}

    def key(item: Any) -> tuple[Any, ...]:
        return item.path, item.start, item.end, collapse(item.text)

    for gate in ("branches", "clauses", "tests_conditions", "formula", "control"):
        old_by_key = {key(item): item for item in old[gate]}
        new_by_key = {key(item): item for item in new[gate]}
        removed = [old_by_key[item] for item in old_by_key.keys() - new_by_key.keys()]
        added = [new_by_key[item] for item in new_by_key.keys() - old_by_key.keys()]
        if removed or added:
            changes[gate] = {
                "removed": [
                    {
                        **branch_record(item),
                        "evidence": structural_removal_evidence(
                            item, new[gate], new["clauses"], text, outline, gate
                        ),
                    }
                    for item in sorted(
                        removed, key=lambda item: (item.start, item.path)
                    )
                ],
                "added": [
                    branch_record(item)
                    for item in sorted(added, key=lambda item: (item.start, item.path))
                ],
            }
    return changes


def glue_violations(new: Any, text: str, outline: Any) -> list[dict[str, Any]]:
    """Detect marker characters retained anywhere in an authenticated excerpt."""
    violations = []
    records = [
        (gate, item.start, item.end, item.text)
        for gate in ("branches", "clauses", "tests_conditions", "formula", "control")
        for item in new[gate]
    ]
    for gate, start, end, value in records:
        for label in outline:
            if not start <= label.start < label.marker_end <= end:
                continue
            # Offset-preserving masks allow direct checking; stripped excerpts
            # are checked against the canonical masked slice instead.
            expected = mask_bare_labels(text, outline)[start:end]
            if collapse(value) != collapse(expected):
                violations.append(
                    {
                        "gate": gate,
                        "path": list(label.path),
                        "token": label.token,
                        "text": collapse(value),
                    }
                )
    return violations


def slice_result(
    module: Any, text: str, citation: str, path: tuple[str, ...]
) -> str | None:
    try:
        return module._slice_parent_body(
            text,
            requested_path=citation + "/" + "/".join(path),
            resolved_path=citation,
        )
    except module.CorpusSourceSliceError:
        return None


def analyze_row(
    row: dict[str, Any], baseline: Any, baseline_resolver: Any
) -> dict[str, Any]:
    text, citation = row["body"], row["citation_path"]
    outline = bare_outline(text) or ()
    profile = _numeric_profile_for_citation_path(citation)

    def extractor(value: str) -> Any:
        return extract_typed_numeric_inventory_occurrences_from_text(
            value, profile=profile
        )

    old_recall = baseline.authoritative_numeric_recall_text(
        text, corpus_citation_path=citation
    )
    new_recall = current.authoritative_numeric_recall_text(
        text, corpus_citation_path=citation
    )
    old_numbers = anchored_occurrences(text, old_recall, extractor(old_recall))
    new_numbers = anchored_occurrences(text, new_recall, extractor(new_recall))
    removed = [old_numbers[key] for key in old_numbers.keys() - new_numbers.keys()]
    added = [new_numbers[key] for key in new_numbers.keys() - old_numbers.keys()]
    removed_records = []
    for item in sorted(removed, key=lambda item: item["start"]):
        label = next(
            (
                label
                for label in outline
                if item["source_start"] is not None
                and label.start
                <= item["source_start"]
                < item["source_end"]
                <= label.marker_end
            ),
            None,
        )
        removed_records.append(
            {
                **item,
                "authenticated_label": None
                if label is None
                else {
                    "token": label.token,
                    "path": list(label.path),
                    "line": text[label.start : label.marker_end],
                    "followed_by": text[label.marker_end :].splitlines()[1],
                },
            }
        )
    old_gates = gates(baseline, text, citation, extractor)
    new_gates = gates(current, text, citation, extractor)
    slice_changes = []
    for label in outline:
        before = slice_result(baseline_resolver, text, citation, label.path)
        after = slice_result(current_resolver, text, citation, label.path)
        slice_changes.append(
            {
                "path": list(label.path),
                "previously_sliced": before is not None,
                "now_slices": after is not None,
                "changed_existing_slice": before is not None and before != after,
                "excerpt": after,
            }
        )
    gate_changes = compare_gates(old_gates, new_gates, text, outline)
    old_values = {item["value"] for item in old_numbers.values()}
    new_values = {item["value"] for item in new_numbers.values()}
    # A former condition may split or acquire a real branch owner, but its cue
    # must still belong to an obligation. Arithmetic-only cues may be owned by
    # the existing formula gate instead, as in the normal complete-source walk.
    lost_condition_cues = [
        {
            "start": match.start(),
            "end": match.end(),
            "text": match.group(),
        }
        for match in baseline._source_exception_or_applicability_matches(text)
        if any(
            branch.start <= match.start() and match.end() <= branch.end
            for branch in old_gates["tests_conditions"]
        )
        and not any(
            branch.start <= match.start() and match.end() <= branch.end
            for branch in (*new_gates["tests_conditions"], *new_gates["formula"])
        )
    ]
    return {
        "citation": citation,
        "body_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_chars": len(text),
        "candidate_bare_lines": len(
            re.findall(r"(?m)^[ \t]*(?:[1-9]\d*[A-Z]?|[a-z]{1,3})[ \t]*$", text)
        ),
        "authenticated_labels": len(outline),
        "slices": slice_changes,
        "numeric_removed": removed_records,
        "numeric_added": sorted(added, key=lambda item: item["start"]),
        "numeric_demand_values_removed": sorted(old_values - new_values),
        "numeric_demand_values_added": sorted(new_values - old_values),
        "unanchored_numeric_occurrences_before": sum(
            item["source_start"] is None for item in old_numbers.values()
        ),
        "unanchored_numeric_occurrences_after": sum(
            item["source_start"] is None for item in new_numbers.values()
        ),
        "gate_changes": gate_changes,
        "lost_condition_cues": lost_condition_cues,
        "glue_violations": glue_violations(new_gates, text, outline),
        "ambiguous_unchanged": not outline
        and old_recall == new_recall
        and not gate_changes,
        "gate_counts_before": {key: len(value) for key, value in old_gates.items()},
        "gate_counts_after": {key: len(value) for key, value in new_gates.items()},
    }


def add_counts(counts: Counter[str], result: dict[str, Any]) -> None:
    counts["bodies"] += 1
    counts["authenticated_bodies"] += bool(result["authenticated_labels"])
    counts["candidate_bare_bodies"] += bool(result["candidate_bare_lines"])
    counts["ambiguous_bodies_unchanged"] += bool(
        result["candidate_bare_lines"] and result["ambiguous_unchanged"]
    )
    counts["authenticated_labels"] += result["authenticated_labels"]
    counts["numeric_occurrences_removed"] += len(result["numeric_removed"])
    counts["numeric_occurrences_added"] += len(result["numeric_added"])
    counts["numeric_demand_values_removed"] += len(
        result["numeric_demand_values_removed"]
    )
    counts["numeric_demand_values_added"] += len(result["numeric_demand_values_added"])
    counts["lost_condition_cues"] += len(result["lost_condition_cues"])
    for suffix in ("before", "after"):
        counts[f"unanchored_numeric_occurrences_{suffix}"] += result[
            f"unanchored_numeric_occurrences_{suffix}"
        ]
    counts["previously_failed_slicing_bodies_now_sliced"] += any(
        not item["previously_sliced"] and item["now_slices"]
        for item in result["slices"]
    )
    for suffix in ("before", "after"):
        for gate, value in result[f"gate_counts_{suffix}"].items():
            counts[f"{gate}_obligations_{suffix}"] += value
    counts["numeric_removals_without_label_evidence"] += sum(
        item["authenticated_label"] is None for item in result["numeric_removed"]
    )
    counts["glue_violations"] += len(result["glue_violations"])
    counts["ambiguous_bodies_changed"] += bool(
        not result["authenticated_labels"] and not result["ambiguous_unchanged"]
    )
    for item in result["slices"]:
        counts["requested_paths"] += 1
        counts["previously_failed_paths_now_sliced"] += bool(
            not item["previously_sliced"] and item["now_slices"]
        )
        counts["recognized_paths_still_failing"] += not item["now_slices"]
        counts["existing_slices_changed"] += item["changed_existing_slice"]
        path = "/".join(item["path"])
        if not item["previously_sliced"] and item["now_slices"]:
            counts[f"new_slice_path:{path}"] += 1
    for gate, change in result["gate_changes"].items():
        counts[f"{gate}_removed"] += len(change["removed"])
        counts[f"{gate}_added"] += len(change["added"])
        counts[f"{gate}_unexplained_removals"] += sum(
            not item["evidence"]["structural_artifact"] for item in change["removed"]
        )


def failure_count(counts: dict[str, int]) -> int:
    return sum(
        value
        for key, value in counts.items()
        if key
        in {
            "numeric_occurrences_added",
            "numeric_demand_values_added",
            "numeric_removals_without_label_evidence",
            "unanchored_numeric_occurrences_before",
            "unanchored_numeric_occurrences_after",
            "lost_condition_cues",
            "glue_violations",
            "ambiguous_bodies_changed",
            "recognized_paths_still_failing",
            "existing_slices_changed",
        }
        or key.endswith("_unexplained_removals")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="cb2399746^")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=480)
    parser.add_argument("--max-bodies", type=int)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 480:
        parser.error("--max-seconds must be between 0 and 480")
    started = time.monotonic()
    checkpoint = json.loads(args.checkpoint.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "checkpoint.json"
    changes_path = args.output_dir / "changes.jsonl"
    bodies_path = args.output_dir / "bodies.jsonl"
    summary_path = args.output_dir / "summary.json"
    file_paths = [Path(value) for value in checkpoint["files"]]
    manifest = [
        {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in file_paths
    ]
    implementation = {
        relative: hashlib.sha256(Path(relative).read_bytes()).hexdigest()
        for relative in (
            "src/axiom_encode/bare_structure.py",
            "src/axiom_encode/corpus_resolver.py",
            "src/axiom_encode/harness/source_completeness.py",
            "src/axiom_encode/harness/validator_pipeline.py",
            "scripts/nz_bare_structure_differential.py",
        )
    }
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "release": checkpoint["release"],
            "manifest": checkpoint["manifest"],
            "baseline_ref": args.baseline_ref,
            "files": manifest,
            "implementation": implementation,
            "next_row": 0,
            "artifact_bytes": {"bodies": 0, "changes": 0},
            "counts": {},
            "per_file_counts": {},
            "complete": False,
        }
    )
    if (
        state["files"] != manifest
        or state["baseline_ref"] != args.baseline_ref
        or state.get("implementation") != implementation
    ):
        raise ValueError(
            "Resume checkpoint differs from corpus, baseline, or implementation"
        )
    state["artifacts"] = {
        "bodies": bodies_path.name,
        "changes": changes_path.name,
        "checkpoint": state_path.name,
        "summary": summary_path.name,
    }
    write_json(state_path, state)
    if state["complete"]:
        print("Differential already complete", flush=True)
        return 1 if failure_count(state["counts"]) else 0
    # A killed process may have appended its next row before replacing the
    # checkpoint. Restore the committed byte offsets before resuming, so an
    # interrupted batch never duplicates or leaves a partial JSONL record.
    for artifact, path in (("bodies", bodies_path), ("changes", changes_path)):
        if path.exists():
            with path.open("r+b") as output:
                output.truncate(state["artifact_bytes"][artifact])
    baseline = load_baseline(
        args.baseline_ref,
        "src/axiom_encode/harness/source_completeness.py",
        "axiom_encode.harness._nz_bare_baseline",
    )
    baseline_resolver = load_baseline(
        args.baseline_ref,
        "src/axiom_encode/corpus_resolver.py",
        "axiom_encode._nz_bare_resolver_baseline",
    )
    counts = Counter(state["counts"])
    per_file = {
        name: Counter(value) for name, value in state["per_file_counts"].items()
    }
    row_index = 0
    batch_bodies = 0
    stopped = False
    with changes_path.open("a") as changes, bodies_path.open("a") as bodies:
        for path in file_paths:
            with path.open() as source:
                for line_number, raw in enumerate(source, 1):
                    row = json.loads(raw)
                    if (
                        not isinstance(row.get("body"), str)
                        or not row["body"].strip()
                        or row.get("document_class") != "statute"
                    ):
                        continue
                    if row_index < state["next_row"]:
                        row_index += 1
                        continue
                    if time.monotonic() - started > args.max_seconds or (
                        args.max_bodies is not None and batch_bodies >= args.max_bodies
                    ):
                        stopped = True
                        break
                    if (
                        changes_path.stat().st_size + bodies_path.stat().st_size
                        > 500_000_000
                    ):
                        raise RuntimeError(
                            "Stopped before differential artifacts could exceed 1 GB"
                        )
                    result = analyze_row(row, baseline, baseline_resolver)
                    result.update(file=path.name, line=line_number)
                    add_counts(counts, result)
                    file_counts = per_file.setdefault(path.name, Counter())
                    add_counts(file_counts, result)
                    changed = (
                        result["slices"]
                        or result["numeric_removed"]
                        or result["numeric_added"]
                        or result["gate_changes"]
                        or result["lost_condition_cues"]
                        or result["glue_violations"]
                    )
                    if changed:
                        changes.write(json.dumps(result, ensure_ascii=False) + "\n")
                        changes.flush()
                    bodies.write(
                        json.dumps(
                            {
                                key: value
                                for key, value in result.items()
                                if key
                                not in {
                                    "slices",
                                    "numeric_removed",
                                    "numeric_added",
                                    "gate_changes",
                                    "glue_violations",
                                    "lost_condition_cues",
                                }
                            }
                        )
                        + "\n"
                    )
                    bodies.flush()
                    row_index += 1
                    batch_bodies += 1
                    state.update(
                        next_row=row_index,
                        artifact_bytes={
                            "bodies": bodies.tell(),
                            "changes": changes.tell(),
                        },
                        counts=dict(counts),
                        per_file_counts={
                            key: dict(value) for key, value in per_file.items()
                        },
                    )
                    write_json(state_path, state)
                    if batch_bodies % 100 == 0:
                        print(
                            f"Audited {row_index} bodies; {counts['previously_failed_paths_now_sliced']} new slices; {counts['numeric_occurrences_removed']} label numbers removed; {time.monotonic() - started:.1f}s",
                            flush=True,
                        )
                if stopped:
                    break
    state["complete"] = not stopped
    state["last_batch_seconds"] = round(time.monotonic() - started, 3)
    write_json(state_path, state)
    write_json(summary_path, state)
    print(
        json.dumps(
            {
                "complete": state["complete"],
                "next_row": row_index,
                "batch_bodies": batch_bodies,
                "seconds": state["last_batch_seconds"],
                "counts": {
                    key: value
                    for key, value in counts.items()
                    if not key.startswith("new_slice_path:")
                },
            }
        ),
        flush=True,
    )
    return 1 if failure_count(dict(counts)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
