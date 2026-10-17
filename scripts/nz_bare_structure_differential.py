"""Audit every statute body in a pinned NZ release against the pre-change code.

Run repeatedly with the same output directory to resume bounded batches. The
checkpoint lists exact provision files; this command never searches the corpus.
Every change is retained in JSONL, including numeric label evidence and each
before/after gate obligation. All public artifacts use corpus-relative names.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import functools
import hashlib
import importlib.util
import inspect
import json
import re
import signal
import subprocess
import sys
import time
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, SimpleNamespace
from typing import Any

from axiom_encode import corpus_resolver as current_resolver
from axiom_encode.bare_structure import (
    BareLabel,
    bare_definition_headers,
    bare_outline,
    mask_bare_labels,
)
from axiom_encode.harness import source_completeness as current
from axiom_encode.harness import validator_pipeline as current_validator

GATES = (
    "branches",
    "clauses",
    "tests_conditions",
    "tests_boundaries",
    "formula",
    "control",
    "deferral_targets",
)
BARE_TOKEN = re.compile(r"(?m)^[ \t]*(?P<token>[1-9]\d*[A-Z]?|[a-z]+)[ \t]*$")
ROMAN_TOKEN = re.compile(
    r"(?=[ivxlcdm]+\Z)m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})\Z"
)
_CITATION_FRAGMENT = (
    r"(?:(?:(?-i:[A-Z]{1,3})[ \t]+)?\d+[A-Za-z]*(?:\([0-9A-Za-z]+\))*"
    r"|\([0-9A-Za-z]+\))(?![\w(]|\.\d)"
)
CITATION_REFERENCE = re.compile(
    r"\b(?:sections?|subsections?|paragraphs?|clauses?)(?P<reference_gap>\s+)"
    rf"(?P<references>{_CITATION_FRAGMENT}"
    rf"(?:(?:\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or|to)\s+){_CITATION_FRAGMENT})*)",
    re.IGNORECASE,
)
DURATION_CONVERTERS = (("_YEAR_DURATION_PATTERN", 12), ("_WEEK_DURATION_PATTERN", 7))
RESTORATION_OPERAND = re.compile(r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+\.[0-9]+)")


class BatchDeadlineReached(Exception):
    """Stop a bounded batch even when an individual body is unusually slow."""


@dataclass(frozen=True)
class BoundaryGateBranch:
    path: tuple[str, ...]
    kind: str
    label: str
    text: str
    start: int
    end: int
    requirement: dict[str, Any]


def stop_batch(_signum: int, _frame: Any) -> None:
    raise BatchDeadlineReached


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
    exec(compile(source, relative, "exec", dont_inherit=True), module.__dict__)
    module._audit_baseline_source = source
    module._audit_baseline_ref = ref
    return module


def collapse(text: str) -> str:
    return " ".join(text.split())


def branch_record(branch: Any) -> dict[str, Any]:
    result = {
        "path": list(branch.path),
        "kind": branch.kind,
        "label": branch.label,
        "start": branch.start,
        "end": branch.end,
        "text": collapse(branch.text),
    }
    if hasattr(branch, "requirement"):
        result["numeric_requirement"] = branch.requirement
    if getattr(branch, "computation_chapeaux", ()):
        result["computation_chapeaux"] = [
            {"start": start, "end": end, "text": collapse(text)}
            for start, end, text in branch.computation_chapeaux
        ]
    return result


def occurrence_record(item: Any) -> dict[str, Any]:
    record = {
        "raw": item.raw,
        "value": item.value,
        "start": item.start,
        "end": item.end,
    }
    record.update(
        {
            name: getattr(item, name)
            for name in (
                "source_value",
                "alternative_values",
                "is_word_number",
                "has_rate_context",
                "has_temporal_context",
                "has_structural_context",
                "requires_rate_context",
            )
            if hasattr(item, name)
        }
    )
    return record


def numeric_semantic_identity(item: dict[str, Any]) -> tuple[Any, ...]:
    """Retain every field that changes the extractor's numeric demands."""
    return tuple(
        (name, tuple(item[name]) if name == "alternative_values" else item[name])
        for name in (
            "raw",
            "value",
            "source_value",
            "alternative_values",
            "is_word_number",
            "has_rate_context",
            "has_temporal_context",
            "has_structural_context",
            "requires_rate_context",
        )
        if name in item
    )


def numeric_requirement_identity(item: dict[str, Any]) -> tuple[Any, ...] | None:
    """Bind a complete numeric demand to its verified physical occurrence."""
    if item["source_start"] is None or item["source_end"] is None:
        return None
    return (
        item["source_start"],
        item["source_end"],
        numeric_semantic_identity(item),
    )


def partial_token_source_starts(
    source_tokens: Any, recall_tokens: Any, blocks: Any
) -> dict[int, int]:
    """Anchor unique retained suffixes inside small unmatched token gaps.

    Citation cleanup can turn ``10D(1)`` into ``(1)``. Equal-token alignment
    cannot map that surviving token. Require every recalled token in the gap
    to occur exactly once in its source gap and preserve their physical order;
    repeated suffixes and overlapping mappings remain unauthenticated. Bounds
    keep this fallback independent of the size of long statute definitions.
    """
    result = {}
    source_index = recall_index = 0
    for block in blocks:
        if (
            0 < block.b - recall_index <= 64
            and 0 < block.a - source_index <= 64
            and sum(
                len(token.group()) for token in source_tokens[source_index : block.a]
            )
            <= 4096
        ):
            pending = {}
            previous_end = -1
            for index in range(recall_index, block.b):
                needle = recall_tokens[index].group()
                candidates = []
                for token in source_tokens[source_index : block.a]:
                    offset = token.group().find(needle)
                    while offset != -1:
                        candidates.append(token.start() + offset)
                        if len(candidates) > 1:
                            break
                        offset = token.group().find(needle, offset + 1)
                    if len(candidates) > 1:
                        break
                if len(candidates) != 1 or candidates[0] < previous_end:
                    break
                pending[index] = candidates[0]
                previous_end = candidates[0] + len(needle)
            else:
                result.update(pending)
        source_index = block.a + block.size
        recall_index = block.b + block.size
    return result


class SourceTokenMatcher(difflib.SequenceMatcher):
    """Keep difflib's block choices without its quadratic repeated-token scan.

    With no junk or popular-token exclusions, each difflib search is the
    longest common substring, with ties resolved by the earliest source start
    and then the earliest recall start. A suffix automaton computes that exact
    match in linear time and space per search. The inherited block recursion
    and adjacent-block merging remain unchanged.
    """

    def __init__(self, source: list[str], recall: list[str]) -> None:
        super().__init__(a=source, b=recall, autojunk=False)

    def find_longest_match(
        self, alo: int = 0, ahi: int | None = None, blo: int = 0, bhi: int | None = None
    ) -> Any:
        ahi = len(self.a) if ahi is None else ahi
        bhi = len(self.b) if bhi is None else bhi
        # Retain difflib for small windows, where building an index costs more
        # than its bounded nested scan. Every larger search uses the automaton.
        if (ahi - alo) * (bhi - blo) <= 4096:
            return super().find_longest_match(alo, ahi, blo, bhi)

        transitions: list[dict[str, int]] = [{}]
        links = [-1]
        lengths = [0]
        first_ends = [-1]
        previous = 0
        for index in range(blo, bhi):
            token = self.b[index]
            state = len(transitions)
            transitions.append({})
            lengths.append(lengths[previous] + 1)
            links.append(0)
            first_ends.append(index)
            parent = previous
            while parent != -1 and token not in transitions[parent]:
                transitions[parent][token] = state
                parent = links[parent]
            if parent != -1:
                successor = transitions[parent][token]
                if lengths[parent] + 1 == lengths[successor]:
                    links[state] = successor
                else:
                    clone = len(transitions)
                    transitions.append(transitions[successor].copy())
                    lengths.append(lengths[parent] + 1)
                    links.append(links[successor])
                    # A clone has the same occurrence endpoints as the state
                    # it splits, including its first physical recall endpoint.
                    first_ends.append(first_ends[successor])
                    while parent != -1 and transitions[parent].get(token) == successor:
                        transitions[parent][token] = clone
                        parent = links[parent]
                    links[successor] = links[state] = clone
            previous = state

        state = size = best_size = 0
        best_source, best_recall = alo, blo
        for index in range(alo, ahi):
            token = self.a[index]
            while state and token not in transitions[state]:
                state = links[state]
                size = lengths[state]
            successor = transitions[state].get(token)
            if successor is None:
                state = size = 0
                continue
            state = successor
            size += 1
            source_start = index - size + 1
            recall_start = first_ends[state] - size + 1
            if size > best_size or (
                size == best_size
                and (source_start, recall_start) < (best_source, best_recall)
            ):
                best_source, best_recall, best_size = source_start, recall_start, size
        return difflib.Match(best_source, best_recall, best_size)


def uncached_source_token_alignment(text: str, recall: str) -> tuple[Any, ...]:
    """Return the existing deterministic alignment as immutable index tuples."""
    source_tokens = list(re.finditer(r"\S+", text))
    recall_tokens = list(re.finditer(r"\S+", recall))
    blocks = SourceTokenMatcher(
        [token.group() for token in source_tokens],
        [token.group() for token in recall_tokens],
    ).get_matching_blocks()
    token_map = {
        block.b + index: block.a + index
        for block in blocks
        for index in range(block.size)
    }
    partial_starts = partial_token_source_starts(source_tokens, recall_tokens, blocks)
    return (
        tuple(token.start() for token in source_tokens),
        tuple(token.start() for token in recall_tokens),
        tuple(token_map.get(index) for index in range(len(recall_tokens))),
        tuple(partial_starts.get(index) for index in range(len(recall_tokens))),
    )


@functools.lru_cache(maxsize=4)
def cached_source_token_alignment(text: str, recall: str) -> tuple[Any, ...]:
    """Reuse only a pure alignment for the exact original input strings."""
    return uncached_source_token_alignment(text, recall)


def source_token_alignment(text: str, recall: str) -> tuple[Any, ...]:
    # Small formula fragments must not evict a large body's repeated mapping.
    function = (
        cached_source_token_alignment
        if max(len(text), len(recall)) >= 16_384
        else uncached_source_token_alignment
    )
    return function(text, recall)


def anchored_occurrences(
    text: str, recall: str, occurrences: Any
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Restore corpus offsets after the legacy recall cleanup deletes citations."""
    if text == recall:
        result = {}
        for item in occurrences:
            verified = text[item.start : item.end] == item.raw
            start, end = (item.start, item.end) if verified else (None, None)
            record = occurrence_record(item)
            result[
                (
                    start,
                    end,
                    numeric_semantic_identity(record),
                    None if verified else item.start,
                    None if verified else item.end,
                )
            ] = {
                **record,
                "source_start": start,
                "source_end": end,
            }
        return result
    # Align non-whitespace tokens rather than individual letters. Character
    # matching on long statutes full of repeated prose is quadratic; token
    # blocks retain context, and every restored span is verified character for
    # character below. Unverifiable mappings are reported as audit failures.
    source_starts, recall_starts, token_map, partial_starts = source_token_alignment(
        text, recall
    )
    result = {}
    for item in occurrences:
        first = bisect_right(recall_starts, item.start) - 1
        last = bisect_left(recall_starts, item.end) - 1
        source_first = token_map[first] if 0 <= first < len(token_map) else None
        source_last = token_map[last] if 0 <= last < len(token_map) else None
        start = end = None
        if (
            first >= 0
            and last >= first
            and recall[item.start : item.end] == item.raw
            and source_first is not None
            and source_last is not None
            and source_last - source_first == last - first
            and all(
                token_map[index] == source_first + index - first
                for index in range(first, last + 1)
            )
        ):
            candidate_start = (
                source_starts[source_first] + item.start - recall_starts[first]
            )
            candidate_end = source_starts[source_last] + item.end - recall_starts[last]
            if text[candidate_start:candidate_end] == recall[item.start : item.end]:
                start, end = candidate_start, candidate_end
        elif (
            first == last
            and 0 <= first < len(partial_starts)
            and partial_starts[first] is not None
        ):
            candidate_start = partial_starts[first] + item.start - recall_starts[first]
            candidate_end = candidate_start + item.end - item.start
            if (
                recall[item.start : item.end] == item.raw
                and text[candidate_start:candidate_end] == item.raw
            ):
                start, end = candidate_start, candidate_end
        record = {**occurrence_record(item), "source_start": start, "source_end": end}
        result[
            (
                start,
                end,
                numeric_semantic_identity(record),
                item.start if start is None else None,
                item.end if end is None else None,
            )
        ] = record
    return result


def gates(module: Any, text: str, citation: str, extractor: Any) -> dict[str, Any]:
    branches = module.recognize_source_structure(text, corpus_citation_path=citation)
    shared = {
        "branches": branches,
        "active_branches": branches,
        "deferred_paths": set(),
    }
    formula = module._source_formula_branches(text, **shared)
    exceptions = module._source_exception_branches(
        text, **shared, formula_branches=formula
    )
    control = module._source_control_branches(
        text,
        **shared,
        formula_branches=formula,
        extract_numeric_occurrences=extractor,
    )
    boundary_branches = module._active_or_root_source_branches(text, **shared)

    @functools.lru_cache(maxsize=128)
    def boundary_source_map(source: str, recall: str) -> dict[tuple[Any, ...], Any]:
        return {
            (item["start"], item["end"], numeric_semantic_identity(item)): item
            for item in anchored_occurrences(source, recall, extractor(recall)).values()
        }

    boundaries = []
    for branch, boundary in module._source_boundary_obligations(
        boundary_branches,
        narrative_formula_branches=formula,
        extract_numeric_occurrences=extractor,
        **({"source_text": text} if module is current else {}),
    ):
        if module._source_boundary_is_temporal(branch, boundary):
            continue
        # Narrative formula bounds use their first line in production; ordinary
        # structural bounds use the owner's direct text before descendants.
        direct = (
            branch.text.splitlines()[0]
            if branch.kind == "formula-clause" and branch.text.splitlines()
            else module._source_branch_direct_text(branch, branches=boundary_branches)
        )
        recall = module.authoritative_numeric_recall_text(direct)
        anchored = boundary_source_map(
            text[branch.start : branch.start + len(direct)], recall
        ).get(
            (
                boundary.start,
                boundary.end,
                numeric_semantic_identity(occurrence_record(boundary)),
            ),
            {"source_start": None, "source_end": None},
        )
        source_start, source_end = anchored["source_start"], anchored["source_end"]
        requirement = {
            **occurrence_record(boundary),
            "source_start": None
            if source_start is None
            else branch.start + source_start,
            "source_end": None if source_end is None else branch.start + source_end,
        }
        boundaries.append(
            BoundaryGateBranch(
                branch.path,
                "numeric-boundary",
                f"{branch.label}={boundary.value:g}@{requirement['source_start']}:{requirement['source_end']}",
                branch.text,
                branch.start,
                branch.end,
                requirement,
            )
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
        "tests_boundaries": tuple(boundaries),
        "formula": formula,
        "control": control,
        "deferral_targets": boundary_branches,
        "_condition_cues": tuple(
            {"start": match.start(), "end": match.end(), "text": match.group()}
            for match in module._source_exception_or_applicability_matches(text)
        ),
        # A removal proof may replay the original grouping on a newly isolated
        # physical clause. The observed module is the unmodified pinned source;
        # current gate extraction must never grade its own changed obligations.
        "_baseline_condition_module": module
        if module is not current and getattr(module, "_audit_baseline_source", None)
        else None,
    }


def citation_boundary_reownership(
    removed: Any, replacements: Any, text: str, outline: Any, new_gates: Any
) -> dict[str, Any] | None:
    """Prove that a false citation-interior boundary gained its actual owner."""
    if not outline or new_gates is None:
        return None
    boundaries = [
        {
            "offset": boundary,
            "citation_start": match.start(),
            "citation_end": match.end(),
            "citation": match.group(),
            "boundary_kind": (
                "citation-interior"
                if match.start("references") <= boundary
                else "trimmed-citation-whitespace"
            ),
            "trimmed_whitespace": (
                None
                if match.start("references") <= boundary
                else {
                    "start": boundary,
                    "end": match.start("references"),
                    "text": text[boundary : match.start("references")],
                }
            ),
        }
        for match in CITATION_REFERENCE.finditer(text)
        for boundary in (removed.start, removed.end)
        if match.start("references") <= boundary < match.end("references")
        or (
            boundary == removed.end == match.start("reference_gap")
            and text[match.start("references")].isdigit()
            and text[boundary : match.start("references")].isspace()
        )
    ]
    if not boundaries or collapse(removed.text) != collapse(
        text[removed.start : removed.end]
    ):
        return None
    masked = mask_bare_labels(text, outline)
    first_label = min(label.start for label in outline)
    owners = [
        branch
        for branch in new_gates["branches"]
        if branch.text == masked[branch.start : branch.end]
        and (
            any(
                branch.path == tuple(part.lower() for part in label.path)
                and branch.start == label.start
                and branch.end == label.end
                for label in outline
            )
            or (
                branch.path == ()
                and branch.kind == "source-unit"
                and branch.start == 0
                and branch.end == first_label
                and first_label > 0
            )
        )
    ]
    supersets = []
    for item in replacements:
        if not (
            item.start <= removed.start < removed.end <= item.end
            and (item.start < removed.start or removed.end < item.end)
            and item.text == masked[item.start : item.end]
        ):
            continue
        owner = max(
            (
                branch
                for branch in owners
                if branch.start <= item.start < item.end <= branch.end
                and (item.kind == "excerpt" or branch.path == item.path)
            ),
            key=lambda branch: len(branch.path),
            default=None,
        )
        if owner is not None:
            supersets.append(
                {
                    "path": list(item.path),
                    "start": item.start,
                    "end": item.end,
                    "owner_path": list(owner.path),
                    "owner_start": owner.start,
                    "owner_end": owner.end,
                }
            )
    return (
        {"citation_boundaries": boundaries, "source_aligned_supersets": supersets}
        if supersets
        else None
    )


def condition_group_reownership(
    start: int,
    end: int,
    clauses: Any,
    text: str,
    outline: Any,
    old_gates: Any,
    new_gates: Any,
) -> dict[str, Any] | None:
    """Replay original complete condition groups inside an authentic clause.

    A glued legacy obligation can begin in an earlier paragraph, then consume
    the unconditional action before a later paragraph's grouped conditions.
    Only the pinned parser's complete groups may replace that overbroad span;
    the complete action and clause must retain their exact structural owner.
    """
    baseline = (old_gates or {}).get("_baseline_condition_module")
    baseline_source = getattr(baseline, "_audit_baseline_source", None)
    baseline_ref = getattr(baseline, "_audit_baseline_ref", None)
    if (
        not outline
        or new_gates is None
        or not baseline_source
        or baseline_ref != "86a1f911a"
        or baseline is current
    ):
        return None
    masked = mask_bare_labels(text, outline)
    clause = next(
        (
            clause
            for clause in clauses
            if clause.start < end
            and start < clause.end
            and end <= clause.end
            and not masked[start : clause.start].strip()
            and clause.text == masked[clause.start : clause.end]
            and clause.text == text[clause.start : clause.end]
        ),
        None,
    )
    if clause is None:
        return None
    label = max(
        (
            label
            for label in outline
            if label.start <= clause.start < clause.end <= label.end
        ),
        key=lambda label: len(label.path),
        default=None,
    )
    if label is None:
        return None
    owner = next(
        (
            branch
            for branch in new_gates["branches"]
            if branch.start <= clause.start < clause.end <= branch.end
            and branch.text == masked[branch.start : branch.end]
            and branch.path == tuple(part.lower() for part in label.path)
            and (branch.start, branch.end) == (label.start, label.end)
        ),
        None,
    )
    if owner is None:
        return None
    # Replaying one whole, unchanged physical clause prevents a shorter input
    # from manufacturing a condition boundary or losing a later qualifier.
    original_clauses = tuple(baseline._source_clause_spans(clause.text, branches=()))
    if original_clauses != ((0, len(clause.text), clause.text),):
        return None
    groups = baseline._source_exception_branches(
        clause.text,
        branches=(),
        active_branches=(),
        deferred_paths=set(),
    )
    if len(groups) < 2:
        return None
    conditions = []
    for group in groups:
        condition = baseline._source_exception_condition_text(group.text)
        if (
            not condition
            or condition.strip() != group.text.strip()
            or clause.text[group.start : group.end] != group.text
        ):
            return None
        group_start, group_end = clause.start + group.start, clause.start + group.end
        if not any(
            branch.path == owner.path
            and branch.start <= group_start < group_end <= branch.end
            and branch.text == masked[branch.start : branch.end]
            for branch in new_gates["tests_conditions"]
        ):
            return None
        conditions.append({"start": group_start, "end": group_end, "text": group.text})
    required = [
        group for group in conditions if group["start"] < end and start < group["end"]
    ]
    if not required:
        return None
    return {
        "baseline_ref": baseline_ref,
        "baseline_source_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
        "owner_path": list(owner.path),
        "owner_start": owner.start,
        "owner_end": owner.end,
        "complete_source_clause": branch_record(clause),
        "all_original_condition_groups": conditions,
        "required_condition_groups": required,
        "all_groups_retained_in_actual_conditions": True,
        "complete_action_retained_in_owned_source_clause": True,
    }


def structural_removal_evidence(
    removed: Any,
    replacements: Any,
    clauses: Any,
    text: str,
    outline: Any,
    gate: str,
    *,
    old_gates: Any = None,
    new_gates: Any = None,
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
    retained_formula_cues = [
        current.SourceStructureBranch(
            item.path, "computation-context", item.label, cue_text, start, end
        )
        for item in replacements
        for start, end, cue_text in getattr(item, "computation_chapeaux", ())
        if gate == "formula"
        and start < removed.end
        and removed.start < end
        and cue_text == masked[start:end]
    ]
    uncovered_gate_offsets = [
        offset
        for offset in range(removed.start, removed.end)
        if not masked[offset].isspace()
        and not any(
            item.start <= offset < item.end
            for item in (*aligned_replacements, *retained_formula_cues)
        )
    ]
    ownership_only = any(
        collapse(item.text) == collapse(removed.text) and item.path != removed.path
        for item in overlapping
    )
    citation_reownership = (
        citation_boundary_reownership(removed, overlapping, text, outline, new_gates)
        if not affected and not ownership_only
        else None
    )
    connective_clauses = [
        item
        for item in retained_clauses
        if re.fullmatch(r"(?:and|or|but)", collapse(item.text), re.IGNORECASE)
        and any(
            item.end <= label.start and not text[item.end : label.start].strip()
            for label in affected
        )
    ]
    uncovered_after_connectives = [
        offset
        for offset in uncovered_gate_offsets
        if not any(item.start <= offset < item.end for item in connective_clauses)
    ]
    requirement_proof = None
    if (
        old_gates is not None
        and new_gates is not None
        and gate in {"control", "tests_boundaries"}
    ):
        old_boundaries = [
            item.requirement
            for item in old_gates["tests_boundaries"]
            if item.path == removed.path
            and item.start == removed.start
            and item.end == removed.end
            and (gate != "tests_boundaries" or item.label == removed.label)
        ]

        new_identities = {
            numeric_requirement_identity(item.requirement)
            for item in new_gates["tests_boundaries"]
            if gate != "control"
            or any(
                control.path == item.path
                and control.start == item.start
                and control.end == item.end
                for control in new_gates["control"]
            )
        }
        missing_boundaries = [
            item
            for item in old_boundaries
            if numeric_requirement_identity(item) is None
            or numeric_requirement_identity(item) not in new_identities
        ]
        cues = (
            [
                cue
                for cue in old_gates["_condition_cues"]
                if removed.start <= cue["start"] and cue["end"] <= removed.end
            ]
            if gate == "control"
            else []
        )
        missing_cues = [
            cue
            for cue in cues
            if not any(
                branch.start <= cue["start"] and cue["end"] <= branch.end
                for branch in new_gates["control"]
            )
        ]
        requirement_proof = {
            "boundaries": old_boundaries,
            "missing_boundaries": missing_boundaries,
            "condition_cues": cues,
            "missing_condition_cues": missing_cues,
            "all_requirements_retained": bool(old_boundaries or cues)
            and not missing_boundaries
            and not missing_cues,
        }
    condition_proof = None
    if gate == "tests_conditions":
        # A legacy condition can accidentally consume the next bare paragraph.
        # Its own physical structural fragment must remain an actual condition;
        # retaining only the cue word or keeping the prose in general clauses
        # cannot certify removal of a tests obligation.
        cues = (
            old_gates["_condition_cues"]
            if old_gates is not None
            else tuple(
                {"start": match.start(), "end": match.end()}
                for match in current._source_exception_or_applicability_matches(text)
            )
        )
        cuts = sorted(
            {
                removed.start,
                removed.end,
                *(
                    point
                    for label in outline
                    for point in (label.start, label.marker_end, label.end)
                    if removed.start < point < removed.end
                ),
                *(
                    start
                    for start, _term in bare_definition_headers(text, outline)
                    if removed.start < start < removed.end
                ),
            }
        )
        fragments = [
            (start, end)
            for start, end in zip(cuts, cuts[1:])
            if any(start <= cue["start"] and cue["end"] <= end for cue in cues)
        ]
        group_reownership = [
            condition_group_reownership(
                start, end, clauses, text, outline, old_gates, new_gates
            )
            for start, end in fragments
        ]
        required_regions = [
            (group["start"], group["end"])
            for (start, end), proof in zip(fragments, group_reownership)
            for group in (
                proof["required_condition_groups"]
                if proof
                else ({"start": start, "end": end},)
            )
        ]
        missing_condition_offsets = [
            offset
            for start, end in required_regions
            for offset in range(start, end)
            if not masked[offset].isspace()
            and not any(
                item.start <= offset < item.end for item in aligned_replacements
            )
        ]
        condition_proof = {
            "physical_fragments": [
                {"start": start, "end": end, "text": masked[start:end]}
                for start, end in fragments
            ],
            "uncovered_condition_offsets": missing_condition_offsets,
            "original_condition_group_reownership": [
                proof for proof in group_reownership if proof
            ],
            "all_conditions_retained": bool(fragments)
            and not missing_condition_offsets,
        }
    gate_retained = not uncovered_after_connectives
    if condition_proof is not None:
        gate_retained = condition_proof["all_conditions_retained"]
    if requirement_proof is not None and (
        requirement_proof["boundaries"] or requirement_proof["condition_cues"]
    ):
        gate_retained = requirement_proof["all_requirements_retained"]
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
        "retained_standalone_connectives": [
            branch_record(item) for item in connective_clauses
        ],
        "retained_physical_formula_cues": [
            branch_record(item) for item in retained_formula_cues
        ],
        "requirement_identity_proof": requirement_proof,
        "condition_identity_proof": condition_proof,
        "ownership_only": ownership_only,
        "citation_boundary_reownership": citation_reownership,
        "structural_artifact": bool(affected or ownership_only or citation_reownership)
        and bool(overlapping or retained_formula_cues)
        and not uncovered
        and gate_retained,
    }


def compare_gates(old: Any, new: Any, text: str, outline: Any) -> dict[str, Any]:
    changes = {}

    def key(item: Any) -> tuple[Any, ...]:
        return (
            item.path,
            item.kind,
            item.label,
            item.start,
            item.end,
            collapse(item.text),
            tuple(getattr(item, "computation_chapeaux", ())),
            (
                item.requirement["source_start"],
                item.requirement["source_end"],
                numeric_semantic_identity(item.requirement),
            )
            if hasattr(item, "requirement")
            else None,
        )

    for gate in GATES:
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
                            item,
                            new[gate],
                            new["clauses"],
                            text,
                            outline,
                            gate,
                            old_gates=old,
                            new_gates=new,
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


def glue_violations(
    new: Any, text: str, outline: Any, *, explicit_parenthetical: bool = False
) -> list[dict[str, Any]]:
    """Detect marker characters retained anywhere in an authenticated excerpt."""
    violations = []
    records = [
        (gate, item.start, item.end, item.text) for gate in GATES for item in new[gate]
    ]
    records.extend(
        ("formula-context", start, end, cue_text)
        for item in new["formula"]
        for start, end, cue_text in getattr(item, "computation_chapeaux", ())
    )
    for gate, start, end, value in records:
        for label in outline:
            if not start <= label.start < label.marker_end <= end:
                continue
            # Offset-preserving masks allow direct checking; stripped excerpts
            # are checked against the canonical masked slice instead.
            expected = mask_bare_labels(text, outline)[start:end]
            if collapse(value) != collapse(expected) and not (
                explicit_parenthetical and collapse(value) == collapse(text[start:end])
            ):
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


def formula_context_records(
    formula: Any, text: str, outline: Any
) -> list[dict[str, Any]]:
    masked = mask_bare_labels(text, outline)
    return [
        {
            "owner_path": list(branch.path),
            "owner_start": branch.start,
            "owner_end": branch.end,
            "start": start,
            "end": end,
            "text": cue_text,
            "offset_aligned_source_excerpt": 0 <= start < end <= len(text)
            and cue_text == masked[start:end],
        }
        for branch in formula
        for start, end, cue_text in getattr(branch, "computation_chapeaux", ())
    ]


def missing_test_condition_cues(
    baseline: Any, text: str, old_gates: Any, new_gates: Any
) -> list[dict[str, Any]]:
    """Require old condition cues in the actual tests-condition gate."""
    return [
        {"start": match.start(), "end": match.end(), "text": match.group()}
        for match in baseline._source_exception_or_applicability_matches(text)
        if any(
            branch.start <= match.start() and match.end() <= branch.end
            for branch in old_gates["tests_conditions"]
        )
        and not any(
            branch.start <= match.start() and match.end() <= branch.end
            for branch in new_gates["tests_conditions"]
        )
    ]


def candidate_paths(
    text: str, outline: Any, old_branches: Any
) -> dict[tuple[str, ...], str]:
    """Probe unauthenticated label-shaped lines without treating them as labels.

    Every standalone token is requested as a potential root. Letter and roman
    lines also get the lexical parent suggested by preceding lines. These are
    deliberately overinclusive diagnostic requests: authentication comes only
    from the shared outline, never from this enumeration. Existing recognized
    parenthetical paths are included to check their unchanged behavior.
    """
    paths = {
        branch.path: "existing-layout"
        for branch in old_branches
        if branch.path
        and all(re.fullmatch(r"[1-9]\d*[A-Za-z]?|[a-z]+", part) for part in branch.path)
    }
    number = None
    letter = None
    for match in BARE_TOKEN.finditer(text):
        token = match.group("token")
        paths.setdefault((token,), "candidate-bare-root")
        if token[0].isdigit():
            number, letter = token, None
        else:
            if number is not None:
                paths.setdefault((number, token), "candidate-bare-child")
            if letter is not None and ROMAN_TOKEN.fullmatch(token):
                parent = (number, letter) if number is not None else (letter,)
                paths.setdefault((*parent, token), "candidate-bare-grandchild")
            letter = token
    for label in outline:
        paths[label.path] = "authenticated-bare"
    return paths


def numeric_removal_evidence(
    item: dict[str, Any], text: str, outline: Any, *, baseline_validator: Any = None
) -> Any:
    """Certify label-derived tokens even when a unit suffix extends into prose.

    The legacy tokenizer can interpret ``4\nAn`` as four French years, returning
    48 months before its blanket line cleanup. Authenticate the actual numeric
    constituent, while retaining the article ``An`` and every other nonlabel
    character. Any additional unlabelled number or word-number provenance fails.
    """
    start, end = item["source_start"], item["source_end"]
    if start is None or end is None or item.get("is_word_number", False):
        return None
    numbers = list(re.finditer(r"\d+", text[start:end]))
    if not numbers:
        return None
    constituents = []
    for number in numbers:
        number_start, number_end = start + number.start(), start + number.end()
        label = next(
            (
                label
                for label in outline
                if label.start <= number_start < number_end <= label.marker_end
            ),
            None,
        )
        if label is None:
            return None
        constituents.append(
            {
                "raw": number.group(),
                "start": number_start,
                "end": number_end,
                "token": label.token,
                "path": list(label.path),
                "line": text[label.start : label.marker_end],
                "followed_by": text[label.marker_end :].splitlines()[1],
            }
        )
    # For converted tokens, source_value binds the numeric operand before unit
    # normalization. A source scalar or an unconverted token must be exactly
    # the authenticated operand; do not explain arbitrary normalized values.
    operand = float(constituents[0]["raw"])
    provenance = None
    if len(constituents) != 1:
        return None
    if operand == float(
        item.get("source_value")
        if item.get("source_value") is not None
        else item["value"]
    ):
        provenance = {"kind": "source-scalar", "operand": operand}
    elif baseline_validator is not None:
        for pattern_name, multiplier in DURATION_CONVERTERS:
            pattern = getattr(baseline_validator, pattern_name)
            match = pattern.fullmatch(item["raw"])
            if (
                match is not None
                and start + match.start("number") == constituents[0]["start"]
                and start + match.end("number") == constituents[0]["end"]
                and baseline_validator._parse_belgian_numeric_phrase(
                    match.group("number")
                )
                == operand
                and operand * multiplier == item["value"]
            ):
                provenance = {
                    "kind": pattern_name,
                    "operand": operand,
                    "multiplier": multiplier,
                    "normalized_value": item["value"],
                }
                break
    if provenance is None:
        return None
    masked = mask_bare_labels(text, outline)
    retained_offsets = [
        offset
        for offset in range(start, end)
        if not any(label.start <= offset < label.marker_end for label in outline)
    ]
    if any(masked[offset] != text[offset] for offset in retained_offsets):
        return None
    return {
        **constituents[0],
        "numeric_constituents": constituents,
        "numeric_constituent_count": len(constituents),
        "legacy_numeric_provenance": provenance,
        "masked_raw_span": masked[start:end],
        "retained_nonlabel_characters": len(retained_offsets),
    }


def numeric_restoration_record(
    item: dict[str, Any], text: str, outline: Any, baseline_validator: Any
) -> dict[str, Any]:
    """Certify restoration of the complete operand on a legacy-excluded line."""
    start, end = item["source_start"], item["source_end"]
    label = next(
        (
            label
            for label in outline
            if start is not None and label.start <= start < label.marker_end
        ),
        None,
    )
    line_start = None if start is None else text.rfind("\n", 0, start) + 1
    line_end = None if start is None else text.find("\n", start)
    if line_end == -1:
        line_end = len(text)
    line = "" if line_start is None else text[line_start:line_end]
    stripped = line.strip()
    quote_chars = baseline_validator._STRUCTURAL_SOURCE_QUOTE_CHARS
    legacy_structural_line = stripped.strip(quote_chars)
    constituent = re.fullmatch(
        r"(?P<number>\d+)(?P<suffix>[A-Za-z]?)", legacy_structural_line
    )
    proof = None
    if (
        label is None
        and start is not None
        and end is not None
        and 0 <= start < end <= len(text)
        and constituent is not None
        and baseline_validator._STRUCTURAL_SOURCE_LINE_PATTERN.fullmatch(
            legacy_structural_line
        )
        and not item.get("is_word_number", False)
    ):
        token_start = (
            line_start
            + len(line)
            - len(line.lstrip())
            + len(stripped)
            - len(stripped.lstrip(quote_chars))
        )
        number_start = token_start + constituent.start("number")
        number_end = token_start + constituent.end("number")
        token_end = token_start + constituent.end()
        operand = float(constituent.group("number"))
        # The suffix is a printed structural spelling, not another operand.
        # Accept an extractor that retains it in raw or normalizes it away,
        # but never a prefix/subspan of the complete digit run or a new value.
        if (
            start == number_start
            and end in {number_end, token_end}
            and text[start:end] == item["raw"]
            and item["value"] == operand
            and item.get("source_value") in {None, operand}
            and all(value == operand for value in item.get("alternative_values", ()))
        ):
            proof = {
                "raw": constituent.group("number"),
                "source_start": number_start,
                "source_end": number_end,
                "value": operand,
                "suffix": constituent.group("suffix"),
            }
    return {
        "previously_blanket_excluded_numeric_line": proof is not None,
        "restored_numeric_constituent": proof,
        "line_start": line_start,
        "line_end": line_end,
        "line": line,
        "legacy_structural_line": legacy_structural_line,
        "authenticated_label": None if label is None else list(label.path),
    }


def complete_numeric_constituent_span(text: str, start: int, end: int) -> bool:
    """Reject a partial digit/decimal/grouping token while allowing punctuation."""
    return not (
        (start > 0 and text[start - 1].isdigit())
        or (end < len(text) and text[end].isdigit())
        or (start > 1 and text[start - 1] in ".," and text[start - 2].isdigit())
        or (end + 1 < len(text) and text[end] in ".," and text[end + 1].isdigit())
    )


def duration_suppressor_provenance(
    old: dict[str, Any], text: str, recall: str, baseline_validator: Any
) -> dict[str, Any] | None:
    """Bind a converted legacy occurrence to its complete numeric operand."""
    if old.get("source_value") != old["value"] or old.get("is_word_number", False):
        return None
    for pattern_name, multiplier in DURATION_CONVERTERS:
        pattern = getattr(baseline_validator, pattern_name, None)
        match = None if pattern is None else pattern.fullmatch(old["raw"])
        if match is None:
            continue
        number = match.group("number")
        # Authenticate a complete digit operand, including a decimal separator
        # when present. The unit/article can never enter a numeric mask.
        if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", number) is None:
            continue
        operand = baseline_validator._parse_belgian_numeric_phrase(number)
        if operand is None or operand * multiplier != old["value"]:
            continue
        source_start = old["source_start"] + match.start("number")
        source_end = old["source_start"] + match.end("number")
        recall_start = old["start"] + match.start("number")
        recall_end = old["start"] + match.end("number")
        if (
            text[source_start:source_end] != number
            or recall[recall_start:recall_end] != number
            or not complete_numeric_constituent_span(text, source_start, source_end)
        ):
            continue
        return {
            "kind": pattern_name,
            "operand": operand,
            "multiplier": multiplier,
            "normalized_value": old["value"],
            "numeric_constituents": [
                {
                    "raw": number,
                    "source_start": source_start,
                    "source_end": source_end,
                    "recall_start": recall_start,
                    "recall_end": recall_end,
                }
            ],
        }
    return None


def captured_special_suppressor_provenance(
    old: dict[str, Any], text: str, recall: str, snapshot: Any
) -> dict[str, Any] | None:
    """Authenticate percent conversion or an actual legacy grouped prefix."""
    if snapshot is None or not snapshot.get("traced_final_matches_normal"):
        return None
    if not any(
        numeric_requirement_identity(value) == numeric_requirement_identity(old)
        and (value["start"], value["end"]) == (old["start"], old["end"])
        for value in snapshot["occurrences"]
    ):
        return None
    start, end = old["source_start"], old["source_end"]
    recall_start, recall_end = old["start"], old["end"]
    raw = old["raw"]
    if (
        not 0 <= start < end <= len(text)
        or not 0 <= recall_start < recall_end <= len(recall)
        or text[start:end] != raw
        or recall[recall_start:recall_end] != raw
        or old.get("is_word_number", False)
        or old.get("alternative_values", ())
    ):
        return None
    if (
        re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw)
        and complete_numeric_constituent_span(text, start, end)
        and text[end : end + 1] == recall[recall_end : recall_end + 1] == "%"
        and old.get("source_value") == float(raw)
        and old["value"] == float(raw) / 100
        and old.get("has_rate_context", False)
        and old.get("requires_rate_context", False)
        and not old.get("has_temporal_context", False)
        and not old.get("has_structural_context", False)
    ):
        return {
            "kind": "captured-ascii-percent",
            "operand": float(raw),
            "multiplier": 0.01,
            "normalized_value": old["value"],
            "preserved_percent_source_span": [end, end + 1],
            "numeric_constituents": [
                {
                    "raw": raw,
                    "source_start": start,
                    "source_end": end,
                    "recall_start": recall_start,
                    "recall_end": recall_end,
                }
            ],
        }
    grouped = re.match(r"[0-9]{1,3}(?:,[0-9]{3})+", text[start:])
    if grouped is None:
        return None
    enclosing = grouped.group()
    enclosing_end, enclosing_recall_end = (
        start + len(enclosing),
        recall_start + len(enclosing),
    )
    if (
        raw != enclosing.split(",")[0]
        or re.fullmatch(r"[0-9]{1,3}", raw) is None
        or old["value"] != float(raw)
        or old.get("source_value") != float(raw)
        or not complete_numeric_constituent_span(text, start, enclosing_end)
        or recall[recall_start:enclosing_recall_end] != enclosing
        or any(
            old.get(name, False)
            for name in (
                "has_rate_context",
                "has_temporal_context",
                "has_structural_context",
                "requires_rate_context",
            )
        )
    ):
        return None
    return {
        "kind": "captured-legacy-first-group-prefix",
        "legacy_partial_capture": {
            "raw": raw,
            "value": old["value"],
            "source_value": old["source_value"],
            "source_start": start,
            "source_end": end,
            "recall_start": recall_start,
            "recall_end": recall_end,
        },
        "operand": float(enclosing.replace(",", "")),
        "numeric_constituents": [
            {
                "raw": enclosing,
                "source_start": start,
                "source_end": enclosing_end,
                "recall_start": recall_start,
                "recall_end": enclosing_recall_end,
            }
        ],
    }


@functools.lru_cache(maxsize=8)
def baseline_grouped_capture_stage(source: str) -> dict[str, Any] | None:
    """Locate the exact pinned legacy suppression predicate by its full AST."""
    assignment = ast.parse(
        "grouped_value = _parse_belgian_numeric_phrase(match.group(1))"
    ).body[0]
    predicate = ast.parse(
        "grouped_value is not None and not math.isclose(grouped_value, value, "
        "rel_tol=0, abs_tol=NUMERIC_GROUNDING_ABS_TOLERANCE) and "
        "any(math.isclose(grouped_value, occurrence.value, rel_tol=0, "
        "abs_tol=NUMERIC_GROUNDING_ABS_TOLERANCE) for occurrence in collector.inventory)",
        mode="eval",
    ).body
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_tokenize_numeric_occurrences_from_text"
    ]
    if len(functions) != 1:
        return None
    function = functions[0]
    stages = []
    for loop in ast.walk(function):
        if not (
            isinstance(loop, ast.For)
            and isinstance(loop.iter, ast.Name)
            and loop.iter.id == "percentage_context_number_matches"
        ):
            continue
        for block in ast.walk(loop):
            # The pinned assignment is inside the loop's ValueError guard.
            # Keep the loop ancestry and require that exact guard as well.
            if not (
                isinstance(block, ast.With)
                and len(block.items) == 1
                and ast.dump(block.items[0].context_expr)
                == ast.dump(
                    ast.parse("contextlib.suppress(ValueError)", mode="eval").body
                )
            ):
                continue
            for left, right in zip(block.body, block.body[1:]):
                if (
                    ast.dump(left) == ast.dump(assignment)
                    and isinstance(right, ast.If)
                    and ast.dump(right.test) == ast.dump(predicate)
                ):
                    stages.append(right.test.values[0].lineno)
    if len(stages) != 1:
        return None
    return {
        "function": function.name,
        "function_first_line": min(
            [function.lineno, *(node.lineno for node in function.decorator_list)]
        ),
        "line": stages[0],
        "predicate": ast.unparse(predicate),
        "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


@functools.lru_cache(maxsize=8)
def baseline_grouped_capture_code(source: str, filename: str) -> CodeType | None:
    """Compile the unchanged source independently of this audit's future flags."""
    codes = [
        value
        for value in compile(source, filename, "exec", dont_inherit=True).co_consts
        if isinstance(value, CodeType)
        and value.co_name == "_tokenize_numeric_occurrences_from_text"
    ]
    return codes[0] if len(codes) == 1 else None


def baseline_code_identity(value: Any) -> Any:
    """Retain all immutable code fields without marshal reference-state flags."""
    if isinstance(value, CodeType):
        return {
            "type": "code",
            **{
                name: baseline_code_identity(getattr(value, name))
                for name in (
                    "co_argcount",
                    "co_posonlyargcount",
                    "co_kwonlyargcount",
                    "co_nlocals",
                    "co_stacksize",
                    "co_flags",
                    "co_code",
                    "co_consts",
                    "co_names",
                    "co_varnames",
                    "co_filename",
                    "co_name",
                    "co_qualname",
                    "co_firstlineno",
                    "co_linetable",
                    "co_exceptiontable",
                    "co_freevars",
                    "co_cellvars",
                )
            },
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [baseline_code_identity(x) for x in value]}
    if isinstance(value, frozenset):
        return {
            "type": "frozenset",
            "items": sorted(
                (baseline_code_identity(x) for x in value),
                key=lambda x: json.dumps(x, sort_keys=True),
            ),
        }
    return {"type": type(value).__name__, "repr": repr(value)}


def baseline_grouped_capture_snapshot(
    item: dict[str, Any],
    text: str,
    recall: str,
    baseline_validator: Any,
    old_extractor: Any,
    *,
    profile: str = "legacy",
) -> dict[str, Any] | None:
    """Observe only the original collector state at this exact grouped token."""
    source = getattr(baseline_validator, "_audit_baseline_source", None)
    if source is None:
        return None
    stage = baseline_grouped_capture_stage(source)
    if stage is None:
        return None
    tokenizer = inspect.unwrap(
        baseline_validator._tokenize_numeric_occurrences_from_text
    )
    code = tokenizer.__code__
    expected_code = baseline_grouped_capture_code(source, code.co_filename)
    code_identity = baseline_code_identity(code)
    if (
        code.co_name != stage["function"]
        or code.co_firstlineno != stage["function_first_line"]
        or expected_code is None
        or code_identity != baseline_code_identity(expected_code)
    ):
        return None
    stage = {
        **stage,
        "runtime_code_sha256": hashlib.sha256(
            json.dumps(code_identity, sort_keys=True).encode()
        ).hexdigest(),
        "runtime_code_flags": code.co_flags,
        "compiled_without_inherited_flags": True,
    }
    captures = []

    def observe(frame: Any, event: str, _argument: Any) -> Any:
        if frame.f_code is code and event == "line" and frame.f_lineno == stage["line"]:
            local = frame.f_locals
            match = local["match"]
            if match.group(1) == item["raw"] and local["value"] == item["value"]:
                # The view carries offsets through the original cleanup. Bind
                # this actual current match to its physical corpus occurrence.
                recall_start, recall_end = local["cleaned_view"].source_span(
                    local["span"]
                )
                occurrence = SimpleNamespace(
                    raw=recall[recall_start:recall_end],
                    value=local["value"],
                    start=recall_start,
                    end=recall_end,
                )
                anchored = tuple(
                    anchored_occurrences(text, recall, (occurrence,)).values()
                )
                if len(anchored) == 1 and (
                    anchored[0]["source_start"],
                    anchored[0]["source_end"],
                ) == (item["source_start"], item["source_end"]):
                    captures.append(tuple(local["collector"].inventory))
        return observe if frame.f_code is code else None

    normal = tuple(old_extractor(recall))
    previous_trace = sys.gettrace()
    try:
        sys.settrace(observe)
        traced = tokenizer(recall, profile=profile)
    finally:
        sys.settrace(previous_trace)
    if len(captures) != 1 or [
        occurrence_record(value) for value in traced.inventory
    ] != [occurrence_record(value) for value in normal]:
        return None
    alternate = baseline_validator._parse_belgian_numeric_phrase(item["raw"])
    records = tuple(anchored_occurrences(text, recall, captures[0]).values())
    return {
        "stage": {**stage, "baseline_ref": baseline_validator._audit_baseline_ref},
        "profile": profile,
        "target": {
            "raw": item["raw"],
            "value": item["value"],
            "source_start": item["source_start"],
            "source_end": item["source_end"],
        },
        "snapshot_size": len(captures[0]),
        "snapshot_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True).encode()
        ).hexdigest(),
        "occurrences": tuple(value for value in records if value["value"] == alternate),
        "traced_final_matches_normal": True,
    }


def _cross_occurrence_restoration_from_candidates(
    item: dict[str, Any],
    text: str,
    recall: str,
    old_numbers: Any,
    baseline_validator: Any,
    old_extractor: Any,
    *,
    stage_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Replay the exact legacy cross-occurrence grouped-number suppression."""
    start, end = item["source_start"], item["source_end"]
    raw = item["raw"]
    if (
        start is None
        or end is None
        or not 0 <= start < end <= len(text)
        or text[start:end] != raw
        or RESTORATION_OPERAND.fullmatch(raw) is None
        or not complete_numeric_constituent_span(text, start, end)
        or item.get("is_word_number", False)
    ):
        return None
    operand = float(raw.replace(",", ""))
    if (
        item["value"] != operand
        or item.get("source_value") not in {None, operand}
        or any(value != operand for value in item.get("alternative_values", ()))
    ):
        return None
    alternate = baseline_validator._parse_belgian_numeric_phrase(raw)
    if alternate is None or alternate == operand:
        return None
    # Any existing reading at the grouped token is a different case: this
    # certificate concerns only suppression caused by distinct previously
    # collected tokens. A stage snapshot can prove a physically later token
    # was already collected by an earlier semantic pass.
    if any(
        old.get("source_start") == start and old.get("source_end") == end
        for old in old_numbers
    ):
        return None
    suppressors = []
    verified_original = (
        {
            (value["start"], value["end"], numeric_requirement_identity(value))
            for value in old_numbers
        }
        if stage_snapshot is not None
        else None
    )
    for old in old_numbers:
        source_start, source_end = old.get("source_start"), old.get("source_end")
        if not (
            source_start is not None
            and source_end is not None
            and 0 <= source_start < source_end <= len(text)
            and (
                source_end <= start
                or (stage_snapshot is not None and end <= source_start)
            )
            and text[source_start:source_end] == old["raw"]
            and old["value"] == alternate
            and not old.get("is_word_number", False)
            and 0 <= old["start"] < old["end"] <= len(recall)
            and recall[old["start"] : old["end"]] == old["raw"]
        ):
            continue
        if (
            re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", old["raw"])
            and complete_numeric_constituent_span(text, source_start, source_end)
            and float(old["raw"]) == old["value"]
            and old.get("source_value") in {None, alternate}
        ):
            provenance = {
                "kind": "source-scalar",
                "operand": old["value"],
                "numeric_constituents": [
                    {
                        "raw": old["raw"],
                        "source_start": source_start,
                        "source_end": source_end,
                        "recall_start": old["start"],
                        "recall_end": old["end"],
                    }
                ],
            }
        else:
            provenance = duration_suppressor_provenance(
                old, text, recall, baseline_validator
            )
            if provenance is None:
                provenance = captured_special_suppressor_provenance(
                    old, text, recall, stage_snapshot
                )
            if provenance is None:
                continue
            if verified_original is None:
                verified_original = {
                    (
                        item["start"],
                        item["end"],
                        numeric_requirement_identity(item),
                    )
                    for item in anchored_occurrences(
                        text, recall, old_extractor(recall)
                    ).values()
                }
            if (
                old["start"],
                old["end"],
                numeric_requirement_identity(old),
            ) not in verified_original:
                continue
        if any(
            not (
                constituent["source_end"] <= start or end <= constituent["source_start"]
            )
            for constituent in provenance["numeric_constituents"]
        ):
            continue
        suppressors.append(
            {
                **old,
                "suppression_provenance": provenance,
                "physical_relation": "before" if source_end <= start else "after",
            }
        )
    if not suppressors:
        return None
    masked_spans = sorted(
        {
            (constituent["recall_start"], constituent["recall_end"])
            for old in suppressors
            for constituent in old["suppression_provenance"]["numeric_constituents"]
        }
    )
    if any(left[1] > right[0] for left, right in zip(masked_spans, masked_spans[1:])):
        return None
    pieces = []
    offset = 0
    for mask_start, mask_end in masked_spans:
        pieces.extend((recall[offset:mask_start], " " * (mask_end - mask_start)))
        offset = mask_end
    pieces.append(recall[offset:])
    counterfactual = "".join(pieces)
    restored = next(
        (
            occurrence
            for occurrence in anchored_occurrences(
                text, counterfactual, old_extractor(counterfactual)
            ).values()
            if numeric_requirement_identity(occurrence)
            == numeric_requirement_identity(item)
        ),
        None,
    )
    if restored is None:
        return None
    return {
        "raw": raw,
        "source_start": start,
        "source_end": end,
        "operand": operand,
        "operand_kind": "comma-thousands" if "," in raw else "ascii-dot-decimal",
        "legacy_alternate_reading": alternate,
        "distinct_prior_suppressors": suppressors,
        "counterfactual_masked_recall_spans": [list(span) for span in masked_spans],
        "counterfactual_only_suppressors_masked": all(
            character == recall[index]
            for index, character in enumerate(counterfactual)
            if not any(left <= index < right for left, right in masked_spans)
        ),
        "counterfactual_restored_occurrence": restored,
        "collector_snapshot": stage_snapshot,
    }


def cross_occurrence_restoration_evidence(
    item: dict[str, Any],
    text: str,
    recall: str,
    old_numbers: Any,
    baseline_validator: Any,
    old_extractor: Any,
    *,
    profile: str = "legacy",
) -> dict[str, Any] | None:
    """Retain the simple proof, then observe an exact hidden collector cause."""
    # The corrected suppression guard belongs only to the legacy tokenizer;
    # other profiles cannot borrow evidence from a different numeric parser.
    if profile != "legacy" or RESTORATION_OPERAND.fullmatch(item["raw"]) is None:
        return None
    old_numbers = tuple(old_numbers)
    evidence = (
        _cross_occurrence_restoration_from_candidates(
            item, text, recall, old_numbers, baseline_validator, old_extractor
        )
        if "," in item["raw"]
        else None
    )
    if evidence is not None:
        return evidence
    snapshot = baseline_grouped_capture_snapshot(
        item, text, recall, baseline_validator, old_extractor, profile=profile
    )
    if snapshot is None:
        return None
    return _cross_occurrence_restoration_from_candidates(
        item,
        text,
        recall,
        snapshot["occurrences"],
        baseline_validator,
        old_extractor,
        stage_snapshot=snapshot,
    )


def bare_slice_projection(text: str, label: Any, outline: Any) -> dict[str, Any]:
    """Certify the only allowed source transformations for a selected unit."""
    raw = text[label.marker_end : label.end]
    source_start = label.marker_end + len(raw) - len(raw.lstrip())
    source_end = label.end - len(raw) + len(raw.rstrip())
    expected = text[source_start:source_end]
    descendants = tuple(
        BareLabel(
            child.token,
            child.path[len(label.path) :],
            child.start - source_start,
            child.marker_end - source_start,
            min(child.end, source_end) - source_start,
        )
        for child in outline
        if len(child.path) > len(label.path)
        and child.path[: len(label.path)] == label.path
        and source_start <= child.start < child.marker_end <= source_end
    )
    found = bare_outline(expected) or ()
    if descendants == found:
        return {
            "mode": "bare" if descendants else "leaf",
            "text": expected,
            "outline": descendants,
            "source_start": source_start,
            "source_end": source_end,
        }
    definitions = [
        {"start": start, "term": term}
        for start, term in bare_definition_headers(text, outline)
        if source_start <= start < label.end
    ]
    if definitions:
        return {
            "mode": "reject-detached-definition-projection",
            "source_start": source_start,
            "source_end": source_end,
            "definitions": definitions,
            "expected_descendant_paths": [list(item.path) for item in descendants],
            "independently_recognized_paths": [list(item.path) for item in found],
        }
    # Build an offset map for the explicitly delimited descendant markers.
    # Every source character outside authenticated markers remains unchanged.
    parts = []
    replacements = []
    offset = 0
    shift = 0
    for child in descendants:
        replacement = f"({child.token})"
        parts.extend((expected[offset : child.start], replacement))
        replacements.append((child, child.start + shift, len(replacement)))
        shift += len(replacement) - (child.marker_end - child.start)
        offset = child.marker_end
    parts.append(expected[offset:])
    rendered = "".join(parts)

    def projected_offset(position: int) -> int:
        return position + sum(
            replacement_size - (child.marker_end - child.start)
            for child, _start, replacement_size in replacements
            if child.marker_end <= position
        )

    projected = tuple(
        BareLabel(
            child.token,
            child.path,
            start,
            start + replacement_size,
            projected_offset(child.end),
        )
        for child, start, replacement_size in replacements
    )
    return {
        "mode": "parenthetical-descendants",
        "text": rendered,
        "outline": projected,
        "source_start": source_start,
        "source_end": source_end,
        "marker_replacements": [
            {
                "source_start": source_start + child.start,
                "source_end": source_start + child.marker_end,
                "rendered_start": start,
                "rendered_end": start + replacement_size,
            }
            for child, start, replacement_size in replacements
        ],
    }


def audit_bare_slice(
    text: str, label: Any, outline: Any, sliced: str, citation: str, extractor: Any
) -> dict[str, Any]:
    """Audit downstream gates and recall after removing a selected own marker."""
    projection = bare_slice_projection(text, label, outline)
    expected = projection.get("text", "")
    descendants = projection.get("outline", ())
    parenthetical = projection["mode"] == "parenthetical-descendants"
    slice_gates = gates(current, sliced, citation, extractor)
    found = bare_outline(sliced) or ()
    if parenthetical:
        found = tuple(
            BareLabel(item.token, item.path, item.start, item.marker_end, branch.end)
            for item in descendants
            for branch in slice_gates["branches"]
            if branch.path == tuple(part.lower() for part in item.path)
            and branch.start == item.start
        )
    recall = current.authoritative_numeric_recall_text(
        sliced, corpus_citation_path=citation
    )
    expected_recall = current.authoritative_numeric_recall_text(
        mask_bare_labels(expected, descendants), corpus_citation_path=citation
    )
    own_removed = sliced == expected
    glue = (
        glue_violations(
            slice_gates,
            sliced,
            descendants,
            explicit_parenthetical=parenthetical,
        )
        if own_removed
        else []
    )

    def source_inventory(recall_text: str) -> tuple[dict[str, Any], ...]:
        records = []
        for item in anchored_occurrences(
            sliced, recall_text, extractor(recall_text)
        ).values():
            start, end = item["source_start"], item["source_end"]

            def parent_offset(position: int | None) -> int | None:
                if position is None:
                    return None
                return (
                    projection["source_start"]
                    + position
                    - sum(
                        replacement["rendered_end"]
                        - replacement["rendered_start"]
                        - (replacement["source_end"] - replacement["source_start"])
                        for replacement in projection.get("marker_replacements", ())
                        if replacement["rendered_end"] <= position
                    )
                )

            parent_start, parent_end = parent_offset(start), parent_offset(end)
            records.append(
                {
                    **{
                        key: value
                        for key, value in item.items()
                        if key not in {"start", "end"}
                    },
                    "parent_source_start": parent_start,
                    "parent_source_end": parent_end,
                    "parent_source_span_verified": parent_start is not None
                    and parent_end is not None
                    and text[parent_start:parent_end] == item["raw"],
                }
            )
        return tuple(records)

    expected_inventory = source_inventory(expected_recall)
    actual_inventory = source_inventory(recall)
    result = {
        "path": list(label.path),
        "projection": projection["mode"],
        "source_start": projection["source_start"],
        "source_end": projection["source_end"],
        "own_authenticated_marker_removed": own_removed,
        "operative_text_preserved": own_removed,
        "expected_descendant_labels": len(descendants),
        "descendant_outline_preserved": descendants == found,
        "numeric_inventory_matches_authenticated_slice": expected_inventory
        == actual_inventory
        and all(item["parent_source_span_verified"] for item in actual_inventory),
        "numeric_occurrences_audited": len(actual_inventory),
        "numeric_occurrences_at_verified_parent_offsets": sum(
            item["parent_source_span_verified"] for item in actual_inventory
        ),
        "glue_violations": glue,
        "formula_contexts": formula_context_records(
            slice_gates["formula"], sliced, descendants
        ),
    }
    if not result["descendant_outline_preserved"]:
        result["expected_outline"] = [
            {
                "token": item.token,
                "path": list(item.path),
                "start": item.start,
                "marker_end": item.marker_end,
                "end": item.end,
            }
            for item in descendants
        ]
        result["recognized_outline"] = [
            {
                "token": item.token,
                "path": list(item.path),
                "start": item.start,
                "marker_end": item.marker_end,
                "end": item.end,
            }
            for item in found
        ]
    if not result["numeric_inventory_matches_authenticated_slice"]:
        result["expected_inventory"] = expected_inventory
        result["actual_inventory"] = actual_inventory
    return result


def analyze_row(
    row: dict[str, Any], baseline: Any, baseline_resolver: Any, baseline_validator: Any
) -> dict[str, Any]:
    text, citation = row["body"], row["citation_path"]
    outline = bare_outline(text) or ()
    profile = current_validator._numeric_profile_for_citation_path(citation)

    @functools.lru_cache(maxsize=128)
    def extractor(value: str) -> Any:
        return tuple(
            current_validator.extract_typed_numeric_inventory_occurrences_from_text(
                value, profile=profile
            )
        )

    @functools.lru_cache(maxsize=128)
    def old_extractor(value: str) -> Any:
        return tuple(
            baseline_validator.extract_typed_numeric_inventory_occurrences_from_text(
                value, profile=profile
            )
        )

    old_recall = baseline.authoritative_numeric_recall_text(
        text, corpus_citation_path=citation
    )
    new_recall = current.authoritative_numeric_recall_text(
        text, corpus_citation_path=citation
    )
    old_occurrences = old_extractor(old_recall)
    new_occurrences = extractor(new_recall)
    # Equal recall text proves that every occurrence is unchanged without an
    # expensive alignment of very large, repetitive statute definitions. Source
    # anchoring is needed only for removals/additions that require evidence.
    same_occurrences = [occurrence_record(item) for item in old_occurrences] == [
        occurrence_record(item) for item in new_occurrences
    ]
    anchoring_required = old_recall != new_recall or not same_occurrences
    if not anchoring_required:
        old_numbers = {
            index: occurrence_record(item) for index, item in enumerate(old_occurrences)
        }
        new_numbers = old_numbers
        removed, added = [], []
    else:
        old_numbers = anchored_occurrences(text, old_recall, old_occurrences)
        new_numbers = anchored_occurrences(text, new_recall, new_occurrences)
        removed = [old_numbers[key] for key in old_numbers.keys() - new_numbers.keys()]
        added = [new_numbers[key] for key in new_numbers.keys() - old_numbers.keys()]
    removed_records = []
    for item in sorted(removed, key=lambda item: item["start"]):
        removed_records.append(
            {
                **item,
                "authenticated_label": numeric_removal_evidence(
                    item, text, outline, baseline_validator=baseline_validator
                ),
            }
        )
    added_records = []
    for item in sorted(added, key=lambda item: item["start"]):
        added_records.append(
            {
                **item,
                **numeric_restoration_record(item, text, outline, baseline_validator),
                "cross_occurrence_restoration": cross_occurrence_restoration_evidence(
                    item,
                    text,
                    old_recall,
                    old_numbers.values(),
                    baseline_validator,
                    old_extractor,
                    profile=profile,
                ),
            }
        )
    old_gates = gates(baseline, text, citation, old_extractor)
    new_gates = gates(current, text, citation, extractor)
    slice_changes = []
    slice_layout_audits = []
    labels_by_path = {label.path: label for label in outline}
    for path, evidence in sorted(
        candidate_paths(text, outline, old_gates["branches"]).items()
    ):
        before = slice_result(baseline_resolver, text, citation, path)
        after = slice_result(current_resolver, text, citation, path)
        projection_rejection = None
        if evidence == "authenticated-bare" and after is None and before is None:
            projection = bare_slice_projection(text, labels_by_path[path], outline)
            if projection["mode"] == "reject-detached-definition-projection":
                projection_rejection = projection
        slice_changes.append(
            {
                "path": list(path),
                "evidence": evidence,
                "previously_sliced": before is not None,
                "now_slices": after is not None,
                "changed_existing_slice": before is not None and before != after,
                "excerpt": after if before != after else None,
                "certified_projection_rejection": projection_rejection,
            }
        )
        if evidence == "authenticated-bare" and after is not None:
            slice_layout_audits.append(
                audit_bare_slice(
                    text,
                    labels_by_path[path],
                    outline,
                    after,
                    citation + "/" + "/".join(path),
                    extractor,
                )
            )
    gate_changes = compare_gates(old_gates, new_gates, text, outline)
    old_values = {item["value"] for item in old_numbers.values()}
    new_values = {item["value"] for item in new_numbers.values()}
    # A former condition may split or acquire a real branch owner, but its cue
    # must still belong to the actual tests-condition obligation inventory.
    lost_condition_cues = missing_test_condition_cues(
        baseline, text, old_gates, new_gates
    )
    return {
        "citation": citation,
        "body_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_chars": len(text),
        "candidate_bare_lines": len(list(BARE_TOKEN.finditer(text))),
        "authenticated_labels": len(outline),
        "slices": slice_changes,
        "slice_layout_audits": slice_layout_audits,
        "numeric_removed": removed_records,
        "numeric_added": added_records,
        "numeric_demand_values_before": sorted(old_values),
        "numeric_demand_values_after": sorted(new_values),
        "numeric_demand_values_removed": sorted(old_values - new_values),
        "numeric_demand_values_added": sorted(new_values - old_values),
        "recall_matches_baseline_of_authenticated_mask": new_recall
        == baseline.authoritative_numeric_recall_text(
            mask_bare_labels(text, outline), corpus_citation_path=citation
        ),
        "numeric_occurrences_before": len(old_occurrences),
        "numeric_occurrences_after": len(new_occurrences),
        "numeric_source_anchoring_required": anchoring_required,
        "unanchored_numeric_occurrences_before": sum(
            item.get("source_start", 0) is None for item in old_numbers.values()
        ),
        "unanchored_numeric_occurrences_after": sum(
            item.get("source_start", 0) is None for item in new_numbers.values()
        ),
        "unanchored_changed_numeric_occurrences": sum(
            item["source_start"] is None for item in (*removed, *added)
        ),
        "unanchored_unchanged_numeric_occurrences": [
            old_numbers[key]
            for key in old_numbers.keys() & new_numbers.keys()
            if old_numbers[key].get("source_start", 0) is None
        ],
        "gate_changes": gate_changes,
        "lost_condition_cues": lost_condition_cues,
        "glue_violations": glue_violations(new_gates, text, outline),
        "formula_contexts": formula_context_records(
            new_gates["formula"], text, outline
        ),
        "ambiguous_unchanged": not outline
        and old_recall == new_recall
        and not any(
            gate in gate_changes
            for gate in (
                "branches",
                "clauses",
                "tests_conditions",
                "formula",
                "deferral_targets",
            )
        ),
        "ambiguous_restored_numeric_gate_additions": not outline
        and any(
            gate_changes.get(gate, {}).get("added")
            for gate in ("tests_boundaries", "control")
        ),
        "gate_counts_before": {key: len(old_gates[key]) for key in GATES},
        "gate_counts_after": {key: len(new_gates[key]) for key in GATES},
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
    counts["numeric_additions_without_restoration_evidence"] += sum(
        not item["previously_blanket_excluded_numeric_line"]
        and not item["cross_occurrence_restoration"]
        for item in result["numeric_added"]
    )
    counts["numeric_additions_restored_cross_occurrence"] += sum(
        bool(item["cross_occurrence_restoration"]) for item in result["numeric_added"]
    )
    counts["numeric_demand_values_removed"] += len(
        result["numeric_demand_values_removed"]
    )
    counts["numeric_demand_values_added"] += len(result["numeric_demand_values_added"])
    counts["numeric_mask_baseline_mismatches"] += not result[
        "recall_matches_baseline_of_authenticated_mask"
    ]
    counts["numeric_occurrences_before"] += result["numeric_occurrences_before"]
    counts["numeric_occurrences_after"] += result["numeric_occurrences_after"]
    counts["lost_condition_cues"] += len(result["lost_condition_cues"])
    counts["unanchored_changed_numeric_occurrences"] += result[
        "unanchored_changed_numeric_occurrences"
    ]
    for item in result["slice_layout_audits"]:
        counts["sliced_source_units_audited"] += 1
        counts[f"sliced_projection:{item['projection']}"] += 1
        counts["slice_own_markers_retained"] += not item[
            "own_authenticated_marker_removed"
        ]
        counts["slice_operative_text_changed"] += not item["operative_text_preserved"]
        counts["slice_descendant_outline_mismatches"] += not item[
            "descendant_outline_preserved"
        ]
        counts["slice_numeric_inventory_mismatches"] += not item[
            "numeric_inventory_matches_authenticated_slice"
        ]
        counts["slice_glue_violations"] += len(item["glue_violations"])
        counts["slice_unaligned_formula_contexts"] += sum(
            not cue["offset_aligned_source_excerpt"] for cue in item["formula_contexts"]
        )
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
    counts["formula_inherited_physical_cues"] += len(result["formula_contexts"])
    counts["unaligned_formula_contexts"] += sum(
        not cue["offset_aligned_source_excerpt"] for cue in result["formula_contexts"]
    )
    counts["ambiguous_bodies_changed"] += bool(
        not result["authenticated_labels"] and not result["ambiguous_unchanged"]
    )
    counts["ambiguous_bodies_with_restored_numeric_gate_additions"] += result[
        "ambiguous_restored_numeric_gate_additions"
    ]
    for item in result["slices"]:
        counts["requested_paths"] += 1
        authenticated = item["evidence"] == "authenticated-bare"
        counts["authenticated_requested_paths"] += authenticated
        counts["unauthenticated_candidate_paths_rejected"] += bool(
            not authenticated and not item["now_slices"]
        )
        counts["unauthenticated_candidate_paths_newly_accepted"] += bool(
            not authenticated and not item["previously_sliced"] and item["now_slices"]
        )
        counts["previously_failed_paths_now_sliced"] += bool(
            not item["previously_sliced"] and item["now_slices"]
        )
        counts["recognized_paths_still_failing"] += (
            authenticated
            and not item["now_slices"]
            and not item["certified_projection_rejection"]
        )
        counts["authenticated_paths_conservatively_rejected"] += bool(
            item["certified_projection_rejection"]
        )
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
            "numeric_additions_without_restoration_evidence",
            "numeric_mask_baseline_mismatches",
            "numeric_removals_without_label_evidence",
            "unanchored_changed_numeric_occurrences",
            "lost_condition_cues",
            "glue_violations",
            "ambiguous_bodies_changed",
            "recognized_paths_still_failing",
            "existing_slices_changed",
            "unauthenticated_candidate_paths_newly_accepted",
            "slice_own_markers_retained",
            "slice_operative_text_changed",
            "slice_descendant_outline_mismatches",
            "slice_numeric_inventory_mismatches",
            "slice_glue_violations",
            "slice_unaligned_formula_contexts",
            "unaligned_formula_contexts",
        }
        or key.endswith("_unexplained_removals")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="86a1f911a")
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
    if "expected_artifacts" in checkpoint:
        expected = [
            {key: artifact[key] for key in ("file", "sha256")}
            for artifact in checkpoint["expected_artifacts"]
        ]
        if manifest != expected:
            raise ValueError("Corpus artifacts differ from the pinned release manifest")
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
            "manifest_provision_rows": sum(
                artifact["rows"]
                for artifact in checkpoint.get("expected_artifacts", ())
            ),
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
    print(
        json.dumps(
            {
                "stage": "manifest-verified",
                "resume_row": state["next_row"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        ),
        flush=True,
    )
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
    baseline_validator = load_baseline(
        args.baseline_ref,
        "src/axiom_encode/harness/validator_pipeline.py",
        "axiom_encode.harness._nz_bare_validator_baseline",
    )
    analysis_started = time.monotonic()
    print(
        json.dumps(
            {
                "stage": "baseline-loaded",
                "elapsed_seconds": round(analysis_started - started, 3),
            }
        ),
        flush=True,
    )
    counts = Counter(state["counts"])
    per_file = {
        name: Counter(value) for name, value in state["per_file_counts"].items()
    }
    row_index = 0
    batch_bodies = 0
    stopped = False
    deadline_row = None
    previous_alarm_handler = signal.signal(signal.SIGALRM, stop_batch)
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
                    signal.setitimer(
                        signal.ITIMER_REAL,
                        max(0.001, args.max_seconds - (time.monotonic() - started)),
                    )
                    try:
                        result = analyze_row(
                            row, baseline, baseline_resolver, baseline_validator
                        )
                    except BatchDeadlineReached:
                        stopped = True
                        deadline_row = {
                            "file": path.name,
                            "line": line_number,
                            "citation": row["citation_path"],
                        }
                        break
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
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
                                    "slice_layout_audits",
                                    "numeric_removed",
                                    "numeric_added",
                                    "gate_changes",
                                    "glue_violations",
                                    "lost_condition_cues",
                                    "formula_contexts",
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
                    # Commit bounded groups of rows. A killed batch restores
                    # these byte offsets and reruns at most 24 completed rows,
                    # avoiding a growing checkpoint rewrite after every body.
                    if batch_bodies % 25 == 0:
                        write_json(state_path, state)
                    if batch_bodies % 100 == 0:
                        print(
                            f"Audited {row_index} bodies; {counts['previously_failed_paths_now_sliced']} new slices; {counts['numeric_occurrences_removed']} label numbers removed; {time.monotonic() - started:.1f}s",
                            flush=True,
                        )
                if stopped:
                    break
    state["complete"] = not stopped
    state["deadline_row"] = deadline_row
    signal.signal(signal.SIGALRM, previous_alarm_handler)
    state["last_batch_seconds"] = round(time.monotonic() - started, 3)
    state["last_batch_analysis_seconds"] = round(time.monotonic() - analysis_started, 3)
    state["last_batch_bodies"] = batch_bodies
    write_json(state_path, state)
    write_json(summary_path, state)
    print(
        json.dumps(
            {
                "complete": state["complete"],
                "next_row": row_index,
                "batch_bodies": batch_bodies,
                "seconds": state["last_batch_seconds"],
                "analysis_seconds": state["last_batch_analysis_seconds"],
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
