"""Audit every statute body in a pinned NZ release against the pre-change code.

Run repeatedly with the same output directory to resume bounded batches. The
checkpoint lists exact provision files; this command never searches the corpus.
Every change is retained in JSONL, including numeric label evidence and each
before/after gate obligation. All public artifacts use corpus-relative names.
"""

from __future__ import annotations

import argparse
import difflib
import functools
import hashlib
import importlib.util
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
    exec(compile(source, relative, "exec"), module.__dict__)
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


def anchored_occurrences(
    text: str, recall: str, occurrences: Any
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Restore corpus offsets after the legacy recall cleanup deletes citations."""
    if text == recall:
        result = {}
        for item in occurrences:
            verified = text[item.start : item.end] == item.raw
            start, end = (item.start, item.end) if verified else (None, None)
            result[
                (
                    start,
                    end,
                    item.raw,
                    item.value,
                    None if verified else item.start,
                    None if verified else item.end,
                )
            ] = {
                **occurrence_record(item),
                "source_start": start,
                "source_end": end,
            }
        return result
    # Align non-whitespace tokens rather than individual letters. Character
    # matching on long statutes full of repeated prose is quadratic; token
    # blocks retain context, and every restored span is verified character for
    # character below. Unverifiable mappings are reported as audit failures.
    source_tokens = list(re.finditer(r"\S+", text))
    recall_tokens = list(re.finditer(r"\S+", recall))
    blocks = difflib.SequenceMatcher(
        a=[token.group() for token in source_tokens],
        b=[token.group() for token in recall_tokens],
        autojunk=False,
    ).get_matching_blocks()
    token_map = {
        block.b + index: block.a + index
        for block in blocks
        for index in range(block.size)
    }
    recall_starts = [token.start() for token in recall_tokens]
    result = {}
    for item in occurrences:
        first = bisect_right(recall_starts, item.start) - 1
        last = bisect_left(recall_starts, item.end) - 1
        source_first = token_map.get(first)
        source_last = token_map.get(last)
        start = end = None
        if (
            first >= 0
            and last >= first
            and recall[item.start : item.end] == item.raw
            and source_first is not None
            and source_last is not None
            and source_last - source_first == last - first
            and all(
                token_map.get(index) == source_first + index - first
                for index in range(first, last + 1)
            )
        ):
            candidate_start = (
                source_tokens[source_first].start()
                + item.start
                - recall_tokens[first].start()
            )
            candidate_end = (
                source_tokens[source_last].start()
                + item.end
                - recall_tokens[last].start()
            )
            if text[candidate_start:candidate_end] == recall[item.start : item.end]:
                start, end = candidate_start, candidate_end
        record = {**occurrence_record(item), "source_start": start, "source_end": end}
        result[
            (
                start,
                end,
                item.raw,
                item.value,
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
            (item["start"], item["end"], item["raw"], item["value"]): item
            for item in anchored_occurrences(source, recall, extractor(recall)).values()
        }

    boundaries = []
    for branch, boundary in module._source_boundary_obligations(
        boundary_branches,
        narrative_formula_branches=formula,
        extract_numeric_occurrences=extractor,
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
            (boundary.start, boundary.end, boundary.raw, boundary.value),
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

        def identity(item: dict[str, Any]) -> tuple[Any, ...] | None:
            if item["source_start"] is None or item["source_end"] is None:
                return None
            return tuple(
                item.get(key)
                for key in (
                    "source_start",
                    "source_end",
                    "raw",
                    "value",
                    "source_value",
                    "has_rate_context",
                    "has_temporal_context",
                    "requires_rate_context",
                )
            ) + (tuple(item.get("alternative_values", ())),)

        new_identities = {
            identity(item.requirement)
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
            if identity(item) is None or identity(item) not in new_identities
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
    gate_retained = (
        not uncovered_after_connectives
        or gate == "tests_conditions"
        or bool(requirement_proof and requirement_proof["all_requirements_retained"])
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
        "retained_standalone_connectives": [
            branch_record(item) for item in connective_clauses
        ],
        "retained_physical_formula_cues": [
            branch_record(item) for item in retained_formula_cues
        ],
        "requirement_identity_proof": requirement_proof,
        "ownership_only": ownership_only,
        "structural_artifact": bool(affected or ownership_only)
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
        for pattern_name, multiplier in (
            ("_YEAR_DURATION_PATTERN", 12),
            ("_WEEK_DURATION_PATTERN", 7),
        ):
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
        start = item["source_start"]
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
        legacy_structural_line = line.strip().strip(
            baseline_validator._STRUCTURAL_SOURCE_QUOTE_CHARS
        )
        restored = bool(
            label is None
            and start is not None
            and line_start <= start < item["source_end"] <= line_end
            and re.fullmatch(r"\d+[A-Za-z]?", legacy_structural_line)
            and baseline_validator._STRUCTURAL_SOURCE_LINE_PATTERN.fullmatch(
                legacy_structural_line
            )
        )
        added_records.append(
            {
                **item,
                "previously_blanket_excluded_numeric_line": restored,
                "line_start": line_start,
                "line_end": line_end,
                "line": line,
                "legacy_structural_line": legacy_structural_line,
                "authenticated_label": None if label is None else list(label.path),
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
        for item in result["numeric_added"]
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
