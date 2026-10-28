"""Pin the near-linear source-completeness scans to main's results and growth.

The ``_old_*`` functions below are verbatim copies of the scans these paths
replaced (main at cbc6b74fe), with module globals read through ``sc``.
``_old_mode`` swaps them back into the module, so each differential compares
whole analyses and helper results, old against new, on the same inputs. The
growth tests bound CPU time per doubling of the input: a linear scan doubles,
and a regression to a quadratic one quadruples.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import gc
import inspect
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
import types
import weakref
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from unittest import mock

import pytest
import yaml
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.coverage_index import format_coverage_index
from axiom_encode.harness.source_completeness import SourceStructureBranch
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

FIXTURES = Path(__file__).parent / "fixtures/source_completeness"
SNAP_CITATION = "us/statute/7/2014"


def _old_reason_dependency_occurrence_is_contextual(
    reason: str,
    match: re.Match[str],
    *,
    current_usc_title: str | None,
) -> bool:
    """Reject context attached to this citation without crossing into another one."""

    groups = match.groupdict()
    title = groups.get("title")
    section = groups.get("section")
    if title and section:
        dependencies = sc._reason_dependencies(reason)
        candidates = (
            *sc._qualified_usc_dependencies(reason),
            *sc._RELATIVE_USC_DEFERRAL_DEPENDENCY.finditer(reason),
            *sc._qualified_louisiana_rs_dependencies(reason),
        )
        for candidate in candidates:
            if _old_usc_dependency_occurrences_match(
                reason,
                candidate,
                match,
                current_usc_title=current_usc_title,
            ) and (
                sc._CONTEXTUAL_AUTHORITY_LANGUAGE.search(
                    _old_reason_dependency_local_context(
                        reason,
                        candidate,
                        dependencies=dependencies,
                    )
                )
            ):
                return True

    remainder = reason[match.end() :]
    stop = min(
        (
            position
            for separator in (".", ";", "\n")
            if (position := remainder.find(separator)) >= 0
        ),
        default=-1,
    )
    if stop < 0:
        return False
    trailing_clause = re.split(r"[.;\n]", remainder[stop + 1 :], maxsplit=1)[0]
    return bool(
        not sc._reason_dependencies(trailing_clause)
        and sc._has_adversative_language(trailing_clause)
    )


def _old_reason_dependency_local_context(
    reason: str,
    match: re.Match[str],
    *,
    dependencies: list[re.Match[str]],
) -> str:
    clause_start, clause_end = _old_reason_clause_bounds(reason, match)
    previous_end = max(
        (
            dependency.end()
            for dependency in dependencies
            if clause_start <= dependency.start() and dependency.end() <= match.start()
        ),
        default=clause_start,
    )
    next_start = min(
        (
            dependency.start()
            for dependency in dependencies
            if match.end() <= dependency.start() < clause_end
        ),
        default=clause_end,
    )
    before = reason[previous_end : match.start()]
    after = reason[match.end() : next_start]
    preceding_coordinations = list(sc._DEPENDENCY_CONTEXT_COORDINATION.finditer(before))
    if preceding_coordinations:
        before = before[preceding_coordinations[-1].end() :]
    following_coordination = sc._DEPENDENCY_CONTEXT_COORDINATION.search(after)
    if following_coordination:
        after = after[: following_coordination.start()]
    return before + after


def _old_usc_dependencies_match(left: re.Match[str], right: re.Match[str]) -> bool:
    left_groups = left.groupdict()
    right_groups = right.groupdict()
    return bool(
        sc._dependency_citation_family(left) == sc._dependency_citation_family(right)
        and left_groups.get("title")
        and right_groups.get("title")
        and left_groups["title"].lower() == right_groups["title"].lower()
        and sc.normalize_rulespec_path_segment(left_groups["section"])
        == sc.normalize_rulespec_path_segment(right_groups["section"])
        and sc._usc_dependency_fragments(left) == sc._usc_dependency_fragments(right)
    )


def _old_usc_dependency_occurrences_match(
    reason: str,
    left: re.Match[str],
    right: re.Match[str],
    *,
    current_usc_title: str | None,
) -> bool:
    if _old_usc_dependencies_match(left, right):
        return True
    left_groups = left.groupdict()
    right_groups = right.groupdict()
    if (
        left_groups.get("title")
        or not right_groups.get("title")
        or not current_usc_title
        or right_groups["title"].lower() != current_usc_title.lower()
    ):
        return False
    if not re.match(
        r"\s+of\s+this\s+title\b",
        reason[left.end() :],
        flags=re.IGNORECASE,
    ):
        return False
    return bool(
        sc.normalize_rulespec_path_segment(left_groups["section"])
        == sc.normalize_rulespec_path_segment(right_groups["section"])
        and sc._usc_dependency_fragments(left) == sc._usc_dependency_fragments(right)
    )


def _old_reason_clause_bounds(
    reason: str,
    match: re.Match[str],
) -> tuple[int, int]:
    masked = list(reason)
    for dependency in (
        *sc._qualified_usc_dependencies(reason),
        *sc._qualified_louisiana_rs_dependencies(reason),
    ):
        for position in range(dependency.start(), dependency.end()):
            if masked[position] in {".", ";"}:
                masked[position] = " "
    masked_reason = "".join(masked)
    preceding_stops = [
        position
        for separator in (".", ";", "\n")
        if (position := masked_reason.rfind(separator, 0, match.start())) >= 0
    ]
    following_stops = [
        position
        for separator in (".", ";", "\n")
        if (position := masked_reason.find(separator, match.end())) >= 0
    ]
    return (
        max(preceding_stops, default=-1) + 1,
        min(following_stops, default=len(reason)),
    )


def _old_is_marker_only_container(
    branch: SourceStructureBranch,
    *,
    branches: Sequence[SourceStructureBranch],
    source_text: str,
) -> bool:
    """Ignore a parent wrapper only when it has no operative chapeau text."""

    direct_child_starts = [
        candidate.start
        for candidate in branches
        if len(candidate.path) == len(branch.path) + 1
        and candidate.path[: len(branch.path)] == branch.path
        and branch.start <= candidate.start < branch.end
    ]
    if not direct_child_starts:
        return False
    chapeau = source_text[branch.start : min(direct_child_starts)]
    if branch.kind == "guidance-subsection" and chapeau.strip() == branch.label:
        return True
    return re.search(r"\w", sc._strip_source_clause_marker(chapeau)) is None


def _old_source_clause_spans(
    source_text: str,
    *,
    branches: Sequence[SourceStructureBranch],
) -> Iterable[tuple[int, int, str]]:
    """Yield offset-preserving clauses split at punctuation and structure."""

    # Mask only authenticated structural labels, preserving offsets. The
    # resulting excerpts retain every operative character of the source.
    printed_rows = sc._unconflicted_printed_rows(source_text, branches)
    bare_labels = sc.bare_outline(source_text) or ()
    display_text = sc.mask_bare_labels(source_text, bare_labels)
    boundary_text = sc._mask_spaced_german_sentence_labels(display_text)
    boundary = re.compile(
        r";|:(?=\s*(?i:provided)\s*,?\s*(?i:that)\b)|"
        r"[.!?](?=(?:[ \t]+[A-ZÄÖÜ(]|\s*$))",
        flags=re.MULTILINE,
    )
    inline_operand_list_spans = sc._formula_inline_operand_list_spans(source_text)
    boundary_matches = (
        match
        for match in boundary.finditer(boundary_text)
        if not _old_source_clause_boundary_splits_state_code_citation(
            source_text, match
        )
        and not (
            match.group() == "."
            and re.search(r"(?<!\w)\d+\.$", source_text[: match.end()])
            and re.match(r"\s+Lebensjahr(?:es|e|en)?\b", source_text[match.end() :])
        )
        and not any(
            start < match.end() < end for start, end in inline_operand_list_spans
        )
        and not (
            match.group() == ";"
            and sc._FORMULA_DIRECT_FOLLOWING_OPERANDS.search(
                source_text[max(0, match.start() - 300) : match.start()]
            )
            and ":" in source_text[max(0, match.start() - 300) : match.start()]
            and not sc._FORMULA_FOLLOWING_OPERAND_PROVISO.match(
                source_text[match.end() :]
            )
        )
    )
    split_points = {
        0,
        len(source_text),
        *sc._work_chart_heading_starts(source_text),
        *sc._printed_chart_arithmetic_starts(source_text),
        *sc._printed_page_header_boundaries(source_text),
        *(
            point
            for span in sc._spaced_german_sentence_label_spans(source_text)
            for point in span
        ),
        *(
            match.end()
            for match in boundary_matches
            if printed_rows is None
            or not any(a < match.end() < b for a, b in printed_rows.atomic_spans)
        ),
        *(
            point
            for row in sc._worksheet_arithmetic_rows(source_text)
            for point in (row.start(), row.end())
        ),
        *(
            match.start()
            for match in re.finditer(
                r"\b(?:Guidance\s+documents\s+lack\s+the\s+force\s+and\s+"
                r"effect\s+of\s+law|USDA\s+may\s+not\s+cite,\s*use,\s*or\s+"
                r"rely\s+on\s+any\s+guidance)\b",
                source_text,
                flags=re.IGNORECASE,
            )
        ),
        *(branch.start for branch in branches),
        *(branch.end for branch in branches),
        *(label.start for label in bare_labels),
        *(label.marker_end for label in bare_labels),
        *(
            (
                start
                for start, _term in sc.bare_definition_headers(source_text, bare_labels)
            )
            if bare_labels
            else ()
        ),
    }
    if printed_rows is not None:
        split_points.update(
            point
            for point in printed_rows.boundaries
            if not any(a < point < b for a, b in printed_rows.atomic_spans)
        )
    for start, end in zip(sorted(split_points), sorted(split_points)[1:]):
        raw = boundary_text[start:end]
        left_trimmed = len(raw) - len(raw.lstrip())
        right_trimmed = len(raw.rstrip())
        if right_trimmed > left_trimmed:
            yield (
                start + left_trimmed,
                start + right_trimmed,
                display_text[start + left_trimmed : start + right_trimmed],
            )


def _old_source_clause_boundary_splits_state_code_citation(
    source_text: str,
    boundary: re.Match[str],
) -> bool:
    """Keep a line-wrapped ``R.S. 47:...`` citation in one legal clause."""

    return bool(
        boundary.group(0) == "."
        and re.search(r"\bR\.S\.$", source_text[: boundary.end()], re.IGNORECASE)
        and re.match(
            r"\s*\d+[A-Za-z]?:\d+[A-Za-z]?",
            source_text[boundary.end() :],
        )
    )


def _old_span_is_deferred(
    start: int,
    end: int,
    *,
    branches: Sequence[SourceStructureBranch],
    deferred_paths: set[tuple[str, ...]],
) -> bool:
    if not branches:
        return () in deferred_paths
    return any(
        branch.start <= start
        and end <= branch.end
        and sc._path_is_deferred(branch.path, deferred_paths)
        for branch in branches
    )


def _old_most_specific_containing_branch(
    start: int,
    end: int,
    *,
    branches: Sequence[SourceStructureBranch],
) -> SourceStructureBranch | None:
    candidates = [
        branch for branch in branches if branch.start <= start and end <= branch.end
    ]
    return max(
        candidates,
        key=lambda branch: (
            len(branch.path),
            -(branch.end - branch.start),
        ),
        default=None,
    )


def _old_source_branch_direct_text(
    branch: SourceStructureBranch,
    *,
    branches: Sequence[SourceStructureBranch],
) -> str:
    """Return one branch's own chapeau without nested child bodies.

    ``SourceStructureBranch.text`` intentionally spans descendants so structural
    and formula analysis can retain parent context. Boundary inventory is
    different: counting descendant thresholds again on every ancestor creates
    impossible parent-level obligations and obscures the child rule that owns
    each comparator. Preserve the original prefix offsets while stopping at the
    first descendant; leaf branches continue to use their complete text. Using
    any descendant is necessary because callers may already have removed
    marker-only intermediate containers from the candidate branch sequence.
    """

    descendant_starts = [
        candidate.start
        for candidate in branches
        if (
            (
                len(candidate.path) > len(branch.path)
                and candidate.path[: len(branch.path)] == branch.path
            )
            or (
                # German sentence labels and numbered items share paragraph
                # paths, even when the sentence text contains the whole list.
                branch.kind == "sentence"
                and branch.path
                and branch.path[-1].startswith("satz-")
                and candidate.kind in {"number", "letter"}
                and len(candidate.path) >= len(branch.path)
                and candidate.path[: len(branch.path) - 1] == branch.path[:-1]
                and branch.start < candidate.start
                and candidate.end <= branch.end
            )
        )
        and branch.start <= candidate.start < branch.end
    ]
    if not descendant_starts:
        return branch.text
    direct_end = max(0, min(descendant_starts) - branch.start)
    return branch.text[:direct_end]


def _old_path_is_deferred(
    path: tuple[str, ...],
    deferred_paths: set[tuple[str, ...]],
) -> bool:
    """Let a precise non-root deferral cover its structural descendants."""

    if not path:
        return () in deferred_paths
    return any(
        deferred_path and path[: len(deferred_path)] == deferred_path
        for deferred_path in deferred_paths
    )


class _NaiveSpanContainment:
    """Main's linear ``any`` scans, behind ``_SpanContainment``'s interface."""

    def __init__(self, spans: Iterable[tuple[int, int]]) -> None:
        self._spans = tuple(spans)

    def contains(self, start: int, end: int) -> bool:
        return any(a <= start and end <= b for a, b in self._spans)

    def straddles(self, point: int) -> bool:
        return any(a < point < b for a, b in self._spans)


@functools.cache
def _cache_clears() -> tuple[Callable[[], None], ...]:
    """Every module-level cache in the loaded ``axiom_encode`` modules."""

    return tuple(
        value.cache_clear
        for name, module in tuple(sys.modules.items())
        if name.startswith("axiom_encode")
        for value in tuple(vars(module).values())
        if callable(getattr(value, "cache_clear", None))
    )


def _clear_caches() -> None:
    """Start cold, so one mode never reads a result cached by the other."""

    for cache_clear in _cache_clears():
        cache_clear()
    re.purge()


@contextlib.contextmanager
def _old_mode():
    """Run ``sc`` with main's scans in place of the near-linear ones."""

    with mock.patch.multiple(
        sc,
        _reason_dependency_occurrence_is_contextual=(
            _old_reason_dependency_occurrence_is_contextual
        ),
        _reason_clause_bounds=_old_reason_clause_bounds,
        _is_marker_only_container=_old_is_marker_only_container,
        _source_clause_spans=_old_source_clause_spans,
        _source_clause_boundary_splits_state_code_citation=(
            _old_source_clause_boundary_splits_state_code_citation
        ),
        _span_is_deferred=_old_span_is_deferred,
        _most_specific_containing_branch=_old_most_specific_containing_branch,
        _source_branch_direct_text=_old_source_branch_direct_text,
        _path_is_deferred=_old_path_is_deferred,
        _source_branch_index=lambda branches: None,
        _SpanContainment=_NaiveSpanContainment,
    ):
        yield


def _outcome(call: Callable[[], object]) -> object:
    """A call's value, or the exception it raised, so both can be compared."""

    try:
        return call()
    except Exception as error:  # noqa: BLE001 - the differential compares errors
        return (type(error).__name__, str(error))


def _old_and_new(call: Callable[[], object]) -> tuple[object, object, object]:
    """Main's outcome, then the new one cold, then the new one warm.

    The new ones share one analysis scope, as one analysis's helpers do, so
    the warm call reads the indexes the cold one built. An analyzer call runs
    inside it.
    """

    _clear_caches()
    with _old_mode():
        old = _outcome(call)
    _clear_caches()
    with sc._source_index_scope():
        cold = _outcome(call)
        warm = _outcome(call)
    return old, cold, warm


def _analyze(
    content: str,
    source: str,
    *,
    citation: str,
    test_cases: list[object] | None = None,
    profile: str = "en-US",
):
    return sc.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=test_cases or [],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=(
            functools.partial(
                extract_typed_numeric_occurrences_from_text, profile=profile
            )
            if profile == "en-US"
            else None
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _no_rules(citation: str, deferrals: Sequence[tuple[str, str]] = ()) -> str:
    records = "".join(
        f"    - output: {output}\n      reason: {json.dumps(reason)}\n"
        for output, reason in deferrals
    )
    return (
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {citation}\n"
        + (f"  deferred_outputs:\n{records}" if records else "")
        + "rules: []\n"
    )


def _wide_outline(n: int) -> str:
    """Eight lettered subsections of numbered paragraphs, n lines in all."""

    lines = []
    for letter in "abcdefgh":
        lines.append(f"({letter}) Provision.")
        lines.extend(f"({number}) Provision." for number in range(1, n // 8))
    return "\n".join(lines)


def _formula_outline(n: int) -> str:
    """``_wide_outline`` with a formula and an exception in every paragraph."""

    lines = []
    for letter in "abcdefgh":
        lines.append(f"({letter}) Benefit calculation.")
        lines.extend(
            f"({number}) The benefit equals income multiplied by 2 unless "
            "residence is unverified."
            for number in range(1, n // 8)
        )
    return "\n".join(lines)


def _deep_outline(n: int) -> str:
    per = max(1, n // 32)
    lines = []
    for letter in "abcdefgh":
        lines.append(f"({letter}) Provision.")
        for number in range(1, 5):
            lines.append(f"({number}) Provision.")
            lines.extend(
                f"({chr(ord('A') + item)}) Provision." for item in range(per - 1)
            )
    return "\n".join(lines)


def _cited_outline(n: int) -> str:
    return "\n".join(
        f"({k}) The agency shall determine eligibility under section {3000 + k}."
        for k in range(1, n + 1)
    )


def _half_deferred(n: int, *, reason: str) -> str:
    """Precisely defer every odd paragraph of ``_cited_outline(n)``."""

    return _no_rules(
        SNAP_CITATION,
        [
            (
                f"us:statutes/7/2014/{k}#eligibility_{k}",
                reason.format(k=k, section=3000 + k),
            )
            for k in range(1, n + 1, 2)
        ],
    )


GERMAN_SOURCE = (
    "(1) Ein Kind wird bis zur Vollendung des 18. Lebensjahres berücksichtigt. "
    "Satz 1 gilt auch nach dem 25.\nLebensjahr. Ein Kind, das das 21. Lebensjahr "
    "noch nicht vollendet hat; es gilt R.S.\n47:297 entsprechend.\n"
    "(2) 1. Ein Betrag von 250 Euro. 2. Ein Betrag von 300 Euro; zuzüglich "
    "der Summe aus: Absatz 1 und Absatz 2.\n(3) Satz 1. Satz 2 gilt nicht."
)


def _fixture_cases():
    """Every analyzer input the source-completeness fixtures hold."""

    for folder in sorted(FIXTURES.glob("signed_reencode_*")):
        recorded = json.loads((folder / "issues.json").read_text())
        source = (folder / "source.txt").read_text()
        yield pytest.param(
            (folder / "candidate.yaml").read_text(),
            source,
            recorded["citation"],
            yaml.safe_load((folder / "candidate.test.yaml").read_text()),
            "en-US",
            id=folder.name,
        )
        yield pytest.param(
            _no_rules(recorded["citation"]),
            source,
            recorded["citation"],
            None,
            "en-US",
            id=f"{folder.name}-no-rules",
        )
    estg = FIXTURES / "estg_66_rounding_input_stage"
    for variant in ("rejected", "minimally_fixed"):
        yield pytest.param(
            (estg / variant / "66.yaml").read_text(),
            (estg / "source.txt").read_text().removesuffix("\n"),
            "de/statute/estg/66",
            yaml.safe_load((estg / variant / "66.test.yaml").read_text()),
            "de-DE",
            id=f"estg_66-{variant}",
        )
    housing = FIXTURES / "irs_housing_formula"
    yield pytest.param(
        (housing / "rule.yaml").read_text(),
        (housing / "source.txt").read_text(),
        "us/guidance/irs/rev-proc-2025-32/page-15",
        yaml.safe_load((housing / "rule.test.yaml").read_text()),
        "en-US",
        id="irs_housing_formula",
    )
    rev_proc = FIXTURES / "irs_rev_proc_2025_32"
    page_14 = json.loads((rev_proc / "page-14.json").read_text())["body"]
    for variant in ("max_formula", "precise_deferral"):
        yield pytest.param(
            (rev_proc / variant / "child-tax-credit.yaml").read_text(),
            page_14,
            "us/guidance/irs/rev-proc-2025-32/page-14",
            yaml.safe_load(
                (rev_proc / variant / "child-tax-credit.test.yaml").read_text()
            ),
            "en-US",
            id=f"irs_rev_proc_2025_32-{variant}",
        )
    for name, citation in (
        (
            "fns_snap_fy2026_cola_memo_page_3",
            "us/guidance/usda/fns/snap-fy2026-cola/page-3",
        ),
        (
            "fns_snap_fy2026_income_standards_page_2",
            "us/guidance/usda/fns/snap-fy2026-income-eligibility-standards/page-2",
        ),
        (
            "fns_snap_fy2026_income_standards_composed",
            "us/guidance/usda/fns/snap-fy2026-income-eligibility-standards/page-2",
        ),
        ("irs_rev_proc_2025_32_page_15", "us/guidance/irs/rev-proc-2025-32/page-15"),
        ("quebec_2025_work_charts", "ca/policy/revenu-quebec/tp1-2025/main-return"),
        ("quebec_work_charts_prefix", "ca/policy/revenu-quebec/tp1-2025/main-return"),
    ):
        yield pytest.param(
            _no_rules(citation),
            (FIXTURES / f"{name}.txt").read_text(),
            citation,
            None,
            "en-US",
            id=f"{name}-no-rules",
        )
    yield pytest.param(
        _no_rules(SNAP_CITATION),
        _wide_outline(48),
        SNAP_CITATION,
        None,
        "en-US",
        id="wide-outline",
    )
    yield pytest.param(
        _no_rules(SNAP_CITATION),
        _deep_outline(96),
        SNAP_CITATION,
        None,
        "en-US",
        id="deep-outline",
    )
    for label, reason in (
        (
            "precise",
            "Paragraph ({k}) cannot be encoded until eligibility under section "
            "{section} is available.",
        ),
        (
            "of-this-title",
            "Paragraph ({k}) cannot be encoded until eligibility under section "
            "{section} of this title is available; 7 U.S.C. {section} is quoted "
            "only for context, but 42 U.S.C. 1382a(a)(2) is missing.",
        ),
        (
            "contextual-usc",
            "7 U.S.C. {section} is cited for orientation. Paragraph ({k}) cannot "
            "be encoded until 7 U.S.C. {section} is available, although section "
            "{section} is merely illustrative.",
        ),
    ):
        yield pytest.param(
            _half_deferred(24, reason=reason),
            _cited_outline(24),
            SNAP_CITATION,
            None,
            "en-US",
            id=f"half-deferred-{label}",
        )
    # Attached markers nest one level each, so branch paths run 25 deep.
    attached = (
        "A.\n" + "(a)" * 24 + " Eligibility under section 3001.\n"
        "B. Eligibility under section 3002."
    )
    for label, deferrals in (
        ("no-rules", ()),
        (
            "deep-deferral",
            (
                (
                    "us:statutes/7/2014/a/a/a#eligibility_a",
                    "Paragraph (a) cannot be encoded until eligibility under "
                    "section 3001 is available.",
                ),
            ),
        ),
    ):
        yield pytest.param(
            _no_rules(SNAP_CITATION, deferrals),
            attached,
            SNAP_CITATION,
            None,
            "en-US",
            id=f"attached-markers-{label}",
        )
    yield pytest.param(
        _no_rules("de/statute/estg/32"),
        GERMAN_SOURCE,
        "de/statute/estg/32",
        None,
        "de-DE",
        id="german-sentences",
    )


FIXTURE_CASES = tuple(_fixture_cases())


@pytest.mark.parametrize(
    ("content", "source", "citation", "test_cases", "profile"), FIXTURE_CASES
)
def test_fixture_analysis_matches_main(content, source, citation, test_cases, profile):
    old, cold, warm = _old_and_new(
        lambda: _analyze(
            content, source, citation=citation, test_cases=test_cases, profile=profile
        )
    )

    assert cold == old
    assert warm == cold


def test_fixture_differential_reaches_every_replaced_scan():
    """The fixtures run main's copy of every replaced scan, not a bypass."""

    reached: set[str] = set()

    def counted(name: str, function: Callable):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            reached.add(name)
            return function(*args, **kwargs)

        return wrapper

    # Main's ``_source_clause_spans`` calls its state-code splitter directly,
    # so the splitter runs inside it; ``test_clause_spans_match_main`` compares
    # the splitter itself.
    names = (
        "_reason_dependency_occurrence_is_contextual",
        "_reason_clause_bounds",
        "_is_marker_only_container",
        "_source_clause_spans",
        "_span_is_deferred",
        "_most_specific_containing_branch",
        "_source_branch_direct_text",
        "_path_is_deferred",
    )
    _clear_caches()
    with _old_mode():
        with mock.patch.multiple(
            sc, **{name: counted(name, getattr(sc, name)) for name in names}
        ):
            for case in FIXTURE_CASES:
                content, source, citation, test_cases, profile = case.values
                _outcome(
                    lambda: _analyze(
                        content,
                        source,
                        citation=citation,
                        test_cases=test_cases,
                        profile=profile,
                    )
                )

    assert reached == set(names)


def _fixture_sources():
    seen = set()
    for case in FIXTURE_CASES:
        _content, source, citation, _tests, _profile = case.values
        if (source, citation) not in seen:
            seen.add((source, citation))
            yield pytest.param(source, citation, id=case.id)


def _deferral_sets(branches: Sequence[SourceStructureBranch]):
    paths = sorted({branch.path for branch in branches})
    for deferred in (
        set(),
        {()},
        set(paths[::2]),
        {path[:1] for path in paths if path},
        {path for path in paths if len(path) > 1},
    ):
        # An analysis passes a frozen set, which is indexed; others are scanned.
        yield deferred
        yield frozenset(deferred)


@pytest.mark.parametrize(("source", "citation"), tuple(_fixture_sources()))
def test_fixture_branch_helpers_match_main(source, citation):
    branches = sc.recognize_source_structure(source, corpus_citation_path=citation)

    def helpers(pool):
        spans = [
            (start, end)
            for start, end, _text in sc._source_clause_spans(source, branches=pool)
        ]
        spans += [(branch.start, branch.end) for branch in pool]
        deferral_sets = tuple(_deferral_sets(pool))
        return (
            spans,
            [
                (
                    sc._span_is_deferred(
                        start, end, branches=pool, deferred_paths=deferred
                    )
                    for deferred in deferral_sets
                )
                for start, end in spans
            ],
            [
                _identity(
                    pool, sc._most_specific_containing_branch(start, end, branches=pool)
                )
                for start, end in spans
            ],
            [
                (
                    sc._is_marker_only_container(
                        branch, branches=pool, source_text=source
                    ),
                    sc._source_branch_direct_text(branch, branches=pool),
                    tuple(
                        sc._path_is_deferred(branch.path, deferred)
                        for deferred in deferral_sets
                    ),
                )
                for branch in pool
            ],
        )

    for pool in (branches, list(branches), ()):
        old, cold, warm = _old_and_new(lambda: _materialize(helpers(pool)))
        assert cold == old
        assert warm == cold


def _materialize(value):
    if isinstance(value, list | tuple):
        return [_materialize(item) for item in value]
    if hasattr(value, "__next__"):
        return [_materialize(item) for item in value]
    return value


def _identity(pool: Sequence[SourceStructureBranch], branch) -> int | None:
    """Which object a helper returned, since equal branches can repeat."""

    if branch is None:
        return None
    return next(index for index, candidate in enumerate(pool) if candidate is branch)


def _reason_matches(reason: str) -> list[re.Match[str]]:
    return [
        *sc._reason_dependencies(reason),
        *sc._PRECISE_DEFERRAL_DEPENDENCY.finditer(reason),
        *sc._qualified_usc_dependencies(reason),
        *sc._qualified_louisiana_rs_dependencies(reason),
        *sc._RELATIVE_USC_DEFERRAL_DEPENDENCY.finditer(reason),
    ]


def _reason_helpers(reason: str) -> list[object]:
    return [
        (
            sc._reason_clause_bounds(reason, match),
            tuple(
                sc._reason_dependency_occurrence_is_contextual(
                    reason, match, current_usc_title=title
                )
                for title in (None, "7", "42", "47")
            ),
        )
        for match in _reason_matches(reason)
    ] + [
        sc._reason_dependency_is_source_bound(
            reason, scope, corpus_citation_path=citation, path=path
        )
        for scope, citation, path in (
            ("(a) Under section 345.", SNAP_CITATION, ("a",)),
            ("(1) Under R.S. 47:297.", "us-la/statute/47/293", ("1",)),
            ("(a) Under 42 U.S.C. 1382a.", "us/statute/42/1382a", None),
        )
    ]


def _fixture_reasons() -> list[str]:
    reasons = []
    for case in FIXTURE_CASES:
        payload = yaml.safe_load(case.values[0])
        module = payload.get("module") if isinstance(payload, dict) else None
        for record in (module or {}).get("deferred_outputs") or ():
            if isinstance(record, dict) and isinstance(record.get("reason"), str):
                reasons.append(record["reason"])
    return reasons


def test_fixture_reason_helpers_match_main():
    reasons = _fixture_reasons()
    assert reasons

    for reason in reasons:
        old, cold, warm = _old_and_new(lambda: _reason_helpers(reason))
        assert cold == old, reason
        assert warm == cold, reason


_REASON_FRAGMENTS = (
    "section 346",
    "section 5",
    "sections 3 and 4",
    "section 2015(d) of this title",
    "section 2015",
    "section 2015(d)(1)",
    "section 1382a(a)(2)",
    "of this title",
    "7 U.S.C. 2015",
    "7 U.S.C. 2015(d)(1)",
    "42 U.S.C. 1382a(a)(2)",
    "R.S. 47:297",
    "R.S. 47:293(1)",
    "subsection (b)",
    "paragraph (2)",
    "is missing",
    "is not yet available",
    "cannot be encoded until",
    "Paragraph (a)",
    "but",
    "although",
    "unless",
    "and",
    "or",
    "under",
    "is quoted only for context",
    "legislative history",
    "for orientation",
    "merely illustrative",
    ".",
    ";",
    "\n",
    ",",
)


# A relative section is the same citation only when "of this title" follows it.
_THIS_TITLE_REASONS = (
    "7 U.S.C. 2015 is missing. section 2015 is quoted only for context.",
    "7 U.S.C. 2015(d) is missing; section 2015(d) of this title is quoted only "
    "for context.",
    "section 2015(d)(1) of this title, for orientation, and 7 U.S.C. 2015(d)(1) "
    "is missing.",
)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.sampled_from(_REASON_FRAGMENTS), max_size=24).map(" ".join))
@example(_THIS_TITLE_REASONS[0])
@example(_THIS_TITLE_REASONS[1])
@example(_THIS_TITLE_REASONS[2])
def test_reason_helpers_match_main(reason):
    old, cold, warm = _old_and_new(lambda: _reason_helpers(reason))

    assert cold == old
    assert warm == cold


class _SubclassReason(str):
    """A reason string of a subclass that changes nothing."""


class _UnhashableReason(str):
    """A reason string that cannot be hashed."""

    __hash__ = None  # type: ignore[assignment]


class _CaseFoldReason(str):
    """A reason string equal to, and hashed as, any of its type and folded text."""

    def __hash__(self) -> int:
        return hash(self.casefold())

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and self.casefold() == other.casefold()

    def __ne__(self, other: object) -> bool:
        return not self == other


class _LooseReason(str):
    """A reason string equal to everything, all hashed alike."""

    def __hash__(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        return True


class _BackwardReason(str):
    """A reason string whose items and slices read it backwards."""

    def __getitem__(self, key):
        return str.__str__(self)[::-1][key]


def _reason_outcomes(reasons: Sequence[str]) -> list[object]:
    """Each match's clause bounds and contextual verdicts, raised or returned."""

    return [
        (
            _outcome(lambda: sc._reason_clause_bounds(reason, match)),
            *(
                _outcome(
                    lambda: sc._reason_dependency_occurrence_is_contextual(
                        reason, match, current_usc_title=title
                    )
                )
                for title in (None, "7", "42", "47")
            ),
        )
        for reason in reasons
        for match in _reason_matches(reason)
    ]


_SUBCLASS_REASONS = (
    "missing section 346.",
    "ß; missing section 346.",
    "ss; missing section 346.",
    "Paragraph (a) under 7 U.S.C. 2015 is missing; section 346 is missing.",
    *_THIS_TITLE_REASONS,
)


@pytest.mark.parametrize(
    "make",
    [
        _SubclassReason,
        _UnhashableReason,
        _CaseFoldReason,
        _LooseReason,
        _BackwardReason,
    ],
)
def test_reason_helpers_match_main_on_string_subclasses(make):
    # Recognition's reasons are plain strings. One of a subclass may hash,
    # compare or slice unlike its text, so it is neither indexed nor cached:
    # each reason, here in one analysis, is scanned as main scanned it.
    reasons = [make(reason) for reason in _SUBCLASS_REASONS]

    old, cold, warm = _old_and_new(lambda: _reason_outcomes(reasons))

    assert cold == old
    assert warm == cold


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(
        st.tuples(
            st.sampled_from(
                [
                    _SubclassReason,
                    _UnhashableReason,
                    _CaseFoldReason,
                    _LooseReason,
                    _BackwardReason,
                ]
            ),
            st.lists(st.sampled_from(_REASON_FRAGMENTS), max_size=12).map(" ".join),
        ),
        max_size=4,
    )
)
def test_reason_helpers_match_main_on_mixed_string_subclasses(cases):
    reasons = [make(reason) for make, reason in cases]

    old, cold, warm = _old_and_new(lambda: _reason_outcomes(reasons))

    assert cold == old
    assert warm == cold


@pytest.mark.parametrize(
    ("reason", "contextual"),
    [
        # The adversative clause is the second after the citation, not the next.
        (
            "section 346 is missing. It is not yet available. But the rate is known.",
            False,
        ),
        # The next clause is adversative, and a later one cites a dependency.
        ("section 346 is missing. But the rate is known; section 5 is pending.", True),
    ],
)
def test_trailing_clause_ends_at_the_next_separator(reason, contextual):
    match = sc._reason_dependencies(reason)[0]

    old, cold, warm = _old_and_new(
        lambda: sc._reason_dependency_occurrence_is_contextual(
            reason, match, current_usc_title=None
        )
    )

    assert old is contextual
    assert cold is old
    assert warm is cold


def _local_contexts(reason: str) -> tuple[list[str], list[str]]:
    """Each occurrence's local context, from the index and from main's scan."""

    index = sc._ReasonOccurrenceIndex(reason)
    dependencies = sc._reason_dependencies(reason)
    matches = [*index.candidates, *dependencies]
    return (
        [index.local_context(match) for match in matches],
        [
            _old_reason_dependency_local_context(
                reason, match, dependencies=dependencies
            )
            for match in matches
        ],
    )


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.sampled_from(_REASON_FRAGMENTS), max_size=24).map(" ".join))
@example("section 2015(d)(1) is quoted only for context.")
@example("7 U.S.C. 2015(d)(1) and section 2015 of this title, for orientation.")
def test_local_context_matches_main(reason):
    # Nested citations (a relative section inside a qualified one, or a
    # section inside its own subsection) make the latest-ending dependency
    # before a match differ from the latest-starting one.
    new, old = _local_contexts(reason)

    assert new == old


def test_fixture_local_contexts_match_main():
    for reason in (*_fixture_reasons(), *_THIS_TITLE_REASONS):
        new, old = _local_contexts(reason)
        assert new == old, reason


def test_sentence_direct_text_stops_at_its_numbered_items():
    """A German sentence label shares its paragraph path with the items in it."""

    text = "(1) Satz 1 gilt für: 1. erstens, 2. zweitens. Satz 2 gilt nicht."

    def span(path, kind, label, start_text, end_text=None):
        start = text.index(start_text)
        end = text.index(end_text) if end_text else len(text)
        return SourceStructureBranch(
            path=path,
            kind=kind,
            label=label,
            text=text[start:end],
            start=start,
            end=end,
        )

    sentence = span(("1", "satz-1"), "sentence", "Satz 1", "Satz 1", " Satz 2")
    branches = (
        span(("1",), "paragraph", "(1)", "(1)"),
        sentence,
        span(("1", "1"), "number", "1.", "1. erstens", "2. zweitens"),
        span(("1", "2"), "number", "2.", "2. zweitens", " Satz 2"),
        span(("1", "satz-2"), "sentence", "Satz 2", "Satz 2"),
    )

    assert sc._source_branch_direct_text(sentence, branches=branches) == (
        "Satz 1 gilt für: "
    )
    for pool in (branches, list(branches)):
        old, cold, warm = _old_and_new(
            lambda: [
                sc._source_branch_direct_text(branch, branches=pool) for branch in pool
            ]
        )
        assert cold == old
        assert warm == cold


_PATH_PARTS = ("a", "b", "1", "2", "satz-1", "satz-2")
_KINDS = (
    "paragraph",
    "subsection",
    "sentence",
    "number",
    "letter",
    "guidance-subsection",
)


@st.composite
def _branch_pools(draw):
    text = draw(st.text(alphabet="ab (1).;\n", max_size=40))
    positions = st.integers(0, len(text))
    branches = []
    for _ in range(draw(st.integers(0, 10))):
        start = draw(positions)
        end = draw(st.one_of(st.integers(start, len(text)), positions))
        branches.append(
            SourceStructureBranch(
                path=tuple(draw(st.lists(st.sampled_from(_PATH_PARTS), max_size=4))),
                kind=draw(st.sampled_from(_KINDS)),
                label=draw(st.sampled_from(("", "(a)", "(1)", "a"))),
                text=text[start:end],
                start=start,
                end=end,
            )
        )
    if branches and draw(st.booleans()):
        # One object twice and an equal copy: ties must break the same way.
        repeated = draw(st.sampled_from(branches))
        branches.append(repeated)
        branches.insert(
            draw(st.integers(0, len(branches))),
            SourceStructureBranch(
                path=repeated.path,
                kind=repeated.kind,
                label=repeated.label,
                text=repeated.text,
                start=repeated.start,
                end=repeated.end,
            ),
        )
    paths = [branch.path for branch in branches]
    prefixes = [path[:cut] for path in paths for cut in range(len(path) + 1)]
    deferred = draw(st.sets(st.sampled_from(prefixes))) if prefixes else set()
    spans = draw(
        st.lists(
            st.tuples(st.integers(-1, len(text) + 1), st.integers(-1, len(text) + 1)),
            max_size=8,
        )
    )
    return text, tuple(branches), deferred, spans


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_branch_pools())
def test_branch_helpers_match_main_on_random_pools(case):
    text, branches, deferred_paths, spans = case
    spans = spans + [(branch.start, branch.end) for branch in branches]

    def helpers(pool, deferred):
        return (
            [
                (
                    sc._span_is_deferred(
                        start, end, branches=pool, deferred_paths=deferred
                    ),
                    _identity(
                        pool,
                        sc._most_specific_containing_branch(start, end, branches=pool),
                    ),
                )
                for start, end in spans
            ],
            [
                (
                    sc._is_marker_only_container(
                        branch, branches=pool, source_text=text
                    ),
                    sc._source_branch_direct_text(branch, branches=pool),
                    sc._path_is_deferred(branch.path, deferred),
                )
                for branch in pool
            ],
        )

    for pool in (branches, list(branches)):
        for deferred in (deferred_paths, frozenset(deferred_paths)):
            old, cold, warm = _old_and_new(lambda: helpers(pool, deferred))
            assert cold == old
            assert warm == cold


def _path_filter(
    branches: Sequence[SourceStructureBranch], path: tuple[str, ...]
) -> list[SourceStructureBranch]:
    """Main's descendant test: a longer path that starts with ``path``."""

    return [
        candidate
        for candidate in branches
        if len(candidate.path) > len(path) and candidate.path[: len(path)] == path
    ]


def _first_start(
    branches: Sequence[SourceStructureBranch], branch: SourceStructureBranch
) -> int | None:
    return min(
        (
            candidate.start
            for candidate in _path_filter(branches, branch.path)
            if branch.start <= candidate.start < branch.end
        ),
        default=None,
    )


@pytest.mark.parametrize(("source", "citation"), tuple(_fixture_sources()))
def test_fixture_branch_index_descendants_are_the_path_filter(source, citation):
    """Where the index answers, its descendants are exactly main's filter's."""

    branches = sc.recognize_source_structure(source, corpus_citation_path=citation)
    if not branches:
        return
    with sc._source_index_scope():
        index = sc._source_branch_index(branches)
        assert index._knows_lowest_starts
        for branch in branches:
            for cut in range(len(branch.path) + 1):
                path = branch.path[:cut]
                assert list(map(id, index.descendants(path))) == list(
                    map(id, _path_filter(branches, path))
                )
            assert index.first_descendant_start(branch) == _first_start(
                branches, branch
            )


_ODD_STARTS = st.one_of(
    st.integers(-1, 41), st.booleans(), st.floats(-1, 41), st.none()
)


@st.composite
def _odd_branch_pools(draw):
    """Pools the index must refuse or answer exactly as main's scan does."""

    branches = []
    for _ in range(draw(st.integers(0, 10))):
        parts = draw(st.lists(st.sampled_from(_PATH_PARTS[:3]), max_size=5))
        shape = draw(
            st.sampled_from(("tuple",) * 6 + ("string", "list-part", "nested"))
        )
        path = {
            "tuple": tuple(parts),
            # A trie would split a string into characters.
            "string": "".join(parts),
            "list-part": (*parts, ["x"]),
            "nested": (*parts, ("a",)),
        }[shape]
        integral = draw(st.booleans()) or draw(st.booleans())
        start = draw(st.integers(0, 40)) if integral else draw(_ODD_STARTS)
        end = draw(st.integers(0, 41)) if integral else draw(_ODD_STARTS)
        branches.append(
            SourceStructureBranch(
                path=path,
                kind=draw(st.sampled_from(_KINDS)),
                label="(a)",
                text="x" * 41,
                start=start,
                end=end,
            )
        )
    # Branches from another tuple: an equal path in a new object, which the
    # index must look up by value, and paths no branch in the pool has.
    strangers = [
        dataclasses.replace(branch, path=tuple(list(branch.path)))
        for branch in branches
        if type(branch.path) is tuple
    ]
    strangers += [
        SourceStructureBranch(
            path=tuple(draw(st.lists(st.sampled_from(_PATH_PARTS[:3]), max_size=3))),
            kind="paragraph",
            label="(a)",
            text="x" * 41,
            start=draw(st.integers(0, 40)),
            end=draw(st.integers(0, 41)),
        )
        for _ in range(draw(st.integers(0, 3)))
    ]
    # Paths the index must not answer for, whatever the pool holds.
    strangers += [
        SourceStructureBranch(
            path=path,
            kind="paragraph",
            label="(a)",
            text="x" * 41,
            start=draw(st.integers(0, 40)),
            end=draw(st.integers(0, 41)),
        )
        for path in draw(
            st.lists(
                st.sampled_from([("a", ("a",)), ("a", ["x"]), ("a", 1), "ab", ""]),
                max_size=2,
            )
        )
    ]
    return tuple(branches), strangers


@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_odd_branch_pools())
@example(
    # A child that starts before its parent: the least start is outside the
    # parent's span, so the index must find the first one inside it.
    (
        (
            SourceStructureBranch(("a",), "paragraph", "(a)", "x" * 41, 10, 30),
            SourceStructureBranch(("a", "b"), "letter", "(b)", "x" * 41, 2, 5),
            SourceStructureBranch(("a", "1"), "letter", "(1)", "x" * 41, 20, 25),
        ),
        [],
    )
)
@example(
    # The same, but no descendant starts inside the parent's span, though one
    # starts after it: the parent keeps its whole text.
    (
        (
            SourceStructureBranch(("a",), "paragraph", "(a)", "x" * 41, 10, 15),
            SourceStructureBranch(("a", "b"), "letter", "(b)", "x" * 41, 2, 5),
            SourceStructureBranch(("a", "1"), "letter", "(1)", "x" * 41, 20, 25),
        ),
        [],
    )
)
@example(
    # An empty string path has no last component for the trie to walk to.
    (
        (SourceStructureBranch("", "paragraph", "(a)", "x" * 41, 0, 0),),
        [],
    )
)
def test_direct_text_matches_main_on_odd_pools(case):
    branches, strangers = case
    text = "x" * 41

    def direct_texts(pool):
        return [
            (
                _outcome(lambda: sc._source_branch_direct_text(branch, branches=pool)),
                _outcome(
                    lambda: sc._is_marker_only_container(
                        branch, branches=pool, source_text=text
                    )
                ),
            )
            for branch in (*branches, *strangers)
        ]

    for pool in (branches, list(branches)):
        old, cold, warm = _old_and_new(lambda: direct_texts(pool))
        assert cold == old
        assert warm == cold
    if not branches:
        return
    with sc._source_index_scope():
        index = sc._source_branch_index(branches)
        if not index._knows_lowest_starts:
            return
        for branch in (*branches, *strangers):
            if type(branch.start) is not int or type(branch.end) is not int:
                continue
            if sc._plain_path(branch.path):
                assert index.first_descendant_start(branch) == _first_start(
                    branches, branch
                )
            else:
                assert index.first_descendant_start(branch) is sc._NO_INDEX_ANSWER


@settings(max_examples=300, deadline=None)
@given(
    st.lists(st.tuples(st.integers(-3, 30), st.integers(-3, 30)), max_size=12),
    st.lists(st.tuples(st.integers(-4, 31), st.integers(-4, 31)), max_size=12),
)
def test_span_containment_matches_any_scan(spans, queries):
    fast = sc._SpanContainment(spans)
    naive = _NaiveSpanContainment(spans)

    for start, end in queries:
        assert fast.contains(start, end) == naive.contains(start, end)
        assert fast.straddles(start) == naive.straddles(start)


_SOURCE_FRAGMENTS = (
    "R.S.",
    "r.s.",
    " 47:297",
    "47:293A",
    "12.",
    "2.",
    " Lebensjahr",
    " Lebensjahres",
    " Lebensjahren",
    "\n",
    ";",
    ".",
    ". ",
    ": ",
    "Provided, that",
    " provided that",
    " if",
    " when",
    "(a)",
    "(1)",
    "(A)",
    "Satz 1",
    " The",
    " amount",
    "x7",
    "$5",
    " plus",
    " the sum of:",
    "(i)",
    " Guidance documents lack the force and effect of law",
)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.sampled_from(_SOURCE_FRAGMENTS), max_size=30).map("".join))
def test_clause_spans_match_main(text):
    branches = sc.recognize_source_structure(text)

    def spans():
        return [
            list(sc._source_clause_spans(text, branches=pool))
            for pool in ((), branches, list(branches))
        ] + [
            sc._source_clause_boundary_splits_state_code_citation(text, boundary)
            for boundary in re.finditer(r"[.;]", text)
        ]

    old, cold, warm = _old_and_new(spans)

    assert cold == old
    assert warm == cold


@settings(max_examples=300, deadline=None)
@given(st.lists(st.sampled_from(_SOURCE_FRAGMENTS), max_size=12).map("".join))
# ``$`` also matches before a final newline, so the period can sit one before.
@example("12.\n")
@example("R.S.\n47:297")
def test_windowed_matches_equal_sliced_matches(text):
    for end in range(len(text) + 1):
        assert sc._number_ends_before(text, end) == bool(
            re.search(r"(?<!\w)\d+\.$", text[:end])
        )
        assert bool(
            sc._STATE_CODE_CITATION_TITLE.search(text, max(0, end - 6), end)
        ) == bool(re.search(r"\bR\.S\.$", text[:end], re.IGNORECASE))
        assert bool(sc._STATE_CODE_CITATION_SECTION.match(text, end)) == bool(
            re.match(r"\s*\d+[A-Za-z]?:\d+[A-Za-z]?", text[end:])
        )
        assert bool(sc._LEBENSJAHR_FOLLOWS.match(text, end)) == bool(
            re.match(r"\s+Lebensjahr(?:es|e|en)?\b", text[end:])
        )
        assert bool(sc._FORMULA_FOLLOWING_OPERAND_PROVISO.match(text, end)) == bool(
            sc._FORMULA_FOLLOWING_OPERAND_PROVISO.match(text[end:])
        )
        assert bool(sc._OF_THIS_TITLE.match(text, end)) == bool(
            re.match(r"\s+of\s+this\s+title\b", text[end:], flags=re.IGNORECASE)
        )


def test_branch_index_is_kept_per_tuple_for_one_scope():
    branches = sc.recognize_source_structure(_wide_outline(32))
    equal_copy = tuple(list(branches))

    # Outside an analysis the helpers scan.
    assert sc._source_branch_index(branches) is None
    with sc._source_index_scope():
        index = sc._source_branch_index(branches)
        assert index is not None
        assert sc._source_branch_index(branches) is index
        with sc._source_index_scope():
            assert sc._source_branch_index(branches) is index
        assert sc._source_branch_index(equal_copy) is not index
        assert sc._source_branch_index(list(branches)) is None
        assert sc._source_branch_index(()) is None
    assert sc._SOURCE_INDEXES.get() is None
    with sc._source_index_scope():
        assert sc._source_branch_index(branches) is not index


class _RaisingComponent:
    """A hashable path component whose comparison raises; recognition never builds one."""

    def __hash__(self) -> int:
        return 7

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("compared")


class _Text(str):
    """A string subclass, which recognition never builds."""


class _Loose(str):
    """A string subclass equal to everything but hashed as its own text."""

    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        return True


class _LooseTuple(tuple):
    """A tuple subclass equal to everything but hashed as its own items."""

    __hash__ = tuple.__hash__

    def __eq__(self, other: object) -> bool:
        return True


class _Reversed(tuple):
    """A tuple subclass whose items and slices read it backwards."""

    def __getitem__(self, key):
        return tuple(reversed(tuple(self)))[key]


class _PassesForStr(type):
    """A metaclass whose classes hash and compare as ``str`` itself does."""

    def __hash__(cls) -> int:
        return hash(str)

    def __eq__(cls, other: object) -> bool:
        return other is str


class _DisguisedLoose(str, metaclass=_PassesForStr):
    """A loose string subclass whose type a set of types takes for ``str``."""

    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        return True


class _UnhashableType(type):
    """A metaclass whose classes cannot be hashed."""

    __hash__ = None  # type: ignore[assignment]


class _UnhashablyTyped(metaclass=_UnhashableType):
    """An ordinary object, whose class cannot be hashed."""


class _HashRaises:
    """A branch kind whose hashing raises; recognition never builds one."""

    def __hash__(self) -> int:
        raise RuntimeError("hashed")


_RAISING = _RaisingComponent()
_DISGUISED = _DisguisedLoose("z")
_UNHASHABLY_TYPED = _UnhashablyTyped()
_ANY_PATHS = [
    ["a", "b"],
    "ab",
    ("a", "b"),
    ("b",),
    (),
    (["x"],),
    ("a", ["x"]),
    ("ab",),
    (_RAISING,),
    ("a", _RAISING),
    (_Text("a"),),
    (_Loose("z"),),
    _Reversed(("b", "a")),
    _LooseTuple(("z",)),
    (_DISGUISED,),
    (_UNHASHABLY_TYPED,),
    ("a", _UNHASHABLY_TYPED),
]


@pytest.mark.parametrize("path", _ANY_PATHS)
@pytest.mark.parametrize(
    "deferred",
    [
        frozenset({("a",)}),
        {("a",)},
        frozenset({()}),
        frozenset(),
        frozenset({"ab"}),
        frozenset({"a", ("a",)}),
        frozenset({("a", "b")}),
        frozenset({(_RAISING,)}),
        frozenset({(_Text("a"),)}),
        frozenset({(_Loose("z"),)}),
        frozenset({_Reversed(("b", "a"))}),
        frozenset({_LooseTuple(("z",))}),
        frozenset({(_DISGUISED,)}),
        frozenset({("a", _UNHASHABLY_TYPED)}),
    ],
)
def test_path_is_deferred_matches_main_on_any_path(path, deferred):
    # Main compared slices, so a list or string path never matched a tuple, and
    # a string deferral never matched a tuple path; neither raised. A component
    # whose comparison raises raised only where main compared it.
    old = _outcome(lambda: _old_path_is_deferred(path, deferred))

    assert _outcome(lambda: sc._path_is_deferred(path, deferred)) == old
    with sc._source_index_scope():
        assert _outcome(lambda: sc._path_is_deferred(path, deferred)) == old
        assert _outcome(lambda: sc._path_is_deferred(path, deferred)) == old


def test_branch_helpers_match_main_on_unhashable_paths():
    source = _wide_outline(16)
    branches = sc.recognize_source_structure(source)
    # One branch path with a list component, which no dict can key.
    odd = tuple(
        dataclasses.replace(branch, path=(*branch.path[:-1], [branch.path[-1]]))
        if position == 3
        else branch
        for position, branch in enumerate(branches)
    )
    deferred = frozenset({branches[0].path, branches[5].path})

    def helpers():
        return [
            (
                _outcome(
                    lambda: sc._is_marker_only_container(
                        branch, branches=odd, source_text=source
                    )
                ),
                _outcome(lambda: sc._source_branch_direct_text(branch, branches=odd)),
                _outcome(
                    lambda: sc._span_is_deferred(
                        branch.start, branch.end, branches=odd, deferred_paths=deferred
                    )
                ),
                _outcome(
                    lambda: _identity(
                        odd,
                        sc._most_specific_containing_branch(
                            branch.start, branch.end, branches=odd
                        ),
                    )
                ),
                _outcome(lambda: sc._path_is_deferred(branch.path, deferred)),
            )
            for branch in odd
        ]

    old, cold, warm = _old_and_new(helpers)

    assert len(branches) > 5
    assert cold == old
    assert warm == cold


def test_span_is_deferred_tries_containers_in_branch_order():
    # Main tried containers in branch order and stopped at the first deferred
    # one, so a container whose path cannot be compared raised only when it
    # came first. The lazy search must try them in the same order.
    outer = SourceStructureBranch(("a",), "paragraph", "(a)", "x" * 10, 0, 10)
    odd = SourceStructureBranch(5, "paragraph", "(b)", "x" * 8, 1, 9)
    deferred = frozenset({("a",)})
    in_order = (outer, odd)
    out_of_order = (odd, outer)

    for pool, expected in ((in_order, True), (out_of_order, "TypeError")):
        old, cold, warm = _old_and_new(
            lambda: sc._span_is_deferred(2, 8, branches=pool, deferred_paths=deferred)
        )
        assert old == cold == warm
        assert (old if old is True else old[0]) == expected
    with sc._source_index_scope():
        assert sc._source_branch_index(in_order)._spans[4]
        assert not sc._source_branch_index(out_of_order)._spans[4]


def test_recognized_branches_take_the_lazy_containment_search():
    for case in FIXTURE_CASES:
        _content, source, citation, _tests, _profile = case.values
        branches = sc.recognize_source_structure(source, corpus_citation_path=citation)
        if branches:
            with sc._source_index_scope():
                assert sc._source_branch_index(branches)._spans[4], case.id


def test_deferral_index_is_kept_per_frozen_set_for_one_scope():
    source = _cited_outline(8)
    branches = sc.recognize_source_structure(source, corpus_citation_path=SNAP_CITATION)
    payload = yaml.safe_load(_half_deferred(8, reason=_PRECISE_REASON))
    covered, issues = sc._deferred_coverage(
        payload,
        corpus_citation_path=SNAP_CITATION,
        source_text=source,
        branches=branches,
    )

    assert type(covered) is frozenset
    assert covered == {("1",), ("3",), ("5",), ("7",)}
    assert not issues
    assert sc._deferred_path_index(covered) is None
    with sc._source_index_scope():
        index = sc._deferred_path_index(covered)
        assert index is not None
        assert sc._deferred_path_index(covered) is index
        # ``frozenset(covered)`` would return ``covered`` itself; copy via a list.
        assert sc._deferred_path_index(frozenset(list(covered))) is not index
        # A set holding a string is scanned, never split into characters.
        assert sc._deferred_path_index(frozenset({"ab"})) is None
    assert type(sc._deferred_coverage({}, **_COVERAGE_ARGS)[0]) is frozenset


def test_entry_points_build_indexes_in_a_scope_and_drop_them():
    built: list[weakref.ref] = []
    scoped: list[bool] = []
    real_index = sc._source_branch_index

    class RecordedBranchIndex(sc._SourceBranchIndex):
        def __init__(self, branches):
            super().__init__(branches)
            built.append(weakref.ref(self))

    class RecordedDeferredIndex(sc._DeferredPathIndex):
        __slots__ = ("__weakref__",)

        def __init__(self, deferred_paths):
            super().__init__(deferred_paths)
            built.append(weakref.ref(self))

    def recorded_index(branches):
        scoped.append(sc._open_source_indexes() is not None)
        return real_index(branches)

    source = _cited_outline(8)
    with mock.patch.multiple(
        sc,
        _SourceBranchIndex=RecordedBranchIndex,
        _DeferredPathIndex=RecordedDeferredIndex,
        _source_branch_index=recorded_index,
    ):
        _analyze(
            _half_deferred(8, reason=_PRECISE_REASON), source, citation=SNAP_CITATION
        )
        analyzer_built = len(built)
        sc.manual_outcome_exception_advice(
            "(a) Households are eligible if residence is verified.\n"
            "(b) Households are not eligible unless income is verified.",
            corpus_citation_path="us-or/manual/odhs/open/page-644",
        )
        advice_built = len(built)
        format_coverage_index(source, SNAP_CITATION)
    gc.collect()

    assert analyzer_built > 1
    # The coverage formatter reaches the branch helpers from its own module.
    assert len(built) > advice_built
    assert scoped and all(scoped)
    # Nothing outlives the call that built it.
    assert all(ref() is None for ref in built)
    assert sc._SOURCE_INDEXES.get() is None


def test_fixture_coverage_index_matches_main():
    for case in FIXTURE_CASES:
        content, source, citation, _tests, _profile = case.values
        old, cold, warm = _old_and_new(
            functools.partial(
                format_coverage_index, source, citation, candidate=content
            )
        )

        assert cold == old, case.id
        assert warm == cold, case.id


def _in_thread(call: Callable[[], object], context: contextvars.Context) -> object:
    """``call``'s outcome, run in a new thread inside ``context``."""

    outcomes: list[object] = []
    thread = threading.Thread(
        target=context.run, args=(lambda: outcomes.append(_outcome(call)),)
    )
    thread.start()
    thread.join(60)
    assert not thread.is_alive()
    return outcomes[0]


def test_a_copied_context_neither_reads_nor_keeps_an_analysis_index():
    branches = sc.recognize_source_structure(_wide_outline(32))
    deferred = frozenset({branches[0].path})

    def analysis_here():
        with sc._source_index_scope():
            return sc._source_branch_index(branches), sc._SOURCE_INDEXES.get()

    with sc._source_index_scope():
        scope = sc._SOURCE_INDEXES.get()
        index = sc._source_branch_index(branches)
        assert index is not None
        assert sc._deferred_path_index(deferred) is not None
        context = contextvars.copy_context()

        # Another thread with this context scans, and an analysis there
        # opens a scope of its own.
        assert _in_thread(
            lambda: (
                sc._source_branch_index(branches),
                sc._deferred_path_index(deferred),
            ),
            context,
        ) == (None, None)
        own, own_scope = _in_thread(analysis_here, context)
        assert own is not None and own is not index
        assert own_scope is not scope
        index_ref = weakref.ref(index)
        del index, own

    gc.collect()
    # The kept context still holds the ended scope, closed and empty, so the
    # indexes are freed and nothing reads it again.
    assert index_ref() is None
    assert context.run(sc._SOURCE_INDEXES.get) is scope
    assert not scope.is_open and not scope.branches and not scope.deferrals
    assert context.run(sc._source_branch_index, branches) is None
    assert context.run(sc._deferred_path_index, deferred) is None
    later, later_scope = context.run(analysis_here)
    assert later is not None and later_scope is not scope
    assert _in_thread(analysis_here, context)[1] is not scope
    assert not scope.branches and not scope.deferrals


@pytest.mark.parametrize("raises", [False, True], ids=["returns", "raises"])
def test_an_analysis_frees_indexes_a_copied_context_still_refers_to(raises):
    kept: list[contextvars.Context] = []
    seen: list[int] = []

    def extract(text):
        if not kept:
            seen.append(len(sc._SOURCE_INDEXES.get().branches))
            kept.append(contextvars.copy_context())
            if raises:
                raise RuntimeError("callback failed")
        return extract_typed_numeric_inventory_occurrences_from_text(
            text, profile="en-US"
        )

    def analysis():
        return sc.analyze_complete_source_unit(
            _no_rules(SNAP_CITATION),
            _cited_outline(8),
            corpus_citation_path=SNAP_CITATION,
            test_cases=[],
            extract_numeric_occurrences=extract,
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        )

    outcome = _outcome(analysis)
    assert (outcome == ("RuntimeError", "callback failed")) is raises
    assert kept and seen[0] > 0
    scope = kept[0].run(sc._SOURCE_INDEXES.get)

    assert sc._SOURCE_INDEXES.get() is None
    assert not scope.is_open and not scope.branches and not scope.deferrals
    # Later analyses in that context, here or in another thread, keep their
    # own indexes and leave the ended scope empty.
    kept[0].run(_outcome, analysis)
    _in_thread(analysis, kept[0])
    assert not scope.branches and not scope.deferrals


# Field values recognition never gives a branch.
_OTHER_FIELDS = [
    pytest.param({"path": ""}, id="string-path"),
    pytest.param({"path": ["a"]}, id="list-path"),
    pytest.param({"path": ("a", _RAISING)}, id="raising-component"),
    pytest.param({"path": (_Text("a"),)}, id="string-subclass-component"),
    pytest.param({"path": (_Loose("z"),)}, id="loose-string-component"),
    pytest.param({"path": _Reversed(("1", "a"))}, id="tuple-subclass-path"),
    pytest.param({"path": _LooseTuple(("z",))}, id="loose-tuple-path"),
    pytest.param({"path": (_DISGUISED, "child")}, id="disguised-string-component"),
    pytest.param({"path": ("a", _UNHASHABLY_TYPED)}, id="unhashable-type-component"),
    pytest.param({"path": (1,)}, id="int-component"),
    pytest.param({"start": None}, id="none-start"),
    pytest.param({"end": None}, id="none-end"),
    pytest.param({"start": 0.0}, id="float-start"),
    pytest.param({"end": True}, id="bool-end"),
    pytest.param({"kind": None}, id="none-kind"),
]


@pytest.mark.parametrize("change", _OTHER_FIELDS)
@pytest.mark.parametrize("position", [0, 3], ids=["first", "inner"])
def test_branch_helpers_match_main_on_other_field_types(change, position):
    source = _wide_outline(16)
    branches = sc.recognize_source_structure(source)
    odd = tuple(
        dataclasses.replace(branch, **change) if index == position else branch
        for index, branch in enumerate(branches)
    )
    deferred_sets = [frozenset({branches[0].path, branches[5].path}), frozenset()]
    with contextlib.suppress(TypeError):
        deferred_sets.append(frozenset({odd[position].path, branches[5].path}))

    def helpers():
        return [
            (
                _outcome(
                    lambda: sc._is_marker_only_container(
                        branch, branches=odd, source_text=source
                    )
                ),
                _outcome(lambda: sc._source_branch_direct_text(branch, branches=odd)),
                _outcome(
                    lambda: _identity(
                        odd,
                        sc._most_specific_containing_branch(
                            branch.start, branch.end, branches=odd
                        ),
                    )
                ),
                *(
                    _outcome(
                        lambda: sc._span_is_deferred(
                            branch.start,
                            branch.end,
                            branches=odd,
                            deferred_paths=deferred,
                        )
                    )
                    for deferred in deferred_sets
                ),
                *(
                    _outcome(lambda: sc._path_is_deferred(branch.path, deferred))
                    for deferred in deferred_sets
                ),
            )
            for branch in odd
        ]

    old, cold, warm = _old_and_new(helpers)

    assert cold == old
    assert warm == cold


_OTHER_QUESTION_PATHS = [
    pytest.param("a", id="string"),
    pytest.param(["a"], id="list"),
    pytest.param((["a"],), id="list-component"),
    pytest.param((_RAISING,), id="raising-component"),
    pytest.param((_Text("a"),), id="string-subclass-component"),
    pytest.param((_Loose("z"),), id="loose-string-component"),
    pytest.param(_Reversed(("1", "a")), id="tuple-subclass-path"),
    pytest.param(_LooseTuple(("z",)), id="loose-tuple-path"),
    pytest.param((_DISGUISED,), id="disguised-string-component"),
    pytest.param((_UNHASHABLY_TYPED,), id="unhashable-type-component"),
    pytest.param((1,), id="int-component"),
]


@pytest.mark.parametrize("path", _OTHER_QUESTION_PATHS)
def test_branch_questions_match_main_from_outside_the_pool(path):
    # The pool's paths are plain, so its trie is built; the branch asked about
    # is not in it, and its path is of another type.
    source = _wide_outline(16)
    branches = sc.recognize_source_structure(source)
    asked = dataclasses.replace(branches[0], path=path)

    def children(candidate):
        return (
            len(candidate.path) == len(path) + 1 and candidate.path[: len(path)] == path
        )

    def descendants(candidate):
        return len(candidate.path) > len(path) and candidate.path[: len(path)] == path

    old, cold, warm = _old_and_new(
        lambda: _outcome(
            lambda: sc._is_marker_only_container(
                asked, branches=branches, source_text=source
            )
        )
    )
    assert cold == old
    assert warm == cold

    # Each index question still returns a pool its filter turns into main's.
    with sc._source_index_scope():
        index = sc._source_branch_index(branches)
        assert index is not None
        for question, keep in (
            (index.children, children),
            (index.descendants, descendants),
        ):
            expected = _outcome(lambda: [b for b in branches if keep(b)])
            assert _outcome(lambda: [b for b in question(path) if keep(b)]) == expected


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"kind": _RAISING}, id="raising-kind"),
        pytest.param({"kind": "sentence", "path": (1,)}, id="sentence-int-component"),
        pytest.param(
            {"kind": "sentence", "path": (["a"],)}, id="sentence-list-component"
        ),
    ],
)
def test_direct_text_tests_kind_and_path_only_where_main_did(change):
    # Main tests the branch's kind and last path component only for a
    # candidate that is not its descendant. With only descendants in the pool
    # it never does, so a kind or component it cannot test is never reached.
    source = _wide_outline(16)
    branches = sc.recognize_source_structure(source)
    parent = branches[0]
    asked = dataclasses.replace(parent, **change)
    pool = tuple(
        dataclasses.replace(branch, path=(*asked.path, *branch.path[1:]))
        for branch in branches[1:]
        if branch.path[: len(parent.path)] == parent.path
    )
    assert pool

    old, cold, warm = _old_and_new(
        lambda: _outcome(lambda: sc._source_branch_direct_text(asked, branches=pool))
    )

    assert not isinstance(old, tuple)
    assert cold == old
    assert warm == cold


_SENTENCE = SourceStructureBranch(
    ("a", "satz-1"), "sentence", "", "Some sentence.", 0, 14
)


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("paragraph", id="plain"),
        pytest.param([], id="list"),
        pytest.param(_HashRaises(), id="raising-hash"),
        pytest.param(_Loose("number"), id="loose-string"),
        pytest.param(_DISGUISED, id="disguised-string"),
    ],
)
def test_sentence_text_tests_every_kind_main_tested(kind):
    # For a sentence, main tests the kind of every branch that is not its
    # descendant, even one outside its paragraph, so the pool may leave such a
    # branch out only when every kind is a plain string.
    item = SourceStructureBranch(("a", "1"), "number", "", "sentence.", 5, 14)
    outside = SourceStructureBranch(("z",), kind, "", "", 0, 14)
    pool = (_SENTENCE, item, outside)

    old, cold, warm = _old_and_new(
        lambda: _outcome(
            lambda: sc._source_branch_direct_text(_SENTENCE, branches=pool)
        )
    )

    assert cold == old
    assert warm == cold
    if kind == "paragraph":
        assert old == "Some "


def test_branch_helpers_match_main_on_other_branch_objects():
    # A tuple holding anything but source branches is scanned.
    source = _wide_outline(16)
    branches = sc.recognize_source_structure(source)
    odd = tuple(
        types.SimpleNamespace(**dataclasses.asdict(branch)) if index == 3 else branch
        for index, branch in enumerate(branches)
    )
    deferred = frozenset({branches[0].path, branches[5].path})

    def helpers():
        return [
            (
                _outcome(
                    lambda: sc._is_marker_only_container(
                        branch, branches=odd, source_text=source
                    )
                ),
                _outcome(lambda: sc._source_branch_direct_text(branch, branches=odd)),
                _outcome(
                    lambda: sc._span_is_deferred(
                        branch.start, branch.end, branches=odd, deferred_paths=deferred
                    )
                ),
                _outcome(
                    lambda: _identity(
                        odd,
                        sc._most_specific_containing_branch(
                            branch.start, branch.end, branches=odd
                        ),
                    )
                ),
            )
            for branch in odd
        ]

    old, cold, warm = _old_and_new(helpers)

    assert cold == old
    assert warm == cold
    with sc._source_index_scope():
        assert sc._source_branch_index(odd) is None


@pytest.mark.parametrize(
    ("start", "end"),
    [(0.0, 3), (1, 2.5), (True, 3), (None, 3), (2, None), ("1", 3)],
)
def test_span_helpers_match_main_on_other_offset_types(start, end):
    branches = sc.recognize_source_structure(_wide_outline(16))
    deferred = frozenset({branches[0].path})

    old, cold, warm = _old_and_new(
        lambda: (
            _outcome(
                lambda: sc._span_is_deferred(
                    start, end, branches=branches, deferred_paths=deferred
                )
            ),
            _outcome(
                lambda: _identity(
                    branches,
                    sc._most_specific_containing_branch(start, end, branches=branches),
                )
            ),
        )
    )

    assert cold == old
    assert warm == cold


_MIXED_COMPONENTS = st.sampled_from(
    [
        "a",
        "b",
        "ab",
        _Text("a"),
        _Loose("z"),
        _DISGUISED,
        _UNHASHABLY_TYPED,
        1,
        1.0,
        True,
        None,
        b"a",
    ]
)
_MIXED_PATHS = st.lists(_MIXED_COMPONENTS, max_size=4).flatmap(
    lambda parts: st.sampled_from([tuple(parts), _LooseTuple(parts), _Reversed(parts)])
)


@settings(max_examples=400, deadline=None)
@given(
    paths=st.lists(_MIXED_PATHS, max_size=6),
    deferred=st.frozensets(_MIXED_PATHS, max_size=4),
)
def test_path_is_deferred_matches_main_on_mixed_components(paths, deferred):
    # Equal components of other types (1, 1.0, True), string and tuple
    # subclasses that compare their own way, match as main compared them;
    # plain-string paths take the index.
    expected = [_old_path_is_deferred(path, deferred) for path in paths]

    with sc._source_index_scope():
        assert [sc._path_is_deferred(path, deferred) for path in paths] == expected
        assert [sc._path_is_deferred(path, deferred) for path in paths] == expected


_COVERAGE_ARGS = {
    "corpus_citation_path": SNAP_CITATION,
    "source_text": "",
    "branches": (),
}


# CPU-time growth per doubling. A linear scan grows about 2x, a quadratic one
# about 4x; main grew 3.2-4x on these inputs, and 6-8x on qualified U.S.C.
# citations. The bound leaves room for timer noise and per-call overhead.
MAX_GROWTH_PER_DOUBLING = 2.6
# Each size is timed as the least of `_ROUNDS` rounds, interleaved so a busy
# host slows every size alike. A round repeats the cold call the same number
# of times at every size, enough for the smallest to spend `_MIN_ROUND_CPU`,
# so timer and scheduler noise stays a small share of every round.
_ROUNDS = 5
_MIN_ROUND_CPU = 0.05


def _cold_cpu_seconds(call: Callable[[], object], *, budget: float) -> float:
    """The CPU time of one ``call`` with every cache cold, in a fresh scope.

    A ``call`` with a ``warm`` attribute runs it in the scope first, off the
    clock, to build what the family does not time.
    """

    # Patterns built per reason are purged too: `re` keeps 512, which holds a
    # small size's patterns between calls but not a large one's.
    _clear_caches()
    with sc._source_index_scope():
        getattr(call, "warm", lambda: None)()
        start = time.process_time()
        call()
        elapsed = time.process_time() - start
    if elapsed > budget:
        pytest.fail(
            f"one call took {elapsed:.2f}s of CPU, over its {budget:.2f}s "
            "budget: the scan is no longer near-linear"
        )
    return elapsed


@contextlib.contextmanager
def _without_tracing() -> Iterator[None]:
    """Suspend this thread's trace function, so the timings are of the code.

    CI measures coverage of the package, not of this file, so under its line
    tracer the new code pays a cost per line that main's copies here do not.
    On CPython 3.13 the deep-deferral sweep read 4x main's that way (3.2x on
    CI), against 1.7x untraced. Only this thread's function is suspended, so
    coverage keeps measuring other threads; stopping and restarting coverage
    would leave theirs stopped. Coverage through ``sys.monitoring`` (3.14's
    default) sets no trace function, and disables each event once it fires.
    """

    tracer = sys.gettrace()
    if tracer is None:
        yield
        return
    sys.settrace(None)
    try:
        yield
    finally:
        sys.settrace(tracer)


def _trace_nothing(frame, event, arg):
    return None


def test_timings_suspend_only_this_threads_tracer():
    """Inside, this thread has no tracer and another keeps its own; after, both do."""

    own = sys.gettrace()
    if own is None:
        sys.settrace(_trace_nothing)
    mine = sys.gettrace()
    ready, paused = threading.Event(), threading.Event()
    seen: dict[str, object] = {}

    def worker():
        if sys.gettrace() is None:
            sys.settrace(_trace_nothing)
        seen["before"] = sys.gettrace()
        ready.set()
        assert paused.wait(30)
        seen["during"] = sys.gettrace()

    thread = threading.Thread(target=worker)
    try:
        thread.start()
        assert ready.wait(30)
        with _without_tracing():
            assert sys.gettrace() is None
            with _without_tracing():
                assert sys.gettrace() is None
            assert sys.gettrace() is None
            paused.set()
            thread.join(30)
        assert sys.gettrace() is mine
        with contextlib.suppress(RuntimeError), _without_tracing():
            raise RuntimeError
        assert sys.gettrace() is mine
    finally:
        paused.set()
        thread.join(30)
        if own is None:
            sys.settrace(None)
    assert not thread.is_alive()
    assert seen["before"] is not None
    assert seen["during"] is seen["before"]


_COVERAGE_PROBE = """
import contextlib, json, sys, threading
from collections.abc import Iterator
import coverage
import target

{helper}

started, release = threading.Event(), threading.Event()


def worker():
    target.before()
    started.set()
    assert release.wait(30)
    target.after()


cov = coverage.Coverage(data_file=None, config_file=False, include=[target.__file__])
cov.start()
thread = threading.Thread(target=worker)
thread.start()
assert started.wait(30)
with _without_tracing():
    target.paused()
target.resumed()
release.set()
thread.join(30)
cov.stop()
print(json.dumps(sorted(cov.get_data().lines(target.__file__))))
"""


@pytest.mark.parametrize("core", ["ctrace", "pytrace"])
def test_coverage_still_measures_other_threads_after_a_timing(core, tmp_path):
    """Real coverage keeps every thread but the paused call, under each tracer."""

    pytest.importorskip("coverage")
    functions = ("before", "after", "paused", "resumed")
    (tmp_path / "target.py").write_text(
        "".join(f"def {name}():\n    return {name!r}\n" for name in functions)
    )
    (tmp_path / "probe.py").write_text(
        _COVERAGE_PROBE.format(
            helper=textwrap.dedent(inspect.getsource(_without_tracing))
        )
    )
    # A child of a measured run would start its own measurement; this one
    # starts only the probe's, with the tracer under test.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COVERAGE_", "COV_CORE_"))
    }
    env["COVERAGE_CORE"] = core
    result = subprocess.run(
        [sys.executable, "probe.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    # Each function's body is the line after its def: lines 2, 4, 6 and 8.
    body = {name: 2 * position + 2 for position, name in enumerate(functions)}
    assert json.loads(result.stdout) == [
        body["before"],
        body["after"],
        body["resumed"],
    ]


def _cpu_times(calls: Sequence[Callable[[], object]], *, budget: float) -> list[float]:
    """The least CPU time of each call's round, over interleaved rounds."""

    with _without_tracing():
        smallest = min(_cold_cpu_seconds(calls[0], budget=budget) for _ in range(3))
        repeats = max(1, math.ceil(_MIN_ROUND_CPU / max(smallest, 1e-6)))
        best = [math.inf] * len(calls)
        for _round in range(_ROUNDS):
            for position, call in enumerate(calls):
                spent = sum(
                    _cold_cpu_seconds(call, budget=budget) for _repeat in range(repeats)
                )
                best[position] = min(best[position], spent)
    return best


def _assert_near_linear(
    family: Callable[[int], Callable[[], object]],
    sizes: Sequence[int],
    *,
    budget: float,
) -> None:
    times = _cpu_times([family(size) for size in sizes], budget=budget)
    growth = (times[-1] / times[0]) ** (1 / math.log2(sizes[-1] / sizes[0]))
    measured = ", ".join(f"n={size}: {cpu:.4f}s" for size, cpu in zip(sizes, times))

    assert growth <= MAX_GROWTH_PER_DOUBLING, (
        f"CPU time grew {growth:.2f}x per doubling ({measured})"
    )


def _reason_family(clause: str):
    """``_reason_dependency_is_source_bound`` over n repeats of ``clause``."""

    def family(n: int):
        return functools.partial(
            sc._reason_dependency_is_source_bound,
            clause * n,
            "(a) Under section 345.",
            corpus_citation_path=SNAP_CITATION,
            path=("a",),
        )

    return family


def _clause_bounds_family(n: int):
    """The clause bounds of every dependency in a reason of n clauses."""

    reason = "missing section 346. " * n
    matches = tuple(sc._PRECISE_DEFERRAL_DEPENDENCY.finditer(reason))
    return lambda: [sc._reason_clause_bounds(reason, match) for match in matches]


def _analyzer_family(outline: Callable[[int], str], content: Callable[[int], str]):
    """The whole analyzer over an n-line ``outline``."""

    def family(n: int):
        return functools.partial(
            _analyze, content(n), outline(n), citation=SNAP_CITATION
        )

    return family


def _coverage_index_family(n: int):
    """The generation coverage index over an n-line ``_formula_outline``.

    It runs in an empty context, as a caller outside any analysis runs it, so
    the formatter must open its own scope.
    """

    source = _formula_outline(n)
    return lambda: contextvars.Context().run(
        format_coverage_index, source, SNAP_CITATION
    )


def _branch_helper_family(
    helper: Callable[[str, tuple[SourceStructureBranch, ...], set], object],
):
    """One per-branch helper over every branch of an n-line wide outline."""

    def family(n: int):
        source = _wide_outline(n)
        branches = sc.recognize_source_structure(source)
        # Frozen, as `_deferred_coverage` returns it to an analysis.
        deferred = frozenset(branch.path for branch in branches[::2])
        return functools.partial(helper, source, branches, deferred)

    return family


def _deep_path_family(deferred: frozenset | set):
    """``_path_is_deferred`` over an outline of n attached markers, n deep."""

    def family(n: int):
        branches = _attached_markers(n)
        return lambda: [
            sc._path_is_deferred(branch.path, deferred) for branch in branches
        ]

    return family


def _deep_span_family(n: int):
    """``_span_is_deferred`` over n attached markers, the outermost deferred."""

    branches = _attached_markers(n)
    # Branches come sorted by start, so the first holds every other one.
    deferred = frozenset({branches[0].path})
    return lambda: [
        sc._span_is_deferred(
            branch.start, branch.end, branches=branches, deferred_paths=deferred
        )
        for branch in branches
    ]


def _deep_direct_text_family(n: int):
    """``_source_branch_direct_text`` over n attached markers, n deep.

    These paths hold n(n+1)/2 components in all, and building the branch index
    reads each one, so the index is built off the clock: each branch's lookup
    is timed, and it must not read its descendants' paths.
    """

    branches = _attached_markers(n)

    def sweep():
        return [
            sc._source_branch_direct_text(branch, branches=branches)
            for branch in branches
        ]

    sweep.warm = lambda: sc._source_branch_index(branches)._knows_lowest_starts
    return sweep


def _attached_markers(n: int) -> tuple[SourceStructureBranch, ...]:
    """The branches of n attached markers, each inside the last: n deep."""

    return sc.recognize_source_structure("A.\n" + "(a)" * n + " x\nB. x")


def test_deep_deferral_costs_at_most_a_constant_times_main():
    # Main compares each path to the deferral in C, cubic in the depth here, so
    # a growth bound cannot separate the two; time both, interleaved, instead.
    branches = _attached_markers(400)
    deferred = frozenset({max(branches, key=lambda branch: len(branch.path)).path})

    def sweep():
        return [
            sc._span_is_deferred(
                branch.start, branch.end, branches=branches, deferred_paths=deferred
            )
            for branch in branches
        ]

    def main_sweep():
        with _old_mode():
            return sweep()

    with sc._source_index_scope():
        assert sweep() == main_sweep()
    new, main = _cpu_times([sweep, main_sweep], budget=10.0)

    assert new <= 3 * main, f"new {new:.4f}s, main {main:.4f}s of CPU"


def test_deep_direct_text_costs_a_fraction_of_main():
    # Main slices every descendant's path for every branch, about n³/6
    # components here; the new sweep, index build included, reads each path
    # once. Time both, interleaved, main first so its rounds set the repeats.
    branches = _attached_markers(400)

    def sweep():
        return [
            sc._source_branch_direct_text(branch, branches=branches)
            for branch in branches
        ]

    def main_sweep():
        with _old_mode():
            return sweep()

    with sc._source_index_scope():
        assert sweep() == main_sweep()
    main, new = _cpu_times([main_sweep, sweep], budget=10.0)

    assert new * 4 <= main, f"new {new:.4f}s, main {main:.4f}s of CPU"


_PRECISE_REASON = (
    "Paragraph ({k}) cannot be encoded until eligibility under section "
    "{section} is available."
)


@pytest.mark.parametrize(
    ("family", "sizes", "budget"),
    (
        pytest.param(
            _reason_family("missing section 346. "), (500, 1000, 2000), 1.0, id="prose"
        ),
        pytest.param(
            _reason_family("7 U.S.C. 2015 is missing. "), (128, 256, 512), 1.0, id="usc"
        ),
        pytest.param(
            _reason_family("missing 42 U.S.C. 1382a(a)(2). "),
            (128, 256, 512),
            1.0,
            id="usc-other-title",
        ),
        pytest.param(_clause_bounds_family, (500, 1000, 2000), 1.0, id="clause-bounds"),
    ),
)
def test_deferral_reason_cpu_growth_is_near_linear(family, sizes, budget):
    _assert_near_linear(family, sizes, budget=budget)


@pytest.mark.parametrize(
    ("family", "sizes", "budget"),
    (
        pytest.param(
            _analyzer_family(_wide_outline, lambda n: _no_rules(SNAP_CITATION)),
            (400, 800, 1600),
            8.0,
            id="analyzer-wide-outline",
        ),
        pytest.param(
            _analyzer_family(
                _cited_outline, lambda n: _half_deferred(n, reason=_PRECISE_REASON)
            ),
            (200, 400, 800),
            12.0,
            id="analyzer-half-deferred",
        ),
        # The coverage formatter calls the branch helpers from its own module.
        pytest.param(
            _coverage_index_family, (400, 800, 1600), 4.0, id="coverage-index"
        ),
        pytest.param(
            _branch_helper_family(
                lambda source, branches, deferred: [
                    sc._span_is_deferred(
                        branch.start,
                        branch.end,
                        branches=branches,
                        deferred_paths=deferred,
                    )
                    for branch in branches
                ]
            ),
            (400, 800, 1600),
            1.0,
            id="span-is-deferred",
        ),
        # Main stops at the deferred outermost branch, the first it tries for
        # every span here; the search must stop there too, not collect every
        # container first.
        pytest.param(
            _deep_span_family, (400, 800, 1600), 1.0, id="span-is-deferred-deep-outer"
        ),
        pytest.param(
            _branch_helper_family(
                lambda source, branches, deferred: [
                    sc._path_is_deferred(branch.path, deferred) for branch in branches
                ]
            ),
            (400, 800, 1600),
            1.0,
            id="path-is-deferred",
        ),
        # Branch paths grow with n here, so a lookup must not copy each prefix.
        pytest.param(
            _deep_path_family(frozenset()),
            (200, 400, 800),
            1.0,
            id="path-is-deferred-deep-none",
        ),
        pytest.param(
            _deep_path_family(frozenset({("z",)})),
            (200, 400, 800),
            1.0,
            id="path-is-deferred-deep-unrelated",
        ),
        pytest.param(
            _deep_path_family({("z",)}),
            (200, 400, 800),
            1.0,
            id="path-is-deferred-deep-unrelated-unfrozen",
        ),
        pytest.param(
            _branch_helper_family(
                lambda source, branches, deferred: [
                    sc._most_specific_containing_branch(
                        branch.start, branch.end, branches=branches
                    )
                    for branch in branches
                ]
            ),
            (400, 800, 1600),
            1.0,
            id="most-specific-containing-branch",
        ),
        pytest.param(
            _branch_helper_family(
                lambda source, branches, deferred: [
                    sc._is_marker_only_container(
                        branch, branches=branches, source_text=source
                    )
                    for branch in branches
                ]
            ),
            (400, 800, 1600),
            1.0,
            id="marker-only-container",
        ),
        pytest.param(
            _branch_helper_family(
                lambda source, branches, deferred: [
                    sc._source_branch_direct_text(branch, branches=branches)
                    for branch in branches
                ]
            ),
            (400, 800, 1600),
            1.0,
            id="source-branch-direct-text",
        ),
        # Each branch here holds every deeper one, so its first descendant must
        # come from the index, not from a slice of every descendant's path:
        # main's slices cost about n³/6 and grew 4-6x per doubling.
        pytest.param(
            _deep_direct_text_family,
            (400, 800, 1600),
            1.0,
            id="source-branch-direct-text-deep",
        ),
        pytest.param(
            _branch_helper_family(
                lambda source, branches, deferred: list(
                    sc._source_clause_spans(source, branches=branches)
                )
            ),
            (400, 800, 1600),
            1.0,
            id="source-clause-spans",
        ),
    ),
)
def test_source_branch_cpu_growth_is_near_linear(family, sizes, budget):
    _assert_near_linear(family, sizes, budget=budget)
