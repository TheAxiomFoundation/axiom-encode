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
import functools
import json
import math
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from unittest import mock

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
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
    """Main's outcome, then the new one cold, then the new one warm."""

    _clear_caches()
    with _old_mode():
        old = _outcome(call)
    _clear_caches()
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
    yield set()
    yield {()}
    yield set(paths[::2])
    yield {path[:1] for path in paths if path}
    yield {path for path in paths if len(path) > 1}


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


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.sampled_from(_REASON_FRAGMENTS), max_size=24).map(" ".join))
def test_reason_helpers_match_main(reason):
    old, cold, warm = _old_and_new(lambda: _reason_helpers(reason))

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
    text, branches, deferred, spans = case
    spans = spans + [(branch.start, branch.end) for branch in branches]

    def helpers(pool):
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
        old, cold, warm = _old_and_new(lambda: helpers(pool))
        assert cold == old
        assert warm == cold


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


def test_branch_index_is_cached_per_tuple_and_skips_other_sequences():
    branches = sc.recognize_source_structure(_wide_outline(32))
    equal_copy = tuple(list(branches))

    assert sc._source_branch_index(branches) is sc._source_branch_index(branches)
    assert sc._source_branch_index(equal_copy) is not sc._source_branch_index(branches)
    assert sc._source_branch_index(list(branches)) is None
    assert sc._source_branch_index(()) is None


# CPU-time growth per doubling. A linear scan grows about 2x, a quadratic one
# about 4x; main grew 3.2-4x on these inputs, and 6-8x on qualified U.S.C.
# citations. The bound leaves room for timer noise and per-call overhead.
MAX_GROWTH_PER_DOUBLING = 2.6
_CPU_FLOOR = 0.05


def _cpu_seconds(call: Callable[[], object], *, budget: float) -> float:
    """The least CPU time of one cold ``call`` over enough runs to be stable."""

    best = math.inf
    spent = 0.0
    runs = 0
    while runs < 3 or (spent < _CPU_FLOOR and runs < 200):
        _clear_caches()
        start = time.process_time()
        call()
        elapsed = time.process_time() - start
        if elapsed > budget:
            pytest.fail(
                f"one call took {elapsed:.2f}s of CPU, over its {budget:.2f}s "
                "budget: the scan is no longer near-linear"
            )
        best = min(best, elapsed)
        spent += elapsed
        runs += 1
    return best


def _assert_near_linear(
    family: Callable[[int], Callable[[], object]],
    sizes: Sequence[int],
    *,
    budget: float,
) -> None:
    times = [_cpu_seconds(family(size), budget=budget) for size in sizes]
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


def _branch_helper_family(
    helper: Callable[[str, tuple[SourceStructureBranch, ...], set], object],
):
    """One per-branch helper over every branch of an n-line wide outline."""

    def family(n: int):
        source = _wide_outline(n)
        branches = sc.recognize_source_structure(source)
        deferred = {branch.path for branch in branches[::2]}
        return functools.partial(helper, source, branches, deferred)

    return family


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
