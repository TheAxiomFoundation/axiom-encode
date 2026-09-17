"""Fold verifier results into a judge leaderboard.

Comparability contract (refused, not reinterpreted): every folded payload
must carry the same suite sha256 (which already binds the suite name, source
kind, corpus release, mutator version, provision window and the ordered case
identities), and no runner name may appear twice. ``allow_partial`` folds
payloads whose coverage is incomplete; rates then cover only what each
runner scored, and the board says so.

Headline: per-kind detection AUC on each judge's kind channel, subject to a
false-alarm ceiling on the judge's native verdict. A judge whose native flag
rate on clean controls exceeds the ceiling is shown but not ranked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import BOARD_SCHEMA, DEFECT_KINDS
from .judges.base import CHANNEL_NATIVE, VERDICT_FLAG
from .metrics import (
    auc,
    detection_at_false_alarm_ceiling,
    mean_or_none,
    mean_paired_delta,
    median_or_none,
    paired_rise_rate,
    rate,
)
from .results import load_results

DEFAULT_FALSE_ALARM_CEILING = 0.25


class VerifierBoardError(ValueError):
    """A board input is unreadable, malformed, incomplete, or not comparable."""


@dataclass
class KindStats:
    kind: str
    pairs: int = 0
    defective_scored: int = 0
    control_scored: int = 0
    complete_pairs: int = 0
    channel: str = CHANNEL_NATIVE
    kind_auc: Optional[float] = None
    verdict_auc: Optional[float] = None
    kind_rise_rate: Optional[float] = None
    kind_mean_delta: Optional[float] = None
    detection_at_ceiling: Optional[float] = None
    native_detection_rate: Optional[float] = None
    native_false_alarm_rate: Optional[float] = None
    localization_rate: Optional[float] = None
    localized: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class RunnerStats:
    runner: str
    family: str
    model: str
    source: str
    cases_expected: int
    cases_scored: int
    errors: int
    native_false_alarm_rate: Optional[float]
    native_detection_rate: Optional[float]
    mean_kind_auc: Optional[float]
    mean_verdict_auc: Optional[float]
    localization_rate: Optional[float]
    median_latency_seconds: Optional[float]
    mean_tokens_input: Optional[float]
    mean_tokens_output: Optional[float]
    mean_cost_usd: Optional[float]
    total_cost_usd: Optional[float]
    pricing_source: Optional[str]
    over_ceiling: bool
    complete: bool
    kinds: dict[str, KindStats] = field(default_factory=dict)
    verdict_fallback_kinds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {k: v for k, v in self.__dict__.items() if k != "kinds"}
        payload["kinds"] = {kind: stats.to_dict() for kind, stats in self.kinds.items()}
        return payload


@dataclass
class VerifierBoard:
    suite: dict[str, Any]
    false_alarm_ceiling: float
    runners: list[RunnerStats]
    sources: dict[str, str]
    incomplete_sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def ordered_runners(self) -> list[RunnerStats]:
        """Under-ceiling judges first by mean kind AUC, then the rest."""

        def key(stats: RunnerStats):
            auc_value = stats.mean_kind_auc if stats.mean_kind_auc is not None else -1
            return (
                stats.over_ceiling,
                -auc_value,
                stats.median_latency_seconds
                if stats.median_latency_seconds is not None
                else float("inf"),
                stats.runner,
            )

        return sorted(self.runners, key=key)


def _scored(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not row.get("error")]


def _kind_stats(kind: str, rows: list[dict[str, Any]]) -> KindStats:
    stats = KindStats(kind=kind)
    kind_rows = [row for row in rows if row.get("defect_kind") == kind]
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in kind_rows:
        pairs.setdefault(str(row["pair_id"]), {})[str(row["variant"])] = row
    stats.pairs = len(pairs)
    stats.errors = sum(1 for row in kind_rows if row.get("error"))
    scored = _scored(kind_rows)
    defective = [row for row in scored if row.get("variant") == "defective"]
    control = [row for row in scored if row.get("variant") == "control"]
    stats.defective_scored = len(defective)
    stats.control_scored = len(control)
    channels = {
        str((row.get("kind_score_channels") or {}).get(kind, CHANNEL_NATIVE))
        for row in scored
    }
    stats.channel = (
        CHANNEL_NATIVE
        if channels == {CHANNEL_NATIVE} or not channels
        else "verdict_fallback"
    )

    def kind_score(row: dict[str, Any]) -> Optional[float]:
        value = (row.get("kind_scores") or {}).get(kind)
        return float(value) if isinstance(value, (int, float)) else None

    def verdict_score(row: dict[str, Any]) -> Optional[float]:
        value = row.get("verdict_score")
        return float(value) if isinstance(value, (int, float)) else None

    pos_kind = [s for s in (kind_score(r) for r in defective) if s is not None]
    neg_kind = [s for s in (kind_score(r) for r in control) if s is not None]
    pos_verdict = [s for s in (verdict_score(r) for r in defective) if s is not None]
    neg_verdict = [s for s in (verdict_score(r) for r in control) if s is not None]
    stats.kind_auc = auc(pos_kind, neg_kind)
    stats.verdict_auc = auc(pos_verdict, neg_verdict)
    paired: list[tuple[float, float]] = []
    for members in pairs.values():
        d = members.get("defective")
        c = members.get("control")
        if d is None or c is None or d.get("error") or c.get("error"):
            continue
        ds, cs = kind_score(d), kind_score(c)
        if ds is None or cs is None:
            continue
        paired.append((ds, cs))
    stats.complete_pairs = len(paired)
    stats.kind_rise_rate = paired_rise_rate(paired)
    stats.kind_mean_delta = mean_paired_delta(paired)
    stats.native_detection_rate = rate(
        sum(1 for row in defective if row.get("verdict") == VERDICT_FLAG),
        len(defective),
    )
    stats.native_false_alarm_rate = rate(
        sum(1 for row in control if row.get("verdict") == VERDICT_FLAG), len(control)
    )
    localizable = [row for row in defective if row.get("localized") is not None]
    stats.localized = sum(1 for row in localizable if row.get("localized") is True)
    stats.localization_rate = rate(stats.localized, len(localizable))
    return stats


def fold_verifier_board(
    inputs: list[Path],
    *,
    false_alarm_ceiling: float = DEFAULT_FALSE_ALARM_CEILING,
    allow_partial: bool = False,
) -> VerifierBoard:
    if not inputs:
        raise VerifierBoardError("the verifier board needs at least one results input")
    if not 0.0 <= false_alarm_ceiling <= 1.0:
        raise VerifierBoardError("false-alarm ceiling must be within [0, 1]")

    reference_suite: Optional[dict[str, Any]] = None
    reference_identities: Optional[list[dict[str, Any]]] = None
    reference_source = ""
    runner_sources: dict[str, str] = {}
    runners: list[RunnerStats] = []
    sources: dict[str, str] = {}
    incomplete: list[str] = []
    notes: list[str] = []

    for raw in inputs:
        path = Path(raw)
        source = str(path if path.is_file() else path / "results.json")
        try:
            payload = load_results(path)
        except Exception as exc:  # noqa: BLE001 - re-raised as a board error
            raise VerifierBoardError(str(exc)) from exc
        suite = payload["suite"]
        identities = payload["case_identities"]
        if reference_suite is None:
            reference_suite = suite
            reference_identities = identities
            reference_source = source
        else:
            if suite.get("sha256") != reference_suite.get("sha256"):
                raise VerifierBoardError(
                    "Results are not comparable: suite identity in "
                    f"{source} ({suite.get('name')!r}, mutator "
                    f"{suite.get('mutator_version')!r}, sha256 "
                    f"{str(suite.get('sha256'))[:12]}) does not match "
                    f"{reference_source} ({reference_suite.get('name')!r}, mutator "
                    f"{reference_suite.get('mutator_version')!r}, sha256 "
                    f"{str(reference_suite.get('sha256'))[:12]}); boards fold only "
                    "runs of the identical suite"
                )
            if identities != reference_identities:
                raise VerifierBoardError(
                    f"Results are not comparable: case identities in {source} do "
                    f"not match {reference_source}"
                )
        runner = payload["runner"]
        name = str(runner["name"])
        if name in runner_sources:
            raise VerifierBoardError(
                f"Runner {name!r} appears in both {runner_sources[name]} and "
                f"{source}; two runs of one judge are two boards, not one"
            )
        runner_sources[name] = source
        sources[source] = str(suite.get("name"))
        coverage = payload["coverage"]
        complete = bool(coverage.get("complete"))
        if not complete:
            if not allow_partial:
                raise VerifierBoardError(
                    f"Results are incomplete ({coverage.get('scored')} scored, "
                    f"{coverage.get('errors')} errors of {coverage.get('expected')}): "
                    f"{source}. Finish the run (--resume) or pass --allow-partial."
                )
            incomplete.append(source)
        rows = payload["results"]
        scored = _scored(rows)
        controls = [row for row in scored if row.get("variant") == "control"]
        defectives = [row for row in scored if row.get("variant") == "defective"]
        kinds = {kind: _kind_stats(kind, rows) for kind in DEFECT_KINDS}
        kind_aucs = [s.kind_auc for s in kinds.values() if s.kind_auc is not None]
        verdict_aucs = [
            s.verdict_auc for s in kinds.values() if s.verdict_auc is not None
        ]
        for kind, stats in kinds.items():
            pos = [
                float((r.get("kind_scores") or {}).get(kind))
                for r in defectives
                if r.get("defect_kind") == kind
                and isinstance((r.get("kind_scores") or {}).get(kind), (int, float))
            ]
            neg = [
                float((r.get("kind_scores") or {}).get(kind))
                for r in controls
                if r.get("defect_kind") == kind
                and isinstance((r.get("kind_scores") or {}).get(kind), (int, float))
            ]
            stats.detection_at_ceiling = detection_at_false_alarm_ceiling(
                pos, neg, false_alarm_ceiling
            )
        native_far = rate(
            sum(1 for r in controls if r.get("verdict") == VERDICT_FLAG), len(controls)
        )
        localizable = [r for r in defectives if r.get("localized") is not None]
        latencies = [
            float(r["latency_ms"]) / 1000.0
            for r in scored
            if isinstance(r.get("latency_ms"), (int, float))
        ]
        tokens_in = [float((r.get("tokens") or {}).get("input") or 0) for r in scored]
        tokens_out = [float((r.get("tokens") or {}).get("output") or 0) for r in scored]
        costs = [
            float(r["cost_usd"])
            for r in scored
            if isinstance(r.get("cost_usd"), (int, float))
        ]
        pricing = payload.get("pricing") or None
        stats = RunnerStats(
            runner=name,
            family=str(runner.get("family")),
            model=str(runner.get("model")),
            source=source,
            cases_expected=int(coverage.get("expected") or len(identities)),
            cases_scored=len(scored),
            errors=len(rows) - len(scored),
            native_false_alarm_rate=native_far,
            native_detection_rate=rate(
                sum(1 for r in defectives if r.get("verdict") == VERDICT_FLAG),
                len(defectives),
            ),
            mean_kind_auc=mean_or_none(kind_aucs),
            mean_verdict_auc=mean_or_none(verdict_aucs),
            localization_rate=rate(
                sum(1 for r in localizable if r.get("localized") is True),
                len(localizable),
            ),
            median_latency_seconds=median_or_none(latencies),
            mean_tokens_input=mean_or_none(tokens_in),
            mean_tokens_output=mean_or_none(tokens_out),
            mean_cost_usd=mean_or_none(costs) if costs else None,
            total_cost_usd=round(sum(costs), 6) if costs else None,
            pricing_source=pricing.get("source") if pricing else None,
            over_ceiling=native_far is not None and native_far > false_alarm_ceiling,
            complete=complete,
            kinds=kinds,
            verdict_fallback_kinds=[
                kind for kind, s in kinds.items() if s.channel != CHANNEL_NATIVE
            ],
        )
        runners.append(stats)

    assert reference_suite is not None
    if incomplete:
        notes.append(
            "Partial fold: incomplete runs were included with --allow-partial; "
            "rates cover only the cases each judge scored."
        )
    return VerifierBoard(
        suite=dict(reference_suite),
        false_alarm_ceiling=false_alarm_ceiling,
        runners=runners,
        sources=sources,
        incomplete_sources=incomplete,
        notes=notes,
    )


# -- rendering -----------------------------------------------------------------


def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.0%}"


def _num(value: Optional[float], template: str = "{:.3f}") -> str:
    return "—" if value is None else template.format(value)


def board_to_json(board: VerifierBoard) -> dict[str, Any]:
    return {
        "schema": BOARD_SCHEMA,
        "suite": board.suite,
        "false_alarm_ceiling": board.false_alarm_ceiling,
        "sources": board.sources,
        "incomplete_sources": board.incomplete_sources,
        "notes": board.notes,
        "defect_kinds": list(DEFECT_KINDS),
        "runners": [stats.to_dict() for stats in board.ordered_runners()],
    }


def render_board_markdown(board: VerifierBoard) -> str:
    ordered = board.ordered_runners()
    suite = board.suite
    lines: list[str] = []
    lines.append(f"# EncodeBench verifier board — {suite.get('name')}")
    lines.append("")
    lines.append(
        f"Suite `{str(suite.get('sha256'))[:12]}`, source `{suite.get('source_kind')}`"
        + (
            f", corpus release `{suite.get('corpus_release')}`"
            if suite.get("corpus_release")
            else ""
        )
        + (
            f", mutator `{suite.get('mutator_version')}`"
            if suite.get("mutator_version")
            else ""
        )
        + f", provision window {suite.get('provision_chars')} chars, "
        f"{suite.get('pair_count')} pairs ({suite.get('case_count')} cases)."
    )
    lines.append("")
    for note in board.notes:
        lines.append(f"> {note}")
        lines.append("")
    lines.append(
        f"Headline: per-kind detection AUC on each judge's kind channel, subject to a "
        f"false-alarm ceiling of {board.false_alarm_ceiling:.0%} on the judge's native "
        "verdict (flag rate on clean controls). Judges over the ceiling are shown but "
        "not ranked (†). AUC is pooled Mann-Whitney within a kind, ties count 0.5. "
        "`det@ceil` is the share of defective cases scoring above the control score "
        "that admits at most the ceiling's share of false alarms. Localization counts "
        "a finding that names the mutated rule or the edited token; probability-only "
        "judges score blank there by construction. Kinds marked ‡ have no "
        "kind-specific question for that judge and fall back to the verdict score."
    )
    lines.append("")
    short = {
        "amount_changed": "amount",
        "boundary_flipped": "boundary",
        "conjunct_dropped": "conjunct",
        "polarity_swapped": "polarity",
        "date_or_period_wrong": "date/period",
        "entity_wrong": "entity",
    }
    header = (
        "| judge | model | native FAR | native det | "
        + " | ".join(f"AUC {short[k]}" for k in DEFECT_KINDS)
        + " | mean AUC | localize | median s | tokens in/out | cost/case |"
    )
    lines.append(header)
    lines.append("|" + "---|" * (9 + len(DEFECT_KINDS)))
    for stats in ordered:
        mark = "†" if stats.over_ceiling else ""
        cells = []
        for kind in DEFECT_KINDS:
            ks = stats.kinds[kind]
            fallback = "‡" if ks.channel != CHANNEL_NATIVE else ""
            cells.append(_num(ks.kind_auc) + fallback)
        tokens = (
            f"{stats.mean_tokens_input:,.0f}/{stats.mean_tokens_output:,.0f}"
            if stats.mean_tokens_input is not None
            else "—"
        )
        lines.append(
            f"| {stats.runner}{mark} | {stats.model} | "
            f"{_pct(stats.native_false_alarm_rate)} | {_pct(stats.native_detection_rate)} | "
            + " | ".join(cells)
            + f" | {_num(stats.mean_kind_auc)} | {_pct(stats.localization_rate)} | "
            f"{_num(stats.median_latency_seconds, '{:.2f}')} | {tokens} | "
            f"{_num(stats.mean_cost_usd, '${:.5f}')} |"
        )
    lines.append("")
    lines.append("## Per kind")
    lines.append("")
    lines.append(
        "| judge | kind | pairs | kind AUC | verdict AUC | paired rise | mean Δ | "
        "det@ceil | native det | native FAR | localize | errors |"
    )
    lines.append("|" + "---|" * 12)
    for stats in ordered:
        for kind in DEFECT_KINDS:
            ks = stats.kinds[kind]
            fallback = "‡" if ks.channel != CHANNEL_NATIVE else ""
            lines.append(
                f"| {stats.runner} | {short[kind]}{fallback} | {ks.complete_pairs}/{ks.pairs} | "
                f"{_num(ks.kind_auc)} | {_num(ks.verdict_auc)} | "
                f"{_pct(ks.kind_rise_rate)} | {_num(ks.kind_mean_delta, '{:+.3f}')} | "
                f"{_pct(ks.detection_at_ceiling)} | {_pct(ks.native_detection_rate)} | "
                f"{_pct(ks.native_false_alarm_rate)} | {_pct(ks.localization_rate)} | "
                f"{ks.errors} |"
            )
    lines.append("")
    lines.append("## Spend and coverage")
    lines.append("")
    lines.append("| judge | scored | errors | total cost | price source |")
    lines.append("|---|---|---|---|---|")
    for stats in ordered:
        lines.append(
            f"| {stats.runner} | {stats.cases_scored}/{stats.cases_expected} | "
            f"{stats.errors} | {_num(stats.total_cost_usd, '${:.4f}')} | "
            f"{stats.pricing_source or 'no published price recorded; cost blank'} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_board_text(board: VerifierBoard) -> str:
    ordered = board.ordered_runners()
    lines = [
        f"Suite: {board.suite.get('name')}  ({board.suite.get('pair_count')} pairs, "
        f"mutator {board.suite.get('mutator_version')})",
        f"False-alarm ceiling on native verdict: {board.false_alarm_ceiling:.0%}",
        "",
    ]
    width = max((len(s.runner) for s in ordered), default=6)
    for stats in ordered:
        mark = " (over ceiling, unranked)" if stats.over_ceiling else ""
        lines.append(
            f"{stats.runner:<{width}}  mean kind AUC {_num(stats.mean_kind_auc)}  "
            f"native FAR {_pct(stats.native_false_alarm_rate)}  "
            f"native det {_pct(stats.native_detection_rate)}  "
            f"localize {_pct(stats.localization_rate)}  "
            f"median {_num(stats.median_latency_seconds, '{:.2f}s')}  "
            f"cost/case {_num(stats.mean_cost_usd, '${:.5f}')}{mark}"
        )
        for kind in DEFECT_KINDS:
            ks = stats.kinds[kind]
            lines.append(
                f"  {kind:<22} pairs {ks.complete_pairs:>3}  kind AUC {_num(ks.kind_auc)}"
                f"  verdict AUC {_num(ks.verdict_auc)}  rise {_pct(ks.kind_rise_rate)}"
                f"  det@ceil {_pct(ks.detection_at_ceiling)}"
                + ("  [verdict fallback]" if ks.channel != CHANNEL_NATIVE else "")
            )
    return "\n".join(lines)
