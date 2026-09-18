"""Fold verifier results into a judge leaderboard.

Comparability contract (refused, not reinterpreted): every folded payload
must carry the same suite sha256 (which binds the suite name, source kind,
corpus release, mutator version, provision window, derivation and the
ordered case identities with their content digests); no runner name and no
runner identity may appear twice; incomplete runs fold only with
``allow_partial``, and the board then says so.

Headline: per-kind detection AUC on each judge's kind channel, subject to a
false-alarm ceiling on the judge's native verdict. A judge whose native flag
rate on clean controls exceeds the ceiling is shown but not ranked. The
ceiling is only applied when the suite's controls are gate-verified: for a
real-defects suite the controls are post-fix artifacts that are not proven
clean, so a flag on one may be right, and the board says so instead of
unranking anyone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import BOARD_SCHEMA, DEFECT_KINDS
from .canonical import canonical_json_sha256
from .judges.base import CHANNEL_NATIVE, CHANNEL_VERDICT_FALLBACK, VERDICT_FLAG
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
OTHER_KIND_PREFIX = "other:"


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
    served_models: list[str]
    source: str
    cases_expected: int
    cases_scored: int
    errors: int
    native_false_alarm_rate: Optional[float]
    native_detection_rate: Optional[float]
    mean_kind_auc: Optional[float]
    mean_native_kind_auc: Optional[float]
    mean_verdict_auc: Optional[float]
    localization_rate: Optional[float]
    median_latency_seconds: Optional[float]
    mean_tokens_input: Optional[float]
    mean_tokens_output: Optional[float]
    mean_cost_usd: Optional[float]
    total_cost_usd: Optional[float]
    unpriced_rows: int
    pricing_source: Optional[str]
    rank_status: str  # ranked | over_ceiling | unrankable
    complete: bool
    coerced_verdicts: int = 0
    same_family_rows: int = 0
    kinds: dict[str, KindStats] = field(default_factory=dict)
    verdict_fallback_kinds: list[str] = field(default_factory=list)
    missing_kind_aucs: list[str] = field(default_factory=list)

    @property
    def over_ceiling(self) -> bool:
        return self.rank_status == "over_ceiling"

    @property
    def ranked(self) -> bool:
        return self.rank_status == "ranked"

    def to_dict(self) -> dict[str, Any]:
        payload = {k: v for k, v in self.__dict__.items() if k != "kinds"}
        payload["over_ceiling"] = self.over_ceiling
        payload["kinds"] = {kind: stats.to_dict() for kind, stats in self.kinds.items()}
        return payload


@dataclass
class VerifierBoard:
    suite: dict[str, Any]
    false_alarm_ceiling: float
    ceiling_applied: bool
    kinds: list[str]
    runners: list[RunnerStats]
    sources: dict[str, str]
    incomplete_sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def ordered_runners(self) -> list[RunnerStats]:
        """Ranked judges first by mean kind AUC, then over-ceiling, then unrankable."""

        tier = {"ranked": 0, "over_ceiling": 1, "unrankable": 2}

        def key(stats: RunnerStats):
            auc_value = stats.mean_kind_auc if stats.mean_kind_auc is not None else -1
            return (
                tier.get(stats.rank_status, 3),
                -auc_value,
                stats.median_latency_seconds
                if stats.median_latency_seconds is not None
                else float("inf"),
                stats.runner,
            )

        return sorted(self.runners, key=key)


def _scored(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not row.get("error")]


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _kind_score(row: dict[str, Any], kind: str) -> Optional[float]:
    if kind.startswith(OTHER_KIND_PREFIX):
        # Kinds outside the taxonomy have no kind-specific question anywhere;
        # every judge is scored on its verdict for them.
        return _number(row.get("verdict_score"))
    return _number((row.get("kind_scores") or {}).get(kind))


def _kind_channel(row: dict[str, Any], kind: str) -> str:
    if kind.startswith(OTHER_KIND_PREFIX):
        return CHANNEL_VERDICT_FALLBACK
    return str((row.get("kind_score_channels") or {}).get(kind, CHANNEL_NATIVE))


def _kind_stats(kind: str, rows: list[dict[str, Any]], ceiling: float) -> KindStats:
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
    channels = {_kind_channel(row, kind) for row in scored}
    stats.channel = (
        CHANNEL_NATIVE if channels <= {CHANNEL_NATIVE} else CHANNEL_VERDICT_FALLBACK
    )
    pos_kind = [s for s in (_kind_score(r, kind) for r in defective) if s is not None]
    neg_kind = [s for s in (_kind_score(r, kind) for r in control) if s is not None]
    pos_verdict = [
        s for s in (_number(r.get("verdict_score")) for r in defective) if s is not None
    ]
    neg_verdict = [
        s for s in (_number(r.get("verdict_score")) for r in control) if s is not None
    ]
    stats.kind_auc = auc(pos_kind, neg_kind)
    stats.verdict_auc = auc(pos_verdict, neg_verdict)
    stats.detection_at_ceiling = detection_at_false_alarm_ceiling(
        pos_kind, neg_kind, ceiling
    )
    paired: list[tuple[float, float]] = []
    for members in pairs.values():
        d = members.get("defective")
        c = members.get("control")
        if d is None or c is None or d.get("error") or c.get("error"):
            continue
        ds, cs = _kind_score(d, kind), _kind_score(c, kind)
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


def _suite_kinds(identities: list[dict[str, Any]]) -> list[str]:
    """Taxonomy kinds first, then any ``other:`` kinds a real suite carries."""

    present = {str(item.get("defect_kind")) for item in identities}
    kinds = [kind for kind in DEFECT_KINDS if kind in present]
    kinds += sorted(k for k in present if k.startswith(OTHER_KIND_PREFIX))
    return kinds


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
    identity_sources: dict[str, str] = {}
    runners: list[RunnerStats] = []
    sources: dict[str, str] = {}
    incomplete: list[str] = []
    notes: list[str] = []
    kinds: list[str] = []
    ceiling_applied = True

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
            kinds = _suite_kinds(identities)
            controls = {
                str(item.get("control_clean"))
                for item in identities
                if item.get("variant") == "control"
            }
            ceiling_applied = controls <= {"known_good_gate"}
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
        identity_digest = canonical_json_sha256(runner.get("identity"))
        if identity_digest in identity_sources:
            raise VerifierBoardError(
                f"Runner {name!r} in {source} has the same judge identity "
                f"(family, model, prompt or question set, window) as the runner in "
                f"{identity_sources[identity_digest]}; two runs of one judge are two "
                "boards, not one, whatever they are named"
            )
        runner_sources[name] = source
        identity_sources[identity_digest] = source
        sources[source] = str(suite.get("name"))
        coverage = payload["coverage"]
        complete = bool(coverage.get("complete"))
        if not complete:
            if not allow_partial:
                raise VerifierBoardError(
                    f"Results are incomplete ({coverage.get('scored')} scored, "
                    f"{coverage.get('errors')} errors of {coverage.get('expected')}): "
                    f"{source}. Re-run the same `run` command (it resumes from "
                    "cases.jsonl and retries error rows) or pass --allow-partial."
                )
            incomplete.append(source)
        rows = payload["results"]
        scored = _scored(rows)
        controls_scored = [row for row in scored if row.get("variant") == "control"]
        defectives = [row for row in scored if row.get("variant") == "defective"]
        kind_stats = {
            kind: _kind_stats(kind, rows, false_alarm_ceiling) for kind in kinds
        }
        kind_aucs = {k: s.kind_auc for k, s in kind_stats.items()}
        missing = [k for k, v in kind_aucs.items() if v is None]
        mean_kind_auc = mean_or_none(list(kind_aucs.values())) if not missing else None
        native = [
            s.kind_auc
            for s in kind_stats.values()
            if s.channel == CHANNEL_NATIVE and s.kind_auc is not None
        ]
        verdict_aucs = [
            s.verdict_auc for s in kind_stats.values() if s.verdict_auc is not None
        ]
        native_far = rate(
            sum(1 for r in controls_scored if r.get("verdict") == VERDICT_FLAG),
            len(controls_scored),
        )
        if not ceiling_applied:
            rank_status = "ranked"
        elif native_far is None:
            rank_status = "unrankable"
        elif native_far > false_alarm_ceiling:
            rank_status = "over_ceiling"
        else:
            rank_status = "ranked"
        if mean_kind_auc is None:
            rank_status = "unrankable"
        localizable = [r for r in defectives if r.get("localized") is not None]
        # Spend and latency cover every row, errors included: an error still
        # cost tokens and time (timeouts are the longest calls of all).
        latencies = [
            float(r["latency_ms"]) / 1000.0
            for r in rows
            if _number(r.get("latency_ms")) is not None
        ]
        tokens_in = [
            float(v)
            for v in ((r.get("tokens") or {}).get("input") for r in rows)
            if _number(v) is not None
        ]
        tokens_out = [
            float(v)
            for v in ((r.get("tokens") or {}).get("output") for r in rows)
            if _number(v) is not None
        ]
        costs = [
            float(r["cost_usd"]) for r in rows if _number(r.get("cost_usd")) is not None
        ]
        pricing = payload.get("pricing") or None
        stats = RunnerStats(
            runner=name,
            family=str(runner.get("family")),
            model=str(runner.get("model")),
            served_models=list(runner.get("served_models") or []),
            source=source,
            cases_expected=len(identities),
            cases_scored=len(scored),
            errors=len(rows) - len(scored),
            native_false_alarm_rate=native_far,
            native_detection_rate=rate(
                sum(1 for r in defectives if r.get("verdict") == VERDICT_FLAG),
                len(defectives),
            ),
            mean_kind_auc=mean_kind_auc,
            mean_native_kind_auc=mean_or_none(native),
            mean_verdict_auc=mean_or_none(verdict_aucs),
            localization_rate=rate(
                sum(1 for r in localizable if r.get("localized") is True),
                len(localizable),
            ),
            median_latency_seconds=median_or_none(latencies),
            mean_tokens_input=mean_or_none(tokens_in) if tokens_in else None,
            mean_tokens_output=mean_or_none(tokens_out) if tokens_out else None,
            mean_cost_usd=mean_or_none(costs) if costs else None,
            total_cost_usd=round(sum(costs), 6) if costs else None,
            unpriced_rows=len(rows) - len(costs),
            pricing_source=pricing.get("source") if pricing else None,
            rank_status=rank_status,
            complete=complete,
            coerced_verdicts=sum(
                1
                for r in scored
                if (r.get("raw") or {}).get("verdict_coerced_by_findings") is True
            ),
            same_family_rows=sum(
                1
                for r in rows
                if (r.get("raw") or {}).get("same_family_as_generator") is True
            ),
            kinds=kind_stats,
            verdict_fallback_kinds=[
                k for k, s in kind_stats.items() if s.channel != CHANNEL_NATIVE
            ],
            missing_kind_aucs=missing,
        )
        runners.append(stats)

    assert reference_suite is not None
    if incomplete:
        notes.append(
            "Partial fold: incomplete runs were included with --allow-partial; "
            "rates cover only the cases each judge scored."
        )
    if not ceiling_applied:
        notes.append(
            "Controls in this suite are post-fix artifacts that are not proven clean "
            "(control_clean = unverified). The native false-alarm rate is the flag "
            "rate on those unverified controls, and the false-alarm ceiling is not "
            "applied: no judge is unranked for flagging them."
        )
    derived = reference_suite.get("derived_from")
    if isinstance(derived, dict):
        notes.append(
            f"Derived suite: filtered from parent {str(derived.get('parent_suite_sha256'))[:12]} "
            f"({derived.get('parent_suite_name')!r}) by {derived.get('filter')}; "
            f"{len(derived.get('dropped_pairs') or [])} pair(s) dropped."
        )
    other = [k for k in kinds if k.startswith(OTHER_KIND_PREFIX)]
    if other:
        notes.append(
            "Defect kinds outside the synthetic taxonomy are scored on each judge's "
            f"verdict channel: {', '.join(other)}."
        )
    unrankable = [s.runner for s in runners if s.rank_status == "unrankable"]
    if unrankable:
        notes.append(
            "Unrankable (no scored controls, or a kind with no AUC): "
            + ", ".join(unrankable)
            + ". Shown last, without a rank."
        )
    same_family = [s.runner for s in runners if s.same_family_rows]
    if same_family:
        notes.append(
            "Some cases were generated by a model in the same family as the judge "
            "(same-family rows are counted per judge in the JSON output): "
            + ", ".join(same_family)
            + "."
        )
    return VerifierBoard(
        suite=dict(reference_suite),
        false_alarm_ceiling=false_alarm_ceiling,
        ceiling_applied=ceiling_applied,
        kinds=kinds,
        runners=runners,
        sources=sources,
        incomplete_sources=incomplete,
        notes=notes,
    )


# -- rendering -----------------------------------------------------------------

_SHORT = {
    "amount_changed": "amount",
    "boundary_flipped": "boundary",
    "conjunct_dropped": "conjunct",
    "polarity_swapped": "polarity",
    "date_or_period_wrong": "date/period",
    "entity_wrong": "entity",
}


def _short(kind: str) -> str:
    return _SHORT.get(kind, kind)


def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.0%}"


def _num(value: Optional[float], template: str = "{:.3f}") -> str:
    return "—" if value is None else template.format(value)


def _mark(stats: RunnerStats) -> str:
    return {"over_ceiling": "†", "unrankable": "§"}.get(stats.rank_status, "")


def board_to_json(board: VerifierBoard) -> dict[str, Any]:
    return {
        "schema": BOARD_SCHEMA,
        "suite": board.suite,
        "false_alarm_ceiling": board.false_alarm_ceiling,
        "ceiling_applied": board.ceiling_applied,
        "sources": board.sources,
        "incomplete_sources": board.incomplete_sources,
        "notes": board.notes,
        "defect_kinds": list(board.kinds),
        "runners": [stats.to_dict() for stats in board.ordered_runners()],
    }


def render_board_markdown(board: VerifierBoard) -> str:
    ordered = board.ordered_runners()
    suite = board.suite
    kinds = board.kinds
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
    ceiling_text = (
        f"subject to a false-alarm ceiling of {board.false_alarm_ceiling:.0%} on the "
        "judge's native verdict (flag rate on clean controls). Judges over the ceiling "
        "are shown but not ranked (†)"
        if board.ceiling_applied
        else "with the false-alarm ceiling not applied because the controls are not "
        "proven clean (see note)"
    )
    lines.append(
        f"Headline: per-kind detection AUC on each judge's kind channel, {ceiling_text}. "
        "AUC is pooled Mann-Whitney within a kind, ties count 0.5. `det@ceil` is the "
        "share of defective cases scoring above the control score that admits at most "
        "the ceiling's share of false alarms. Localization counts a finding that names "
        "the mutated rule or the edited token; probability-only judges score blank "
        "there by construction. Kinds marked ‡ have no kind-specific question for that "
        "judge and fall back to the verdict score; a mean AUC marked ‡ includes such "
        "kinds. Judges marked § could not be ranked (no scored controls or a kind with "
        "no AUC). Tokens, latency and cost cover every call, errors included; a blank "
        "cost means no published price or no reported usage, never zero."
    )
    lines.append("")
    header = (
        "| judge | model | native FAR | native det | "
        + " | ".join(f"AUC {_short(k)}" for k in kinds)
        + " | mean AUC | localize | coerced | median s | tokens in/out | cost/case |"
    )
    lines.append(header)
    lines.append("|" + "---|" * (10 + len(kinds)))
    for stats in ordered:
        cells = []
        for kind in kinds:
            ks = stats.kinds[kind]
            fallback = "‡" if ks.channel != CHANNEL_NATIVE else ""
            cells.append(_num(ks.kind_auc) + fallback)
        mean_mark = (
            "‡"
            if stats.verdict_fallback_kinds and stats.mean_kind_auc is not None
            else ""
        )
        tokens = (
            f"{stats.mean_tokens_input:,.0f}/{stats.mean_tokens_output:,.0f}"
            if stats.mean_tokens_input is not None
            and stats.mean_tokens_output is not None
            else "—"
        )
        lines.append(
            f"| {stats.runner}{_mark(stats)} | {stats.model} | "
            f"{_pct(stats.native_false_alarm_rate)} | {_pct(stats.native_detection_rate)} | "
            + " | ".join(cells)
            + f" | {_num(stats.mean_kind_auc)}{mean_mark} | {_pct(stats.localization_rate)} | "
            f"{stats.coerced_verdicts} | {_num(stats.median_latency_seconds, '{:.2f}')} | "
            f"{tokens} | {_num(stats.mean_cost_usd, '${:.5f}')} |"
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
        for kind in kinds:
            ks = stats.kinds[kind]
            fallback = "‡" if ks.channel != CHANNEL_NATIVE else ""
            lines.append(
                f"| {stats.runner} | {_short(kind)}{fallback} | {ks.complete_pairs}/{ks.pairs} | "
                f"{_num(ks.kind_auc)} | {_num(ks.verdict_auc)} | "
                f"{_pct(ks.kind_rise_rate)} | {_num(ks.kind_mean_delta, '{:+.3f}')} | "
                f"{_pct(ks.detection_at_ceiling)} | {_pct(ks.native_detection_rate)} | "
                f"{_pct(ks.native_false_alarm_rate)} | {_pct(ks.localization_rate)} | "
                f"{ks.errors} |"
            )
    lines.append("")
    lines.append("## Spend and coverage")
    lines.append("")
    lines.append(
        "| judge | served model(s) | scored | errors | total cost | unpriced rows | price source |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for stats in ordered:
        lines.append(
            f"| {stats.runner} | {', '.join(stats.served_models) or '—'} | "
            f"{stats.cases_scored}/{stats.cases_expected} | {stats.errors} | "
            f"{_num(stats.total_cost_usd, '${:.4f}')} | {stats.unpriced_rows} | "
            f"{stats.pricing_source or 'no published price recorded; cost blank'} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_board_text(board: VerifierBoard) -> str:
    ordered = board.ordered_runners()
    lines = [
        f"Suite: {board.suite.get('name')}  ({board.suite.get('pair_count')} pairs, "
        f"mutator {board.suite.get('mutator_version')})",
        (
            f"False-alarm ceiling on native verdict: {board.false_alarm_ceiling:.0%}"
            if board.ceiling_applied
            else "False-alarm ceiling not applied: controls are not proven clean"
        ),
        "",
    ]
    for note in board.notes:
        lines.append(f"note: {note}")
    if board.notes:
        lines.append("")
    width = max((len(s.runner) for s in ordered), default=6)
    status = {
        "over_ceiling": " (over ceiling, unranked)",
        "unrankable": " (unrankable)",
    }
    for stats in ordered:
        mean_mark = (
            "‡"
            if stats.verdict_fallback_kinds and stats.mean_kind_auc is not None
            else ""
        )
        lines.append(
            f"{stats.runner:<{width}}  mean kind AUC {_num(stats.mean_kind_auc)}{mean_mark}  "
            f"native FAR {_pct(stats.native_false_alarm_rate)}  "
            f"native det {_pct(stats.native_detection_rate)}  "
            f"localize {_pct(stats.localization_rate)}  "
            f"median {_num(stats.median_latency_seconds, '{:.2f}s')}  "
            f"cost/case {_num(stats.mean_cost_usd, '${:.5f}')}"
            f"{status.get(stats.rank_status, '')}"
        )
        for kind in board.kinds:
            ks = stats.kinds[kind]
            lines.append(
                f"  {kind:<22} pairs {ks.complete_pairs:>3}  kind AUC {_num(ks.kind_auc)}"
                f"  verdict AUC {_num(ks.verdict_auc)}  rise {_pct(ks.kind_rise_rate)}"
                f"  det@ceil {_pct(ks.detection_at_ceiling)}"
                + ("  [verdict fallback]" if ks.channel != CHANNEL_NATIVE else "")
            )
    return "\n".join(lines)
