"""Render the runbook's boards block from committed board JSON.

Every number in the runbook's boards section comes from here, so it can be
regenerated after a re-fold and checked in CI: ``report --check`` fails when
the block in ``docs/encodebench-verifier.md`` no longer matches what the
committed ``board.json`` files say. Interpretation that is not a number
(why controls get flagged, what a binary channel implies) stays in the
hand-written prose around the block; statements the block makes (who leads,
who is fastest, how many judges rank) are computed, never typed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

BEGIN = "<!-- begin generated boards: verifier.py report; edit boards, not this -->"
END = "<!-- end generated boards -->"

_SHORT = {
    "amount_changed": "amount",
    "boundary_flipped": "boundary",
    "conjunct_dropped": "conjunct",
    "polarity_swapped": "polarity",
    "date_or_period_wrong": "date or period",
    "entity_wrong": "entity",
}


class ReportError(ValueError):
    """The runbook has no generated block, or a board is unreadable."""


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _auc(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _money(value: Optional[float], places: int = 5) -> str:
    return "n/a" if value is None else f"${value:.{places}f}"


def _kind_label(kind: str) -> str:
    if kind.startswith("other:"):
        return kind.split(":", 1)[1].replace("_", " ")
    return _SHORT.get(kind, kind)


def _judge_mark(runner: dict[str, Any]) -> str:
    status = runner.get("rank_status")
    if status == "unrankable":
        return "§"
    if status == "over_ceiling":
        return "†"
    return ""


def load_board(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "board.json"
    try:
        board = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"could not read board {path}: {exc}") from exc
    if not isinstance(board, dict) or not isinstance(board.get("runners"), list):
        raise ReportError(f"{path} is not a verifier board")
    return board


def _suite_lines(board: dict[str, Any]) -> list[str]:
    suite = board["suite"]
    line = (
        f"{suite['pair_count']} pairs ({suite['case_count']} cases), suite "
        f"`{suite['sha256'][:12]}`, source `{suite['source_kind']}`"
    )
    if suite.get("mutator_version"):
        line += f", built with mutator {suite['mutator_version']}"
    line += f", provision window {suite['provision_chars']:,} characters."
    lines = [line]
    derived = suite.get("derived_from")
    if derived:
        dropped = derived.get("dropped_pairs") or []
        reason = (derived.get("filter") or {}).get("reason")
        text = (
            f"Filtered from suite `{derived['parent_suite_sha256'][:12]}` "
            f"({derived['parent_suite_name']}): {len(dropped)} pair(s) dropped"
        )
        text += f", because they {reason}." if reason else "."
        lines.append(text)
    return lines


def _table(board: dict[str, Any]) -> list[str]:
    lines = [
        "| judge | model | native FAR | native det | mean kind AUC | verdict AUC "
        "| localize | median s | cost/case | total |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in board["runners"]:
        mean = _auc(r.get("mean_kind_auc"))
        if r.get("mean_kind_auc") is not None and r.get("verdict_fallback_kinds"):
            mean += " ‡"
        localize = (
            "blank by construction"
            if r.get("localization_rate") is None
            else _pct(r["localization_rate"])
        )
        median = r.get("median_latency_seconds")
        total = _money(r.get("total_cost_usd"), 2)
        if r.get("cases_scored") != r.get("cases_expected"):
            total += f" ({r.get('cases_scored')}/{r.get('cases_expected')} scored)"
        lines.append(
            f"| {r['runner']}{_judge_mark(r)} | {r['model']} "
            f"| {_pct(r.get('native_false_alarm_rate'))} "
            f"| {_pct(r.get('native_detection_rate'))} | {mean} "
            f"| {_auc(r.get('mean_verdict_auc'))} | {localize} "
            f"| {'n/a' if median is None else f'{median:.2f}'} "
            f"| {_money(r.get('mean_cost_usd'))} | {total} |"
        )
    return lines


def _per_kind(board: dict[str, Any]) -> str:
    runners = board["runners"]
    names = " / ".join(r["runner"] for r in runners)
    parts = []
    for kind in board["defect_kinds"]:
        pairs = max(
            (r["kinds"].get(kind, {}).get("pairs", 0) for r in runners), default=0
        )
        values = []
        for r in runners:
            stats = r["kinds"].get(kind) or {}
            value = _auc(stats.get("kind_auc"))
            if stats.get("channel") == "verdict_fallback" and value != "n/a":
                value += " ‡"
            values.append(value)
        parts.append(f"{_kind_label(kind)} (n={pairs}) {' / '.join(values)}")
    return f"Per-kind kind-channel AUC ({names}): " + "; ".join(parts) + "."


def _facts(board: dict[str, Any]) -> list[str]:
    runners = board["runners"]
    ceiling = board.get("false_alarm_ceiling")
    scored = [r for r in runners if r.get("mean_kind_auc") is not None]
    facts = []
    ranked = [r for r in runners if r.get("rank_status") == "ranked"]
    with_far = [r for r in runners if r.get("native_false_alarm_rate") is not None]
    lowest_far = min(with_far, key=lambda r: r["native_false_alarm_rate"], default=None)
    if not board.get("ceiling_applied"):
        facts.append(
            "The false-alarm ceiling is not applied: this suite's controls are "
            "not gate-verified, so flagging them unranks no one."
        )
    elif ranked:
        facts.append(
            f"{len(ranked)} of {len(runners)} judges stay under the "
            f"{_pct(ceiling)} false-alarm ceiling and rank."
        )
    elif lowest_far is not None:
        facts.append(
            f"No judge ranks: every judge flags more than {_pct(ceiling)} of the "
            f"clean controls at its native verdict. The lowest rate is "
            f"{lowest_far['runner']}'s, at "
            f"{_pct(lowest_far['native_false_alarm_rate'])}."
        )
    if scored:
        top = max(scored, key=lambda r: r["mean_kind_auc"])
        sentence = (
            f"Highest mean kind-channel AUC: {top['runner']} "
            f"({_auc(top['mean_kind_auc'])}), at {_money(top.get('mean_cost_usd'))} "
            f"and {top.get('median_latency_seconds'):.2f} s a case."
        )
        referees = [r for r in scored if r.get("family") == "referee" and r is not top]
        if referees:
            best = max(referees, key=lambda r: r["mean_kind_auc"])
            sentence += (
                f" The best referee configuration is {best['runner']} "
                f"({_auc(best['mean_kind_auc'])}), at "
                f"{_money(best.get('mean_cost_usd'))} and "
                f"{best.get('median_latency_seconds'):.2f} s."
            )
        facts.append(sentence)
        kind_aucs = [
            (stats["kind_auc"], kind)
            for kind, stats in top["kinds"].items()
            if stats.get("kind_auc") is not None
        ]
        weakest = sorted(kind_aucs)[:2]
        if weakest:
            facts.append(
                f"{top['runner']}'s weakest kinds: "
                + " and ".join(f"{_kind_label(k)} ({_auc(a)})" for a, k in weakest)
                + "."
            )
    localizing = [r for r in runners if r.get("localization_rate") is not None]
    if localizing:
        best_loc = max(localizing, key=lambda r: r["localization_rate"])
        facts.append(
            f"Best localization: {best_loc['runner']}, "
            f"{_pct(best_loc['localization_rate'])} of defective cases with a "
            "finding naming the mutated rule or token."
        )
    timed = [r for r in runners if r.get("median_latency_seconds") is not None]
    if timed:
        slowest = max(timed, key=lambda r: r["median_latency_seconds"])
        facts.append(
            f"Slowest median call: {slowest['runner']}, "
            f"{slowest['median_latency_seconds']:.2f} s."
        )
    coerced = sum(int(r.get("coerced_verdicts") or 0) for r in runners)
    facts.append(
        f"Coerced verdicts (a raw pass that carried findings): {coerced} in total."
    )
    total = sum(float(r.get("total_cost_usd") or 0.0) for r in runners)
    facts.append(f"Spend recorded in this board's results: {_money(total, 2)}.")
    return facts


def render_board(board: dict[str, Any]) -> str:
    lines = [f"### {board['suite']['name']}", ""]
    lines += _suite_lines(board) + [""]
    lines += _table(board) + [""]
    lines.append(_per_kind(board))
    lines.append("")
    lines.append("Computed from the board:")
    lines.append("")
    lines += [f"- {fact}" for fact in _facts(board)]
    return "\n".join(lines)


def render_block(boards: list[dict[str, Any]]) -> str:
    body = "\n\n".join(render_board(board) for board in boards)
    return f"{BEGIN}\n\n{body}\n\n{END}"


def splice(runbook: str, block: str) -> str:
    start, end = runbook.find(BEGIN), runbook.find(END)
    if start < 0 or end < start:
        raise ReportError("the runbook has no generated boards block to replace")
    return runbook[:start] + block + runbook[end + len(END) :]


def current_block(runbook: str) -> str:
    start, end = runbook.find(BEGIN), runbook.find(END)
    if start < 0 or end < start:
        raise ReportError("the runbook has no generated boards block")
    return runbook[start : end + len(END)]
