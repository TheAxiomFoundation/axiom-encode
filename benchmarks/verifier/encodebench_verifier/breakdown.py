"""Paired detection broken down by case properties.

A board answers "does the judge separate defective from control?". This
answers "on which cases?": paired rise (defective scored strictly above its
own control on the kind channel) and pooled AUC per bucket of module size,
relative size of the fix, fix stage or triage confidence. Buckets come from
the suite (texts and origin metadata), scores from a run; the two are joined
on case id with the content digests checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .cases import CaseSuite
from .metrics import auc, rate
from .results import ResultsError, load_results


class BreakdownError(ValueError):
    """The run does not belong to the suite, or the property is unknown."""


@dataclass
class Bucket:
    label: str
    pairs: int
    rises: int
    rise_rate: Optional[float]
    kind_auc: Optional[float]
    verdict_auc: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Breakdown:
    runner: str
    model: str
    suite: str
    property: str
    complete_pairs: int
    buckets: list[Bucket] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner": self.runner,
            "model": self.model,
            "suite": self.suite,
            "property": self.property,
            "complete_pairs": self.complete_pairs,
            "buckets": [b.to_dict() for b in self.buckets],
        }

    def render(self) -> str:
        lines = [
            f"{self.runner} ({self.model}) on {self.suite}: paired detection by "
            f"{self.property}, {self.complete_pairs} complete pairs",
            "| bucket | pairs | paired rise | kind AUC | verdict AUC |",
            "|---|---|---|---|---|",
        ]
        for b in self.buckets:
            lines.append(
                f"| {b.label} | {b.pairs} | "
                f"{'—' if b.rise_rate is None else f'{b.rise_rate:.0%}'} | "
                f"{'—' if b.kind_auc is None else f'{b.kind_auc:.3f}'} | "
                f"{'—' if b.verdict_auc is None else f'{b.verdict_auc:.3f}'} |"
            )
        return "\n".join(lines)


def _numeric_buckets(
    edges: list[float], fmt: Callable[[float], str]
) -> list[tuple[str, Callable[[float], bool]]]:
    out = []
    for lo, hi in zip(edges, edges[1:]):
        label = f"{fmt(lo)} to {fmt(hi)}" if hi != float("inf") else f"{fmt(lo)} and up"
        out.append((label, (lambda v, lo=lo, hi=hi: lo <= v < hi)))
    return out


def _chars(value: float) -> str:
    return f"{int(value):,}" if value != float("inf") else "inf"


def _pct(value: float) -> str:
    return f"{value:.0%}" if value != float("inf") else "inf"


# property name -> (value extractor over (defective case, control case), buckets)
def _properties() -> dict[
    str, tuple[Callable[..., Any], list[tuple[str, Callable[[Any], bool]]]]
]:
    return {
        "size": (
            lambda d, c: max(
                len(d.provision_text) + len(d.artifact_text),
                len(c.provision_text) + len(c.artifact_text),
            ),
            _numeric_buckets(
                [0, 10_000, 30_000, 60_000, 100_000, float("inf")], _chars
            ),
        ),
        "diff": (
            lambda d, c: (
                abs(len(d.artifact_text) - len(c.artifact_text))
                / max(1, len(d.artifact_text))
            ),
            _numeric_buckets([0, 0.02, 0.1, 0.3, float("inf")], _pct),
        ),
        "confidence": (
            lambda d, c: float(d.origin.get("confidence") or 0.0),
            _numeric_buckets([0, 0.6, 0.8, 0.9, 1.0001], lambda v: f"{v:.2f}"),
        ),
        "fix_stage": (
            lambda d, c: str(d.origin.get("fix_stage") or "unknown"),
            [],  # categorical: buckets discovered from the data
        ),
        "kind": (lambda d, c: d.defect_kind, []),
        "jurisdiction": (
            lambda d, c: d.citation.split("/")[0].split(":")[0] or "unknown",
            [],
        ),
    }


PROPERTIES = tuple(_properties())


def breakdown(suite: CaseSuite, run: Path, prop: str) -> Breakdown:
    if prop not in PROPERTIES:
        raise BreakdownError(f"unknown property {prop!r}; choose from {PROPERTIES}")
    try:
        payload = load_results(Path(run))
    except ResultsError as exc:
        raise BreakdownError(str(exc)) from exc
    if payload["suite"].get("sha256") != suite.sha256:
        raise BreakdownError(
            "run was judged against a different suite "
            f"({str(payload['suite'].get('sha256'))[:12]} vs {suite.sha256[:12]})"
        )
    rows = {r["case_id"]: r for r in payload["results"] if not r.get("error")}
    by_pair: dict[str, dict[str, Any]] = {}
    for case in suite.cases:
        by_pair.setdefault(case.pair_id, {})[case.variant] = case
    extract, buckets = _properties()[prop]
    samples: list[tuple[Any, float, float, float, float]] = []
    for pair_id, members in by_pair.items():
        d, c = members.get("defective"), members.get("control")
        if d is None or c is None:
            continue
        rd, rc = rows.get(d.case_id), rows.get(c.case_id)
        if rd is None or rc is None:
            continue
        kind = d.defect_kind
        dk = rd["kind_scores"].get(kind, rd["verdict_score"])
        ck = rc["kind_scores"].get(kind, rc["verdict_score"])
        if dk is None or ck is None:
            dk, ck = rd["verdict_score"], rc["verdict_score"]
        samples.append(
            (
                extract(d, c),
                float(dk),
                float(ck),
                float(rd["verdict_score"]),
                float(rc["verdict_score"]),
            )
        )
    if not buckets:
        labels = sorted({str(s[0]) for s in samples})
        buckets = [
            (label, (lambda v, label=label: str(v) == label)) for label in labels
        ]
    result = Breakdown(
        runner=str(payload["runner"]["name"]),
        model=str(payload["runner"]["model"]),
        suite=suite.name,
        property=prop,
        complete_pairs=len(samples),
    )
    for label, member in buckets:
        sel = [s for s in samples if member(s[0])]
        if not sel:
            continue
        rises = sum(1 for s in sel if s[1] > s[2])
        result.buckets.append(
            Bucket(
                label=label,
                pairs=len(sel),
                rises=rises,
                rise_rate=rate(rises, len(sel)),
                kind_auc=auc([s[1] for s in sel], [s[2] for s in sel]),
                verdict_auc=auc([s[3] for s in sel], [s[4] for s in sel]),
            )
        )
    return result
