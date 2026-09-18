"""Run-to-run self-agreement of a judge on identical case text.

Two runs of the same judge (same family, model, prompt or question set, and
window) over cases whose provision and artifact digests are identical form a
test-retest sample. Rows are joined on those content digests, never on case
ids, so runs over different suites that happen to share text still compare.
A judge that changes its verdict on the same text between calls is noisy in a
way no detection AUC shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Optional

from .results import load_results


class AgreementError(ValueError):
    """The two runs cannot be compared as test and retest."""


@dataclass
class AgreementReport:
    runner_a: str
    runner_b: str
    model: str
    same_identity: bool
    joined: int
    verdict_agreement: Optional[float]
    verdict_agreements: int
    finding_kind_set_agreement: Optional[float]
    finding_kind_set_agreements: Optional[int]
    median_abs_verdict_score_delta: Optional[float]
    max_abs_verdict_score_delta: Optional[float]
    median_abs_kind_score_delta: dict[str, Optional[float]] = field(
        default_factory=dict
    )
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def render(self) -> str:
        lines = [
            f"Self-agreement of {self.model}: {self.runner_a} vs {self.runner_b}"
            + ("" if self.same_identity else "  (runner identities differ)"),
            f"identical texts judged in both runs: {self.joined}",
        ]
        if self.joined:
            lines.append(
                f"verdict agreement: {self.verdict_agreements}/{self.joined} "
                f"({self.verdict_agreement:.0%})"
            )
            if self.finding_kind_set_agreement is not None:
                lines.append(
                    f"identical finding-kind sets: {self.finding_kind_set_agreements}/"
                    f"{self.joined} ({self.finding_kind_set_agreement:.0%})"
                )
            lines.append(
                f"|Δ verdict score|: median {self.median_abs_verdict_score_delta:.3f}, "
                f"max {self.max_abs_verdict_score_delta:.3f}"
            )
            for kind, value in self.median_abs_kind_score_delta.items():
                if value is not None:
                    lines.append(f"  median |Δ kind score| {kind}: {value:.3f}")
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)


def _scored_by_text(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload["results"]:
        if row.get("error"):
            continue
        key = (str(row.get("provision_sha256")), str(row.get("artifact_sha256")))
        rows[key] = row
    return rows


def compare_runs(path_a: Path, path_b: Path) -> AgreementReport:
    a = load_results(Path(path_a))
    b = load_results(Path(path_b))
    runner_a, runner_b = a["runner"], b["runner"]
    if runner_a.get("model") != runner_b.get("model") or runner_a.get(
        "family"
    ) != runner_b.get("family"):
        raise AgreementError(
            "self-agreement compares two runs of one judge; these runs name "
            f"{runner_a.get('family')}:{runner_a.get('model')} and "
            f"{runner_b.get('family')}:{runner_b.get('model')}"
        )
    same_identity = runner_a.get("identity") == runner_b.get("identity")
    rows_a = _scored_by_text(a)
    rows_b = _scored_by_text(b)
    keys = sorted(set(rows_a) & set(rows_b))
    notes: list[str] = []
    if not same_identity:
        notes.append(
            "Runner identities differ (prompt, schema, window or max_tokens); this "
            "compares two configurations of one model, not test and retest."
        )
    if a["suite"].get("sha256") != b["suite"].get("sha256"):
        notes.append(
            "Runs come from different suites; only cases with identical provision "
            "and artifact text were joined."
        )
    report = AgreementReport(
        runner_a=str(runner_a.get("name")),
        runner_b=str(runner_b.get("name")),
        model=str(runner_a.get("model")),
        same_identity=same_identity,
        joined=len(keys),
        verdict_agreement=None,
        verdict_agreements=0,
        finding_kind_set_agreement=None,
        finding_kind_set_agreements=None,
        median_abs_verdict_score_delta=None,
        max_abs_verdict_score_delta=None,
        notes=notes,
    )
    if not keys:
        return report
    pairs = [(rows_a[k], rows_b[k]) for k in keys]
    report.verdict_agreements = sum(x["verdict"] == y["verdict"] for x, y in pairs)
    report.verdict_agreement = round(report.verdict_agreements / len(pairs), 6)
    deltas = [
        abs(float(x["verdict_score"]) - float(y["verdict_score"])) for x, y in pairs
    ]
    report.median_abs_verdict_score_delta = round(float(median(deltas)), 6)
    report.max_abs_verdict_score_delta = round(max(deltas), 6)
    if any(x.get("findings") or y.get("findings") for x, y in pairs):
        same = sum(
            {f.get("kind") for f in x.get("findings", [])}
            == {f.get("kind") for f in y.get("findings", [])}
            for x, y in pairs
        )
        report.finding_kind_set_agreements = same
        report.finding_kind_set_agreement = round(same / len(pairs), 6)
    kinds = sorted(set(pairs[0][0].get("kind_scores") or {}))
    for kind in kinds:
        kd = [
            abs(float(x["kind_scores"][kind]) - float(y["kind_scores"][kind]))
            for x, y in pairs
            if x["kind_scores"].get(kind) is not None
            and y["kind_scores"].get(kind) is not None
        ]
        report.median_abs_kind_score_delta[kind] = (
            round(float(median(kd)), 6) if kd else None
        )
    return report
