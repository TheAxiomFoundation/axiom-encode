"""Calibration harness for the statutory-fidelity referee.

Replays the referee over N>=30 historical generations from ``encodings.db`` and
reports false-positive / false-negative rates. This is the evidence the PR body
carries and the basis on which promotion from advisory to a hard gate is later
decided.

Labels come from the recorded encoding outcome:

* **good** — ``outcome_json.status == "apply_applied"`` (the generation validated
  and merged cleanly).
* **bad**  — ``outcome_json.status == "apply_blocked_validation"`` (the compile /
  CI gate rejected the generated RuleSpec).

The referee "flags" a generation when its verdict is ``flag``. Then:

* good + flag  = false positive (referee flagged a known-good generation)
* bad  + pass  = false negative (referee missed a known-bad generation)

A ``judge_error`` verdict is neither pass nor flag: fail-open is banned, so an
error is reported separately and never counted as a pass (which would inflate
the miss/FN picture into a false clean bill).
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import statutory_fidelity
from .client import JudgeClient
from .run_log import TokenCounts, Verdict

DEFAULT_DB_PATH = Path("encodings.db")
GOOD_STATUS = "apply_applied"
BAD_STATUS = "apply_blocked_validation"


@dataclass
class CalibrationCase:
    run_id: str
    citation: str
    label: str  # "good" | "bad"
    source_text: str
    rulespec_content: str


def load_cases(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    n: int = 30,
    seed: Optional[int] = 0,
    generator_model: str = "gpt-5.5",
) -> list[CalibrationCase]:
    """Load a balanced sample of known-good and known-bad historical generations."""

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        ids = {"good": [], "bad": []}
        for label, status in (("good", GOOD_STATUS), ("bad", BAD_STATUS)):
            rows = conn.execute(
                """
                SELECT id FROM encoding_runs
                WHERE source_text IS NOT NULL AND source_text != ''
                  AND rulespec_content IS NOT NULL AND rulespec_content != ''
                  AND agent_model = ?
                  AND json_extract(outcome_json, '$.status') = ?
                """,
                (generator_model, status),
            ).fetchall()
            ids[label] = [r["id"] for r in rows]

        rng = random.Random(seed)
        rng.shuffle(ids["good"])
        rng.shuffle(ids["bad"])
        half = max(n // 2, 1)

        cases: list[CalibrationCase] = []
        for label, chosen in (("good", ids["good"][:half]), ("bad", ids["bad"][:half])):
            for run_id in chosen:
                row = conn.execute(
                    "SELECT id, citation, source_text, rulespec_content "
                    "FROM encoding_runs WHERE id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    continue
                cases.append(
                    CalibrationCase(
                        run_id=row["id"],
                        citation=row["citation"] or "",
                        label=label,
                        source_text=row["source_text"],
                        rulespec_content=row["rulespec_content"],
                    )
                )
        return cases
    finally:
        conn.close()


@dataclass
class CalibrationReport:
    n: int = 0
    n_good: int = 0
    n_bad: int = 0
    true_positive: int = 0  # bad + flag
    true_negative: int = 0  # good + pass
    false_positive: int = 0  # good + flag
    false_negative: int = 0  # bad + pass
    errors: int = 0  # judge_error (excluded from FP/FN)
    tokens: TokenCounts = field(default_factory=TokenCounts)
    per_case: list[dict[str, Any]] = field(default_factory=list)

    @property
    def false_positive_rate(self) -> Optional[float]:
        denom = self.true_negative + self.false_positive
        return self.false_positive / denom if denom else None

    @property
    def false_negative_rate(self) -> Optional[float]:
        denom = self.true_positive + self.false_negative
        return self.false_negative / denom if denom else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_good": self.n_good,
            "n_bad": self.n_bad,
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "errors": self.errors,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "tokens": self.tokens.to_dict(),
            "per_case": self.per_case,
        }

    def summary(self) -> str:
        fpr = self.false_positive_rate
        fnr = self.false_negative_rate
        return (
            f"Calibration over {self.n} historical generations "
            f"({self.n_good} good, {self.n_bad} bad):\n"
            f"  TP={self.true_positive} TN={self.true_negative} "
            f"FP={self.false_positive} FN={self.false_negative} "
            f"errors={self.errors}\n"
            f"  false-positive rate = "
            f"{'n/a' if fpr is None else f'{fpr:.1%}'} "
            f"(good generations wrongly flagged)\n"
            f"  false-negative rate = "
            f"{'n/a' if fnr is None else f'{fnr:.1%}'} "
            f"(bad generations missed)\n"
            f"  judge tokens: in={self.tokens.input} out={self.tokens.output}"
        )


def run_calibration(
    cases: list[CalibrationCase],
    *,
    client: Optional[JudgeClient] = None,
) -> CalibrationReport:
    """Replay the fidelity referee over the labelled cases and score FP/FN."""

    client = client or JudgeClient()
    report = CalibrationReport()
    for case in cases:
        event = statutory_fidelity.run(
            case.source_text,
            case.rulespec_content,
            citation=case.citation,
            run_id=case.run_id,
            client=client,
        )
        report.n += 1
        report.tokens = report.tokens + event.tokens
        if case.label == "good":
            report.n_good += 1
        else:
            report.n_bad += 1

        outcome: str
        if event.verdict == Verdict.ERROR:
            report.errors += 1
            outcome = "error"
        elif event.verdict == Verdict.FLAG:
            outcome = "flag"
            if case.label == "bad":
                report.true_positive += 1
            else:
                report.false_positive += 1
        else:  # PASS
            outcome = "pass"
            if case.label == "good":
                report.true_negative += 1
            else:
                report.false_negative += 1

        report.per_case.append(
            {
                "run_id": case.run_id,
                "citation": case.citation,
                "label": case.label,
                "verdict": event.verdict.value,
                "outcome": outcome,
                "confidence": event.confidence,
                "n_findings": len(event.findings),
                "escalated": event.escalated,
            }
        )
    return report


def calibrate(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    n: int = 30,
    seed: Optional[int] = 0,
    client: Optional[JudgeClient] = None,
) -> CalibrationReport:
    """Convenience: load a balanced sample and run the replay."""

    cases = load_cases(db_path, n=n, seed=seed)
    return run_calibration(cases, client=client)


def _main(argv: Optional[list[str]] = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(
        description="Statutory-fidelity calibration replay"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = calibrate(args.db, n=args.n, seed=args.seed)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
