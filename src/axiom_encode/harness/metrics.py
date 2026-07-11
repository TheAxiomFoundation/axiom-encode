"""
Review metrics for encoding runs.

Tracks reviewer pass rates and checklist item rates over time so prompt and
harness changes can be compared against a stable signal.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .encoding_db import EncodingDB


@dataclass
class CalibrationMetrics:
    """Trend metrics for a specific reviewer or oracle score."""

    metric_name: str = ""
    n_samples: int = 0
    predicted_mean: float = 0.0
    actual_mean: float = 0.0
    mse: float = 0.0  # Mean squared error
    mae: float = 0.0  # Mean absolute error
    bias: float = 0.0  # Systematic over/under prediction
    correlation: Optional[float] = None  # Pearson correlation


@dataclass
class CalibrationSnapshot:
    """Point-in-time calibration across all metrics."""

    timestamp: datetime = field(default_factory=datetime.now)
    metrics: dict[str, CalibrationMetrics] = field(default_factory=dict)
    total_runs: int = 0
    pass_rate: float = 0.0  # % of runs where all validators passed


def _compute_metric(name: str, pairs: list[tuple[float, float]]) -> CalibrationMetrics:
    """Compute calibration metrics for a list of (predicted, actual) pairs."""
    n = len(pairs)
    if n == 0:
        return CalibrationMetrics(
            metric_name=name,
            n_samples=0,
            predicted_mean=0,
            actual_mean=0,
            mse=0,
            mae=0,
            bias=0,
        )

    preds = [p for p, _ in pairs]
    actuals = [a for _, a in pairs]

    pred_mean = sum(preds) / n
    actual_mean = sum(actuals) / n

    # MSE
    mse = sum((p - a) ** 2 for p, a in pairs) / n

    # MAE
    mae = sum(abs(p - a) for p, a in pairs) / n

    # Bias (positive = overconfident)
    bias = pred_mean - actual_mean

    # Correlation (requires >= 3 non-constant samples)
    correlation = None
    if n >= 3:
        pred_std = (sum((p - pred_mean) ** 2 for p in preds) / n) ** 0.5
        actual_std = (sum((a - actual_mean) ** 2 for a in actuals) / n) ** 0.5

        if pred_std > 0 and actual_std > 0:
            cov = sum((p - pred_mean) * (a - actual_mean) for p, a in pairs) / n
            correlation = cov / (pred_std * actual_std)

    return CalibrationMetrics(
        metric_name=name,
        n_samples=n,
        predicted_mean=pred_mean,
        actual_mean=actual_mean,
        mse=mse,
        mae=mae,
        bias=bias,
        correlation=correlation,
    )


def compute_calibration(
    db: EncodingDB,
    min_samples: int = 10,
) -> CalibrationSnapshot:
    """
    Compute calibration metrics from encoding database.

    Uses review_results (checklist-based) to compute pass rates per reviewer.

    Args:
        db: Encoding database
        min_samples: Minimum samples required per metric

    Returns:
        CalibrationSnapshot with all metrics
    """
    runs = db.get_recent_runs(limit=1000)

    # Filter to runs with review results
    runs_with_reviews = [r for r in runs if r.review_results]

    if not runs_with_reviews:
        return CalibrationSnapshot(
            timestamp=datetime.now(),
            metrics={},
            total_runs=0,
            pass_rate=0.0,
        )

    # Extract pass rate pairs for each reviewer
    # Reuse the existing metric shape by treating the expected review outcome
    # as pass=1.0 and measuring the checklist item pass rate as the observation.
    metric_pairs: dict[str, list[tuple[float, float]]] = {}

    for run in runs_with_reviews:
        for review in run.review_results.reviews:
            pass_val = 1.0 if review.passed else 0.0
            items_rate = (
                review.items_passed / review.items_checked
                if review.items_checked > 0
                else pass_val
            )
            metric_pairs.setdefault(review.reviewer, []).append((1.0, items_rate))

        # Oracle dimensions
        if run.review_results.policyengine_match is not None:
            metric_pairs.setdefault("policyengine_match", []).append(
                (1.0, run.review_results.policyengine_match)
            )

    # Compute metrics for each reviewer (only if enough samples)
    metrics = {}
    for name, pairs in metric_pairs.items():
        if len(pairs) >= min_samples:
            metrics[name] = _compute_metric(name, pairs)

    # Overall pass rate (all reviews passed)
    pass_count = sum(1 for r in runs_with_reviews if r.review_results.passed)
    pass_rate = pass_count / len(runs_with_reviews)

    return CalibrationSnapshot(
        timestamp=datetime.now(),
        metrics=metrics,
        total_runs=len(runs_with_reviews),
        pass_rate=pass_rate,
    )


def print_calibration_report(snapshot: CalibrationSnapshot) -> str:
    """Generate human-readable calibration report."""
    lines = [
        "=" * 60,
        "CALIBRATION REPORT",
        f"Generated: {snapshot.timestamp.isoformat()}",
        f"Total Runs: {snapshot.total_runs}",
        f"Pass Rate: {snapshot.pass_rate * 100:.1f}%",
        "=" * 60,
        "",
    ]

    if not snapshot.metrics:
        lines.append("No calibration data available yet.")
        return "\n".join(lines)

    # Header
    lines.append(
        f"{'Metric':<25} {'N':>5} {'Pred':>7} {'Actual':>7} {'Bias':>7} {'MSE':>7}"
    )
    lines.append("-" * 60)

    for name, m in sorted(snapshot.metrics.items()):
        bias_str = f"+{m.bias:.3f}" if m.bias > 0 else f"{m.bias:.3f}"
        lines.append(
            f"{name:<25} {m.n_samples:>5} {m.predicted_mean:>7.2f} "
            f"{m.actual_mean:>7.2f} {bias_str:>7} {m.mse:>7.4f}"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  Bias > 0: Agent overconfident (predicts higher than actual)")
    lines.append("  Bias < 0: Agent underconfident")
    lines.append("  Lower MSE = better calibration")

    return "\n".join(lines)


def save_calibration_snapshot(
    db_path: Path,
    snapshot: CalibrationSnapshot,
) -> None:
    """Save calibration snapshot to database for trend analysis."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calibration_snapshots (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            predicted_mean REAL,
            actual_mean REAL,
            mse REAL,
            n_samples INTEGER
        )
    """)

    for name, m in snapshot.metrics.items():
        cursor.execute(
            """
            INSERT INTO calibration_snapshots (
                id, timestamp, metric_name, predicted_mean, actual_mean, mse, n_samples
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                f"{snapshot.timestamp.isoformat()}_{name}",
                snapshot.timestamp.isoformat(),
                name,
                m.predicted_mean,
                m.actual_mean,
                m.mse,
                m.n_samples,
            ),
        )

    conn.commit()
    conn.close()


def get_calibration_trend(
    db_path: Path,
    metric_name: str,
    limit: int = 30,
) -> list[tuple[datetime, float, float]]:
    """
    Get calibration trend over time for a specific metric.

    Returns list of (timestamp, predicted_mean, actual_mean).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calibration_snapshots (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            predicted_mean REAL,
            actual_mean REAL,
            mse REAL,
            n_samples INTEGER
        )
    """)
    conn.commit()

    cursor.execute(
        """
        SELECT timestamp, predicted_mean, actual_mean
        FROM calibration_snapshots
        WHERE metric_name = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """,
        (metric_name, limit),
    )

    rows = cursor.fetchall()
    conn.close()

    return [(datetime.fromisoformat(ts), pred, actual) for ts, pred, actual in rows]
