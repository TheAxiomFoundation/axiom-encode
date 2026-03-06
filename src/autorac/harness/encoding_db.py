"""
Encoding Database - tracks encoding runs for continuous improvement.

Key insight: We learn from the JOURNEY (errors, fixes, iterations),
not from comparing predictions to actuals.

Now also tracks full session transcripts for replay and analysis.
"""

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional


@dataclass
class TokenUsage:
    """Token usage for a session or run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Cost estimate using Opus 4.5 pricing (updated Feb 2026).

        Rates: $5/M input, $25/M output, $0.50/M cache read, $6.25/M cache create.
        Source: https://platform.claude.com/docs/en/about-claude/pricing
        """
        return (
            self.input_tokens * 5 / 1_000_000
            + self.output_tokens * 25 / 1_000_000
            + self.cache_read_tokens * 0.50 / 1_000_000
            + self.cache_creation_tokens * 6.25 / 1_000_000
        )


# Session event types
EventType = Literal[
    "session_start",
    "session_end",
    "user_prompt",
    "assistant_response",
    "tool_call",
    "tool_result",
    "subagent_start",
    "subagent_end",
    # Validation events (3-tier pipeline)
    "validation_ci_start",
    "validation_ci_end",
    "validation_oracle_start",
    "validation_oracle_end",
    "validation_llm_start",
    "validation_llm_end",
]


@dataclass
class SessionEvent:
    """A single event in a session transcript."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    session_id: str = ""
    sequence: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""  # EventType
    tool_name: Optional[str] = None
    content: str = ""  # Main content (prompt, response, tool input/output)
    metadata: dict = field(default_factory=dict)  # Extra data (tokens, duration, etc.)


@dataclass
class Session:
    """A full Claude Code session transcript."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_id: Optional[str] = None  # FK to EncodingRun if this is an encoding session
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    model: str = ""
    cwd: str = ""
    event_count: int = 0
    total_tokens: int = 0


@dataclass
class ComplexityFactors:
    """Upfront analysis of statute complexity."""

    cross_references: list[str] = field(default_factory=list)  # ["1402(a)", "164(f)"]
    has_nested_structure: bool = False
    has_numeric_thresholds: bool = False
    has_phase_in_out: bool = False
    estimated_variables: int = 1
    estimated_parameters: int = 0


@dataclass
class IterationError:
    """An error encountered during encoding."""

    error_type: str  # "parse", "test", "import", "style", "other"
    message: str
    variable: Optional[str] = None  # Which variable failed, if applicable
    fix_applied: Optional[str] = None  # What fix was attempted


@dataclass
class Iteration:
    """A single encoding attempt."""

    attempt: int
    duration_ms: int
    errors: list[IterationError] = field(default_factory=list)
    success: bool = False


@dataclass
class OracleResult:
    """Detailed result from an oracle validator."""

    name: str  # "policyengine" or "taxsim"
    score: Optional[float] = None  # Match rate 0-1
    passed: bool = False
    issues: list[str] = field(default_factory=list)
    duration_ms: int = 0
    test_cases_run: int = 0
    test_cases_passed: int = 0


@dataclass
class ReviewResult:
    """Result from a single reviewer (checklist-based)."""

    reviewer: str = ""  # "rac", "formula", "parameter", "integration"
    passed: bool = False
    items_checked: int = 0
    items_passed: int = 0
    critical_issues: list[str] = field(default_factory=list)
    important_issues: list[str] = field(default_factory=list)
    minor_issues: list[str] = field(default_factory=list)
    lessons: str = ""


@dataclass
class ReviewResults:
    """Aggregated results from all reviewers."""

    reviews: list[ReviewResult] = field(default_factory=list)
    policyengine_match: Optional[float] = None
    taxsim_match: Optional[float] = None
    oracle_context: dict = field(default_factory=dict)
    lessons: str = ""

    @property
    def passed(self) -> bool:
        """All reviews passed."""
        return all(r.passed for r in self.reviews) if self.reviews else False

    @property
    def total_critical_issues(self) -> int:
        """Total critical issues across all reviews."""
        return sum(len(r.critical_issues) for r in self.reviews)


# Backward-compat aliases for old code that references these types
FinalScores = ReviewResults
PredictedScores = ReviewResults
ActualScores = ReviewResults
AgentSuggestion = ReviewResult


@dataclass
class EncodingRun:
    """A complete encoding run from start to finish."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)

    # What we're encoding
    citation: str = ""
    file_path: str = ""
    statute_text: Optional[str] = None

    # Upfront analysis
    complexity: ComplexityFactors = field(default_factory=ComplexityFactors)

    # Review results (replaces predicted/actual/final_scores/suggestions)
    review_results: Optional[ReviewResults] = None
    lessons: str = ""

    # Iteration tracking
    iteration: int = 1
    parent_run_id: Optional[str] = None

    # The journey
    iterations: list[Iteration] = field(default_factory=list)

    # Final result
    total_duration_ms: int = 0
    rac_content: str = ""

    # Agent info
    agent_type: str = "encoder"
    agent_model: str = ""
    autorac_version: str = ""

    # Session linkage
    session_id: Optional[str] = None

    # Backward-compat properties for old code that sets these
    @property
    def predicted(self) -> Optional[ReviewResults]:
        return self.review_results

    @predicted.setter
    def predicted(self, value):
        if value is not None:
            self.review_results = value

    @property
    def actual(self) -> Optional[ReviewResults]:
        return self.review_results

    @actual.setter
    def actual(self, value):
        if value is not None:
            self.review_results = value

    @property
    def final_scores(self) -> Optional[ReviewResults]:
        return self.review_results

    @final_scores.setter
    def final_scores(self, value):
        if value is not None:
            self.review_results = value

    @property
    def predicted_scores(self) -> Optional[ReviewResults]:
        """Alias for backwards compatibility."""
        return self.review_results

    @predicted_scores.setter
    def predicted_scores(self, value):
        if value is not None:
            self.review_results = value

    @property
    def suggestions(self) -> list:
        """Backward compat: return empty list."""
        return []

    @suggestions.setter
    def suggestions(self, value):
        """Backward compat: ignore."""
        pass

    @property
    def iterations_needed(self) -> int:
        return len(self.iterations)

    @property
    def success(self) -> bool:
        return self.iterations and self.iterations[-1].success

    @property
    def all_errors(self) -> list[IterationError]:
        errors = []
        for it in self.iterations:
            errors.extend(it.errors)
        return errors


def create_run(
    file_path: str,
    citation: str,
    agent_type: str,
    agent_model: str,
    rac_content: str,
    statute_text: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    review_results: Optional[ReviewResults] = None,
    lessons: str = "",
) -> EncodingRun:
    """Factory function to create an EncodingRun with defaults."""
    from autorac import __version__

    return EncodingRun(
        file_path=file_path,
        citation=citation,
        agent_type=agent_type,
        agent_model=agent_model,
        rac_content=rac_content,
        statute_text=statute_text,
        iteration=2 if parent_run_id else 1,
        parent_run_id=parent_run_id,
        review_results=review_results,
        lessons=lessons,
        autorac_version=__version__,
    )


class EncodingDB:
    """SQLite-based encoding database."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Migrate: rename 'runs' to 'encoding_runs' if old table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
        )
        if cursor.fetchone():
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='encoding_runs'"
            )
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE runs RENAME TO encoding_runs")

        # Encoding runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS encoding_runs (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                citation TEXT,
                file_path TEXT,
                complexity_json TEXT,
                iterations_json TEXT,
                total_duration_ms INTEGER,
                final_scores_json TEXT,
                agent_type TEXT,
                agent_model TEXT,
                rac_content TEXT,
                predicted_scores_json TEXT,
                session_id TEXT,
                iteration INTEGER DEFAULT 1,
                parent_run_id TEXT,
                actual_scores_json TEXT,
                suggestions_json TEXT
            )
        """)

        # Add columns if they don't exist (for migration)
        for col, col_type, default in [
            ("predicted_scores_json", "TEXT", None),
            ("session_id", "TEXT", None),
            ("iteration", "INTEGER", "1"),
            ("parent_run_id", "TEXT", None),
            ("actual_scores_json", "TEXT", None),
            ("suggestions_json", "TEXT", None),
            ("review_results_json", "TEXT", None),
            ("lessons", "TEXT", "''"),
            ("autorac_version", "TEXT", "''"),
        ]:
            try:
                stmt = f"ALTER TABLE encoding_runs ADD COLUMN {col} {col_type}"
                if default is not None:
                    stmt += f" DEFAULT {default}"
                cursor.execute(stmt)
            except sqlite3.OperationalError:
                pass  # Column already exists

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citation ON encoding_runs(citation)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON encoding_runs(timestamp)
        """)

        # Calibration snapshots table (per-metric rows for trend analysis)
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

        # Sessions table - full Claude Code session transcripts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_run ON sessions(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_started ON sessions(started_at)
        """)

        # Session events table - individual events within a session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_session ON session_events(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type ON session_events(event_type)
        """)

        # =====================================================================
        # Add token usage columns to sessions (migration)
        # =====================================================================
        for col in [
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
            "estimated_cost_usd",
        ]:
            try:
                cursor.execute(
                    f"ALTER TABLE sessions ADD COLUMN {col} INTEGER DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists

        conn.commit()
        conn.close()

    def log_run(self, run: EncodingRun) -> str:
        """Log a completed encoding run. Returns the run ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convert dataclasses to JSON
        complexity_json = json.dumps(
            {
                "cross_references": run.complexity.cross_references,
                "has_nested_structure": run.complexity.has_nested_structure,
                "has_numeric_thresholds": run.complexity.has_numeric_thresholds,
                "has_phase_in_out": run.complexity.has_phase_in_out,
                "estimated_variables": run.complexity.estimated_variables,
                "estimated_parameters": run.complexity.estimated_parameters,
            }
        )

        iterations_json = json.dumps(
            [
                {
                    "attempt": it.attempt,
                    "duration_ms": it.duration_ms,
                    "success": it.success,
                    "errors": [
                        {
                            "error_type": e.error_type,
                            "message": e.message,
                            "variable": e.variable,
                            "fix_applied": e.fix_applied,
                        }
                        for e in it.errors
                    ],
                }
                for it in run.iterations
            ]
        )

        # Serialize review_results (new checklist-based format)
        review_results_json = None
        if run.review_results:
            review_results_json = json.dumps(
                {
                    "reviews": [
                        {
                            "reviewer": r.reviewer,
                            "passed": r.passed,
                            "items_checked": r.items_checked,
                            "items_passed": r.items_passed,
                            "critical_issues": r.critical_issues,
                            "important_issues": r.important_issues,
                            "minor_issues": r.minor_issues,
                            "lessons": r.lessons,
                        }
                        for r in run.review_results.reviews
                    ],
                    "policyengine_match": run.review_results.policyengine_match,
                    "taxsim_match": run.review_results.taxsim_match,
                    "oracle_context": run.review_results.oracle_context,
                    "lessons": run.review_results.lessons,
                }
            )

        cursor.execute(
            """
            INSERT OR REPLACE INTO encoding_runs
            (id, timestamp, citation, file_path, complexity_json, iterations_json,
             total_duration_ms, final_scores_json, agent_type, agent_model, rac_content,
             predicted_scores_json, session_id, iteration, parent_run_id,
             actual_scores_json, suggestions_json, review_results_json, lessons,
             autorac_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                run.id,
                run.timestamp.isoformat(),
                run.citation,
                run.file_path,
                complexity_json,
                iterations_json,
                run.total_duration_ms,
                None,  # final_scores_json (legacy, no longer written)
                run.agent_type,
                run.agent_model,
                run.rac_content,
                None,  # predicted_scores_json (legacy, no longer written)
                run.session_id,
                run.iteration,
                run.parent_run_id,
                None,  # actual_scores_json (legacy, no longer written)
                None,  # suggestions_json (legacy, no longer written)
                review_results_json,
                run.lessons,
                run.autorac_version,
            ),
        )

        conn.commit()
        conn.close()

        return run.id

    def get_run(self, run_id: str) -> Optional[EncodingRun]:
        """Get a specific run by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM encoding_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_run(row)

    def get_recent_runs(self, limit: int = 20) -> list[EncodingRun]:
        """Get recent runs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM encoding_runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def get_runs_for_citation(self, citation: str) -> list[EncodingRun]:
        """Get all runs for a specific citation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM encoding_runs WHERE citation = ? ORDER BY timestamp DESC",
            (citation,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def update_review_results(self, run_id: str, review_results: ReviewResults) -> None:
        """Update a run with review results after validation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        review_results_json = json.dumps(
            {
                "reviews": [
                    {
                        "reviewer": r.reviewer,
                        "passed": r.passed,
                        "items_checked": r.items_checked,
                        "items_passed": r.items_passed,
                        "critical_issues": r.critical_issues,
                        "important_issues": r.important_issues,
                        "minor_issues": r.minor_issues,
                        "lessons": r.lessons,
                    }
                    for r in review_results.reviews
                ],
                "policyengine_match": review_results.policyengine_match,
                "taxsim_match": review_results.taxsim_match,
                "oracle_context": review_results.oracle_context,
                "lessons": review_results.lessons,
            }
        )

        cursor.execute(
            "UPDATE encoding_runs SET review_results_json = ? WHERE id = ?",
            (review_results_json, run_id),
        )

        conn.commit()
        conn.close()

    def get_error_stats(self) -> dict:
        """Get error type distribution."""
        runs = self.get_recent_runs(limit=100)

        error_counts = {}
        for run in runs:
            for err in run.all_errors:
                error_counts[err.error_type] = error_counts.get(err.error_type, 0) + 1

        total = sum(error_counts.values())
        return {
            "counts": error_counts,
            "percentages": {
                k: v / total * 100 if total > 0 else 0 for k, v in error_counts.items()
            },
            "total_runs": len(runs),
            "total_errors": total,
        }

    def get_iteration_stats(self) -> dict:
        """Get iteration distribution."""
        runs = self.get_recent_runs(limit=100)

        iteration_counts = {}
        for run in runs:
            n = run.iterations_needed
            iteration_counts[n] = iteration_counts.get(n, 0) + 1

        total = len(runs)
        avg = (
            sum(n * c for n, c in iteration_counts.items()) / total if total > 0 else 0
        )

        return {
            "distribution": iteration_counts,
            "average": avg,
            "first_try_rate": iteration_counts.get(1, 0) / total * 100
            if total > 0
            else 0,
            "total_runs": total,
        }

    def _row_to_run(self, row) -> EncodingRun:
        """Convert database row to EncodingRun.

        Handles various schema versions:
        - 11 columns: original schema
        - 13 columns: added predicted_scores, session_id
        - 17 columns: added iteration, parent_run_id, actual_scores, suggestions
        - 19 columns: added review_results_json, lessons
        """
        # Extract columns by position with defaults for missing columns
        autorac_version = ""
        if len(row) == 11:
            (
                id,
                timestamp,
                citation,
                file_path,
                complexity_json,
                iterations_json,
                total_duration_ms,
                final_scores_json,
                agent_type,
                agent_model,
                rac_content,
            ) = row
            predicted_scores_json = None
            session_id = None
            iteration = 1
            parent_run_id = None
            actual_scores_json = None
            suggestions_json = None
            review_results_json = None
            lessons = ""
        elif len(row) == 13:
            (
                id,
                timestamp,
                citation,
                file_path,
                complexity_json,
                iterations_json,
                total_duration_ms,
                final_scores_json,
                agent_type,
                agent_model,
                rac_content,
                predicted_scores_json,
                session_id,
            ) = row
            iteration = 1
            parent_run_id = None
            actual_scores_json = None
            suggestions_json = None
            review_results_json = None
            lessons = ""
        elif len(row) == 17:
            (
                id,
                timestamp,
                citation,
                file_path,
                complexity_json,
                iterations_json,
                total_duration_ms,
                final_scores_json,
                agent_type,
                agent_model,
                rac_content,
                predicted_scores_json,
                session_id,
                iteration,
                parent_run_id,
                actual_scores_json,
                suggestions_json,
            ) = row
            review_results_json = None
            lessons = ""
        else:
            # 19+ columns (new schema with review_results_json, lessons)
            (
                id,
                timestamp,
                citation,
                file_path,
                complexity_json,
                iterations_json,
                total_duration_ms,
                final_scores_json,
                agent_type,
                agent_model,
                rac_content,
                predicted_scores_json,
                session_id,
                iteration,
                parent_run_id,
                actual_scores_json,
                suggestions_json,
                review_results_json,
                lessons,
            ) = row[:19]
            autorac_version = row[19] if len(row) > 19 else ""

        # Parse complexity
        c = json.loads(complexity_json) if complexity_json else {}
        complexity = ComplexityFactors(
            cross_references=c.get("cross_references", []),
            has_nested_structure=c.get("has_nested_structure", False),
            has_numeric_thresholds=c.get("has_numeric_thresholds", False),
            has_phase_in_out=c.get("has_phase_in_out", False),
            estimated_variables=c.get("estimated_variables", 1),
            estimated_parameters=c.get("estimated_parameters", 0),
        )

        # Parse iterations
        iterations = []
        if iterations_json:
            for it_data in json.loads(iterations_json):
                errors = [
                    IterationError(
                        error_type=e["error_type"],
                        message=e["message"],
                        variable=e.get("variable"),
                        fix_applied=e.get("fix_applied"),
                    )
                    for e in it_data.get("errors", [])
                ]
                iterations.append(
                    Iteration(
                        attempt=it_data["attempt"],
                        duration_ms=it_data["duration_ms"],
                        errors=errors,
                        success=it_data.get("success", False),
                    )
                )

        # Parse review_results (new format first, fall back to legacy)
        review_results = None
        if review_results_json:
            rr = json.loads(review_results_json)
            review_results = ReviewResults(
                reviews=[
                    ReviewResult(
                        reviewer=r.get("reviewer", ""),
                        passed=r.get("passed", False),
                        items_checked=r.get("items_checked", 0),
                        items_passed=r.get("items_passed", 0),
                        critical_issues=r.get("critical_issues", []),
                        important_issues=r.get("important_issues", []),
                        minor_issues=r.get("minor_issues", []),
                        lessons=r.get("lessons", ""),
                    )
                    for r in rr.get("reviews", [])
                ],
                policyengine_match=rr.get("policyengine_match"),
                taxsim_match=rr.get("taxsim_match"),
                oracle_context=rr.get("oracle_context", {}),
                lessons=rr.get("lessons", ""),
            )
        elif final_scores_json or actual_scores_json or predicted_scores_json:
            # Backward compat: convert old numeric scores to ReviewResults
            # Derive pass_rate from scores (score >= 7.0 = passed)
            reviews = []
            # Use actual_scores if available, else final_scores, else predicted
            score_data = None
            for json_str in [
                actual_scores_json,
                final_scores_json,
                predicted_scores_json,
            ]:
                if json_str:
                    score_data = json.loads(json_str)
                    break

            if score_data:
                for reviewer_name in [
                    "rac_reviewer",
                    "formula_reviewer",
                    "parameter_reviewer",
                    "integration_reviewer",
                ]:
                    score = score_data.get(reviewer_name, 0)
                    if score is not None:
                        reviews.append(
                            ReviewResult(
                                reviewer=reviewer_name,
                                passed=float(score) >= 7.0,
                                items_checked=10,
                                items_passed=int(float(score)),
                            )
                        )

                review_results = ReviewResults(
                    reviews=reviews,
                    policyengine_match=score_data.get("policyengine_match"),
                    taxsim_match=score_data.get("taxsim_match"),
                    oracle_context=score_data.get("oracle_context", {}),
                )

        return EncodingRun(
            id=id,
            timestamp=datetime.fromisoformat(timestamp),
            citation=citation,
            file_path=file_path,
            complexity=complexity,
            review_results=review_results,
            lessons=lessons or "",
            iteration=iteration or 1,
            parent_run_id=parent_run_id,
            iterations=iterations,
            total_duration_ms=total_duration_ms or 0,
            agent_type=agent_type or "encoder",
            agent_model=agent_model or "",
            autorac_version=autorac_version or "",
            rac_content=rac_content or "",
            session_id=session_id,
        )

    # =========================================================================
    # Session Logging Methods
    # =========================================================================

    def start_session(
        self, model: str = "", cwd: str = "", session_id: str = None
    ) -> Session:
        """Start a new session and return it."""
        session = Session(
            model=model,
            cwd=cwd or os.getcwd(),
        )
        # Allow custom session_id for SDK orchestrator
        if session_id:
            session.id = session_id

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO sessions (id, started_at, model, cwd, event_count, total_tokens)
            VALUES (?, ?, ?, ?, 0, 0)
        """,
            (session.id, session.started_at.isoformat(), session.model, session.cwd),
        )

        conn.commit()
        conn.close()

        return session

    def end_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE sessions SET ended_at = ? WHERE id = ?
        """,
            (datetime.now().isoformat(), session_id),
        )

        conn.commit()
        conn.close()

    def log_event(
        self,
        session_id: str,
        event_type: str,
        content: str = "",
        tool_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SessionEvent:
        """Log an event to a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get next sequence number
        cursor.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM session_events WHERE session_id = ?",
            (session_id,),
        )
        sequence = cursor.fetchone()[0]

        event = SessionEvent(
            session_id=session_id,
            sequence=sequence,
            event_type=event_type,
            tool_name=tool_name,
            content=content,
            metadata=metadata or {},
        )

        cursor.execute(
            """
            INSERT INTO session_events (id, session_id, sequence, timestamp, event_type, tool_name, content, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                event.id,
                event.session_id,
                event.sequence,
                event.timestamp.isoformat(),
                event.event_type,
                event.tool_name,
                event.content,
                json.dumps(event.metadata),
            ),
        )

        # Update event count
        cursor.execute(
            """
            UPDATE sessions SET event_count = event_count + 1 WHERE id = ?
        """,
            (session_id,),
        )

        conn.commit()
        conn.close()

        return event

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return Session(
            id=row[0],
            run_id=row[1],
            started_at=datetime.fromisoformat(row[2]) if row[2] else datetime.now(),
            ended_at=datetime.fromisoformat(row[3]) if row[3] else None,
            model=row[4] or "",
            cwd=row[5] or "",
            event_count=row[6] or 0,
            total_tokens=row[7] or 0,
        )

    def get_session_events(self, session_id: str) -> list[SessionEvent]:
        """Get all events for a session, ordered by sequence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, session_id, sequence, timestamp, event_type, tool_name, content, metadata_json
            FROM session_events
            WHERE session_id = ?
            ORDER BY sequence
        """,
            (session_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        events = []
        for row in rows:
            events.append(
                SessionEvent(
                    id=row[0],
                    session_id=row[1],
                    sequence=row[2],
                    timestamp=datetime.fromisoformat(row[3])
                    if row[3]
                    else datetime.now(),
                    event_type=row[4] or "",
                    tool_name=row[5],
                    content=row[6] or "",
                    metadata=json.loads(row[7]) if row[7] else {},
                )
            )

        return events

    def get_recent_sessions(self, limit: int = 20) -> list[Session]:
        """Get recent sessions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        sessions = []
        for row in rows:
            sessions.append(
                Session(
                    id=row[0],
                    run_id=row[1],
                    started_at=datetime.fromisoformat(row[2])
                    if row[2]
                    else datetime.now(),
                    ended_at=datetime.fromisoformat(row[3]) if row[3] else None,
                    model=row[4] or "",
                    cwd=row[5] or "",
                    event_count=row[6] or 0,
                    total_tokens=row[7] or 0,
                )
            )

        return sessions

    def get_session_stats(self) -> dict:
        """Get session statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total = cursor.fetchone()[0]

        # Event type distribution
        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM session_events
            GROUP BY event_type
            ORDER BY count DESC
        """)
        event_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Tool usage
        cursor.execute("""
            SELECT tool_name, COUNT(*) as count
            FROM session_events
            WHERE tool_name IS NOT NULL
            GROUP BY tool_name
            ORDER BY count DESC
            LIMIT 20
        """)
        tool_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Average events per session
        cursor.execute("SELECT AVG(event_count) FROM sessions")
        avg_events = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_sessions": total,
            "event_type_counts": event_counts,
            "tool_usage": tool_counts,
            "avg_events_per_session": round(avg_events, 1),
        }

    def update_session_tokens(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        """Update token usage for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        estimated_cost = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        ).estimated_cost_usd

        cursor.execute(
            """
            UPDATE sessions
            SET input_tokens = ?,
                output_tokens = ?,
                cache_read_tokens = ?,
                cache_creation_tokens = ?,
                total_tokens = ?,
                estimated_cost_usd = ?
            WHERE id = ?
        """,
            (
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                input_tokens + output_tokens,
                estimated_cost,
                session_id,
            ),
        )

        conn.commit()
        conn.close()



# Backward compatibility alias
ExperimentDB = EncodingDB
