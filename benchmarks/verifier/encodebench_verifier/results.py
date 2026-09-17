"""Run a judge over a suite and persist ``results.json`` with its contract.

Each results payload binds the suite identity (name, sha256, source kind,
corpus release, mutator version, provision window) and the runner identity
(family, model, prompt or question-set digest) so the board can refuse to
fold what is not comparable. Rows are appended to ``cases.jsonl`` as they
finish, so an interrupted run resumes without re-spending.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from . import DEFECT_KINDS, RESULTS_SCHEMA, SUITE_SCHEMA, __version__
from .canonical import canonical_json_sha256, utc_now_iso
from .cases import CaseSuite, VerifierCase
from .judges.base import JudgeResponse, JudgeRunner, error_response
from .localization import localize
from .pricing import Price, cost_usd

RESULT_SHA256_FIELD = "result_sha256"
CASES_FILE = "cases.jsonl"
RESULTS_FILE = "results.json"


class ResultsError(ValueError):
    """A results payload is unreadable, malformed or internally inconsistent."""


def result_row(
    index: int,
    case: VerifierCase,
    response: JudgeResponse,
    *,
    price: Optional[Price],
    supports_localization: bool = True,
) -> dict[str, Any]:
    localized, evidence = (None, None)
    if case.is_defective and supports_localization and response.ok:
        localized, evidence = localize(case.locator, response.findings)
    row: dict[str, Any] = {
        "index": index,
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "variant": case.variant,
        "defect_kind": case.defect_kind,
        "verdict": response.verdict,
        "verdict_score": response.verdict_score,
        "kind_scores": {kind: response.kind_scores.get(kind) for kind in DEFECT_KINDS},
        "kind_score_channels": dict(response.kind_score_channels),
        "findings": list(response.findings),
        "localized": localized,
        "localization_evidence": evidence,
        "latency_ms": response.latency_ms,
        "tokens": {"input": response.tokens_input, "output": response.tokens_output},
        "cost_usd": cost_usd(price, response.tokens_input, response.tokens_output),
        "model": response.model,
        "error": response.error,
        "raw": dict(response.raw),
    }
    row[RESULT_SHA256_FIELD] = canonical_json_sha256(row)
    return row


def _verify_row(row: dict[str, Any], context: str) -> None:
    unsigned = dict(row)
    digest = unsigned.pop(RESULT_SHA256_FIELD, None)
    if not isinstance(digest, str) or digest != canonical_json_sha256(unsigned):
        raise ResultsError(f"{context} is missing its result_sha256 or does not match")


def _restamp(row: dict[str, Any], index: int) -> dict[str, Any]:
    """Bind a row to its position in the suite being assembled.

    Rows judged against a parent suite fold into a derived (filtered) suite
    without re-judging; only their position changes, so the digest is
    recomputed over the re-indexed row.
    """

    if row.get("index") == index:
        return row
    restamped = {k: v for k, v in row.items() if k != RESULT_SHA256_FIELD}
    restamped["index"] = index
    restamped[RESULT_SHA256_FIELD] = canonical_json_sha256(restamped)
    return restamped


def load_completed_rows(out_dir: Path) -> dict[str, dict[str, Any]]:
    path = Path(out_dir) / CASES_FILE
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResultsError(f"{path}:{line_no} is not JSON: {exc}") from exc
        _verify_row(row, f"{path}:{line_no}")
        rows[str(row["case_id"])] = row
    return rows


def run_suite(
    suite: CaseSuite,
    runner: JudgeRunner,
    out_dir: Path,
    *,
    price: Optional[Price],
    workers: int = 4,
    resume: bool = True,
    limit: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
    retry_errors: bool = True,
) -> dict[str, Any]:
    """Judge every case (or the first ``limit``) and write results.json."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = list(suite.cases)
    if limit is not None:
        cases = cases[: max(0, limit)]
    completed = load_completed_rows(out_dir) if resume else {}
    if retry_errors:
        completed = {k: v for k, v in completed.items() if not v.get("error")}
    wanted = {case.case_id for case in cases}
    todo = [
        (index, case)
        for index, case in enumerate(suite.cases, 1)
        if case.case_id in wanted and case.case_id not in completed
    ]
    lock = threading.Lock()
    jsonl_path = out_dir / CASES_FILE
    if not resume and jsonl_path.exists():
        jsonl_path.unlink()

    def work(item: tuple[int, VerifierCase]) -> dict[str, Any]:
        index, case = item
        try:
            response = runner.judge(case)
        except Exception as exc:  # noqa: BLE001 - fail closed, keep the run alive
            # A runner is expected to return an error response itself; if it
            # raises instead, the case is still recorded as an error (never a
            # pass, never silently dropped) and is retried on resume.
            response = error_response(
                runner.model, f"runner_exception:{type(exc).__name__}", str(exc)[:500]
            )
        row = result_row(
            index,
            case,
            response,
            price=price,
            supports_localization=getattr(runner, "supports_localization", True),
        )
        with lock:
            with jsonl_path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed[case.case_id] = row
        if progress:
            status = (
                row["verdict"] if not row["error"] else f"error:{row['error']['type']}"
            )
            progress(f"[{len(completed)}/{len(cases)}] {case.case_id} -> {status}")
        return row

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(work, item) for item in todo]
            for future in as_completed(futures):
                future.result()

    payload = assemble_results(suite, runner, completed, price=price, limit=limit)
    (out_dir / RESULTS_FILE).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False)
    )
    return payload


def assemble_results(
    suite: CaseSuite,
    runner: JudgeRunner,
    completed: dict[str, dict[str, Any]],
    *,
    price: Optional[Price],
    limit: Optional[int] = None,
) -> dict[str, Any]:
    expected = suite.cases if limit is None else suite.cases[: max(0, limit)]
    position = {case.case_id: index for index, case in enumerate(suite.cases, 1)}
    rows = [
        _restamp(completed[c.case_id], position[c.case_id])
        for c in expected
        if c.case_id in completed
    ]
    errors = sum(1 for row in rows if row.get("error"))
    return {
        "schema": RESULTS_SCHEMA,
        "generated_at": utc_now_iso(),
        "verifier_version": __version__,
        "suite": suite.summary(),
        "case_identities": suite.case_identities(),
        "runner": {
            "name": runner.name,
            "family": runner.family,
            "model": runner.model,
            "identity": runner.identity(),
        },
        "pricing": price.to_dict() if price else None,
        "results": rows,
        "coverage": {
            "complete": len(rows) == len(suite.cases) and errors == 0,
            "expected": len(suite.cases),
            "requested": len(expected),
            "scored": len(rows) - errors,
            "errors": errors,
            "results_sha256": canonical_json_sha256(rows),
        },
    }


def load_results(path: Path) -> dict[str, Any]:
    """Load one results payload with structural and digest checks."""

    path = Path(path)
    if path.is_dir():
        path = path / RESULTS_FILE
    if not path.is_file():
        raise ResultsError(f"results file not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultsError(f"could not read results {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResultsError(f"results must be a JSON object: {path}")
    if payload.get("schema") != RESULTS_SCHEMA:
        raise ResultsError(
            f"results {path} carry schema {payload.get('schema')!r}; the board "
            f"folds only {RESULTS_SCHEMA!r}"
        )
    for key in ("suite", "case_identities", "runner", "results", "coverage"):
        if key not in payload:
            raise ResultsError(f"results {path} are missing the '{key}' section")
    suite = payload["suite"]
    if not isinstance(suite, dict) or not isinstance(suite.get("sha256"), str):
        raise ResultsError(f"results {path} carry no suite identity digest")
    runner = payload["runner"]
    if not isinstance(runner, dict) or not runner.get("name"):
        raise ResultsError(f"results {path} carry no runner name")
    rows = payload["results"]
    if not isinstance(rows, list):
        raise ResultsError(f"results {path} carry no result rows")
    identities = payload["case_identities"]
    if not isinstance(identities, list):
        raise ResultsError(f"results {path} carry malformed case identities")
    # The suite digest must be the digest of the identities this payload
    # carries: a payload cannot keep a suite's sha256 while altering its cases.
    recomputed = canonical_json_sha256(
        {
            "schema": SUITE_SCHEMA,
            "name": suite.get("name"),
            "source_kind": suite.get("source_kind"),
            "corpus_release": suite.get("corpus_release"),
            "mutator_version": suite.get("mutator_version"),
            "provision_chars": suite.get("provision_chars"),
            "case_identities": identities,
        }
    )
    if recomputed != suite["sha256"]:
        raise ResultsError(
            f"results {path} carry a suite sha256 that does not match their own "
            "suite identity and case identities"
        )
    by_case = {str(item.get("case_id")): item for item in identities}
    seen: set[str] = set()
    for position, row in enumerate(rows, 1):
        context = f"result row #{position} in {path}"
        if not isinstance(row, dict):
            raise ResultsError(f"{context} is malformed")
        _verify_row(row, context)
        case_id = str(row.get("case_id"))
        if case_id not in by_case:
            raise ResultsError(f"{context} names case {case_id!r} not in the suite")
        identity = by_case[case_id]
        for field in ("pair_id", "variant", "defect_kind"):
            if row.get(field) != identity.get(field):
                raise ResultsError(
                    f"{context} disagrees with the suite on {field} for {case_id}"
                )
        if case_id in seen:
            raise ResultsError(f"{context} duplicates case {case_id}")
        seen.add(case_id)
        if row.get("verdict") not in ("pass", "flag", "error"):
            raise ResultsError(
                f"{context} carries unknown verdict {row.get('verdict')!r}"
            )
        if row.get("verdict") == "error" and not row.get("error"):
            raise ResultsError(f"{context} is an error verdict without an error")
        if row.get("error") and row.get("verdict") != "error":
            raise ResultsError(f"{context} carries an error on a non-error verdict")
        score = row.get("verdict_score")
        if score is not None and not (
            isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0
        ):
            raise ResultsError(f"{context} verdict_score is not within [0, 1]")
    coverage = payload["coverage"]
    if not isinstance(coverage, dict):
        raise ResultsError(f"results {path} carry malformed coverage")
    errors = sum(1 for row in rows if row.get("error"))
    claimed = coverage.get("complete")
    actual = len(rows) == len(identities) and errors == 0
    if claimed is not actual:
        raise ResultsError(
            f"results {path} claim coverage.complete={claimed!r} but the rows say "
            f"{actual!r}"
        )
    if coverage.get("results_sha256") != canonical_json_sha256(rows):
        raise ResultsError(f"results {path} coverage digest does not match its rows")
    return payload
