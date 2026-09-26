"""Run a judge over a suite and persist ``results.json`` with its contract.

Each results payload binds the suite identity (name, sha256, source kind,
corpus release, mutator version, provision window, derivation) and the runner
identity (family, model, prompt or question-set digest) so the board can
refuse to fold what is not comparable. Rows are appended to ``cases.jsonl``
as they finish, each bound to the case's content digests and to the runner's
identity digest, so an interrupted run resumes without re-spending and a
stale row (another judge, a changed prompt, a rebuilt artifact) is never
attributed to the current run. The payload carries a digest over everything
except its timestamp.
"""

from __future__ import annotations

import json
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from . import DEFECT_KINDS, RESULTS_SCHEMA, SUITE_SCHEMA, __version__
from .canonical import canonical_json_sha256, utc_now_iso
from .cases import CaseSuite, VerifierCase
from .judges.base import (
    CHANNELS,
    VERDICT_ERROR,
    VERDICTS,
    JudgeResponse,
    JudgeRunner,
    checked,
    error_response,
)
from .localization import localize
from .pricing import Price, cost_usd

RESULT_SHA256_FIELD = "result_sha256"
PAYLOAD_SHA256_FIELD = "payload_sha256"
CASES_FILE = "cases.jsonl"
RESULTS_FILE = "results.json"


class ResultsError(ValueError):
    """A results payload is unreadable, malformed or internally inconsistent."""


def runner_identity_sha256(runner: JudgeRunner) -> str:
    return canonical_json_sha256(
        {
            "name": runner.name,
            "family": runner.family,
            "model": runner.model,
            "identity": runner.identity(),
        }
    )


def _sign(row: dict[str, Any]) -> dict[str, Any]:
    unsigned = {k: v for k, v in row.items() if k != RESULT_SHA256_FIELD}
    unsigned[RESULT_SHA256_FIELD] = canonical_json_sha256(unsigned)
    return unsigned


def result_row(
    index: int,
    case: VerifierCase,
    response: JudgeResponse,
    *,
    price: Optional[Price],
    supports_localization: bool = True,
    runner_identity_digest: str = "",
    runner_name: str = "",
) -> dict[str, Any]:
    response = checked(response)
    localized, evidence = (None, None)
    if case.is_defective and supports_localization and response.ok:
        localized, evidence = localize(case.locator, response.findings)
    row: dict[str, Any] = {
        "index": index,
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "variant": case.variant,
        "defect_kind": case.defect_kind,
        "provision_sha256": case.provision_sha256,
        "artifact_sha256": case.artifact_sha256,
        "runner_name": runner_name,
        "runner_identity_sha256": runner_identity_digest,
        "verdict": response.verdict,
        "verdict_score": response.verdict_score,
        "kind_scores": {kind: response.kind_scores.get(kind) for kind in DEFECT_KINDS},
        "kind_score_channels": dict(response.kind_score_channels),
        "findings": list(response.findings),
        "localized": localized,
        "localization_evidence": evidence,
        "latency_ms": int(response.latency_ms),
        "tokens": {"input": response.tokens_input, "output": response.tokens_output},
        "cost_usd": cost_usd(price, response.tokens_input, response.tokens_output),
        "model": response.model,
        "error": response.error,
        "raw": dict(response.raw),
    }
    return _sign(row)


def _verify_row(row: dict[str, Any], context: str) -> None:
    digest = row.get(RESULT_SHA256_FIELD)
    unsigned = {k: v for k, v in row.items() if k != RESULT_SHA256_FIELD}
    if not isinstance(digest, str) or digest != canonical_json_sha256(unsigned):
        raise ResultsError(f"{context} is missing its result_sha256 or does not match")


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read ``cases.jsonl`` rows, splitting on newline bytes only.

    JSON text is written with ``ensure_ascii=False``, so U+2028 and friends
    may appear inside a row; they are not record separators here. Exactly
    one unparseable *final* line without a trailing newline (a write cut off
    by a crash) is dropped with a warning; anything else is refused.
    """

    text = path.read_text(encoding="utf-8")
    parts = text.split("\n")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(parts, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_no == len(parts) and not text.endswith("\n"):
                warnings.warn(
                    f"{path}: dropped one truncated trailing row (line {line_no})",
                    stacklevel=2,
                )
                continue
            raise ResultsError(f"{path}:{line_no} is not JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ResultsError(f"{path}:{line_no} is not a JSON object")
        _verify_row(row, f"{path}:{line_no}")
        rows.append(row)
    return rows


def load_completed_rows(
    out_dir: Path,
    *,
    suite: Optional[CaseSuite] = None,
    runner_identity_digest: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    """Rows in ``cases.jsonl`` that belong to this suite's content and runner.

    Later rows win for a case id. Rows whose provision or artifact digest
    differs from the suite's case, or whose runner identity digest differs
    from the current runner, are ignored (never reused, never counted).
    """

    path = Path(out_dir) / CASES_FILE
    if not path.is_file():
        return {}
    by_case = {case.case_id: case for case in suite.cases} if suite else None
    rows: dict[str, dict[str, Any]] = {}
    for row in read_rows(path):
        case_id = str(row.get("case_id"))
        if by_case is not None:
            case = by_case.get(case_id)
            if case is None:
                continue
            if (
                row.get("provision_sha256") != case.provision_sha256
                or row.get("artifact_sha256") != case.artifact_sha256
            ):
                continue
        if (
            runner_identity_digest is not None
            and row.get("runner_identity_sha256") != runner_identity_digest
        ):
            continue
        rows[case_id] = row
    return rows


def _rotate(path: Path) -> None:
    if path.exists():
        stamp = utc_now_iso().replace(":", "").replace("+", "").replace(".", "")
        path.rename(path.with_name(f"{path.name}.{stamp}.bak"))


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
    """Judge every case (or the first ``limit``) and write results.json.

    Rows for cases outside ``limit`` that were completed earlier are kept, so
    a spot check never downgrades a finished run. An interrupt cancels queued
    cases, writes what finished, and re-raises.
    """

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / CASES_FILE
    if not resume:
        _rotate(jsonl_path)
        _rotate(out_dir / RESULTS_FILE)
    identity_digest = runner_identity_sha256(runner)
    supports_localization = bool(getattr(runner, "supports_localization", True))
    requested = suite.cases if limit is None else suite.cases[:limit]
    completed = (
        load_completed_rows(
            out_dir, suite=suite, runner_identity_digest=identity_digest
        )
        if resume
        else {}
    )
    if retry_errors:
        completed = {k: v for k, v in completed.items() if not v.get("error")}
    wanted = {case.case_id for case in requested}
    todo = [
        (index, case)
        for index, case in enumerate(suite.cases, 1)
        if case.case_id in wanted and case.case_id not in completed
    ]
    lock = threading.Lock()
    stop = threading.Event()

    def work(item: tuple[int, VerifierCase]) -> Optional[dict[str, Any]]:
        index, case = item
        if stop.is_set():
            return None
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
            supports_localization=supports_localization,
            runner_identity_digest=identity_digest,
            runner_name=runner.name,
        )
        with lock:
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            completed[case.case_id] = row
        if progress:
            status = (
                row["verdict"] if not row["error"] else f"error:{row['error']['type']}"
            )
            progress(f"[{len(completed)}/{len(requested)}] {case.case_id} -> {status}")
        return row

    if todo:
        pool = ThreadPoolExecutor(max_workers=max(1, workers))
        try:
            futures = [pool.submit(work, item) for item in todo]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
            payload = assemble_results(suite, runner, completed, price=price)
            (out_dir / RESULTS_FILE).write_text(
                json.dumps(payload, indent=1, ensure_ascii=False)
            )
            raise
        pool.shutdown(wait=True)

    payload = assemble_results(suite, runner, completed, price=price)
    (out_dir / RESULTS_FILE).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False)
    )
    load_results(out_dir / RESULTS_FILE)  # our own output must satisfy the contract
    return payload


def _finalize_row(
    row: dict[str, Any], index: int, price: Optional[Price]
) -> dict[str, Any]:
    """Bind a row to its position in the suite being assembled and re-cost it.

    Rows judged against a parent suite fold into a derived (filtered) suite
    without re-judging: only their position changes. Cost is recomputed from
    the row's reported tokens and the payload's price, so every row prices
    under the one source the payload names.
    """

    tokens = row.get("tokens") or {}
    cost = cost_usd(price, tokens.get("input"), tokens.get("output"))
    if row.get("index") == index and row.get("cost_usd") == cost:
        return row
    updated = dict(row)
    updated["index"] = index
    updated["cost_usd"] = cost
    return _sign(updated)


def payload_sha256(payload: dict[str, Any]) -> str:
    """Digest over every section except the timestamp and the digest itself."""

    body = {
        k: v
        for k, v in payload.items()
        if k not in (PAYLOAD_SHA256_FIELD, "generated_at")
    }
    return canonical_json_sha256(body)


def assemble_results(
    suite: CaseSuite,
    runner: JudgeRunner,
    completed: dict[str, dict[str, Any]],
    *,
    price: Optional[Price],
) -> dict[str, Any]:
    rows = [
        _finalize_row(completed[case.case_id], index, price)
        for index, case in enumerate(suite.cases, 1)
        if case.case_id in completed
    ]
    errors = sum(1 for row in rows if row.get("error"))
    served = sorted({str(row.get("model")) for row in rows if not row.get("error")})
    payload: dict[str, Any] = {
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
            "identity_sha256": runner_identity_sha256(runner),
            "served_models": served,
        },
        "pricing": price.to_dict() if price else None,
        "results": rows,
        "coverage": {
            "complete": bool(suite.cases)
            and len(rows) == len(suite.cases)
            and errors == 0,
            "expected": len(suite.cases),
            "scored": len(rows) - errors,
            "errors": errors,
            "results_sha256": canonical_json_sha256(rows),
        },
    }
    payload[PAYLOAD_SHA256_FIELD] = payload_sha256(payload)
    return payload


def _unit(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


def _nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def load_results(path: Path) -> dict[str, Any]:
    """Load one results payload with structural and digest checks."""

    path = Path(path)
    if path.is_dir():
        path = path / RESULTS_FILE
    if not path.is_file():
        raise ResultsError(f"results file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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
    if payload.get(PAYLOAD_SHA256_FIELD) != payload_sha256(payload):
        raise ResultsError(
            f"results {path} payload digest is missing or does not match; the "
            "runner, pricing, coverage or suite sections were edited"
        )
    suite = payload["suite"]
    if not isinstance(suite, dict) or not isinstance(suite.get("sha256"), str):
        raise ResultsError(f"results {path} carry no suite identity digest")
    runner = payload["runner"]
    if not isinstance(runner, dict) or not runner.get("name"):
        raise ResultsError(f"results {path} carry no runner name")
    if runner.get("identity_sha256") != canonical_json_sha256(
        {
            "name": runner.get("name"),
            "family": runner.get("family"),
            "model": runner.get("model"),
            "identity": runner.get("identity"),
        }
    ):
        raise ResultsError(f"results {path} runner identity digest does not match")
    rows = payload["results"]
    if not isinstance(rows, list):
        raise ResultsError(f"results {path} carry no result rows")
    identities = payload["case_identities"]
    if not isinstance(identities, list) or not identities:
        raise ResultsError(f"results {path} carry no case identities")
    recomputed = canonical_json_sha256(
        {
            "schema": SUITE_SCHEMA,
            "name": suite.get("name"),
            "source_kind": suite.get("source_kind"),
            "corpus_release": suite.get("corpus_release"),
            "mutator_version": suite.get("mutator_version"),
            "provision_chars": suite.get("provision_chars"),
            "derived_from": suite.get("derived_from"),
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
        for field in (
            "pair_id",
            "variant",
            "defect_kind",
            "provision_sha256",
            "artifact_sha256",
        ):
            if row.get(field) != identity.get(field):
                raise ResultsError(
                    f"{context} disagrees with the suite on {field} for {case_id}"
                )
        if row.get("runner_identity_sha256") != runner.get("identity_sha256"):
            raise ResultsError(f"{context} was judged by a different runner identity")
        if row.get("runner_name") != runner.get("name"):
            raise ResultsError(f"{context} was judged under a different runner name")
        if case_id in seen:
            raise ResultsError(f"{context} duplicates case {case_id}")
        seen.add(case_id)
        verdict = row.get("verdict")
        if verdict not in VERDICTS:
            raise ResultsError(f"{context} carries unknown verdict {verdict!r}")
        kind_scores = row.get("kind_scores")
        if not isinstance(kind_scores, dict) or set(kind_scores) != set(DEFECT_KINDS):
            raise ResultsError(f"{context} kind_scores do not cover the defect kinds")
        if verdict == VERDICT_ERROR:
            if not row.get("error"):
                raise ResultsError(f"{context} is an error verdict without an error")
            if row.get("verdict_score") is not None or any(
                v is not None for v in kind_scores.values()
            ):
                raise ResultsError(f"{context} is an error verdict carrying scores")
        else:
            if row.get("error"):
                raise ResultsError(f"{context} carries an error on a non-error verdict")
            if not _unit(row.get("verdict_score")):
                raise ResultsError(
                    f"{context} verdict_score is missing or outside [0, 1]"
                )
            channels = row.get("kind_score_channels") or {}
            for kind in DEFECT_KINDS:
                if not _unit(kind_scores.get(kind)):
                    raise ResultsError(
                        f"{context} kind score for {kind} is missing or outside [0, 1]"
                    )
                if channels.get(kind) not in CHANNELS:
                    raise ResultsError(
                        f"{context} kind score channel for {kind} is invalid"
                    )
        if not _nonneg_int(row.get("latency_ms")):
            raise ResultsError(f"{context} latency_ms is not a non-negative integer")
        tokens = row.get("tokens")
        if not isinstance(tokens, dict):
            raise ResultsError(f"{context} carries no tokens object")
        for name in ("input", "output"):
            value = tokens.get(name)
            if value is not None and not _nonneg_int(value):
                raise ResultsError(
                    f"{context} {name} tokens are not a non-negative integer"
                )
        cost = row.get("cost_usd")
        if cost is not None and (
            not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0
        ):
            raise ResultsError(f"{context} cost_usd is not a non-negative number")
    coverage = payload["coverage"]
    if not isinstance(coverage, dict):
        raise ResultsError(f"results {path} carry malformed coverage")
    errors = sum(1 for row in rows if row.get("error"))
    derived = {
        "complete": bool(identities) and len(rows) == len(identities) and errors == 0,
        "expected": len(identities),
        "scored": len(rows) - errors,
        "errors": errors,
        "results_sha256": canonical_json_sha256(rows),
    }
    for key, value in derived.items():
        if coverage.get(key) != value:
            raise ResultsError(
                f"results {path} coverage.{key}={coverage.get(key)!r} but the rows "
                f"say {value!r}"
            )
    return payload
