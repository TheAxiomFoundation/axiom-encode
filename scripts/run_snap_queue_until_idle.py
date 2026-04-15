#!/usr/bin/env python3
"""Run the local SNAP queue until idle.

This is the canonical repo-backed copy of the local queue runner that powers the
event-driven Codex automation. The live automation may invoke a thin wrapper from
`~/.codex`, but logic changes should land here first.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


CODEX_HOME = Path(
    os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))
).expanduser().resolve()
AUTORAC_ROOT = Path(
    os.environ.get("AUTORAC_ROOT", str(Path(__file__).resolve().parents[1]))
).expanduser().resolve()
AUTOMATION_DIR = Path(
    os.environ.get(
        "AUTORAC_SNAP_AUTOMATION_DIR",
        str(CODEX_HOME / "automations" / "hourly-snap-encode"),
    )
).expanduser().resolve()
QUEUE_PATH = AUTOMATION_DIR / "queue.json"
MEMORY_PATH = AUTOMATION_DIR / "memory.md"
RUN_LEDGER_PATH = AUTOMATION_DIR / "run_ledger.ndjson"
LOCK_PATH = AUTOMATION_DIR / "runner.lock"
TMP_ROOT = Path(
    os.environ.get("AUTORAC_TMP_ROOT", str(Path.home() / "tmp"))
).expanduser().resolve()
DEFAULT_ARCHIVE_ROOT = Path(
    os.environ.get(
        "AUTORAC_EVAL_ARCHIVE_ROOT",
        str(AUTORAC_ROOT / "artifacts" / "eval-suites"),
    )
).expanduser().resolve()
DEFAULT_ATLAS_ARCH_ROOT = Path.home() / ".arch"
BENCHMARK_GLOB = "us_snap_*_refresh.yaml"
POLICYENGINE_CANDIDATES = [
    Path.home() / "worktrees" / "policyengine-us-main-view",
    Path.home() / "PolicyEngine" / "policyengine-us",
]
REQUIRED_PATH_ENTRIES = [
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    str(Path.home() / ".bun" / "bin"),
    str(Path.home() / ".local" / "bin"),
    str(Path.home() / "bin"),
]
UV_CANDIDATES = [
    Path("/opt/homebrew/bin/uv"),
    Path("/usr/local/bin/uv"),
]
POLICYENGINE_US_PYTHON_CANDIDATES = [
    Path.home() / "worktrees" / "policyengine-us-main-view" / ".venv" / "bin" / "python",
    Path.home() / "PolicyEngine" / "policyengine-us" / ".venv" / "bin" / "python",
]
WORKSPACES = [
    AUTORAC_ROOT,
    *sorted(path for path in AUTORAC_ROOT.parent.glob("rac-*") if path.is_dir()),
]
RETRYABLE_PATTERNS = (
    "usage limit",
    "rate limit",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "temporary failure",
    "dns",
    "could not resolve",
    "connection reset",
    "connection aborted",
    "connection refused",
    "502",
    "503",
    "504",
)


@dataclass
class ActiveState:
    status: str
    action: str
    manifest: str = "none"
    target: str = "none"
    source_repo: str = "none"
    source_file: str = "none"
    output_dir: str = "none"
    archive_path: str = "none"
    started_at: str = "none"
    finished_at: str = "none"
    progress: str = "none"
    outcome: str = "none"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def now_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "snap-task"


def load_queue(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_queue(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(paths: list[Path | None]) -> str | None:
    existing = [
        path.resolve()
        for path in paths
        if path is not None and path.exists() and path.is_file()
    ]
    if not existing:
        return None
    if len(existing) == 1:
        return sha256_file(existing[0])

    digest = hashlib.sha256()
    for resolved in existing:
        digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        file_digest = sha256_file(resolved)
        if file_digest is not None:
            digest.update(file_digest.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def atlas_archive_root() -> Path:
    override = os.environ.get("AUTORAC_ARCH_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_ATLAS_ARCH_ROOT.resolve()


def git_head(repo: Path | None) -> str | None:
    if repo is None or not repo.exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_root_for_path(path: Path | None) -> Path | None:
    if path is None or not path.exists():
        return None
    target = path if path.is_dir() else path.parent
    result = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()) if result.stdout.strip() else None


def detect_policyengine_root() -> Path | None:
    for candidate in POLICYENGINE_CANDIDATES:
        if (candidate / ".git").exists():
            return candidate
    return None


def append_event(data: dict[str, Any], message: str) -> None:
    log = data.setdefault("event_log", [])
    log.append({"timestamp": now_utc(), "message": message})
    if len(log) > 50:
        del log[:-50]


def build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = [entry for entry in env.get("PATH", "").split(":") if entry]
    merged: list[str] = []
    for entry in [*REQUIRED_PATH_ENTRIES, *existing]:
        if entry not in merged:
            merged.append(entry)
    env["PATH"] = ":".join(merged)
    pe_python = resolve_policyengine_us_python()
    if pe_python:
        env["AUTORAC_POLICYENGINE_US_PYTHON"] = pe_python
    return env


def resolve_uv_bin() -> str:
    uv_from_path = shutil.which("uv")
    if uv_from_path:
        return uv_from_path
    for candidate in UV_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return "uv"


def resolve_policyengine_us_python() -> str | None:
    for candidate in POLICYENGINE_US_PYTHON_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def infer_repo(source_file: str | None) -> str:
    if not source_file:
        return "none"
    path = Path(source_file)
    git_root = git_root_for_path(path)
    if git_root and git_root.name.startswith("rac-"):
        return git_root.name
    for workspace in WORKSPACES:
        try:
            path.resolve().relative_to(workspace.resolve())
            return workspace.name
        except ValueError:
            continue
    return path.parents[0].name if path.exists() else "unknown"


def resolve_manifest_source_file(manifest_path: Path, source_file: str | None) -> Path | None:
    if not source_file:
        return None
    candidate = Path(source_file)
    if not candidate.is_absolute():
        candidate = (manifest_path.parent / candidate).resolve()
    return candidate


def resolve_companion_metadata_file(source_file: Path | None) -> Path | None:
    if source_file is None:
        return None
    candidates = [
        source_file.with_name(f"{source_file.stem}.meta.yaml"),
        source_file.with_name(f"{source_file.stem}.meta.yml"),
        source_file.with_suffix(source_file.suffix + ".meta.yaml"),
        source_file.with_suffix(source_file.suffix + ".meta.yml"),
    ]
    return next((path for path in candidates if path.exists()), None)


def resolve_metadata_akn_file(metadata_file: Path | None) -> Path | None:
    if metadata_file is None or not metadata_file.exists():
        return None
    try:
        payload = yaml.safe_load(metadata_file.read_text()) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(payload, dict):
        return None
    backing = payload.get("source_backing")
    if not isinstance(backing, dict):
        return None
    arch_path = backing.get("arch_path")
    if arch_path is not None:
        path = Path(str(arch_path)).expanduser()
        if not path.is_absolute():
            path = atlas_archive_root() / path
        return path.resolve()
    akn_file = backing.get("akn_file")
    if akn_file is None:
        return None
    path = Path(str(akn_file)).expanduser()
    if not path.is_absolute():
        path = metadata_file.parent / path
    return path.resolve()


def resolve_manifest_case_source_inputs(
    manifest_path: Path,
    case: dict[str, Any],
) -> tuple[Path | None, list[Path]]:
    kind = str(case.get("kind") or "")
    if kind == "source":
        source_file = resolve_manifest_source_file(manifest_path, case.get("source_file"))
        metadata_file = resolve_companion_metadata_file(source_file)
        archive_akn_file = resolve_metadata_akn_file(metadata_file)
        source_inputs = [
            path for path in (source_file, metadata_file, archive_akn_file) if path is not None
        ]
        return source_file, source_inputs
    if kind == "akn_section":
        metadata_file = resolve_manifest_source_file(
            manifest_path, case.get("metadata_file")
        )
        akn_file = resolve_manifest_source_file(manifest_path, case.get("akn_file"))
        archive_akn_file = resolve_metadata_akn_file(metadata_file)
        primary = metadata_file or akn_file or archive_akn_file
        source_inputs = [
            path
            for path in (metadata_file, akn_file, archive_akn_file)
            if path is not None
        ]
        return primary, source_inputs
    return None, []


def iter_manifest_queue_candidates() -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for manifest_path in sorted((AUTORAC_ROOT / "benchmarks").glob(BENCHMARK_GLOB)):
        try:
            manifest = yaml.safe_load(manifest_path.read_text()) or {}
        except yaml.YAMLError:
            continue
        cases = manifest.get("cases") or []
        if not isinstance(cases, list):
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            name = case.get("name")
            source_file, source_inputs = resolve_manifest_case_source_inputs(
                manifest_path, case
            )
            if not name or source_file is None or not source_inputs:
                continue
            source_repo = infer_repo(str(source_file))
            if source_repo in {"none", "unknown", "rac-us"}:
                continue
            candidates.append(
                {
                    "name": str(name),
                    "manifest": str(manifest_path),
                    "source_file": str(source_file),
                    "source_inputs": [str(path) for path in source_inputs],
                    "source_repo": source_repo,
                }
            )
    return candidates


def sync_queue_with_manifests(
    data: dict[str, Any],
) -> tuple[bool, list[str], list[str], list[str]]:
    items = data.setdefault("items", [])
    candidates = iter_manifest_queue_candidates()
    candidate_names = {candidate["name"] for candidate in candidates}
    by_name = {
        item.get("name"): item
        for item in items
        if isinstance(item, dict) and item.get("name")
    }
    added: list[str] = []
    retired: list[str] = []
    refreshed: list[str] = []
    changed = False
    for candidate in candidates:
        manifest_sha = sha256_file(Path(candidate["manifest"]))
        source_sha = sha256_paths(
            [Path(path) for path in candidate.get("source_inputs", [])]
        )
        existing = by_name.get(candidate["name"])
        if existing is None:
            items.append(
                {
                    "name": candidate["name"],
                    "status": "queued",
                    "manifest": candidate["manifest"],
                    "source_file": candidate["source_file"],
                    "source_inputs": candidate.get("source_inputs", []),
                    "manifest_sha256": manifest_sha,
                    "source_sha256": source_sha,
                    "note": "queued from manifest sync",
                }
            )
            added.append(candidate["name"])
            changed = True
            continue
        previous_manifest_sha = existing.get("manifest_sha256")
        previous_source_sha = existing.get("source_sha256")
        for key in ("manifest", "source_file", "source_inputs"):
            if existing.get(key) != candidate[key]:
                existing[key] = candidate[key]
                changed = True
        if existing.get("manifest_sha256") != manifest_sha:
            existing["manifest_sha256"] = manifest_sha
            changed = True
        if existing.get("source_sha256") != source_sha:
            existing["source_sha256"] = source_sha
            changed = True
        if (
            previous_manifest_sha is not None
            and previous_source_sha is not None
            and (
                previous_manifest_sha != manifest_sha
                or previous_source_sha != source_sha
            )
            and existing.get("status") in {"done", "blocked", "retryable"}
        ):
            existing["status"] = "queued"
            existing["started_at"] = None
            existing["finished_at"] = None
            existing["output_dir"] = None
            existing["archive_path"] = None
            existing["note"] = "requeued after manifest/source change"
            refreshed.append(candidate["name"])
            changed = True
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        manifest = item.get("manifest")
        status = item.get("status")
        if not name or name in candidate_names or not manifest:
            continue
        if status in {"running", "done", "retired"}:
            continue
        if Path(manifest).exists():
            continue
        item["status"] = "retired"
        item["finished_at"] = now_utc()
        item["note"] = "manifest removed from autorac benchmarks; retired from queue"
        retired.append(str(name))
        changed = True
    return changed, added, retired, refreshed


def find_active_eval_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "-Ao", "pid=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    active: list[str] = []
    for line in lines:
        if "run_queue_until_idle.py" in line:
            continue
        command = line.split(maxsplit=1)[1] if " " in line else ""
        if re.search(r"(^|\\s)uv run autorac eval-suite(\\s|$)", command) or re.search(
            r"(^|\\s)\\S+/autorac eval-suite(\\s|$)",
            command,
        ):
            active.append(line)
    return active


def build_output_dir(name: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    return TMP_ROOT / f"autorac-{slugify(name)}-{timestamp}"


def classify_status(returncode: int, summary: dict[str, Any] | None, results: dict[str, Any] | None) -> tuple[str, str]:
    if summary and summary.get("all_ready"):
        return "done", "closed fully ready"

    combined_parts: list[str] = []
    if returncode != 0:
        combined_parts.append(f"eval-suite exited with code {returncode}")
    if results:
        for result in results.get("results", []):
            error = result.get("error")
            if error:
                combined_parts.append(str(error))
            metrics = result.get("metrics") or {}
            combined_parts.extend(metrics.get("compile_issues") or [])
            combined_parts.extend(metrics.get("ci_issues") or [])
            combined_parts.extend(metrics.get("generalist_review_issues") or [])
            combined_parts.extend(metrics.get("policyengine_issues") or [])
    combined = " | ".join(part for part in combined_parts if part).lower()
    if any(pattern in combined for pattern in RETRYABLE_PATTERNS):
        return "retryable", combined_parts[0] if combined_parts else "retryable infrastructure failure"
    return "blocked", combined_parts[0] if combined_parts else "not ready"


def run_eval_item(item: dict[str, Any], backend: str, reviewer_cli: str, output_dir: Path) -> int:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    env = build_subprocess_env()
    env["AUTORAC_REVIEWER_CLI"] = reviewer_cli
    cmd = [
        resolve_uv_bin(),
        "run",
        "autorac",
        "eval-suite",
        item["manifest"],
        "--gpt-backend",
        backend,
        "--output",
        str(output_dir),
    ]
    try:
        result = subprocess.run(cmd, cwd=AUTORAC_ROOT, env=env, check=False)
    except FileNotFoundError:
        return 127
    return result.returncode


def archive_eval(output_dir: Path) -> Path | None:
    cmd = [
        resolve_uv_bin(),
        "run",
        "autorac",
        "eval-suite-archive",
        str(output_dir),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=AUTORAC_ROOT,
            env=build_subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"Archived eval suite to (.+)", result.stdout)
    return Path(match.group(1).strip()) if match else None


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def load_run_ledger_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id:
            ids.add(run_id)
    return ids


def compute_run_id(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("name") or "unknown"),
        str(item.get("started_at") or "none"),
        str(item.get("output_dir") or item.get("archive_path") or "none"),
    ]
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:24]


def classify_failure_class(
    status: str,
    returncode: int | None,
    summary: dict[str, Any] | None,
    results: dict[str, Any] | None,
    note: str | None,
) -> str:
    if summary and summary.get("all_ready"):
        return "ready"
    lowered = " | ".join(
        str(part).lower()
        for part in [note, returncode]
        if part not in (None, "")
    )
    if "usage limit" in lowered or "rate limit" in lowered:
        return "retryable_quota"
    if any(token in lowered for token in ("dns", "timeout", "temporarily unavailable", "connection")):
        return "retryable_transport"
    for result in (results or {}).get("results", []):
        error = str(result.get("error") or "").lower()
        metrics = result.get("metrics") or {}
        if "no rac content returned" in error:
            return "generation_no_content"
        if metrics.get("compile_pass") is False:
            return "compile"
        if metrics.get("ci_pass") is False:
            return "ci"
        if metrics.get("generalist_review_pass") is False:
            return "review"
        if metrics.get("policyengine_pass") is False:
            return "policyengine"
    if status == "retryable":
        return "retryable_unknown"
    if status == "blocked":
        return "blocked_unknown"
    return "completed_unknown"


def build_run_record(
    item: dict[str, Any],
    *,
    backend: str,
    reviewer_cli: str,
    returncode: int | None,
    summary: dict[str, Any] | None,
    results: dict[str, Any] | None,
    archive_path: Path | None,
    status: str,
    note: str,
    backfilled: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(item["manifest"]).resolve() if item.get("manifest") else None
    source_path = Path(item["source_file"]).resolve() if item.get("source_file") else None
    source_inputs = [
        Path(path).resolve() for path in (item.get("source_inputs") or []) if path
    ]
    output_dir = Path(item["output_dir"]).resolve() if item.get("output_dir") else None
    policy_repo_root = git_root_for_path(source_path)
    policyengine_root = detect_policyengine_root()
    effective_runner = None
    readiness_block = None
    if summary:
        effective_runners = (summary.get("manifest") or {}).get("effective_runners") or []
        effective_runner = effective_runners[0] if effective_runners else None
        readiness = summary.get("readiness") or {}
        if effective_runner and isinstance(readiness, dict):
            readiness_block = readiness.get(effective_runner)
        elif isinstance(readiness, dict) and len(readiness) == 1:
            readiness_block = next(iter(readiness.values()))

    result_rows = (results or {}).get("results") or []
    first_result = result_rows[0] if result_rows else {}
    metrics = first_result.get("metrics") or {}
    token_fields = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    estimated_cost_usd = 0.0
    actual_cost_usd: float | None = None
    duration_ms = 0
    issue_counts = {
        "compile_issue_count": 0,
        "ci_issue_count": 0,
        "generalist_review_issue_count": 0,
        "policyengine_issue_count": 0,
    }
    errors: list[str] = []
    for row in result_rows:
        for key in token_fields:
            token_fields[key] += int(row.get(key) or 0)
        estimated_cost_usd += float(row.get("estimated_cost_usd") or 0.0)
        if row.get("actual_cost_usd") is not None:
            actual_cost_usd = (actual_cost_usd or 0.0) + float(row["actual_cost_usd"])
        duration_ms += int(row.get("duration_ms") or 0)
        row_metrics = row.get("metrics") or {}
        issue_counts["compile_issue_count"] += len(row_metrics.get("compile_issues") or [])
        issue_counts["ci_issue_count"] += len(row_metrics.get("ci_issues") or [])
        issue_counts["generalist_review_issue_count"] += len(row_metrics.get("generalist_review_issues") or [])
        issue_counts["policyengine_issue_count"] += len(row_metrics.get("policyengine_issues") or [])
        if row.get("error"):
            errors.append(str(row["error"]))

    return {
        "schema_version": 1,
        "recorded_at": now_utc(),
        "run_id": compute_run_id(item),
        "target": item.get("name"),
        "status": status,
        "failure_class": classify_failure_class(status, returncode, summary, results, note),
        "backfilled_from_queue": backfilled,
        "backend": backend,
        "effective_runner": effective_runner,
        "reviewer_cli": reviewer_cli,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_sha256": sha256_file(manifest_path),
        "source_file": str(source_path) if source_path else None,
        "source_sha256": sha256_paths(source_inputs or [source_path]),
        "source_repo": infer_repo(item.get("source_file")),
        "autorac_sha": git_head(AUTORAC_ROOT),
        "policy_repo_sha": git_head(policy_repo_root),
        "policyengine_sha": git_head(policyengine_root),
        "started_at": item.get("started_at"),
        "finished_at": item.get("finished_at"),
        "returncode": returncode,
        "duration_ms": duration_ms or None,
        "output_dir": str(output_dir) if output_dir else None,
        "archive_path": str(archive_path.resolve()) if archive_path else item.get("archive_path"),
        "note": note,
        "all_ready": bool(summary.get("all_ready")) if summary else False,
        "readiness": readiness_block,
        "result_count": len(result_rows),
        "first_error": errors[0] if errors else None,
        "generation_prompt_sha256": first_result.get("generation_prompt_sha256"),
        "generalist_review_prompt_sha256": metrics.get(
            "generalist_review_prompt_sha256"
        ),
        "metrics": {
            "compile_pass": metrics.get("compile_pass"),
            "ci_pass": metrics.get("ci_pass"),
            "generalist_review_pass": metrics.get("generalist_review_pass"),
            "policyengine_pass": metrics.get("policyengine_pass"),
            "success": first_result.get("success"),
            "zero_ungrounded": (metrics.get("ungrounded_numeric_count") == 0) if "ungrounded_numeric_count" in metrics else None,
            "generalist_review_score": metrics.get("generalist_review_score"),
            "policyengine_score": metrics.get("policyengine_score"),
            "estimated_cost_usd": estimated_cost_usd if result_rows else None,
            "actual_cost_usd": actual_cost_usd,
            **token_fields,
            **issue_counts,
        },
    }


def append_run_record(path: Path, record: dict[str, Any], known_ids: set[str]) -> None:
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id or run_id in known_ids:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    known_ids.add(run_id)


def backfill_run_ledger(data: dict[str, Any], known_ids: set[str]) -> None:
    for item in data.get("items", []):
        if item.get("status") not in {"done", "blocked", "retryable"}:
            continue
        if not item.get("output_dir") and not item.get("archive_path"):
            continue
        output_dir = Path(item["output_dir"]).resolve() if item.get("output_dir") else None
        archive_path = Path(item["archive_path"]).resolve() if item.get("archive_path") else None
        summary = load_json(output_dir / "summary.json") if output_dir else None
        results = load_json(output_dir / "results.json") if output_dir else None
        if summary is None and archive_path:
            summary = load_json(archive_path / "summary.json")
        if results is None and archive_path:
            results = load_json(archive_path / "results.json")
        record = build_run_record(
            item,
            backend=str(data.get("default_backend") or "codex"),
            reviewer_cli=str(data.get("default_reviewer_cli") or "claude"),
            returncode=None,
            summary=summary,
            results=results,
            archive_path=archive_path,
            status=str(item.get("status") or "unknown"),
            note=str(item.get("note") or ""),
            backfilled=True,
        )
        append_run_record(RUN_LEDGER_PATH, record, known_ids)


def render_memory(data: dict[str, Any], active: ActiveState) -> str:
    items = data.get("items", [])
    next_item = next(
        (item["name"] for item in items if item.get("status") in {"queued", "retryable"}),
        "none queued or retryable",
    )
    lines = [
        "# Hourly SNAP Encode Memory",
        "",
        f"Last refreshed: {now_utc()}",
        f"Run ledger: `{RUN_LEDGER_PATH}`",
        "Queue seeding: auto-sync from checked-in `autorac` SNAP refresh manifests on runner wakeup",
        "",
        "## Active SNAP eval",
        "",
        f"- status: {active.status}",
        f"- action: {active.action}",
        f"- manifest: {active.manifest}",
        f"- target: {active.target}",
        f"- source repo: {active.source_repo}",
        f"- source file: {active.source_file}",
        f"- output dir: {active.output_dir}",
        f"- archive path: {active.archive_path}",
        f"- suite started_at: {active.started_at}",
        f"- suite finished_at: {active.finished_at}",
        f"- suite progress: {active.progress}",
        f"- outcome: {active.outcome}",
        "",
        "## Queue state",
        "",
    ]
    for item in items:
        lines.append(f"- `{item['name']}`: {item.get('status', 'unknown')}")
        if item.get("manifest"):
            lines.append(f"  manifest: `{item['manifest']}`")
        if item.get("source_file"):
            lines.append(f"  source file: `{item['source_file']}`")
        if item.get("output_dir"):
            lines.append(f"  output dir: `{item['output_dir']}`")
        if item.get("archive_path"):
            lines.append(f"  archive path: `{item['archive_path']}`")
        if item.get("note"):
            lines.append(f"  note: {item['note']}")
    lines.append(f"- next eligible target: {next_item}")
    lines.extend(["", "## Notes", ""])
    for event in data.get("event_log", [])[-25:]:
        lines.append(f"- {event['timestamp']}: {event['message']}")
    lines.append("")
    return "\n".join(lines)


def write_memory(data: dict[str, Any], active: ActiveState) -> None:
    MEMORY_PATH.write_text(render_memory(data, active))


def reconcile_stale_running_items(data: dict[str, Any], active_processes: list[str]) -> bool:
    if active_processes:
        return False

    changed = False
    for item in data.get("items", []):
        if item.get("status") != "running":
            continue
        item["status"] = "retryable"
        item["finished_at"] = now_utc()
        item["note"] = "runner exited before this queued eval finished; marked retryable for relaunch"
        changed = True
        append_event(
            data,
            f"{item['name']} was left in `running` without a live eval process; marked retryable for relaunch.",
        )
    return changed


def process_queue(queue_path: Path) -> int:
    data = load_queue(queue_path)
    sync_changed, added_items, retired_items, refreshed_items = sync_queue_with_manifests(data)
    if sync_changed:
        save_queue(queue_path, data)
    if added_items:
        append_event(
            data,
            "Queued newly discovered SNAP manifests: " + ", ".join(added_items),
        )
        save_queue(queue_path, data)
    if retired_items:
        append_event(
            data,
            "Retired queue items whose manifests were removed: " + ", ".join(retired_items),
        )
        save_queue(queue_path, data)
    if refreshed_items:
        append_event(
            data,
            "Requeued queue items after manifest/source updates: " + ", ".join(refreshed_items),
        )
        save_queue(queue_path, data)
    backend = str(data.get("default_backend") or "codex")
    reviewer_cli = str(data.get("default_reviewer_cli") or "claude")
    known_run_ids = load_run_ledger_ids(RUN_LEDGER_PATH)
    backfill_run_ledger(data, known_run_ids)

    while True:
        active_processes = find_active_eval_processes()
        if reconcile_stale_running_items(data, active_processes):
            save_queue(queue_path, data)
        if active_processes:
            active = ActiveState(
                status="waiting",
                action="another SNAP-focused `autorac eval-suite` is already running; waiting for it to finish before starting the next queued item",
                outcome=f"external eval already active at {now_local()}",
            )
            write_memory(data, active)
            time.sleep(30)
            data = load_queue(queue_path)
            sync_changed, added_items, retired_items, refreshed_items = sync_queue_with_manifests(data)
            if sync_changed:
                if added_items:
                    append_event(
                        data,
                        "Queued newly discovered SNAP manifests: " + ", ".join(added_items),
                    )
                if retired_items:
                    append_event(
                        data,
                        "Retired queue items whose manifests were removed: " + ", ".join(retired_items),
                    )
                if refreshed_items:
                    append_event(
                        data,
                        "Requeued queue items after manifest/source updates: " + ", ".join(refreshed_items),
                    )
                save_queue(queue_path, data)
            continue

        item = next(
            (candidate for candidate in data.get("items", []) if candidate.get("status") in {"queued", "retryable"}),
            None,
        )
        if item is None:
            active = ActiveState(
                status="idle",
                action="no active SNAP eval at refresh time; the queue has no queued or retryable target to launch",
                outcome=f"queue is idle at {now_local()}",
            )
            write_memory(data, active)
            return 0

        output_dir = build_output_dir(item["name"])
        item["status"] = "running"
        item["started_at"] = now_utc()
        item["output_dir"] = str(output_dir)
        item["archive_path"] = None
        item.pop("finished_at", None)
        item["note"] = f"started with backend `{backend}`"
        save_queue(queue_path, data)
        active = ActiveState(
            status="running",
            action="running the next queued SNAP eval until completion",
            manifest=item.get("manifest", "none"),
            target=item["name"],
            source_repo=infer_repo(item.get("source_file")),
            source_file=item.get("source_file", "none"),
            output_dir=str(output_dir),
            started_at=item["started_at"],
            outcome=f"started at {now_local()} with backend `{backend}`",
        )
        write_memory(data, active)

        returncode = run_eval_item(item, backend=backend, reviewer_cli=reviewer_cli, output_dir=output_dir)
        summary = load_json(output_dir / "summary.json")
        results = load_json(output_dir / "results.json")
        archive_path = archive_eval(output_dir) if (output_dir / "suite-run.json").exists() else None
        new_status, reason = classify_status(returncode, summary, results)
        item["status"] = new_status
        item["output_dir"] = str(output_dir)
        item["archive_path"] = str(archive_path) if archive_path else item.get("archive_path")
        item["finished_at"] = now_utc()
        item["note"] = reason
        append_event(
            data,
            f"{item['name']} finished with status `{new_status}`; output `{output_dir}`"
            + (f" and archive `{archive_path}`" if archive_path else ""),
        )
        record = build_run_record(
            item,
            backend=backend,
            reviewer_cli=reviewer_cli,
            returncode=returncode,
            summary=summary,
            results=results,
            archive_path=archive_path,
            status=new_status,
            note=reason,
        )
        append_run_record(RUN_LEDGER_PATH, record, known_run_ids)
        save_queue(queue_path, data)
        active = ActiveState(
            status="completed",
            action="last queued SNAP eval finished",
            manifest=item.get("manifest", "none"),
            target=item["name"],
            source_repo=infer_repo(item.get("source_file")),
            source_file=item.get("source_file", "none"),
            output_dir=str(output_dir),
            archive_path=str(archive_path) if archive_path else "none",
            started_at=item.get("started_at", "none"),
            finished_at=item["finished_at"],
            progress="1 case complete",
            outcome=f"{item['name']} finished `{new_status}` at {now_local()}",
        )
        write_memory(data, active)

        data = load_queue(queue_path)
        sync_changed, added_items, retired_items, refreshed_items = sync_queue_with_manifests(data)
        if sync_changed:
            if added_items:
                append_event(
                    data,
                    "Queued newly discovered SNAP manifests: " + ", ".join(added_items),
                )
            if retired_items:
                append_event(
                    data,
                    "Retired queue items whose manifests were removed: " + ", ".join(retired_items),
                )
            if refreshed_items:
                append_event(
                    data,
                    "Requeued queue items after manifest/source updates: " + ", ".join(refreshed_items),
                )
            save_queue(queue_path, data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SNAP queue until idle.")
    parser.add_argument("--queue", default=str(QUEUE_PATH))
    args = parser.parse_args()

    queue_path = Path(args.queue).resolve()
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        return process_queue(queue_path)


if __name__ == "__main__":
    sys.exit(main())
