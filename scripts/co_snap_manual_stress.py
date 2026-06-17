#!/usr/bin/env python3
"""Stress-run Axiom encoding across the Colorado SNAP rule manual.

This is intentionally a local harness, not an installer. It calls the public
`axiom-encode encode` CLI for each corpus provision, keeps generated artifacts
under a scratch output root, and records deterministic compile/CI results without
running the slower LLM reviewer tier.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from axiom_encode.harness.evals import (
    _rulespec_validation_target,
    _validation_policy_repo_root,
)
from axiom_encode.harness.validator_pipeline import ValidatorPipeline

DEFAULT_CORPUS = (
    Path.home()
    / "TheAxiomFoundation/axiom-corpus/data/corpus/provisions/us-co/regulation/"
    / "2026-04-29-10-ccr-2506-1.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/axiom-co-snap-manual-stress/bulk")
DEFAULT_AXIOM_CORPUS = Path.home() / "TheAxiomFoundation/axiom-corpus"
DEFAULT_RULES_ENGINE = Path.home() / "TheAxiomFoundation/axiom-rules-engine"
DEFAULT_POLICY_REPO = Path.home() / "TheAxiomFoundation/rulespec-us-co"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--axiom-corpus", type=Path, default=DEFAULT_AXIOM_CORPUS)
    parser.add_argument("--axiom-rules-engine", type=Path, default=DEFAULT_RULES_ENGINE)
    parser.add_argument("--policy-repo", type=Path, default=DEFAULT_POLICY_REPO)
    parser.add_argument("--backend", choices=["openai", "codex", "claude"], default="openai")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--mode", choices=["cold", "repo-augmented"], default="repo-augmented")
    parser.add_argument("--timeout-seconds", type=int, default=210)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--filter-regex")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_provisions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        body = str(payload.get("body") or "")
        if not body:
            continue
        records.append(
            {
                "line": line_number,
                "citation_label": payload.get("citation_label"),
                "citation_path": payload.get("citation_path"),
                "heading": payload.get("heading"),
                "body_length": len(body),
            }
        )
    return records


def runner_name(backend: str, model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", f"{backend}-{model}")


def expected_rulespec_path(output_root: Path, runner: str, citation_path: str) -> Path:
    parts = citation_path.split("/")
    if len(parts) < 4:
        return output_root / runner / f"{safe_slug(citation_path)}.yaml"
    jurisdiction, document_class, *rest = parts
    del jurisdiction
    class_dir = {
        "regulation": "regulations",
        "statute": "statutes",
        "policy": "policies",
        "manual": "manuals",
    }.get(document_class, f"{document_class}s")
    if not rest:
        return output_root / runner / class_dir / "index.yaml"
    return output_root / runner / class_dir / Path(*rest[:-1]) / f"{rest[-1]}.yaml"


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    env: dict[str, str],
    expected_artifact: Path,
) -> tuple[int | None, bool, bool, float, str, str]:
    start = time.time()
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    terminated_after_artifact = False
    artifact_snapshot: tuple[int, int] | None = None
    artifact_seen_since: float | None = None
    try:
        while process.poll() is None:
            if time.time() - start > timeout_seconds:
                timed_out = True
                break
            if expected_artifact.exists() and expected_artifact.stat().st_size > 0:
                stat = expected_artifact.stat()
                snapshot = (stat.st_size, stat.st_mtime_ns)
                if snapshot == artifact_snapshot:
                    artifact_seen_since = artifact_seen_since or time.time()
                    if time.time() - artifact_seen_since >= 1:
                        terminated_after_artifact = True
                        break
                else:
                    artifact_snapshot = snapshot
                    artifact_seen_since = time.time()
            time.sleep(0.5)
        if timed_out or terminated_after_artifact:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        raise
    return (
        process.returncode,
        timed_out,
        terminated_after_artifact,
        time.time() - start,
        stdout,
        stderr,
    )


def tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def validate_generated(
    rules_file: Path,
    *,
    policy_repo: Path,
    axiom_rules_engine: Path,
) -> dict[str, Any]:
    try:
        with _rulespec_validation_target(rules_file, policy_repo) as validation_file:
            validation_policy_repo = _validation_policy_repo_root(
                validation_file, policy_repo
            )
            pipeline = ValidatorPipeline(
                policy_repo_path=validation_policy_repo,
                axiom_rules_path=axiom_rules_engine,
                enable_oracles=False,
                require_policy_proofs=True,
            )
            compile_result = pipeline._run_compile_check(validation_file)
            ci_result = pipeline._run_ci(validation_file)
    except Exception as exc:
        return {
            "compile_pass": False,
            "ci_pass": False,
            "validation_error": str(exc),
        }
    return {
        "compile_pass": compile_result.passed,
        "compile_error": compile_result.error,
        "compile_issues": compile_result.issues,
        "ci_pass": ci_result.passed,
        "ci_error": ci_result.error,
        "ci_issues": ci_result.issues,
    }


def existing_completed(results_path: Path) -> set[str]:
    completed: set[str] = set()
    if not results_path.exists():
        return completed
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        with contextlib.suppress(json.JSONDecodeError):
            payload = json.loads(line)
            citation_path = payload.get("citation_path")
            if citation_path:
                completed.add(str(citation_path))
    return completed


def main() -> int:
    args = parse_args()
    provisions = load_provisions(args.manifest)
    if args.filter_regex:
        pattern = re.compile(args.filter_regex)
        provisions = [
            item
            for item in provisions
            if pattern.search(str(item["citation_path"]))
            or pattern.search(str(item.get("heading") or ""))
        ]
    provisions = provisions[args.start_index :]
    if args.limit is not None:
        provisions = provisions[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_root / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_root / "results.jsonl"
    completed = existing_completed(results_path) if args.resume else set()
    runner = runner_name(args.backend, args.model)

    env = os.environ.copy()
    env.setdefault("AXIOM_ENCODE_CODEX_TIMEOUT_SECONDS", "90")
    env.setdefault("AXIOM_ENCODE_CODEX_IDLE_TIMEOUT_SECONDS", "35")
    env.setdefault("AXIOM_ENCODE_CODEX_LONG_TIMEOUT_SECONDS", "150")
    env.setdefault("AXIOM_ENCODE_CODEX_LONG_IDLE_TIMEOUT_SECONDS", "45")

    print(
        json.dumps(
            {
                "event": "stress_start",
                "count": len(provisions),
                "output_root": str(args.output_root),
                "backend": args.backend,
                "model": args.model,
                "runner": runner,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for index, item in enumerate(provisions, start=args.start_index):
        citation_path = str(item["citation_path"])
        if citation_path in completed:
            continue
        expected = expected_rulespec_path(args.output_root, runner, citation_path)
        slug = safe_slug(citation_path)
        stdout_log = logs_dir / f"{index:03d}-{slug}.stdout.log"
        stderr_log = logs_dir / f"{index:03d}-{slug}.stderr.log"
        cmd = [
            "uv",
            "run",
            "axiom-encode",
            "encode",
            citation_path,
            "--backend",
            args.backend,
            "--model",
            args.model,
            "--output",
            str(args.output_root),
            "--corpus-path",
            str(args.axiom_corpus),
            "--axiom-rules-engine-path",
            str(args.axiom_rules_engine),
            "--policy-repo-path",
            str(args.policy_repo),
            "--mode",
            args.mode,
            "--no-sync",
        ]
        print(
            json.dumps(
                {
                    "event": "provision_start",
                    "index": index,
                    "citation_path": citation_path,
                    "expected_file": str(expected),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        (
            returncode,
            timed_out,
            terminated_after_artifact,
            duration,
            stdout,
            stderr,
        ) = run_command(
            cmd,
            cwd=Path.cwd(),
            timeout_seconds=args.timeout_seconds,
            env=env,
            expected_artifact=expected,
        )
        stdout_log.write_text(stdout)
        stderr_log.write_text(stderr)
        generated = expected.exists()
        test_file = expected.with_name(f"{expected.stem}.test.yaml")
        validation = (
            validate_generated(
                expected,
                policy_repo=args.policy_repo,
                axiom_rules_engine=args.axiom_rules_engine,
            )
            if generated
            else {}
        )
        result = {
            **item,
            "index": index,
            "backend": args.backend,
            "model": args.model,
            "returncode": returncode,
            "timed_out": timed_out,
            "terminated_after_artifact": terminated_after_artifact,
            "duration_seconds": round(duration, 3),
            "generated": generated,
            "generated_file": str(expected) if generated else None,
            "generated_test": test_file.exists(),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "stdout_tail": tail(stdout),
            "stderr_tail": tail(stderr),
            **validation,
        }
        with results_path.open("a") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)

    print(
        json.dumps({"event": "stress_done", "results": str(results_path)}, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
