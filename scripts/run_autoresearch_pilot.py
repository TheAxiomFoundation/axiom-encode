#!/usr/bin/env python3
"""Run the autoresearch-style pilot against frozen repair manifests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _default_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("/tmp") / f"autorac-autoresearch-pilot-{timestamp}"


def main() -> int:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from autorac.harness.autoresearch_pilot import (
        autorac_repo_root,
        extract_primary_runner_summary,
        load_suite_summary,
        pilot_editable_paths,
        pilot_manifest_paths,
        program_path,
        score_readiness_summary,
        seed_legislation_cache,
        shared_legislation_cache_root,
        sync_legislation_cache,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run frozen UK repair manifests and emit a single scalar score for "
            "autoresearch-style prompt tuning."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_output_root(),
        help="Directory for per-manifest eval outputs and aggregate pilot report",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Override the frozen manifest set with one or more specific manifests",
    )
    parser.add_argument(
        "--gpt-backend",
        choices=["codex", "openai"],
        default=None,
        help="Override GPT runner backend for local-vs-API eval execution",
    )
    parser.add_argument(
        "--shared-legislation-cache-root",
        type=Path,
        default=None,
        help=(
            "Persistent directory for reusing legislation.gov.uk payloads across "
            "pilot runs. Defaults to AUTORAC_SHARED_LEGISLATION_CACHE or "
            "~/tmp/autorac-shared-legislation-cache."
        ),
    )
    args = parser.parse_args()

    repo_root = autorac_repo_root()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    legislation_cache_root = (
        args.shared_legislation_cache_root.resolve()
        if args.shared_legislation_cache_root is not None
        else shared_legislation_cache_root()
    )

    manifests = (
        [Path(item).resolve() for item in args.manifest]
        if args.manifest
        else pilot_manifest_paths(repo_root)
    )

    env = os.environ.copy()
    pythonpath_entries = [str(repo_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    if args.gpt_backend:
        env["AUTORAC_GPT_BACKEND"] = args.gpt_backend

    report: dict[str, object] = {
        "repo_root": str(repo_root),
        "program": str(program_path(repo_root)),
        "editable_files": [str(path) for path in pilot_editable_paths(repo_root)],
        "output_root": str(output_root),
        "shared_legislation_cache_root": str(legislation_cache_root),
        "manifests": [str(path) for path in manifests],
        "results": [],
    }

    manifest_scores: list[float] = []

    for manifest in manifests:
        run_root = output_root / manifest.stem
        run_root.mkdir(parents=True, exist_ok=True)
        preseeded = seed_legislation_cache(
            run_root,
            shared_root=legislation_cache_root,
        )
        stdout_path = run_root / "autoresearch-pilot.stdout"
        stderr_path = run_root / "autoresearch-pilot.stderr"
        cmd = [
            sys.executable,
            "-m",
            "autorac.cli",
            "eval-suite",
            str(manifest),
            "--output",
            str(run_root),
            "--json",
        ]
        process = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
        )
        stdout_path.write_text(process.stdout)
        stderr_path.write_text(process.stderr)
        synced_cache = sync_legislation_cache(
            run_root,
            shared_root=legislation_cache_root,
        )

        result_record: dict[str, object] = {
            "manifest": str(manifest),
            "run_root": str(run_root),
            "command": cmd,
            "returncode": process.returncode,
            "preseeded_cache_files": preseeded,
            "synced_cache_files": synced_cache,
            "stdout_file": str(stdout_path),
            "stderr_file": str(stderr_path),
        }

        summary_path = run_root / "summary.json"
        if summary_path.exists():
            payload = load_suite_summary(summary_path)
            runner, summary = extract_primary_runner_summary(payload)
            score = score_readiness_summary(summary)
            manifest_scores.append(score)
            result_record.update(
                {
                    "runner": runner,
                    "summary": summary,
                    "score": score,
                }
            )
        else:
            result_record["error"] = "Missing summary.json"

        report["results"].append(result_record)

    aggregate_score = (
        round(sum(manifest_scores) / len(manifest_scores), 6)
        if manifest_scores
        else None
    )
    report["aggregate_score"] = aggregate_score

    report_path = output_root / "autoresearch-pilot-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    if aggregate_score is not None:
        print(f"AUTORESEARCH_SCORE={aggregate_score}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
