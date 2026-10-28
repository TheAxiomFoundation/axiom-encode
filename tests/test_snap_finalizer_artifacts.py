from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.environment_assertions import (
    Names,
    environment_variable,
)

ROOT = Path(__file__).parents[1]
RUN_ID = 123


def _artifact(attempt: int | None, *, expired: bool = False) -> dict:
    suffix = "" if attempt is None else f"-{attempt}"
    return {
        "name": f"targeted-reencode-{RUN_ID}{suffix}",
        "expired": expired,
        "archive_download_url": f"https://example.test/artifacts/{attempt}",
        "workflow_run": {"id": RUN_ID},
    }


def _lookup(
    root: Path,
    *,
    pages: list[list[dict]],
    target_attempt: object = 1,
    run_attempt: int = 1,
) -> subprocess.CompletedProcess[str]:
    """Execute the workflow's real lookup with local, read-only GitHub fixtures."""
    if shutil.which("jq") is None:
        pytest.skip("jq is required to exercise the workflow artifact lookup")
    root.mkdir()
    fixtures = root / "fixtures"
    fixtures.mkdir()
    (fixtures / "artifacts.json").write_text(
        json.dumps([{"artifacts": page} for page in pages])
    )
    (fixtures / "run.json").write_text(
        json.dumps({"id": RUN_ID, "run_attempt": run_attempt})
    )
    (fixtures / "jobs.json").write_text(json.dumps({"jobs": []}))
    binaries = root / "bin"
    binaries.mkdir()
    gh = binaries / "gh"
    gh.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        '  *"/artifacts?"*)\n'
        '    case "$*" in\n'
        '      *"--slurp"*) cat "$FIXTURE_ROOT/artifacts.json" ;;\n'
        '      *) jq ".[0]" "$FIXTURE_ROOT/artifacts.json" ;;\n'
        "    esac ;;\n"
        '  *"/jobs?"*) cat "$FIXTURE_ROOT/jobs.json" ;;\n'
        '  *) cat "$FIXTURE_ROOT/run.json" ;;\n'
        "esac\n"
    )
    gh.chmod(0o755)
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/finalize-signed-snap-queue.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["finalize"]["steps"]
        if step.get("name") == "Fetch live finalization evidence"
    )
    lookup = command[
        command.index('item_id="$(jq') : command.index('gh api "$archive_url"')
    ]
    target = {
        "id": "ut-0001",
        "target_run_url": (
            f"https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/{RUN_ID}"
        ),
    }
    if target_attempt is not None:
        target["target_run_attempt"] = target_attempt
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + lookup + 'printf "%s" "$archive_url"'],
        cwd=ROOT,
        env=Names(
            {
                **os.environ,
                "FIXTURE_ROOT": str(fixtures),
                "GITHUB_REPOSITORY": "TheAxiomFoundation/axiom-encode",
                "PATH": str(binaries) + os.pathsep + environment_variable("PATH"),
                "RUNNER_TEMP": str(root),
                "api_version": "2026-03-10",
                "target": json.dumps(target),
            }
        ),
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_target_lookup_matches_uploader_and_keeps_identified_attempt(
    tmp_path: Path,
) -> None:
    uploader = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    upload = next(
        step
        for step in uploader["jobs"]["encode"]["steps"]
        if step.get("with", {}).get("name")
        == "targeted-reencode-${{ github.run_id }}-${{ github.run_attempt }}"
    )
    uploaded_name = (
        upload["with"]["name"]
        .replace("${{ github.run_id }}", str(RUN_ID))
        .replace("${{ github.run_attempt }}", "1")
    )
    artifact = _artifact(1)
    assert artifact["name"] == uploaded_name
    for run_attempt in (1, 2):
        result = _lookup(
            tmp_path / f"run-{run_attempt}",
            pages=[[artifact]],
            run_attempt=run_attempt,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == artifact["archive_download_url"]


def test_target_lookup_rejects_missing_or_mismatched_attempt(tmp_path: Path) -> None:
    # A legacy artifact must never rescue an unidentified or missing attempt.
    cases = [
        (None, [_artifact(None)]),
        (2, [_artifact(None)]),
        ("1", [_artifact(None)]),
        (True, [_artifact(None)]),
        (0, [_artifact(None)]),
        (1, []),
        (1, [_artifact(None)]),
        (1, [_artifact(2)]),
        (1, [_artifact(1, expired=True)]),
        (None, [_artifact(1)]),
        (2, [_artifact(2)]),
    ]
    for index, (attempt, artifacts) in enumerate(cases):
        result = _lookup(
            tmp_path / f"case-{index}", pages=[artifacts], target_attempt=attempt
        )
        assert result.returncode != 0, (attempt, artifacts, result.stdout)


def test_target_lookup_rejects_competing_uploads_across_pages(tmp_path: Path) -> None:
    cases = [
        # Old lookup accepts the legacy name without noticing other attempts.
        [[_artifact(None)], [_artifact(1), _artifact(2)]],
        [[_artifact(1)], [_artifact(2)]],
        [[_artifact(1)], [_artifact(2, expired=True)]],
        [[_artifact(1)], [_artifact(1)]],
        [[_artifact(1)], [_artifact(None)]],
    ]
    for index, pages in enumerate(cases):
        result = _lookup(tmp_path / f"case-{index}", pages=pages)
        assert result.returncode != 0, (pages, result.stdout)
