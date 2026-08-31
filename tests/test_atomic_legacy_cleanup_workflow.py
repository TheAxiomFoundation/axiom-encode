"""Contract tests for the isolated atomic legacy-cleanup workflow."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/atomic-legacy-cleanup.yml"


def _workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _step(job: str, name: str) -> dict:
    return next(
        step
        for step in _workflow()["jobs"][job]["steps"]
        if step.get("name") == name
    )


def _job_text(job: str) -> str:
    return yaml.safe_dump(_workflow()["jobs"][job], sort_keys=True)


def test_workflow_is_manual_main_only_with_read_scoped_contents() -> None:
    workflow = _workflow()

    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"actions": "read", "contents": "read"}
    assert set(workflow["jobs"]) == {"validate", "signer", "publisher"}
    for job in workflow["jobs"].values():
        assert "github.event_name == 'workflow_dispatch'" in job["if"]
        assert "github.ref == 'refs/heads/main'" in job["if"]
        assert job["permissions"]["contents"] == "read"
    assert workflow["on"]["workflow_dispatch"]["inputs"]["open_draft_pr"] == {
        "description": "Publish the exact branch and open its draft pull request",
        "required": "false",
        "default": "true",
        "type": "boolean",
    }
    assert "inputs.open_draft_pr" in workflow["jobs"]["publisher"]["if"]
    draft_step = _step(
        "publisher", "Open draft pull request linked only to issue 1557"
    )
    assert "if" not in draft_step


def test_signer_and_publisher_credentials_are_strictly_separated() -> None:
    signer = _job_text("signer")
    publisher = _job_text("publisher")
    cleanup = _step("signer", "Create one signed receipt and exact deletions at B")[
        "run"
    ]

    assert "environment: production-signing" in signer
    assert "AXIOM_ENCODE_APPLY_SIGNING_KEY" in signer
    assert "axiom-encode-apply-signer run" in cleanup
    assert "AXIOM_REPO_TOKEN" not in signer
    assert "OPENAI_API_KEY" not in signer
    assert "git push" not in signer
    assert "gh auth" not in signer

    assert "environment: legacy-cleanup-publishing" in publisher
    assert "AXIOM_REPO_TOKEN" in publisher
    assert "AXIOM_ENCODE_APPLY_SIGNING_KEY" not in publisher
    assert "OPENAI_API_KEY" not in publisher
    assert "axiom-encode-apply-signer" not in publisher
    assert "Provision verification-only runtime" in publisher


def test_signer_creates_atomic_cleanup_without_fast_or_model_pipeline() -> None:
    signer = _job_text("signer")
    cleanup = _step("signer", "Create one signed receipt and exact deletions at B")[
        "run"
    ]

    assert cleanup.count("cleanup-unmanifested-legacy") == 1
    assert "--base-ref \"$RULESPEC_BASE\"" in cleanup
    assert "--expected-encoder-checkout" in cleanup
    assert "--axiom-rules-engine-path" in cleanup
    assert "--rulespec-dependency-root" in cleanup
    assert "--fast" not in signer
    assert "targeted-signed-reencode" not in signer
    assert "encode --apply" not in signer
    assert " cleanup-unmanifested-legacy" in cleanup
    assert "\n  retire" not in cleanup
    assert (
        "TheAxiomFoundation/axiom-encode/.github/workflows/"
        "atomic-legacy-cleanup.yml@refs/heads/main"
    ) in cleanup


def test_transport_is_exact_receipt_plus_deletion_inventory() -> None:
    package = _step("signer", "Package the exact receipt and deletion inventory")[
        "run"
    ]
    upload = _step("signer", "Upload exact cleanup transport")
    materialize = _step(
        "publisher", "Validate transport and materialize only its exact contraction"
    )["run"]

    assert "scripts/legacy_cleanup_transport.py build" in package
    assert 'find "$RUNNER_TEMP/legacy-cleanup-artifact" -type f' in package
    assert "= 2" in package
    assert "deletion-inventory.json" in package
    assert ".axiom/legacy-rulespec-deletion-receipts" in package
    assert upload["with"]["include-hidden-files"] == "true"
    assert upload["with"]["if-no-files-found"] == "error"
    assert "scripts/legacy_cleanup_transport.py materialize" in materialize
    assert "--base-ref \"$RULESPEC_BASE\"" in materialize
    assert "--trusted-signing-roots" in materialize
    assert "rsync" not in package + materialize


def test_publisher_stages_exact_set_and_creates_one_guarded_b_to_h_commit() -> None:
    stage = _step(
        "publisher", "Reverify and stage only receipt plus authorized deletions"
    )["run"]
    commit = _step(
        "publisher", "Create exactly one H with sole parent B and run committed guard"
    )["run"]

    assert "stage-signed-backfill" in stage
    assert "--legacy-cleanup-base-ref \"$RULESPEC_BASE\"" in stage
    assert "--expected-encoder-checkout" in stage
    assert "--axiom-rules-engine-path" in stage
    assert "git add -A" not in stage
    assert "git add ." not in stage
    assert "guard-generated" in commit
    assert "--base-ref \"$RULESPEC_BASE\"" in commit
    assert "--head-ref \"$head\"" in commit
    assert 'rev-parse HEAD^)" = "$RULESPEC_BASE"' in commit
    assert 'rev-list --count "$RULESPEC_BASE..$head"' in commit
    assert "log -1 --format='%P'" in commit
    assert "git merge" not in commit
    assert "git rebase" not in commit


def test_publisher_rechecks_main_nonforce_pushes_and_opens_only_a_draft() -> None:
    push = _step(
        "publisher", "Recheck main and non-force push same-repository branch"
    )["run"]
    pull_request = _step(
        "publisher", "Open draft pull request linked only to issue 1557"
    )["run"]

    assert push.count("refs/heads/main") == 2
    assert push.count('test "$remote_main" = "$RULESPEC_BASE"') == 2
    assert 'push origin "HEAD:refs/heads/$BRANCH"' in push
    assert "--force" not in push
    assert "--force-with-lease" not in push
    assert "branch already exists" in push
    assert "-F draft=true" in pull_request
    assert '.draft == true' in pull_request
    assert '.base.sha == $base' in pull_request
    assert '.head.repo.full_name == $repo' in pull_request
    assert pull_request.count("axiom-encode#1557") == 1
    assert re.findall(r"(?:#|issues/)(\d+)", pull_request) == ["1557"]
    assert "gh pr merge" not in pull_request


def test_target_checkout_is_fresh_exact_b_and_never_persists_credentials() -> None:
    workflow = _workflow()
    for job_name in ("signer", "publisher"):
        checkout = next(
            step
            for step in workflow["jobs"][job_name]["steps"]
            if step.get("name")
            in {
                "Checkout exact protected RuleSpec base",
                "Checkout fresh exact RuleSpec base with publisher credential",
            }
        )
        assert checkout["with"]["ref"] == "${{ inputs.rulespec_base }}"
        assert checkout["with"]["fetch-depth"] == "0"
        assert checkout["with"]["persist-credentials"] == "false"
    authenticate = _step(
        "publisher", "Authenticate fresh exact B before materialization"
    )["run"]
    assert 'rev-parse HEAD)\" = \"$RULESPEC_BASE"' in authenticate
    assert "status --porcelain --ignored" in authenticate
    assert "refs/heads/main" in authenticate


def test_every_external_action_is_pinned_to_a_full_commit() -> None:
    workflow = _workflow()
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            uses = step.get("uses")
            if uses is None:
                continue
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", uses), uses


def test_every_workflow_shell_step_has_valid_bash_syntax(tmp_path: Path) -> None:
    workflow = _workflow()
    for job_name, job in workflow["jobs"].items():
        for index, step in enumerate(job.get("steps", [])):
            command = step.get("run")
            if command is None:
                continue
            script = tmp_path / f"{job_name}-{index}.bash"
            script.write_text(command, encoding="utf-8")
            subprocess.run(["bash", "-n", str(script)], check=True)


def test_preflight_rejects_non_full_refs_before_any_dynamic_checkout(
    tmp_path: Path,
) -> None:
    command = _step("validate", "Validate bounded immutable inputs")["run"]
    environment = {
        **os.environ,
        "COUNTRY": "us",
        "ENCODER_REF": "a" * 40,
        "RULESPEC_BASE": "b" * 39,
        "CORPUS_REF": "c" * 40,
        "RULES_ENGINE_REF": "d" * 40,
        "PRIMARY_PATHS_JSON": '["us/statutes/legacy.yaml"]',
        "REASON": "Remove exact legacy group",
        "RULESPEC_DEPENDENCIES_JSON": "{}",
    }

    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "RULESPEC_BASE must be a full lowercase commit SHA" in completed.stderr
