"""End-to-end ``repoint-legacy-successor`` on a real-shaped fixture checkout.

Runs the real command (signing through the test broker, with only the rules
engine validators stubbed), then ``guard-generated`` on the result, then the
workflow's repoint packaging check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.cli import (
    SUCCESSOR_REPOINT_DEPENDENT_TOOL,
    SUCCESSOR_REPOINT_RETIRED_TOOL,
    _load_verified_applied_encoding_manifest_payload,
    _successor_repoint_manifest_issues,
    cmd_guard_generated,
    guard_generated_change_issues,
)
from tests.successor_repoint_fixtures import (
    BROKER,
    DEPENDENT,
    DEPENDENT_COMPANION,
    DEPENDENT_MANIFEST,
    DEPENDENT_RELATIVE_V1,
    ENVELOPE,
    LEGACY,
    LEGACY_COMPANION,
    LEGACY_V1_MANIFEST,
    PROGRAM_SPEC,
    RETIRED_MANIFEST,
    SUCCESSOR,
    SUCCESSOR_IDENTITY,
    SUCCESSOR_MANIFEST,
    TRANSITIVE,
    build_repoint_fixture,
    git,
    run_repoint,
)

RECEIPT_DIR = ".axiom/legacy-successor-repoints"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def repointed(tmp_path, monkeypatch):
    fixture = build_repoint_fixture(tmp_path, monkeypatch)
    run_repoint(fixture)
    return fixture


def _changed(fixture) -> dict[str, str]:
    """Return ``path -> status`` for the whole working-tree change set."""

    git(fixture.repo, "add", "-A", "--intent-to-add")
    out = git(fixture.repo, "diff", "--name-status", "--no-renames", fixture.base)
    return {
        line.split("\t", 1)[1]: line.split("\t", 1)[0]
        for line in out.splitlines()
        if line
    }


def _receipt(fixture) -> tuple[str, dict]:
    receipts = sorted((fixture.repo / RECEIPT_DIR).iterdir())
    assert len(receipts) == 1
    relative = receipts[0].relative_to(fixture.repo).as_posix()
    return relative, json.loads(receipts[0].read_text())


class TestRepointEndToEnd:
    def test_rewrites_the_dependent_onto_the_successor(self, repointed):
        before = repointed.preimages[DEPENDENT].decode("utf-8")
        after = (repointed.repo / DEPENDENT).read_text()
        successor_sha256 = _sha256((repointed.repo / SUCCESSOR).read_bytes())
        expected = (
            before.replace(
                "us:policies/irs/rev-proc-2025-32/earned-income-credit",
                SUCCESSOR_IDENTITY,
            )
            .replace(
                f"sha256:{_sha256(repointed.preimages[LEGACY])}",
                f"sha256:{successor_sha256}",
            )
            .replace(
                "eitc_earned_income_amounts",
                "earned_income_credit_earned_income_amounts",
            )
            .replace(
                "eitc_maximum_credit_amounts",
                "earned_income_credit_maximum_credit_amounts",
            )
            .replace(
                "eitc_maximum_investment_income",
                "earned_income_credit_maximum_investment_income",
            )
        )
        assert after == expected
        # The companion names nothing retired, so it is untouched.
        assert (repointed.repo / DEPENDENT_COMPANION).read_bytes() == (
            repointed.preimages[DEPENDENT_COMPANION]
        )

    def test_retires_the_legacy_group_and_its_v1_manifests(self, repointed):
        for relative in (LEGACY, LEGACY_COMPANION, LEGACY_V1_MANIFEST):
            assert not (repointed.repo / relative).exists()
        assert not (repointed.repo / DEPENDENT_RELATIVE_V1).exists()

    def test_leaves_the_successor_model_manifest_untouched(self, repointed):
        assert (repointed.repo / SUCCESSOR_MANIFEST).read_bytes() == (
            repointed.successor_manifest_bytes
        )

    def test_writes_both_manifest_classes(self, repointed):
        receipt_path, _receipt_payload = _receipt(repointed)
        retired = json.loads((repointed.repo / RETIRED_MANIFEST).read_text())
        assert retired["tool"] == SUCCESSOR_REPOINT_RETIRED_TOOL
        assert retired["applied_files"] == [
            {"path": LEGACY_COMPANION, "deleted": True},
            {"path": LEGACY, "deleted": True},
        ]
        dependent = json.loads((repointed.repo / DEPENDENT_MANIFEST).read_text())
        assert dependent["tool"] == SUCCESSOR_REPOINT_DEPENDENT_TOOL
        assert dependent["applied_files"] == [
            {
                "path": DEPENDENT,
                "sha256": _sha256((repointed.repo / DEPENDENT).read_bytes()),
            },
            {
                "path": DEPENDENT_COMPANION,
                "sha256": _sha256(repointed.preimages[DEPENDENT_COMPANION]),
            },
        ]
        for payload in (retired, dependent):
            assert payload["successor_repoint"]["receipt_path"] == receipt_path
            assert payload["successor_repoint"]["receipt_sha256"] == _sha256(
                (repointed.repo / receipt_path).read_bytes()
            )
            # Neither class claims a live digest for shared metadata.
            claimed = {item["path"] for item in payload["applied_files"]}
            assert not claimed & {PROGRAM_SPEC, "known-validation-gaps.yaml"}

    def test_the_receipt_records_the_whole_transaction(self, repointed):
        _path, receipt = _receipt(repointed)
        assert receipt["request"] == ENVELOPE
        assert receipt["repository"]["base_commit"] == repointed.base
        assert receipt["legacy"]["manifests"] == [
            {
                "path": LEGACY_V1_MANIFEST,
                "sha256": _sha256(
                    git(
                        repointed.repo, "show", f"{repointed.base}:{LEGACY_V1_MANIFEST}"
                    ).encode()
                ),
                "owner_class": "v1-hmac-untrusted",
            }
        ]
        assert "owner_class" not in receipt["legacy"]
        assert [
            (item["path"], item["owner_class"])
            for item in receipt["dependents"][0]["manifests"]
        ] == [
            (DEPENDENT_MANIFEST, "v1-manual-hmac-untrusted"),
            (DEPENDENT_RELATIVE_V1, "v1-deterministic-hmac-untrusted"),
        ]
        assert receipt["successor"]["manifest_sha256"] == _sha256(
            repointed.successor_manifest_bytes
        )
        semantics = receipt["semantics"]
        assert semantics["successor_window"] == {
            "effective_from": "2026-01-01",
            "effective_to": "2026-12-31",
        }
        assert semantics["post_window_behavior_change"] is True
        assert semantics["pre_window_behavior_change"] is False
        assert semantics["behavior_change_outside_successor_window"] is True
        assert [
            (item["concept"], item["engine_lowering"])
            for item in semantics["runtime_behavior_outside_successor_window"]
        ] == [
            ("earned_income_credit_earned_income_amounts", "indexed_parameter"),
            ("earned_income_credit_maximum_credit_amounts", "indexed_parameter"),
            ("earned_income_credit_maximum_investment_income", "scalar_parameter"),
        ]
        assert sorted(item["path"] for item in receipt["metadata_reconciliations"]) == (
            sorted(
                [
                    ".axiom/index/provisions_to_rules.json",
                    ".axiom/pending-validation-fingerprints.json",
                    ".axiom/toolchain.toml",
                    ".axiom/upstream-source-check-baseline.txt",
                    "known-missing-money-atoms.yaml",
                    "known-validation-gaps.yaml",
                ]
            )
        )
        assert receipt["program_scope_reconciliations"][0]["removed"] == [
            "policies/irs/rev-proc-2025-32/earned-income-credit"
        ]

    def test_reconciles_metadata_and_the_program_spec(self, repointed):
        waivers = yaml.safe_load(
            (repointed.repo / "known-validation-gaps.yaml").read_text()
        )
        assert set(waivers["validate_failures"]) == {TRANSITIVE}
        waiver_sha256 = _sha256(
            (repointed.repo / "known-validation-gaps.yaml").read_bytes()
        )
        assert (
            f'validation_waiver_set_sha256 = "{waiver_sha256}"'
            in (repointed.repo / ".axiom/toolchain.toml").read_text()
        )
        index = json.loads(
            (repointed.repo / ".axiom/index/provisions_to_rules.json").read_text()
        )
        modules = {
            record["module"]
            for records in index["provisions"].values()
            for record in records
        }
        assert LEGACY not in modules and SUCCESSOR in modules and DEPENDENT in modules
        pending = json.loads(
            (repointed.repo / ".axiom/pending-validation-fingerprints.json").read_text()
        )
        assert LEGACY not in pending["modules"]
        # Historical prose survives; only structured records are removed.
        assert LEGACY in pending["generated_from"]["divergence_note"]
        assert (
            LEGACY
            not in (
                repointed.repo / ".axiom/upstream-source-check-baseline.txt"
            ).read_text()
        )
        assert (
            LEGACY
            not in (repointed.repo / "known-missing-money-atoms.yaml").read_text()
        )
        spec = yaml.safe_load((repointed.repo / PROGRAM_SPEC).read_text())
        assert spec["scope"]["federal"] == [
            "policies/irs/rev-proc-2025-32/page-15",
            "statutes/26/24/d",
            "statutes/26/32",
        ]

    def test_changes_exactly_the_declared_files(self, repointed):
        receipt_path, _receipt_payload = _receipt(repointed)
        assert _changed(repointed) == {
            LEGACY: "D",
            LEGACY_COMPANION: "D",
            LEGACY_V1_MANIFEST: "D",
            DEPENDENT_RELATIVE_V1: "D",
            DEPENDENT: "M",
            DEPENDENT_MANIFEST: "M",
            RETIRED_MANIFEST: "A",
            receipt_path: "A",
            PROGRAM_SPEC: "M",
            "known-validation-gaps.yaml": "M",
            ".axiom/toolchain.toml": "M",
            ".axiom/index/provisions_to_rules.json": "M",
            ".axiom/pending-validation-fingerprints.json": "M",
            ".axiom/upstream-source-check-baseline.txt": "M",
            "known-missing-money-atoms.yaml": "M",
        }

    def test_guard_generated_accepts_the_tree(self, repointed):
        issues = guard_generated_change_issues(
            repointed.repo,
            corpus_path=repointed.corpus,
            base_ref=repointed.base,
            head_ref="HEAD",
        )
        assert issues == [], "\n".join(issues)

    def test_guard_generated_command_accepts_the_tree(self, repointed, capsys):
        from types import SimpleNamespace

        with pytest.raises(SystemExit) as exit_info:
            cmd_guard_generated(
                SimpleNamespace(
                    repo=repointed.repo,
                    corpus_path=repointed.corpus,
                    base_ref=repointed.base,
                    head_ref="HEAD",
                    json=True,
                    all=False,
                    expected_encoder_checkout=None,
                )
            )
        assert exit_info.value.code == 0
        assert json.loads(capsys.readouterr().out)["passed"] is True

    def test_guard_refuses_a_live_postimage_that_drifts_in_the_same_change(
        self, repointed
    ):
        spec = repointed.repo / PROGRAM_SPEC
        spec.write_text(spec.read_text() + "# drift\n")
        issues = guard_generated_change_issues(
            repointed.repo,
            corpus_path=repointed.corpus,
            base_ref=repointed.base,
            head_ref="HEAD",
        )
        assert any(
            f"postimage is not live: {PROGRAM_SPEC}" in issue for issue in issues
        ), issues

    def test_guard_refuses_a_restored_legacy_file(self, repointed):
        (repointed.repo / LEGACY).write_bytes(repointed.preimages[LEGACY])
        issues = guard_generated_change_issues(
            repointed.repo,
            corpus_path=repointed.corpus,
            base_ref=repointed.base,
            head_ref="HEAD",
        )
        assert issues

    def test_unrelated_later_edits_leave_no_stale_repoint_manifest(self, repointed):
        """A committed repoint stays verifiable after shared-file edits."""

        git(repointed.repo, "add", "-A")
        git(repointed.repo, "commit", "-q", "-m", "repoint")
        # Unrelated edits to every shared file the transaction reconciled.
        waivers = repointed.repo / "known-validation-gaps.yaml"
        waivers.write_text(
            waivers.read_text().replace(
                f'  "{TRANSITIVE}":\n', f'  "{TRANSITIVE}":\n', 1
            )
            + "# reviewed 2026-09-23\n"
        )
        toolchain = repointed.repo / ".axiom/toolchain.toml"
        toolchain.write_text(
            toolchain.read_text().replace(
                toolchain.read_text().split('validation_waiver_set_sha256 = "')[1][:64],
                _sha256(waivers.read_bytes()),
            )
        )
        index = repointed.repo / ".axiom/index/provisions_to_rules.json"
        payload = json.loads(index.read_text())
        payload["description"] = "Reverse index (regenerated)."
        index.write_text(json.dumps(payload, indent=2) + "\n")
        spec = repointed.repo / PROGRAM_SPEC
        spec.write_text(spec.read_text() + "  state: []\n")
        git(repointed.repo, "commit", "-q", "-am", "unrelated edits")

        for relative in (RETIRED_MANIFEST, DEPENDENT_MANIFEST):
            verified, _prefix, _digest, issues = (
                _load_verified_applied_encoding_manifest_payload(
                    repointed.repo,
                    relative,
                    signing_broker=BROKER,
                )
            )
            assert verified is not None and issues == [], issues

    def test_a_dependent_manifest_reverifies_the_successors_own_signature(
        self, repointed
    ):
        manifest = repointed.repo / SUCCESSOR_MANIFEST
        payload = json.loads(manifest.read_text())
        payload["citation"] = "us/guidance/irs/rev-proc-2025-32/page-16"
        manifest.write_text(json.dumps(payload, indent=2) + "\n")
        dependent = json.loads((repointed.repo / DEPENDENT_MANIFEST).read_text())
        issues = _successor_repoint_manifest_issues(
            dependent,
            repo_path=repointed.repo,
            manifest_label=DEPENDENT_MANIFEST,
            signing_broker=BROKER,
            local_corpus_release=None,
        )
        assert any(
            "successor model manifest does not verify" in issue for issue in issues
        )

    def test_refuses_to_run_twice(self, repointed):
        git(repointed.repo, "add", "-A")
        git(repointed.repo, "commit", "-q", "-m", "repoint")
        with pytest.raises(SystemExit, match="refused"):
            run_repoint(repointed)


# ---------------------------------------------------------------------------
# Workflow packaging and commit of a repoint dispatch
# ---------------------------------------------------------------------------

WORKFLOW = Path(__file__).resolve().parents[1] / (
    ".github/workflows/targeted-signed-reencode.yml"
)


def _step(name: str) -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    run = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == name
    )
    # A developer machine may carry a stale provisioned trusted runtime; force
    # the step's documented test fallback (AXIOM_TEST_PYTHON) instead.
    trusted = "/opt/axiom-verification/python/bin/python"
    assert trusted in run
    return run.replace(trusted, "/nonexistent/axiom-verification/python/bin/python")


def _workspace(fixture) -> dict[str, str]:
    """Lay out the runner workspace the repoint steps expect."""

    import os
    import sys

    root = fixture.tmp_path
    encoder = root / "axiom-encode"
    if not encoder.exists():
        encoder.mkdir()
        (encoder / "scripts").symlink_to(WORKFLOW.parents[2] / "scripts")
        for checkout in (encoder, fixture.corpus, fixture.engine):
            git(checkout, "init", "-q")
            git(checkout, "config", "user.email", "test@example.com")
            git(checkout, "config", "user.name", "Test User")
            git(checkout, "add", "-A")
            git(checkout, "commit", "-q", "--allow-empty", "-m", "pin")
    runner = root / "runner"
    runner.mkdir(exist_ok=True)
    (runner / "successor-repoint-request.json").write_text(json.dumps(ENVELOPE))
    (runner / "guard-generated.json").write_text(
        json.dumps({"passed": True, "issues": [], "repo": str(fixture.repo)})
    )
    return {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", str(root)),
        "RUNNER_TEMP": str(runner),
        "GITHUB_RUN_ID": "35000000001",
        "GITHUB_RUN_ATTEMPT": "1",
        "AXIOM_TEST_PYTHON": sys.executable,
        "PYTHONPATH": str(WORKFLOW.parents[2] / "src"),
        "RULESPEC_CHECKOUT": "rulespec-us",
        "RULESPEC_REF": fixture.base,
        "COUNTRY": "us",
        "CORPUS_REF": "0" * 40,
        "RULES_ENGINE_REF": "0" * 40,
    }


def _run_step(fixture, name: str, *, request: dict | None = None):
    import subprocess

    environment = _workspace(fixture)
    if request is not None:
        (fixture.tmp_path / "runner/successor-repoint-request.json").write_text(
            json.dumps(request)
        )
    return subprocess.run(
        ["bash", "-c", _step(name)],
        capture_output=True,
        text=True,
        cwd=fixture.tmp_path,
        env=environment,
        check=False,
    )


class TestRepointWorkflowPackaging:
    def test_packages_exactly_the_receipt(self, repointed):
        completed = _run_step(repointed, "Package successor repoint changes")
        assert completed.returncode == 0, completed.stderr
        artifact = repointed.tmp_path / "runner/targeted-reencode"
        inventory = json.loads(
            (artifact / "successor-repoint-changes.json").read_text()
        )
        receipt_path, _receipt_payload = _receipt(repointed)
        assert inventory["receipt_path"] == receipt_path
        assert inventory["legacy_primary"] == LEGACY
        assert inventory["successor_primary"] == SUCCESSOR
        assert {item["path"]: item["status"] for item in inventory["changes"]} == (
            _changed(repointed)
        )
        metadata = json.loads((artifact / "metadata.json").read_text())
        assert metadata["schema"] == "axiom-encode/successor-repoint-artifact/v1"
        assert metadata["receipt_sha256"] == inventory["receipt_sha256"]
        assert metadata["rulespec_base"] == repointed.base
        assert "citation" not in metadata
        assert (artifact / "successor-repoint-receipt.json").read_bytes() == (
            repointed.repo / receipt_path
        ).read_bytes()

    def test_refuses_an_extra_file(self, repointed):
        (repointed.repo / "notes.txt").write_text("stray\n")
        completed = _run_step(repointed, "Package successor repoint changes")
        assert completed.returncode != 0
        assert "unexpected=['notes.txt']" in completed.stderr

    def test_refuses_a_postimage_that_is_not_the_receipts(self, repointed):
        spec = repointed.repo / PROGRAM_SPEC
        spec.write_text(spec.read_text() + "# drift\n")
        completed = _run_step(repointed, "Package successor repoint changes")
        assert completed.returncode != 0
        assert f"not its receipt postimage: {PROGRAM_SPEC}" in completed.stderr

    def test_refuses_a_receipt_for_another_request(self, repointed):
        other = dict(ENVELOPE, program_scope_updates=[])
        completed = _run_step(
            repointed, "Package successor repoint changes", request=other
        )
        assert completed.returncode != 0
        assert "not for the dispatched request" in completed.stderr

    def test_commits_exactly_the_packaged_inventory(self, repointed):
        assert _run_step(repointed, "Package successor repoint changes").returncode == 0
        completed = _run_step(repointed, "Commit successor repoint locally")
        assert completed.returncode == 0, completed.stderr
        assert (
            git(repointed.repo, "status", "--porcelain", "--untracked-files=all") == ""
        )
        assert git(repointed.repo, "log", "-1", "--format=%s").strip() == (
            f"Repoint {LEGACY} onto {SUCCESSOR}"
        )
        assert git(repointed.repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == (
            "axiom/signed-backfill-us-35000000001-1"
        )
        committed = git(
            repointed.repo,
            "diff",
            "--name-status",
            "--no-renames",
            repointed.base,
            "HEAD",
        )
        inventory = json.loads(
            (
                repointed.tmp_path
                / "runner/targeted-reencode/successor-repoint-changes.json"
            ).read_text()
        )
        assert committed == "".join(
            f"{item['status']}\t{item['path']}\n" for item in inventory["changes"]
        )
        # The committed tree still passes the guard against the same base.
        assert (
            guard_generated_change_issues(
                repointed.repo,
                corpus_path=repointed.corpus,
                base_ref=repointed.base,
                head_ref="HEAD",
            )
            == []
        )

    def test_commit_refuses_a_change_made_after_packaging(self, repointed):
        assert _run_step(repointed, "Package successor repoint changes").returncode == 0
        (repointed.repo / TRANSITIVE).write_text("format: rulespec/v1\nrules: []\n")
        completed = _run_step(repointed, "Commit successor repoint locally")
        assert completed.returncode != 0
