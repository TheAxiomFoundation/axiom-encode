"""Tests for the pre-approval RuleSpec encoder pin admission check."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

import scripts.check_encoder_pin_admission as admission_mod
from scripts.check_encoder_pin_admission import (
    AdmissionCheckError,
    EncoderPin,
    evaluate_admission,
    git_is_ancestor,
    git_on_first_parent_main,
    read_encoder_pin,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/targeted-signed-reencode.yml"


def _git(repo: Path, *args: str) -> str:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "commit", "--allow-empty", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def history(tmp_path: Path) -> dict[str, str | Path]:
    """Encoder history with the #1859 floor, the #1883 fix and a merged branch.

    main (first parent):
      old_pin -> floor (#1859) -> window_pin -> fix (#1883) -> pin
      -> merge(feature) -> tip
    feature (second parent of the merge): feature_commit, branched at pin
    """

    repo = tmp_path / "axiom-encode"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    old_pin = _commit(repo, "old pin")
    floor = _commit(repo, "producer-ancestry admission (#1859)")
    window_pin = _commit(repo, "pin without the trusted-wrapper fix")
    fix = _commit(repo, "trusted Git wrapper runs the proof (#1883)")
    pin = _commit(repo, "newer pin")
    _git(repo, "switch", "-q", "-c", "feature")
    feature_commit = _commit(repo, "feature work")
    _git(repo, "switch", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge feature", "feature")
    tip = _commit(repo, "main tip")
    return {
        "repo": repo,
        "old_pin": old_pin,
        "floor": floor,
        "window_pin": window_pin,
        "fix": fix,
        "pin": pin,
        "feature_commit": feature_commit,
        "tip": tip,
    }


def _evaluate(
    history: dict, *, pin: str, commit: str, main_ref: str = "refs/heads/main"
):
    repo = history["repo"]
    return evaluate_admission(
        pin=EncoderPin(ref=pin, version="0.2.1"),
        encoder_commit=commit,
        is_ancestor=git_is_ancestor(repo),
        on_first_parent_main=git_on_first_parent_main(repo, main_ref),
        ancestry_admission_commit=history["floor"],
        supervised_admission_fix_commit=history["fix"],
    )


def _toolchain(ref: str | None, version: str | None = "0.2.2160") -> str:
    lines = ["[workflow_toolchain]"]
    if version is not None:
        lines.append(f'axiom_encode_version = "{version}"')
    if ref is not None:
        lines.append(f'axiom_encode_ref = "{ref}"')
    lines.append(f'axiom_corpus_ref = "{"c" * 40}"')
    return "\n".join(lines) + "\n"


class TestReadEncoderPin:
    def test_reads_ref_and_version(self) -> None:
        pin = read_encoder_pin(_toolchain("a" * 40))
        assert pin == EncoderPin(ref="a" * 40, version="0.2.2160")

    def test_missing_pin_is_none(self) -> None:
        assert read_encoder_pin(_toolchain(None, None)) == EncoderPin(None, None)

    @pytest.mark.parametrize("ref", ["A" * 40, "a" * 39, "main", "v0.2.2160"])
    def test_rejects_non_sha_pin(self, ref: str) -> None:
        with pytest.raises(AdmissionCheckError, match="full lowercase commit SHA"):
            read_encoder_pin(_toolchain(ref))

    def test_rejects_missing_section(self) -> None:
        with pytest.raises(AdmissionCheckError, match="workflow_toolchain"):
            read_encoder_pin('[toolchain]\naxiom_corpus_release = "x"\n')

    def test_rejects_invalid_toml(self) -> None:
        with pytest.raises(AdmissionCheckError, match="not valid TOML"):
            read_encoder_pin("[workflow_toolchain\n")


class TestEvaluateAdmission:
    def test_exact_pin_is_admitted_even_before_the_floor(self, history) -> None:
        result = _evaluate(history, pin=history["old_pin"], commit=history["old_pin"])
        assert result.admissible is True
        assert "exact pin" in result.reason

    def test_pin_before_the_floor_admits_only_itself(self, history) -> None:
        result = _evaluate(history, pin=history["old_pin"], commit=history["tip"])
        assert result.admissible is False
        assert "predates producer-ancestry admission" in result.reason
        assert "#1859" in result.reason

    def test_first_parent_main_descendant_is_admitted(self, history) -> None:
        result = _evaluate(history, pin=history["pin"], commit=history["tip"])
        assert result.admissible is True
        assert "descends from the pin" in result.reason

    def test_fix_itself_as_pin_admits_descendants(self, history) -> None:
        result = _evaluate(history, pin=history["fix"], commit=history["tip"])
        assert result.admissible is True

    @pytest.mark.parametrize("pin_key", ["floor", "window_pin"])
    @pytest.mark.parametrize("commit_key", ["tip", None])
    def test_pin_between_1859_and_1883_admits_nothing(
        self, history, pin_key: str, commit_key: str | None
    ) -> None:
        # Under the verification runtime such a pin's guard cannot run its own
        # Git checks, so CI rejects every changed manifest, exact pin included.
        commit = history[commit_key] if commit_key else history[pin_key]
        result = _evaluate(history, pin=history[pin_key], commit=commit)
        assert result.admissible is False
        assert "#1883" in result.reason
        assert "the exact pin included" in result.reason

    def test_descendant_admitted_from_a_full_clone_remote_main(
        self, history, tmp_path: Path
    ) -> None:
        clone = tmp_path / "clone"
        subprocess.run(
            ["git", "clone", "-q", f"file://{history['repo']}", str(clone)],
            check=True,
            capture_output=True,
        )
        result = evaluate_admission(
            pin=EncoderPin(ref=history["pin"], version="0.2.1"),
            encoder_commit=history["tip"],
            is_ancestor=git_is_ancestor(clone),
            on_first_parent_main=git_on_first_parent_main(
                clone, "refs/remotes/origin/main"
            ),
            ancestry_admission_commit=history["floor"],
            supervised_admission_fix_commit=history["fix"],
        )
        assert result.admissible is True

    def test_merged_second_parent_commit_is_refused(self, history) -> None:
        result = _evaluate(
            history, pin=history["pin"], commit=history["feature_commit"]
        )
        assert result.admissible is False
        assert "first-parent" in result.reason

    def test_encoder_older_than_the_pin_is_refused(self, history) -> None:
        result = _evaluate(history, pin=history["pin"], commit=history["floor"])
        assert result.admissible is False
        assert "does not descend from the pinned" in result.reason

    def test_unknown_pin_raises(self, history) -> None:
        with pytest.raises(AdmissionCheckError, match="does not contain commit"):
            _evaluate(history, pin="f" * 40, commit=history["tip"])

    def test_no_pin_is_not_judged(self) -> None:
        def unreachable(*_args: str) -> bool:
            raise AssertionError("history must not be consulted without a pin")

        result = evaluate_admission(
            pin=EncoderPin(None, None),
            encoder_commit="a" * 40,
            is_ancestor=unreachable,
            on_first_parent_main=unreachable,
        )
        assert result.admissible is None

    def test_malformed_encoder_commit_raises(self) -> None:
        with pytest.raises(AdmissionCheckError, match="encoder commit"):
            evaluate_admission(
                pin=EncoderPin("a" * 40, None),
                encoder_commit="HEAD",
                is_ancestor=lambda *_: True,
                on_first_parent_main=lambda *_: True,
            )

    def test_shallow_history_raises(self, history, tmp_path: Path) -> None:
        shallow = tmp_path / "shallow"
        subprocess.run(
            [
                "git",
                "clone",
                "-q",
                "--depth",
                "1",
                f"file://{history['repo']}",
                str(shallow),
            ],
            check=True,
            capture_output=True,
        )
        with pytest.raises(AdmissionCheckError, match="shallow"):
            git_on_first_parent_main(shallow, "refs/remotes/origin/main")(
                history["tip"]
            )


class TestMain:
    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        history: dict,
        *,
        pin: str | None,
        commit: str,
        override: str = "",
        open_pr: str = "true",
    ) -> tuple[int, str]:
        toolchain = tmp_path / "workflow-toolchain.toml"
        toolchain.write_text(_toolchain(pin), encoding="utf-8")
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        monkeypatch.setenv("ENCODER_PIN_ADMISSION_OVERRIDE", override)
        monkeypatch.setattr(
            admission_mod, "PRODUCER_ANCESTRY_ADMISSION_COMMIT", history["floor"]
        )
        monkeypatch.setattr(
            admission_mod, "SUPERVISED_ADMISSION_FIX_COMMIT", history["fix"]
        )
        code = admission_mod.main(
            [
                "--toolchain",
                str(toolchain),
                "--encoder-checkout",
                str(history["repo"]),
                "--encoder-commit",
                commit,
                "--main-ref",
                "refs/heads/main",
                "--open-pr",
                open_pr,
            ]
        )
        return code, summary.read_text() if summary.exists() else ""

    def test_inadmissible_encoder_blocks_with_triage(
        self, monkeypatch, tmp_path, history, capsys
    ) -> None:
        code, summary = self._run(
            monkeypatch,
            tmp_path,
            history,
            pin=history["old_pin"],
            commit=history["tip"],
        )
        assert code == 1
        assert "Blocked before the signing environment and any model call" in summary
        assert "axiom_encode_ref" in summary
        assert "::error title=Encoder not admissible" in capsys.readouterr().out

    def test_admissible_encoder_passes_silently(
        self, monkeypatch, tmp_path, history
    ) -> None:
        code, summary = self._run(
            monkeypatch, tmp_path, history, pin=history["pin"], commit=history["tip"]
        )
        assert code == 0
        assert summary == ""

    def test_override_reports_without_blocking(
        self, monkeypatch, tmp_path, history
    ) -> None:
        code, summary = self._run(
            monkeypatch,
            tmp_path,
            history,
            pin=history["old_pin"],
            commit=history["tip"],
            override="true",
        )
        assert code == 0
        assert "Report-only" in summary
        assert "ENCODER_PIN_ADMISSION_OVERRIDE=true" in summary

    def test_window_pin_blocks_a_pr_run(self, monkeypatch, tmp_path, history) -> None:
        code, summary = self._run(
            monkeypatch,
            tmp_path,
            history,
            pin=history["window_pin"],
            commit=history["tip"],
        )
        assert code == 1
        assert "#1883" in summary

    def test_artifact_only_run_is_reported_not_blocked(
        self, monkeypatch, tmp_path, history
    ) -> None:
        code, summary = self._run(
            monkeypatch,
            tmp_path,
            history,
            pin=history["old_pin"],
            commit=history["tip"],
            open_pr="false",
        )
        assert code == 0
        assert "Report-only: this run opens no RuleSpec pull request" in summary

    def test_missing_pin_passes_and_is_reported(
        self, monkeypatch, tmp_path, history
    ) -> None:
        code, summary = self._run(
            monkeypatch, tmp_path, history, pin=None, commit=history["tip"]
        )
        assert code == 0
        assert "declares no axiom_encode_ref" in summary
        assert "Not judged" in summary

    def test_unprovable_history_blocks(self, monkeypatch, tmp_path, history) -> None:
        code, summary = self._run(
            monkeypatch, tmp_path, history, pin="f" * 40, commit=history["tip"]
        )
        assert code == 1
        assert "could not be established" in summary

    def test_missing_toolchain_blocks(self, monkeypatch, tmp_path, history) -> None:
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        code = admission_mod.main(
            [
                "--toolchain",
                str(tmp_path / "absent.toml"),
                "--encoder-checkout",
                str(history["repo"]),
                "--encoder-commit",
                history["tip"],
            ]
        )
        assert code == 1


def test_admission_commits_are_the_1859_and_1883_main_commits() -> None:
    assert (
        admission_mod.PRODUCER_ANCESTRY_ADMISSION_COMMIT
        == "630aa73931e5aa87c51d25fe448e38dbcfaed50f"
    )
    assert (
        admission_mod.SUPERVISED_ADMISSION_FIX_COMMIT
        == "b5e85ab4135be6ed4acaf2b2cd2b783648af2f6e"
    )


def test_workflow_checks_admission_before_the_protected_job() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    job = workflow["jobs"]["encoder_pin_admission"]
    assert job["name"] == "Check RuleSpec encoder pin admission"
    # Unprotected and pre-approval: no environment, read-only token, all runs.
    assert "environment" not in job
    assert "permissions" not in job
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert "queue_id" not in job["if"]
    checkouts = [step for step in job["steps"] if "uses" in step]
    assert all(step["with"]["persist-credentials"] is False for step in checkouts)
    assert all("token" not in step["with"] for step in checkouts)
    encoder_checkout, rulespec_checkout = checkouts
    assert encoder_checkout["with"]["fetch-depth"] == 0
    assert rulespec_checkout["with"]["repository"] == (
        "TheAxiomFoundation/rulespec-${{ inputs.country }}"
    )
    assert rulespec_checkout["with"]["ref"] == "${{ inputs.rulespec_ref }}"
    assert rulespec_checkout["with"]["sparse-checkout"] == (
        ".axiom/workflow-toolchain.toml"
    )
    check = job["steps"][-1]
    assert check["env"]["ENCODER_PIN_ADMISSION_OVERRIDE"] == (
        "${{ vars.ENCODER_PIN_ADMISSION_OVERRIDE }}"
    )
    assert check["env"]["OPEN_PR"] == "${{ inputs.open_pr }}"
    assert '--open-pr "$OPEN_PR"' in check["run"]
    assert "scripts/check_encoder_pin_admission.py" in check["run"]
    assert '--encoder-commit "$GITHUB_SHA"' in check["run"]
    assert '.axiom/workflow-toolchain.toml"' in check["run"]
    assert (
        'test "$(git -C axiom-encode rev-parse HEAD)" = "$GITHUB_SHA"' in (check["run"])
    )

    encode = workflow["jobs"]["encode"]
    assert encode["needs"] == ["attempt_budget", "encoder_pin_admission"]
    assert "needs.encoder_pin_admission.result == 'success'" in encode["if"]
    # The protected job keeps its own fail-closed routing conditions.
    assert "github.run_attempt == 1" in encode["if"]
    assert encode["environment"] == "production-signing"
