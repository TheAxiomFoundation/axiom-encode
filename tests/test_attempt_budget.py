"""Tests for the failed-attempt budget guard (axiom-encode#1495)."""

from __future__ import annotations

import http.server
import io
import json
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

import scripts.enforce_attempt_budget as budget_mod
from scripts.enforce_attempt_budget import (
    ENCODE_JOB_NAME,
    ENCODE_STEP_NAME,
    evaluate_attempt_budget,
    make_encode_job_checker,
    run_citation,
)

CITATION = "us-ky/statute/krs/141.020/document-1"
DE_CITATION = "de/statute/estg/78"


def _run(
    run_id: int,
    created_at: str,
    conclusion: str,
    *,
    citation: str = CITATION,
    queue: str = "adhoc:adhoc:adhoc",
    status: str = "completed",
) -> dict:
    return {
        "id": run_id,
        "name": f"Targeted signed RuleSpec re-encode [{queue}] {citation}",
        "created_at": created_at,
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://github.com/org/repo/actions/runs/{run_id}",
    }


class TestRunCitation:
    def test_adhoc_run_name(self) -> None:
        name = f"Targeted signed RuleSpec re-encode [adhoc:adhoc:adhoc] {CITATION}"
        assert run_citation(name) == CITATION

    def test_queue_run_name(self) -> None:
        name = f"Targeted signed RuleSpec re-encode [q1:item2:abc123] {CITATION}"
        assert run_citation(name) == CITATION

    def test_name_without_bracket_separator(self) -> None:
        assert run_citation("some unrelated run") is None

    def test_non_string_name(self) -> None:
        assert run_citation(None) is None


class TestEvaluateAttemptBudget:
    def test_consecutive_failures_reach_budget(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "failure"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=3, current_run_id=99
        )
        assert decision.streak == 3
        assert decision.exhausted
        assert [run["id"] for run in decision.counted_runs] == [3, 2, 1]

    def test_success_resets_streak(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "success"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=3, current_run_id=99
        )
        assert decision.streak == 1
        assert not decision.exhausted

    def test_cancelled_runs_do_not_break_or_count(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "cancelled"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=2, current_run_id=99
        )
        assert decision.streak == 2
        assert decision.exhausted

    def test_timed_out_counts_as_failure(self) -> None:
        runs = [_run(1, "2026-08-12T01:00:00Z", "timed_out")]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=1, current_run_id=99
        )
        assert decision.streak == 1
        assert decision.exhausted

    def test_unknown_conclusion_is_neutral(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "some_future_conclusion"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=3, current_run_id=99
        )
        assert decision.streak == 2

    def test_current_run_excluded(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=2, current_run_id=2
        )
        assert decision.streak == 1
        assert not decision.exhausted

    def test_other_citations_ignored(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(
                2,
                "2026-08-12T02:00:00Z",
                "failure",
                citation="us-ms/statute/27-7-5",
            ),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=2, current_run_id=99
        )
        assert decision.streak == 1

    def test_incomplete_runs_ignored(self) -> None:
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "", status="in_progress"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=1, current_run_id=99
        )
        assert decision.streak == 1

    def test_order_independent_input(self) -> None:
        runs = [
            _run(3, "2026-08-12T03:00:00Z", "failure"),
            _run(1, "2026-08-12T01:00:00Z", "success"),
            _run(2, "2026-08-12T02:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs, citation=CITATION, budget=3, current_run_id=99
        )
        assert decision.streak == 2

    def test_no_matching_runs(self) -> None:
        decision = evaluate_attempt_budget(
            [], citation=CITATION, budget=3, current_run_id=99
        )
        assert decision.streak == 0
        assert not decision.exhausted


class TestEncodeJobChecker:
    """Guard-blocked runs must not feed back into the streak."""

    @staticmethod
    def _jobs_fixture(jobs_by_run: dict[int, list[dict]]):
        calls: list[int] = []

        def fetch(run_id: int) -> list[dict]:
            calls.append(run_id)
            return jobs_by_run[run_id]

        return fetch, calls

    def test_blocked_runs_are_neutral(self) -> None:
        # Runs 2 and 3 were blocked by the guard itself: run-level failure,
        # encode job skipped. Only run 1 actually reached the model.
        fetch, _ = self._jobs_fixture(
            {
                1: [self._encode_job("failure")],
                2: [self._encode_job("skipped")],
                3: [{"name": "Enforce failed-attempt budget", "conclusion": "failure"}],
            }
        )
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "failure"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=3,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch),
        )
        assert decision.streak == 1
        assert not decision.exhausted

    @staticmethod
    def _encode_job(step_conclusion: str, *, job_conclusion: str = "failure") -> dict:
        return {
            "name": ENCODE_JOB_NAME,
            "conclusion": job_conclusion,
            "steps": [
                {"name": ENCODE_STEP_NAME, "conclusion": step_conclusion},
            ],
        }

    def test_preflight_failure_inside_encode_job_is_neutral(self) -> None:
        fetch, _ = self._jobs_fixture(
            {
                1: [
                    {
                        "name": ENCODE_JOB_NAME,
                        "conclusion": "failure",
                        "steps": [
                            {
                                "name": "Resolve trusted prior-run repair candidate",
                                "conclusion": "failure",
                            },
                            {"name": ENCODE_STEP_NAME, "conclusion": "skipped"},
                        ],
                    }
                ]
            }
        )
        decision = evaluate_attempt_budget(
            [_run(1, "2026-08-12T01:00:00Z", "failure")],
            citation=CITATION,
            budget=1,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch),
        )
        assert decision.streak == 0
        assert not decision.exhausted

    def test_real_failures_still_count(self) -> None:
        fetch, _ = self._jobs_fixture(
            {run_id: [self._encode_job("failure")] for run_id in (1, 2, 3)}
        )
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "failure"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=3,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch),
        )
        assert decision.streak == 3
        assert decision.exhausted

    def test_lookup_cap_counts_conservatively(self) -> None:
        fetch, calls = self._jobs_fixture(
            {
                run_id: [self._encode_job("skipped", job_conclusion="skipped")]
                for run_id in range(1, 6)
            }
        )
        runs = [
            _run(run_id, f"2026-08-12T0{run_id}:00:00Z", "failure")
            for run_id in range(1, 6)
        ]
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=3,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch, max_lookups=2),
        )
        # Two newest runs verified as blocked (neutral); the remaining three
        # count unverified once the cap is hit.
        assert len(calls) == 2
        assert decision.streak == 3

    def test_fetch_error_counts_conservatively(self) -> None:
        def fetch(run_id: int) -> list[dict]:
            raise RuntimeError("jobs API down")

        runs = [_run(1, "2026-08-12T01:00:00Z", "failure")]
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=1,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch),
        )
        assert decision.streak == 1

    def test_unverified_success_past_cap_stays_neutral(self) -> None:
        # Ten blocked failures (all verified neutral) exhaust the lookup
        # cap; an unverifiable phantom success below them must not reset
        # the streak fed by genuine failures further down.
        jobs_by_run: dict[int, list[dict]] = {
            run_id: [self._encode_job("skipped", job_conclusion="skipped")]
            for run_id in range(5, 15)
        }
        fetch, calls = self._jobs_fixture(jobs_by_run)
        runs = (
            # Genuine failures, oldest.
            [
                _run(run_id, f"2026-08-12T0{run_id}:00:00Z", "failure")
                for run_id in (1, 2, 3)
            ]
            # Phantom success above them (encode skipped, unverifiable
            # once the cap is spent).
            + [_run(4, "2026-08-12T04:00:00Z", "success")]
            # Ten newest guard-blocked failures that consume the cap.
            + [
                _run(run_id, f"2026-08-12T{run_id}:00:00Z", "failure")
                for run_id in range(5, 15)
            ]
        )
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=3,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch, max_lookups=10),
        )
        assert len(calls) == 10
        # Blocked pile neutral, phantom success neutral (unverified),
        # genuine failures below count unverified: streak reaches budget.
        assert decision.streak == 3
        assert decision.exhausted

    def test_phantom_success_does_not_reset_streak(self) -> None:
        # A queue run re-run: encode requires attempt 1, so attempt 2 has
        # only successful/skipped jobs and the run concludes "success"
        # without any encode having succeeded. It must not lift the block.
        fetch, _ = self._jobs_fixture(
            {
                1: [self._encode_job("failure")],
                2: [self._encode_job("failure")],
                3: [self._encode_job("skipped", job_conclusion="skipped")],
                4: [self._encode_job("failure")],
            }
        )
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "failure"),
            _run(3, "2026-08-12T03:00:00Z", "success"),
            _run(4, "2026-08-12T04:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=3,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch),
        )
        assert decision.streak == 3
        assert decision.exhausted

    def test_real_success_still_resets_streak(self) -> None:
        fetch, _ = self._jobs_fixture(
            {
                1: [self._encode_job("failure")],
                2: [self._encode_job("success", job_conclusion="success")],
                3: [self._encode_job("failure")],
            }
        )
        runs = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "success"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        decision = evaluate_attempt_budget(
            runs,
            citation=CITATION,
            budget=3,
            current_run_id=99,
            spent_model_tokens=make_encode_job_checker(fetch),
        )
        assert decision.streak == 1
        assert not decision.exhausted


class TestMainExitContract:
    """The workflow consumes only main()'s exit code — pin it."""

    FAILING_HISTORY = [
        _run(run_id, f"2026-08-12T0{run_id}:00:00Z", "failure") for run_id in (1, 2, 3)
    ]

    def _set_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        queue_id: str = "",
        repair_run_id: str = "",
        override: str = "",
        budget: str = "",
    ) -> None:
        monkeypatch.setenv("CITATION", CITATION)
        monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")
        monkeypatch.setenv("GH_TOKEN", "test-token")
        monkeypatch.setenv("GITHUB_RUN_ID", "99")
        monkeypatch.setenv("QUEUE_ID", queue_id)
        monkeypatch.setenv("REPAIR_RUN_ID", repair_run_id)
        monkeypatch.setenv("ATTEMPT_BUDGET_OVERRIDE", override)
        monkeypatch.setenv("ATTEMPT_BUDGET", budget)
        monkeypatch.delenv("ATTEMPT_BUDGET_BY_CITATION_JSON", raising=False)
        monkeypatch.delenv("REPEATED_FAILURE_OVERRIDE_JSON", raising=False)
        monkeypatch.delenv("SOURCE_BUNDLE_JSON", raising=False)
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    def _stub_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
        runs: list[dict],
        jobs: list[dict] | None = None,
        rule_codes: dict[int, frozenset[str] | None] | None = None,
    ) -> None:
        monkeypatch.setattr(budget_mod, "_fetch_completed_runs", lambda **kwargs: runs)
        monkeypatch.setattr(
            budget_mod,
            "_fetch_failure_rule_codes",
            lambda **kwargs: (rule_codes or {}).get(kwargs["run"]["id"]),
        )
        monkeypatch.setattr(
            budget_mod,
            "_fetch_run_jobs",
            lambda **kwargs: (
                jobs
                if jobs is not None
                else [TestEncodeJobChecker._encode_job("failure")]
            ),
        )

    def test_blocked_adhoc_exits_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch)
        self._stub_api(monkeypatch, self.FAILING_HISTORY)
        assert budget_mod.main() == 1

    def test_queue_dispatch_reports_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch, queue_id="q-123")
        self._stub_api(monkeypatch, self.FAILING_HISTORY)
        assert budget_mod.main() == 0

    def test_repair_replay_reports_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch, repair_run_id="32201076681")
        self._stub_api(monkeypatch, self.FAILING_HISTORY)
        assert budget_mod.main() == 0

    def test_override_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch, override="true")
        self._stub_api(monkeypatch, self.FAILING_HISTORY)
        assert budget_mod.main() == 0

    def test_scoped_limit_keeps_us_blocked_and_blocks_fourth_de_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_env(monkeypatch)
        monkeypatch.setenv(
            "ATTEMPT_BUDGET_BY_CITATION_JSON",
            json.dumps(
                {
                    DE_CITATION: {
                        "budget": 4,
                        "expires_at": (
                            datetime.now(timezone.utc) + timedelta(days=1)
                        ).isoformat(),
                        "reason": "Reviewed dependency-link fix before one additional retry",
                    }
                }
            ),
        )
        de_history = [
            _run(i, f"2026-08-12T{i:02d}:00:00Z", "failure", citation=DE_CITATION)
            for i in (11, 12, 13)
        ]
        self._stub_api(monkeypatch, self.FAILING_HISTORY + de_history)
        assert budget_mod.main() == 1  # US remains at three, with enforcement on.
        monkeypatch.setenv("CITATION", DE_CITATION)
        assert budget_mod.main() == 0
        de_history.append(
            _run(14, "2026-08-12T14:00:00Z", "failure", citation=DE_CITATION)
        )
        self._stub_api(monkeypatch, self.FAILING_HISTORY + de_history)
        assert budget_mod.main() == 1  # A numeric limit, never report-only.

    def test_invalid_scoped_configuration_keeps_guard_enforced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_env(monkeypatch)
        monkeypatch.setenv("ATTEMPT_BUDGET_BY_CITATION_JSON", "{broken")
        self._stub_api(monkeypatch, self.FAILING_HISTORY)
        assert budget_mod.main() == 1

    def test_under_budget_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch)
        self._stub_api(monkeypatch, self.FAILING_HISTORY[:2])
        assert budget_mod.main() == 0

    def test_invalid_budget_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch, budget="not-a-number")
        self._stub_api(monkeypatch, self.FAILING_HISTORY)
        assert budget_mod.main() == 1

    def test_api_error_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch)

        def boom(**kwargs: object) -> list[dict]:
            raise RuntimeError("api down")

        monkeypatch.setattr(budget_mod, "_fetch_completed_runs", boom)
        assert budget_mod.main() == 0

    def test_missing_env_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "CITATION",
            "GITHUB_REPOSITORY",
            "GH_TOKEN",
            "QUEUE_ID",
            "REPAIR_RUN_ID",
            "ATTEMPT_BUDGET",
            "ATTEMPT_BUDGET_OVERRIDE",
            "ATTEMPT_BUDGET_BY_CITATION_JSON",
        ):
            monkeypatch.delenv(name, raising=False)
        assert budget_mod.main() == 0


class TestConfiguredCitationBudget:
    NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)
    ENTRY = {
        "budget": 4,
        "expires_at": "2026-09-10T00:00:00Z",
        "reason": "Triage complete",
    }

    @pytest.mark.parametrize(
        "citation", [CITATION, DE_CITATION + "/child", "DE/statute/estg/78"]
    )
    def test_only_exact_citation_matches(self, citation: str) -> None:
        assert (
            budget_mod.configured_citation_budget(
                json.dumps({DE_CITATION: self.ENTRY}),
                citation=citation,
                fallback=3,
                now=self.NOW,
            )
            == 3
        )

    @pytest.mark.parametrize(
        "expires_at",
        ["2026-09-08T23:59:59Z", "2026-09-09T00:00:00Z", "2026-09-09T02:00:00+02:00"],
    )
    def test_expired_entry_restores_existing_limit(self, expires_at: str) -> None:
        entry = {**self.ENTRY, "expires_at": expires_at}
        assert (
            budget_mod.configured_citation_budget(
                json.dumps({DE_CITATION: entry}),
                citation=DE_CITATION,
                fallback=2,
                now=self.NOW,
            )
            == 2
        )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("budget", True),
            ("budget", 0),
            ("budget", -1),
            ("budget", 4.0),
            ("budget", "4"),
            ("reason", " "),
            ("reason", None),
            ("expires_at", "2026-09-10"),
            ("expires_at", None),
            ("expires_at", "tomorrow"),
            ("extra", "typo"),
        ],
    )
    def test_invalid_selected_entry_is_rejected(
        self, field: str, value: object
    ) -> None:
        with pytest.raises(ValueError):
            budget_mod.configured_citation_budget(
                json.dumps({DE_CITATION: {**self.ENTRY, field: value}}),
                citation=DE_CITATION,
                fallback=3,
                now=self.NOW,
            )


TESTS_ISSUE = (
    "regulations/7-cfr/273/4.yaml: ci: [complete-source-unit:tests] Source-stated "
    "exceptions require paired cases; missing: us/regulation/7/273/4(a) "
    "[Absatz a, Nummer 4, Nummer iii] [change]: `or an unmarried disabled child`."
)
STRUCTURE_ISSUE = (
    "regulations/7-cfr/273/4.yaml: ci: [complete-source-unit:structure] Branch "
    "[Absatz b] has no output."
)


class TestIssueRuleCode:
    @pytest.mark.parametrize(
        "issue,expected",
        [
            (TESTS_ISSUE, "complete-source-unit:tests"),
            (STRUCTURE_ISSUE, "complete-source-unit:structure"),
            (
                "statutes/7/2017/a.yaml: ci: [existing-target-oracle-contract] x",
                "existing-target-oracle-contract",
            ),
            (
                "statutes/7/2017/a.yaml: ci: [temporal-dependency-coverage] x",
                "temporal-dependency-coverage",
            ),
        ],
    )
    def test_named_issues(self, issue: str, expected: str) -> None:
        assert budget_mod.issue_rule_code(issue) == expected

    @pytest.mark.parametrize(
        "issue",
        [
            # Output defects that a fresh generation often clears name no rule.
            "statutes/7/2017/a.yaml: compile: Axiom rules engine compile failed: "
            "failed to load RuleSpec module",
            "policies/x.yaml: ci: Proof source evidence not found: rule `max`",
            "regulations/7-cfr/273/4.yaml: ci: 4.test.yaml YAML parse failed: x",
            "statutes/7/2017/a.yaml: ci: Test case `one_person` output "
            "`allotment` expected 23 got 24",
            "Generated target must directly import at least one exported symbol",
            "policies/x.yaml: ci: Source scope mismatch: `expanded` is wider",
            # A bracketed provision label is not a rule code.
            "policies/x.yaml: ci: missing [Absatz a, Nummer 4] coverage",
            None,
            42,
        ],
    )
    def test_unnamed_issues_are_unknown(self, issue: object) -> None:
        assert budget_mod.issue_rule_code(issue) is None

    def test_union_across_lanes(self) -> None:
        assert budget_mod.rule_codes_from_issue_payloads(
            [{"issues": [TESTS_ISSUE]}, {"issues": [STRUCTURE_ISSUE, TESTS_ISSUE]}]
        ) == frozenset({"complete-source-unit:tests", "complete-source-unit:structure"})

    @pytest.mark.parametrize(
        "payloads",
        [
            [],
            [{"issues": []}],
            [{"issues": "not a list"}],
            ["not an object"],
            [{"issues": [TESTS_ISSUE]}, {"issues": ["Unclassified failure"]}],
        ],
    )
    def test_any_unknown_lane_makes_the_set_unknown(self, payloads: list) -> None:
        assert budget_mod.rule_codes_from_issue_payloads(payloads) is None


def _failure_artifact(
    tmp_path: Path,
    members: dict[str, bytes],
    *,
    symlink: str | None = None,
    extra_zip_member: bool = False,
) -> str:
    packaged = io.BytesIO()
    with tarfile.open(fileobj=packaged, mode="w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if symlink is not None:
            info = tarfile.TarInfo(symlink)
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            archive.addfile(info)
    zip_path = tmp_path / "artifact.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(budget_mod.FAILURE_ARCHIVE_NAME, packaged.getvalue())
        if extra_zip_member:
            archive.writestr("other.txt", b"x")
    return str(zip_path)


def _issues(*issues: str) -> bytes:
    return json.dumps({"issues": list(issues), "citation": CITATION}).encode()


class TestFailureArtifact:
    LANE = "./generated/target-preflight/final-rejected-candidate/issues.json"

    def test_reads_every_lane_and_ignores_other_files(self, tmp_path: Path) -> None:
        zip_path = _failure_artifact(
            tmp_path,
            {
                "./metadata.json": b"{}",
                self.LANE: _issues(TESTS_ISSUE),
                "./generated/dependent/final-rejected-candidate/issues.json": _issues(
                    STRUCTURE_ISSUE
                ),
                "./generated/target/openai-gpt-5.6/issues.json": _issues(
                    "Unclassified"
                ),
            },
        )
        assert budget_mod.rule_codes_from_failure_artifact(zip_path) == frozenset(
            {"complete-source-unit:tests", "complete-source-unit:structure"}
        )

    def test_no_final_candidate_is_unknown(self, tmp_path: Path) -> None:
        zip_path = _failure_artifact(tmp_path, {"./metadata.json": b"{}"})
        assert budget_mod.rule_codes_from_failure_artifact(zip_path) is None

    def test_symlinked_issues_file_is_unknown(self, tmp_path: Path) -> None:
        zip_path = _failure_artifact(tmp_path, {}, symlink=self.LANE)
        assert budget_mod.rule_codes_from_failure_artifact(zip_path) is None

    def test_oversized_issues_file_is_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(budget_mod, "MAX_ISSUES_BYTES", 16)
        zip_path = _failure_artifact(tmp_path, {self.LANE: _issues(TESTS_ISSUE)})
        assert budget_mod.rule_codes_from_failure_artifact(zip_path) is None

    def test_invalid_json_is_unknown(self, tmp_path: Path) -> None:
        zip_path = _failure_artifact(tmp_path, {self.LANE: b"{broken"})
        assert budget_mod.rule_codes_from_failure_artifact(zip_path) is None

    def test_unexpected_zip_layout_is_unknown(self, tmp_path: Path) -> None:
        zip_path = _failure_artifact(
            tmp_path, {self.LANE: _issues(TESTS_ISSUE)}, extra_zip_member=True
        )
        assert budget_mod.rule_codes_from_failure_artifact(zip_path) is None


class TestEvaluateRepeatedFailure:
    TESTS = frozenset({"complete-source-unit:tests"})
    BOTH = frozenset({"complete-source-unit:tests", "complete-source-unit:structure"})

    def _runs(self, *ids: int) -> list[dict]:
        return [{"id": run_id} for run_id in ids]

    def test_two_newest_failures_with_one_set_repeat(self) -> None:
        codes = {3: self.TESTS, 2: self.TESTS, 1: self.BOTH}
        repeated = budget_mod.evaluate_repeated_failure(
            self._runs(3, 2, 1), lambda run: codes[run["id"]]
        )
        assert repeated is not None
        assert repeated.rule_codes == self.TESTS
        assert [run["id"] for run in repeated.runs] == [3, 2]

    def test_different_sets_do_not_repeat(self) -> None:
        codes = {3: self.TESTS, 2: self.BOTH}
        assert (
            budget_mod.evaluate_repeated_failure(
                self._runs(3, 2), lambda run: codes[run["id"]]
            )
            is None
        )

    def test_an_unknown_set_does_not_repeat(self) -> None:
        codes = {3: self.TESTS, 2: None}
        assert (
            budget_mod.evaluate_repeated_failure(
                self._runs(3, 2), lambda run: codes[run["id"]]
            )
            is None
        )

    def test_one_failure_does_not_repeat(self) -> None:
        assert (
            budget_mod.evaluate_repeated_failure(self._runs(3), lambda run: self.TESTS)
            is None
        )


class TestRepeatedFailureMain:
    _set_env = TestMainExitContract._set_env
    _stub_api = TestMainExitContract._stub_api
    TWO_FAILURES = [
        _run(run_id, f"2026-08-12T0{run_id}:00:00Z", "failure") for run_id in (1, 2)
    ]
    TESTS = frozenset({"complete-source-unit:tests"})

    def _repeat(self, monkeypatch: pytest.MonkeyPatch, codes: dict) -> None:
        self._stub_api(monkeypatch, self.TWO_FAILURES, rule_codes=codes)

    def test_same_rule_set_blocks_under_the_numeric_budget(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._set_env(monkeypatch)
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        self._repeat(monkeypatch, {1: self.TESTS, 2: self.TESTS})
        assert budget_mod.main() == 1
        text = summary.read_text()
        assert "### Repeated failure" in text
        assert "`complete-source-unit:tests`" in text
        assert "REPEATED_FAILURE_OVERRIDE_JSON" in text
        assert "::error title=Repeated final rule codes::" in capsys.readouterr().out

    def test_repair_replays_are_also_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_env(monkeypatch, repair_run_id="32201076681")
        self._repeat(monkeypatch, {1: self.TESTS, 2: self.TESTS})
        assert budget_mod.main() == 1

    def test_different_rule_sets_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch)
        self._repeat(
            monkeypatch,
            {1: self.TESTS, 2: frozenset({"complete-source-unit:structure"})},
        )
        assert budget_mod.main() == 0

    def test_unknown_rule_set_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch)
        self._repeat(monkeypatch, {1: self.TESTS})
        assert budget_mod.main() == 0

    def test_a_success_in_between_resets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch)
        history = [
            _run(1, "2026-08-12T01:00:00Z", "failure"),
            _run(2, "2026-08-12T02:00:00Z", "success"),
            _run(3, "2026-08-12T03:00:00Z", "failure"),
        ]
        self._stub_api(
            monkeypatch,
            history,
            jobs=[TestEncodeJobChecker._encode_job("success")],
            rule_codes={1: self.TESTS, 3: self.TESTS},
        )
        assert budget_mod.main() == 0

    def test_lookup_error_fails_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch)
        self._stub_api(monkeypatch, self.TWO_FAILURES)

        def boom(**kwargs: object) -> frozenset[str]:
            raise RuntimeError("artifact API down")

        monkeypatch.setattr(budget_mod, "_fetch_failure_rule_codes", boom)
        assert budget_mod.main() == 0

    def test_queue_dispatch_reports_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_env(monkeypatch, queue_id="q-123")
        self._repeat(monkeypatch, {1: self.TESTS, 2: self.TESTS})
        assert budget_mod.main() == 0

    def test_model_free_metadata_migration_reports_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_env(monkeypatch)
        monkeypatch.setenv(
            "SOURCE_BUNDLE_JSON",
            json.dumps(
                {
                    "schema_version": (
                        "axiom-encode/retired-source-metadata-migration-plan/v2"
                    )
                }
            ),
        )
        self._repeat(monkeypatch, {1: self.TESTS, 2: self.TESTS})
        assert budget_mod.main() == 0

    def test_repository_override_lifts_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_env(monkeypatch, override="true")
        self._repeat(monkeypatch, {1: self.TESTS, 2: self.TESTS})
        assert budget_mod.main() == 0

    def test_exact_citation_override_lifts_it_until_expiry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_env(monkeypatch)
        self._repeat(monkeypatch, {1: self.TESTS, 2: self.TESTS})
        entry = {"reason": "Prompt fix for paired tests landed in #1900"}
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        monkeypatch.setenv(
            "REPEATED_FAILURE_OVERRIDE_JSON",
            json.dumps({CITATION: {**entry, "expires_at": future}}),
        )
        assert budget_mod.main() == 0
        monkeypatch.setenv(
            "REPEATED_FAILURE_OVERRIDE_JSON",
            json.dumps({CITATION: {**entry, "expires_at": past}}),
        )
        assert budget_mod.main() == 1
        monkeypatch.setenv(
            "REPEATED_FAILURE_OVERRIDE_JSON",
            json.dumps({DE_CITATION: {**entry, "expires_at": future}}),
        )
        assert budget_mod.main() == 1

    @pytest.mark.parametrize(
        "raw",
        [
            "{broken",
            "[]",
            json.dumps({CITATION: {"expires_at": "2099-01-01T00:00:00Z"}}),
            json.dumps({CITATION: {"expires_at": "2099-01-01", "reason": "x"}}),
            json.dumps(
                {CITATION: {"expires_at": "2099-01-01T00:00:00Z", "reason": " "}}
            ),
        ],
    )
    def test_invalid_override_keeps_the_rule(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        self._set_env(monkeypatch)
        monkeypatch.setenv("REPEATED_FAILURE_OVERRIDE_JSON", raw)
        self._repeat(monkeypatch, {1: self.TESTS, 2: self.TESTS})
        assert budget_mod.main() == 1


class TestFetchFailureRuleCodes:
    """The bearer token must never follow the redirect to blob storage."""

    def test_follows_the_redirect_without_the_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        zip_bytes = Path(
            _failure_artifact(
                tmp_path,
                {TestFailureArtifact.LANE: _issues(TESTS_ISSUE)},
            )
        ).read_bytes()
        name = "targeted-reencode-failure-7-1"
        seen: list[urllib.request.Request] = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                self.close()

        def urlopen(request, timeout=None):  # noqa: ANN001
            seen.append(request)
            if "/actions/runs/7/artifacts" in request.full_url:
                return Response(
                    json.dumps(
                        {
                            "artifacts": [
                                {
                                    "name": name,
                                    "expired": False,
                                    "size_in_bytes": len(zip_bytes),
                                    "archive_download_url": (
                                        "https://api.github.com/repos/org/repo/"
                                        "actions/artifacts/9/zip"
                                    ),
                                }
                            ]
                        }
                    ).encode()
                )
            assert request.full_url == "https://blob.example/artifact.zip"
            return Response(zip_bytes)

        class Opener:
            def open(self, request, timeout=None):  # noqa: ANN001
                seen.append(request)
                raise urllib.error.HTTPError(
                    request.full_url,
                    302,
                    "Found",
                    {"Location": "https://blob.example/artifact.zip"},
                    None,
                )

        monkeypatch.setattr(budget_mod.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(
            budget_mod.urllib.request, "build_opener", lambda *handlers: Opener()
        )
        codes = budget_mod._fetch_failure_rule_codes(
            repo="org/repo", token="secret-token", run={"id": 7, "run_attempt": 1}
        )
        assert codes == frozenset({"complete-source-unit:tests"})
        assert "name=targeted-reencode-failure-7-1" in seen[0].full_url
        assert seen[1].get_header("Authorization") == "Bearer secret-token"
        assert seen[2].get_header("Authorization") is None


def test_workflow_passes_the_repeated_failure_override() -> None:
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / ".github/workflows/targeted-signed-reencode.yml"
        ).read_text()
    )
    step = next(
        step
        for step in workflow["jobs"]["attempt_budget"]["steps"]
        if step.get("name") == "Enforce failed-attempt budget"
    )
    assert step["env"]["REPEATED_FAILURE_OVERRIDE_JSON"] == (
        "${{ vars.REPEATED_FAILURE_OVERRIDE_JSON }}"
    )
    assert step["env"]["REPAIR_RUN_ID"] == "${{ inputs.repair_run_id }}"


def test_no_redirect_handler_stops_at_the_redirect_on_a_real_server() -> None:
    """Python's own redirect handler would forward the Authorization header."""

    requests: list[tuple[str, str | None]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            requests.append((self.path, self.headers.get("Authorization")))
            if self.path == "/zip":
                self.send_response(302)
                self.send_header(
                    "Location", f"http://127.0.0.1:{self.server.server_port}/blob"
                )
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"blob")

        def log_message(self, *args: object) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        opener = urllib.request.build_opener(budget_mod._NoRedirect)
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/zip",
            headers={"Authorization": "Bearer secret-token"},
        )
        with pytest.raises(urllib.error.HTTPError) as raised:
            opener.open(request, timeout=10)
        assert raised.value.code == 302
        assert raised.value.headers["Location"].endswith("/blob")
        raised.value.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
    assert requests == [("/zip", "Bearer secret-token")]


def test_a_spent_lookup_deadline_skips_the_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched: list[str] = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def urlopen(request, timeout=None):  # noqa: ANN001
        fetched.append(request.full_url)
        return Response(
            json.dumps(
                {
                    "artifacts": [
                        {
                            "name": "targeted-reencode-failure-7-1",
                            "expired": False,
                            "size_in_bytes": 10,
                            "archive_download_url": (
                                "https://api.github.com/repos/org/repo/"
                                "actions/artifacts/9/zip"
                            ),
                        }
                    ]
                }
            ).encode()
        )

    class Opener:
        def open(self, request, timeout=None):  # noqa: ANN001
            raise urllib.error.HTTPError(
                request.full_url,
                302,
                "Found",
                {"Location": "https://blob.example/artifact.zip"},
                None,
            )

    monkeypatch.setattr(budget_mod.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(
        budget_mod.urllib.request, "build_opener", lambda *handlers: Opener()
    )
    codes = budget_mod._fetch_failure_rule_codes(
        repo="org/repo",
        token="secret-token",
        run={"id": 7, "run_attempt": 1},
        deadline=time.monotonic() - 1,
    )
    assert codes is None
    assert all("blob.example" not in url for url in fetched)
