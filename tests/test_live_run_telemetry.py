"""Tests for live encode-run presence telemetry."""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from axiom_encode import live_run_telemetry
from axiom_encode.live_run_telemetry import (
    ENCODE_PHASES,
    PHASE_APPLY,
    PHASE_GENERATE,
    PHASE_RESOLVE,
    PHASE_REVIEW,
    PHASE_VALIDATE,
    LiveRunTelemetry,
    abnormal_exit_outcome,
    first_typed_check_id,
    github_run_identity,
    report_phase,
    runner_identity,
    telemetry_mode,
)

_GITHUB_ENV_VARS = (
    "GITHUB_ACTIONS",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_SERVER_URL",
    "GITHUB_REPOSITORY",
    "GITHUB_WORKFLOW",
)


def _clear_github_env(monkeypatch):
    # The suite itself runs in GitHub Actions; identity tests must not see
    # the real run.
    for name in _GITHUB_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _github_env(monkeypatch, **overrides):
    _clear_github_env(monkeypatch)
    values = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_RUN_ID": "12345678901",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "TheAxiomFoundation/axiom-encode",
        "GITHUB_WORKFLOW": "Targeted signed re-encode",
    }
    values.update(overrides)
    for name, value in values.items():
        if value is not None:
            monkeypatch.setenv(name, value)


def _mock_client():
    client = MagicMock()
    table = client.schema.return_value.table.return_value
    table.insert.return_value.execute.return_value = MagicMock()
    query = table.update.return_value
    query.eq.return_value = query
    query.execute.return_value = MagicMock()
    return client, table


def _configured_env(monkeypatch):
    monkeypatch.setenv("AXIOM_ENCODE_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("AXIOM_ENCODE_SUPABASE_SECRET_KEY", "secret")
    # These tests exercise the enabled path with mocked transports; the
    # explicit "on" override is the only way past the in-test detection.
    monkeypatch.setenv("AXIOM_ENCODE_TELEMETRY", "on")


def _ingest_env(monkeypatch):
    monkeypatch.delenv("AXIOM_ENCODE_SUPABASE_URL", raising=False)
    monkeypatch.delenv("AXIOM_ENCODE_SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.setenv("AXIOM_ENCODE_TELEMETRY", "on")


def _mock_urlopen(status=204):
    response = MagicMock()
    response.status = status
    response.__enter__ = lambda self: self
    response.__exit__ = lambda self, *args: None
    return patch(
        "axiom_encode.live_run_telemetry.urllib.request.urlopen",
        return_value=response,
    )


def _ingest_payloads(urlopen_mock):
    return [
        json.loads(call.args[0].data.decode("utf-8"))
        for call in urlopen_mock.call_args_list
    ]


def _phase_updates(table):
    """The phase carried by each live-row update, in order (None if absent)."""
    return [call.args[0].get("phase") for call in table.update.call_args_list]


def _wait_for_calls(mock, count, timeout=5.0, *, exact=True):
    """Wait for the heartbeat thread to have sent ``count`` requests."""
    deadline = time.monotonic() + timeout
    while mock.call_count < count and time.monotonic() < deadline:
        time.sleep(0.002)
    if exact:
        assert mock.call_count == count, (mock.call_count, count)
    else:
        assert mock.call_count >= count, (mock.call_count, count)


class TestRunnerIdentity:
    def test_contains_machine_fields(self, monkeypatch):
        _clear_github_env(monkeypatch)
        identity = runner_identity()
        assert set(identity) == {"hostname", "username", "platform", "pid", "is_ci"}
        assert isinstance(identity["pid"], int)
        assert isinstance(identity["is_ci"], bool)

    def test_carries_github_run_identity_inside_actions(self, monkeypatch):
        _github_env(monkeypatch)
        identity = runner_identity()
        assert identity["is_ci"] is True
        assert identity["github_run_id"] == "12345678901"
        assert identity["github_run_attempt"] == 2
        assert identity["github_run_url"] == (
            "https://github.com/TheAxiomFoundation/axiom-encode"
            "/actions/runs/12345678901"
        )
        assert identity["github_workflow"] == "Targeted signed re-encode"


class TestGithubRunIdentity:
    def test_full_actions_environment(self, monkeypatch):
        _github_env(monkeypatch)
        assert github_run_identity() == {
            "github_run_id": "12345678901",
            "github_run_attempt": 2,
            "github_run_url": (
                "https://github.com/TheAxiomFoundation/axiom-encode"
                "/actions/runs/12345678901"
            ),
            "github_workflow": "Targeted signed re-encode",
        }

    def test_absent_outside_actions(self, monkeypatch):
        _clear_github_env(monkeypatch)
        assert github_run_identity() == {}

    def test_run_vars_without_github_actions_flag_are_ignored(self, monkeypatch):
        _github_env(monkeypatch, GITHUB_ACTIONS=None)
        assert github_run_identity() == {}
        _github_env(monkeypatch, GITHUB_ACTIONS="false")
        assert github_run_identity() == {}

    def test_missing_or_malformed_run_id_reports_nothing(self, monkeypatch):
        _github_env(monkeypatch, GITHUB_RUN_ID=None)
        assert github_run_identity() == {}
        for bad in ("", "abc", "12 34", "-1", "1" * 21):
            _github_env(monkeypatch, GITHUB_RUN_ID=bad)
            assert github_run_identity() == {}, bad

    def test_partial_environment_reports_only_what_is_well_formed(self, monkeypatch):
        _github_env(
            monkeypatch,
            GITHUB_RUN_ATTEMPT=None,
            GITHUB_SERVER_URL=None,
            GITHUB_WORKFLOW=None,
        )
        assert github_run_identity() == {"github_run_id": "12345678901"}

        _github_env(monkeypatch, GITHUB_REPOSITORY=None)
        identity = github_run_identity()
        assert "github_run_url" not in identity
        assert identity["github_run_attempt"] == 2

        for bad_attempt in ("0", "x", "", "-2"):
            _github_env(monkeypatch, GITHUB_RUN_ATTEMPT=bad_attempt)
            assert "github_run_attempt" not in github_run_identity(), bad_attempt

        _github_env(monkeypatch, GITHUB_SERVER_URL="not a url")
        assert "github_run_url" not in github_run_identity()
        _github_env(monkeypatch, GITHUB_REPOSITORY="no-slash")
        assert "github_run_url" not in github_run_identity()

    def test_enterprise_server_url_and_trailing_slash(self, monkeypatch):
        _github_env(
            monkeypatch,
            GITHUB_SERVER_URL="https://ghe.example.com:8443/",
            GITHUB_RUN_ATTEMPT="1",
        )
        identity = github_run_identity()
        assert identity["github_run_url"] == (
            "https://ghe.example.com:8443/TheAxiomFoundation/axiom-encode"
            "/actions/runs/12345678901"
        )
        assert identity["github_run_attempt"] == 1

    def test_workflow_name_is_bounded(self, monkeypatch):
        _github_env(monkeypatch, GITHUB_WORKFLOW="w" * 500)
        assert github_run_identity()["github_workflow"] == "w" * 120


class TestTelemetryMode:
    def test_defaults_to_ingest_without_credentials(self, monkeypatch):
        _ingest_env(monkeypatch)
        assert telemetry_mode() == "ingest"

    def test_direct_with_credentials(self, monkeypatch):
        _configured_env(monkeypatch)
        assert telemetry_mode() == "direct"

    def test_off_under_pytest_or_explicit_optout(self, monkeypatch):
        _ingest_env(monkeypatch)
        monkeypatch.setenv("AXIOM_ENCODE_TELEMETRY", "off")
        assert telemetry_mode() == "off"
        monkeypatch.setenv("AXIOM_ENCODE_TELEMETRY", "false")
        assert telemetry_mode() == "off"
        # Without the explicit "on" override, in-process test detection wins
        # even when hermetic tests have scrubbed every env marker: the
        # pytest module itself is the signal.
        monkeypatch.delenv("AXIOM_ENCODE_TELEMETRY")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        assert telemetry_mode() == "off"

    def test_explicit_off_beats_explicit_on_semantics(self, monkeypatch):
        _ingest_env(monkeypatch)
        assert telemetry_mode() == "ingest"
        monkeypatch.setenv("AXIOM_ENCODE_TELEMETRY", "disabled")
        assert telemetry_mode() == "off"


class TestLiveRunTelemetry:
    def test_ingest_mode_reports_lifecycle_without_credentials(self, monkeypatch):
        _ingest_env(monkeypatch)
        with _mock_urlopen() as urlopen_mock:
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.2.1677",
            ) as live:
                live.set_attempt(2, "gpt-5.5-max")
                live.finish("completed", run_id="abc12345")

        payloads = _ingest_payloads(urlopen_mock)
        assert payloads[0]["op"] == "start"
        assert payloads[0]["citation"] == "us/statute/26/32"
        assert payloads[0]["runner"]["hostname"] == runner_identity()["hostname"]
        # Finish repeats current state even if the heartbeat took the change
        # but has not delivered it yet.
        carriers = [payload for payload in payloads[1:] if "attempt" in payload]
        assert carriers
        assert all(payload["attempt"] == 2 for payload in carriers)
        assert all(payload["model"] == "gpt-5.5-max" for payload in carriers)
        assert payloads[-1]["op"] == "finish"
        assert payloads[-1]["status"] == "completed"
        assert payloads[-1]["run_id"] == "abc12345"
        request = urlopen_mock.call_args_list[0].args[0]
        assert request.full_url.startswith("https://axiom.org/")

    def test_ingest_url_override(self, monkeypatch):
        _ingest_env(monkeypatch)
        monkeypatch.setenv(
            "AXIOM_ENCODE_TELEMETRY_INGEST_URL", "https://staging.example/ingest"
        )
        with _mock_urlopen() as urlopen_mock:
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.0.0",
            ):
                pass
        assert urlopen_mock.call_args_list[0].args[0].full_url == (
            "https://staging.example/ingest"
        )

    def test_ingest_start_failure_disables_telemetry(self, monkeypatch):
        _ingest_env(monkeypatch)
        with patch(
            "axiom_encode.live_run_telemetry.urllib.request.urlopen",
            side_effect=OSError("unreachable"),
        ) as urlopen_mock:
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.0.0",
            ) as live:
                live.finish("completed")
        # Only the failed start attempt — no heartbeat or finish posts after.
        assert urlopen_mock.call_count == 1

    def test_noop_under_pytest_even_with_credentials(self, monkeypatch):
        monkeypatch.setenv("AXIOM_ENCODE_SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("AXIOM_ENCODE_SUPABASE_SECRET_KEY", "secret")
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_x.py::test_y")
        with patch("axiom_encode.supabase_sync.get_supabase_client") as mock_get:
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.0.0",
            ) as live:
                assert live._client is None
        mock_get.assert_not_called()

    def test_noop_when_disabled(self, monkeypatch):
        _configured_env(monkeypatch)
        with patch("axiom_encode.supabase_sync.get_supabase_client") as mock_get:
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.0.0",
                enabled=False,
            ) as live:
                assert live._client is None
        mock_get.assert_not_called()

    def test_inserts_running_row_and_finishes_completed(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.1.0",
            ) as live:
                live.finish("completed", run_id="abc12345")

        inserted = table.insert.call_args[0][0]
        assert inserted["citation"] == "us/statute/26/32"
        assert inserted["status"] == "running"
        assert inserted["backend"] == "codex"
        assert inserted["model"] == "gpt-5.5"
        assert inserted["runner"]["hostname"] == runner_identity()["hostname"]
        assert inserted["id"].startswith("live-")

        finished = table.update.call_args[0][0]
        assert finished["status"] == "completed"
        assert finished["run_id"] == "abc12345"
        assert finished["finished_at"]

    def test_exit_without_finish_marks_failed(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            try:
                with LiveRunTelemetry(
                    citation="us/statute/26/32",
                    backend="codex",
                    model="gpt-5.5",
                    encoder_version="0.1.0",
                ):
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                pass

        finished = table.update.call_args[0][0]
        assert finished["status"] == "failed"
        assert "run_id" not in finished

    def test_finish_is_idempotent(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.1.0",
            ) as live:
                live.finish("failed")
        # __exit__ must not overwrite the explicit finish.
        assert table.update.call_count == 1
        assert table.update.call_args[0][0]["status"] == "failed"

    def test_set_attempt_updates_row(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.1.0",
            ) as live:
                live.set_attempt(2, "gpt-5.5-max")
                live.finish("completed")
        attempt_update = table.update.call_args_list[0][0][0]
        assert attempt_update["attempt"] == 2
        assert attempt_update["model"] == "gpt-5.5-max"
        assert attempt_update["last_heartbeat_at"]

    def test_insert_failure_disables_telemetry(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        table.insert.return_value.execute.side_effect = RuntimeError("supabase down")
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="codex",
                model="gpt-5.5",
                encoder_version="0.1.0",
            ) as live:
                assert live._client is None
                live.finish("completed")
        table.update.assert_not_called()

    def test_direct_start_row_carries_github_run_identity(self, monkeypatch):
        _configured_env(monkeypatch)
        _github_env(monkeypatch)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="openai",
                model="gpt-5.5",
                encoder_version="0.1.0",
            ):
                pass
        runner = table.insert.call_args[0][0]["runner"]
        assert runner["github_run_id"] == "12345678901"
        assert runner["github_run_attempt"] == 2
        assert runner["github_run_url"].endswith("/actions/runs/12345678901")

    def test_ingest_start_carries_github_run_identity(self, monkeypatch):
        _ingest_env(monkeypatch)
        _github_env(monkeypatch)
        with _mock_urlopen() as urlopen_mock:
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="openai",
                model="gpt-5.5",
                encoder_version="0.1.0",
            ):
                pass
        start = _ingest_payloads(urlopen_mock)[0]
        assert start["runner"]["github_run_id"] == "12345678901"
        assert start["runner"]["github_run_attempt"] == 2


class TestLiveRunPhases:
    def test_phase_names_are_short_stable_lowercase(self):
        assert ENCODE_PHASES == ("resolve", "generate", "validate", "review", "apply")
        for phase in ENCODE_PHASES:
            assert phase == phase.lower()
            assert phase.isalpha()

    def test_start_row_carries_initial_phase(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="openai",
                model="gpt-5.5",
                encoder_version="0.1.0",
                phase=PHASE_RESOLVE,
            ):
                pass
        assert table.insert.call_args[0][0]["phase"] == "resolve"

    def test_transitions_send_one_update_each_and_repeats_are_free(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="openai",
                model="gpt-5.5",
                encoder_version="0.1.0",
                phase=PHASE_RESOLVE,
            ) as live:
                sent = 0
                steps = (
                    (PHASE_RESOLVE, False),  # already there: no update
                    (PHASE_GENERATE, True),
                    (PHASE_VALIDATE, True),
                    (PHASE_VALIDATE, False),  # a repair round: no update
                    (PHASE_REVIEW, True),
                    # A validator retry restarts from resolve in the same
                    # update that records the new attempt.
                    ("retry", True),
                    (PHASE_GENERATE, True),
                    (PHASE_VALIDATE, True),
                    (PHASE_APPLY, True),
                )
                for step, transition in steps:
                    if step == "retry":
                        live.set_attempt(2, "gpt-5.5-max", phase=PHASE_RESOLVE)
                    else:
                        report_phase(step)
                    sent += transition
                    _wait_for_calls(table.update, sent)
                live.finish("completed", run_id="abc12345")

        assert _phase_updates(table) == [
            "generate",
            "validate",
            "review",
            "resolve",
            "generate",
            "validate",
            "apply",
            "apply",  # the finish records the final phase
        ]
        retry_update = table.update.call_args_list[3].args[0]
        assert retry_update["attempt"] == 2
        assert retry_update["model"] == "gpt-5.5-max"
        for call in table.update.call_args_list[:-1]:
            assert set(call.args[0]) <= {
                "phase",
                "attempt",
                "model",
                "last_heartbeat_at",
            }

    def test_ingest_phase_transition_is_one_heartbeat(self, monkeypatch):
        _ingest_env(monkeypatch)
        with _mock_urlopen() as urlopen_mock:
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="openai",
                model="gpt-5.5",
                encoder_version="0.1.0",
                phase=PHASE_RESOLVE,
            ) as live:
                report_phase(PHASE_GENERATE)
                _wait_for_calls(urlopen_mock, 2)
                report_phase(PHASE_GENERATE)
                live.finish("completed")
        payloads = _ingest_payloads(urlopen_mock)
        assert [payload["op"] for payload in payloads] == [
            "start",
            "heartbeat",
            "finish",
        ]
        assert payloads[0]["phase"] == "resolve"
        assert payloads[1] == {"op": "heartbeat", "id": live.id, "phase": "generate"}

    def test_heartbeat_repeats_current_phase(self, monkeypatch):
        _configured_env(monkeypatch)
        monkeypatch.setattr(live_run_telemetry, "HEARTBEAT_INTERVAL_SECONDS", 0.01)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="openai",
                model="gpt-5.5",
                encoder_version="0.1.0",
                phase=PHASE_VALIDATE,
            ):
                _wait_for_calls(table.update, 2, exact=False)
        heartbeat = table.update.call_args_list[0].args[0]
        assert heartbeat["phase"] == "validate"
        assert heartbeat["last_heartbeat_at"]
        assert "attempt" not in heartbeat

    def test_heartbeat_without_phase_sends_only_liveness(self, monkeypatch):
        _configured_env(monkeypatch)
        monkeypatch.setattr(live_run_telemetry, "HEARTBEAT_INTERVAL_SECONDS", 0.01)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="openai",
                model="gpt-5.5",
                encoder_version="0.1.0",
            ):
                _wait_for_calls(table.update, 1, exact=False)
        assert set(table.update.call_args_list[0].args[0]) == {"last_heartbeat_at"}

    def test_phase_changes_never_wait_on_the_network(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        release = threading.Event()
        in_flight = threading.Event()

        def slow_update():
            in_flight.set()
            release.wait(10)
            return MagicMock()

        table.update.return_value.eq.return_value.execute.side_effect = slow_update
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="openai",
                model="gpt-5.5",
                encoder_version="0.1.0",
                phase=PHASE_RESOLVE,
            ):
                report_phase(PHASE_GENERATE)
                assert in_flight.wait(5)
                # The heartbeat thread is stuck on a slow Supabase; the encode
                # thread's transitions still return immediately.
                started = time.monotonic()
                report_phase(PHASE_VALIDATE)
                report_phase(PHASE_REVIEW)
                assert time.monotonic() - started < 0.5
                release.set()
                # The stalled transitions coalesce into one update, in order.
                _wait_for_calls(table.update, 2)
        assert _phase_updates(table)[:2] == ["generate", "review"]

    def test_delayed_heartbeat_cannot_overwrite_finished_row(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        row = {}
        row_lock = threading.Lock()
        in_flight = threading.Event()
        release = threading.Event()

        class Query:
            def __init__(self, data, *, insert=False):
                self.data = dict(data)
                self.insert = insert
                self.filters = []

            def eq(self, key, value):
                self.filters.append((key, value))
                return self

            def execute(self):
                if not self.insert and "status" not in self.data:
                    in_flight.set()
                    if not release.wait(10):
                        raise TimeoutError("test heartbeat was not released")
                # PostgREST evaluates predicates when the write executes, not
                # when the client constructs a request before the network wait.
                with row_lock:
                    if self.insert or all(
                        row.get(key) == value for key, value in self.filters
                    ):
                        row.update(self.data)
                return MagicMock()

        table.insert.side_effect = lambda data: Query(data, insert=True)
        table.update.side_effect = Query
        live = LiveRunTelemetry(
            citation="us/statute/26/32",
            backend="codex",
            model="initial-model",
            encoder_version="0.1.0",
            phase=PHASE_RESOLVE,
        )
        try:
            with patch(
                "axiom_encode.supabase_sync.get_supabase_client", return_value=client
            ):
                live.__enter__()
                live.set_attempt(2, "escalated-model", phase=PHASE_GENERATE)
                assert in_flight.wait(5)
                live.set_phase(PHASE_APPLY)
                started = time.monotonic()
                live.finish("completed", run_id="final-run")
                assert time.monotonic() - started < 4
                assert live._thread.is_alive()
                terminal = dict(row)
                assert terminal["status"] == "completed"
                assert terminal["phase"] == PHASE_APPLY
                assert terminal["attempt"] == 2
                assert terminal["model"] == "escalated-model"
                assert terminal["run_id"] == "final-run"
                assert terminal["started_at"]
                assert terminal["finished_at"]
                assert terminal["last_heartbeat_at"]
                release.set()
                live._thread.join(timeout=5)
                assert not live._thread.is_alive()
                assert row == terminal
        finally:
            release.set()
            live.finish("failed")
            if live._thread is not None:
                live._thread.join(timeout=5)

    def test_report_phase_is_a_noop_without_an_active_run(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="openai",
                model="gpt-5.5",
                encoder_version="0.1.0",
            ) as live:
                assert live_run_telemetry._active_run is live
            # Finished runs deregister; later reports go nowhere.
            assert live_run_telemetry._active_run is None
            update_count = table.update.call_count
            report_phase(PHASE_GENERATE)
        assert table.update.call_count == update_count

    def test_report_phase_is_a_noop_when_telemetry_is_off(self, monkeypatch):
        # No explicit "on": in-test detection turns telemetry off, so the run
        # never registers and phase reports never reach a transport.
        monkeypatch.delenv("AXIOM_ENCODE_TELEMETRY", raising=False)
        with patch("axiom_encode.supabase_sync.get_supabase_client") as mock_get:
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="openai",
                model="gpt-5.5",
                encoder_version="0.1.0",
                phase=PHASE_RESOLVE,
            ):
                assert live_run_telemetry._active_run is None
                report_phase(PHASE_GENERATE)
        mock_get.assert_not_called()

    def test_failed_start_does_not_register(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        table.insert.return_value.execute.side_effect = RuntimeError("down")
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with LiveRunTelemetry(
                citation="us/statute/26/32",
                backend="openai",
                model="gpt-5.5",
                encoder_version="0.1.0",
            ):
                assert live_run_telemetry._active_run is None
                report_phase(PHASE_GENERATE)
        table.update.assert_not_called()


class _MissingColumnError(Exception):
    """PostgREST's unknown-column error, as supabase-py raises it."""

    code = "PGRST204"


def _live_run():
    return LiveRunTelemetry(
        citation="us/statute/26/32",
        backend="codex",
        model="gpt-5.5",
        encoder_version="0.1.0",
    )


def _finish_via(monkeypatch, transport, body):
    """Run ``body(live)`` inside a live run; return the finish update sent."""
    if transport == "direct":
        _configured_env(monkeypatch)
        client, table = _mock_client()
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            try:
                with _live_run() as live:
                    body(live)
            except BaseException:
                pass
        finishes = [
            call.args[0]
            for call in table.update.call_args_list
            if "finished_at" in call.args[0]
        ]
    else:
        _ingest_env(monkeypatch)
        with _mock_urlopen() as urlopen_mock:
            try:
                with _live_run() as live:
                    body(live)
            except BaseException:
                pass
        finishes = [
            payload
            for payload in _ingest_payloads(urlopen_mock)
            if payload["op"] == "finish"
        ]
    assert len(finishes) == 1, finishes
    return finishes[0]


def _raise(error):
    def body(_live):
        raise error

    return body


@pytest.mark.parametrize("transport", ["direct", "ingest"])
class TestTerminalOutcome:
    def test_normal_finish_carries_outcome_fields(self, monkeypatch, transport):
        finished = _finish_via(
            monkeypatch,
            transport,
            lambda live: live.finish(
                "failed",
                run_id="abc12345",
                outcome_status="apply_blocked_validation",
                error_rule="complete-source-unit:tests/output-unasserted",
            ),
        )
        assert finished["status"] == "failed"
        assert finished["run_id"] == "abc12345"
        assert finished["outcome_status"] == "apply_blocked_validation"
        assert finished["error_rule"] == "complete-source-unit:tests/output-unasserted"

    def test_successful_finish_sends_no_error_rule(self, monkeypatch, transport):
        finished = _finish_via(
            monkeypatch,
            transport,
            lambda live: live.finish(
                "completed", run_id="abc12345", outcome_status="apply_applied"
            ),
        )
        assert finished["outcome_status"] == "apply_applied"
        assert "error_rule" not in finished

    def test_finish_without_outcome_sends_neither_field(self, monkeypatch, transport):
        finished = _finish_via(
            monkeypatch, transport, lambda live: live.finish("completed")
        )
        assert "outcome_status" not in finished
        assert "error_rule" not in finished

    def test_escaped_exception_is_aborted_with_its_class(self, monkeypatch, transport):
        secret = "/Users/someone/secret/path could not be read"
        finished = _finish_via(monkeypatch, transport, _raise(RuntimeError(secret)))
        assert finished["status"] == "failed"
        assert finished["outcome_status"] == "aborted"
        assert finished["error_rule"] == "exception:RuntimeError"
        assert "run_id" not in finished
        assert secret not in json.dumps(finished)

    def test_system_exit_code_is_exited(self, monkeypatch, transport):
        finished = _finish_via(monkeypatch, transport, _raise(SystemExit(1)))
        assert finished["status"] == "failed"
        assert finished["outcome_status"] == "exited"
        assert finished["error_rule"] == "exit:1"

    def test_system_exit_without_code_is_exit_zero(self, monkeypatch, transport):
        finished = _finish_via(monkeypatch, transport, _raise(SystemExit()))
        assert finished["outcome_status"] == "exited"
        assert finished["error_rule"] == "exit:0"

    def test_system_exit_message_is_never_sent(self, monkeypatch, transport):
        message = "Policy checkout /private/repo is dirty"
        finished = _finish_via(monkeypatch, transport, _raise(SystemExit(message)))
        assert finished["outcome_status"] == "exited"
        assert finished["error_rule"] == "exit:message"
        assert message not in json.dumps(finished)
        assert "/private/repo" not in json.dumps(finished)

    def test_keyboard_interrupt_is_interrupted(self, monkeypatch, transport):
        finished = _finish_via(monkeypatch, transport, _raise(KeyboardInterrupt()))
        assert finished["status"] == "failed"
        assert finished["outcome_status"] == "interrupted"
        assert finished["error_rule"] == "exception:KeyboardInterrupt"

    @pytest.mark.parametrize(
        "outcome_status, error_rule",
        [
            ("x" * 65, "e" * 161),
            ("has space", "Could not read /tmp/file: denied"),
            ("", ""),
            (7, {"check": "a"}),
            ("ünïcode", "exception:Ünï"),
            ("line\nbreak", "a;b"),
        ],
    )
    def test_oversize_or_unsafe_values_are_dropped(
        self, monkeypatch, transport, outcome_status, error_rule
    ):
        finished = _finish_via(
            monkeypatch,
            transport,
            lambda live: live.finish(
                "failed", outcome_status=outcome_status, error_rule=error_rule
            ),
        )
        assert finished["status"] == "failed"
        assert "outcome_status" not in finished
        assert "error_rule" not in finished

    def test_values_at_the_bounds_are_kept(self, monkeypatch, transport):
        finished = _finish_via(
            monkeypatch,
            transport,
            lambda live: live.finish(
                "failed", outcome_status="s" * 64, error_rule="A-z0_9.:/" + "r" * 151
            ),
        )
        assert finished["outcome_status"] == "s" * 64
        assert len(finished["error_rule"]) == 160

    def test_one_unsafe_value_does_not_drop_the_other(self, monkeypatch, transport):
        finished = _finish_via(
            monkeypatch,
            transport,
            lambda live: live.finish(
                "failed", outcome_status="standalone_failed", error_rule="not safe"
            ),
        )
        assert finished["outcome_status"] == "standalone_failed"
        assert "error_rule" not in finished

    def test_first_finish_wins_over_the_exit_outcome(self, monkeypatch, transport):
        def body(live):
            live.finish(
                "failed", run_id="abc12345", outcome_status="apply_blocked_generation"
            )
            # The encode loop exits right after its explicit finish.
            raise SystemExit(1)

        finished = _finish_via(monkeypatch, transport, body)
        assert finished["outcome_status"] == "apply_blocked_generation"
        assert finished["run_id"] == "abc12345"
        assert "error_rule" not in finished

    def test_heartbeats_never_carry_outcome_fields(self, monkeypatch, transport):
        if transport == "direct":
            _configured_env(monkeypatch)
            client, table = _mock_client()
            with patch(
                "axiom_encode.supabase_sync.get_supabase_client", return_value=client
            ):
                with _live_run() as live:
                    live.set_phase(PHASE_GENERATE)
                    _wait_for_calls(table.update, 1)
                    live.finish("completed", outcome_status="standalone_validated")
            sent = [call.args[0] for call in table.update.call_args_list]
            heartbeats = [data for data in sent if "finished_at" not in data]
        else:
            _ingest_env(monkeypatch)
            with _mock_urlopen() as urlopen_mock:
                with _live_run() as live:
                    live.set_phase(PHASE_GENERATE)
                    _wait_for_calls(urlopen_mock, 2)
                    live.finish("completed", outcome_status="standalone_validated")
            heartbeats = [
                payload
                for payload in _ingest_payloads(urlopen_mock)
                if payload["op"] != "finish"
            ]
        assert heartbeats
        for data in heartbeats:
            assert "outcome_status" not in data
            assert "error_rule" not in data


class TestTerminalOutcomeDirectFallback:
    def _run(self, monkeypatch, side_effect, **finish_kwargs):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        table.update.return_value.execute.side_effect = side_effect
        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=client
        ):
            with _live_run() as live:
                live.finish("failed", run_id="abc12345", **finish_kwargs)
        return [call.args[0] for call in table.update.call_args_list]

    def test_unknown_column_retries_once_without_outcome_fields(self, monkeypatch):
        sent = self._run(
            monkeypatch,
            [
                _MissingColumnError("Could not find the 'error_rule' column"),
                MagicMock(),
            ],
            outcome_status="aborted",
            error_rule="exception:RuntimeError",
        )
        assert len(sent) == 2
        assert sent[0]["outcome_status"] == "aborted"
        assert sent[0]["error_rule"] == "exception:RuntimeError"
        retry = sent[1]
        assert "outcome_status" not in retry
        assert "error_rule" not in retry
        # Everything else survives, so the row still closes.
        assert retry["status"] == "failed"
        assert retry["run_id"] == "abc12345"
        assert retry["finished_at"]
        assert {key: value for key, value in sent[0].items() if key in retry} == retry

    def test_code_less_unknown_column_message_also_retries(self, monkeypatch):
        sent = self._run(
            monkeypatch,
            [
                Exception('column "outcome_status" of relation x does not exist'),
                MagicMock(),
            ],
            outcome_status="aborted",
        )
        assert len(sent) == 2
        assert "outcome_status" not in sent[1]

    def test_retry_failure_is_swallowed_and_not_repeated(self, monkeypatch):
        sent = self._run(
            monkeypatch,
            _MissingColumnError("Could not find the 'error_rule' column"),
            outcome_status="aborted",
            error_rule="exception:RuntimeError",
        )
        assert len(sent) == 2

    def test_other_errors_do_not_retry(self, monkeypatch):
        sent = self._run(
            monkeypatch,
            RuntimeError("connection reset"),
            outcome_status="aborted",
        )
        assert len(sent) == 1

    def test_unknown_column_without_outcome_fields_does_not_retry(self, monkeypatch):
        sent = self._run(
            monkeypatch,
            _MissingColumnError("Could not find the 'phase' column"),
        )
        assert len(sent) == 1

    def test_missing_table_does_not_retry(self, monkeypatch):
        error = _MissingColumnError("Could not find the table")
        error.code = "PGRST205"
        sent = self._run(monkeypatch, error, outcome_status="aborted")
        assert len(sent) == 1


class TestAbnormalExitOutcome:
    def test_bool_exit_code_reads_as_its_integer(self):
        assert abnormal_exit_outcome(SystemExit, SystemExit(True)) == (
            "exited",
            "exit:1",
        )

    def test_subclass_reports_its_own_class_name(self):
        class CorpusUnavailable(RuntimeError):
            pass

        assert abnormal_exit_outcome(CorpusUnavailable, CorpusUnavailable("x")) == (
            "aborted",
            "exception:CorpusUnavailable",
        )

    def test_telemetry_never_raises_on_exit(self, monkeypatch):
        _ingest_env(monkeypatch)
        with (
            _mock_urlopen() as urlopen_mock,
            patch.object(
                live_run_telemetry,
                "abnormal_exit_outcome",
                side_effect=RuntimeError("boom"),
            ),
        ):
            with pytest.raises(ValueError):
                with _live_run():
                    raise ValueError("original")
        finished = _ingest_payloads(urlopen_mock)[-1]
        assert finished["op"] == "finish"
        assert finished["status"] == "failed"
        assert "outcome_status" not in finished


class TestFirstTypedCheckId:
    @staticmethod
    def _run(*documents):
        from axiom_encode.harness.encoding_db import Iteration, IterationError

        earlier = Iteration(
            attempt=1,
            duration_ms=1,
            errors=[
                IterationError(
                    error_type="validation",
                    message="earlier",
                    issue_document={"issues": [{"check": "earlier:check"}]},
                )
            ],
        )
        final = Iteration(
            attempt=2,
            duration_ms=1,
            errors=[
                IterationError(
                    error_type="validation", message="m", issue_document=document
                )
                for document in documents
            ],
        )
        return MagicMock(iterations=[earlier, final])

    def test_reads_first_typed_issue_of_the_final_attempt(self):
        run = self._run(
            {
                "issues": [
                    {"check": None, "message": "untyped"},
                    {"check": "complete-source-unit:tests/output-unasserted"},
                    {"check": "later:check"},
                ]
            }
        )
        assert (
            first_typed_check_id(run) == "complete-source-unit:tests/output-unasserted"
        )

    def test_real_typed_issue_document_round_trips(self):
        from axiom_encode import checks

        check_id = "complete-source-unit:tests/output-unasserted"
        issue = checks.ValidationIssue("Output never asserted", check=check_id)
        run = self._run(checks.typed_issue_document([("ci", [issue])]))
        assert first_typed_check_id(run) == check_id

    def test_none_without_a_typed_issue(self):
        assert first_typed_check_id(self._run(None)) is None
        assert first_typed_check_id(self._run({"issues": [{"check": None}]})) is None
        assert first_typed_check_id(self._run()) is None
        assert first_typed_check_id(MagicMock(iterations=[])) is None
        assert first_typed_check_id(None) is None

    def test_never_raises_on_malformed_runs(self):
        assert first_typed_check_id(self._run({"issues": 5})) is None
        assert first_typed_check_id(self._run({"issues": ["text"]})) is None
        assert first_typed_check_id(MagicMock(iterations=7)) is None
