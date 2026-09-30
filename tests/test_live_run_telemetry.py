"""Tests for live encode-run presence telemetry."""

import json
from unittest.mock import MagicMock, patch

from axiom_encode import live_run_telemetry
from axiom_encode.live_run_telemetry import (
    ENCODE_PHASES,
    PHASE_APPLY,
    PHASE_GENERATE,
    PHASE_RESOLVE,
    PHASE_REVIEW,
    PHASE_VALIDATE,
    LiveRunTelemetry,
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
    table.update.return_value.eq.return_value.execute.return_value = MagicMock()
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
        assert payloads[1] == {
            "op": "heartbeat",
            "id": live.id,
            "attempt": 2,
            "model": "gpt-5.5-max",
        }
        assert payloads[2]["op"] == "finish"
        assert payloads[2]["status"] == "completed"
        assert payloads[2]["run_id"] == "abc12345"
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
                report_phase(PHASE_RESOLVE)  # already there: no update
                report_phase(PHASE_GENERATE)
                report_phase(PHASE_VALIDATE)
                report_phase(PHASE_VALIDATE)  # a repair round: no update
                report_phase(PHASE_REVIEW)
                # A validator retry restarts from resolve in the same update
                # that records the new attempt.
                live.set_attempt(2, "gpt-5.5-max", phase=PHASE_RESOLVE)
                report_phase(PHASE_GENERATE)
                report_phase(PHASE_VALIDATE)
                report_phase(PHASE_APPLY)
                live.finish("completed", run_id="abc12345")

        assert _phase_updates(table) == [
            "generate",
            "validate",
            "review",
            "resolve",
            "generate",
            "validate",
            "apply",
            None,  # finish
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
        client, table = _mock_client()
        live = LiveRunTelemetry(
            citation="us/statute/26/32",
            backend="openai",
            model="gpt-5.5",
            encoder_version="0.1.0",
            phase=PHASE_VALIDATE,
        )
        live._client = client
        live._stop = MagicMock()
        # One interval elapses, then the run stops.
        live._stop.wait.side_effect = [False, True]
        live._heartbeat_loop()
        (update,) = table.update.call_args_list
        assert update.args[0]["phase"] == "validate"
        assert update.args[0]["last_heartbeat_at"]

    def test_heartbeat_without_phase_sends_only_liveness(self, monkeypatch):
        _configured_env(monkeypatch)
        client, table = _mock_client()
        live = LiveRunTelemetry(
            citation="us/statute/26/32",
            backend="openai",
            model="gpt-5.5",
            encoder_version="0.1.0",
        )
        live._client = client
        live._stop = MagicMock()
        live._stop.wait.side_effect = [False, True]
        live._heartbeat_loop()
        (update,) = table.update.call_args_list
        assert set(update.args[0]) == {"last_heartbeat_at"}

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
