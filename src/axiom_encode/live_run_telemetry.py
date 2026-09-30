"""
Live encode-run presence for the ops dashboard.

Maintains one row in encodings.live_encoding_runs per in-flight
`axiom-encode encode` invocation: inserted when the run starts, heartbeated
by a daemon thread while the run is active, and closed with a pointer to the
final encodings.encoding_runs row. The dashboard treats a 'running' row with
a stale heartbeat as a dead encoder.

Two transports, chosen automatically:

- **direct**: environments holding Supabase write credentials
  (`AXIOM_ENCODE_SUPABASE_URL` + `AXIOM_ENCODE_SUPABASE_SECRET_KEY`)
  write rows directly, as trusted telemetry.
- **ingest**: everyone else — including third-party encoders — reports
  credential-free to the public ops ingest endpoint, which stamps rows
  as self-reported. This is the default: no setup, no tokens.

Set `AXIOM_ENCODE_TELEMETRY=off` (or pass `--no-sync`) to opt out.
Telemetry is strictly best-effort: every network failure is swallowed so
neither a Supabase outage nor an unreachable ingest endpoint can ever
fail an encode.

Phases
------
`live_encoding_runs.phase` names what an encode is doing right now. The
values are short, stable, lowercase strings the ops dashboard matches on;
add new ones rather than renaming these:

- ``resolve``: loading the corpus release and resolving the source unit,
  replacement target, and prompt workspace. A run starts here, and every
  validator retry or escalation re-enters it.
- ``generate``: the model call that writes the candidate RuleSpec.
- ``validate``: deterministic compile, CI, and oracle checks of a
  candidate, including in-place repair rounds and the retained-candidate
  preflight.
- ``review``: the LLM generalist reviewer, only when it actually runs.
- ``apply``: ``--apply`` only; overlay validation against the policy
  checkout and the signed apply transaction.

Only transitions reach the network: re-entering the current phase is free,
and a transition costs one small update. The periodic heartbeat carries the
current phase too, so a dropped transition update heals within one
interval.

GitHub Actions
--------------
Inside GitHub Actions the runner identity also carries ``github_run_id``,
``github_run_attempt``, ``github_run_url``, and ``github_workflow`` (see
`github_run_identity`), so a live run joins to its workflow run exactly.
"""

import getpass
import json
import os
import platform
import re
import socket
import sys
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Optional

HEARTBEAT_INTERVAL_SECONDS = 30.0
DEFAULT_INGEST_URL = "https://axiom.org/api/ops/encoding/ingest"
INGEST_TIMEOUT_SECONDS = 10.0

_CI_ENV_VARS = ("CI", "GITHUB_ACTIONS", "BUILDKITE", "CIRCLECI")
_TELEMETRY_OFF_VALUES = {"off", "0", "false", "disabled"}
_TELEMETRY_ON_VALUES = {"on", "1", "true"}

# Encode phases; the module docstring documents what each one covers.
PHASE_RESOLVE = "resolve"
PHASE_GENERATE = "generate"
PHASE_VALIDATE = "validate"
PHASE_REVIEW = "review"
PHASE_APPLY = "apply"
ENCODE_PHASES = (
    PHASE_RESOLVE,
    PHASE_GENERATE,
    PHASE_VALIDATE,
    PHASE_REVIEW,
    PHASE_APPLY,
)

_GITHUB_RUN_ID_RE = re.compile(r"[0-9]{1,20}")
_GITHUB_RUN_ATTEMPT_RE = re.compile(r"[1-9][0-9]{0,8}")
_GITHUB_SERVER_URL_RE = re.compile(r"https?://[A-Za-z0-9.-]+(?::[0-9]{1,5})?")
_GITHUB_REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_GITHUB_WORKFLOW_MAX_LENGTH = 120


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def runner_identity() -> dict:
    """Machine identity attached to a live run (shown on the ops dashboard)."""
    try:
        hostname = socket.gethostname()
    except OSError:
        hostname = ""
    try:
        username = getpass.getuser()
    except (KeyError, OSError):
        username = ""
    return {
        "hostname": hostname,
        "username": username,
        "platform": platform.system().lower(),
        "pid": os.getpid(),
        "is_ci": any(os.environ.get(var) for var in _CI_ENV_VARS),
        **github_run_identity(),
    }


def github_run_identity() -> dict:
    """The GitHub Actions run this process belongs to, or ``{}`` outside it.

    Read from the standard runner variables (``GITHUB_ACTIONS``,
    ``GITHUB_RUN_ID``, ``GITHUB_RUN_ATTEMPT``, ``GITHUB_SERVER_URL``,
    ``GITHUB_REPOSITORY``, ``GITHUB_WORKFLOW``). ``github_run_id`` anchors
    the identity: without ``GITHUB_ACTIONS=true`` and a numeric run id
    nothing is reported. The other keys appear only when their inputs are
    present and well-formed, so a partial environment yields a partial
    identity rather than a malformed one: ``github_run_attempt`` (int),
    ``github_run_url`` (the run page, needing the server URL and repository),
    and ``github_workflow`` (the workflow name).
    """
    if os.environ.get("GITHUB_ACTIONS", "").strip().lower() != "true":
        return {}
    run_id = os.environ.get("GITHUB_RUN_ID", "").strip()
    if not _GITHUB_RUN_ID_RE.fullmatch(run_id):
        return {}
    identity: dict = {"github_run_id": run_id}
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "").strip()
    if _GITHUB_RUN_ATTEMPT_RE.fullmatch(attempt):
        identity["github_run_attempt"] = int(attempt)
    server_url = os.environ.get("GITHUB_SERVER_URL", "").strip().rstrip("/")
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    url_parts_valid = bool(
        _GITHUB_SERVER_URL_RE.fullmatch(server_url)
        and _GITHUB_REPOSITORY_RE.fullmatch(repository)
    )
    if url_parts_valid:
        identity["github_run_url"] = f"{server_url}/{repository}/actions/runs/{run_id}"
    workflow = os.environ.get("GITHUB_WORKFLOW", "").strip()
    if workflow:
        identity["github_workflow"] = workflow[:_GITHUB_WORKFLOW_MAX_LENGTH]
    return identity


def running_under_tests() -> bool:
    """Test-suite invocations of the encode path must never reach the real
    dashboard. The env marker alone is not enough: hermetic tests clear
    os.environ (in-process) or spawn the CLI with scrubbed envs, so the
    in-process signal is the pytest module itself.
    """
    return "pytest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST"))


def telemetry_blocked_for_tests() -> bool:
    """True when test detection should suppress telemetry. The explicit
    `AXIOM_ENCODE_TELEMETRY=on` override bypasses detection — used by
    telemetry tests exercising the (mocked) transports, never in
    production config."""
    override = os.environ.get("AXIOM_ENCODE_TELEMETRY", "").strip().lower()
    return override not in _TELEMETRY_ON_VALUES and running_under_tests()


def telemetry_mode() -> str:
    """Resolve the transport: 'direct', 'ingest', or 'off'."""
    override = os.environ.get("AXIOM_ENCODE_TELEMETRY", "").strip().lower()
    if override in _TELEMETRY_OFF_VALUES:
        return "off"
    if telemetry_blocked_for_tests():
        return "off"
    if os.environ.get("AXIOM_ENCODE_SUPABASE_URL") and os.environ.get(
        "AXIOM_ENCODE_SUPABASE_SECRET_KEY"
    ):
        return "direct"
    return "ingest"


def _ingest_url() -> str:
    return (
        os.environ.get("AXIOM_ENCODE_TELEMETRY_INGEST_URL", "").strip()
        or DEFAULT_INGEST_URL
    )


# The live run of this process, registered while its row is open. Encode
# stages deep in the harness report phases through `report_phase` instead of
# threading the telemetry object through every call.
_active_run: Optional["LiveRunTelemetry"] = None
_active_run_lock = threading.Lock()


def report_phase(phase: str) -> None:
    """Move this process's live run to ``phase``; a no-op without one.

    Safe to call from any code path: outside an ``encode`` invocation (or
    with telemetry off) nothing is registered and nothing is sent.
    """
    run = _active_run
    if run is not None:
        run.set_phase(phase)


class LiveRunTelemetry:
    """Context manager owning one live_encoding_runs row and its heartbeat.

    Constructed unconditionally; becomes a no-op when telemetry is off or
    the initial announce fails.
    """

    def __init__(
        self,
        *,
        citation: str,
        backend: str,
        model: str,
        encoder_version: str,
        enabled: bool = True,
        phase: Optional[str] = None,
    ):
        self.id = f"live-{uuid.uuid4().hex[:12]}"
        self.citation = citation
        self.backend = backend
        self.model = model
        self.encoder_version = encoder_version
        self.phase = phase
        self._mode = telemetry_mode() if enabled else "off"
        self._client = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._finished = False

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "LiveRunTelemetry":
        if self._mode == "off":
            return self
        started = (
            self._start_direct() if self._mode == "direct" else self._start_ingest()
        )
        if not started:
            self._mode = "off"
            self._client = None
            return self
        global _active_run
        with _active_run_lock:
            _active_run = self
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="live-run-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Interrupted or crashed runs finish as 'failed'; normal completion
        # already called finish() with the real outcome.
        self.finish("failed" if exc_type is not None else "completed")

    # -- updates -------------------------------------------------------------

    def set_attempt(
        self, attempt: int, model: str, *, phase: Optional[str] = None
    ) -> None:
        """Record a retry/escalation so the dashboard shows current state.

        ``phase`` rides in the same update, so a retry that restarts from
        ``resolve`` costs one call rather than two.
        """
        self.model = model
        fields: dict = {"attempt": attempt, "model": model}
        if phase is not None:
            self.phase = phase
            fields["phase"] = phase
        self._send("heartbeat", fields)

    def set_phase(self, phase: str) -> None:
        """Move the run to ``phase`` (one of `ENCODE_PHASES`).

        Re-entering the current phase sends nothing; a transition sends one
        small update, and later heartbeats keep repeating it.
        """
        if phase == self.phase:
            return
        self.phase = phase
        self._send("heartbeat", {"phase": phase})

    def finish(self, status: str, run_id: Optional[str] = None) -> None:
        """Close the live row; idempotent, first call wins."""
        if self._finished:
            return
        self._finished = True
        global _active_run
        with _active_run_lock:
            if _active_run is self:
                _active_run = None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        fields: dict = {"status": status}
        if run_id:
            fields["run_id"] = run_id
        self._send("finish", fields, force=True)

    # -- internals -----------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            phase = self.phase
            self._send("heartbeat", {"phase": phase} if phase else {})

    def _send(self, kind: str, fields: dict, *, force: bool = False) -> None:
        if self._mode == "off" or (self._finished and not force):
            return
        try:
            if self._mode == "direct":
                self._send_direct(kind, fields)
            else:
                self._send_ingest(kind, fields)
        except Exception:
            # Best-effort: a missed heartbeat shows as staleness, nothing more.
            pass

    # -- direct transport (trusted environments) -----------------------------

    def _start_direct(self) -> bool:
        try:
            from .supabase_sync import ENCODINGS_SCHEMA, get_supabase_client

            self._client = get_supabase_client()
            now = _utcnow()
            self._client.schema(ENCODINGS_SCHEMA).table("live_encoding_runs").insert(
                {
                    "id": self.id,
                    "citation": self.citation,
                    "status": "running",
                    "started_at": now,
                    "last_heartbeat_at": now,
                    "backend": self.backend,
                    "model": self.model,
                    "attempt": 1,
                    "phase": self.phase,
                    "encoder_version": self.encoder_version or None,
                    "runner": runner_identity(),
                }
            ).execute()
            return True
        except Exception as exc:
            print(f"live-run telemetry disabled: {exc}", file=sys.stderr)
            return False

    def _send_direct(self, kind: str, fields: dict) -> None:
        if self._client is None:
            return
        from .supabase_sync import ENCODINGS_SCHEMA

        data = dict(fields)
        data["last_heartbeat_at"] = _utcnow()
        if kind == "finish":
            data["finished_at"] = _utcnow()
        self._client.schema(ENCODINGS_SCHEMA).table("live_encoding_runs").update(
            data
        ).eq("id", self.id).execute()

    # -- ingest transport (credential-free default) --------------------------

    def _start_ingest(self) -> bool:
        ok = self._post_ingest(
            {
                "op": "start",
                "id": self.id,
                "citation": self.citation,
                "backend": self.backend,
                "model": self.model,
                "attempt": 1,
                "phase": self.phase,
                "encoder_version": self.encoder_version or None,
                "runner": runner_identity(),
            }
        )
        if not ok:
            print(
                "live-run telemetry disabled: ops ingest endpoint unreachable",
                file=sys.stderr,
            )
        return ok

    def _send_ingest(self, kind: str, fields: dict) -> None:
        self._post_ingest({"op": kind, "id": self.id, **fields})

    def _post_ingest(self, payload: dict) -> bool:
        try:
            request = urllib.request.Request(
                _ingest_url(),
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(
                request, timeout=INGEST_TIMEOUT_SECONDS
            ) as response:
                return 200 <= response.status < 300
        except Exception:
            return False
