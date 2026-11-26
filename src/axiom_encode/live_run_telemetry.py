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
and a transition costs one small update. Phase and attempt changes never
wait on the network: they record state and wake the heartbeat thread, the
only sender while the run is live, so updates reach the row in order and a
slow Supabase or ingest endpoint cannot stall the encode. Every heartbeat
carries the current phase, so a dropped update heals within one interval.

GitHub Actions
--------------
Inside GitHub Actions the runner identity also carries ``github_run_id``,
``github_run_attempt``, ``github_run_url``, and ``github_workflow`` (see
`github_run_identity`), so a live run joins to its workflow run exactly.

Terminal outcome
----------------
The ``finish`` update, and only it, may carry two more nullable text fields,
so a run that ends before any encoding_runs row exists still says why:

- ``outcome_status``: a short machine token. On the normal path it is the
  encode outcome's ``status`` (``standalone_validated``, ``apply_applied``,
  ``apply_blocked_validation``, ...). When the encode leaves the live-run
  context without an explicit finish it is ``aborted`` (an exception
  escaped), ``exited`` (``SystemExit``), or ``interrupted``
  (``KeyboardInterrupt``).
- ``error_rule``: a typed, bounded identifier of the cause, never a message
  or a path. Abnormal exits send ``exception:<ExceptionClassName>`` or
  ``exit:<int code>`` (``exit:message`` for a non-integer ``SystemExit``
  code; the message itself is never sent). A failed normal finish sends the
  check id of the final attempt's first typed validation issue, when there
  is one. Successful runs send none.

An exit that reaches the end of the live-run context without an explicit
finish always closes the row as ``failed``, also for ``sys.exit(0)`` or a
bare ``SystemExit()`` (``exited`` / ``exit:0``); the encode loop has no such
path today, because it finishes explicitly before it exits.

Both are bounded before sending: ``outcome_status`` to 64 characters,
``error_rule`` to 160, both to ``[A-Za-z0-9_.:/-]``. A value that does not
fit is dropped, not truncated. The columns come from
``migrations/009_live_encoding_run_outcome.sql``; until it is applied, the
direct transport retries a rejected finish once without the two fields. The
ingest endpoint accepts and stores ``outcome_status`` and ``error_rule`` on
``finish`` (since axiom.org#407), and still ignores fields it does not know.

Reviewed candidate promotion
----------------------------
``axiom-encode promote-reviewed-candidate`` makes no model call and never
runs ``encode``, so `PromotionLiveRun` opens its row instead. The row has the
same columns and transports; ``backend`` is ``reviewed-candidate`` and
``model`` is ``reviewed-candidate-promotion-v1``, fixed markers that say no
model ran. Its stages reuse the encode phases:

- ``admission`` (phase ``resolve``): every check before validation: the
  exact checkout, the admission record and its live revalidation, the
  binding, the clean checkout, and source resolution.
- ``validation`` (phase ``validate``): complete-source overlay validation
  of the exact candidate bytes.
- ``signing`` (phase ``apply``): the last admission revalidation, the
  signature, and the manifest install, until the install transaction
  returns.
- ``post_install`` (phase ``apply``): the final check after the signed
  manifest is installed. A failure here means "signed, then the final check
  failed": the manifest is on disk.

A promotion that completes closes ``completed`` with ``outcome_status``
``promotion_signed``. One that raises closes ``failed`` with
``promotion_failed_<stage>`` and ``error_rule`` ``exception:<ClassName>``.
``SystemExit`` and ``KeyboardInterrupt`` keep the ``exited`` and
``interrupted`` tokens above. As for ``encode``, that also holds for
``sys.exit(0)``: it would close the row ``failed`` / ``exited`` /
``exit:0``; the promotion body has no ``sys.exit`` today.

The promotion runs in the signing path, so it never waits long for its row.
The row is opened on a helper thread and the command waits at most
`PROMOTION_START_TIMEOUT_SECONDS` for it, then continues. A start that
completes later still gets the current stage, and the terminal outcome if
the command has ended by then. ``encode`` is unchanged: it opens its row on
the main thread.

The citation is sent before the command validates it, so the promotion row
bounds it to `PROMOTION_CITATION_MAX_LENGTH` characters (the ingest
endpoint's limit); a longer one is cut and ends with ``...[truncated]``.

There is no fallback between transports. With credentials present the
promotion uses the direct transport only: if that start fails, the run has
no row. An ingest row cannot join to its workflow run, because the ingest
endpoint drops the GitHub run fields of ``runner``.
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

# Terminal outcome fields; the module docstring documents the contract.
OUTCOME_STATUS_MAX_LENGTH = 64
ERROR_RULE_MAX_LENGTH = 160
OUTCOME_ABORTED = "aborted"
OUTCOME_EXITED = "exited"
OUTCOME_INTERRUPTED = "interrupted"
TERMINAL_OUTCOME_COLUMNS = ("outcome_status", "error_rule")
_OUTCOME_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:/-]+")

# Reviewed candidate promotion; the module docstring documents the contract.
PROMOTION_BACKEND = "reviewed-candidate"
PROMOTION_MODEL = "reviewed-candidate-promotion-v1"
PROMOTION_STAGE_ADMISSION = "admission"
PROMOTION_STAGE_VALIDATION = "validation"
PROMOTION_STAGE_SIGNING = "signing"
PROMOTION_STAGE_POST_INSTALL = "post_install"
PROMOTION_STAGE_PHASES = {
    PROMOTION_STAGE_ADMISSION: PHASE_RESOLVE,
    PROMOTION_STAGE_VALIDATION: PHASE_VALIDATE,
    PROMOTION_STAGE_SIGNING: PHASE_APPLY,
    PROMOTION_STAGE_POST_INSTALL: PHASE_APPLY,
}
PROMOTION_START_TIMEOUT_SECONDS = 3.0
PROMOTION_CITATION_MAX_LENGTH = 300
PROMOTION_CITATION_TRUNCATED_MARKER = "...[truncated]"
PROMOTION_OUTCOME_SIGNED = "promotion_signed"
PROMOTION_OUTCOME_FAILED_PREFIX = "promotion_failed_"


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


def _bounded_token(value: object, max_length: int) -> Optional[str]:
    """``value`` when it is a safe token within ``max_length``, else ``None``.

    A value that does not fit is dropped whole: a truncated identifier would
    name a different, wrong cause.
    """
    if not isinstance(value, str) or len(value) > max_length:
        return None
    if not _OUTCOME_TOKEN_RE.fullmatch(value):
        return None
    return value


def abnormal_exit_outcome(
    exc_type: Optional[type], exc: Optional[BaseException]
) -> tuple[str, str]:
    """``(outcome_status, error_rule)`` for an exit that skipped `finish`.

    Carries the exception class or the integer exit code only; never the
    exception message or a non-integer ``SystemExit`` code.
    """
    if exc_type is not None and issubclass(exc_type, SystemExit):
        code = getattr(exc, "code", None)
        if code is None:
            return OUTCOME_EXITED, "exit:0"
        if isinstance(code, int):
            return OUTCOME_EXITED, f"exit:{int(code)}"
        return OUTCOME_EXITED, "exit:message"
    name = getattr(exc_type, "__name__", "") or "BaseException"
    if exc_type is not None and issubclass(exc_type, KeyboardInterrupt):
        return OUTCOME_INTERRUPTED, f"exception:{name}"
    return OUTCOME_ABORTED, f"exception:{name}"


def first_typed_check_id(run: object) -> Optional[str]:
    """Check id of the first typed issue of a logged run's final attempt.

    Reads the ``issue_document`` the run log already stored for that attempt
    (see `axiom_encode.checks.typed_issue_document`); ``None`` when the final
    attempt recorded no typed issue. Never raises.
    """
    try:
        iterations = getattr(run, "iterations", None) or ()
        if not iterations:
            return None
        for error in getattr(iterations[-1], "errors", None) or ():
            document = getattr(error, "issue_document", None)
            if not isinstance(document, dict):
                continue
            for record in document.get("issues") or ():
                check = record.get("check") if isinstance(record, dict) else None
                if isinstance(check, str) and check:
                    return check
    except Exception:
        return None
    return None


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
        self.attempt = 1
        self.encoder_version = encoder_version
        self.phase = phase
        self._mode = telemetry_mode() if enabled else "off"
        self._client = None
        self._stop = threading.Event()
        # Wakes the heartbeat thread early to publish a state change.
        self._wake = threading.Event()
        self._state_lock = threading.Lock()
        # attempt/model/phase changes not yet taken for an update.
        self._unsent: dict = {}
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
        # already called finish() with the real outcome, and that call wins.
        if exc_type is None:
            self.finish("completed")
            return
        try:
            outcome_status, error_rule = abnormal_exit_outcome(exc_type, exc)
        except Exception:
            outcome_status, error_rule = None, None
        self.finish("failed", outcome_status=outcome_status, error_rule=error_rule)

    # -- updates -------------------------------------------------------------

    def set_attempt(
        self, attempt: int, model: str, *, phase: Optional[str] = None
    ) -> None:
        """Record a retry/escalation so the dashboard shows current state.

        Like `set_phase`, this never waits on the network. ``phase`` rides in
        the same update, so a retry that restarts from ``resolve`` costs one
        call rather than two.
        """
        with self._state_lock:
            self.attempt = attempt
            self.model = model
            self._unsent.update(attempt=attempt, model=model)
            if phase is not None:
                self.phase = phase
                self._unsent["phase"] = phase
        self._wake.set()

    def set_phase(self, phase: str) -> None:
        """Move the run to ``phase`` (one of `ENCODE_PHASES`).

        Re-entering the current phase does nothing. A transition records the
        phase and wakes the heartbeat thread, which sends one small update;
        later heartbeats keep repeating it.
        """
        with self._state_lock:
            if phase == self.phase:
                return
            self.phase = phase
            self._unsent["phase"] = phase
        self._wake.set()

    def finish(
        self,
        status: str,
        run_id: Optional[str] = None,
        *,
        outcome_status: Optional[str] = None,
        error_rule: Optional[str] = None,
    ) -> None:
        """Close the live row; idempotent, first call wins.

        ``outcome_status`` and ``error_rule`` are the terminal outcome fields
        the module docstring describes; a value outside their bounds is
        dropped.
        """
        if self._finished:
            return
        self._finished = True
        global _active_run
        with _active_run_lock:
            if _active_run is self:
                _active_run = None
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # Repeat current state: a heartbeat may have taken an update but still
        # be in flight after the bounded join. It must not erase terminal state.
        fields = self._take_update(final=True)
        fields["status"] = status
        if run_id:
            fields["run_id"] = run_id
        outcome_status = _bounded_token(outcome_status, OUTCOME_STATUS_MAX_LENGTH)
        if outcome_status:
            fields["outcome_status"] = outcome_status
        error_rule = _bounded_token(error_rule, ERROR_RULE_MAX_LENGTH)
        if error_rule:
            fields["error_rule"] = error_rule
        self._send("finish", fields, force=True)

    # -- internals -----------------------------------------------------------

    def _take_update(self, *, final: bool = False) -> dict:
        """Unsent changes and phase, or full final state; clears the unsent set."""
        with self._state_lock:
            fields = self._unsent
            self._unsent = {}
            if final:
                fields.update(attempt=self.attempt, model=self.model)
            if self.phase:
                fields["phase"] = self.phase
        return fields

    def _heartbeat_loop(self) -> None:
        # The only sender while the run is live: it wakes every interval to
        # prove liveness, or early to publish a change.
        while True:
            self._wake.wait(HEARTBEAT_INTERVAL_SECONDS)
            self._wake.clear()
            if self._stop.is_set():
                return
            # Forced: changes taken here must go out even if finish() starts
            # meanwhile. The join is bounded; the transport rejects a delayed
            # heartbeat once the row is terminal.
            self._send("heartbeat", self._take_update(), force=True)

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
        from .supabase_sync import ENCODINGS_SCHEMA, _is_missing_column_error

        data = dict(fields)
        data["last_heartbeat_at"] = _utcnow()
        if kind == "finish":
            data["finished_at"] = _utcnow()

        def update(payload: dict) -> None:
            query = (
                self._client.schema(ENCODINGS_SCHEMA)
                .table("live_encoding_runs")
                .update(payload)
                .eq("id", self.id)
            )
            if kind == "heartbeat":
                query = query.eq("status", "running")
            query.execute()

        try:
            update(data)
        except Exception as error:
            # Before migration 009 the table lacks the terminal outcome
            # columns: close the row without them rather than leave it open.
            if not _is_missing_column_error(error) or not any(
                column in data for column in TERMINAL_OUTCOME_COLUMNS
            ):
                raise
            update(
                {
                    key: value
                    for key, value in data.items()
                    if key not in TERMINAL_OUTCOME_COLUMNS
                }
            )

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


# The promotion of this process, registered while its context is open, so the
# command reports stages without threading the object through its body.
_active_promotion: Optional["PromotionLiveRun"] = None


def report_promotion_stage(stage: str) -> None:
    """Move this process's promotion to ``stage``; a no-op without one."""
    promotion = _active_promotion
    if promotion is not None:
        promotion.stage(stage)


def bounded_promotion_citation(citation: str) -> str:
    """``citation`` within `PROMOTION_CITATION_MAX_LENGTH`, marked when cut."""
    if len(citation) <= PROMOTION_CITATION_MAX_LENGTH:
        return citation
    marker = PROMOTION_CITATION_TRUNCATED_MARKER
    return citation[: PROMOTION_CITATION_MAX_LENGTH - len(marker)] + marker


class PromotionLiveRun:
    """Live row of one ``promote-reviewed-candidate`` invocation.

    Wraps `LiveRunTelemetry` and never changes the command: no telemetry
    error leaves this class, an exception from the command always propagates
    unchanged, and the command never waits more than
    `PROMOTION_START_TIMEOUT_SECONDS` for the row to open.
    """

    def __init__(self, *, citation: str, encoder_version: str):
        self._stage = PROMOTION_STAGE_ADMISSION
        self._lock = threading.Lock()
        # The row is open (its start returned), and the terminal fields once
        # the command has ended. Whichever of the two comes second closes it.
        self._opened = False
        self._final: Optional[dict] = None
        self._previous_promotion: Optional["PromotionLiveRun"] = None
        self._previous_run: Optional[LiveRunTelemetry] = None
        self._live: Optional[LiveRunTelemetry] = None
        try:
            self._live = LiveRunTelemetry(
                citation=bounded_promotion_citation(citation),
                backend=PROMOTION_BACKEND,
                model=PROMOTION_MODEL,
                encoder_version=encoder_version,
                phase=PROMOTION_STAGE_PHASES[PROMOTION_STAGE_ADMISSION],
            )
        except Exception:
            self._live = None

    def __enter__(self) -> "PromotionLiveRun":
        global _active_promotion
        self._previous_promotion = _active_promotion
        self._previous_run = _active_run
        _active_promotion = self
        if self._live is not None:
            try:
                returned = threading.Event()
                threading.Thread(
                    target=self._open,
                    args=(returned,),
                    name="promotion-live-run-start",
                    daemon=True,
                ).start()
                returned.wait(PROMOTION_START_TIMEOUT_SECONDS)
            except Exception:
                self._live = None
        return self

    def stage(self, stage: str) -> None:
        """Enter ``stage`` (a key of `PROMOTION_STAGE_PHASES`); others are ignored."""
        phase = PROMOTION_STAGE_PHASES.get(stage)
        if phase is None:
            return
        self._stage = stage
        if self._live is not None:
            try:
                # Records state only; a row that opens later starts from it.
                self._live.set_phase(phase)
            except Exception:
                pass

    def __exit__(self, exc_type, exc, tb) -> None:
        global _active_promotion
        if _active_promotion is self:
            _active_promotion = self._previous_promotion
        if self._live is None:
            return
        try:
            if exc_type is None:
                final = {
                    "status": "completed",
                    "outcome_status": PROMOTION_OUTCOME_SIGNED,
                }
            elif issubclass(exc_type, Exception):
                final = {
                    "status": "failed",
                    "outcome_status": PROMOTION_OUTCOME_FAILED_PREFIX + self._stage,
                    "error_rule": f"exception:{exc_type.__name__}",
                }
            else:
                # SystemExit and KeyboardInterrupt: `exited` / `interrupted`.
                outcome_status, error_rule = abnormal_exit_outcome(exc_type, exc)
                final = {
                    "status": "failed",
                    "outcome_status": outcome_status,
                    "error_rule": error_rule,
                }
            with self._lock:
                self._final = final
                opened = self._opened
            if opened:
                self._close(final)
        except Exception:
            pass

    # -- internals -----------------------------------------------------------

    def _open(self, returned: threading.Event) -> None:
        """Open the row off the main thread; close it if the command ended."""
        try:
            self._live.__enter__()
            with self._lock:
                self._opened = True
                final = self._final
            if final is not None:
                self._close(final)
        except Exception:
            pass
        finally:
            returned.set()

    def _close(self, final: dict) -> None:
        try:
            self._live.finish(
                final["status"],
                outcome_status=final.get("outcome_status"),
                error_rule=final.get("error_rule"),
            )
        finally:
            self._restore_outer_run()

    def _restore_outer_run(self) -> None:
        """Give `report_phase` back to the run that was live before this one."""
        global _active_run
        previous = self._previous_run
        if previous is None or previous._finished:
            return
        with _active_run_lock:
            if _active_run is None or _active_run is self._live:
                _active_run = previous
