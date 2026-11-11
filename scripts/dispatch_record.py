#!/usr/bin/env python3
"""Write one dispatch record for a targeted signed re-encode run.

The ``dispatch_record`` job of ``targeted-signed-reencode.yml`` runs this for
every dispatch, outside the protected environment and without any secret. The
record says how the run was triggered (queue, repair, promotion or ad hoc) and
which publication the dispatcher asked for, so a dashboard does not have to
infer that from job names or from a failure bundle that only failed encode
jobs upload.

This is telemetry: it never blocks a run. Every problem becomes a ``null``
field plus an entry in ``invalid_inputs``, and the process always exits 0.

Only values that match a narrow allowlist pattern are copied. Free-text inputs
(review findings, source bundles, import lists, paths) are recorded as
``true``/``false`` presence flags, never by value.

No package imports: the job runs it with the runner's system Python.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "axiom-encode/dispatch-record/v1"
RECORD_NAME = "dispatch-record.json"
REVIEWED_PR_PROMOTION_SCHEMA = "axiom-encode/reviewed-pr-promotion/v1"
#: The only transaction schema whose ``reviewed_candidate_promotion`` the
#: workflow honours.
PROMOTION_TRANSACTION_SCHEMA = "axiom-encode/atomic-source-transaction/v5"
#: A larger bundle is not parsed; the promotion fields are then unknown.
MAX_BUNDLE_BYTES = 1_000_000
#: The compact tag is published as a step name, so it stays short.
MAX_TAG_VALUE = 80

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[1-9][0-9]{0,19}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_BRANCH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}")
#: The canonical citation grammar of ``corpus_resolver``: a jurisdiction, then
#: hierarchy segments that may hold a space, ``.``, ``:``, ``-`` or an en dash.
_CITATION = re.compile(
    r"(?=.{1,300}\Z)[a-z]{2,3}(?:-[a-z]{2,32})*(?:/[A-Za-z0-9][A-Za-z0-9 .:\-\u2013]*)+"
)
_COUNTRY = re.compile(r"[a-z][a-z0-9-]{0,31}")
_ACTOR = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,99}(\[bot\])?")
_REPOSITORY = re.compile(r"[A-Za-z0-9._-]{1,100}/[A-Za-z0-9._-]{1,100}")
_REF = re.compile(r"refs/[A-Za-z0-9._/-]{1,200}")
_EVENT = re.compile(r"[a-z_]{1,40}")

#: Inputs copied by value when they match their pattern.
VALUE_INPUTS: dict[str, re.Pattern[str]] = {
    "citation": _CITATION,
    "country": _COUNTRY,
    "rulespec_ref": _SHA1,
    "corpus_ref": _SHA1,
    "rules_engine_ref": _SHA1,
    "pr_base_branch": _BRANCH,
    "repair_run_id": _RUN_ID,
    "queue_id": _TOKEN,
    "queue_item_id": _TOKEN,
    "queue_manifest_sha256": _SHA256,
    "queue_item_generation_sha256": _SHA256,
    "queue_dispatcher_run_id": _RUN_ID,
}

#: Inputs recorded only as present or absent. ``[]`` counts as absent for the
#: JSON list inputs, because that is their default.
PRESENCE_INPUTS: tuple[str, ...] = (
    "review_finding",
    "source_bundle_json",
    "existing_signed_imports_json",
    "replace_rulespec_path",
    "replace_legacy_rulespec_path",
    "legacy_exact_dependent_rulespec_path",
    "second_legacy_exact_dependent_rulespec_path",
    "legacy_retained_successor_rulespec_paths_json",
    "dependent_citation",
    "dependent_review_finding",
    "second_dependent_citation",
    "second_dependent_review_finding",
)

#: Run identity from the runner environment.
RUN_FIELDS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("repository", "GITHUB_REPOSITORY", _REPOSITORY),
    ("run_id", "GITHUB_RUN_ID", _RUN_ID),
    ("run_attempt", "GITHUB_RUN_ATTEMPT", _RUN_ID),
    ("event_name", "GITHUB_EVENT_NAME", _EVENT),
    ("ref", "GITHUB_REF", _REF),
    ("head_sha", "GITHUB_SHA", _SHA1),
    ("actor", "GITHUB_ACTOR", _ACTOR),
    ("triggering_actor", "GITHUB_TRIGGERING_ACTOR", _ACTOR),
)


def _text(value: Any) -> str:
    """An input as text. GitHub passes booleans and numbers as JSON scalars."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    return ""


def _present(name: str, value: Any) -> bool:
    text = _text(value).strip()
    if name.endswith("_json") and text == "[]":
        return False
    return bool(text)


def _promotion(bundle_text: str) -> tuple[bool | None, bool | None]:
    """``(live_reviewed_pr, reviewed_candidate_promotion)`` of a source bundle.

    Mirrors how the workflow reads ``source_bundle_json``: an object with the
    reviewed-PR promotion schema carries the transaction in ``transaction``;
    any other object is the transaction itself. ``None`` means the bundle
    could not be read, so the answer is unknown.
    """
    text = bundle_text.strip()
    if not text or text == "[]":
        return False, False
    if len(text.encode("utf-8", "replace")) > MAX_BUNDLE_BYTES:
        return None, None
    try:
        payload = json.loads(text)
    except (ValueError, RecursionError):
        return None, None
    if not isinstance(payload, dict):
        return False, False
    live_reviewed_pr = payload.get("schema") == REVIEWED_PR_PROMOTION_SCHEMA
    transaction = payload.get("transaction") if live_reviewed_pr else payload
    promoted = (
        isinstance(transaction, dict)
        and transaction.get("schema") == PROMOTION_TRANSACTION_SCHEMA
        and transaction.get("reviewed_candidate_promotion") is True
    )
    return live_reviewed_pr, promoted


def trigger_kind(*, is_promotion: bool | None, is_repair: bool, is_queue: bool) -> str:
    """One label per run. The three flags stay in the record beside it.

    A run can be several things at once (a queue item that replays a repair
    candidate). The label names the most specific one: promotion, then
    repair, then queue; a run that is none of them is ad hoc.
    """
    if is_promotion:
        return "promotion"
    if is_repair:
        return "repair"
    if is_queue:
        return "queue"
    return "adhoc"


def build_record(
    inputs: Any, environ: dict[str, str], *, recorded_at: str | None = None
) -> dict[str, Any]:
    """The dispatch record. Never raises on malformed inputs."""
    invalid: list[str] = []
    if not isinstance(inputs, dict):
        inputs = {}
        invalid.append("inputs")

    record: dict[str, Any] = {"schema": SCHEMA}
    for field, name, pattern in RUN_FIELDS:
        value = environ.get(name, "")
        record[field] = value if pattern.fullmatch(value) else None

    values: dict[str, str | None] = {}
    for name, pattern in VALUE_INPUTS.items():
        text = _text(inputs.get(name)).strip()
        if not text:
            values[name] = None
        elif pattern.fullmatch(text):
            values[name] = text
        else:
            values[name] = None
            invalid.append(name)

    open_pr_raw = inputs.get("open_pr")
    open_pr_text = _text(open_pr_raw).strip().lower()
    if open_pr_text in {"true", "false"}:
        open_pr: bool | None = open_pr_text == "true"
    elif open_pr_text == "":
        open_pr = False
    else:
        open_pr = None
        invalid.append("open_pr")

    live_reviewed_pr, promoted = _promotion(_text(inputs.get("source_bundle_json")))
    if live_reviewed_pr is None:
        invalid.append("source_bundle_json")
        is_promotion: bool | None = None
    else:
        is_promotion = bool(live_reviewed_pr or promoted)

    # Presence, not validity: a malformed repair_run_id still makes the
    # workflow take its repair path (and refuse there). The workflow compares
    # the raw input with '', so a whitespace-only value counts as present.
    is_repair = _text(inputs.get("repair_run_id")) != ""
    is_queue = _text(inputs.get("queue_id")) != ""

    record.update(
        {
            "trigger_kind": trigger_kind(
                is_promotion=is_promotion, is_repair=is_repair, is_queue=is_queue
            ),
            "is_queue": is_queue,
            "is_repair": is_repair,
            "is_promotion": is_promotion,
            "live_reviewed_pr": live_reviewed_pr,
            "reviewed_candidate_promotion": promoted,
            "open_pr": open_pr,
            **values,
            "inputs_present": {
                name: _present(name, inputs.get(name)) for name in PRESENCE_INPUTS
            },
            "invalid_inputs": sorted(set(invalid)),
            "recorded_at": recorded_at
            or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    return record


def _tag_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    # Values are already allowlisted; this keeps the tag one token per field.
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", text) or len(text) > MAX_TAG_VALUE:
        return "invalid"
    return text


def record_tag(record: dict[str, Any]) -> str:
    """``kind=repair open_pr=true pr_base_branch=main repair_run_id=123``.

    The workflow puts this in a step name, which the jobs API returns for
    every run. A reader gets the main fields without downloading the artifact.
    """
    return " ".join(
        f"{field}={_tag_value(record.get(source))}"
        for field, source in (
            ("kind", "trigger_kind"),
            ("open_pr", "open_pr"),
            ("pr_base_branch", "pr_base_branch"),
            ("repair_run_id", "repair_run_id"),
        )
    )


def _append(path: str, text: str) -> None:
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    environ = dict(os.environ)
    try:
        inputs = json.loads(environ.get("DISPATCH_JSON", "") or "{}")
    except (ValueError, RecursionError):
        inputs = None
    record = build_record(inputs, environ)
    tag = record_tag(record)
    text = json.dumps(record, sort_keys=True, separators=(",", ":"))
    print(f"dispatch_record {text}")
    if args:
        try:
            out_dir = Path(args[0])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / RECORD_NAME).write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError as error:
            print(f"dispatch record was not written: {error.strerror}")
    _append(environ.get("GITHUB_OUTPUT", ""), f"tag={tag}\n")
    _append(
        environ.get("GITHUB_STEP_SUMMARY", ""),
        f"**Dispatch record** `{tag}`\n\n",
    )
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as error:  # noqa: BLE001 - telemetry never fails a run
        print(f"dispatch record failed: {type(error).__name__}")
        code = 0
    raise SystemExit(code)
