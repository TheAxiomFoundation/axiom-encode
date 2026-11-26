"""The published review record of one reviewed candidate unit (#2319).

A candidate that the local encode produced is reviewed by the generalist
model reviewer on the machine that encoded it. Until this record existed the
verdict reached the candidate pull request as free text only: nothing bound
it to the bytes that were later signed, and the reviewer's findings were not
kept in a form a program could read.

The record states, for one unit (a module and its companion test):

- the citation and the module path;
- the SHA-256 of the module and of the companion **as reviewed**;
- the encoder version and commit that ran the review;
- each reviewer (today one, the generalist reviewer) with its backend, model,
  prompt digest, status and score;
- the overall status, which is the status of the embedded
  ``axiom-encode/generalist-review/v2`` record (``generalist_review_record``);
- every finding with its reviewer, its severity (``blocking`` or
  ``non_blocking``) and its text;
- the run id, the attempt number and the time of the review.

The record travels in the body of the candidate pull request, between two
fixed marker lines (``render_block``). ``extract_records`` reads it back, and
``unit_record_issues`` decides whether a record admits a unit: the status is
``passed`` and the hashes, the citation and the path are exactly those of the
bytes that are about to be signed.

**What the record is not.** It is not signed. Whoever can edit the candidate
pull request body can write one by hand. A matching record proves that a
hash-bound review record was published for exactly these bytes; it does not
prove that a model produced it. See docs/reviewed-pr-promotion.md, "Review
record".

Unlike the embedded v2 record, this record carries reviewer prose (the
findings). It is therefore never placed in a signed manifest. The text is
bounded, and ``render_block`` shows it as an inert code span, so that it
cannot form a marker line, a code fence, a mention or a link in the pull
request body. The JSON block is the record; ``extract_records`` refuses a
block whose readable part is not exactly the rendering of its JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

from axiom_encode.generalist_review_chunks import (
    MAX_REVIEW_CALLS,
    chunked_prompt_sha256,
)
from axiom_encode.generalist_review_record import (
    GENERALIST_REVIEW_FIELD,
    GENERALIST_REVIEW_SCHEMA,
    GENERALIST_REVIEWER_NAME,
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_PASSED,
    STATUSES,
    companion_test_manifest_path,
    generalist_review_record_issues,
)

SCHEMA = "axiom-encode/candidate-review-record/v1"
READ_SCHEMA = "axiom-encode/candidate-review-record-read/v1"
BEGIN_MARKER = "<!-- axiom-encode:candidate-review-record:v1:begin -->"
END_MARKER = "<!-- axiom-encode:candidate-review-record:v1:end -->"
_MARKER_PREFIX = "<!-- axiom-encode:candidate-review-record:"

SEVERITY_BLOCKING = "blocking"
SEVERITY_NON_BLOCKING = "non_blocking"
_SEVERITIES = frozenset({SEVERITY_BLOCKING, SEVERITY_NON_BLOCKING})
_NON_BLOCKING_PREFIX = "[non-blocking] "
# The closed grammar of one issue text of the generalist reviewer, as
# ``harness.validator_pipeline`` writes it: an optional call mark of a
# chunked review, then an optional non-blocking mark, then the text. A text
# without the non-blocking mark is blocking, also when it does not look like
# a finding at all: an unknown form never becomes non-blocking.
_ISSUE_TEXT = re.compile(
    r"(?:\[review call (?P<index>[1-9][0-9]{0,2}) of (?P<total>[1-9][0-9]{0,2})\] )?"
    r"(?P<non_blocking>\[non-blocking\] )?(?P<text>.*)",
    re.DOTALL,
)
STATUS_ERROR = "error"
CALL_PASSED = "passed"
CALL_FAILED = "failed"
CALL_ERROR = "error"
_CALL_STATUSES = frozenset({CALL_PASSED, CALL_FAILED, CALL_ERROR})
MAX_ERROR_CHARS = 2000

DEFAULT_MAX_FINDING_CHARS = 1000
MAX_FINDING_CHARS = 4000
MAX_FINDINGS = 64
# GitHub refuses a pull request body above 65,536 characters.
MAX_BODY_BLOCK_CHARS = 60_000
MAX_RECORDS_PER_BODY = 64

_FIELDS = frozenset(
    {
        "schema",
        "citation",
        "rulespec_path",
        "reviewed",
        "encoder",
        "status",
        "reviewers",
        "findings",
        "findings_sha256",
        "calls",
        "error",
        "raw_output_sha256",
        GENERALIST_REVIEW_FIELD,
        "run",
        "created_at",
    }
)
_REVIEWED_FIELDS = frozenset({"rulespec_sha256", "companion_sha256", "commit"})
_ENCODER_FIELDS = frozenset({"version", "commit"})
_REVIEWER_FIELDS = frozenset(
    {"name", "backend", "model", "prompt_sha256", "status", "score"}
)
_FINDING_FIELDS = frozenset(
    {"reviewer", "severity", "text", "truncated", "review_call"}
)
_REVIEW_CALL_FIELDS = frozenset({"index", "total"})
_CALL_FIELDS = frozenset({"index", "total", "status", "score", "prompt_sha256"})
_RUN_FIELDS = frozenset({"id", "attempt"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_VERSION = re.compile(r"(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})")
# No ``@``, ``/`` or ``+``: a run id is shown in a pull request body.
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,99}")
_CREATED_AT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_PATH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@+-]{0,511}")
_BODY_FENCE = re.compile(r" {0,3}(?P<fence>`{3,}|~{3,})(?P<rest>.*)")
_FENCE = re.compile(
    r"\n```json\n(?P<json>[^\n]*)\n```\n\Z",
)


class ReviewRecordError(ValueError):
    """A review record, or the text that should hold one, cannot be used."""


def _version_tuple(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, str):
        return None
    match = _VERSION.fullmatch(value)
    return None if match is None else tuple(int(part) for part in match.groups())


def parse_issue_text(issue: str) -> tuple[dict[str, int] | None, str, str]:
    """Split one reviewer issue text into (review call, severity, text).

    The call is ``{"index": N, "total": M}`` for an issue of a chunked
    review, else ``None``. The severity is ``non_blocking`` only for the
    exact mark of the pipeline; everything else is ``blocking``.
    """

    match = _ISSUE_TEXT.fullmatch(issue)
    assert match is not None  # the grammar ends in ``.*``
    call = None
    if match.group("index") is not None:
        index, total = int(match.group("index")), int(match.group("total"))
        if 1 <= index <= total <= MAX_REVIEW_CALLS:
            call = {"index": index, "total": total}
        else:
            # Not a call mark of a real plan: the whole text is the finding.
            return None, SEVERITY_BLOCKING, issue
    severity = (
        SEVERITY_NON_BLOCKING
        if match.group("non_blocking") is not None
        else SEVERITY_BLOCKING
    )
    return call, severity, match.group("text")


def findings_from_issues(
    issues: object,
    *,
    reviewer: str = GENERALIST_REVIEWER_NAME,
    max_chars: int = DEFAULT_MAX_FINDING_CHARS,
) -> list[dict[str, object]]:
    """Turn the reviewer's issue list into findings, in the reviewer's order.

    The pipeline gives issues as text only. ``parse_issue_text`` reads the
    two marks it writes; the call of a chunked review becomes the field
    ``review_call`` and is not part of the text. A text above ``max_chars``
    is cut and marked ``truncated``; a list above ``MAX_FINDINGS`` is
    refused, so a record never silently drops a finding.
    """

    if not 1 <= max_chars <= MAX_FINDING_CHARS:
        raise ReviewRecordError(
            f"the finding text bound must be from 1 to {MAX_FINDING_CHARS}"
        )
    if not isinstance(issues, list):
        return []
    if len(issues) > MAX_FINDINGS:
        raise ReviewRecordError(
            f"the review has {len(issues)} findings; a record holds at most "
            f"{MAX_FINDINGS}"
        )
    findings: list[dict[str, object]] = []
    for issue in issues:
        raw = issue if isinstance(issue, str) else json.dumps(issue, sort_keys=True)
        call, severity, text = parse_issue_text(raw)
        text = "".join(
            character if character == "\n" or character.isprintable() else " "
            for character in text.replace("\r\n", "\n").replace("\r", "\n")
        ).strip()
        truncated = len(text) > max_chars
        findings.append(
            {
                "reviewer": reviewer,
                "severity": severity,
                "text": text[:max_chars],
                "truncated": truncated,
                "review_call": call,
            }
        )
    return findings


def review_calls(
    details: object,
    *,
    passed: object,
    score: object,
    prompt_sha256: object,
) -> list[dict[str, object]]:
    """The calls of one review, in order, as a record keeps them.

    A chunked review reports its calls itself (``details["review_calls"]``).
    A review of one call reports none; its single call is the review.
    ``prompt_sha256`` is the digest of the exact prompt of the call, so of
    everything that call was shown.
    """

    def status(call_passed: object, call_score: object) -> str:
        if call_score is None:
            return CALL_ERROR
        return CALL_PASSED if call_passed is True else CALL_FAILED

    def digest(value: object) -> str | None:
        return value if isinstance(value, str) and _SHA256.fullmatch(value) else None

    def number(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if math.isfinite(value) else None

    reported = details.get("review_calls") if isinstance(details, Mapping) else None
    if isinstance(reported, list) and reported:
        return [
            {
                "index": call.get("index"),
                "total": call.get("total"),
                "status": status(call.get("passed"), number(call.get("score"))),
                "score": number(call.get("score")),
                "prompt_sha256": digest(call.get("prompt_sha256")),
            }
            for call in reported
            if isinstance(call, Mapping)
        ]
    return [
        {
            "index": 1,
            "total": 1,
            "status": status(passed, number(score)),
            "score": number(score),
            "prompt_sha256": digest(prompt_sha256),
        }
    ]


def _raw_digest(raw_output: object) -> str | None:
    if not isinstance(raw_output, str):
        return None
    return hashlib.sha256(raw_output.encode("utf-8")).hexdigest()


def build_candidate_review_record(
    *,
    citation: str,
    rulespec_path: str,
    rulespec: bytes,
    companion: bytes,
    reviewed_commit: str | None,
    encoder_version: str,
    encoder_commit: str | None,
    generalist_review: Mapping[str, object],
    issues: object,
    run_id: str | None,
    attempt: int,
    calls: Sequence[Mapping[str, object]] | None = None,
    raw_output: object = None,
    max_finding_chars: int = DEFAULT_MAX_FINDING_CHARS,
    now: datetime | None = None,
) -> dict[str, object]:
    """Build the record of one reviewed unit from its v2 review record.

    The overall status, the reviewer identity, the prompt digest and the
    score are copied from ``generalist_review``; nothing here decides a
    verdict. The result is checked with ``record_issues`` before it is
    returned, so this function cannot emit a record that a reader refuses.
    A review that cannot be recorded this way is kept by
    ``build_error_record``.
    """

    reviewer = generalist_review.get("reviewer")
    reviewer = reviewer if isinstance(reviewer, Mapping) else {}
    verdict = generalist_review.get("verdict")
    verdict = verdict if isinstance(verdict, Mapping) else {}
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    findings = findings_from_issues(issues, max_chars=max_finding_chars)
    if calls is None:
        calls = review_calls(
            None,
            passed=verdict.get("passed"),
            score=verdict.get("score"),
            prompt_sha256=generalist_review.get("prompt_sha256"),
        )
    record: dict[str, object] = {
        "schema": SCHEMA,
        "citation": citation,
        "rulespec_path": rulespec_path,
        "reviewed": {
            "rulespec_sha256": hashlib.sha256(rulespec).hexdigest(),
            "companion_sha256": hashlib.sha256(companion).hexdigest(),
            "commit": reviewed_commit,
        },
        "encoder": {"version": encoder_version, "commit": encoder_commit},
        "status": generalist_review.get("status"),
        "reviewers": [
            {
                "name": GENERALIST_REVIEWER_NAME,
                "backend": reviewer.get("backend"),
                "model": reviewer.get("model"),
                "prompt_sha256": generalist_review.get("prompt_sha256"),
                "status": generalist_review.get("status"),
                "score": verdict.get("score"),
            }
        ],
        "findings": findings,
        "findings_sha256": findings_sha256(findings),
        "calls": [dict(call) for call in calls],
        "error": None,
        "raw_output_sha256": _raw_digest(raw_output),
        # ``required`` says whether the run that made the v2 record would
        # have refused without a pass. Here the requirement belongs to the
        # signing run, so the published record always says False; a failed
        # review is then a well-formed record that admits nothing.
        GENERALIST_REVIEW_FIELD: {**generalist_review, "required": False},
        "run": {"id": run_id, "attempt": attempt},
        "created_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    issues_found = record_issues(record)
    if issues_found:
        raise ReviewRecordError(
            "the review cannot be recorded: " + "; ".join(issues_found)
        )
    return record


def build_error_record(
    *,
    citation: str,
    rulespec_path: str,
    rulespec: bytes,
    companion: bytes,
    reviewed_commit: str | None,
    encoder_version: str,
    encoder_commit: str | None,
    error: str,
    run_id: str | None,
    attempt: int,
    generalist_review: Mapping[str, object] | None = None,
    issues: object = None,
    calls: Sequence[Mapping[str, object]] = (),
    raw_output: object = None,
    max_finding_chars: int = DEFAULT_MAX_FINDING_CHARS,
    now: datetime | None = None,
) -> dict[str, object]:
    """The record of a review attempt that gave no usable record.

    The status is ``error`` and ``error`` says why. It keeps what is known:
    the two file digests, the findings that could be read, the calls, the v2
    record when one was built, and the digest of the raw reviewer output. An
    attempt therefore always leaves a record. It admits nothing.
    """

    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        findings = findings_from_issues(issues, max_chars=max_finding_chars)
    except ReviewRecordError as exc:
        findings = []
        error = f"{error}; {exc}"
    flat = " ".join("".join(c if c.isprintable() else " " for c in str(error)).split())
    record: dict[str, object] = {
        "schema": SCHEMA,
        "citation": citation,
        "rulespec_path": rulespec_path,
        "reviewed": {
            "rulespec_sha256": hashlib.sha256(rulespec).hexdigest(),
            "companion_sha256": hashlib.sha256(companion).hexdigest(),
            "commit": reviewed_commit,
        },
        "encoder": {"version": encoder_version, "commit": encoder_commit},
        "status": STATUS_ERROR,
        "reviewers": [],
        "findings": findings,
        "findings_sha256": findings_sha256(findings),
        "calls": [dict(call) for call in calls],
        "error": flat[:MAX_ERROR_CHARS] or "unknown error",
        "raw_output_sha256": _raw_digest(raw_output),
        GENERALIST_REVIEW_FIELD: (
            None if generalist_review is None else dict(generalist_review)
        ),
        "run": {"id": run_id, "attempt": attempt},
        "created_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    issues_found = record_issues(record)
    if issues_found:
        # Malformed calls or a v2 record that is not JSON data: keep less.
        record["calls"] = []
        record[GENERALIST_REVIEW_FIELD] = None
        issues_found = record_issues(record)
    if issues_found:
        raise ReviewRecordError(
            "the review error cannot be recorded: " + "; ".join(issues_found)
        )
    return record


def _call_shape_ok(calls: object) -> bool:
    return (
        isinstance(calls, list)
        and len(calls) <= MAX_REVIEW_CALLS
        and all(
            _exact(call, _CALL_FIELDS)
            and type(call["index"]) is int
            and type(call["total"]) is int
            and 1 <= call["index"] <= call["total"] <= MAX_REVIEW_CALLS
            and call["status"] in _CALL_STATUSES
            and (
                call["score"] is None
                or (
                    isinstance(call["score"], (int, float))
                    and not isinstance(call["score"], bool)
                    and math.isfinite(call["score"])
                )
            )
            and (
                call["prompt_sha256"] is None
                or (
                    isinstance(call["prompt_sha256"], str)
                    and _SHA256.fullmatch(call["prompt_sha256"]) is not None
                )
            )
            for call in calls
        )
    )


def _exact(value: object, fields: frozenset[str]) -> bool:
    return isinstance(value, dict) and set(value) == fields


def record_issues(record: object) -> list[str]:
    """Return every reason ``record`` is not a well-formed review record.

    The check uses the record alone. It is strict: an unknown or missing
    field, a value of another type, and a field that contradicts the embedded
    v2 record each give an issue. A record with an issue admits nothing.
    """

    if not _exact(record, _FIELDS):
        return ["the record does not have exactly the fields of " + SCHEMA]
    assert isinstance(record, dict)
    if record["schema"] != SCHEMA:
        return [f"the record schema is not {SCHEMA}"]
    issues: list[str] = []
    citation = record["citation"]
    path = record["rulespec_path"]
    if not isinstance(citation, str) or _PATH.fullmatch(citation) is None:
        issues.append("the citation is not a plain citation path")
    if (
        not isinstance(path, str)
        or _PATH.fullmatch(path) is None
        or not path.endswith(".yaml")
        or path.endswith(".test.yaml")
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        issues.append("the module path is not a plain RuleSpec module path")
        return issues
    reviewed = record["reviewed"]
    if (
        not _exact(reviewed, _REVIEWED_FIELDS)
        or any(
            not isinstance(reviewed[name], str)
            or _SHA256.fullmatch(reviewed[name]) is None
            for name in ("rulespec_sha256", "companion_sha256")
        )
        or not (
            reviewed["commit"] is None
            or (
                isinstance(reviewed["commit"], str)
                and _COMMIT.fullmatch(reviewed["commit"]) is not None
            )
        )
    ):
        issues.append("the reviewed digests are malformed")
        return issues
    encoder = record["encoder"]
    if (
        not _exact(encoder, _ENCODER_FIELDS)
        or _version_tuple(encoder["version"]) is None
        or not (
            encoder["commit"] is None
            or (
                isinstance(encoder["commit"], str)
                and _COMMIT.fullmatch(encoder["commit"]) is not None
            )
        )
    ):
        issues.append("the encoder identity is malformed")
    status = record["status"]
    if not isinstance(status, str) or status not in STATUSES | {STATUS_ERROR}:
        issues.append("the status is not a known review status")
    error = record["error"]
    if (status == STATUS_ERROR) is not (
        isinstance(error, str) and 0 < len(error) <= MAX_ERROR_CHARS
    ) or not (error is None or isinstance(error, str)):
        issues.append("an error text is required for the status error and for no other")
    raw_digest = record["raw_output_sha256"]
    if not (
        raw_digest is None
        or (isinstance(raw_digest, str) and _SHA256.fullmatch(raw_digest))
    ):
        issues.append("the raw output digest is malformed")
    calls = record["calls"]
    if not _call_shape_ok(calls):
        issues.append("the reviewer calls are malformed")
        calls = None
    run = record["run"]
    if (
        not _exact(run, _RUN_FIELDS)
        or not (
            run["id"] is None
            or (isinstance(run["id"], str) and _RUN_ID.fullmatch(run["id"]))
        )
        or type(run["attempt"]) is not int
        or not 1 <= run["attempt"] <= 999_999_999
    ):
        issues.append("the run id or the attempt number is malformed")
    created = record["created_at"]
    if not isinstance(created, str) or _CREATED_AT.fullmatch(created) is None:
        issues.append("the time is not in the form YYYY-MM-DDTHH:MM:SSZ")
    else:
        try:
            datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            issues.append("the time is not a calendar time")

    findings = record["findings"]
    if (
        not isinstance(findings, list)
        or len(findings) > MAX_FINDINGS
        or any(
            not _exact(finding, _FINDING_FIELDS)
            or finding["reviewer"] != GENERALIST_REVIEWER_NAME
            or finding["severity"] not in _SEVERITIES
            or not isinstance(finding["text"], str)
            or len(finding["text"]) > MAX_FINDING_CHARS
            or type(finding["truncated"]) is not bool
            or not (
                finding["review_call"] is None
                or (
                    _exact(finding["review_call"], _REVIEW_CALL_FIELDS)
                    and type(finding["review_call"]["index"]) is int
                    and type(finding["review_call"]["total"]) is int
                    and 1
                    <= finding["review_call"]["index"]
                    <= finding["review_call"]["total"]
                    <= MAX_REVIEW_CALLS
                )
            )
            for finding in findings
        )
    ):
        issues.append("the findings are malformed")
        findings = None
    elif record["findings_sha256"] != findings_sha256(findings):
        issues.append(
            "the findings differ from their digest: a finding was edited, "
            "added or removed after the record was made"
        )

    embedded = record[GENERALIST_REVIEW_FIELD]
    if status == STATUS_ERROR:
        # An error record keeps what is known and is checked for shape only:
        # it admits nothing, so nothing in it has to be consistent.
        if not (embedded is None or isinstance(embedded, dict)):
            issues.append("the embedded review record is not an object or null")
        if record["reviewers"] != []:
            issues.append("an error record names no reviewer")
        return issues
    companion_path = companion_test_manifest_path(path)
    embedded_issues = generalist_review_record_issues(
        embedded,
        [
            (path, reviewed["rulespec_sha256"]),
            (companion_path, reviewed["companion_sha256"]),
        ],
    )
    if embedded_issues:
        issues.append(
            f"the embedded {GENERALIST_REVIEW_SCHEMA} record is not valid for "
            "the two reviewed files: "
            + "; ".join(str(issue) for issue in embedded_issues[:3])
        )
        return issues
    assert isinstance(embedded, dict)
    if embedded.get("status") != status:
        issues.append("the status differs from the embedded review record")
    reviewer = embedded.get("reviewer")
    reviewer = reviewer if isinstance(reviewer, dict) else {}
    verdict = embedded.get("verdict")
    verdict = verdict if isinstance(verdict, dict) else {}
    expected_reviewer = {
        "name": GENERALIST_REVIEWER_NAME,
        "backend": reviewer.get("backend"),
        "model": reviewer.get("model"),
        "prompt_sha256": embedded.get("prompt_sha256"),
        "status": embedded.get("status"),
        "score": verdict.get("score"),
    }
    reviewers = record["reviewers"]
    if (
        not isinstance(reviewers, list)
        or len(reviewers) != 1
        or not _exact(reviewers[0], _REVIEWER_FIELDS)
        or reviewers[0] != expected_reviewer
        or (
            isinstance(reviewers[0].get("score"), float)
            and not math.isfinite(reviewers[0]["score"])
        )
    ):
        issues.append(
            "the reviewer list is not exactly the reviewer of the embedded "
            "review record"
        )
    if findings is not None:
        count = verdict.get("issue_count")
        # An unavailable or skipped review has no scored verdict: its v2
        # record counts no issue, and its findings say why no verdict exists.
        if (
            status in (STATUS_PASSED, STATUS_PARTIAL, STATUS_FAILED)
            and type(count) is int
            and count != len(findings)
        ):
            issues.append(
                "the number of findings differs from the issue count of the "
                "embedded review record"
            )
        if status == STATUS_PASSED and any(
            finding["severity"] == SEVERITY_BLOCKING for finding in findings
        ):
            issues.append("a passed record carries a blocking finding")
    if calls is not None and status == STATUS_PASSED:
        # A pass needs every planned call: made, passed, in order, and with
        # prompt digests that give the prompt digest of the review. The v2
        # record separately requires that the calls showed every byte of
        # both files (its ``coverage`` and ``files[].complete``).
        coverage = embedded.get("coverage")
        coverage = coverage if isinstance(coverage, dict) else {}
        digests = tuple(call["prompt_sha256"] for call in calls)
        total = len(calls)
        if (
            total == 0
            or [call["index"] for call in calls] != list(range(1, total + 1))
            or any(call["total"] != total for call in calls)
            or any(call["status"] != CALL_PASSED for call in calls)
            or coverage.get("calls_made") != total
            or coverage.get("calls_planned") != total
            or any(digest is None for digest in digests)
            or embedded.get("prompt_sha256")
            not in (
                {digests[0], chunked_prompt_sha256(digests)}
                if total == 1
                else {chunked_prompt_sha256(digests)}
            )
        ):
            issues.append(
                "a passed record does not list every planned reviewer call "
                "as made and passed, with the prompt digests of the review"
            )
    if calls is not None and findings is not None:
        made = {call["index"] for call in calls}
        totals = {call["total"] for call in calls}
        if any(
            finding["review_call"] is not None
            and (
                finding["review_call"]["index"] not in made
                or {finding["review_call"]["total"]} != totals
            )
            for finding in findings
        ):
            issues.append("a finding names a reviewer call that the record lacks")
    return issues


def unit_record_issues(
    record: object,
    *,
    citation: str,
    rulespec_path: str,
    rulespec_sha256: str,
    companion_sha256: str,
    encoder_version: str,
) -> list[str]:
    """Return every reason ``record`` does not admit this unit for signing.

    The unit is admitted only when the record is well formed, names this
    citation and this module path, has the status ``passed``, carries exactly
    the two digests of the bytes that are about to be signed, and comes from
    an admissible encoder version: one that is not newer than
    ``encoder_version``, the encoder that signs. (A record cannot come from an
    encoder that the signing encoder does not know.)
    """

    issues = record_issues(record)
    if issues:
        return issues
    assert isinstance(record, dict)
    if record["citation"] != citation:
        issues.append(
            f"the record is for citation {record['citation']}, not {citation}"
        )
    if record["rulespec_path"] != rulespec_path:
        issues.append(
            f"the record is for module {record['rulespec_path']}, not {rulespec_path}"
        )
    if record["status"] != STATUS_PASSED:
        embedded = record[GENERALIST_REVIEW_FIELD]
        reason = record["error"] or (
            embedded.get("reason") if isinstance(embedded, dict) else None
        )
        issues.append(
            f"the review status is {record['status']}"
            + (f" ({reason})" if reason else "")
            + ", not passed"
        )
    reviewed = record["reviewed"]
    if reviewed["rulespec_sha256"] != rulespec_sha256:
        issues.append(
            "the module changed after the review: reviewed sha256 "
            f"{reviewed['rulespec_sha256']}, candidate sha256 {rulespec_sha256}"
        )
    if reviewed["companion_sha256"] != companion_sha256:
        issues.append(
            "the companion test changed after the review: reviewed sha256 "
            f"{reviewed['companion_sha256']}, candidate sha256 {companion_sha256}"
        )
    signing = _version_tuple(encoder_version)
    recorded = _version_tuple(record["encoder"]["version"])
    if signing is None or recorded is None or recorded > signing:
        issues.append(
            f"the record names encoder version {record['encoder']['version']}, "
            f"which is newer than the signing encoder {encoder_version}"
        )
    return issues


def select_unit_record(
    records: Sequence[object], *, citation: str, rulespec_path: str
) -> dict:
    """Return the one record that names this unit, or refuse.

    A record names the unit when its citation or its module path is the
    unit's. Two such records are refused, also when they are equal: the
    author must publish one record for one unit.
    """

    named = [
        record
        for record in records
        if isinstance(record, dict)
        and (
            record.get("citation") == citation
            or record.get("rulespec_path") == rulespec_path
        )
    ]
    if not named:
        raise ReviewRecordError(
            f"no review record for {citation} ({rulespec_path}) was found"
        )
    if len(named) > 1:
        raise ReviewRecordError(
            f"{len(named)} review records name {citation} ({rulespec_path}); "
            "exactly one is required"
        )
    return named[0]


def require_passed_unit_records(
    records: Sequence[object],
    units: Iterable[tuple[str, str, str, str]],
    *,
    encoder_version: str,
) -> list[dict]:
    """Return the admitting record of every unit, or refuse the whole set.

    ``units`` holds (citation, module path, module sha256, companion sha256).
    One unit without an admitting record refuses every unit: the error names
    each refused unit and its reasons.
    """

    admitted: list[dict] = []
    refusals: list[str] = []
    for citation, rulespec_path, rulespec_sha256, companion_sha256 in units:
        try:
            record = select_unit_record(
                records, citation=citation, rulespec_path=rulespec_path
            )
        except ReviewRecordError as exc:
            refusals.append(str(exc))
            continue
        issues = unit_record_issues(
            record,
            citation=citation,
            rulespec_path=rulespec_path,
            rulespec_sha256=rulespec_sha256,
            companion_sha256=companion_sha256,
            encoder_version=encoder_version,
        )
        if issues:
            refusals.append(f"{citation} ({rulespec_path}): " + "; ".join(issues))
        else:
            admitted.append(record)
    if refusals:
        raise ReviewRecordError(
            "a passed review record bound to the exact candidate bytes is "
            "required for every unit and is missing: " + " | ".join(refusals)
        )
    return admitted


def canonical_json(value: object) -> str:
    """One-line JSON that cannot hold a marker, a fence or a line break.

    ``<``, ``>``, ``&`` and the backtick are written as ``\\uXXXX`` escapes,
    which a JSON reader turns back into the same characters.
    """

    text = json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    for character in "<>&`":
        text = text.replace(character, f"\\u{ord(character):04x}")
    return text


def record_sha256(record: object) -> str:
    """The SHA-256 of the canonical one-line JSON form of ``record``."""

    return hashlib.sha256(canonical_json(record).encode("ascii")).hexdigest()


def findings_sha256(findings: object) -> str:
    """The SHA-256 of the canonical one-line JSON of a findings list."""

    return hashlib.sha256(canonical_json(findings).encode("ascii")).hexdigest()


def _prose(text: str) -> str:
    """Finding text as one inert code span of pull request Markdown.

    Inside a code span GitHub renders no markup, no HTML, no link and no
    ``@`` mention. A backtick would end the span and an HTML comment opener
    could look like a marker line, so both are respelled here. The readable
    part is for people; the JSON block holds the exact text.
    """

    flat = " ".join(text.split()).replace("`", "'")
    flat = flat.replace("<!--", "<! --").replace("-->", "-- >")
    return f"` {flat} `"


def render_block(record: object) -> str:
    """Render one record as the fixed block of a candidate pull request body.

    The block starts and ends with a marker line. Between them it holds a
    short readable part (status, digests, reviewer, findings) and one fenced
    ``json`` block with the record on a single line. Only the JSON is read
    back; the readable part is derived from it and is for people.
    """

    issues = record_issues(record)
    if issues:
        raise ReviewRecordError(
            "the review record cannot be rendered: " + "; ".join(issues)
        )
    assert isinstance(record, dict)
    reviewed = record["reviewed"]
    reviewer = (
        record["reviewers"][0]
        if record["reviewers"]
        else {
            "name": GENERALIST_REVIEWER_NAME,
            "backend": None,
            "model": None,
            "prompt_sha256": None,
            "score": None,
        }
    )
    score = reviewer["score"]
    calls = record["calls"]
    call_line = (
        "- Reviewer calls: none recorded"
        if not calls
        else f"- Reviewer calls: {len(calls)} made of {calls[0]['total']} planned ("
        + ", ".join(f"{call['index']} {call['status']}" for call in calls)
        + ")"
    )
    lines = [
        BEGIN_MARKER,
        f"### Review record: `{record['citation']}`",
        "",
        f"- Status: **{record['status']}**"
        + ("" if score is None else f" (score {score:g}/10)"),
        f"- Module: `{record['rulespec_path']}`",
        f"- Module sha256 as reviewed: `{reviewed['rulespec_sha256']}`",
        f"- Companion sha256 as reviewed: `{reviewed['companion_sha256']}`",
        f"- Reviewer: {reviewer['name']}, backend {reviewer['backend'] or 'none'}, "
        f"model {reviewer['model'] or 'none'}, prompt sha256 "
        f"`{reviewer['prompt_sha256'] or 'none'}`",
        call_line,
        *([f"- Error: {_prose(record['error'])}"] if record["error"] else []),
        f"- Encoder: {record['encoder']['version']}"
        + (
            f" (`{record['encoder']['commit']}`)" if record["encoder"]["commit"] else ""
        ),
        f"- Run: `{record['run']['id'] or 'not named'}`, attempt "
        f"{record['run']['attempt']}, {record['created_at']}",
        f"- Record sha256: `{record_sha256(record)}`",
        "",
    ]
    findings = record["findings"]
    if findings:
        lines.append(f"Findings ({len(findings)}):")
        lines.append("")
        for number, finding in enumerate(findings, start=1):
            mark = (
                "BLOCKING"
                if finding["severity"] == SEVERITY_BLOCKING
                else "non-blocking"
            )
            lines.append(
                f"{number}. **{mark}** ({finding['reviewer']}"
                + (
                    ""
                    if finding["review_call"] is None
                    else f", call {finding['review_call']['index']} of "
                    f"{finding['review_call']['total']}"
                )
                + "): "
                + _prose(finding["text"])
                + (" [text cut]" if finding["truncated"] else "")
            )
    else:
        lines.append("Findings: none.")
    lines.extend(
        [
            "",
            "This record is not signed. It binds a published review to the "
            "exact bytes above; it does not prove that a model produced it.",
            "",
            "```json",
            canonical_json(record),
            "```",
            END_MARKER,
        ]
    )
    return "\n".join(lines) + "\n"


def render_blocks(records: Sequence[object]) -> str:
    """Render several records, in the given order, for one pull request body."""

    text = "\n".join(render_block(record) for record in records)
    if len(text) > MAX_BODY_BLOCK_CHARS:
        raise ReviewRecordError(
            f"the rendered records take {len(text)} characters; a pull request "
            f"body holds at most {MAX_BODY_BLOCK_CHARS} of them. Create the "
            "records again with a smaller --max-finding-chars"
        )
    return text


def extract_records(body: object) -> list[dict]:
    """Read every review record block from a pull request body.

    Returns the parsed JSON of each block, in order, without judging it
    (``unit_record_issues`` does that). It refuses, for the whole body: a
    marker without its partner, a block inside a block, a block without
    exactly one single-line fenced JSON object at its end, JSON that is not
    an object or repeats a key, and a marker of another record version. A
    body without any marker has no record: the result is empty.
    """

    if not isinstance(body, str):
        raise ReviewRecordError("the pull request body is not text")
    text = body.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    records: list[dict] = []
    start: int | None = None
    fence: str | None = None
    details = 0
    for index, raw_line in enumerate(lines):
        if start is None:
            # Position in the body outside a block: a marker inside a code
            # fence or a <details> element is not at the top level.
            opened = _BODY_FENCE.match(raw_line)
            if fence is None and opened is not None:
                fence = opened.group("fence")
                continue
            if fence is not None:
                if _MARKER_PREFIX in raw_line:
                    raise ReviewRecordError(
                        "a review record marker is inside a code fence"
                    )
                closed = _BODY_FENCE.match(raw_line)
                if (
                    closed is not None
                    and closed.group("fence")[0] == fence[0]
                    and len(closed.group("fence")) >= len(fence)
                    and not closed.group("rest").strip()
                ):
                    fence = None
                continue
            lowered = raw_line.lower()
            details += lowered.count("<details") - lowered.count("</details")
        if _MARKER_PREFIX in raw_line and raw_line not in (BEGIN_MARKER, END_MARKER):
            raise ReviewRecordError(
                "a review record marker is not exactly one of the two marker "
                "lines (no other text, no leading or trailing space, version v1)"
            )
        if raw_line == BEGIN_MARKER:
            if start is not None:
                raise ReviewRecordError(
                    "a review record block starts inside another block"
                )
            if details > 0:
                raise ReviewRecordError(
                    "a review record block is inside a <details> element"
                )
            start = index
        elif raw_line == END_MARKER:
            if start is None:
                raise ReviewRecordError(
                    "a review record block ends without a start marker"
                )
            block = "\n".join(lines[start + 1 : index]) + "\n"
            whole = "\n".join(lines[start : index + 1]) + "\n"
            start = None
            match = _FENCE.search(block)
            if match is None or block.count("```") != 2:
                raise ReviewRecordError(
                    "a review record block does not end with exactly one "
                    "single-line fenced json block"
                )
            record = _load_record_json(match.group("json"))
            # The JSON is the record. The readable part must be exactly what
            # ``render_block`` writes for that JSON, so the text a person
            # reads cannot differ from the record a program checks. A record
            # that cannot be rendered is kept: ``record_issues`` names why.
            try:
                expected = render_block(record)
            except ReviewRecordError:
                expected = None
            if expected is not None and whole != expected:
                raise ReviewRecordError(
                    "a review record block is not in the exact rendered form: "
                    "its readable part differs from its JSON. Render the "
                    "record again with review-record render"
                )
            records.append(record)
            if len(records) > MAX_RECORDS_PER_BODY:
                raise ReviewRecordError("the body holds too many review records")
    if start is not None:
        raise ReviewRecordError("a review record block has no end marker")
    return records


def _load_record_json(text: str) -> dict:
    def unique(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ReviewRecordError(
                    "the review record JSON repeats the key " + json.dumps(key)
                )
            result[key] = value
        return result

    def refuse_constant(name: str) -> object:
        raise ReviewRecordError("the review record JSON holds " + name)

    try:
        value = json.loads(
            text, object_pairs_hook=unique, parse_constant=refuse_constant
        )
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, ReviewRecordError):
            raise
        raise ReviewRecordError("the review record is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ReviewRecordError("the review record JSON is not an object")
    return value


def load_record_file_text(text: str) -> dict:
    """Read one record from the text of a JSON record file."""

    return _load_record_json(text)


def build_read_document(
    *,
    rulespec_pr: int,
    head_sha: str,
    body: str,
    records: Sequence[object],
    run_id: str,
    run_attempt: int,
    now: datetime | None = None,
) -> dict[str, object]:
    """What the signing run read from the candidate pull request, once."""

    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "schema": READ_SCHEMA,
        "rulespec_pr": rulespec_pr,
        "head_sha": head_sha,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "body_chars": len(body),
        "read_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "signing_run_id": run_id,
        "signing_run_attempt": run_attempt,
        "records": list(records),
    }


_READ_FIELDS = frozenset(
    {
        "schema",
        "rulespec_pr",
        "head_sha",
        "body_sha256",
        "body_chars",
        "read_at",
        "signing_run_id",
        "signing_run_attempt",
        "records",
    }
)


def load_read_document(text: str) -> dict:
    """Read the document that ``build_read_document`` wrote, strictly."""

    document = _load_record_json(text)
    if (
        set(document) != _READ_FIELDS
        or document["schema"] != READ_SCHEMA
        or type(document["rulespec_pr"]) is not int
        or not isinstance(document["head_sha"], str)
        or _COMMIT.fullmatch(document["head_sha"]) is None
        or not isinstance(document["body_sha256"], str)
        or _SHA256.fullmatch(document["body_sha256"]) is None
        or not isinstance(document["signing_run_id"], str)
        or type(document["signing_run_attempt"]) is not int
        or not isinstance(document["records"], list)
        or len(document["records"]) > MAX_RECORDS_PER_BODY
    ):
        raise ReviewRecordError(
            f"the review record file is not a {READ_SCHEMA} document"
        )
    return document


def render_read_summary(
    document: Mapping[str, object], admitted: Sequence[dict]
) -> str:
    """The section of the run summary and of the lane pull request body.

    It names what was read (pull request, head, body digest), the signing
    run and attempt, and each admitted record in full. When the records do
    not fit a pull request body, it lists status and digests only and points
    to the run artifact.
    """

    head = [
        "## Review record",
        "",
        "Review record required: yes",
        f"Candidate pull request: #{document['rulespec_pr']} at head "
        f"`{document['head_sha']}`",
        f"Pull request body sha256 (read once, {document['read_at']}): "
        f"`{document['body_sha256']}`",
        f"Signing run: {document['signing_run_id']}, attempt "
        f"{document['signing_run_attempt']}",
        "",
        "Each unit below has a published review record with the status "
        "`passed` whose digests equal the signed bytes. The record is not "
        "signed: it does not prove that a model produced it.",
        "",
    ]
    full = "\n".join(head) + "\n".join(render_block(record) for record in admitted)
    if len(full) <= MAX_BODY_BLOCK_CHARS:
        return full
    short = list(head)
    for record in admitted:
        short.append(
            f"- `{record['citation']}`: status {record['status']}, module sha256 "
            f"`{record['reviewed']['rulespec_sha256']}`, companion sha256 "
            f"`{record['reviewed']['companion_sha256']}`, record attempt "
            f"{record['run']['attempt']}, {len(record['findings'])} findings, "
            f"record sha256 `{record_sha256(record)}`"
        )
    short.extend(
        [
            "",
            "The records are too long for this body. They are in "
            "`candidate-review-records.json` of the run artifact and in the "
            "candidate pull request.",
        ]
    )
    return "\n".join(short) + "\n"
