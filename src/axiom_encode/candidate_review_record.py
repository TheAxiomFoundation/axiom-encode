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
bounded, and ``render_block`` escapes it so that it cannot form a marker line
or a code fence in the pull request body.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

from axiom_encode.generalist_review_record import (
    GENERALIST_REVIEW_FIELD,
    GENERALIST_REVIEW_SCHEMA,
    GENERALIST_REVIEWER_NAME,
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
_FINDING_FIELDS = frozenset({"reviewer", "severity", "text", "truncated"})
_RUN_FIELDS = frozenset({"id", "attempt"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_VERSION = re.compile(r"(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}")
_CREATED_AT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_PATH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@+-]{0,511}")
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


def findings_from_issues(
    issues: object,
    *,
    reviewer: str = GENERALIST_REVIEWER_NAME,
    max_chars: int = DEFAULT_MAX_FINDING_CHARS,
) -> list[dict[str, object]]:
    """Turn the reviewer's issue list into findings, in the reviewer's order.

    The generalist reviewer returns blocking and non-blocking issues; the
    validator pipeline joins them into one list and marks a non-blocking one
    with a ``[non-blocking] `` prefix. Every other entry is blocking. A text
    above ``max_chars`` is cut and marked ``truncated``; a list above
    ``MAX_FINDINGS`` is refused, so a record never silently drops a finding.
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
        text = issue if isinstance(issue, str) else json.dumps(issue, sort_keys=True)
        severity = SEVERITY_BLOCKING
        if text.startswith(_NON_BLOCKING_PREFIX):
            severity = SEVERITY_NON_BLOCKING
            text = text[len(_NON_BLOCKING_PREFIX) :]
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
            }
        )
    return findings


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
    max_finding_chars: int = DEFAULT_MAX_FINDING_CHARS,
    now: datetime | None = None,
) -> dict[str, object]:
    """Build the record of one reviewed unit from its v2 review record.

    The overall status, the reviewer identity, the prompt digest and the
    score are copied from ``generalist_review``; nothing here decides a
    verdict. The result is checked with ``record_issues`` before it is
    returned, so this function cannot emit a record that a reader refuses.
    """

    reviewer = generalist_review.get("reviewer")
    reviewer = reviewer if isinstance(reviewer, Mapping) else {}
    verdict = generalist_review.get("verdict")
    verdict = verdict if isinstance(verdict, Mapping) else {}
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
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
        "findings": findings_from_issues(issues, max_chars=max_finding_chars),
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
    if not isinstance(status, str) or status not in STATUSES:
        issues.append("the status is not a known review status")
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
            for finding in findings
        )
    ):
        issues.append("the findings are malformed")
        findings = None

    embedded = record[GENERALIST_REVIEW_FIELD]
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
        if type(count) is int and count != len(findings):
            issues.append(
                "the number of findings differs from the issue count of the "
                "embedded review record"
            )
        if status == STATUS_PASSED and any(
            finding["severity"] == SEVERITY_BLOCKING for finding in findings
        ):
            issues.append("a passed record carries a blocking finding")
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
        reason = record[GENERALIST_REVIEW_FIELD].get("reason")
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


def _prose(text: str) -> str:
    """Finding text as one inert line of pull request Markdown."""

    return "".join(
        f"&#{ord(character)};" if character in "<>&`*_[]|~#\\" else character
        for character in " ".join(text.split())
    )


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
    reviewer = record["reviewers"][0]
    score = reviewer["score"]
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
        f"- Encoder: {record['encoder']['version']}"
        + (
            f" (`{record['encoder']['commit']}`)" if record["encoder"]["commit"] else ""
        ),
        f"- Run: {record['run']['id'] or 'not named'}, attempt "
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
                f"{number}. **{mark}** ({finding['reviewer']}): "
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
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line.startswith(_MARKER_PREFIX) and line not in (BEGIN_MARKER, END_MARKER):
            raise ReviewRecordError(
                "the pull request body holds a review record marker of an unknown form"
            )
        if line == BEGIN_MARKER:
            if start is not None:
                raise ReviewRecordError(
                    "a review record block starts inside another block"
                )
            start = index
        elif line == END_MARKER:
            if start is None:
                raise ReviewRecordError(
                    "a review record block ends without a start marker"
                )
            block = "\n".join(lines[start + 1 : index]) + "\n"
            start = None
            match = _FENCE.search(block)
            if match is None or block.count("```") != 2:
                raise ReviewRecordError(
                    "a review record block does not end with exactly one "
                    "single-line fenced json block"
                )
            records.append(_load_record_json(match.group("json")))
            if len(records) > MAX_RECORDS_PER_BODY:
                raise ReviewRecordError("the body holds too many review records")
        elif _MARKER_PREFIX in raw_line:
            raise ReviewRecordError(
                "a review record marker is not on a line of its own"
            )
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
            f"- `{record['citation']}`: {record['status']}, module sha256 "
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
