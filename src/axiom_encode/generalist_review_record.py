"""The generalist review record of one encode (#2146).

An encode runs one generalist model review of its candidate. Until this
record existed, a review that could not start, could not be read or was
skipped left no trace in the signed apply manifest: the module was applied
and nothing downstream could tell whether a model had reviewed it.

The record states what happened, in one of four statuses:

``passed``
    The reviewer ran, its output was read, it passed with a finite score of
    at least 7.0, and the content it reviewed is the content that was applied.
``failed``
    The same, but the reviewer did not pass.
``unavailable``
    No usable verdict on the applied content: the reviewer could not start,
    its output could not be read, the result was malformed, or the reviewed
    content differs from the applied content.
``skipped``
    The reviewer was not asked: ``--skip-reviewers``, or the encoder skipped
    it because deterministic validation had rejected the standalone candidate.

Everything that is not a complete, readable, content-bound pass is something
other than ``passed``. No code path here turns a missing or unreadable value
into ``passed``.

"The same content" is decided by a content digest, not by file bytes. Between
the review and the apply, ``encode --apply`` always rewrites the candidate
once: it stamps ``module.source_verification`` with the resolver's attested
corpus path and source hash and dumps the YAML again. That rewrite changes
every byte digest and no rule. The content digest is the SHA-256 of the parsed
YAML document in a canonical form, with those two stamped keys left out, so it
survives the stamp and changes with any other edit (a repaired formula, a
changed test case). The record also names the byte digests on both sides.

The record holds only enumerated words, numbers and digests. Reviewer prose
(issues, reasoning) never enters it, so model output cannot place a string of
its choice in a signed manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping

import yaml

GENERALIST_REVIEW_FIELD = "generalist_review"
GENERALIST_REVIEW_SCHEMA = "axiom-encode/generalist-review/v1"
GENERALIST_REVIEWER_NAME = "generalist-reviewer"
REQUIRE_REVIEW_ENV = "AXIOM_ENCODE_REQUIRE_REVIEW"

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_SKIPPED = "skipped"
STATUSES = frozenset({STATUS_PASSED, STATUS_FAILED, STATUS_UNAVAILABLE, STATUS_SKIPPED})

REASON_SKIP_FLAG = "skip_reviewers_flag"
REASON_NOT_RUN = "reviewer_not_run"
REASON_REVIEWER_ERROR = "reviewer_error"
REASON_UNREADABLE = "unreadable_review_result"
REASON_CONTENT_NOT_APPLIED = "reviewed_content_not_applied"
_REASONS_BY_STATUS = {
    STATUS_PASSED: frozenset({None}),
    STATUS_FAILED: frozenset({None}),
    STATUS_UNAVAILABLE: frozenset(
        {REASON_REVIEWER_ERROR, REASON_UNREADABLE, REASON_CONTENT_NOT_APPLIED}
    ),
    STATUS_SKIPPED: frozenset({REASON_SKIP_FLAG, REASON_NOT_RUN}),
}

PASS_SCORE_THRESHOLD = 7.0
_MAX_SCORE = 10.0
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVIEWER_BACKENDS = frozenset({"claude", "codex"})
_REVIEWER_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}")

_RECORD_FIELDS = frozenset(
    {
        "schema",
        "status",
        "reason",
        "required",
        "reviewer",
        "verdict",
        "prompt_sha256",
        "reviewed",
        "applied",
    }
)
_REVIEWER_FIELDS = frozenset({"name", "backend", "model"})
_VERDICT_FIELDS = frozenset({"passed", "score", "issue_count"})
_DIGEST_FIELDS = (
    "rulespec_sha256",
    "rulespec_content_sha256",
    "test_sha256",
    "test_content_sha256",
)
_REVIEWED_FIELDS = frozenset({*_DIGEST_FIELDS, "content_complete"})
_APPLIED_FIELDS = frozenset(_DIGEST_FIELDS)
# The two keys `encode --apply` stamps into module.source_verification.
_SOURCE_STAMP_KEYS = ("corpus_citation_path", "source_sha256")


def _canonical(value: object) -> object:
    """Return a JSON value that differs whenever the YAML value differs."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__float__": repr(value)}
    if isinstance(value, date):  # datetime is a date
        return {"__date__": type(value).__name__ + ":" + value.isoformat()}
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"__set__": sorted(json.dumps(_canonical(item)) for item in value)}
    if isinstance(value, dict):
        return {
            "__map__": sorted(
                [json.dumps(_canonical(key), sort_keys=True), _canonical(item)]
                for key, item in value.items()
            )
        }
    raise TypeError(f"unsupported YAML value {type(value).__name__}")


def review_content_sha256(raw: bytes | None, *, rulespec: bool) -> str | None:
    """Digest the content of one reviewed or applied YAML file.

    ``raw`` None is an absent file, which digests like an empty document, so
    an absent companion test equals the empty one that apply writes. A file
    that cannot be decoded or parsed has no content digest (None), and a
    verdict without one is never bound.
    """

    if raw is None:
        document: object = None
    else:
        try:
            document = yaml.safe_load(raw.decode("utf-8"))
        except (UnicodeError, yaml.YAMLError, RecursionError, ValueError):
            return None
    if rulespec and isinstance(document, dict):
        document = dict(document)
        module = document.get("module")
        if isinstance(module, dict):
            module = dict(module)
            verification = module.get("source_verification")
            if isinstance(verification, dict):
                verification = {
                    key: item
                    for key, item in verification.items()
                    if key not in _SOURCE_STAMP_KEYS
                }
                if verification:
                    module["source_verification"] = verification
                else:
                    del module["source_verification"]
            document["module"] = module
        if document.get("module", 0) in (None, {}):
            del document["module"]
    if document in (None, [], {}, ""):
        document = None
    try:
        canonical = json.dumps(
            _canonical(document),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        return None
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


_ABSENT_TEST_CONTENT_SHA256 = review_content_sha256(None, rulespec=False)


@dataclass(frozen=True)
class AppliedPair:
    """One applied RuleSpec file and its companion test, by digest."""

    rulespec_sha256: str
    rulespec_content_sha256: str | None
    test_sha256: str | None
    test_content_sha256: str | None


def applied_pair(rulespec: bytes, test: bytes | None) -> AppliedPair:
    """Digest one RuleSpec file and its companion test (None when absent)."""

    return AppliedPair(
        rulespec_sha256=hashlib.sha256(rulespec).hexdigest(),
        rulespec_content_sha256=review_content_sha256(rulespec, rulespec=True),
        test_sha256=None if test is None else hashlib.sha256(test).hexdigest(),
        test_content_sha256=review_content_sha256(test, rulespec=False),
    )


def review_evidence_digests(rulespec: bytes, test: bytes | None) -> dict[str, object]:
    """The digests a reviewer call records for the files it read."""

    pair = applied_pair(rulespec, test)
    return {
        "reviewed_rulespec_sha256": pair.rulespec_sha256,
        "reviewed_rulespec_content_sha256": pair.rulespec_content_sha256,
        "reviewed_test_sha256": pair.test_sha256,
        "reviewed_test_content_sha256": pair.test_content_sha256,
    }


@dataclass(frozen=True)
class GeneralistReviewState:
    """What one encode attempt knows about its generalist review.

    ``metrics_pass``, ``score``, ``issue_count`` and ``prompt_sha256`` are the
    values the eval pipeline already produces. ``evidence`` is what the
    reviewer call itself measured (digests of the bytes it read, and the CLI
    it started); it is absent when no scored review happened.
    """

    metrics_pass: object = None
    score: object = None
    issue_count: object = 0
    prompt_sha256: object = None
    evidence: object = None
    skipped_by_flag: bool = False
    required: bool = False


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _finite_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > _MAX_SCORE:
        return None
    return number


def _issue_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _reviewer_identity(evidence: Mapping[str, Any]) -> dict[str, object]:
    backend = evidence.get("reviewer_backend")
    model = evidence.get("reviewer_model")
    if backend not in _REVIEWER_BACKENDS:
        backend = None
    if not isinstance(model, str) or _REVIEWER_MODEL.fullmatch(model) is None:
        model = None
    return {
        "name": GENERALIST_REVIEWER_NAME,
        "backend": backend,
        "model": model if backend is not None else None,
    }


def _unscored_record(status: str, reason: str, *, required: bool) -> dict[str, object]:
    return {
        "schema": GENERALIST_REVIEW_SCHEMA,
        "status": status,
        "reason": reason,
        "required": required,
        "reviewer": {
            "name": GENERALIST_REVIEWER_NAME,
            "backend": None,
            "model": None,
        },
        "verdict": {"passed": None, "score": None, "issue_count": 0},
        "prompt_sha256": None,
        "reviewed": {**dict.fromkeys(_DIGEST_FIELDS), "content_complete": None},
        "applied": dict.fromkeys(_DIGEST_FIELDS),
    }


def build_generalist_review_record(
    state: GeneralistReviewState,
    applied: Iterable[AppliedPair],
) -> dict[str, object]:
    """Return the record for ``state`` against the files that are applied.

    ``applied`` are the RuleSpec files the encode applies (or is about to
    apply), each with its companion test. A scored verdict counts only when
    one of them has the content the reviewer read, in both files.
    """

    required = state.required is True
    if state.skipped_by_flag is True:
        return _unscored_record(STATUS_SKIPPED, REASON_SKIP_FLAG, required=required)

    verdict = state.metrics_pass
    raw_score = state.score
    if verdict is True and raw_score is None and state.prompt_sha256 is None:
        # The eval pipeline's placeholder for a reviewer it did not call.
        return _unscored_record(STATUS_SKIPPED, REASON_NOT_RUN, required=required)
    if verdict is False and raw_score is None:
        return _unscored_record(
            STATUS_UNAVAILABLE, REASON_REVIEWER_ERROR, required=required
        )

    score = _finite_score(raw_score)
    evidence = state.evidence
    if (
        not isinstance(verdict, bool)
        or score is None
        or not _is_sha256(state.prompt_sha256)
        or not isinstance(evidence, Mapping)
        or not _is_sha256(evidence.get("reviewed_rulespec_sha256"))
        or not _is_sha256(evidence.get("reviewed_rulespec_content_sha256"))
        or not (
            evidence.get("reviewed_test_sha256") is None
            or _is_sha256(evidence.get("reviewed_test_sha256"))
        )
        or not _is_sha256(evidence.get("reviewed_test_content_sha256"))
        or (verdict is True and score < PASS_SCORE_THRESHOLD)
    ):
        return _unscored_record(
            STATUS_UNAVAILABLE, REASON_UNREADABLE, required=required
        )

    reviewed = {field: evidence.get(f"reviewed_{field}") for field in _DIGEST_FIELDS}
    bound_pair = next(
        (
            pair
            for pair in applied
            if isinstance(pair, AppliedPair)
            and _is_sha256(pair.rulespec_sha256)
            and pair.rulespec_content_sha256 == reviewed["rulespec_content_sha256"]
            and pair.test_content_sha256 == reviewed["test_content_sha256"]
            and (pair.test_sha256 is None or _is_sha256(pair.test_sha256))
        ),
        None,
    )
    content_complete = evidence.get("content_complete")
    if bound_pair is not None:
        status = STATUS_PASSED if verdict else STATUS_FAILED
        reason = None
        applied_digests: dict[str, object] = {
            field: getattr(bound_pair, field) for field in _DIGEST_FIELDS
        }
    else:
        status = STATUS_UNAVAILABLE
        reason = REASON_CONTENT_NOT_APPLIED
        applied_digests = dict.fromkeys(_DIGEST_FIELDS)
    return {
        "schema": GENERALIST_REVIEW_SCHEMA,
        "status": status,
        "reason": reason,
        "required": required,
        "reviewer": _reviewer_identity(evidence),
        "verdict": {
            "passed": verdict,
            "score": score,
            "issue_count": _issue_count(state.issue_count),
        },
        "prompt_sha256": state.prompt_sha256,
        "reviewed": {
            **reviewed,
            "content_complete": (
                content_complete if isinstance(content_complete, bool) else None
            ),
        },
        "applied": applied_digests,
    }


def generalist_review_record_issues(
    record: object,
    applied_sha256s: Iterable[str],
) -> list[str]:
    """Return every reason ``record`` is not a well-formed, bound record.

    This is the reader's check, from the manifest alone. A record that says
    ``passed`` or ``failed`` must carry a matching verdict, equal reviewed and
    applied content digests, and applied byte digests that are among
    ``applied_sha256s`` (the manifest's own file digests). A record that says
    the reviewed content was not applied must name no applied file.
    """

    if not isinstance(record, dict):
        return ["generalist_review is not an object"]
    if frozenset(record) != _RECORD_FIELDS:
        return ["generalist_review does not match the exact v1 field set"]
    issues: list[str] = []
    if record["schema"] != GENERALIST_REVIEW_SCHEMA:
        issues.append("generalist_review schema is unsupported")
    status = record["status"]
    reason = record["reason"]
    if not isinstance(status, str) or status not in STATUSES:
        return issues + ["generalist_review status is unsupported"]
    if not (reason is None or isinstance(reason, str)) or (
        reason not in _REASONS_BY_STATUS[status]
    ):
        issues.append("generalist_review reason does not match its status")
    if not isinstance(record["required"], bool):
        issues.append("generalist_review required is not a boolean")
    elif record["required"] and status != STATUS_PASSED:
        issues.append(
            "generalist_review was required but the recorded status is " + status
        )

    reviewer = record["reviewer"]
    verdict = record["verdict"]
    reviewed = record["reviewed"]
    applied_block = record["applied"]
    if (
        not isinstance(reviewer, dict)
        or frozenset(reviewer) != _REVIEWER_FIELDS
        or not isinstance(verdict, dict)
        or frozenset(verdict) != _VERDICT_FIELDS
        or not isinstance(reviewed, dict)
        or frozenset(reviewed) != _REVIEWED_FIELDS
        or not isinstance(applied_block, dict)
        or frozenset(applied_block) != _APPLIED_FIELDS
    ):
        return issues + ["generalist_review has a malformed nested object"]

    backend = reviewer["backend"]
    model = reviewer["model"]
    if (
        reviewer["name"] != GENERALIST_REVIEWER_NAME
        or not (backend is None or backend in _REVIEWER_BACKENDS)
        or not (
            model is None
            or (
                backend is not None
                and isinstance(model, str)
                and _REVIEWER_MODEL.fullmatch(model) is not None
            )
        )
    ):
        issues.append("generalist_review reviewer identity is malformed")

    passed = verdict["passed"]
    score = verdict["score"]
    issue_count = verdict["issue_count"]
    prompt_sha256 = record["prompt_sha256"]
    content_complete = reviewed["content_complete"]
    if isinstance(issue_count, bool) or not isinstance(issue_count, int):
        issues.append("generalist_review issue_count is not an integer")
    elif issue_count < 0:
        issues.append("generalist_review issue_count is negative")

    scored = reason in {None, REASON_CONTENT_NOT_APPLIED}
    if not scored:
        if (
            passed is not None
            or score is not None
            or issue_count != 0
            or prompt_sha256 is not None
            or any(reviewed[field] is not None for field in _DIGEST_FIELDS)
            or any(applied_block[field] is not None for field in _DIGEST_FIELDS)
            or content_complete is not None
            or backend is not None
            or model is not None
        ):
            issues.append(
                "generalist_review carries a verdict for a review that has none"
            )
        return issues

    if (
        not isinstance(passed, bool)
        or _finite_score(score) is None
        or not _is_sha256(prompt_sha256)
        or not _is_sha256(reviewed["rulespec_sha256"])
        or not _is_sha256(reviewed["rulespec_content_sha256"])
        or not (reviewed["test_sha256"] is None or _is_sha256(reviewed["test_sha256"]))
        or not _is_sha256(reviewed["test_content_sha256"])
        or not (content_complete is None or isinstance(content_complete, bool))
    ):
        return issues + ["generalist_review scored verdict is malformed"]
    if passed and float(score) < PASS_SCORE_THRESHOLD:
        issues.append("generalist_review verdict passes below the pass threshold")
    if status == STATUS_PASSED and not passed:
        issues.append("generalist_review status passed contradicts its verdict")
    if status == STATUS_FAILED and passed:
        issues.append("generalist_review status failed contradicts its verdict")
    if status == STATUS_UNAVAILABLE:
        if any(applied_block[field] is not None for field in _DIGEST_FIELDS):
            issues.append(
                "generalist_review says the reviewed content was not applied, "
                "but names an applied file"
            )
        return issues
    manifest_digests = {digest for digest in applied_sha256s if _is_sha256(digest)}
    if (
        applied_block["rulespec_content_sha256"] != reviewed["rulespec_content_sha256"]
        or applied_block["test_content_sha256"] != reviewed["test_content_sha256"]
    ):
        issues.append(
            "generalist_review verdict is on content that differs from the "
            "applied content"
        )
    if applied_block["rulespec_sha256"] not in manifest_digests or not (
        applied_block["test_sha256"] in manifest_digests
        or (
            applied_block["test_sha256"] is None
            and applied_block["test_content_sha256"] == _ABSENT_TEST_CONTENT_SHA256
        )
    ):
        issues.append(
            "generalist_review verdict is not bound to the applied file digests"
        )
    return issues


def require_review_setting(
    flag: object,
    environ: Mapping[str, str],
) -> bool:
    """Return whether this encode requires a passed generalist review.

    ``--require-review`` or ``AXIOM_ENCODE_REQUIRE_REVIEW=1`` turns it on.
    An unset, empty or ``0`` variable leaves the flag in charge. Any other
    value refuses: a typo must not silently mean "not required".
    """

    raw = environ.get(REQUIRE_REVIEW_ENV)
    if raw is None or raw in {"", "0"}:
        return flag is True
    if raw == "1":
        return True
    raise ValueError(
        f"{REQUIRE_REVIEW_ENV} must be 1, 0 or unset; refusing an unreadable value"
    )


def review_requirement_refusal(record: Mapping[str, object]) -> str | None:
    """Return why a required review blocks this encode, or None when it passed."""

    if record.get("status") == STATUS_PASSED:
        return None
    status = record.get("status")
    reason = record.get("reason")
    detail = f"status {status}" + (f" ({reason})" if reason else "")
    return (
        "A passed generalist review is required (--require-review) and this "
        f"encode has none: {detail}. Nothing was applied."
    )
