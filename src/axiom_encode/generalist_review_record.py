"""The generalist review record of one encode (#2146).

An encode runs one generalist model review of its candidate. Until this
record existed, a review that could not start, could not be read or was
skipped left no trace in the signed apply manifest: the module was applied
and nothing downstream could tell whether a model had reviewed it.

The record states what happened, in one of five statuses:

``passed``
    The reviewer ran, its output was read, it passed with a finite score of
    at least 7.0, and every file the encode applies was reviewed in full.
``partial``
    The reviewer passed the candidate it read, but that is not everything
    that is applied: another file is applied that it never read, or its
    prompt showed it only the start of a file.
``failed``
    The reviewer ran on applied content and did not pass.
``unavailable``
    No usable verdict on the applied content: the reviewer could not start,
    its output could not be read, the result was malformed, or the reviewed
    content differs from the applied content.
``skipped``
    The reviewer was not asked: ``--skip-reviewers``, or the encoder skipped
    it because deterministic validation had rejected the standalone candidate.

Only ``passed`` satisfies ``encode --apply --require-review``. No code path
here turns a missing or unreadable value into ``passed``.

"The same content" is decided by a content digest, not by file bytes. Between
the review and the apply, ``encode --apply`` always rewrites the candidate
once: it stamps ``module.source_verification`` with the resolver's attested
corpus path and source hash and dumps the YAML again. That rewrite changes
every byte digest and no rule. The content digest is the SHA-256 of the YAML
node tree (each scalar with its resolved type and its exact text), with those
two stamped keys left out. It survives the stamp, key order, quoting style
and comments, and it changes with any other edit, including one that only a
YAML 1.1 reader treats as equal (``yes`` and ``true``, ``010`` and ``8``).
The record also names the byte digests on both sides.

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
from typing import Any, Iterable, Mapping

import yaml

from axiom_encode.checks import ValidationIssue, declare_check

GENERALIST_REVIEW_FIELD = "generalist_review"
GENERALIST_REVIEW_SCHEMA = "axiom-encode/generalist-review/v1"
GENERALIST_REVIEWER_NAME = "generalist-reviewer"

STATUS_PASSED = "passed"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_SKIPPED = "skipped"
STATUSES = frozenset(
    {
        STATUS_PASSED,
        STATUS_PARTIAL,
        STATUS_FAILED,
        STATUS_UNAVAILABLE,
        STATUS_SKIPPED,
    }
)

REASON_SKIP_FLAG = "skip_reviewers_flag"
REASON_NOT_RUN = "reviewer_not_run"
REASON_REVIEWER_ERROR = "reviewer_error"
REASON_UNREADABLE = "unreadable_review_result"
REASON_CONTENT_NOT_APPLIED = "reviewed_content_not_applied"
REASON_UNREVIEWED_FILES = "unreviewed_applied_files"
REASON_TRUNCATED = "review_saw_truncated_file"
_REASONS_BY_STATUS = {
    STATUS_PASSED: frozenset({None}),
    STATUS_PARTIAL: frozenset({REASON_UNREVIEWED_FILES, REASON_TRUNCATED}),
    STATUS_FAILED: frozenset({None}),
    STATUS_UNAVAILABLE: frozenset(
        {REASON_REVIEWER_ERROR, REASON_UNREADABLE, REASON_CONTENT_NOT_APPLIED}
    ),
    STATUS_SKIPPED: frozenset({REASON_SKIP_FLAG, REASON_NOT_RUN}),
}
# Statuses whose verdict is on applied content.
_BOUND_STATUSES = frozenset({STATUS_PASSED, STATUS_PARTIAL, STATUS_FAILED})

_RECORD_WHY = (
    "The record is the only statement in a signed manifest about whether a "
    "model reviewed the applied module; a wrong record reads as a review."
)
_RECORD_FIX = (
    "Do not edit a signed manifest. Run encode --apply again so the encoder "
    "writes and signs the record itself."
)
_RECORD_SITE = (
    "axiom_encode/generalist_review_record.py:generalist_review_record_issues"
)
CHECK_REVIEW_RECORD_SHAPE = declare_check(
    "manifest:generalist-review/shape",
    title="Generalist review record is malformed",
    requires=(
        "The record has the exact v1 fields, a supported status and reason, "
        "and well-formed reviewer, verdict, digest and file values."
    ),
    why=_RECORD_WHY,
    fix=_RECORD_FIX,
    cause="encoder_output",
    gate="manifest",
    site=_RECORD_SITE,
)
CHECK_REVIEW_RECORD_VERDICT = declare_check(
    "manifest:generalist-review/verdict",
    title="Generalist review status contradicts its verdict",
    requires=(
        "A passed, partial or failed status carries the matching verdict, a "
        "pass is at or above the threshold, an unscored status carries no "
        "verdict, and a required review is passed."
    ),
    why=_RECORD_WHY,
    fix=_RECORD_FIX,
    cause="encoder_output",
    gate="manifest",
    site=_RECORD_SITE,
)
CHECK_REVIEW_RECORD_BINDING = declare_check(
    "manifest:generalist-review/binding",
    title="Generalist review verdict is not bound to the applied files",
    requires=(
        "The record lists exactly the files the manifest applies; a passed "
        "record marks every one of them reviewed in full; a scored record has "
        "equal reviewed and applied content digests."
    ),
    why=_RECORD_WHY,
    fix=_RECORD_FIX,
    cause="encoder_output",
    gate="manifest",
    site=_RECORD_SITE,
)

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
        "files",
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
_PAIR_FIELDS = frozenset(_DIGEST_FIELDS)
_FILE_FIELDS = frozenset({"sha256", "reviewed", "complete"})
# The two keys `encode --apply` stamps into module.source_verification.
_SOURCE_STAMP_KEYS = ("corpus_citation_path", "source_sha256")
_NULL_TAG = "tag:yaml.org,2002:null"
_BOOL_TAG = "tag:yaml.org,2002:bool"


def _is_empty(node: yaml.Node | None) -> bool:
    if node is None:
        return True
    if isinstance(node, yaml.ScalarNode):
        return node.tag == _NULL_TAG
    return not node.value


def _mapping_item(node: yaml.Node | None, key: str) -> yaml.Node | None:
    if not isinstance(node, yaml.MappingNode):
        return None
    for key_node, value_node in node.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
            return value_node
    return None


def _canonical_node(
    node: yaml.Node | None, *, drop: Mapping[int, frozenset[str]]
) -> object:
    """Return a JSON value that differs whenever the YAML node differs.

    A scalar keeps its resolved type tag and its exact text, so ``'1'`` and
    ``1`` differ, and so do ``yes`` and ``true``. ``drop`` names, per mapping
    node, the keys to leave out.
    """

    if node is None:
        return None
    if isinstance(node, yaml.ScalarNode):
        value = node.value
        if node.tag == _NULL_TAG:
            value = ""  # `key:`, `~` and `null` are one value, dumped as `null`
        elif node.tag == _BOOL_TAG:
            value = value.lower()  # `True` is dumped as `true`; `yes` stays `yes`
        return ["scalar", node.tag, value]
    if isinstance(node, yaml.SequenceNode):
        return ["sequence", [_canonical_node(item, drop=drop) for item in node.value]]
    if isinstance(node, yaml.MappingNode):
        dropped = drop.get(id(node), frozenset())
        return [
            "mapping",
            sorted(
                [
                    json.dumps(_canonical_node(key_node, drop=drop), sort_keys=True),
                    _canonical_node(value_node, drop=drop),
                ]
                for key_node, value_node in node.value
                if not (
                    isinstance(key_node, yaml.ScalarNode) and key_node.value in dropped
                )
            ),
        ]
    raise TypeError(f"unsupported YAML node {type(node).__name__}")


def review_content_sha256(raw: bytes | None, *, rulespec: bool) -> str | None:
    """Digest the content of one reviewed or applied YAML file.

    ``raw`` None is an absent file, which digests like an empty document, so
    an absent companion test equals the empty one that apply writes. A file
    that cannot be decoded or parsed has no content digest (None), and a
    verdict without one is never bound.
    """

    if raw is None:
        root: yaml.Node | None = None
    else:
        try:
            root = yaml.compose(raw.decode("utf-8"), Loader=yaml.SafeLoader)
        except (UnicodeError, yaml.YAMLError, RecursionError, ValueError):
            return None
    drop: dict[int, frozenset[str]] = {}
    if rulespec and isinstance(root, yaml.MappingNode):
        module = _mapping_item(root, "module")
        verification = _mapping_item(module, "source_verification")
        if isinstance(verification, yaml.MappingNode):
            drop[id(verification)] = frozenset(_SOURCE_STAMP_KEYS)
            if all(
                isinstance(key_node, yaml.ScalarNode)
                and key_node.value in _SOURCE_STAMP_KEYS
                for key_node, _value in verification.value
            ):
                # Nothing but the stamp: the mapping itself came from apply.
                drop[id(module)] = frozenset({"source_verification"})
                if len(module.value) == 1:
                    drop[id(root)] = frozenset({"module"})
        elif module is not None and _is_empty(module):
            drop[id(root)] = frozenset({"module"})
    try:
        canonical_value = _canonical_node(root, drop=drop)
        if _is_empty(root) or canonical_value == ["mapping", []]:
            canonical_value = None
        canonical = json.dumps(
            canonical_value,
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
    reviewer call itself measured (digests of the files it read, whether its
    prompt showed each one whole, and the CLI it started); it is absent when
    no scored review happened.
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


def _file_entries(
    sha256s: Iterable[object], reviewed: Mapping[str, bool]
) -> list[dict[str, object]]:
    """One entry per applied file: reviewed at all, and reviewed in full."""

    return [
        {
            "sha256": sha256,
            "reviewed": sha256 in reviewed,
            "complete": reviewed.get(sha256, False),
        }
        for sha256 in sorted({item for item in sha256s if _is_sha256(item)})
    ]


def _unscored_record(
    status: str,
    reason: str,
    *,
    required: bool,
    files: list[dict[str, object]],
) -> dict[str, object]:
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
        "reviewed": dict.fromkeys(_DIGEST_FIELDS),
        "applied": dict.fromkeys(_DIGEST_FIELDS),
        "files": files,
    }


def build_generalist_review_record(
    state: GeneralistReviewState,
    applied: Iterable[AppliedPair],
    applied_sha256s: Iterable[object] | None = None,
) -> dict[str, object]:
    """Return the record for ``state`` against the files that are applied.

    ``applied`` are the RuleSpec files the encode applies (or is about to
    apply), each with its companion test. ``applied_sha256s`` are the byte
    digests of every applied file, whether or not it is in a pair; left out,
    they are the files of the pairs. A scored verdict counts only when one
    pair has the content the reviewer read, and it is ``passed`` only when
    that pair is everything that is applied and the reviewer saw it whole.
    """

    pairs = [pair for pair in applied if isinstance(pair, AppliedPair)]
    if applied_sha256s is None:
        applied_sha256s = [
            digest
            for pair in pairs
            for digest in (pair.rulespec_sha256, pair.test_sha256)
            if digest is not None
        ]
    all_files = list(applied_sha256s)
    unreviewed = _file_entries(all_files, {})
    required = state.required is True
    if state.skipped_by_flag is True:
        return _unscored_record(
            STATUS_SKIPPED, REASON_SKIP_FLAG, required=required, files=unreviewed
        )

    verdict = state.metrics_pass
    raw_score = state.score
    if verdict is True and raw_score is None and state.prompt_sha256 is None:
        # The eval pipeline's placeholder for a reviewer it did not call.
        return _unscored_record(
            STATUS_SKIPPED, REASON_NOT_RUN, required=required, files=unreviewed
        )
    if verdict is False and raw_score is None:
        return _unscored_record(
            STATUS_UNAVAILABLE,
            REASON_REVIEWER_ERROR,
            required=required,
            files=unreviewed,
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
            STATUS_UNAVAILABLE,
            REASON_UNREADABLE,
            required=required,
            files=unreviewed,
        )

    reviewed = {field: evidence.get(f"reviewed_{field}") for field in _DIGEST_FIELDS}
    bound_pair = next(
        (
            pair
            for pair in pairs
            if _is_sha256(pair.rulespec_sha256)
            and pair.rulespec_content_sha256 == reviewed["rulespec_content_sha256"]
            and pair.test_content_sha256 == reviewed["test_content_sha256"]
            and (pair.test_sha256 is None or _is_sha256(pair.test_sha256))
        ),
        None,
    )
    if bound_pair is None:
        status = STATUS_UNAVAILABLE
        reason: str | None = REASON_CONTENT_NOT_APPLIED
        applied_digests: dict[str, object] = dict.fromkeys(_DIGEST_FIELDS)
        files = unreviewed
    else:
        applied_digests = {
            field: getattr(bound_pair, field) for field in _DIGEST_FIELDS
        }
        # Only an explicit True says the prompt showed the whole file.
        seen_whole = {
            bound_pair.rulespec_sha256: evidence.get("rulespec_complete") is True
        }
        if bound_pair.test_sha256 is not None:
            seen_whole[bound_pair.test_sha256] = evidence.get("test_complete") is True
        files = _file_entries(all_files, seen_whole)
        if not verdict:
            status, reason = STATUS_FAILED, None
        elif not files or not all(entry["reviewed"] for entry in files):
            status, reason = STATUS_PARTIAL, REASON_UNREVIEWED_FILES
        elif not all(entry["complete"] for entry in files):
            status, reason = STATUS_PARTIAL, REASON_TRUNCATED
        else:
            status, reason = STATUS_PASSED, None
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
        "reviewed": reviewed,
        "applied": applied_digests,
        "files": files,
    }


def generalist_review_record_issues(
    record: object,
    applied_sha256s: Iterable[str],
) -> list[ValidationIssue]:
    """Return every reason ``record`` is not a well-formed, bound record.

    This is the reader's check, from the manifest alone. The record must list
    exactly ``applied_sha256s`` (the manifest's own file digests). A record
    that says ``passed`` must mark every one of them reviewed in full. A
    scored record must carry the matching verdict and equal reviewed and
    applied content digests, and may mark as reviewed only the files of the
    pair it names. Any other record must mark no file reviewed.
    """

    def shape(message: str) -> ValidationIssue:
        return ValidationIssue(message, check=CHECK_REVIEW_RECORD_SHAPE)

    def verdict_issue(message: str) -> ValidationIssue:
        return ValidationIssue(message, check=CHECK_REVIEW_RECORD_VERDICT)

    def binding(message: str) -> ValidationIssue:
        return ValidationIssue(message, check=CHECK_REVIEW_RECORD_BINDING)

    if not isinstance(record, dict):
        return [shape("generalist_review is not an object")]
    if frozenset(record) != _RECORD_FIELDS:
        return [shape("generalist_review has an unexpected field set")]
    found: list[ValidationIssue] = []
    if record["schema"] != GENERALIST_REVIEW_SCHEMA:
        found += [shape("generalist_review schema is unsupported")]
    status = record["status"]
    reason = record["reason"]
    if not isinstance(status, str) or status not in STATUSES:
        return found + [shape("generalist_review status is unsupported")]
    if not (reason is None or isinstance(reason, str)) or (
        reason not in _REASONS_BY_STATUS[status]
    ):
        found += [shape("generalist_review reason does not match its status")]
    if not isinstance(record["required"], bool):
        found += [shape("generalist_review required is not a boolean")]
    elif record["required"] and status != STATUS_PASSED:
        found += [
            verdict_issue(
                "generalist_review was required but the recorded status is " + status
            )
        ]

    reviewer = record["reviewer"]
    verdict = record["verdict"]
    reviewed = record["reviewed"]
    applied_block = record["applied"]
    files = record["files"]
    if (
        not isinstance(reviewer, dict)
        or frozenset(reviewer) != _REVIEWER_FIELDS
        or not isinstance(verdict, dict)
        or frozenset(verdict) != _VERDICT_FIELDS
        or not isinstance(reviewed, dict)
        or frozenset(reviewed) != _PAIR_FIELDS
        or not isinstance(applied_block, dict)
        or frozenset(applied_block) != _PAIR_FIELDS
        or not isinstance(files, list)
        or not all(
            isinstance(entry, dict)
            and frozenset(entry) == _FILE_FIELDS
            and _is_sha256(entry["sha256"])
            and isinstance(entry["reviewed"], bool)
            and isinstance(entry["complete"], bool)
            for entry in files
        )
    ):
        return found + [shape("generalist_review has a malformed nested object")]

    listed = [entry["sha256"] for entry in files]
    if listed != sorted(set(listed)):
        found += [shape("generalist_review files are not sorted and unique")]
    if any(entry["complete"] and not entry["reviewed"] for entry in files):
        found += [shape("generalist_review marks an unreviewed file complete")]
    if set(listed) != {digest for digest in applied_sha256s if _is_sha256(digest)}:
        found += [
            binding("generalist_review does not list exactly the applied file digests")
        ]

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
        found += [shape("generalist_review reviewer identity is malformed")]

    passed = verdict["passed"]
    score = verdict["score"]
    issue_count = verdict["issue_count"]
    prompt_sha256 = record["prompt_sha256"]
    if isinstance(issue_count, bool) or not isinstance(issue_count, int):
        found += [shape("generalist_review issue_count is not an integer")]
    elif issue_count < 0:
        found += [shape("generalist_review issue_count is negative")]
    reviewed_files = {entry["sha256"] for entry in files if entry["reviewed"]}

    scored = status in _BOUND_STATUSES or reason == REASON_CONTENT_NOT_APPLIED
    if not scored:
        if (
            passed is not None
            or score is not None
            or issue_count != 0
            or prompt_sha256 is not None
            or any(reviewed[field] is not None for field in _DIGEST_FIELDS)
            or any(applied_block[field] is not None for field in _DIGEST_FIELDS)
            or reviewed_files
            or backend is not None
            or model is not None
        ):
            found += [
                verdict_issue(
                    "generalist_review carries a verdict for a review that has none"
                )
            ]
        return found

    if (
        not isinstance(passed, bool)
        or _finite_score(score) is None
        or not _is_sha256(prompt_sha256)
        or not _is_sha256(reviewed["rulespec_sha256"])
        or not _is_sha256(reviewed["rulespec_content_sha256"])
        or not (reviewed["test_sha256"] is None or _is_sha256(reviewed["test_sha256"]))
        or not _is_sha256(reviewed["test_content_sha256"])
    ):
        return found + [shape("generalist_review scored verdict is malformed")]
    if passed and float(score) < PASS_SCORE_THRESHOLD:
        found += [
            verdict_issue("generalist_review verdict passes below the pass threshold")
        ]
    if status in {STATUS_PASSED, STATUS_PARTIAL} and not passed:
        found += [
            verdict_issue(f"generalist_review status {status} contradicts its verdict")
        ]
    if status == STATUS_FAILED and passed:
        found += [
            verdict_issue("generalist_review status failed contradicts its verdict")
        ]
    if status == STATUS_UNAVAILABLE:
        if (
            any(applied_block[field] is not None for field in _DIGEST_FIELDS)
            or reviewed_files
        ):
            found += [
                binding(
                    "generalist_review says the reviewed content was not applied, "
                    "but names an applied file"
                )
            ]
        return found

    if (
        applied_block["rulespec_content_sha256"] != reviewed["rulespec_content_sha256"]
        or applied_block["test_content_sha256"] != reviewed["test_content_sha256"]
    ):
        found += [
            binding(
                "generalist_review verdict is on content that differs from the "
                "applied content"
            )
        ]
    pair_files = {
        digest
        for digest in (applied_block["rulespec_sha256"], applied_block["test_sha256"])
        if digest is not None
    }
    if (
        not _is_sha256(applied_block["rulespec_sha256"])
        or reviewed_files != pair_files
        or (
            applied_block["test_sha256"] is None
            and applied_block["test_content_sha256"] != _ABSENT_TEST_CONTENT_SHA256
        )
    ):
        found += [
            binding(
                "generalist_review verdict is not bound to the applied file digests"
            )
        ]
    everything_in_full = bool(files) and all(
        entry["reviewed"] and entry["complete"] for entry in files
    )
    if status == STATUS_PASSED and not everything_in_full:
        found += [
            binding(
                "generalist_review status passed, but an applied file was not "
                "reviewed in full"
            )
        ]
    if status == STATUS_PARTIAL:
        all_reviewed = bool(files) and all(entry["reviewed"] for entry in files)
        expected = REASON_TRUNCATED if all_reviewed else REASON_UNREVIEWED_FILES
        if everything_in_full or reason != expected:
            found += [
                binding("generalist_review status partial contradicts its file list")
            ]
    return found


def review_requirement_refusal(record: Mapping[str, object]) -> str | None:
    """Return why a required review blocks this encode, or None when it passed."""

    if record.get("status") == STATUS_PASSED:
        return None
    status = record.get("status")
    reason = record.get("reason")
    detail = f"status {status}" + (f" ({reason})" if reason else "")
    return (
        "A passed generalist review of every applied file is required "
        f"(--require-review) and this encode has none: {detail}. "
        "Nothing was applied."
    )
