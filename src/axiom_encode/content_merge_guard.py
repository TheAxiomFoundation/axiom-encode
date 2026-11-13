"""Content-bound merge guard: evaluate one RuleSpec pull request.

INACTIVE. Nothing in the encoder, a workflow or a RuleSpec repository calls
this module, no check run is written, and the manifest schema it reads is not
emitted by any writer. See ``docs/content-bound-merge-guard.md``.

Applicability is decided from the paths and modes of the complete tree delta,
never from a label inside a file the pull request supplies. A pull request
that changes RuleSpec content or an apply manifest is judged. When it carries
one signed ``applied-rulespec/v6`` promotion unit, the evaluator decides
whether it also carries a live approval, from an enrolled owner who is
independent of the candidate author, of the dispatcher and of every commit
author, on a commit whose primary, companion and manifest blobs are the blobs
at the current head. Every read is live and goes through one injected
``fetch`` callable. Any error, truncation or disagreement gives ``fail``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

from axiom_encode.prepare_signed_backfill import (
    RULESPEC_ATOMIC_ROOTS,
    citation_rulespec_path,
)
from axiom_encode.reviewed_pr_admission import (
    TRUSTED_SIGNING_REVIEWERS,
    AdmissionError,
    PromotionSelector,
    _leaf_entries,
    _object,
    _pages,
    _pr_identity,
    _require_immutable_delta,
    github_json,
)

# Schemas. The v6 manifest and its attestation scope do not exist yet; the
# evaluator accepts exactly the shape below and nothing else.
VERDICT_SCHEMA = "axiom-encode/content-bound-merge-guard-verdict/v1"
MERGE_REQUIREMENT = "axiom-encode/content-bound-merge-guard/v1"
MANIFEST_SCHEMA_V6 = "axiom-encode/applied-rulespec/v6"
ATTESTATION_SCOPE_SCHEMA = "axiom-encode/attestation-scope/v1"
ATTESTATION_VALIDATION = "encoder-main-complete-source"
ATTESTATION_OWNER_REVIEW = "none"
ATTESTATION_ORIGIN = "local-candidate"
PROMOTION_TOOL = "axiom-encode promote-reviewed-candidate"
MANIFEST_DIR = ".axiom/encoding-manifests"
MANIFEST_PREFIX = MANIFEST_DIR + "/"
# Earlier apply-manifest schemas. Their lanes keep their own controls (the
# approval is checked before signing); this guard does not judge them.
EARLIER_MANIFEST_SCHEMAS = frozenset({"axiom-encode/applied-rulespec/v5"})

_V6_COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "generated_at",
        "tool",
        "axiom_encode_version",
        "axiom_encode_git",
        "run_id",
        "citation",
        "candidate_rulespec_ref",
        "applied_files",
        "source_attestation",
        "validation_execution",
        "signature",
        "attestation_scope",
    }
)
# The repository waiver-set digest, or the per-file waiver bindings (#2092).
V6_MANIFEST_FIELD_VARIANTS = (
    _V6_COMMON_FIELDS | {"validation_waiver_set_sha256"},
    _V6_COMMON_FIELDS | {"validation_waiver_bindings"},
)
ATTESTATION_SCOPE_FIELDS = frozenset(
    {
        "schema",
        "validation",
        "owner_review",
        "origin",
        "candidate",
        "dispatcher",
        "merge_requirement",
    }
)
ATTESTATION_CANDIDATE_FIELDS = frozenset(
    {"repository", "pull_request", "head_sha", "base_sha", "author_id"}
)
ATTESTATION_DISPATCHER_FIELDS = frozenset({"id", "login"})

# Live policy the evaluator checks against.
ORGANISATION = "TheAxiomFoundation"
ENCODER_REPOSITORY = ORGANISATION + "/axiom-encode"
ENCODER_ROOT = "/repos/" + ENCODER_REPOSITORY
SIGNING_WORKFLOW_PATH = ".github/workflows/targeted-signed-reencode.yml"
REQUIRED_CHECK_NAME = "validate / validate"
REQUIRED_CHECK_APP_ID = 15368  # GitHub Actions
REQUIRED_CHECK_WORKFLOW_PATH = ".github/workflows/repository-checks.yml"
BLOCKING_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
MAX_MANIFEST_BLOBS = 256
MAX_BLOB_BYTES = 10 * 1024 * 1024

PASS = "pass"
FAIL = "fail"
NOT_APPLICABLE = "not_applicable"
EXIT_CODES = {PASS: 0, FAIL: 1, NOT_APPLICABLE: 3}

# Reason codes. One per refusal; tests assert them.
R_PASS = "content_bound_approval"
R_NO_GUARDED_CONTENT = "no_guarded_content"
R_EARLIER_LANE = "earlier_lane_v5"
R_GUARDED_LEAF = "guarded_leaf_not_regular"
R_MANIFEST_REMOVED = "manifest_removed"
R_CONTENT_WITHOUT_MANIFEST = "content_without_manifest"
R_CONTENT_NOT_COVERED = "content_not_covered"
R_UNKNOWN_SCHEMA = "unknown_manifest_schema"
R_MIXED_SCHEMAS = "mixed_manifest_schemas"
R_UNSUPPORTED_REPOSITORY = "unsupported_repository"
R_INVALID_REQUEST = "invalid_request"
R_READ_FAILED = "github_read_failed"
R_EVALUATION_ERROR = "evaluation_error"
R_PR_IDENTITY = "pr_identity"
R_PR_DRAFT = "pr_is_draft"
R_BASE_NOT_CURRENT = "base_not_current_main"
R_DELTA_INCOMPLETE = "delta_incomplete"
R_DELTA_DISAGREES = "delta_disagrees"
R_TOO_MANY_MANIFESTS = "too_many_manifests"
R_BLOB_UNREADABLE = "blob_unreadable"
R_MANIFEST_UNPARSEABLE = "manifest_unparseable"
R_MULTIPLE_V6 = "multiple_v6_manifests"
R_MANIFEST_SHAPE = "manifest_shape"
R_MANIFEST_CANDIDATE = "manifest_candidate_mismatch"
R_CITATION = "citation_not_canonical"
R_MANIFEST_PATH = "manifest_path_not_canonical"
R_SCOPE_PATHS = "scope_paths"
R_SCOPE_NOT_NEW = "scope_not_new_module"
R_APPLIED_FILES = "applied_files_mismatch"
R_RUN = "signing_run_not_bound"
R_DISPATCHER_MISMATCH = "dispatcher_mismatch"
R_DISPATCHER_NOT_ENROLLED = "dispatcher_not_enrolled"
R_REVIEWS = "reviews_unreadable"
R_CHANGES_REQUESTED = "changes_requested"
R_COMMITS = "commits_unreadable"
R_NO_APPROVAL = "no_approval"
R_NOT_ENROLLED = "approver_not_enrolled"
R_IS_AUTHOR = "approver_is_author"
R_IS_DISPATCHER = "approver_is_dispatcher"
R_AUTHORED_COMMITS = "approver_authored_commits"
R_NOT_CONTENT_BOUND = "approval_not_content_bound"
R_NO_COUNTING_APPROVAL = "no_independent_content_bound_approval"
R_CI = "ci_not_green"
R_CHANGED = "changed_during_evaluation"
R_MERGE_GROUP = "merge_group_not_implemented"

_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LOGIN = re.compile(r"[A-Za-z0-9-]{1,39}")
_RUN_ID = re.compile(r"[1-9][0-9]{0,19}")
# rulespec-<country> and rulespec-<country>-<subdivision> repositories.
_REPOSITORY = re.compile(r"rulespec-([a-z]{2})(-[a-z0-9]+)*")
_REGULAR = ("blob", "100644")


@dataclass(frozen=True)
class Verdict:
    """One evaluation result. ``evidence`` holds what was established."""

    verdict: str
    reason: str
    detail: str = ""
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "schema": VERDICT_SCHEMA,
            "verdict": self.verdict,
            "reason": self.reason,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


class _Refusal(AdmissionError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


@contextmanager
def _refuse_as(reason: str):
    """Give a reason code to a refusal raised by a shared admission helper."""
    try:
        yield
    except _Refusal:
        raise
    except AdmissionError as exc:
        raise _Refusal(reason, str(exc)) from exc


@dataclass(frozen=True)
class _Target:
    """The fields ``_pr_identity`` reads from a promotion selector."""

    pr: int
    head: str
    branch: str
    repository: str


@dataclass(frozen=True)
class _Manifest:
    citation: str
    run_id: str
    applied: dict
    dispatcher_id: int
    dispatcher_login: str
    candidate_head: str
    candidate_base: str


def _reader(fetch: Callable[[str], object]) -> Callable[[str], object]:
    def read(path: str) -> object:
        try:
            return fetch(path)
        except _Refusal:
            raise
        except Exception as exc:  # HTTP error, redirect, rate limit, bad JSON
            raise _Refusal(R_READ_FAILED, type(exc).__name__) from exc

    return read


def _strict_int(value: object) -> bool:
    return type(value) is int and value > 0


def _delta(base: dict[str, dict], head: dict[str, dict]) -> dict[str, str]:
    """The leaf delta of two complete trees, with no mode restriction.

    ``_require_immutable_delta`` refuses any changed non-regular blob, which is
    right for a promotion and wrong for deciding applicability: a maintenance
    pull request may add an executable script.
    """

    def identity(entry):
        return None if entry is None else (entry["type"], entry["mode"], entry["sha"])

    delta = {}
    for path in base.keys() | head.keys():
        old, new = base.get(path), head.get(path)
        if identity(old) != identity(new):
            delta[path] = (
                "added" if old is None else "removed" if new is None else "modified"
            )
    return delta


def _blob(read: Callable[[str], object], root: str, sha: str) -> bytes:
    """Read one blob and prove its bytes hash to the requested Git object id."""
    with _refuse_as(R_BLOB_UNREADABLE):
        blob = _object(read(f"{root}/git/blobs/{sha}"), "blob")
        if blob.get("sha") != sha or blob.get("encoding") != "base64":
            raise AdmissionError("blob identity or encoding differs")
        try:
            raw = base64.b64decode("".join(blob["content"].splitlines()), validate=True)
        except (KeyError, TypeError, AttributeError, ValueError, binascii.Error) as exc:
            raise AdmissionError("blob encoding is invalid") from exc
        if len(raw) != blob.get("size") or len(raw) > MAX_BLOB_BYTES:
            raise AdmissionError("blob size differs or exceeds its bound")
        if hashlib.sha1(b"blob %d\x00" % len(raw) + raw).hexdigest() != sha:
            raise AdmissionError("blob bytes do not hash to the Git object id")
    return raw


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("non-finite JSON number")


def _parse_json_object(raw: bytes) -> dict:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError) as exc:
        raise _Refusal(R_MANIFEST_UNPARSEABLE, type(exc).__name__) from exc
    if not isinstance(payload, dict):
        raise _Refusal(R_MANIFEST_UNPARSEABLE, "manifest is not a JSON object")
    return payload


def _parse_manifest(
    payload: dict, *, repository: str, number: int, author_id: int
) -> _Manifest:
    """Accept exactly the v6 promotion shape of the design, section 5."""

    def shape(condition: bool, detail: str):
        if not condition:
            raise _Refusal(R_MANIFEST_SHAPE, detail)

    shape(
        any(set(payload) == fields for fields in V6_MANIFEST_FIELD_VARIANTS),
        "field set is not the v6 promotion contract",
    )
    shape(payload["schema_version"] == MANIFEST_SCHEMA_V6, "schema version")
    shape(payload["tool"] == PROMOTION_TOOL, "tool is not the promotion command")
    citation, run_id = payload["citation"], payload["run_id"]
    shape(isinstance(citation, str) and bool(citation), "citation")
    shape(isinstance(run_id, str) and bool(_RUN_ID.fullmatch(run_id)), "run_id")
    ref = payload["candidate_rulespec_ref"]
    shape(isinstance(ref, str) and bool(_SHA.fullmatch(ref)), "candidate ref")
    for name in ("source_attestation", "validation_execution", "signature"):
        shape(isinstance(payload[name], dict) and bool(payload[name]), name)

    applied = payload["applied_files"]
    shape(isinstance(applied, list) and len(applied) == 2, "applied_files")
    digests = {}
    for item in applied:
        shape(
            isinstance(item, dict)
            and set(item) == {"path", "sha256"}
            and isinstance(item["path"], str)
            and isinstance(item["sha256"], str)
            and bool(_SHA256.fullmatch(item["sha256"]))
            and item["path"] not in digests,
            "applied_files entry",
        )
        digests[item["path"]] = item["sha256"]

    scope = payload["attestation_scope"]
    shape(
        isinstance(scope, dict) and set(scope) == ATTESTATION_SCOPE_FIELDS,
        "attestation_scope fields",
    )
    shape(scope["schema"] == ATTESTATION_SCOPE_SCHEMA, "attestation_scope schema")
    shape(scope["validation"] == ATTESTATION_VALIDATION, "attestation validation")
    shape(scope["owner_review"] == ATTESTATION_OWNER_REVIEW, "attestation owner_review")
    shape(scope["origin"] == ATTESTATION_ORIGIN, "attestation origin")
    shape(scope["merge_requirement"] == MERGE_REQUIREMENT, "merge_requirement")
    candidate, dispatcher = scope["candidate"], scope["dispatcher"]
    shape(
        isinstance(candidate, dict)
        and set(candidate) == ATTESTATION_CANDIDATE_FIELDS
        and _strict_int(candidate["pull_request"])
        and _strict_int(candidate["author_id"])
        and all(
            isinstance(candidate[name], str) and bool(_SHA.fullmatch(candidate[name]))
            for name in ("head_sha", "base_sha")
        ),
        "attestation candidate",
    )
    shape(
        isinstance(dispatcher, dict)
        and set(dispatcher) == ATTESTATION_DISPATCHER_FIELDS
        and _strict_int(dispatcher["id"])
        and isinstance(dispatcher["login"], str)
        and bool(_LOGIN.fullmatch(dispatcher["login"])),
        "attestation dispatcher",
    )
    if (
        candidate["repository"] != repository
        or candidate["pull_request"] != number
        or candidate["author_id"] != author_id
        or candidate["head_sha"] != ref
    ):
        raise _Refusal(
            R_MANIFEST_CANDIDATE,
            "manifest names another repository, pull request, author or candidate",
        )
    return _Manifest(
        citation=citation,
        run_id=run_id,
        applied=digests,
        dispatcher_id=dispatcher["id"],
        dispatcher_login=dispatcher["login"],
        candidate_head=candidate["head_sha"],
        candidate_base=candidate["base_sha"],
    )


def _effective_reviews(reviews: list[dict]) -> dict[int, dict]:
    """The latest decisive review of each reviewer, as admission computes it.

    A review whose account was deleted (``user`` is null) is ignored: it can
    never count as an approval and is not an enrolled or associated blocker.
    """
    decisive: dict[int, dict] = {}
    for review in reviews:
        if not _strict_int(review.get("id")):
            raise AdmissionError("review lacks a numeric identity")
    for review in sorted(reviews, key=lambda item: item["id"]):
        state = review.get("state")
        if state not in {
            "CHANGES_REQUESTED",
            "APPROVED",
            "DISMISSED",
            "COMMENTED",
            "PENDING",
        }:
            raise AdmissionError("unknown review state")
        if review.get("user") is None:
            continue
        identity = _object(review["user"], "reviewer").get("id")
        if not _strict_int(identity):
            raise AdmissionError("reviewer lacks a numeric identity")
        if state in {"CHANGES_REQUESTED", "APPROVED", "DISMISSED"}:
            decisive[identity] = review
    return decisive


def _actor_ids(commit: dict) -> set[int]:
    """GitHub-attributed author and committer. Unattributed commits give none."""
    ids = set()
    for role in ("author", "committer"):
        account = commit.get(role)
        if account is None:
            continue
        identity = _object(account, "commit account").get("id")
        if not _strict_int(identity):
            raise AdmissionError("commit account lacks a numeric identity")
        ids.add(identity)
    return ids


def _commit_shas(commits: list[dict]) -> list[str]:
    shas = [commit.get("sha") for commit in commits]
    if len(set(shas)) != len(shas) or not all(
        isinstance(sha, str) and _SHA.fullmatch(sha) for sha in shas
    ):
        raise AdmissionError("invalid or duplicate commit list")
    return shas


def _main_tip(read: Callable[[str], object], root: str) -> str:
    main = _object(read(root + "/git/ref/heads/main"), "main ref")
    sha = _object(main.get("object"), "main object").get("sha")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise AdmissionError("main ref is not an exact commit")
    return sha


def _is_ancestor(
    read: Callable[[str], object], root: str, ancestor: str, descendant: str
) -> bool:
    """Whether ``ancestor`` is ``descendant`` or one of its ancestors."""
    if ancestor == descendant:
        return True
    comparison = _object(read(f"{root}/compare/{ancestor}...{descendant}"), "ancestry")
    return (
        comparison.get("status") == "ahead"
        and _object(comparison.get("merge_base_commit"), "merge base").get("sha")
        == ancestor
    )


def _signing_run(read: Callable[[str], object], run_id: str) -> tuple[int, int, str]:
    """The actor ids of a successful signing dispatch from encoder ``main``."""
    run = _object(read(f"{ENCODER_ROOT}/actions/runs/{run_id}"), "signing run")
    actor = _object(run.get("actor"), "run actor")
    trigger = _object(run.get("triggering_actor"), "triggering actor")
    commit = run.get("head_sha")
    if (
        str(run.get("id")) != run_id
        or _object(run.get("repository"), "run repository").get("full_name")
        != ENCODER_REPOSITORY
        or run.get("event") != "workflow_dispatch"
        or run.get("path") != SIGNING_WORKFLOW_PATH
        or run.get("head_branch") != "main"
        or not isinstance(commit, str)
        or not _SHA.fullmatch(commit)
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or not _strict_int(actor.get("id"))
        or not _strict_int(trigger.get("id"))
    ):
        raise AdmissionError(
            "manifest run is not a successful central main signing dispatch"
        )
    # ``head_branch`` is a name; a tag or another ref may carry it. The run's
    # commit must be on the encoder's protected ``main`` history.
    if not _is_ancestor(read, ENCODER_ROOT, commit, _main_tip(read, ENCODER_ROOT)):
        raise AdmissionError("signing run commit is not on encoder main")
    return actor["id"], trigger["id"], trigger.get("login")


def _require_green_ci(
    read: Callable[[str], object], root: str, head: str, number: int
) -> None:
    name = urllib.parse.quote(REQUIRED_CHECK_NAME, safe="")
    listing = _object(
        read(
            f"{root}/commits/{head}/check-runs"
            f"?check_name={name}&filter=latest&per_page=100"
        ),
        "check runs",
    )
    runs = listing.get("check_runs")
    if (
        not isinstance(runs, list)
        or not runs
        or listing.get("total_count") != len(runs)
    ):
        raise AdmissionError("required check is absent or its listing is incomplete")
    repository = root.removeprefix("/repos/")
    pattern = re.compile(
        re.escape(f"https://github.com/{repository}/actions/runs/")
        + r"([1-9][0-9]{0,19})/job/[1-9][0-9]{0,19}"
    )
    for run in runs:
        run = _object(run, "check run")
        match = pattern.fullmatch(str(run.get("details_url", "")))
        if (
            run.get("name") != REQUIRED_CHECK_NAME
            or _object(run.get("app"), "check app").get("id") != REQUIRED_CHECK_APP_ID
            or run.get("head_sha") != head
            or run.get("status") != "completed"
            or run.get("conclusion") != "success"
            or match is None
        ):
            raise AdmissionError("required check is not green on the current head")
        workflow = _object(
            read(f"{root}/actions/runs/{match.group(1)}"), "check workflow run"
        )
        pulls = workflow.get("pull_requests")
        if (
            str(workflow.get("id")) != match.group(1)
            or workflow.get("event") != "pull_request"
            or workflow.get("path") != REQUIRED_CHECK_WORKFLOW_PATH
            or workflow.get("head_sha") != head
            or _object(workflow.get("repository"), "run repository").get("full_name")
            != repository
            or _object(workflow.get("head_repository"), "run head repository").get(
                "full_name"
            )
            != repository
            or not isinstance(pulls, list)
            or not any(
                isinstance(pull, dict)
                and pull.get("number") == number
                and _object(pull.get("head"), "run PR head").get("sha") == head
                for pull in pulls
            )
        ):
            raise AdmissionError(
                "required check did not come from this PR's pull_request run"
            )


def _is_guarded(path: str) -> bool:
    """RuleSpec content or apply-manifest territory, by path alone.

    The content roots are the encoder's ``RULESPEC_ATOMIC_ROOTS``, at the
    repository root or one level below a jurisdiction directory. A leaf at
    ``.axiom`` or at the manifest directory itself replaces a directory.
    """
    parts = path.split("/")
    return (
        path in {".axiom", MANIFEST_DIR}
        or path.startswith(MANIFEST_PREFIX)
        or parts[0] in RULESPEC_ATOMIC_ROOTS
        or (len(parts) >= 2 and parts[1] in RULESPEC_ATOMIC_ROOTS)
    )


def _applied_paths(payload: dict) -> set[str]:
    applied = payload.get("applied_files")
    if not isinstance(applied, list):
        return set()
    return {
        item["path"]
        for item in applied
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }


def evaluate(
    repository: str, pr: int, *, fetch: Callable[[str], object] = github_json
) -> Verdict:
    """Evaluate one pull request of ``TheAxiomFoundation/<repository>``.

    ``repository`` is a short name: ``rulespec-us``, ``rulespec-us-tn``.
    Never raises.
    """
    evidence: dict = {"pull_request": pr}
    try:
        return _evaluate(repository, pr, _reader(fetch), evidence)
    except _Refusal as refusal:
        return Verdict(FAIL, refusal.reason, refusal.detail, evidence)
    except Exception as exc:  # fail closed on anything unforeseen
        return Verdict(FAIL, R_EVALUATION_ERROR, type(exc).__name__, evidence)


def evaluate_merge_group(*_args, **_kwargs) -> Verdict:
    """STUB. Merge-group evaluation (design section 6) is not implemented."""
    return Verdict(FAIL, R_MERGE_GROUP, "merge-group evaluation is not implemented")


def _evaluate(name, number, read, evidence) -> Verdict:
    match = _REPOSITORY.fullmatch(name) if isinstance(name, str) else None
    if match is None or not _strict_int(number):
        raise _Refusal(R_INVALID_REQUEST, "invalid repository or pull request number")
    repository = f"{ORGANISATION}/{name}"
    root = "/repos/" + repository
    pr_path = f"{root}/pulls/{number}"
    evidence["repository"] = repository

    # 1. PR identity: open, unmerged, same-repository head, base ``main``,
    # strictly ahead of the live ``main`` tip.
    with _refuse_as(R_PR_IDENTITY):
        before = _object(read(pr_path), "PR")
        pr_head = _object(before.get("head"), "PR head")
        head, branch = pr_head.get("sha"), pr_head.get("ref")
        if (
            not isinstance(head, str)
            or not _SHA.fullmatch(head)
            or not isinstance(branch, str)
            or not branch
        ):
            raise AdmissionError("PR head is not an exact commit on a branch")
        target = _Target(number, head, branch, repository)
        identity = _pr_identity(before, target)
        author_id = identity[1]
        if not _strict_int(author_id):
            raise AdmissionError("PR author lacks a numeric identity")
    evidence["head"] = head
    with _refuse_as(R_BASE_NOT_CURRENT):
        base = _main_tip(read, root)
        evidence["base"] = base
        comparison = _object(read(f"{root}/compare/{base}...{head}"), "ancestry")
        if (
            comparison.get("status") != "ahead"
            or _object(comparison.get("merge_base_commit"), "merge base").get("sha")
            != base
        ):
            raise AdmissionError("head must descend from the current main tip")

    # 2. Complete delta, from two complete trees.
    with _refuse_as(R_DELTA_INCOMPLETE):
        base_tree = _object(read(f"{root}/git/trees/{base}?recursive=1"), "base tree")
        head_tree = _object(read(f"{root}/git/trees/{head}?recursive=1"), "head tree")
        base_leaves = _leaf_entries(base_tree)
        head_leaves = _leaf_entries(head_tree)
    delta = _delta(base_leaves, head_leaves)

    # 3. Applicability, from paths and modes only. A label inside a file the
    # pull request supplies never makes the guard stand aside.
    def regular(path):
        entry = head_leaves.get(path)
        return entry is None or (entry["type"], entry["mode"]) == _REGULAR

    guarded = sorted(
        path
        for path in delta
        # A non-regular top-level leaf can alias a jurisdiction directory.
        if _is_guarded(path) or ("/" not in path and not regular(path))
    )
    if not guarded:
        return Verdict(
            NOT_APPLICABLE,
            R_NO_GUARDED_CONTENT,
            "the delta changes no RuleSpec content and no apply manifest",
            evidence,
        )
    for path in guarded:
        if not regular(path) or path in {".axiom", MANIFEST_DIR}:
            raise _Refusal(R_GUARDED_LEAF, path)
    manifests = [path for path in guarded if path.startswith(MANIFEST_PREFIX)]
    content = [path for path in guarded if not path.startswith(MANIFEST_PREFIX)]
    if not manifests:
        raise _Refusal(
            R_CONTENT_WITHOUT_MANIFEST, "RuleSpec content changes with no manifest"
        )
    if any(delta[path] == "removed" for path in manifests):
        raise _Refusal(R_MANIFEST_REMOVED, "an apply manifest is removed")
    if len(manifests) > MAX_MANIFEST_BLOBS:
        raise _Refusal(R_TOO_MANY_MANIFESTS, "changed manifests exceed the read bound")
    parsed = {}
    for path in manifests:
        raw = _blob(read, root, head_leaves[path]["sha"])
        parsed[path] = (raw, _parse_json_object(raw))
    labels = [payload.get("schema_version") for _raw, payload in parsed.values()]
    if not all(
        isinstance(label, str)
        and label in EARLIER_MANIFEST_SCHEMAS | {MANIFEST_SCHEMA_V6}
        for label in labels
    ):
        raise _Refusal(R_UNKNOWN_SCHEMA, "a changed manifest has an unknown schema")
    schemas = set(labels)
    if MANIFEST_SCHEMA_V6 not in schemas:
        covered = set().union(
            *(_applied_paths(payload) for _raw, payload in parsed.values())
        )
        if not set(content) <= covered:
            raise _Refusal(
                R_CONTENT_NOT_COVERED,
                "RuleSpec content changes outside the changed manifests",
            )
        return Verdict(
            NOT_APPLICABLE,
            R_EARLIER_LANE,
            "an earlier-schema lane; its approval is checked before signing",
            evidence,
        )
    if len(schemas) != 1:
        raise _Refusal(R_MIXED_SCHEMAS, "v6 and earlier manifests in one PR")
    if len(parsed) != 1:
        raise _Refusal(R_MULTIPLE_V6, "exactly one v6 manifest is allowed per PR")
    if match.group(2) is not None:
        raise _Refusal(
            R_UNSUPPORTED_REPOSITORY,
            "the promotion lane publishes only to rulespec-<country>",
        )
    country = match.group(1)
    if before.get("draft") is not False:
        raise _Refusal(R_PR_DRAFT, "a draft pull request has no merge verdict")

    # 4. Scope and canonical paths: the delta is exactly one new unit.
    ((manifest_path, (manifest_raw, payload)),) = parsed.items()
    with _refuse_as(R_DELTA_DISAGREES):
        files = _pages(read, pr_path + "/files")
        if before.get("changed_files") != len(files):
            raise AdmissionError("PR file count differs from its file list")
        _require_immutable_delta(base_tree, head_tree, files, comparison)
    manifest = _parse_manifest(
        payload, repository=repository, number=number, author_id=author_id
    )
    evidence["dispatcher_id"] = manifest.dispatcher_id
    evidence["run_id"] = manifest.run_id
    evidence["candidate_head"] = manifest.candidate_head
    try:
        primary = str(citation_rulespec_path(manifest.citation))
        selector = PromotionSelector(
            country,
            number,
            head,
            manifest.citation,
            primary,
            branch,
            manifest.dispatcher_login,
            manifest.run_id,
        )
    except Exception as exc:
        raise _Refusal(R_CITATION, type(exc).__name__) from exc
    companion = selector.companion
    if manifest_path != MANIFEST_PREFIX + primary.removesuffix(".yaml") + ".json":
        raise _Refusal(R_MANIFEST_PATH, manifest_path)
    unit = (primary, companion, manifest_path)
    evidence["paths"] = {
        "primary": primary,
        "companion": companion,
        "manifest": manifest_path,
    }
    if set(delta) != set(unit):
        raise _Refusal(
            R_SCOPE_PATHS,
            "delta is not exactly the primary, companion and manifest",
        )
    if any(delta[path] != "added" for path in unit):
        raise _Refusal(R_SCOPE_NOT_NEW, "promotion paths already exist on main")
    digests = {
        primary: hashlib.sha256(
            _blob(read, root, head_leaves[primary]["sha"])
        ).hexdigest(),
        companion: hashlib.sha256(
            _blob(read, root, head_leaves[companion]["sha"])
        ).hexdigest(),
    }
    evidence["primary_sha256"] = digests[primary]
    evidence["companion_sha256"] = digests[companion]
    evidence["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    if manifest.applied != digests:
        raise _Refusal(
            R_APPLIED_FILES, "manifest applied_files differ from the head blobs"
        )

    # The PR's commits: the candidate the manifest names, and who wrote them.
    with _refuse_as(R_COMMITS):
        commits = _pages(read, pr_path + "/commits")
        commit_shas = _commit_shas(commits)
        if before.get("commits") != len(commits) or head not in commit_shas:
            raise AdmissionError("PR commit list is incomplete")
        commit_actors: set[int] = set()
        for commit in commits:
            commit_actors |= _actor_ids(commit)

    def entries_at(commit, label):
        return _leaf_entries(
            _object(read(f"{root}/git/trees/{commit}?recursive=1"), label)
        )

    def same_blobs(leaves, paths):
        return all(
            path in leaves
            and (leaves[path]["type"], leaves[path]["mode"], leaves[path]["sha"])
            == (
                head_leaves[path]["type"],
                head_leaves[path]["mode"],
                head_leaves[path]["sha"],
            )
            for path in paths
        )

    # The candidate the signer validated is a commit of this PR that carries
    # the head's primary and companion, cut from an ancestor of this base.
    with _refuse_as(R_MANIFEST_CANDIDATE):
        if (
            manifest.candidate_head not in commit_shas
            or not same_blobs(
                entries_at(manifest.candidate_head, "candidate tree"),
                (primary, companion),
            )
            or not _is_ancestor(read, root, manifest.candidate_base, base)
        ):
            raise AdmissionError(
                "manifest candidate is not a commit of this PR with the head content"
            )

    # 6 (first half). The signing run and the dispatcher it authenticates.
    with _refuse_as(R_RUN):
        run_actor_id, trigger_id, trigger_login = _signing_run(read, manifest.run_id)
    if (
        trigger_id != manifest.dispatcher_id
        or trigger_login != manifest.dispatcher_login
    ):
        raise _Refusal(
            R_DISPATCHER_MISMATCH, "manifest dispatcher is not the run's actor"
        )
    if (
        manifest.dispatcher_id not in TRUSTED_SIGNING_REVIEWERS
        or TRUSTED_SIGNING_REVIEWERS[manifest.dispatcher_id] != trigger_login
    ):
        raise _Refusal(R_DISPATCHER_NOT_ENROLLED, "dispatcher is not enrolled")
    dispatchers = {manifest.dispatcher_id, run_actor_id, trigger_id}

    # 5 and 7 (reviews). Effective reviews; a blocking review refuses.
    with _refuse_as(R_REVIEWS):
        reviews = _pages(read, pr_path + "/reviews")
        effective = _effective_reviews(reviews)
    for reviewer_id, review in effective.items():
        if review["state"] == "CHANGES_REQUESTED" and (
            reviewer_id in TRUSTED_SIGNING_REVIEWERS
            or review.get("author_association") in BLOCKING_ASSOCIATIONS
            or not isinstance(review.get("author_association"), str)
        ):
            raise _Refusal(R_CHANGES_REQUESTED, f"review {review['id']}")

    # 5 and 6. A counting approval is enrolled, independent and content-bound.
    def content_bound(commit):
        if not isinstance(commit, str) or commit not in commit_shas:
            return False
        if commit == head:
            return True
        with _refuse_as(R_DELTA_INCOMPLETE):
            ancestry = _object(read(f"{root}/compare/{base}...{commit}"), "ancestry")
            fork = _object(ancestry.get("merge_base_commit"), "merge base").get("sha")
            if not isinstance(fork, str) or not _SHA.fullmatch(fork):
                raise AdmissionError("approved commit has no merge base with main")
            approved = entries_at(commit, "approved tree")
            scope = set(_delta(entries_at(fork, "fork tree"), approved))
        return scope == set(unit) and same_blobs(approved, unit)

    approvals, rejected = [], []
    for reviewer_id in sorted(effective):
        review = effective[reviewer_id]
        if review["state"] != "APPROVED":
            continue
        reviewer = review["user"]
        if (
            reviewer_id not in TRUSTED_SIGNING_REVIEWERS
            or reviewer.get("login") != TRUSTED_SIGNING_REVIEWERS[reviewer_id]
            or reviewer.get("type") != "User"
        ):
            reason = R_NOT_ENROLLED
        elif reviewer_id == author_id:
            reason = R_IS_AUTHOR
        elif reviewer_id in dispatchers:
            reason = R_IS_DISPATCHER
        elif reviewer_id in commit_actors:
            reason = R_AUTHORED_COMMITS
        elif not content_bound(review.get("commit_id")):
            reason = R_NOT_CONTENT_BOUND
        else:
            approvals.append(
                {
                    "review_id": review["id"],
                    "reviewer_id": reviewer_id,
                    "commit_id": review["commit_id"],
                }
            )
            continue
        rejected.append(
            {"review_id": review["id"], "reviewer_id": reviewer_id, "reason": reason}
        )
    if not approvals:
        evidence["rejected_approvals"] = rejected
        reasons = {item["reason"] for item in rejected}
        raise _Refusal(
            R_NO_APPROVAL
            if not rejected
            else reasons.pop()
            if len(reasons) == 1
            else R_NO_COUNTING_APPROVAL,
            "no enrolled, independent approval is bound to the head content",
        )
    evidence["approvals"] = approvals
    evidence["review_id"] = approvals[0]["review_id"]
    evidence["reviewer_id"] = approvals[0]["reviewer_id"]

    # 7. Current-head CI from this PR's pull_request run.
    with _refuse_as(R_CI):
        _require_green_ci(read, root, head, number)

    # 8. Re-read the mutable state; any movement refuses this evaluation.
    with _refuse_as(R_CHANGED):
        after = _object(read(pr_path), "PR recheck")
        if (
            _pr_identity(after, target) != identity
            or after.get("draft") is not False
            or _main_tip(read, root) != base
            or _pages(read, pr_path + "/reviews") != reviews
        ):
            raise AdmissionError("PR, main or reviews changed during evaluation")
    return Verdict(PASS, R_PASS, "", evidence)


def command(args) -> int:
    """Print one verdict as one JSON line.

    Exit 0 for pass, 1 for fail, 3 for not_applicable. (2 is a usage error.)
    """
    verdict = evaluate(args.repository, args.pr, fetch=github_json)
    print(json.dumps(verdict.as_dict(), sort_keys=True, separators=(",", ":")))
    return EXIT_CODES[verdict.verdict]
