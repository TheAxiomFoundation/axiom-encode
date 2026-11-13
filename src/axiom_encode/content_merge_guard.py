"""Content-bound merge guard: evaluate one RuleSpec pull request.

INACTIVE. Nothing in the encoder, a workflow or a RuleSpec repository calls
this module, no check run is written, and the manifest schema it reads is not
emitted by any writer. See ``docs/content-bound-merge-guard.md``.

The evaluator decides whether one open pull request that carries a signed
``applied-rulespec/v6`` promotion manifest also carries a live approval, from
an enrolled owner who is independent of the candidate author and of the
dispatcher, on a commit whose primary, companion and manifest blobs are the
blobs at the current head. Every read is live and goes through one injected
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

from axiom_encode.prepare_signed_backfill import citation_rulespec_path
from axiom_encode.reviewed_pr_admission import (
    INDEX_PATH,
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
MANIFEST_PREFIX = ".axiom/encoding-manifests/"

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
ENCODER_ROOT = "/repos/TheAxiomFoundation/axiom-encode"
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

# Reason codes. One per refusal; tests assert them.
R_PASS = "content_bound_approval"
R_NO_V6_MANIFEST = "no_v6_manifest"
R_INVALID_REQUEST = "invalid_request"
R_READ_FAILED = "github_read_failed"
R_EVALUATION_ERROR = "evaluation_error"
R_PR_IDENTITY = "pr_identity"
R_PR_DRAFT = "pr_is_draft"
R_BASE_NOT_CURRENT = "base_not_current_main"
R_DELTA_INCOMPLETE = "delta_incomplete"
R_DELTA_DISAGREES = "delta_disagrees"
R_TOO_MANY_MANIFESTS = "too_many_manifests"
R_MANIFEST_NOT_REGULAR = "manifest_not_regular_blob"
R_BLOB_UNREADABLE = "blob_unreadable"
R_MANIFEST_UNPARSEABLE = "manifest_unparseable"
R_MULTIPLE_V6 = "multiple_v6_manifests"
R_MANIFEST_SHAPE = "manifest_shape"
R_MANIFEST_CANDIDATE = "manifest_candidate_mismatch"
R_CITATION = "citation_not_canonical"
R_MANIFEST_PATH = "manifest_path_not_canonical"
R_SCOPE_PATHS = "scope_paths"
R_SCOPE_NOT_NEW = "scope_not_new_module"
R_SCOPE_NOT_REGULAR = "scope_not_regular_blob"
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
    author_id: int


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
        and isinstance(candidate["repository"], str)
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
    ):
        raise _Refusal(
            R_MANIFEST_CANDIDATE,
            "manifest names another repository, pull request or author",
        )
    return _Manifest(
        citation=citation,
        run_id=run_id,
        applied=digests,
        dispatcher_id=dispatcher["id"],
        dispatcher_login=dispatcher["login"],
        author_id=candidate["author_id"],
    )


def _effective_reviews(reviews: list[dict]) -> dict[int, dict]:
    """The latest decisive review of each reviewer, as admission computes it."""
    decisive: dict[int, dict] = {}
    for review in reviews:
        if not _strict_int(review.get("id")):
            raise AdmissionError("review lacks a numeric identity")
    for review in sorted(reviews, key=lambda item: item["id"]):
        identity = _object(review.get("user"), "reviewer").get("id")
        if not _strict_int(identity):
            raise AdmissionError("reviewer lacks a numeric identity")
        state = review.get("state")
        if state in {"CHANGES_REQUESTED", "APPROVED", "DISMISSED"}:
            decisive[identity] = review
        elif state not in {"COMMENTED", "PENDING"}:
            raise AdmissionError("unknown review state")
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


def _path_history(
    read: Callable[[str], object], root: str, head: str, path: str
) -> list[dict]:
    """Commits reachable from the head that change one path."""
    query = (
        f"{root}/commits?sha={head}&path={urllib.parse.quote(path, safe='')}"
        "&per_page=100&page="
    )
    # ``_pages`` owns the page bound; it appends ``?per_page=100&page=N``.
    return _pages(lambda url: read(query + url.rpartition("=")[2]), "")


def _main_tip(read: Callable[[str], object], root: str) -> str:
    main = _object(read(root + "/git/ref/heads/main"), "main ref")
    sha = _object(main.get("object"), "main object").get("sha")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise AdmissionError("main ref is not an exact commit")
    return sha


def _signing_run(read: Callable[[str], object], run_id: str) -> tuple[int, int, str]:
    run = _object(read(f"{ENCODER_ROOT}/actions/runs/{run_id}"), "signing run")
    actor = _object(run.get("actor"), "run actor")
    trigger = _object(run.get("triggering_actor"), "triggering actor")
    if (
        str(run.get("id")) != run_id
        or run.get("event") != "workflow_dispatch"
        or run.get("path") != SIGNING_WORKFLOW_PATH
        or run.get("head_branch") != "main"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or not _strict_int(actor.get("id"))
        or not _strict_int(trigger.get("id"))
        or not isinstance(trigger.get("login"), str)
    ):
        raise AdmissionError(
            "manifest run is not a successful central main signing dispatch"
        )
    return actor["id"], trigger["id"], trigger["login"]


def _require_green_ci(read: Callable[[str], object], root: str, head: str) -> None:
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
        or len(runs) > 100
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
        if (
            str(workflow.get("id")) != match.group(1)
            or workflow.get("path") != REQUIRED_CHECK_WORKFLOW_PATH
            or workflow.get("head_sha") != head
        ):
            raise AdmissionError("required check did not come from the protected run")


def evaluate(
    country: str, pr: int, *, fetch: Callable[[str], object] = github_json
) -> Verdict:
    """Evaluate one pull request of ``rulespec-<country>``. Never raises."""
    evidence: dict = {"pull_request": pr}
    try:
        return _evaluate(country, pr, _reader(fetch), evidence)
    except _Refusal as refusal:
        return Verdict(FAIL, refusal.reason, refusal.detail, evidence)
    except Exception as exc:  # fail closed on anything unforeseen
        return Verdict(FAIL, R_EVALUATION_ERROR, type(exc).__name__, evidence)


def evaluate_merge_group(*_args, **_kwargs) -> Verdict:
    """STUB. Merge-group evaluation (design section 6) is not implemented."""
    return Verdict(FAIL, R_MERGE_GROUP, "merge-group evaluation is not implemented")


def _evaluate(country, number, read, evidence) -> Verdict:
    if (
        not isinstance(country, str)
        or not re.fullmatch(r"[a-z]{2}", country)
        or not _strict_int(number)
    ):
        raise _Refusal(R_INVALID_REQUEST, "invalid country or pull request number")
    repository = f"TheAxiomFoundation/rulespec-{country}"
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

    # 3. Applicability: every changed manifest blob is read and parsed.
    manifest_paths = sorted(
        path
        for path, status in delta.items()
        if path.startswith(MANIFEST_PREFIX) and status != "removed"
    )
    if len(manifest_paths) > MAX_MANIFEST_BLOBS:
        raise _Refusal(R_TOO_MANY_MANIFESTS, "changed manifests exceed the read bound")
    promotions = {}
    for path in manifest_paths:
        entry = head_leaves[path]
        if (entry["type"], entry["mode"]) != ("blob", "100644"):
            raise _Refusal(R_MANIFEST_NOT_REGULAR, path)
        raw = _blob(read, root, entry["sha"])
        payload = _parse_json_object(raw)
        if payload.get("schema_version") == MANIFEST_SCHEMA_V6:
            promotions[path] = (raw, payload)
    if not promotions:
        return Verdict(
            NOT_APPLICABLE,
            R_NO_V6_MANIFEST,
            "the delta adds or changes no v6 promotion manifest",
            evidence,
        )
    if len(promotions) != 1:
        raise _Refusal(R_MULTIPLE_V6, "exactly one v6 manifest is allowed per PR")
    if before.get("draft") is not False:
        raise _Refusal(R_PR_DRAFT, "a draft pull request has no merge verdict")

    # 4. Scope and canonical paths.
    ((manifest_path, (manifest_raw, payload)),) = promotions.items()
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
    if set(delta) - {INDEX_PATH} != set(unit) or delta.get(INDEX_PATH) == "removed":
        raise _Refusal(
            R_SCOPE_PATHS,
            "delta is not exactly the primary, companion, manifest and optional index",
        )
    if any(delta[path] != "added" or path in base_leaves for path in unit):
        raise _Refusal(R_SCOPE_NOT_NEW, "promotion paths already exist on main")
    for path in (*unit, INDEX_PATH):
        entry = head_leaves.get(path)
        if entry is not None and (entry["type"], entry["mode"]) != ("blob", "100644"):
            raise _Refusal(R_SCOPE_NOT_REGULAR, path)
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
    if TRUSTED_SIGNING_REVIEWERS.get(manifest.dispatcher_id) != trigger_login:
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

    with _refuse_as(R_COMMITS):
        commits = _pages(read, pr_path + "/commits")
        commit_shas = _commit_shas(commits)
        if before.get("commits") != len(commits) or head not in commit_shas:
            raise AdmissionError("PR commit list is incomplete")
        content_authors: set[int] = set()
        for path in (primary, companion):
            history = _path_history(read, root, head, path)
            if not history or not set(_commit_shas(history)) <= set(commit_shas):
                raise AdmissionError("path history disagrees with the PR commits")
            for commit in history:
                content_authors |= _actor_ids(commit)

    # 5 and 6. A counting approval is enrolled, independent and content-bound.
    approvals, rejected = [], []
    for reviewer_id in sorted(effective):
        review = effective[reviewer_id]
        if review["state"] != "APPROVED":
            continue
        reviewer = review["user"]
        if (
            TRUSTED_SIGNING_REVIEWERS.get(reviewer_id) != reviewer.get("login")
            or reviewer.get("type") != "User"
        ):
            reason = R_NOT_ENROLLED
        elif reviewer_id in {author_id, manifest.author_id}:
            reason = R_IS_AUTHOR
        elif reviewer_id in dispatchers:
            reason = R_IS_DISPATCHER
        elif reviewer_id in content_authors:
            reason = R_AUTHORED_COMMITS
        elif not _content_bound(
            read,
            root,
            review.get("commit_id"),
            head,
            base,
            commit_shas,
            unit,
            head_leaves,
        ):
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

    # 7. Current-head CI from the protected workflow definition.
    with _refuse_as(R_CI):
        _require_green_ci(read, root, head)

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


def _content_bound(
    read, root, commit, head, base, commit_shas, unit, head_leaves
) -> bool:
    """Whether the unit blobs at the approved commit are those at the head."""
    if not isinstance(commit, str) or commit not in commit_shas:
        return False
    if commit == head:
        return True
    with _refuse_as(R_DELTA_INCOMPLETE):
        comparison = _object(read(f"{root}/compare/{base}...{commit}"), "ancestry")
        fork = _object(comparison.get("merge_base_commit"), "merge base").get("sha")
        if not isinstance(fork, str) or not _SHA.fullmatch(fork):
            raise AdmissionError("approved commit has no merge base with main")
        fork_leaves = _leaf_entries(
            _object(read(f"{root}/git/trees/{fork}?recursive=1"), "fork tree")
        )
        commit_leaves = _leaf_entries(
            _object(read(f"{root}/git/trees/{commit}?recursive=1"), "approved tree")
        )
    if set(_delta(fork_leaves, commit_leaves)) - {INDEX_PATH} != set(unit):
        return False

    def identity(entry):
        return (entry["type"], entry["mode"], entry["sha"])

    return all(
        identity(commit_leaves[path]) == identity(head_leaves[path]) for path in unit
    )


def add_arguments(parser) -> None:
    parser.add_argument("--country", required=True, help="Two-letter country code")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")


def command(args) -> int:
    """Print one verdict as JSON. Exit 0 for pass or not_applicable, else 1."""
    verdict = evaluate(args.country, args.pr, fetch=github_json)
    print(json.dumps(verdict.as_dict(), indent=2, sort_keys=True))
    return 0 if verdict.verdict in {PASS, NOT_APPLICABLE} else 1
