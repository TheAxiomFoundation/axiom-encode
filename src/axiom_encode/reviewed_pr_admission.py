"""Live GitHub authorization for one protected, reviewed candidate promotion.

Selectors and cached JSON are never authority. The trusted provisioner and the
protected promotion command each fetch GitHub directly. Public repositories
need no credential, so no repository token enters the signing runtime.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Final

# The same human owners who authorize encoder signing-critical changes. Numeric
# GitHub identities prevent a recycled login from gaining signing authority.
# Changes to this policy require code-owner review, not candidate PR review.
TRUSTED_SIGNING_REVIEWERS = {6076111: "MaxGhenis", 110687043: "PavelMakarchuk"}
SCHEMA = "axiom-encode/live-reviewed-pr-admission/v1"
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_PAGES = 10
INDEX_PATH = ".axiom/index/provisions_to_rules.json"

# --- The unreviewed-candidate lane. SWITCHED OFF. ----------------------------
#
# Change 5 of ``docs/content-bound-merge-guard.md`` section 8 (#2172): admit a
# candidate WITHOUT the owner approval before signing, and let the promotion
# writer emit ``applied-rulespec/v6``. The owner approval then happens once,
# on the signed pull request, and the content-bound merge guard requires it.
#
# The switch rule (design, section 8, row 2b):
#   The reader may be switched on while no v6 writer exists: nothing new can
#   land. The v6 writer and the admission without approval must stay off
#   until the guard check is required. With the reader on, the writer on and
#   the guard check not required, an enrolled dispatcher lands their own
#   unreviewed candidate with no approval, and any account with write access
#   re-lands signed v6 bytes from a closed pull request.
#
# Three locks, all on encoder main, none with an environment variable, CLI
# flag, dispatch input or repository setting:
#
# 1. ``V6_WRITER_ENABLED``. While it is not exactly ``True`` this module
#    behaves as before the lane existed: the same reads, results and errors.
# 2. ``GUARD_APP_ID``. ``None`` until the guard GitHub App exists. A reviewed
#    change records its numeric App ID here. While it is ``None`` the lane
#    refuses, also with lock 1 open.
# 3. ``require_guard_enforced``: a live read at every admission and every
#    revalidation. It refuses unless ``main`` of the RuleSpec repository
#    requires the check ``GUARD_CHECK_NAME`` from exactly ``GUARD_APP_ID``,
#    for administrators too. So an open switch does not help while the guard
#    is not a required check (decision D2d).
#
# OPERATOR PRECONDITION, not checkable here: the read of lock 3 proves the
# requirement at admission and before signing. A repository administrator can
# remove a required check afterwards; that adversary is out of scope (design,
# section 9). Do not turn lock 1 before activation checklist step 4 is done.
V6_WRITER_ENABLED: Final[bool] = False
# The enforcing check of the merge guard, and the App that must publish it.
# The shadow check of the runner has another name and never satisfies this.
GUARD_CHECK_NAME: Final = "content-bound-merge-guard"
GUARD_APP_ID: Final[int | None] = None
# GitHub Actions. Any workflow of the RuleSpec repository can publish a check
# as this App, so it can never be the guard's publisher.
GITHUB_ACTIONS_APP_ID: Final = 15368
CANDIDATE_SCHEMA: Final = "axiom-encode/live-candidate-pr-admission/v1"
CANDIDATE_FIELDS: Final = frozenset(
    {
        "schema",
        "selector",
        "record",
        "owner_review",
        "candidate",
        "dispatcher",
        "guard",
        "workflow_commit",
    }
)
CANDIDATE_OWNER_REVIEW: Final = "none"
CANDIDATE_PURPOSE: Final = (
    "Live scope-checked candidate on the RuleSpec PR; NO owner approval before "
    "signing; the content-bound merge guard requires the approval before merge; "
    "new-module promotion only."
)


class AdmissionError(ValueError):
    """The requested candidate has no current independent authorization."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AdmissionError("GitHub admission requests must not redirect")


def github_json(path: str) -> object:
    """Read a bounded response only from the fixed GitHub API origin."""
    if not path.startswith("/repos/TheAxiomFoundation/") or "#" in path:
        raise AdmissionError("invalid admission API path")
    request = urllib.request.Request(
        "https://api.github.com" + path,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "axiom-encode-reviewed-pr-admission",
        },
    )
    # Ignore environment proxies: the API origin, TLS trust and redirects are
    # not dispatch inputs. HTTP errors/rate limits fail closed.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=30) as response:
        if response.status != 200:
            raise AdmissionError("GitHub admission query did not succeed")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise AdmissionError("GitHub admission response exceeds its bound")
    return json.loads(raw)


@dataclass(frozen=True)
class PromotionSelector:
    country: str
    pr: int
    head: str
    citation: str
    rulespec_path: str
    branch: str
    actor: str
    run_id: str = ""

    def __post_init__(self):
        if not re.fullmatch(r"[a-z]{2}", self.country):
            raise AdmissionError("invalid country")
        if type(self.pr) is not int or self.pr <= 0:
            raise AdmissionError("invalid candidate PR number")
        if not re.fullmatch(r"[0-9a-f]{40}", self.head):
            raise AdmissionError("candidate head must be an exact commit")
        for value in (self.citation, self.rulespec_path, self.branch):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 512
                or not re.fullmatch(r"[A-Za-z0-9._/-]+", value)
                or any(part in {"", ".", ".."} for part in value.split("/"))
            ):
                raise AdmissionError("invalid candidate selector path or branch")
        if not re.fullmatch(r"[A-Za-z0-9-]{1,39}", self.actor):
            raise AdmissionError("invalid dispatch actor")
        if not re.fullmatch(r"[1-9][0-9]{0,19}", self.run_id):
            raise AdmissionError("invalid protected workflow run ID")
        if not self.citation.startswith(
            self.country + "/"
        ) and not self.citation.startswith(self.country + "-"):
            raise AdmissionError("citation country differs from candidate country")
        if not self.rulespec_path.endswith(".yaml") or self.rulespec_path.endswith(
            ".test.yaml"
        ):
            raise AdmissionError("candidate must name a primary YAML")

    @property
    def repository(self) -> str:
        return f"TheAxiomFoundation/rulespec-{self.country}"

    @property
    def companion(self) -> str:
        return self.rulespec_path.removesuffix(".yaml") + ".test.yaml"


def _object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise AdmissionError(f"GitHub {label} is not an object")
    return value


def _leaf_entries(tree: dict) -> dict[str, dict]:
    """Retain every immutable leaf, including symlinks and submodules."""
    if tree.get("truncated") is not False or not isinstance(tree.get("tree"), list):
        raise AdmissionError("Git tree cannot be inspected completely")
    leaves = {}
    seen = set()
    for entry in tree["tree"]:
        entry = _object(entry, "tree entry")
        path = entry.get("path")
        if (
            not isinstance(path, str)
            or not path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or path in seen
            or (entry.get("type"), entry.get("mode"))
            not in {
                ("tree", "040000"),
                ("blob", "100644"),
                ("blob", "100755"),
                ("blob", "120000"),
                ("commit", "160000"),
            }
            or not re.fullmatch(r"[0-9a-f]{40}", str(entry.get("sha", "")))
        ):
            raise AdmissionError("invalid or duplicate Git tree entry")
        seen.add(path)
        if entry["type"] != "tree":
            leaves[path] = entry
    return leaves


def _require_immutable_delta(
    base_tree: dict, head_tree: dict, files: list[dict], comparison: dict
) -> dict[str, dict]:
    base_entries = _leaf_entries(base_tree)
    head_entries = _leaf_entries(head_tree)
    delta = {}
    for path in base_entries.keys() | head_entries.keys():
        old, new = base_entries.get(path), head_entries.get(path)

        def identity(entry):
            return (
                None if entry is None else (entry["type"], entry["mode"], entry["sha"])
            )

        if identity(old) != identity(new):
            delta[path] = (
                "added" if old is None else "removed" if new is None else "modified"
            )
            if new is not None and (new["type"] != "blob" or new["mode"] != "100644"):
                raise AdmissionError("changed candidate file is not a regular blob")
    # PR files can race or contradict an immutable comparison. Neither can
    # substitute a permitted index for a workflow changed at the approved SHA.
    for evidence in (files, comparison.get("files")):
        if not isinstance(evidence, list):
            raise AdmissionError("immutable comparison lacks its file delta")
        observed = {}
        for item in evidence:
            item = _object(item, "changed file")
            path = item.get("filename")
            if not isinstance(path, str) or path in observed:
                raise AdmissionError("invalid or duplicate changed file")
            observed[path] = item.get("status")
        if observed != delta:
            raise AdmissionError(
                "file scope differs from complete immutable tree delta"
            )
    return head_entries


def selector_from_payload(value: object) -> PromotionSelector:
    payload = _object(value, "promotion selector")
    if set(payload) != set(PromotionSelector.__dataclass_fields__):
        raise AdmissionError("promotion selector has unexpected fields")
    return PromotionSelector(**payload)


def revalidate_admission(
    expected: object, *, fetch: Callable[[str], object] = github_json
) -> dict:
    payload = _object(expected, "provisioned admission")
    if V6_WRITER_ENABLED is True and payload.get("schema") == CANDIDATE_SCHEMA:
        return _revalidate_candidate_admission(payload, fetch=fetch)
    if (
        set(payload) != {"schema", "selector", "record", "approvals", "workflow_commit"}
        or payload.get("schema") != SCHEMA
    ):
        raise AdmissionError("invalid provisioned admission")
    selector = selector_from_payload(payload["selector"])
    root = "/repos/" + selector.repository
    run = _object(
        fetch(f"/repos/TheAxiomFoundation/axiom-encode/actions/runs/{selector.run_id}"),
        "protected run",
    )
    if (
        run.get("status") != "in_progress"
        or run.get("head_sha") != payload["workflow_commit"]
        or _object(run.get("triggering_actor"), "triggering actor").get("login")
        != selector.actor
    ):
        raise AdmissionError("protected run changed after provisioning")
    pr_path = f"{root}/pulls/{selector.pr}"
    pr = _object(fetch(pr_path), "PR")
    base, author_id = _pr_identity(pr, selector)
    main = _object(fetch(root + "/git/ref/heads/main"), "main ref")
    reviews = _pages(fetch, pr_path + "/reviews")
    approvals = _approved_reviews(reviews, selector, author_id)
    if (
        base != payload["record"]["base_sha"]
        or _object(main.get("object"), "main object").get("sha") != base
        or approvals != payload["approvals"]
    ):
        raise AdmissionError(
            "live approval or candidate differs from provisioned admission"
        )
    # The head and base are immutable Git objects already fully inspected by
    # the root provisioner. The signer separately rehashes the local primary
    # and companion; re-fetch only mutable authorization, not all source bytes.
    return payload


def candidate_lane_enabled() -> bool:
    """Whether the unreviewed-candidate lane may be used at all (lock 1)."""

    return V6_WRITER_ENABLED is True


def is_candidate_admission(payload: object) -> bool:
    """Whether ``payload`` is an admission of the unreviewed-candidate lane.

    ``False`` for everything while the switch is off. It reads the label only:
    call ``revalidate_admission`` first, which checks the whole record live.
    """

    return (
        V6_WRITER_ENABLED is True
        and isinstance(payload, dict)
        and payload.get("schema") == CANDIDATE_SCHEMA
    )


def require_guard_enforced(
    repository: str, *, fetch: Callable[[str], object] = github_json
) -> dict:
    """Refuse unless ``main`` requires the guard's check from the guard App.

    Locks 2 and 3. The read is the public branch record, so the signing
    runtime needs no credential. Any read error, a missing field or a shape
    this function does not know refuses. A required check that comes from a
    repository ruleset is not in this record and does not count.
    """

    if (
        type(GUARD_APP_ID) is not int
        or GUARD_APP_ID <= 0
        or GUARD_APP_ID == GITHUB_ACTIONS_APP_ID
    ):
        raise AdmissionError(
            "the merge guard App is not recorded on encoder main (GUARD_APP_ID)"
        )
    if re.fullmatch(r"TheAxiomFoundation/rulespec-[a-z]{2}", str(repository)) is None:
        raise AdmissionError("invalid repository for the merge guard requirement")
    try:
        branch = fetch(f"/repos/{repository}/branches/main")
    except AdmissionError:
        raise
    except Exception as exc:  # HTTP error, rate limit, bad JSON: never "absent"
        raise AdmissionError(
            "branch protection of the RuleSpec repository cannot be read"
        ) from exc
    branch = _object(branch, "main branch")
    protection = _object(branch.get("protection"), "main branch protection")
    required = _object(
        protection.get("required_status_checks"), "required status checks"
    )
    checks = required.get("checks")
    if (
        branch.get("name") != "main"
        or branch.get("protected") is not True
        or protection.get("enabled") is not True
        or required.get("enforcement_level") != "everyone"
        or not isinstance(checks, list)
    ):
        raise AdmissionError(
            "main does not enforce required checks for everyone, administrators "
            "included"
        )
    named = [
        _object(check, "required check")
        for check in checks
        if _object(check, "required check").get("context") == GUARD_CHECK_NAME
    ]
    if (
        len(named) != 1
        or type(named[0].get("app_id")) is not int
        or named[0]["app_id"] != GUARD_APP_ID
    ):
        raise AdmissionError(
            "the content-bound merge guard is not a required check of main, "
            "bound to the guard App"
        )
    return {"check": GUARD_CHECK_NAME, "app_id": GUARD_APP_ID}


def _enrolled_dispatcher(run: dict, selector: PromotionSelector) -> dict:
    """The run's triggering actor, by numeric id, enrolled (decision D3)."""

    actor = _object(run.get("triggering_actor"), "triggering actor")
    identity, login = actor.get("id"), actor.get("login")
    if (
        type(identity) is not int
        or identity not in TRUSTED_SIGNING_REVIEWERS
        or login != TRUSTED_SIGNING_REVIEWERS[identity]
        or login != selector.actor
        or actor.get("type") != "User"
    ):
        raise AdmissionError("candidate dispatcher is not an enrolled identity")
    return {"id": identity, "login": login}


def _refuse_changes_requested(reviews: list[dict]) -> None:
    """The blocking half of ``_approved_reviews``; no approval is required."""

    decisive: dict[object, dict] = {}
    for review in sorted(reviews, key=lambda item: item.get("id", 0)):
        reviewer = _object(review.get("user"), "reviewer")
        state = review.get("state")
        if state in {"CHANGES_REQUESTED", "APPROVED", "DISMISSED"}:
            decisive[reviewer.get("id")] = review
        elif state not in {"COMMENTED", "PENDING"}:
            raise AdmissionError("unknown review state")
    if any(review["state"] == "CHANGES_REQUESTED" for review in decisive.values()):
        raise AdmissionError("candidate has unresolved changes requested")


def _revalidate_candidate_admission(
    payload: dict, *, fetch: Callable[[str], object]
) -> dict:
    """Recheck the mutable state of an unreviewed-candidate admission."""

    if (
        V6_WRITER_ENABLED is not True
        or set(payload) != CANDIDATE_FIELDS
        or payload.get("schema") != CANDIDATE_SCHEMA
        or payload.get("owner_review") != CANDIDATE_OWNER_REVIEW
    ):
        raise AdmissionError("invalid provisioned admission")
    selector = selector_from_payload(payload["selector"])
    record = _object(payload["record"], "admission record")
    candidate = _object(payload["candidate"], "admission candidate")
    root = "/repos/" + selector.repository
    run = _object(
        fetch(f"/repos/TheAxiomFoundation/axiom-encode/actions/runs/{selector.run_id}"),
        "protected run",
    )
    if (
        run.get("status") != "in_progress"
        or run.get("head_sha") != payload["workflow_commit"]
        or _enrolled_dispatcher(run, selector) != payload["dispatcher"]
    ):
        raise AdmissionError("protected run changed after provisioning")
    guard = require_guard_enforced(selector.repository, fetch=fetch)
    pr_path = f"{root}/pulls/{selector.pr}"
    pr = _object(fetch(pr_path), "PR")
    base, author_id = _pr_identity(pr, selector)
    main = _object(fetch(root + "/git/ref/heads/main"), "main ref")
    _refuse_changes_requested(_pages(fetch, pr_path + "/reviews"))
    if (
        set(candidate) != {"author_id"}
        or base != record.get("base_sha")
        or _object(main.get("object"), "main object").get("sha") != base
        or author_id != candidate["author_id"]
        or guard != payload["guard"]
    ):
        raise AdmissionError("live candidate differs from provisioned admission")
    return payload


def _pages(fetch: Callable[[str], object], path: str) -> list[dict]:
    result = []
    for page in range(1, MAX_PAGES + 1):
        items = fetch(f"{path}?per_page=100&page={page}")
        if not isinstance(items, list) or len(items) > 100:
            raise AdmissionError("invalid paginated GitHub response")
        result.extend(_object(item, "list entry") for item in items)
        if len(items) < 100:
            return result
    raise AdmissionError("GitHub admission pagination limit reached")


def _pr_identity(pr: dict, selector: PromotionSelector) -> tuple[str, int]:
    head = _object(pr.get("head"), "PR head")
    base = _object(pr.get("base"), "PR base")
    author = _object(pr.get("user"), "PR author")
    if (
        pr.get("number") != selector.pr
        or pr.get("state") != "open"
        or pr.get("merged") is not False
        or head.get("sha") != selector.head
        or head.get("ref") != selector.branch
        or _object(head.get("repo"), "head repository").get("full_name")
        != selector.repository
        or _object(base.get("repo"), "base repository").get("full_name")
        != selector.repository
        or base.get("ref") != "main"
        or not re.fullmatch(r"[0-9a-f]{40}", str(base.get("sha", "")))
        or type(author.get("id")) is not int
    ):
        raise AdmissionError("candidate PR identity, head, or main base differs")
    return base["sha"], author["id"]


def _approved_reviews(
    reviews: list[dict], selector: PromotionSelector, author_id: int
) -> list[dict]:
    decisive: dict[int, dict] = {}
    for review in sorted(reviews, key=lambda item: item.get("id", 0)):
        reviewer = _object(review.get("user"), "reviewer")
        identity = reviewer.get("id")
        state = review.get("state")
        if state in {"CHANGES_REQUESTED", "APPROVED", "DISMISSED"}:
            decisive[identity] = review
        elif state not in {"COMMENTED", "PENDING"}:
            raise AdmissionError("unknown review state")
    if any(review["state"] == "CHANGES_REQUESTED" for review in decisive.values()):
        raise AdmissionError("candidate has unresolved changes requested")
    approvals = []
    for identity, review in decisive.items():
        reviewer = review["user"]
        if (
            identity in TRUSTED_SIGNING_REVIEWERS
            and reviewer.get("login") == TRUSTED_SIGNING_REVIEWERS[identity]
            and reviewer.get("type") == "User"
            and identity != author_id
            and reviewer["login"].lower() != selector.actor.lower()
            and review.get("state") == "APPROVED"
            and review.get("commit_id") == selector.head
            and type(review.get("id")) is int
            and review["id"] > 0
        ):
            approvals.append({"reviewer_id": identity, "review_id": review["id"]})
    if not approvals:
        raise AdmissionError(
            "candidate lacks an independent exact-head signing-owner approval"
        )
    return sorted(approvals, key=lambda value: value["reviewer_id"])


def verify_reviewed_pr(
    selector: PromotionSelector,
    *,
    fetch: Callable[[str], object] = github_json,
) -> dict:
    """Derive a module-bound admission from live, independently approved bytes.

    New modules only. The exact primary/companion and optional derived index
    are the only allowed changes. No fork, stacked base, deleted/renamed file,
    signed-manifest change, workflow change or unrelated candidate is admitted.
    """
    run = _object(
        fetch(f"/repos/TheAxiomFoundation/axiom-encode/actions/runs/{selector.run_id}"),
        "protected run",
    )
    if (
        str(run.get("id")) != selector.run_id
        or run.get("event") != "workflow_dispatch"
        or run.get("path") != ".github/workflows/targeted-signed-reencode.yml"
        or run.get("head_branch") != "main"
        or run.get("status") != "in_progress"
        or not re.fullmatch(r"[0-9a-f]{40}", str(run.get("head_sha", "")))
        or _object(run.get("triggering_actor"), "triggering actor").get("login")
        != selector.actor
    ):
        raise AdmissionError(
            "admission is not bound to the live central main signing run"
        )
    candidate_lane = V6_WRITER_ENABLED is True
    if candidate_lane:
        # Before any candidate read: the dispatcher is enrolled by numeric id,
        # and the merge guard is already a required check of main.
        dispatcher = _enrolled_dispatcher(run, selector)
        guard = require_guard_enforced(selector.repository, fetch=fetch)
    root = "/repos/" + selector.repository
    pr_path = f"{root}/pulls/{selector.pr}"
    before = _object(fetch(pr_path), "PR")
    base, author_id = _pr_identity(before, selector)
    main = _object(fetch(root + "/git/ref/heads/main"), "main ref")
    if _object(main.get("object"), "main object").get("sha") != base:
        raise AdmissionError("PR base is not the current main head")
    comparison = _object(
        fetch(f"{root}/compare/{base}...{selector.head}"), "candidate ancestry"
    )
    if (
        comparison.get("status") != "ahead"
        or _object(comparison.get("merge_base_commit"), "merge base").get("sha") != base
    ):
        raise AdmissionError("candidate must descend from the current main head")
    branch = urllib.parse.quote(selector.branch, safe="")
    tip = _object(fetch(root + "/git/ref/heads/" + branch), "candidate ref")
    if _object(tip.get("object"), "candidate object").get("sha") != selector.head:
        raise AdmissionError("candidate branch moved")
    files = _pages(fetch, pr_path + "/files")
    expected = {selector.rulespec_path, selector.companion}
    base_tree = _object(fetch(f"{root}/git/trees/{base}?recursive=1"), "base tree")
    if base_tree.get("truncated") is not False or not isinstance(
        base_tree.get("tree"), list
    ):
        raise AdmissionError("base tree cannot be inspected completely")
    if any(
        entry.get("path") in expected
        for entry in base_tree["tree"]
        if isinstance(entry, dict)
    ):
        raise AdmissionError("new candidate paths already exist on main")
    paths = [item.get("filename") for item in files]
    if (
        len(set(paths)) != len(paths)
        or not expected.issubset(paths)
        or not set(paths).issubset(expected | {INDEX_PATH})
        or before.get("changed_files") != len(files)
        or any(
            item.get("status") != "added"
            for item in files
            if item.get("filename") in expected
        )
        or any(item.get("status") not in {"added", "modified"} for item in files)
    ):
        raise AdmissionError("candidate PR changes files outside its new module scope")

    reviews = _pages(fetch, pr_path + "/reviews")
    if candidate_lane:
        _refuse_changes_requested(reviews)
    else:
        approvals = _approved_reviews(reviews, selector, author_id)

    digests = {}
    tree = _object(
        fetch(f"{root}/git/trees/{selector.head}?recursive=1"), "candidate tree"
    )
    if tree.get("truncated") is not False or not isinstance(tree.get("tree"), list):
        raise AdmissionError("candidate tree cannot be inspected completely")
    entries = _require_immutable_delta(base_tree, tree, files, comparison)
    for path in sorted(expected):
        entry = entries.get(path, {})
        if entry.get("type") != "blob" or entry.get("mode") != "100644":
            raise AdmissionError("candidate file is not a non-executable regular blob")
        blob = _object(
            fetch(
                f"{root}/contents/{urllib.parse.quote(path, safe='/')}?ref={selector.head}"
            ),
            "candidate file",
        )
        if (
            blob.get("type") != "file"
            or blob.get("path") != path
            or blob.get("encoding") != "base64"
            or blob.get("sha") != entry.get("sha")
        ):
            raise AdmissionError("candidate content is not a regular file")
        try:
            raw = base64.b64decode("".join(blob["content"].splitlines()), validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise AdmissionError("candidate content encoding is invalid") from exc
        if len(raw) != blob.get("size") or len(raw) > 10 * 1024 * 1024:
            raise AdmissionError("candidate content size differs")
        digests[path] = hashlib.sha256(raw).hexdigest()

    # Refuse changes during the API sequence rather than carrying an old
    # approval across a new head/base. Before signing and publication, recheck
    # mutable authorization; the signer rehashes local immutable source bytes.
    after = _object(fetch(pr_path), "PR recheck")
    if _pr_identity(after, selector) != (base, author_id):
        raise AdmissionError("candidate PR moved during admission")
    if _pages(fetch, pr_path + "/reviews") != reviews:
        raise AdmissionError("candidate reviews changed during admission")
    if candidate_lane:
        if require_guard_enforced(selector.repository, fetch=fetch) != guard:
            raise AdmissionError("merge guard requirement changed during admission")
        # No ``approvals`` key and another schema: nobody can read this as a
        # reviewed admission.
        return {
            "schema": CANDIDATE_SCHEMA,
            "workflow_commit": run["head_sha"],
            "selector": selector.__dict__,
            "owner_review": CANDIDATE_OWNER_REVIEW,
            "candidate": {"author_id": author_id},
            "dispatcher": dispatcher,
            "guard": guard,
            "record": {
                "schema": "axiom-encode/reviewed-rulespec-admission/v1",
                "country": selector.country,
                "rulespec_ref": selector.head,
                "rulespec_pr": selector.pr,
                "pr_base_branch": selector.branch,
                "base_sha": base,
                "binding": {
                    "citation": selector.citation,
                    "rulespec_path": selector.rulespec_path,
                    "rulespec_sha256": digests[selector.rulespec_path],
                    "companion_sha256": digests[selector.companion],
                },
                "declared": None,
                "purpose": CANDIDATE_PURPOSE,
            },
        }
    return {
        "schema": SCHEMA,
        "workflow_commit": run["head_sha"],
        "selector": selector.__dict__,
        "approvals": sorted(approvals, key=lambda value: value["reviewer_id"]),
        "record": {
            "schema": "axiom-encode/reviewed-rulespec-admission/v1",
            "country": selector.country,
            "rulespec_ref": selector.head,
            "rulespec_pr": selector.pr,
            "pr_base_branch": selector.branch,
            "base_sha": base,
            "binding": {
                "citation": selector.citation,
                "rulespec_path": selector.rulespec_path,
                "rulespec_sha256": digests[selector.rulespec_path],
                "companion_sha256": digests[selector.companion],
            },
            "declared": None,
            "purpose": "Live exact-head independent signing-owner approval on the RuleSpec PR; new-module promotion only.",
        },
    }
