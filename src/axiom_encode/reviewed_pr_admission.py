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
# How to read "no v6 writer exists" since change 5: the writer exists and is
# locked by the three locks below. For this rule a locked writer counts as
# no writer; the rule's sentences are kept as written because a test pins
# them to the design.
#
# Three locks. Locks 1 and 2 are constants on encoder main: no encoder-side
# environment variable, CLI flag, dispatch input or setting opens them. Lock
# 3 is code on encoder main that separately reads the branch protection of
# the RuleSpec repository, and refuses unless that protection requires the
# guard check from the recorded guard App:
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
#    is not a required check (proposed constraint D2d).
#
# Change 5c: a SECOND accepted form of locks 2 and 3, for the mechanism that
# has no guard App (an organization ruleset "Require workflows to pass before
# merging"; a recommendation, not accepted). Exactly one of ``GUARD_APP_ID``
# and ``GUARD_WORKFLOW`` may be recorded: with neither, and with both, the
# lane refuses before any read. Both are ``None`` here. In the ruleset form
# locks 2 and 3 are:
#
# 2. ``GUARD_WORKFLOW``. ``None`` until a reviewed change records the
#    repository id, the path and the reviewed commit SHAs of the guard
#    workflow. While it is ``None`` (and ``GUARD_APP_ID`` too) the lane
#    refuses, also with lock 1 open.
# 3. ``_require_guard_workflow_ruleset``: a live read of the active branch
#    rules of ``main`` of the RuleSpec repository, at every admission and
#    every revalidation of a candidate-lane run (the reviewed lane never
#    calls it). It refuses unless the listing shows exactly one
#    ``workflows`` rule, from a ruleset of the organization, that pins the
#    recorded workflow at a recorded commit, and a ``merge_queue`` rule of
#    the RuleSpec repository's own ruleset.
#    This read proves LESS than the App form's ("for administrators too"). It
#    proves that the two rules were in the listing at each read. It does not
#    show who may bypass them: no read of this module returns a bypass list,
#    so a bypass entry that exists before the dispatch is invisible at every
#    read. An organization owner (the enrolled signers may be owners) can add
#    such an entry before a run, and only the organization audit log shows
#    it. The activation checklist has a person confirm the bypass lists
#    once; nothing confirms them again.
#    The merge queue half is weaker still. GitHub offers "Require merge
#    queue" in repository rulesets only, so the queue rule lives in a ruleset
#    that the administrators of the RuleSpec repository can edit, disable or
#    give a bypass list. The lock proves that the rule was present at each of
#    its reads (start and end of admission, each revalidation), no more.
#
# OPERATOR PRECONDITION, not checkable here: the read of lock 3 proves the
# requirement at admission and before signing (in the ruleset form: that the
# rules are listed, not that they bind the dispatcher; see above). A
# repository administrator can remove a required check afterwards; that
# adversary is out of scope (design, section 9). Do not turn lock 1 before
# activation checklist step 4 is done.
#
# RESIDUAL RISK the owners would accept by opening lock 1 (a proposal: they
# have not accepted it). Lock 3 is read again, never
# cached, at five points: preflight (start and end), root provisioner (start
# and end), before validation, immediately before signing, before
# publication. AFTER publication only the GitHub setting protects the merge.
# Today the approval before signing is enforced by this module, independent
# of repository settings. After activation an enrolled owner who is also a
# repository administrator can sign their own candidate, then remove the
# required check (or lower its enforcement level, or use the guard App key)
# and merge with no second approval. What exists against that: the
# organisation audit log records the settings change, and the push to main
# still runs validation. Neither prevents the merge or supplies a review.
# In the ruleset form the same owner has two more ways, both invisible to
# lock 3: a bypass entry of either ruleset that is set BEFORE the run (then
# no setting changes after signing), and, as an administrator of the RuleSpec
# repository alone, the merge queue's ruleset.
#
# KNOWN LIMIT (availability only). Every read is anonymous. One signing run
# of this lane makes at least 45 reads (7 of them the branch record) and one
# of the reviewed lane at least 38, counted by the reviewer at 880238c7 with
# every list on one page; pagination adds reads. GitHub allows 60 per hour
# per address, so a second run in the same hour from the same address is
# refused. Follow-up: send the reads with the
# workflow token when one is available. The record is public, so a token
# only raises the limit; any error must still refuse.
V6_WRITER_ENABLED: Final[bool] = False
# The enforcing check of the merge guard, and the App that must publish it.
# The shadow check of the runner has another name and never satisfies this.
GUARD_CHECK_NAME: Final = "content-bound-merge-guard"
GUARD_APP_ID: Final[int | None] = None
# GitHub Actions. Any workflow of the RuleSpec repository can publish a check
# as this App, so it can never be the guard's publisher.
GITHUB_ACTIONS_APP_ID: Final = 15368


@dataclass(frozen=True, slots=True)
class GuardWorkflow:
    """The guard workflow that an organization ruleset must require (5c).

    ``repository_id``: the numeric id of the repository that holds the
    workflow file (axiom-encode). ``path``: exactly ``GUARD_WORKFLOW_PATH``.
    ``shas``: the reviewed commits the ruleset may pin, a non-empty
    ``frozenset`` of 40 lowercase hex characters each. The dataclass does not
    check its fields; ``_recorded_guard_workflow`` does, at every use.
    """

    repository_id: int
    path: str
    shas: frozenset[str]


# The ruleset form of lock 2 (change 5c). ``None`` until a reviewed activation
# change records ``GuardWorkflow(...)`` here. No environment variable, CLI
# flag, dispatch input or setting sets it.
GUARD_WORKFLOW: Final[GuardWorkflow | None] = None
GUARD_WORKFLOW_PATH: Final = ".github/workflows/content-bound-merge-guard.yml"
# The organization whose ruleset must carry the rule. A ruleset of the
# RuleSpec repository itself does not count: its administrators can edit it.
GUARD_RULESET_ORGANIZATION: Final = "TheAxiomFoundation"
GUARD_RULESET_FORM: Final = "organization_ruleset_workflow"
# The fields of the REST schemas this lock knows (github/rest-api-description
# at c9c95fe3, read on 2026-11-17): ``repository-rule-workflows`` parameters
# and ``repository-rule-params-workflow-file-reference``. Any other key
# refuses: an unknown field could change which file runs.
# A rule of the listing (``repository-rule-detailed``) is the rule itself
# (``type``, ``parameters``) joined with ``repository-rule-ruleset-info``.
# Any other key at rule level refuses too, for the ``workflows`` rule and for
# a ``merge_queue`` rule: an unknown field could say whether the rule binds
# (``enforcement``, ``bypass_actors``, ``current_user_can_bypass``).
_BRANCH_RULE_FIELDS: Final = frozenset(
    {"type", "parameters", "ruleset_source_type", "ruleset_source", "ruleset_id"}
)
_WORKFLOWS_PARAMETER_FIELDS: Final = frozenset(
    {"workflows", "do_not_enforce_on_create"}
)
_WORKFLOW_ENTRY_FIELDS: Final = frozenset({"path", "repository_id", "ref", "sha"})
# --- Multi-unit ("batch") form of the v6 lane. SWITCHED OFF. ------------------
#
# Issue #2301, option A; design: ``docs/content-bound-merge-guard.md``, section
# 5, "Multi-unit candidates". With the switch on, one candidate pull request
# may carry up to ``MAX_V6_UNITS`` new modules, each with its own primary, its
# own companion and its own signed v6 manifest, under one owner approval.
#
# ``MULTI_UNIT_V6_ENABLED`` is the one switch of both halves: this module's
# admission of one unit out of a batch candidate (``verify_reviewed_pr``) and
# the merge guard's evaluation of the final head
# (``content_merge_guard._evaluate``). It is a constant on encoder main: no
# environment variable, CLI flag, dispatch input or setting turns it. While
# it is not exactly ``True`` both halves behave as before it existed: the
# same reads, results and errors. It opens nothing by itself: the admission
# half also needs every lock of the v6 writer above, and a multi-unit pass of
# the guard carries its own reason code, which the enforcing runner does not
# accept until a reviewed change adds it to the runner's success set.
MULTI_UNIT_V6_ENABLED: Final[bool] = False
# The most units of one batch candidate. Bounded by the request budget of the
# enforcing guard run (design, "Multi-unit candidates", "Request budget"): one
# evaluation of ``u`` units costs at most ``20 + 9u`` GitHub requests with one
# page per list, and a pull-request run makes two evaluations and at most 101
# polls under a cap of 350 requests: 2 x (20 + 9u) + 101 <= 350 gives u <= 11.
# 10 leaves 29 requests for further pages.
MAX_V6_UNITS: Final[int] = 10
# Where a v6 manifest sits, and the roots a RuleSpec module lives under. This
# file is loaded by path, outside the package, by the root provisioner, so it
# imports nothing from the package: both are copies, pinned by a test to
# ``content_merge_guard.MANIFEST_PREFIX`` and
# ``rulespec_citation_paths.RULESPEC_ATOMIC_ROOTS``.
_BATCH_MANIFEST_PREFIX: Final = ".axiom/encoding-manifests/"
_BATCH_ATOMIC_ROOTS: Final = frozenset(
    {"legislation", "policies", "regulations", "statutes"}
)
_BATCH_JURISDICTION: Final = re.compile(r"[a-z]{2}(-[a-z0-9]+)*")
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

    Change 5c: exactly one of ``GUARD_APP_ID`` (the App form, described
    above) and ``GUARD_WORKFLOW`` (the organization-ruleset form,
    ``_require_guard_workflow_ruleset``) may be recorded. Neither, or both,
    refuses before any read. The two forms return records with different
    keys; callers compare the record as a whole and read no key of it.
    """

    if GUARD_APP_ID is None and GUARD_WORKFLOW is None:
        raise AdmissionError(
            "the merge guard requirement is not recorded on encoder main "
            "(neither GUARD_APP_ID nor GUARD_WORKFLOW)"
        )
    if GUARD_APP_ID is not None and GUARD_WORKFLOW is not None:
        raise AdmissionError(
            "ambiguous merge guard configuration on encoder main: both "
            "GUARD_APP_ID and GUARD_WORKFLOW are recorded"
        )
    if GUARD_WORKFLOW is not None:
        return _require_guard_workflow_ruleset(repository, fetch=fetch)
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


def _recorded_guard_workflow() -> GuardWorkflow:
    """``GUARD_WORKFLOW``, or refuse when it is not the exact structure."""

    recorded = GUARD_WORKFLOW
    malformed = "the recorded merge guard workflow is malformed (GUARD_WORKFLOW)"
    # ``type(...) is``: a subclass, a mapping with the same keys, ``True`` as
    # an id, a list of SHAs and a subclass of ``frozenset`` (it could answer
    # ``in`` as it likes) are all refused.
    if type(recorded) is not GuardWorkflow:
        raise AdmissionError(malformed)
    try:
        repository_id, path, shas = recorded.repository_id, recorded.path, recorded.shas
    except AttributeError as exc:  # an instance made without its fields
        raise AdmissionError(malformed) from exc
    if (
        type(repository_id) is not int
        or repository_id <= 0
        or type(path) is not str
        or path != GUARD_WORKFLOW_PATH
        or type(shas) is not frozenset
        or not shas
        or any(
            type(sha) is not str or re.fullmatch(r"[0-9a-f]{40}", sha) is None
            for sha in shas
        )
    ):
        raise AdmissionError(malformed)
    return recorded


def _require_guard_workflow_ruleset(
    repository: str, *, fetch: Callable[[str], object]
) -> dict:
    """Refuse unless an organization ruleset requires the guard workflow on ``main``.

    The ruleset form of locks 2 and 3 (change 5c). The read is
    ``GET /repos/{repository}/rules/branches/main``, anonymous. GitHub's REST
    documentation (https://docs.github.com/en/rest/repos/rules, "Get rules
    for a branch", read on 2026-11-17) says: "Returns all active rules that
    apply to the specified branch. [...] Rules in rulesets with "evaluate" or
    "disabled" enforcement statuses are not returned." That sentence is about
    the rules of such a ruleset; the listing can still hold rules of other
    rulesets. When the guard's ruleset is in one of those modes the listing
    has no ``workflows`` rule, and the count below refuses.

    Accepted: exactly one rule of type ``workflows`` in the whole listing,
    with no key outside ``_BRANCH_RULE_FIELDS``, from a ruleset of the
    organization ``GUARD_RULESET_ORGANIZATION``, with exactly one workflow
    entry that names the recorded repository id, the recorded path and a
    ``sha`` of the recorded set, and no ``ref``; and at least one rule of
    type ``merge_queue``, each of them a well-formed rule of a ruleset of
    ``repository`` itself. Without the merge queue a pass on the pull request
    stays green after a dismissed approval; only the merge-group run reads
    the reviews again at merge time.

    Decisions, each the stricter reading until experiment X14 records a live
    response:
    - An entry with a ``ref`` that is not null refuses, also next to a good
      ``sha``: which of the two GitHub follows is not documented.
    - Unknown keys in the rule's ``parameters`` or in the workflow entry
      refuse.
    - Unknown keys of the ``workflows`` rule itself refuse, in particular
      ``enforcement``, ``bypass_actors`` and ``current_user_can_bypass``:
      they are fields of a ruleset in the REST schema, not of a rule of this
      listing, and what they would mean here is not documented.
    - ``do_not_enforce_on_create`` must be absent or exactly ``False``. The
      schema describes ``True`` as "Allow repositories and branches to be
      created if a check would otherwise prohibit it"; whether that touches a
      merge into an existing ``main`` is not documented, so it refuses.

    The merge queue rule. GitHub's documentation of the rule "Require merge
    queue" says: "This rule is not available for rulesets created at the
    organization level." (docs.github.com, "Available rules for rulesets",
    read on 2026-11-17; source file in github/docs at 49a012fa, last changed
    in 1e6ab0ef.) So the organization source that is required of the
    ``workflows`` rule cannot be required of the queue. What is required
    instead, of EVERY rule of type ``merge_queue`` in the listing:
    - no key outside ``_BRANCH_RULE_FIELDS`` and ``parameters`` a JSON
      object, as in the one recorded live rule (axiom-encode, 2026-11-17);
    - ``ruleset_source_type`` exactly ``"Repository"``, ``ruleset_source``
      exactly ``repository`` and a positive integer ``ruleset_id``. A queue
      rule "from another repository's ruleset" is not a state GitHub is
      documented to report, and refuses.
    The queue's parameters are NOT checked (``merge_method``,
    ``grouping_strategy``, ``max_entries_to_build``, ``max_entries_to_merge``,
    ``min_entries_to_merge``, ``min_entries_to_merge_wait_minutes``,
    ``check_response_timeout_minutes``; all seven of
    ``repository-rule-merge-queue`` were examined). Reason: the merge-group
    evaluation of the enforcing guard (``evaluate_merge_group`` of change 3b; on
    this branch it is a stub that fails) passes only when
    the group is built on the live ``main`` tip and its tree is the tree of
    one judged pull request head. A group of two or more pull requests has
    another tree or another base and fails at the guard, with ``ALLGREEN``
    and with ``HEADGREEN``, and whatever the merge method, because trees are
    compared. The size, wait and concurrency parameters only decide whether
    and when a group forms; a check that does not report within the timeout
    counts as failed (schema text). So no parameter value lets a merge pass
    the merge-group check without a judged tree; a wrong value stops merges.
    This rests on that guard, which this function does not read.

    What this read cannot see, and what the lock therefore does not prove:
    - The bypass list of the organization ruleset.
      ``GET /repos/{owner}/{repo}/rulesets/{id}`` returns ``bypass_actors``
      only to a caller with write access to the ruleset (same documentation
      page), so an anonymous reader never sees it. An actor on that list
      merges without the workflow.
    - Anything about who controls the merge queue rule. It is in a
      repository ruleset, which the administrators of the RuleSpec
      repository can edit, disable or give a bypass list; that bypass list
      is not readable here either. An administrator on it merges outside the
      queue, with the pull-request pass that stays green after a dismissed
      approval. The lock proves only that the rule was present at each of
      its reads: the start and the end of admission and each revalidation.
    - That the pinned file is the guard. The path is a name. The recorded
      SHA set is the ONLY evidence: the reviewed change that records a SHA
      in ``GUARD_WORKFLOW`` asserts that the file at that path in that commit
      is the reviewed guard workflow. This function does not read the file,
      does not check that the commit exists in the recorded repository, and
      compares the repository by its numeric id only.
    - Whether the response of this endpoint for a ``workflows`` rule has the
      shape assumed here. The schema is documented, an example is not; the
      test fixture is SYNTHETIC.
      [experiment X14]: must be replaced by the recorded live response before
      activation.
    - That GitHub runs the file of exactly the pinned commit (experiment X1),
      and everything after the read: an organization owner can change the
      ruleset afterwards (design, section 9).
    """

    recorded = _recorded_guard_workflow()
    # An exact ``str``: the queue rule below is compared with this value, and
    # an object with its own ``__eq__`` could match another repository's rule.
    if (
        type(repository) is not str
        or re.fullmatch(r"TheAxiomFoundation/rulespec-[a-z]{2}", repository) is None
    ):
        raise AdmissionError("invalid repository for the merge guard requirement")
    try:
        rules = _pages(fetch, f"/repos/{repository}/rules/branches/main")
    except AdmissionError:
        raise
    except Exception as exc:  # HTTP error, rate limit, bad JSON: never "absent"
        raise AdmissionError(
            "branch rules of the RuleSpec repository cannot be read"
        ) from exc
    if any(type(rule.get("type")) is not str for rule in rules):
        raise AdmissionError("a branch rule of main has no type")
    required = [rule for rule in rules if rule["type"] == "workflows"]
    if len(required) != 1:
        raise AdmissionError(
            "main must have exactly one active rule that requires workflows; "
            f"the branch rules show {len(required)}"
        )
    rule = required[0]
    if not set(rule) <= _BRANCH_RULE_FIELDS:
        raise AdmissionError(
            "the rule that requires workflows on main has an unknown field"
        )
    ruleset_id = rule.get("ruleset_id")
    if (
        rule.get("ruleset_source_type") != "Organization"
        or rule.get("ruleset_source") != GUARD_RULESET_ORGANIZATION
        or type(ruleset_id) is not int
        or ruleset_id <= 0
    ):
        raise AdmissionError(
            "the rule that requires workflows on main is not from a ruleset of "
            f"the organization {GUARD_RULESET_ORGANIZATION}"
        )
    parameters = _object(rule.get("parameters"), "required workflows parameters")
    entries = parameters.get("workflows")
    if not set(parameters) <= _WORKFLOWS_PARAMETER_FIELDS or type(entries) is not list:
        raise AdmissionError("the required workflows rule has an unknown shape")
    if parameters.get("do_not_enforce_on_create", False) is not False:
        raise AdmissionError(
            "the required workflows rule is not enforced on creation "
            "(do_not_enforce_on_create)"
        )
    if len(entries) != 1:
        raise AdmissionError(
            "the organization ruleset must require exactly one workflow; it "
            f"names {len(entries)}"
        )
    entry = _object(entries[0], "required workflow")
    if not set(entry) <= _WORKFLOW_ENTRY_FIELDS:
        raise AdmissionError("the required workflow entry has an unknown field")
    if (
        type(entry.get("repository_id")) is not int
        or entry["repository_id"] != recorded.repository_id
        or entry.get("path") != recorded.path
    ):
        raise AdmissionError(
            "the required workflow is not the recorded merge guard workflow "
            "(repository id or path differs)"
        )
    if entry.get("ref") is not None:
        raise AdmissionError(
            "the required workflow follows a ref; only a pinned commit SHA is accepted"
        )
    sha = entry.get("sha")
    if type(sha) is not str or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        raise AdmissionError("the required workflow is not pinned to a commit SHA")
    if sha not in recorded.shas:
        raise AdmissionError(
            "the required workflow is pinned to a commit that is not recorded "
            "(GUARD_WORKFLOW)"
        )
    queues = [rule for rule in rules if rule["type"] == "merge_queue"]
    if not queues:
        raise AdmissionError(
            "main does not require a merge queue; the merge guard reads the "
            "reviews again only on the merge group"
        )
    for queue in queues:
        if (
            not set(queue) <= _BRANCH_RULE_FIELDS
            or type(queue.get("parameters")) is not dict
        ):
            raise AdmissionError("a merge queue rule of main has an unknown shape")
        if (
            queue.get("ruleset_source_type") != "Repository"
            or queue.get("ruleset_source") != repository
            or type(queue.get("ruleset_id")) is not int
            or queue["ruleset_id"] <= 0
        ):
            raise AdmissionError(
                "a merge queue rule of main is not from a ruleset of the "
                "RuleSpec repository itself"
            )
    return {
        "form": GUARD_RULESET_FORM,
        "ruleset_id": ruleset_id,
        "repository_id": recorded.repository_id,
        "path": recorded.path,
        "sha": sha,
    }


def _enrolled_dispatcher(run: dict, selector: PromotionSelector) -> dict:
    """The run's triggering actor, by numeric id, enrolled (proposed constraint D3)."""

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

    # An unknown shape refuses. A review without a numeric id has no place in
    # the order, and reviewers without a numeric id share one key: the later
    # review of one would replace, and so cancel, the changes another
    # requested. ``type(...) is int`` also keeps ``True`` apart from ``1``.
    for review in reviews:
        if type(review.get("id")) is not int:
            raise AdmissionError("review lacks a numeric identity")
    decisive: dict[int, dict] = {}
    for review in sorted(reviews, key=lambda item: item["id"]):
        reviewer = _object(review.get("user"), "reviewer")
        identity = reviewer.get("id")
        if type(identity) is not int:
            raise AdmissionError("reviewer lacks a numeric identity")
        state = review.get("state")
        if state in {"CHANGES_REQUESTED", "APPROVED", "DISMISSED"}:
            decisive[identity] = review
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


def multi_unit_v6_enabled() -> bool:
    """Whether a v6 candidate may carry more than one unit (the one switch)."""

    return MULTI_UNIT_V6_ENABLED is True


def _batch_unit_member(path: str) -> tuple[str, str] | None:
    """``(module stem, role)`` of a path that can belong to a promotion unit.

    The stem is the primary without ``.yaml``. A unit is
    ``<stem>.yaml`` (``primary``), ``<stem>.test.yaml`` (``companion``) and
    ``.axiom/encoding-manifests/<stem>.json`` (``manifest``), where the stem
    is ``<atomic root>/...`` or ``<jurisdiction>/<atomic root>/...`` with no
    hidden segment. Everything else is ``None``.
    """

    if path.startswith(_BATCH_MANIFEST_PREFIX) and path.endswith(".json"):
        stem, role = path[len(_BATCH_MANIFEST_PREFIX) : -len(".json")], "manifest"
    elif path.endswith(".test.yaml"):
        stem, role = path[: -len(".test.yaml")], "companion"
    elif path.endswith(".yaml"):
        stem, role = path[: -len(".yaml")], "primary"
    else:
        return None
    parts = stem.split("/")
    if (
        any(not part or part.startswith(".") for part in parts)
        or stem.endswith(".test")
        or not (
            (parts[0] in _BATCH_ATOMIC_ROOTS and len(parts) >= 2)
            or (
                len(parts) >= 3
                and _BATCH_JURISDICTION.fullmatch(parts[0]) is not None
                and parts[1] in _BATCH_ATOMIC_ROOTS
            )
        )
    ):
        return None
    return stem, role


def _batch_sibling_paths(paths: list, selector: PromotionSelector) -> frozenset[str]:
    """The paths of the other units of a batch candidate, or a refusal.

    Called only with every lock of the v6 writer open and the multi-unit
    switch on. ``paths`` is the candidate pull request's file list. Besides
    this run's own primary and companion, every path must belong to a
    complete sibling unit: a primary and its companion, and optionally the
    manifest at the canonical position of that primary (a sibling that an
    earlier run of this batch already signed). The run's own unit must not
    have a manifest yet. The whole batch holds at most ``MAX_V6_UNITS``
    units. Judged from paths alone: the merge guard judges the final head,
    and the sibling bytes are not this run's subject.

    Returns the empty set for a candidate that holds no file of another
    unit. Such a candidate is not a batch: the caller judges it exactly as a
    single-unit candidate, with the single-unit words.
    """

    own = {selector.rulespec_path, selector.companion}
    own_stem = selector.rulespec_path.removesuffix(".yaml")
    members = {
        path: _batch_unit_member(path) if isinstance(path, str) else None
        for path in paths
        if path not in own
    }
    if not any(
        member is not None and member[0] != own_stem for member in members.values()
    ):
        return frozenset()
    siblings: dict[str, set[str]] = {}
    for member in members.values():
        if member is None:
            raise AdmissionError(
                "batch candidate changes a file that belongs to no promotion unit"
            )
        siblings.setdefault(member[0], set()).add(member[1])
    if own_stem in siblings:
        raise AdmissionError("batch candidate already holds a manifest for this unit")
    if any(not {"primary", "companion"} <= roles for roles in siblings.values()):
        raise AdmissionError("batch candidate holds an incomplete sibling unit")
    limit = MAX_V6_UNITS
    if type(limit) is not int or len(siblings) + 1 > limit:
        raise AdmissionError("batch candidate holds more units than the batch limit")
    return frozenset(members)


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
    # The multi-unit form (switched off): the other files of the candidate
    # may be complete sibling units, all new. With the switch off, or with no
    # file of another unit, the allowed set is the one of a single-unit
    # candidate.
    siblings: frozenset[str] = frozenset()
    if (
        candidate_lane
        and MULTI_UNIT_V6_ENABLED is True
        and len(set(paths)) == len(paths)
        and expected.issubset(paths)
    ):
        siblings = _batch_sibling_paths(paths, selector)
    allowed = expected | siblings if siblings else expected | {INDEX_PATH}
    if (
        len(set(paths)) != len(paths)
        or not expected.issubset(paths)
        or not set(paths).issubset(allowed)
        or before.get("changed_files") != len(files)
        or any(
            item.get("status") != "added"
            for item in files
            if item.get("filename") in expected | siblings
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
        # No ``approvals`` key and another outer ``schema``: nobody can read
        # this as a reviewed admission. The nested ``record.schema`` is the
        # same in both lanes.
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
