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

# What authorizes one promotion (#2301). Read at call time; there is no
# environment variable, CLI flag or repository setting.
#   False: no APPROVED review is required. The accountable act is the dispatch
#     of the protected signing run by an enrolled owner account: the live run's
#     actor (who chose the inputs) and its triggering actor (who started this
#     attempt) must be the same ``TRUSTED_SIGNING_REVIEWERS`` identity (numeric
#     id, login and type ``User``). This identifies an account, not a person:
#     a tool or agent session that holds an owner's token passes. No second
#     person sees the candidate before it is signed. An outstanding
#     changes-requested review still refuses.
#   True (or any value that is not exactly ``False``): an independent enrolled
#     owner must have approved the exact candidate head, and any account with
#     write access may dispatch. This is the rule that was in force before.
# Every other check is the same under both values.
OWNER_APPROVAL_REQUIRED: Final[bool] = False
BASIS_OWNER_APPROVAL: Final = "owner_approval"
BASIS_OWNER_DISPATCH: Final = "owner_dispatch"
# The one field of an admission that holds the evidence for each basis. A
# record names its basis; it never carries an empty approvals list.
_BASIS_EVIDENCE_FIELD: Final = {
    BASIS_OWNER_APPROVAL: "approvals",
    BASIS_OWNER_DISPATCH: "dispatcher",
}
# v2 added ``admission_basis`` and the per-basis evidence field.
SCHEMA = "axiom-encode/live-reviewed-pr-admission/v2"
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_PAGES = 10
INDEX_PATH = ".axiom/index/provisions_to_rules.json"

# --- Several units in one candidate ("batch"). SWITCHED OFF. ------------------
#
# Issue #2301; ``docs/reviewed-pr-promotion.md``, "Several units in one
# candidate". With the switch on, one candidate pull request may carry up to
# ``MAX_PROMOTION_UNITS`` new modules, each with its own primary and its own
# companion. One protected run admits, validates and signs all of them, and
# writes one signed manifest per unit in the form of a single-unit promotion.
#
# The switch is a constant on encoder main, read at call time: no environment
# variable, CLI flag, dispatch input or repository setting turns it. While it
# is not exactly ``True`` this module does not know the batch forms: a selector
# with ``units`` and an admission with ``records`` are refused with the errors
# they had before the switch existed, and every single-unit read, result and
# error is unchanged.
MULTI_UNIT_PROMOTION_ENABLED: Final[bool] = False
# The most units of one batch. The bound is the anonymous GitHub API budget of
# one signing run (60 requests an hour for one address; the signing runtime
# holds no token). One admission of ``k`` units reads ``11 + 2k`` times with
# one page per list (two content reads per unit); a revalidation reads 4
# times. One run admits twice (preflight, root provisioner) and revalidates
# three times (before validation, before signing, before publication):
#   2 x (11 + 2k) + 3 x 4 = 34 + 4k <= 60 gives k <= 6.
# A single-unit promotion costs 38 of the 60. A larger batch needs fewer or
# authenticated reads first, in a reviewed change of this constant.
# ``tests/test_multi_unit_promotion_admission.py`` pins the arithmetic.
MAX_PROMOTION_UNITS: Final[int] = 6
ANONYMOUS_REQUESTS_PER_HOUR: Final = 60
BATCH_SCHEMA: Final = "axiom-encode/live-reviewed-pr-batch-admission/v1"
# The root provisioner reads at most this many bytes of a selector file.
MAX_SELECTOR_BYTES: Final = 4096


class AdmissionError(ValueError):
    """The requested candidate has no current authorization."""


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


_BATCH_SELECTOR_FIELDS: Final = frozenset(
    {"country", "pr", "head", "branch", "actor", "run_id", "units"}
)


@dataclass(frozen=True)
class BatchPromotionSelector:
    """One candidate pull request and every unit it carries.

    ``units`` is ``((citation, rulespec_path), ...)`` in strictly ascending
    order of ``rulespec_path``, so no primary is named twice and one batch has
    one selector. Every unit is checked as the selector of a single-unit
    promotion is.
    """

    country: str
    pr: int
    head: str
    branch: str
    actor: str
    units: tuple[tuple[str, str], ...]
    run_id: str = ""

    def __post_init__(self):
        if (
            not isinstance(self.units, tuple)
            or not self.units
            or any(
                not isinstance(unit, tuple)
                or len(unit) != 2
                or not all(isinstance(value, str) for value in unit)
                for unit in self.units
            )
        ):
            raise AdmissionError("batch selector must name at least one unit")
        selectors = self.unit_selectors()
        paths = [unit.rulespec_path for unit in selectors]
        if any(left >= right for left, right in zip(paths, paths[1:])):
            raise AdmissionError(
                "batch units must be in ascending path order and name no primary twice"
            )
        citations = [unit.citation for unit in selectors]
        if len(set(citations)) != len(citations):
            raise AdmissionError("batch units must not name a citation twice")

    def unit_selectors(self) -> tuple[PromotionSelector, ...]:
        """Each unit as the selector of a single-unit promotion of it."""
        return tuple(
            PromotionSelector(
                country=self.country,
                pr=self.pr,
                head=self.head,
                citation=citation,
                rulespec_path=rulespec_path,
                branch=self.branch,
                actor=self.actor,
                run_id=self.run_id,
            )
            for citation, rulespec_path in self.units
        )

    @property
    def repository(self) -> str:
        return f"TheAxiomFoundation/rulespec-{self.country}"

    def payload(self) -> dict:
        """The JSON form: what ``selector_from_payload`` reads back."""
        return {
            "country": self.country,
            "pr": self.pr,
            "head": self.head,
            "branch": self.branch,
            "actor": self.actor,
            "run_id": self.run_id,
            "units": [
                {"citation": citation, "rulespec_path": rulespec_path}
                for citation, rulespec_path in self.units
            ],
        }


def _require_batch_in_force(selector: BatchPromotionSelector) -> None:
    """Refuse a batch while the switch is off, and one over the unit limit."""
    if MULTI_UNIT_PROMOTION_ENABLED is not True:
        raise AdmissionError("multi-unit promotion is switched off")
    limit = MAX_PROMOTION_UNITS
    if type(limit) is not int or len(selector.units) > limit:
        raise AdmissionError(
            f"batch names {len(selector.units)} units; the limit is {limit}"
        )
    if (
        len(json.dumps(selector.payload(), sort_keys=True).encode()) + 1
        > MAX_SELECTOR_BYTES
    ):
        raise AdmissionError("batch selector exceeds the provisioner's size bound")


def _batch_selector_from_payload(payload: dict) -> BatchPromotionSelector:
    if set(payload) != _BATCH_SELECTOR_FIELDS or not isinstance(payload["units"], list):
        raise AdmissionError("batch promotion selector has unexpected fields")
    units = []
    for unit in payload["units"]:
        unit = _object(unit, "batch unit")
        if set(unit) != {"citation", "rulespec_path"}:
            raise AdmissionError("batch unit has unexpected fields")
        units.append((unit["citation"], unit["rulespec_path"]))
    selector = BatchPromotionSelector(
        **{key: value for key, value in payload.items() if key != "units"},
        units=tuple(units),
    )
    _require_batch_in_force(selector)
    return selector


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


def selector_from_payload(
    value: object,
) -> PromotionSelector | BatchPromotionSelector:
    payload = _object(value, "promotion selector")
    if MULTI_UNIT_PROMOTION_ENABLED is True and "units" in payload:
        return _batch_selector_from_payload(payload)
    if set(payload) != set(PromotionSelector.__dataclass_fields__):
        raise AdmissionError("promotion selector has unexpected fields")
    return PromotionSelector(**payload)


def _admission_basis() -> str:
    """The basis the policy in force admits. Only exactly ``False`` drops the approval."""
    return (
        BASIS_OWNER_DISPATCH
        if OWNER_APPROVAL_REQUIRED is False
        else BASIS_OWNER_APPROVAL
    )


def _enrolled_run_account(run: dict, field: str, label: str) -> tuple[int, str]:
    account = _object(run.get(field), label)
    identity = account.get("id")
    if (
        type(identity) is not int
        or identity not in TRUSTED_SIGNING_REVIEWERS
        or account.get("login") != TRUSTED_SIGNING_REVIEWERS[identity]
        or account.get("type") != "User"
    ):
        raise AdmissionError("dispatch actor is not a verified enrolled signing owner")
    return identity, account["login"]


def _enrolled_dispatcher(run: dict, selector: PromotionSelector) -> dict:
    """The live run's dispatcher as one enrolled owner account, or refuse.

    Two accounts are read from the live run and must be the same enrolled
    identity: ``actor``, who dispatched the run and chose its inputs, and
    ``triggering_actor``, who started this attempt. So an owner's re-run of
    another account's dispatch refuses, and so does another account's re-run
    of an owner's dispatch. For each, the numeric id is the identity, the
    login must be the enrolled login of that id, and the account type must be
    ``User``. The selector's actor must be that login. A missing or malformed
    field refuses.

    Type ``User`` excludes Apps and bots. It does not prove that a person
    acted: any session that holds the owner's token is that ``User``.
    """
    actor_id, _login = _enrolled_run_account(run, "actor", "run actor")
    triggering_id, login = _enrolled_run_account(
        run, "triggering_actor", "triggering actor"
    )
    if actor_id != triggering_id or selector.actor != login:
        raise AdmissionError("dispatch actor is not a verified enrolled signing owner")
    return {"login": login, "actor_id": actor_id, "triggering_actor_id": triggering_id}


def revalidate_admission(
    expected: object, *, fetch: Callable[[str], object] = github_json
) -> dict:
    payload = _object(expected, "provisioned admission")
    basis = payload.get("admission_basis")
    if not isinstance(basis, str) or basis not in _BASIS_EVIDENCE_FIELD:
        raise AdmissionError("unknown admission basis")
    # The batch form exists only while its switch is on.
    batch = MULTI_UNIT_PROMOTION_ENABLED is True and "records" in payload
    if set(payload) != {
        "schema",
        "selector",
        "records" if batch else "record",
        "workflow_commit",
        "admission_basis",
        _BASIS_EVIDENCE_FIELD[basis],
    } or payload.get("schema") != (BATCH_SCHEMA if batch else SCHEMA):
        raise AdmissionError("invalid provisioned admission")
    if basis != _admission_basis():
        raise AdmissionError("admission basis is not the policy in force")
    selector = selector_from_payload(payload["selector"])
    if isinstance(selector, BatchPromotionSelector) is not batch:
        raise AdmissionError("invalid provisioned admission")
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
    if (
        basis == BASIS_OWNER_DISPATCH
        and _enrolled_dispatcher(run, selector) != payload["dispatcher"]
    ):
        raise AdmissionError("dispatching owner changed after provisioning")
    pr_path = f"{root}/pulls/{selector.pr}"
    pr = _object(fetch(pr_path), "PR")
    base, author_id = _pr_identity(pr, selector)
    main = _object(fetch(root + "/git/ref/heads/main"), "main ref")
    reviews = _pages(fetch, pr_path + "/reviews")
    if basis == BASIS_OWNER_DISPATCH:
        # No approval to compare; the veto is still read live.
        _decisive_reviews(reviews)
        evidence_unchanged = True
    else:
        evidence_unchanged = (
            _approved_reviews(reviews, selector, author_id) == payload["approvals"]
        )
    if (
        base
        != (
            _batch_records_base(payload["records"], selector)
            if batch
            else payload["record"]["base_sha"]
        )
        or _object(main.get("object"), "main object").get("sha") != base
        or not evidence_unchanged
    ):
        raise AdmissionError(
            "live approval or candidate differs from provisioned admission"
        )
    # The head and base are immutable Git objects already fully inspected by
    # the root provisioner. The signer separately rehashes the local primary
    # and companion; re-fetch only mutable authorization, not all source bytes.
    return payload


def _batch_records_base(records: object, selector: BatchPromotionSelector) -> str:
    """The one main base of a batch admission's records, or refuse.

    One record per unit, in the selector's order, each bound to its unit, all
    on one base.
    """
    if not isinstance(records, list) or len(records) != len(selector.units):
        raise AdmissionError("batch admission lacks one record per unit")
    bases = []
    for record, (citation, rulespec_path) in zip(records, selector.units):
        binding = _object(_object(record, "batch record").get("binding"), "binding")
        if (
            binding.get("citation") != citation
            or binding.get("rulespec_path") != rulespec_path
        ):
            raise AdmissionError(
                f"batch record is not bound to its unit {citation} ({rulespec_path})"
            )
        bases.append(record.get("base_sha"))
    # Compared, not hashed: a base of any JSON type refuses as an admission.
    if not isinstance(bases[0], str) or any(base != bases[0] for base in bases):
        raise AdmissionError("batch records differ in their main base")
    return bases[0]


def _batch_scope_issue(
    units: tuple[PromotionSelector, ...], files: list[dict]
) -> str | None:
    """Name the unit or the file that makes a batch's change set wrong.

    The change set of the candidate must be exactly the named units: each
    unit's primary and companion, both added, and at most the derived index
    besides. A unit the operator did not name shows as a file of no named
    unit; a named unit the candidate lacks shows as a missing file.
    """
    status = {}
    for item in files:
        path = item.get("filename")
        if not isinstance(path, str) or path in status:
            return "candidate PR lists an invalid or duplicate changed file"
        status[path] = item.get("status")
    owned = {}
    for unit in units:
        for path in (unit.rulespec_path, unit.companion):
            owned[path] = unit
            label = f"unit {unit.citation} ({unit.rulespec_path})"
            if path not in status:
                return f"candidate PR lacks {path} of {label}"
            if status[path] != "added":
                return f"candidate PR does not add {path} of {label}"
    for path in sorted(status):
        if path not in owned and path != INDEX_PATH:
            return (
                f"candidate PR changes {path}, which belongs to no unit named "
                "in the dispatch"
            )
    return None


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


def _decisive_reviews(reviews: list[dict]) -> dict[int, dict]:
    """Each reviewer's latest decisive review; refuse on the veto.

    Any reviewer's outstanding changes-requested review refuses, under both
    admission bases. It is not limited to enrolled owners.
    """
    # Order and reviewer identity decide the veto, so neither may be guessed.
    for review in reviews:
        if any(
            type(value) is not int or value <= 0
            for value in (
                review.get("id"),
                _object(review.get("user"), "reviewer").get("id"),
            )
        ):
            raise AdmissionError("review or reviewer identity is malformed")
    decisive: dict[int, dict] = {}
    for review in sorted(reviews, key=lambda item: item["id"]):
        reviewer = review["user"]
        identity = reviewer["id"]
        state = review.get("state")
        if state in {"CHANGES_REQUESTED", "APPROVED", "DISMISSED"}:
            decisive[identity] = review
        elif state not in {"COMMENTED", "PENDING"}:
            raise AdmissionError("unknown review state")
    if any(review["state"] == "CHANGES_REQUESTED" for review in decisive.values()):
        raise AdmissionError("candidate has unresolved changes requested")
    return decisive


def _approved_reviews(
    reviews: list[dict], selector: PromotionSelector, author_id: int
) -> list[dict]:
    decisive = _decisive_reviews(reviews)
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
    selector: PromotionSelector | BatchPromotionSelector,
    *,
    fetch: Callable[[str], object] = github_json,
) -> dict:
    """Derive a module-bound admission from live GitHub state.

    ``OWNER_APPROVAL_REQUIRED`` decides the basis: an enrolled owner's dispatch
    of the run (``False``) or an independent enrolled owner's approval of the
    exact head (``True``). The admission names its basis.

    New modules only. The exact primary/companion and optional derived index
    are the only allowed changes. No fork, stacked base, deleted/renamed file,
    signed-manifest change, workflow change or unrelated candidate is admitted.

    A ``BatchPromotionSelector`` (``MULTI_UNIT_PROMOTION_ENABLED``) names every
    unit of the candidate. The same reads and rules apply; the change set must
    be exactly the primaries and companions of the named units, and the
    admission carries one record per unit, each equal to the record of a
    single-unit admission of that unit.
    """
    batch = isinstance(selector, BatchPromotionSelector)
    if batch:
        _require_batch_in_force(selector)
    units = selector.unit_selectors() if batch else (selector,)
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
    basis = _admission_basis()
    dispatcher = (
        _enrolled_dispatcher(run, selector) if basis == BASIS_OWNER_DISPATCH else None
    )
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
    expected = {path for unit in units for path in (unit.rulespec_path, unit.companion)}
    base_tree = _object(fetch(f"{root}/git/trees/{base}?recursive=1"), "base tree")
    if base_tree.get("truncated") is not False or not isinstance(
        base_tree.get("tree"), list
    ):
        raise AdmissionError("base tree cannot be inspected completely")
    if batch:
        on_main = {
            entry.get("path") for entry in base_tree["tree"] if isinstance(entry, dict)
        }
        for unit in units:
            for path in (unit.rulespec_path, unit.companion):
                if path in on_main:
                    raise AdmissionError(
                        f"{path} of unit {unit.citation} ({unit.rulespec_path}) "
                        "already exists on main"
                    )
    if any(
        entry.get("path") in expected
        for entry in base_tree["tree"]
        if isinstance(entry, dict)
    ):
        raise AdmissionError("new candidate paths already exist on main")
    paths = [item.get("filename") for item in files]
    if batch:
        # The same rules as below, with the unit or the file named.
        issue = _batch_scope_issue(units, files)
        if issue is not None:
            raise AdmissionError(issue)
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
    if basis == BASIS_OWNER_DISPATCH:
        # No approval is required; an outstanding changes-requested review
        # still refuses.
        _decisive_reviews(reviews)
        evidence: object = dispatcher
        purpose = (
            "Enrolled signing-owner dispatch of the protected signing run; no "
            "owner approval of the RuleSpec PR; new-module promotion only."
        )
    else:
        evidence = _approved_reviews(reviews, selector, author_id)
        purpose = "Live exact-head independent signing-owner approval on the RuleSpec PR; new-module promotion only."

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

    def record(unit: PromotionSelector) -> dict:
        return {
            "schema": "axiom-encode/reviewed-rulespec-admission/v1",
            "country": unit.country,
            "rulespec_ref": unit.head,
            "rulespec_pr": unit.pr,
            "pr_base_branch": unit.branch,
            "base_sha": base,
            "binding": {
                "citation": unit.citation,
                "rulespec_path": unit.rulespec_path,
                "rulespec_sha256": digests[unit.rulespec_path],
                "companion_sha256": digests[unit.companion],
            },
            "declared": None,
            "purpose": purpose,
        }

    if batch:
        return {
            "schema": BATCH_SCHEMA,
            "workflow_commit": run["head_sha"],
            "selector": selector.payload(),
            "admission_basis": basis,
            _BASIS_EVIDENCE_FIELD[basis]: evidence,
            "records": [record(unit) for unit in units],
        }
    return {
        "schema": SCHEMA,
        "workflow_commit": run["head_sha"],
        "selector": selector.__dict__,
        "admission_basis": basis,
        _BASIS_EVIDENCE_FIELD[basis]: evidence,
        "record": record(selector),
    }
