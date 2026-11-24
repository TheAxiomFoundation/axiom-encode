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


# The command that binds an existing candidate (docs/reviewed-pr-promotion.md).
SOURCE_PIN_COMMAND: Final = (
    "axiom-encode bind-source-verification --repo <rulespec checkout> "
    "--citation <citation> --corpus-path <corpus checkout>"
)
_PIN_KEY_LINE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):(?: +(.*))?")
_PIN_PLAIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@+-]*")
_PIN_NUMBER_LIKE = re.compile(r"[-+]?[0-9][0-9_]*(?:[eE][-+]?[0-9]+)?")
_PIN_VALUE = re.compile(
    r"""(?:(?P<plain>[^\s'"#][^\s#]*)|'(?P<single>[^'\\]*)'|"(?P<double>[^"\\]*)")"""
    r"""(?:\s+#.*)?"""
)


def _pin_refusal(detail: str) -> AdmissionError:
    return AdmissionError(
        f"candidate module is not bound to its source: {detail}. Run "
        f"`{SOURCE_PIN_COMMAND}` on the candidate branch, push the change and "
        "dispatch the new head"
    )


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_content(line: str) -> bool:
    stripped = line.strip(" ")
    return bool(stripped) and not stripped.startswith("#")


def _block_after(lines: list[str], header: int, parent_indent: int) -> list[str]:
    """The content lines that belong to the block mapping under ``lines[header]``."""
    block = []
    for line in lines[header + 1 :]:
        if not _is_content(line):
            continue
        if _indent(line) <= parent_indent:
            break
        block.append(line)
    return block


def read_candidate_source_pin(raw: bytes) -> tuple[str | None, str | None]:
    """Read ``module.source_verification`` by a strict line read; no YAML parser.

    Returns ``(corpus_citation_path, source_sha256)``; a missing key is
    ``None``. Standard library only: this module also runs where the encoder
    package and PyYAML are absent. It accepts one shape, the plain block form
    that the encoder and ``bind-source-verification`` write::

        module:
          source_verification:
            corpus_citation_path: <path>
            source_sha256: <64 hex>

    one ``module:`` line at column 0, one ``source_verification:`` line among
    its keys, and one ``key: value`` line per entry with a one-line plain or
    quoted value. Every other shape refuses (flow style, anchor, alias, tag,
    merge key, duplicate key, tab indentation, multi-line value, document
    marker). A refusal is not a claim that the YAML is wrong; it means this
    reader will not vouch for it.

    This is an early refusal, not the binding proof. It does not prove that a
    YAML parser reads the same values (a multi-line quoted scalar elsewhere can
    hide these lines), and it does not know the corpus. The protected job
    still parses the module and compares the pin with the resolver's
    attestation before it signs.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise _pin_refusal("the module is not UTF-8") from None
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    if any(mark in line for line in lines for mark in "\r\x85\u2028\u2029"):
        raise _pin_refusal("the module uses a line break other than LF or CRLF")
    if text.startswith("\ufeff") or any(
        line.startswith(("---", "...", "%")) for line in lines
    ):
        raise _pin_refusal(
            "the module has a byte order mark, a directive or a document marker"
        )
    top_level = [
        index
        for index, line in enumerate(lines)
        if re.match(r"""["']?module["']?\s*:""", line)
    ]
    if len(top_level) != 1 or lines[top_level[0]].rstrip(" ") != "module:":
        raise _pin_refusal("the file has no single plain block `module:` mapping")
    if any(line.startswith("\t") for line in lines):
        raise _pin_refusal("a line of the file is indented with tabs")
    module = _block_after(lines, top_level[0], 0)
    if not module:
        raise _pin_refusal("`module` is empty")
    key_indent = _indent(module[0])
    if any(_indent(line) < key_indent for line in module):
        raise _pin_refusal("`module` keys are not indented evenly")
    keys = [line[key_indent:] for line in module if _indent(line) == key_indent]
    if any(key.startswith(("<<", "?", "&", "*", "!", "{", "[", "\t")) for key in keys):
        raise _pin_refusal(
            "`module` uses a merge key, anchor, tag, flow style or tab indentation"
        )
    headers = [
        key for key in keys if re.match(r"""["']?source_verification["']?\s*:""", key)
    ]
    if not headers:
        raise _pin_refusal("the module has no `module.source_verification` block")
    if len(headers) != 1 or headers[0].rstrip(" ") != "source_verification:":
        raise _pin_refusal(
            "`module.source_verification` is not a single plain block mapping"
        )
    header = lines.index(" " * key_indent + headers[0], top_level[0])
    entries = _block_after(lines, header, key_indent)
    if not entries:
        raise _pin_refusal("`module.source_verification` is empty")
    entry_indent = _indent(entries[0])
    values: dict[str, str] = {}
    for line in entries:
        match = (
            _PIN_KEY_LINE.fullmatch(line[entry_indent:].rstrip(" "))
            if _indent(line) == entry_indent and "\t" not in line
            else None
        )
        value = _PIN_VALUE.fullmatch(match.group(2) or "") if match else None
        if match is None or value is None or match.group(1) in values:
            raise _pin_refusal(
                "`module.source_verification` is not in the plain block form "
                "(one `key: value` line per entry, no duplicate key)"
            )
        plain = value.group("plain")
        if plain is not None and (
            _PIN_PLAIN.fullmatch(plain) is None
            or _PIN_NUMBER_LIKE.fullmatch(plain) is not None
        ):
            raise _pin_refusal(
                "a `module.source_verification` value is not a plain string"
            )
        values[match.group(1)] = next(
            group
            for group in value.group("plain", "single", "double")
            if group is not None
        )
    return values.get("corpus_citation_path"), values.get("source_sha256")


def require_candidate_source_pin(raw: bytes, citation: str) -> None:
    """Refuse a candidate primary that a promotion could not bind to its source.

    The module must name the promotion's citation and carry a 64-character
    lowercase hex ``source_sha256``. Whether that pin is the right one is
    decided later, by the protected binding check against the corpus.
    """
    path, pin = read_candidate_source_pin(raw)
    if path != citation:
        raise _pin_refusal(
            "`module.source_verification.corpus_citation_path` is missing or is "
            "not the promotion citation"
        )
    if pin is None or re.fullmatch(r"[0-9a-f]{64}", pin) is None:
        raise _pin_refusal(
            "`module.source_verification.source_sha256` is missing or is not 64 "
            "lowercase hex characters"
        )


def selector_from_payload(value: object) -> PromotionSelector:
    payload = _object(value, "promotion selector")
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
    if (
        set(payload)
        != {
            "schema",
            "selector",
            "record",
            "workflow_commit",
            "admission_basis",
            _BASIS_EVIDENCE_FIELD[basis],
        }
        or payload.get("schema") != SCHEMA
    ):
        raise AdmissionError("invalid provisioned admission")
    if basis != _admission_basis():
        raise AdmissionError("admission basis is not the policy in force")
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
        base != payload["record"]["base_sha"]
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
    selector: PromotionSelector,
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
    contents = {}
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
        contents[path] = raw
    # Under both admission bases: a candidate without the source pin fails the
    # protected binding check after the signing approval (signing run
    # 45328443323). Refuse it here, before the admission exists.
    require_candidate_source_pin(contents[selector.rulespec_path], selector.citation)

    # Refuse changes during the API sequence rather than carrying an old
    # approval across a new head/base. Before signing and publication, recheck
    # mutable authorization; the signer rehashes local immutable source bytes.
    after = _object(fetch(pr_path), "PR recheck")
    if _pr_identity(after, selector) != (base, author_id):
        raise AdmissionError("candidate PR moved during admission")
    if _pages(fetch, pr_path + "/reviews") != reviews:
        raise AdmissionError("candidate reviews changed during admission")
    return {
        "schema": SCHEMA,
        "workflow_commit": run["head_sha"],
        "selector": selector.__dict__,
        "admission_basis": basis,
        _BASIS_EVIDENCE_FIELD[basis]: evidence,
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
            "purpose": purpose,
        },
    }
