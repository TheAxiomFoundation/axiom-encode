#!/usr/bin/env python3
"""Build the real-defects corpus from the triage record.

Inputs, all under the corpus directory:

* ``triage/triage_merged.json``: one row per module a candidate commit
  touched, with keep and drop decisions (``tools/merge_triage.py``).
* ``triage/case_ids.json``: the case id of every row built so far. Ids are
  assigned from this registry before any case is filtered out, so a case
  keeps its id whatever flags a build runs with; a row the registry has not
  seen gets the next free number of its jurisdiction.
* ``triage/provision_review.json``: the per-case provision review
  (``tools/provision_review.py``), when it exists.

Outputs: ``cases/<id>/`` directories with ``case.json``, ``pre_fix.yaml``,
``post_fix.yaml`` and ``provision.txt``; ``index.json``;
``triage/build_log.json`` (rows that failed to build, and cases the flags
removed); and the generated counts block of ``README.md``.

Steps, in order:

1. Build every kept row: module bytes from Git, the provision of the
   module's first citation from the signed corpus release.
2. Assign ids from the registry and group the cases into families.
3. Filter: ``--drop-unclear`` removes ``triage_status: unclear`` cases and
   ``--family-members representatives`` removes every family member but the
   representative. Both are logged, with the case id, in the build log.
4. Apply the shipping policy and the provision review. A reviewed case whose
   decisive text sits under another citation gets that citation's text
   appended to its provision, resolved from the same release.
5. Run the mechanical evidence check, then write the index, the build log
   and the README counts.

Everything the build derives comes from Git (the rulespec checkouts and the
axiom-corpus checkout) and from signed corpus release objects fetched from the
public registry. Nothing is hand-written.

Usage (from the axiom-encode checkout; the committed corpus is this command)::

    uv run python benchmarks/verifier/real_defects_v0/tools/build_real_defects.py \
        --corpus-dir benchmarks/verifier/real_defects_v0 \
        --rulespec-us ../rulespec-us --rulespec-uk ../rulespec-uk \
        --axiom-corpus ../axiom-corpus --release-cache ~/.cache/axiom-real-defects \
        --drop-unclear --family-members representatives

``--refresh-review`` skips steps 1 to 3: it re-applies the provision review,
the evidence check, the index and the README counts to the cases already
built (it needs the release cache and the axiom-corpus checkout only when a
provision has to be extended).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[4]
_SPEC = importlib.util.spec_from_file_location(
    "verify_real_defects", ROOT / "scripts" / "verify_real_defects.py"
)
assert _SPEC is not None and _SPEC.loader is not None
lib = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lib)
_EVIDENCE_SPEC = importlib.util.spec_from_file_location(
    "check_evidence", Path(__file__).resolve().parent / "check_evidence.py"
)
assert _EVIDENCE_SPEC is not None and _EVIDENCE_SPEC.loader is not None
evidence = importlib.util.module_from_spec(_EVIDENCE_SPEC)
_EVIDENCE_SPEC.loader.exec_module(evidence)

MAIN_REF = "origin/main"
FIRST_PARENT_PROBE = 400
FAMILY_MEMBER_CHOICES = ("all", "representatives")
REMOVED_UNCLEAR = "triage_unclear"
REMOVED_NON_REPRESENTATIVE = "family_non_representative"
COUNTS_BEGIN = (
    "<!-- begin generated counts: tools/build_real_defects.py; "
    "edit the build, not this -->"
)
COUNTS_END = "<!-- end generated counts -->"


def slugify(path: str) -> str:
    stem = re.sub(r"\.ya?ml$", "", path)
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:80]


def list_registry_releases(prefix: str, cache_dir: Path) -> list[dict[str, Any]]:
    """Return every registry release for one jurisdiction prefix, oldest first."""

    listing = cache_dir / "registry_releases.json"
    if listing.exists():
        rows = json.loads(listing.read_text(encoding="utf-8"))
    else:
        url_base, key = lib.registry_credentials()
        url = (
            f"{url_base.rstrip('/')}/rest/v1/release_objects"
            "?select=release_name,content_sha256,created_at&order=release_name.asc&limit=1000"
        )
        request = urllib.request.Request(
            url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Accept-Profile": "corpus",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            rows = json.loads(response.read())
        cache_dir.mkdir(parents=True, exist_ok=True)
        listing.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    releases = []
    for row in rows:
        name = row["release_name"]
        if not name.startswith(prefix + "-"):
            continue
        payload = lib.fetch_release_object(name, row["content_sha256"], cache_dir)
        git_meta = payload["content"]["git"]
        releases.append(
            {
                "name": name,
                "content_sha256": row["content_sha256"],
                "commit": git_meta["commit"],
                "committed_at": git_meta.get("committed_at", ""),
                "created_at": row.get("created_at", ""),
                "payload": payload,
            }
        )
    releases.sort(key=lambda item: (item["committed_at"], item["created_at"]))
    return releases


def toolchain_at(repo: Path, commit: str) -> dict[str, Any]:
    raw = lib.git_blob(repo, commit, ".axiom/toolchain.toml")
    if raw is None:
        return {}
    try:
        return tomllib.loads(raw.decode("utf-8")).get("toolchain", {})
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return {}


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    probe = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant]
    )
    return probe.returncode == 0


def select_releases(
    jurisdiction: str,
    repo: Path,
    corpus_repo: Path,
    commit: str,
    releases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Order the candidate releases for one fix commit.

    The fix commit's toolchain decides: a signed release pin wins; otherwise
    the latest release whose corpus commit is at or before the toolchain's
    ``axiom_corpus_ref``; otherwise the earliest release after it.
    """

    toolchain = toolchain_at(repo, commit)
    pin = toolchain.get("axiom_corpus_release")
    ref = toolchain.get("axiom_corpus_ref")
    if pin:
        first = [r for r in releases if r["name"] == pin]
        rest = [
            r
            for r in releases
            if r["name"] != pin
            and (not first or r["committed_at"] >= first[0]["committed_at"])
        ]
        return first + rest, "toolchain_release_pin", ref
    if ref:
        before = [
            r
            for r in releases
            if r["commit"] == ref or is_ancestor(corpus_repo, r["commit"], ref)
        ]
        after = [r for r in releases if r not in before]
        if before:
            return (
                list(reversed(before)) + after,
                "latest_release_at_or_before_toolchain_corpus_ref",
                ref,
            )
        return after, "earliest_release_after_toolchain_corpus_ref", ref
    return list(releases), "no_toolchain_corpus_pin", None


def fix_time_corpus_match(
    corpus_repo: Path, corpus_ref: str | None, resolution: dict[str, Any]
) -> str:
    """Does the same provision row at the toolchain corpus ref carry the same body?"""

    if not corpus_ref:
        return "no_corpus_ref"
    if corpus_ref == resolution["corpus_commit"]:
        return "same_commit"
    raw = lib.git_blob(corpus_repo, corpus_ref, resolution["provision_file"])
    if raw is None:
        return "provision_file_absent_at_corpus_ref"
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("citation_path") != resolution["resolved_citation_path"]:
            continue
        body = record.get("body")
        if not isinstance(body, str) or not body.strip():
            return "row_body_empty_at_corpus_ref"
        if lib.sha256_text(body) == resolution["stored_body_sha256"]:
            return "same_body"
        return "different_body"
    return "row_absent_at_corpus_ref"


def citation_paths(module_text: str) -> list[str]:
    try:
        doc = yaml.safe_load(module_text)
    except yaml.YAMLError:
        return []
    if not isinstance(doc, dict):
        return []
    module = doc.get("module") or {}
    verification = module.get("source_verification") or {}
    paths: list[str] = []
    single = verification.get("corpus_citation_path")
    if isinstance(single, str):
        paths.append(single)
    plural = verification.get("corpus_citation_paths")
    if isinstance(plural, list):
        paths.extend(p for p in plural if isinstance(p, str))
    return paths


def hunk_ranges(repo: Path, parent: str, commit: str, path: str) -> dict[str, list]:
    diff = lib.git(repo, "diff", "--no-color", "-U0", parent, commit, "--", path)
    pre: list[list[int]] = []
    post: list[list[int]] = []
    for match in re.finditer(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", diff, flags=re.M
    ):
        a, b, c, d = match.groups()
        b = int(b) if b is not None else 1
        d = int(d) if d is not None else 1
        if b:
            pre.append([int(a), int(a) + b - 1])
        if d:
            post.append([int(c), int(c) + d - 1])
    return {"pre_fix_lines": pre, "post_fix_lines": post}


def fix_stage(repo: Path, parent: str, path: str) -> str:
    """post_merge when the pre-fix blob was ever on the first-parent history of main.

    Otherwise ``pre_merge_review``: the correction landed on the branch before
    its pull request merged. The caller has already read the pre-fix blob, so
    a blob that cannot be named here is an error, not a third stage.
    """

    pre_blob = lib.git(repo, "rev-parse", f"{parent}:{path}").strip()
    history = lib.git(
        repo,
        "log",
        "--first-parent",
        f"-n{FIRST_PARENT_PROBE}",
        "--format=%H",
        MAIN_REF,
        "--",
        path,
    ).split()
    for candidate in history:
        try:
            blob = lib.git(repo, "rev-parse", f"{candidate}:{path}").strip()
        except subprocess.CalledProcessError:
            continue
        if blob == pre_blob:
            return "post_merge"
    return "pre_merge_review"


ARTIFACT_FILES = ("pre_fix.yaml", "post_fix.yaml", "provision.txt")
PROVISION_FILE = "provision.txt"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any, indent: int = 2) -> None:
    path.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------
# Case ids
# --------------------------------------------------------------------------


def row_key(jurisdiction: str, commit: str, module_path: str) -> tuple[str, str, str]:
    return (jurisdiction, commit[:10], module_path)


class CaseIds:
    """The registry of case ids (``triage/case_ids.json``).

    An id is ``<jurisdiction>-<seq>-<commit8>-<module slug>``. The registry
    maps each built row to its sequence number, so ids do not depend on the
    order rows are built in, on which cases a build filters out, or on rows
    added later: a row the registry has not seen takes the next free number
    of its jurisdiction.
    """

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.entries = list(entries)
        self._by_key = {
            row_key(e["jurisdiction"], e["commit"], e["module_path"]): e
            for e in self.entries
        }
        self._next = {"us": 1, "uk": 1}
        for entry in self.entries:
            jurisdiction = entry["jurisdiction"]
            self._next[jurisdiction] = max(
                self._next.get(jurisdiction, 1), entry["seq"] + 1
            )
        if len(self._by_key) != len(self.entries):
            raise ValueError("case id registry names a row twice")
        if len({e["id"] for e in self.entries}) != len(self.entries):
            raise ValueError("case id registry repeats an id")

    @classmethod
    def load(cls, path: Path) -> CaseIds:
        return cls(read_json(path)["ids"] if path.exists() else [])

    def id_for(self, case: dict[str, Any]) -> str:
        key = row_key(case["jurisdiction"], case["commit"], case["module_path"])
        entry = self._by_key.get(key)
        if entry is None:
            jurisdiction = case["jurisdiction"]
            seq = self._next.get(jurisdiction, 1)
            self._next[jurisdiction] = seq + 1
            entry = {
                "id": f"{jurisdiction}-{seq:03d}-{case['commit'][:8]}-"
                f"{slugify(case['module_path'])}",
                "jurisdiction": jurisdiction,
                "seq": seq,
                "commit": case["commit"],
                "module_path": case["module_path"],
            }
            self.entries.append(entry)
            self._by_key[key] = entry
        return entry["id"]

    def save(self, path: Path) -> None:
        ordered = sorted(self.entries, key=lambda e: (e["jurisdiction"], e["seq"]))
        write_json(
            path,
            {
                "about": (
                    "The case id of every triage row a build has turned into a "
                    "case, including cases a later build filtered out. "
                    "tools/build_real_defects.py reads it before filtering and "
                    "appends rows it has not seen."
                ),
                "ids": ordered,
            },
            indent=1,
        )


# --------------------------------------------------------------------------
# Families
# --------------------------------------------------------------------------


def _directly_verified(case: dict[str, Any]) -> bool:
    return case["triage"].get("verifier_inferred_from_module_index") is None


def _own_kind(case: dict[str, Any]) -> str:
    """The kind the case's own readers gave it, before any family settlement."""

    return case["triage"].get("kind_before_family_settlement") or case["defect_kind"]


def settle_family_kind(members: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    """One defect kind for a family, and the votes behind it.

    The kind most of the directly verified members carry (every member, when
    none was verified directly). A tie goes to the kind of the tied member
    with the first module path. ``members`` is sorted by module path.
    """

    voters = [c for c in members if _directly_verified(c)] or members
    votes = Counter(_own_kind(c) for c in voters)
    top = max(votes.values())
    tied = {kind for kind, count in votes.items() if count == top}
    kind = next(_own_kind(c) for c in voters if _own_kind(c) in tied)
    return kind, dict(sorted(votes.items()))


def assign_families(cases: list[dict[str, Any]]) -> None:
    """Group cases that carry one correction applied to many modules.

    A family is one commit and one rule path: the generator regenerations
    touch 100 chapter compositions with the same hunk. Readers labelled some
    members of one family with different kinds, so the family settles on one
    (:func:`settle_family_kind`); a member whose own label differed keeps it
    in ``triage.kind_before_family_settlement``. The representative is the
    first module path among the ``fidelity`` members, or the first of all
    when none is. ``family_size`` counts every member built, whatever a
    later filter removes.
    """

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for case in cases:
        key = (
            case["jurisdiction"],
            case["commit"][:8],
            case["locator"].get("rule_path") or "",
        )
        groups.setdefault(key, []).append(case)
    for (jurisdiction, commit8, rule_path), members in groups.items():
        members.sort(key=lambda c: c["module_path"])
        kind, votes = settle_family_kind(members)
        # The id folds in the settled kind as well as the commit and rule
        # path; a family whose members agreed keeps the id it had before
        # families were keyed on commit and rule path alone.
        digest = lib.sha256_text(f"{jurisdiction}-{commit8}-{kind}-{rule_path}")[:12]
        family_id = f"{jurisdiction}-{commit8}-{digest}"
        fidelity = [c for c in members if c["triage_status"] == "fidelity"]
        representative = (fidelity or members)[0]
        for case in members:
            original = _own_kind(case)
            case["triage"]["kind_before_family_settlement"] = (
                original if original != kind else None
            )
            case["triage"]["family_kind_votes"] = votes if len(votes) > 1 else None
            if case["defect_kind"] != kind:
                case["defect_kind"] = kind
                if kind != "other":
                    case["other_kind"] = None
            case["family_id"] = family_id
            case["family_size"] = len(members)
            case["family_representative"] = case is representative


def filter_cases(
    built: list[dict[str, Any]], *, drop_unclear: bool, family_members: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the selection flags; return ``(kept, removed log rows)``."""

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in built:
        case = item["case"]
        reason = None
        if drop_unclear and case["triage_status"] == "unclear":
            reason = REMOVED_UNCLEAR
        elif family_members == "representatives" and not case["family_representative"]:
            reason = REMOVED_NON_REPRESENTATIVE
        if reason is None:
            kept.append(item)
            continue
        removed.append(
            {
                "id": case["id"],
                "jurisdiction": case["jurisdiction"],
                "commit": case["commit"],
                "module_path": case["module_path"],
                "reason": reason,
                "family_id": case["family_id"],
            }
        )
    return kept, removed


def apply_shipping_policy(built: list[dict[str, Any]], policy: str) -> None:
    """Decide which cases keep pre_fix.yaml, post_fix.yaml and provision.txt.

    ``all`` keeps every case's files. ``verified`` (the default) keeps them for
    every case a verifier read directly and drops them for cases whose
    verdict was inherited from a sibling (the generator-family members), which
    stay as metadata-only records reproducible from Git and the release
    object. ``case.json`` records the outcome under ``artifacts_shipped``.
    """

    for item in built:
        keep = policy == "all" or _directly_verified(item["case"])
        item["case"]["artifacts_shipped"] = keep
        if not keep:
            item["files"] = None


# --------------------------------------------------------------------------
# Provision review and evidence
# --------------------------------------------------------------------------


class ReleaseResolver:
    """Resolve further citations from a case's own signed release."""

    def __init__(
        self,
        corpus_repo: Path | None,
        release_cache: Path | None,
        roots_dir: Path | None,
    ) -> None:
        self.corpus_repo = corpus_repo
        self.release_cache = release_cache
        self.roots_dir = roots_dir
        self._payloads: dict[str, dict[str, Any]] = {}
        self._resolved: dict[tuple[str, str], dict[str, Any]] = {}

    def resolve(self, case: dict[str, Any], citation: str) -> dict[str, Any]:
        if self.corpus_repo is None or self.release_cache is None:
            raise RuntimeError(
                f"{case['id']}: extending the provision with {citation} needs "
                "--axiom-corpus and --release-cache"
            )
        release = case["corpus_release"]
        key = (release, citation)
        if key not in self._resolved:
            if release not in self._payloads:
                self._payloads[release] = lib.fetch_release_object(
                    release, case["corpus_release_content_sha256"], self.release_cache
                )
            roots = self.roots_dir or self.release_cache / "roots"
            self._resolved[key] = lib.resolve_provision(
                self._payloads[release], self.corpus_repo, roots, citation
            )
        return dict(self._resolved[key])


def first_citation_text(case: dict[str, Any], provision_text: str) -> str:
    """The first citation's own text, whether or not the provision is extended."""

    extension = case.get("provision_extension")
    if not extension:
        return provision_text
    components = lib.provision_components(case)
    parts = lib.split_provision(
        provision_text, [(c["citation_path"], c["chars"]) for c in components]
    )
    if parts is None:
        raise RuntimeError(f"{case['id']}: provision.txt is not its composition")
    return parts[0]


def _default_window() -> tuple[int, Any]:
    from axiom_encode.judges.client import DEFAULT_PROVISION_CHARS, truncate_provision

    return DEFAULT_PROVISION_CHARS, truncate_provision


def apply_provision_review(
    item: dict[str, Any],
    record: dict[str, Any] | None,
    review_meta: dict[str, Any],
    resolver: ReleaseResolver,
) -> None:
    """Write the review outcome into a case, extending its provision if needed.

    ``record`` is the case's entry in ``triage/provision_review.json`` (None
    when the case was not reviewed). The provision is reset to the first
    citation's text first, so applying a changed record is the same as
    applying it to a fresh build.
    """

    case, files = item["case"], item["files"]
    if files is None:
        if record is not None:
            raise RuntimeError(f"{case['id']}: reviewed, but ships no artifacts")
        case["provision_extension"] = None
        case["judgeable_from_provision"] = None
        case["provision_review"] = None
        return
    text = first_citation_text(case, files[PROVISION_FILE].decode("utf-8"))
    case["provision_extension"] = None
    case["provision_sha256"] = lib.sha256_text(text)
    case["provision_chars"] = len(text)
    files[PROVISION_FILE] = text.encode("utf-8")
    if record is None:
        case["judgeable_from_provision"] = None
        case["provision_review"] = None
        return
    verdict = record["verdict"]
    pre = files["pre_fix.yaml"].decode("utf-8", errors="replace")
    post = files["post_fix.yaml"].decode("utf-8", errors="replace")
    mechanical, _check = evidence.check_case(case, pre, post, text)
    first = case["corpus_citation_path"]
    components = [(first, text)]
    if verdict == "in_other_citation":
        added: list[dict[str, Any]] = []
        for quote in record["decisive_quotes"]:
            citation = quote["citation_path"]
            if citation == first or any(a["citation_path"] == citation for a in added):
                continue
            resolution = resolver.resolve(case, citation)
            added_text = resolution.pop("text")
            added.append(
                {
                    "citation_path": citation,
                    "text_sha256": lib.sha256_text(added_text),
                    "chars": len(added_text),
                    "resolution": resolution,
                }
            )
            components.append((citation, added_text))
        if not added:
            raise RuntimeError(f"{case['id']}: in_other_citation names no other one")
        case["provision_extension"] = {
            "composition": lib.PROVISION_COMPOSITION,
            "first_citation_text_sha256": lib.sha256_text(text),
            "first_citation_chars": len(text),
            "added": added,
        }
        text = lib.compose_provision(components)
        case["provision_sha256"] = lib.sha256_text(text)
        case["provision_chars"] = len(text)
        files[PROVISION_FILE] = text.encode("utf-8")
    judgeable = verdict != "not_in_sources"
    # Each decisive quote is located inside the text of the citation it was
    # quoted from, so a passage that also occurs in an earlier component is
    # still recorded where the reader found it.
    regions: dict[str, tuple[int, str]] = {}
    position = 0
    for index, (citation, part) in enumerate(components):
        if len(components) > 1:
            position += len("\n\n" if index else "") + len(
                lib.provision_source_header(citation) + "\n"
            )
        regions[citation] = (position, part)
        position += len(part)
    quotes = []
    for quote in record["decisive_quotes"] if judgeable else []:
        citation = quote["citation_path"]
        if citation not in regions:
            raise RuntimeError(
                f"{case['id']}: quote cites {citation}, not in the provision"
            )
        offset, part = regions[citation]
        span = lib.locate_quote(quote["quote"], part)
        if span is None:
            raise RuntimeError(
                f"{case['id']}: decisive quote is not in {citation}: "
                f"{quote['quote'][:80]!r}"
            )
        quotes.append(
            {
                "citation_path": citation,
                "quote": quote["quote"],
                "span": [offset + span[0], offset + span[1]],
            }
        )
    window = None
    if judgeable:
        chars, truncate = _default_window()
        shown = truncate(text, chars)
        window = {
            "chars": chars,
            "decisive_text_visible": all(
                text[q["span"][0] : q["span"][1]] in shown for q in quotes
            ),
        }
    case["judgeable_from_provision"] = judgeable
    case["provision_review"] = {
        "method": review_meta["method"],
        "reviewed_on": review_meta["reviewed_on"],
        "verdict": verdict,
        "outcome": lib.PROVISION_REVIEW_OUTCOMES[verdict],
        "defect_real": record.get("defect_real"),
        "decisive_quotes": quotes,
        "nearest_quotes": record.get("nearest_quotes") or [],
        "why": record.get("why") or "",
        "missing_basis": record.get("missing_basis") or "",
        "pre_fix_restates_it": bool(record.get("pre_fix_restates_it")),
        "basis": record.get("basis"),
        "readers": record.get("readers") or {},
        "mechanical_before_review": mechanical,
        "default_window": window,
    }


def apply_evidence_check(item: dict[str, Any]) -> None:
    """Write ``evidence_in_provision`` and ``evidence_check`` into a case.

    Runs after the shipping policy and the provision review: a metadata-only
    case has no files to test and is ``unknown``, and an extended provision
    is tested as shipped (see ``tools/check_evidence.py``).
    """

    case, files = item["case"], item["files"]
    if files is None:
        status, check = evidence.check_case(case, None, None, None)
    else:
        status, check = evidence.check_case(
            case,
            files["pre_fix.yaml"].decode("utf-8", errors="replace"),
            files["post_fix.yaml"].decode("utf-8", errors="replace"),
            files[PROVISION_FILE].decode("utf-8"),
        )
    case["evidence_in_provision"] = status
    case["evidence_check"] = check


CASE_KEY_ORDER = (
    "id",
    "jurisdiction",
    "repo",
    "commit",
    "parent_commit",
    "commit_date",
    "commit_subject",
    "pr_url",
    "module_path",
    "corpus_citation_path",
    "corpus_citation_paths_all",
    "corpus_release",
    "corpus_release_content_sha256",
    "corpus_commit",
    "defect_kind",
    "other_kind",
    "confidence",
    "description",
    "description_source",
    "locator",
    "pre_fix_artifact_sha256",
    "post_fix_artifact_sha256",
    "provision_sha256",
    "provision_chars",
    "provision_resolution",
    "provision_extension",
    "evidence_in_provision",
    "evidence_check",
    "judgeable_from_provision",
    "provision_review",
    "fix_stage",
    "artifacts_shipped",
    "family_id",
    "family_size",
    "family_representative",
    "triage_status",
    "triage",
    "triage_notes",
)


def ordered_case(case: dict[str, Any]) -> dict[str, Any]:
    """The case with its keys in the documented order."""

    missing = [key for key in CASE_KEY_ORDER if key not in case]
    extra = [key for key in case if key not in CASE_KEY_ORDER]
    if missing or extra:
        raise RuntimeError(f"{case.get('id')}: keys missing {missing}, unknown {extra}")
    return {key: case[key] for key in CASE_KEY_ORDER}


def write_cases(cases_dir: Path, built: list[dict[str, Any]]) -> None:
    """Replace ``cases/`` with exactly the given cases."""

    if cases_dir.exists():
        shutil.rmtree(cases_dir)
    cases_dir.mkdir(parents=True)
    for item in built:
        case_dir = cases_dir / item["case"]["id"]
        case_dir.mkdir()
        for name, raw in (item["files"] or {}).items():
            (case_dir / name).write_bytes(raw)
        write_json(case_dir / "case.json", ordered_case(item["case"]))


def load_cases(corpus_dir: Path) -> list[dict[str, Any]]:
    """The cases on disk, as the build holds them (for ``--refresh-review``)."""

    built = []
    for case_path in sorted((corpus_dir / "cases").glob("*/case.json")):
        case = read_json(case_path)
        files = None
        if case.get("artifacts_shipped", True):
            files = {
                name: (case_path.parent / name).read_bytes() for name in ARTIFACT_FILES
            }
        built.append({"case": case, "files": files})
    return built


# --------------------------------------------------------------------------
# Index and README counts
# --------------------------------------------------------------------------


def _tally(cases: list[dict[str, Any]], key: str, values: Any) -> dict[str, int]:
    return {value: sum(1 for c in cases if c.get(key) == value) for value in values}


def is_board_eligible(case: dict[str, Any]) -> bool:
    """A shipped family representative with ``triage_status: fidelity``."""

    return bool(
        case["family_representative"]
        and case["triage_status"] == "fidelity"
        and case["artifacts_shipped"]
    )


def _window_visible(case: dict[str, Any]) -> bool | None:
    window = (case.get("provision_review") or {}).get("default_window")
    return None if window is None else window["decisive_text_visible"]


def build_index(
    cases: list[dict[str, Any]],
    *,
    shipping_policy: str,
    flags: dict[str, Any],
    dropped_at_build: int,
    removed: list[dict[str, Any]],
) -> dict[str, Any]:
    """The corpus index: per-case entries and every count, from the case records."""

    representatives = [c for c in cases if c["family_representative"]]
    eligible = [c for c in cases if is_board_eligible(c)]
    reviewed = [c for c in cases if c["provision_review"] is not None]
    judgeable = [c for c in eligible if c["judgeable_from_provision"] is True]
    verdicts = lib.PROVISION_REVIEW_VERDICTS
    return {
        "schema_version": "real_defects/v0",
        "shipping_policy": shipping_policy,
        "build_flags": flags,
        "generated_from": {
            "triage_record": "triage/triage_merged.json",
            "case_ids": "triage/case_ids.json",
            "provision_review": "triage/provision_review.json",
            "rulespec_main_ref": MAIN_REF,
        },
        "counts": {
            "cases": len(cases),
            "built_before_filters": len(cases) + len(removed),
            "removed_by_flag": dict(
                sorted(Counter(r["reason"] for r in removed).items())
            ),
            "by_jurisdiction": _tally(cases, "jurisdiction", ("us", "uk")),
            "by_kind": _tally(cases, "defect_kind", lib.DEFECT_KINDS),
            "by_triage_status": _tally(cases, "triage_status", lib.TRIAGE_STATUSES),
            "by_fix_stage": _tally(cases, "fix_stage", lib.FIX_STAGES),
            "dropped_at_build": dropped_at_build,
            "artifacts_shipped": sum(1 for c in cases if c["artifacts_shipped"]),
            "metadata_only": sum(1 for c in cases if not c["artifacts_shipped"]),
            "families": len({c["family_id"] for c in cases}),
            "family_representatives": len(representatives),
            "by_kind_representatives": _tally(
                representatives, "defect_kind", lib.DEFECT_KINDS
            ),
            "fidelity_representatives": len(eligible),
            "by_evidence_in_provision": _tally(
                cases, "evidence_in_provision", evidence.STATUSES
            ),
            "by_evidence_in_provision_fidelity_representatives": _tally(
                eligible, "evidence_in_provision", evidence.STATUSES
            ),
            "provision_reviewed": len(reviewed),
            "fidelity_representatives_not_reviewed": sum(
                1 for c in eligible if c["provision_review"] is None
            ),
            "by_provision_review_verdict": {
                verdict: sum(
                    1 for c in reviewed if c["provision_review"]["verdict"] == verdict
                )
                for verdict in verdicts
            },
            "by_provision_review_defect_real": dict(
                sorted(
                    Counter(
                        str(c["provision_review"]["defect_real"]) for c in reviewed
                    ).items()
                )
            ),
            "by_provision_review_basis": dict(
                sorted(
                    Counter(c["provision_review"]["basis"] for c in reviewed).items()
                )
            ),
            "mechanical_before_review_by_verdict": {
                status: {
                    verdict: sum(
                        1
                        for c in reviewed
                        if c["provision_review"]["mechanical_before_review"] == status
                        and c["provision_review"]["verdict"] == verdict
                    )
                    for verdict in verdicts
                }
                for status in evidence.STATUSES
            },
            "provision_extended": sum(1 for c in cases if c["provision_extension"]),
            "judgeable_fidelity_representatives": len(judgeable),
            "by_kind_judgeable": _tally(judgeable, "defect_kind", lib.DEFECT_KINDS),
            "judgeable_with_decisive_text_in_default_window": sum(
                1 for c in judgeable if _window_visible(c)
            ),
            "by_release_selection_basis": dict(
                sorted(
                    Counter(
                        c["provision_resolution"]["selection_basis"] for c in cases
                    ).items()
                )
            ),
            "provisions_over_default_window": sum(
                1
                for c in cases
                if c["artifacts_shipped"]
                and c["provision_chars"] > _default_window()[0]
            ),
        },
        "cases": [
            {
                "id": c["id"],
                "jurisdiction": c["jurisdiction"],
                "commit": c["commit"],
                "module_path": c["module_path"],
                "corpus_citation_path": c["corpus_citation_path"],
                "corpus_release": c["corpus_release"],
                "defect_kind": c["defect_kind"],
                "confidence": c["confidence"],
                "triage_status": c["triage_status"],
                "fix_stage": c["fix_stage"],
                "provision_chars": c["provision_chars"],
                "evidence_in_provision": c["evidence_in_provision"],
                "judgeable_from_provision": c["judgeable_from_provision"],
                "provision_review_verdict": (c["provision_review"] or {}).get(
                    "verdict"
                ),
                "provision_extended": bool(c["provision_extension"]),
                "decisive_text_in_default_window": _window_visible(c),
                "pr_url": c["pr_url"],
                "family_id": c["family_id"],
                "family_size": c["family_size"],
                "family_representative": c["family_representative"],
                "artifacts_shipped": c["artifacts_shipped"],
            }
            for c in cases
        ],
    }


def _counter_line(counts: dict[str, int]) -> str:
    return ", ".join(f"{key} {value}" for key, value in counts.items() if value)


def render_counts(
    index: dict[str, Any], summary: dict[str, Any], build_log: dict[str, Any]
) -> str:
    """The README's generated counts block, from the index and the triage logs."""

    counts = index["counts"]
    cases = index["cases"]
    flags = index["build_flags"]
    dropped = Counter(row["reason"] for row in build_log["dropped"])
    removed = counts["removed_by_flag"]
    lines = [
        COUNTS_BEGIN,
        "",
        f"Module changes classified by the triage readers: {summary['module_rows']:,} "
        f"({_counter_line(dict(sorted(summary['by_classification'].items())))}). "
        f"Verifier verdicts on the candidates: "
        f"{_counter_line(dict(sorted(summary['verifier_verdicts'].items())))}. "
        f"Dropped at merge: {_counter_line(dict(sorted(summary['dropped_by_reason'].items())))}. "
        f"Kept after verification: {summary['kept']}.",
        "",
        f"Dropped at build ({len(build_log['dropped'])} of {summary['kept']} kept rows): "
        f"{_counter_line(dict(sorted(dropped.items())))}. "
        f"Cases built: {counts['built_before_filters']}.",
        "",
        "Removed by the build flags "
        f"(`--drop-unclear`: {'on' if flags['drop_unclear'] else 'off'}; "
        f"`--family-members {flags['family_members']}`): "
        + (f"{_counter_line(removed)}. " if removed else "none. ")
        + f"Cases in the corpus: **{counts['cases']}** "
        f"({counts['by_jurisdiction']['us']} US, {counts['by_jurisdiction']['uk']} UK), "
        f"in {counts['families']} families.",
        "",
        "| Kind | US | UK | All | Judgeable from the provision |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for kind in lib.DEFECT_KINDS:
        us = sum(
            1 for c in cases if c["defect_kind"] == kind and c["jurisdiction"] == "us"
        )
        uk = sum(
            1 for c in cases if c["defect_kind"] == kind and c["jurisdiction"] == "uk"
        )
        lines.append(
            f"| `{kind}` | {us} | {uk} | {us + uk} | {counts['by_kind_judgeable'][kind]} |"
        )
    lines += [
        f"| Total | {counts['by_jurisdiction']['us']} | {counts['by_jurisdiction']['uk']} "
        f"| {counts['cases']} | {counts['judgeable_fidelity_representatives']} |",
        "",
        "How the corpus release was chosen (`selection_basis`): "
        f"{_counter_line(counts['by_release_selection_basis'])}.",
        "",
        f"Triage status: {_counter_line(counts['by_triage_status'])}. "
        f"Fix stage: {_counter_line(counts['by_fix_stage'])}. "
        f"Artifacts: {counts['artifacts_shipped']} cases ship their module and "
        f"provision files, {counts['metadata_only']} are metadata-only.",
        "",
        f"Provision review ({counts['provision_reviewed']} cases reviewed; "
        f"{counts['fidelity_representatives_not_reviewed']} board-eligible cases not "
        "reviewed):",
        "",
        "| Where the decisive text sits | Cases | Outcome |",
        "| --- | ---: | --- |",
        f"| In the packaged provision (`in_provision`) | "
        f"{counts['by_provision_review_verdict']['in_provision']} | kept as built |",
        f"| Under another citation (`in_other_citation`) | "
        f"{counts['by_provision_review_verdict']['in_other_citation']} | provision "
        "extended with that citation |",
        f"| In none of the cited sources (`not_in_sources`) | "
        f"{counts['by_provision_review_verdict']['not_in_sources']} | "
        "`judgeable_from_provision: false` |",
        "",
        f"Board-eligible cases (shipped family representatives with `triage_status: "
        f"fidelity`): {counts['fidelity_representatives']}. Judgeable from the "
        f"provision: {counts['judgeable_fidelity_representatives']}. Of those, "
        f"{counts['judgeable_with_decisive_text_in_default_window']} keep their decisive "
        f"text inside the judges' default {_default_window()[0]:,}-character window. "
        f"Provisions longer than that window: {counts['provisions_over_default_window']} "
        f"of {counts['artifacts_shipped']} shipped.",
        "",
        f"How the reviews were settled: {_counter_line(counts['by_provision_review_basis'])}. "
        "The readers' call on whether the bundle's sources confirm the defect "
        f"(`defect_real`): {_counter_line(counts['by_provision_review_defect_real'])}.",
        "",
        "The mechanical check (`tools/check_evidence.py`) against the review, on "
        "each reviewed case's first-citation provision:",
        "",
        "| Mechanical result | `in_provision` | `in_other_citation` | `not_in_sources` |",
        "| --- | ---: | ---: | ---: |",
    ]
    for status in evidence.STATUSES:
        row = counts["mechanical_before_review_by_verdict"][status]
        if any(row.values()):
            lines.append(
                f"| `{status}` | {row['in_provision']} | {row['in_other_citation']} "
                f"| {row['not_in_sources']} |"
            )
    lines += [
        "",
        "Mechanical `evidence_in_provision` on the provisions as shipped (after "
        f"extension): {_counter_line(counts['by_evidence_in_provision'])}.",
        "",
        COUNTS_END,
    ]
    return "\n".join(lines)


def splice_counts(readme: str, block: str) -> str:
    """``readme`` with its generated counts block replaced by ``block``."""

    start = readme.index(COUNTS_BEGIN)
    end = readme.index(COUNTS_END) + len(COUNTS_END)
    return readme[:start] + block + readme[end:]


def finalize(
    corpus_dir: Path,
    built: list[dict[str, Any]],
    *,
    shipping_policy: str,
    flags: dict[str, Any],
    dropped: list[dict[str, Any]],
    removed: list[dict[str, Any]],
    resolver: ReleaseResolver,
) -> dict[str, Any]:
    """Steps 4 and 5: review, evidence, cases on disk, index, log, README."""

    triage_dir = corpus_dir / "triage"
    review_path = triage_dir / "provision_review.json"
    review = read_json(review_path) if review_path.exists() else {"cases": {}}
    ids = {item["case"]["id"] for item in built}
    strangers = sorted(set(review["cases"]) - ids)
    if strangers:
        raise RuntimeError(
            f"provision review names cases not in the build: {strangers}"
        )
    built.sort(key=lambda item: item["case"]["id"])
    for item in built:
        apply_provision_review(
            item, review["cases"].get(item["case"]["id"]), review, resolver
        )
        apply_evidence_check(item)
    write_cases(corpus_dir / "cases", built)
    cases = [item["case"] for item in built]
    index = build_index(
        cases,
        shipping_policy=shipping_policy,
        flags=flags,
        dropped_at_build=len(dropped),
        removed=removed,
    )
    write_json(corpus_dir / "index.json", index)
    build_log = {
        "flags": flags,
        "built": len(cases) + len(removed),
        "kept": len(cases),
        "dropped": dropped,
        "removed": removed,
    }
    write_json(triage_dir / "build_log.json", build_log)
    readme_path = corpus_dir / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        block = render_counts(index, read_json(triage_dir / "summary.json"), build_log)
        readme_path.write_text(splice_counts(readme, block), encoding="utf-8")
    return index


def build_case(
    row: dict[str, Any],
    *,
    repos: dict[str, Path],
    corpus_repo: Path,
    releases_by_jur: dict[str, list[dict[str, Any]]],
    roots_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    jur = row["jurisdiction"]
    repo = repos[jur]
    if row.get("defect_kind") not in lib.DEFECT_KINDS:
        return None, None, "defect_kind_missing"
    commit = lib.git(repo, "rev-parse", row["commit"]).strip()
    parents = lib.git(repo, "rev-list", "--parents", "-n", "1", commit).split()[1:]
    if not parents:
        return None, None, "root_commit"
    parent = parents[0]
    if not is_ancestor(repo, commit, MAIN_REF):
        return None, None, "commit_not_on_main"
    path = row["module_path"]
    pre = lib.git_blob(repo, parent, path)
    post = lib.git_blob(repo, commit, path)
    if pre is None:
        return None, None, "module_added_in_fix_commit"
    if post is None:
        return None, None, "module_deleted_in_fix_commit"
    if pre == post:
        return None, None, "module_unchanged"
    pre_text = pre.decode("utf-8", errors="replace")
    post_text = post.decode("utf-8", errors="replace")
    if not post_text.startswith("format: rulespec/v1") or not pre_text.startswith(
        "format: rulespec/v1"
    ):
        return None, None, "not_rulespec_v1_on_both_sides"
    citations = citation_paths(post_text) or citation_paths(pre_text)
    if not citations:
        return None, None, "no_corpus_citation_path"
    ordered, basis, corpus_ref = select_releases(
        jur, repo, corpus_repo, commit, releases_by_jur[jur]
    )
    resolution = None
    chosen = None
    attempts = []
    for index, release in enumerate(ordered):
        try:
            resolution = lib.resolve_provision(
                release["payload"], corpus_repo, roots_dir, citations[0]
            )
        except Exception as exc:  # noqa: BLE001
            attempts.append(
                {
                    "release": release["name"],
                    "error": lib.portable_error(exc, roots_dir / release["name"]),
                }
            )
            continue
        chosen = dict(release)
        chosen["fallback_index"] = index
        break
    if resolution is None or chosen is None:
        return None, {"attempts": attempts[:5]}, "provision_unresolved"
    provision_text = resolution.pop("text")
    resolution["fix_time_corpus_match"] = fix_time_corpus_match(
        corpus_repo, corpus_ref, resolution
    )
    resolution["selection_basis"] = basis
    resolution["fallback_index"] = chosen["fallback_index"]
    resolution["toolchain_corpus_ref"] = corpus_ref
    resolution["requested_citation_path"] = citations[0]
    resolution["release_errors_before_success"] = attempts
    locator = hunk_ranges(repo, parent, commit, path)
    locator["rule_names"] = row.get("rule_names") or []
    locator["rule_path"] = row.get("rule_path") or ""
    verifier = row.get("verifier") or {}
    # Date and subject always come from the commit itself. The triage row's
    # own fields are not used: for screen-flagged commits the 2026-09 workflow
    # recorded an empty date and the screen's paraphrase as the subject.
    meta = lib.commit_metadata(repo, [commit])[commit]
    case = {
        "id": None,
        "jurisdiction": jur,
        "repo": lib.REPO_SLUGS[jur],
        "commit": commit,
        "parent_commit": parent,
        "commit_date": meta["date"],
        "commit_subject": meta["subject"],
        "pr_url": row.get("pr_url"),
        "module_path": path,
        "corpus_citation_path": citations[0],
        "corpus_citation_paths_all": citations,
        "corpus_release": chosen["name"],
        "corpus_release_content_sha256": chosen["content_sha256"],
        "corpus_commit": resolution["corpus_commit"],
        "defect_kind": row["defect_kind"],
        "other_kind": row.get("other_kind") or None,
        "confidence": round(float(row["confidence"]), 3),
        "description": row.get("description") or "",
        "description_source": row.get("description_source") or "none",
        "locator": locator,
        "pre_fix_artifact_sha256": lib.sha256_bytes(pre),
        "post_fix_artifact_sha256": lib.sha256_bytes(post),
        "provision_sha256": lib.sha256_text(provision_text),
        "provision_chars": len(provision_text),
        "provision_resolution": resolution,
        "fix_stage": fix_stage(repo, parent, path),
        "triage_status": row.get("triage_status"),
        "triage": {
            "classification": row.get("classification"),
            "pre_fix_wrong_because": row.get("pre_fix_wrong_because") or "",
            "triage_notes": row.get("triage_notes") or "",
            "verifier_verdict": verifier.get("verdict"),
            "verifier_defect_kind": verifier.get("defect_kind"),
            "verifier_confidence": verifier.get("confidence"),
            "verifier_justification": verifier.get("justification") or "",
            "verifier_quote": verifier.get("quote") or "",
            "verifier_notes": verifier.get("notes") or "",
            "verifier_inferred_from_module_index": verifier.get(
                "inferred_from_module_index"
            ),
            "candidate_source": row.get("candidate_source"),
            "screen_reason": row.get("screen_reason"),
            "review_override": row.get("review_override"),
        },
        "triage_notes": row.get("triage_notes_combined")
        or row.get("triage_notes")
        or "",
    }
    files = {
        "pre_fix.yaml": pre,
        "post_fix.yaml": post,
        "provision.txt": provision_text.encode("utf-8"),
    }
    return case, files, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--rulespec-us", type=Path)
    parser.add_argument("--rulespec-uk", type=Path)
    parser.add_argument("--axiom-corpus", type=Path)
    parser.add_argument("--release-cache", type=Path)
    parser.add_argument("--roots-dir", type=Path)
    parser.add_argument(
        "--ship-artifacts",
        choices=["verified", "all"],
        default="verified",
        help="which cases keep pre_fix/post_fix/provision files on disk",
    )
    parser.add_argument(
        "--drop-unclear",
        action="store_true",
        help="leave out cases with triage_status unclear (logged as triage_unclear)",
    )
    parser.add_argument(
        "--family-members",
        choices=FAMILY_MEMBER_CHOICES,
        default="all",
        help="representatives: leave out every family member but the "
        "representative (logged as family_non_representative)",
    )
    parser.add_argument(
        "--refresh-review",
        action="store_true",
        help="skip the build; re-apply the provision review, the evidence check, "
        "the index and the README counts to the cases on disk",
    )
    args = parser.parse_args(argv)
    corpus_dir = args.corpus_dir
    triage_dir = corpus_dir / "triage"
    resolver = ReleaseResolver(args.axiom_corpus, args.release_cache, args.roots_dir)
    if args.refresh_review:
        previous = read_json(corpus_dir / "index.json")
        log = read_json(triage_dir / "build_log.json")
        index = finalize(
            corpus_dir,
            load_cases(corpus_dir),
            shipping_policy=previous["shipping_policy"],
            flags=previous["build_flags"],
            dropped=log["dropped"],
            removed=log["removed"],
            resolver=resolver,
        )
        print(json.dumps(index["counts"], indent=1))
        return 0
    for name in ("rulespec_us", "rulespec_uk", "axiom_corpus", "release_cache"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for a build")
    repos = {"us": args.rulespec_us, "uk": args.rulespec_uk}
    roots_dir = args.roots_dir or args.release_cache / "roots"
    resolver.roots_dir = roots_dir
    rows = read_json(triage_dir / "triage_merged.json")
    releases_by_jur = {
        jur: list_registry_releases(jur, args.release_cache) for jur in ("us", "uk")
    }
    kept_rows = [r for r in rows if r.get("keep")]
    kept_rows.sort(
        key=lambda r: (r["jurisdiction"], r["date"], r["commit"], r["module_path"])
    )
    case_ids = CaseIds.load(triage_dir / "case_ids.json")
    built: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in kept_rows:
        case, files, reason = build_case(
            row,
            repos=repos,
            corpus_repo=args.axiom_corpus,
            releases_by_jur=releases_by_jur,
            roots_dir=roots_dir,
        )
        if case is None or files is None:
            dropped.append(
                {
                    "jurisdiction": row["jurisdiction"],
                    "commit": row["commit"],
                    "module_path": row["module_path"],
                    "reason": reason,
                    "detail": files,
                }
            )
            print(
                f"drop {row['jurisdiction']} {row['commit'][:10]} {row['module_path']}: {reason}",
                flush=True,
            )
            continue
        # Ids come from the registry before anything is filtered out.
        case["id"] = case_ids.id_for(case)
        built.append({"case": case, "files": files})
        print(
            f"case {case['id']} kind={case['defect_kind']} conf={case['confidence']}",
            flush=True,
        )
    assign_families([item["case"] for item in built])
    flags = {"drop_unclear": args.drop_unclear, "family_members": args.family_members}
    built, removed = filter_cases(
        built, drop_unclear=args.drop_unclear, family_members=args.family_members
    )
    apply_shipping_policy(built, args.ship_artifacts)
    index = finalize(
        corpus_dir,
        built,
        shipping_policy=args.ship_artifacts,
        flags=flags,
        dropped=dropped,
        removed=removed,
        resolver=resolver,
    )
    case_ids.save(triage_dir / "case_ids.json")
    print(
        f"built {index['counts']['built_before_filters']} cases, kept "
        f"{index['counts']['cases']}, dropped {len(dropped)} rows at build"
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
