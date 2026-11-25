"""State machine and document schemas of the central encode queue.

Pure functions, no I/O. Three kinds of document hold the queue:

``packages.json``
    The priority list people maintain: one entry per (program, jurisdiction)
    package with a rank and a status.
``units/<package>.json``
    The derived membership of one package. Only ``build`` writes it.
``ledger/<package>.json``
    One entry per unit of the package that is not simply ``ready``: the
    lease, the owner and the outcome. One shard per package, so a write
    carries only the shard it changed.

A unit with no ledger entry is ``ready``. The state of a unit is always read
through :func:`effective_state`, so an expired lease returns the unit to
``ready`` with no sweeper process.

Every document is validated when it is read (:func:`validate_packages`,
:func:`validate_units`, :func:`validate_ledger`). The queue branch can be
edited by anyone with write access, so no stored value is trusted: unknown
fields are refused, citations must match the citation grammar, and the
RuleSpec target path is never stored. :func:`target_path` recomputes it from
the citation each time.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

PACKAGES_SCHEMA = "axiom-encode/central-queue/packages/v1"
UNITS_SCHEMA = "axiom-encode/central-queue/units/v2"
LEDGER_SCHEMA = "axiom-encode/central-queue/ledger/v2"

PACKAGES_DOC = "packages.json"

READY = "ready"
RESERVED = "reserved"
ENCODING = "encoding"
CANDIDATE = "candidate"
SIGNING = "signing"
IN_TRAIN = "in_train"
LANDED = "landed"
REFUSED = "refused"
BLOCKED = "blocked"

STATES = (
    READY,
    RESERVED,
    ENCODING,
    CANDIDATE,
    SIGNING,
    IN_TRAIN,
    LANDED,
    REFUSED,
    BLOCKED,
)
# States that hold a lease: the owner must heartbeat or the unit returns.
ACTIVE_LEASE_STATES = (RESERVED, ENCODING)

# from-state -> states a report may move the unit to.
TRANSITIONS: dict[str, tuple[str, ...]] = {
    READY: (BLOCKED,),
    RESERVED: (ENCODING, CANDIDATE, REFUSED, BLOCKED, READY),
    ENCODING: (CANDIDATE, REFUSED, BLOCKED, READY),
    CANDIDATE: (SIGNING, IN_TRAIN, LANDED, REFUSED, BLOCKED, READY),
    SIGNING: (IN_TRAIN, LANDED, CANDIDATE, REFUSED, BLOCKED, READY),
    IN_TRAIN: (LANDED, CANDIDATE, REFUSED, BLOCKED, READY),
    REFUSED: (READY, BLOCKED),
    BLOCKED: (READY,),
    LANDED: (),
}

PACKAGE_ACTIVE = "active"
PACKAGE_PAUSED = "paused"
PACKAGE_BLOCKED = "blocked"
PACKAGE_STATUSES = (PACKAGE_ACTIVE, PACKAGE_PAUSED, PACKAGE_BLOCKED)

DEFAULT_LEASE_SECONDS = 30 * 60
MIN_LEASE_SECONDS = 60
MAX_LEASE_SECONDS = 6 * 60 * 60
# The lease clock is the server's (the store gives the time). The grace
# covers the delay between a decision and its commit.
LEASE_GRACE_SECONDS = 120
# A refused unit is ready again under another encoder version, this many
# times in all. After that it is blocked until a person requeues it.
MAX_ATTEMPTS = 3
MAX_CLAIM_COUNT = 40
MAX_TEXT = 500
MAX_CITATION = 300

_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,62}")
_OWNER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,62}")
_CATEGORY_RE = re.compile(r"[a-z0-9][a-z0-9_.:/-]{0,119}")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,39}")
_REQUEST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{7,79}")
_LEASE_RE = re.compile(r"([a-z0-9][a-z0-9._-]{0,62}):[0-9a-f]{32}")
_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,99}")
_JURISDICTION_RE = re.compile(r"[a-z]{2}(-[a-z0-9]{2,8})?")
# The characters that corpus citation paths use today. No control
# character, quote, backslash or shell metacharacter.
_SEGMENT_RE = re.compile(r"[A-Za-z0-9(][A-Za-z0-9 ._:()–-]*")


class QueueError(Exception):
    """A queue request that is refused. The queue is unchanged."""


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def format_time(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text: Any) -> datetime:
    if not isinstance(text, str):
        raise QueueError(f"invalid time {text!r}")
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        raise QueueError(f"invalid time {text!r}") from None


# ------------------------------------------------------------------ identifiers


def check_package_id(identifier: Any) -> str:
    if not isinstance(identifier, str) or not _ID_RE.fullmatch(identifier):
        raise QueueError(f"invalid package id: {identifier!r}")
    return identifier


def package_id(program: str, jurisdiction: str) -> str:
    if not isinstance(program, str) or not isinstance(jurisdiction, str):
        raise QueueError("program and jurisdiction must be text")
    return check_package_id(f"{program}-{jurisdiction}".lower())


def units_doc(package: str) -> str:
    return f"units/{check_package_id(package)}.json"


def ledger_doc(package: str) -> str:
    return f"ledger/{check_package_id(package)}.json"


def new_lease_id(package: str, token: str) -> str:
    return f"{check_package_id(package)}:{token}"


def lease_package(lease_id: Any) -> str:
    """The package of a lease id (``<package>:<32 hex>``)."""

    match = _LEASE_RE.fullmatch(lease_id) if isinstance(lease_id, str) else None
    if not match:
        raise QueueError(f"invalid lease id: {lease_id!r}")
    return match.group(1)


def check_owner(machine: Any, person: Any) -> dict[str, str]:
    for label, value in (("machine", machine), ("person", person)):
        if not isinstance(value, str) or not _OWNER_RE.fullmatch(value):
            raise QueueError(
                f"{label} must be 1-63 characters of letters, digits, '.', '_' or '-'"
            )
    return {"machine": machine, "person": person}


def check_person(person: Any) -> str:
    return check_owner("any", person)["person"]


def check_category(category: Any) -> str:
    if not isinstance(category, str) or not _CATEGORY_RE.fullmatch(category):
        raise QueueError(
            "a refused unit needs a category such as "
            "'validation/tests-coverage' (lower case, no spaces)"
        )
    return category


def check_version(version: Any) -> str:
    if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
        raise QueueError(f"invalid encoder version: {version!r}")
    return version


def check_request_id(request_id: Any) -> str:
    if not isinstance(request_id, str) or not _REQUEST_RE.fullmatch(request_id):
        raise QueueError(
            "request id must be 8-80 characters of letters, digits, '.', '_' or '-'"
        )
    return request_id


def check_lease_seconds(seconds: Any) -> int:
    if (
        not isinstance(seconds, int)
        or isinstance(seconds, bool)
        or not MIN_LEASE_SECONDS <= seconds <= MAX_LEASE_SECONDS
    ):
        raise QueueError(
            f"lease must be {MIN_LEASE_SECONDS} to {MAX_LEASE_SECONDS} seconds"
        )
    return seconds


def check_text(value: Any, label: str, *, required: bool = False) -> str:
    """Free text of bounded length on one line set (no control characters)."""

    if value is None:
        value = ""
    if not isinstance(value, str):
        raise QueueError(f"{label} must be text")
    value = value.strip()
    if len(value) > MAX_TEXT:
        raise QueueError(f"{label} is longer than {MAX_TEXT} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise QueueError(f"{label} has a control character")
    if required and not value:
        raise QueueError(f"{label} is required")
    return value


def check_citation(citation: Any) -> str:
    """A corpus citation path: ``<jurisdiction>/<kind>/<segment>...``."""

    if not isinstance(citation, str) or not 0 < len(citation) <= MAX_CITATION:
        raise QueueError(f"invalid citation: {citation!r:.80}")
    segments = citation.split("/")
    if (
        len(segments) < 3
        or not _JURISDICTION_RE.fullmatch(segments[0])
        or any(
            not _SEGMENT_RE.fullmatch(segment)
            or segment != segment.strip()
            or set(segment) <= {"."}
            for segment in segments[1:]
        )
    ):
        raise QueueError(f"invalid citation: {citation!r:.80}")
    return citation


def target_path(citation: str) -> str:
    """The RuleSpec path of a unit, computed from its citation.

    The path is never read from a queue document.
    """

    from axiom_encode.rulespec_citation_paths import citation_rulespec_path

    check_citation(citation)
    try:
        path = citation_rulespec_path(citation)
    except ValueError as error:
        raise QueueError(f"{citation} has no RuleSpec path: {error}") from None
    text = path.as_posix()
    if path.is_absolute() or ".." in path.parts or not text.endswith(".yaml"):
        raise QueueError(f"{citation} has no safe RuleSpec path")
    return text


# -------------------------------------------------------------------- documents


def _keys(
    value: Any,
    where: str,
    required: dict[str, type | tuple[type, ...]],
    optional: dict[str, type | tuple[type, ...]] | None = None,
) -> dict[str, Any]:
    """Refuse a mapping with a missing, unknown or wrongly typed field."""

    optional = optional or {}
    if not isinstance(value, dict):
        raise QueueError(f"{where} is not a mapping")
    unknown = sorted(set(value) - set(required) - set(optional))
    if unknown:
        raise QueueError(f"{where} has unknown fields {unknown}")
    missing = sorted(set(required) - set(value))
    if missing:
        raise QueueError(f"{where} lacks fields {missing}")
    for key, item in value.items():
        expected = required.get(key) or optional[key]
        if isinstance(item, bool) and expected is not bool:
            raise QueueError(f"{where}.{key} has the wrong type")
        if not isinstance(item, expected):
            raise QueueError(f"{where}.{key} has the wrong type")
    return value


def _count(value: Any, where: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise QueueError(f"{where} is not a count")
    return value


def empty_packages() -> dict[str, Any]:
    return {"schema": PACKAGES_SCHEMA, "packages": []}


def empty_ledger(package: str) -> dict[str, Any]:
    return {"schema": LEDGER_SCHEMA, "package": package, "units": {}}


def validate_packages(document: Any) -> dict[str, Any]:
    where = PACKAGES_DOC
    _keys(document, where, {"schema": str, "packages": list}, {"moved_to": str})
    if document["schema"] != PACKAGES_SCHEMA:
        raise QueueError(f"{where} does not have schema {PACKAGES_SCHEMA}")
    if "moved_to" in document and not _REF_RE.fullmatch(document["moved_to"]):
        raise QueueError(f"{where}.moved_to is not a branch name")
    seen: set[str] = set()
    ranks: set[int] = set()
    for index, entry in enumerate(document["packages"]):
        item = f"{where}.packages[{index}]"
        _keys(
            entry,
            item,
            {
                "id": str,
                "program": str,
                "jurisdiction": str,
                "rank": int,
                "status": str,
                "unblock": str,
                "note": str,
                "added_by": str,
                "added_at": str,
            },
            {"status_by": str, "status_at": str},
        )
        if package_id(entry["program"], entry["jurisdiction"]) != entry["id"]:
            raise QueueError(f"{item}.id does not match program and jurisdiction")
        if entry["id"] in seen or entry["rank"] in ranks or entry["rank"] < 1:
            raise QueueError(f"{item} repeats an id or a rank")
        seen.add(entry["id"])
        ranks.add(entry["rank"])
        if entry["status"] not in PACKAGE_STATUSES:
            raise QueueError(f"{item}.status is unknown")
        check_text(entry["unblock"], f"{item}.unblock")
        check_text(entry["note"], f"{item}.note")
        check_person(entry["added_by"])
        parse_time(entry["added_at"])
        if "status_by" in entry:
            check_person(entry["status_by"])
        if "status_at" in entry:
            parse_time(entry["status_at"])
    return document


def validate_units(document: Any, package: str) -> dict[str, Any]:
    where = units_doc(package)
    _keys(
        document,
        where,
        {
            "schema": str,
            "package": str,
            "built_at": str,
            "inputs": dict,
            "on_main": int,
            "units": list,
        },
        {"documents": dict, "blocked": str},
    )
    if document["schema"] != UNITS_SCHEMA or document["package"] != package:
        raise QueueError(f"{where} does not have schema {UNITS_SCHEMA} for {package}")
    parse_time(document["built_at"])
    _count(document["on_main"], f"{where}.on_main")
    _keys(
        document["inputs"],
        f"{where}.inputs",
        {},
        {
            "corpus_ref": str,
            "rulespec_ref": str,
            "bundle_as_of": str,
            "open_prs_listed": bool,
        },
    )
    for key in ("corpus_ref", "rulespec_ref", "bundle_as_of"):
        check_text(document["inputs"].get(key), f"{where}.inputs.{key}")
    if "documents" in document:
        counts = _keys(
            document["documents"],
            f"{where}.documents",
            {"in_scope": int, "held": int, "missing": list},
        )
        _count(counts["in_scope"], f"{where}.documents.in_scope")
        _count(counts["held"], f"{where}.documents.held")
        for key in counts["missing"]:
            check_text(key, f"{where}.documents.missing", required=True)
    if "blocked" in document:
        check_text(document["blocked"], f"{where}.blocked", required=True)
    seen: set[str] = set()
    ranks: set[int] = set()
    for index, unit in enumerate(document["units"]):
        item = f"{where}.units[{index}]"
        _keys(
            unit,
            item,
            {"citation": str, "rank": int},
            {
                "document": str,
                "part": str,
                "heading": str,
                "source_chars": int,
                "money_or_percent": bool,
                "open_pr": int,
            },
        )
        check_citation(unit["citation"])
        if unit["citation"] in seen or unit["rank"] in ranks or unit["rank"] < 1:
            raise QueueError(f"{item} repeats a citation or a rank")
        seen.add(unit["citation"])
        ranks.add(unit["rank"])
        if "document" in unit:
            check_citation(unit["document"])
        check_text(unit.get("part"), f"{item}.part")
        check_text(unit.get("heading"), f"{item}.heading")
        if "open_pr" in unit:
            _count(unit["open_pr"], f"{item}.open_pr", minimum=1)
        if "source_chars" in unit:
            _count(unit["source_chars"], f"{item}.source_chars")
    return document


_ENTRY_OPTIONAL: dict[str, type | tuple[type, ...]] = {
    "owner": dict,
    "lease_id": str,
    "request_id": str,
    "claimed_at": str,
    "heartbeat_at": str,
    "lease_expires_at": str,
    "lease_seconds": int,
    "encoder_version": str,
    "refused_versions": list,
    "closed_lease_id": str,
    "superseded": bool,
    "category": str,
    "unblock": str,
    "candidate_pr": int,
    "signing_run": int,
    "train_pr": int,
    "note": str,
    "reported_by": str,
}
_LEASE_FIELDS = (
    "owner",
    "lease_id",
    "request_id",
    "claimed_at",
    "heartbeat_at",
    "lease_expires_at",
    "lease_seconds",
    "encoder_version",
)


def validate_ledger(document: Any, package: str, now: datetime) -> dict[str, Any]:
    where = ledger_doc(package)
    _keys(document, where, {"schema": str, "package": str, "units": dict})
    if document["schema"] != LEDGER_SCHEMA or document["package"] != package:
        raise QueueError(f"{where} does not have schema {LEDGER_SCHEMA} for {package}")
    for citation, entry in document["units"].items():
        check_citation(citation)
        item = f"{where} entry {citation}"
        _keys(
            entry,
            item,
            {"state": str, "attempts": int, "updated_at": str},
            _ENTRY_OPTIONAL,
        )
        if entry["state"] not in STATES:
            raise QueueError(f"{item} has unknown state {entry['state']!r}")
        _count(entry["attempts"], f"{item}.attempts")
        parse_time(entry["updated_at"])
        for version in entry.get("refused_versions", []):
            check_version(version)
        if "owner" in entry:
            owner = _keys(
                entry["owner"], f"{item}.owner", {"machine": str, "person": str}
            )
            check_owner(owner["machine"], owner["person"])
        if "encoder_version" in entry:
            check_version(entry["encoder_version"])
        if "closed_lease_id" in entry and lease_package(entry["closed_lease_id"]) != (
            package
        ):
            raise QueueError(f"{item} has a lease id of another package")
        if "category" in entry:
            check_category(entry["category"])
        for key in ("unblock", "note"):
            check_text(entry.get(key), f"{item}.{key}")
        if "reported_by" in entry:
            check_person(entry["reported_by"])
        for key in ("candidate_pr", "signing_run", "train_pr"):
            if key in entry:
                _count(entry[key], f"{item}.{key}", minimum=1)
        if entry["state"] in ACTIVE_LEASE_STATES:
            missing = [key for key in _LEASE_FIELDS if key not in entry]
            if missing:
                raise QueueError(f"{item} is a lease without {missing}")
            if lease_package(entry["lease_id"]) != package:
                raise QueueError(f"{item} has a lease id of another package")
            check_request_id(entry["request_id"])
            seconds = check_lease_seconds(entry["lease_seconds"])
            claimed = parse_time(entry["claimed_at"])
            beat = parse_time(entry["heartbeat_at"])
            expires = parse_time(entry["lease_expires_at"])
            # A lease never ends later than one lease length after its last
            # heartbeat, and no heartbeat is from the future.
            if (
                expires > beat + timedelta(seconds=seconds)
                or beat > now + timedelta(seconds=LEASE_GRACE_SECONDS)
                or claimed > beat
            ):
                raise QueueError(f"{item} has an invalid lease time")
        if entry["state"] == REFUSED and "category" not in entry:
            raise QueueError(f"{item} is refused without a category")
        if entry["state"] == BLOCKED and not entry.get("unblock"):
            raise QueueError(f"{item} is blocked without what unblocks it")
    return document


# ------------------------------------------------------------------------ state


def lease_is_live(entry: dict[str, Any], now: datetime) -> bool:
    expires = entry.get("lease_expires_at")
    if not expires:
        return False
    return now <= parse_time(expires) + timedelta(seconds=LEASE_GRACE_SECONDS)


def effective_state(
    entry: dict[str, Any] | None,
    now: datetime,
    *,
    encoder_version: str | None = None,
    open_pr: int | None = None,
) -> str:
    """The state of one unit at ``now``.

    ``entry`` is the ledger entry or ``None``. ``open_pr`` is the open
    candidate pull request that the last build found for the unit.
    ``encoder_version`` is the version of the worker that asks. A unit is
    never ready for a version that it was refused under, also after later
    claims and releases by other versions.
    """

    if entry is None:
        return CANDIDATE if open_pr else READY
    state = entry["state"]
    if state in ACTIVE_LEASE_STATES and lease_is_live(entry, now):
        return state
    if state in ACTIVE_LEASE_STATES or state in (READY, REFUSED):
        if entry["attempts"] >= MAX_ATTEMPTS:
            return BLOCKED
        refused_versions = entry.get("refused_versions", [])
        if state == REFUSED and encoder_version is None:
            return REFUSED
        if encoder_version is not None and encoder_version in refused_versions:
            return REFUSED
        return CANDIDATE if open_pr else READY
    return state


def blocked_reason(entry: dict[str, Any] | None) -> str:
    """What unblocks a unit that :func:`effective_state` calls blocked."""

    if entry is None:
        return ""
    if entry["state"] != BLOCKED:
        return (
            f"refused {entry['attempts']} times "
            f"(last category {entry.get('category')}); a person must requeue it"
        )
    return str(entry.get("unblock") or "")


def check_transition(current: str, target: str) -> None:
    if target not in TRANSITIONS.get(current, ()):
        raise QueueError(f"a unit cannot move from {current} to {target}")
