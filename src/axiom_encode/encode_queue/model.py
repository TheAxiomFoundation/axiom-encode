"""State machine of the central encode queue (pure functions, no I/O).

Three documents hold the queue:

``packages.json``
    The priority list people maintain: one entry per (program, jurisdiction)
    package with a rank and a status.
``units/<package>.json``
    The derived membership of one package. Only ``build`` writes it.
``ledger.json``
    One entry per unit that is not simply ``ready``: the lease, the owner and
    the outcome.

A unit with no ledger entry is ``ready``. The state of a unit is always read
through :func:`effective_state`, so an expired lease returns the unit to
``ready`` with no sweeper process.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

PACKAGES_SCHEMA = "axiom-encode/central-queue/packages/v1"
UNITS_SCHEMA = "axiom-encode/central-queue/units/v1"
LEDGER_SCHEMA = "axiom-encode/central-queue/ledger/v1"

PACKAGES_DOC = "packages.json"
LEDGER_DOC = "ledger.json"

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
# States a worker reports while it holds the lease of the unit.
LEASE_OUTCOMES = (CANDIDATE, REFUSED, BLOCKED)
# States after the encode. No lease: the candidate exists outside the worker.
LANDING_STATES = (CANDIDATE, SIGNING, IN_TRAIN)

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
# Worker clocks differ. A lease is taken from its owner only this long after
# it expired.
LEASE_GRACE_SECONDS = 120
# A refused unit is ready again under another encoder version, this many
# times in all. After that it is blocked until a person requeues it.
MAX_ATTEMPTS = 3
MAX_CLAIM_COUNT = 40

_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,62}")
_OWNER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,62}")
_CATEGORY_RE = re.compile(r"[a-z0-9][a-z0-9_.:/-]{0,119}")


class QueueError(Exception):
    """A queue request that is refused. The queue is unchanged."""


def utc_now() -> datetime:
    return datetime.now(UTC)


def format_time(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def package_id(program: str, jurisdiction: str) -> str:
    identifier = f"{program}-{jurisdiction}".lower()
    if not _ID_RE.fullmatch(identifier):
        raise QueueError(f"invalid package id: {identifier!r}")
    return identifier


def units_doc(package: str) -> str:
    if not _ID_RE.fullmatch(package):
        raise QueueError(f"invalid package id: {package!r}")
    return f"units/{package}.json"


def check_owner(machine: str, person: str) -> dict[str, str]:
    for label, value in (("machine", machine), ("person", person)):
        if not isinstance(value, str) or not _OWNER_RE.fullmatch(value):
            raise QueueError(
                f"{label} must be 1-63 characters of letters, digits, '.', '_' or '-'"
            )
    return {"machine": machine, "person": person}


def check_category(category: str | None) -> str:
    if not category or not _CATEGORY_RE.fullmatch(category):
        raise QueueError(
            "a refused unit needs a category such as "
            "'validation/tests-coverage' (lower case, no spaces)"
        )
    return category


def check_lease_seconds(seconds: int) -> int:
    if not MIN_LEASE_SECONDS <= seconds <= MAX_LEASE_SECONDS:
        raise QueueError(
            f"lease must be {MIN_LEASE_SECONDS} to {MAX_LEASE_SECONDS} seconds"
        )
    return seconds


def empty_packages() -> dict[str, Any]:
    return {"schema": PACKAGES_SCHEMA, "packages": []}


def empty_ledger() -> dict[str, Any]:
    return {"schema": LEDGER_SCHEMA, "units": {}}


def check_schema(document: dict[str, Any], schema: str, name: str) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema") != schema:
        raise QueueError(f"{name} does not have schema {schema}")
    return document


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
    ``encoder_version`` is the version of the worker that asks: a unit
    refused under another version is ready for it again.
    """

    if entry is None:
        return CANDIDATE if open_pr else READY
    state = entry.get("state")
    if state in ACTIVE_LEASE_STATES:
        if lease_is_live(entry, now):
            return state
        return CANDIDATE if open_pr else READY
    if state == REFUSED:
        if int(entry.get("attempts", 1)) >= MAX_ATTEMPTS:
            return BLOCKED
        if encoder_version and entry.get("encoder_version") != encoder_version:
            return CANDIDATE if open_pr else READY
        return REFUSED
    if state in STATES:
        return state
    raise QueueError(f"ledger entry has unknown state {state!r}")


def blocked_reason(entry: dict[str, Any] | None, now: datetime) -> str:
    """What unblocks a unit that :func:`effective_state` calls blocked."""

    if entry is None:
        return ""
    if entry.get("state") == REFUSED:
        return (
            f"refused {entry.get('attempts')} times "
            f"(last category {entry.get('category')}); a person must requeue it"
        )
    return str(entry.get("unblock") or "")


def check_transition(current: str, target: str) -> None:
    if target not in TRANSITIONS.get(current, ()):
        raise QueueError(f"a unit cannot move from {current} to {target}")
