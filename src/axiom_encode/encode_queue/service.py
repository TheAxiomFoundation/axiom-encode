"""Operations of the central encode queue over a :class:`QueueStore`.

Every write is ``read snapshot -> change documents -> commit at the read
revision``. A lost race raises ``RevisionConflict`` in the store; the
operation then reads again and decides again, so it never acts on stale
state.
"""

from __future__ import annotations

import random
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from axiom_encode.encode_queue import model
from axiom_encode.encode_queue.model import QueueError
from axiom_encode.encode_queue.store import (
    Document,
    QueueStore,
    RevisionConflict,
    Snapshot,
)

MAX_COMMIT_ATTEMPTS = 40

# What one operation decides on a snapshot: documents to write, a commit
# message and the value to return. ``None`` changes mean "nothing to write".
_Decision = tuple[dict[str, Document | None], str, Any]


class EncodeQueue:
    def __init__(
        self,
        store: QueueStore,
        *,
        clock: Callable[[], datetime] = model.utc_now,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._store = store
        self._clock = clock
        self._sleep = sleep

    # ------------------------------------------------------------------ core

    def _transact(self, decide: Callable[[Snapshot, datetime], _Decision]) -> Any:
        for attempt in range(MAX_COMMIT_ATTEMPTS):
            snapshot = self._store.snapshot()
            changes, message, result = decide(snapshot, self._clock())
            if not changes:
                return result
            try:
                self._store.commit(snapshot.revision, changes, message)
            except RevisionConflict:
                self._sleep(random.uniform(0.05, 0.5) * min(attempt + 1, 8))  # noqa: S311
                continue
            return result
        raise QueueError(
            f"the queue store stayed busy for {MAX_COMMIT_ATTEMPTS} attempts; try again"
        )

    @staticmethod
    def _packages(snapshot: Snapshot) -> dict[str, Any]:
        document = snapshot.read(model.PACKAGES_DOC)
        if document is None:
            return model.empty_packages()
        return model.check_schema(document, model.PACKAGES_SCHEMA, model.PACKAGES_DOC)

    @staticmethod
    def _ledger(snapshot: Snapshot) -> dict[str, Any]:
        document = snapshot.read(model.LEDGER_DOC)
        if document is None:
            return model.empty_ledger()
        return model.check_schema(document, model.LEDGER_SCHEMA, model.LEDGER_DOC)

    @staticmethod
    def _units(snapshot: Snapshot, package: str) -> dict[str, Any] | None:
        document = snapshot.read(model.units_doc(package))
        if document is None:
            return None
        return model.check_schema(document, model.UNITS_SCHEMA, f"units of {package}")

    @staticmethod
    def _ranked(packages: dict[str, Any]) -> list[dict[str, Any]]:
        return sorted(packages["packages"], key=lambda p: (p["rank"], p["id"]))

    def _find_unit(
        self, snapshot: Snapshot, citation: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        for package in self._ranked(self._packages(snapshot)):
            units = self._units(snapshot, package["id"])
            for unit in (units or {}).get("units", []):
                if unit["citation"] == citation:
                    return package, unit
        raise QueueError(f"{citation} is not a unit of the queue")

    # -------------------------------------------------------------- packages

    def add_package(
        self,
        program: str,
        jurisdiction: str,
        *,
        person: str,
        before: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Add a (program, jurisdiction) package to the priority list.

        The package goes to the end of the list, or in front of ``before``.
        It has no units until the next ``build``.
        """

        identifier = model.package_id(program, jurisdiction)
        model.check_owner("any", person)

        def decide(snapshot: Snapshot, now: datetime) -> _Decision:
            packages = self._packages(snapshot)
            ordered = self._ranked(packages)
            if any(p["id"] == identifier for p in ordered):
                raise QueueError(f"package {identifier} is already in the queue")
            entry = {
                "id": identifier,
                "program": program.lower(),
                "jurisdiction": jurisdiction.lower(),
                "status": model.PACKAGE_ACTIVE,
                "unblock": "",
                "note": note,
                "added_by": person,
                "added_at": model.format_time(now),
            }
            self._insert(ordered, entry, before)
            packages["packages"] = ordered
            return (
                {model.PACKAGES_DOC: packages},
                f"queue: add package {identifier} ({person})",
                entry,
            )

        return self._transact(decide)

    @staticmethod
    def _insert(
        ordered: list[dict[str, Any]], entry: dict[str, Any], before: str | None
    ) -> None:
        if before is None:
            ordered.append(entry)
        else:
            index = next((i for i, p in enumerate(ordered) if p["id"] == before), None)
            if index is None:
                raise QueueError(f"package {before} is not in the queue")
            ordered.insert(index, entry)
        for rank, package in enumerate(ordered, start=1):
            package["rank"] = rank

    def move_package(
        self, package: str, *, person: str, before: str | None = None
    ) -> list[str]:
        """Move a package in front of ``before``, or to the end. Returns the order."""

        model.check_owner("any", person)

        def decide(snapshot: Snapshot, now: datetime) -> _Decision:
            packages = self._packages(snapshot)
            ordered = self._ranked(packages)
            entry = next((p for p in ordered if p["id"] == package), None)
            if entry is None:
                raise QueueError(f"package {package} is not in the queue")
            if before == package:
                raise QueueError("a package cannot move in front of itself")
            ordered.remove(entry)
            self._insert(ordered, entry, before)
            packages["packages"] = ordered
            order = [p["id"] for p in ordered]
            return (
                {model.PACKAGES_DOC: packages},
                f"queue: move package {package} ({person})",
                order,
            )

        return self._transact(decide)

    def set_package_status(
        self, package: str, status: str, *, person: str, unblock: str = ""
    ) -> dict[str, Any]:
        """Set a package active, paused or blocked.

        A blocked package must say what unblocks it. Units of a package that
        is not active are never claimed; live leases run to their end.
        """

        model.check_owner("any", person)
        if status not in model.PACKAGE_STATUSES:
            raise QueueError(f"status must be one of {model.PACKAGE_STATUSES}")
        if status == model.PACKAGE_BLOCKED and not unblock.strip():
            raise QueueError("a blocked package must say what unblocks it")

        def decide(snapshot: Snapshot, now: datetime) -> _Decision:
            packages = self._packages(snapshot)
            entry = next((p for p in packages["packages"] if p["id"] == package), None)
            if entry is None:
                raise QueueError(f"package {package} is not in the queue")
            entry["status"] = status
            entry["unblock"] = unblock.strip() if status != model.PACKAGE_ACTIVE else ""
            entry["status_by"] = person
            entry["status_at"] = model.format_time(now)
            return (
                {model.PACKAGES_DOC: packages},
                f"queue: package {package} {status} ({person})",
                entry,
            )

        return self._transact(decide)

    # ----------------------------------------------------------------- build

    def apply_build(
        self, built: dict[str, dict[str, Any]], *, person: str
    ) -> dict[str, Any]:
        """Replace the derived membership with the result of a build.

        ``built`` maps a package id to its units document
        (:func:`axiom_encode.encode_queue.build.derive_package`). Packages
        that are not in ``built`` keep their units. A ledger entry whose unit
        left the membership (it landed on main, or it left the pinned corpus)
        is removed, so a landed unit disappears. A live lease is kept until
        it ends: the worker still gets an answer to its report.
        """

        model.check_owner("any", person)

        def decide(snapshot: Snapshot, now: datetime) -> _Decision:
            packages = self._packages(snapshot)
            known = {p["id"] for p in packages["packages"]}
            unknown = sorted(set(built) - known)
            if unknown:
                raise QueueError(f"build has packages that are not queued: {unknown}")
            ledger = self._ledger(snapshot)
            changes: dict[str, Document | None] = {}
            members: set[str] = set()
            for package in sorted(known):
                document = built.get(package) or self._units(snapshot, package)
                if package in built:
                    model.check_schema(document, model.UNITS_SCHEMA, package)
                    changes[model.units_doc(package)] = document
                members.update(u["citation"] for u in (document or {}).get("units", []))
            removed = []
            for citation in sorted(ledger["units"]):
                entry = ledger["units"][citation]
                if entry.get("package") not in built or citation in members:
                    continue
                if entry.get("state") in model.ACTIVE_LEASE_STATES and (
                    model.lease_is_live(entry, now)
                ):
                    continue
                del ledger["units"][citation]
                removed.append(citation)
            changes[model.LEDGER_DOC] = ledger
            result = {
                "packages": sorted(built),
                "units": {p: len(built[p]["units"]) for p in sorted(built)},
                "ledger_entries_removed": removed,
            }
            return changes, f"queue: build {len(built)} packages ({person})", result

        return self._transact(decide)

    # ----------------------------------------------------------------- claim

    def claim(
        self,
        *,
        machine: str,
        person: str,
        encoder_version: str,
        count: int = 1,
        package: str | None = None,
        lease_seconds: int = model.DEFAULT_LEASE_SECONDS,
    ) -> list[dict[str, Any]]:
        """Reserve the next ``count`` ready units for one owner.

        Order: package rank, then unit rank. Skipped: packages that are not
        active, units with a live lease, a candidate, an outcome, or a block.
        The result can be shorter than ``count``, or empty.
        """

        owner = model.check_owner(machine, person)
        model.check_lease_seconds(lease_seconds)
        if not 1 <= count <= model.MAX_CLAIM_COUNT:
            raise QueueError(f"count must be 1 to {model.MAX_CLAIM_COUNT}")
        if not encoder_version:
            raise QueueError("claim needs the encoder version of the worker")

        def decide(snapshot: Snapshot, now: datetime) -> _Decision:
            packages = self._packages(snapshot)
            if package is not None and not any(
                p["id"] == package for p in packages["packages"]
            ):
                raise QueueError(f"package {package} is not in the queue")
            ledger = self._ledger(snapshot)
            claimed: list[dict[str, Any]] = []
            for entry in self._ranked(packages):
                if len(claimed) == count:
                    break
                if package is not None and entry["id"] != package:
                    continue
                if entry["status"] != model.PACKAGE_ACTIVE:
                    continue
                units = self._units(snapshot, entry["id"])
                if units is None or units.get("blocked"):
                    continue
                for unit in sorted(units["units"], key=lambda u: u["rank"]):
                    if len(claimed) == count:
                        break
                    citation = unit["citation"]
                    previous = ledger["units"].get(citation)
                    state = model.effective_state(
                        previous,
                        now,
                        encoder_version=encoder_version,
                        open_pr=unit.get("open_pr"),
                    )
                    if state != model.READY:
                        continue
                    attempts = int((previous or {}).get("attempts", 0))
                    lease = {
                        "state": model.RESERVED,
                        "package": entry["id"],
                        "owner": owner,
                        "lease_id": uuid.uuid4().hex,
                        "claimed_at": model.format_time(now),
                        "heartbeat_at": model.format_time(now),
                        "lease_expires_at": model.format_time(
                            now + timedelta(seconds=lease_seconds)
                        ),
                        "lease_seconds": lease_seconds,
                        "encoder_version": encoder_version,
                        "attempts": attempts,
                        "updated_at": model.format_time(now),
                    }
                    ledger["units"][citation] = lease
                    claimed.append(
                        {
                            "citation": citation,
                            "package": entry["id"],
                            "target_path": unit.get("target_path"),
                            "lease_id": lease["lease_id"],
                            "lease_expires_at": lease["lease_expires_at"],
                        }
                    )
            if not claimed:
                return {}, "", []
            message = (
                f"queue: {machine}/{person} claims {len(claimed)} "
                f"({claimed[0]['package']})"
            )
            return {model.LEDGER_DOC: ledger}, message, claimed

        return self._transact(decide)

    def _held(
        self,
        ledger: dict[str, Any],
        citation: str,
        lease_id: str,
        now: datetime,
    ) -> dict[str, Any]:
        """The ledger entry of a lease the caller still holds.

        The lease is identified by its id. An expired lease that nobody
        claimed again is still the caller's: a slow encode keeps its result.
        After another claim the entry carries another lease id.
        """

        entry = ledger["units"].get(citation)
        if (
            entry is None
            or entry.get("lease_id") != lease_id
            or entry.get("state") not in model.ACTIVE_LEASE_STATES
        ):
            raise QueueError(
                f"lease {lease_id} on {citation} is gone: the lease expired and "
                "the unit was claimed again, or it was released. Stop this unit."
            )
        return entry

    def heartbeat(
        self,
        leases: dict[str, str],
        *,
        encoding: bool = False,
    ) -> dict[str, str]:
        """Extend leases. ``leases`` maps citation to lease id.

        One call (one commit) extends every lease of a machine. With
        ``encoding`` the units move from reserved to encoding. Returns the
        new expiry per citation; a lost lease is a ``QueueError``.
        """

        if not leases:
            raise QueueError("heartbeat needs one lease or more")

        def decide(snapshot: Snapshot, now: datetime) -> _Decision:
            ledger = self._ledger(snapshot)
            expiries = {}
            for citation, lease_id in sorted(leases.items()):
                entry = self._held(ledger, citation, lease_id, now)
                if encoding:
                    entry["state"] = model.ENCODING
                expiry = model.format_time(
                    now + timedelta(seconds=int(entry["lease_seconds"]))
                )
                entry["heartbeat_at"] = model.format_time(now)
                entry["lease_expires_at"] = expiry
                entry["updated_at"] = model.format_time(now)
                expiries[citation] = expiry
            return (
                {model.LEDGER_DOC: ledger},
                f"queue: heartbeat {len(leases)} leases",
                expiries,
            )

        return self._transact(decide)

    def release(self, citation: str, lease_id: str) -> None:
        """Give a reserved or encoding unit back. It is ready at once."""

        def decide(snapshot: Snapshot, now: datetime) -> _Decision:
            ledger = self._ledger(snapshot)
            entry = self._held(ledger, citation, lease_id, now)
            if int(entry.get("attempts", 0)) > 0:
                # Keep the count of earlier refusals.
                ledger["units"][citation] = {
                    "state": model.READY,
                    "package": entry["package"],
                    "attempts": entry["attempts"],
                    "updated_at": model.format_time(now),
                }
            else:
                del ledger["units"][citation]
            return {model.LEDGER_DOC: ledger}, f"queue: release {citation}", None

        return self._transact(decide)

    # ---------------------------------------------------------------- report

    def report(
        self,
        citation: str,
        state: str,
        *,
        lease_id: str | None = None,
        person: str | None = None,
        category: str | None = None,
        unblock: str | None = None,
        candidate_pr: int | None = None,
        signing_run: int | None = None,
        train_pr: int | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Record the outcome of an encode, or a later landing step.

        A worker that holds the lease passes ``lease_id`` and reports
        ``candidate``, ``refused`` (with a category) or ``blocked`` (with
        what unblocks it). Landing steps (``signing``, ``in_train``,
        ``landed``, back to ``candidate``) and ``ready`` (requeue, unblock)
        have no lease; they pass ``person``.
        """

        if state not in model.STATES or state in (model.RESERVED, model.ENCODING):
            raise QueueError(f"report cannot set state {state!r}")
        if state == model.REFUSED:
            model.check_category(category)
        if state == model.BLOCKED and not (unblock or "").strip():
            raise QueueError("a blocked unit must say what unblocks it")
        if lease_id is None:
            if person is None:
                raise QueueError("report needs a lease id or a person")
            model.check_owner("any", person)

        def decide(snapshot: Snapshot, now: datetime) -> _Decision:
            _package, unit = self._find_unit(snapshot, citation)
            ledger = self._ledger(snapshot)
            if lease_id is not None:
                entry = self._held(ledger, citation, lease_id, now)
                current = entry["state"]
            else:
                entry = ledger["units"].get(citation)
                current = model.effective_state(entry, now, open_pr=unit.get("open_pr"))
                if current in model.ACTIVE_LEASE_STATES:
                    raise QueueError(
                        f"{citation} is {current} under a live lease; "
                        "only the lease holder reports on it"
                    )
                if entry is None or entry.get("state") in model.ACTIVE_LEASE_STATES:
                    # Ready (an expired lease included), or a candidate that
                    # only the build knows. Keep the count of refusals.
                    entry = {
                        "package": _package["id"],
                        "attempts": int((entry or {}).get("attempts", 0)),
                    }
            model.check_transition(current, state)
            if state == model.READY:
                # Requeue or unblock: the unit starts again with no history.
                ledger["units"].pop(citation, None)
                return (
                    {model.LEDGER_DOC: ledger},
                    f"queue: {citation} ready ({person or 'lease holder'})",
                    {"citation": citation, "state": model.READY},
                )
            updated: dict[str, Any] = {
                "state": state,
                "package": entry["package"],
                "attempts": int(entry.get("attempts", 0)),
                "updated_at": model.format_time(now),
            }
            for key in (
                "owner",
                "encoder_version",
                "candidate_pr",
                "signing_run",
                "train_pr",
            ):
                if entry.get(key) is not None:
                    updated[key] = entry[key]
            if lease_id is None:
                updated["reported_by"] = person
            if state == model.REFUSED:
                updated["attempts"] += 1
                updated["category"] = category
            if state == model.BLOCKED:
                updated["unblock"] = (unblock or "").strip()
                if category:
                    updated["category"] = category
            for key, value in (
                ("candidate_pr", candidate_pr),
                ("signing_run", signing_run),
                ("train_pr", train_pr),
            ):
                if value is not None:
                    updated[key] = int(value)
            if note:
                updated["note"] = note
            ledger["units"][citation] = updated
            return (
                {model.LEDGER_DOC: ledger},
                f"queue: {citation} {state}",
                {"citation": citation, **updated},
            )

        return self._transact(decide)

    # ---------------------------------------------------------------- status

    def status(self, *, encoder_version: str | None = None) -> dict[str, Any]:
        """The whole queue at this moment, for people and for the ops page."""

        snapshot = self._store.snapshot()
        now = self._clock()
        packages = self._packages(snapshot)
        ledger = self._ledger(snapshot)
        rows = []
        active = []
        for package in self._ranked(packages):
            units = self._units(snapshot, package["id"])
            counts = dict.fromkeys(model.STATES, 0)
            unit_rows = []
            package_open = package["status"] == model.PACKAGE_ACTIVE and not (
                units or {}
            ).get("blocked")
            for unit in sorted((units or {}).get("units", []), key=lambda u: u["rank"]):
                entry = ledger["units"].get(unit["citation"])
                state = model.effective_state(
                    entry,
                    now,
                    encoder_version=encoder_version,
                    open_pr=unit.get("open_pr"),
                )
                if state == model.READY and not package_open:
                    state = model.BLOCKED
                counts[state] += 1
                row = {"citation": unit["citation"], "state": state}
                if state in model.ACTIVE_LEASE_STATES and entry:
                    row.update(
                        owner=entry["owner"],
                        claimed_at=entry["claimed_at"],
                        heartbeat_at=entry["heartbeat_at"],
                        lease_expires_at=entry["lease_expires_at"],
                    )
                    active.append({"package": package["id"], **row})
                elif state == model.REFUSED and entry:
                    row.update(
                        category=entry.get("category"),
                        encoder_version=entry.get("encoder_version"),
                    )
                elif state == model.BLOCKED:
                    row["unblock"] = (
                        model.blocked_reason(entry, now)
                        or package.get("unblock")
                        or (units or {}).get("blocked")
                        or f"package is {package['status']}"
                    )
                elif entry:
                    for key in ("candidate_pr", "signing_run", "train_pr"):
                        if entry.get(key) is not None:
                            row[key] = entry[key]
                if state == model.CANDIDATE and unit.get("open_pr"):
                    row.setdefault("candidate_pr", unit["open_pr"])
                unit_rows.append(row)
            rows.append(
                {
                    "id": package["id"],
                    "rank": package["rank"],
                    "program": package["program"],
                    "jurisdiction": package["jurisdiction"],
                    "status": package["status"],
                    "unblock": package.get("unblock")
                    or (units or {}).get("blocked")
                    or "",
                    "built": units is not None,
                    "built_at": (units or {}).get("built_at"),
                    "inputs": (units or {}).get("inputs"),
                    "landed_on_main": (units or {}).get("on_main", 0),
                    "counts": counts,
                    "units": unit_rows,
                }
            )
        return {
            "schema": "axiom-encode/central-queue/status/v1",
            "revision": snapshot.revision,
            "at": model.format_time(now),
            "active": active,
            "packages": rows,
        }
