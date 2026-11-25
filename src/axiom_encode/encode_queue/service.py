"""Operations of the central encode queue over a :class:`QueueStore`.

Every write is ``read snapshot -> change documents -> commit at the read
revision``. A lost race raises ``RevisionConflict`` in the store; the
operation then reads again and decides again, so it never acts on stale
state.

Worker operations (claim, heartbeat, report, release) go through
:meth:`EncodeQueue.apply`, which takes a list of operations and writes them
in **one commit**. A machine with several workers sends one batch per
interval, so the number of writes depends on the number of machines, not on
the number of workers or units.

Worker operations are idempotent. A claim carries a request id: the same
request id returns the same units. A report or a release that is sent again
after it was applied gets the same answer. So a worker that got no answer
repeats its request safely.
"""

from __future__ import annotations

import copy
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

MAX_COMMIT_ATTEMPTS = 20
MAX_BATCH_OPERATIONS = 200

_Decision = tuple[str, Any]


class _Work:
    """The documents of one snapshot, validated on first read."""

    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        self.now = snapshot.now
        self.docs: dict[str, Document | None] = {}
        self.dirty: set[str] = set()

    def checkpoint(self) -> tuple[dict[str, Document | None], set[str]]:
        return copy.deepcopy(self.docs), set(self.dirty)

    def rollback(self, saved: tuple[dict[str, Document | None], set[str]]) -> None:
        self.docs, self.dirty = saved

    def packages(self) -> dict[str, Any]:
        if model.PACKAGES_DOC not in self.docs:
            document = self.snapshot.read(model.PACKAGES_DOC)
            if document is None:
                document = model.empty_packages()
            model.validate_packages(document)
            if document.get("moved_to"):
                raise QueueError(
                    f"this queue is closed; it moved to branch {document['moved_to']}"
                )
            self.docs[model.PACKAGES_DOC] = document
        return self.docs[model.PACKAGES_DOC]

    def ranked(self) -> list[dict[str, Any]]:
        return sorted(self.packages()["packages"], key=lambda p: p["rank"])

    def package(self, identifier: str) -> dict[str, Any]:
        for entry in self.packages()["packages"]:
            if entry["id"] == identifier:
                return entry
        raise QueueError(f"package {identifier} is not in the queue")

    def units(self, package: str) -> dict[str, Any] | None:
        path = model.units_doc(package)
        if path not in self.docs:
            document = self.snapshot.read(path)
            if document is not None:
                model.validate_units(document, package)
            self.docs[path] = document
        return self.docs[path]

    def ledger(self, package: str) -> dict[str, Any]:
        path = model.ledger_doc(package)
        if path not in self.docs:
            document = self.snapshot.read(path)
            if document is None:
                document = model.empty_ledger(package)
            model.validate_ledger(document, package, self.now)
            self.docs[path] = document
        return self.docs[path]

    def touch(self, path: str) -> None:
        self.dirty.add(path)

    def changes(self) -> dict[str, Document | None]:
        return {path: self.docs[path] for path in sorted(self.dirty)}


def _field(op: dict[str, Any], key: str, kind: type, default: Any = None) -> Any:
    value = op.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) and kind is not bool:
        raise QueueError(f"operation field {key} has the wrong type")
    if not isinstance(value, kind):
        raise QueueError(f"operation field {key} has the wrong type")
    return value


_OP_FIELDS = {
    "claim": {
        "machine", "person", "encoder_version", "count", "package",
        "lease_seconds", "request_id",
    },
    "heartbeat": {"machine", "person", "leases", "encoding"},
    "release": {"machine", "person", "citation", "lease_id"},
    "report": {
        "machine", "person", "citation", "state", "lease_id", "category",
        "unblock", "candidate_pr", "signing_run", "train_pr", "note",
    },
}  # fmt: skip


class EncodeQueue:
    def __init__(
        self,
        store: QueueStore,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._store = store
        self._sleep = sleep

    # ------------------------------------------------------------------ core

    def _transact(self, decide: Callable[[_Work], _Decision]) -> Any:
        for attempt in range(MAX_COMMIT_ATTEMPTS):
            work = _Work(self._store.snapshot())
            message, result = decide(work)
            changes = work.changes()
            if not changes:
                return result
            try:
                self._store.commit(work.snapshot.revision, changes, message)
            except RevisionConflict:
                self._sleep(random.uniform(0.1, 1.0) * min(attempt + 1, 8))  # noqa: S311
                continue
            return result
        raise QueueError(
            f"the queue store stayed busy for {MAX_COMMIT_ATTEMPTS} attempts; try again"
        )

    def _find_unit(self, work: _Work, citation: str) -> tuple[str, dict[str, Any]]:
        model.check_citation(citation)
        for package in work.ranked():
            units = work.units(package["id"])
            for unit in (units or {}).get("units", []):
                if unit["citation"] == citation:
                    return package["id"], unit
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
        model.check_person(person)
        note = model.check_text(note, "note")

        def decide(work: _Work) -> _Decision:
            ordered = work.ranked()
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
                "added_at": model.format_time(work.now),
            }
            self._insert(ordered, entry, before)
            work.packages()["packages"] = ordered
            work.touch(model.PACKAGES_DOC)
            return f"queue: add package {identifier} ({person})", entry

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

        model.check_person(person)

        def decide(work: _Work) -> _Decision:
            ordered = work.ranked()
            entry = work.package(package)
            if before == package:
                raise QueueError("a package cannot move in front of itself")
            ordered.remove(entry)
            self._insert(ordered, entry, before)
            work.packages()["packages"] = ordered
            work.touch(model.PACKAGES_DOC)
            return (
                f"queue: move package {package} ({person})",
                [p["id"] for p in ordered],
            )

        return self._transact(decide)

    def set_package_status(
        self, package: str, status: str, *, person: str, unblock: str = ""
    ) -> dict[str, Any]:
        """Set a package active, paused or blocked.

        A blocked package must say what unblocks it. Units of a package that
        is not active are never claimed; live leases run to their end.
        """

        model.check_person(person)
        if status not in model.PACKAGE_STATUSES:
            raise QueueError(f"status must be one of {model.PACKAGE_STATUSES}")
        unblock = model.check_text(unblock, "unblock")
        if status == model.PACKAGE_BLOCKED and not unblock:
            raise QueueError("a blocked package must say what unblocks it")

        def decide(work: _Work) -> _Decision:
            entry = work.package(package)
            entry["status"] = status
            entry["unblock"] = unblock if status != model.PACKAGE_ACTIVE else ""
            entry["status_by"] = person
            entry["status_at"] = model.format_time(work.now)
            work.touch(model.PACKAGES_DOC)
            return f"queue: package {package} {status} ({person})", entry

        return self._transact(decide)

    # ----------------------------------------------------------------- build

    def apply_build(
        self, built: dict[str, dict[str, Any]], *, person: str
    ) -> dict[str, Any]:
        """Replace the derived membership with the result of a build.

        ``built`` maps a package id to its units document
        (:func:`axiom_encode.encode_queue.build.derive_package`). Packages
        that are not in ``built`` keep their units.

        - A unit belongs to one package: the highest-ranked one that lists
          it. It is removed from the others.
        - A ledger entry whose unit left the membership (it landed on main,
          or it left the pinned corpus) is removed, so a landed unit
          disappears.
        - A unit under a live lease is never removed. Its entry is marked
          ``superseded``; the owner sees that in the answer to its next
          heartbeat or report and stops. The entry goes at release, at
          report, or at the first build after the lease expired.
        """

        model.check_person(person)

        def decide(work: _Work) -> _Decision:
            known = [p["id"] for p in work.ranked()]
            unknown = sorted(set(built) - set(known))
            if unknown:
                raise QueueError(f"build has packages that are not queued: {unknown}")
            seen: set[str] = set()
            sizes: dict[str, int] = {}
            removed: list[str] = []
            superseded: list[str] = []
            for package in known:
                path = model.units_doc(package)
                if package in built:
                    document = model.validate_units(
                        copy.deepcopy(built[package]), package
                    )
                    work.docs[path] = document
                    work.touch(path)
                else:
                    document = work.units(package)
                if document is None:
                    continue
                kept = [u for u in document["units"] if u["citation"] not in seen]
                if len(kept) != len(document["units"]):
                    document["units"] = kept
                    work.touch(path)
                seen.update(u["citation"] for u in kept)
                if package in built:
                    sizes[package] = len(kept)
                members = {u["citation"] for u in kept}
                ledger = work.ledger(package)
                for citation in sorted(ledger["units"]):
                    entry = ledger["units"][citation]
                    live = entry["state"] in model.ACTIVE_LEASE_STATES and (
                        model.lease_is_live(entry, work.now)
                    )
                    if citation in members:
                        if entry.pop("superseded", None):
                            work.touch(model.ledger_doc(package))
                        if (
                            not live
                            and entry["state"]
                            in (model.READY, *model.ACTIVE_LEASE_STATES)
                            and entry["attempts"] == 0
                            and not entry.get("refused_versions")
                        ):
                            # Nothing to remember about this ready unit.
                            del ledger["units"][citation]
                            work.touch(model.ledger_doc(package))
                        continue
                    if live:
                        if not entry.get("superseded"):
                            entry["superseded"] = True
                            superseded.append(citation)
                            work.touch(model.ledger_doc(package))
                        continue
                    del ledger["units"][citation]
                    removed.append(citation)
                    work.touch(model.ledger_doc(package))
            result = {
                "packages": sorted(built),
                "units": sizes,
                "ledger_entries_removed": removed,
                "leases_superseded": superseded,
            }
            return f"queue: build {len(built)} packages ({person})", result

        return self._transact(decide)

    # ------------------------------------------------------ worker operations

    def apply(self, operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run worker operations in order and write them in one commit.

        Each operation is a mapping with ``op`` = ``claim``, ``heartbeat``,
        ``report`` or ``release`` and the fields of that operation. The
        result has one mapping per operation: ``{"ok": true, ...}`` or
        ``{"ok": false, "error": "..."}``. A refused operation changes
        nothing and does not stop the others.
        """

        if not isinstance(operations, list) or not operations:
            raise QueueError("apply needs a list of operations")
        if len(operations) > MAX_BATCH_OPERATIONS:
            raise QueueError(f"apply takes at most {MAX_BATCH_OPERATIONS} operations")
        handlers = {
            "claim": self._claim,
            "heartbeat": self._heartbeat,
            "release": self._release,
            "report": self._report,
        }

        def decide(work: _Work) -> _Decision:
            work.packages()  # a closed or damaged queue refuses the whole batch
            results = []
            counts: dict[str, int] = {}
            for op in operations:
                saved = work.checkpoint()
                try:
                    if not isinstance(op, dict) or op.get("op") not in handlers:
                        raise QueueError("operation needs op = " + "|".join(handlers))
                    unknown = sorted(set(op) - _OP_FIELDS[op["op"]] - {"op"})
                    if unknown:
                        raise QueueError(f"operation has unknown fields {unknown}")
                    results.append({"ok": True, **handlers[op["op"]](work, op)})
                    counts[op["op"]] = counts.get(op["op"], 0) + 1
                except QueueError as error:
                    work.rollback(saved)
                    results.append({"ok": False, "error": str(error)})
            summary = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
            owner = next(
                (
                    f"{op.get('machine')}/{op.get('person')}"
                    for op in operations
                    if isinstance(op, dict) and op.get("machine")
                ),
                "person",
            )
            return f"queue: {owner}: {summary}", results

        return self._transact(decide)

    def _one(self, op: dict[str, Any]) -> dict[str, Any]:
        result = self.apply([{k: v for k, v in op.items() if v is not None}])[0]
        if not result["ok"]:
            raise QueueError(result["error"])
        return result

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
        request_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Reserve the next ``count`` ready units for one owner.

        Order: package rank, then unit rank. Skipped: packages that are not
        active, units with a live lease, a candidate, an outcome, or a block.
        The result can be shorter than ``count``, or empty. The same
        ``request_id`` returns the same units while their leases are held.
        """

        return self._one(
            {
                "op": "claim",
                "machine": machine,
                "person": person,
                "encoder_version": encoder_version,
                "count": count,
                "package": package,
                "lease_seconds": lease_seconds,
                "request_id": request_id,
            }
        )["claimed"]

    def _claim(self, work: _Work, op: dict[str, Any]) -> dict[str, Any]:
        owner = model.check_owner(op.get("machine"), op.get("person"))
        version = model.check_version(op.get("encoder_version"))
        count = _field(op, "count", int, 1)
        if not 1 <= count <= model.MAX_CLAIM_COUNT:
            raise QueueError(f"count must be 1 to {model.MAX_CLAIM_COUNT}")
        lease_seconds = model.check_lease_seconds(
            _field(op, "lease_seconds", int, model.DEFAULT_LEASE_SECONDS)
        )
        request_id = model.check_request_id(op.get("request_id") or uuid.uuid4().hex)
        only = _field(op, "package", str)
        if only is not None:
            work.package(only)
        packages = [
            p["id"]
            for p in work.ranked()
            if (only is None or p["id"] == only) and p["status"] == model.PACKAGE_ACTIVE
        ]
        now = work.now

        def answer(
            citation: str, package: str, entry: dict[str, Any]
        ) -> dict[str, Any]:
            return {
                "citation": citation,
                "package": package,
                "target_path": model.target_path(citation),
                "lease_id": entry["lease_id"],
                "lease_expires_at": entry["lease_expires_at"],
            }

        if op.get("request_id"):
            # The same request again: the same units, nothing new.
            held = [
                answer(citation, package, entry)
                for package in packages
                for citation, entry in sorted(work.ledger(package)["units"].items())
                if entry.get("request_id") == request_id
                and entry.get("owner") == owner
                and entry["state"] in model.ACTIVE_LEASE_STATES
                and model.lease_is_live(entry, now)
            ]
            if held:
                return {"claimed": held, "request_id": request_id, "repeated": True}
        claimed: list[dict[str, Any]] = []
        for package in packages:
            if len(claimed) == count:
                break
            units = work.units(package)
            if units is None or units.get("blocked"):
                continue
            ledger = work.ledger(package)
            for unit in sorted(units["units"], key=lambda u: u["rank"]):
                if len(claimed) == count:
                    break
                citation = unit["citation"]
                previous = ledger["units"].get(citation)
                state = model.effective_state(
                    previous, now, encoder_version=version, open_pr=unit.get("open_pr")
                )
                if state != model.READY:
                    continue
                try:
                    model.target_path(citation)
                except QueueError:
                    continue  # never hand out a unit with no safe target
                entry: dict[str, Any] = {
                    "state": model.RESERVED,
                    "owner": owner,
                    "lease_id": model.new_lease_id(package, uuid.uuid4().hex),
                    "request_id": request_id,
                    "claimed_at": model.format_time(now),
                    "heartbeat_at": model.format_time(now),
                    "lease_expires_at": model.format_time(
                        now + timedelta(seconds=lease_seconds)
                    ),
                    "lease_seconds": lease_seconds,
                    "encoder_version": version,
                    "attempts": (previous or {}).get("attempts", 0),
                    "updated_at": model.format_time(now),
                }
                if (previous or {}).get("refused_versions"):
                    entry["refused_versions"] = previous["refused_versions"]
                ledger["units"][citation] = entry
                work.touch(model.ledger_doc(package))
                claimed.append(answer(citation, package, entry))
        return {"claimed": claimed, "request_id": request_id, "repeated": False}

    # ---------------------------------------------------------------- leases

    @staticmethod
    def _held(
        work: _Work, citation: Any, lease_id: Any, owner: dict[str, str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """The ledger shard and the entry of a lease the caller still holds.

        The caller proves the lease with its id and its owner identity. An
        expired lease that nobody claimed again is still the caller's: a
        slow encode keeps its result. After another claim the entry carries
        another lease id.
        """

        model.check_citation(citation)
        ledger = work.ledger(model.lease_package(lease_id))
        entry = ledger["units"].get(citation)
        if (
            entry is None
            or entry.get("lease_id") != lease_id
            or entry["state"] not in model.ACTIVE_LEASE_STATES
        ):
            raise QueueError(
                f"lease {lease_id} on {citation} is gone: the lease expired and "
                "the unit was claimed again, or it was released. Stop this unit."
            )
        if entry["owner"] != owner:
            raise QueueError(f"lease {lease_id} on {citation} has another owner")
        return ledger, entry

    def heartbeat(
        self,
        leases: dict[str, str],
        *,
        machine: str,
        person: str,
        encoding: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Extend leases. ``leases`` maps citation to lease id.

        The answer has one result per lease. A valid lease is extended also
        when another lease of the call is lost:
        ``{"ok": true, "lease_expires_at": ..., "superseded": false}`` or
        ``{"ok": false, "error": ...}``. ``superseded`` means the unit left
        the queue (for example it landed on main): stop it and release it.
        With ``encoding`` the units move from reserved to encoding.
        """

        return self._one(
            {
                "op": "heartbeat",
                "machine": machine,
                "person": person,
                "leases": leases,
                "encoding": encoding,
            }
        )["leases"]

    def _heartbeat(self, work: _Work, op: dict[str, Any]) -> dict[str, Any]:
        owner = model.check_owner(op.get("machine"), op.get("person"))
        leases = _field(op, "leases", dict)
        if not leases:
            raise QueueError("heartbeat needs one lease or more")
        encoding = bool(_field(op, "encoding", bool, False))
        now = work.now
        results: dict[str, dict[str, Any]] = {}
        for citation, lease_id in sorted(leases.items()):
            try:
                ledger, entry = self._held(work, citation, lease_id, owner)
            except QueueError as error:
                results[str(citation)] = {"ok": False, "error": str(error)}
                continue
            seconds = entry["lease_seconds"]
            remaining = (
                model.parse_time(entry["lease_expires_at"]) - now
            ).total_seconds()
            flip = encoding and entry["state"] != model.ENCODING
            # A lease with more than half its time left is not rewritten.
            if flip or remaining < seconds / 2:
                if flip:
                    entry["state"] = model.ENCODING
                entry["heartbeat_at"] = model.format_time(now)
                entry["lease_expires_at"] = model.format_time(
                    now + timedelta(seconds=seconds)
                )
                entry["updated_at"] = model.format_time(now)
                work.touch(model.ledger_doc(ledger["package"]))
            results[citation] = {
                "ok": True,
                "lease_expires_at": entry["lease_expires_at"],
                "superseded": bool(entry.get("superseded")),
            }
        return {"leases": results}

    @staticmethod
    def _closed(entry: dict[str, Any], lease_id: str, now: datetime) -> dict[str, Any]:
        """What stays of a lease entry after the lease ended."""

        closed: dict[str, Any] = {
            "state": model.READY,
            "attempts": entry["attempts"],
            "closed_lease_id": lease_id,
            "updated_at": model.format_time(now),
        }
        if entry.get("refused_versions"):
            closed["refused_versions"] = entry["refused_versions"]
        return closed

    def release(
        self, citation: str, lease_id: str, *, machine: str, person: str
    ) -> dict[str, Any]:
        """Give a reserved or encoding unit back. It is ready at once."""

        return self._one(
            {
                "op": "release",
                "machine": machine,
                "person": person,
                "citation": citation,
                "lease_id": lease_id,
            }
        )

    def _release(self, work: _Work, op: dict[str, Any]) -> dict[str, Any]:
        owner = model.check_owner(op.get("machine"), op.get("person"))
        citation, lease_id = op.get("citation"), op.get("lease_id")
        model.check_citation(citation)
        package = model.lease_package(lease_id)
        answer = {"citation": citation, "state": model.READY}
        previous = work.ledger(package)["units"].get(citation)
        if (
            previous is not None
            and previous.get("closed_lease_id") == lease_id
            and previous["state"] == model.READY
        ):
            return {**answer, "repeated": True}
        ledger, entry = self._held(work, citation, lease_id, owner)
        if entry.get("superseded"):
            del ledger["units"][citation]
        else:
            ledger["units"][citation] = self._closed(entry, lease_id, work.now)
        work.touch(model.ledger_doc(package))
        return {**answer, "repeated": False}

    # ---------------------------------------------------------------- report

    def report(
        self,
        citation: str,
        state: str,
        *,
        lease_id: str | None = None,
        machine: str | None = None,
        person: str | None = None,
        category: str | None = None,
        unblock: str | None = None,
        candidate_pr: int | None = None,
        signing_run: int | None = None,
        train_pr: int | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Record the outcome of an encode, or a later landing step.

        A worker that holds the lease passes ``lease_id`` with its machine
        and person, and reports ``candidate``, ``refused`` (with a category)
        or ``blocked`` (with what unblocks it). Landing steps (``signing``,
        ``in_train``, ``landed``, back to ``candidate``) and ``ready``
        (requeue, unblock) have no lease; they pass ``person``.
        """

        return self._one(
            {
                "op": "report",
                "citation": citation,
                "state": state,
                "lease_id": lease_id,
                "machine": machine,
                "person": person,
                "category": category,
                "unblock": unblock,
                "candidate_pr": candidate_pr,
                "signing_run": signing_run,
                "train_pr": train_pr,
                "note": note or None,
            }
        )

    def _report(self, work: _Work, op: dict[str, Any]) -> dict[str, Any]:
        citation = model.check_citation(op.get("citation"))
        state = op.get("state")
        if state not in model.STATES or state in model.ACTIVE_LEASE_STATES:
            raise QueueError(f"report cannot set state {state!r}")
        lease_id = op.get("lease_id")
        category = op.get("category")
        if state == model.REFUSED or category is not None:
            model.check_category(category)
        unblock = model.check_text(
            op.get("unblock"), "unblock", required=state == model.BLOCKED
        )
        note = model.check_text(op.get("note"), "note")
        numbers = {}
        for key in ("candidate_pr", "signing_run", "train_pr"):
            value = _field(op, key, int)
            if value is not None:
                if value < 1:
                    raise QueueError(f"{key} must be a positive number")
                numbers[key] = value
        now = work.now
        if lease_id is not None:
            owner = model.check_owner(op.get("machine"), op.get("person"))
            package = model.lease_package(lease_id)
            ledger = work.ledger(package)
            previous = ledger["units"].get(citation)
            if previous is not None and previous.get("closed_lease_id") == lease_id:
                if previous["state"] == state:
                    # The same report again: the same answer, no write.
                    return {"citation": citation, **previous, "repeated": True}
                raise QueueError(
                    f"lease {lease_id} on {citation} was already closed "
                    f"as {previous['state']}"
                )
            ledger, entry = self._held(work, citation, lease_id, owner)
            if entry.get("superseded"):
                del ledger["units"][citation]
                work.touch(model.ledger_doc(package))
                return {
                    "citation": citation,
                    "state": "superseded",
                    "note": "the unit left the queue during the lease; "
                    "the result is not recorded",
                }
            current = entry["state"]
            open_pr = None
        else:
            person = model.check_person(op.get("person"))
            package, unit = self._find_unit(work, citation)
            ledger = work.ledger(package)
            entry = ledger["units"].get(citation)
            open_pr = unit.get("open_pr")
            current = model.effective_state(entry, now, open_pr=open_pr)
            if current in model.ACTIVE_LEASE_STATES:
                raise QueueError(
                    f"{citation} is {current} under a live lease; "
                    "only the lease holder reports on it"
                )
            if entry is None:
                entry = {"attempts": 0}
        model.check_transition(current, state)
        work.touch(model.ledger_doc(package))
        if state == model.READY and lease_id is None:
            # Requeue or unblock by a person: the unit starts again.
            ledger["units"].pop(citation, None)
            return {"citation": citation, "state": model.READY}
        if state == model.READY:
            ledger["units"][citation] = self._closed(entry, lease_id, now)
            return {"citation": citation, "state": model.READY}
        updated: dict[str, Any] = {
            "state": state,
            "attempts": entry["attempts"],
            "updated_at": model.format_time(now),
        }
        for key in (
            "owner",
            "encoder_version",
            "refused_versions",
            "candidate_pr",
            "signing_run",
            "train_pr",
        ):
            if entry.get(key) is not None:
                updated[key] = entry[key]
        if lease_id is not None:
            updated["closed_lease_id"] = lease_id
        else:
            updated["reported_by"] = person
            if entry.get("closed_lease_id"):
                updated["closed_lease_id"] = entry["closed_lease_id"]
        if state == model.REFUSED:
            updated["attempts"] += 1
            updated["category"] = category
            version = entry.get("encoder_version")
            versions = list(entry.get("refused_versions", []))
            if version and version not in versions:
                versions.append(version)
            if versions:
                updated["refused_versions"] = versions
        if state == model.BLOCKED:
            updated["unblock"] = unblock
            if category:
                updated["category"] = category
        updated.update(numbers)
        if note:
            updated["note"] = note
        ledger["units"][citation] = updated
        return {"citation": citation, **updated}

    # ---------------------------------------------------------------- status

    def status(self, *, encoder_version: str | None = None) -> dict[str, Any]:
        """The whole queue at this moment, for people and for the ops page."""

        if encoder_version is not None:
            model.check_version(encoder_version)
        work = _Work(self._store.snapshot())
        now = work.now
        rows = []
        active = []
        for package in work.ranked():
            units = work.units(package["id"])
            ledger = work.ledger(package["id"])["units"]
            counts = dict.fromkeys(model.STATES, 0)
            unit_rows = []
            package_open = package["status"] == model.PACKAGE_ACTIVE and not (
                units or {}
            ).get("blocked")
            members = set()
            for unit in sorted((units or {}).get("units", []), key=lambda u: u["rank"]):
                members.add(unit["citation"])
                entry = ledger.get(unit["citation"])
                state = model.effective_state(
                    entry,
                    now,
                    encoder_version=encoder_version,
                    open_pr=unit.get("open_pr"),
                )
                if state == model.READY and not package_open:
                    state = model.BLOCKED
                    entry = None
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
                        refused_versions=entry.get("refused_versions", []),
                    )
                elif state == model.BLOCKED:
                    row["unblock"] = (
                        model.blocked_reason(entry)
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
            for citation, entry in sorted(ledger.items()):
                # A lease on a unit that left the membership (superseded).
                if (
                    citation not in members
                    and entry["state"] in model.ACTIVE_LEASE_STATES
                    and model.lease_is_live(entry, now)
                ):
                    active.append(
                        {
                            "package": package["id"],
                            "citation": citation,
                            "state": entry["state"],
                            "owner": entry["owner"],
                            "claimed_at": entry["claimed_at"],
                            "heartbeat_at": entry["heartbeat_at"],
                            "lease_expires_at": entry["lease_expires_at"],
                            "superseded": True,
                        }
                    )
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
            "revision": work.snapshot.revision,
            "at": model.format_time(now),
            "active": active,
            "packages": rows,
        }
