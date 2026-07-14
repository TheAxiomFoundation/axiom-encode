"""RuleSpec-side declared oracle-coverage pending classification.

The oracle-coverage gate fails any executable RuleSpec output whose legal ID
has no PolicyEngine oracle mapping (status ``unmapped``). Adding a mapping
means editing axiom-oracles and bumping its pinned commit through
axiom-encode — a per-output cross-repo pin dance that does not scale to bulk
encoding.

A rulespec repository may temporarily *declare* new outputs as pending
classification in a top-level ``oracle-coverage-pending.yaml``. Reports then
treat a declared output as ``pending_classification`` — visible, counted,
and accountable — instead of silently ``unmapped``. Release gates use
``oracle-coverage --fail-on-pending`` and do not admit this temporary state.
A classification sweep
later drains declared entries into real axiom-oracles mappings (one batch pin
bump, not one per PR); once an output is mapped upstream it stops being
``unmapped``, the declaration goes stale, and the ratchet forces its removal.
So the debt only ratchets down.

This module is the single source of truth for the file format, the
report reclassification and the ratchet checks. It is deliberately
dependency-light so the
``oracle-coverage`` command can post-process a report the axiom-oracles
bridge produced without any axiom-oracles change.
"""

from __future__ import annotations

import datetime as _dt
import os
import stat
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ...repo_routing import is_policy_repo_root

PENDING_FILENAME = "oracle-coverage-pending.yaml"
# Report status assigned to a declared output. Distinct from ``unmapped`` so
# release gates can reject it explicitly with ``--fail-on-pending`` while
# reports and humans still see accountable debt rather than a silent gap.
PENDING_STATUS = "pending_classification"
_UNMAPPED_STATUS = "unmapped"
_ALLOWED_SOURCES = {"bulk", "manual", "migration", "backfill"}


class PendingDeclarationError(ValueError):
    """Raised when an ``oracle-coverage-pending.yaml`` file is malformed."""


@dataclass(frozen=True)
class PendingEntry:
    """One declared output awaiting oracle-coverage classification."""

    legal_id: str
    source: str
    since: str
    note: str | None
    repo: str
    file: str

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "legal_id": self.legal_id,
            "source": self.source,
            "since": self.since,
        }
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass(frozen=True)
class PendingFile:
    """A parsed ``oracle-coverage-pending.yaml`` for one repository."""

    path: Path
    repo: str
    entries: tuple[PendingEntry, ...]
    ceiling: int | None
    issue: str | None


def _coerce_date(value: Any, *, where: str) -> str:
    """Normalise a ``since`` value to a ``YYYY-MM-DD`` string.

    YAML parses an unquoted ``2026-07-07`` as a :class:`datetime.date`, so
    accept date/datetime objects as well as ISO strings.
    """
    if isinstance(value, _dt.datetime):
        return value.date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if not isinstance(value, str) or not value.strip():
        raise PendingDeclarationError(f"{where}: since must be a non-empty date")
    text = value.strip()
    try:
        _dt.date.fromisoformat(text)
    except ValueError as exc:
        raise PendingDeclarationError(
            f"{where}: since must be an ISO date (YYYY-MM-DD), got {text!r}"
        ) from exc
    return text


def parse_pending_payload(payload: Any, *, path: Path, repo: str) -> PendingFile:
    """Validate a loaded YAML payload into a :class:`PendingFile`."""
    where = str(path)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise PendingDeclarationError(f"{where}: top-level YAML must be a mapping")
    if payload.get("version") != 1:
        raise PendingDeclarationError(f"{where}: must set version: 1")

    ceiling = payload.get("ceiling")
    if ceiling is not None and (
        not isinstance(ceiling, int) or isinstance(ceiling, bool)
    ):
        raise PendingDeclarationError(f"{where}: ceiling must be an integer")
    if isinstance(ceiling, int) and ceiling < 0:
        raise PendingDeclarationError(f"{where}: ceiling must not be negative")

    issue = payload.get("issue")
    if issue is not None and not isinstance(issue, str):
        raise PendingDeclarationError(f"{where}: issue must be a string URL")

    raw_entries = payload.get("entries")
    if raw_entries is None:
        raw_entries = []
    if not isinstance(raw_entries, list):
        raise PendingDeclarationError(f"{where}: entries must be a list")

    entries: list[PendingEntry] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_entries):
        item_where = f"{where}: entries[{index}]"
        if not isinstance(raw, dict):
            raise PendingDeclarationError(f"{item_where} must be a mapping")
        legal_id = raw.get("legal_id")
        if not isinstance(legal_id, str) or not legal_id.strip():
            raise PendingDeclarationError(
                f"{item_where}: legal_id must be a non-empty string"
            )
        legal_id = legal_id.strip()
        if legal_id in seen:
            raise PendingDeclarationError(
                f"{item_where}: duplicate legal_id {legal_id}"
            )
        seen.add(legal_id)
        source = raw.get("source")
        if not isinstance(source, str) or not source.strip():
            raise PendingDeclarationError(
                f"{item_where}: source must be a non-empty string"
            )
        source = source.strip()
        if source not in _ALLOWED_SOURCES:
            raise PendingDeclarationError(
                f"{item_where}: source {source!r} must be one of "
                f"{sorted(_ALLOWED_SOURCES)}"
            )
        since = _coerce_date(raw.get("since"), where=item_where)
        note = raw.get("note")
        if note is not None and not isinstance(note, str):
            raise PendingDeclarationError(f"{item_where}: note must be a string")
        entries.append(
            PendingEntry(
                legal_id=legal_id,
                source=source,
                since=since,
                note=note.strip() if isinstance(note, str) else None,
                repo=repo,
                file=where,
            )
        )
    return PendingFile(
        path=path,
        repo=repo,
        entries=tuple(entries),
        ceiling=ceiling if isinstance(ceiling, int) else None,
        issue=issue,
    )


def load_pending_file(path: Path, *, repo: str | None = None) -> PendingFile:
    payload = yaml.safe_load(path.read_text())
    return parse_pending_payload(payload, path=path, repo=repo or path.parent.name)


def iter_pending_file_paths(root: Path) -> list[Path]:
    """Return the pending declaration in one explicit canonical checkout.

    GitHub Actions places the repository at ``rulespec-us/rulespec-us`` and
    passes the outer canonical checkout root to oracle coverage. Support that
    exact nested checkout as well as the direct layout used by local callers.
    Never scan sibling repositories.
    """

    raw_root = Path(root).expanduser()
    if not is_policy_repo_root(raw_root):
        raise PendingDeclarationError(
            "Pending declarations require the exact canonical "
            f"rulespec-<country> checkout: {raw_root}"
        )
    checkout = raw_root.resolve(strict=True)
    nested_checkout = checkout / checkout.name
    if nested_checkout.is_symlink():
        raise PendingDeclarationError(
            "Nested RuleSpec checkout must be a regular directory, not a symlink: "
            f"{nested_checkout}"
        )

    candidates = [checkout / PENDING_FILENAME]
    if nested_checkout.is_dir():
        candidates.append(nested_checkout / PENDING_FILENAME)

    found: list[Path] = []
    for path in candidates:
        if path.is_symlink():
            raise PendingDeclarationError(
                f"Pending declaration must be a regular file, not a symlink: {path}"
            )
        if path.is_file():
            found.append(path)
    if len(found) > 1:
        raise PendingDeclarationError(
            "Pending declaration is ambiguous; keep exactly one file in the direct "
            f"or nested canonical checkout: {', '.join(str(path) for path in found)}"
        )
    return found


def load_pending_files(root: Path) -> list[PendingFile]:
    return [load_pending_file(path) for path in iter_pending_file_paths(root)]


def declarations_from_files(
    files: list[PendingFile],
) -> dict[str, PendingEntry]:
    """Merge per-repo pending files into a legal_id -> entry map.

    Legal IDs are globally unique (prefixed by jurisdiction), so a collision
    across two files is a declaration bug and is rejected.
    """
    declared: dict[str, PendingEntry] = {}
    for pending_file in files:
        for entry in pending_file.entries:
            existing = declared.get(entry.legal_id)
            if existing is not None:
                raise PendingDeclarationError(
                    f"{entry.file}: legal_id {entry.legal_id} is also declared "
                    f"in {existing.file}"
                )
            declared[entry.legal_id] = entry
    return declared


def load_pending_declarations(root: Path) -> dict[str, PendingEntry]:
    return declarations_from_files(load_pending_files(root))


def apply_pending_to_report(
    report: dict[str, Any], declared: dict[str, PendingEntry]
) -> dict[str, Any]:
    """Reclassify declared unmapped outputs and annotate the report in place.

    Every report item whose status is ``unmapped`` and whose legal_id is
    declared is rewritten to ``pending_classification`` with a ``pending``
    provenance block; ``status_counts`` is recomputed. A ``pending`` summary
    is attached to the report describing what was declared, applied, and what
    is stale (declared but no longer unmapped — the ratchet's removal list).
    Returns the summary that was attached.
    """
    items = report.get("items") or []
    unmapped_ids = {
        item.get("legal_id") for item in items if item.get("status") == _UNMAPPED_STATUS
    }
    present_ids = {item.get("legal_id") for item in items}

    applied: list[str] = []
    for item in items:
        legal_id = item.get("legal_id")
        if item.get("status") == _UNMAPPED_STATUS and legal_id in declared:
            entry = declared[legal_id]
            item["status"] = PENDING_STATUS
            item["pending"] = {
                "source": entry.source,
                "since": entry.since,
                "repo": entry.repo,
                **({"note": entry.note} if entry.note else {}),
            }
            applied.append(legal_id)

    stale: list[dict[str, str]] = []
    for legal_id, entry in declared.items():
        if legal_id in unmapped_ids:
            continue
        if legal_id in present_ids:
            reason = "already classified upstream — remove this declaration"
        else:
            reason = "output not found (removed or renamed) — remove this declaration"
        stale.append({"legal_id": legal_id, "reason": reason, "repo": entry.repo})

    if items:
        report["status_counts"] = dict(
            sorted(Counter(item.get("status") for item in items).items())
        )

    summary = {
        "declared": len(declared),
        "applied": sorted(applied),
        "stale": sorted(stale, key=lambda row: row["legal_id"]),
        "sources": dict(sorted(Counter(declared[a].source for a in applied).items())),
    }
    report["pending"] = summary
    return summary


def ratchet_problems(
    report: dict[str, Any],
    files: list[PendingFile],
) -> list[str]:
    """Both-ways ratchet, mirroring the known-validation-gaps mechanism.

    Fails when an output is unmapped but undeclared (nothing silent), when a
    declaration is stale (drained upstream — remove it, ratchets debt down),
    or when a file exceeds its declared ceiling. ``report`` must already be
    reclassified via :func:`apply_pending_to_report`. The caller supplies one
    exact country checkout, so cross-repository scoping is neither needed nor
    accepted.
    """
    problems: list[str] = []

    undeclared = sorted(
        item.get("legal_id")
        for item in report.get("items") or []
        if item.get("status") == _UNMAPPED_STATUS
    )
    problems.extend(
        f"unmapped output is not declared in {PENDING_FILENAME}: {legal_id}"
        for legal_id in undeclared
    )

    pending = report.get("pending") or {}
    problems.extend(
        f"{PENDING_FILENAME} entry is fixed — remove it: {row['legal_id']} "
        f"({row['reason']})"
        for row in pending.get("stale") or []
    )

    for pending_file in files:
        if (
            pending_file.ceiling is not None
            and len(pending_file.entries) > pending_file.ceiling
        ):
            problems.append(
                f"{pending_file.path}: {len(pending_file.entries)} entries exceed "
                f"ceiling {pending_file.ceiling}"
            )
    return problems


def _dump_pending_file(
    *,
    entries: list[PendingEntry],
    ceiling: int,
    issue: str | None,
) -> str:
    header = [
        "# Declared oracle-coverage debt: executable RuleSpec outputs awaiting",
        "# classification in axiom-oracles PolicyEngine mappings. The",
        "# oracle-coverage gate treats these as pending_classification (visible,",
        "# counted, accountable) instead of silently unmapped. Ratchets both",
        "# ways: a new unmapped output must be declared here or the gate fails;",
        "# an entry that is no longer unmapped must be removed.",
        "# Maintained by: axiom-encode oracle-coverage-pending sync.",
    ]
    payload: dict[str, Any] = {"version": 1}
    if issue:
        payload["issue"] = issue
    payload["ceiling"] = ceiling
    payload["entries"] = [
        entry.as_dict() for entry in sorted(entries, key=lambda item: item.legal_id)
    ]
    body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    return "\n".join(header) + "\n" + body


def _pending_sync_path(root: Path) -> Path:
    """Return the one authorized declaration path beneath an exact checkout."""
    existing = iter_pending_file_paths(root)
    if existing:
        return existing[0]
    checkout = Path(root).resolve(strict=True)
    nested_checkout = checkout / checkout.name
    if nested_checkout.is_symlink():
        raise PendingDeclarationError(
            f"Pending sync nested checkout must not be a symlink: {nested_checkout}"
        )
    if nested_checkout.is_dir():
        return nested_checkout / PENDING_FILENAME
    return checkout / PENDING_FILENAME


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` atomically so an interrupted sync keeps a valid file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.is_file() else 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def sync_repo_pending(
    *,
    repo_root: Path,
    unmapped_legal_ids: list[str],
    source: str,
    since: str,
    issue: str | None = None,
) -> dict[str, Any]:
    """Atomically rewrite one exact checkout's pending declaration.

    Existing entries retain their provenance while still unmapped. Newly
    unmapped outputs receive ``source`` and ``since``; classified or removed
    outputs are drained. The ceiling is reset to the resulting entry count.
    """
    root = Path(repo_root)
    if not is_policy_repo_root(root):
        raise PendingDeclarationError(
            "Pending sync requires the exact canonical "
            f"rulespec-<country> checkout: {root}"
        )
    if source not in _ALLOWED_SOURCES:
        raise PendingDeclarationError(
            f"source {source!r} must be one of {sorted(_ALLOWED_SOURCES)}"
        )
    since = _coerce_date(since, where="sync")
    path = _pending_sync_path(root)
    existing_by_id: dict[str, PendingEntry] = {}
    if path.is_file():
        existing = load_pending_file(path, repo=path.parent.name)
        existing_by_id = {entry.legal_id: entry for entry in existing.entries}
        issue = issue if issue is not None else existing.issue

    wanted = sorted(set(unmapped_legal_ids))
    entries: list[PendingEntry] = []
    added: list[str] = []
    for legal_id in wanted:
        prior = existing_by_id.get(legal_id)
        if prior is not None:
            entries.append(prior)
            continue
        entries.append(
            PendingEntry(
                legal_id=legal_id,
                source=source,
                since=since,
                note=None,
                repo=path.parent.name,
                file=str(path),
            )
        )
        added.append(legal_id)
    dropped = sorted(set(existing_by_id) - set(wanted))

    text = _dump_pending_file(entries=entries, ceiling=len(entries), issue=issue)
    changed = not path.is_file() or path.read_text(encoding="utf-8") != text
    if changed:
        _atomic_write_text(path, text)
    return {
        "path": str(path),
        "count": len(entries),
        "added": added,
        "dropped": dropped,
        "changed": changed,
    }
