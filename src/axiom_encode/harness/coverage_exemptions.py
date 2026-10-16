"""Reviewed coverage exceptions bound to the exact text the checklist scans.

There is deliberately no source interpretation here. The scanner still emits
main's demands; only a literal registry key and its literal window can remove
one. Offsets and windows count Python Unicode characters; hashes cover UTF-8.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from importlib.resources import files
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

REASONS = frozenset(
    {"owned_by_other_provision", "repealed_tombstone", "nested_under_numbered_parent"}
)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_window(source_text: str, physical_start: int) -> str:
    return source_text[max(0, physical_start - 80) : physical_start + 80]


def _literal(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "*?[]"):
        raise ValueError(f"{field} must be a nonempty literal without wildcards")
    return value


def _digest(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{field} must be a lowercase sha256")
    return value


def _offset(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class CoverageDemand:
    label: str
    body: str
    physical_start: int


@dataclass(frozen=True)
class CoverageExemption:
    citation_path: str
    requested_scope: tuple[str, ...] | None
    source_text_sha256: str
    label: str
    physical_start: int
    window_sha256: str
    window: str
    reason: str
    owner_evidence: Mapping[str, Any]
    corpus_artifact: str
    artifact_sha256: str
    row: int
    reviewer: str
    reviewed_at: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CoverageExemption:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("coverage exemption has missing or unknown fields")
        data = dict(value)
        for field in ("citation_path", "label", "corpus_artifact"):
            _literal(data[field], field)
        artifact = PurePosixPath(data["corpus_artifact"])
        if artifact.is_absolute() or ".." in artifact.parts:
            raise ValueError("corpus_artifact must be a relative corpus path")
        scope = data["requested_scope"]
        if scope is not None:
            if not isinstance(scope, (list, tuple)) or not scope:
                raise ValueError(
                    "requested_scope must be null or a nonempty label list"
                )
            data["requested_scope"] = tuple(
                _literal(x, "requested_scope") for x in scope
            )
        for field in ("source_text_sha256", "window_sha256", "artifact_sha256"):
            _digest(data[field], field)
        _offset(data["physical_start"], "physical_start")
        _offset(data["row"], "row", 1)
        if data["reason"] not in REASONS:
            raise ValueError("unknown coverage exemption reason")
        window = data["window"]
        if not isinstance(window, str) or not window or len(window) > 160:
            raise ValueError("window must contain at most 160 source characters")
        if text_sha256(window) != data["window_sha256"]:
            raise ValueError("window_sha256 does not match window")
        owner = data["owner_evidence"]
        if not isinstance(owner, Mapping) or set(owner) != {
            "citation_path",
            "physical_start",
        }:
            raise ValueError(
                "owner_evidence needs an owner citation and physical_start"
            )
        _literal(owner["citation_path"], "owner_evidence.citation_path")
        _offset(owner["physical_start"], "owner_evidence.physical_start")
        if not isinstance(data["reviewer"], str) or not data["reviewer"].strip():
            raise ValueError("reviewer is required")
        if not isinstance(data["reviewed_at"], str):
            raise ValueError("reviewed_at is required")
        date.fromisoformat(data["reviewed_at"])
        return cls(**data)

    @property
    def key(self) -> tuple[str, tuple[str, ...] | None, str, str, int]:
        return (
            self.citation_path,
            self.requested_scope,
            self.source_text_sha256,
            self.label,
            self.physical_start,
        )


@lru_cache(maxsize=1)
def load_coverage_exemptions() -> tuple[CoverageExemption, ...]:
    raw = (
        files("axiom_encode.harness")
        .joinpath("data/coverage_exemptions.jsonl")
        .read_text(encoding="utf-8")
    )
    entries = tuple(
        CoverageExemption.from_mapping(json.loads(line))
        for line in raw.splitlines()
        if line
    )
    if len({entry.key for entry in entries}) != len(entries):
        raise ValueError("duplicate coverage exemption key")
    return entries


def _matching_entries(
    *,
    citation_path: str,
    requested_scope: tuple[str, ...] | None,
    source_text: str,
    exemptions: Iterable[CoverageExemption] | None,
) -> dict[tuple[str, int], CoverageExemption]:
    digest = text_sha256(source_text)
    return {
        (entry.label, entry.physical_start): entry
        for entry in (load_coverage_exemptions() if exemptions is None else exemptions)
        if entry.citation_path == citation_path
        and entry.requested_scope == requested_scope
        and entry.source_text_sha256 == digest
        and 0 <= entry.physical_start < len(source_text)
        and entry.window == source_window(source_text, entry.physical_start)
        and entry.window_sha256
        == text_sha256(source_window(source_text, entry.physical_start))
    }


def filter_coverage_demands(
    demands: Iterable[CoverageDemand],
    *,
    citation_path: str,
    requested_scope: tuple[str, ...] | None,
    source_text: str,
    exemptions: Iterable[CoverageExemption] | None = None,
) -> list[CoverageDemand]:
    matches = _matching_entries(
        citation_path=citation_path,
        requested_scope=requested_scope,
        source_text=source_text,
        exemptions=exemptions,
    )
    return [d for d in demands if (d.label, d.physical_start) not in matches]


def has_reviewed_repeal(
    *,
    citation_path: str,
    requested_scope: tuple[str, ...],
    source_text: str,
) -> bool:
    if not requested_scope:
        return False
    return any(
        entry.reason == "repealed_tombstone" and entry.label == requested_scope[0]
        for entry in _matching_entries(
            citation_path=citation_path,
            requested_scope=requested_scope,
            source_text=source_text,
            exemptions=None,
        ).values()
    )
