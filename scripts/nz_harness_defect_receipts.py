"""Replay receipts for the NZ harness defects fixed against `BASE_REF`.

Compare pinned base modules with the current ones over every plain-text
fixture in `tests/fixtures`, from every jurisdiction. The base side is the
module text at `BASE_REF`, loaded from git, so each comparison is a true
two-implementation differential.

- Outlines: each fixture under one complete-section path per jurisdiction.
  Record the body hash, both outlines and, where they differ or for NZ
  bodies, both numeric recalls.
- Conjunctive fact gates: every line-and-sentence clause of every fixture.
  Record the clause count and each clause whose gates differ.

Usage: uv run python scripts/nz_harness_defect_receipts.py [--write]
Without `--write` the command exits nonzero if the committed receipts differ.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
RECEIPTS = FIXTURES / "nz_bare_labels" / "nz_harness_defect_receipts.json"
BASE_REF = "4bd60776cb624bd0f6d9df0e09950789026e9fa9"
BASE_MODULE = "src/axiom_encode/bare_structure.py"
BASE_COMPLETENESS = "src/axiom_encode/harness/source_completeness.py"
NZ_PREFIX = "nz_bare_labels/"
# One complete-section path per jurisdiction. Only the `.../section/<id>`
# shape matters to the outline; the prefix selects jurisdiction cleaners.
SECTION_PATHS = (
    "nz/statute/act/public/2018/0032/section/220",
    "uk/statute/example/section/15",
    "us/statute/example/section/15",
    "de/statute/example/section/15",
    "fr/statute/example/section/15",
    "dk/statute/example/section/15",
    "ca/statute/example/section/15",
)
# The named defects whose verdicts may change: full bare s220 outlines and
# the clause 6A counterfactual's ceteris paribus gate.
EXPECTED_CHANGES = frozenset(
    ("nz_bare_labels/s220.txt", path) for path in SECTION_PATHS
)
EXPECTED_GATE_CHANGES = frozenset(
    {"nz_bare_labels/amendment_2026_0054_s15_clause6a.txt"}
)
_SENTENCE = re.compile(r"[^.;!?]+[.;!?]?")


def load_base(
    ref: str = BASE_REF,
    relative: str = BASE_MODULE,
    name: str = "_nz_harness_base_bare_structure",
    package: str | None = None,
) -> ModuleType:
    source = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "show", f"{ref}:{relative}"],
        cwd=ROOT,
        text=True,
    )
    spec = importlib.util.spec_from_loader(name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__file__ = relative
    if package is not None:
        module.__package__ = package
    sys.modules[name] = module
    exec(compile(source, relative, "exec", dont_inherit=True), module.__dict__)
    return module


def load_base_completeness(ref: str = BASE_REF) -> ModuleType:
    return load_base(
        ref,
        BASE_COMPLETENESS,
        "axiom_encode.harness._nz_harness_base_source_completeness",
        "axiom_encode.harness",
    )


def fixture_clauses(text: str) -> list[str]:
    """Each whole line, then each sentence-like clause of it."""
    clauses: list[str] = []
    for line in text.splitlines():
        if line.strip():
            clauses.append(line.strip())
            clauses.extend(
                part.strip() for part in _SENTENCE.findall(line) if part.strip()
            )
    return list(dict.fromkeys(clauses))


def gate_record(gates: Any) -> list[list[list[str]]]:
    return [[sorted(entities), sorted(predicates)] for entities, predicates in gates]


def fixture_bodies() -> list[tuple[str, str]]:
    return [
        (path.relative_to(FIXTURES).as_posix(), path.read_text(encoding="utf-8"))
        for path in sorted(FIXTURES.rglob("*.txt"))
    ]


def outline_record(outline: Any) -> list[list[Any]] | None:
    if outline is None:
        return None
    return [
        [list(label.path), label.start, label.marker_end, label.end]
        for label in outline
    ]


@contextmanager
def base_masking(base: ModuleType) -> Iterator[None]:
    """Run numeric recall with the base bare-label mask in place."""
    from axiom_encode.harness import source_completeness as sc

    current = sc.mask_bare_labels

    def mask(text: str, outline: Any = None, *, corpus_citation_path: str = "") -> str:
        return base.mask_bare_labels(text, outline)

    sc.mask_bare_labels = mask
    try:
        yield
    finally:
        sc.mask_bare_labels = current


def recall_numbers(text: str, path: str) -> list[float]:
    from axiom_encode.harness import source_completeness as sc
    from axiom_encode.harness import validator_pipeline as vp

    cleaned = sc.authoritative_numeric_recall_text(text, corpus_citation_path=path)
    return [float(value) for value in vp.extract_numeric_occurrences_from_text(cleaned)]


def build_receipts(
    base: ModuleType | None = None, base_completeness: ModuleType | None = None
) -> dict[str, Any]:
    from axiom_encode import bare_structure as current
    from axiom_encode.harness import source_completeness as sc

    base = load_base() if base is None else base
    if base_completeness is None:
        base_completeness = load_base_completeness()
    rows = []
    for name, text in fixture_bodies():
        base_outline = base.bare_outline(text)
        for path in SECTION_PATHS:
            new_outline = current.bare_outline(text, corpus_citation_path=path)
            row: dict[str, Any] = {
                "fixture": name,
                "body_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "corpus_citation_path": path,
                "base_outline": outline_record(base_outline),
                "outline": outline_record(new_outline),
            }
            # Labels from the two module copies are distinct classes, so
            # compare their recorded values, never the objects.
            if row["base_outline"] != row["outline"] or name.startswith(NZ_PREFIX):
                with base_masking(base):
                    row["base_recall"] = recall_numbers(text, path)
                row["recall"] = recall_numbers(text, path)
            rows.append(row)
    clause_count = 0
    gate_rows = []
    for name, text in fixture_bodies():
        for clause in fixture_clauses(text):
            clause_count += 1
            before = base_completeness._source_conjunctive_fact_gates(clause)
            after = sc._source_conjunctive_fact_gates(clause)
            if before != after:
                gate_rows.append(
                    {
                        "fixture": name,
                        "clause": clause,
                        "base_gates": gate_record(before),
                        "gates": gate_record(after),
                    }
                )
    return {
        "base_ref": BASE_REF,
        "base_modules": [BASE_MODULE, BASE_COMPLETENESS],
        "rows": rows,
        "gate_clauses_compared": clause_count,
        "gate_changes": gate_rows,
    }


def committed_view(receipts: dict[str, Any]) -> dict[str, Any]:
    """The part of a replay worth committing: NZ bodies and every change.

    Unchanged rows of other jurisdictions' fixtures are recomputed by the
    tests on every run, so editing or adding one never needs new receipts.
    """
    changes = changed_rows(receipts)
    return {
        "base_ref": receipts["base_ref"],
        "base_modules": receipts["base_modules"],
        "rows": [
            row
            for row in receipts["rows"]
            if row["fixture"].startswith(NZ_PREFIX)
            or (row["fixture"], row["corpus_citation_path"]) in changes
        ],
        "gate_changes": receipts["gate_changes"],
    }


def changed_rows(receipts: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (row["fixture"], row["corpus_citation_path"])
        for row in receipts["rows"]
        if row["base_outline"] != row["outline"]
        or row.get("base_recall") != row.get("recall")
    }


def changed_gate_fixtures(receipts: dict[str, Any]) -> set[str]:
    return {row["fixture"] for row in receipts["gate_changes"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    receipts = build_receipts()
    view = committed_view(receipts)
    rendered = json.dumps(view, indent=1, sort_keys=True) + "\n"
    if args.write:
        RECEIPTS.write_text(rendered, encoding="utf-8")
        print(
            f"wrote {len(view['rows'])} of {len(receipts['rows'])} outline rows "
            f"and {len(view['gate_changes'])} gate changes"
        )
        return 0
    if RECEIPTS.read_text(encoding="utf-8") != rendered:
        print("committed receipts differ from a fresh replay", file=sys.stderr)
        return 1
    changes = changed_rows(receipts)
    if changes != EXPECTED_CHANGES:
        print(f"unexpected verdict changes: {sorted(changes ^ EXPECTED_CHANGES)}")
        return 1
    gate_changes = changed_gate_fixtures(receipts)
    if gate_changes != EXPECTED_GATE_CHANGES:
        print(
            f"unexpected gate changes: {sorted(gate_changes ^ EXPECTED_GATE_CHANGES)}"
        )
        return 1
    print(
        f"{len(receipts['rows'])} outline rows and "
        f"{receipts['gate_clauses_compared']} clauses match; only named defects change"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
