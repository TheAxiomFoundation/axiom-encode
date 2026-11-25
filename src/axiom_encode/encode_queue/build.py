"""Derive the membership of a package. Nobody maintains the unit list by hand.

    units of a package =
        rows of the pinned corpus under the in-scope documents that the
        program bundle (``manifests/program-bundles/<program>.yaml`` in the
        corpus checkout) lists for the jurisdiction
      - units that are already on the RuleSpec main checkout
        (the module file exists, or a module cites the unit, a parent of it
        or a part of it)

A unit with an open candidate pull request stays in the list with
``open_pr`` set, so the queue shows it as a candidate and never hands it out.

All inputs are local checkouts and one JSON file. The build makes no network
call and no model call.
"""

from __future__ import annotations

import bisect
import json
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from axiom_encode.encode_queue import model
from axiom_encode.encode_queue.model import QueueError
from axiom_encode.rulespec_citation_paths import citation_rulespec_path

MIN_SOURCE_CHARS = 150
MAX_SOURCE_CHARS = 3000
COMPUTATION_PARTS = frozenset({"Income", "Deductions", "Benefit amount", "Assets"})

_SKIP_HEADING = re.compile(
    r"^(Legal Authorit|Examples?$|Table of Contents|Contents$|Index$|Revision"
    r"|Quick Access)",
    re.I,
)
_MONEY_OR_PERCENT = re.compile(r"\$\s?\d|\d\s?%|percent", re.I)
_RULESPEC_FORMAT = re.compile(r"^format:\s*rulespec/", re.M)
_CITATION = re.compile(r"corpus_citation_path:\s*[\"']?([^\s\"']+)")
_CITATION_BLOCK = re.compile(r"corpus_citation_paths:\s*\n((?:\s*-\s*\S+\n?)+)")
_CITATION_ITEM = re.compile(r"-\s*[\"']?([^\s\"']+)")


def load_bundle(corpus_root: Path, program: str) -> dict[str, Any]:
    path = corpus_root / "manifests" / "program-bundles" / f"{program}.yaml"
    if not path.is_file():
        raise QueueError(f"the corpus checkout has no program bundle for {program!r}")
    bundle = yaml.safe_load(path.read_text())
    if not isinstance(bundle, dict) or not isinstance(bundle.get("layers"), list):
        raise QueueError(f"program bundle {program!r} has no layers")
    return bundle


def in_scope_documents(
    bundle: dict[str, Any], jurisdiction: str
) -> list[dict[str, Any]]:
    """The in-scope documents of one jurisdiction layer, one entry per key."""

    documents: dict[str, dict[str, Any]] = {}
    for layer in bundle["layers"]:
        if layer.get("jurisdiction") != jurisdiction:
            continue
        for value in layer.values():
            if not isinstance(value, list):
                continue
            for document in value:
                if not isinstance(document, dict) or "key" not in document:
                    continue
                entry = documents.setdefault(document["key"], dict(document))
                if document.get("scope") == "in":
                    entry["scope"] = "in"
    return [d for d in documents.values() if d.get("scope") == "in"]


def index_corpus(
    corpus_root: Path, jurisdictions: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Index the provision rows of the given jurisdictions. No body is kept."""

    provisions = corpus_root / "data" / "corpus" / "provisions"
    index: dict[str, dict[str, Any]] = {}
    for jurisdiction in sorted(set(jurisdictions)):
        directory = provisions / jurisdiction
        if not directory.is_dir():
            continue
        for file in sorted(directory.rglob("*.jsonl")):
            with file.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    citation = row.get("citation_path")
                    if not isinstance(citation, str):
                        continue
                    body = row.get("body") or ""
                    index[citation] = {
                        "chars": len(body),
                        "heading": (row.get("heading") or "")[:80],
                        "money": bool(_MONEY_OR_PERCENT.search(body)),
                    }
    return index


def scan_rulespec(rulespec_root: Path) -> tuple[set[str], set[str]]:
    """Module files and cited corpus paths of a RuleSpec checkout."""

    modules: set[str] = set()
    cited: set[str] = set()
    for file in sorted(rulespec_root.rglob("*.yaml")):
        relative = file.relative_to(rulespec_root)
        if relative.parts[0].startswith(".") or file.name.endswith(".test.yaml"):
            continue
        text = file.read_text(errors="replace")
        if not _RULESPEC_FORMAT.search(text):
            continue
        modules.add(relative.as_posix())
        cited.update(match.group(1) for match in _CITATION.finditer(text))
        for block in _CITATION_BLOCK.finditer(text):
            cited.update(_CITATION_ITEM.findall(block.group(1)))
    return modules, cited


def open_pr_paths(open_prs: list[dict[str, Any]]) -> dict[str, int]:
    """Primary RuleSpec path -> lowest open pull request that touches it.

    ``open_prs`` is the output of
    ``gh pr list --state open --json number,files`` on the RuleSpec repo.
    """

    paths: dict[str, int] = {}
    for pull in open_prs:
        number = int(pull["number"])
        for file in pull.get("files") or []:
            path = file.get("path", "")
            if path.endswith(".yaml") and not path.endswith(".test.yaml"):
                paths[path] = min(number, paths.get(path, number))
    return paths


def _under(sorted_paths: list[str], document: str) -> list[str]:
    rows = []
    start = bisect.bisect_left(sorted_paths, document)
    if start < len(sorted_paths) and sorted_paths[start] == document:
        rows.append(document)
    prefix = document + "/"
    position = bisect.bisect_left(sorted_paths, prefix)
    while position < len(sorted_paths) and sorted_paths[position].startswith(prefix):
        rows.append(sorted_paths[position])
        position += 1
    return rows


def _on_main(citation: str, cited: set[str], sorted_cited: list[str]) -> bool:
    if citation in cited:
        return True
    parts = citation.split("/")
    if any("/".join(parts[:n]) in cited for n in range(len(parts) - 1, 2, -1)):
        return True
    position = bisect.bisect_left(sorted_cited, citation + "/")
    return position < len(sorted_cited) and sorted_cited[position].startswith(
        citation + "/"
    )


def derive_package(
    program: str,
    jurisdiction: str,
    *,
    bundle: dict[str, Any],
    corpus_index: dict[str, dict[str, Any]],
    main_modules: set[str],
    main_cited: set[str],
    open_prs: dict[str, int],
    now: datetime,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """The units document of one package."""

    package = model.package_id(program, jurisdiction)
    documents = in_scope_documents(bundle, jurisdiction)
    sorted_paths = sorted(corpus_index)
    sorted_cited = sorted(main_cited)
    held = 0
    missing: list[str] = []
    on_main = 0
    candidates: dict[str, dict[str, Any]] = {}
    for document in documents:
        root = document.get("citation_path")
        rows = (
            [p for p in _under(sorted_paths, root) if corpus_index[p]["chars"] > 0]
            if root
            else []
        )
        if not rows:
            missing.append(document["key"])
            continue
        held += 1
        sized = {
            p
            for p in rows
            if MIN_SOURCE_CHARS <= corpus_index[p]["chars"] <= MAX_SOURCE_CHARS
        }
        for citation in rows:
            if citation not in sized or citation in candidates:
                continue
            parts = citation.split("/")
            # Keep the largest unit: skip a row whose parent is itself a unit.
            if any("/".join(parts[:n]) in sized for n in range(len(parts) - 1, 2, -1)):
                continue
            row = corpus_index[citation]
            if _SKIP_HEADING.search(row["heading"].strip()):
                continue
            try:
                target = citation_rulespec_path(citation).as_posix()
            except ValueError:
                continue
            if target in main_modules or _on_main(citation, main_cited, sorted_cited):
                on_main += 1
                continue
            unit = {
                "citation": citation,
                "target_path": target,
                "document": root,
                "part": document.get("part") or "",
                "heading": row["heading"],
                "source_chars": row["chars"],
                "money_or_percent": row["money"],
            }
            if target in open_prs:
                unit["open_pr"] = open_prs[target]
            candidates[citation] = unit
    ordered = sorted(
        candidates.values(),
        key=lambda u: (
            not u["money_or_percent"],
            u["part"] not in COMPUTATION_PARTS,
            not 300 <= u["source_chars"] <= 2500,
            u["citation"],
        ),
    )
    for rank, unit in enumerate(ordered, start=1):
        unit["rank"] = rank
    result: dict[str, Any] = {
        "schema": model.UNITS_SCHEMA,
        "package": package,
        "built_at": model.format_time(now),
        "inputs": inputs,
        "documents": {
            "in_scope": len(documents),
            "held": held,
            "missing": sorted(missing),
        },
        "on_main": on_main,
        "units": ordered,
    }
    if not documents:
        result["blocked"] = (
            f"program bundle: no in-scope document for {jurisdiction} in the "
            f"{program} bundle; add the documents to the bundle in axiom-corpus"
        )
    elif not held:
        result["blocked"] = (
            f"corpus: none of the {len(documents)} in-scope documents is in the "
            "pinned corpus release; a release that holds them and a pin move "
            "unblock it"
        )
    return result
