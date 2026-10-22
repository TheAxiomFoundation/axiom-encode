"""Check commencement bounds on executable rules grounded in amendments."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date

import yaml

_EXECUTABLE_KINDS = frozenset(
    {"parameter", "derived", "derived_relation", "data_relation"}
)
_VERSION_PATH = re.compile(r"^versions\[(\d+)\](?:\.|$)")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


def amendment_timing_issues(
    content: str, amendment_documents: Sequence[object]
) -> list[str]:
    """Reject amendment-backed versions which predate their commencement.

    Amendment documents expose ``citation_path`` and ``metadata`` attributes;
    the latter binds ``commencement_date`` to authenticated corpus evidence.
    Documents without that field are supporting or legacy context, rather
    than commencement bounds; direct amendment requests require a bound
    before context assembly admits them.
    Principal-only versions remain free to retain their historical dates.
    Rule-level proof atoms naming a version apply only to that version;
    use these indexed paths to anchor principal history separately from
    amendment-derived behavior. Version proof adds evidence conservatively
    and never erases authenticated rule-level evidence.
    An amendment proof for ``effective_to`` establishes the end of an era,
    rather than deriving that era's earlier behavior. Ambiguous
    Act-level proof inherits the commencement bound of a contained amendment;
    exact citations to unrelated provisions avoid that ambiguity.

    Ordinary YAML/schema validation owns malformed RuleSpec documents, and
    proof validation owns source authentication. This check adds timing
    diagnostics only for identifiable executable rules.
    """
    if not amendment_documents:
        return []
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    bounds: list[tuple[str, date | None, str]] = []
    for document in amendment_documents:
        citation = str(getattr(document, "citation_path", "") or "").strip(" /")
        metadata = getattr(document, "metadata", None)
        if not citation:
            continue
        if not isinstance(metadata, Mapping) or "commencement_date" not in metadata:
            continue
        bounds.append(
            (
                citation,
                _parse_date(metadata.get("commencement_date")),
                str(metadata.get("commencement_citation_path") or "").strip(),
            )
        )

    issues: list[str] = []
    for rule_index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        if rule.get("kind") not in _EXECUTABLE_KINDS and not rule.get("versions"):
            continue
        rule_name = str(rule.get("name") or f"rules[{rule_index}]")
        rule_atoms = _proof_atoms(rule)
        versions = rule.get("versions")
        if not isinstance(versions, list) or not versions:
            versions = [rule]
        for version_index, version in enumerate(versions):
            version_mapping = version if isinstance(version, dict) else {}
            version_atoms = [
                atom
                for atom in _proof_atoms(version_mapping)
                if _atom_applies_to_version(atom, version_index)
            ]
            atoms = version_atoms + [
                atom
                for atom in rule_atoms
                if _atom_applies_to_version(atom, version_index)
            ]
            cited_paths = _source_paths(atoms)
            label = f"rule `{rule_name}` versions[{version_index}]"
            for citation, commencement, commencement_citation in bounds:
                if not any(
                    path == citation
                    or path.startswith(citation + "/")
                    or citation.startswith(path + "/")
                    for path in cited_paths
                ):
                    continue
                if commencement is None:
                    issues.append(
                        f"Amendment timing unresolved: {label} cites `{citation}`, "
                        "whose commencement_date is missing or invalid."
                    )
                    continue
                effective = _parse_date(version_mapping.get("effective_from"))
                evidence = (
                    f" (commencement evidence `{commencement_citation}`)"
                    if commencement_citation
                    else ""
                )
                if effective is None:
                    issues.append(
                        f"Amendment timing: {label} cites `{citation}` and must "
                        "declare a valid effective_from on or after "
                        f"{commencement.isoformat()}{evidence}."
                    )
                elif effective < commencement:
                    issues.append(
                        f"Amendment timing: {label} effective_from "
                        f"{effective.isoformat()} predates commencement "
                        f"{commencement.isoformat()} for `{citation}`{evidence}."
                    )
    return issues


def _parse_date(value: object) -> date | None:
    if type(value) is date:
        return value
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _proof_atoms(value: Mapping[str, object]) -> list[object]:
    metadata = value.get("metadata")
    proof = metadata.get("proof") if isinstance(metadata, Mapping) else None
    if not isinstance(proof, Mapping):
        proof = value.get("proof")
    atoms = proof.get("atoms") if isinstance(proof, Mapping) else None
    return atoms if isinstance(atoms, list) else []


def _atom_applies_to_version(atom: object, version_index: int) -> bool:
    if not isinstance(atom, Mapping):
        return False
    path = re.sub(r"\s+", "", str(atom.get("path") or ""))
    match = _VERSION_PATH.match(path)
    return match is None or int(match.group(1)) == version_index


def _local_atom_path(atom: object) -> str | None:
    if not isinstance(atom, Mapping) or not isinstance(atom.get("path"), str):
        return None
    path = re.sub(r"\s+", "", atom["path"]).rstrip(".")
    match = _VERSION_PATH.match(path)
    if match:
        path = path[match.end() :]
    elif path.startswith("versions."):
        path = path.removeprefix("versions.")
    return path or None


def _source_paths(atoms: Sequence[object]) -> list[str]:
    paths: list[str] = []
    for atom in atoms:
        if _local_atom_path(atom) == "effective_to":
            continue
        if path := _source_citation_path(atom):
            paths.append(path)
    return paths


def _source_citation_path(atom: object) -> str | None:
    source = atom.get("source") if isinstance(atom, Mapping) else None
    if not isinstance(source, Mapping):
        return None
    path = source.get("corpus_citation_path")
    if not isinstance(path, str):
        return None
    return path.strip(" /") or None
