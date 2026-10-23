"""Check commencement bounds on executable rules grounded in amendments."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date

import yaml

_EXECUTABLE_KINDS = frozenset(
    {"parameter", "derived", "derived_relation", "data_relation"}
)
_VERSION_PATH = re.compile(r"^versions\[(-?\d+)\](?:\.|$)")
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
    Indexed proof paths must name an existing version before any evidence
    is scoped or excluded. Rule-level atoms naming a version apply only there;
    use these indexed paths to anchor principal history separately from
    amendment-derived behavior. Version-owned indexed paths must name their
    owning version. Both proof blocks add evidence conservatively and never
    erase authenticated evidence through ownership or block precedence.
    An amendment proof for a real, valid ``effective_to`` before commencement
    establishes the end of a historical era. Other cutoff spellings retain
    the amendment bound. Ambiguous
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
    except (yaml.YAMLError, ValueError, KeyError, AttributeError):
        # PyYAML raises KeyError or AttributeError for `!!bool` or
        # `!!timestamp` on other text; the YAML preflight reports those tags.
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
        proof_owners = [(f"rule `{rule_name}`", rule, None)]
        if isinstance(versions, list):
            proof_owners.extend(
                (f"rule `{rule_name}` versions[{index}]", version, index)
                for index, version in enumerate(versions)
                if isinstance(version, dict)
            )
        for owner_label, owner, owner_index in proof_owners:
            for atom_index, atom in enumerate(_proof_atoms(owner)):
                if not isinstance(atom, Mapping):
                    continue
                issue = indexed_proof_path_issue(
                    str(atom.get("path") or ""),
                    versions=versions,
                    label=f"{owner_label} proof atom {atom_index}",
                    owner_version_index=owner_index,
                )
                if issue is not None:
                    issues.append(issue)
        if not isinstance(versions, list) or not versions:
            versions = [rule]
        for version_index, version in enumerate(versions):
            version_mapping = version if isinstance(version, dict) else {}
            # A mismatched local index is rejected above. Keep its evidence
            # at the owner too, so even malformed ownership cannot erase it.
            version_atoms = _proof_atoms(version_mapping)
            atoms = version_atoms + [
                atom
                for atom in rule_atoms
                if _atom_applies_to_version(atom, version_index)
            ]
            label = f"rule `{rule_name}` versions[{version_index}]"
            for citation, commencement, commencement_citation in bounds:
                cited_paths = _source_paths(
                    atoms,
                    version=version_mapping,
                    version_index=version_index,
                    commencement=commencement,
                )
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
    proofs = [
        metadata.get("proof") if isinstance(metadata, Mapping) else None,
        value.get("proof"),
    ]
    return [
        atom
        for proof in proofs
        if isinstance(proof, Mapping) and isinstance(proof.get("atoms"), list)
        for atom in proof["atoms"]
    ]


def normalize_proof_location(path: str) -> str:
    """Share proof validation's single-version shorthand with timing checks."""
    normalized = re.sub(r"\s+", "", path).rstrip(".")
    return re.sub(r"^versions\.", "versions[0].", normalized)


def indexed_proof_path_issue(
    path: str,
    *,
    versions: object,
    label: str,
    owner_version_index: int | None = None,
) -> str | None:
    """Reject an indexed proof location before its index can exclude evidence.

    This applies to every anchor field, including dates and Boolean formulas.
    Field-specific shape checks remain the responsibility of proof validation.
    """
    normalized = normalize_proof_location(path)
    match = _VERSION_PATH.match(normalized)
    if match is None:
        return None
    version_index = int(match.group(1))
    if not isinstance(versions, list) or not 0 <= version_index < len(versions):
        return (
            f"Proof atom anchor invalid: {label} anchors `{normalized}`, "
            "but the rule has no such version."
        )
    if not isinstance(versions[version_index], Mapping):
        return (
            f"Proof atom anchor invalid: {label} anchors `{normalized}`, "
            "but that version is malformed."
        )
    if owner_version_index is not None and version_index != owner_version_index:
        return (
            f"Proof atom anchor invalid: {label} anchors `{normalized}`, "
            f"but its owning version is versions[{owner_version_index}]."
        )
    return None


def _atom_applies_to_version(atom: object, version_index: int) -> bool:
    if not isinstance(atom, Mapping):
        return False
    path = normalize_proof_location(str(atom.get("path") or ""))
    match = _VERSION_PATH.match(path)
    return match is None or int(match.group(1)) == version_index


def _local_atom_path(atom: object) -> str | None:
    if not isinstance(atom, Mapping) or not isinstance(atom.get("path"), str):
        return None
    path = normalize_proof_location(atom["path"])
    match = _VERSION_PATH.match(path)
    if match:
        path = path[match.end() :]
    return path or None


def _is_principal_history_cutoff(
    atom: object,
    *,
    version: Mapping[str, object],
    version_index: int,
    commencement: date | None,
) -> bool:
    if (
        _local_atom_path(atom) != "effective_to"
        or not _atom_applies_to_version(atom, version_index)
        or commencement is None
    ):
        return False
    effective_from = _parse_date(version.get("effective_from"))
    effective_to = _parse_date(version.get("effective_to"))
    return (
        effective_from is not None
        and effective_to is not None
        and effective_from <= effective_to < commencement
    )


def _source_paths(
    atoms: Sequence[object],
    *,
    version: Mapping[str, object],
    version_index: int,
    commencement: date | None,
) -> list[str]:
    paths: list[str] = []
    for atom in atoms:
        if _is_principal_history_cutoff(
            atom,
            version=version,
            version_index=version_index,
            commencement=commencement,
        ):
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
