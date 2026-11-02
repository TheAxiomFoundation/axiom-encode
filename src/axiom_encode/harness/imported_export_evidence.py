"""Resolver snapshots and direct, source-owned entered-observation bindings.

These records never evaluate an import or grant a source-condition exemption.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from axiom_encode.printed_source_rows import recognize_printed_rows
from axiom_encode.rulespec_version_dates import version_date_fields


@dataclass(frozen=True)
class ResolvedImportedExport:
    import_uri: str
    canonical_target: str | None
    export_name: str
    resolved_path: str
    jurisdiction_root: str | None
    content_sha256: str
    content: str
    consumer_imports: tuple[str, ...]


@dataclass(frozen=True)
class EnteredObservationVersion:
    start: str
    end: str
    year: int
    frame_citation: str
    frame_sha256: str
    row_citation: str
    row_sha256: str
    row_span: tuple[int, int]
    predicate_citation: str
    predicate_sha256: str
    predicate_span: tuple[int, int]


@dataclass(frozen=True)
class BoundEnteredObservation:
    export_uri: str
    input_uri: str
    provider_path: str
    provider_sha256: str
    coordinate: int
    entity: str
    period: str
    versions: tuple[EnteredObservationVersion, ...]


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader: _UniqueLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError("Ambiguous YAML mapping")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
_URI = re.compile(
    r"[a-z][a-z0-9-]*:(?:policies|statutes|regulations)/[A-Za-z0-9_/-]+#[A-Za-z_][A-Za-z0-9_]*"
)
_ROW = re.compile(r"Line [1-9]\d{0,2} and (?:line )?([1-9]\d{4}): [^\r\n]+")
_BULLET = re.compile(
    r"You entered an amount for (?P<label>[A-Za-z][A-Za-z ()-]+?) on line (?P<first>[1-9]\d{4})"
    r"(?: or (?P<other>[A-Za-z][A-Za-z ()-]+?) on line (?P<second>[1-9]\d{4}))?"
    r" and the amount on line (?P<income>[1-9]\d{4}) is more than \$(?P<amount>\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _payload(text: str) -> dict[str, Any] | None:
    try:
        value = yaml.load(text, Loader=_UniqueLoader)
        return (
            value
            if isinstance(value, dict) and value.get("format") == "rulespec/v1"
            else None
        )
    except (yaml.YAMLError, ValueError, TypeError, RecursionError):
        return None


def _named(payload: Mapping[str, Any], field: str) -> dict[str, dict[str, Any]] | None:
    items = payload.get(field, [])
    if not isinstance(items, list):
        return None
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return None
        name = item["name"]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in result:
            return None
        result[name] = item
    return result


def _atoms(rule: Mapping[str, Any], path: str, kind: str) -> list[dict[str, Any]]:
    metadata = rule.get("metadata")
    proof = metadata.get("proof") if isinstance(metadata, dict) else None
    atoms = proof.get("atoms") if isinstance(proof, dict) else None
    return (
        [
            a
            for a in atoms
            if isinstance(a, dict) and a.get("path") == path and a.get("kind") == kind
        ]
        if isinstance(atoms, list)
        else []
    )


def _source(
    atom: Mapping[str, Any], context: Mapping[str, str | None]
) -> tuple[str, str, str, int] | None:
    source = atom.get("source")
    if not isinstance(source, dict):
        return None
    citation, excerpt = source.get("corpus_citation_path"), source.get("excerpt")
    if not isinstance(citation, str) or not isinstance(excerpt, str) or not excerpt:
        return None
    body = context.get(citation)
    if not isinstance(body, str) or body.count(excerpt) != 1:
        return None
    return citation, body, excerpt, body.index(excerpt)


def _frame(
    rule: Mapping[str, Any],
    index: int,
    version: Mapping[str, Any],
    context: Mapping[str, str | None],
) -> tuple[int, str, str] | None:
    date_fields = version_date_fields(version)
    if date_fields is None:
        return None
    frames = []
    for field, actual_field in zip(
        ("effective_from", "effective_to"), date_fields, strict=True
    ):
        atoms = _atoms(rule, f"versions[{index}].{actual_field}", "effective_period")
        if len(atoms) != 1 or (source := _source(atoms[0], context)) is None:
            return None
        citation, body, excerpt, start = source
        if (start and body[start - 1] != "\n") or (
            start + len(excerpt) < len(body)
            and body[start + len(excerpt)] not in "\r\n"
        ):
            return None
        header = re.fullmatch(r"[A-Z][A-Z0-9-]{0,20} ([1-9]\d{3})", excerpt)
        if header is None:
            return None
        year = int(header[1])
        expected = (
            f"{year:04d}-01-01" if field == "effective_from" else f"{year:04d}-12-31"
        )
        if version.get(actual_field) != expected:
            return None
        frames.append((year, citation, _sha(body)))
    return frames[0] if frames[0] == frames[1] else None


def _entered_roles(body: str) -> dict[str, frozenset[int]]:
    structure = recognize_printed_rows(body.strip())
    if structure is None:
        return {}
    roles: dict[str, frozenset[int]] = {}
    delta = len(body) - len(body.lstrip())
    for start, end in structure.atomic_spans:
        group = body[start + delta : end + delta]
        lines = [line[2:] for line in group.splitlines() if line.startswith("- ")]
        if (
            not 2 <= len(lines) <= 6
            or "if one or more of the following apply:" not in group
        ):
            continue
        matches = [_BULLET.fullmatch(line) for line in lines]
        if any(m is None for m in matches):
            continue
        for line, match in zip(lines, matches):
            assert match is not None
            labels = match["label"] + " " + (match["other"] or "")
            if re.search(
                r"\b(?:if|unless|provided|except|entered|more|less)\b", labels, re.I
            ):
                return {}
            if line in roles or body.count(line) != 1:
                return {}
            offset = body.index(line)
            if any(
                m.start() < offset + len(line) and offset < m.end()
                for m in re.finditer(r'"[^\"]*"|“[^”]*”', body)
            ):
                return {}
            roles[line] = frozenset(
                int(match[k]) for k in ("first", "second") if match[k]
            )
    return roles


def _canonical_record(record: ResolvedImportedExport) -> bool:
    if record.jurisdiction_root is None:
        return False
    root, path = Path(record.jurisdiction_root), Path(record.resolved_path)
    if not root.is_absolute() or not path.is_absolute() or ".." in path.parts:
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if (
        relative.suffix != ".yaml"
        or not relative.parts
        or relative.parts[0] != "policies"
    ):
        return False
    expected = f"{root.name}:{relative.with_suffix('').as_posix()}#{record.export_name}"
    return record.import_uri == record.canonical_target == expected


def bind_entered_observations(
    consumer_content: str,
    *,
    owner: str,
    consumer_citation: str,
    source_context: Mapping[str, str | None],
    imported_exports: Sequence[ResolvedImportedExport],
) -> tuple[BoundEnteredObservation, ...]:
    """Bind only transparent imported observations; no execution or exemption."""
    consumer = _payload(consumer_content)
    if consumer is None:
        return ()
    rules, inputs = _named(consumer, "rules"), _named(consumer, "inputs")
    imports = consumer.get("imports", [])
    if (
        rules is None
        or inputs is None
        or set(rules) & set(inputs)
        or owner not in rules
    ):
        return ()
    if not isinstance(imports, list) or any(
        not isinstance(i, str) or _URI.fullmatch(i) is None for i in imports
    ):
        return ()
    aliases = [i.rsplit("#", 1)[1] for i in imports]
    if len(set(aliases)) != len(aliases) or (set(aliases) & (set(rules) | set(inputs))):
        return ()
    if any(r.consumer_imports != tuple(imports) for r in imported_exports):
        return ()
    if len({r.import_uri for r in imported_exports}) != len(imported_exports):
        return ()
    module = consumer.get("module")
    verification = (
        module.get("source_verification") if isinstance(module, dict) else None
    )
    if (
        not isinstance(verification, dict)
        or verification.get("corpus_citation_path") != consumer_citation
    ):
        return ()
    snapshots: dict[str, tuple[str, str]] = {}
    modules: dict[str, str] = {}
    for record in imported_exports:
        snapshot = (record.content_sha256, record.content)
        if snapshots.setdefault(record.resolved_path, snapshot) != snapshot:
            return ()
        if record.canonical_target is not None:
            module_id = record.canonical_target.rsplit("#", 1)[0]
            if (
                modules.setdefault(module_id, record.resolved_path)
                != record.resolved_path
            ):
                return ()
    body = source_context.get(consumer_citation)
    roles = _entered_roles(body) if isinstance(body, str) else {}
    if not roles:
        return ()
    target = rules[owner]
    versions = target.get("versions")
    if (
        target.get("entity") != "Person"
        or target.get("period") != "Year"
        or not isinstance(versions, list)
        or not versions
    ):
        return ()
    target_frames = [
        _frame(target, i, v, source_context) if isinstance(v, dict) else None
        for i, v in enumerate(versions)
    ]
    if None in target_frames or len(set(target_frames)) != len(target_frames):
        return ()
    results = []
    for record in imported_exports:
        if (
            not _canonical_record(record)
            or record.import_uri not in imports
            or record.canonical_target != record.import_uri
            or record.export_name != record.import_uri.rsplit("#", 1)[1]
            or _sha(record.content) != record.content_sha256
        ):
            continue
        provider = _payload(record.content)
        if provider is None:
            continue
        exports, facts = _named(provider, "rules"), _named(provider, "inputs")
        if (
            exports is None
            or facts is None
            or set(exports) & set(facts)
            or record.export_name not in exports
        ):
            continue
        provider_imports = provider.get("imports", [])
        if not isinstance(provider_imports, list) or any(
            not isinstance(i, str) or _URI.fullmatch(i) is None
            for i in provider_imports
        ):
            continue
        provider_aliases = [i.rsplit("#", 1)[1] for i in provider_imports]
        if len(set(provider_aliases)) != len(provider_aliases) or set(
            provider_aliases
        ) & (set(exports) | set(facts)):
            continue
        rule = exports[record.export_name]
        vv = rule.get("versions")
        if (
            set(rule)
            - {
                "name",
                "kind",
                "dtype",
                "entity",
                "period",
                "versions",
                "metadata",
                "description",
                "source",
            }
            or (
                "source" in rule
                and (not isinstance(rule["source"], str) or not rule["source"].strip())
            )
            or rule.get("kind") != "derived"
            or rule.get("dtype") != "Judgment"
            or rule.get("entity") != "Person"
            or rule.get("period") != "Year"
            or "unit" in rule
            or not isinstance(vv, list)
            or not vv
        ):
            continue
        bound = []
        input_names, coordinates = set(), set()
        for index, version in enumerate(vv):
            if not isinstance(version, dict) or set(version) - {
                "formula",
                "effective_from",
                "effective_to",
                "from",
                "to",
            }:
                break
            formula = version.get("formula")
            try:
                node = (
                    ast.parse(formula, mode="eval").body
                    if isinstance(formula, str)
                    else None
                )
            except (SyntaxError, ValueError, RecursionError):
                node = None
            if not isinstance(node, ast.Name) or node.id not in facts:
                break
            fact = facts[node.id]
            if (
                set(fact) - {"name", "dtype", "entity", "period", "description"}
                or fact.get("dtype") != "Bool"
                or fact.get("entity") != "Person"
                or fact.get("period") != "Year"
            ):
                break
            frame = _frame(rule, index, version, source_context)
            if frame is None or frame not in target_frames:
                break
            row_atoms = _atoms(rule, f"versions[{index}].formula", "formula")
            predicates = _atoms(rule, f"versions[{index}].formula", "predicate")
            if len(row_atoms) != 1 or len(predicates) != 1:
                break
            row = _source(row_atoms[0], source_context)
            predicate = _source(predicates[0], source_context)
            if row is None or predicate is None:
                break
            row_citation, row_body, row_text, row_start = row
            predicate_citation, predicate_body, quote, predicate_start = predicate
            match = _ROW.fullmatch(row_text)
            module = provider.get("module")
            verification = (
                module.get("source_verification") if isinstance(module, dict) else None
            )
            if (
                not isinstance(verification, dict)
                or verification.get("corpus_citation_path") != row_citation
                or recognize_printed_rows(row_body.strip()) is None
                or match is None
                or predicate_citation != consumer_citation
                or int(match[1]) not in roles.get(quote, ())
            ):
                break
            expected_row_citation = record.import_uri.rsplit("#", 1)[0].replace(
                ":policies/", "/policy/"
            )
            if (
                row_citation != expected_row_citation
                or row_citation.rsplit("/", 1)[0] != consumer_citation.rsplit("/", 1)[0]
                or frame[1].rsplit("/", 1)[0] != consumer_citation.rsplit("/", 1)[0]
            ):
                break
            coordinate = int(match[1])
            # The row must be a physical row, not a matching quoted substring.
            if (row_start and row_body[row_start - 1] != "\n") or (
                row_start + len(row_text) < len(row_body)
                and row_body[row_start + len(row_text)] not in "\r\n"
            ):
                break
            bound.append(
                EnteredObservationVersion(
                    str(version.get("effective_from", version.get("from"))),
                    str(version.get("effective_to", version.get("to"))),
                    frame[0],
                    frame[1],
                    frame[2],
                    row_citation,
                    _sha(row_body),
                    (row_start, row_start + len(row_text)),
                    predicate_citation,
                    _sha(predicate_body),
                    (predicate_start, predicate_start + len(quote)),
                )
            )
            input_names.add(node.id)
            coordinates.add(coordinate)
        if (
            len(bound) != len(vv)
            or len(input_names) != 1
            or len(coordinates) != 1
            or len({v.year for v in bound}) != len(bound)
            or {v.year for v in bound} != {f[0] for f in target_frames if f is not None}
        ):
            continue
        results.append(
            BoundEnteredObservation(
                record.import_uri,
                record.import_uri.rsplit("#", 1)[0] + "#" + next(iter(input_names)),
                record.resolved_path,
                record.content_sha256,
                next(iter(coordinates)),
                "Person",
                "Year",
                tuple(bound),
            )
        )
    # A single primitive observation cannot acquire two different printed roles
    # merely because two direct exports provide different proof metadata.
    identities: dict[str, tuple[int, tuple[EnteredObservationVersion, ...]]] = {}
    for result in results:
        identity = (result.coordinate, result.versions)
        previous = identities.setdefault(result.input_uri, identity)
        if previous != identity:
            return ()
    return tuple(results)
