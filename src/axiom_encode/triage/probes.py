"""The five deterministic probes of the failure-triage evidence bundle.

See docs/design/encode-failure-triage.md. Each probe reads only the run's
recorded material and two read-only views (a RuleSpec state and a local
corpus checkout). No probe calls a model, the network, the rules engine or
the encoder's validation, and no probe changes what an encode does.

Every probe returns the same shape::

    {"status": "ok" | "skipped", "reason": str | None,
     "counts": {finding status: n}, "findings": [...],
     "truncated_findings": n}

and every finding carries ``refs`` that point into the evidence bundle (see
``axiom_encode.triage.evidence.resolve_ref``).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from ..concepts.registry import ConceptRegistry
from ..harness.dependency_stubs import UnsafeRulespecContextPath
from ..harness.pre_encode_gate import (
    PRE_ENCODE_CHECKS,
    GateSource,
    evaluate_pre_encode_gate,
    explicit_federal_references,
)
from .issues import MAX_SIGNATURE_CHARS, issue_signature, parse_issue
from .material import TriageMaterial, module_key, split_module_path
from .views import LocalCorpusIndex, RulespecSnapshot

PROBE_NAMES = ("gate", "registry", "corpus", "import", "persistence")
MAX_FINDINGS_PER_PROBE = 64
MAX_CORPUS_LOOKUPS = 24
MAX_REGISTRY_TARGETS = 64

_TARGET_KEY_RE = re.compile(r"^(?P<prefix>[a-z]{2}(?:-[a-z0-9]+)*):(?P<path>[^#\s]+)")
_CORPUS_CITATION_IN_YAML = re.compile(
    r"corpus_citation_path:\s*['\"]?(?P<path>[a-z]{2}[a-z0-9-]*/[^\s'\"#,}\]]+)"
)
_COMPILE_MODULE_PATTERNS = (
    re.compile(r"failed to load RuleSpec module `(?P<path>[^`]+)`"),
    re.compile(r"compile failed: (?P<path>[^\s`]+\.ya?ml):"),
    re.compile(r"`(?P<path>[^`\s]*rulespec-[a-z]+/[^`\s]+\.ya?ml)`"),
)
_QUOTED = re.compile(r"'[^'\n]*'|\"[^\"\n]*\"")
_CHECKOUT_PATH = re.compile(
    r"rulespec-[a-z]+/(?P<prefix>[a-z]{2}(?:-[a-z0-9]+)*)/(?P<rest>.+)"
)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _probe_result(
    findings: Sequence[dict[str, Any]],
    *,
    status: str = "ok",
    reason: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    counts = Counter(str(item.get("status")) for item in findings)
    kept = list(findings[:MAX_FINDINGS_PER_PROBE])
    result: dict[str, Any] = {
        "status": status,
        "reason": reason,
        "counts": dict(sorted(counts.items())),
        "findings": kept,
        "truncated_findings": max(0, len(findings) - len(kept)),
    }
    if extra:
        result.update(extra)
    return result


def _skipped(reason: str, **extra: Any) -> dict[str, Any]:
    return _probe_result([], status="skipped", reason=reason, extra=extra or None)


def offset_ref(name: str, text: str, index: int) -> str:
    """Return ``<name>:L<n>`` for the line holding character ``index``."""

    return f"{name}:L{text.count(chr(10), 0, index) + 1}"


def line_ref(
    name: str, text: str | None, needle: str | None, start: int = 0
) -> str | None:
    """Return ``<name>:L<n>`` for the first line of ``text`` holding ``needle``."""

    if not text or not needle:
        return None
    index = text.find(needle, start)
    if index < 0:
        return None
    return offset_ref(name, text, index)


def _refs(*items: str | None) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _target_key(target: str, policy_prefix: str | None) -> str | None:
    """Return the canonical ``prefix:path`` key of a RuleSpec target."""

    text = target.split("#", 1)[0].strip().strip("'\"")
    match = _TARGET_KEY_RE.match(text)
    if match is not None:
        return f"{match.group('prefix')}:{match.group('path').removesuffix('.yaml')}"
    if policy_prefix and text and ":" not in text:
        return f"{policy_prefix}:{text.strip('/').removesuffix('.yaml')}"
    return None


def _load_yaml(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        payload = yaml.safe_load(text)
    except (yaml.YAMLError, RecursionError, ValueError, TypeError):
        # Constructor errors (an invalid date) raise plain ValueError.
        return {}
    return payload if isinstance(payload, dict) else {}


MAX_YAML_NODES = 200_000


class YamlScalar(NamedTuple):
    path: tuple[Any, ...]
    value: str
    line: int  # 1-based line where the scalar starts
    start: int  # character span of the scalar in the source text
    end: int


MAX_YAML_DEPTH = 64


def yaml_scalars(text: str | None) -> list[YamlScalar] | None:
    """Every scalar of a YAML document with its key path and source span.

    An alias node is walked once (a recursive alias is not expanded).
    Returns ``None`` when the text does not compose as one safe YAML
    document or exceeds the node or depth bound; callers then fall back to a
    lexical reading.
    """

    if not text:
        return []
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except (yaml.YAMLError, RecursionError):
        # Deep nesting exhausts the parser's recursion before any bound here.
        return None
    if root is None:
        return []
    found: list[YamlScalar] = []
    # Each entry carries its ancestors: a node is skipped only when it is its
    # own ancestor (a recursive alias), so a reused anchor is read at every
    # path where it occurs.
    stack: list[tuple[tuple[Any, ...], Any, frozenset[int]]] = [((), root, frozenset())]
    seen = 0
    while stack:
        path, node, ancestors = stack.pop()
        seen += 1
        if seen > MAX_YAML_NODES or len(path) > MAX_YAML_DEPTH:
            return None
        if isinstance(node, yaml.ScalarNode):
            found.append(
                YamlScalar(
                    path,
                    str(node.value),
                    node.start_mark.line + 1,
                    node.start_mark.index,
                    node.end_mark.index,
                )
            )
            continue
        if id(node) in ancestors:
            continue
        inner = ancestors | {id(node)}
        if isinstance(node, yaml.SequenceNode):
            for index, child in reversed(list(enumerate(node.value))):
                stack.append(((*path, index), child, inner))
        elif isinstance(node, yaml.MappingNode):
            for key, child in reversed(node.value):
                label = key.value if isinstance(key, yaml.ScalarNode) else "?"
                stack.append(((*path, label), child, inner))
    return found


def _is_blocker_path(path: tuple[Any, ...]) -> bool:
    tail = path[-4:]
    return (
        len(tail) == 4
        and tail[0] == "deferred_outputs"
        and isinstance(tail[1], int)
        and tail[2] == "blocked_by"
        and isinstance(tail[3], int)
        and path[:-4] in {(), ("module",)}
    )


# --------------------------------------------------------------------------
# Gate probe
# --------------------------------------------------------------------------


def _corpus_labels(
    material: TriageMaterial, corpus: LocalCorpusIndex | None
) -> tuple[tuple[str, ...], str | None]:
    """Row labels for the gate, and the repealed status of the target row."""

    labels: list[str] = []
    repealed: str | None = None
    citation_path = material.citation_path
    if corpus is not None and citation_path:
        chosen = _target_rows(corpus.rows(citation_path), _attested_version(material))
        for row in chosen:
            labels.extend(
                value for value in (row.heading, row.citation_label, row.title) if value
            )
            if row.status == "repealed":
                repealed = (
                    f"local corpus row {citation_path} ({row.version}) has "
                    "metadata.status repealed"
                )
    title = (material.provision_metadata or {}).get("title")
    if isinstance(title, str) and title.strip():
        labels.append(title)
    return tuple(dict.fromkeys(labels)), repealed


def _target_rows(rows: Sequence[Any], attested: str | None) -> list[Any]:
    """The target's own rows: the attested version only, when one is attested.

    Without an attestation a single row is taken; several rows of different
    versions select none. Another version is never substituted for the
    attested one.
    """

    if attested:
        return [row for row in rows if row.version == attested]
    return list(rows) if len(rows) == 1 else []


def _attested_version(material: TriageMaterial) -> str | None:
    attestation = material.source_attestation or {}
    row = attestation.get("row")
    if isinstance(row, Mapping) and isinstance(row.get("version"), str):
        return row["version"]
    return None


def _target_module_relative(material: TriageMaterial) -> str | None:
    from ..harness.evals import _resolve_eval_output_path

    if not material.policy_prefix or not material.citation:
        return None
    try:
        relative = _resolve_eval_output_path(material.citation).as_posix()
    except ValueError:
        return None
    return f"{material.policy_prefix}/{relative}"


def cited_module_status(snapshot: RulespecSnapshot | None, cited_path: str) -> str:
    """Whether a RuleSpec module exists at or under a cited corpus path.

    Mirrors the gate's ``_module_exists`` on the sparse snapshot.
    """

    from ..harness.evals import _resolve_eval_output_path
    from ..harness.pre_encode_gate import _module_exists

    if snapshot is None:
        return "not_checked"
    try:
        relative = _resolve_eval_output_path(cited_path).as_posix()
    except ValueError:
        return "not_checked"
    jurisdiction = cited_path.split("/", 1)[0]
    file_path = f"{jurisdiction}/{relative}"
    snapshot.materialize(file_path)
    snapshot.materialize_dir_marker(file_path.removesuffix(".yaml"))
    directory = file_path.removesuffix(".yaml") + "/"
    if any(
        item == file_path or item.startswith(directory) for item in snapshot.unreadable
    ):
        return "unreadable"
    content_root = snapshot.checkout_root / jurisdiction
    return "present" if _module_exists(content_root, Path(relative)) else "absent"


def probe_gate(
    material: TriageMaterial,
    *,
    snapshot_factory: Callable[[], RulespecSnapshot | None] | None = None,
    corpus: LocalCorpusIndex | None = None,
) -> dict[str, Any]:
    """Re-run the Stage 1 pre-encode gate in report mode on the run's target."""

    if material.source_text is None or not material.citation:
        return _skipped("the run has no recorded source text")
    labels, repealed = _corpus_labels(material, corpus)
    source = GateSource(
        requested=material.citation,
        citation_path=material.citation_path or material.citation,
        body=material.source_text,
        labels=labels,
        amendment_context=bool(
            material.amendment_request
            or material.amendment_documents
            or material.dropped_amendment_documents
        ),
    )
    snapshot = snapshot_factory() if snapshot_factory is not None else None
    # unencoded_reference needs the run's release-bound active corpus rows,
    # which a local checkout cannot establish; the corpus probe reports each
    # cited row and whether a module exists for it instead.
    not_run: dict[str, str] = {
        "unencoded_reference": "the run's active corpus scope is not available"
    }
    try:
        policy_repo_path: Path | None = None
        target_module = _target_module_relative(material)
        if snapshot is not None and target_module is not None:
            snapshot.materialize(target_module)
            if snapshot.unreadable:
                not_run["existing_module"] = "the target module could not be read"
            else:
                policy_repo_path = snapshot.content_root(str(material.policy_prefix))
        else:
            not_run["existing_module"] = "no RuleSpec state for the run"
        gate = evaluate_pre_encode_gate(
            source,
            mode="report",
            policy_repo_path=policy_repo_path,
            replacement_target=material.replacement_target,
            repealed_reason=repealed,
        )
    finally:
        if snapshot is not None:
            snapshot.close()
    findings = [
        {
            "kind": finding.check,
            "status": "finding",
            "subject": material.citation_path or material.citation,
            "class": finding.failure_class,
            "subclass": finding.subclass,
            "enforceable": PRE_ENCODE_CHECKS[finding.check][1],
            "detail": finding.reason,
            "evidence": list(finding.evidence),
            "refs": _refs("target", "source"),
        }
        for finding in gate.findings
    ]
    return _probe_result(
        findings,
        extra={
            "checks_not_run": sorted(not_run),
            "checks_not_run_reasons": dict(sorted(not_run.items())),
            "label_count": len(labels),
            "amendment_context": source.amendment_context,
        },
    )


# --------------------------------------------------------------------------
# Registry probe
# --------------------------------------------------------------------------


def _deferral_blockers(payload: Mapping[str, Any]) -> list[str]:
    records: list[Any] = []
    module = payload.get("module")
    if isinstance(module, Mapping) and isinstance(module.get("deferred_outputs"), list):
        records.extend(module["deferred_outputs"])
    if isinstance(payload.get("deferred_outputs"), list):
        records.extend(payload["deferred_outputs"])
    blockers: list[str] = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("blocked_by"), list
        ):
            continue
        blockers.extend(
            str(item).strip()
            for item in record["blocked_by"]
            if str(item or "").strip()
        )
    return blockers


def _free_formula_identifiers(
    payload: Mapping[str, Any], imported: set[str]
) -> list[str]:
    from ..harness.validator_pipeline import _formula_local_identifiers

    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []
    local = {
        str(rule.get("name") or "").strip()
        for rule in rules
        if isinstance(rule, Mapping)
    }
    free: set[str] = set()
    for rule in rules:
        if not isinstance(rule, Mapping) or not isinstance(rule.get("versions"), list):
            continue
        for version in rule["versions"]:
            if isinstance(version, Mapping) and isinstance(version.get("formula"), str):
                free.update(_formula_local_identifiers(version["formula"]))
    return sorted(free - local - imported)


def _registry_targets(
    material: TriageMaterial, registry: ConceptRegistry | None
) -> list[dict[str, Any]]:
    from ..harness.validator_pipeline import _extract_import_items_from_content

    assert material.candidate is not None
    content = material.candidate.rulespec
    payload = _load_yaml(content)
    targets: list[dict[str, Any]] = []
    imported_symbols: set[str] = set()
    scalars = yaml_scalars(content)
    imports: list[tuple[str, int | None]]
    blockers: list[tuple[str, int | None]]
    if scalars is not None:
        # The parsed document, in every accepted form (block or flow
        # sequence, alias mapping), with each item's own line.
        imports = [
            (item.value.strip(), item.line)
            for item in scalars
            if item.path
            and item.path[0] == "imports"
            and len(item.path) >= 2
            and item.value.strip()
        ]
        blockers = [
            (item.value.strip(), item.line)
            for item in scalars
            if _is_blocker_path(item.path) and item.value.strip()
        ]
    else:
        # Malformed YAML: the validator's own lexical reading of imports.
        imports = [(item, None) for item in _extract_import_items_from_content(content)]
        blockers = [(item, None) for item in _deferral_blockers(payload)]
    for item, line in imports:
        base, _sep, symbol = item.partition("#")
        symbol = symbol.strip() or None
        if symbol:
            imported_symbols.add(symbol)
        targets.append(
            {
                "origin": "import",
                "target": item,
                "path": base.strip(),
                "symbol": symbol,
                "line": line,
            }
        )
    for blocker, line in blockers:
        base, _sep, symbol = blocker.partition("#")
        targets.append(
            {
                "origin": "deferral_blocked_by",
                "target": blocker,
                "path": base.strip(),
                "symbol": symbol.strip() or None,
                "line": line,
            }
        )
    formula_spans = [
        (item.start, item.end)
        for item in scalars or ()
        if item.path and item.path[-1] == "formula"
    ]

    def formula_line(name: str) -> int | None:
        # Searched in each formula's own source span, so block and folded
        # scalars keep their physical lines.
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
        for start, end in formula_spans:
            span = _QUOTED.sub(lambda m: " " * len(m.group(0)), content[start:end])
            # The scalar's own opening quote is not a string literal.
            if content[start : start + 1] in {"'", '"'}:
                span = _QUOTED.sub(
                    lambda m: " " * len(m.group(0)), content[start + 1 : end - 1]
                )
                span = " " + span + " "
            match = pattern.search(span)
            if match is not None:
                return content.count("\n", 0, start + match.start()) + 1
        return None

    if registry is not None:
        for name in _free_formula_identifiers(payload, imported_symbols):
            concept = registry.concept_for_name(name)
            if concept is None:
                continue
            canonical = concept.canonical_name
            anchors = concept.producer_anchors if concept.has_producer else ()
            if not anchors:
                targets.append(
                    {
                        "origin": "input",
                        "target": name,
                        "path": None,
                        "symbol": canonical,
                        "concept": concept.id,
                        "blocked_synonym": canonical != name,
                        "line": formula_line(name),
                    }
                )
            for anchor in anchors:
                targets.append(
                    {
                        "origin": "input",
                        "target": f"{anchor}#{canonical}",
                        "path": anchor,
                        "symbol": canonical,
                        "concept": concept.id,
                        "input_name": name,
                        "blocked_synonym": canonical != name,
                        "line": formula_line(name),
                    }
                )
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for target in targets:
        unique.setdefault((target["origin"], target["target"]), target)
    return list(unique.values())


def _module_key_from_relative(relative: str) -> str | None:
    """``us/statutes/7/a.yaml`` -> ``us:statutes/7/a``."""

    prefix, _sep, rest = relative.partition("/")
    if not rest or not _TARGET_KEY_RE.match(f"{prefix}:{rest}"):
        return None
    return f"{prefix}:{rest.removesuffix('.yaml')}"


def _resolve_target(
    target: Mapping[str, Any],
    material: TriageMaterial,
    snapshot: RulespecSnapshot,
) -> tuple[str, str | None]:
    from ..harness.validator_pipeline import (
        _resolve_rulespec_import_file_static,
        _rulespec_import_file_candidates_static,
        _rulespec_rule_names_from_content,
    )

    assert material.candidate is not None and material.policy_prefix
    path = target.get("path")
    if not path:
        # Only a registered input can lack a producer; an import or blocker
        # with no module path is an invalid target.
        if target.get("origin") == "input":
            return "no_producer_registered", None
        return "unresolvable_target", None
    policy_root = snapshot.content_root(material.policy_prefix)
    split = split_module_path(material.candidate.module_path, material.policy_prefix)
    rules_file = (
        snapshot.content_root(split[0]) / split[1]
        if split is not None
        else policy_root / material.candidate.module_path
    )
    try:
        candidates = list(
            _rulespec_import_file_candidates_static(
                path,
                rules_file=rules_file,
                policy_repo_path=policy_root,
                rulespec_dependency_roots=(),
            )
        )
        if not candidates:
            return "unresolvable_target", None
        # In the resolver's own order: the first candidate that exists wins,
        # so an unreadable one matters only before that.
        for candidate in candidates:
            snapshot.materialize_path(candidate)
            if snapshot.relative(candidate) in snapshot.unreadable:
                return "module_unreadable", None
            if candidate.is_file():
                break
        resolved = _resolve_rulespec_import_file_static(
            path,
            rules_file=rules_file,
            policy_repo_path=policy_root,
            rulespec_dependency_roots=(),
        )
    except (UnsafeRulespecContextPath, ValueError, OSError):
        return "unresolvable_target", None
    if resolved is None:
        return "module_missing", None
    module = snapshot.relative(resolved)
    try:
        text = resolved.read_bytes().decode("utf-8")
    except (OSError, UnicodeError):
        return "module_unreadable", module
    symbol = target.get("symbol")
    if not symbol:
        return "module_present", module
    try:
        names = _rulespec_rule_names_from_content(text)
    except RecursionError:
        return "module_unreadable", module
    return ("exported" if symbol in names else "symbol_missing"), module


_REGISTRY_SECTIONS = {
    "import": "imports",
    "deferral_blocked_by": "blocked_by",
    "input": "rules:",
}


def _registry_ref(content: str, target: Mapping[str, Any]) -> str | None:
    """The candidate line of the occurrence a target came from."""

    if target.get("line"):
        return f"candidate:L{target['line']}"
    section = content.find(_REGISTRY_SECTIONS[target["origin"]])
    start = max(section, 0)
    if target["origin"] == "input":
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(target['input_name'] if 'input_name' in target else target['target'])}(?![A-Za-z0-9_])"
        )
        match = pattern.search(content, start) or pattern.search(content)
        return offset_ref("candidate", content, match.start()) if match else None
    return line_ref("candidate", content, target["target"], start) or line_ref(
        "candidate", content, target["target"]
    )


def probe_registry(
    material: TriageMaterial,
    *,
    snapshot_factory: Callable[[], RulespecSnapshot | None] | None = None,
    registry: ConceptRegistry | None = None,
) -> dict[str, Any]:
    """Check every import, deferral provider and registered input at the ref."""

    if material.candidate is None:
        return _skipped("the run has no final candidate")
    targets = _registry_targets(material, registry)
    truncated_targets = max(0, len(targets) - MAX_REGISTRY_TARGETS)
    targets = targets[:MAX_REGISTRY_TARGETS]
    snapshot = (
        snapshot_factory()
        if snapshot_factory is not None and material.policy_prefix
        else None
    )
    workspace = set(material.workspace_context)
    outputs = set(material.run_outputs)
    findings: list[dict[str, Any]] = []
    try:
        for target in targets:
            key = (
                _target_key(target["path"], material.policy_prefix)
                if target.get("path")
                else None
            )
            if snapshot is None:
                status, module = "not_checked", None
            else:
                status, module = _resolve_target(target, material, snapshot)
            if module:
                # The module the resolver picked (it may be a relative
                # fallback), not the lexical target.
                key = _module_key_from_relative(module) or key
            finding = {
                "kind": target["origin"],
                "status": status,
                "subject": target["target"],
                "module": module,
                "symbol": target.get("symbol"),
                "in_workspace_context": key in workspace if key else False,
                "authored_by_run": key in outputs if key else False,
                "refs": _refs(_registry_ref(material.candidate.rulespec, target)),
            }
            for extra in ("concept", "input_name", "blocked_synonym"):
                if extra in target:
                    finding[extra] = target[extra]
            findings.append(finding)
    finally:
        if snapshot is not None:
            snapshot.close()
    reason = None
    if snapshot is None:
        reason = (
            "no RuleSpec state at the run's ref is available locally"
            if material.policy_prefix
            else "the run has no policy prefix"
        )
    return _probe_result(
        findings,
        reason=reason,
        extra={
            "target_count": len(targets),
            "truncated_targets": truncated_targets,
        },
    )


# --------------------------------------------------------------------------
# Corpus probe
# --------------------------------------------------------------------------


def _federal_reference_offsets(source: str) -> dict[str, int]:
    """First offset of each citation ``explicit_federal_references`` returns."""

    from ..harness.pre_encode_gate import _CFR_TEXT_RE, _USC_TEXT_RE

    offsets: dict[str, int] = {}
    for pattern, label in ((_USC_TEXT_RE, "USC"), (_CFR_TEXT_RE, "CFR")):
        for match in pattern.finditer(source):
            citation = f"{match.group('title')} {label} {match.group('section')}"
            offsets[citation] = min(offsets.get(citation, match.start()), match.start())
    return offsets


def _corpus_subjects(material: TriageMaterial) -> list[tuple[str, str, str | None]]:
    """Return ``(citation path, origin, ref)`` for each referenced citation."""

    from ..harness.evals import _citation_to_corpus_citation_path

    subjects: list[tuple[str, str, str | None]] = []
    if material.citation_path:
        subjects.append((material.citation_path, "target", "target"))
    source = material.source_text or ""
    offsets = _federal_reference_offsets(source)
    for citation in explicit_federal_references(source):
        try:
            path = _citation_to_corpus_citation_path(citation)
        except ValueError:
            continue
        index = offsets.get(citation)
        subjects.append(
            (
                path,
                "source_reference",
                offset_ref("source", source, index) if index is not None else None,
            )
        )
    if material.candidate is not None:
        content = material.candidate.rulespec
        scalars = yaml_scalars(content)
        if scalars is not None:
            for item in scalars:
                if item.path and item.path[-1] == "corpus_citation_path":
                    if item.value.strip():
                        subjects.append(
                            (
                                item.value.strip(),
                                "candidate_citation",
                                f"candidate:L{item.line}",
                            )
                        )
        else:
            for match in _CORPUS_CITATION_IN_YAML.finditer(content):
                subjects.append(
                    (
                        match.group("path"),
                        "candidate_citation",
                        offset_ref("candidate", content, match.start("path")),
                    )
                )
    unique: dict[str, tuple[str, str, str | None]] = {}
    for subject in subjects:
        unique.setdefault(subject[0], subject)
    return list(unique.values())


def probe_corpus(
    material: TriageMaterial,
    *,
    corpus: LocalCorpusIndex | None = None,
    snapshot_factory: Callable[[], RulespecSnapshot | None] | None = None,
) -> dict[str, Any]:
    """Look up every referenced citation in the local corpus checkout.

    For each USC/CFR reference in the source it also says whether a RuleSpec
    module exists at or under the cited path at the run's ref (the facts the
    gate's ``unencoded_reference`` combines).
    """

    subjects = _corpus_subjects(material)
    if not subjects:
        return _skipped("the run names no citation")
    if corpus is None:
        return _skipped(
            "no local corpus checkout was given", subject_count=len(subjects)
        )
    attested = _attested_version(material)
    attested_present: bool | None = None
    attestation = material.source_attestation or {}
    row = attestation.get("row")
    if isinstance(row, Mapping) and attested:
        attested_present = corpus.version_present(
            str(row.get("jurisdiction") or ""),
            str(row.get("document_class") or ""),
            attested,
        )
    findings = []
    truncated = max(0, len(subjects) - MAX_CORPUS_LOOKUPS)
    snapshot = snapshot_factory() if snapshot_factory is not None else None
    try:
        for path, origin, ref in subjects[:MAX_CORPUS_LOOKUPS]:
            findings.append(
                _corpus_finding(corpus, path, origin, ref, attested, snapshot)
            )
    finally:
        if snapshot is not None:
            snapshot.close()
    return _probe_result(
        findings,
        reason=("rows are read from the local corpus checkout, not a verified release"),
        extra={
            "attested_version": attested,
            "attested_version_present_locally": attested_present,
            "truncated_subjects": truncated,
        },
    )


def _corpus_finding(
    corpus: LocalCorpusIndex,
    path: str,
    origin: str,
    ref: str | None,
    attested: str | None,
    snapshot: RulespecSnapshot | None,
) -> dict[str, Any]:
    rows = corpus.rows(path)
    selected = _target_rows(rows, attested if origin == "target" else None)
    if not selected and not corpus.complete_for(path):
        # Part of the bucket could not be read: absence is unknown.
        status = "unavailable"
    elif not rows:
        status = "absent"
    elif origin == "target" and attested and not selected:
        # Rows exist, but not at the version the run read.
        status = "absent_at_attested_version"
    elif not selected:
        status = (
            "repealed"
            if all(r.status == "repealed" for r in rows)
            else "present_multiple_versions"
        )
    else:
        status = "repealed" if selected[0].status == "repealed" else "present"
    finding: dict[str, Any] = {
        "kind": origin,
        "status": status,
        "subject": path,
        "rows": [r.to_dict() for r in rows[:4]],
        "row_count": len(rows),
        "refs": _refs(ref),
    }
    if origin == "source_reference":
        finding["rulespec_module"] = cited_module_status(snapshot, path)
    return finding


# --------------------------------------------------------------------------
# Import probe
# --------------------------------------------------------------------------


def _compile_module_key(text: str) -> str | None:
    """The module a compile error is about.

    The engine wraps a dependency's error in the compilation root's
    ``failed to load RuleSpec module <root>``; the innermost (last) module
    named is the one at fault.
    """

    found: list[tuple[int, str]] = []
    for pattern in _COMPILE_MODULE_PATTERNS:
        for match in pattern.finditer(text):
            located = _CHECKOUT_PATH.search(match.group("path"))
            if located is not None:
                rest = located.group("rest").removesuffix(".yaml").removesuffix(".yml")
                found.append((match.start("path"), f"{located.group('prefix')}:{rest}"))
    return max(found)[1] if found else None


def _is_compile_issue(text: str, gate: str | None) -> bool:
    lowered = text.lower()
    return gate == "compile" or "compile failed" in lowered


def probe_import(material: TriageMaterial) -> dict[str, Any]:
    """Place each compile error in the target's own file or another module."""

    if not material.final_issues:
        return _skipped("the run has no recorded final issues")
    parsed = [
        (index, parse_issue(text)) for index, text in enumerate(material.final_issues)
    ]
    compile_issues = [
        (index, issue)
        for index, issue in parsed
        if _is_compile_issue(issue.body, issue.gate)
    ]
    target = material.target_key
    siblings = set()
    if material.candidate is not None and material.policy_prefix:
        siblings = {
            key
            for path in material.candidate.sibling_modules
            if (key := module_key(path, material.policy_prefix))
        }
    outputs = set(material.run_outputs)
    workspace = set(material.workspace_context)
    findings = []
    for index, issue in compile_issues:
        key = _compile_module_key(issue.body)
        if key is None:
            status = "unlocated"
        elif target is not None and key == target:
            status = "own_module"
        elif key in siblings or key in outputs:
            status = "authored_sibling"
        elif key in workspace:
            status = "workspace_context_module"
        else:
            status = "imported_not_authored"
        findings.append(
            {
                "kind": "compile_error",
                "status": status,
                "subject": key,
                "target": target,
                "refs": _refs(f"issue:final:{index}"),
            }
        )
    return _probe_result(findings, extra={"compile_issue_count": len(compile_issues)})


# --------------------------------------------------------------------------
# Persistence probe
# --------------------------------------------------------------------------


def probe_persistence(material: TriageMaterial) -> dict[str, Any]:
    """Split issues into those every attempt had and those that changed.

    ``persistent`` needs the complete history: every attempt of the run is
    recorded with its whole issue list. Otherwise an issue seen in every
    recorded attempt is only ``in_all_recorded``.
    """

    recorded = [attempt for attempt in material.attempts if attempt.issues is not None]
    if len(recorded) < 2:
        return _skipped(
            "fewer than two attempts have recorded issues",
            attempts_with_issues=len(recorded),
            attempt_count=material.attempt_count,
        )
    per_attempt: list[dict[str, list[int]]] = []
    tags_per_attempt: list[set[str]] = []
    for attempt in recorded:
        signatures: dict[str, list[int]] = {}
        tags: set[str] = set()
        for index, text in enumerate(attempt.issues or ()):
            signatures.setdefault(issue_signature(text, None), []).append(index)
            tags.add(parse_issue(text).tag or "untagged")
        per_attempt.append(signatures)
        tags_per_attempt.append(tags)
    numbers = [attempt.attempt for attempt in material.attempts]
    complete = (
        len(recorded) == len(material.attempts)
        and numbers == list(range(1, len(numbers) + 1))
        and (material.attempt_count is None or material.attempt_count == len(numbers))
        and all(
            attempt.issue_count is None
            or attempt.issue_count <= len(attempt.issues or ())
            for attempt in recorded
        )
    )
    in_all = "persistent" if complete else "in_all_recorded"
    union: set[str] = set().union(*(set(item) for item in per_attempt))
    persistent = set.intersection(*(set(item) for item in per_attempt))
    final_signatures = {
        issue_signature(text, None): index
        for index, text in enumerate(material.final_issues)
    }
    findings = []
    for signature in sorted(union, key=lambda item: (item not in persistent, item)):
        refs = [
            f"attempt:{attempt.attempt}:issue:{signatures[signature][0]}"
            for attempt, signatures in zip(recorded, per_attempt, strict=True)
            if signature in signatures
        ]
        if signature in final_signatures:
            refs.append(f"issue:final:{final_signatures[signature]}")
        findings.append(
            {
                "kind": "issue",
                "status": in_all if signature in persistent else "changed",
                "subject": signature[:MAX_SIGNATURE_CHARS],
                "signature_sha256": hashlib.sha256(signature.encode()).hexdigest(),
                "tag": parse_issue(signature).tag,
                "attempts_present": sum(
                    signature in signatures for signatures in per_attempt
                ),
                "refs": refs,
            }
        )
    tags_in_all = sorted(set.intersection(*tags_per_attempt))
    return _probe_result(
        findings,
        extra={
            "attempts_with_issues": len(recorded),
            "history_complete": complete,
            "persistent_tags": tags_in_all if complete else [],
            "tags_in_all_recorded": tags_in_all,
        },
    )
