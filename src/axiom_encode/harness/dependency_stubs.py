"""Shared planning and materialization for canonical dependency stubs."""

from __future__ import annotations

import contextlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from axiom_encode.repo_routing import canonical_rulespec_repo_name


@dataclass(frozen=True)
class ResolvedDefinedTerm:
    """One canonical legal term resolved to an import target."""

    term: str
    import_target: str
    symbol: str
    citation: str
    entity: str
    period: str
    dtype: str
    label: str


@dataclass(frozen=True)
class ResolvedCanonicalConcept:
    """One high-confidence nearby canonical concept resolved from the corpus."""

    term: str
    import_target: str
    symbol: str
    citation: str
    entity: str
    period: str
    dtype: str
    label: str
    rulespec_file: Path


_REGISTERED_DEFINED_TERM_PATTERNS: tuple[
    tuple[re.Pattern[str], ResolvedDefinedTerm], ...
] = (
    (
        re.compile(r"\bmixed-age couple\b", re.IGNORECASE),
        ResolvedDefinedTerm(
            term="mixed-age couple",
            import_target="legislation/ukpga/2002/16/section/3ZA/3#is_member_of_mixed_age_couple",
            symbol="is_member_of_mixed_age_couple",
            citation="State Pension Credit Act 2002 section 3ZA(3)",
            entity="Person",
            period="Day",
            dtype="Boolean",
            label=(
                "`mixed-age couple` -> import "
                "`legislation/ukpga/2002/16/section/3ZA/3#is_member_of_mixed_age_couple` "
                "(State Pension Credit Act 2002 section 3ZA(3))"
            ),
        ),
    ),
)

_REGISTERED_STUBS_BY_KEY = {
    (term.import_target.split("#", 1)[0], term.symbol): term
    for _, term in _REGISTERED_DEFINED_TERM_PATTERNS
}
_EMBEDDED_SOURCE_PATTERN = re.compile(r'\s*"""(.*?)"""\s*', re.DOTALL)
_QUOTED_DEFINITION_PATTERN = re.compile(
    r"[\"“]([^\"”]+)[\"”](?:\s+or\s+[\"“]([^\"”]+)[\"”])?\s+means\b",
    re.IGNORECASE,
)
_SOURCE_ROOT_SEGMENTS = {"legislation", "statutes", "regulation"}
_CORPUS_PROVISION_KINDS = {
    "guidance",
    "legislation",
    "policy",
    "regulation",
    "regulations",
    "statute",
    "statutes",
}


def resolve_defined_terms_from_text(text: str) -> list[ResolvedDefinedTerm]:
    """Resolve registered legally-defined terms mentioned in source text."""
    resolved: list[ResolvedDefinedTerm] = []
    for pattern, term in _REGISTERED_DEFINED_TERM_PATTERNS:
        if pattern.search(text) and term not in resolved:
            resolved.append(term)
    return resolved


def resolve_canonical_concepts_from_text(
    text: str,
    corpus_root: Path,
    *,
    current_file: Path | None = None,
) -> list[ResolvedCanonicalConcept]:
    """Resolve high-confidence reusable legal concepts from nearby corpus files."""
    index: dict[str, list[ResolvedCanonicalConcept]] = {}

    for candidate_file in sorted(corpus_root.rglob("*.yaml")):
        if candidate_file.name.endswith(".test.yaml"):
            continue
        if (
            current_file is not None
            and candidate_file.resolve() == current_file.resolve()
        ):
            continue

        candidate = _build_canonical_concept_candidate(candidate_file, corpus_root)
        if candidate is None:
            continue

        for term in _extract_defined_concept_terms_from_source(
            _extract_embedded_source_text(candidate_file.read_text())
        ):
            index.setdefault(term, []).append(_with_concept_term(candidate, term))

    resolved: list[ResolvedCanonicalConcept] = []
    seen_targets: set[tuple[str, str]] = set()

    for term, candidates in index.items():
        if len(candidates) != 1:
            continue
        if not _source_text_mentions_term(text, term):
            continue

        candidate = candidates[0]
        key = (candidate.import_target, candidate.symbol)
        if key in seen_targets:
            continue
        seen_targets.add(key)
        resolved.append(candidate)

    return resolved


def find_registered_stub_specs(
    import_path: str,
    symbol_names: Sequence[str],
) -> list[ResolvedDefinedTerm]:
    """Return registered canonical stub specs for one unresolved import target."""
    normalized_import = import_path.strip().strip('"').strip("'")
    if normalized_import.endswith((".yaml", ".yml")):
        normalized_import = str(Path(normalized_import).with_suffix(""))
    specs: list[ResolvedDefinedTerm] = []
    for symbol in symbol_names:
        spec = _REGISTERED_STUBS_BY_KEY.get((normalized_import, symbol))
        if spec is None:
            return []
        specs.append(spec)
    return specs


def import_target_to_relative_rulespec_path(import_target: str) -> Path:
    """Convert an import target like legislation/...#name into a .yaml path."""
    normalized = import_target.strip().strip('"').strip("'").split("#", 1)[0]
    if ":" in normalized:
        prefix, rest = normalized.split(":", 1)
        if re.fullmatch(r"[a-z][a-z0-9-]*", prefix) and rest:
            normalized = rest
    if normalized.endswith((".yaml", ".yml")):
        return Path(normalized)
    return Path(f"{normalized}.yaml")


def build_registered_stub_content(specs: Sequence[ResolvedDefinedTerm]) -> str:
    """Return deterministic stub file content for one registered dependency file."""
    if not specs:
        raise ValueError("At least one stub spec is required")

    base_paths = {
        str(Path(spec.import_target.split("#", 1)[0]).with_suffix(""))
        if spec.import_target.split("#", 1)[0].endswith((".yaml", ".yml"))
        else spec.import_target.split("#", 1)[0]
        for spec in specs
    }
    if len(base_paths) != 1:
        raise ValueError(
            "All registered stub specs must belong to the same target file"
        )

    summary = (
        f"Canonical definition stub for `{specs[0].term}`. "
        f"Resolved to {specs[0].citation}."
        if len(specs) == 1
        else "Canonical definition stubs. Resolved to "
        + ", ".join(sorted({spec.citation for spec in specs}))
        + "."
    )
    payload = {
        "format": "rulespec/v1",
        "module": {
            "summary": summary,
            "status": "stub",
        },
        "rules": [
            {
                "name": spec.symbol,
                "kind": "derived",
                "entity": spec.entity,
                "period": spec.period,
                "dtype": spec.dtype,
                "versions": [
                    {
                        "effective_from": "0001-01-01",
                        "formula": "false" if spec.dtype == "Boolean" else "0",
                    }
                ],
            }
            for spec in specs
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False)


def materialize_registered_stub(
    root: Path,
    specs: Sequence[ResolvedDefinedTerm],
    *,
    prefix: Path | None = None,
) -> Path:
    """Write one deterministic canonical stub file under the given root."""
    if not specs:
        raise ValueError("At least one stub spec is required")

    relative_path = import_target_to_relative_rulespec_path(specs[0].import_target)
    target = (
        root / prefix / relative_path if prefix is not None else root / relative_path
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_registered_stub_content(specs))
    return target


def rulespec_content_has_stub_status(content: str) -> bool:
    """Return whether a RuleSpec file declares `module.status: stub`."""
    with contextlib.suppress(
        yaml.YAMLError, TypeError, ValueError, AttributeError, NameError
    ):
        payload = yaml.safe_load(content)
        if isinstance(payload, dict):
            module = payload.get("module")
            if isinstance(module, dict):
                return str(module.get("status", "")).strip() == "stub"
    return False


def rulespec_file_has_stub_status(rules_file: Path) -> bool:
    """Return whether a RuleSpec file exists and declares `module.status: stub`."""
    try:
        return rules_file.exists() and rulespec_content_has_stub_status(
            rules_file.read_text()
        )
    except OSError:
        return False


def has_corpus_provision_for_import_target(
    import_target: str, rules_repo_root: Path
) -> bool:
    """Return whether the import target has a local corpus.provisions row."""
    return bool(find_corpus_provision_artifacts(import_target, rules_repo_root))


def find_corpus_provision_artifacts(
    import_target: str,
    rules_repo_root: Path,
) -> list[Path]:
    """Locate corpus.provisions files containing the import target."""
    artifacts: list[Path] = []
    seen: set[Path] = set()
    citation_paths = _candidate_corpus_paths_for_import_target(
        import_target,
        rules_repo_root,
    )
    if not citation_paths:
        return []

    for provisions_root in _candidate_corpus_provisions_roots(rules_repo_root):
        for citation_path in citation_paths:
            for candidate in _candidate_corpus_provision_files(
                provisions_root,
                citation_path,
            ):
                if not _corpus_file_contains_citation_path(candidate, citation_path):
                    continue
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    artifacts.append(resolved)

    return artifacts


def _candidate_corpus_provisions_roots(rules_repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    env_root = os.environ.get("AXIOM_CORPUS_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    root = Path(rules_repo_root).expanduser().resolve()
    candidates.extend([root.parent / "axiom-corpus", root.parent / "corpus"])

    provisions_roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        base = candidate.expanduser()
        possible_roots = (
            base,
            base / "provisions",
            base / "data" / "corpus",
            base / "data" / "corpus" / "provisions",
        )
        for possible_root in possible_roots:
            provisions_root = (
                possible_root
                if possible_root.name == "provisions"
                else possible_root / "provisions"
            )
            if not provisions_root.is_dir():
                continue
            resolved = provisions_root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                provisions_roots.append(resolved)
    return provisions_roots


def _candidate_corpus_paths_for_import_target(
    import_target: str,
    rules_repo_root: Path,
) -> tuple[str, ...]:
    normalized = import_target.strip().strip('"').strip("'").split("#", 1)[0]
    if normalized.endswith((".yaml", ".yml")):
        normalized = str(Path(normalized).with_suffix(""))
    normalized = normalized.strip("/")
    if not normalized:
        return ()

    if _looks_like_corpus_citation_path(normalized):
        primary = normalized
    else:
        jurisdiction = _jurisdiction_for_rules_repo(rules_repo_root)
        relative = normalized
        if ":" in normalized:
            prefix, relative = normalized.split(":", 1)
            if prefix.strip():
                jurisdiction = prefix.strip()

        parts = Path(relative).parts
        if not parts:
            return ()
        kind, rest = _corpus_kind_and_rest(parts)
        primary = "/".join((jurisdiction, kind, *rest))

    candidates: list[str] = []

    def add(candidate: str) -> None:
        cleaned = candidate.strip().strip("/")
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    add(primary)
    parts = primary.split("/")
    for end in range(len(parts) - 1, 2, -1):
        add("/".join(parts[:end]))
    return tuple(candidates)


def _looks_like_corpus_citation_path(identifier: str) -> bool:
    parts = identifier.split("/")
    return len(parts) >= 3 and parts[1] in _CORPUS_PROVISION_KINDS


def _jurisdiction_for_rules_repo(rules_repo_root: Path) -> str:
    name = Path(rules_repo_root).resolve().name
    if name.startswith("rulespec-"):
        return name.removeprefix("rulespec-")
    return "us"


def _corpus_kind_and_rest(parts: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    head = parts[0]
    if head in {"statute", "statutes"}:
        return "statute", parts[1:]
    if head in {"regulation", "regulations"}:
        return "regulation", parts[1:]
    if head in {"policy", "policies"}:
        return "policy", parts[1:]
    if head in {"guidance", "legislation"}:
        return head, parts[1:]
    return "policy", parts


def _candidate_corpus_provision_files(
    provisions_root: Path,
    citation_path: str,
) -> list[Path]:
    parts = citation_path.split("/")
    if len(parts) < 2:
        return []
    bucket = provisions_root / parts[0] / parts[1]
    if not bucket.is_dir():
        return []
    return sorted(bucket.glob("*.jsonl"))


def _corpus_file_contains_citation_path(
    provision_file: Path,
    citation_path: str,
) -> bool:
    with contextlib.suppress(OSError):
        for line in provision_file.read_text().splitlines():
            if not line.strip():
                continue
            with contextlib.suppress(json.JSONDecodeError):
                payload = json.loads(line)
                if payload.get("citation_path") == citation_path:
                    return True
    return False


def _extract_embedded_source_text(content: str) -> str:
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if isinstance(payload, dict):
            module = payload.get("module")
            if isinstance(module, dict):
                summary = module.get("summary")
                if isinstance(summary, str) and summary.strip():
                    return summary.strip()
    match = _EMBEDDED_SOURCE_PATTERN.match(content)
    return match.group(1).strip() if match else ""


def _extract_defined_concept_terms_from_source(source_text: str) -> list[str]:
    terms: list[str] = []
    for match in _QUOTED_DEFINITION_PATTERN.finditer(source_text):
        for group in match.groups():
            raw = " ".join((group or "").split()).strip()
            normalized = _normalize_concept_term(group)
            if not normalized or not _is_high_confidence_term(raw):
                continue
            if normalized not in terms:
                terms.append(normalized)
    return terms


def _normalize_concept_term(term: str | None) -> str:
    if not term:
        return ""
    return " ".join(term.split()).strip().lower()


def _is_high_confidence_term(raw_term: str) -> bool:
    if not raw_term:
        return False
    if len(raw_term.split()) >= 2:
        return True
    return raw_term.isupper()


def _source_text_mentions_term(source_text: str, term: str) -> bool:
    if not term:
        return False
    escaped = re.escape(term)
    patterns = (
        rf"[\"“]{escaped}[\"”]",
        rf"(?im)^\s*{escaped}(?=$|[\s),.;:])",
        rf"\b(?:a|an|the|this|that|these|those|such|any|each|every)\s+{escaped}(?=$|[\s),.;:])",
        rf"\b(?:of|for|to|by|in|from|with|under|on|into)\s+(?:(?:a|an|the|this|that|these|those)\s+)?{escaped}(?=$|[\s),.;:])",
    )
    return any(
        re.search(pattern, source_text, flags=re.IGNORECASE) for pattern in patterns
    )


def _build_canonical_concept_candidate(
    candidate_file: Path,
    corpus_root: Path,
) -> ResolvedCanonicalConcept | None:
    content = candidate_file.read_text()
    source_text = _extract_embedded_source_text(content)
    if not source_text:
        return None

    metadata = _extract_principal_symbol_metadata(content)
    if metadata is None:
        return None

    symbol, entity, period, dtype = metadata
    try:
        relative = candidate_file.resolve().relative_to(corpus_root.resolve())
    except ValueError:
        return None

    if not relative.parts or relative.parts[0] not in _SOURCE_ROOT_SEGMENTS:
        return None

    citation = _first_nonempty_line(source_text) or relative.as_posix()
    import_base = relative.with_suffix("").as_posix()
    repo_name = canonical_rulespec_repo_name(candidate_file)
    if repo_name and len(repo_name) > len("rulespec-"):
        import_base = f"{repo_name.removeprefix('rulespec-')}:{import_base}"
    return ResolvedCanonicalConcept(
        term="",
        import_target=f"{import_base}#{symbol}",
        symbol=symbol,
        citation=citation,
        entity=entity,
        period=period,
        dtype=dtype,
        label="",
        rulespec_file=candidate_file,
    )


def _extract_principal_symbol_metadata(
    content: str,
) -> tuple[str, str, str, str] | None:
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
            return None
        principal: tuple[str, str, str, str] | None = None
        for rule in payload["rules"]:
            if not isinstance(rule, dict):
                continue
            name = str(rule.get("name") or "").strip()
            entity = str(rule.get("entity") or "").strip()
            period = str(rule.get("period") or "").strip()
            dtype = str(rule.get("dtype") or "").strip()
            if name and entity and period and dtype:
                principal = (name, entity, period, dtype)
        return principal
    return None


def _first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _with_concept_term(
    candidate: ResolvedCanonicalConcept,
    term: str,
) -> ResolvedCanonicalConcept:
    return ResolvedCanonicalConcept(
        term=term,
        import_target=candidate.import_target,
        symbol=candidate.symbol,
        citation=candidate.citation,
        entity=candidate.entity,
        period=candidate.period,
        dtype=candidate.dtype,
        label=(
            f"`{term}` -> import `{candidate.import_target}` ({candidate.citation})"
        ),
        rulespec_file=candidate.rulespec_file,
    )
