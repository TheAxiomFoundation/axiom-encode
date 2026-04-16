"""Shared planning and materialization for canonical dependency stubs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


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
    source_file: Path


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
_FILE_LEVEL_STUB_STATUS_PATTERN = re.compile(r"(?m)^status:\s*stub\s*$")
_TOP_LEVEL_SYMBOL_PATTERN = re.compile(r"^([A-Za-z_]\w*):\s*$")
_METADATA_PATTERN = re.compile(r"^\s+(entity|period|dtype):\s*(.+?)\s*$")
_QUOTED_DEFINITION_PATTERN = re.compile(
    r"[\"“]([^\"”]+)[\"”](?:\s+or\s+[\"“]([^\"”]+)[\"”])?\s+means\b",
    re.IGNORECASE,
)
_SOURCE_ROOT_SEGMENTS = {"legislation", "statute", "regulation"}
_SOURCE_SLICE_EXTENSIONS = (".txt", ".xml", ".akn", ".html", ".json", ".md")


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

    for candidate_file in sorted(corpus_root.rglob("*.rac")):
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
    normalized_import = import_path.strip().strip('"').strip("'").removesuffix(".rac")
    specs: list[ResolvedDefinedTerm] = []
    for symbol in symbol_names:
        spec = _REGISTERED_STUBS_BY_KEY.get((normalized_import, symbol))
        if spec is None:
            return []
        specs.append(spec)
    return specs


def import_target_to_relative_rac_path(import_target: str) -> Path:
    """Convert an import target like legislation/...#name into a .rac path."""
    normalized = import_target.strip().strip('"').strip("'").split("#", 1)[0]
    if normalized.endswith(".rac"):
        return Path(normalized)
    return Path(f"{normalized}.rac")


def build_registered_stub_content(specs: Sequence[ResolvedDefinedTerm]) -> str:
    """Return deterministic stub file content for one registered dependency file."""
    if not specs:
        raise ValueError("At least one stub spec is required")

    base_paths = {
        spec.import_target.split("#", 1)[0].removesuffix(".rac") for spec in specs
    }
    if len(base_paths) != 1:
        raise ValueError(
            "All registered stub specs must belong to the same target file"
        )

    if len(specs) == 1:
        header = (
            f'"""\nCanonical definition stub for `{specs[0].term}`.\n'
            f"Resolved to {specs[0].citation}.\n"
            '"""\n\n'
        )
    else:
        citations = ", ".join(sorted({spec.citation for spec in specs}))
        header = f'"""\nCanonical definition stubs.\nResolved to {citations}.\n"""\n\n'

    blocks = []
    for spec in specs:
        blocks.append(
            "\n".join(
                [
                    f"{spec.symbol}:",
                    f"    stub_for: {spec.import_target}",
                    f"    entity: {spec.entity}",
                    f"    period: {spec.period}",
                    f"    dtype: {spec.dtype}",
                ]
            )
        )

    return header + "status: stub\n\n" + "\n\n".join(blocks) + "\n"


def materialize_registered_stub(
    root: Path,
    specs: Sequence[ResolvedDefinedTerm],
    *,
    prefix: Path | None = None,
) -> Path:
    """Write one deterministic canonical stub file under the given root."""
    if not specs:
        raise ValueError("At least one stub spec is required")

    relative_path = import_target_to_relative_rac_path(specs[0].import_target)
    target = (
        root / prefix / relative_path if prefix is not None else root / relative_path
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_registered_stub_content(specs))
    return target


def rac_content_has_stub_status(content: str) -> bool:
    """Return whether a RAC file declares file-level `status: stub`."""
    return bool(_FILE_LEVEL_STUB_STATUS_PATTERN.search(content))


def rac_file_has_stub_status(rac_file: Path) -> bool:
    """Return whether a RAC file exists and declares file-level `status: stub`."""
    try:
        return rac_file.exists() and rac_content_has_stub_status(rac_file.read_text())
    except OSError:
        return False


def has_ingested_source_for_import_target(
    import_target: str, corpus_root: Path
) -> bool:
    """Return whether the import target's official source is present locally."""
    return bool(find_ingested_source_artifacts(import_target, corpus_root))


def find_ingested_source_artifacts(import_target: str, corpus_root: Path) -> list[Path]:
    """Locate source files proving the import target has been ingested locally."""
    corpus_root = corpus_root.resolve()
    sources_root = corpus_root / "sources"
    if not sources_root.exists():
        return []

    relative = import_target_to_relative_rac_path(import_target).with_suffix("")
    artifacts: list[Path] = []
    seen: set[Path] = set()

    for candidate in _iter_source_slice_candidates(sources_root, relative):
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            artifacts.append(resolved)

    for root in _iter_official_source_roots(sources_root, relative):
        if not root.exists():
            continue
        for candidate in sorted(root.rglob("*")):
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                artifacts.append(resolved)

    return artifacts


def _extract_embedded_source_text(content: str) -> str:
    match = _EMBEDDED_SOURCE_PATTERN.match(content)
    return match.group(1).strip() if match else ""


def _iter_source_slice_candidates(sources_root: Path, relative: Path) -> list[Path]:
    base = sources_root / "slices" / relative
    candidates: list[Path] = []
    if base.exists() and base.is_file():
        candidates.append(base)
    for extension in _SOURCE_SLICE_EXTENSIONS:
        candidate = base.with_suffix(extension)
        if candidate.exists():
            candidates.append(candidate)
    return candidates


def _iter_official_source_roots(sources_root: Path, relative: Path) -> list[Path]:
    parts = relative.parts
    if not parts:
        return []

    if parts[0] == "statute" and len(parts) >= 3:
        return [sources_root / "official" / parts[0] / parts[1] / parts[2]]

    if parts[0] == "legislation" and len(parts) >= 4:
        return [sources_root / "official" / parts[1] / parts[2] / parts[3]]

    if parts[0] == "regulation" and len(parts) >= 2:
        return [sources_root / "official" / parts[1]]

    return []


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
    return ResolvedCanonicalConcept(
        term="",
        import_target=f"{import_base}#{symbol}",
        symbol=symbol,
        citation=citation,
        entity=entity,
        period=period,
        dtype=dtype,
        label="",
        source_file=candidate_file,
    )


def _extract_principal_symbol_metadata(
    content: str,
) -> tuple[str, str, str, str] | None:
    current_name: str | None = None
    current_entity: str | None = None
    current_period: str | None = None
    current_dtype: str | None = None
    principal: tuple[str, str, str, str] | None = None

    for line in content.splitlines():
        header_match = _TOP_LEVEL_SYMBOL_PATTERN.match(line)
        if header_match:
            if (
                current_name is not None
                and current_entity is not None
                and current_period is not None
                and current_dtype is not None
            ):
                principal = (
                    current_name,
                    current_entity,
                    current_period,
                    current_dtype,
                )
            current_name = header_match.group(1)
            current_entity = None
            current_period = None
            current_dtype = None
            continue

        metadata_match = _METADATA_PATTERN.match(line)
        if metadata_match and current_name is not None:
            key, value = metadata_match.groups()
            cleaned = value.strip().strip('"').strip("'")
            if key == "entity":
                current_entity = cleaned
            elif key == "period":
                current_period = cleaned
            elif key == "dtype":
                current_dtype = cleaned

    if (
        current_name is not None
        and current_entity is not None
        and current_period is not None
        and current_dtype is not None
    ):
        principal = (
            current_name,
            current_entity,
            current_period,
            current_dtype,
        )

    return principal


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
        source_file=candidate.source_file,
    )
