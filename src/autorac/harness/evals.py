"""Model comparison evals for statute and source-slice encoding."""

from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Literal, Sequence
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import requests
import yaml

from autorac.statute import (
    CitationParts,
    citation_to_relative_rac_path,
    find_citation_text,
    parse_usc_citation,
)

from .dependency_stubs import (
    ResolvedCanonicalConcept,
    ResolvedDefinedTerm,
    import_target_to_relative_rac_path,
    materialize_registered_stub,
    resolve_canonical_concepts_from_text,
    resolve_defined_terms_from_text,
)
from .encoding_db import TokenUsage
from .eval_prompt_surface import (
    render_date_silent_scaffold_guidance,
    render_single_amount_row_guidance,
    render_uk_legislation_guidance,
)
from .observability import emit_eval_result, extract_reasoning_output_tokens
from .pricing import estimate_usage_cost_usd
from .validator_pipeline import (
    ValidationResult,
    ValidatorPipeline,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_named_scalar_occurrences,
    extract_numbers_from_text,
    extract_numeric_occurrences_from_text,
)

EvalMode = Literal["cold", "repo-augmented"]
EvalOracleMode = Literal["none", "policyengine", "all"]
IMPORT_ITEM_PATTERN = re.compile(r"^\s*-\s*(['\"]?)([^'\"]+?)\1\s*$")
AKN_NS = {"akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}
AKN_CONTAINER_TAGS = {
    "hcontainer",
    "section",
    "subsection",
    "level",
    "article",
    "paragraph",
    "subparagraph",
    "point",
    "subpoint",
    "part",
    "chapter",
}
SUPPORTED_EVAL_ENTITIES = (
    "Payment",
    "Person",
    "TaxUnit",
    "Household",
    "Family",
    "TanfUnit",
    "SnapUnit",
    "SPMUnit",
    "Corporation",
    "Business",
    "Asset",
)
SUPPORTED_EVAL_PERIODS = ("Year", "Month", "Week", "Day")
SUPPORTED_EVAL_DTYPES = (
    "Money",
    "Rate",
    "Boolean",
    "Integer",
    "Count",
    "String",
    "Decimal",
    "Float",
)
_PURE_NUMERIC_EXPRESSION_PATTERN = re.compile(r"^[\d\s()+\-*/.,]+$")
_ISO_WEEK_PERIOD_PATTERN = re.compile(r"^\d{4}-W\d{2}(?:-\d)?$")
_CONDITIONAL_AMOUNT_SLICE_PATTERN = re.compile(
    r"\b(?:if|where|unless|except|subject to|treated as paid)\b",
    re.IGNORECASE,
)
_LOCAL_IMPORT_ROOT_TOKENS = {"legislation", "statute", "regulation"}
_DEFAULT_SHARED_LEGISLATION_CACHE_ROOT = (
    Path.home() / "tmp" / "autorac-shared-legislation-cache"
).resolve()


@dataclass(frozen=True)
class EvalRunnerSpec:
    """How to invoke a model in an eval."""

    name: str
    backend: str
    model: str


@dataclass
class GroundingMetric:
    """A numeric grounding decision."""

    line: int
    raw: str
    value: float
    grounded: bool


@dataclass
class EvalArtifactMetrics:
    """Deterministic checks over a produced RAC artifact."""

    compile_pass: bool
    compile_issues: list[str]
    ci_pass: bool
    ci_issues: list[str]
    embedded_source_present: bool
    grounded_numeric_count: int
    ungrounded_numeric_count: int
    grounding: list[GroundingMetric]
    source_numeric_occurrence_count: int = 0
    covered_source_numeric_occurrence_count: int = 0
    missing_source_numeric_occurrence_count: int = 0
    numeric_occurrence_issues: list[str] = field(default_factory=list)
    generalist_review_pass: bool | None = None
    generalist_review_score: float | None = None
    generalist_review_issues: list[str] = field(default_factory=list)
    policyengine_pass: bool | None = None
    policyengine_score: float | None = None
    policyengine_issues: list[str] = field(default_factory=list)
    taxsim_pass: bool | None = None
    taxsim_score: float | None = None
    taxsim_issues: list[str] = field(default_factory=list)


@dataclass
class EvalContextFile:
    """A context file copied into the eval workspace."""

    source_path: str
    workspace_path: str
    import_path: str
    kind: str
    label: str | None = None


@dataclass
class EvalWorkspace:
    """Prepared workspace bundle for an eval run."""

    root: Path
    source_file: Path
    manifest_file: Path
    context_files: list[EvalContextFile] = field(default_factory=list)


@dataclass
class EvalPromptResponse:
    """Raw model output and trace from a prompt-only eval run."""

    text: str
    duration_ms: int
    tokens: TokenUsage | None = None
    estimated_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    trace: dict | None = None
    unexpected_accesses: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvalResult:
    """One citation x runner result."""

    citation: str
    runner: str
    backend: str
    model: str
    mode: EvalMode
    output_file: str
    trace_file: str
    context_manifest_file: str
    duration_ms: int
    success: bool
    error: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    reasoning_output_tokens: int
    estimated_cost_usd: float | None
    actual_cost_usd: float | None
    retrieved_files: list[str]
    unexpected_accesses: list[str]
    metrics: EvalArtifactMetrics | None

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.metrics is not None:
            data["metrics"] = asdict(self.metrics)
        return data


@dataclass(frozen=True)
class EvalReadinessGates:
    """Thresholds that determine whether a benchmark suite is bulk-ready."""

    min_cases: int = 1
    min_success_rate: float | None = None
    min_compile_pass_rate: float | None = None
    min_ci_pass_rate: float | None = None
    min_zero_ungrounded_rate: float | None = None
    min_generalist_review_pass_rate: float | None = 1.0
    min_policyengine_pass_rate: float | None = None
    max_mean_estimated_cost_usd: float | None = None


@dataclass
class EvalSuiteCase:
    """One manifest entry in an eval suite."""

    kind: Literal["citation", "source", "akn_section", "uk_legislation"]
    name: str
    mode: EvalMode
    allow_context: list[Path] = field(default_factory=list)
    allow_parent: bool = False
    citation: str | None = None
    source_id: str | None = None
    source_file: Path | None = None
    akn_file: Path | None = None
    section_eid: str | None = None
    table_row_query: str | None = None
    policyengine_rac_var_hint: str | None = None
    source_ref: str | None = None
    oracle: EvalOracleMode = "none"
    policyengine_country: str = "auto"


@dataclass
class EvalSuiteManifest:
    """Manifest describing a benchmark suite and its readiness gates."""

    name: str
    path: Path
    runners: list[str]
    mode: EvalMode
    allow_context: list[Path]
    gates: EvalReadinessGates
    cases: list[EvalSuiteCase]


@dataclass(frozen=True)
class EvalReadinessGateResult:
    """Outcome of one readiness threshold."""

    name: str
    comparator: Literal["min", "max"]
    threshold: float | int
    actual: float | int | None
    passed: bool


@dataclass
class EvalReadinessSummary:
    """Aggregated readiness summary for one runner across a suite."""

    total_cases: int
    success_rate: float
    compile_pass_rate: float
    ci_pass_rate: float
    zero_ungrounded_rate: float
    generalist_review_pass_rate: float
    mean_generalist_review_score: float | None
    policyengine_case_count: int
    policyengine_pass_rate: float | None
    mean_policyengine_score: float | None
    mean_estimated_cost_usd: float | None
    gate_results: list[EvalReadinessGateResult]
    ready: bool


@dataclass(frozen=True)
class FetchedLegislationGovUkDocument:
    """Official legislation.gov.uk sources fetched for one content URL."""

    source_id: str
    content_url: str
    akn_file: Path
    clml_file: Path


def parse_runner_spec(spec: str) -> EvalRunnerSpec:
    """Parse `[name=]backend:model` into a structured runner spec."""
    alias = ""
    target = spec
    if "=" in spec:
        alias, target = spec.split("=", 1)

    if ":" not in target:
        raise ValueError(
            f"Invalid runner spec '{spec}'. Expected [name=]backend:model."
        )

    backend, model = target.split(":", 1)
    backend = backend.strip()
    model = model.strip()
    name = alias.strip() or re.sub(r"[^a-zA-Z0-9._-]+", "-", f"{backend}-{model}")

    if backend not in {"claude", "codex", "openai"}:
        raise ValueError(f"Unsupported backend '{backend}' in runner spec '{spec}'")

    return EvalRunnerSpec(name=name, backend=backend, model=model)


def run_model_eval(
    citations: list[str],
    runner_specs: list[str],
    output_root: Path,
    rac_path: Path,
    atlas_path: Path,
    mode: EvalMode = "repo-augmented",
    extra_context_paths: list[Path] | None = None,
) -> list[EvalResult]:
    """Run a deterministic comparison over one or more citations."""
    xml_root = atlas_path / "data" / "uscode"
    results: list[EvalResult] = []

    for runner in [parse_runner_spec(spec) for spec in runner_specs]:
        for citation in citations:
            results.append(
                _run_single_eval(
                    citation=citation,
                    runner=runner,
                    output_root=output_root,
                    rac_path=rac_path,
                    xml_root=xml_root,
                    mode=mode,
                    extra_context_paths=extra_context_paths or [],
                )
            )

    return results


def run_source_eval(
    source_id: str,
    source_text: str,
    runner_specs: list[str],
    output_root: Path,
    rac_path: Path,
    mode: EvalMode = "repo-augmented",
    extra_context_paths: list[Path] | None = None,
    oracle: EvalOracleMode = "none",
    policyengine_country: str = "auto",
    policyengine_rac_var_hint: str | None = None,
) -> list[EvalResult]:
    """Run a deterministic comparison over one arbitrary source slice."""
    results: list[EvalResult] = []

    for runner in [parse_runner_spec(spec) for spec in runner_specs]:
        results.append(
            _run_single_source_eval(
                source_id=source_id,
                source_text=source_text,
                runner=runner,
                output_root=output_root,
                rac_path=rac_path,
                mode=mode,
                extra_context_paths=extra_context_paths or [],
                oracle=oracle,
                policyengine_country=policyengine_country,
                policyengine_rac_var_hint=policyengine_rac_var_hint,
            )
        )

    return results


def _collapse_whitespace(text: str) -> str:
    """Normalize extracted XML text into one readable line."""
    return " ".join(text.split())


def _akn_child_text(parent: ET.Element, child_tag: str) -> str:
    """Return normalized text from a direct child tag."""
    child = parent.find(f"akn:{child_tag}", AKN_NS)
    if child is None:
        return ""
    return _collapse_whitespace("".join(child.itertext()))


def _akn_local_tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _is_akn_container(element: ET.Element) -> bool:
    return _akn_local_tag(element) in AKN_CONTAINER_TAGS


def _find_akn_section(root: ET.Element, section_eid: str) -> ET.Element:
    for element in root.iter():
        if element.get("eId") == section_eid and _is_akn_container(element):
            return element
    raise ValueError(f"Section eId not found: {section_eid}")


def _direct_akn_child_sections(section: ET.Element) -> list[ET.Element]:
    return [child for child in list(section) if _is_akn_container(child)]


def _akn_parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in list(parent)}


def _akn_section_own_text(section: ET.Element) -> str:
    parts: list[str] = []
    title = _akn_title(section)
    if title:
        parts.append(title)
    for tag in ("intro", "content", "wrapUp"):
        node = section.find(f"akn:{tag}", AKN_NS)
        if node is not None:
            _append_akn_content_block_text(node, parts)
    return "\n\n".join(part for part in parts if part).strip()


def _shared_single_numeric_sibling_eids(akn_file: Path, section_eid: str) -> list[str]:
    tree = ET.parse(akn_file)
    root = tree.getroot()
    section = _find_akn_section(root, section_eid)
    target_children = _direct_akn_child_sections(section)
    if target_children:
        return []

    parent = _akn_parent_map(root).get(section)
    if parent is None or not _is_akn_container(parent):
        return []

    siblings = [
        child
        for child in _direct_akn_child_sections(parent)
        if not _direct_akn_child_sections(child)
    ]
    if len(siblings) < 2:
        return []

    signatures: dict[str, list[str]] = {}
    target_signature: str | None = None
    for sibling in siblings:
        numbers = {
            round(value, 9)
            for value in extract_numbers_from_text(_akn_section_own_text(sibling))
        }
        for value in sorted(numbers):
            scaled = round(value * 100, 9)
            if value <= 1 and scaled in numbers:
                numbers.discard(scaled)
        if len(numbers) != 1:
            continue
        signature = f"{next(iter(numbers)):.9f}"
        sibling_eid = sibling.get("eId") or ""
        signatures.setdefault(signature, []).append(sibling_eid)
        if sibling is section:
            target_signature = signature

    if target_signature is None:
        return []
    matching = sorted(eid for eid in signatures.get(target_signature, []) if eid)
    if len(matching) < 2:
        return []
    return matching


def _validate_uk_shared_scalar_sibling_sets(
    manifest: EvalSuiteManifest,
    output_root: Path,
) -> None:
    cases_by_source: dict[str, list[EvalSuiteCase]] = {}
    for case in manifest.cases:
        if case.kind != "uk_legislation" or not case.source_ref or not case.section_eid:
            continue
        cases_by_source.setdefault(case.source_ref, []).append(case)

    for source_ref, cases in cases_by_source.items():
        fetched = _fetch_legislation_gov_uk_document(
            source_ref,
            output_root,
            fetch_cache_root=_resolve_legislation_gov_uk_fetch_cache_root(output_root),
        )
        selected_eids = {case.section_eid for case in cases if case.section_eid}
        for case in cases:
            sibling_eids = _shared_single_numeric_sibling_eids(
                fetched.akn_file,
                case.section_eid or "",
            )
            if not sibling_eids:
                continue
            missing = [eid for eid in sibling_eids if eid not in selected_eids]
            if not missing:
                continue
            raise ValueError(
                f"UK legislation case '{case.name}' targets {case.section_eid}, which is part of "
                f"a repeated-scalar sibling set under the same parent. Include the full sibling set: "
                f"{', '.join(sibling_eids)}. Missing: {', '.join(missing)}."
            )


def _find_primary_akn_section_eid(akn_file: Path) -> str:
    """Pick the sole top-level AKN content node from a document-level source."""
    tree = ET.parse(akn_file)
    root = tree.getroot()

    for body_tag in ("mainBody", "body"):
        body = root.find(f".//akn:{body_tag}", AKN_NS)
        if body is None:
            continue
        children = _direct_akn_child_sections(body)
        if len(children) == 1:
            section_eid = children[0].get("eId")
            if not section_eid:
                raise ValueError("Primary AKN section is missing an eId")
            return section_eid
        if len(children) > 1:
            labels = ", ".join(child.get("eId", "<unknown>") for child in children[:8])
            raise ValueError(
                "AKN document has multiple top-level sections. "
                f"Pass --section-eid explicitly. Candidates: {labels}"
            )

    raise ValueError("Could not identify a primary AKN section in the document")


def _normalize_legislation_gov_uk_source_ref(source_ref: str) -> tuple[str, str]:
    """Normalize a legislation.gov.uk URL or path to a canonical content URL."""
    raw = source_ref.strip()
    if not raw:
        raise ValueError("Empty legislation.gov.uk source reference")

    if "://" not in raw:
        raw = f"https://www.legislation.gov.uk/{raw.lstrip('/')}"

    parsed = urlparse(raw)
    if parsed.netloc not in {"www.legislation.gov.uk", "legislation.gov.uk"}:
        raise ValueError(
            f"Unsupported source host '{parsed.netloc}'. Expected legislation.gov.uk"
        )

    path = parsed.path.rstrip("/")
    for suffix in (
        "/data.akn",
        "/data.xml",
        "/data.html",
        "/data.htm",
        "/data.xht",
        "/data.pdf",
    ):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break

    if not path or path == "/":
        raise ValueError(f"Invalid legislation.gov.uk source reference: {source_ref}")

    source_id = path.lstrip("/")
    return source_id, f"https://www.legislation.gov.uk/{source_id}"


def _resolve_legislation_gov_uk_fetch_cache_root(output_root: Path) -> Path:
    """Prefer a persistent local fetch cache when one is configured or already exists."""
    override = os.getenv("AUTORAC_SHARED_LEGISLATION_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    if _DEFAULT_SHARED_LEGISLATION_CACHE_ROOT.exists():
        return _DEFAULT_SHARED_LEGISLATION_CACHE_ROOT
    return Path(output_root).resolve()


def _fetch_legislation_gov_uk_document(
    source_ref: str,
    output_root: Path,
    *,
    fetch_cache_root: Path | None = None,
) -> FetchedLegislationGovUkDocument:
    """Fetch official AKN and CLML files from legislation.gov.uk."""
    source_id, content_url = _normalize_legislation_gov_uk_source_ref(source_ref)
    source_base_dir = (
        Path(fetch_cache_root) / "_legislation_gov_uk_cache"
        if fetch_cache_root is not None and Path(fetch_cache_root).resolve() != Path(output_root).resolve()
        else Path(output_root) / "_legislation_gov_uk"
    )
    source_dir = source_base_dir / _slugify(source_id)
    source_dir.mkdir(parents=True, exist_ok=True)

    akn_file = source_dir / "source.akn"
    clml_file = source_dir / "source.xml"
    if (
        akn_file.exists()
        and clml_file.exists()
        and akn_file.stat().st_size > 0
        and clml_file.stat().st_size > 0
    ):
        return FetchedLegislationGovUkDocument(
            source_id=source_id,
            content_url=content_url,
            akn_file=akn_file,
            clml_file=clml_file,
        )

    akn_file.write_text(_fetch_legislation_gov_uk_text(f"{content_url}/data.akn"))
    try:
        clml_file.write_text(_fetch_legislation_gov_uk_text(f"{content_url}/data.xml"))
    except requests.RequestException as exc:
        clml_file.write_text(f"<!-- source.xml unavailable: {exc} -->\n")

    return FetchedLegislationGovUkDocument(
        source_id=source_id,
        content_url=content_url,
        akn_file=akn_file,
        clml_file=clml_file,
    )


def _fetch_legislation_gov_uk_text(
    url: str,
    *,
    attempts: int = 6,
    timeout: int = 30,
) -> str:
    """Fetch one legislation.gov.uk payload with retry for transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            retriable = status_code in {429, 500, 502, 503, 504} or status_code is None
            if attempt >= attempts or not retriable:
                raise
            time.sleep(min(2 ** (attempt - 1), 10))

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch legislation.gov.uk resource: {url}")


def _akn_title(element: ET.Element) -> str:
    return " ".join(
        item for item in (_akn_child_text(element, "num"), _akn_child_text(element, "heading")) if item
    ).strip()


def _akn_ancestor_titles(root: ET.Element, section: ET.Element) -> list[str]:
    parent_by_child = {child: parent for parent in root.iter() for child in list(parent)}
    titles: list[str] = []
    current = parent_by_child.get(section)
    while current is not None:
        if _is_akn_container(current):
            title = _akn_title(current)
            if title:
                titles.append(title)
        current = parent_by_child.get(current)
    titles.reverse()
    return titles


def _table_rows_from_element(table: ET.Element) -> list[list[str]]:
    """Extract normalized cell text from an AKN or XHTML table."""
    rows: list[list[str]] = []
    for row in table.iter():
        if _akn_local_tag(row) != "tr":
            continue
        cells = [
            _collapse_whitespace("".join(cell.itertext()))
            for cell in list(row)
            if _akn_local_tag(cell) in {"td", "th"}
        ]
        if any(cells):
            rows.append(cells)
    return rows


def _table_row_has_amount(row: list[str]) -> bool:
    """Return True when a row looks like a value-bearing table row."""
    if not row:
        return False
    return bool(re.search(r"\d", row[-1]))


def _normalize_table_match_text(value: str) -> str:
    """Normalize table text for resilient row-query matching."""
    return re.sub(r"\W+", "", _collapse_whitespace(value).lower())


def _select_table_rows(
    rows: list[list[str]],
    table_row_query: str | None = None,
) -> list[list[str]]:
    """Filter table rows to one matched row plus nearby grouping context."""
    if not table_row_query:
        return rows

    query = _normalize_table_match_text(table_row_query)
    if not query:
        return rows

    matched_indexes = [
        index
        for index, row in enumerate(rows)
        if query in _normalize_table_match_text(" | ".join(row))
    ]
    if not matched_indexes:
        return rows

    selected_indexes: set[int] = {0} if rows else set()
    for index in matched_indexes:
        selected_indexes.add(index)
        context_index = index - 1
        while context_index > 0 and not _table_row_has_amount(rows[context_index]):
            selected_indexes.add(context_index)
            context_index -= 1

    return [rows[index] for index in sorted(selected_indexes)]


def _append_akn_content_block_text(
    parent: ET.Element,
    parts: list[str],
    table_row_query: str | None = None,
) -> None:
    for child in list(parent):
        local_tag = _akn_local_tag(child)
        if local_tag == "p":
            if table_row_query:
                continue
            paragraph = _collapse_whitespace("".join(child.itertext()))
            if paragraph:
                parts.append(paragraph)
            continue

        tables = [node for node in child.iter() if _akn_local_tag(node) == "table"]
        if local_tag == "table" and child not in tables:
            tables.insert(0, child)
        if not tables:
            continue

        for table in tables:
            selected_rows = _select_table_rows(
                _table_rows_from_element(table),
                table_row_query=table_row_query,
            )
            formatted_rows = [" | ".join(cell for cell in row if cell) for row in selected_rows]
            if formatted_rows:
                parts.append("Structured table:\n" + "\n".join(formatted_rows))


def _akn_ancestor_intro_text(root: ET.Element, section: ET.Element) -> list[str]:
    parent_by_child = {child: parent for parent in root.iter() for child in list(parent)}
    intros: list[str] = []
    current = parent_by_child.get(section)
    lineage: list[ET.Element] = []
    while current is not None:
        if _is_akn_container(current):
            lineage.append(current)
        current = parent_by_child.get(current)
    lineage.reverse()
    for ancestor in lineage:
        for tag in ("intro", "wrapUp"):
            node = ancestor.find(f"akn:{tag}", AKN_NS)
            if node is not None:
                _append_akn_content_block_text(node, intros)
    return intros


def _akn_expression_valid_from(root: ET.Element) -> str:
    """Return the AKN expression-level valid-from date when present."""
    date_node = root.find(
        ".//akn:FRBRExpression/akn:FRBRdate[@name='validFrom']",
        AKN_NS,
    )
    if date_node is None:
        return ""
    return (date_node.get("date") or "").strip()


def _resolve_akn_section_eid(
    akn_file: Path,
    section_eid: str,
    allow_parent: bool = False,
) -> str:
    """Resolve an AKN section target, rejecting parent nodes by default."""
    tree = ET.parse(akn_file)
    root = tree.getroot()
    section = _find_akn_section(root, section_eid)
    child_sections = _direct_akn_child_sections(section)
    if allow_parent or not child_sections:
        return section_eid

    child_summaries: list[str] = []
    for child in child_sections[:8]:
        child_eid = child.get("eId", "<unknown>")
        label = " ".join(
            item
            for item in (_akn_child_text(child, "num"), _akn_child_text(child, "heading"))
            if item
        ).strip()
        child_summaries.append(f"{child_eid} ({label or 'child'})")

    suggestions = ", ".join(child_summaries)
    raise ValueError(
        f"Section {section_eid} has child sections. "
        f"Choose an atomic child section instead: {suggestions}. "
        "Pass allow_parent=True only when you intentionally need the parent layer."
    )


def extract_akn_section_text(
    akn_file: Path,
    section_eid: str,
    table_row_query: str | None = None,
) -> str:
    """Extract one Akoma Ntoso section as plain source text for evals."""
    tree = ET.parse(akn_file)
    root = tree.getroot()
    section = _find_akn_section(root, section_eid)

    parts: list[str] = [title for title in _akn_ancestor_titles(root, section) if title]
    valid_from = _akn_expression_valid_from(root)
    if valid_from:
        parts.append(f"Editorial note: current text valid from {valid_from}.")
    parts.extend(_akn_ancestor_intro_text(root, section))
    title = _akn_title(section)
    if title:
        parts.append(title)

    for remark in section.findall("akn:remark", AKN_NS):
        if remark.get("status") != "editorial":
            continue
        remark_text = _collapse_whitespace("".join(remark.itertext()))
        if remark_text:
            parts.append(remark_text)

    for tag in ("intro", "content", "wrapUp"):
        node = section.find(f"akn:{tag}", AKN_NS)
        if node is not None:
            _append_akn_content_block_text(
                node,
                parts,
                table_row_query=table_row_query,
            )

    return "\n\n".join(parts).strip()


def run_akn_section_eval(
    source_id: str,
    akn_file: Path,
    section_eid: str,
    runner_specs: list[str],
    output_root: Path,
    rac_path: Path,
    mode: EvalMode = "repo-augmented",
    extra_context_paths: list[Path] | None = None,
    allow_parent: bool = False,
    table_row_query: str | None = None,
    oracle: EvalOracleMode = "none",
    policyengine_country: str = "auto",
    policyengine_rac_var_hint: str | None = None,
) -> list[EvalResult]:
    """Run a deterministic comparison on one section extracted from AKN XML."""
    resolved_section_eid = _resolve_akn_section_eid(
        akn_file,
        section_eid,
        allow_parent=allow_parent,
    )
    return run_source_eval(
        source_id=source_id,
        source_text=extract_akn_section_text(
            akn_file,
            resolved_section_eid,
            table_row_query=table_row_query,
        ),
        runner_specs=runner_specs,
        output_root=output_root,
        rac_path=rac_path,
        mode=mode,
        extra_context_paths=extra_context_paths,
        oracle=oracle,
        policyengine_country=policyengine_country,
        policyengine_rac_var_hint=policyengine_rac_var_hint,
    )


def run_legislation_gov_uk_section_eval(
    source_ref: str,
    runner_specs: list[str],
    output_root: Path,
    rac_path: Path,
    mode: EvalMode = "repo-augmented",
    extra_context_paths: list[Path] | None = None,
    section_eid: str | None = None,
    allow_parent: bool = False,
    table_row_query: str | None = None,
    oracle: EvalOracleMode = "none",
    policyengine_country: str = "auto",
    policyengine_rac_var_hint: str | None = None,
    fetch_cache_root: Path | None = None,
) -> list[EvalResult]:
    """Fetch official UK legislation XML and run an AKN section eval."""
    resolved_fetch_cache_root = (
        fetch_cache_root
        if fetch_cache_root is not None
        else _resolve_legislation_gov_uk_fetch_cache_root(output_root)
    )
    fetched = _fetch_legislation_gov_uk_document(
        source_ref,
        output_root,
        fetch_cache_root=resolved_fetch_cache_root,
    )
    target_section_eid = section_eid or _find_primary_akn_section_eid(fetched.akn_file)
    return run_akn_section_eval(
        source_id=fetched.source_id,
        akn_file=fetched.akn_file,
        section_eid=target_section_eid,
        runner_specs=runner_specs,
        output_root=output_root,
        rac_path=rac_path,
        mode=mode,
        extra_context_paths=extra_context_paths,
        allow_parent=allow_parent,
        table_row_query=table_row_query,
        oracle=oracle,
        policyengine_country=policyengine_country,
        policyengine_rac_var_hint=policyengine_rac_var_hint,
    )


def load_eval_suite_manifest(path: Path) -> EvalSuiteManifest:
    """Load a manifest describing a benchmark suite and readiness gates."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Eval suite manifest must be a mapping: {path}")

    base_dir = Path(path).resolve().parent
    default_mode = _coerce_eval_mode(raw.get("mode", "repo-augmented"))
    default_context = [
        _resolve_manifest_path(base_dir, entry)
        for entry in raw.get("allow_context", []) or []
    ]
    runners = [str(item) for item in (raw.get("runners") or ["codex:gpt-5.4"])]

    gates_raw = raw.get("gates") or {}
    gates = EvalReadinessGates(
        min_cases=int(gates_raw.get("min_cases", 1)),
        min_success_rate=_optional_float(gates_raw.get("min_success_rate")),
        min_compile_pass_rate=_optional_float(
            gates_raw.get("min_compile_pass_rate")
        ),
        min_ci_pass_rate=_optional_float(gates_raw.get("min_ci_pass_rate")),
        min_zero_ungrounded_rate=_optional_float(
            gates_raw.get("min_zero_ungrounded_rate")
        ),
        min_generalist_review_pass_rate=_optional_float(
            gates_raw.get("min_generalist_review_pass_rate", 1.0)
        ),
        min_policyengine_pass_rate=_optional_float(
            gates_raw.get("min_policyengine_pass_rate")
        ),
        max_mean_estimated_cost_usd=_optional_float(
            gates_raw.get("max_mean_estimated_cost_usd")
        ),
    )

    cases_raw = raw.get("cases") or []
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError(f"Eval suite manifest has no cases: {path}")

    cases: list[EvalSuiteCase] = []
    for index, item in enumerate(cases_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Eval suite case #{index} must be a mapping")
        kind = str(item.get("kind", "")).strip()
        if kind not in {"citation", "source", "akn_section", "uk_legislation"}:
            raise ValueError(f"Unsupported eval suite case kind '{kind}'")

        case_mode = _coerce_eval_mode(item.get("mode", default_mode))
        name = (
            str(item.get("name", "")).strip()
            or str(
                item.get("citation")
                or item.get("source_id")
                or item.get("source_ref")
                or item.get("section_eid")
                or f"case-{index}"
            )
        )

        case = EvalSuiteCase(
            kind=kind,
            name=name,
            mode=case_mode,
            allow_context=[
                _resolve_manifest_path(base_dir, entry)
                for entry in item.get("allow_context", []) or []
            ],
            allow_parent=bool(item.get("allow_parent", False)),
            citation=item.get("citation"),
            source_id=item.get("source_id"),
            source_file=(
                _resolve_manifest_path(base_dir, item["source_file"])
                if item.get("source_file")
                else None
            ),
            akn_file=(
                _resolve_manifest_path(base_dir, item["akn_file"])
                if item.get("akn_file")
                else None
            ),
            section_eid=item.get("section_eid"),
            table_row_query=(
                str(item.get("table_row_query")).strip()
                if item.get("table_row_query") is not None
                else None
            ),
            policyengine_rac_var_hint=(
                str(item.get("policyengine_rac_var_hint")).strip()
                if item.get("policyengine_rac_var_hint") is not None
                else None
            ),
            source_ref=item.get("source_ref"),
            oracle=str(item.get("oracle", "none")),
            policyengine_country=str(item.get("policyengine_country", "auto")),
        )
        _validate_eval_suite_case(case, index)
        cases.append(case)

    return EvalSuiteManifest(
        name=str(raw.get("name") or Path(path).stem),
        path=Path(path).resolve(),
        runners=runners,
        mode=default_mode,
        allow_context=default_context,
        gates=gates,
        cases=cases,
    )


def run_eval_suite(
    manifest: EvalSuiteManifest,
    output_root: Path,
    rac_path: Path,
    atlas_path: Path | None = None,
    runner_specs: list[str] | None = None,
    suite_retry_attempts: int = 2,
    resume_existing: bool = False,
) -> list[EvalResult]:
    """Run every case in a benchmark suite manifest."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_runners = runner_specs or manifest.runners
    parsed_runners = [parse_runner_spec(spec) for spec in resolved_runners]
    results: list[EvalResult] = []
    started_at = _utc_now_iso()
    completed_case_indexes: set[int] = set()
    completed_cases = 0
    last_case_name: str | None = None
    active_case_index: int | None = None
    active_case_name: str | None = None
    active_case_started_at: str | None = None
    active_case_output_root: Path | None = None
    if resume_existing:
        (
            started_at,
            results,
            completed_case_indexes,
        ) = _load_eval_suite_resume_state(
            output_root=output_root,
            manifest=manifest,
            resolved_runners=resolved_runners,
            runner_count=len(parsed_runners),
        )
        completed_cases = _contiguous_completed_case_count(
            completed_case_indexes, len(manifest.cases)
        )
        if completed_cases > 0:
            last_case_name = manifest.cases[completed_cases - 1].name
    _write_eval_suite_run_state(
        output_root=output_root,
        manifest=manifest,
        resolved_runners=resolved_runners,
        status="running",
        started_at=started_at,
        completed_cases=completed_cases,
        result_count=len(results),
        last_case_name=last_case_name,
    )
    _validate_uk_shared_scalar_sibling_sets(manifest, Path(output_root))
    try:
        for index, case in enumerate(manifest.cases, start=1):
            if index in completed_case_indexes:
                continue
            case_output_root = output_root / f"{index:02d}-{_slugify(case.name)}"
            extra_context = [*manifest.allow_context, *case.allow_context]
            attempts = max(suite_retry_attempts, 0) + 1
            active_case_index = index
            active_case_name = case.name
            active_case_started_at = _utc_now_iso()
            active_case_output_root = case_output_root
            _write_eval_suite_run_state(
                output_root=output_root,
                manifest=manifest,
                resolved_runners=resolved_runners,
                status="running",
                started_at=started_at,
                completed_cases=completed_cases,
                result_count=len(results),
                last_case_name=last_case_name,
                active_case_index=active_case_index,
                active_case_name=active_case_name,
                active_case_started_at=active_case_started_at,
                active_case_output_root=active_case_output_root,
            )
            for attempt_index in range(attempts):
                try:
                    if case.kind == "citation":
                        if atlas_path is None:
                            raise ValueError(
                                "atlas_path is required for citation eval suite cases"
                            )
                        case_results = run_model_eval(
                            citations=[case.citation or ""],
                            runner_specs=resolved_runners,
                            output_root=case_output_root,
                            rac_path=rac_path,
                            atlas_path=atlas_path,
                            mode=case.mode,
                            extra_context_paths=extra_context,
                        )
                    elif case.kind == "source":
                        case_results = run_source_eval(
                            source_id=case.source_id or case.name,
                            source_text=(case.source_file or Path()).read_text(),
                            runner_specs=resolved_runners,
                            output_root=case_output_root,
                            rac_path=rac_path,
                            mode=case.mode,
                            extra_context_paths=extra_context,
                            oracle=case.oracle,
                            policyengine_country=case.policyengine_country,
                            policyengine_rac_var_hint=case.policyengine_rac_var_hint,
                        )
                    elif case.kind == "akn_section":
                        case_results = run_akn_section_eval(
                            source_id=case.source_id or case.name,
                            akn_file=case.akn_file or Path(),
                            section_eid=case.section_eid or "",
                            runner_specs=resolved_runners,
                            output_root=case_output_root,
                            rac_path=rac_path,
                            mode=case.mode,
                            extra_context_paths=extra_context,
                            allow_parent=case.allow_parent,
                            table_row_query=case.table_row_query,
                            oracle=case.oracle,
                            policyengine_country=case.policyengine_country,
                            policyengine_rac_var_hint=case.policyengine_rac_var_hint,
                        )
                    else:
                        case_results = run_legislation_gov_uk_section_eval(
                            source_ref=case.source_ref or "",
                            section_eid=case.section_eid,
                            runner_specs=resolved_runners,
                            output_root=case_output_root,
                            rac_path=rac_path,
                            mode=case.mode,
                            extra_context_paths=extra_context,
                            allow_parent=case.allow_parent,
                            table_row_query=case.table_row_query,
                            oracle=case.oracle,
                            policyengine_country=case.policyengine_country,
                            policyengine_rac_var_hint=case.policyengine_rac_var_hint,
                            fetch_cache_root=_resolve_legislation_gov_uk_fetch_cache_root(
                                output_root
                            ),
                        )
                except Exception as exc:
                    case_results = _suite_case_failure_results(case, parsed_runners, exc)

                if (
                    attempt_index >= attempts - 1
                    or not _suite_case_results_should_retry(case_results)
                ):
                    break

            for result in case_results:
                if case.name and case.name != result.citation:
                    result.citation = f"{case.name} ({result.citation})"
            results.extend(case_results)
            completed_case_indexes.add(index)
            completed_cases = index
            last_case_name = case.name
            active_case_index = None
            active_case_name = None
            active_case_started_at = None
            active_case_output_root = None
            _append_eval_suite_case_results(output_root, index, case, case_results)
            if _suite_case_results_hit_usage_limit(case_results):
                _write_eval_suite_run_state(
                    output_root=output_root,
                    manifest=manifest,
                    resolved_runners=resolved_runners,
                    status="failed",
                    started_at=started_at,
                    completed_cases=completed_cases,
                    result_count=len(results),
                    last_case_name=last_case_name,
                    error=(
                        "Usage limit reached while running "
                        f"case '{case.name}'. Stop the suite and retry after quota resets."
                    ),
                )
                return results
            _write_eval_suite_run_state(
                output_root=output_root,
                manifest=manifest,
                resolved_runners=resolved_runners,
                status="running",
                started_at=started_at,
                completed_cases=completed_cases,
                result_count=len(results),
                last_case_name=last_case_name,
            )
    except BaseException as exc:
        _write_eval_suite_run_state(
            output_root=output_root,
            manifest=manifest,
            resolved_runners=resolved_runners,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            started_at=started_at,
            completed_cases=completed_cases,
            result_count=len(results),
            last_case_name=last_case_name,
            error=_format_suite_exception(exc),
            active_case_index=active_case_index,
            active_case_name=active_case_name,
            active_case_started_at=active_case_started_at,
            active_case_output_root=active_case_output_root,
        )
        raise

    _write_eval_suite_run_state(
        output_root=output_root,
        manifest=manifest,
        resolved_runners=resolved_runners,
        status="completed",
        started_at=started_at,
        completed_cases=completed_cases,
        result_count=len(results),
        last_case_name=last_case_name,
    )
    return results


def _contiguous_completed_case_count(
    completed_case_indexes: set[int],
    total_cases: int,
) -> int:
    """Return the largest completed case prefix represented in the ledger."""
    completed = 0
    for index in range(1, total_cases + 1):
        if index not in completed_case_indexes:
            break
        completed = index
    return completed


def _utc_now_iso() -> str:
    """Return the current UTC time in a stable JSON-friendly format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_suite_exception(exc: BaseException) -> str:
    """Return a non-empty error string for suite state logs."""
    message = str(exc).strip()
    return message or exc.__class__.__name__


def _load_eval_suite_resume_state(
    output_root: Path,
    manifest: EvalSuiteManifest,
    resolved_runners: list[str],
    runner_count: int,
) -> tuple[str, list[EvalResult], set[int]]:
    """Load prior suite state and completed case results for resumption."""
    state_path = output_root / "suite-run.json"
    ledger_path = output_root / "suite-results.jsonl"
    started_at = _utc_now_iso()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        manifest_payload = state.get("manifest") or {}
        existing_path = manifest_payload.get("path")
        if existing_path and existing_path != str(manifest.path):
            raise ValueError(
                "Cannot resume eval suite with a different manifest path: "
                f"{existing_path}"
            )
        existing_runners = manifest_payload.get("effective_runners")
        if existing_runners and list(existing_runners) != list(resolved_runners):
            raise ValueError(
                "Cannot resume eval suite with different effective runners: "
                f"{existing_runners}"
            )
        started_at = state.get("started_at") or started_at

    if not ledger_path.exists():
        return started_at, [], set()

    rows_by_case: dict[int, list[dict]] = defaultdict(list)
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        case_index = int(payload.get("case_index", 0) or 0)
        if case_index <= 0:
            continue
        rows_by_case[case_index].append(payload)

    completed_case_indexes: set[int] = set()
    results: list[EvalResult] = []
    for case_index in sorted(rows_by_case):
        rows = rows_by_case[case_index]
        if len(rows) < runner_count:
            continue
        completed_case_indexes.add(case_index)
        for payload in rows[:runner_count]:
            results.append(_eval_result_from_payload(payload.get("result") or {}))

    return started_at, results, completed_case_indexes


def _write_eval_suite_run_state(
    output_root: Path,
    manifest: EvalSuiteManifest,
    resolved_runners: list[str],
    status: str,
    started_at: str,
    completed_cases: int,
    result_count: int,
    last_case_name: str | None = None,
    error: str | None = None,
    active_case_index: int | None = None,
    active_case_name: str | None = None,
    active_case_started_at: str | None = None,
    active_case_output_root: Path | None = None,
) -> None:
    """Persist suite lifecycle state so interrupted runs remain inspectable."""
    payload = {
        "manifest": {
            "name": manifest.name,
            "path": str(manifest.path),
            "runners": manifest.runners,
            "effective_runners": resolved_runners,
        },
        "status": status,
        "started_at": started_at,
        "updated_at": _utc_now_iso(),
        "total_cases": len(manifest.cases),
        "completed_cases": completed_cases,
        "result_count": result_count,
    }
    if last_case_name:
        payload["last_case_name"] = last_case_name
    if error:
        payload["error"] = error
    if active_case_index is not None and active_case_name:
        payload["active_case"] = {
            "index": active_case_index,
            "name": active_case_name,
            "started_at": active_case_started_at,
            "output_root": str(active_case_output_root)
            if active_case_output_root is not None
            else None,
        }
    if status != "running":
        payload["finished_at"] = payload["updated_at"]
    (output_root / "suite-run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _append_eval_suite_case_results(
    output_root: Path,
    case_index: int,
    case: EvalSuiteCase,
    case_results: list[EvalResult],
) -> None:
    """Append finalized case results to a durable JSONL ledger."""
    ledger_path = output_root / "suite-results.jsonl"
    with ledger_path.open("a", encoding="utf-8") as handle:
        for result in case_results:
            payload = {
                "case_index": case_index,
                "case_name": case.name,
                "case_kind": case.kind,
                "result": result.to_dict(),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _eval_result_from_payload(payload: dict) -> EvalResult:
    """Rehydrate an EvalResult from a persisted JSON payload."""
    metrics_payload = payload.get("metrics")
    metrics = None
    if isinstance(metrics_payload, dict):
        grounding = [
            GroundingMetric(
                line=int(item.get("line", 0) or 0),
                raw=str(item.get("raw", "")),
                value=float(item.get("value", 0.0) or 0.0),
                grounded=bool(item.get("grounded", False)),
            )
            for item in metrics_payload.get("grounding") or []
        ]
        metrics = EvalArtifactMetrics(
            compile_pass=bool(metrics_payload.get("compile_pass", False)),
            compile_issues=list(metrics_payload.get("compile_issues") or []),
            ci_pass=bool(metrics_payload.get("ci_pass", False)),
            ci_issues=list(metrics_payload.get("ci_issues") or []),
            embedded_source_present=bool(
                metrics_payload.get("embedded_source_present", False)
            ),
            grounded_numeric_count=int(
                metrics_payload.get("grounded_numeric_count", 0) or 0
            ),
            ungrounded_numeric_count=int(
                metrics_payload.get("ungrounded_numeric_count", 0) or 0
            ),
            grounding=grounding,
            source_numeric_occurrence_count=int(
                metrics_payload.get("source_numeric_occurrence_count", 0) or 0
            ),
            covered_source_numeric_occurrence_count=int(
                metrics_payload.get("covered_source_numeric_occurrence_count", 0) or 0
            ),
            missing_source_numeric_occurrence_count=int(
                metrics_payload.get("missing_source_numeric_occurrence_count", 0) or 0
            ),
            numeric_occurrence_issues=list(
                metrics_payload.get("numeric_occurrence_issues") or []
            ),
            generalist_review_pass=metrics_payload.get("generalist_review_pass"),
            generalist_review_score=metrics_payload.get("generalist_review_score"),
            generalist_review_issues=list(
                metrics_payload.get("generalist_review_issues") or []
            ),
            policyengine_pass=metrics_payload.get("policyengine_pass"),
            policyengine_score=metrics_payload.get("policyengine_score"),
            policyengine_issues=list(metrics_payload.get("policyengine_issues") or []),
            taxsim_pass=metrics_payload.get("taxsim_pass"),
            taxsim_score=metrics_payload.get("taxsim_score"),
            taxsim_issues=list(metrics_payload.get("taxsim_issues") or []),
        )

    return EvalResult(
        citation=str(payload.get("citation", "")),
        runner=str(payload.get("runner", "")),
        backend=str(payload.get("backend", "")),
        model=str(payload.get("model", "")),
        mode=payload.get("mode", "cold"),
        output_file=str(payload.get("output_file", "")),
        trace_file=str(payload.get("trace_file", "")),
        context_manifest_file=str(payload.get("context_manifest_file", "")),
        duration_ms=int(payload.get("duration_ms", 0) or 0),
        success=bool(payload.get("success", False)),
        error=payload.get("error"),
        input_tokens=int(payload.get("input_tokens", 0) or 0),
        output_tokens=int(payload.get("output_tokens", 0) or 0),
        cache_read_tokens=int(payload.get("cache_read_tokens", 0) or 0),
        cache_creation_tokens=int(payload.get("cache_creation_tokens", 0) or 0),
        reasoning_output_tokens=int(payload.get("reasoning_output_tokens", 0) or 0),
        estimated_cost_usd=payload.get("estimated_cost_usd"),
        actual_cost_usd=payload.get("actual_cost_usd"),
        retrieved_files=list(payload.get("retrieved_files") or []),
        unexpected_accesses=list(payload.get("unexpected_accesses") or []),
        metrics=metrics,
    )


def _suite_case_failure_results(
    case: EvalSuiteCase,
    runners: list[EvalRunnerSpec],
    exc: Exception,
) -> list[EvalResult]:
    """Convert an exception into explicit failed results for each runner."""
    return [
        EvalResult(
            citation=case.name,
            runner=runner.name,
            backend=runner.backend,
            model=runner.model,
            mode=case.mode,
            output_file="",
            trace_file="",
            context_manifest_file="",
            duration_ms=0,
            success=False,
            error=str(exc),
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            reasoning_output_tokens=0,
            estimated_cost_usd=None,
            actual_cost_usd=None,
            retrieved_files=[],
            unexpected_accesses=[],
            metrics=None,
        )
        for runner in runners
    ]


def _suite_case_results_should_retry(case_results: list[EvalResult]) -> bool:
    """Return True when a suite case likely failed for a transient reason."""
    if _suite_case_results_hit_usage_limit(case_results):
        return False
    return any(
        result.error is not None
        or result.metrics is None
        or _eval_result_indicates_retryable_timeout(result)
        for result in case_results
    )


def _suite_case_results_hit_usage_limit(case_results: list[EvalResult]) -> bool:
    """Return True when a case result indicates hard quota exhaustion."""
    return any(_eval_result_indicates_usage_limit(result) for result in case_results)


def _eval_result_indicates_usage_limit(result: EvalResult) -> bool:
    """Return True when one result contains a non-retryable usage-limit error."""
    texts: list[str] = []
    if result.error:
        texts.append(result.error)
    if result.metrics is not None:
        texts.extend(result.metrics.compile_issues)
        texts.extend(result.metrics.ci_issues)
        texts.extend(result.metrics.generalist_review_issues)
        texts.extend(result.metrics.policyengine_issues)
        texts.extend(result.metrics.taxsim_issues)

    return any("usage limit" in text.lower() for text in texts)


def _eval_result_indicates_retryable_timeout(result: EvalResult) -> bool:
    """Return True when one result failed due to a transient timeout."""
    texts: list[str] = []
    if result.error:
        texts.append(result.error)
    if result.metrics is not None:
        texts.extend(result.metrics.compile_issues)
        texts.extend(result.metrics.ci_issues)
        texts.extend(result.metrics.generalist_review_issues)
        texts.extend(result.metrics.policyengine_issues)
        texts.extend(result.metrics.taxsim_issues)

    lowered = [text.lower() for text in texts]
    return any("timeout after" in text or "timed out" in text for text in lowered)


def summarize_readiness(
    results: list[EvalResult],
    gates: EvalReadinessGates,
) -> EvalReadinessSummary:
    """Summarize suite readiness for one runner."""
    total_cases = len(results)
    success_rate = _fraction(sum(1 for result in results if result.success), total_cases)
    compile_pass_rate = _fraction(
        sum(
            1
            for result in results
            if result.metrics is not None and result.metrics.compile_pass
        ),
        total_cases,
    )
    ci_pass_rate = _fraction(
        sum(
            1
            for result in results
            if result.metrics is not None and result.metrics.ci_pass
        ),
        total_cases,
    )
    zero_ungrounded_rate = _fraction(
        sum(
            1
            for result in results
            if result.metrics is not None and result.metrics.ungrounded_numeric_count == 0
        ),
        total_cases,
    )
    generalist_review_pass_rate = _fraction(
        sum(
            1
            for result in results
            if result.metrics is not None and result.metrics.generalist_review_pass
        ),
        total_cases,
    )
    generalist_scores = [
        result.metrics.generalist_review_score
        for result in results
        if result.metrics is not None
        and result.metrics.generalist_review_score is not None
    ]
    mean_generalist_review_score = (
        round(mean(generalist_scores), 6) if generalist_scores else None
    )

    policyengine_results = [
        result
        for result in results
        if result.metrics is not None and result.metrics.policyengine_score is not None
    ]
    policyengine_case_count = len(policyengine_results)
    policyengine_pass_rate = (
        _fraction(
            sum(
                1
                for result in policyengine_results
                if result.metrics is not None and result.metrics.policyengine_pass
            ),
            policyengine_case_count,
        )
        if policyengine_case_count
        else None
    )
    mean_policyengine_score = (
        round(
            mean(
                result.metrics.policyengine_score
                for result in policyengine_results
                if result.metrics is not None
                and result.metrics.policyengine_score is not None
            ),
            6,
        )
        if policyengine_case_count
        else None
    )

    costs = [
        result.estimated_cost_usd
        for result in results
        if result.estimated_cost_usd is not None
    ]
    mean_estimated_cost_usd = round(mean(costs), 6) if costs else None

    gate_results: list[EvalReadinessGateResult] = [
        _min_gate("min_cases", total_cases, gates.min_cases),
    ]
    if gates.min_success_rate is not None:
        gate_results.append(
            _min_gate("min_success_rate", success_rate, gates.min_success_rate)
        )
    if gates.min_compile_pass_rate is not None:
        gate_results.append(
            _min_gate(
                "min_compile_pass_rate",
                compile_pass_rate,
                gates.min_compile_pass_rate,
            )
        )
    if gates.min_ci_pass_rate is not None:
        gate_results.append(
            _min_gate("min_ci_pass_rate", ci_pass_rate, gates.min_ci_pass_rate)
        )
    if gates.min_zero_ungrounded_rate is not None:
        gate_results.append(
            _min_gate(
                "min_zero_ungrounded_rate",
                zero_ungrounded_rate,
                gates.min_zero_ungrounded_rate,
            )
        )
    if gates.min_generalist_review_pass_rate is not None:
        gate_results.append(
            _min_gate(
                "min_generalist_review_pass_rate",
                generalist_review_pass_rate,
                gates.min_generalist_review_pass_rate,
            )
        )
    if gates.min_policyengine_pass_rate is not None:
        gate_results.append(
            _min_gate(
                "min_policyengine_pass_rate",
                policyengine_pass_rate,
                gates.min_policyengine_pass_rate,
            )
        )
    if gates.max_mean_estimated_cost_usd is not None:
        gate_results.append(
            _max_gate(
                "max_mean_estimated_cost_usd",
                mean_estimated_cost_usd,
                gates.max_mean_estimated_cost_usd,
            )
        )

    return EvalReadinessSummary(
        total_cases=total_cases,
        success_rate=success_rate,
        compile_pass_rate=compile_pass_rate,
        ci_pass_rate=ci_pass_rate,
        zero_ungrounded_rate=zero_ungrounded_rate,
        generalist_review_pass_rate=generalist_review_pass_rate,
        mean_generalist_review_score=mean_generalist_review_score,
        policyengine_case_count=policyengine_case_count,
        policyengine_pass_rate=policyengine_pass_rate,
        mean_policyengine_score=mean_policyengine_score,
        mean_estimated_cost_usd=mean_estimated_cost_usd,
        gate_results=gate_results,
        ready=all(result.passed for result in gate_results),
    )


def _coerce_eval_mode(value: str) -> EvalMode:
    """Validate a manifest eval mode."""
    normalized = str(value).strip()
    if normalized not in {"cold", "repo-augmented"}:
        raise ValueError(f"Unsupported eval mode '{value}'")
    return normalized  # type: ignore[return-value]


def _optional_float(value: object) -> float | None:
    """Convert optional numeric manifest values to float."""
    if value is None:
        return None
    return float(value)


def _resolve_manifest_path(base_dir: Path, value: object) -> Path:
    """Resolve a manifest path entry relative to the manifest file."""
    path = Path(str(value))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _validate_eval_suite_case(case: EvalSuiteCase, index: int) -> None:
    """Validate one suite case after parsing."""
    if case.kind == "citation" and not case.citation:
        raise ValueError(f"Eval suite case #{index} is missing 'citation'")
    if case.kind == "source":
        if not case.source_id:
            raise ValueError(f"Eval suite case #{index} is missing 'source_id'")
        if case.source_file is None:
            raise ValueError(f"Eval suite case #{index} is missing 'source_file'")
    if case.kind == "akn_section":
        if not case.source_id:
            raise ValueError(f"Eval suite case #{index} is missing 'source_id'")
        if case.akn_file is None:
            raise ValueError(f"Eval suite case #{index} is missing 'akn_file'")
        if not case.section_eid:
            raise ValueError(f"Eval suite case #{index} is missing 'section_eid'")
    if case.kind == "uk_legislation" and not case.source_ref:
        raise ValueError(f"Eval suite case #{index} is missing 'source_ref'")


def _fraction(numerator: int, denominator: int) -> float:
    """Return a rounded fraction or 0 when the denominator is empty."""
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _min_gate(
    name: str,
    actual: float | int | None,
    threshold: float | int,
) -> EvalReadinessGateResult:
    """Evaluate a lower-bound readiness gate."""
    return EvalReadinessGateResult(
        name=name,
        comparator="min",
        threshold=threshold,
        actual=actual,
        passed=actual is not None and actual >= threshold,
    )


def _max_gate(
    name: str,
    actual: float | int | None,
    threshold: float | int,
) -> EvalReadinessGateResult:
    """Evaluate an upper-bound readiness gate."""
    return EvalReadinessGateResult(
        name=name,
        comparator="max",
        threshold=threshold,
        actual=actual,
        passed=actual is not None and actual <= threshold,
    )


def select_context_files(
    citation: str | CitationParts,
    rac_us_root: Path,
    max_files: int = 6,
) -> list[Path]:
    """Select canonical implementation precedent files for repo-augmented evals."""
    parts = citation if isinstance(citation, CitationParts) else parse_usc_citation(citation)
    section_root = Path(rac_us_root) / parts.title / parts.section
    target_rel = citation_to_relative_rac_path(parts)
    target_path = Path(rac_us_root) / target_rel

    candidates: list[Path] = []
    if section_root.exists():
        candidates.extend(
            sorted(
                path
                for path in section_root.rglob("*.rac")
                if path.resolve() != target_path.resolve()
            )
        )

    if not candidates:
        title_root = Path(rac_us_root) / parts.title
        candidates.extend(
            sorted(path for path in title_root.rglob("*.rac") if path != target_path)
        )

    # Bias toward nearby files first, then shallower paths for readability.
    candidates.sort(
        key=lambda path: (
            0 if path.parent == section_root else 1,
            len(path.relative_to(rac_us_root).parts),
            str(path),
        )
    )

    selected: list[Path] = []
    for candidate in candidates:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= max_files:
            break
    return selected


def prepare_eval_workspace(
    citation: str,
    runner: EvalRunnerSpec,
    output_root: Path,
    source_text: str,
    rac_path: Path,
    mode: EvalMode,
    extra_context_paths: list[Path] | None = None,
) -> EvalWorkspace:
    """Create an isolated workspace bundle for a single eval."""
    slug = _slugify(citation)
    workspace_root = (
        Path(output_root) / "_eval_workspaces" / runner.name / slug / "workspace"
    )
    if workspace_root.parent.exists():
        shutil.rmtree(workspace_root.parent)
    workspace_root.mkdir(parents=True, exist_ok=True)

    source_file = workspace_root / "source.txt"
    source_file.write_text(source_text.strip() + "\n")

    context_files: list[EvalContextFile] = []
    context_root = workspace_root / "context"
    target_rel = _target_rel_for_eval_identifier(citation)
    current_file = rac_path / target_rel if target_rel is not None else None
    for resolved_term in resolve_defined_terms_from_text(source_text):
        context_files.append(
            _materialize_resolved_definition_stub(
                context_root=context_root,
                resolved_term=resolved_term,
                workspace_root=workspace_root,
            )
        )
    for resolved_concept in resolve_canonical_concepts_from_text(
        source_text,
        rac_path,
        current_file=current_file,
    ):
        context_files.append(
            _materialize_resolved_canonical_concept(
                context_root=context_root,
                resolved_concept=resolved_concept,
                workspace_root=workspace_root,
            )
        )

    rac_us_root = rac_path.parent / "rac-us" / "statute"

    if mode == "repo-augmented":
        selected = _auto_select_context_files(citation, rac_us_root)
        for extra_path in extra_context_paths or []:
            path = Path(extra_path)
            if path.exists():
                selected.append(path)

        expanded_context = _expand_context_files(selected, rac_us_root, target_rel)

        for source_path, kind in expanded_context:
            relative_target = _context_import_relative_target(source_path, rac_path)

            workspace_relative_path = Path("context") / relative_target
            workspace_path = workspace_root / workspace_relative_path
            workspace_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, workspace_path)
            context_files.append(
                EvalContextFile(
                    source_path=str(source_path),
                    workspace_path=str(workspace_relative_path),
                    import_path=_relative_rac_path_to_import_target(relative_target),
                    kind=kind,
                )
            )

    manifest_file = workspace_root / "context-manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "citation": citation,
                "mode": mode,
                "source_file": str(source_file.relative_to(workspace_root)),
                "context_files": [asdict(item) for item in context_files],
            },
            indent=2,
            sort_keys=True,
        )
    )

    return EvalWorkspace(
        root=workspace_root,
        source_file=source_file,
        manifest_file=manifest_file,
        context_files=context_files,
    )


def _materialize_resolved_definition_stub(
    *,
    context_root: Path,
    resolved_term: ResolvedDefinedTerm,
    workspace_root: Path,
) -> EvalContextFile:
    """Write one resolved definition stub into the eval workspace context."""
    relative_target = Path("context") / import_target_to_relative_rac_path(
        resolved_term.import_target
    )
    materialize_registered_stub(
        workspace_root,
        [resolved_term],
        prefix=Path("context"),
    )
    return EvalContextFile(
        source_path=resolved_term.citation,
        workspace_path=str(relative_target),
        import_path=_relative_rac_path_to_import_target(
            relative_target.relative_to("context")
        ),
        kind="definition_stub",
        label=resolved_term.label,
    )


def _materialize_resolved_canonical_concept(
    *,
    context_root: Path,
    resolved_concept: ResolvedCanonicalConcept,
    workspace_root: Path,
) -> EvalContextFile:
    """Copy one resolved canonical concept file into the eval workspace context."""
    relative_target = Path("context") / import_target_to_relative_rac_path(
        resolved_concept.import_target
    )
    target = workspace_root / relative_target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_concept.source_file, target)
    return EvalContextFile(
        source_path=str(resolved_concept.source_file),
        workspace_path=str(relative_target),
        import_path=_relative_rac_path_to_import_target(
            relative_target.relative_to("context")
        ),
        kind="canonical_concept",
        label=resolved_concept.label,
    )


def _auto_select_context_files(citation: str, rac_us_root: Path) -> list[Path]:
    """Best-effort auto-context selection for statute citations only."""
    try:
        return select_context_files(citation, rac_us_root)
    except Exception:
        return []


def _context_import_relative_target(source_path: Path, rac_path: Path) -> Path:
    """Prefer canonical repo-relative import targets for copied precedent files."""
    repo_parent = rac_path.parent.resolve()
    resolved_source = source_path.resolve()

    for candidate in sorted(repo_parent.glob("rac*")):
        if not candidate.is_dir():
            continue
        resolved_candidate = candidate.resolve()
        with contextlib.suppress(ValueError):
            return resolved_source.relative_to(resolved_candidate)

    return Path("external") / resolved_source.name


def _relative_rac_path_to_import_target(path: Path) -> str:
    """Convert a relative RAC file path into a bare import target."""
    normalized = path.with_suffix("") if path.suffix == ".rac" else path
    return normalized.as_posix()


def _target_rel_for_eval_identifier(citation: str) -> Path | None:
    """Return the canonical RAC target path for USC citations when parseable."""
    try:
        return citation_to_relative_rac_path(citation)
    except Exception:
        return None


def evaluate_artifact(
    rac_file: Path,
    rac_root: Path,
    rac_path: Path,
    source_text: str,
    oracle: EvalOracleMode = "none",
    policyengine_country: str = "auto",
    policyengine_rac_var_hint: str | None = None,
) -> EvalArtifactMetrics:
    """Evaluate one RAC file with deterministic checks plus optional oracles."""
    pipeline = ValidatorPipeline(
        rac_us_path=rac_root,
        rac_path=rac_path,
        enable_oracles=oracle != "none",
        policyengine_country=policyengine_country,
        policyengine_rac_var_hint=policyengine_rac_var_hint,
    )
    compile_result = pipeline._run_compile_check(rac_file)
    ci_result = pipeline._run_ci(rac_file)

    policyengine_result = None
    taxsim_result = None
    if oracle in ("policyengine", "all"):
        try:
            policyengine_result = pipeline._run_policyengine(rac_file)
        except Exception as exc:
            policyengine_result = ValidationResult(
                validator_name="policyengine",
                passed=False,
                error=str(exc),
                issues=[str(exc)],
            )
    if oracle == "all":
        try:
            taxsim_result = pipeline._run_taxsim(rac_file)
        except Exception as exc:
            taxsim_result = ValidationResult(
                validator_name="taxsim",
                passed=False,
                error=str(exc),
                issues=[str(exc)],
            )

    oracle_context: dict[str, dict[str, object]] = {}
    if policyengine_result is not None:
        oracle_context["policyengine"] = {
            "score": policyengine_result.score,
            "passed": policyengine_result.passed,
            "issues": policyengine_result.issues,
            "duration_ms": policyengine_result.duration_ms,
        }
    if taxsim_result is not None:
        oracle_context["taxsim"] = {
            "score": taxsim_result.score,
            "passed": taxsim_result.passed,
            "issues": taxsim_result.issues,
            "duration_ms": taxsim_result.duration_ms,
        }
    review_context = (
        "This review is running inside an eval-suite benchmark workspace. "
        "The artifact file path is generic benchmark output and is not itself the legal citation. "
        "Benchmark directory labels may be stale, generic, or misleading and must be ignored as legal cues. "
        "The benchmark target is an atomic source slice, so judge fidelity to exactly this slice rather than demanding omitted sibling limbs or parent consequences unless the RAC claims to encode them. "
        "Judge citation fidelity against the embedded source-text docstring and this authoritative source excerpt:\n\n"
        f"{source_text.strip()[:4000]}"
    )
    if re.search(
        r"\bon the first day\b|\bnext benefit week\b|\bon or after the day\b",
        source_text,
        flags=re.IGNORECASE,
    ):
        review_context += (
            "\n\nThis is a temporal timing clause. RAC does not expose a native date-valued output in this eval, "
            "so a boolean day-predicate helper on `period: Day`, plus explicit trigger preconditions from the source text, "
            "is an acceptable representation."
        )
    try:
        generalist_review_result = pipeline._run_reviewer(
            "generalist-reviewer",
            rac_file,
            oracle_context or None,
            review_context=review_context,
        )
    except Exception as exc:
        generalist_review_result = ValidationResult(
            validator_name="generalist-reviewer",
            passed=False,
            error=str(exc),
            issues=[f"Reviewer error: {exc}"],
        )

    content = rac_file.read_text()
    embedded_source = extract_embedded_source_text(content)
    source_numbers = extract_numbers_from_text(embedded_source or source_text)
    source_numeric_occurrences = Counter(
        extract_numeric_occurrences_from_text(embedded_source or source_text)
    )
    named_scalar_occurrences = Counter(
        item.value for item in extract_named_scalar_occurrences(content)
    )

    grounding_metrics: list[GroundingMetric] = []
    for line, raw, value in extract_grounding_values(content):
        grounding_metrics.append(
            GroundingMetric(
                line=line,
                raw=raw,
                value=value,
                grounded=value in source_numbers,
            )
        )

    numeric_occurrence_issues: list[str] = []
    covered_source_numeric_occurrence_count = 0
    missing_source_numeric_occurrence_count = 0
    for value, expected_count in sorted(source_numeric_occurrences.items()):
        covered_count = min(expected_count, named_scalar_occurrences.get(value, 0))
        covered_source_numeric_occurrence_count += covered_count
        if covered_count < expected_count:
            missing_count = expected_count - covered_count
            missing_source_numeric_occurrence_count += missing_count
            numeric_occurrence_issues.append(
                f"Source numeric value {value:g} appears {expected_count} time(s), "
                f"but only {covered_count} named scalar definition(s) with that value were found."
            )

    ci_issues = list(ci_result.issues) + numeric_occurrence_issues
    ci_pass = ci_result.passed and not numeric_occurrence_issues

    return EvalArtifactMetrics(
        compile_pass=compile_result.passed,
        compile_issues=compile_result.issues,
        ci_pass=ci_pass,
        ci_issues=ci_issues,
        embedded_source_present=bool(embedded_source),
        grounded_numeric_count=sum(1 for item in grounding_metrics if item.grounded),
        ungrounded_numeric_count=sum(
            1 for item in grounding_metrics if not item.grounded
        ),
        grounding=grounding_metrics,
        source_numeric_occurrence_count=sum(source_numeric_occurrences.values()),
        covered_source_numeric_occurrence_count=covered_source_numeric_occurrence_count,
        missing_source_numeric_occurrence_count=missing_source_numeric_occurrence_count,
        numeric_occurrence_issues=numeric_occurrence_issues,
        generalist_review_pass=generalist_review_result.passed,
        generalist_review_score=generalist_review_result.score,
        generalist_review_issues=generalist_review_result.issues,
        policyengine_pass=(
            policyengine_result.passed if policyengine_result is not None else None
        ),
        policyengine_score=(
            policyengine_result.score if policyengine_result is not None else None
        ),
        policyengine_issues=(
            policyengine_result.issues if policyengine_result is not None else []
        ),
        taxsim_pass=taxsim_result.passed if taxsim_result is not None else None,
        taxsim_score=taxsim_result.score if taxsim_result is not None else None,
        taxsim_issues=taxsim_result.issues if taxsim_result is not None else [],
    )


def _run_single_eval(
    citation: str,
    runner: EvalRunnerSpec,
    output_root: Path,
    rac_path: Path,
    xml_root: Path,
    mode: EvalMode,
    extra_context_paths: list[Path],
) -> EvalResult:
    source_text = find_citation_text(citation, xml_root)
    if not source_text:
        raise ValueError(f"No statute text found for {citation}")

    workspace = prepare_eval_workspace(
        citation=citation,
        runner=runner,
        output_root=output_root,
        source_text=source_text,
        rac_path=rac_path,
        mode=mode,
        extra_context_paths=extra_context_paths,
    )

    relative_output = citation_to_relative_rac_path(citation)
    prompt = _build_eval_prompt(
        citation,
        mode,
        workspace,
        workspace.context_files,
        target_file_name=relative_output.name,
        include_tests=False,
        runner_backend=runner.backend,
    )
    response = _run_prompt_eval(runner, workspace, prompt)
    output_file = Path(output_root) / runner.name / relative_output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    wrote_artifact = _materialize_eval_artifact(
        response.text,
        output_file,
        source_text=source_text,
        workspace_root=workspace.root,
    )
    if wrote_artifact:
        _hydrate_eval_root(output_file.parents[1], workspace)

    trace_file = (
        Path(output_root)
        / "traces"
        / runner.name
        / f"{_slugify(citation)}.json"
    )
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(json.dumps(response.trace or {}, indent=2, sort_keys=True))

    metrics = None
    if output_file.exists():
        metrics = evaluate_artifact(
            rac_file=output_file,
            rac_root=output_file.parents[1],
            rac_path=rac_path,
            source_text=source_text,
        )

    tokens = response.tokens
    result = EvalResult(
        citation=citation,
        runner=runner.name,
        backend=runner.backend,
        model=runner.model,
        mode=mode,
        output_file=str(output_file),
        trace_file=str(trace_file),
        context_manifest_file=str(workspace.manifest_file),
        duration_ms=response.duration_ms,
        success=wrote_artifact and response.error is None,
        error=response.error or (None if wrote_artifact else "No RAC content returned"),
        input_tokens=tokens.input_tokens if tokens else 0,
        output_tokens=tokens.output_tokens if tokens else 0,
        cache_read_tokens=tokens.cache_read_tokens if tokens else 0,
        cache_creation_tokens=tokens.cache_creation_tokens if tokens else 0,
        reasoning_output_tokens=tokens.reasoning_output_tokens if tokens else 0,
        estimated_cost_usd=response.estimated_cost_usd,
        actual_cost_usd=response.actual_cost_usd,
        retrieved_files=[item.source_path for item in workspace.context_files],
        unexpected_accesses=response.unexpected_accesses,
        metrics=metrics,
    )
    emit_eval_result(result, response.trace)
    return result


def _run_single_source_eval(
    source_id: str,
    source_text: str,
    runner: EvalRunnerSpec,
    output_root: Path,
    rac_path: Path,
    mode: EvalMode,
    extra_context_paths: list[Path],
    oracle: EvalOracleMode,
    policyengine_country: str,
    policyengine_rac_var_hint: str | None,
) -> EvalResult:
    """Run one eval on an arbitrary source slice rather than a USC citation."""
    workspace = prepare_eval_workspace(
        citation=source_id,
        runner=runner,
        output_root=output_root,
        source_text=source_text,
        rac_path=rac_path,
        mode=mode,
        extra_context_paths=extra_context_paths,
    )

    relative_output = _source_identifier_to_relative_rac_path(source_id)
    prompt = _build_eval_prompt(
        source_id,
        mode,
        workspace,
        workspace.context_files,
        target_file_name=relative_output.name,
        include_tests=True,
        runner_backend=runner.backend,
        policyengine_rac_var_hint=policyengine_rac_var_hint,
    )
    response = _run_prompt_eval(runner, workspace, prompt)
    output_file = Path(output_root) / runner.name / relative_output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    wrote_artifact = _materialize_eval_artifact(
        response.text,
        output_file,
        source_text=source_text,
        workspace_root=workspace.root,
    )
    if wrote_artifact:
        _hydrate_eval_root(output_file.parents[1], workspace)

    trace_file = (
        Path(output_root)
        / "traces"
        / runner.name
        / f"{_slugify(source_id)}.json"
    )
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(json.dumps(response.trace or {}, indent=2, sort_keys=True))

    metrics = None
    if output_file.exists():
        metrics = evaluate_artifact(
            rac_file=output_file,
            rac_root=output_file.parents[1],
            rac_path=rac_path,
            source_text=source_text,
            oracle=oracle,
            policyengine_country=policyengine_country,
            policyengine_rac_var_hint=policyengine_rac_var_hint,
        )

    tokens = response.tokens
    result = EvalResult(
        citation=source_id,
        runner=runner.name,
        backend=runner.backend,
        model=runner.model,
        mode=mode,
        output_file=str(output_file),
        trace_file=str(trace_file),
        context_manifest_file=str(workspace.manifest_file),
        duration_ms=response.duration_ms,
        success=wrote_artifact and response.error is None,
        error=response.error or (None if wrote_artifact else "No RAC content returned"),
        input_tokens=tokens.input_tokens if tokens else 0,
        output_tokens=tokens.output_tokens if tokens else 0,
        cache_read_tokens=tokens.cache_read_tokens if tokens else 0,
        cache_creation_tokens=tokens.cache_creation_tokens if tokens else 0,
        reasoning_output_tokens=tokens.reasoning_output_tokens if tokens else 0,
        estimated_cost_usd=response.estimated_cost_usd,
        actual_cost_usd=response.actual_cost_usd,
        retrieved_files=[item.source_path for item in workspace.context_files],
        unexpected_accesses=response.unexpected_accesses,
        metrics=metrics,
    )
    emit_eval_result(result, response.trace)
    return result


def _source_identifier_to_relative_rac_path(source_id: str) -> Path:
    """Map an arbitrary source identifier to a stable eval artifact path."""
    return Path("source") / f"{_slugify(source_id)}.rac"


def _build_eval_prompt(
    citation: str,
    mode: EvalMode,
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
    target_file_name: str,
    include_tests: bool = False,
    runner_backend: str = "codex",
    policyengine_rac_var_hint: str | None = None,
) -> str:
    """Build a prompt-only eval request with explicit provenance rules."""
    source_text = workspace.source_file.read_text().strip()
    single_amount_table_slice = _is_single_amount_table_slice(source_text)
    definition_context_files = [
        item for item in context_files if item.kind == "definition_stub"
    ]
    canonical_concept_context_files = [
        item for item in context_files if item.kind == "canonical_concept"
    ]
    precedent_context_files = [
        item
        for item in context_files
        if item.kind not in {"definition_stub", "canonical_concept"}
    ]

    inline_source_section = ""
    if runner_backend == "openai":
        inline_source_section = f"""
You do not have filesystem tool access in this eval. The exact contents of `./source.txt`
are copied inline below and must be treated as the contents of that file.

=== BEGIN SOURCE.TXT ===
{source_text}
=== END SOURCE.TXT ===
"""

    definition_section = ""
    if definition_context_files:
        listed_definitions = "\n".join(
            _format_context_file_listing(item, include_label=True)
            for item in definition_context_files
        )
        inline_definition_copies = ""
        if runner_backend == "openai":
            inline_definition_copies = f"""

Inline resolved definition file copies:
{_format_inline_context_snippets(workspace, definition_context_files)}
"""
        definition_section = f"""
Resolved definition files are available below.
If `./source.txt` uses one of these defined terms, import the listed canonical definition instead of inventing a local helper:
{listed_definitions}

Inspect the copied file at the listed `./context/...` path when needed, but emit any RAC `imports:` entry using the listed import target rather than the `./context/` inspection path.

Exact RAC import syntax for a resolved definition:
resolved_term_local_name:
    imports:
        - legislation/ukpga/2002/16/section/3ZA/3#is_member_of_mixed_age_couple
    entity: Person
    period: Day
    dtype: Boolean
    from 2025-03-21:
        is_member_of_mixed_age_couple

Do not replace that import with a local deferred stub or a path-mangled variable name like
`legislation_ukpga_2002_16_section_3ZA_3_is_member_of_mixed_age_couple`.
{inline_definition_copies}
"""

    canonical_concept_section = ""
    if canonical_concept_context_files:
        listed_concepts = "\n".join(
            _format_context_file_listing(item, include_label=True)
            for item in canonical_concept_context_files
        )
        inline_concept_copies = ""
        if runner_backend == "openai":
            inline_concept_copies = f"""

Inline canonical concept file copies:
{_format_inline_context_snippets(workspace, canonical_concept_context_files)}
"""
        canonical_concept_section = f"""
Resolved canonical concept files from this corpus are available below.
If `./source.txt` uses one of these legal concepts, import the listed canonical definition instead of restating that concept locally:
{listed_concepts}

Inspect the copied file at the listed `./context/...` path when needed, but emit any RAC `imports:` entry using the listed import target rather than the `./context/` inspection path.

Exact RAC import syntax for a copied canonical concept:
local_fact_name:
    imports:
        - statute/crs/26-2-703/12#is_individual_responsibility_contract
    entity: Person
    period: Month
    dtype: Boolean
    from 2026-04-03:
        is_individual_responsibility_contract

Because the canonical concept file already exists in this workspace, do not keep that concept as a leaf-local helper when the listed import target matches the source text.
{inline_concept_copies}
"""

    context_section = ""
    if precedent_context_files:
        listed = "\n".join(
            _format_context_file_listing(item) for item in precedent_context_files
        )
        scaffold_dates = _collect_scaffold_dates(workspace, precedent_context_files)
        scaffold_date_lines = ""
        if scaffold_dates:
            dates = ", ".join(f"`{date}`" for date in scaffold_dates)
            scaffold_date_lines = f"""
Allowed temporal scaffold dates copied from precedent:
- {dates}

If the source text is silent on effective dates but RAC syntax requires a `from YYYY-MM-DD:` clause,
you may use one scaffold date from the copied precedent files.
Prefer the earliest scaffold date unless a matching copied concept clearly uses a later one.
Numeric grounding rules do not apply to the YYYY-MM-DD tokens inside that temporal clause.
"""
        if runner_backend == "openai":
            context_section = f"""
Implementation precedent files are available below as inline copies.
You may use ONLY the copied files below for syntax, naming, entity, import, re-export, and style conventions.
They are not legal authority and may not justify new substantive numeric values.
Prefer importing or re-exporting a copied concept instead of inventing a fresh stub input when a matching concept already exists.
When you emit a RAC `imports:` entry based on one of these copied files, use the listed import target rather than the `./context/...` inspection path.
{scaffold_date_lines}

Available precedent files:
{listed}

Inline precedent file copies:
{_format_inline_context_snippets(workspace, context_files)}
"""
        else:
            context_section = f"""
Implementation precedent files are available under `./context/`.
You may use ONLY the copied files below for syntax, naming, entity, import, re-export, and style conventions.
They are not legal authority and may not justify new substantive numeric values.
Prefer importing or re-exporting a copied concept instead of inventing a fresh stub input when a matching concept already exists.
When you emit a RAC `imports:` entry based on one of these copied files, use the listed import target rather than the `./context/...` inspection path.
You may inspect `./source.txt`, `./context-manifest.json`, and the listed copied files only when needed.
{scaffold_date_lines}

Available precedent files:
{listed}
"""

    file_output_rules = f"""
- Return ONLY raw `.rac` file content for `{target_file_name}`.
- Do not include markdown fences or explanation.
"""
    if include_tests:
        test_file_name = Path(target_file_name).with_suffix(".rac.test").name
        file_output_rules = f"""
- Return exactly this two-file bundle and nothing else:
=== FILE: {target_file_name} ===
<raw .rac content>
=== FILE: {test_file_name} ===
<raw .rac.test YAML>
- The `.rac.test` file must contain 1-4 cases, unless the `.rac` file is fully `status: deferred` or `status: entity_not_supported` with no assertable outputs.
- For a fully deferred or `entity_not_supported` fallback file with no assertable outputs, leave `.rac.test` empty instead of emitting `output: {{}}` or assertions against deferred symbols.
- If `./source.txt` is omitted/repealed text shown only by ellipses or otherwise contains no operative rule content for the target slice, emit only a top-level `status: deferred` (or `status: entity_not_supported` when appropriate), keep the embedded source/docstring showing that omission, and emit no local rule blocks.
- For ordinary source slices, the `.rac.test` file should usually contain 3-4 cases covering true/applicable, false/inapplicable, and boundary or alternate factual branches.
- Only a single fixed-amount source slice may use 1-2 cases.
- The `.rac.test` file must be a YAML list of cases beginning with `- name:` entries, not a top-level mapping keyed by case names.
- For a single fixed-amount source slice, a base case is sufficient.
- Add an effective-date boundary only when the period supports a meaningful point-in-time boundary.
- Add an alternate branch only when `./source.txt` states another grounded branch condition or amount.
- Test inputs must contain factual predicates or quantities, not the output variable being asserted.
- In `.rac.test`, input and output values must be plain scalars or simple mappings, not inline variable declarations with keys like `entity`, `period`, `dtype`, `values`, or `from ...`.
- Use `output:` mappings in `.rac.test` cases, not `expect:` blocks.
- Use concrete ISO calendar dates like `2025-03-21` in `.rac.test` `period:` fields; do not use ISO week strings like `2025-W13`.
- The `.rac.test` file must contain YAML only, with no trailing notes or prose.
- Do not include markdown fences or explanation.
"""

    schema_rules = f"""
- Do not invent new entities, periods, or dtypes.
- Allowed `entity:` values are {", ".join(f"`{entity}`" for entity in SUPPORTED_EVAL_ENTITIES)}.
- Allowed `period:` values are {", ".join(f"`{period}`" for period in SUPPORTED_EVAL_PERIODS)}.
- Allowed `dtype:` values are {", ".join(f"`{dtype}`" for dtype in SUPPORTED_EVAL_DTYPES)}, or `Enum[Name]`.
- If the source cannot be represented faithfully with the supported schema, prefer a compileable fallback with `status: entity_not_supported` or `status: deferred` instead of inventing a new ontology.
"""

    uk_guidance = ""
    if re.match(r"^(?:ukpga|uksi|asp|ssi|wsi|nisi|anaw|asc)/", citation):
        uk_guidance = render_uk_legislation_guidance()
    single_amount_row_guidance = ""
    if single_amount_table_slice:
        single_amount_row_guidance = render_single_amount_row_guidance()
    date_silent_scaffold_guidance = render_date_silent_scaffold_guidance()
    target_hint_guidance = ""
    if policyengine_rac_var_hint:
        target_hint_guidance = f"""
- Prefer `{policyengine_rac_var_hint}` as the principal output variable name for this encoding unless the source text clearly requires a materially different concept.
"""

    return f"""You are participating in an encoding eval for {citation}.

Primary legal authority:
- `./source.txt` contains the complete source text for this target source slice.
{inline_source_section}

Context mode: `{mode}`
{definition_section}{canonical_concept_section}{context_section}

Rules:
- Do not inspect or rely on any path outside this workspace.
- Treat `./source.txt` as the only legal source.
- Any numeric literal in your output must appear in `./source.txt`, unless it is -1, 0, 1, 2, or 3.
- Every substantive numeric occurrence in `./source.txt` must be represented by a named scalar definition in RAC, even when the same numeric value repeats.
- If the same numeric value appears twice in materially different legal roles, declare separate named scalar variables for those separate occurrences instead of reusing a single scalar everywhere.
- If a legal scalar amount, threshold, cap, or limit appears in a formula or conditional branch, first declare it as its own named variable and then reference that variable from the formula.
- Once you declare a substantive numeric scalar, reuse that named scalar everywhere the rule compares against or computes with that number; do not restate the raw literal inline in formulas, comparisons, or tests.
- If `./source.txt` says someone is "aged 18 or over", "under 25", or gives another numeric eligibility threshold, model that threshold as a named scalar variable rather than only burying the number inside a helper name.
- Do not create scalar variables for citation numbers that only appear inside section, paragraph, regulation, schedule, or similar legal cross-references.
- Do not invent `dtype: String` variables just to restate the effective date or to hold quoted date text from `./source.txt`.
- Do not decompose legal dates into numeric `year`, `month`, or `day` scalar variables; keep date references semantic inside boolean/fact-shaped helpers instead.
- For phrases like `1st September following the person's 19th birthday`, keep the calendar-date portion semantic inside a boolean/fact helper; do not invent numeric `1`/`September` scalars, but do preserve the substantive `19` age threshold as a named scalar if your logic uses it.
- Include the source text in a triple-quoted docstring.
- Use RAC DSL conventions.
- If `./source.txt` explicitly cites another section or source for a definition, emit the upstream import instead of restating the concept locally.
- When `./source.txt` says a value is determined `in accordance with section X`, `under section X`, or another cited upstream computation, and a copied precedent file from that cited section exports the matching computed concept, import that exported concept instead of inventing a fresh local `*_under_section_X` input or helper.
- For example, if the source says an allotment is reduced by household income `as determined in accordance with section 2014(d) and (e)` and a copied precedent file exports `statute/7/2014/e#snap_net_income`, import `snap_net_income` rather than inventing a local input like `snap_household_income_under_2014_d_and_e`.
- If `./source.txt` uses a legally-defined term for which a resolved canonical definition file is provided above, import that canonical definition instead of inventing a leaf-local helper.
- If `./source.txt` uses a legal concept for which a copied canonical concept file is provided above, import or re-export that exact canonical concept instead of duplicating it locally.
- If `./source.txt` is an annual publication, table, or schedule that updates values for canonical concepts already defined in copied context, author it as an amendment layer targeting those canonical symbols rather than redefining the canonical outputs locally.
- For example, if copied context already defines `snap_one_person_thrifty_food_plan_cost`, `snap_minimum_allotment`, or `snap_maximum_allotment`, emit dated `amend` blocks for those canonical symbols instead of a second full local definition of the same output.
- Do not import a canonical output like `snap_one_person_thrifty_food_plan_cost` or `snap_maximum_allotment` and then redeclare that same variable locally in the same file. That creates duplicate-variable failures once the import closure is compiled.
- Wrong for annual parameter tables: importing `statute/...#snap_maximum_allotment` and then defining a new local `snap_maximum_allotment:` rule body. Right: keep the canonical symbol in context and emit `amend snap_maximum_allotment:` entries that update its values for the publication period.
- When a publication table keys values by household size, region, filing status, bracket row, or another schedule index, do not create documentary scalar constants like `snap_household_size_four: 4` just to restate the row labels. Compare directly against the canonical input or derive one helper like `additional_household_members_above_eight` only when the source actually requires that arithmetic.
- Right pattern for the USDA SNAP FY2026 table:
```rac
imports:
  - statute/7/2017/a#snap_household_size
  - statute/7/2017/a#snap_region
  - statute/7/2017/a#snap_one_person_thrifty_food_plan_cost
  - statute/7/2017/a#snap_minimum_allotment
  - statute/7/2017/a#snap_maximum_allotment

amend snap_one_person_thrifty_food_plan_cost:
    from 2025-10-01:
        match snap_region:
            "CONTIGUOUS_US" => 298
            ...

amend snap_minimum_allotment:
    from 2025-10-01:
        if snap_household_size <= 0: 0
        elif snap_household_size == 1 or snap_household_size == 2:
            match snap_region:
                "CONTIGUOUS_US" => 24
                ...
        else: 0

amend snap_maximum_allotment:
    from 2025-10-01:
        match snap_region:
            "CONTIGUOUS_US" =>
                if snap_household_size == 1: 298
                elif snap_household_size == 2: 546
                ...
                else: 1789 + ((snap_household_size - 8) * 218)
```
- Wrong pattern for that same table:
```rac
imports:
  - statute/7/2017/a#snap_maximum_allotment

snap_household_size_four:
    from 2025-10-01: 4

snap_maximum_allotment:
    from 2025-10-01:
        ...
```
- For publication tables that update canonical statute outputs, prefer one `amend` per canonical output and direct comparisons like `snap_household_size == 4`; do not introduce helper constants solely for row labels.
- For resolved definition files listed above, the required syntax is an `imports:` block that references the exact `path#symbol` target.
- For copied canonical concept files listed above, the required syntax is an `imports:` block that references the exact `path#symbol` target.
- In any `imports:` block, emit bare import targets like `- regulation/9-CCR-2503-6/3.606.1/F#need_standard_for_assistance_unit`; do not wrap import targets in quotes.
- Do not replace a resolved canonical import with a local deferred symbol whose name is just a mangled version of the import target.
- If that cited upstream file is absent from this workspace, still emit the unresolved import path; the external-stub workflow is expected to fill it in later.
- If the source text only implies a shared concept, import an existing canonical concept only when one is actually present in the workspace; otherwise keep the helper local to this leaf.
- For isolated amount/rate leaves that cite same-instrument conditions or exceptions, do not fabricate sibling-file imports just because the text mentions another paragraph or schedule test. Model those cited conditions as local booleans or fact-shaped inputs unless the exact canonical import file is already present in this workspace.
- If no resolved definition file or copied precedent file shows you the import syntax, do not guess. Keep cited same-instrument conditions local instead of inventing `import` statements or `imports:` blocks.
- When the source states factual predicates that this leaf depends on, expose those predicates as plain fact-shaped inputs (`entity`, `period`, `dtype`) unless they are imported from a canonical definition.
- Do not encode such local factual predicates as placeholder constants like `true` or `false`.
- Do not encode such local factual predicates as `status: deferred`; if they are not imported, leave them as plain input stubs instead.
- When the source text says an amount is tested only after cited disregards, deductions, or other adjustments from an unavailable provision, preserve that post-adjustment quantity directly as an input/helper instead of silently switching to the raw pre-adjustment amount.
- For example, if the source says gross income must not exceed a need standard `after disregards have been applied`, prefer an input like `countable_gross_earned_income_after_disregards` over raw `gross_earned_income`, unless the cited disregard rule is actually present in the workspace and can be modeled.
- When the source text uses a month-day cutoff like `after the 15th day of a month`, keep that cutoff semantic in a fact-shaped input or comparison helper; do not decompose it into separate numeric `*_day`, `*_month`, or `*_year` scalar definitions.
- Do not add a documentary scalar like `*_cutoff_day: 15` just to restate that month-day cutoff. If the source only uses the date to define applicability, keep the cutoff inside the boolean/fact-shaped helper and omit the dead numeric constant entirely.
- If `./source.txt` states that a fixed supplement, allowance, or addition is payable only while an eligibility condition holds, do not leave that money output unconditional; make the amount depend on that eligibility condition, usually with `else: 0` when the source states no alternate amount.
- If `./source.txt` itself states the concrete facts that make someone eligible, do not collapse those facts into an opaque local input like `*_eligible_for_*` or `*_qualifies_for_*`. Expose the source-stated facts directly and derive eligibility from them only if needed.
- For example, if the source says `pregnant parents are eligible ... through the month in which the pregnancy ends`, prefer direct facts like `client_is_pregnant_parent` and a month-end boundary fact/helper over a black-box `person_is_eligible_for_pregnancy_allowance`.
- For textual instructions like `drop the cents`, `drop any cents`, or `truncate`, model truncation toward zero rather than toward negative infinity.
- For instructions like `rounded to the nearest whole dollar` or `nearest whole dollar increment`, do not rely on Python-style `round(...)` if the .5 behavior matters. Model explicit half-up rounding instead.
- A safe RAC pattern is `floor(amount + 0.5)` when the amount is non-negative; if negative values are possible, use a sign-aware half-up equivalent rather than banker’s rounding.
- If negative values are possible, use a sign-aware RAC expression such as `if amount >= 0: floor(amount) else: ceil(amount)` instead of bare `floor(amount)`.
- Reserve bare `floor(...)` for instructions that explicitly say `round down` or for complete-band/counting rules, and do not use unsupported operators such as `%`.
- When a rule drops cents or truncates and the computed amount may be negative, include a `.rac.test` case with a negative fractional amount so `floor(-1.25)` versus truncation-to-`-1` is actually exercised.
- When a copied precedent file supplies chart values, thresholds, or standard amounts that your artifact imports, do not guess contradictory `.rac.test` expectations for those imported values. Choose test rows that match explicit imported chart values, or assert only relationships that do not require guessing the imported amount.
- If you import variables like `need_standard_for_assistance_unit` or `grant_standard_for_assistance_unit` from a copied chart/standard file, keep `.rac.test` inputs and expected outputs consistent with the rows visible in that imported file rather than inventing zero or placeholder standards.
- If an imported chart file keys those values by household composition, pick `.rac.test` households from explicit chart rows that visibly exist in the copied file. Do not invent degenerate placeholder rows like `number_of_children_in_assistance_unit: 0` plus `number_of_caretakers_in_assistance_unit: 0` unless the imported chart explicitly defines that exact row for the imported symbol.
- Do not assert an exact zero imported standard, grant, or threshold unless that exact imported row is visible in the copied chart file. When the chart row is not visible, prefer relational assertions over guessed exact imported outputs.
- Do not use a `0 children / 0 caretakers` household as the primary threshold test for imported `need_standard_for_assistance_unit` or `grant_standard_for_assistance_unit`; instead choose a non-degenerate row that is visibly grounded in the copied chart.
- Wrong (`.rac.test` guesses a degenerate chart row):
  - name: zero_need_standard_exceeded
    input:
      number_of_children_in_assistance_unit: 0
      number_of_caretakers_in_assistance_unit: 0
    output:
      gross_income_is_within_need_standard_for_basic_cash_assistance: false
- Right (`.rac.test` uses a visible chart row like one child / no caretaker):
  - name: one_child_income_within_need_standard
    input:
      number_of_children_in_assistance_unit: 1
      number_of_caretakers_in_assistance_unit: 0
    output:
      gross_income_is_within_need_standard_for_basic_cash_assistance: true
- Wrong:
  some_paragraph_applies:
      entity: Person
      period: Day
      dtype: Boolean
      from 2025-03-21:
          false
- Right:
  some_paragraph_applies:
      entity: Person
      period: Day
      dtype: Boolean
- Wrong:
  current_day_is_first_day_of_next_benefit_week:
      entity: Person
      period: Day
      dtype: Boolean
      from 2025-03-21:
          false
- Right:
  current_day_is_first_day_of_next_benefit_week:
      entity: Person
      period: Day
      dtype: Boolean
- Do not invent schema keys like `namespace:`, `parameter`, `variable`, or `rule:`.
{schema_rules}{uk_guidance}{single_amount_row_guidance}{date_silent_scaffold_guidance}{target_hint_guidance}
- Prefer standard RAC blocks shaped like:
  example_name:
      entity: TaxUnit
      period: Month
      dtype: Money
      unit: USD
      from 2024-07-01:
          165
- For conditionals, RAC uses inline conditional expressions like `if condition: value else: other_value`.
- Use `==` for equality comparisons inside RAC expressions; never use bare `=` as an expression operator.
- Do not append a multiline conditional directly onto another expression like `base_amount + if condition: ...`; factor the conditional into its own helper variable or make the whole formula a single conditional expression.
- For derived values, keep using normal RAC blocks with `entity`, `period`, `dtype`, and `from YYYY-MM-DD:` formulas.
- For `dtype: Rate`, encode percentages as decimal ratios like `0.60` or `0.40`, never as `%` literals.
- Do not use Python inline ternaries like `x if cond else y`; use RAC conditional expressions instead.
- Do not use YAML-style `if:` / `then:` / `else:` blocks.
{file_output_rules}
- Do not respond with summaries like `Both files written`, `Done`, or bullet recaps in place of the requested files.
- For bundled file output, the `.rac` body must begin with RAC content, not prose.
- Do not use inline assignment syntax like `:=` inside `from` blocks or formulas.
- If a helper value is needed, declare it as its own top-level RAC variable block instead of assigning it inline.
"""


def _is_single_amount_table_slice(source_text: str) -> bool:
    """Return True when source text is a single amount-bearing table row slice."""
    marker = "Structured table:"
    if marker in source_text:
        table_section = source_text.split(marker, 1)[1]
        lines = [line.strip() for line in table_section.splitlines() if line.strip()]
        amount_rows = [line for line in lines if "|" in line and re.search(r"\d", line)]
        if len(amount_rows) == 1:
            return True

    money_matches = re.findall(r"[£$€]\s*\d[\d,]*(?:\.\d+)?", source_text)
    if len(money_matches) != 1:
        return False
    if _CONDITIONAL_AMOUNT_SLICE_PATTERN.search(source_text):
        return False

    money_lines = [
        line.strip()
        for line in source_text.splitlines()
        if re.search(r"[£$€]\s*\d[\d,]*(?:\.\d+)?", line)
    ]
    return len(money_lines) == 1


def _format_inline_context_snippets(
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
    max_chars_per_file: int = 6000,
) -> str:
    """Inline copied precedent files for non-tool backends like Responses API."""
    snippets: list[str] = []
    for item in context_files:
        path = workspace.root / item.workspace_path
        try:
            content = path.read_text().strip()
        except OSError:
            continue
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file].rstrip() + "\n... [truncated]"
        snippets.append(f"=== FILE: {item.workspace_path} ===\n{content}")
    return "\n\n".join(snippets)


def _format_context_file_listing(
    item: EvalContextFile,
    *,
    include_label: bool = False,
) -> str:
    """Format one copied context file for prompt display."""
    details = f": {item.label or item.source_path}" if include_label else ""
    if item.workspace_path == item.import_path:
        return f"- `{item.workspace_path}`{details}"
    return (
        f"- inspect `{item.workspace_path}`; import target `{item.import_path}`"
        f"{details}"
    )


def _collect_scaffold_dates(
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
) -> list[str]:
    """Collect usable temporal scaffold dates from copied precedent files."""
    dates: set[str] = set()
    for item in context_files:
        path = workspace.root / item.workspace_path
        try:
            text = path.read_text()
        except OSError:
            continue
        dates.update(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
    return sorted(dates)


def _expand_context_files(
    selected_paths: list[Path],
    rac_us_root: Path,
    target_rel: Path | None,
) -> list[tuple[Path, str]]:
    """Expand selected precedent files with their transitive canonical imports."""
    expanded: list[tuple[Path, str]] = []
    pending: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    for path in selected_paths:
        kind = (
            "implementation_precedent"
            if _is_under_root(path, rac_us_root)
            else "implementation_external"
        )
        pending.append((path, kind))

    while pending:
        source_path, kind = pending.pop(0)
        resolved = source_path.resolve()
        if resolved in seen:
            continue
        if target_rel is not None and _relative_to_root(source_path, rac_us_root) == target_rel:
            continue
        seen.add(resolved)
        expanded.append((source_path, kind))

        if not _is_under_root(source_path, rac_us_root):
            continue

        for dependency in _resolve_context_imports(source_path, rac_us_root):
            if dependency.resolve() in seen:
                continue
            pending.append((dependency, "implementation_dependency"))

    return expanded


def _resolve_context_imports(source_path: Path, rac_us_root: Path) -> list[Path]:
    """Resolve canonical import targets for one copied precedent file."""
    dependencies: list[Path] = []
    for import_target in _extract_import_targets(source_path.read_text()):
        target_path = _import_target_to_path(import_target)
        candidates = [rac_us_root / target_path]
        if target_path.parts:
            first = target_path.parts[0]
            if first == rac_us_root.name:
                candidates.append(rac_us_root / Path(*target_path.parts[1:]))
            if first in _LOCAL_IMPORT_ROOT_TOKENS:
                candidates.append(rac_us_root.parent / target_path)

        for candidate in candidates:
            if candidate.exists():
                dependencies.append(candidate)
                break
    return dependencies


def _extract_import_targets(content: str) -> list[str]:
    """Extract file-level import targets from RAC imports blocks."""
    targets: list[str] = []
    in_imports = False
    imports_indent = 0

    for line in content.splitlines():
        imports_match = re.match(r"^(\s*)imports:\s*$", line)
        if imports_match:
            in_imports = True
            imports_indent = len(imports_match.group(1))
            continue

        if not in_imports:
            continue

        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())
        if indent <= imports_indent:
            in_imports = False
            continue

        item_match = IMPORT_ITEM_PATTERN.match(line)
        if not item_match:
            continue

        item = item_match.group(2).strip()
        import_target = item.split("#", 1)[0].strip()
        if import_target:
            targets.append(import_target)

    return targets


def _import_target_to_path(import_target: str) -> Path:
    """Convert an import target like 26/24/c#name into 26/24/c.rac."""
    normalized = import_target.strip().strip('"').strip("'")
    if normalized.endswith(".rac"):
        return Path(normalized)
    return Path(f"{normalized}.rac")


def _is_under_root(path: Path, root: Path) -> bool:
    """Return True when a path is within the given root."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _relative_to_root(path: Path, root: Path) -> Path | None:
    """Return the path relative to the root when possible."""
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def _hydrate_eval_root(eval_root: Path, workspace: EvalWorkspace) -> None:
    """Copy allowed precedent files into the eval RAC root so imports resolve."""
    for item in workspace.context_files:
        workspace_path = Path(item.workspace_path)
        if not workspace_path.parts or workspace_path.parts[0] != "context":
            continue

        target = eval_root / _import_target_to_path(item.import_path)
        if target.exists():
            continue

        source = workspace.root / workspace_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _run_prompt_eval(
    runner: EvalRunnerSpec,
    workspace: EvalWorkspace,
    prompt: str,
) -> EvalPromptResponse:
    """Run one prompt-only eval through the selected local CLI."""
    if runner.backend == "claude":
        return _run_claude_prompt_eval(runner, workspace, prompt)
    if runner.backend == "codex":
        return _run_codex_prompt_eval(runner, workspace, prompt)
    if runner.backend == "openai":
        return _run_openai_prompt_eval(runner, workspace, prompt)
    raise ValueError(f"Unsupported backend: {runner.backend}")


def _run_claude_prompt_eval(
    runner: EvalRunnerSpec,
    workspace: EvalWorkspace,
    prompt: str,
) -> EvalPromptResponse:
    """Run prompt-only eval via Claude CLI."""
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        runner.model,
        "-p",
        prompt,
    ]

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=workspace.root,
        timeout=600,
    )
    duration_ms = int((time.time() - start) * 1000)

    trace: dict = {}
    text = result.stdout + result.stderr
    tokens = None
    actual_cost = None
    error = None

    try:
        payload = json.loads(text)
        trace = {
            "provider": "anthropic",
            "backend": "claude-print",
            "model": runner.model,
            "json_result": payload,
        }
        usage = payload.get("usage", {}) or {}
        tokens = TokenUsage(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            cache_creation_tokens=int(
                usage.get("cache_creation_input_tokens", 0) or 0
            ),
        )
        actual_cost = payload.get("total_cost_usd")
        text = payload.get("result", "") or ""
        if payload.get("is_error"):
            error = text or "Claude eval returned an error"
    except json.JSONDecodeError:
        trace = {
            "provider": "anthropic",
            "backend": "claude-print",
            "model": runner.model,
            "raw_output": result.stdout + result.stderr,
        }
        if result.returncode != 0:
            error = (result.stdout + result.stderr).strip() or "Claude eval failed"

    if result.returncode != 0 and not error:
        error = (result.stdout + result.stderr).strip() or "Claude eval failed"

    return EvalPromptResponse(
        text=text,
        duration_ms=duration_ms,
        tokens=tokens,
        estimated_cost_usd=estimate_usage_cost_usd(runner.model, tokens),
        actual_cost_usd=actual_cost,
        trace=trace,
        error=error,
    )


def _run_codex_prompt_eval(
    runner: EvalRunnerSpec,
    workspace: EvalWorkspace,
    prompt: str,
) -> EvalPromptResponse:
    """Run prompt-only eval via Codex CLI."""
    last_message_file = workspace.root / ".codex-last-message.txt"
    if last_message_file.exists():
        last_message_file.unlink()

    cmd = [
        "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "-o",
        str(last_message_file),
        "-m",
        runner.model,
        "-c",
        'reasoning_effort="low"',
        "-C",
        str(workspace.root),
        "-s",
        "read-only",
        prompt,
    ]

    start = time.time()
    terminated_after_output = False
    timed_out = False
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as stdout_file, tempfile.NamedTemporaryFile(
        mode="w+", delete=False
    ) as stderr_file:
        stdout_path = Path(stdout_file.name)
        stderr_path = Path(stderr_file.name)
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            cwd=workspace.root,
        )
        try:
            terminated_after_output = _wait_for_codex_process(
                process,
                last_message_file=last_message_file,
                timeout=600,
                heartbeat_paths=[stdout_path, stderr_path],
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()

    stdout_text = stdout_path.read_text()
    stderr_text = stderr_path.read_text()
    stdout_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)
    duration_ms = int((time.time() - start) * 1000)

    events: list[dict] = []
    assistant_messages: list[str] = []
    usage_payload: dict | None = None
    unexpected_accesses: list[str] = []
    error = None

    for line in (stdout_text + stderr_text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        events.append(payload)
        if payload.get("type") == "item.completed":
            item = payload.get("item", {}) or {}
            if item.get("type") == "agent_message" and item.get("text"):
                assistant_messages.append(item["text"])
            if item.get("type") == "command_execution":
                command = item.get("command", "")
                if _command_looks_out_of_bounds(command, workspace.root):
                    unexpected_accesses.append(command)
        elif payload.get("type") == "turn.completed":
            usage_payload = payload.get("usage") or {}
        elif payload.get("type") == "error":
            error = payload.get("message") or "Codex eval error"

    tokens = None
    if usage_payload is not None:
        tokens = TokenUsage(
            input_tokens=int(usage_payload.get("input_tokens", 0) or 0),
            output_tokens=int(usage_payload.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage_payload.get("cached_input_tokens", 0) or 0),
            reasoning_output_tokens=extract_reasoning_output_tokens(
                {
                    "provider": "openai",
                    "events": events,
                }
            ),
        )

    final_text = "\n".join(assistant_messages).strip()
    if last_message_file.exists():
        file_text = last_message_file.read_text().strip()
        if file_text:
            final_text = file_text

    if timed_out and not error and not final_text:
        error = "Codex eval timed out"

    if process.returncode != 0 and not error and not (
        (terminated_after_output and final_text) or (timed_out and final_text)
    ):
        error = (stdout_text + stderr_text).strip() or "Codex eval failed"

    return EvalPromptResponse(
        text=final_text,
        duration_ms=duration_ms,
        tokens=tokens,
        estimated_cost_usd=estimate_usage_cost_usd(runner.model, tokens),
        trace={
            "provider": "openai",
            "backend": "codex-exec",
            "model": runner.model,
            "events": events,
        },
        unexpected_accesses=unexpected_accesses,
        error=error,
    )


def _wait_for_codex_process(
    process: subprocess.Popen[str],
    last_message_file: Path,
    timeout: int,
    *,
    heartbeat_paths: Sequence[Path] | None = None,
    settle_seconds: float = 5.0,
    max_output_wait_seconds: float = 30.0,
    max_idle_seconds: float = 120.0,
    poll_interval: float = 0.5,
) -> bool:
    """Wait for Codex CLI, terminating it once output is stable or persistent."""
    start = time.time()
    last_snapshot: tuple[int, int] | None = None
    stable_since: float | None = None
    output_seen_at: float | None = None
    last_activity_at = start
    heartbeat_snapshot: tuple[tuple[int, int, int], ...] | None = None

    def _snapshot_activity() -> tuple[tuple[int, int, int], ...]:
        files = [last_message_file, *(heartbeat_paths or [])]
        snapshot: list[tuple[int, int, int]] = []
        for path in files:
            if not path.exists():
                snapshot.append((0, 0, 0))
                continue
            try:
                stat = path.stat()
            except OSError:
                snapshot.append((0, 0, 0))
                continue
            snapshot.append((1, stat.st_size, stat.st_mtime_ns))
        return tuple(snapshot)

    while True:
        if process.poll() is not None:
            return False

        now = time.time()
        if now - start > timeout:
            raise subprocess.TimeoutExpired(process.args, timeout)

        current_heartbeat_snapshot = _snapshot_activity()
        if current_heartbeat_snapshot != heartbeat_snapshot:
            heartbeat_snapshot = current_heartbeat_snapshot
            last_activity_at = now
        elif now - last_activity_at >= max_idle_seconds:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise subprocess.TimeoutExpired(process.args, max_idle_seconds)

        if last_message_file.exists():
            try:
                text = last_message_file.read_text().strip()
                stat = last_message_file.stat()
            except OSError:
                text = ""
                stat = None

            if text and stat is not None:
                output_seen_at = output_seen_at or now
                snapshot = (stat.st_size, stat.st_mtime_ns)
                if snapshot == last_snapshot:
                    stable_since = stable_since or now
                    if now - stable_since >= settle_seconds:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        return True
                else:
                    last_snapshot = snapshot
                    stable_since = None
                    if (
                        output_seen_at is not None
                        and now - output_seen_at >= max_output_wait_seconds
                    ):
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        return True

        time.sleep(poll_interval)


def _extract_openai_response_text(payload: dict) -> str:
    """Flatten a Responses API payload into assistant text."""
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            continue
        if item.get("type") == "message":
            for content_item in item.get("content", []) or []:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in {"output_text", "text"}:
                    text = content_item.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    return "\n\n".join(texts).strip()


def _run_openai_prompt_eval(
    runner: EvalRunnerSpec,
    workspace: EvalWorkspace,
    prompt: str,
) -> EvalPromptResponse:
    """Run prompt-only eval via the OpenAI Responses API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return EvalPromptResponse(
            text="",
            duration_ms=0,
            trace={
                "provider": "openai",
                "backend": "responses",
                "model": runner.model,
            },
            error="OPENAI_API_KEY is not set",
        )

    body = {
        "model": runner.model,
        "input": prompt,
        "max_output_tokens": 16384,
        "reasoning": {
            "effort": "low",
            "summary": "auto",
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start = time.time()
    try:
        response = _post_openai_eval_request(headers=headers, body=body)
    except requests.RequestException as exc:
        duration_ms = int((time.time() - start) * 1000)
        return EvalPromptResponse(
            text="",
            duration_ms=duration_ms,
            trace={
                "provider": "openai",
                "backend": "responses",
                "model": runner.model,
                "request_body": body,
            },
            error=str(exc),
        )
    duration_ms = int((time.time() - start) * 1000)

    request_id = response.headers.get("x-request-id")
    try:
        payload = response.json()
    except ValueError:
        payload = {
            "error": {
                "message": response.text or f"HTTP {response.status_code}",
            }
        }

    trace = {
        "provider": "openai",
        "backend": "responses",
        "model": runner.model,
        "request_id": request_id,
        "request_body": body,
        "json_result": payload,
        "status_code": response.status_code,
    }

    if response.status_code >= 400:
        error = payload.get("error") or {}
        return EvalPromptResponse(
            text="",
            duration_ms=duration_ms,
            trace=trace,
            error=error.get("message") or response.text or "OpenAI eval failed",
        )

    usage = payload.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    tokens = TokenUsage(
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cache_read_tokens=int(input_details.get("cached_tokens", 0) or 0),
    )
    tokens.reasoning_output_tokens = int(
        ((usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
    )

    return EvalPromptResponse(
        text=_extract_openai_response_text(payload),
        duration_ms=duration_ms,
        tokens=tokens,
        estimated_cost_usd=estimate_usage_cost_usd(runner.model, tokens),
        trace=trace,
    )


def _post_openai_eval_request(
    headers: dict[str, str],
    body: dict[str, object],
    attempts: int = 6,
) -> requests.Response:
    """POST a Responses API eval request with transient retry handling."""
    last_response: requests.Response | None = None
    last_error: requests.RequestException | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=body,
                timeout=(30, 180),
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 10))
            continue

        last_response = response
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == attempts:
            return response
        time.sleep(min(2 ** (attempt - 1), 10))

    if last_response is not None:
        return last_response
    if last_error is not None:
        raise last_error
    raise requests.RequestException("OpenAI eval request failed without response")


def _command_looks_out_of_bounds(command: str, workspace_root: Path) -> bool:
    """Heuristic for whether a Codex shell command accessed paths outside workspace."""
    if not command:
        return False

    if re.search(r"(^|[\s'\"`])\.\.(?:/|[\s'\"`]|$)", command):
        return True

    abs_paths = re.findall(r"(?:(?<=^)|(?<=[\s'\"`]))(/[^\s'\"`]+)", command)
    for raw_path in abs_paths:
        path = Path(raw_path)
        try:
            path.resolve().relative_to(workspace_root.resolve())
        except ValueError:
            return True
        except FileNotFoundError:
            return True

    return False


def _extract_rac_content(llm_response: str) -> str | None:
    """Extract raw RAC content from a model response."""
    if not llm_response or not llm_response.strip():
        return None

    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", llm_response)

    fence_match = re.search(r"```(?:yaml|rac|text)?\s*\n(.*?)\n```", cleaned, re.DOTALL)
    if fence_match:
        content = fence_match.group(1).strip()
        if content:
            return content + "\n"

    stripped = cleaned.strip()
    if stripped.startswith("#") or stripped.startswith('"""'):
        return stripped + ("\n" if not stripped.endswith("\n") else "")

    lines = stripped.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("#") or line.startswith('"""') or line.startswith("status:"):
            return "\n".join(lines[i:]).strip() + "\n"

    rac_keywords = ("status:", "entity:", "imports:", "period:", "dtype:")
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if any(stripped_line.startswith(kw) for kw in rac_keywords):
            return "\n".join(lines[i:]).strip() + "\n"

    return None


def _extract_generated_file_bundle(llm_response: str) -> dict[str, str]:
    """Extract a small multi-file bundle emitted by eval backends."""
    if not llm_response or "=== FILE:" not in llm_response:
        return {}

    pattern = re.compile(r"^=== FILE:\s*(?P<name>.+?)\s*===\s*$", re.MULTILINE)
    matches = list(pattern.finditer(llm_response))
    if not matches:
        return {}

    files: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(
            llm_response
        )
        content = llm_response[start:end].strip()
        if content:
            files[match.group("name").strip()] = _clean_generated_file_content(content)
    return files


def _clean_generated_file_content(content: str) -> str:
    """Strip common wrapper noise from bundled file content."""
    stripped = content.strip()
    fenced = re.match(
        r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n```(?:\s|$)",
        stripped,
        re.DOTALL,
    )
    if fenced:
        stripped = fenced.group(1).strip()
    stripped = re.sub(
        r"(?<![\d.])(-?\d+(?:,\d{3})*(?:\.\d+)?)\s+(GBP|USD|EUR)\b",
        lambda match: match.group(1),
        stripped,
    )
    return stripped + ("\n" if stripped and not stripped.endswith("\n") else "")


def _normalize_comma_numeric_literals(content: str) -> str:
    """Strip thousands separators from numeric literals without touching prose."""
    return re.sub(
        r"(?<![\d.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d.])",
        lambda match: match.group(0).replace(",", ""),
        content,
    )


def _normalize_rac_code_numeric_literals(content: str) -> str:
    """Strip thousands separators from executable RAC code, preserving docstring prose."""
    content = _normalize_source_text_wrapper_rac_content(content)
    if content.startswith('"""'):
        closing_index = content.find('"""', 3)
        if closing_index != -1:
            code_start = closing_index + 3
            return (
                content[:code_start]
                + _normalize_direct_scalar_numeric_expressions(
                    _normalize_comma_numeric_literals(content[code_start:])
                )
            )
    return _normalize_direct_scalar_numeric_expressions(
        _normalize_comma_numeric_literals(content)
    )


def _normalize_direct_scalar_numeric_expressions(content: str) -> str:
    """Collapse direct scalar arithmetic like `1 / 10` into decimal literals."""
    lines = content.splitlines()
    rewritten: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        inline_match = re.match(r"^(\s*from\s+\d{4}-\d{2}-\d{2}:\s*)(.+?)\s*$", line)
        if inline_match:
            expression = inline_match.group(2).strip()
            if _PURE_NUMERIC_EXPRESSION_PATTERN.fullmatch(expression):
                if (formatted := _format_safe_numeric_expression(expression)) is not None:
                    rewritten.append(f"{inline_match.group(1)}{formatted}")
                    index += 1
                    continue

        block_match = re.match(r"^(\s*from\s+\d{4}-\d{2}-\d{2}:)\s*$", line)
        if block_match and index + 1 < len(lines):
            next_line = lines[index + 1]
            expression = next_line.strip()
            if (
                expression
                and _PURE_NUMERIC_EXPRESSION_PATTERN.fullmatch(expression)
                and (
                    len(next_line) - len(next_line.lstrip())
                    > len(line) - len(line.lstrip())
                )
            ):
                if (formatted := _format_safe_numeric_expression(expression)) is not None:
                    rewritten.append(f"{block_match.group(1)} {formatted}")
                    index += 2
                    continue

        rewritten.append(line)
        index += 1

    return "\n".join(rewritten) + ("\n" if content.endswith("\n") else "")


_SOURCE_TEXT_WRAPPER_PATTERN = re.compile(
    r"^source_text:\s*\n"
    r"(?:^[ \t]+(?:entity|period|dtype):.*\n)+"
    r"^[ \t]+from\s+\d{4}-\d{2}-\d{2}:\s*\n"
    r"^[ \t]+\"\"\"\n"
    r"(?P<doc>.*?)"
    r"^[ \t]+\"\"\"\s*\n?",
    re.MULTILINE | re.DOTALL,
)


def _normalize_source_text_wrapper_rac_content(content: str) -> str:
    """Rewrite mistaken `source_text` string wrappers into leading docstrings."""
    match = _SOURCE_TEXT_WRAPPER_PATTERN.match(content)
    if not match:
        return content

    doc_lines = [line[8:] if line.startswith("        ") else line for line in match.group("doc").splitlines()]
    docstring = '"""\n' + "\n".join(doc_lines).strip("\n") + '\n"""\n'
    remainder = content[match.end() :].lstrip("\n")
    if remainder:
        return docstring + "\n" + remainder
    return docstring


def _normalize_single_amount_row_rac_content(content: str) -> str:
    """Collapse one-row conditional encodings into grounded constants."""
    normalized = _normalize_rac_code_numeric_literals(content)
    normalized = re.sub(
        r"(^\s*from\s+\d{4}-\d{2}-\d{2}:\s*)if\b.+?:\s*(.+?)\s*(?:else:\s*0(?:\.0+)?)?\s*$",
        lambda match: (
            f"{match.group(1)}{formatted}"
            if (formatted := _format_safe_numeric_expression(match.group(2))) is not None
            else match.group(0)
        ),
        normalized,
        flags=re.MULTILINE,
    )

    lines = normalized.splitlines()
    rewritten: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        from_match = re.match(r"^(\s*from\s+\d{4}-\d{2}-\d{2}:)\s*$", line)
        if (
            from_match
            and index + 1 < len(lines)
            and (
                cond_match := re.match(
                    r"^\s*(?:if\b.+?:\s*)?(.+?)\s*(?:else:\s*0(?:\.0+)?)?\s*$",
                    lines[index + 1],
                )
            )
        ):
            if (
                formatted := _format_safe_numeric_expression(cond_match.group(1))
            ) is not None:
                rewritten.append(f"{from_match.group(1)} {formatted}")
                index += 2
                continue
        rewritten.append(line)
        index += 1
    return "\n".join(rewritten) + ("\n" if normalized.endswith("\n") else "")


def _format_safe_numeric_expression(expression: str) -> str | None:
    """Evaluate a numeric literal or simple arithmetic expression safely."""
    try:
        value = _evaluate_safe_numeric_expression(expression.strip())
    except (ValueError, SyntaxError, ZeroDivisionError):
        return None

    if float(value).is_integer():
        return str(int(value))
    return format(value, "g")


def _evaluate_safe_numeric_expression(expression: str) -> float:
    """Return the numeric value of a simple arithmetic expression."""
    node = ast.parse(expression, mode="eval")

    def visit(current: ast.AST) -> float:
        if isinstance(current, ast.Expression):
            return visit(current.body)
        if isinstance(current, ast.Constant) and isinstance(current.value, (int, float)):
            return float(current.value)
        if isinstance(current, ast.UnaryOp) and isinstance(
            current.op, (ast.UAdd, ast.USub)
        ):
            operand = visit(current.operand)
            return operand if isinstance(current.op, ast.UAdd) else -operand
        if isinstance(current, ast.BinOp) and isinstance(
            current.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = visit(current.left)
            right = visit(current.right)
            if isinstance(current.op, ast.Add):
                return left + right
            if isinstance(current.op, ast.Sub):
                return left - right
            if isinstance(current.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError(f"Unsupported numeric expression: {expression}")

    return visit(node)


def _normalize_single_amount_row_test_content(
    content: str,
    rac_content: str | None = None,
    source_text: str | None = None,
) -> str:
    """Drop alternate-branch tests for one-row fixed-amount slices."""
    normalized = _normalize_comma_numeric_literals(content)
    try:
        payload = yaml.safe_load(normalized)
    except yaml.YAMLError:
        return normalized

    if payload is None:
        return normalized

    annual_period = bool(
        rac_content and re.search(r"^\s*period:\s*Year\s*$", rac_content, flags=re.MULTILINE)
    )
    effective_date = _extract_effective_date_for_tests(
        rac_content=rac_content,
        source_text=source_text,
    )

    def should_keep(case_name: str | None) -> bool:
        if not case_name:
            return True
        lowered = case_name.lower()
        if "alternate" in lowered:
            return False
        if annual_period and "effective_date_boundary" in lowered:
            return False
        return True

    def normalize_case(case: object) -> object:
        if not isinstance(case, dict):
            return case
        normalized_case = dict(case)
        if annual_period and effective_date is not None:
            normalized_case["period"] = _normalize_annual_test_period_value(
                normalized_case.get("period"),
                effective_date,
            )
        output = normalized_case.get("output")
        if isinstance(output, dict) and len(output) > 1:
            numeric_output = {
                key: value
                for key, value in output.items()
                if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))
            }
            if numeric_output:
                normalized_case["output"] = numeric_output
        return normalized_case

    cases = _coerce_test_payload_to_case_list(payload)
    if cases is not None:
        filtered = [
            normalize_case(case)
            for case in cases
            if not isinstance(case, dict) or should_keep(case.get("name"))
        ]
        return yaml.safe_dump(filtered, sort_keys=False).strip() + "\n"

    return normalized


def _extract_effective_date_for_tests(
    rac_content: str | None,
    source_text: str | None,
) -> date | None:
    """Return the earliest explicit effective date available for test normalization."""
    if rac_content and (
        from_match := re.search(r"\bfrom\s+(\d{4}-\d{2}-\d{2}):", rac_content)
    ):
        return date.fromisoformat(from_match.group(1))
    if source_text and (
        source_match := re.search(
            r"\b(?:text|current text)\s+valid\s+from\s+(\d{4}-\d{2}-\d{2})\b",
            source_text,
            flags=re.IGNORECASE,
        )
    ):
        return date.fromisoformat(source_match.group(1))
    return None


def _normalize_annual_test_period_value(
    period: object,
    effective_date: date,
) -> object:
    """Convert bare annual periods into concrete dates on or after the effective date."""
    year: int | None = None
    if period is None:
        year = effective_date.year
    elif isinstance(period, int):
        year = period
    elif isinstance(period, str) and re.fullmatch(r"\d{4}", period):
        year = int(period)

    if year is None:
        return period
    if year < effective_date.year:
        return period
    if year == effective_date.year:
        return effective_date.isoformat()
    return f"{year}-01-01"


def _extract_rac_period_granularity(rac_content: str | None) -> str | None:
    """Return the first declared RAC period name."""
    if rac_content is None:
        return None
    match = re.search(
        r"^\s*period:\s*(Year|Month|Week|Day)\s*$",
        rac_content,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def _default_test_period_for_granularity(
    granularity: str | None,
    effective_date: date,
) -> str:
    """Return a concrete test period compatible with the RAC runner."""
    return effective_date.isoformat()


def _normalize_nonannual_test_period_value(
    period: object,
    effective_date: date,
) -> object:
    """Convert non-annual periods to concrete dates on or after the effective date."""
    if period is None:
        return effective_date.isoformat()
    if isinstance(period, date):
        if period < effective_date:
            return effective_date.isoformat()
        return period.isoformat()
    if isinstance(period, int):
        if period == effective_date.year:
            return effective_date.isoformat()
        if period > effective_date.year:
            return f"{period}-01-01"
        return period
    if isinstance(period, str):
        if re.fullmatch(r"\d{4}", period):
            year = int(period)
            if year == effective_date.year:
                return effective_date.isoformat()
            if year > effective_date.year:
                return f"{year}-01-01"
            return period
        if _ISO_WEEK_PERIOD_PATTERN.fullmatch(period):
            week_year = int(period[:4])
            if week_year == effective_date.year:
                return effective_date.isoformat()
            if week_year > effective_date.year:
                return f"{week_year}-01-01"
            return period
        if re.fullmatch(r"\d{4}-\d{2}", period):
            if period == effective_date.strftime("%Y-%m"):
                return effective_date.isoformat()
            return period
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", period):
            try:
                parsed = date.fromisoformat(period)
            except ValueError:
                return period
            if parsed < effective_date:
                return effective_date.isoformat()
            return period
    return period


def _coerce_test_payload_to_case_list(payload: object) -> list[object] | None:
    """Return test payloads as a plain list of case objects when recognizable."""
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return None

    tests_payload = payload.get("tests")
    if isinstance(tests_payload, list):
        return tests_payload

    case_like_keys = {"name", "period", "input", "inputs", "output", "expect"}
    if case_like_keys & set(payload):
        return [payload]

    if not payload or not all(isinstance(value, dict) for value in payload.values()):
        return None

    cases: list[object] = []
    for key, value in payload.items():
        case = dict(value)
        if isinstance(key, str) and "name" not in case:
            case["name"] = key
        cases.append(case)
    return cases


def _normalize_test_case_value(value: object) -> object:
    """Collapse entity/time wrappers in generated .rac.test values to plain scalars."""
    if isinstance(value, list):
        return [_normalize_test_case_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = {
        key: _normalize_test_case_value(inner) for key, inner in value.items()
    }
    lowered_keys = {str(key).lower() for key in normalized.keys()}
    metadata_keys = {"entity", "period", "dtype", "unit", "label", "description", "default"}

    values_entries = [
        inner for key, inner in normalized.items() if str(key).lower() == "values"
    ]
    if len(values_entries) == 1 and lowered_keys.issubset(metadata_keys | {"values"}):
        values_entry = values_entries[0]
        if isinstance(values_entry, dict) and len(values_entry) == 1:
            return _normalize_test_case_value(next(iter(values_entry.values())))
        return _normalize_test_case_value(values_entry)

    from_entries = [
        inner
        for key, inner in normalized.items()
        if str(key).lower().startswith("from ")
    ]
    if from_entries and lowered_keys.issubset(metadata_keys | {str(key).lower() for key in normalized.keys() if str(key).lower().startswith("from ")}):
        if len(from_entries) == 1:
            return _normalize_test_case_value(from_entries[0])

    if len(normalized) == 1:
        only_key, only_value = next(iter(normalized.items()))
        if re.fullmatch(r"\d{4}(?:-\d{2})?(?:-\d{2})?", str(only_key)):
            return _normalize_test_case_value(only_value)
        if not isinstance(only_value, (dict, list)):
            return only_value

    return normalized


def _normalize_test_periods_to_effective_dates(
    content: str,
    rac_content: str | None = None,
    source_text: str | None = None,
) -> str:
    """Normalize annual test periods to concrete dates that survive effective-date compilation."""
    normalized = _normalize_comma_numeric_literals(content)
    granularity = _extract_rac_period_granularity(rac_content)
    effective_date = _extract_effective_date_for_tests(
        rac_content=rac_content,
        source_text=source_text,
    )
    if effective_date is None and granularity != "Year":
        return normalized

    try:
        payload = yaml.safe_load(normalized)
    except yaml.YAMLError:
        return normalized

    if payload is None:
        return normalized

    def normalize_case(case: object) -> object:
        if not isinstance(case, dict):
            return case
        normalized_case = dict(case)
        if granularity == "Year" and effective_date is not None:
            normalized_case["period"] = _normalize_annual_test_period_value(
                normalized_case.get("period"),
                effective_date,
            )
        elif effective_date is not None:
            normalized_case["period"] = _normalize_nonannual_test_period_value(
                normalized_case.get("period"),
                effective_date,
            )

        for key in ("input", "inputs", "output"):
            if key in normalized_case and isinstance(normalized_case[key], dict):
                normalized_case[key] = {
                    child_key: _normalize_test_case_value(child_value)
                    for child_key, child_value in normalized_case[key].items()
                }
        if "expect" in normalized_case:
            normalized_case["expect"] = _normalize_test_case_value(
                normalized_case["expect"]
            )
        return normalized_case

    cases = _coerce_test_payload_to_case_list(payload)
    if cases is not None:
        return yaml.safe_dump(
            [normalize_case(case) for case in cases],
            sort_keys=False,
        ).strip() + "\n"

    return normalized


def _materialize_eval_artifact(
    llm_response: str,
    expected_path: Path,
    source_text: str | None = None,
    workspace_root: Path | None = None,
) -> bool:
    """Write an eval artifact and optional companion test file from model output."""
    single_amount_table_slice = bool(
        source_text and _is_single_amount_table_slice(source_text)
    )
    expected_test_path = expected_path.with_suffix(".rac.test")

    if workspace_root is not None:
        wrote_from_workspace = _materialize_workspace_artifacts(
            expected_path=expected_path,
            expected_test_path=expected_test_path,
            workspace_root=workspace_root,
            single_amount_table_slice=single_amount_table_slice,
            source_text=source_text,
        )
        if wrote_from_workspace:
            return True

    bundle = _extract_generated_file_bundle(llm_response)
    if bundle:
        wrote_main = False
        bundle_by_candidate_name = {
            Path(file_name).name: content for file_name, content in bundle.items()
        }
        for file_name, content in bundle.items():
            candidate_name = Path(file_name).name
            if candidate_name == expected_path.name:
                target_path = expected_path
            elif candidate_name == expected_test_path.name:
                target_path = expected_test_path
            else:
                continue
            if single_amount_table_slice:
                if target_path == expected_path:
                    content = _normalize_single_amount_row_rac_content(content)
                elif target_path == expected_test_path:
                    content = _normalize_single_amount_row_test_content(
                        content,
                        rac_content=bundle_by_candidate_name.get(expected_path.name),
                        source_text=source_text,
                    )
            elif target_path == expected_path:
                content = _normalize_rac_code_numeric_literals(content)
            elif target_path == expected_test_path:
                content = _normalize_test_periods_to_effective_dates(
                    content,
                    rac_content=bundle_by_candidate_name.get(expected_path.name),
                    source_text=source_text,
                )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content)
            if target_path == expected_path:
                wrote_main = True
        if wrote_main or expected_path.exists():
            return True

    rac_content = _extract_rac_content(llm_response)
    if not rac_content:
        return False
    if single_amount_table_slice:
        rac_content = _normalize_single_amount_row_rac_content(rac_content)
    else:
        rac_content = _normalize_rac_code_numeric_literals(rac_content)

    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_text(rac_content)
    return True


def _materialize_workspace_artifacts(
    expected_path: Path,
    expected_test_path: Path,
    workspace_root: Path,
    single_amount_table_slice: bool,
    source_text: str | None,
) -> bool:
    """Salvage eval artifacts that a model wrote directly into the workspace."""
    workspace_main = workspace_root / expected_path.name
    workspace_test = workspace_root / expected_test_path.name
    if not workspace_main.exists():
        return False

    main_content = workspace_main.read_text()
    if single_amount_table_slice:
        main_content = _normalize_single_amount_row_rac_content(main_content)
    else:
        main_content = _normalize_rac_code_numeric_literals(main_content)

    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_text(main_content)

    if workspace_test.exists():
        test_content = workspace_test.read_text()
        if single_amount_table_slice:
            test_content = _normalize_single_amount_row_test_content(
                test_content,
                rac_content=main_content,
                source_text=source_text,
            )
        else:
            test_content = _normalize_test_periods_to_effective_dates(
                test_content,
                rac_content=main_content,
                source_text=source_text,
            )
        expected_test_path.write_text(test_content)

    return True


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "eval"
