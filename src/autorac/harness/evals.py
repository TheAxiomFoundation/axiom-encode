"""Model comparison evals for statute and source-slice encoding."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Literal
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

from .encoding_db import TokenUsage
from .observability import emit_eval_result, extract_reasoning_output_tokens
from .pricing import estimate_usage_cost_usd
from .validator_pipeline import (
    ResolvedDefinedTerm,
    ValidationResult,
    ValidatorPipeline,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_named_scalar_occurrences,
    extract_numbers_from_text,
    extract_numeric_occurrences_from_text,
    resolve_defined_terms_from_text,
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
            fetch_cache_root=output_root,
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
    policyengine_rac_var_hint: str | None = None,
    fetch_cache_root: Path | None = None,
) -> list[EvalResult]:
    """Fetch official UK legislation XML and run an AKN section eval."""
    fetched = _fetch_legislation_gov_uk_document(
        source_ref,
        output_root,
        fetch_cache_root=fetch_cache_root,
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
        oracle="policyengine",
        policyengine_country="uk",
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
) -> list[EvalResult]:
    """Run every case in a benchmark suite manifest."""
    resolved_runners = runner_specs or manifest.runners
    parsed_runners = [parse_runner_spec(spec) for spec in resolved_runners]
    results: list[EvalResult] = []
    _validate_uk_shared_scalar_sibling_sets(manifest, Path(output_root))

    for index, case in enumerate(manifest.cases, start=1):
        case_output_root = Path(output_root) / f"{index:02d}-{_slugify(case.name)}"
        extra_context = [*manifest.allow_context, *case.allow_context]
        attempts = max(suite_retry_attempts, 0) + 1
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
                        policyengine_rac_var_hint=case.policyengine_rac_var_hint,
                        fetch_cache_root=Path(output_root),
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

    return results


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
    return any(result.error is not None or result.metrics is None for result in case_results)


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
    for resolved_term in resolve_defined_terms_from_text(source_text):
        context_files.append(
            _materialize_resolved_definition_stub(
                context_root=context_root,
                resolved_term=resolved_term,
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

        target_rel = _target_rel_for_eval_identifier(citation)
        expanded_context = _expand_context_files(selected, rac_us_root, target_rel)

        for source_path, kind in expanded_context:
            try:
                if source_path.is_relative_to(rac_us_root):
                    relative_target = source_path.relative_to(rac_us_root)
                else:
                    relative_target = Path("external") / source_path.name
            except ValueError:
                relative_target = Path("external") / source_path.name

            workspace_path = context_root / relative_target
            workspace_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, workspace_path)
            context_files.append(
                EvalContextFile(
                    source_path=str(source_path),
                    workspace_path=str(workspace_path.relative_to(workspace_root)),
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


def _import_target_to_relative_rac_path(import_target: str) -> Path:
    """Convert an import target like legislation/...#name into a .rac path."""
    normalized = import_target.strip().strip('"').strip("'")
    normalized = normalized.split("#", 1)[0]
    if normalized.endswith(".rac"):
        return Path(normalized)
    return Path(f"{normalized}.rac")


def _build_resolved_definition_stub_content(resolved_term: ResolvedDefinedTerm) -> str:
    """Return a compile-friendly stub file for one resolved legal term."""
    return (
        f'"""\nCanonical definition stub for `{resolved_term.term}`.\n'
        f"Resolved to {resolved_term.citation}.\n"
        '"""\n\n'
        "status: stub\n\n"
        f"{resolved_term.symbol}:\n"
        f"    stub_for: {resolved_term.import_target}\n"
        f"    entity: {resolved_term.entity}\n"
        f"    period: {resolved_term.period}\n"
        f"    dtype: {resolved_term.dtype}\n"
    )


def _materialize_resolved_definition_stub(
    *,
    context_root: Path,
    resolved_term: ResolvedDefinedTerm,
    workspace_root: Path,
) -> EvalContextFile:
    """Write one resolved definition stub into the eval workspace context."""
    relative_target = Path("context") / _import_target_to_relative_rac_path(
        resolved_term.import_target
    )
    workspace_path = workspace_root / relative_target
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path.write_text(_build_resolved_definition_stub_content(resolved_term))
    return EvalContextFile(
        source_path=resolved_term.citation,
        workspace_path=str(relative_target),
        kind="definition_stub",
        label=resolved_term.label,
    )


def _auto_select_context_files(citation: str, rac_us_root: Path) -> list[Path]:
    """Best-effort auto-context selection for statute citations only."""
    try:
        return select_context_files(citation, rac_us_root)
    except Exception:
        return []


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
    precedent_context_files = [
        item for item in context_files if item.kind != "definition_stub"
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
            f"- `{item.workspace_path}`: {item.label or item.source_path}"
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

    context_section = ""
    if precedent_context_files:
        listed = "\n".join(
            f"- `{item.workspace_path}`" for item in precedent_context_files
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
- The `.rac.test` file must contain 1-4 cases.
- For a single fixed-amount source slice, a base case is sufficient.
- Add an effective-date boundary only when the period supports a meaningful point-in-time boundary.
- Add an alternate branch only when `./source.txt` states another grounded branch condition or amount.
- Test inputs must contain factual predicates or quantities, not the output variable being asserted.
- Use `output:` mappings in `.rac.test` cases, not `expect:` blocks.
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
        uk_guidance = """
- For UK legislation, do not invent custom provision-level entities like `Provision`, `Section`, or `Regulation`, and do not invent periods like `Instant`.
- Prefer `Person` when the source states an amount or condition "in respect of" a child, qualifying young person, or other individual.
- Use `Family` only when the encoded quantity is explicitly aggregate at claimant or benefit-unit level.
- For UK rate leaves with one grounded monetary amount, encode the directly payable person-level or unit-level amount described by the text; do not collapse it into an unconditional family-level constant.
- For UK branch leaves like `(a)`, `(b)`, or `80A(2)(c)`, encode the branch identity in the output variable name. Do not reuse generic parent variable names like `child_benefit_weekly_rate`, `standard_minimum_guarantee`, or `benefit_cap` for a branch-specific leaf.
- In `.rac.test`, use helper/input names that expose the actual legal facts from the source text. Prefer names like `child_benefit_is_only_person`, `child_benefit_is_elder_or_eldest_person`, `claimant_has_partner`, `is_single_claimant`, `is_joint_claimant`, `resident_in_greater_london`, or `responsible_for_child_or_qualifying_young_person`.
- In `.rac.test`, avoid opaque placeholders like `*_condition`, `*_eligibility_flag`, or `family_has_partner` when a more direct legal-fact name is available from the source text.
- In `.rac.test`, choose periods on or after the explicit effective date in `./source.txt`.
- Do not add speculative future-period tests that would rely on uprating, later amendments, or rates not stated in `./source.txt`.
"""
    single_amount_row_guidance = ""
    if single_amount_table_slice:
        single_amount_row_guidance = """
- `./source.txt` is already a single table row or atomic branch with one grounded amount. Encode that branch-specific amount directly.
- Use a descriptive legal variable name, not a path- or source-id-derived placeholder like `uksi_2013_...`.
- For a one-row fixed-amount slice, do not invent a fresh `*_applies` helper or unrelated eligibility booleans unless the source text itself states them.
- For a one-row fixed-amount slice, do not invent alternate zero-amount tests.
- For a one-row fixed-amount slice, Do not emit `otherwise:`.
- For a one-row fixed-amount slice, Do not emit `before YYYY-MM-DD: 0`.
- For a one-row fixed-amount slice, Do not emit stray blocks like `from 0:`.
- For a one-row fixed-amount slice, use exactly one grounded `from YYYY-MM-DD:` clause unless `./source.txt` itself states multiple grounded dates or amounts.
- For a one-row fixed-amount slice, the principal amount variable should usually be a grounded constant under `from YYYY-MM-DD:`; do not wrap it in a conditional formula unless `./source.txt` itself states a second grounded amount or branch inside the same row.
- For a one-row fixed-amount slice, do not disguise the grounded amount as arithmetic like `2025 * 11 - 255`; emit the grounded constant directly.
- In `.rac.test` for a one-row fixed-amount slice, use boolean or fact-shaped helper inputs that mirror the row text.
- Do not invent sample ages like `2`, `3`, `24`, or `25` just to witness a row condition; if the row says "aged under 25", prefer a helper like `claimant_aged_under_25`.
- For a one-row fixed-amount slice with a single canonical subject, keep `.rac.test` outputs scalar instead of nested wrappers like `{person: 1, value: ...}`.
- For a one-row fixed-amount slice, every `.rac.test` case should keep the row-defining conditions satisfied; do not negate them in alternate tests unless `./source.txt` states another grounded amount for that alternate branch.
- For a one-row fixed-amount slice with `period: Year`, a base case is sufficient; do not synthesize an `effective_date_boundary` test.
- For a one-row fixed-amount slice with non-annual periods, the allowed `.rac.test` shapes are base case and effective-date boundary.
- Add a later same-amount case only when `./source.txt` explicitly says the amount remains unchanged through that later date.
- Do not include `alternate_branch_*` tests unless `./source.txt` states a second grounded amount.
- Do not use thousands separators in RAC numeric literals or `.rac.test` outputs; write `2500`, not `2,500`.
"""
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
{definition_section}{context_section}

Rules:
- Do not inspect or rely on any path outside this workspace.
- Treat `./source.txt` as the only legal source.
- Any numeric literal in your output must appear in `./source.txt`, unless it is -1, 0, 1, 2, or 3.
- Every substantive numeric occurrence in `./source.txt` must be represented by a named scalar definition in RAC, even when the same numeric value repeats.
- If the same numeric value appears twice in materially different legal roles, declare separate named scalar variables for those separate occurrences instead of reusing a single scalar everywhere.
- If a legal scalar amount, threshold, cap, or limit appears in a formula or conditional branch, first declare it as its own named variable and then reference that variable from the formula.
- If `./source.txt` says someone is "aged 18 or over", "under 25", or gives another numeric eligibility threshold, model that threshold as a named scalar variable rather than only burying the number inside a helper name.
- Do not create scalar variables for citation numbers that only appear inside section, paragraph, regulation, schedule, or similar legal cross-references.
- Do not invent `dtype: String` variables just to restate the effective date or to hold quoted date text from `./source.txt`.
- Do not decompose legal dates into numeric `year`, `month`, or `day` scalar variables; keep date references semantic inside boolean/fact-shaped helpers instead.
- Include the source text in a triple-quoted docstring.
- Use RAC DSL conventions.
- If `./source.txt` explicitly cites another section or source for a definition, emit the upstream import instead of restating the concept locally.
- If `./source.txt` uses a legally-defined term for which a resolved canonical definition file is provided above, import that canonical definition instead of inventing a leaf-local helper.
- For resolved definition files listed above, the required syntax is an `imports:` block that references the exact `path#symbol` target.
- Do not replace a resolved canonical import with a local deferred symbol whose name is just a mangled version of the import target.
- If that cited upstream file is absent from this workspace, still emit the unresolved import path; the external-stub workflow is expected to fill it in later.
- If the source text only implies a shared concept, import an existing canonical concept only when one is actually present in the workspace; otherwise keep the helper local to this leaf.
- For isolated amount/rate leaves that cite same-instrument conditions or exceptions, do not fabricate sibling-file imports just because the text mentions another paragraph or schedule test. Model those cited conditions as local booleans or fact-shaped inputs unless the exact canonical import file is already present in this workspace.
- If no resolved definition file or copied precedent file shows you the import syntax, do not guess. Keep cited same-instrument conditions local instead of inventing `import` statements or `imports:` blocks.
- Do not invent schema keys like `namespace:`, `parameter`, `variable`, or `rule:`.
{schema_rules}{uk_guidance}{single_amount_row_guidance}{target_hint_guidance}
- Prefer standard RAC blocks shaped like:
  example_name:
      entity: TaxUnit
      period: Month
      dtype: Money
      unit: USD
      from 2024-07-01:
          165
- For conditionals, RAC uses inline conditional expressions like `if condition: value else: other_value`.
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
        target = rac_us_root / _import_target_to_path(import_target)
        if target.exists():
            dependencies.append(target)
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

        target = eval_root.joinpath(*workspace_path.parts[1:])
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
    settle_seconds: float = 5.0,
    poll_interval: float = 0.5,
) -> bool:
    """Wait for Codex CLI, terminating it once the last message file is stable."""
    start = time.time()
    last_snapshot: tuple[int, int] | None = None
    stable_since: float | None = None

    while True:
        if process.poll() is not None:
            return False

        if time.time() - start > timeout:
            raise subprocess.TimeoutExpired(process.args, timeout)

        if last_message_file.exists():
            try:
                text = last_message_file.read_text().strip()
                stat = last_message_file.stat()
            except OSError:
                text = ""
                stat = None

            if text and stat is not None:
                snapshot = (stat.st_size, stat.st_mtime_ns)
                if snapshot == last_snapshot:
                    stable_since = stable_since or time.time()
                    if time.time() - stable_since >= settle_seconds:
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
    if content.startswith('"""'):
        closing_index = content.find('"""', 3)
        if closing_index != -1:
            code_start = closing_index + 3
            return (
                content[:code_start]
                + _normalize_comma_numeric_literals(content[code_start:])
            )
    return _normalize_comma_numeric_literals(content)


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
    except ValueError:
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
    annual_base_period = None
    if annual_period:
        if rac_content and (
            from_match := re.search(r"\bfrom\s+(\d{4})-\d{2}-\d{2}:", rac_content)
        ):
            annual_base_period = int(from_match.group(1))
        elif source_text and (
            source_match := re.search(
                r"\b(?:text|current text)\s+valid\s+from\s+(\d{4})-\d{2}-\d{2}\b",
                source_text,
                flags=re.IGNORECASE,
            )
        ):
            annual_base_period = int(source_match.group(1))

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
        if annual_period and annual_base_period is not None and "period" not in normalized_case:
            normalized_case["period"] = annual_base_period
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

    if isinstance(payload, list):
        filtered = [
            normalize_case(case)
            for case in payload
            if not isinstance(case, dict) or should_keep(case.get("name"))
        ]
        return yaml.safe_dump(filtered, sort_keys=False).strip() + "\n"

    if isinstance(payload, dict):
        if isinstance(payload.get("tests"), list):
            payload["tests"] = [
                normalize_case(case)
                for case in payload["tests"]
                if not isinstance(case, dict) or should_keep(case.get("name"))
            ]
            return yaml.safe_dump(payload, sort_keys=False).strip() + "\n"

        filtered_items: dict[str, object] = {}
        for key, value in payload.items():
            case_name = key if isinstance(key, str) else None
            if should_keep(case_name):
                filtered_items[key] = normalize_case(value)
        return yaml.safe_dump(filtered_items, sort_keys=False).strip() + "\n"

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
                        rac_content=bundle.get(expected_path.name),
                        source_text=source_text,
                    )
            elif target_path == expected_path:
                content = _normalize_rac_code_numeric_literals(content)
            elif target_path == expected_test_path:
                content = _normalize_comma_numeric_literals(content)
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
            test_content = _normalize_comma_numeric_literals(test_content)
        expected_test_path.write_text(test_content)

    return True


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "eval"
