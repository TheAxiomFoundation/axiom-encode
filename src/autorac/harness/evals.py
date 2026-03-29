"""Model comparison evals for statute and source-slice encoding."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

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
    ValidatorPipeline,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_numbers_from_text,
)

EvalMode = Literal["cold", "repo-augmented"]
IMPORT_ITEM_PATTERN = re.compile(r"^\s*-\s*(['\"]?)([^'\"]+?)\1\s*$")


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


@dataclass
class EvalContextFile:
    """A context file copied into the eval workspace."""

    source_path: str
    workspace_path: str
    kind: str


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

    if backend not in {"claude", "codex"}:
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
            )
        )

    return results


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
) -> EvalArtifactMetrics:
    """Evaluate one RAC file with deterministic checks only."""
    pipeline = ValidatorPipeline(
        rac_us_path=rac_root,
        rac_path=rac_path,
        enable_oracles=False,
    )
    compile_result = pipeline._run_compile_check(rac_file)
    ci_result = pipeline._run_ci(rac_file)

    content = rac_file.read_text()
    embedded_source = extract_embedded_source_text(content)
    source_numbers = extract_numbers_from_text(embedded_source or source_text)

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

    return EvalArtifactMetrics(
        compile_pass=compile_result.passed,
        compile_issues=compile_result.issues,
        ci_pass=ci_result.passed,
        ci_issues=ci_result.issues,
        embedded_source_present=bool(embedded_source),
        grounded_numeric_count=sum(1 for item in grounding_metrics if item.grounded),
        ungrounded_numeric_count=sum(
            1 for item in grounding_metrics if not item.grounded
        ),
        grounding=grounding_metrics,
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

    prompt = _build_eval_prompt(citation, mode, workspace, workspace.context_files)
    response = _run_prompt_eval(runner, workspace, prompt)
    rac_content = _extract_rac_content(response.text)

    relative_output = citation_to_relative_rac_path(citation)
    output_file = Path(output_root) / runner.name / relative_output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if rac_content:
        output_file.write_text(rac_content)
        _hydrate_eval_root(output_file.parents[2], workspace)

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
            rac_root=output_file.parents[2],
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
        success=bool(rac_content) and response.error is None,
        error=response.error or (None if rac_content else "No RAC content returned"),
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

    prompt = _build_eval_prompt(source_id, mode, workspace, workspace.context_files)
    response = _run_prompt_eval(runner, workspace, prompt)
    rac_content = _extract_rac_content(response.text)

    relative_output = _source_identifier_to_relative_rac_path(source_id)
    output_file = Path(output_root) / runner.name / relative_output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if rac_content:
        output_file.write_text(rac_content)
        _hydrate_eval_root(output_file.parents[2], workspace)

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
            rac_root=output_file.parents[2],
            rac_path=rac_path,
            source_text=source_text,
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
        success=bool(rac_content) and response.error is None,
        error=response.error or (None if rac_content else "No RAC content returned"),
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
) -> str:
    """Build a prompt-only eval request with explicit provenance rules."""
    context_section = ""
    if context_files:
        listed = "\n".join(f"- `{item.workspace_path}`" for item in context_files)
        scaffold_dates = _collect_scaffold_dates(workspace, context_files)
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

    return f"""You are participating in an encoding eval for {citation}.

Primary legal authority:
- `./source.txt` contains the complete source text for this target source slice.

Context mode: `{mode}`
{context_section}

Rules:
- Do not inspect or rely on any path outside this workspace.
- Treat `./source.txt` as the only legal source.
- Any numeric literal in your output must appear in `./source.txt`, unless it is -1, 0, 1, 2, or 3.
- Return ONLY raw `.rac` file content for a single file.
- Do not include markdown fences or explanation.
- Include the source text in a triple-quoted docstring.
- Use RAC DSL conventions.
- Do not create tests.
"""


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
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=workspace.root,
        timeout=600,
    )
    duration_ms = int((time.time() - start) * 1000)

    events: list[dict] = []
    assistant_messages: list[str] = []
    usage_payload: dict | None = None
    unexpected_accesses: list[str] = []
    error = None

    for line in (result.stdout + result.stderr).splitlines():
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

    if result.returncode != 0 and not error:
        error = (result.stdout + result.stderr).strip() or "Codex eval failed"

    final_text = "\n".join(assistant_messages).strip()
    if last_message_file.exists():
        file_text = last_message_file.read_text().strip()
        if file_text:
            final_text = file_text

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


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "eval"
