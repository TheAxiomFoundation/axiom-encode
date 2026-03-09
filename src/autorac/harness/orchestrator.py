"""
Encoding Orchestrator -- self-contained pipeline for statute encoding.

Replaces the plugin-dependent SDKOrchestrator and encoding-orchestrator agent.
All prompts are embedded -- `pip install autorac` is sufficient.

Supports two backends:
- Claude Code CLI (subprocess) -- works with Max subscription
- Claude API (via anthropic SDK) -- works on Modal or any server

Usage:
    from autorac.harness.orchestrator import Orchestrator

    orchestrator = Orchestrator(model="opus")
    run = await orchestrator.encode("26 USC 21")
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

from autorac.constants import DEFAULT_CLI_MODEL, DEFAULT_MODEL
from autorac.prompts.encoder import ENCODER_PROMPT
from autorac.prompts.reviewers import (
    FORMULA_REVIEWER_PROMPT,
    INTEGRATION_REVIEWER_PROMPT,
    PARAMETER_REVIEWER_PROMPT,
    RAC_REVIEWER_PROMPT,
    get_formula_reviewer_prompt,
    get_integration_reviewer_prompt,
    get_parameter_reviewer_prompt,
    get_rac_reviewer_prompt,
)
from autorac.prompts.validator import VALIDATOR_PROMPT

from .encoding_db import EncodingDB, TokenUsage
from .validator_pipeline import ValidatorPipeline


class Phase(Enum):
    ANALYSIS = "analysis"
    ENCODING = "encoding"
    RESOLVE_EXTERNALS = "resolve_externals"
    ORACLE = "oracle"
    REVIEW = "review"
    REPORT = "report"


class Backend(Enum):
    """Backend for running Claude."""

    CLI = "cli"  # Claude Code CLI (subprocess)
    API = "api"  # Claude API (anthropic SDK)


@dataclass
class AgentMessage:
    """A single message in an agent conversation."""

    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tokens: Optional[TokenUsage] = None
    tool_name: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class AgentRun:
    """Complete record of a single agent invocation."""

    agent_type: str
    prompt: str
    phase: Phase
    messages: List[AgentMessage] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    total_tokens: Optional[TokenUsage] = None
    total_cost: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class EncodingRun:
    """Complete record of an orchestration run."""

    citation: str
    session_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None

    agent_runs: List[AgentRun] = field(default_factory=list)

    files_created: List[str] = field(default_factory=list)
    stubs_created: List[str] = field(default_factory=list)
    oracle_pe_match: Optional[float] = None
    oracle_taxsim_match: Optional[float] = None
    discrepancies: List[dict] = field(default_factory=list)

    total_tokens: Optional[TokenUsage] = None
    total_cost_usd: float = 0.0
    autorac_version: str = ""


STUB_GENERATOR_PROMPT = """You generate .rac stub files for external dependencies.

You categorize variables into three types:
1. **Computable from statute** (status: encoded) — the statute defines a clear formula
2. **Stub** (status: stub) — statutory definition exists but full encoding needs more context
3. **Input** (status: input) — an observable real-world fact no statute defines (wages, age, etc.)

Always output raw .rac content with no markdown fencing or explanation."""


# Agent prompt mapping -- all embedded, no plugin needed
AGENT_PROMPTS = {
    "encoder": ENCODER_PROMPT,
    "validator": VALIDATOR_PROMPT,
    "rac_reviewer": RAC_REVIEWER_PROMPT,
    "formula_reviewer": FORMULA_REVIEWER_PROMPT,
    "parameter_reviewer": PARAMETER_REVIEWER_PROMPT,
    "integration_reviewer": INTEGRATION_REVIEWER_PROMPT,
    "stub_generator": STUB_GENERATOR_PROMPT,
}

# DSL cheatsheet appended to encoder prompts for subsection-level encoding
DSL_CHEATSHEET = """
## RAC DSL quick reference (unified syntax)

### Declaration types
- `name:` with `from yyyy-mm-dd:` -- policy value with temporal entries
- `name:` with `formula:` -- computed value
- `name:` with `default:` -- user-provided input
- `enum Name:` -- enumeration with `values:` list

### Fields (all required unless noted)
- `entity:` Person | TaxUnit | Household | Family
- `period:` Year | Month | Day
- `dtype:` Money | Rate | Boolean | Integer | String | Enum[Name]
- `formula: |` -- Python-like formula (see below)
- `imports:` -- list of `path#name` (optional)

### Temporal syntax
```yaml
ctc_base_amount:
  from 2018-01-01: 2000
  from 2025-01-01: 2500
```

### Formula syntax
**Allowed:** `if`/`elif`/`else`, `return`, `and`/`or`/`not`, `=` assignment
**Operators:** `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`
**Built-in functions:** `min(a,b)`, `max(a,b)`, `abs(x)`, `floor(x)`, `ceil(x)`, `round(x,n)`, `clamp(x,lo,hi)`, `sum(...)`, `len(...)`
**FORBIDDEN:** `for`/`while` loops, list comprehensions, `def`/`lambda`, `try`/`except`, `+=`, imports
**Numeric literals:** ONLY -1, 0, 1, 2, 3 allowed. ALL other numbers must come from named definitions.
"""


@dataclass
class SubsectionTask:
    """A single subsection to encode."""

    subsection_id: str
    title: str
    file_name: str
    dependencies: list = field(default_factory=list)
    wave: int = 0


class Orchestrator:
    """Orchestrates the full encoding workflow in Python.

    Self-contained -- all prompts are embedded, no external plugin needed.
    Works with `pip install autorac` on Modal or anywhere.
    """

    def __init__(
        self,
        model: str | None = None,
        db_path: Path | None = None,
        backend: Backend | str = Backend.CLI,
        api_key: str | None = None,
        atlas_path: Path | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            model: Model to use. For CLI backend, short names like "opus".
                   For API backend, full model IDs like "claude-opus-4-6".
            db_path: Path to encoding database. None to skip logging.
            backend: "cli" for Claude Code CLI, "api" for direct API calls.
            api_key: Anthropic API key (required for API backend).
            atlas_path: Path to atlas repo. Falls back to ATLAS_PATH env var,
                        then ~/RulesFoundation/atlas.
        """
        if isinstance(backend, str):
            backend = Backend(backend)

        self.backend = backend
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if self.backend == Backend.API and not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required for API backend")

        if self.backend == Backend.CLI:
            self.model = model or DEFAULT_CLI_MODEL
        else:
            self.model = model or DEFAULT_MODEL

        self.encoding_db = EncodingDB(db_path) if db_path else None

        # Resolve atlas path: explicit > env var > default
        if atlas_path:
            self.atlas_path = Path(atlas_path)
        elif os.environ.get("ATLAS_PATH"):
            self.atlas_path = Path(os.environ["ATLAS_PATH"])
        else:
            self.atlas_path = Path.home() / "RulesFoundation" / "atlas"

        # Cache env without CLAUDECODE (prevents nested launch errors)
        # Preserve PYTHONPATH so subprocess can import atlas and other packages
        self._cli_env = (
            {
                k: v
                for k, v in os.environ.items()
                if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
            }
            if self.backend == Backend.CLI
            else None
        )
        self._context_section = self._build_context_section()

    async def encode(
        self,
        citation: str,
        output_path: Path | None = None,
        statute_text: str | None = None,
    ) -> EncodingRun:
        """Full pipeline: encode -> validate -> review -> log.

        Args:
            citation: Legal citation, e.g. "26 USC 21"
            output_path: Directory for output .rac files.
                         Defaults to ~/RulesFoundation/rac-us/statute/{title}/{section}/
            statute_text: Pre-fetched statute text. If None, encoder will fetch it.

        Returns:
            EncodingRun with full results.
        """
        # Derive output path from citation if not provided
        if output_path is None:
            citation_clean = (
                citation.replace("USC", "")
                .replace("usc", "")
                .replace("\u00a7", "")
                .strip()
            )
            parts = citation_clean.split()
            if len(parts) >= 2:
                title = parts[0]
                section = parts[1]
            else:
                path_parts = citation_clean.replace(" ", "/").split("/")
                title = path_parts[0]
                section = "/".join(path_parts[1:])
            output_path = (
                Path.home()
                / "RulesFoundation"
                / "rac-us"
                / "statute"
                / title
                / section.replace("(", "/").replace(")", "")
            )

        from autorac import __version__

        run = EncodingRun(
            citation=citation,
            session_id=f"orch-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            autorac_version=__version__,
        )

        try:
            # Create DB session
            if self.encoding_db:
                self.encoding_db.start_session(
                    model=self.model,
                    cwd=str(Path.cwd()),
                    session_id=run.session_id,
                )

            # Pre-fetch statute text if not provided
            if not statute_text:
                statute_text = self._fetch_statute_text(citation)

            # Phase 1: Analysis
            print(
                f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] ANALYSIS: {citation}",
                flush=True,
            )
            analysis_prompt = self._build_analyzer_prompt(
                citation, statute_text=statute_text
            )
            analysis = await self._run_agent(
                agent_key="encoder",
                prompt=analysis_prompt,
                phase=Phase.ANALYSIS,
            )
            run.agent_runs.append(analysis)
            self._log_agent_run(run.session_id, analysis)

            # Phase 2: Encoding
            if analysis.result:
                encoding_runs = await self._run_encoding_parallel(
                    citation, output_path, statute_text, analysis.result
                )
                for enc_run in encoding_runs:
                    run.agent_runs.append(enc_run)
                    self._log_agent_run(run.session_id, enc_run)

                if not encoding_runs:
                    # Fallback: single encoder
                    print(
                        "  No subsections parsed, falling back to single encoder",
                        flush=True,
                    )
                    encode_prompt = self._build_fallback_encode_prompt(
                        citation, output_path, statute_text
                    )
                    encoding = await self._run_agent(
                        agent_key="encoder",
                        prompt=encode_prompt,
                        phase=Phase.ENCODING,
                    )
                    run.agent_runs.append(encoding)
                    self._log_agent_run(run.session_id, encoding)
            else:
                encode_prompt = self._build_fallback_encode_prompt(
                    citation, output_path, statute_text
                )
                encoding = await self._run_agent(
                    agent_key="encoder",
                    prompt=encode_prompt,
                    phase=Phase.ENCODING,
                )
                run.agent_runs.append(encoding)
                self._log_agent_run(run.session_id, encoding)

            # Check created files
            if output_path.exists():
                run.files_created = [str(f) for f in output_path.rglob("*.rac")]

            # Phase 2.5: Resolve external dependencies (create stubs)
            stubs = await self._resolve_external_dependencies(output_path)
            if stubs:
                run.stubs_created = [str(s) for s in stubs]
                # Re-scan to include stubs in files_created
                run.files_created = [str(f) for f in output_path.rglob("*.rac")]
                # Also include stubs outside the output_path
                for stub_path in stubs:
                    stub_str = str(stub_path)
                    if stub_str not in run.files_created:
                        run.files_created.append(stub_str)

            # Phase 3: Oracle validation
            oracle_context = await self._run_oracle_validation(output_path)
            run.oracle_pe_match = oracle_context.get("pe_match")
            run.oracle_taxsim_match = oracle_context.get("taxsim_match")
            run.discrepancies = oracle_context.get("discrepancies", [])

            # Phase 4: LLM Review (parallel)
            oracle_summary = self._format_oracle_summary(oracle_context)
            review_runs = await self._run_reviews_parallel(citation, oracle_summary)
            for rev_run in review_runs:
                run.agent_runs.append(rev_run)
                self._log_agent_run(run.session_id, rev_run)

            # Phase 5: Report
            run.ended_at = datetime.utcnow()
            run.total_tokens = self._sum_tokens(run.agent_runs)
            run.total_cost_usd = self._sum_cost(run.agent_runs)

            if self.encoding_db:
                self._log_to_db(run)

        except Exception as e:
            run.ended_at = datetime.utcnow()
            if run.agent_runs:
                run.agent_runs[-1].error = str(e)
            print(f"  ERROR: {e}", flush=True)

        return run

    async def _run_agent(
        self,
        agent_key: str,
        prompt: str,
        phase: Phase,
    ) -> AgentRun:
        """Run a single agent using the configured backend."""
        system_prompt = AGENT_PROMPTS.get(agent_key, "")
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n# TASK\n\n{prompt}"
        else:
            full_prompt = prompt

        agent_run = AgentRun(
            agent_type=agent_key,
            prompt=prompt,
            phase=phase,
        )

        start_time = datetime.utcnow()
        print(
            f"\n[{start_time.strftime('%H:%M:%S')}] {phase.value.upper()}: {agent_key}",
            flush=True,
        )

        try:
            if self.backend == Backend.CLI:
                result = await self._run_via_cli(full_prompt)
            else:
                result = await self._run_via_api(full_prompt, system_prompt, prompt)

            agent_run.result = result.get("text", "")
            agent_run.total_tokens = result.get("tokens")
            agent_run.total_cost = result.get("cost")

        except Exception as e:
            agent_run.error = str(e)
            print(f"  ERROR: {e}", flush=True)

        agent_run.ended_at = datetime.utcnow()
        duration = (agent_run.ended_at - start_time).total_seconds()
        print(f"  DONE ({duration:.1f}s)", flush=True)

        return agent_run

    async def _run_via_cli(self, prompt: str) -> dict:
        """Run via Claude Code CLI subprocess."""
        import subprocess

        cmd = [
            "claude",
            "--print",
            "--model",
            self.model,
            "--mcp-config",
            '{"mcpServers":{}}',
            "--strict-mcp-config",
            "-p",
            prompt,
        ]

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=self._cli_env,
                ),
            )
            text = result.stdout + result.stderr
            return {"text": text}
        except subprocess.TimeoutExpired:
            return {"text": "Timeout after 600s"}
        except FileNotFoundError:
            return {
                "text": "Claude CLI not found - install with: npm install -g @anthropic-ai/claude-code"
            }

    async def _run_via_api(
        self, full_prompt: str, system_prompt: str, user_prompt: str
    ) -> dict:
        """Run via Claude API directly (for Modal/server deployments)."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)

            messages = [{"role": "user", "content": user_prompt}]
            kwargs = {
                "model": self.model,
                "max_tokens": 16384,
                "messages": messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = await client.messages.create(**kwargs)

            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            tokens = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            return {"text": text, "tokens": tokens}

        except ImportError:
            raise ImportError("anthropic SDK not installed. Run: pip install anthropic")

    async def _run_oracle_validation(self, output_path: Path) -> dict:
        """Run oracle validation using ValidatorPipeline."""
        print(
            f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] ORACLE: ValidatorPipeline",
            flush=True,
        )
        oracle_start = time.time()

        rac_files = list(output_path.rglob("*.rac")) if output_path.exists() else []

        oracle_context = {
            "pe_match": None,
            "taxsim_match": None,
            "discrepancies": [],
        }

        if rac_files:
            pipeline = ValidatorPipeline(
                rac_us_path=output_path.parent.parent
                if "statute" in str(output_path)
                else output_path,
                rac_path=Path(__file__).parent.parent.parent.parent / "rac",
                enable_oracles=True,
                max_workers=2,
            )

            pe_scores = []
            taxsim_scores = []
            all_issues = []

            for rac_file in rac_files:
                print(f"  Validating: {rac_file.name}", flush=True)
                try:
                    pe_result = pipeline._run_policyengine(rac_file)
                    if pe_result.score is not None:
                        pe_scores.append(pe_result.score)
                        print(f"    PE: {pe_result.score:.1%}", flush=True)
                    all_issues.extend(pe_result.issues)

                    taxsim_result = pipeline._run_taxsim(rac_file)
                    if taxsim_result.score is not None:
                        taxsim_scores.append(taxsim_result.score)
                        print(f"    TAXSIM: {taxsim_result.score:.1%}", flush=True)
                    all_issues.extend(taxsim_result.issues)
                except Exception as e:
                    print(f"    Error: {e}", flush=True)
                    all_issues.append(str(e))

            if pe_scores:
                oracle_context["pe_match"] = sum(pe_scores) / len(pe_scores) * 100
            if taxsim_scores:
                oracle_context["taxsim_match"] = (
                    sum(taxsim_scores) / len(taxsim_scores) * 100
                )
            oracle_context["files_tested"] = len(pe_scores) + len(taxsim_scores)
            oracle_context["files_total"] = len(rac_files)
            oracle_context["files_untested"] = len(rac_files) - max(
                len(pe_scores), len(taxsim_scores)
            )
            oracle_context["discrepancies"] = [
                {"description": issue} for issue in all_issues[:10]
            ]

            # Microdata benchmark: run PE microsimulation for target variable
            pe_variable = self._infer_pe_variable(output_path)
            if pe_variable:
                print(
                    f"  Microdata benchmark: {pe_variable} (CPS)",
                    flush=True,
                )
                try:
                    benchmark = pipeline._run_microdata_benchmark(
                        output_path, pe_variable=pe_variable
                    )
                    oracle_context["microdata_benchmark"] = {
                        "variable": pe_variable,
                        "score": benchmark.score,
                        "issues": benchmark.issues,
                    }
                    if benchmark.raw_output:
                        oracle_context["microdata_stats"] = benchmark.raw_output
                    for issue in benchmark.issues:
                        print(f"    {issue}", flush=True)
                except Exception as e:
                    print(f"    Benchmark error: {e}", flush=True)
        else:
            print("  No RAC files found to validate", flush=True)

        duration = time.time() - oracle_start
        total = oracle_context.get("files_total", 0)
        untested = oracle_context.get("files_untested", total)
        pe_str = (
            f"{oracle_context['pe_match']:.1f}%"
            if oracle_context.get("pe_match") is not None
            else "UNTESTED"
        )
        taxsim_str = (
            f"{oracle_context['taxsim_match']:.1f}%"
            if oracle_context.get("taxsim_match") is not None
            else "UNTESTED"
        )
        print(
            f"  DONE: PE={pe_str}, TAXSIM={taxsim_str} "
            f"({untested}/{total} files had no tests) ({duration:.1f}s)",
            flush=True,
        )

        return oracle_context

    async def _run_reviews_parallel(
        self, citation: str, oracle_summary: str
    ) -> List[AgentRun]:
        """Run all 4 reviewers in parallel."""
        reviewers = [
            ("rac_reviewer", get_rac_reviewer_prompt(citation, oracle_summary)),
            ("formula_reviewer", get_formula_reviewer_prompt(citation, oracle_summary)),
            (
                "parameter_reviewer",
                get_parameter_reviewer_prompt(citation, oracle_summary),
            ),
            (
                "integration_reviewer",
                get_integration_reviewer_prompt(citation, oracle_summary),
            ),
        ]

        tasks = [
            self._run_agent(
                agent_key=key,
                prompt=prompt,
                phase=Phase.REVIEW,
            )
            for key, prompt in reviewers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        runs = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"  FAILED ({reviewers[i][0]}): {r}", flush=True)
            else:
                runs.append(r)

        return runs

    # ========================================================================
    # Analysis and encoding helpers
    # ========================================================================

    def _build_analyzer_prompt(
        self, citation: str, statute_text: str | None = None
    ) -> str:
        """Build the analysis prompt with structured output instructions."""
        text_section = ""
        if statute_text:
            text_section = f"""

## Statute Text

The following is the AUTHORITATIVE text of {citation}. Use ONLY this text to identify subsections.

<statute>
{statute_text}
</statute>
"""
        else:
            text_section = f"""

NOTE: Statute text for {citation} was not available. Fetch it using WebFetch or WebSearch.
"""

        return f"""Analyze {citation}. Report: subsection tree, encoding order, dependencies.
{text_section}
After your markdown analysis, include a machine-readable block:
<!-- STRUCTURED_OUTPUT
{{"subsections": [{{"id": "a", "title": "...", "disposition": "ENCODE", "file": "a.rac"}}, ...],
 "dependencies": {{"b": ["a"], "d": ["a"]}},
 "encoding_order": ["a", "c", "e", "f", "b", "d"]}}
-->

Valid dispositions: "ENCODE", "SKIP", "OBSOLETE"
List dependencies as subsection IDs that must be encoded first."""

    def _build_fallback_encode_prompt(
        self,
        citation: str,
        output_path: Path,
        statute_text: str | None,
    ) -> str:
        """Build encoding prompt for single-agent fallback."""
        return f"""Encode {citation} into RAC format.

Output path: {output_path}
{f"Statute text: {statute_text[:5000]}" if statute_text else "Fetch statute text as needed."}

## CRITICAL RULES:

1. **FILEPATH = CITATION** - File names MUST be subsection names
2. **One subsection per file**
3. **Only statute values** - No indexed/derived/computed values
4. **COMPILE CHECK** - Run `autorac compile <file>` after writing each .rac file to verify it parses and compiles correctly.
5. **WRITE TESTS** - Write 3-5 test cases in a companion `.rac.test` file next to your `.rac` file. Tests should cover the main computation, edge cases (zero values, thresholds), and boundary conditions.
6. **PARENT IMPORTS FROM CHILDREN** - Parent files MUST import from their children using `from ./{{child}}` — NEVER re-define parameters or formulas that exist in child files. Parents are aggregators/routers only.
7. **INDEXED_BY FOR INFLATION** - For parameters subject to inflation/COLA adjustments (e.g., dollar thresholds in 26 USC 1(f)), include `indexed_by: <index_variable>` in the parameter definition.

Write .rac files to the output path. Run `autorac test` after each file.
{DSL_CHEATSHEET}
{self._context_section}"""

    def _parse_analyzer_output(self, analysis_text: str) -> list[SubsectionTask]:
        """Parse analyzer output into structured subsection tasks."""
        if not analysis_text or not analysis_text.strip():
            return []

        tasks = []

        # Primary: look for STRUCTURED_OUTPUT JSON block
        json_match = re.search(
            r"<!--\s*STRUCTURED_OUTPUT\s*\n(.*?)\n\s*-->",
            analysis_text,
            re.DOTALL,
        )
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                deps = data.get("dependencies", {})
                for s in data.get("subsections", []):
                    if s.get("disposition", "").upper() != "ENCODE":
                        continue
                    tasks.append(
                        SubsectionTask(
                            subsection_id=s["id"],
                            title=s.get("title", ""),
                            file_name=s.get("file", f"{s['id']}.rac"),
                            dependencies=deps.get(s["id"], []),
                        )
                    )
                return tasks
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: parse markdown table rows
        row_pattern = re.compile(
            r"\|\s*\((\w+)\)\s*\|\s*([^|]+?)\s*\|\s*(\w+)\s*\|\s*([^|]+?)\s*\|"
        )
        for m in row_pattern.finditer(analysis_text):
            sub_id, title, disposition, file_name = (
                m.group(1),
                m.group(2).strip(),
                m.group(3).strip(),
                m.group(4).strip(),
            )
            if disposition.upper() != "ENCODE":
                continue
            tasks.append(
                SubsectionTask(
                    subsection_id=sub_id,
                    title=title,
                    file_name=file_name if file_name != "-" else f"{sub_id}.rac",
                )
            )

        return tasks

    def _compute_waves(self, tasks: List[SubsectionTask]) -> List[List[SubsectionTask]]:
        """Topological sort into parallel batches (waves)."""
        if not tasks:
            return []

        assigned = set()
        waves = []

        while len(assigned) < len(tasks):
            wave = []
            for t in tasks:
                if t.subsection_id in assigned:
                    continue
                if all(d in assigned for d in t.dependencies):
                    t.wave = len(waves)
                    wave.append(t)
            if not wave:
                remaining = [t for t in tasks if t.subsection_id not in assigned]
                for t in remaining:
                    t.wave = len(waves)
                waves.append(remaining)
                break
            assigned.update(t.subsection_id for t in wave)
            waves.append(wave)

        return waves

    async def _run_encoding_parallel(
        self,
        citation: str,
        output_path: Path,
        statute_text: str | None,
        analysis_result: str,
        max_concurrent: int = 5,
    ) -> List[AgentRun]:
        """Encode subsections in parallel waves."""
        tasks = self._parse_analyzer_output(analysis_result)
        if not tasks:
            return []

        waves = self._compute_waves(tasks)
        semaphore = asyncio.Semaphore(max_concurrent)
        all_runs: List[AgentRun] = []

        for wave_idx, wave in enumerate(waves):
            wave_ids = ", ".join(f"({t.subsection_id})" for t in wave)
            print(
                f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] "
                f"ENCODING WAVE {wave_idx}: {wave_ids}",
                flush=True,
            )

            async def encode_one(task: SubsectionTask) -> AgentRun:
                async with semaphore:
                    prompt = self._build_subsection_prompt(
                        task, citation, output_path, statute_text
                    )
                    return await self._run_agent("encoder", prompt, Phase.ENCODING)

            results = await asyncio.gather(
                *[encode_one(t) for t in wave],
                return_exceptions=True,
            )

            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    print(f"  FAILED ({wave[i].subsection_id}): {r}", flush=True)
                else:
                    all_runs.append(r)

        # Final aggregation wave: produce root .rac file
        if len(tasks) > 1:
            print(
                f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] "
                f"ENCODING WAVE {len(waves)}: ROOT AGGREGATOR",
                flush=True,
            )
            aggregator_prompt = self._build_aggregator_prompt(
                citation, output_path, tasks
            )
            aggregator_run = await self._run_agent(
                "encoder", aggregator_prompt, Phase.ENCODING
            )
            all_runs.append(aggregator_run)

        return all_runs

    def _build_subsection_prompt(
        self,
        task: SubsectionTask,
        citation: str,
        output_path: Path,
        statute_text: str | None = None,
    ) -> str:
        """Build a focused encoding prompt for a single subsection."""
        # Strip leading prefix from file_name if it duplicates output_path's tail.
        # E.g. output_path=.../24/d and file_name=d/1.rac → file_name=1.rac
        file_name = task.file_name
        tail = output_path.name
        if file_name.startswith(f"{tail}/"):
            file_name = file_name[len(tail) + 1 :]

        parts = [
            f"Encode {citation} subsection ({task.subsection_id}) - "
            f'"{task.title}" into RAC format.',
            "",
            f"Output: {output_path / file_name}",
            f"Scope: ONLY this subsection. One file: {file_name}",
            "",
            "## CRITICAL RULES:",
            "",
            "1. **FILEPATH = CITATION** - File names MUST be subsection names",
            "2. **One subsection per file**",
            "3. **Only statute values** - No indexed/derived/computed values",
            "4. **COMPILE CHECK** - Run `autorac compile <file>` after writing "
            "each .rac file to verify it parses and compiles correctly.",
            "5. **WRITE TESTS** - Write 3-5 test cases in a companion `.rac.test` "
            "file next to your `.rac` file. Tests should cover the main computation, "
            "edge cases (zero values, thresholds), and boundary conditions.",
            "6. **PARENT IMPORTS FROM CHILDREN** - Parent files MUST import from "
            "their children using `from ./{child}` — NEVER re-define parameters or "
            "formulas that exist in child files. Parents are aggregators/routers only.",
            "7. **INDEXED_BY FOR INFLATION** - For parameters subject to "
            "inflation/COLA adjustments (e.g., dollar thresholds in 26 USC 1(f)), "
            "include `indexed_by: <index_variable>` in the parameter definition.",
            "",
            "Write the .rac file to the output path. Run `autorac test` after writing.",
        ]

        if task.dependencies:
            dep_files = ", ".join(f"{d}.rac" for d in task.dependencies)
            parts.append("")
            parts.append(
                f"Note: This subsection depends on {dep_files} "
                f"(already encoded). You may import from those files."
            )

        if statute_text:
            subsection_text = self._extract_subsection_text(
                statute_text, task.subsection_id
            )
            if subsection_text:
                parts.append("")
                parts.append(
                    f"## Statute text for subsection ({task.subsection_id}):\n"
                    f"{subsection_text}"
                )
            else:
                # Fallback: provide truncated full text
                parts.append("")
                parts.append(f"Full statute text (excerpt):\n{statute_text[:5000]}")

        parts.append(DSL_CHEATSHEET)
        if self._context_section:
            parts.append(self._context_section)

        return "\n".join(parts)

    def _extract_subsection_text(
        self, statute_text: str, subsection_id: str
    ) -> str | None:
        """Extract the text for a specific subsection from the full statute.

        Looks for patterns like "(a)" in the statute text and extracts from
        that marker to the next sibling subsection marker.
        """
        if not statute_text or not subsection_id:
            return None

        clean_id = subsection_id.strip("()")

        escaped_id = re.escape(clean_id)
        start_pattern = rf"\({escaped_id}\)"
        start_match = re.search(start_pattern, statute_text)
        if not start_match:
            return None

        start_pos = start_match.start()

        # Find next sibling subsection to determine end boundary
        if clean_id.isalpha() and len(clean_id) == 1:
            next_letter = chr(ord(clean_id) + 1)
            end_pattern = rf"\({re.escape(next_letter)}\)"
        elif clean_id.isdigit():
            next_num = str(int(clean_id) + 1)
            end_pattern = rf"\({re.escape(next_num)}\)"
        else:
            end_pos = min(start_pos + 10000, len(statute_text))
            return statute_text[start_pos:end_pos].strip()

        end_match = re.search(end_pattern, statute_text[start_pos + 1 :])
        if end_match:
            end_pos = start_pos + 1 + end_match.start()
        else:
            end_pos = len(statute_text)

        extracted = statute_text[start_pos:end_pos].strip()

        if len(extracted) > 15000:
            extracted = extracted[:15000] + "\n... [truncated]"

        return extracted if extracted else None

    def _build_aggregator_prompt(
        self,
        citation: str,
        output_path: Path,
        tasks: list[SubsectionTask],
    ) -> str:
        """Build prompt for the final aggregation wave that produces the root .rac file."""
        citation_clean = (
            citation.replace("USC", "").replace("usc", "").replace("\u00a7", "").strip()
        )
        parts_list = citation_clean.split()
        section = parts_list[-1] if parts_list else "root"
        # Strip parentheses for filepath — "24(d)" → "d" (last path component)
        section_clean = section.replace("(", "/").replace(")", "").rstrip("/")
        root_name = (
            section_clean.split("/")[-1] if "/" in section_clean else section_clean
        )
        root_file = f"{root_name}.rac"

        children_lines = []
        for t in tasks:
            children_lines.append(
                f"- `{t.file_name}`: subsection ({t.subsection_id}) - {t.title}"
            )
        children_list = "\n".join(children_lines)

        parts = [
            f"Create the ROOT aggregator file for {citation}.",
            "",
            f"Output: {output_path / root_file}",
            f"File name: {root_file}",
            "",
            "## Purpose",
            "",
            "This root file imports from all subsection files and composes the",
            "final top-level computation. It should NOT duplicate any logic from",
            "the subsection files -- only import and combine them.",
            "",
            "## Subsection files (already encoded):",
            "",
            children_list,
            "",
            "## CRITICAL RULES:",
            "",
            "1. Import each subsection file using `imports:` with `path#name` syntax",
            "2. Do NOT re-encode any subsection logic -- only import and compose",
            "3. The root file defines the top-level variable(s) that combine subsection results",
            "4. Use the same entity/period/dtype patterns as the subsection files",
            "",
            DSL_CHEATSHEET,
        ]

        if self._context_section:
            parts.append(self._context_section)

        return "\n".join(parts)

    # ========================================================================
    # Encoding context for agents
    # ========================================================================

    def _build_context_section(self) -> str:
        """Build a context section pointing agents to past encodings.

        Only useful for CLI backend (agents have filesystem/shell access).
        API backend agents can't run sqlite3, so returns empty string.
        """
        if self.backend != Backend.CLI:
            return ""

        db_path = (
            self.encoding_db.db_path
            if self.encoding_db
            else Path.home() / "RulesFoundation" / "autorac" / "encodings.db"
        )
        rac_us_path = Path.home() / "RulesFoundation" / "rac-us" / "statute"

        return f"""

## Past encoding reference (explore if useful for complex sections)

Encoding DB: {db_path}
Key table: encoding_runs — columns: citation, rac_content, lessons, iterations_json, review_results_json

Example queries:
  sqlite3 {db_path} "SELECT citation, lessons FROM encoding_runs WHERE lessons != '' ORDER BY timestamp DESC LIMIT 5"
  sqlite3 {db_path} "SELECT rac_content FROM encoding_runs WHERE citation LIKE '%21%' AND rac_content != '' LIMIT 1"

Existing .rac files: {rac_us_path}/
Read any .rac file for reference on style and patterns."""

    # ========================================================================
    # External dependency resolution
    # ========================================================================

    # Map IRC section → PolicyEngine variable for microdata benchmarking
    _PE_VARIABLE_MAP = {
        "32": "eitc",
        "24": "ctc",
        "21": "cdcc",  # child and dependent care credit
        "36B": "premium_tax_credit",
        "63": "standard_deduction",
        "1": "income_tax",
        "62": "adjusted_gross_income",
    }

    def _infer_pe_variable(self, output_path: Path) -> str | None:
        """Infer the PolicyEngine variable from the output path.

        Maps IRC section numbers to PE variable names for benchmarking.
        """
        # Extract section from path like .../statute/26/32/...
        parts = output_path.parts
        try:
            statute_idx = parts.index("statute")
            if statute_idx + 2 < len(parts):
                section = parts[statute_idx + 2]  # e.g., "32"
                return self._PE_VARIABLE_MAP.get(section)
        except ValueError:
            pass
        return None

    _IMPORT_RE = re.compile(r"^\s*-\s+(\S+)#(\S+)", re.MULTILINE)

    def _find_statute_root(self, output_path: Path) -> Path:
        """Find the statute/ root directory by walking up from output_path."""
        current = output_path
        while current != current.parent:
            if current.name == "statute":
                return current
            current = current.parent
        return Path.home() / "RulesFoundation" / "rac-us" / "statute"

    def _scan_unresolved_imports(
        self, output_path: Path
    ) -> list[tuple[str, str, Path]]:
        """Scan .rac files for imports whose target files don't exist.

        Returns list of (citation_path, variable_name, expected_file_path) tuples.
        """
        rac_us_root = self._find_statute_root(output_path)
        unresolved = []
        seen = set()

        # Scan all .rac files in the output directory
        for rac_file in output_path.rglob("*.rac"):
            content = rac_file.read_text()
            for match in self._IMPORT_RE.finditer(content):
                import_path = match.group(1)
                var_name = match.group(2)

                # Strip " as alias" if present
                if " as " in var_name:
                    var_name = var_name.split(" as ")[0]

                key = (import_path, var_name)
                if key in seen:
                    continue
                seen.add(key)

                # Check if target file exists
                # import_path like "26/62" → statute/26/62/62.rac or statute/26/62.rac
                target_candidates = [
                    rac_us_root / import_path / f"{import_path.split('/')[-1]}.rac",
                    rac_us_root / f"{import_path}.rac",
                ]
                # Also check parent dir for subsection imports
                # e.g., "26/21/b" → statute/26/21/b.rac
                parts = import_path.split("/")
                if len(parts) >= 2:
                    parent = "/".join(parts[:-1])
                    target_candidates.append(rac_us_root / parent / f"{parts[-1]}.rac")

                found = False
                for candidate in target_candidates:
                    if candidate.exists():
                        # Verify the variable is actually defined in the file
                        try:
                            target_content = candidate.read_text()
                            if f"{var_name}:" in target_content:
                                found = True
                                break
                        except Exception:
                            pass

                if not found:
                    # Determine where the file should go
                    expected = rac_us_root / f"{import_path}.rac"
                    # If path looks like a section (e.g., "26/62"), use dir/section.rac
                    if len(parts) == 2:
                        expected = rac_us_root / parts[0] / parts[1] / f"{parts[1]}.rac"
                    unresolved.append((import_path, var_name, expected))

        return unresolved

    def _citation_from_path(self, import_path: str) -> str:
        """Convert an import path like '26/62' to a citation like '26 USC 62'."""
        parts = import_path.split("/")
        if len(parts) < 2:
            return import_path
        title = parts[0]
        section = parts[1]
        # Build subsection notation: 26/21/b/1/C → 26 USC 21(b)(1)(C)
        subsections = "".join(f"({p})" for p in parts[2:])
        return f"{title} USC {section}{subsections}"

    async def _resolve_external_dependencies(self, output_path: Path) -> list[Path]:
        """Scan encoded files for unresolved imports, create stubs for missing ones.

        For each unresolved import:
        1. Fetch statute text from atlas
        2. Ask LLM to determine: computable (encode), stub (needs future work), or input
        3. Write the appropriate .rac file

        Returns list of stub file paths created.
        """
        unresolved = self._scan_unresolved_imports(output_path)
        if not unresolved:
            return []

        print(
            f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] RESOLVE_EXTERNALS: "
            f"{len(unresolved)} unresolved import(s)",
            flush=True,
        )

        # Group by target file (multiple variables may come from same section)
        from collections import defaultdict

        by_file: dict[Path, list[tuple[str, str]]] = defaultdict(list)
        for import_path, var_name, expected_path in unresolved:
            by_file[expected_path].append((import_path, var_name))

        created: list[Path] = []

        for expected_path, vars_list in by_file.items():
            import_path = vars_list[0][0]
            var_names = [v[1] for v in vars_list]
            citation = self._citation_from_path(import_path)

            print(f"  Creating stub for {citation}: {', '.join(var_names)}", flush=True)

            # Fetch statute text
            statute_text = self._fetch_statute_text(citation)

            # Build prompt for LLM to generate the stub
            prompt = self._build_stub_prompt(
                citation, import_path, var_names, statute_text
            )

            agent_run = await self._run_agent(
                agent_key="stub_generator",
                prompt=prompt,
                phase=Phase.RESOLVE_EXTERNALS,
            )

            if agent_run.result and not agent_run.error:
                # Extract .rac content from LLM response
                rac_content = self._extract_rac_content(agent_run.result)
                if rac_content:
                    expected_path.parent.mkdir(parents=True, exist_ok=True)
                    expected_path.write_text(rac_content)
                    created.append(expected_path)
                    print(f"    Wrote: {expected_path}", flush=True)
                else:
                    print(
                        "    Warning: could not extract .rac from response", flush=True
                    )
            else:
                print(f"    Warning: stub generation failed for {citation}", flush=True)

        if created:
            print(f"  Created {len(created)} stub file(s)", flush=True)

        return created

    def _build_stub_prompt(
        self,
        citation: str,
        import_path: str,
        var_names: list[str],
        statute_text: str | None,
    ) -> str:
        """Build prompt for LLM to generate a stub .rac file."""
        statute_section = ""
        if statute_text:
            statute_section = f"""
## Statute text

```
{statute_text[:5000]}
```
"""

        vars_section = "\n".join(f"- `{v}`" for v in var_names)

        return f"""Generate a .rac stub file for {citation} (import path: {import_path}).

The following variable(s) are imported from this section by other encoded files:
{vars_section}

{statute_section}

## Your task

For each variable listed above, determine which category it falls into:

1. **Computable from statute**: The statute text defines this variable with a clear formula
   or rule. Write `status: encoded` and include the formula.

2. **Defined but not yet encodable**: The statute defines this concept but encoding the full
   formula requires other sections not yet available. Write `status: stub` with entity/period/dtype
   metadata but no formula.

3. **True input**: No statute defines this — it's an observable fact about the taxpayer
   (wages earned, age, months of school attendance, etc.). Write `status: input` with
   entity/period/dtype metadata and a `default:` value (usually 0 for Money/Integer,
   false for Boolean).

## Output format

Return ONLY the .rac file content. No markdown fences, no explanation. The file should:
- Start with `# {citation}` header
- Include the statute text in a triple-quoted docstring
- Set `status:` to encoded, stub, or input
- Define each variable with proper entity, period, dtype, label, description
- For computable variables, include temporal formula blocks
- For input variables, include `default:` field
- Follow RAC DSL conventions (expression-based formulas, no `return` keyword,
  only literals -1, 0, 1, 2, 3 allowed)
"""

    def _extract_rac_content(self, llm_response: str) -> str | None:
        """Extract .rac file content from LLM response.

        Handles raw content, markdown-fenced responses, and CLI artifacts.
        """
        if not llm_response or not llm_response.strip():
            return None

        # Strip common CLI artifacts (timestamps, status lines, ANSI codes)
        cleaned = re.sub(r"\x1b\[[0-9;]*m", "", llm_response)  # ANSI codes

        # Try to extract from code fence (flexible: yaml, rac, or bare)
        fence_match = re.search(
            r"```(?:yaml|rac|text)?\s*\n(.*?)\n```", cleaned, re.DOTALL
        )
        if fence_match:
            content = fence_match.group(1).strip()
            if content:
                return content + "\n"

        # If response starts with # (raw .rac content), use as-is
        stripped = cleaned.strip()
        if stripped.startswith("#"):
            return stripped + "\n"

        # Look for the first line that starts with # (skip preamble)
        lines = stripped.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("#"):
                return "\n".join(lines[i:]).strip() + "\n"

        # Last resort: look for RAC structure keywords (status:, entity:, imports:)
        rac_keywords = ("status:", "entity:", "imports:", "period:", "dtype:")
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if any(stripped_line.startswith(kw) for kw in rac_keywords):
                return "\n".join(lines[i:]).strip() + "\n"

        return None

    # ========================================================================
    # Statute text fetching
    # ========================================================================

    def _fetch_statute_text(self, citation: str) -> str | None:
        """Fetch statute text from atlas or local USC XML."""
        # Try atlas first
        try:
            try:
                from atlas import Arch
            except ImportError:
                # Try adding atlas repo to path if installed locally
                atlas_src = self.atlas_path
                if (atlas_src / "atlas").is_dir():
                    import sys

                    sys.path.insert(0, str(atlas_src))
                    from atlas import Arch
                else:
                    raise

            db_path = self.atlas_path / "atlas.db"
            if db_path.exists():
                a = Arch(db_path=db_path)
                section = a.get(citation)
                if section and section.text:
                    print(
                        f"  Statute text loaded from atlas ({len(section.text)} chars)",
                        flush=True,
                    )
                    return section.text

                # Try stripping subsection — e.g. "26 USC 24(d)" → "26 USC 24"
                base_citation = re.sub(r"\([^)]+\)\s*$", "", citation).strip()
                if base_citation != citation:
                    section = a.get(base_citation)
                    if section and section.text:
                        print(
                            f"  Statute text loaded from atlas via {base_citation} ({len(section.text)} chars)",
                            flush=True,
                        )
                        return section.text
        except ImportError:
            print(
                "  Warning: atlas not installed, cannot fetch statute text", flush=True
            )
        except Exception as e:
            print(f"  Warning: atlas fetch failed: {e}", flush=True)

        # Fallback to legacy XML
        text = self._fetch_statute_text_legacy(citation)
        if text:
            print(f"  Statute text loaded from USC XML ({len(text)} chars)", flush=True)
        else:
            print(f"  Warning: no statute text found for {citation}", flush=True)
        return text

    def _fetch_statute_text_legacy(self, citation: str) -> str | None:
        """Fetch statute text from local USC XML."""
        import html as html_mod

        xml_path = self.atlas_path / "data" / "uscode"

        citation_clean = (
            citation.upper().replace("USC", "").replace("\u00a7", "").strip()
        )
        parts = re.split(r"[\s/]+", citation_clean)
        if len(parts) < 2:
            return None

        try:
            title = int(parts[0])
        except ValueError:
            return None
        section = parts[1]

        xml_file = xml_path / f"usc{title}.xml"
        if not xml_file.exists():
            return None

        try:
            content = xml_file.read_text()
        except Exception:
            return None

        identifier = f"/us/usc/t{title}/s{section}"
        start_pattern = rf'<section[^>]*identifier="{re.escape(identifier)}"[^>]*>'
        start_match = re.search(start_pattern, content)
        if not start_match:
            return None

        start_pos = start_match.start()
        depth = 0
        end_pos = start_pos
        i = start_pos
        while i < len(content):
            if content[i : i + 8] == "<section":
                depth += 1
            elif content[i : i + 10] == "</section>":
                depth -= 1
                if depth == 0:
                    end_pos = i + 10
                    break
            i += 1

        xml_section = content[start_pos:end_pos]

        text = re.sub(r"<[^>]+>", " ", xml_section)
        text = html_mod.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()

        return text if text else None

    # ========================================================================
    # Logging and reporting
    # ========================================================================

    def _format_oracle_summary(self, context: dict) -> str:
        """Format oracle context for reviewer prompts."""
        parts = []
        total = context.get("files_total", 0)
        untested = context.get("files_untested", total)
        if untested > 0:
            parts.append(
                f"WARNING: {untested}/{total} files had NO inline tests "
                f"and could not be validated against oracles."
            )
        if context.get("pe_match") is not None:
            parts.append(f"PE match: {context['pe_match']}%")
        else:
            parts.append("PE: UNTESTED (no inline tests with expected values)")
        if context.get("taxsim_match") is not None:
            parts.append(f"TAXSIM match: {context['taxsim_match']}%")
        else:
            parts.append("TAXSIM: UNTESTED (no inline tests)")
        if context.get("discrepancies"):
            parts.append(f"Discrepancies: {len(context['discrepancies'])}")
            for d in context["discrepancies"][:3]:
                parts.append(f"  - {d['description'][:100]}")
        return "\n".join(parts) if parts else "No oracle data available"

    def _sum_tokens(self, runs: List[AgentRun]) -> TokenUsage:
        """Sum token usage across all agent runs."""
        total = TokenUsage()
        for run in runs:
            if run.total_tokens:
                total.input_tokens += run.total_tokens.input_tokens
                total.output_tokens += run.total_tokens.output_tokens
                total.cache_read_tokens += run.total_tokens.cache_read_tokens
        return total

    def _sum_cost(self, runs: List[AgentRun]) -> float:
        """Sum costs across agent runs."""
        total = 0.0
        for run in runs:
            if run.total_cost is not None:
                total += run.total_cost
            elif run.total_tokens:
                total += run.total_tokens.estimated_cost_usd
        return total

    def _log_agent_run(self, session_id: str, agent_run: AgentRun) -> None:
        """Log a single agent run to the DB."""
        if not self.encoding_db:
            return

        self.encoding_db.log_event(
            session_id=session_id,
            event_type="agent_start",
            content=agent_run.prompt,
            metadata={
                "agent_type": agent_run.agent_type,
                "phase": agent_run.phase.value,
            },
        )

        phase_cost = (
            agent_run.total_cost
            if agent_run.total_cost is not None
            else (
                agent_run.total_tokens.estimated_cost_usd
                if agent_run.total_tokens
                else 0
            )
        )

        self.encoding_db.log_event(
            session_id=session_id,
            event_type="agent_end",
            content=agent_run.result if agent_run.result else "",
            metadata={
                "agent_type": agent_run.agent_type,
                "phase": agent_run.phase.value,
                "error": agent_run.error,
                "total_tokens": {
                    "input": agent_run.total_tokens.input_tokens,
                    "output": agent_run.total_tokens.output_tokens,
                }
                if agent_run.total_tokens
                else None,
                "cost_usd": phase_cost,
            },
        )

    def _log_to_db(self, run: EncodingRun) -> None:
        """Finalize session in DB with totals."""
        if not self.encoding_db:
            return

        if run.total_tokens:
            self.encoding_db.update_session_tokens(
                session_id=run.session_id,
                input_tokens=run.total_tokens.input_tokens,
                output_tokens=run.total_tokens.output_tokens,
                cache_read_tokens=run.total_tokens.cache_read_tokens,
            )

    def print_report(self, run: EncodingRun) -> str:
        """Generate human-readable report."""
        lines = [
            f"# Encoding Report: {run.citation}",
            f"Session: {run.session_id}",
            f"Duration: {(run.ended_at - run.started_at).total_seconds():.1f}s"
            if run.ended_at
            else "In progress",
            "",
            "## Oracle Match Rates",
            f"- PolicyEngine: {run.oracle_pe_match}%"
            if run.oracle_pe_match
            else "- PolicyEngine: N/A",
            f"- TAXSIM: {run.oracle_taxsim_match}%"
            if run.oracle_taxsim_match
            else "- TAXSIM: N/A",
            "",
            "## Files Created",
        ]
        for f in run.files_created:
            lines.append(f"- {f}")

        lines.extend(["", "## Agent Runs"])
        for agent_run in run.agent_runs:
            cost = agent_run.total_cost
            tokens = agent_run.total_tokens
            if cost is not None:
                lines.append(
                    f"- {agent_run.phase.value}: {agent_run.agent_type} (${cost:.4f})"
                )
            elif tokens:
                lines.append(
                    f"- {agent_run.phase.value}: {agent_run.agent_type} "
                    f"({tokens.total_tokens} tokens)"
                )
            else:
                lines.append(f"- {agent_run.phase.value}: {agent_run.agent_type}")

        if run.total_tokens:
            lines.extend(
                [
                    "",
                    "## Totals",
                    f"- Input tokens: {run.total_tokens.input_tokens:,}",
                    f"- Output tokens: {run.total_tokens.output_tokens:,}",
                    f"- Total cost: ${run.total_cost_usd:.4f}",
                ]
            )

        return "\n".join(lines)
