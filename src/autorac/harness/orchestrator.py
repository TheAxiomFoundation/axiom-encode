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
    oracle_pe_match: Optional[float] = None
    oracle_taxsim_match: Optional[float] = None
    discrepancies: List[dict] = field(default_factory=list)

    total_tokens: Optional[TokenUsage] = None
    total_cost_usd: float = 0.0
    autorac_version: str = ""


# Agent prompt mapping -- all embedded, no plugin needed
AGENT_PROMPTS = {
    "encoder": ENCODER_PROMPT,
    "validator": VALIDATOR_PROMPT,
    "rac_reviewer": RAC_REVIEWER_PROMPT,
    "formula_reviewer": FORMULA_REVIEWER_PROMPT,
    "parameter_reviewer": PARAMETER_REVIEWER_PROMPT,
    "integration_reviewer": INTEGRATION_REVIEWER_PROMPT,
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
    ):
        """Initialize the orchestrator.

        Args:
            model: Model to use. For CLI backend, short names like "opus".
                   For API backend, full model IDs like "claude-opus-4-6".
            db_path: Path to encoding database. None to skip logging.
            backend: "cli" for Claude Code CLI, "api" for direct API calls.
            api_key: Anthropic API key (required for API backend).
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

        # Cache env without CLAUDECODE (prevents nested launch errors)
        self._cli_env = (
            {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
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

        cmd = ["claude", "--print", "--model", self.model, "-p", prompt]

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
            oracle_context["discrepancies"] = [
                {"description": issue} for issue in all_issues[:10]
            ]
        else:
            print("  No RAC files found to validate", flush=True)

        duration = time.time() - oracle_start
        print(
            f"  DONE: PE={oracle_context.get('pe_match', 'N/A')}%, "
            f"TAXSIM={oracle_context.get('taxsim_match', 'N/A')}% ({duration:.1f}s)",
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

Write .rac files to the output path. Run tests after each file.
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

        return all_runs

    def _build_subsection_prompt(
        self,
        task: SubsectionTask,
        citation: str,
        output_path: Path,
        statute_text: str | None = None,
    ) -> str:
        """Build a focused encoding prompt for a single subsection."""
        parts = [
            f"Encode {citation} subsection ({task.subsection_id}) - "
            f'"{task.title}" into RAC format.',
            "",
            f"Output: {output_path / task.file_name}",
            f"Scope: ONLY this subsection. One file: {task.file_name}",
            "",
            "## CRITICAL RULES:",
            "",
            "1. **FILEPATH = CITATION** - File names MUST be subsection names",
            "2. **One subsection per file**",
            "3. **Only statute values** - No indexed/derived/computed values",
            "",
            "Write the .rac file to the output path. Run tests after writing.",
        ]

        if task.dependencies:
            dep_files = ", ".join(f"{d}.rac" for d in task.dependencies)
            parts.append("")
            parts.append(
                f"Note: This subsection depends on {dep_files} "
                f"(already encoded). You may import from those files."
            )

        if statute_text:
            parts.append("")
            parts.append(f"Full statute text (excerpt):\n{statute_text[:5000]}")

        parts.append(DSL_CHEATSHEET)
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
    # Statute text fetching
    # ========================================================================

    def _fetch_statute_text(self, citation: str) -> str | None:
        """Fetch statute text from atlas or local USC XML."""
        # Try atlas first
        try:
            from atlas import Arch

            db_path = Path.home() / "RulesFoundation" / "atlas" / "atlas.db"
            if db_path.exists():
                a = Arch(db_path=db_path)
                section = a.get(citation)
                if section and section.text:
                    return section.text
        except ImportError:
            pass

        # Fallback to legacy XML
        return self._fetch_statute_text_legacy(citation)

    def _fetch_statute_text_legacy(self, citation: str) -> str | None:
        """Fetch statute text from local USC XML."""
        import html as html_mod

        xml_path = Path.home() / "RulesFoundation" / "atlas" / "data" / "uscode"

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
        if context.get("pe_match") is not None:
            parts.append(f"PE match: {context['pe_match']}%")
        if context.get("taxsim_match") is not None:
            parts.append(f"TAXSIM match: {context['taxsim_match']}%")
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
            content=agent_run.prompt[:2000],
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
            content=agent_run.result[:2000] if agent_run.result else "",
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
