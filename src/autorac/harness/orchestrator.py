"""
Encoding Orchestrator -- self-contained pipeline for statute encoding.

Replaces the plugin-dependent SDKOrchestrator and encoding-orchestrator agent.
All prompts are embedded -- `pip install autorac` is sufficient.

Supports three backends:
- Claude Code CLI (subprocess) -- works with Max subscription
- Claude API (via anthropic SDK) -- works on Modal or any server
- OpenAI Responses API (direct HTTPS) -- captures reasoning summaries and usage

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
from typing import Any, List, Optional

from autorac.constants import DEFAULT_CLI_MODEL, DEFAULT_MODEL, DEFAULT_OPENAI_MODEL
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
from autorac.statute import find_citation_text, parse_usc_citation

from .backends import parse_claude_cli_json_output
from .dependency_stubs import (
    build_registered_stub_content,
    find_ingested_source_artifacts,
    find_registered_stub_specs,
)
from .encoding_db import EncodingDB, TokenUsage
from .observability import emit_agent_run, extract_reasoning_entries
from .pricing import estimate_usage_cost_usd
from .validator_pipeline import (
    ValidatorPipeline,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_numbers_from_text,
)


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
    OPENAI = "openai"  # OpenAI Responses API


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
    provider_trace: Optional[dict[str, Any]] = None


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


PROVENANCE_VALUE_METADATA_KEYS = {
    "imports",
    "status",
    "description",
    "label",
    "entity",
    "period",
    "dtype",
    "unit",
    "indexed_by",
    "tests",
}


def _summarize_agent_message(content: str) -> str:
    """Generate a concise summary for a provider response."""
    if not content:
        return "Empty response"

    if "STRUCTURED_OUTPUT" in content:
        return "Produced structured subsection plan"
    if '"score"' in content and '"passed"' in content:
        return "Produced structured review result"
    if "```" in content:
        return f"Code-heavy response ({len(content):,} chars)"
    if "Error" in content or "error" in content:
        return "Reported an error"
    if "Created" in content or "Successfully" in content:
        return "Completed successfully"

    lines = [
        line.strip()
        for line in content.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return f"Response ({len(content):,} chars)"

    first = lines[0][:120]
    return first + ("..." if len(lines[0]) > 120 else "")


def _build_cli_trace_command(model: str) -> list[str]:
    """Describe the CLI invocation without duplicating the full prompt."""
    return [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--model",
        model,
        "--mcp-config",
        "<omitted>",
        "--strict-mcp-config",
        "-p",
        "<prompt omitted>",
    ]


STUB_GENERATOR_PROMPT = """You generate .rac stub files for external dependencies.

You categorize variables into three types:
1. **Computable from statute** (status: encoded) — the statute defines a clear formula
2. **Stub** (status: stub) — statutory definition exists but full encoding needs more context
3. **Input** (status: input) — an observable real-world fact no statute defines (wages, age, etc.)

Always output raw .rac content with no markdown fencing or explanation.

Stub files must use the same RAC DSL as normal section files:
- Do NOT use a `variable:` keyword
- Put `status:` immediately after the docstring for file-level stubs
- Use normal definition blocks like `some_name:`
- Use `entity: Payment|TaxUnit|Person|Household|Family`, `period: Year|Month|Day`, `dtype: Money|Boolean|Integer|Rate|String`
- Quote `label:` and `description:` strings
- Use 4 spaces for fields under a definition
- Never emit lowercase schema names like `tax_unit` or `taxable_year`"""


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

### Required file shape
- Every `.rac` file MUST include `status:` explicitly
- Put `status:` immediately after the docstring when a docstring exists
- Use 4 spaces for fields under each definition
- Use 8 spaces for expressions nested under `from yyyy-mm-dd:`
- Quote all `description:` and `label:` string values
- Do NOT emit generic 2-space YAML indentation

### Declaration types
- `name:` with `from yyyy-mm-dd:` -- policy value with temporal entries
- `name:` with `from yyyy-mm-dd:` expression -- computed value
- `name:` with `default:` -- user-provided input
- `enum Name:` -- enumeration with `values:` list

### Fields (all required unless noted)
- `entity:` Payment | Person | TaxUnit | Household | Family
- `period:` Year | Month | Day
- `dtype:` Money | Rate | Boolean | Integer | String | Enum[Name]
- `imports:` -- list of `path#name` (optional)

### Temporal syntax
```yaml
ctc_base_amount:
  from 2018-01-01: 2000
  from 2025-01-01: 2500
```

### Formula syntax
**Write formulas as temporal expressions after `from yyyy-mm-dd:`**
**Allowed:** `if`/`elif`/`else`, `and`/`or`/`not`, `=` assignment, final expression value
**Operators:** `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`
**Built-in functions:** `min(a,b)`, `max(a,b)`, `abs(x)`, `floor(x)`, `ceil(x)`, `round(x,n)`, `clamp(x,lo,hi)`, `sum(...)`, `len(...)`
**FORBIDDEN:** `return`, `formula: |`, `for`/`while` loops, list comprehensions, list literals (`[...]`), membership tests with `in`, `def`/`lambda`, `try`/`except`, `+=`, imports
**Numeric literals:** ONLY -1, 0, 1, 2, 3 allowed. ALL other numbers must come from named definitions.

### Conditional shape rules
- Use repo-style conditional expressions like `if cond: value`, `elif cond: value`, `else: value`
- A conditional block may return a value directly; it must NOT assign helper variables inside branches
- Straight-line assignments before the final expression are allowed
- If a branch only selects among upstream values, make that selector its own variable or imported dependency

Valid:
```yaml
threshold_amount:
    from 2021-01-01:
        if is_joint_return: threshold_joint
        elif is_head_of_household: threshold_head_of_household
        else: threshold_other
```

Invalid:
```yaml
threshold_amount:
    from 2021-01-01:
        if is_joint_return:
            selected_threshold = threshold_joint
        elif is_head_of_household:
            selected_threshold = threshold_head_of_household
        else:
            selected_threshold = threshold_other
        selected_threshold
```

Use explicit enum comparisons like `status == joint or status == surviving_spouse`.
Do NOT write `status in [joint, surviving_spouse]`.
"""

MAX_COMPILE_REPAIR_ATTEMPTS = 2


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
                   For Anthropic API backend, full model IDs like "claude-opus-4-6".
                   For OpenAI backend, use Responses-compatible models like "gpt-5.4".
            db_path: Path to encoding database. None to skip logging.
            backend: "cli" for Claude Code CLI, "api" for Anthropic API,
                     "openai" for OpenAI Responses API.
            api_key: Provider API key (Anthropic for "api", OpenAI for "openai").
            atlas_path: Path to atlas repo. Falls back to ATLAS_PATH env var,
                        then ~/RulesFoundation/atlas.
        """
        if isinstance(backend, str):
            backend = Backend(backend)

        self.backend = backend
        if self.backend == Backend.API:
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        elif self.backend == Backend.OPENAI:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        else:
            self.api_key = api_key

        if self.backend == Backend.API and not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required for API backend")
        if self.backend == Backend.OPENAI and not self.api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI backend")

        if self.backend == Backend.CLI:
            self.model = model or DEFAULT_CLI_MODEL
        elif self.backend == Backend.OPENAI:
            self.model = model or DEFAULT_OPENAI_MODEL
        else:
            self.model = model or DEFAULT_MODEL

        self.encoding_db = EncodingDB(db_path) if db_path else None

        # Resolve atlas path: explicit > discovered checkout
        if atlas_path:
            self.atlas_path = Path(atlas_path)
        else:
            self.atlas_path = self._resolve_atlas_repo_path()
        self.rac_repo_path = self._resolve_rac_repo_path()

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
        self._logged_artifact_paths: dict[str, set[str]] = {}
        self._current_citation: str | None = None

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

        self._current_citation = citation

        try:
            # Create DB session
            if self.encoding_db:
                self.encoding_db.start_session(
                    model=self.model,
                    cwd=str(Path.cwd()),
                    session_id=run.session_id,
                    autorac_version=run.autorac_version,
                )
            self._logged_artifact_paths[run.session_id] = set()

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
            self._log_analysis_provenance(
                run.session_id, citation, analysis.result or ""
            )

            # Phase 2: Encoding
            if analysis.result:
                encoding_runs = await self._run_encoding_parallel(
                    citation,
                    output_path,
                    statute_text,
                    analysis.result,
                    session_id=run.session_id,
                )
                for enc_run in encoding_runs:
                    run.agent_runs.append(enc_run)

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
                    fallback_file = self._resolve_aggregator_output_file(
                        citation, output_path
                    )
                    self._materialize_agent_artifact(encoding, fallback_file)
                    self._compile_existing_artifacts(
                        run.session_id,
                        output_path,
                        Phase.ENCODING,
                    )
                    self._log_provenance_decision(
                        run.session_id,
                        "Falling back to single-agent encoding",
                        [
                            f"Citation: {citation}",
                            "Analyzer did not yield structured subsection tasks.",
                            f"Output path: {output_path}",
                        ],
                        Phase.ENCODING,
                    )
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
                fallback_file = self._resolve_aggregator_output_file(
                    citation, output_path
                )
                self._materialize_agent_artifact(encoding, fallback_file)
                self._compile_existing_artifacts(
                    run.session_id,
                    output_path,
                    Phase.ENCODING,
                )
                self._log_provenance_decision(
                    run.session_id,
                    "Encoding proceeded without analyzer output",
                    [
                        f"Citation: {citation}",
                        "Analyzer returned no result text.",
                        f"Output path: {output_path}",
                    ],
                    Phase.ENCODING,
                )

            # Check created files
            if output_path.exists():
                run.files_created = [str(f) for f in output_path.rglob("*.rac")]
                self._log_artifact_provenance_records(
                    run.session_id,
                    [Path(f) for f in run.files_created],
                    Phase.ENCODING,
                    fallback_source_text=statute_text,
                )

            # Phase 2.5: Resolve external dependencies (create stubs)
            stubs = await self._resolve_external_dependencies(
                output_path, session_id=run.session_id
            )
            if stubs:
                run.stubs_created = [str(s) for s in stubs]
                # Re-scan to include stubs in files_created
                run.files_created = [str(f) for f in output_path.rglob("*.rac")]
                # Also include stubs outside the output_path
                for stub_path in stubs:
                    stub_str = str(stub_path)
                    if stub_str not in run.files_created:
                        run.files_created.append(stub_str)
                self._log_artifact_provenance_records(
                    run.session_id,
                    stubs,
                    Phase.RESOLVE_EXTERNALS,
                )

            # Phase 3: Oracle validation
            oracle_context = await self._run_oracle_validation(
                output_path, session_id=run.session_id
            )
            run.oracle_pe_match = oracle_context.get("pe_match")
            run.oracle_taxsim_match = oracle_context.get("taxsim_match")
            run.discrepancies = oracle_context.get("discrepancies", [])

            # Phase 4: LLM Review (parallel)
            oracle_summary = self._format_oracle_summary(oracle_context)
            review_runs = await self._run_reviews_parallel(
                citation,
                oracle_summary,
                [Path(file_path) for file_path in run.files_created],
            )
            for rev_run in review_runs:
                run.agent_runs.append(rev_run)
                self._log_agent_run(run.session_id, rev_run)
                self._log_review_provenance(run.session_id, rev_run)

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
        finally:
            if self.encoding_db:
                self.encoding_db.end_session(run.session_id)
            self._logged_artifact_paths.pop(run.session_id, None)
            self._current_citation = None

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
            elif self.backend == Backend.OPENAI:
                result = await self._run_via_openai_responses(
                    full_prompt, system_prompt, prompt
                )
            else:
                result = await self._run_via_api(full_prompt, system_prompt, prompt)

            agent_run.result = result.get("text", "")
            agent_run.total_tokens = result.get("tokens")
            agent_run.total_cost = result.get("cost")
            agent_run.provider_trace = result.get("trace")
            if agent_run.result:
                agent_run.messages.append(
                    AgentMessage(
                        role="assistant",
                        content=agent_run.result,
                        tokens=agent_run.total_tokens,
                        summary=_summarize_agent_message(agent_run.result),
                    )
                )

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
            "--output-format",
            "json",
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
            parsed = parse_claude_cli_json_output(text, self.model)
            return {
                "text": parsed["text"] if parsed else text,
                "tokens": parsed["tokens"] if parsed else None,
                "cost": parsed["cost_usd"] if parsed else None,
                "trace": (
                    parsed["trace"]
                    if parsed
                    else {
                        "provider": "anthropic",
                        "backend": Backend.CLI.value,
                        "model": self.model,
                        "command": _build_cli_trace_command(self.model),
                        "returncode": result.returncode,
                        "raw_output": text,
                    }
                ),
            }
        except subprocess.TimeoutExpired:
            timeout_text = "Timeout after 600s"
            return {
                "text": timeout_text,
                "trace": {
                    "provider": "anthropic",
                    "backend": Backend.CLI.value,
                    "model": self.model,
                    "command": _build_cli_trace_command(self.model),
                    "timeout_seconds": 600,
                    "raw_output": timeout_text,
                },
            }
        except FileNotFoundError:
            not_found_text = "Claude CLI not found - install with: npm install -g @anthropic-ai/claude-code"
            return {
                "text": not_found_text,
                "trace": {
                    "provider": "anthropic",
                    "backend": Backend.CLI.value,
                    "model": self.model,
                    "command": _build_cli_trace_command(self.model),
                    "raw_output": not_found_text,
                },
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
            response_blocks = []
            for block in response.content:
                block_type = getattr(block, "type", block.__class__.__name__)
                block_payload: dict[str, Any] = {"type": block_type}
                for attr in ("id", "name", "text", "thinking", "signature"):
                    value = getattr(block, attr, None)
                    if value is not None:
                        block_payload[attr] = value
                block_input = getattr(block, "input", None)
                if block_input is not None:
                    block_payload["input"] = block_input
                response_blocks.append(block_payload)
                if hasattr(block, "text"):
                    text += block.text

            tokens = TokenUsage(
                input_tokens=int(getattr(response.usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(response.usage, "output_tokens", 0) or 0),
                cache_read_tokens=int(
                    getattr(response.usage, "cache_read_input_tokens", 0) or 0
                ),
                cache_creation_tokens=int(
                    getattr(response.usage, "cache_creation_input_tokens", 0) or 0
                ),
            )

            return {
                "text": text,
                "tokens": tokens,
                "cost": estimate_usage_cost_usd(self.model, tokens),
                "trace": {
                    "provider": "anthropic",
                    "backend": Backend.API.value,
                    "model": self.model,
                    "system_prompt": system_prompt or None,
                    "user_prompt": user_prompt,
                    "response_blocks": response_blocks,
                },
            }

        except ImportError:
            raise ImportError("anthropic SDK not installed. Run: pip install anthropic")

    async def _run_via_openai_responses(
        self, full_prompt: str, system_prompt: str, user_prompt: str
    ) -> dict:
        """Run via the OpenAI Responses API over HTTPS."""
        import requests

        body: dict[str, Any] = {
            "model": self.model,
            "input": user_prompt if system_prompt else full_prompt,
            "max_output_tokens": 16384,
            "reasoning": {
                "effort": "low",
                "summary": "auto",
            },
        }
        if system_prompt:
            body["instructions"] = system_prompt

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        def _post_response() -> requests.Response:
            return requests.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=body,
                timeout=600,
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _post_response)

        request_id = response.headers.get("x-request-id")
        try:
            payload = response.json()
        except ValueError:
            payload = {
                "error": {
                    "message": response.text or f"HTTP {response.status_code}",
                }
            }

        if response.status_code >= 400:
            error = payload.get("error") or {}
            message = error.get("message") or response.text or "OpenAI request failed"
            return {
                "text": message,
                "trace": {
                    "provider": "openai",
                    "backend": Backend.OPENAI.value,
                    "model": self.model,
                    "request_id": request_id,
                    "request_body": body,
                    "json_result": payload,
                    "status_code": response.status_code,
                },
            }

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

        text = self._extract_openai_response_text(payload)

        return {
            "text": text,
            "tokens": tokens,
            "cost": estimate_usage_cost_usd(self.model, tokens),
            "trace": {
                "provider": "openai",
                "backend": Backend.OPENAI.value,
                "model": self.model,
                "request_id": request_id,
                "request_body": body,
                "json_result": payload,
                "status_code": response.status_code,
            },
        }

    async def _run_oracle_validation(
        self, output_path: Path, session_id: str | None = None
    ) -> dict:
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
                rac_path=self.rac_repo_path,
                enable_oracles=True,
                max_workers=2,
            )

            pe_scores = []
            taxsim_scores = []
            all_issues = []

            for rac_file in rac_files:
                print(f"  Validating: {rac_file.name}", flush=True)
                file_issues: list[str] = []
                pe_score = None
                taxsim_score = None
                try:
                    pe_result = pipeline._run_policyengine(rac_file)
                    if pe_result.score is not None:
                        pe_scores.append(pe_result.score)
                        pe_score = pe_result.score
                        print(f"    PE: {pe_result.score:.1%}", flush=True)
                    all_issues.extend(pe_result.issues)
                    file_issues.extend(pe_result.issues)

                    taxsim_result = pipeline._run_taxsim(rac_file)
                    if taxsim_result.score is not None:
                        taxsim_scores.append(taxsim_result.score)
                        taxsim_score = taxsim_result.score
                        print(f"    TAXSIM: {taxsim_result.score:.1%}", flush=True)
                    all_issues.extend(taxsim_result.issues)
                    file_issues.extend(taxsim_result.issues)
                except Exception as e:
                    print(f"    Error: {e}", flush=True)
                    all_issues.append(str(e))
                    file_issues.append(str(e))

                if session_id:
                    self._log_validation_provenance(
                        session_id,
                        rac_file,
                        pe_score,
                        taxsim_score,
                        file_issues,
                    )

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
        if session_id:
            self._log_session_event(
                session_id=session_id,
                event_type="provenance_validation",
                content=(
                    "Validation summary\n"
                    f"PolicyEngine aggregate: {pe_str}\n"
                    f"TAXSIM aggregate: {taxsim_str}\n"
                    f"Files tested: {oracle_context.get('files_tested', 0)} / {total}\n"
                    f"Discrepancies captured: {len(oracle_context.get('discrepancies', []))}"
                ),
                metadata={
                    "phase": Phase.ORACLE.value,
                    "aggregate": True,
                    "oracle_context": oracle_context,
                },
            )

        return oracle_context

    async def _run_reviews_parallel(
        self, citation: str, oracle_summary: str, rac_files: list[Path]
    ) -> List[AgentRun]:
        """Run all 4 reviewers in parallel."""
        review_context = self._build_review_file_context(citation, rac_files)
        reviewers = [
            (
                "rac_reviewer",
                get_rac_reviewer_prompt(citation, oracle_summary, review_context),
            ),
            (
                "formula_reviewer",
                get_formula_reviewer_prompt(citation, oracle_summary, review_context),
            ),
            (
                "parameter_reviewer",
                get_parameter_reviewer_prompt(citation, oracle_summary, review_context),
            ),
            (
                "integration_reviewer",
                get_integration_reviewer_prompt(
                    citation, oracle_summary, review_context
                ),
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

    def _build_review_file_context(
        self, citation: str, rac_files: list[Path], statute_text: str | None = None
    ) -> str:
        """Inline the actual RAC artifacts for API reviewers."""
        files = []
        seen = set()
        for rac_file in rac_files:
            if not rac_file.exists() or rac_file.suffix != ".rac":
                continue
            path_key = str(rac_file.resolve())
            if path_key in seen:
                continue
            seen.add(path_key)
            files.append(rac_file)

        if not files:
            return ""

        if statute_text is None:
            statute_text = self._fetch_statute_text(citation)

        parts = [
            "## Files Under Review",
            "Use the exact file contents below. Do not say the file path or file contents were missing.",
        ]
        if statute_text:
            statute_preview = statute_text[:8000]
            if len(statute_text) > 8000:
                statute_preview += "\n... [truncated]"
            parts.extend(
                [
                    "",
                    "## Authoritative Statute Text",
                    "```text",
                    statute_preview,
                    "```",
                ]
            )

        for rac_file in sorted(files):
            try:
                content = rac_file.read_text()
            except Exception:
                continue
            if len(content) > 12000:
                content = content[:12000] + "\n... [truncated]"
            parts.extend(
                [
                    "",
                    f"## File: {self._artifact_relative_path(rac_file)}",
                    f"Absolute path: {rac_file}",
                    "```rac",
                    content,
                    "```",
                ]
            )
            test_file = rac_file.with_suffix(".rac.test")
            if test_file.exists():
                try:
                    test_content = test_file.read_text()
                except Exception:
                    test_content = None
                if test_content is not None:
                    if len(test_content) > 12000:
                        test_content = test_content[:12000] + "\n... [truncated]"
                    parts.extend(
                        [
                            "",
                            f"## Companion Test File: {test_file.name}",
                            "```yaml",
                            test_content,
                            "```",
                        ]
                    )

        return "\n".join(parts)

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
        target_file = self._resolve_aggregator_output_file(citation, output_path)
        test_file = target_file.with_suffix(".rac.test")
        output_instructions = (
            "Write .rac files to the output path. Run `python -m rac.test_runner <file>` after each file."
            if self.backend == Backend.CLI
            else (
                "Return EXACTLY two files with no markdown fences and no explanation:\n"
                f"=== FILE: {target_file.name} ===\n"
                "<raw .rac content>\n"
                f"=== FILE: {test_file.name} ===\n"
                "<raw .rac.test YAML>\n"
                "The orchestrator will write both files."
            )
        )
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
8. **CROSS-STATUTE DEFINITIONS MUST BE IMPORTED** - If the source text says a term is defined in another section, or relies on another section's definition/eligibility rule, import that upstream definition or predicate. Do NOT restate it locally or invent a leaf-local stand-in. If the cited upstream file is missing, still emit the best import path so the external-stub workflow can create it. If the source text only implies a shared concept, import an existing canonical nearby concept only when one already exists; otherwise keep the helper local to this leaf.

{output_instructions}
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
        session_id: str | None = None,
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
            wave_failures: list[str] = []

            async def encode_one(
                task: SubsectionTask,
            ) -> tuple[SubsectionTask, AgentRun]:
                async with semaphore:
                    prompt = self._build_subsection_prompt(
                        task, citation, output_path, statute_text
                    )
                    run = await self._run_agent("encoder", prompt, Phase.ENCODING)
                    return task, run

            pending = [asyncio.create_task(encode_one(task)) for task in wave]
            for completed in asyncio.as_completed(pending):
                try:
                    task, run = await completed
                except Exception as exc:
                    print(f"  FAILED: {exc}", flush=True)
                    wave_failures.append(str(exc))
                    continue

                all_runs.append(run)
                if not session_id:
                    continue

                self._log_agent_run(session_id, run)
                expected_file = self._resolve_task_output_file(
                    output_path, task.file_name
                )
                self._materialize_agent_artifact(run, expected_file)
                if not expected_file.exists():
                    issue = (
                        f"Expected output file missing after subsection "
                        f"({task.subsection_id}): {expected_file}"
                    )
                    self._log_provenance_decision(
                        session_id,
                        "Encoder did not create the expected artifact",
                        [issue],
                        Phase.ENCODING,
                        metadata={
                            "subsection_id": task.subsection_id,
                            "expected_file": str(expected_file),
                        },
                    )
                    wave_failures.append(issue)
                    continue

                self._log_artifact_provenance_records(
                    session_id,
                    [expected_file],
                    Phase.ENCODING,
                    fallback_source_text=statute_text,
                )
                compile_result = self._run_compile_gate(expected_file)
                self._log_compile_validation(
                    session_id,
                    expected_file,
                    Phase.ENCODING,
                    compile_result.passed,
                    compile_result.issues,
                    compile_result.raw_output,
                )
                repair_attempts = 0
                while (
                    not compile_result.passed
                    and repair_attempts < MAX_COMPILE_REPAIR_ATTEMPTS
                ):
                    repair_attempts += 1
                    self._log_provenance_decision(
                        session_id,
                        f"Compile repair attempt {repair_attempts} for subsection ({task.subsection_id})",
                        compile_result.issues
                        or [compile_result.error or "compile failed"],
                        Phase.ENCODING,
                        metadata={
                            "subsection_id": task.subsection_id,
                            "artifact_path": str(expected_file),
                            "attempt": repair_attempts,
                        },
                    )
                    repair_prompt = self._build_compile_fix_prompt(
                        task=task,
                        citation=citation,
                        output_path=output_path,
                        statute_text=statute_text,
                        artifact_path=expected_file,
                        compile_issues=compile_result.issues
                        or [compile_result.error or "compile failed"],
                    )
                    repair_run = await self._run_agent(
                        "encoder", repair_prompt, Phase.ENCODING
                    )
                    all_runs.append(repair_run)
                    self._log_agent_run(session_id, repair_run)
                    self._materialize_agent_artifact(repair_run, expected_file)
                    self._log_artifact_provenance_records(
                        session_id,
                        [expected_file],
                        Phase.ENCODING,
                        fallback_source_text=statute_text,
                    )
                    compile_result = self._run_compile_gate(expected_file)
                    self._log_compile_validation(
                        session_id,
                        expected_file,
                        Phase.ENCODING,
                        compile_result.passed,
                        compile_result.issues,
                        compile_result.raw_output,
                    )
                if not compile_result.passed:
                    wave_failures.extend(
                        f"{expected_file}: {issue}"
                        for issue in (
                            compile_result.issues
                            or [compile_result.error or "compile failed"]
                        )
                    )

            if wave_failures:
                raise RuntimeError(
                    "Wave compile gate failed:\n" + "\n".join(wave_failures[:10])
                )

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
            if session_id:
                self._log_agent_run(session_id, aggregator_run)
                aggregator_file = self._resolve_aggregator_output_file(
                    citation, output_path
                )
                self._materialize_agent_artifact(aggregator_run, aggregator_file)
                if not aggregator_file.exists():
                    raise RuntimeError(
                        f"Aggregator did not create expected file: {aggregator_file}"
                    )
                self._log_artifact_provenance_records(
                    session_id,
                    [aggregator_file],
                    Phase.ENCODING,
                    fallback_source_text=statute_text,
                )
                compile_result = self._run_compile_gate(aggregator_file)
                self._log_compile_validation(
                    session_id,
                    aggregator_file,
                    Phase.ENCODING,
                    compile_result.passed,
                    compile_result.issues,
                    compile_result.raw_output,
                )
                if not compile_result.passed:
                    raise RuntimeError(
                        "Aggregator compile gate failed:\n"
                        + "\n".join(
                            compile_result.issues
                            or [compile_result.error or "compile failed"]
                        )
                    )

        return all_runs

    def _build_subsection_prompt(
        self,
        task: SubsectionTask,
        citation: str,
        output_path: Path,
        statute_text: str | None = None,
    ) -> str:
        """Build a focused encoding prompt for a single subsection."""
        file_name = str(
            self._resolve_task_output_file(output_path, task.file_name).relative_to(
                output_path
            )
        )

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
            "8. **CROSS-STATUTE DEFINITIONS MUST BE IMPORTED** - If the source "
            "text says a term is defined in another section, or relies on another "
            "section's definition/eligibility rule, import that upstream definition "
            "or predicate. Do NOT restate it locally or invent a leaf-local stand-in. "
            "If the cited upstream file is missing, still emit the best import path "
            "so the external-stub workflow can create it. If the source text only "
            "implies a shared concept, import an existing canonical nearby concept "
            "only when one already exists; otherwise keep the helper local to this leaf.",
            "",
            (
                "Write the .rac file to the output path. Run `python -m rac.test_runner <file>` after writing."
                if self.backend == Backend.CLI
                else (
                    "Return EXACTLY two files with no markdown fences and no explanation:\n"
                    f"=== FILE: {Path(file_name).name} ===\n"
                    "<raw .rac content>\n"
                    f"=== FILE: {Path(file_name).with_suffix('.rac.test').name} ===\n"
                    "<raw .rac.test YAML>\n"
                    "The orchestrator will write both files."
                )
            ),
        ]

        if task.dependencies:
            dep_files = ", ".join(f"{d}.rac" for d in task.dependencies)
            parts.append("")
            parts.append(
                f"Note: This subsection depends on {dep_files} "
                f"(already encoded). You may import from those files."
            )

        if statute_text:
            subsection_text = self._lookup_precise_subsection_text(
                citation, task.subsection_id
            ) or self._extract_subsection_text(statute_text, task.subsection_id)
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

    def _citation_for_subsection(self, citation: str, subsection_id: str) -> str | None:
        """Append subsection fragments to a USC citation."""
        if "USC" not in citation.upper() or not subsection_id:
            return None

        try:
            parts = parse_usc_citation(citation)
        except ValueError:
            return None

        subsection_fragments = [
            fragment.strip("()")
            for fragment in subsection_id.strip("/").split("/")
            if fragment.strip("()")
        ]
        if not subsection_fragments:
            return citation

        base_fragments = list(parts.fragments)
        if (
            base_fragments
            and subsection_fragments[: len(base_fragments)] == base_fragments
        ):
            all_fragments = subsection_fragments
        else:
            all_fragments = base_fragments + subsection_fragments

        return f"{parts.title} USC {parts.section}" + "".join(
            f"({fragment})" for fragment in all_fragments
        )

    def _lookup_precise_subsection_text(
        self, citation: str, subsection_id: str
    ) -> str | None:
        """Fetch exact subsection text from USC XML when available."""
        subsection_citation = self._citation_for_subsection(citation, subsection_id)
        if not subsection_citation:
            return None

        xml_root = self.atlas_path / "data" / "uscode"
        if not xml_root.exists():
            return None

        try:
            return find_citation_text(subsection_citation, xml_root)
        except Exception:
            return None

    def _build_compile_fix_prompt(
        self,
        task: SubsectionTask,
        citation: str,
        output_path: Path,
        statute_text: str | None,
        artifact_path: Path,
        compile_issues: list[str],
    ) -> str:
        """Build a targeted repair prompt for a compile-failing subsection."""
        relative_file = artifact_path.relative_to(output_path)
        test_path = artifact_path.with_suffix(".rac.test")
        precise_text = self._lookup_precise_subsection_text(
            citation, task.subsection_id
        ) or (
            self._extract_subsection_text(statute_text, task.subsection_id)
            if statute_text
            else None
        )

        if self.backend == Backend.CLI:
            output_instructions = (
                "Rewrite the files in place and run `autorac compile <file>` again "
                "before returning."
            )
        else:
            output_instructions = (
                "Return EXACTLY two files with no markdown fences and no explanation:\n"
                f"=== FILE: {relative_file.name} ===\n"
                "<corrected .rac content>\n"
                f"=== FILE: {test_path.name} ===\n"
                "<corrected .rac.test YAML>\n"
                "The orchestrator will write both files."
            )

        prompt_parts = [
            f"Fix the compile failure for {citation} subsection ({task.subsection_id}).",
            "",
            f"Output: {artifact_path}",
            f"Scope: ONLY this subsection. Keep the same files: {relative_file} and {test_path.name}",
            "",
            "## Compile Errors",
            *[f"- {issue}" for issue in compile_issues],
            "",
            "## Repair Requirements",
            "- Preserve the same legal scope and file names",
            "- Keep only literals grounded in the source text",
            "- Quote all `description:` and `label:` values",
            "- Do NOT use list literals or `in` membership tests; compare enum values explicitly with `==` and `or`",
            "- Do NOT emit Python-style branch-local assignment blocks inside conditionals",
            output_instructions,
        ]

        if precise_text:
            prompt_parts.extend(
                [
                    "",
                    f"## Exact statute text for subsection ({task.subsection_id})",
                    precise_text,
                ]
            )
        elif statute_text:
            prompt_parts.extend(
                ["", "## Full statute text (excerpt)", statute_text[:5000]]
            )

        prompt_parts.extend(
            [
                "",
                f"=== FILE: {relative_file.name} ===",
                artifact_path.read_text() if artifact_path.exists() else "",
                f"=== FILE: {test_path.name} ===",
                test_path.read_text() if test_path.exists() else "",
                "",
                DSL_CHEATSHEET,
            ]
        )
        if self._context_section:
            prompt_parts.extend(["", self._context_section])

        return "\n".join(prompt_parts)

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
            (
                "Write the root `.rac` file to the output path."
                if self.backend == Backend.CLI
                else (
                    "Return EXACTLY two files with no markdown fences and no explanation:\n"
                    f"=== FILE: {root_file} ===\n"
                    "<raw .rac content>\n"
                    f"=== FILE: {Path(root_file).with_suffix('.rac.test').name} ===\n"
                    "<raw .rac.test YAML>\n"
                    "The orchestrator will write both files."
                )
            ),
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

    def _resolve_rac_repo_path(self) -> Path:
        """Find the rac repo checkout used for compile/test validation."""
        candidates = []
        if os.environ.get("RAC_PATH"):
            candidates.append(Path(os.environ["RAC_PATH"]))
        candidates.extend(
            [
                Path(__file__).resolve().parents[4] / "rac",
                Path.home() / "RulesFoundation" / "rac",
            ]
        )

        for candidate in candidates:
            if (candidate / "src" / "rac").exists():
                return candidate

        return candidates[0]

    def _resolve_atlas_repo_path(self) -> Path:
        """Find the atlas repo checkout used for statute lookup."""
        candidates = []
        if os.environ.get("ATLAS_PATH"):
            candidates.append(Path(os.environ["ATLAS_PATH"]))
        candidates.extend(
            [
                Path(__file__).resolve().parents[4] / "atlas",
                Path.home() / "RulesFoundation" / "atlas",
            ]
        )

        for candidate in candidates:
            if (
                (candidate / "src" / "atlas").exists()
                or (candidate / "atlas").exists()
                or (candidate / "atlas.db").exists()
            ):
                return candidate

        return candidates[0]

    def _resolve_task_output_file(self, output_path: Path, file_name: str) -> Path:
        """Resolve a task's file name to the concrete output file path."""
        normalized = file_name
        tail = output_path.name
        if normalized.startswith(f"{tail}/"):
            normalized = normalized[len(tail) + 1 :]
        return output_path / normalized

    def _resolve_aggregator_output_file(self, citation: str, output_path: Path) -> Path:
        """Resolve the root aggregator file path for a citation."""
        citation_clean = (
            citation.replace("USC", "").replace("usc", "").replace("\u00a7", "").strip()
        )
        parts_list = citation_clean.split()
        section = parts_list[-1] if parts_list else "root"
        section_clean = section.replace("(", "/").replace(")", "").rstrip("/")
        root_name = (
            section_clean.split("/")[-1] if "/" in section_clean else section_clean
        )
        return output_path / f"{root_name}.rac"

    def _run_compile_gate(self, rac_file: Path):
        """Run the engine compile check used to stop bad files from propagating."""
        pipeline = ValidatorPipeline(
            rac_us_path=rac_file.parent,
            rac_path=self.rac_repo_path,
            enable_oracles=False,
            max_workers=1,
        )
        return pipeline._run_compile_check(rac_file)

    def _log_compile_validation(
        self,
        session_id: str,
        rac_file: Path,
        phase: Phase,
        passed: bool,
        issues: list[str],
        raw_output: str | None = None,
    ) -> None:
        """Persist compile-gate outcomes as provenance validation events."""
        relative_path = self._artifact_relative_path(rac_file)
        lines = [
            f"Compile check for {relative_path}",
            f"Passed: {passed}",
        ]
        if raw_output:
            lines.append(raw_output)
        if issues:
            lines.append("Issues:")
            lines.extend(f"- {issue}" for issue in issues[:10])

        self._log_session_event(
            session_id=session_id,
            event_type="provenance_validation",
            content="\n".join(lines),
            metadata={
                "phase": phase.value,
                "validation_type": "compile",
                "artifact_path": str(rac_file),
                "relative_artifact_path": relative_path,
                "passed": passed,
                "issues": issues,
            },
        )

    def _compile_existing_artifacts(
        self,
        session_id: str,
        output_path: Path,
        phase: Phase,
    ) -> None:
        """Compile-gate all RAC files currently present under an output path."""
        failures = []
        for rac_file in sorted(output_path.rglob("*.rac")):
            result = self._run_compile_gate(rac_file)
            self._log_compile_validation(
                session_id,
                rac_file,
                phase,
                result.passed,
                result.issues,
                result.raw_output,
            )
            if not result.passed:
                failures.extend(
                    f"{rac_file}: {issue}"
                    for issue in (result.issues or [result.error or "compile failed"])
                )

        if failures:
            raise RuntimeError("Compile gate failed:\n" + "\n".join(failures[:10]))

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
        """Find the corpus root used for import resolution."""
        current = output_path
        while current != current.parent:
            if (current / "sources").exists():
                return current
            if current.name == "statute":
                return current
            if current.name in {"regulation", "legislation"}:
                parent = current.parent
                if (parent / "sources").exists():
                    return parent
                return current
            current = current.parent
        parts = list(output_path.parts)
        for idx in range(len(parts) - 1):
            if re.fullmatch(r"\d+", parts[idx]) and re.fullmatch(
                r"[0-9A-Za-z]+", parts[idx + 1]
            ):
                prefix = Path(parts[0])
                for part in parts[1:idx]:
                    prefix /= part
                return prefix
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

    async def _resolve_external_dependencies(
        self, output_path: Path, session_id: str | None = None
    ) -> list[Path]:
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

        if session_id:
            self._log_provenance_decision(
                session_id,
                "Resolving external dependencies",
                [
                    f"Unresolved imports: {len(unresolved)}",
                    *[
                        f"{import_path} -> {var_name}"
                        for import_path, var_name, _ in unresolved[:10]
                    ],
                ],
                Phase.RESOLVE_EXTERNALS,
            )

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
            registered_specs = find_registered_stub_specs(import_path, var_names)
            citation = (
                registered_specs[0].citation
                if registered_specs
                else self._citation_from_path(import_path)
            )

            if expected_path.exists():
                message = f"Skipping stub generation for existing file: {expected_path}"
                print(f"  {message}", flush=True)
                if session_id:
                    self._log_provenance_decision(
                        session_id,
                        "Skipped external stub generation because target file already exists",
                        [
                            f"Citation: {citation}",
                            f"Existing file: {expected_path}",
                            f"Requested variables: {', '.join(var_names)}",
                        ],
                        Phase.RESOLVE_EXTERNALS,
                        metadata={
                            "citation": citation,
                            "artifact_path": str(expected_path),
                            "variable_names": var_names,
                        },
                    )
                continue

            ingested_source_artifacts = find_ingested_source_artifacts(
                import_path, self._find_statute_root(output_path)
            )
            if ingested_source_artifacts:
                source_examples = ", ".join(
                    str(path) for path in ingested_source_artifacts[:3]
                )
                message = (
                    "Refusing RAC stub creation because official source is already ingested: "
                    f"{citation} -> {source_examples}"
                )
                print(f"  {message}", flush=True)
                if session_id:
                    self._log_provenance_decision(
                        session_id,
                        "Blocked external stub generation because official source already exists",
                        [
                            f"Citation: {citation}",
                            f"Import path: {import_path}",
                            f"Requested variables: {', '.join(var_names)}",
                            f"Source artifacts: {source_examples}",
                        ],
                        Phase.RESOLVE_EXTERNALS,
                        metadata={
                            "citation": citation,
                            "import_path": import_path,
                            "variable_names": var_names,
                            "source_artifacts": [
                                str(path) for path in ingested_source_artifacts
                            ],
                        },
                    )
                raise RuntimeError(
                    "Official source already ingested for "
                    f"{import_path}; encode the upstream RAC file instead of creating a stub"
                )

            print(f"  Creating stub for {citation}: {', '.join(var_names)}", flush=True)

            if registered_specs:
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                expected_path.write_text(
                    build_registered_stub_content(registered_specs)
                )
                created.append(expected_path)
                print(f"    Wrote registered stub: {expected_path}", flush=True)
                if session_id:
                    self._log_artifact_provenance_records(
                        session_id,
                        [expected_path],
                        Phase.RESOLVE_EXTERNALS,
                        fallback_source_text=citation,
                    )
                continue

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
                    if session_id:
                        self._log_artifact_provenance_records(
                            session_id,
                            [expected_path],
                            Phase.RESOLVE_EXTERNALS,
                            fallback_source_text=statute_text,
                        )
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
- Define each variable as a normal RAC block like `some_name:`
- Use `entity: TaxUnit|Person|Household|Family`
- Use `period: Year|Month|Day`
- Use `dtype: Money|Boolean|Integer|Rate|String`
- Quote all `label:` and `description:` strings
- Use 4 spaces for fields under each definition
- NEVER use a `variable:` keyword
- NEVER write lowercase schema names like `tax_unit`, `taxable_year`, `boolean`
- For computable variables, include temporal formula blocks
- For input variables, include `default:` field
- Follow RAC DSL conventions (expression-based formulas, no `return` keyword,
  only literals -1, 0, 1, 2, 3 allowed)
"""

    def _extract_openai_response_text(self, payload: dict[str, Any]) -> str:
        """Flatten a Responses API payload into plain assistant text."""
        texts: list[str] = []

        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        for item in payload.get("output", []) or []:
            if not isinstance(item, dict):
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

            if item.get("type") == "reasoning":
                continue

            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

        return "\n\n".join(texts).strip()

    def _materialize_agent_artifact(
        self,
        agent_run: AgentRun,
        expected_path: Path,
    ) -> bool:
        """Write extracted RAC content when a non-CLI backend returns raw file text."""
        if not agent_run.result or agent_run.error:
            return False

        bundle = self._extract_generated_file_bundle(agent_run.result)
        if bundle:
            wrote_main = False
            expected_test_path = expected_path.with_suffix(".rac.test")
            for file_name, content in bundle.items():
                candidate_name = Path(file_name).name
                if candidate_name == expected_path.name:
                    target_path = expected_path
                elif candidate_name == expected_test_path.name:
                    target_path = expected_test_path
                else:
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(content)
                if target_path == expected_path:
                    wrote_main = True
            if wrote_main or expected_path.exists():
                return True

        if expected_path.exists():
            return True

        rac_content = self._extract_rac_content(agent_run.result)
        if not rac_content:
            return False

        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(rac_content)
        return True

    def _extract_generated_file_bundle(self, llm_response: str) -> dict[str, str]:
        """Extract a small multi-file bundle emitted by API backends."""
        if not llm_response or "=== FILE:" not in llm_response:
            return {}

        pattern = re.compile(r"^=== FILE:\s*(?P<name>.+?)\s*===\s*$", re.MULTILINE)
        matches = list(pattern.finditer(llm_response))
        if not matches:
            return {}

        files: dict[str, str] = {}
        for index, match in enumerate(matches):
            start = match.end()
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(llm_response)
            )
            content = llm_response[start:end].strip()
            if content:
                files[match.group("name").strip()] = content + "\n"
        return files

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
        if stripped.startswith('"""') or stripped.startswith("'''"):
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
        xml_root = self.atlas_path / "data" / "uscode"
        if "USC" in citation.upper() and xml_root.exists():
            try:
                precise_text = find_citation_text(citation, xml_root)
            except Exception:
                precise_text = None
            if precise_text:
                print(
                    f"  Statute text loaded from USC XML ({len(precise_text)} chars)",
                    flush=True,
                )
                return precise_text

        # Try atlas first
        try:
            try:
                from atlas import Arch
            except ImportError:
                # Try adding atlas repo to path if installed locally
                atlas_src = self.atlas_path
                atlas_package_roots = []
                if (atlas_src / "src" / "atlas").is_dir():
                    atlas_package_roots.append(atlas_src / "src")
                if (atlas_src / "atlas").is_dir():
                    atlas_package_roots.append(atlas_src)
                if atlas_package_roots:
                    import sys

                    for package_root in atlas_package_roots:
                        package_root_str = str(package_root)
                        if package_root_str not in sys.path:
                            sys.path.insert(0, package_root_str)
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
    # Provenance logging
    # ========================================================================

    def _log_session_event(
        self,
        session_id: str,
        event_type: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        tool_name: str | None = None,
    ) -> None:
        """Write a structured event when DB logging is enabled."""
        if not self.encoding_db:
            return

        self.encoding_db.log_event(
            session_id=session_id,
            event_type=event_type,
            tool_name=tool_name,
            content=content,
            metadata=metadata or {},
        )

    def _log_provenance_decision(
        self,
        session_id: str,
        title: str,
        details: list[str],
        phase: Phase,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a provider-neutral reasoning/decision record."""
        lines = [title, *[line for line in details if line]]
        event_metadata = {"phase": phase.value}
        if metadata:
            event_metadata.update(metadata)
        self._log_session_event(
            session_id=session_id,
            event_type="provenance_decision",
            content="\n".join(lines),
            metadata=event_metadata,
        )

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        """Extract the first flat JSON object from text responses."""
        if not text:
            return None

        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        return parsed if isinstance(parsed, dict) else None

    def _log_analysis_provenance(
        self, session_id: str, citation: str, analysis_text: str
    ) -> None:
        """Log the analyzer's normalized subsection plan."""
        tasks = self._parse_analyzer_output(analysis_text)
        if not tasks:
            self._log_provenance_decision(
                session_id,
                f"Analyzer returned no structured subsection plan for {citation}",
                [
                    "No machine-readable subsection tasks were parsed.",
                    "Subsequent encoding may fall back to a single-file pass.",
                ],
                Phase.ANALYSIS,
                metadata={"citation": citation, "task_count": 0},
            )
            return

        waves = self._compute_waves(tasks)
        lines = [
            f"Encoding plan for {citation}",
            f"Tasks: {len(tasks)}",
            f"Waves: {len(waves)}",
        ]
        for wave_index, wave in enumerate(waves):
            items = []
            for task in wave:
                deps = (
                    f" depends on {', '.join(task.dependencies)}"
                    if task.dependencies
                    else ""
                )
                items.append(
                    f"({task.subsection_id}) {task.title or 'Untitled'} -> {task.file_name}{deps}"
                )
            lines.append(f"Wave {wave_index}: " + "; ".join(items))

        self._log_session_event(
            session_id=session_id,
            event_type="provenance_plan",
            content="\n".join(lines),
            metadata={
                "citation": citation,
                "phase": Phase.ANALYSIS.value,
                "tasks": [
                    {
                        "subsection_id": task.subsection_id,
                        "title": task.title,
                        "file_name": task.file_name,
                        "dependencies": task.dependencies,
                        "wave": task.wave,
                    }
                    for task in tasks
                ],
            },
        )

    def _artifact_relative_path(self, rac_file: Path) -> str:
        """Render a RAC path relative to the statute root when possible."""
        rel_parts = self._citation_path_parts_from_rac_file(rac_file)
        if rel_parts:
            return "/".join(rel_parts)
        try:
            statute_root = self._find_statute_root(rac_file)
            return str(rac_file.relative_to(statute_root))
        except Exception:
            return str(rac_file)

    def _citation_path_parts_from_rac_file(self, rac_file: Path) -> list[str] | None:
        """Extract title/section/subsection path parts from a RAC file path."""
        parts = list(rac_file.parts)
        if "statute" in parts:
            statute_idx = parts.index("statute")
            rel = parts[statute_idx + 1 :]
            return rel if len(rel) >= 3 else None

        for idx in range(len(parts) - 2):
            title = parts[idx]
            section = parts[idx + 1]
            rel = parts[idx:]
            if (
                re.fullmatch(r"\d+", title)
                and re.fullmatch(r"[0-9A-Za-z]+", section)
                and len(rel) >= 3
                and rel[-1].endswith(".rac")
            ):
                return rel

        return None

    def _citation_from_rac_file(self, rac_file: Path) -> str | None:
        """Derive a legal citation from a RAC file path."""
        rel = self._citation_path_parts_from_rac_file(rac_file)
        if not rel or len(rel) < 3:
            return None

        title = rel[0]
        section = rel[1]
        subsection_parts = rel[2:-1]
        stem = rac_file.stem
        if stem != section and (not subsection_parts or stem != subsection_parts[-1]):
            subsection_parts.append(stem)

        suffix = "".join(f"({part})" for part in subsection_parts)
        return f"{title} USC {section}{suffix}"

    def _extract_defined_symbols(self, content: str) -> list[str]:
        """Extract top-level RAC definition names."""
        definitions = []
        for line in content.splitlines():
            match = re.match(r"^([A-Za-z_]\w*):\s*$", line)
            if not match:
                continue
            name = match.group(1)
            if name in PROVENANCE_VALUE_METADATA_KEYS:
                continue
            definitions.append(name)
        return definitions

    def _extract_import_refs(self, content: str) -> list[str]:
        """Extract normalized import references from a RAC file."""
        refs = []
        for match in self._IMPORT_RE.finditer(content):
            refs.append(f"{match.group(1)}#{match.group(2)}")
        return refs

    def _build_artifact_provenance_metadata(
        self,
        rac_file: Path,
        phase: Phase,
        fallback_source_text: str | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Collect deterministic artifact provenance from a RAC file."""
        if not rac_file.exists():
            return None

        try:
            content = rac_file.read_text()
        except Exception:
            return None

        embedded_source_text = extract_embedded_source_text(content)
        source_text = embedded_source_text or fallback_source_text or ""
        source_numbers = (
            extract_numbers_from_text(source_text) if source_text else set()
        )
        numeric_values = extract_grounding_values(content)
        grounded = []
        ungrounded = []
        for line_number, raw_value, numeric_value in numeric_values:
            payload = {
                "line": line_number,
                "raw": raw_value,
                "value": numeric_value,
            }
            if numeric_value in source_numbers:
                grounded.append(payload)
            else:
                ungrounded.append(payload)

        imports = self._extract_import_refs(content)
        definitions = self._extract_defined_symbols(content)
        citation = self._citation_from_rac_file(rac_file)
        relative_path = self._artifact_relative_path(rac_file)
        status_match = re.search(r"^status:\s*(\w+)", content, re.MULTILINE)
        status = status_match.group(1) if status_match else None

        lines = [
            f"Artifact provenance for {relative_path}",
            f"Citation: {citation or 'unknown'}",
            f"Phase: {phase.value}",
            f"Status: {status or 'unknown'}",
            f"Definitions: {', '.join(definitions) if definitions else 'none'}",
            f"Imports: {', '.join(imports) if imports else 'none'}",
            f"Grounded numeric literals: {len(grounded)}",
            f"Ungrounded numeric literals: {len(ungrounded)}",
        ]
        if ungrounded:
            lines.append(
                "Ungrounded values: "
                + ", ".join(
                    f"{item['raw']} (line {item['line']})" for item in ungrounded[:10]
                )
            )
        if embedded_source_text:
            excerpt = embedded_source_text.strip().replace("\n", " ")
            lines.append(
                "Source excerpt: "
                + excerpt[:400]
                + ("..." if len(excerpt) > 400 else "")
            )

        return (
            "\n".join(lines),
            {
                "phase": phase.value,
                "artifact_path": str(rac_file),
                "relative_artifact_path": relative_path,
                "citation": citation,
                "status": status,
                "definitions": definitions,
                "imports": imports,
                "grounded_values": grounded,
                "ungrounded_values": ungrounded,
                "embedded_source_text": embedded_source_text or None,
            },
        )

    def _log_artifact_provenance_records(
        self,
        session_id: str,
        rac_files: list[Path],
        phase: Phase,
        fallback_source_text: str | None = None,
    ) -> None:
        """Emit provenance records for any newly-created RAC artifacts."""
        if not self.encoding_db:
            return

        seen = self._logged_artifact_paths.setdefault(session_id, set())
        for rac_file in rac_files:
            path_key = str(rac_file.resolve()) if rac_file.exists() else str(rac_file)
            if path_key in seen:
                continue

            artifact = self._build_artifact_provenance_metadata(
                rac_file,
                phase,
                fallback_source_text=fallback_source_text,
            )
            if not artifact:
                continue

            content, metadata = artifact
            self._log_session_event(
                session_id=session_id,
                event_type="provenance_artifact",
                content=content,
                metadata=metadata,
            )
            seen.add(path_key)

    def _log_validation_provenance(
        self,
        session_id: str,
        rac_file: Path,
        pe_score: float | None,
        taxsim_score: float | None,
        issues: list[str],
    ) -> None:
        """Log validation outcomes in a provider-neutral structure."""
        relative_path = self._artifact_relative_path(rac_file)
        lines = [
            f"Validation results for {relative_path}",
            f"PolicyEngine: {f'{pe_score:.1%}' if pe_score is not None else 'UNTESTED'}",
            f"TAXSIM: {f'{taxsim_score:.1%}' if taxsim_score is not None else 'UNTESTED'}",
        ]
        if issues:
            lines.append("Issues:")
            lines.extend(f"- {issue}" for issue in issues[:10])
        else:
            lines.append("Issues: none")

        self._log_session_event(
            session_id=session_id,
            event_type="provenance_validation",
            content="\n".join(lines),
            metadata={
                "phase": Phase.ORACLE.value,
                "artifact_path": str(rac_file),
                "relative_artifact_path": relative_path,
                "policyengine_score": pe_score,
                "taxsim_score": taxsim_score,
                "issues": issues,
            },
        )

    def _format_sidecar_trace(self, agent_run: AgentRun) -> str | None:
        """Render a provider-native trace sidecar for display and audit."""
        if not agent_run.provider_trace:
            return None

        trace = agent_run.provider_trace
        summary_lines = [
            f"Provider sidecar trace for {agent_run.agent_type}",
            f"Backend: {trace.get('backend', 'unknown')}",
            f"Provider: {trace.get('provider', 'unknown')}",
        ]
        if trace.get("model"):
            summary_lines.append(f"Model: {trace['model']}")

        trace_json = json.dumps(trace, indent=2, sort_keys=True, default=str)
        return "\n".join([*summary_lines, "", trace_json])

    def _log_provider_reasoning(self, session_id: str, agent_run: AgentRun) -> None:
        """Promote provider-exposed reasoning items into first-class provenance."""
        trace = agent_run.provider_trace or {}
        reasoning_entries = extract_reasoning_entries(trace)
        if not reasoning_entries:
            return

        for index, entry in enumerate(reasoning_entries, start=1):
            preview = entry.text.splitlines()[0].strip()
            if len(preview) > 140:
                preview = preview[:137] + "..."
            self._log_session_event(
                session_id=session_id,
                event_type="provenance_reasoning",
                content=entry.text,
                metadata={
                    "phase": agent_run.phase.value,
                    "agent_type": agent_run.agent_type,
                    "backend": trace.get("backend"),
                    "provider": trace.get("provider"),
                    "model": trace.get("model"),
                    "source": entry.source,
                    "item_id": entry.item_id,
                    "item_type": entry.item_type,
                    "index": index,
                    "reasoning_count": len(reasoning_entries),
                    "summary": preview,
                },
            )

    def _log_sidecar_trace(self, session_id: str, agent_run: AgentRun) -> None:
        """Store raw provider-native traces as optional sidecars."""
        content = self._format_sidecar_trace(agent_run)
        if not content:
            return

        trace = agent_run.provider_trace or {}
        self._log_session_event(
            session_id=session_id,
            event_type="provenance_sidecar",
            content=content,
            metadata={
                "phase": agent_run.phase.value,
                "agent_type": agent_run.agent_type,
                "backend": trace.get("backend"),
                "provider": trace.get("provider"),
                "model": trace.get("model"),
                "summary": "Raw provider-native sidecar trace",
            },
        )

    def _log_review_provenance(self, session_id: str, review_run: AgentRun) -> None:
        """Log normalized review results from reviewer JSON output."""
        parsed = self._extract_json_object(review_run.result or "")
        if not parsed:
            return

        score = parsed.get("score")
        passed = parsed.get("passed")
        issues = parsed.get("issues") if isinstance(parsed.get("issues"), list) else []
        lines = [
            f"Review result for {review_run.agent_type}",
            f"Passed: {passed}",
            f"Score: {score}",
            "Issues: "
            + (", ".join(str(issue) for issue in issues) if issues else "none"),
        ]

        self._log_session_event(
            session_id=session_id,
            event_type="provenance_review",
            content="\n".join(lines),
            metadata={
                "phase": review_run.phase.value,
                "agent_type": review_run.agent_type,
                "score": score,
                "passed": passed,
                "issues": issues,
            },
        )

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
                total.cache_creation_tokens += run.total_tokens.cache_creation_tokens
                total.reasoning_output_tokens += (
                    run.total_tokens.reasoning_output_tokens
                )
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

        for msg in agent_run.messages:
            self.encoding_db.log_event(
                session_id=session_id,
                event_type=f"agent_{msg.role}",
                tool_name=msg.tool_name,
                content=msg.content,
                metadata={
                    "agent_type": agent_run.agent_type,
                    "phase": agent_run.phase.value,
                    "summary": msg.summary,
                    "tokens": {
                        "input": msg.tokens.input_tokens if msg.tokens else 0,
                        "output": msg.tokens.output_tokens if msg.tokens else 0,
                        "cache_read": msg.tokens.cache_read_tokens if msg.tokens else 0,
                        "cache_creation": (
                            msg.tokens.cache_creation_tokens if msg.tokens else 0
                        ),
                        "reasoning_output": (
                            msg.tokens.reasoning_output_tokens if msg.tokens else 0
                        ),
                    }
                    if msg.tokens
                    else None,
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
                "summary": (
                    f"{agent_run.phase.value.upper()}: {len(agent_run.messages)} message(s)"
                    + (f" - ${phase_cost:.2f}" if phase_cost > 0 else "")
                ),
                "total_tokens": {
                    "input": agent_run.total_tokens.input_tokens,
                    "output": agent_run.total_tokens.output_tokens,
                    "cache_read": agent_run.total_tokens.cache_read_tokens,
                    "cache_creation": agent_run.total_tokens.cache_creation_tokens,
                    "reasoning_output": (
                        agent_run.total_tokens.reasoning_output_tokens
                    ),
                }
                if agent_run.total_tokens
                else None,
                "cost_usd": phase_cost,
            },
        )

        self._log_provider_reasoning(session_id, agent_run)
        self._log_sidecar_trace(session_id, agent_run)
        emit_agent_run(
            citation=getattr(self, "_current_citation", None),
            session_id=session_id,
            agent_run=agent_run,
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
                cache_creation_tokens=run.total_tokens.cache_creation_tokens,
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
