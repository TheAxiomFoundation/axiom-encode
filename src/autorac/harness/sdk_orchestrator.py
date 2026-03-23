"""
SDK-based Encoding Orchestrator with full logging.

Uses Claude Agent SDK for complete control over the encoding workflow.
Logs EVERYTHING: every message, tool call, response, token counts.

This is the scientific-grade orchestrator for calibration encodings.
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

from autorac.constants import DEFAULT_MODEL

from .encoding_db import EncodingDB, TokenUsage
from .validator_pipeline import ValidatorPipeline


class Phase(Enum):
    ANALYSIS = "analysis"
    ENCODING = "encoding"
    ORACLE = "oracle"
    REVIEW = "review"
    REPORT = "report"


@dataclass
class SubsectionTask:
    """A single subsection to encode."""

    subsection_id: str  # "a", "b", "c"
    title: str  # "Allowance of credit"
    file_name: str  # "a.rac"
    dependencies: list = field(default_factory=list)  # ["a"] for b.rac importing a.rac
    wave: int = 0


@dataclass
class AnalyzerOutput:
    """Parsed output from the statute analyzer."""

    subsections: List[SubsectionTask] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class AgentMessage:
    """A single message in an agent conversation."""

    role: str  # "user", "assistant", "tool_use", "tool_result"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tokens: Optional[TokenUsage] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    summary: Optional[str] = None  # Human-readable summary of what happened


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
    total_cost: Optional[float] = None  # USD from SDK
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class OrchestratorRun:
    """Complete record of an orchestration run."""

    citation: str
    session_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None

    # All agent runs, in order
    agent_runs: List[AgentRun] = field(default_factory=list)

    # Results
    files_created: List[str] = field(default_factory=list)
    oracle_pe_match: Optional[float] = None
    oracle_taxsim_match: Optional[float] = None
    discrepancies: List[dict] = field(default_factory=list)

    # Totals
    total_tokens: Optional[TokenUsage] = None
    total_cost_usd: float = 0.0


def _summarize_tool_call(
    tool_name: str, tool_input: Optional[dict], tool_output: Optional[str]
) -> str:
    """Generate a human-readable summary of a tool call."""
    if not tool_name:
        return ""

    input_str = ""
    if tool_input:
        # Extract key info based on tool type
        if tool_name == "Read":
            path = tool_input.get("file_path", "unknown")
            input_str = f"'{path.split('/')[-1]}'"
        elif tool_name == "Write":
            path = tool_input.get("file_path", "unknown")
            content = tool_input.get("content", "")
            lines = content.count("\n") + 1 if content else 0
            input_str = f"'{path.split('/')[-1]}' ({lines} lines)"
        elif tool_name == "Edit":
            path = tool_input.get("file_path", "unknown")
            input_str = f"'{path.split('/')[-1]}'"
        elif tool_name == "Grep":
            pattern = tool_input.get("pattern", "")
            input_str = (
                f"pattern='{pattern[:30]}...'"
                if len(pattern) > 30
                else f"pattern='{pattern}'"
            )
        elif tool_name == "Glob":
            pattern = tool_input.get("pattern", "")
            input_str = f"'{pattern}'"
        elif tool_name == "Bash":
            cmd = tool_input.get("command", "")
            input_str = f"'{cmd[:50]}...'" if len(cmd) > 50 else f"'{cmd}'"
        elif tool_name == "Task":
            subagent = tool_input.get("subagent_type", "unknown")
            input_str = f"spawn {subagent}"

    # Add output summary
    output_str = ""
    if tool_output:
        output_len = len(tool_output)
        if output_len > 1000:
            output_str = f" → {output_len:,} chars"
        elif tool_name == "Grep" and "Found" in tool_output:
            # Extract match count
            output_str = f" → {tool_output.split()[1]} files"

    return f"{tool_name} {input_str}{output_str}".strip()


def _summarize_thinking(content: str) -> Optional[str]:
    """Extract first key insight from thinking/reasoning content."""
    if not content:
        return None

    thinking_match = re.search(r"<thinking>([\s\S]*?)</thinking>", content)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        # Get first sentence or first 100 chars
        first_sentence = re.split(r"[.!?\n]", thinking)[0].strip()
        if first_sentence:
            return first_sentence[:150] + ("..." if len(first_sentence) > 150 else "")

    # Look for reasoning patterns
    for prefix in ["I need to", "Let me", "First,", "The statute", "This section"]:
        if prefix.lower() in content.lower()[:500]:
            idx = content.lower().find(prefix.lower())
            snippet = content[idx : idx + 150]
            first_sentence = re.split(r"[.!?\n]", snippet)[0].strip()
            if first_sentence:
                return first_sentence + ("..." if len(first_sentence) > 100 else "")

    return None


def _summarize_assistant_message(content: str) -> str:
    """Summarize an assistant message."""
    if not content:
        return "Empty response"

    # Check for common patterns
    if "Error" in content or "error" in content:
        return "Encountered an error"
    if "Successfully" in content or "Created" in content:
        return "Completed successfully"
    if "```" in content:
        return f"Code block response ({len(content):,} chars)"

    # Get first meaningful line
    lines = [
        line.strip()
        for line in content.split("\n")
        if line.strip() and not line.startswith("#")
    ]
    if lines:
        first = lines[0][:100]
        return first + ("..." if len(lines[0]) > 100 else "")

    return f"Response ({len(content):,} chars)"


class SDKOrchestrator:
    """
    SDK-based orchestrator with full logging.

    Uses Agent SDK to invoke agents with complete control and logging.
    Every message, tool call, and token count is captured.
    Summaries are generated for human readability.
    """

    # Agent definitions from rac-claude plugin
    AGENTS = {
        "analyzer": "rac:Statute Analyzer",
        "encoder": "rac:RAC Encoder",
        "validator": "rac:Encoding Validator",
        "rac_reviewer": "rac:rac-reviewer",
        "formula_reviewer": "rac:Formula Reviewer",
        "parameter_reviewer": "rac:Parameter Reviewer",
        "integration_reviewer": "rac:Integration Reviewer",
    }

    # Map agent keys to their prompt files
    AGENT_PROMPTS = {
        "analyzer": "statute-analyzer.md",
        "encoder": "encoder.md",
        "validator": "validator.md",
        "rac_reviewer": "rac-reviewer.md",
        "formula_reviewer": "formula-reviewer.md",
        "parameter_reviewer": "parameter-reviewer.md",
        "integration_reviewer": "integration-reviewer.md",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str | None = None,
        plugin_path: Optional[Path] = None,
        encoding_db: Optional[EncodingDB] = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        self.model = model or DEFAULT_MODEL
        self.plugin_path = plugin_path or self._find_plugin_path()
        self.encoding_db = encoding_db

    @staticmethod
    def _find_plugin_path() -> Path:
        """Find the rac-claude plugin, checking marketplace and sibling locations."""
        candidates = [
            Path.home() / ".claude" / "plugins" / "marketplaces" / "rac",
            Path.home() / ".claude" / "plugins" / "cache" / "rac" / "rac",
            Path(__file__).parent.parent.parent.parent.parent / "rac-claude",
        ]
        for p in candidates:
            if (p / "agents").exists():
                return p
        return candidates[0]  # Default to marketplace path

    def _load_agent_prompt(self, agent_key: str) -> str:
        """Load the system prompt for an agent from the plugin."""
        prompt_file = self.AGENT_PROMPTS.get(agent_key)
        if not prompt_file:
            return ""
        prompt_path = self.plugin_path / "agents" / prompt_file
        if prompt_path.exists():
            return prompt_path.read_text()
        return ""

    async def encode(
        self,
        citation: str,
        output_path: Path,
        statute_text: Optional[str] = None,
    ) -> OrchestratorRun:
        """
        Run the full 5-phase encoding workflow with complete logging.

        Returns an OrchestratorRun with every agent invocation recorded.
        """
        run = OrchestratorRun(
            citation=citation,
            session_id=f"sdk-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        )

        try:
            # Create DB session FIRST so all agent data survives crashes
            if self.encoding_db:
                from autorac import __version__

                self.encoding_db.start_session(
                    model=self.model,
                    cwd=str(Path.cwd()),
                    session_id=run.session_id,
                    autorac_version=__version__,
                )

            # Pre-fetch statute text if not provided
            if not statute_text:
                statute_text = self._fetch_statute_text(citation)

            # Phase 1: Analysis
            analysis_prompt = self._build_analyzer_prompt(
                citation, statute_text=statute_text
            )

            analysis = await self._run_agent(
                agent_key="analyzer",
                prompt=analysis_prompt,
                phase=Phase.ANALYSIS,
                model=self.model,
            )
            run.agent_runs.append(analysis)
            self._log_agent_run(run.session_id, analysis)

            # Phase 2: Encoding (parallel per-subsection when analysis available)
            if analysis.result:
                encoding_runs = await self._run_encoding_parallel(
                    citation, output_path, statute_text, analysis.result
                )
                for enc_run in encoding_runs:
                    run.agent_runs.append(enc_run)
                    self._log_agent_run(run.session_id, enc_run)

                if not encoding_runs:
                    # Fallback: no subsections parsed, use single-agent
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
                        model=self.model,
                    )
                    run.agent_runs.append(encoding)
                    self._log_agent_run(run.session_id, encoding)
            else:
                # Fallback: no analysis result
                encode_prompt = self._build_fallback_encode_prompt(
                    citation, output_path, statute_text
                )
                encoding = await self._run_agent(
                    agent_key="encoder",
                    prompt=encode_prompt,
                    phase=Phase.ENCODING,
                    model=self.model,
                )
                run.agent_runs.append(encoding)
                self._log_agent_run(run.session_id, encoding)

            # Check what files were created
            if output_path.exists():
                run.files_created = [str(f) for f in output_path.rglob("*.rac")]

            # Phase 3: Oracle validation (use actual ValidatorPipeline, not LLM agent)
            print(
                f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] ORACLE: ValidatorPipeline (PE + TAXSIM)",
                flush=True,
            )
            oracle_start = time.time()

            # Find RAC files to validate
            rac_files = list(output_path.rglob("*.rac")) if output_path.exists() else []

            oracle_context = {
                "pe_match": None,
                "taxsim_match": None,
                "discrepancies": [],
            }

            if rac_files:
                # Use ValidatorPipeline for actual oracle validation
                pipeline = ValidatorPipeline(
                    rac_us_path=output_path.parent.parent
                    if "statute" in str(output_path)
                    else output_path,
                    rac_path=Path(__file__).parent.parent.parent.parent / "rac",
                    enable_oracles=True,
                    max_workers=2,
                )

                # Aggregate results across all RAC files
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

                # Average scores across files
                if pe_scores:
                    oracle_context["pe_match"] = (
                        sum(pe_scores) / len(pe_scores) * 100
                    )  # Convert to percentage
                if taxsim_scores:
                    oracle_context["taxsim_match"] = (
                        sum(taxsim_scores) / len(taxsim_scores) * 100
                    )
                oracle_context["discrepancies"] = [
                    {"description": issue} for issue in all_issues[:10]
                ]
            else:
                print("  No RAC files found to validate", flush=True)

            oracle_duration = time.time() - oracle_start
            print(
                f"  DONE: PE={oracle_context.get('pe_match', 'N/A')}%, TAXSIM={oracle_context.get('taxsim_match', 'N/A')}% ({oracle_duration:.1f}s)",
                flush=True,
            )

            run.oracle_pe_match = oracle_context.get("pe_match")
            run.oracle_taxsim_match = oracle_context.get("taxsim_match")
            run.discrepancies = oracle_context.get("discrepancies", [])

            # Phase 4: LLM Review (parallel)
            oracle_summary = self._format_oracle_summary(oracle_context)

            # Run reviewers sequentially (parallel has async issues with SDK)
            for reviewer_key, reviewer_type in [
                ("formula_reviewer", "formulas"),
                ("parameter_reviewer", "parameters"),
                ("integration_reviewer", "integration"),
            ]:
                review = await self._run_agent(
                    agent_key=reviewer_key,
                    prompt=f"Review {citation} {reviewer_type}. Oracle found: {oracle_summary}",
                    phase=Phase.REVIEW,
                    model=self.model,  # Use configured model
                )
                run.agent_runs.append(review)
                self._log_agent_run(run.session_id, review)

            # Phase 5: Report (computed, not an agent)
            run.ended_at = datetime.utcnow()
            run.total_tokens = self._sum_tokens(run.agent_runs)
            run.total_cost_usd = self._sum_cost(run.agent_runs)

            # Log to encoding DB if available
            if self.encoding_db:
                self._log_to_db(run)

        except Exception as e:
            run.ended_at = datetime.utcnow()
            # Log error but don't lose partial results
            if run.agent_runs:
                run.agent_runs[-1].error = str(e)

        return run

    async def _run_agent(
        self,
        agent_key: str,
        prompt: str,
        phase: Phase,
        model: str,
    ) -> AgentRun:
        """Run a single agent and capture everything."""

        from claude_agent_sdk import ClaudeAgentOptions, query

        agent_type = self.AGENTS.get(agent_key, agent_key)

        # Load the agent's system prompt from plugin
        system_prompt = self._load_agent_prompt(agent_key)
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n# TASK\n\n{prompt}"
        else:
            full_prompt = prompt

        run = AgentRun(
            agent_type=agent_type,
            prompt=prompt,  # Store original prompt, not full
            phase=phase,
        )

        # ALWAYS print phase start with timestamp
        start_time = datetime.utcnow()
        print(
            f"\n[{start_time.strftime('%H:%M:%S')}] {phase.value.upper()}: {agent_type}",
            flush=True,
        )
        print(f"  Model: {model}", flush=True)

        try:
            options = ClaudeAgentOptions(
                model=model,
                allowed_tools=["Read", "Write", "Edit", "Grep", "Glob", "Bash", "Task"],
            )

            # Collect all messages - log tokens in REAL TIME
            total_input = 0
            total_output = 0
            total_cache_read = 0
            event_count = 0

            async for event in query(prompt=full_prompt, options=options):
                event_count += 1
                event_type = type(event).__name__

                msg = AgentMessage(
                    role=getattr(event, "type", event_type),
                    content="",
                )

                # Parse SDK content blocks properly
                if hasattr(event, "content"):
                    content = event.content
                    # content is usually a list of blocks
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            block_type = type(block).__name__
                            if block_type == "TextBlock" and hasattr(block, "text"):
                                text_parts.append(block.text)
                            elif block_type == "ToolUseBlock":
                                # Extract tool info
                                if hasattr(block, "name"):
                                    msg.tool_name = block.name
                                    print(f"  Tool: {block.name}", flush=True)
                                if hasattr(block, "input"):
                                    msg.tool_input = (
                                        block.input
                                        if isinstance(block.input, dict)
                                        else {"raw": str(block.input)}
                                    )
                                # Generate summary
                                msg.summary = _summarize_tool_call(
                                    msg.tool_name, msg.tool_input, None
                                )
                            elif block_type == "ToolResultBlock":
                                # Tool result content
                                if hasattr(block, "content"):
                                    result_text = str(block.content)[:5000]
                                    text_parts.append(result_text)
                                    msg.tool_output = result_text
                        msg.content = (
                            "\n".join(text_parts)
                            if text_parts
                            else str(content)[:10000]
                        )
                        # Generate thinking summary if we have text
                        if text_parts and not msg.summary:
                            msg.summary = _summarize_thinking(
                                "\n".join(text_parts)
                            ) or _summarize_assistant_message("\n".join(text_parts))
                    else:
                        msg.content = str(content)[:10000]
                elif hasattr(event, "result"):
                    msg.content = str(event.result)[:10000]
                    run.result = event.result
                    msg.summary = "Final result"

                # Fallback tool capture from event attributes
                if not msg.tool_name and hasattr(event, "tool_name"):
                    msg.tool_name = event.tool_name
                    print(f"  Tool: {event.tool_name}", flush=True)
                if not msg.tool_input and hasattr(event, "tool_input"):
                    msg.tool_input = event.tool_input
                if not msg.tool_output and hasattr(event, "tool_output"):
                    msg.tool_output = str(event.tool_output)[:10000]

                # Generate summary if not already set
                if not msg.summary and msg.tool_name:
                    msg.summary = _summarize_tool_call(
                        msg.tool_name, msg.tool_input, msg.tool_output
                    )

                # Capture tokens from ResultMessage (final event has real data)
                if event_type == "ResultMessage" and hasattr(event, "usage"):
                    usage = event.usage
                    # usage is a dict, not an object
                    in_tok = usage.get("input_tokens", 0)
                    out_tok = usage.get("output_tokens", 0)
                    cache_create = usage.get("cache_creation_input_tokens", 0)
                    cache_read = usage.get("cache_read_input_tokens", 0)

                    msg.tokens = TokenUsage(
                        input_tokens=in_tok + cache_create,
                        output_tokens=out_tok,
                        cache_read_tokens=cache_read,
                    )
                    total_input = in_tok + cache_create
                    total_output = out_tok
                    total_cache_read = cache_read

                    # Also capture total_cost_usd if available
                    if hasattr(event, "total_cost_usd"):
                        run.total_cost = event.total_cost_usd

                    # Print final token summary
                    print(
                        f"  Tokens: {in_tok + cache_create:,} in (+{cache_read:,} cache), {out_tok:,} out",
                        flush=True,
                    )
                    if hasattr(event, "total_cost_usd"):
                        print(f"  Cost: ${event.total_cost_usd:.4f}", flush=True)

                run.messages.append(msg)

            run.ended_at = datetime.utcnow()
            run.total_tokens = TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                cache_read_tokens=total_cache_read,
            )

            # Print phase summary
            duration = (run.ended_at - start_time).total_seconds()
            total_cost = run.total_tokens.estimated_cost_usd
            print(
                f"  DONE: {total_input:,} in + {total_output:,} out = ${total_cost:.4f} ({duration:.1f}s)",
                flush=True,
            )

        except Exception as e:
            run.error = str(e)
            run.ended_at = datetime.utcnow()
            print(f"  ERROR: {e}", flush=True)
            # Still print whatever tokens we captured
            if total_input or total_output:
                print(
                    f"  Partial tokens: {total_input:,} in + {total_output:,} out",
                    flush=True,
                )

        return run

    def _parse_analyzer_output(self, analysis_text: str) -> AnalyzerOutput:
        """Parse analyzer output into structured subsection tasks.

        Two-layer parsing:
        - Primary: JSON block in <!-- STRUCTURED_OUTPUT {...} --> tags
        - Fallback: Regex on the markdown subsection table
        """
        result = AnalyzerOutput(raw_text=analysis_text)

        if not analysis_text or not analysis_text.strip():
            return result

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
                    task = SubsectionTask(
                        subsection_id=s["id"],
                        title=s.get("title", ""),
                        file_name=s.get("file", f"{s['id']}.rac"),
                        dependencies=deps.get(s["id"], []),
                    )
                    result.subsections.append(task)
                return result
            except (json.JSONDecodeError, KeyError):
                pass  # Fall through to markdown parsing

        # Fallback: parse markdown table rows
        # Pattern: | (subsection_id) | title | disposition | file |
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
            task = SubsectionTask(
                subsection_id=sub_id,
                title=title,
                file_name=file_name if file_name != "-" else f"{sub_id}.rac",
                dependencies=[],
            )
            result.subsections.append(task)

        return result

    def _compute_waves(self, tasks: List[SubsectionTask]) -> List[List[SubsectionTask]]:
        """Topological sort into parallel batches (waves).

        Subsections with no dependencies = wave 0.
        Subsections depending only on wave 0 items = wave 1, etc.
        """
        if not tasks:
            return []

        assigned = set()
        waves = []

        while len(assigned) < len(tasks):
            wave = []
            for t in tasks:
                if t.subsection_id in assigned:
                    continue
                # All deps must be in already-assigned set
                if all(d in assigned for d in t.dependencies):
                    t.wave = len(waves)
                    wave.append(t)
            if not wave:
                # Remaining tasks have unsatisfiable deps — put them all in next wave
                remaining = [t for t in tasks if t.subsection_id not in assigned]
                for t in remaining:
                    t.wave = len(waves)
                waves.append(remaining)
                break
            assigned.update(t.subsection_id for t in wave)
            waves.append(wave)

        return waves

    DSL_CHEATSHEET = """
## RAC DSL quick reference (unified syntax)

### Declaration types
- `name:` with `from yyyy-mm-dd:` — policy value with temporal entries
- `name:` with `formula:` — computed value
- `name:` with `default:` — user-provided input
- `enum Name:` — enumeration with `values:` list

### Fields (all required unless noted)
- `entity:` Person | TaxUnit | Household | Family
- `period:` Year | Month | Day
- `dtype:` Money | Rate | Boolean | Integer | String | Enum[Name]
- `formula: |` — Python-like formula (see below)
- `imports:` — list of `path#name` (optional)

### Temporal syntax
```yaml
ctc_base_amount:
  from 2018-01-01: 2000
  from 2025-01-01: 2500
```

### Text blocks
```
\"\"\"
Statute text goes here...
\"\"\"
```

### Tests go in `.rac.test` files (not inline)

### Formula syntax
**Allowed:** `if`/`elif`/`else`, `return`, `and`/`or`/`not`, `=` assignment
**Operators:** `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`
**Built-in functions:** `min(a,b)`, `max(a,b)`, `abs(x)`, `floor(x)`, `ceil(x)`, `round(x,n)`, `clamp(x,lo,hi)`, `sum(...)`, `len(...)`
**FORBIDDEN:** `for`/`while` loops, list comprehensions, `def`/`lambda`, `try`/`except`, `+=`, imports
**Numeric literals:** ONLY -1, 0, 1, 2, 3 allowed. ALL other numbers must come from named definitions.

### Import syntax
```yaml
imports: [26/24/a#ctc_maximum, 26/24/b#ctc_phaseout as phaseout]
```

### Exemplar
```yaml
ctc_base_amount:
  description: "Base credit per qualifying child per 26 USC 24(a)"
  unit: USD
  from 2018-01-01: 2000

qualifying_child_count:
  entity: TaxUnit
  period: Year
  dtype: Integer
  default: 0

ctc_maximum:
  entity: TaxUnit
  period: Year
  dtype: Money
  formula: |
    return qualifying_child_count * ctc_base_amount
```
"""

    def _build_subsection_prompt(
        self,
        task: SubsectionTask,
        citation: str,
        output_path: Path,
        statute_text: Optional[str] = None,
        subsection_text: Optional[str] = None,
    ) -> str:
        """Build a focused encoding prompt for a single subsection."""
        parts = [
            f"Encode {citation} subsection ({task.subsection_id}) - "
            f'"{task.title}" into RAC format.',
            "",
            f"Output: {output_path / task.file_name}",
            f"Scope: ONLY this subsection. One file: {task.file_name}",
            "",
            "## CRITICAL RULES (violations = encoding failure):",
            "",
            "1. **FILEPATH = CITATION** - File names MUST be subsection names",
            "2. **One subsection per file** - Each .rac encodes exactly one statutory subsection",
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

        if subsection_text:
            parts.append("")
            parts.append(
                f"Statute text for subsection ({task.subsection_id}):\n{subsection_text}"
            )
        elif statute_text:
            parts.append("")
            parts.append(f"Full statute text (excerpt):\n{statute_text[:5000]}")

        parts.append(self.DSL_CHEATSHEET)

        return "\n".join(parts)

    def _get_cached_section(self, citation: str):
        """Get a cached atlas Section for a citation, or None if unavailable."""
        if not hasattr(self, "_section_cache"):
            self._section_cache = {}
        if citation not in self._section_cache:
            try:
                from atlas import Arch

                db_path = Path.home() / "RulesFoundation" / "atlas" / "atlas.db"
                if not db_path.exists():
                    self._section_cache[citation] = None
                else:
                    a = Arch(db_path=db_path)
                    self._section_cache[citation] = a.get(citation)
            except ImportError:
                self._section_cache[citation] = None
        return self._section_cache[citation]

    def _fetch_subsection_text(
        self, citation: str, subsection_id: str
    ) -> Optional[str]:
        """Fetch text for a specific subsection from atlas."""
        section = self._get_cached_section(citation)
        if section is None:
            return None
        return section.get_subsection_text(subsection_id)

    def _fetch_statute_text(
        self,
        citation: str,
        xml_path: Optional[Path] = None,
    ) -> Optional[str]:
        """Fetch statute text — atlas first, fallback to legacy XML regex."""
        section = self._get_cached_section(citation)
        if section:
            return section.text
        return self._fetch_statute_text_legacy(citation, xml_path)

    def _fetch_statute_text_legacy(
        self,
        citation: str,
        xml_path: Optional[Path] = None,
    ) -> Optional[str]:
        """Pre-fetch statute text from local USC XML to avoid per-encoder discovery.

        Returns the statute text as a string, or None if not available.
        """
        import html as html_mod

        if xml_path is None:  # pragma: no cover
            xml_path = Path.home() / "RulesFoundation" / "atlas" / "data" / "uscode"

        # Parse citation: "26 USC 24" or "26 USC 25A"
        citation_clean = citation.upper().replace("USC", "").replace("§", "").strip()
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

        # Find matching closing tag
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

        # Strip tags, preserve text
        text = re.sub(r"<[^>]+>", " ", xml_section)
        text = html_mod.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()

        return text if text else None

    def _batch_small_subsections(
        self,
        tasks: List[SubsectionTask],
        max_batch: int = 4,
    ) -> List[SubsectionTask]:
        """Batch related small subsections to reduce per-encoder overhead.

        Groups sibling subsections (same parent path) into batched tasks,
        unless they have intra-group dependencies.
        """
        if len(tasks) <= 3:
            return list(tasks)

        # Group by parent path (e.g., "g/1", "g/3", "g/4" all have parent "g")
        from collections import defaultdict

        groups: dict = defaultdict(list)
        for t in tasks:
            parts = t.subsection_id.rsplit("/", 1)
            parent = parts[0] if len(parts) > 1 else ""
            groups[parent].append(t)

        result = []
        for parent, group in groups.items():
            if len(group) < 2 or not parent:
                # No batching: single items or top-level subsections
                result.extend(group)
                continue

            # Check for intra-group dependencies
            group_ids = {t.subsection_id for t in group}

            # Partition: tasks with deps on group members go unbatched
            batchable = []
            unbatchable = []
            for t in group:
                has_internal_dep = any(d in group_ids for d in t.dependencies)
                if has_internal_dep:
                    unbatchable.append(t)
                else:
                    batchable.append(t)

            # Create batches of up to max_batch
            for i in range(0, len(batchable), max_batch):
                chunk = batchable[i : i + max_batch]
                if len(chunk) == 1:
                    result.append(chunk[0])
                else:
                    # Merge into a single compound task
                    merged_id = ",".join(t.subsection_id for t in chunk)
                    merged_title = "; ".join(t.title for t in chunk)
                    merged_files = ",".join(t.file_name for t in chunk)
                    # Union of all dependencies (excluding batched members)
                    merged_deps = list(
                        set(
                            d
                            for t in chunk
                            for d in t.dependencies
                            if d not in group_ids
                        )
                    )
                    result.append(
                        SubsectionTask(
                            subsection_id=merged_id,
                            title=merged_title,
                            file_name=merged_files,
                            dependencies=merged_deps,
                        )
                    )

            result.extend(unbatchable)

        return result

    def _build_analyzer_prompt(
        self, citation: str, statute_text: Optional[str] = None
    ) -> str:
        """Build the analysis prompt with structured output instructions."""
        text_section = ""
        if statute_text:
            text_section = f"""

## Statute Text

The following is the AUTHORITATIVE text of {citation}. Use ONLY this text to identify subsections.
Do NOT rely on your training data for the subsection structure — the text below is the source of truth.

<statute>
{statute_text}
</statute>
"""
        else:
            text_section = f"""

NOTE: Statute text for {citation} was not available. You MUST fetch it using WebFetch or WebSearch
before analyzing. Do NOT guess the subsection structure from memory — it may be inaccurate.
"""

        return f"""Analyze {citation}. Report: subsection tree, encoding order, dependencies.
{text_section}
After your markdown analysis, include a machine-readable block:
<!-- STRUCTURED_OUTPUT
{{"subsections": [{{"id": "a", "title": "...", "disposition": "ENCODE", "file": "a.rac"}}, ...],
 "dependencies": {{"b": ["a"], "d": ["a"]}},
 "encoding_order": ["a", "c", "e", "f", "b", "d"]}}
-->

Valid dispositions:
- "ENCODE" — subsection contains encodable rules (parameters, formulas, eligibility)
- "SKIP" — subsection is definitional only or cross-references another section
- "OBSOLETE" — subsection has been repealed, struck, or redesignated (do NOT encode)

List dependencies as subsection IDs that must be encoded first.
For small related subsections under the same parent (e.g., (g)(1), (g)(3), (g)(4)),
consider whether they can share a single file."""

    def _build_fallback_encode_prompt(
        self,
        citation: str,
        output_path: Path,
        statute_text: Optional[str],
    ) -> str:
        """Build encoding prompt for single-agent fallback (original behavior)."""
        return f"""Encode {citation} into RAC format.

Output path: {output_path}
{f"Statute text: {statute_text[:5000]}" if statute_text else "Fetch statute text as needed."}

## CRITICAL RULES (violations = encoding failure):

1. **FILEPATH = CITATION** - File names MUST be subsection names:
   - ✓ `statute/26/1/j.rac` for § 1(j)
   - ✓ `statute/26/1/a.rac` for § 1(a)
   - ❌ `formulas.rac`, `parameters.rac`, `variables.rac` - WRONG

2. **One subsection per file** - Each .rac encodes exactly one statutory subsection

3. **Only statute values** - No indexed/derived/computed values

Write .rac files to the output path. Run tests after each file."""

    async def _run_encoding_parallel(
        self,
        citation: str,
        output_path: Path,
        statute_text: Optional[str],
        analysis_result: str,
        max_concurrent: int = 5,
    ) -> List[AgentRun]:
        """Encode subsections in parallel waves, respecting dependencies."""
        parsed = self._parse_analyzer_output(analysis_result)
        if not parsed.subsections:
            return []

        # Batch small sibling subsections to reduce per-encoder overhead
        batched = self._batch_small_subsections(parsed.subsections)

        waves = self._compute_waves(batched)
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
                    sub_text = self._fetch_subsection_text(citation, task.subsection_id)
                    prompt = self._build_subsection_prompt(
                        task,
                        citation,
                        output_path,
                        statute_text=statute_text,
                        subsection_text=sub_text,
                    )
                    return await self._run_agent(
                        "encoder", prompt, Phase.ENCODING, self.model
                    )

            results = await asyncio.gather(
                *[encode_one(t) for t in wave],
                return_exceptions=True,
            )

            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    print(
                        f"  FAILED ({wave[i].subsection_id}): {r}",
                        flush=True,
                    )
                else:
                    all_runs.append(r)

        return all_runs

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
                total.cache_creation_tokens += run.total_tokens.cache_creation_tokens
                total.reasoning_output_tokens += (
                    run.total_tokens.reasoning_output_tokens
                )
        return total

    def _sum_cost(self, runs: List[AgentRun]) -> float:
        """Sum costs across agent runs, preferring SDK-reported costs.

        Uses SDK total_cost when available, falls back to token-based estimation.
        """
        total = 0.0
        for run in runs:
            if run.total_cost is not None:
                total += run.total_cost
            elif run.total_tokens:
                total += run.total_tokens.estimated_cost_usd
        return total

    def _log_agent_run(self, session_id: str, agent_run: AgentRun) -> None:
        """Log a single agent run to the DB immediately after it completes.

        Called after each _run_agent() so data survives crashes.
        """
        if not self.encoding_db:
            return

        # Log agent start
        self.encoding_db.log_event(
            session_id=session_id,
            event_type="agent_start",
            content=agent_run.prompt,
            metadata={
                "agent_type": agent_run.agent_type,
                "phase": agent_run.phase.value,
            },
        )

        # Log each message
        for msg in agent_run.messages:
            self.encoding_db.log_event(
                session_id=session_id,
                event_type=f"agent_{msg.role}",
                tool_name=msg.tool_name,
                content=msg.content,
                metadata={
                    "agent_type": agent_run.agent_type,
                    "summary": msg.summary,
                    "tool_input": msg.tool_input,
                    "tool_output": msg.tool_output[:5000] if msg.tool_output else None,
                    "tokens": {
                        "input": msg.tokens.input_tokens if msg.tokens else 0,
                        "output": msg.tokens.output_tokens if msg.tokens else 0,
                    }
                    if msg.tokens
                    else None,
                },
            )

        # Calculate phase cost
        phase_cost = (
            agent_run.total_cost
            if agent_run.total_cost is not None
            else (
                agent_run.total_tokens.estimated_cost_usd
                if agent_run.total_tokens
                else 0
            )
        )

        # Generate phase summary
        tool_counts = {}
        for msg in agent_run.messages:
            if msg.tool_name:
                tool_counts[msg.tool_name] = tool_counts.get(msg.tool_name, 0) + 1
        tools_summary = ", ".join(
            f"{t}×{c}" for t, c in sorted(tool_counts.items(), key=lambda x: -x[1])
        )

        phase_summary = (
            f"{agent_run.phase.value.upper()}: {len(agent_run.messages)} events"
        )
        if tools_summary:
            phase_summary += f" ({tools_summary})"
        if phase_cost > 0:
            phase_summary += f" - ${phase_cost:.2f}"

        # Log agent end
        self.encoding_db.log_event(
            session_id=session_id,
            event_type="agent_end",
            content=agent_run.result or "",
            metadata={
                "agent_type": agent_run.agent_type,
                "phase": agent_run.phase.value,
                "summary": phase_summary,
                "error": agent_run.error,
                "total_tokens": {
                    "input": agent_run.total_tokens.input_tokens,
                    "output": agent_run.total_tokens.output_tokens,
                }
                if agent_run.total_tokens
                else None,
                "cost_usd": phase_cost,
                "tools_used": tool_counts,
            },
        )

    def _log_to_db(self, run: OrchestratorRun) -> None:
        """Finalize session in DB with totals. Agent runs already logged incrementally."""
        if not self.encoding_db:
            return

        # Update session totals
        if run.total_tokens:
            self.encoding_db.update_session_tokens(
                session_id=run.session_id,
                input_tokens=run.total_tokens.input_tokens,
                output_tokens=run.total_tokens.output_tokens,
                cache_read_tokens=run.total_tokens.cache_read_tokens,
                cache_creation_tokens=run.total_tokens.cache_creation_tokens,
            )

    def print_report(self, run: OrchestratorRun) -> str:
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

        lines.extend(
            [
                "",
                "## Agent Runs",
            ]
        )
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
