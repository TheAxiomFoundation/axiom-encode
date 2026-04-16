"""
Encoder Harness - wraps encoder agent with prediction and logging.

The harness orchestrates:
1. Agent encodes the statute using embedded prompts
2. Validators run in parallel
3. Everything is logged for calibration

Self-contained -- no dependency on external plugins.
Uses Claude Code CLI (subprocess) for agent calls.
"""

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from autorac.constants import DEFAULT_CLI_MODEL, DEFAULT_MODEL
from autorac.prompts.encoder import get_encoder_prompt

from .encoding_db import (
    EncodingDB,
    EncodingRun,
    create_run,
)
from .validator_pipeline import PipelineResult, ValidatorPipeline


def run_claude_code(
    prompt: str,
    model: str = DEFAULT_CLI_MODEL,
    timeout: int = 300,
    cwd: Optional[Path] = None,
) -> tuple[str, int]:
    """
    Run Claude Code CLI as subprocess.

    Args:
        prompt: The prompt to send (should include system prompt if needed)
        model: Model to use (opus, haiku) - never sonnet
        timeout: Timeout in seconds
        cwd: Working directory

    Returns:
        Tuple of (output text, return code)
    """
    cmd = ["claude", "--print"]

    if model:
        cmd.extend(["--model", model])

    # Add the prompt
    cmd.extend(["-p", prompt])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return f"Timeout after {timeout}s", 1
    except FileNotFoundError:
        return (
            "Claude CLI not found - install with: npm install -g @anthropic-ai/claude-code",
            1,
        )
    except Exception as e:
        return f"Error running Claude CLI: {e}", 1


@dataclass
class EncoderConfig:
    """Configuration for the encoder harness."""

    rac_us_path: Path
    rac_path: Path
    db_path: Path = Path("encodings.db")
    enable_oracles: bool = True
    max_iterations: int = 3
    score_threshold: float = 7.0  # Minimum score to accept


class EncoderHarness:
    """
    Wraps encoder agent with validation and logging.

    The harness implements the encode-validate-learn loop:
    1. Agent encodes statute to RAC using embedded prompts
    2. Validators run in parallel
    3. Results logged for calibration analysis
    4. Agent suggests improvements based on errors
    """

    def __init__(self, config: EncoderConfig):
        self.config = config
        self.db = EncodingDB(config.db_path)
        self.pipeline = ValidatorPipeline(
            rac_us_path=config.rac_us_path,
            rac_path=config.rac_path,
            enable_oracles=config.enable_oracles,
        )

    def encode_with_feedback(
        self,
        citation: str,
        statute_text: str,
        output_path: Path,
        agent_type: str = "autorac:encoder",
        agent_model: str | None = None,
    ) -> tuple[EncodingRun, PipelineResult]:
        """
        Full encode-validate-log cycle.

        Returns the encoding run and validation results.
        """
        agent_model = agent_model or DEFAULT_MODEL
        start = time.time()

        # Step 1: Encode
        rac_content = self._encode(citation, statute_text, output_path)

        encoding_duration = int((time.time() - start) * 1000)

        # Step 2: Validate
        validation_start = time.time()
        validation_result = self.pipeline.validate(output_path)
        validation_duration = int((time.time() - validation_start) * 1000)

        # Step 3: Get lessons from failures
        lessons = self._get_lessons(citation, rac_content, validation_result)

        # Step 4: Log everything
        review_results = validation_result.to_review_results()
        review_results.lessons = lessons

        run = create_run(
            file_path=str(output_path),
            citation=citation,
            agent_type=agent_type,
            agent_model=agent_model,
            rac_content=rac_content,
            statute_text=statute_text,
            review_results=review_results,
            lessons=lessons,
        )
        run.total_duration_ms = encoding_duration + validation_duration

        self.db.log_run(run)

        return run, validation_result

    def iterate_until_pass(
        self,
        citation: str,
        statute_text: str,
        output_path: Path,
        agent_type: str = "autorac:encoder",
        agent_model: str | None = None,
    ) -> list[tuple[EncodingRun, PipelineResult]]:
        """
        Iteratively encode until all validators pass or max iterations.

        Returns list of (run, result) for each iteration.
        """
        agent_model = agent_model or DEFAULT_MODEL
        iterations = []
        parent_run_id = None

        for i in range(self.config.max_iterations):
            run, result = self.encode_with_feedback(
                citation=citation,
                statute_text=statute_text,
                output_path=output_path,
                agent_type=agent_type,
                agent_model=agent_model,
            )

            if parent_run_id:
                run.parent_run_id = parent_run_id
                run.iteration = i + 1

            iterations.append((run, result))

            if result.all_passed:
                break

            # Prepare feedback for next iteration
            parent_run_id = run.id

        return iterations

    def _encode(self, citation: str, statute_text: str, output_path: Path) -> str:
        """
        Invoke Claude Code to encode the statute to RAC format.

        Uses the embedded encoder prompt -- no external plugin needed.
        """
        prompt = get_encoder_prompt(citation, str(output_path))

        # Append statute text to the prompt
        prompt += f"\n\nStatute Text:\n{statute_text}\n"

        try:
            output, returncode = run_claude_code(
                prompt,
                model=DEFAULT_CLI_MODEL,
                timeout=300,
                cwd=self.config.rac_us_path,
            )

            # Check if file was created
            if output_path.exists():
                rac_content = output_path.read_text()
            else:
                # Try to extract RAC content from output
                rac_content = output

            # Clean up any markdown code blocks if present
            rac_content = re.sub(r"^```\w*\n", "", rac_content)
            rac_content = re.sub(r"\n```$", "", rac_content)
            rac_content = rac_content.strip()

            # Ensure output directory exists and write
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rac_content)

            return rac_content

        except Exception as e:
            print(f"Warning: Failed to encode: {e}")

            # Derive variable name from citation for fallback
            var_name = (
                citation.replace("USC", "")
                .replace("(", "_")
                .replace(")", "")
                .replace(" ", "_")
                .lower()
            )
            var_name = re.sub(r"_+", "_", var_name).strip("_")

            fallback = f'''"""
{statute_text}
"""

{var_name}:
  entity: TaxUnit
  period: Year
  dtype: Money
  label: "{citation}"
  description: "Auto-generated placeholder - encoding failed"
  formula: |
    # TODO(#issue-needed): Implement formula for {citation}.
    # This fallback was emitted after the encoder failed. A human (or a
    # retry with a stronger model) needs to author the real formula. See
    # CHANGELOG.md for the tracked list of open encoder TODOs.
    return 0
  default: 0
'''
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(fallback)
            return fallback

    def _get_lessons(
        self,
        citation: str,
        rac_content: str,
        validation_result: PipelineResult,
    ) -> str:
        """
        Ask Claude Code for lessons learned from encoding attempt.

        Based on validation errors, extracts lessons as free text.
        """
        # Only get lessons if there were failures
        failures = [
            (name, result)
            for name, result in validation_result.results.items()
            if not result.passed
        ]

        if not failures:
            return ""

        # Build validation summary
        validation_summary = []
        for name, result in validation_result.results.items():
            status = "PASSED" if result.passed else "FAILED"
            score_str = f" (score: {result.score})" if result.score is not None else ""
            error_str = f" - {result.error}" if result.error else ""
            issues_str = f" Issues: {result.issues}" if result.issues else ""
            validation_summary.append(
                f"  {name}: {status}{score_str}{error_str}{issues_str}"
            )

        prompt = f"""Analyze encoding attempt for {citation} and summarize lessons learned.

Validation Results:
{chr(10).join(validation_summary)}

Write a brief paragraph summarizing what went wrong and what to do differently next time.
Output ONLY the lessons text, no JSON or formatting.
"""

        try:
            output, returncode = run_claude_code(
                prompt,
                model=DEFAULT_CLI_MODEL,
                timeout=60,
                cwd=self.config.rac_us_path,
            )

            return output.strip()

        except Exception as e:
            print(f"Warning: Failed to get lessons: {e}")

            return "; ".join(
                f"{name} failed: {result.error or 'unknown'}"
                for name, result in failures
            )


def run_encoding_experiment(
    citation: str,
    statute_text: str,
    output_dir: Path,
    config: Optional[EncoderConfig] = None,
) -> list[tuple[EncodingRun, PipelineResult]]:
    """
    Convenience function to run a full encoding pipeline.

    Args:
        citation: Legal citation (e.g., "26 USC 1(h)(1)(E)")
        statute_text: Raw statute text to encode
        output_dir: Directory for output RAC file
        config: Optional encoder configuration

    Returns:
        List of (run, result) tuples for each iteration
    """
    if config is None:
        # Auto-detect paths
        rac_us = output_dir
        while rac_us.name != "rac-us" and rac_us.parent != rac_us:
            rac_us = rac_us.parent

        config = EncoderConfig(
            rac_us_path=rac_us,
            rac_path=rac_us.parent / "rac",
        )

    harness = EncoderHarness(config)

    # Derive output path from citation
    # "26 USC 1(h)(1)(E)" -> statute/26/1/h/1/E.rac
    parts = citation.replace("USC", "").replace("(", "/").replace(")", "").split()
    title = parts[0]
    rest = "".join(parts[1:])
    output_path = output_dir / f"statute/{title}/{rest}.rac"

    return harness.iterate_until_pass(
        citation=citation,
        statute_text=statute_text,
        output_path=output_path,
    )
