"""
Encoder backends - abstraction for different Claude invocation methods.

Two backends:
1. ClaudeCodeBackend - uses Claude Code CLI (subprocess), works with Max subscription
2. AgentSDKBackend - uses Claude API (anthropic SDK), enables parallelization

Both use embedded prompts -- no external plugin dependencies.
"""

import asyncio
import json
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from autorac.constants import DEFAULT_CLI_MODEL, DEFAULT_MODEL
from autorac.prompts.encoder import get_encoder_prompt

from .encoding_db import TokenUsage


@dataclass
class EncoderRequest:
    """Input for an encoding operation."""

    citation: str
    statute_text: str
    output_path: Path
    model: str = DEFAULT_CLI_MODEL
    timeout: int = 300


@dataclass
class EncoderResponse:
    """Output from an encoding operation."""

    rac_content: str
    success: bool
    error: Optional[str] = None
    duration_ms: int = 0
    tokens: Optional[TokenUsage] = None


@dataclass
class PredictionScores:
    """Predicted scores from the encoder."""

    rac_reviewer: float = 7.0
    formula_reviewer: float = 7.0
    parameter_reviewer: float = 7.0
    integration_reviewer: float = 7.0
    ci_pass: bool = True
    policyengine_match: Optional[float] = None
    taxsim_match: Optional[float] = None
    confidence: float = 0.5


class EncoderBackend(ABC):
    """Abstract base class for encoder backends."""

    @abstractmethod
    def encode(self, request: EncoderRequest) -> EncoderResponse:
        """Encode a statute to RAC format (synchronous)."""
        pass  # pragma: no cover

    @abstractmethod
    def predict(self, citation: str, statute_text: str) -> PredictionScores:
        """Predict quality scores before encoding."""
        pass  # pragma: no cover


class ClaudeCodeBackend(EncoderBackend):
    """
    Backend using Claude Code CLI (subprocess).

    Works with Max subscription - no API billing.
    Uses embedded prompts -- no external plugin needed.
    """

    def __init__(
        self,
        cwd: Optional[Path] = None,
    ):
        self.cwd = cwd or Path.cwd()

    def encode(self, request: EncoderRequest) -> EncoderResponse:
        """Encode using Claude Code CLI with embedded encoder prompt."""
        start = time.time()

        prompt = get_encoder_prompt(request.citation, str(request.output_path))
        prompt += f"\n\nStatute Text:\n{request.statute_text}\n"

        output, returncode = self._run_claude_code(
            prompt=prompt,
            model=request.model,
            timeout=request.timeout,
        )

        duration_ms = int((time.time() - start) * 1000)

        if returncode != 0:
            return EncoderResponse(
                rac_content="",
                success=False,
                error=output,
                duration_ms=duration_ms,
            )

        # Check if file was created
        if request.output_path.exists():
            rac_content = request.output_path.read_text()
        else:
            rac_content = output

        # Clean up markdown code blocks
        rac_content = re.sub(r"^```\w*\n", "", rac_content)
        rac_content = re.sub(r"\n```$", "", rac_content)
        rac_content = rac_content.strip()

        return EncoderResponse(
            rac_content=rac_content,
            success=True,
            error=None,
            duration_ms=duration_ms,
        )

    def predict(self, citation: str, statute_text: str) -> PredictionScores:
        """Predict scores using Claude Code CLI."""
        prompt = f"""Predict quality scores for encoding the following statute into RAC DSL.

Citation: {citation}

Statute Text:
{statute_text[:2000]}{"..." if len(statute_text) > 2000 else ""}

Score each dimension from 1-10. Output ONLY valid JSON:
{{
  "rac_reviewer": <float 1-10>,
  "formula_reviewer": <float 1-10>,
  "parameter_reviewer": <float 1-10>,
  "integration_reviewer": <float 1-10>,
  "ci_pass": <boolean>,
  "policyengine_match": <float 0-1>,
  "taxsim_match": <float 0-1>,
  "confidence": <float 0-1>
}}
"""

        try:
            output, returncode = self._run_claude_code(
                prompt=prompt,
                model=DEFAULT_CLI_MODEL,
                timeout=60,
            )

            # Parse JSON from output
            json_match = re.search(r"\{[^{}]*\}", output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in output")

            return PredictionScores(
                rac_reviewer=float(data.get("rac_reviewer", 7.0)),
                formula_reviewer=float(data.get("formula_reviewer", 7.0)),
                parameter_reviewer=float(data.get("parameter_reviewer", 7.0)),
                integration_reviewer=float(data.get("integration_reviewer", 7.0)),
                ci_pass=bool(data.get("ci_pass", True)),
                policyengine_match=data.get("policyengine_match"),
                taxsim_match=data.get("taxsim_match"),
                confidence=float(data.get("confidence", 0.5)),
            )

        except Exception:
            # Return defaults on error
            return PredictionScores(confidence=0.3)

    def _run_claude_code(
        self,
        prompt: str,
        model: str = DEFAULT_CLI_MODEL,
        timeout: int = 300,
    ) -> tuple[str, int]:
        """Run Claude Code CLI as subprocess."""
        cmd = ["claude", "--print", "--permission-mode", "acceptEdits"]

        if model:
            cmd.extend(["--model", model])

        cmd.extend(["-p", prompt])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.cwd,
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


class AgentSDKBackend(EncoderBackend):
    """
    Backend using Claude API (anthropic SDK).

    Requires ANTHROPIC_API_KEY - pay per token.
    Enables parallelization for batch encoding.
    Uses embedded prompts -- no external plugin needed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required for AgentSDKBackend")
        self.model = model or DEFAULT_MODEL

    def encode(self, request: EncoderRequest) -> EncoderResponse:
        """Synchronous encode using API (runs async under the hood)."""
        return asyncio.run(self.encode_async(request))

    async def encode_async(self, request: EncoderRequest) -> EncoderResponse:
        """Async encode using Claude API with embedded prompts."""
        start = time.time()

        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)

            prompt = get_encoder_prompt(request.citation, str(request.output_path))
            prompt += f"\n\nStatute Text:\n{request.statute_text}\n"

            response = await client.messages.create(
                model=self.model,
                max_tokens=16384,
                messages=[{"role": "user", "content": prompt}],
            )

            result_content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    result_content += block.text

            token_usage = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            duration_ms = int((time.time() - start) * 1000)

            # Check if file was created
            if request.output_path.exists():
                rac_content = request.output_path.read_text()
            else:
                rac_content = result_content

            return EncoderResponse(
                rac_content=rac_content,
                success=True,
                error=None,
                duration_ms=duration_ms,
                tokens=token_usage if token_usage.total_tokens > 0 else None,
            )

        except ImportError:
            return EncoderResponse(
                rac_content="",
                success=False,
                error="anthropic SDK not installed. Run: pip install anthropic",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return EncoderResponse(
                rac_content="",
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start) * 1000),
            )

    async def encode_batch(
        self,
        requests: List[EncoderRequest],
        max_concurrent: int = 5,
    ) -> List[EncoderResponse]:
        """
        Encode multiple statutes in parallel.

        This is the key advantage of the API backend - parallelization.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def encode_with_limit(req: EncoderRequest) -> EncoderResponse:
            async with semaphore:
                return await self.encode_async(req)

        tasks = [encode_with_limit(req) for req in requests]
        return await asyncio.gather(*tasks)

    def predict(self, citation: str, statute_text: str) -> PredictionScores:
        """Predict scores using API."""
        # For now, use defaults - prediction is less critical than encoding
        return PredictionScores(confidence=0.5)
