"""Replay runner: serve pre-recorded responses from a file (tests, re-folds).

The response file is JSON: either ``{"<case_id>": {...JudgeResponse...}}``
or ``{"responses": {...}, "model": "...", "family": "...",
"supports_localization": true}``. Cases without an entry, and entries that
break the response contract (unknown verdict, scores outside [0, 1], an
error without a reason), become fail-closed error responses, so a partial or
drifted replay is visible as errors rather than as passes. A scored entry
that omits a kind score is not an error: that kind falls back to the verdict
score on the ``verdict_fallback`` channel, as a live judge without a
kind-specific question does. The runner's identity includes the response
file's sha256, so two different recordings never share resumed rows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .. import DEFECT_KINDS
from ..cases import VerifierCase
from .base import (
    CHANNEL_VERDICT_FALLBACK,
    VERDICT_ERROR,
    JudgeResponse,
    checked,
    error_response,
)


class ReplayRunner:
    family = "replay"

    def __init__(self, response_file: Path, *, name: str = "replay") -> None:
        self.response_file = Path(response_file)
        raw = self.response_file.read_bytes()
        self.response_sha256 = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        self.supports_localization = True
        if isinstance(payload, dict) and "responses" in payload:
            self._responses: dict[str, Any] = dict(payload["responses"])
            self.model = str(payload.get("model") or "replay")
            self.family = str(payload.get("family") or "replay")
            self.supports_localization = bool(
                payload.get("supports_localization", True)
            )
        elif isinstance(payload, dict):
            self._responses = dict(payload)
            self.model = "replay"
        else:
            raise ValueError("replay file must be a JSON object")
        self.name = name

    def identity(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "model": self.model,
            "response_file": self.response_file.name,
            "response_sha256": self.response_sha256,
            "response_count": len(self._responses),
            "supports_localization": self.supports_localization,
        }

    def judge(self, case: VerifierCase) -> JudgeResponse:
        entry = self._responses.get(case.case_id)
        if entry is None:
            return error_response(
                self.model, "missing_replay", f"no recorded response for {case.case_id}"
            )
        if not isinstance(entry, dict):
            return error_response(self.model, "replay_schema", "entry is not an object")
        response = JudgeResponse.from_dict(entry)
        if not response.model:
            response.model = self.model
        if response.verdict != VERDICT_ERROR and response.verdict_score is not None:
            # A recording may carry only a verdict score; kinds it omits fall
            # back to it, exactly as a live judge without a kind question does.
            for kind in DEFECT_KINDS:
                if response.kind_scores.get(kind) is None:
                    response.kind_scores[kind] = response.verdict_score
                    response.kind_score_channels[kind] = CHANNEL_VERDICT_FALLBACK
                response.kind_score_channels.setdefault(kind, CHANNEL_VERDICT_FALLBACK)
        return checked(response)
