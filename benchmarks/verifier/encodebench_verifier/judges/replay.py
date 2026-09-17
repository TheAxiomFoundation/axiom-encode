"""Replay runner: serve pre-recorded responses from a file (tests, re-folds).

The response file is JSON: either ``{"<case_id>": {...JudgeResponse...}}``
or ``{"responses": {...}, "model": "...", "family": "..."}``. Cases without
an entry get a fail-closed error response, so a partial replay is visible as
errors rather than as passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..cases import VerifierCase
from .base import JudgeResponse, error_response


class ReplayRunner:
    family = "replay"

    def __init__(self, response_file: Path, *, name: str = "replay") -> None:
        self.response_file = Path(response_file)
        payload = json.loads(self.response_file.read_text())
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
            "response_count": len(self._responses),
            "supports_localization": self.supports_localization,
        }

    def judge(self, case: VerifierCase) -> JudgeResponse:
        entry = self._responses.get(case.case_id)
        if entry is None:
            return error_response(
                self.model, "missing_replay", f"no recorded response for {case.case_id}"
            )
        response = JudgeResponse.from_dict(entry)
        if not response.model:
            response.model = self.model
        return response
