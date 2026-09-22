"""Durable review evidence for admission of retired replacement metadata."""

import hashlib
import json
from pathlib import Path

from .corpus_resolver import read_bounded_regular_file
from .harness.evals import _secure_atomic_eval_write

REPORT_NAME = "retired-source-admission.json"


def write_retired_source_evidence(output_root: Path, evidence: dict) -> None:
    _secure_atomic_eval_write(
        output_root,
        Path(REPORT_NAME),
        (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode(),
    )


def package_retired_source_evidence(
    generated_root: Path, lane: str, artifact_root: Path
) -> dict | None:
    """Copy the optional diagnostic report; it does not replace signed provenance."""
    if len(Path(lane).parts) != 1 or lane in {".", ".."}:
        raise ValueError("Invalid evidence lane")
    source = generated_root / lane / REPORT_NAME
    if not source.exists() and not source.is_symlink():
        return None
    raw = read_bounded_regular_file(
        generated_root, source, label="retired source admission", max_bytes=1024 * 1024
    )
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or payload.get("contract") != "retired-source-containment/v1"
    ):
        raise ValueError("Invalid retired source admission report")
    relative = Path("source-admission") / lane / REPORT_NAME
    _secure_atomic_eval_write(artifact_root, relative, raw)
    return {"path": relative.as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}
