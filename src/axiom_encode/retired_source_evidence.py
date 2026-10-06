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
    if not isinstance(payload, dict) or payload.get("contract") not in {
        "retired-source-containment/v1",
        "retired-source-containment/v2",
        "retired-source-external-parameters/v1",
    }:
        raise ValueError("Invalid retired source admission report")
    relative = Path("source-admission") / lane / REPORT_NAME
    _secure_atomic_eval_write(artifact_root, relative, raw)
    return {"path": relative.as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}


def match_contained_segment(child: str, parent: str) -> dict | None:
    """Return replayable raw parent offsets, allowing only whitespace folding."""
    import re

    if not child.strip():
        return None
    if child in parent:
        start = parent.index(child)
        return {"start": start, "end": start + len(child)}
    normalized_child = " ".join(child.split())
    characters = []
    offsets = []
    for token in re.finditer(r"\S+", parent):
        if characters:
            characters.append(" ")
            offsets.append(token.start() - 1)
        characters.extend(token.group())
        offsets.extend(range(token.start(), token.end()))
    normalized_parent = "".join(characters)
    start = normalized_parent.find(normalized_child)
    if start < 0:
        return None
    end = start + len(normalized_child)
    return {
        "start": offsets[start],
        "end": offsets[end - 1] + 1,
        "normalization": "unicode-whitespace/v1",
        "normalized_segment_sha256": hashlib.sha256(
            normalized_child.encode()
        ).hexdigest(),
    }
