"""Known-good artifacts from the encoder track's own passing outputs.

An ``eval-suite`` output directory carries a canonical ``results.json`` (the
v6 payload ``eval-board`` folds) and, under ``_eval_workspaces/``, the exact
``source.txt`` each runner was shown. A case counts as known-good when it
passed the deterministic gate battery (``result_gate_pass``: encode success,
compile, CI, zero ungrounded numerics), which is the encoder track's headline
definition of a pass.

The payload is loaded through the board's own strict loader so a
non-canonical results file cannot seed a benchmark.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from axiom_encode.harness.eval_board import (
    EvalBoardError,
    load_eval_suite_results,
    resolve_board_input_path,
    result_gate_pass,
)

from . import KnownGoodArtifact


class EvalSuiteSourceError(ValueError):
    """An eval-suite output could not be used as a known-good source."""


def _slugify(value: str) -> str:
    # Copied from axiom_encode.harness.evals._slugify (private there) so this
    # loader resolves the same workspace directory the suite wrote.
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "eval"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_known_good(
    output_dirs: list[Path],
) -> tuple[list[KnownGoodArtifact], dict[str, Any]]:
    """Collect gate-pass artifacts from one or more suite output directories."""

    artifacts: list[KnownGoodArtifact] = []
    suite_names: set[str] = set()
    corpus_releases: set[str] = set()
    results_sha256s: list[str] = []
    for raw in output_dirs:
        results_path = resolve_board_input_path(Path(raw))
        output_root = results_path.parent
        try:
            payload = load_eval_suite_results(results_path)
        except EvalBoardError as exc:
            raise EvalSuiteSourceError(str(exc)) from exc
        results_sha256s.append(_sha256_bytes(results_path.read_bytes()))
        manifest = payload["evidence"]["manifest"]
        suite_names.add(str(manifest.get("name")))
        corpus = payload["evidence"].get("corpus") or {}
        if corpus.get("corpus_release"):
            corpus_releases.add(str(corpus["corpus_release"]))
        for row in payload.get("results", []):
            if not isinstance(row, dict) or not result_gate_pass(row):
                continue
            runner = str(row.get("runner"))
            citation = str(row.get("citation") or "")
            output_file = Path(str(row.get("output_file") or ""))
            if not output_file.is_absolute():
                output_file = output_root / output_file
            if not output_file.is_file():
                raise EvalSuiteSourceError(
                    f"gate-pass result for {runner} / {citation} names a missing "
                    f"artifact {output_file}"
                )
            artifact_bytes = output_file.read_bytes()
            if _sha256_bytes(artifact_bytes) != row.get("generated_output_sha256"):
                raise EvalSuiteSourceError(
                    f"artifact {output_file} does not match its recorded "
                    "generated_output_sha256"
                )
            source_file = (
                output_root
                / "_eval_workspaces"
                / runner
                / _slugify(citation)
                / "workspace"
                / "source.txt"
            )
            if not source_file.is_file():
                raise EvalSuiteSourceError(
                    f"no workspace source text for {runner} / {citation} at "
                    f"{source_file}"
                )
            eval_case = row.get("eval_case") or {}
            artifacts.append(
                KnownGoodArtifact(
                    key=f"{runner}:{citation}",
                    citation=citation,
                    provision_text=source_file.read_text(),
                    artifact_text=artifact_bytes.decode("utf-8"),
                    origin={
                        "source": "eval_suite",
                        "suite_name": manifest.get("name"),
                        "runner": runner,
                        "generator_model": row.get("model"),
                        "backend": row.get("backend"),
                        "case_name": eval_case.get("name"),
                        "case_sha256": eval_case.get("sha256"),
                        "generated_output_sha256": row.get("generated_output_sha256"),
                        "corpus_release": corpus.get("corpus_release"),
                    },
                )
            )
    if not artifacts:
        raise EvalSuiteSourceError("no gate-pass artifacts found in the given outputs")
    identity = {
        "suite_names": sorted(suite_names),
        "corpus_releases": sorted(corpus_releases),
        "results_sha256s": results_sha256s,
        "artifact_count": len(artifacts),
    }
    return artifacts, identity
