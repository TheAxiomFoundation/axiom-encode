"""Regression tests for encoder output-path collisions (issue #71).

Three layers are validated independently:

- Layer 1: path computation splits dotted leaf segments (covered in
  ``tests/test_evals.py::test_source_identifier_handles_dotted_leaf_segments``).
- Layer 2: ``_run_single_eval`` derives the output path from the requested
  citation, not the resolved corpus citation_path, so a resolver fallback
  to a parent provision does not collapse sibling subsections onto one
  file (tests in this module).
- Layer 3: the apply-time guard refuses to overwrite a target whose
  declared ``corpus_citation_path`` differs from the incoming file
  (tests in this module).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from axiom_encode.cli import _enforce_no_apply_collision
from axiom_encode.harness.evals import (
    EvalPromptResponse,
    EvalWorkspace,
    _resolve_eval_output_path,
    _run_single_eval,
    parse_runner_spec,
    resolve_corpus_source_unit,
)
from tests.release_object_fixtures import bind_test_corpus_release

# ---------------------------------------------------------------------------
# Layer 2: requested-not-resolved drives the output path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "valid,malformed",
    [
        ("us-la/statute/47:294", "us-la/statute/47:.294"),
        ("us-la/statute/47:294.4", "us-la/statute/47:294..4"),
    ],
)
def test_louisiana_malformed_dotted_identity_cannot_alias_valid_path(
    valid,
    malformed,
):
    assert _resolve_eval_output_path(valid).suffix == ".yaml"
    with pytest.raises(ValueError, match="Invalid Louisiana title:section"):
        _resolve_eval_output_path(malformed)


def _make_workspace(root: Path) -> EvalWorkspace:
    root.mkdir(parents=True, exist_ok=True)
    source_text_file = root / "source.txt"
    source_text_file.write_text("operative source text\n")
    manifest_file = root / "context-manifest.json"
    manifest_file.write_text("{}")
    return EvalWorkspace(
        root=root,
        source_text_file=source_text_file,
        manifest_file=manifest_file,
    )


def test_run_single_eval_uses_requested_citation_even_when_resolver_falls_back(
    tmp_path: Path,
):
    """Resolver falling back to a parent must not collapse sibling subsections.

    Issue #71's deeper root cause: even after the path strategy splits dotted
    leaves, if the resolver returns a parent ``citation_path`` because the
    subsection-level corpus row is missing or had a body-null shape, the
    eval used to derive the output path from that resolved parent — losing
    the subsection identity. The fix routes the requested citation to the
    path computation.
    """
    requested_citation = "us-ca/regulation/mpp/63-503.132"
    parent_fallback_path = "us-ca/regulation/mpp/63-503"

    output_root = tmp_path / "out"
    workspace = _make_workspace(output_root / "_eval_workspaces" / "workspace")

    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    rules_path = tmp_path / "rules"
    rules_path.mkdir()
    corpus_path = tmp_path / "corpus"
    (corpus_path / "data/corpus/provisions").mkdir(parents=True)
    selector = corpus_path / "manifests/releases/path-collision-test-release.json"
    selector.parent.mkdir(parents=True)
    selector.write_text(
        json.dumps(
            {
                "name": "path-collision-test-release",
                "scopes": [
                    {
                        "jurisdiction": "us-ca",
                        "document_class": "regulation",
                        "version": "test-version",
                    }
                ],
            }
        )
    )
    provision_file = (
        corpus_path / "data/corpus/provisions/us-ca/regulation/test-version.jsonl"
    )
    provision_file.parent.mkdir(parents=True, exist_ok=True)
    provision_file.write_text(
        json.dumps(
            {
                "id": "parent-fallback",
                "citation_path": parent_fallback_path,
                "body": "63-503.132 operative source text\n63-503.133 next text",
                "jurisdiction": "us-ca",
                "document_class": "regulation",
                "version": "test-version",
                "source_path": "sources/us-ca/regulation/test-version",
                "source_as_of": "2026-01-01",
                "expression_date": "2026-01-01",
            }
        )
        + "\n"
    )
    corpus_release = bind_test_corpus_release(
        corpus_path,
        "path-collision-test-release",
        [("us-ca", "regulation", "test-version")],
    )
    parent_source_unit = resolve_corpus_source_unit(
        parent_fallback_path,
        corpus_release,
    )
    fake_source_unit = parent_source_unit

    response = EvalPromptResponse(
        text="format: rulespec/v1\nmodule:\n  summary: stub\nrules: []\n",
        duration_ms=1,
    )

    with (
        patch(
            "axiom_encode.harness.evals.resolve_corpus_source_unit",
            return_value=fake_source_unit,
        ),
        patch(
            "axiom_encode.harness.evals.prepare_eval_workspace",
            return_value=workspace,
        ),
        patch("axiom_encode.harness.evals._run_prompt_eval", return_value=response),
        patch("axiom_encode.harness.evals.evaluate_artifact", return_value=None),
        patch("axiom_encode.harness.evals._hydrate_eval_root"),
    ):
        result = _run_single_eval(
            citation=requested_citation,
            runner=parse_runner_spec("codex:gpt-5.5"),
            output_root=output_root,
            policy_path=policy_path,
            runtime_axiom_rules_path=rules_path,
            corpus_release=corpus_release,
            mode="cold",
            extra_context_paths=[],
            source_unit=fake_source_unit,
        )

    assert result.output_file.endswith("regulations/mpp/63-503/132.yaml"), (
        "Output path must reflect the requested subsection, not the parent "
        f"fallback. Got: {result.output_file}"
    )


# ---------------------------------------------------------------------------
# Layer 3: apply-time collision guard
# ---------------------------------------------------------------------------


def _write_rulespec(path: Path, *, corpus_citation_path: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            f"""\
            format: rulespec/v1
            module:
              summary: stub
              source_verification:
                corpus_citation_path: {corpus_citation_path}
            rules: []
            """
        )
    )
    return path


def test_collision_guard_no_op_when_target_missing(tmp_path: Path):
    source = _write_rulespec(
        tmp_path / "src.yaml",
        corpus_citation_path="us-ca/regulation/mpp/63-503.132",
    )
    _enforce_no_apply_collision(
        source_file=source, target_file=tmp_path / "target.yaml"
    )


def test_collision_guard_allows_overwrite_with_same_citation_path(tmp_path: Path):
    source = _write_rulespec(
        tmp_path / "src.yaml",
        corpus_citation_path="us-ca/regulation/mpp/63-503.132",
    )
    target = _write_rulespec(
        tmp_path / "target.yaml",
        corpus_citation_path="us-ca/regulation/mpp/63-503.132",
    )
    # Re-applying the same encode (re-run after a fix) must succeed.
    _enforce_no_apply_collision(source_file=source, target_file=target)


def test_unicode_dash_citation_reuses_ascii_target_path(tmp_path: Path):
    citation_path = "us/statute/42/1437c\u20131"
    relative_target = _resolve_eval_output_path(citation_path)
    source = _write_rulespec(
        tmp_path / "src.yaml",
        corpus_citation_path=citation_path,
    )
    target = _write_rulespec(
        tmp_path / relative_target,
        corpus_citation_path=citation_path,
    )

    assert relative_target == Path("statutes/42/1437c-1.yaml")
    assert citation_path in source.read_text()
    assert citation_path in target.read_text()
    _enforce_no_apply_collision(source_file=source, target_file=target)


def test_collision_guard_rejects_unattested_uk_alias(tmp_path: Path):
    source = _write_rulespec(
        tmp_path / "src.yaml",
        corpus_citation_path="uk/statute/ukpga/1992/4/8",
    )
    target = _write_rulespec(
        tmp_path / "target.yaml",
        corpus_citation_path="uk/statute/legislation.gov.uk/ukpga/1992/4/section/8",
    )

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        _enforce_no_apply_collision(source_file=source, target_file=target)


def test_collision_guard_refuses_overwrite_with_different_citation_path(
    tmp_path: Path,
):
    source = _write_rulespec(
        tmp_path / "src.yaml",
        corpus_citation_path="us-ca/regulation/mpp/63-503.131",
    )
    target = _write_rulespec(
        tmp_path / "target.yaml",
        corpus_citation_path="us-ca/regulation/mpp/63-503.132",
    )
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        _enforce_no_apply_collision(source_file=source, target_file=target)


def test_collision_guard_silent_when_citation_path_unparseable(tmp_path: Path):
    # If we cannot parse a corpus citation out of either file, the guard
    # defers to the rest of the pipeline rather than blocking on unparseable
    # YAML — the canonical-concept validator already covers that case.
    source = tmp_path / "src.yaml"
    source.write_text("not: [valid yaml: at all")
    target = _write_rulespec(
        tmp_path / "target.yaml",
        corpus_citation_path="us-ca/regulation/mpp/63-503.132",
    )
    _enforce_no_apply_collision(source_file=source, target_file=target)
