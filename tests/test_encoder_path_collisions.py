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

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from axiom_encode.cli import (
    _enforce_no_apply_collision,
    _read_corpus_citation_paths_from_rulespec,
)
from axiom_encode.harness.evals import (
    CorpusSourceUnit,
    EvalPromptResponse,
    EvalWorkspace,
    _run_single_eval,
    parse_runner_spec,
)

# ---------------------------------------------------------------------------
# Layer 2: requested-not-resolved drives the output path
# ---------------------------------------------------------------------------


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

    fake_source_unit = CorpusSourceUnit(
        requested=requested_citation,
        citation_path=parent_fallback_path,
        body="operative source text",
        source="local",
    )
    workspace = _make_workspace(tmp_path / "workspace")

    output_root = tmp_path / "out"
    policy_path = tmp_path / "policy"
    policy_path.mkdir()
    rules_path = tmp_path / "rules"
    rules_path.mkdir()
    corpus_path = tmp_path / "corpus"
    corpus_path.mkdir()

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
            corpus_path=corpus_path,
            mode="cold",
            extra_context_paths=[],
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


def test_collision_guard_allows_uk_legislation_gov_alias(tmp_path: Path):
    source = _write_rulespec(
        tmp_path / "src.yaml",
        corpus_citation_path="uk/statute/ukpga/1992/4/8",
    )
    target = _write_rulespec(
        tmp_path / "target.yaml",
        corpus_citation_path="uk/statute/legislation.gov.uk/ukpga/1992/4/section/8",
    )

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


def test_read_corpus_citation_paths_picks_up_plural_form(tmp_path: Path):
    path = tmp_path / "rs.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            format: rulespec/v1
            module:
              source_verification:
                corpus_citation_paths:
                  - us-ca/regulation/mpp/63-503.132
                  - us-ca/regulation/mpp/63-503
            """
        )
    )
    assert _read_corpus_citation_paths_from_rulespec(path) == {
        "us-ca/regulation/mpp/63-503.132",
        "us-ca/regulation/mpp/63-503",
    }


def test_collision_guard_allows_overwrite_when_paths_share_an_entry(
    tmp_path: Path,
):
    # Re-encoding a child whose target previously declared the same path
    # alongside a parent (multi-path source_verification) should pass.
    source = _write_rulespec(
        tmp_path / "src.yaml",
        corpus_citation_path="us-ca/regulation/mpp/63-503.132",
    )
    target = tmp_path / "target.yaml"
    target.write_text(
        textwrap.dedent(
            """\
            format: rulespec/v1
            module:
              source_verification:
                corpus_citation_paths:
                  - us-ca/regulation/mpp/63-503.132
                  - us-ca/regulation/mpp/63-503
            rules: []
            """
        )
    )
    _enforce_no_apply_collision(source_file=source, target_file=target)
