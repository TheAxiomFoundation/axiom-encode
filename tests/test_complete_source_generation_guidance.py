"""Generator-side complete-source guidance: encode omitted branches first.

These tests cover prompt and retry-feedback text only. No validator rule,
message, or acceptance decision changes with this guidance.
"""

import pytest

from axiom_encode.harness import evals
from axiom_encode.harness.evals import EvalWorkspace


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _workspace(tmp_path, source_text, corpus_citation_path=None):
    source_file = tmp_path / "source.txt"
    source_file.write_text(source_text)
    metadata = (
        {"source_attestation": {"requested_corpus_citation_path": corpus_citation_path}}
        if corpus_citation_path
        else None
    )
    return EvalWorkspace(
        root=tmp_path,
        source_text_file=source_file,
        manifest_file=tmp_path / "context-manifest.json",
        source_metadata=metadata,
    )


def _prompt(tmp_path, *, complete, source_text, corpus_citation_path):
    return evals._build_eval_prompt(
        citation=corpus_citation_path,
        mode="cold",
        workspace=_workspace(tmp_path, source_text, corpus_citation_path),
        context_files=[],
        target_file_name="a.yaml",
        include_tests=True,
        runner_backend="codex",
        require_complete_source_unit=complete,
    )


@pytest.mark.parametrize(
    ("citation", "anchor"),
    [
        ("us/statute/26/2/a", "us:statutes/26/2/a"),
        ("us-nm/statute/7-2-18.13", "us-nm:statutes/7-2-18/13"),
        ("us/regulation/42/457/622", "us:regulations/42-cfr/457/622"),
        (None, "<module path>"),
    ],
)
def test_deferral_anchor_is_the_validators_preferred_anchor(citation, anchor):
    assert evals._complete_source_deferral_anchor(citation) == anchor


@pytest.mark.parametrize(
    "issue",
    [
        "ci: [complete-source-unit:structure] Source branch (2) at "
        "us/statute/26/2/a(2) is neither encoded nor precisely deferred.",
        "ci: [complete-source-unit:deferral] `module.deferred_outputs[0]` "
        "identifies source branch (1) (`1/a`) but its deferral does not name an "
        "exact missing dependency, input, or runtime capability.",
        "ci: [complete-source-unit:formula-output] Source-stated control (1) in "
        "us/statute/26/2/a(1) has no principal derived/relation output.",
    ],
)
def test_completeness_findings_add_encode_first_repair_guidance(issue):
    rendered = evals._format_validation_retry_feedback(
        (issue,), deferral_anchor="us:statutes/26/2/a"
    )
    normalized = _normalized(rendered)

    assert "Completeness repair:" in normalized
    assert "Encoding is the default repair." in normalized
    assert "Do not reword a rejected deferral of such a branch." in normalized
    assert "`output: <existing output>` and `repair_remove: true`" in normalized
    assert "declare each fact the branch tests" in normalized.lower()
    assert "`us:statutes/26/2/a/<marker>/<marker>#<name>`" in normalized
    assert "Never use parentheses such as `a(1)(A)`" in normalized
    # Guidance comes before the verbatim validator findings, which stay intact.
    assert normalized.index("Completeness repair:") < normalized.index(
        "=== BEGIN PRIOR VALIDATION FEEDBACK ==="
    )
    assert evals.json.dumps(issue, ensure_ascii=False) in rendered


@pytest.mark.parametrize(
    "issue",
    [
        "ci: [complete-source-unit:tests] Source-stated exceptions require "
        "paired positive/blocking cases differing in exactly one input",
        "ci: [complete-source-unit:numeric-recall] Authoritative corpus numeric "
        "value 2 has no named scalar representation.",
        "Generated RuleSpec failed CI validation",
    ],
)
def test_other_findings_omit_completeness_repair_guidance(issue):
    rendered = evals._format_validation_retry_feedback((issue,))

    assert "Completeness repair:" not in rendered
    assert "repair_remove: true" not in rendered


def test_retry_feedback_default_anchor_is_a_placeholder():
    rendered = evals._format_validation_retry_feedback(
        ("ci: [complete-source-unit:structure] Source branch is missing",)
    )

    assert "`<module path>/<marker>/<marker>#<name>`" in _normalized(rendered)


def test_empty_retry_feedback_renders_nothing():
    assert evals._format_validation_retry_feedback(()) == ""


def test_checklist_prefers_encoding_only_in_complete_source_mode():
    source = "(a) First category is eligible.\n(b) Second category is ineligible."
    citation = "us-ny/regulation/18-nycrr/387/14/a/5"

    default = evals._format_subparagraph_coverage_checklist(source, citation)
    complete = evals._format_subparagraph_coverage_checklist(
        source, citation, complete_source_unit=True
    )

    assert "When in doubt, prefer (b)" in default
    assert "the legal dependency or scope reason blocking encoding" in default
    assert "When in doubt, prefer (a)" not in default
    assert "When in doubt, prefer (a)" in complete
    assert "When in doubt, prefer (b)" not in complete
    assert "scope reason blocking encoding" not in complete
    assert "a deferral with a\nscope or generic reason is rejected" in complete
    # The exact validator keys are unchanged in both modes.
    assert "us-ny/regulation/18-nycrr/387/14/a/5(a)" in default
    assert "us-ny/regulation/18-nycrr/387/14/a/5(a)" in complete


def test_checklist_default_text_is_unchanged():
    source = "(a) First category is eligible.\n(b) Second category is ineligible."
    rendered = evals._format_subparagraph_coverage_checklist(
        source, "us-ny/regulation/18-nycrr/387/14/a/5"
    )

    assert (
        "whose `reason:` names\n      the legal dependency or scope reason "
        "blocking encoding.\n" in rendered
    )
    assert (
        "When in doubt, prefer (b): a `deferred_outputs` entry with a clear\n"
        "`reason` is honest and cheap to refine later. Silent omission is not.\n"
        "=== END SUBPARAGRAPH COVERAGE CHECKLIST ===" in rendered
    )


def test_complete_source_prompt_adds_branch_coverage_plan(tmp_path):
    source = (
        "(1) A taxpayer qualifies if the spouse died during either of the two "
        "preceding taxable years.\n(2) A taxpayer who remarried does not qualify."
    )
    complete = _prompt(
        tmp_path,
        complete=True,
        source_text=source,
        corpus_citation_path="us/statute/26/2/a",
    )
    normalized = _normalized(complete)

    assert "Branch coverage plan:" in normalized
    assert "ENCODE is the default." in normalized
    assert (
        "A missing input, a missing date type, or a missing relation is not a deferral ground."
        in normalized
    )
    assert "`us:statutes/26/2/a/<marker>/<marker>#<name>`" in normalized
    # The plan sits inside the complete-source block, after the deferral rules.
    assert (
        normalized.index("Complete-source-unit mode is enabled")
        < normalized.index("Never guess a blocker target.")
        < normalized.index("Branch coverage plan:")
        < normalized.index("A child-branch `source:` citation does not cover")
    )


def test_default_prompt_has_no_branch_coverage_plan(tmp_path):
    default = _prompt(
        tmp_path,
        complete=False,
        source_text="(1) The amount is 12.",
        corpus_citation_path="us/statute/26/2/a",
    )

    assert "Branch coverage plan:" not in default
    assert "Completeness repair:" not in default
