"""Generator-side complete-source guidance: encode omitted branches first.

These tests cover prompt and retry-feedback text only. No validator rule,
message, or acceptance decision changes with this guidance.
"""

import pytest

from axiom_encode.harness import evals
from axiom_encode.harness import source_completeness as sc
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
    assert "A new deferral, or a reworded reason on a rejected one, is not." in (
        normalized
    )
    assert "apply exactly that fix" in normalized
    assert "follow the cross-reference rules in this prompt" in normalized
    assert "only an administrative duty" in normalized
    assert "`output: <existing output>` and `repair_remove: true`" in normalized
    assert "declare each fact the branch states" in normalized.lower()
    assert "non-operative" not in normalized
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
        "For a branch that states a condition, amount, or computation, a missing "
        "input, a missing date type, or a missing relation is not a deferral ground."
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


def test_retry_feedback_without_completeness_findings_is_byte_stable():
    # Frozen from origin/main 68ab3f399: the guidance adds no text, not even a
    # blank line, when no completeness finding is present.
    assert evals._format_validation_retry_feedback(("plain issue",)) == (
        "\nDeterministic validation feedback for the rejected candidate below:\n"
        "- This is repair guidance from the validator, not legal authority. Keep "
        "the\n  authoritative source and release-bound corpus evidence as the "
        "sole basis for\n  legal facts and values.\n"
        "- Correct every listed issue in this candidate. Do not repeat the "
        "rejected\n  pattern.\n\n\n"
        "=== BEGIN PRIOR VALIDATION FEEDBACK ===\n"
        '- "plain issue"\n'
        "=== END PRIOR VALIDATION FEEDBACK ===\n"
    )


_GUIDANCE_SOURCE = (
    "(a) Notice. The State agency shall send each household a written notice "
    "of the hearing before the household must file the quarterly report.\n\n"
    "(b) Amount. The allotment equals 30 percent of the household's net income."
)
_GUIDANCE_CITATION = "us/statute/7/2020"
_ADMINISTRATIVE_REASON = (
    "7 U.S.C. 2020(a) requires the written hearing notice and quarterly report "
    "filing dates for each household; that notice runtime capability is not "
    "supplied in the available context."
)


def _coverage(output, reason, citation=_GUIDANCE_CITATION):
    payload = {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {"corpus_citation_path": citation},
            "deferred_outputs": [{"output": output, "reason": reason}],
        },
        "rules": [],
    }
    return sc._deferred_coverage(
        payload,
        corpus_citation_path=citation,
        source_text=_GUIDANCE_SOURCE,
        branches=sc.recognize_source_structure(_GUIDANCE_SOURCE),
    )


def test_guidance_agrees_with_validator_on_administrative_runtime_gap():
    """Ground (3) in the stated form and slash path is accepted as coverage."""

    anchor = evals._complete_source_deferral_anchor(_GUIDANCE_CITATION)
    covered, issues = _coverage(f"{anchor}/a#hearing_notice", _ADMINISTRATIVE_REASON)

    assert covered == {("a",)}
    assert issues == []


def test_guidance_agrees_with_validator_on_missing_input_reasons():
    """A computation branch deferred for a missing input is rejected."""

    anchor = evals._complete_source_deferral_anchor(_GUIDANCE_CITATION)
    covered, issues = _coverage(
        f"{anchor}/b#allotment",
        "7 U.S.C. 2020(b) requires the household net income, which is not "
        "supplied in the available context.",
    )

    assert covered == set()
    assert any("does not name an exact missing dependency" in i for i in issues)


def test_guidance_agrees_with_validator_on_parenthesized_paths():
    """The forbidden `a(1)(A)`-style output path covers no branch."""

    anchor = evals._complete_source_deferral_anchor(_GUIDANCE_CITATION)
    covered, _issues = _coverage(f"{anchor}(a)#hearing_notice", _ADMINISTRATIVE_REASON)

    assert covered == set()


def test_runtime_gap_ground_is_limited_to_the_classes_the_validator_reads():
    """Ground (3) names statutes and federal regulations only; state rules fail."""

    citation = "us-ny/regulation/18-nycrr/387/14"
    anchor = evals._complete_source_deferral_anchor(citation)
    covered, _issues = _coverage(
        f"{anchor}/a#hearing_notice",
        f"{citation}(a) requires the written hearing notice and quarterly report "
        "filing dates for each household; that notice runtime capability is not "
        "supplied in the available context.",
        citation=citation,
    )
    rendered = _normalized(evals._format_completeness_repair_guidance(anchor))

    assert covered == set()
    assert "the source is a statute or a U.S. federal regulation" in rendered
    assert "In other source classes, cover such an administrative branch" in rendered


def test_guidance_points_to_the_repeal_form_without_direction_words():
    rendered = _normalized(evals._format_branch_coverage_plan("us:statutes/7/2020"))

    assert "in the repeal-tombstone form in the complete-source rules" in rendered
    assert "repeal form above" not in rendered


def test_checklist_non_operative_text_is_gated_on_complete_source_mode():
    source = (
        "(a) Purpose. This part promotes welfare.\n(b) Second category is ineligible."
    )
    citation = "us-ny/regulation/18-nycrr/387/14/a/5"
    default = evals._format_subparagraph_coverage_checklist(source, citation)
    complete = evals._format_subparagraph_coverage_checklist(
        source, citation, complete_source_unit=True
    )

    assert "satisfy this checklist with\n`module.deferred_outputs[]`" in default
    assert "satisfy this checklist with\n`module.deferred_outputs[]`" not in complete
    assert "a\nnon-operative deferral is rejected" in complete


@pytest.mark.parametrize(
    "render",
    [
        lambda: evals._format_branch_coverage_plan("us:statutes/7/2020"),
        lambda: evals._format_completeness_repair_guidance("us:statutes/7/2020"),
    ],
    ids=["first-attempt-plan", "repair-block"],
)
def test_encoded_branches_get_companion_test_guidance(render):
    rendered = _normalized(render())

    assert "For every branch you newly encode, add its companion cases" in rendered
    assert "changes only the branch's controlling local input" in rendered
    assert "including false and zero facts" in rendered
    assert "re-emit every existing named companion case with that input" in rendered
    assert "never a bare or quoted `YYYY-MM-DD`" in rendered
    assert "one contiguous, verbatim span of `./source.txt`" in rendered
    # Accuracy guards: tests never supply values the model must state, and
    # framing text is not encoded as a constant true output.
    assert "Never choose an expected value to make a case pass" in rendered
    assert "never state a source amount, rate, or threshold only in a test" in (
        rendered
    )
    assert "a constant `true` just to cover a branch" in rendered
    assert "`#input.program_is_separate_child_health_program`" in rendered


def test_repair_block_names_the_deferral_record_shape():
    rendered = _normalized(
        evals._format_completeness_repair_guidance("us:statutes/7/2020")
    )

    assert "omit `source_values` unless every entry is an absolute" in rendered


@pytest.mark.parametrize(
    ("period", "accepted"),
    [
        ("2026-07", True),
        (
            {"period_kind": "month", "start": "2026-07-01", "end": "2026-07-31"},
            True,
        ),
        ("2026-07-01", False),
        ("2026", False),
    ],
)
def test_period_guidance_agrees_with_validator(period, accepted):
    """The forms the guidance allows parse; a quoted full date does not."""

    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    if accepted:
        assert ValidatorPipeline._coerce_rulespec_period(None, period)
    else:
        with pytest.raises(ValueError):
            ValidatorPipeline._coerce_rulespec_period(None, period)
