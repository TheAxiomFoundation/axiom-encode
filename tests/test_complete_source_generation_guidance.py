"""Generator-side complete-source guidance: encode omitted branches first.

These tests cover prompt and retry-feedback text only. No validator rule,
message, or acceptance decision changes with this guidance.
"""

import functools

import pytest

from axiom_encode import checks
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


def _typed_completeness_feedback():
    """Run the real completeness analyzer on a candidate that omits (a) and
    defers the computation in (b) for a missing input."""

    import yaml

    from axiom_encode.harness.validator_pipeline import (
        extract_named_scalar_occurrences,
        extract_typed_numeric_inventory_occurrences_from_text,
        extract_typed_numeric_occurrences_from_text,
        numeric_value_is_grounded,
    )

    payload = {
        "format": "rulespec/v1",
        "module": {
            "summary": "Notice and allotment.",
            "source_verification": {"corpus_citation_path": _GUIDANCE_CITATION},
            "deferred_outputs": [
                {
                    "output": "us:statutes/7/2020/b#allotment",
                    "reason": "7 U.S.C. 2020(b) requires the household net "
                    "income, which is not supplied in the available context.",
                }
            ],
        },
        "rules": [],
    }
    analysis = sc.analyze_complete_source_unit(
        yaml.safe_dump(payload, sort_keys=False),
        _GUIDANCE_SOURCE,
        corpus_citation_path=_GUIDANCE_CITATION,
        test_cases=None,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return list(analysis.issues)


def test_typed_completeness_retry_feedback_has_one_repair_order():
    """Main's CHECK FIXES block and the completeness repair block agree.

    The declared fixes of the encode-or-defer checks offer deferral (or a
    reworded reason) as an equal repair; with the completeness block present
    they point at its encode-first order instead, so the retry has one order.
    """

    issues = _typed_completeness_feedback()
    check_ids = [getattr(issue, "check", None) for issue in issues]
    assert {
        "complete-source-unit:deferral/reason-imprecise",
        "complete-source-unit:structure/branch-unencoded",
        "complete-source-unit:formula-output/branch-without-output",
        "complete-source-unit:numeric-recall/value-unnamed",
    } <= set(check_ids)
    # The finding asks for a literal citation in `reason`; that hint must not
    # read as an exact fix that outranks encoding.
    assert any("the literal citation required in `reason`" in i for i in issues)

    rendered = evals._format_validation_retry_feedback(
        issues,
        deferral_anchor=evals._complete_source_deferral_anchor(_GUIDANCE_CITATION),
    )
    normalized = _normalized(rendered)

    # Section order: guidance, verbatim findings, then main's check fixes.
    positions = [
        normalized.index(marker)
        for marker in (
            "Completeness repair:",
            "=== BEGIN PRIOR VALIDATION FEEDBACK ===",
            "=== END PRIOR VALIDATION FEEDBACK ===",
            "=== BEGIN CHECK FIXES (declared by each failed check) ===",
            "=== END CHECK FIXES ===",
        )
    ]
    assert positions == sorted(positions)
    assert normalized.count("Completeness repair:") == 1
    assert normalized.count("Keep or add a deferral only when:") == 1

    # The exact-fix carve-out covers paths and proof atoms, not a reason rewrite.
    assert "a required literal citation in `reason`" not in normalized
    assert (
        "A literal citation that a finding requires in `reason` applies only to "
        "a deferral that one of those grounds supports."
    ) in normalized

    check_fixes = rendered.split("=== BEGIN CHECK FIXES", 1)[1]
    overrides = evals._COMPLETENESS_ENCODE_FIRST_FIXES
    for check_id in set(check_ids) & set(overrides):
        assert check_fixes.count(f"- [{check_id}]") == 1
        assert f"  Fix: {overrides[check_id]}" in check_fixes
    # No declared encode-or-defer fix survives next to the encode-first order.
    for check_id in overrides:
        assert checks.get_check(check_id).fix not in check_fixes
    # The subject lines name which subjects to repair, not how.
    assert "per-subject lines name which subjects to repair; this order says how" in (
        normalized
    )
    # Other checks keep their declared fix, and every subject line stays.
    numeric_fix = checks.get_check("complete-source-unit:numeric-recall/value-unnamed")
    assert f"  Fix: {numeric_fix.fix}" in check_fixes
    assert "us/statute/7/2020(a) [Absatz a] chars 0:141; expected encoded" in (
        check_fixes
    )
    assert "field:module.deferred_outputs[0].reason; expected reason naming" in (
        check_fixes
    )


def test_check_fixes_keep_declared_text_without_completeness_guidance():
    """Main's #1987 rendering is unchanged when the repair block is absent."""

    from axiom_encode.checks import IssueSubject, ValidationIssue, load_checks

    load_checks()
    issue = ValidationIssue(
        "[complete-source-unit:tests] Principal output `x` is never asserted by a "
        "companion test.",
        check="complete-source-unit:tests/output-unasserted",
        subject=IssueSubject(citation="us/statutes/7/2014/e", rule="x"),
        expected="at least one asserting companion case",
        found="no case asserts it",
    )
    rendered = evals._format_validation_retry_feedback((issue,))

    assert "Completeness repair:" not in rendered
    assert rendered.endswith(evals._format_check_fix_feedback((issue,)))
    fix = checks.get_check("complete-source-unit:tests/output-unasserted").fix
    assert f"  Fix: {fix}" in rendered


@pytest.mark.parametrize("check_id", sorted(evals._COMPLETENESS_ENCODE_FIRST_FIXES))
def test_encode_first_fix_overrides(check_id):
    """Each override keeps encoding first, names the deferral grounds, and is
    used only when the completeness repair block renders."""

    from axiom_encode.checks import IssueSubject, ValidationIssue, load_checks

    load_checks()
    category = check_id.split("/", 1)[0]
    assert f"[{category}]" in evals._COMPLETENESS_REPAIR_CATEGORIES
    override = evals._COMPLETENESS_ENCODE_FIRST_FIXES[check_id]
    assert override.startswith(("Encode the", "Add a principal derived rule"))
    assert "deferral ground listed" in override
    assert "Completeness repair guidance above" in override or "there" in override

    issue = ValidationIssue(
        f"[{category}] finding.",
        check=check_id,
        subject=IssueSubject(citation="us/statutes/7/2020", branch="b"),
        expected="x",
        found="y",
    )
    declared = checks.get_check(check_id).fix
    # Main's #1987 rendering, unchanged without overrides.
    plain = evals._format_check_fix_feedback((issue,))
    assert f"  Fix: {declared}" in plain
    assert override not in plain
    # The retry renders the override with the completeness block.
    rendered = evals._format_validation_retry_feedback((issue,))
    assert "Completeness repair:" in rendered
    assert f"  Fix: {override}" in rendered
    assert declared not in rendered


def test_issue_fix_feedback_override_replaces_only_named_fix():
    from axiom_encode.checks import IssueSubject, ValidationIssue, load_checks

    load_checks()
    issue = ValidationIssue(
        "[complete-source-unit:tests] Principal output `x` is never asserted.",
        check="complete-source-unit:tests/output-unasserted",
        subject=IssueSubject(citation="us/statutes/7/2014/e", rule="x"),
        expected="e",
        found="f",
    )
    lines = checks.issue_fix_feedback((issue,))
    assert checks.issue_fix_feedback((issue,), fix_overrides={}) == lines
    assert (
        checks.issue_fix_feedback((issue,), fix_overrides={"other/check": "Z"}) == lines
    )
    replaced = checks.issue_fix_feedback(
        (issue,),
        fix_overrides={"complete-source-unit:tests/output-unasserted": "Do Z."},
    )
    assert replaced[1] == "  Fix: Do Z."
    assert replaced[0] == lines[0] and replaced[2:] == lines[2:]
