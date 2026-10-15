"""Keep only the two complete corpus instructions exempt from paired cases.

The previous relaxation's source fixtures remain here. Its extra relation and
proviso recognition is deliberately removed: unsupported wording retains its
paired obligation instead of gaining a new way to discharge coverage.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import re

import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.manual_procedure_differential import (
    manual_procedure_requires_paired_witness,
)

GROUNDING = functools.partial(
    extract_typed_numeric_occurrences_from_text, profile="legacy"
)


# Page 8, characters 0:440, 488:682, 1903:2036 and 2209:2426.
PROCEDURE_CLAUSES = (
    "Chapter: 2600 Calculation of Benefits Program: FS 1 2610.0000 Food Stamps "
    "When the eligibility specialist has determined available income as per "
    "Chapters 1800 and 2400, the policy in this chapter must be used to perform "
    "the budgets and tests to determine eligibility for benefits and the benefit "
    "amount. 2610.0100 BUDGETS AND TEST CALCULATIONS (FS) Each program has "
    "budgets and tests that must be executed in order to determine eligibility.",
    "Refer to Appendix A-1 for the 165% Need Standard Table. 2610.0104.03 Gross "
    "Income Test (FS) When gross income is determined, that income is compared "
    "to the allowable gross income limit for the assistance group's size.",
)
MANUAL_CITATION = (
    "us-fl/manual/dcf/ess-program-policy-manual/2600-calculation-of-benefits/page-8"
)
# Exact existing page-8 body, including its paragraph breaks and no final newline.
AUTHENTICATED_SOURCE = (
    PROCEDURE_CLAUSES[0]
    + " These are discussed in the following sections. The income limits are found "
    "in Appendix A-1. 2610.0103 Budgets and Tests (FS) Assistance groups must meet "
    "the gross income standards to be eligible for food stamps with the following "
    "exceptions:\n\n"
    "1. assistance groups that contain an elderly or disabled member and are not "
    "categorically eligible must meet the net income limits; and 2. standard filing "
    "units (SFUs) that are broad-based categorically eligible must meet the 200% "
    "gross income limits. 2610.0104.01 Income Tests (FS) Income tests for "
    "determining food stamp eligibility are presented in passages 2610.0104.02 "
    "through 2610.0104.04. 2610.0104.02 165% Need Standard Eligibility Test (FS) "
    "A group of individuals who live together and purchase and prepare meals "
    "together normally constitute a single assistance group. However, if an "
    "otherwise eligible assistance group member who is 60 or older is unable to "
    "purchase and prepare meals because they suffer from a disability considered "
    "permanent under the Social Security Act, or suffers a non-disease related "
    "permanent disability, they may be a separate assistance group. A spouse or "
    "children under 18 who live with this individual may not be separate from the "
    "disabled individual.\n\n"
    "The individual (and spouse or children if required) may be a separate "
    "assistance group from the others, provided the income of the others with "
    "whom the individual resides does not exceed 165% of the gross eligibility "
    "standard. When applying the 165% limit test, the income of the elderly or "
    "disabled person and their spouse is excluded from this determination.\n\n"
    "The assistance group is responsible for providing information regarding "
    "the individual's inability to purchase and prepare meals since this is the "
    "key determining factor. "
    + PROCEDURE_CLAUSES[1]
    + " If the gross income is equal to or exceeds the gross income limit, the "
    "assistance group is ineligible. Refer to passage 2610.0103 for assistance "
    "groups not tested against the gross income limit. The gross income limits "
    "can be found in the on-line reference table or in Appendix A-1."
)


def _same_without_exemptions(verdict):
    actual = verdict()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sc, "_MANUAL_PROCEDURE_CLAUSES", frozenset())
        baseline = verdict()
    assert actual == baseline
    return actual


@pytest.mark.parametrize("clause", PROCEDURE_CLAUSES)
def test_complete_corpus_procedure_needs_no_paired_cases(clause):
    assert not manual_procedure_requires_paired_witness(
        clause, corpus_citation_path=MANUAL_CITATION
    )


@pytest.mark.parametrize("clause", PROCEDURE_CLAUSES)
@pytest.mark.parametrize(
    "citation",
    (
        "",
        "de/statute/estg/32a",
        "us-fl/guidance/dcf/ess-program-policy-manual/2600-calculation-of-benefits/page-8",
        "us-fl/manual/dcf/ess-program-policy-manual/2600-calculation-of-benefits/page-9",
        "us-ga/manual/test/section-1",
    ),
)
def test_corpus_wording_keeps_its_obligation_without_the_exact_manual_source(
    clause, citation
):
    assert manual_procedure_requires_paired_witness(
        clause, corpus_citation_path=citation
    )


@pytest.mark.parametrize(
    "clause",
    (
        "The income limits are found in Appendix A-1. 2610.0103 Budgets and Tests "
        "(FS) Assistance groups must meet the gross income standards to be eligible "
        "for food stamps with the following exceptions:",
        "When applying the 165% limit test, the income of the elderly or disabled "
        "person and their spouse is excluded from this determination.",
        "If the gross income is equal to or exceeds the gross income limit, the "
        "assistance group is ineligible.",
        "When income is determined to exceed the limit, the assistance group is "
        "ineligible.",
        "When gross income is determined, the assistance group is ineligible if "
        "it exceeds the limit.",
        "Assistance groups must meet the gross income standards with the "
        "following exceptions: households with an elderly member.",
        "When applying the 165% limit test, the income of the others is counted "
        "unless the individual is under 18.",
        # An operative exclusion glued into the same flattened clause keeps the
        # clause's obligation (2200 Standard Filing Unit, page 21).
        "2210.0324 Income and Assets Excluded (FS) The income and assets of "
        "excluded individuals such as roomers, live-in attendants, boarders, "
        "students, and individuals who do not purchase and prepare meals together "
        "are not considered available to the SFU. 2210.0325 Participation Status "
        "Codes (FS) When the SFU composition has been determined, a participation "
        "code is assigned to each individual.",
        "However, if an otherwise eligible assistance group member who is 60 or "
        "older is unable to purchase and prepare meals because they suffer from "
        "a disability considered permanent under the Social Security Act, or "
        "suffers a non-disease related permanent disability, they may be a "
        "separate assistance group.",
    ),
)
def test_toggleable_conditions_still_need_paired_cases(clause):
    assert manual_procedure_requires_paired_witness(clause)


def test_is_equal_to_or_exceeds_keeps_legacy_unsupported_relation():
    text = (
        "if the gross income is equal to or exceeds the gross income limit, the "
        "assistance group is ineligible."
    )
    assert (
        _same_without_exemptions(
            lambda: sc._source_relational_exception_match_indices(
                text,
                left_name="gross_income",
                relation=">=",
                right_name="gross_income_limit",
            )
        )
        == ()
    )
    (token,) = sc._source_relation_tokens(text)
    assert token.group("unsupported")
    assert _same_without_exemptions(
        lambda: bool(sc._source_relation_tokens(text)[0].group("unsupported"))
    )
    assert manual_procedure_requires_paired_witness(text)


@pytest.mark.parametrize(
    "phrase", ("is equal to or less than", "is not equal to", "does not equal")
)
def test_other_equal_to_phrases_stay_unsupported(phrase):
    (token,) = sc._source_relation_tokens(f"if income {phrase} the limit, stop.")
    assert token.group("unsupported")
    assert _same_without_exemptions(
        lambda: bool(
            sc._source_relation_tokens(f"if income {phrase} the limit, stop.")[0].group(
                "unsupported"
            )
        )
    )


PROVISO = (
    "The individual (and spouse or children if required) may be a separate "
    "assistance group from the others, provided the income of the others with "
    "whom the individual resides does not exceed 165% of the gross eligibility "
    "standard."
)


def _proviso_witness(blocks: bool) -> sc._ExceptionWitness:
    return sc._ExceptionWitness(
        rule_name="income_of_the_others_within_limit",
        selector_name="income_of_the_others",
        active_value=True,
        blocks=blocks,
        boolean_effect=True,
        zeroes=False,
        numeric_transition=(3300.0, 3301.0) if blocks else (3301.0, 3300.0),
        relational_transitions=(
            (
                "income_of_the_others",
                ">",
                "gross_eligibility_standard_need_standard_income_limit_rate",
            ),
        ),
    )


def test_postposed_proviso_does_not_add_withdrawing_side_admission():
    branch = sc.SourceStructureBranch(("1",), "number", "1.", PROVISO, 0, len(PROVISO))
    assert not _same_without_exemptions(
        lambda: sc._numeric_exception_witness_matches_source(
            branch, _proviso_witness(blocks=True), extract_numeric_occurrences=GROUNDING
        )
    )


def test_postposed_proviso_rejects_the_enabling_side():
    branch = sc.SourceStructureBranch(("1",), "number", "1.", PROVISO, 0, len(PROVISO))
    assert not _same_without_exemptions(
        lambda: sc._numeric_exception_witness_matches_source(
            branch,
            _proviso_witness(blocks=False),
            extract_numeric_occurrences=GROUNDING,
        )
    )


def test_unnegated_proviso_is_not_read_as_withdrawal():
    text = "The credit applies, provided the income exceeds the threshold."
    assert manual_procedure_requires_paired_witness(text)


def _analyze_exact_key(source, citation):
    # Reuse the regression suite's literal true formula and single enabled case.
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation}
rules:
  - name: result
    kind: derived
    description:
    source: {citation}
    versions:
      - formula: 'true'
"""
    return sc.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=[
            {
                "name": "{}",
                "period": "2026",
                "input": {},
                "output": {"result": True},
            }
        ],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


@pytest.mark.parametrize("clause", PROCEDURE_CLAUSES)
def test_exact_manual_keys_preserve_main_analyzer_obligations(clause):
    actual = sc._source_exception_requires_paired_witness(
        clause, corpus_citation_path=MANUAL_CITATION, source_text=clause
    )
    result = _analyze_exact_key(clause, MANUAL_CITATION)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sc, "_MANUAL_PROCEDURE_CLAUSES", frozenset())
        baseline = sc._source_exception_requires_paired_witness(
            clause, corpus_citation_path=MANUAL_CITATION, source_text=clause
        )
        baseline_result = _analyze_exact_key(clause, MANUAL_CITATION)
    assert baseline and not actual
    # Main's manual-only numeric-limit filter already omits both procedure
    # clauses before the paired-witness gate. Every other obligation survives.
    assert not any(
        "paired positive/blocking" in issue for issue in baseline_result.issues
    )
    assert result.issues == baseline_result.issues


def _manual_source_demands(source, *, numeric_limits_only=True):
    branches = sc.recognize_source_structure(
        source, corpus_citation_path=MANUAL_CITATION
    )
    exceptions = sc._source_exception_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        numeric_limits_only=numeric_limits_only,
    )
    return {
        (branch.path, branch.kind, branch.start, branch.end, branch.text)
        for branch in exceptions
        if sc._source_exception_requires_paired_witness(
            branch.text,
            branch_path=branch.path,
            corpus_citation_path=MANUAL_CITATION,
            source_text=source,
        )
    }


@pytest.mark.parametrize("numeric_limits_only", (False, True))
def test_authenticated_page_preserves_production_demands_and_exact_gate_keys(
    numeric_limits_only,
):
    assert hashlib.sha256(AUTHENTICATED_SOURCE.encode()).hexdigest() == (
        "a0b4fac87f905e6a6989b505b866f5f0306fe361531801c4226d9184e62723e5"
    )
    assert (
        hashlib.sha256(_normalized_key_text(AUTHENTICATED_SOURCE).encode()).hexdigest()
        == "e17c82b40e09f545b80f2cedfad3bfc91daea3eea87801f9cb3c10188fccf0a0"
    )
    actual_demands = _manual_source_demands(
        AUTHENTICATED_SOURCE, numeric_limits_only=numeric_limits_only
    )
    result = _analyze_exact_key(AUTHENTICATED_SOURCE, MANUAL_CITATION)
    assert all(
        not sc._source_exception_requires_paired_witness(
            clause,
            corpus_citation_path=MANUAL_CITATION,
            source_text=AUTHENTICATED_SOURCE,
        )
        for clause in PROCEDURE_CLAUSES
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sc, "_MANUAL_PROCEDURE_CLAUSES", frozenset())
        baseline_demands = _manual_source_demands(
            AUTHENTICATED_SOURCE, numeric_limits_only=numeric_limits_only
        )
        baseline_result = _analyze_exact_key(AUTHENTICATED_SOURCE, MANUAL_CITATION)
        assert all(
            sc._source_exception_requires_paired_witness(
                clause,
                corpus_citation_path=MANUAL_CITATION,
                source_text=AUTHENTICATED_SOURCE,
            )
            for clause in PROCEDURE_CLAUSES
        )
    expected_removals = {
        key
        for key in baseline_demands
        if _normalized_key_text(key[-1])
        in {_normalized_key_text(clause) for clause in PROCEDURE_CLAUSES}
    }
    # The unfiltered gate still recognizes exactly the two reviewed clauses.
    # Production uses main's numeric-limit filter, so neither is demanded there.
    assert len(expected_removals) == (0 if numeric_limits_only else 2)
    assert baseline_demands - actual_demands == expected_removals
    assert not actual_demands - baseline_demands
    assert tuple(
        issue for issue in result.issues if "paired positive/blocking" not in issue
    ) == tuple(
        issue
        for issue in baseline_result.issues
        if "paired positive/blocking" not in issue
    )
    assert sum("paired positive/blocking" in issue for issue in result.issues) == 1
    assert (
        sum("paired positive/blocking" in issue for issue in baseline_result.issues)
        == 1
    )


def _paired_witness_with_footer_first():
    """Reorder the actual gate's exemption and footer blocks for comparison."""

    tree = ast.parse(inspect.getsource(sc._source_exception_requires_paired_witness))
    (function,) = tree.body
    exemption_index = next(
        index
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Name) and node.id == "_MANUAL_PROCEDURE_CITATION"
            for node in ast.walk(statement.test)
        )
    )
    exemption = function.body.pop(exemption_index)
    footer_index = next(
        index
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Name) and node.id == "without_footer"
            for node in ast.walk(statement.test)
        )
    )
    function.body.insert(footer_index + 1, exemption)
    namespace = vars(sc).copy()
    exec(compile(tree, "<manual-procedure-footer-order>", "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("clause", PROCEDURE_CLAUSES)
@pytest.mark.parametrize("full_page", (False, True))
def test_footer_order_preserves_both_authenticated_manual_keys(clause, full_page):
    source = AUTHENTICATED_SOURCE if full_page else clause
    footer_first = _paired_witness_with_footer_first()
    kwargs = {"corpus_citation_path": MANUAL_CITATION, "source_text": source}
    assert not sc._source_exception_requires_paired_witness(clause, **kwargs)
    assert not footer_first(clause, **kwargs)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "Guidance documents lack the force and effect of law, unless "
            "expressly authorized by statute or incorporated into a contract.",
            False,
        ),
        (
            "USDA may not cite, use, or rely on any guidance that is not "
            "available through its guidance portal, except to establish historical facts.",
            False,
        ),
        (
            "Each additional member $757 $946 $871 through their guidance "
            "portal, except to establish historical facts.",
            False,
        ),
        (
            "Guidance documents lack the force and effect of law, unless expressly "
            "authorized by statute or applicants are eligible if citizens.",
            True,
        ),
        (
            "If household size exceeds eight, add $757 for each additional member. "
            "Guidance documents lack the force and effect of law, unless expressly "
            "authorized by statute or",
            True,
        ),
        *(
            (clause + " incorporated into a contract.", True)
            for clause in PROCEDURE_CLAUSES
        ),
    ),
)
def test_footer_handling_outside_exact_keys_matches_main(source, expected):
    footer_first = _paired_witness_with_footer_first()
    kwargs = {"corpus_citation_path": MANUAL_CITATION, "source_text": source}
    actual = sc._source_exception_requires_paired_witness(source, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sc, "_MANUAL_PROCEDURE_CLAUSES", frozenset())
        baseline = sc._source_exception_requires_paired_witness(source, **kwargs)
    assert actual is baseline is expected
    assert footer_first(source, **kwargs) is baseline


def _perturb_key(text, *, token_edit, operation, position, payload):
    if token_edit:
        tokens = tuple(re.finditer(r"\S+", text))
        token = tokens[position % len(tokens)]
        start, end = token.span()
        replacement = payload
        if operation == "insert":
            end = start
            replacement += " "
    else:
        start = min(position, len(text) - (operation != "insert"))
        end = start + (operation != "insert")
        replacement = payload[0]
    if operation == "delete":
        replacement = ""
    return text[:start] + replacement + text[end:]


def _normalized_key_text(text):
    return re.sub(r"\s+", " ", text).strip().lower()


@settings(max_examples=120, deadline=None, derandomize=True)
@example(
    clause_index=0,
    target="citation",
    token_edit=False,
    operation="replace",
    position=0,
    payload="US",
)
@example(
    clause_index=1,
    target="citation",
    token_edit=False,
    operation="replace",
    position=0,
    payload="US",
)
@example(
    clause_index=0,
    target="clause",
    token_edit=False,
    operation="insert",
    position=len(PROCEDURE_CLAUSES[0]),
    payload="x",
)
@example(
    clause_index=1,
    target="clause",
    token_edit=False,
    operation="insert",
    position=len(PROCEDURE_CLAUSES[1]),
    payload="x",
)
@example(
    clause_index=1,
    target="clause",
    token_edit=True,
    operation="replace",
    position=11,
    payload="Groß",
)
@given(
    clause_index=st.integers(min_value=0, max_value=1),
    target=st.sampled_from(
        (
            "citation",
            "clause",
            "context-prefix",
            "context-suffix",
            "page-context-prefix",
            "page-context-suffix",
        )
    ),
    token_edit=st.booleans(),
    operation=st.sampled_from(("insert", "replace", "delete")),
    position=st.integers(min_value=0, max_value=600),
    payload=st.sampled_from(("x", "resident", "unless", "page-9", "US", "0")),
)
def test_manual_procedure_perturbed_keys_equal_empty_list(
    clause_index, target, token_edit, operation, position, payload
):
    clause = PROCEDURE_CLAUSES[clause_index]
    citation = MANUAL_CITATION
    source = clause
    if target == "citation":
        citation = _perturb_key(
            citation,
            token_edit=token_edit,
            operation=operation,
            position=position,
            payload=payload,
        )
        assume(citation != MANUAL_CITATION)
    elif target == "clause":
        clause = _perturb_key(
            clause,
            token_edit=token_edit,
            operation=operation,
            position=position,
            payload=payload,
        )
        assume(
            _normalized_key_text(clause)
            != _normalized_key_text(PROCEDURE_CLAUSES[clause_index])
        )
        source = clause
    else:
        context_source = AUTHENTICATED_SOURCE if target.startswith("page-") else clause
        if token_edit:
            source = (
                f"{payload} {context_source}"
                if target.endswith("prefix")
                else f"{context_source} {payload}"
            )
        else:
            source = (
                payload[0] + context_source
                if target.endswith("prefix")
                else context_source + payload[0]
            )
        assert _normalized_key_text(source) != _normalized_key_text(context_source)

    actual = sc._source_exception_requires_paired_witness(
        clause, corpus_citation_path=citation, source_text=source
    )
    result = _analyze_exact_key(source, citation)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sc, "_MANUAL_PROCEDURE_CLAUSES", frozenset())
        baseline = sc._source_exception_requires_paired_witness(
            clause, corpus_citation_path=citation, source_text=source
        )
        baseline_result = _analyze_exact_key(source, citation)
    assert actual == baseline
    assert result.issues == baseline_result.issues
