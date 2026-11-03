"""Florida calculation-of-benefits page-8 probes for the paired-condition gate.

Main's US-manual numeric-limit filter already omits the page's two sequencing
instructions before the paired-witness gate, so they need no exemption. The
remaining probes pin the gate's toggleable conditions, its unsupported relation
wording and its proviso effects, and check that the per-part witness changes no
verdict on the authenticated page or its perturbations.
"""

from __future__ import annotations

import functools
import hashlib
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
from tests.paired_condition_differential import (
    PAIRED,
    analyze_with_part_witness_differential,
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


@pytest.mark.parametrize("clause", PROCEDURE_CLAUSES)
def test_page8_sequencing_instructions_have_no_paired_issue_under_the_manual_filter(
    clause,
):
    # The predicate alone still demands pairs; main's numeric-limit filter for
    # US manuals keeps the gate from reaching these two instructions.
    assert sc._source_exception_requires_paired_witness(clause)
    result = _analyze_exact_key(clause, MANUAL_CITATION)
    assert not result.changed
    assert not any(PAIRED in issue for issue in result.issues)


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
    assert sc._source_exception_requires_paired_witness(clause)


def test_is_equal_to_or_exceeds_keeps_legacy_unsupported_relation():
    text = (
        "if the gross income is equal to or exceeds the gross income limit, the "
        "assistance group is ineligible."
    )
    assert (
        sc._source_relational_exception_match_indices(
            text,
            left_name="gross_income",
            relation=">=",
            right_name="gross_income_limit",
        )
        == ()
    )
    (token,) = sc._source_relation_tokens(text)
    assert token.group("unsupported")
    assert sc._source_exception_requires_paired_witness(text)


@pytest.mark.parametrize(
    "phrase", ("is equal to or less than", "is not equal to", "does not equal")
)
def test_other_equal_to_phrases_stay_unsupported(phrase):
    (token,) = sc._source_relation_tokens(f"if income {phrase} the limit, stop.")
    assert token.group("unsupported")


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
    assert not sc._numeric_exception_witness_matches_source(
        branch, _proviso_witness(blocks=True), extract_numeric_occurrences=GROUNDING
    )


def test_postposed_proviso_rejects_the_enabling_side():
    branch = sc.SourceStructureBranch(("1",), "number", "1.", PROVISO, 0, len(PROVISO))
    assert not sc._numeric_exception_witness_matches_source(
        branch,
        _proviso_witness(blocks=False),
        extract_numeric_occurrences=GROUNDING,
    )


def test_unnegated_proviso_is_not_read_as_withdrawal():
    text = "The credit applies, provided the income exceeds the threshold."
    assert sc._source_exception_requires_paired_witness(text)


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
    return analyze_with_part_witness_differential(
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


def test_authenticated_page_keeps_its_single_paired_demand():
    assert hashlib.sha256(AUTHENTICATED_SOURCE.encode()).hexdigest() == (
        "a0b4fac87f905e6a6989b505b866f5f0306fe361531801c4226d9184e62723e5"
    )
    result = _analyze_exact_key(AUTHENTICATED_SOURCE, MANUAL_CITATION)
    assert not result.changed
    assert sum(PAIRED in issue for issue in result.issues) == 1


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
def test_footer_handling_keeps_main_obligations(source, expected):
    assert sc._source_exception_requires_paired_witness(source) is expected


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
def test_perturbed_page8_instructions_keep_the_clause_witness_verdict(
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

    result = _analyze_exact_key(source, citation)
    # No perturbation of the page's instructions gains or loses a demand
    # through the per-part witness.
    assert not result.changed
