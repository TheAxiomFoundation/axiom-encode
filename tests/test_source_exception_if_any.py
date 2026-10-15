from __future__ import annotations

from functools import partial

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CITATION = "us-nm/statute/7-2-18.1"

NM_SPOUSE_SOURCE = """B. Any resident may claim a credit if the resident:
(3) compensates a caregiver for child day care for a qualifying dependent to enable such resident together with the resident's spouse, if any and if not disabled, to be gainfully employed;
"""


def _exception_obligations(source: str) -> tuple[sc.SourceStructureBranch, ...]:
    branches = tuple(
        sc.recognize_source_structure(
            source, corpus_citation_path="us-nm/statute/7-2-18.1"
        )
    )
    return sc._source_exception_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )


def _public_condition_candidate(
    source: str, conditions: tuple[tuple[str, str, bool], ...]
) -> sc.CompleteSourceUnitAnalysis:
    """Prove each condition with its own executed, single-input case pair."""
    expression = " and ".join(
        name if active else f"not {name}" for name, _, active in conditions
    )
    formula = f"if {expression}: true else: false" if expression else "true"
    rule = {
        "name": "condition_applies",
        "kind": "derived",
        "dtype": "Judgment",
        "source": CITATION + "(1)",
        "versions": [{"effective_from": "2026-01-01", "formula": formula}],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": source,
                        },
                    }
                ]
            }
        },
    }
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {"name": name, "dtype": "Boolean", "description": description}
            for name, description, _ in conditions
        ],
        "rules": [rule],
    }
    active_inputs = {name: active for name, _, active in conditions}
    cases = [
        {
            "period": "2026",
            "input": active_inputs,
            "output": {"condition_applies": True},
        }
    ]
    for name, _, active in conditions:
        cases.append(
            {
                "period": "2026",
                "input": {**active_inputs, name: not active},
                "output": {"condition_applies": False},
            }
        )
    return sc.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _assert_public_condition_required(
    source: str,
    conditions: tuple[tuple[str, str, bool], ...],
    *,
    conservative_dependency: bool = False,
) -> None:
    proved = _public_condition_candidate(source, conditions)
    if conservative_dependency:
        assert any(
            "[complete-source-unit:source-explicit-conditions]" in issue
            for issue in proved.issues
        ), proved.issues
    else:
        assert not proved.issues, proved.issues
    missing = _public_condition_candidate(source, conditions[1:])
    assert any(
        "[complete-source-unit:tests]" in issue
        and "exceptions or applicability conditions" in issue
        for issue in missing.issues
    ), missing.issues


def test_nm_spouse_parenthetical_is_not_a_bare_exception_obligation():
    obligations = _exception_obligations(NM_SPOUSE_SOURCE)

    assert not any(ob.text.strip() == "if any and" for ob in obligations)
    spouse_clauses = [ob for ob in obligations if "if not disabled" in ob.text]
    assert len(spouse_clauses) == 1
    spouse_clause = spouse_clauses[0]
    assert "the resident's spouse, if any and if not disabled" in spouse_clause.text
    assert (
        spouse_clause.text == NM_SPOUSE_SOURCE[spouse_clause.start : spouse_clause.end]
    )
    assert sc._source_exception_selector_is_relevant(
        spouse_clause.text, "resident_has_spouse"
    )
    assert sc._source_exception_selector_is_relevant(
        spouse_clause.text, "spouse_is_disabled"
    )


@pytest.mark.parametrize(
    "source",
    [
        NM_SPOUSE_SOURCE + " ",
        NM_SPOUSE_SOURCE.removesuffix("\n"),
        NM_SPOUSE_SOURCE.upper(),
        NM_SPOUSE_SOURCE.replace("if not disabled", "if not employed"),
        NM_SPOUSE_SOURCE.replace("if any and", "if any or"),
    ],
)
def test_nm_parenthetical_exemption_requires_the_exact_reviewed_source_bytes(source):
    cues = sc._source_exception_or_applicability_matches(source)

    assert any(source[cue.start() :].lower().startswith("if any") for cue in cues)
    assert any(
        ob.text.strip().lower() in {"if any and", "if any or"}
        for ob in _exception_obligations(source)
    )


@pytest.mark.parametrize("prefix", ["The credit applies ", "The credit applies, "])
@pytest.mark.parametrize(
    "condition",
    [
        "if any dependent is disabled",
        "if any and all dependents are disabled",
        "if any, but not all, dependents are disabled",
    ],
)
def test_quantified_any_conditions_remain_independent_obligations(prefix, condition):
    source = f"(1) {prefix}{condition} and if the resident is employed;"

    assert len(sc._source_exception_or_applicability_matches(source)) == 2
    obligations = _exception_obligations(source)
    assert len(obligations) == 2
    assert any(condition in ob.text for ob in obligations)
    assert all(
        sc._source_exception_requires_paired_witness(ob.text) for ob in obligations
    )


def test_standalone_parenthetical_presence_condition_is_preserved():
    source = "(1) The claimant's spouse, if any, must be gainfully employed;"

    assert len(sc._source_exception_or_applicability_matches(source)) == 1
    assert [ob.text for ob in _exception_obligations(source)] == [source]


def test_documents_if_any_or_employed_requires_a_presence_witness():
    source = "(1) Documents must be submitted, if any or if the applicant is employed;"

    result = _public_condition_candidate(
        source,
        (("applicant_is_employed", "The applicant is employed.", True),),
    )

    assert result.issues, "Employment alone does not prove document existence."
    assert len(_exception_obligations(source)) == 2


@settings(max_examples=40, deadline=None)
@given(
    subject=st.sampled_from(["Documents", "Invoices", "Care expenses"]),
    action=st.sampled_from(["submitted", "reported"]),
    conjunction=st.sampled_from(["and", "or"]),
    qualifier=st.sampled_from(["employed", "not employed", "disabled"]),
    whitespace=st.sampled_from([" ", "\t", "\n", "  "]),
)
def test_coordinated_if_any_never_accepts_only_the_qualified_condition(
    subject, action, conjunction, qualifier, whitespace
):
    source = (
        f"(1) {subject} must be {action}, if{whitespace}any "
        f"{conjunction} if the applicant is {qualifier};"
    )
    result = _public_condition_candidate(
        source,
        (
            (
                "applicant_is_" + qualifier.removeprefix("not "),
                "The applicant is " + qualifier.removeprefix("not ") + ".",
                not qualifier.startswith("not "),
            ),
        ),
    )

    assert result.issues, "A second qualified condition cannot replace existence."
    assert len(_exception_obligations(source)) == 2


@settings(max_examples=80, deadline=None)
@given(
    subject=st.sampled_from(["spouse", "partner", "caregiver"]),
    condition=st.sampled_from(["not disabled", "gainfully employed", "eligible"]),
    conjunction=st.sampled_from(["and", "or"]),
    cue=st.sampled_from(["if", "when", "unless"]),
    attachment=st.sampled_from([(", ", ""), (" (", ")"), (" [", "]")]),
    whitespace=st.sampled_from([" ", "\t", "\n", "  "]),
    uppercase=st.booleans(),
)
def test_unreviewed_parenthetical_presence_preserves_all_real_condition_cues(
    subject, condition, conjunction, cue, attachment, whitespace, uppercase
):
    opening, closing = attachment
    source = (
        f"(1) The claimant's {subject}{opening}if{whitespace}any{closing}"
        f"{whitespace}{conjunction}{whitespace}{cue}{whitespace}{condition}, "
        "must be gainfully employed;"
    )
    if uppercase:
        source = source.upper()

    cues = sc._source_exception_or_applicability_matches(source)
    assert len(cues) == 2
    assert cues[0].group(0).lower() == "if"
    assert cues[1].group(0).lower() == cue
    obligations = _exception_obligations(source)
    assert len(obligations) == 2
    assert condition.upper() in obligations[1].text.upper()
    assert all(ob.text == source[ob.start : ob.end] for ob in obligations)
    assert all(
        sc._source_exception_requires_paired_witness(ob.text) for ob in obligations
    )

    selector = {
        "not disabled": "is_disabled",
        "gainfully employed": "is_gainfully_employed",
        "eligible": "is_eligible",
    }[condition]
    active = condition != "not disabled"
    if cue == "unless":
        active = not active
    conditions = (
        (
            selector,
            f"The claimant's {subject} is {condition.removeprefix('not ')}.",
            active,
        ),
    )
    proved = _public_condition_candidate(source, conditions)
    assert any(
        "[complete-source-unit:tests]" in issue
        and "exceptions or applicability conditions" in issue
        for issue in proved.issues
    ), proved.issues
    missing = _public_condition_candidate(
        source,
        (),
    )
    assert any(
        "[complete-source-unit:tests]" in issue
        and "exceptions or applicability conditions" in issue
        for issue in missing.issues
    ), missing.issues


@settings(max_examples=80, deadline=None)
@given(
    subject=st.sampled_from(["dependent", "spouse", "caregiver", "child", "resident"]),
    predicate=st.sampled_from(["disabled", "not disabled", "employed", "eligible"]),
    whitespace=st.sampled_from([" ", "\t", "\n", "  "]),
    uppercase=st.booleans(),
)
def test_any_with_a_subject_predicate_always_retains_its_condition_cue(
    subject, predicate, whitespace, uppercase
):
    source = (
        f"(1) Credit applies, if{whitespace}any{whitespace}{subject} "
        f"is {predicate} and if the claimant has income;"
    )
    if uppercase:
        source = source.upper()

    cues = sc._source_exception_or_applicability_matches(source)
    assert len(cues) == 2
    obligations = _exception_obligations(source)
    assert len(obligations) == 2
    first = obligations[0]
    assert f"ANY{whitespace}{subject.upper()}" in first.text.upper()
    assert f"IS {predicate.upper()}" in first.text.upper()
    assert first.text == source[first.start : first.end]
    assert sc._source_exception_requires_paired_witness(first.text)

    _assert_public_condition_required(
        source,
        (
            (
                f"any_{subject}_is_{predicate.replace(' ', '_')}",
                f"Any {subject} is {predicate}.",
                True,
            ),
            ("claimant_has_income", "The claimant has income.", True),
        ),
        # The bare subject noun remains a conservative unknown dependency
        # reading; attaching "if any" must not turn that into acceptance.
        conservative_dependency=subject == "dependent",
    )
