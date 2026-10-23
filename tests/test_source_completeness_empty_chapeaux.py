"""Conditional-list relaxation abstains until full Boolean structure is resolved."""

from functools import partial

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

CITATION = "nz/statute/act/public/2018/0032/section/72"
# Social Security Act 2018, s 72(1) and the chapeau of s 72(2), with
# s 72(2)(a). The child text supplies the predicates introduced by each 'if'.
SOURCE = """1
A person is entitled to a winter energy payment for 1 or more days during the winter period if, for those days,—
a
the person is receiving a qualifying benefit (or a portion of it), either as the person granted that benefit or as the spouse or partner of that person; and
b
the qualifying benefit (or a portion of it) is payable to the person; and
c
the person is not disqualified under subsection (2).
2
However, the person is not entitled to a winter energy payment for 1 or more days during the winter period if, for those days,—
a
the qualifying benefit is payable at a rate prescribed in respect of a person who is in a relationship, and MSD has determined under subsection (3) that the person’s spouse or partner is entitled to a winter energy payment.
"""


def _exceptions(source, *, recognize=True):
    branches = (
        sc.recognize_source_structure(source, corpus_citation_path=CITATION)
        if recognize
        else ()
    )
    return sc._source_exception_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )


def test_nz_parent_chapeaux_retain_unresolved_parent_obligations():
    # A child list is not a parsed Boolean proposition. In particular s 72's
    # compound leaves and operators cannot be authenticated by child count.
    branches = sc.recognize_source_structure(SOURCE, corpus_citation_path=CITATION)
    assert {("1", "a"), ("1", "b"), ("1", "c"), ("2", "a")} <= {
        branch.path for branch in branches
    }
    assert {branch.path for branch in _exceptions(SOURCE)} == {("1",), ("2",)}


@pytest.mark.parametrize(
    "source",
    [
        "A person is entitled to a payment if, for those days,—",
        "A person is entitled to a payment if the person is receiving a benefit: ",
        "A person is entitled to a payment if, for those days,— the person is receiving a benefit.",
    ],
)
def test_unresolved_or_substantive_condition_keeps_paired_obligation(source):
    obligations = _exceptions(source, recognize=False)
    assert obligations
    assert all(
        sc._source_exception_requires_paired_witness(branch.text)
        for branch in obligations
    )


def test_inactive_children_do_not_resolve_a_parent_chapeau():
    branches = sc.recognize_source_structure(SOURCE, corpus_citation_path=CITATION)
    obligations = sc._source_exception_branches(
        SOURCE,
        branches=branches,
        active_branches=tuple(branch for branch in branches if len(branch.path) == 1),
        deferred_paths=set(),
    )
    assert {branch.path for branch in obligations} == {("1",), ("2",)}


JURISDICTIONS = ("uk", "us", "de", "fr", "nz")
ENTITLEMENT_SOURCE = (
    "1\nA person is entitled to a payment if:\na\n"
    "the person is ordinarily resident; and\nb\nthe person is a citizen."
)
ENTITLEMENT_EXCERPTS = (
    "A person is entitled to a payment if:",
    "the person is ordinarily resident; and",
    "the person is a citizen.",
)


def _entitlement_fixture(jurisdiction="uk", formula="ordinarily_resident and citizen"):
    citation = f"{jurisdiction}/statute/2026/example/section/1"
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": citation}},
        "rules": [
            {
                "name": "payment_entitlement",
                "kind": "derived",
                "dtype": "Judgment",
                "period": "Year",
                "source": citation + "(1)",
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "kind": "formula",
                                "source": {
                                    "corpus_citation_path": citation,
                                    "excerpt": excerpt,
                                },
                            }
                            for excerpt in ENTITLEMENT_EXCERPTS
                        ]
                    }
                },
                "versions": [{"effective_from": "2026-01-01", "formula": formula}],
            }
        ],
    }
    cases = [
        {
            "name": name,
            "period": "2026",
            "input": {"ordinarily_resident": resident, "citizen": citizen},
            "output": {
                "payment_entitlement": "holds" if resident and citizen else "not_holds"
            },
        }
        for name, resident, citizen in (
            ("qualifying", True, True),
            ("not resident", False, True),
            ("not citizen", True, False),
        )
    ]
    return citation, payload, cases


def _analyze_entitlement(citation, payload, cases, *, source=ENTITLEMENT_SOURCE):
    return sc.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        source,
        corpus_citation_path=citation,
        test_cases=cases,
        extract_numeric_occurrences=partial(
            vp.extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
    )


def _paired_issues(analysis):
    return tuple(
        issue for issue in analysis.issues if "paired positive/blocking" in issue
    )


# These are the exact round-2 review inputs. Every proof excerpt is an exact
# substring of source, and all four assertions agree with the supplied formula.
# Source authenticity therefore cannot substitute for parsed source semantics.
REVIEW_CONDITIONS = (
    (
        "or_as_and",
        "the person is ordinarily resident; or",
        "the person is a citizen.",
        "ordinarily_resident and citizen",
        lambda resident, citizen: resident and citizen,
        lambda resident, citizen: resident or citizen,
    ),
    (
        "compound_leaf_omits_citizenship",
        "the person is ordinarily resident and is a citizen.",
        None,
        "ordinarily_resident",
        lambda resident, citizen: resident,
        lambda resident, citizen: resident and citizen,
    ),
    (
        "literal_double_negation",
        "the person is not not ordinarily resident; and",
        "the person is a citizen.",
        "not ordinarily_resident and citizen",
        lambda resident, citizen: not resident and citizen,
        lambda resident, citizen: resident and citizen,
    ),
    (
        "natural_double_negation",
        "it is not true that the person is not ordinarily resident; and",
        "the person is a citizen.",
        "not ordinarily_resident and citizen",
        lambda resident, citizen: not resident and citizen,
        lambda resident, citizen: resident and citizen,
    ),
)


def _condition_case(jurisdiction, child, second, formula, result):
    parent = ENTITLEMENT_EXCERPTS[0]
    source = f"1\n{parent}\na\n{child}"
    excerpts = [parent, child]
    if second is not None:
        source += f"\nb\n{second}"
        excerpts.append(second)
    citation, payload, _cases = _entitlement_fixture(jurisdiction, formula=formula)
    payload["rules"][0]["metadata"]["proof"]["atoms"] = [
        {
            "path": "versions[0].formula",
            "kind": "formula",
            "source": {"corpus_citation_path": citation, "excerpt": excerpt},
        }
        for excerpt in excerpts
    ]
    cases = [
        {
            "name": f"resident={resident}, citizen={citizen}",
            "period": "2026",
            "input": {"ordinarily_resident": resident, "citizen": citizen},
            "output": {
                "payment_entitlement": (
                    "holds" if result(resident, citizen) else "not_holds"
                )
            },
        }
        for resident, citizen in (
            (True, True),
            (False, True),
            (True, False),
            (False, False),
        )
    ]
    assert all(excerpt in source for excerpt in excerpts)
    return source, citation, payload, cases


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
@pytest.mark.parametrize(
    "_name,child,second,formula,result,legal_result",
    REVIEW_CONDITIONS,
    ids=[case[0] for case in REVIEW_CONDITIONS],
)
def test_review_boolean_or_negation_false_pass_keeps_parent_obligation(
    jurisdiction, _name, child, second, formula, result, legal_result
):
    source, citation, payload, cases = _condition_case(
        jurisdiction, child, second, formula, result
    )
    assert any(
        result(case["input"]["ordinarily_resident"], case["input"]["citizen"])
        != legal_result(case["input"]["ordinarily_resident"], case["input"]["citizen"])
        for case in cases
    )
    issues = _paired_issues(
        _analyze_entitlement(citation, payload, cases, source=source)
    )
    assert len(issues) == 1
    assert ENTITLEMENT_EXCERPTS[0] in issues[0]


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
@pytest.mark.parametrize("operator", ["and", "or"])
def test_even_correct_child_formulas_cannot_discharge_unparsed_parent(
    jurisdiction, operator
):
    source, citation, payload, cases = _condition_case(
        jurisdiction,
        f"the person is ordinarily resident; {operator}",
        "the person is a citizen.",
        f"ordinarily_resident {operator} citizen",
        (lambda resident, citizen: resident and citizen)
        if operator == "and"
        else (lambda resident, citizen: resident or citizen),
    )
    assert (
        len(
            _paired_issues(
                _analyze_entitlement(citation, payload, cases, source=source)
            )
        )
        == 1
    )


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
@pytest.mark.parametrize(
    "predicate",
    [
        "the person is not ordinarily resident; and",
        "the person is not not ordinarily resident; and",
        "it is not true that the person is not ordinarily resident; and",
        "it is not true that it is true that the person is ordinarily resident; and",
        "the person is ordinarily resident unless the person is not a citizen; and",
    ],
)
def test_unresolved_inherited_negation_scope_never_transfers(jurisdiction, predicate):
    source, citation, payload, cases = _condition_case(
        jurisdiction,
        predicate,
        "the person is a citizen.",
        "not ordinarily_resident and citizen",
        lambda resident, citizen: not resident and citizen,
    )
    assert _paired_issues(_analyze_entitlement(citation, payload, cases, source=source))
    assert any(branch.path == ("1",) for branch in _exceptions(source))


@given(
    cue=st.sampled_from(["if", "when", "IF", "When"]),
    adjunct=st.sampled_from(["", ", for those days,", ", in that period,"]),
    terminator=st.sampled_from([":", "—", "–", "-"]),
    operator=st.sampled_from(["and", "or"]),
    predicate=st.sampled_from(
        [
            "the person is ordinarily resident",
            "the person is ordinarily resident and is a citizen",
            "the person is not not ordinarily resident",
            "it is not true that the person is not ordinarily resident",
        ]
    ),
)
def test_parent_obligation_is_independent_of_unparsed_children(
    cue, adjunct, terminator, operator, predicate
):
    parent = f"A person is entitled to a payment {cue}{adjunct}{terminator}"
    source = f"1\n{parent}\na\n{predicate}; {operator}\nb\nthe person is a citizen."
    # Snapshot of origin/main's parent branch projection: unparsed predicates
    # and connective words never substitute child obligations for the parent.
    obligations = _exceptions(source)
    assert tuple((branch.path, branch.text) for branch in obligations) == (
        (("1",), parent),
    )


def test_explicit_child_condition_retains_both_source_obligations():
    source = ENTITLEMENT_SOURCE.replace(
        "the person is ordinarily resident; and",
        "the person qualifies if ordinarily resident; and",
    )
    obligations = _exceptions(source)
    assert {branch.path for branch in obligations} == {("1",), ("1", "a")}


def test_nested_child_container_retains_parent_obligation():
    source = (
        "(1) A person is entitled to a payment if:\n"
        "(a) both of the following are true:\n"
        "(i) the person is ordinarily resident; and\n"
        "(ii) the person is a citizen."
    )
    citation, payload, cases = _entitlement_fixture(formula="ordinarily_resident")
    assert _paired_issues(_analyze_entitlement(citation, payload, cases, source=source))
