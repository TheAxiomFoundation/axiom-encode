"""Conditional list chapeaux do not invent independently toggleable gates."""

from copy import deepcopy
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


def test_nz_parent_condition_chapeaux_transfer_obligations_to_every_child():
    branches = sc.recognize_source_structure(SOURCE, corpus_citation_path=CITATION)
    assert {("1", "a"), ("1", "b"), ("1", "c"), ("2", "a")} <= {
        branch.path for branch in branches
    }
    obligations = _exceptions(SOURCE)
    assert {branch.path for branch in obligations} == {
        ("1", "a"),
        ("1", "b"),
        ("1", "c"),
        ("2", "a"),
    }
    assert all(branch.condition_chapeau is not None for branch in obligations)


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
        sc._source_exception_requires_paired_witness(x.text) for x in obligations
    )


def test_only_a_sibling_does_not_resolve_an_empty_chapeau():
    source = (
        "1\nA person is entitled to a payment if, for those days,—\n"
        "2\nA payment has the following rates:—\na\nthe single rate; or\n"
        "b\nthe couple rate.\n"
    )
    second = source.index("2\n")
    first_child = source.index("a\n", second)
    last_child = source.index("b\n", first_child)
    branches = (
        sc.SourceStructureBranch(
            ("1",), "subsection", "(1)", source[:second], 0, second
        ),
        sc.SourceStructureBranch(
            ("2",), "subsection", "(2)", source[second:], second, len(source)
        ),
        sc.SourceStructureBranch(
            ("2", "a"),
            "paragraph",
            "(a)",
            source[first_child:last_child],
            first_child,
            last_child,
        ),
        sc.SourceStructureBranch(
            ("2", "b"),
            "paragraph",
            "(b)",
            source[last_child:],
            last_child,
            len(source),
        ),
    )
    obligations = sc._source_exception_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )
    assert len(obligations) == 1
    assert obligations[0].path == ("1",)


def test_inactive_children_do_not_resolve_an_empty_chapeau():
    branches = sc.recognize_source_structure(SOURCE, corpus_citation_path=CITATION)
    active = tuple(branch for branch in branches if len(branch.path) == 1)
    obligations = sc._source_exception_branches(
        SOURCE,
        branches=branches,
        active_branches=active,
        deferred_paths=set(),
    )
    assert {branch.path for branch in obligations} == {("1",), ("2",)}


def test_substantive_parent_condition_keeps_paired_obligation_with_children():
    source = SOURCE.replace(
        "if, for those days,—", "if the person is ordinarily resident,—", 1
    )
    obligations = _exceptions(source)
    parents = [branch for branch in obligations if branch.path == ("1",)]
    assert len(parents) == 1
    assert "ordinarily resident" in parents[0].text
    assert any(branch.path == ("2", "a") for branch in obligations)


def test_explicit_child_condition_still_requires_paired_evidence():
    source = SOURCE.replace(
        "the person is not disqualified under subsection (2).",
        "the person qualifies if ordinarily resident.",
    )
    obligations = _exceptions(source)
    explicit = [
        branch
        for branch in obligations
        if branch.path == ("1", "c") and branch.condition_chapeau is None
    ]
    assert len(explicit) == 1
    assert sc._source_exception_requires_paired_witness(explicit[0].text)


@given(
    cue=st.sampled_from(["if", "when", "IF", "When"]),
    adjunct=st.sampled_from(
        ["", ", for those days,", ", in that period,", ", during those months,"]
    ),
    terminator=st.sampled_from([":", "—", "–", "-"]),
    substantive=st.sampled_from(
        [
            "the person is resident",
            "income is below the limit",
            "the benefit is payable",
        ]
    ),
)
def test_empty_chapeau_recognition_never_discards_a_condition_predicate(
    cue, adjunct, terminator, substantive
):
    assert sc._source_condition_is_empty_list_chapeau(f"{cue}{adjunct}{terminator}")
    assert not sc._source_condition_is_empty_list_chapeau(
        f"{cue}{adjunct} {substantive}{terminator}"
    )


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


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
def test_review_constant_true_probe_does_not_discard_child_conditions(jurisdiction):
    citation, payload, _cases = _entitlement_fixture(jurisdiction, formula=True)
    cases = [
        {
            "name": "positive",
            "period": "2026",
            "input": {},
            "output": {"payment_entitlement": "holds"},
        }
    ]

    issues = _paired_issues(_analyze_entitlement(citation, payload, cases))

    assert len(issues) == 1
    assert "ordinarily resident" in issues[0] and "citizen" in issues[0]


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
def test_child_predicates_and_distinct_pairs_satisfy_inherited_condition(jurisdiction):
    citation, payload, cases = _entitlement_fixture(jurisdiction)
    assert not _analyze_entitlement(citation, payload, cases).issues


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
@pytest.mark.parametrize("missing", ["predicate", "pair"])
def test_each_child_needs_its_own_controlling_predicate_and_pair(jurisdiction, missing):
    citation, payload, cases = _entitlement_fixture(jurisdiction)
    if missing == "predicate":
        payload["rules"][0]["versions"][0]["formula"] = "ordinarily_resident"
        for case in cases:
            case["output"]["payment_entitlement"] = (
                "holds" if case["input"]["ordinarily_resident"] else "not_holds"
            )
    else:
        cases.pop()

    issues = _paired_issues(_analyze_entitlement(citation, payload, cases))

    assert len(issues) == 1
    assert "citizen" in issues[0]


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
@pytest.mark.parametrize("parent_proof", [True, False])
def test_independent_child_outputs_cannot_certify_constant_parent_entitlement(
    jurisdiction, parent_proof
):
    citation, payload, cases = _entitlement_fixture(jurisdiction, formula=True)
    for name, predicate, child_index in (
        ("resident_recorded", "ordinarily_resident", 1),
        ("citizenship_recorded", "citizen", 2),
    ):
        rule = deepcopy(payload["rules"][0])
        rule["name"] = name
        rule["versions"][0]["formula"] = predicate
        atoms = rule["metadata"]["proof"]["atoms"]
        rule["metadata"]["proof"]["atoms"] = [atoms[0], atoms[child_index]]
        payload["rules"].append(rule)
        for case in cases:
            case["output"][name] = "holds" if case["input"][predicate] else "not_holds"
            case["output"]["payment_entitlement"] = "holds"
    if not parent_proof:
        payload["rules"][0]["metadata"]["proof"]["atoms"].pop(0)

    issues = _paired_issues(_analyze_entitlement(citation, payload, cases))

    assert len(issues) == 1
    assert "[output payment_entitlement]" in issues[0]


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
def test_child_owned_constant_entitlement_cannot_borrow_parent_helper_witnesses(
    jurisdiction,
):
    citation, payload, cases = _entitlement_fixture(jurisdiction, formula=True)
    entitlement = payload["rules"][0]
    helper = deepcopy(entitlement)
    helper["name"] = "requirements_recorded"
    helper["versions"][0]["formula"] = "ordinarily_resident and citizen"
    entitlement["source"] = citation + "(1)(a)"
    entitlement["metadata"]["proof"]["atoms"].pop(0)
    payload["rules"].append(helper)
    for case in cases:
        case["output"]["requirements_recorded"] = case["output"]["payment_entitlement"]
        case["output"]["payment_entitlement"] = "holds"

    issues = _paired_issues(_analyze_entitlement(citation, payload, cases))

    assert len(issues) == 1
    assert "[output payment_entitlement]" in issues[0]


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
def test_inherited_condition_preserves_parent_effect_and_child_polarity(jurisdiction):
    citation, payload, cases = _entitlement_fixture(
        jurisdiction, formula="not (ordinarily_resident and citizen)"
    )
    for case in cases:
        case["output"]["payment_entitlement"] = (
            "not_holds" if case["output"]["payment_entitlement"] == "holds" else "holds"
        )
    assert _paired_issues(_analyze_entitlement(citation, payload, cases))


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
def test_negative_parent_chapeau_transfers_its_exclusion_effect(jurisdiction):
    citation, payload, cases = _entitlement_fixture(
        jurisdiction, formula="not (ordinarily_resident and citizen)"
    )
    source = ENTITLEMENT_SOURCE.replace("is entitled", "is not entitled", 1)
    payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
        ENTITLEMENT_EXCERPTS[0].replace("is entitled", "is not entitled", 1)
    )
    for case in cases:
        case["output"]["payment_entitlement"] = (
            "not_holds" if case["output"]["payment_entitlement"] == "holds" else "holds"
        )
    assert not _analyze_entitlement(citation, payload, cases, source=source).issues


@pytest.mark.parametrize("jurisdiction", JURISDICTIONS)
@pytest.mark.parametrize("correct_polarity", [True, False])
def test_child_predicate_retains_its_own_negative_polarity(
    jurisdiction, correct_polarity
):
    citation, payload, _cases = _entitlement_fixture(
        jurisdiction,
        formula=(
            "not ordinarily_resident and citizen"
            if correct_polarity
            else "ordinarily_resident and citizen"
        ),
    )
    source = ENTITLEMENT_SOURCE.replace(
        "is ordinarily resident", "is not ordinarily resident"
    )
    payload["rules"][0]["metadata"]["proof"]["atoms"][1]["source"]["excerpt"] = (
        ENTITLEMENT_EXCERPTS[1].replace(
            "is ordinarily resident", "is not ordinarily resident"
        )
    )
    cases = [
        {
            "name": f"resident={resident}, citizen={citizen}",
            "period": "2026",
            "input": {"ordinarily_resident": resident, "citizen": citizen},
            "output": {
                "payment_entitlement": (
                    "holds"
                    if ((not resident) if correct_polarity else resident) and citizen
                    else "not_holds"
                )
            },
        }
        for resident, citizen in ((False, True), (True, True), (False, False))
    ]
    analysis = _analyze_entitlement(citation, payload, cases, source=source)
    if correct_polarity:
        assert not analysis.issues
    else:
        assert _paired_issues(analysis)


@pytest.mark.parametrize("atom_index", [0, 1, 2])
@pytest.mark.parametrize("mutation", ["missing", "citation", "version", "excerpt"])
def test_inherited_witness_requires_selected_authentic_parent_and_child_proof(
    atom_index, mutation
):
    citation, payload, cases = _entitlement_fixture()
    atoms = payload["rules"][0]["metadata"]["proof"]["atoms"]
    if mutation == "missing":
        atoms.pop(atom_index)
    elif mutation == "citation":
        atoms[atom_index]["source"]["corpus_citation_path"] += "-other"
    elif mutation == "version":
        atoms[atom_index]["path"] = "versions[1].formula"
    else:
        atoms[atom_index]["source"]["excerpt"] = (
            "An unrelated preface. " + atoms[atom_index]["source"]["excerpt"]
        )

    assert _paired_issues(_analyze_entitlement(citation, payload, cases))


def test_partial_inactive_children_cannot_remove_parent_obligation():
    branches = sc.recognize_source_structure(ENTITLEMENT_SOURCE)
    active = tuple(branch for branch in branches if branch.path != ("1", "b"))

    obligations = sc._source_exception_branches(
        ENTITLEMENT_SOURCE,
        branches=branches,
        active_branches=active,
        deferred_paths=set(),
    )

    assert len(obligations) == 1
    assert obligations[0].path == ("1",)


def test_nested_unresolved_child_container_retains_parent_obligation():
    source = (
        "(1) A person is entitled to a payment if:\n"
        "(a) both of the following are true:\n"
        "(i) the person is ordinarily resident; and\n"
        "(ii) the person is a citizen."
    )
    first = source.index("(a)")
    second = source.index("(i)")
    third = source.index("(ii)")
    branches = (
        sc.SourceStructureBranch(("1",), "subsection", "(1)", source, 0, len(source)),
        sc.SourceStructureBranch(
            ("1", "a"), "paragraph", "(a)", source[first:], first, len(source)
        ),
        sc.SourceStructureBranch(
            ("1", "a", "i"),
            "subparagraph",
            "(i)",
            source[second:third],
            second,
            third,
        ),
        sc.SourceStructureBranch(
            ("1", "a", "ii"),
            "subparagraph",
            "(ii)",
            source[third:],
            third,
            len(source),
        ),
    )
    obligations = sc._source_exception_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )
    assert any(branch.path == ("1",) for branch in obligations)
    citation, payload, cases = _entitlement_fixture(formula="ordinarily_resident")
    assert _paired_issues(_analyze_entitlement(citation, payload, cases, source=source))


def test_explicit_child_condition_keeps_inherited_parent_polarity():
    source = ENTITLEMENT_SOURCE.replace(
        "the person is ordinarily resident; and",
        "the person qualifies if ordinarily resident; and",
    )
    obligations = _exceptions(source)
    child_obligations = [branch for branch in obligations if branch.path == ("1", "a")]
    assert any(branch.condition_chapeau is None for branch in child_obligations)
    assert any(branch.condition_chapeau is not None for branch in child_obligations)


def test_child_source_proof_cannot_be_borrowed_from_unselected_formula_version():
    citation, payload, cases = _entitlement_fixture()
    rule = payload["rules"][0]
    rule["versions"].append(
        {"effective_from": "2027-01-01", "formula": "ordinarily_resident and citizen"}
    )
    atoms = deepcopy(rule["metadata"]["proof"]["atoms"])
    for atom in atoms:
        atom["path"] = "versions[1].formula"
    rule["metadata"]["proof"]["atoms"] = atoms
    assert _paired_issues(_analyze_entitlement(citation, payload, cases))
