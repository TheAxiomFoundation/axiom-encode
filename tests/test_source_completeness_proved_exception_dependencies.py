"""Composite exception selectors abstain without complete definition semantics."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

CITATION = "nz/statute/act/public/2018/0032/section/73"
DEFINITION = (
    "The election must be made, and may be revoked, in a form and manner "
    "approved by MSD."
)
CONDITION = (
    "Unless it is revoked, the election stops the person from being entitled "
    "to the payment for any days after the election is made."
)
SOURCE = DEFINITION + "\n" + CONDITION
SELECTOR = "winter_energy_payment_election_revocation_valid"
OUTPUT = "election_stops_winter_energy_payment_entitlement"
JURISDICTION_CITATIONS = (
    "uk/statute/2020/1/section/73",
    "us/statute/2020/1/section/73",
    "de/statute/2020/1/section/73",
    "fr/statute/2020/1/section/73",
    CITATION,
)


def _rule(name, formula, excerpt):
    return {
        "name": name,
        "kind": "derived",
        "dtype": "Judgment",
        "entity": "Person",
        "period": "Day",
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": excerpt,
                        },
                    }
                ]
            }
        },
        "versions": [{"effective_from": "2018-11-26", "formula": formula}],
    }


def _fixture(selector=SELECTOR):
    rules = {
        selector: _rule(
            selector,
            "revocation_made and revocation_in_msd_approved_form "
            "and revocation_in_msd_approved_manner",
            DEFINITION,
        ),
        OUTPUT: _rule(
            OUTPUT,
            "election_valid and day_is_after_election_made and not " + selector,
            CONDITION,
        ),
    }
    cases = [
        {
            "name": "revocation present" if revoked else "revocation absent",
            "period": "2018-11-26",
            "input": {
                "revocation_made": revoked,
                "revocation_in_msd_approved_form": True,
                "revocation_in_msd_approved_manner": True,
                "election_valid": True,
                "day_is_after_election_made": True,
            },
            "output": {selector: revoked, OUTPUT: not revoked},
        }
        for revoked in (False, True)
    ]
    return rules, cases


def _matches(rules, cases, *, source=SOURCE, condition=CONDITION, paths=None):
    asserted = {name: cases for name in rules}
    witnesses = sc._toggled_formula_boolean_selectors(
        rules, asserted_by_rule=asserted, formula_environment={}
    )
    branch = sc.SourceStructureBranch(
        path=("4",),
        label="(4)",
        text=condition,
        start=source.index(condition),
        end=len(source),
        kind="subsection",
    )
    return sc._exception_witnesses_for_branch(
        branch,
        source_text=source,
        corpus_citation_path=CITATION,
        principal_rules=rules,
        principal_rule_paths=paths or {name: {("4",)} for name in rules},
        asserted_by_rule=asserted,
        toggled_exception_selectors=witnesses,
        extract_numeric_occurrences=vp.extract_typed_numeric_inventory_occurrences_from_text,
    )


def test_composite_revocation_keeps_base_obligation_despite_four_witnesses():
    rules, cases = _fixture()
    witnesses = sc._toggled_formula_boolean_selectors(
        rules, asserted_by_rule={name: cases for name in rules}, formula_environment={}
    )

    # The s 73 short fixture still exercises the formulas, but the unparsed
    # definition cannot authenticate the composite selector's source meaning.
    assert len(witnesses) == 4
    assert not _matches(rules, cases)


@pytest.mark.parametrize("citation", JURISDICTION_CITATIONS)
@pytest.mark.parametrize(
    "definition",
    [DEFINITION, "If the election is revoked, the election is no longer active."],
)
def test_dependency_cannot_certify_opposite_exception_polarity(
    monkeypatch, citation, definition
):
    """The review's active-election proof reverses the source-required result."""

    monkeypatch.setitem(globals(), "CITATION", citation)
    selector = "winter_energy_payment_election_active"
    rules, cases = _fixture(selector)
    rules[selector]["versions"][0]["formula"] = "not revocation_made"
    rules[selector]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = definition
    for case in cases:
        revoked = case["input"]["revocation_made"]
        case["output"][selector] = not revoked
        case["output"][OUTPUT] = revoked

    assert not _matches(
        rules,
        cases,
        source=definition + "\n" + CONDITION,
        paths={selector: {("3",)}, OUTPUT: {("4",)}},
    )


@pytest.mark.parametrize("citation", JURISDICTION_CITATIONS)
@pytest.mark.parametrize("sliced_excerpt", [False, True])
@pytest.mark.parametrize(
    "definition",
    [
        "It is false that the election is revoked.",
        "The claim that the election is revoked is false.",
        "A claim that the election is revoked does not establish revocation.",
    ],
)
def test_dependency_cannot_borrow_negated_or_reported_source_action(
    monkeypatch, citation, definition, sliced_excerpt
):
    monkeypatch.setitem(globals(), "CITATION", citation)
    rules, cases = _fixture()
    rules[SELECTOR]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
        "the election is revoked" if sliced_excerpt else definition
    )

    assert not _matches(
        rules,
        cases,
        source=definition + "\n" + CONDITION,
        paths={SELECTOR: {("3",)}, OUTPUT: {("4",)}},
    )


@pytest.mark.parametrize("citation", JURISDICTION_CITATIONS)
@pytest.mark.parametrize(
    "predicate",
    [
        "election_in_msd_approved_form",
        "revocation_in_msd_approved_form",
        "permit_revoked",
    ],
)
def test_dependency_cannot_borrow_revocation_for_other_predicate(
    monkeypatch, citation, predicate
):
    """An approval, action qualifier or another subject is not the action."""

    monkeypatch.setitem(globals(), "CITATION", citation)
    selector = "winter_energy_payment_election_form_approved"
    rules, cases = _fixture(selector)
    rules[selector]["versions"][0]["formula"] = predicate
    for case in cases:
        case["input"][predicate] = case["output"][selector]
        for name in (
            "revocation_made",
            "revocation_in_msd_approved_form",
            "revocation_in_msd_approved_manner",
        ):
            if name != predicate:
                case["input"].pop(name)

    assert not _matches(rules, cases)


@pytest.mark.parametrize("citation", JURISDICTION_CITATIONS)
def test_composite_dependency_abstains_even_with_source_positive_event(
    monkeypatch, citation
):
    monkeypatch.setitem(globals(), "CITATION", citation)
    rules, cases = _fixture()
    witnesses = sc._toggled_formula_boolean_selectors(
        rules, asserted_by_rule={name: cases for name in rules}, formula_environment={}
    )

    assert len(witnesses) == 4
    assert not _matches(rules, cases)


@pytest.mark.parametrize("rule_name", [SELECTOR, OUTPUT])
@pytest.mark.parametrize("mutation", ["no_proof", "citation", "excerpt", "version"])
def test_composite_dependency_requires_selected_authoritative_formula_proofs(
    rule_name, mutation
):
    rules, cases = _fixture()
    atom = rules[rule_name]["metadata"]["proof"]["atoms"][0]
    if mutation == "no_proof":
        rules[rule_name].pop("metadata")
    elif mutation == "citation":
        atom["source"]["corpus_citation_path"] = CITATION.replace("/73", "/74")
    elif mutation == "excerpt":
        atom["source"]["excerpt"] = "Unless it is revoked, the other payment stops."
    else:
        atom["path"] = "versions[1].formula"

    assert not _matches(rules, cases)


def test_composite_dependency_cannot_borrow_unavailable_definition():
    rules, cases = _fixture()
    assert not _matches(rules, cases, source=CONDITION)


def test_composite_dependency_cannot_borrow_consequent_as_antecedent():
    rules, cases = _fixture()
    rules[SELECTOR]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
        "the election stops the person from being entitled to the payment"
    )
    assert not _matches(rules, cases)


@pytest.mark.parametrize(
    "unrelated",
    [
        "A parking permit may be revoked after a warning.",
        "The election does not affect whether a parking permit may be revoked.",
        "The election remains valid. A parking permit may be revoked after a warning.",
        "The election may be challenged if a parking permit is revoked.",
    ],
)
@pytest.mark.parametrize("explicit_subject", [False, True])
def test_composite_dependency_cannot_borrow_unrelated_revocation(
    unrelated, explicit_subject
):
    rules, cases = _fixture()
    condition = (
        CONDITION.replace("it is", "the election is") if explicit_subject else CONDITION
    )
    rules[OUTPUT]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = condition
    rules[SELECTOR]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = unrelated

    assert not _matches(
        rules, cases, source=unrelated + "\n" + condition, condition=condition
    )


@pytest.mark.parametrize("subject", ["application", "permit", "payment election"])
def test_pronoun_dependency_keeps_unresolved_composite_obligation(subject):
    rules, cases = _fixture()
    source = SOURCE.replace("The election", "The " + subject).replace(
        "the election", "the " + subject
    )
    for rule in rules.values():
        atom = rule["metadata"]["proof"]["atoms"][0]
        atom["source"]["excerpt"] = (
            atom["source"]["excerpt"]
            .replace("The election", "The " + subject)
            .replace("the election", "the " + subject)
        )
    matches = _matches(rules, cases, source=source, condition=source.split("\n")[1])

    assert not matches


def test_source_proved_dependency_keeps_output_ownership():
    rules, cases = _fixture()
    assert not _matches(rules, cases, paths={name: {("3",)} for name in rules})


def test_source_proved_dependency_keeps_blocking_effect():
    rules, cases = _fixture()
    rules[OUTPUT]["versions"][0]["formula"] = SELECTOR
    for case in cases:
        case["output"][OUTPUT] = case["output"][SELECTOR]
    assert not _matches(rules, cases)


def test_dependency_proof_from_inactive_version_cannot_supply_meaning():
    rules, cases = _fixture()
    old = deepcopy(rules[SELECTOR]["versions"][0])
    old["effective_to"] = "2018-11-25"
    old["effective_from"] = "2018-01-01"
    rules[SELECTOR]["versions"].insert(0, old)
    assert not _matches(rules, cases)


@given(
    st.lists(
        st.sampled_from(["winter", "energy", "payment", "election"]),
        min_size=3,
        max_size=4,
    )
)
@settings(max_examples=20, deadline=None)
@example(words=["payment", "payment", "payment"])
def test_composite_dependency_context_prefix_equals_base_snapshot(words):
    # Base's existing lexical route can accept some names (including repeated
    # "payment"). Withdrawing composite proof credit must preserve that result.
    snapshots = json.loads(
        (
            Path(__file__).parent / "fixtures/source_completeness/fail_closed_base.json"
        ).read_text()
    )["context_prefix_match_counts"]
    selector = "_".join(words) + "_revocation_valid"
    rules, cases = _fixture(selector)
    assert len(_matches(rules, cases)) == snapshots["_".join(words)]


@pytest.mark.parametrize("citation", JURISDICTION_CITATIONS)
@pytest.mark.parametrize(
    "definition",
    [
        "The election is revoked only if it has been approved by the court.",
        "The election may be revoked if the person has given consent.",
        "The election must be made, and may be revoked, only after the board has approved it.",
        "The election is revoked but the revocation is ineffective.",
    ],
)
def test_unparsed_definition_tail_cannot_authenticate_composite_dependency(
    monkeypatch, citation, definition
):
    # A prefix ending in 'revoked' does not resolve the complete proposition.
    # All selected excerpts are authentic, and assertions agree with formula;
    # neither fact proves the omitted qualifier or its scope.
    monkeypatch.setitem(globals(), "CITATION", citation)
    rules, cases = _fixture()
    rules[SELECTOR]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = definition
    assert not _matches(rules, cases, source=definition + "\n" + CONDITION)
