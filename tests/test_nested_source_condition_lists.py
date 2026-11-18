"""Nested alternatives remain required without becoming simultaneous facts."""

import copy

import pytest

from axiom_encode.harness import source_completeness as module

SOURCE = """Shelter costs of a temporarily unoccupied residence are allowable only when all of the following apply:

● The budgetary unit intends to return to the residence.

● The residence is not leased or rented to others.

● The residence is unoccupied for any of the following reasons:

Illness

Employment or training away from home

Abandonment caused by a natural disaster or casualty loss(g)

Quick access to the types of shelter expenses:

● Shelter Insurance

● Taxes"""

OUTER = (
    "residence_is_temporarily_unoccupied",
    "budgetary_unit_intends_to_return_to_residence",
    "residence_is_leased_to_others",
    "residence_is_rented_to_others",
)
REASONS = (
    "residence_unoccupied_due_to_illness",
    "residence_unoccupied_due_to_employment_away_from_home",
    "residence_unoccupied_due_to_training_away_from_home",
    "residence_unoccupied_due_to_abandonment_caused_by_natural_disaster",
    "residence_unoccupied_due_to_abandonment_caused_by_casualty_loss",
)


def payload(formula):
    return {
        "inputs": [{"name": name, "dtype": "Boolean"} for name in (*OUTER, *REASONS)],
        "rules": [
            {
                "name": "condition_satisfied",
                "kind": "derived",
                "dtype": "Judgment",
                "source": "generic/manual/condition",
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "kind": "condition",
                                "source": {
                                    "corpus_citation_path": "generic/manual/condition",
                                    "excerpt": SOURCE,
                                },
                            }
                        ]
                    }
                },
                "versions": [{"effective_from": "2026-08-01", "formula": formula}],
            }
        ],
    }


def findings(data, source=SOURCE):
    data = copy.deepcopy(data)
    data["rules"][0]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = source
    return module._opaque_same_source_condition_input_issues(
        data,
        source_text=source,
        branches=module.recognize_source_structure(source),
        principal_rules={"condition_satisfied": data["rules"][0]},
        corpus_citation_path="generic/manual/condition",
    )


def formula(reason=REASONS):
    return f"{OUTER[0]} and {OUTER[1]} and not {OUTER[2]} and not {OUTER[3]} and ({' or '.join(reason)})"


@pytest.mark.parametrize("navigation", [True, False])
@pytest.mark.parametrize("reasons", [REASONS, *[(reason,) for reason in REASONS]])
def test_each_explicit_reason_corroborates_nested_gate(navigation, reasons):
    # This gate checks opacity of a reachable path, not OR/AND equivalence or
    # completeness of the complete formula's alternative branches.
    source = SOURCE if navigation else SOURCE.split("Quick access")[0]
    assert not findings(payload(formula(reasons)), source)


@pytest.mark.parametrize(
    "mutation", ["return", "lease", "reason", "opaque", "unqualified_employment"]
)
def test_real_gate_loss_remains_rejected(mutation):
    expression = formula()
    if mutation == "return":
        expression = expression.replace(f" and {OUTER[1]}", "")
    elif mutation == "lease":
        expression = expression.replace(f" and not {OUTER[2]}", "").replace(
            f" and not {OUTER[3]}", ""
        )
    elif mutation == "reason":
        expression = expression.split(" and (")[0]
    elif mutation == "opaque":
        expression = "overall_eligible"
    else:
        expression = formula(("residence_unoccupied_due_to_employment",))
    data = payload(expression)
    data["inputs"].extend(
        [
            {"name": "overall_eligible", "dtype": "Boolean"},
            {"name": "residence_unoccupied_due_to_employment", "dtype": "Boolean"},
        ]
    )
    assert findings(data)


@pytest.mark.parametrize(
    "change",
    [
        "Illness unless another person pays",
        "Illness without assistance",
        "Illness except when another person pays",
        "Illness and disability",
        "Illness: explanation",
        "Illness\n● Another requirement applies.",
    ],
)
def test_unknown_nested_continuation_keeps_conservative_analysis(change):
    source = SOURCE.replace("Illness", change)
    body = source.split("only when ")[1]
    assert module._closed_nested_reason_list(body) is None
    assert findings(payload(formula()), source)


@pytest.mark.parametrize(
    "change",
    [
        "The residence needs Quick access to the types of shelter expenses:",
        "Quick access to the types of shelter expenses:\n● Insurance is required",
        "Quick access to the types of shelter expenses:\nA requirement follows.",
        "Quick access to the types of shelter expenses:\n● Participants pay shelter expenses",
    ],
)
def test_navigation_is_a_closed_standalone_caption_not_incidental_words(change):
    source = SOURCE.replace("Quick access to the types of shelter expenses:", change)
    assert module._closed_nested_reason_list(source.split("only when ")[1]) is None
    assert findings(payload(formula()), source)


def test_parser_removal_regresses_exact_nested_shape(monkeypatch):
    assert not findings(payload(formula()))
    monkeypatch.setattr(module, "_closed_nested_reason_list", lambda body: None)
    assert findings(payload(formula()))


@pytest.mark.parametrize(
    "name",
    [
        "residence_unoccupied_not_due_to_illness",
        "residence_unoccupied_due_to_no_illness",
        "residence_unoccupied_due_to_illness_unless_qualified",
    ],
)
def test_negative_or_qualified_reason_is_not_positive_evidence(name):
    data = payload(formula((name,)))
    data["inputs"].append({"name": name, "dtype": "Boolean"})
    assert findings(data)


def test_negative_description_is_not_positive_reason_evidence():
    data = payload(formula((REASONS[0],)))
    for item in data["inputs"]:
        if item["name"] == REASONS[0]:
            item["description"] = "The residence is not unoccupied due to illness."
    assert findings(data)


def test_separate_exact_proof_atoms_preserve_complete_nested_list():
    data = payload(formula())
    excerpts = [
        SOURCE.split("\n")[0],
        "The budgetary unit intends to return to the residence.",
        "The residence is not leased or rented to others.",
        SOURCE.split("● ")[3].split("Quick access")[0].strip(),
    ]
    data["rules"][0]["metadata"]["proof"]["atoms"] = [
        {
            "path": "versions[0].formula",
            "kind": "condition",
            "source": {
                "corpus_citation_path": "generic/manual/condition",
                "excerpt": excerpt,
            },
        }
        for excerpt in excerpts
    ]
    assert not module._opaque_same_source_condition_input_issues(
        data,
        source_text=SOURCE,
        branches=(),
        principal_rules={"condition_satisfied": data["rules"][0]},
        corpus_citation_path="generic/manual/condition",
    )


def test_reason_marker_cannot_be_dropped_without_regression(monkeypatch):
    original = module._source_conjunctive_fact_gates

    def drop_reason(text, **kwargs):
        return tuple(
            gate
            for gate in original(text, **kwargs)
            if not any(
                token.startswith(module._SOURCE_ANY_REASON_PREFIX) for token in gate[1]
            )
        )

    monkeypatch.setattr(module, "_source_conjunctive_fact_gates", drop_reason)
    with pytest.raises(AssertionError):
        test_real_gate_loss_remains_rejected("reason")
