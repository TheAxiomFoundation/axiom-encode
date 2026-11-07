import copy

import pytest

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.physical_allowance_conditions import (
    read_physical_allowance_condition,
)

SOURCE = """Storage costs for the applicant when expenses of two or more warehouses occur in the same month are allowable only when any of the following applies:

● The applicant occupies and is billed for storage costs on more than one warehouse because of research or training away from office.

● The applicant moves and is billed for storage expenses for more than one warehouse during the same month."""
COUNT = "stored_warehouses_distinct_expense_warehouse_count"
BILLED = "stored_warehouses_billed_for_those_warehouses"
OCCUPIES = "stored_warehouses_occupies_those_warehouses"
MOVES = "stored_warehouses_moves_between_those_warehouses"
RESEARCH = "stored_warehouses_because_of_research_away_from_office"
TRAINING = "stored_warehouses_because_of_training_away_from_office"
FORMULA = f"{COUNT} > 1 and {BILLED} and (({OCCUPIES} and ({RESEARCH} or {TRAINING})) or {MOVES})"


def payload(formula=FORMULA):
    return {
        "inputs": [
            {
                "name": name,
                "dtype": "Count" if name == COUNT else "Boolean",
                "period": "Month",
                "entity": "Household",
            }
            for name in (COUNT, BILLED, OCCUPIES, MOVES, RESEARCH, TRAINING)
        ],
        "rules": [
            {
                "name": "allowance_condition",
                "kind": "derived",
                "dtype": "Judgment",
                "period": "Month",
                "entity": "Household",
                "versions": [{"effective_from": "2026-08-01", "formula": formula}],
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "source": {
                                    "corpus_citation_path": "generic/allowance",
                                    "excerpt": SOURCE,
                                },
                            }
                        ]
                    }
                },
            }
        ],
    }


def findings(data, source=SOURCE):
    return sc._opaque_same_source_condition_input_issues(
        data,
        source_text=source,
        branches=(),
        principal_rules={"allowance_condition": data["rules"][0]},
        corpus_citation_path="generic/allowance",
    )


@pytest.mark.parametrize(
    "formula",
    [
        FORMULA,
        f"{COUNT} >= 2 and {BILLED} and {OCCUPIES} and {RESEARCH}",
        f"{COUNT} > 1 and {BILLED} and {OCCUPIES} and {TRAINING}",
        f"not ({COUNT} <= 1) and {BILLED} and {MOVES}",
    ],
)
def test_complete_positive_granting_paths_are_corroborated(formula):
    # Opacity contract only: this does not establish that all legal alternatives
    # have been encoded, nor certify benefit amounts or raw caller facts.
    assert not findings(payload(formula))


@pytest.mark.parametrize(
    "formula",
    [
        FORMULA.replace(f" and {BILLED}", ""),
        FORMULA.replace(f"{OCCUPIES} and ", ""),
        f"{COUNT} > 1 and {BILLED} and ({OCCUPIES} or {MOVES})",
        FORMULA.replace(f"{COUNT} > 1 and ", ""),
        FORMULA.replace(f"{COUNT} > 1", f"{COUNT} > 0"),
        FORMULA.replace(f"{COUNT} > 1", f"{COUNT} >= 1"),
        FORMULA.replace(f"{COUNT} > 1", f"{COUNT} > 2"),
        f"{COUNT} > 1 and (({OCCUPIES} and {BILLED} and {RESEARCH}) or {MOVES})",
        f"{COUNT} > 1 and (({BILLED} and False) or {MOVES})",
        f"{COUNT} > 1 and {OCCUPIES} and ({BILLED} or {RESEARCH})",
        f"{COUNT} > 1 and {BILLED} and not {MOVES}",
        f"{COUNT} > 1 and {BILLED} and {OCCUPIES} and not {RESEARCH}",
        "overall_eligible",
        "unknown_helper(warehouse_moves)",
    ],
)
def test_missing_wrong_hybrid_negative_or_opaque_paths_fail(formula):
    assert findings(payload(formula))


@pytest.mark.parametrize(
    "field,value", [("period", "Year"), ("entity", "Person"), ("dtype", "Count")]
)
def test_branch_facts_cannot_borrow_another_entity_period_or_type(field, value):
    data = payload()
    next(item for item in data["inputs"] if item["name"] == BILLED)[field] = value
    assert findings(data)


def test_negative_terminal_description_does_not_supply_positive_fact():
    data = payload()
    next(item for item in data["inputs"] if item["name"] == BILLED)["description"] = (
        "Warehouse is not billed."
    )
    assert findings(data)


def test_same_aggregate_flag_cannot_supply_action_and_billing():
    data = payload(f"{COUNT} > 1 and warehouse_moves_billed")
    data["inputs"].append(
        {
            "name": "warehouse_moves_billed",
            "dtype": "Boolean",
            "period": "Month",
            "entity": "Household",
        }
    )
    assert findings(data)


def test_exact_separate_proofs_cover_shared_clause_and_each_branch():
    data = payload()
    condition = read_physical_allowance_condition(SOURCE)
    template = data["rules"][0]["metadata"]["proof"]["atoms"][0]
    atoms = []
    for excerpt in (
        condition.shared,
        *(branch.source for branch in condition.branches),
    ):
        atom = copy.deepcopy(template)
        atom["source"]["excerpt"] = excerpt
        atoms.append(atom)
    data["rules"][0]["metadata"]["proof"]["atoms"] = atoms
    assert not findings(data)
    assert not condition.owns_excerpts((condition.branches[0].source,))
    assert not condition.owns_excerpts((SOURCE + " invented",))


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("any of", "all of"),
        lambda text: text + "\nThe applicant must remain eligible.",
        lambda text: text.replace("research or training", "research unless training"),
        lambda text: text.replace("same month.", "same month except December."),
        lambda text: text.replace("The applicant moves", "The other applicant moves"),
        lambda text: text.replace("storage expenses", "storage expenses and taxes"),
        lambda text: text.replace(
            "more than one warehouse during", "more than two warehouses during"
        ),
    ],
)
def test_unknown_source_conditions_never_receive_alternative_group_analysis(change):
    assert read_physical_allowance_condition(change(SOURCE)) is None


def test_temporal_derived_alias_requires_full_nonoverlapping_coverage():
    data = payload(FORMULA.replace(MOVES, "moving_status"))
    dependency = {
        "name": "moving_status",
        "kind": "derived",
        "dtype": "Judgment",
        "entity": "Household",
        "period": "Month",
        "versions": [{"effective_from": "2026-08-01", "formula": MOVES}],
    }
    data["rules"].append(dependency)
    assert not findings(data)
    dependency["versions"][0]["effective_from"] = "2026-09-01"
    assert findings(data)


@pytest.mark.parametrize(
    "excerpts",
    [(), (SOURCE.splitlines()[0],), (SOURCE.split("● ")[-1],), (SOURCE, SOURCE)],
)
def test_absent_partial_or_duplicate_proofs_fail_closed(excerpts):
    data = payload()
    template = data["rules"][0]["metadata"]["proof"]["atoms"][0]
    atoms = []
    for excerpt in excerpts:
        atom = copy.deepcopy(template)
        atom["source"]["excerpt"] = excerpt
        atoms.append(atom)
    data["rules"][0]["metadata"]["proof"]["atoms"] = atoms
    if len(excerpts) == 2:
        # Different citation cannot contribute missing same-source ownership.
        atoms[0]["source"]["corpus_citation_path"] = "other/source"
        atoms[1]["source"]["excerpt"] = SOURCE.split("● ")[-1]
    assert findings(data)


def test_output_entity_cannot_borrow_another_entity_fact_set():
    data = payload()
    data["rules"][0]["entity"] = "Person"
    assert findings(data)


@pytest.mark.parametrize(
    "name,description",
    [
        (
            COUNT,
            "Number of expense invoices for a single warehouse; duplicate invoices count separately.",
        ),
        (
            BILLED,
            "Warehouse billing exists for one warehouse out of the several expense warehouses.",
        ),
        (
            RESEARCH,
            "Warehouse research away from office is a proposed purpose, actual occupancy is unrelated.",
        ),
        (COUNT, "A free-form description that the closed reader cannot verify."),
    ],
)
def test_contrary_or_unknown_input_contracts_are_not_positive_evidence(
    name, description
):
    data = payload()
    next(item for item in data["inputs"] if item["name"] == name)["description"] = (
        description
    )
    assert findings(data)


@pytest.mark.parametrize(
    "name,replacement",
    [
        (COUNT, COUNT.replace("distinct_", "")),
        (BILLED, BILLED.replace("those_warehouses", "one_warehouse")),
        (OCCUPIES, OCCUPIES.replace("stored_warehouses", "other_warehouses")),
        (OCCUPIES, OCCUPIES.replace("_those_", "_between_those_")),
        (MOVES, MOVES.replace("_between_those_", "_those_")),
        (RESEARCH, RESEARCH.replace("because_of", "interested_in")),
    ],
)
def test_distinct_expense_set_and_causal_name_contracts_are_required(name, replacement):
    data = payload(FORMULA.replace(name, replacement))
    next(item for item in data["inputs"] if item["name"] == name)["name"] = replacement
    assert findings(data)


def test_actor_object_prefix_entity_and_period_are_parameterized():
    source = (
        SOURCE.replace("applicant", "operator")
        .replace("warehouse", "lodging")
        .replace("month", "year")
    )
    data = payload()
    for item in data["inputs"]:
        item["name"] = (
            item["name"]
            .replace("stored_warehouses", "custom_lodgings")
            .replace("warehouse", "lodging")
        )
        item["entity"] = "Person"
        item["period"] = "Year"
    rule = data["rules"][0]
    rule["entity"] = "Person"
    rule["period"] = "Year"
    rule["versions"][0]["formula"] = FORMULA.replace(
        "stored_warehouses", "custom_lodgings"
    ).replace("warehouse", "lodging")
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = source
    assert not findings(data, source)


def test_source_list_caps_decline_incomplete_reading():
    assert read_physical_allowance_condition(SOURCE, limit=2) is None
    assert read_physical_allowance_condition(SOURCE + " " * 12001) is None


def test_undeclared_or_colliding_terminals_fail_closed():
    data = payload()
    data["inputs"] = [item for item in data["inputs"] if item["name"] != BILLED]
    assert findings(data)
    data = payload()
    data["inputs"].append(copy.deepcopy(data["inputs"][1]))
    assert findings(data)
