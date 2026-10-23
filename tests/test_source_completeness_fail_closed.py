"""Reverted relaxations preserve origin/main results, including former wins.

The snapshots were executed against fd31d10a6b8115d95533014c1e0a99579b2fdd5d.
They contain complete public analysis results, not just an expected issue count.
No checkout, network access, or second implementation is needed at test time.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

BASE = "fd31d10a6b8115d95533014c1e0a99579b2fdd5d"
JURISDICTIONS = ("uk", "us", "de", "fr", "nz")
SNAPSHOT_PATH = (
    Path(__file__).parent / "fixtures" / "source_completeness" / "fail_closed_base.json"
)
PARENT = "A person is entitled to a payment if:"
CITIZEN = "the person is a citizen."
DEFINITION = (
    "The election must be made, and may be revoked, in a form and manner "
    "approved by MSD."
)
CONDITION = (
    "Unless it is revoked, the election stops the person from being entitled "
    "to the payment for any days after the election is made."
)
SELECTOR = "winter_energy_payment_election_revocation_valid"
OUTPUT = "election_stops_winter_energy_payment_entitlement"


def _condition_fixture(case, citation):
    predicate, formula = {
        "or_encoded_as_and": (
            "the person is ordinarily resident; or",
            "ordinarily_resident and citizen",
        ),
        "literal_double_negation": (
            "the person is not not ordinarily resident; and",
            "not ordinarily_resident and citizen",
        ),
        "natural_double_negation": (
            "it is not true that the person is not ordinarily resident; and",
            "not ordinarily_resident and citizen",
        ),
        "unresolved_negation": (
            "the person is not known to be ordinarily resident; and",
            "not ordinarily_resident and citizen",
        ),
        "compound_leaf_omits_citizenship": (
            "the person is ordinarily resident and is a citizen.",
            "ordinarily_resident",
        ),
        "ordinary_and": (
            "the person is ordinarily resident; and",
            "ordinarily_resident and citizen",
        ),
    }[case]
    compound = case == "compound_leaf_omits_citizenship"
    source = f"1\n{PARENT}\na\n{predicate}"
    excerpts = [PARENT, predicate]
    if not compound:
        source += f"\nb\n{CITIZEN}"
        excerpts.append(CITIZEN)
    rule = _rule("payment_entitlement", formula, citation, excerpts)
    rule["source"] = citation + "(1)"
    cases = []
    for resident, citizen in (
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    ):
        expected = (
            resident
            if compound
            else (not resident and citizen)
            if formula.startswith("not")
            else resident and citizen
        )
        cases.append(
            {
                "name": f"resident={resident}, citizen={citizen}",
                "period": "2020",
                "input": {"ordinarily_resident": resident, "citizen": citizen},
                "output": {"payment_entitlement": "holds" if expected else "not_holds"},
            }
        )
    return source, [rule], cases, {}


def _schedule_fixture(case, citation):
    prefix = "The amount is income multiplied by 100. "
    source, reason = {
        "excluded": (
            prefix + "The rate in Part 8 of Schedule 4 does not apply.",
            "Part 8 of Schedule 4 is unavailable.",
        ),
        "not_stated": (
            prefix + "The rate is not stated in Part 8 of Schedule 4.",
            "Part 8 of Schedule 4 is unavailable.",
        ),
        "unaffected": (
            prefix + "The rate is unaffected by Part 8 of Schedule 4.",
            "Part 8 of Schedule 4 is unavailable.",
        ),
        "soft_wrapped_title": (
            prefix + "The rate is stated in Part 8 of Schedule 4 of the Actual "
            "Benefits Act 2020\nAmendment Act 2025.",
            "Part 8 of Schedule 4 of the Actual Benefits Act 2020 is unavailable.",
        ),
        "space_compound_title": (
            prefix + "The rate is stated in Part 8 of Schedule 4 of the Actual "
            "Benefits Act 2020 Amendment Act 2025.",
            "Part 8 of Schedule 4 of the Actual Benefits Act 2020 is unavailable.",
        ),
    }.get(case, ("", ""))
    if not source:
        mention = {
            "plural": "Schedules 5 and 6",
            "plural_wrap": "Schedules\n5 and 6",
            "ordinal_plural": "Fifth and Sixth Schedules",
            "ordinal_plural_wrap": "Fifth and Sixth\nSchedules",
            "unparsed_schedule": "Schedule Alpha",
        }[case]
        source = (
            "In Schedule 4, Part 7, replace clause 4 with the following. "
            "The amount is income multiplied by 100 using the rate in clause 1 "
            f"of {mention}."
        )
        reason = "Schedule 4, Part 7, clause 1 is unavailable."
    multiplier = {
        "name": "multiplier",
        "kind": "parameter",
        "dtype": "Decimal",
        "period": "Year",
        "source": citation,
        "versions": [{"effective_from": "2020-01-01", "formula": 100}],
    }
    cases = [
        {"name": "factor", "period": "2020", "input": {}, "output": {"multiplier": 100}}
    ]
    country, locator = citation.split("/statute/", 1)
    module = {
        "deferred_outputs": [
            {"output": f"{country}:statutes/{locator}#amount", "reason": reason}
        ]
    }
    return source, [multiplier], cases, module


CATEGORY_NEGATIVES = {
    "identifier_operand": "The amount is 50 + single_person_income.",
    "attached_identifier_operand": "The amount is 50+ single_person_income.",
    "missing_article": "The amount is 50+ income as defined in section 21.",
    "spaced_operator": "The amount is a total of 50 + income as defined in section 21.",
    "numeric_operand": "The amount is a 50 + 20 supplement.",
    "unreferenced_label": "The person is a 50+ single person.",
    "unknown_reference": "The person is a 50+ single person discussed in section 21.",
    "attached_arithmetic_with_article": (
        "The amount is a 50+ income as defined in section 21."
    ),
    "attached_household_arithmetic": (
        "The amount is a 50+ household income as defined in section 21."
    ),
    "unparsed_definition": "The person is a 50+ single person as defined in a rumour.",
    "excluded_definition": (
        "The person is a 50+ single person as defined in section 21, "
        "except that this definition does not apply."
    ),
    "former_defined_single_person": (
        "the person is aged 50 to 64 years, is a 50+ single person as defined in "
        "section 21 of the Residential Care and Disability Support Services Act "
        "2018 , and is under section 22 of that Act liable to make a contribution "
        "based on income towards the cost of the person’s LTR contracted care "
        "that is less than the maximum contribution (as defined in section 15(2) "
        "of that Act)"
    ),
}


def _category_fixture(case, citation):
    return CATEGORY_NEGATIVES[case], [], [], {}


def _rule(name, formula, citation, excerpts):
    return {
        "name": name,
        "kind": "derived",
        "dtype": "Judgment",
        "period": "Year",
        "source": citation,
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
                    for excerpt in excerpts
                ]
            }
        },
        "versions": [{"effective_from": "2020-01-01", "formula": formula}],
    }


def _dependency_fixture(case, citation):
    definition = {
        "negated_definition": "It is false that the election is revoked.",
        "reported_definition": "The claim that the election is revoked is false.",
        "sliced_negated_definition": "It is false that the election is revoked.",
        "unrelated_subject": "A parking permit may be revoked after a warning.",
        "unresolved_only_if": (
            "The election may be revoked only if the revocation is approved."
        ),
        "unresolved_report": (
            "The election may be revoked, according to an incorrect claim."
        ),
        "unresolved_exclusion": (
            "The election may be revoked, but this does not establish that any "
            "revocation occurred."
        ),
        "unresolved_unless": (
            "The election may be revoked unless no revocation was made."
        ),
        "unresolved_if": (
            "The election may be revoked if the election is not revoked."
        ),
        "unresolved_negative_qualification": (
            "The election may be revoked in a form and manner not approved by MSD."
        ),
        "unresolved_conjunction": (
            "The election may be revoked, and the source does not define revocation."
        ),
    }.get(case, DEFINITION)
    condition = {
        "double_negation": CONDITION.replace("it is revoked", "it is not not revoked"),
        "nested_negation": CONDITION.replace(
            "it is revoked", "it is not true that the election is not revoked"
        ),
    }.get(case, CONDITION)
    selector_formula = (
        "not revocation_made"
        if case == "inverted_action"
        else "revocation_in_msd_approved_form"
        if case == "action_qualifier"
        else "revocation_made and revocation_in_msd_approved_form "
        "and revocation_in_msd_approved_manner"
    )
    dependency = _rule(SELECTOR, selector_formula, citation, [definition])
    output = _rule(
        OUTPUT,
        "election_valid and day_is_after_election_made and not " + SELECTOR,
        citation,
        [condition],
    )
    dependency["source"] = citation + "(3)"
    output["source"] = citation + "(4)"
    atom = dependency["metadata"]["proof"]["atoms"][0]
    if case == "missing_proof":
        dependency.pop("metadata")
    elif case == "wrong_citation":
        atom["source"]["corpus_citation_path"] = citation.replace("/72", "/73")
    elif case == "unselected_proof":
        atom["path"] = "versions[1].formula"
    elif case == "sliced_negated_definition":
        atom["source"]["excerpt"] = "the election is revoked"
    elif case == "missing_condition_proof":
        output.pop("metadata")
    cases = []
    for revoked in (False, True):
        selector = not revoked if case == "inverted_action" else revoked
        cases.append(
            {
                "name": "revocation present" if revoked else "revocation absent",
                "period": "2020",
                "input": {
                    "revocation_made": revoked,
                    "revocation_in_msd_approved_form": (
                        revoked if case == "action_qualifier" else True
                    ),
                    "revocation_in_msd_approved_manner": True,
                    "election_valid": True,
                    "day_is_after_election_made": True,
                },
                "output": {SELECTOR: selector, OUTPUT: not selector},
            }
        )
    return f"3\n{definition}\n4\n{condition}", [dependency, output], cases, {}


DOMAINS = {
    "condition": (
        "or_encoded_as_and",
        "compound_leaf_omits_citizenship",
        "literal_double_negation",
        "natural_double_negation",
        "unresolved_negation",
        "ordinary_and",
    ),
    "schedule": (
        "excluded",
        "not_stated",
        "unaffected",
        "soft_wrapped_title",
        "space_compound_title",
        "plural",
        "plural_wrap",
        "ordinal_plural",
        "ordinal_plural_wrap",
        "unparsed_schedule",
    ),
    "category": tuple(CATEGORY_NEGATIVES),
    "dependency": (
        "missing_proof",
        "wrong_citation",
        "unselected_proof",
        "missing_condition_proof",
        "negated_definition",
        "reported_definition",
        "sliced_negated_definition",
        "unrelated_subject",
        "inverted_action",
        "action_qualifier",
        "double_negation",
        "nested_negation",
        "former_positive",
        "unresolved_only_if",
        "unresolved_report",
        "unresolved_exclusion",
        "unresolved_unless",
        "unresolved_if",
        "unresolved_negative_qualification",
        "unresolved_conjunction",
    ),
}
FACTORIES = {
    "condition": _condition_fixture,
    "schedule": _schedule_fixture,
    "category": _category_fixture,
    "dependency": _dependency_fixture,
}
SNAPSHOT_KEYS = tuple(
    f"{family}/{case}/{country}"
    for family, cases in DOMAINS.items()
    for case in cases
    for country in JURISDICTIONS
)


def _observe(key, module=sc):
    family, case, country = key.split("/")
    citation = f"{country}/statute/2020/1/section/72"
    source, rules, cases, extra_module = FACTORIES[family](case, citation)
    payload = {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {"corpus_citation_path": citation},
            **extra_module,
        },
        "rules": rules,
    }
    analysis = module.analyze_complete_source_unit(
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
    observed = {"analysis": asdict(analysis)}
    if family == "category":
        observed["explicit_computation"] = module.source_states_explicit_computation(
            source
        )
        observed["operation_kinds"] = sorted(module._formula_operation_kinds(source))
    if family == "dependency":
        principal_rules = {rule["name"]: rule for rule in rules}
        asserted = {name: cases for name in principal_rules}
        witnesses = module._toggled_formula_boolean_selectors(
            principal_rules, asserted_by_rule=asserted, formula_environment={}
        )
        condition = source.split("\n4\n", 1)[1]
        branch = module.SourceStructureBranch(
            path=("4",),
            label="(4)",
            text=condition,
            start=source.index(condition),
            end=len(source),
            kind="subsection",
        )
        matches = module._exception_witnesses_for_branch(
            branch,
            source_text=source,
            corpus_citation_path=citation,
            principal_rules=principal_rules,
            principal_rule_paths={SELECTOR: {("3",)}, OUTPUT: {("4",)}},
            asserted_by_rule=asserted,
            toggled_exception_selectors=witnesses,
            extract_numeric_occurrences=vp.extract_typed_numeric_inventory_occurrences_from_text,
        )
        observed["exception_matches"] = sorted(
            [w.rule_name, w.selector_name, w.active_value, w.blocks] for w in matches
        )
    # JSON normalization also makes dataclass tuples comparable to the fixture.
    return json.loads(json.dumps(observed))


@pytest.fixture(scope="module")
def base_snapshots():
    snapshots = json.loads(SNAPSHOT_PATH.read_text())
    assert snapshots["base_commit"] == BASE
    assert set(snapshots["observations"]) == set(SNAPSHOT_KEYS)
    return snapshots["observations"]


@pytest.mark.parametrize("key", SNAPSHOT_KEYS)
def test_outside_positive_grammar_equals_base_snapshot(key, base_snapshots):
    assert _observe(key) == base_snapshots[key]


@given(key=st.sampled_from(SNAPSHOT_KEYS))
@settings(max_examples=70, deadline=None)
def test_abstention_base_equivalence_property(key):
    """The same property spans every relaxation and all five citation families."""

    snapshots = json.loads(SNAPSHOT_PATH.read_text())["observations"]
    assert _observe(key) == snapshots[key]
